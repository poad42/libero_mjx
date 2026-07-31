"""Benchmark CPU (robosuite) vs Warp (GPU) eval throughput.

Measures wall-clock time for a full eval loop on both paths:
  CPU: 1 env, N_EVAL episodes, sequential
  Warp: N_EVAL envs, 1 batch of episodes, parallel

Reports env-steps/second and speedup.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

import importlib.util
spec = importlib.util.spec_from_file_location(
    "render_kernel_patch",
    os.path.join(os.path.dirname(__file__), "..", "libero_mjx", "render_kernel_patch.py"),
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
_mod.patch_render_kernel()

import jax
import jax.numpy as jnp
import mujoco.mjx.warp as mjwarp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()
from libero_mjx import texture_patch  # noqa: F401
from libero_mjx.envs.libero import LiberoEnv
from libero_mjx.envs.base import LiberoState
from libero_mjx.render import WarpRenderer
from libero_mjx.robosuite_patch import patch_robosuite
patch_robosuite()


def get_benchmark_cfg(suite, task_id):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import yaml
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark

    benchmark_name = {
        "spatial": "libero_spatial", "object": "libero_object",
        "goal": "libero_goal", "scene10": "libero_10", "scene90": "libero_90",
    }[suite]
    config_dir = os.path.join(os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
                            "libero/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            "seed=42", f"benchmark_name={benchmark_name}",
            "policy=bc_transformer_policy", "lifelong=single_task",
            f"data.task_order_index={task_id}",
            "train.num_workers=4", "train.n_epochs=50",
            "train.batch_size=32", "eval.use_mp=false",
            "eval.num_procs=1", "eval.eval=false", "use_wandb=false",
        ])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    benchmark = get_benchmark(benchmark_name)(task_id)
    task = benchmark.get_task(task_id)
    init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path, weights_only=False)
    return cfg, benchmark, task, init_states


def bench_cpu(args, cfg, benchmark, task, init_states):
    from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
    from libero.lifelong.utils import torch_load_model, safe_device, get_task_embs, control_seed
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.metric import raw_obs_to_tensor_obs

    control_seed(args.seed)
    device = "cuda"
    demo_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(args.task_id))
    _, shape_meta = get_dataset(
        dataset_path=demo_path, obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True, seq_len=cfg.data.seq_len,
    )
    cfg.shape_meta = shape_meta

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    task_emb = benchmark.get_task_emb(args.task_id)

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), device)
    sd, _, prev = torch_load_model(args.ckpt, map_location=device)
    algo.policy.load_state_dict(sd)
    algo.policy.previous_mask = prev
    algo.eval()

    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": 128, "camera_widths": 128,
    }
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
    env.reset()
    env.seed(args.seed)

    total_steps = 0
    t0 = time.time()
    for ep in range(args.n_eval):
        env.reset()
        idx = ep % init_states.shape[0]
        algo.reset()
        obs = env.set_init_state(init_states[idx:idx + 1])
        dummy = np.zeros((1, 7))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)

        for step in range(args.max_steps):
            data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
            with torch.no_grad():
                action = algo.policy.get_action(data)
            obs, _, done_arr, _ = env.step(action[:1])
            total_steps += 1
            if done_arr[0]:
                break
    elapsed = time.time() - t0
    env.close()
    return elapsed, total_steps


def bench_warp(args, cfg, benchmark, task, init_states):
    from libero.lifelong.utils import torch_load_model, safe_device, get_task_embs, control_seed
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from mujoco import mjx

    control_seed(args.seed)
    N_ENVS = args.n_eval
    device = "cuda"
    torch.distributions.Distribution.set_default_validate_args(False)

    demo_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(args.task_id))
    _, shape_meta = get_dataset(
        dataset_path=demo_path, obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True, seq_len=cfg.data.seq_len,
    )
    cfg.shape_meta = shape_meta

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)
    task_emb = benchmark.get_task_emb(args.task_id)

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), device)
    sd, _, prev = torch_load_model(args.ckpt, map_location=device)
    algo.policy.load_state_dict(sd)
    algo.policy.previous_mask = prev
    algo.eval()

    env = LiberoEnv(suite=args.suite, task_id=args.task_id, impl="warp",
                    n_envs=N_ENVS, optimize_physics=False)
    env.load_init_states(args.task_id)

    m = env._mj_model
    nq, nv = m.nq, m.nv
    renderer = WarpRenderer(m, n_envs=N_ENVS, brightness_boost=args.brightness)

    ep_indices = [i % init_states.shape[0] for i in range(N_ENVS)]
    qpos_batch = jnp.array([init_states[idx][1:1 + nq] for idx in ep_indices], dtype=jnp.float32)
    qvel_batch = jnp.array([init_states[idx][1 + nq:1 + nq + nv] for idx in ep_indices], dtype=jnp.float32)

    def make_state(qpos, qvel, rng):
        d = mjx.make_data(m, impl="warp", naconmax=env._naconmax, njmax=env._njmax)
        d = d.replace(qpos=qpos, qvel=qvel)
        d = mjx.forward(env._mjx_model, d)
        info = {"rng": rng, "step": jnp.array(0, dtype=jnp.int32), "gripper_current_action": jnp.zeros(2)}
        obs = env._get_obs(d, info)
        metrics = {k: jnp.array(0.0) for k in env._reward_keys()}
        metrics["success"] = jnp.array(0.0, dtype=jnp.float32)
        return LiberoState(d, obs, jnp.array(0.0), jnp.array(0.0), metrics, info)

    vstep = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(make_state))(
        qpos_batch, qvel_batch,
        jax.random.split(jax.random.PRNGKey(args.seed), N_ENVS),
    )
    for _ in range(5):
        state = vstep(state, jnp.zeros((N_ENVS, 7)))
    jax.block_until_ready(state.data.qpos)

    algo.policy.reset()

    _policy = algo.policy
    def _gpu_get_action(data):
        _policy.eval()
        with torch.no_grad():
            d = _policy.preprocess_input(data, train_mode=False)
            x = _policy.spatial_encode(d)
            _policy.latent_queue.append(x)
            if len(_policy.latent_queue) > _policy.max_seq_len:
                _policy.latent_queue.pop(0)
            x = torch.cat(_policy.latent_queue, dim=1)
            x = _policy.temporal_encode(x)
            dist = _policy.policy_head(x[:, -1])
        return dist.sample().detach().view(-1, 7)
    _policy.get_action = _gpu_get_action

    total_steps = 0
    done = [False] * N_ENVS
    steps = [0] * N_ENVS
    CHECK_INTERVAL = 50

    t0 = time.time()
    while not all(done) and max(steps) < args.max_steps:
        images = renderer.render(state_data=state.data)
        qpos_t = torch.utils.dlpack.from_dlpack(state.data.qpos.__dlpack__())
        obs = {
            "obs": {
                "agentview_rgb": images["agentview_rgb"],
                "eye_in_hand_rgb": images["eye_in_hand_rgb"],
                "joint_states": qpos_t[:, env._robot_arm_qposadr].to("cuda"),
                "gripper_states": qpos_t[:, 7:9].to("cuda"),
            },
            "task_emb": task_emb.unsqueeze(0).repeat(N_ENVS, 1).to("cuda"),
        }
        with torch.no_grad():
            action = algo.policy.get_action(obs)
            action = torch.nan_to_num(action, nan=0.0)
            action = torch.clamp(action, -1.0, 1.0)
        state = vstep(state, jnp.from_dlpack(action))
        total_steps += N_ENVS

        cur_step = max(steps) + 1
        for i in range(N_ENVS):
            if not done[i]:
                steps[i] += 1

        if cur_step % CHECK_INTERVAL == 0 or all(s >= args.max_steps for s in steps):
            success_arr = np.asarray(jax.block_until_ready(state.metrics["success"]))
            for i in range(N_ENVS):
                if done[i]:
                    continue
                if bool(success_arr[i] > 0.5) or steps[i] >= args.max_steps:
                    done[i] = True
    elapsed = time.time() - t0
    return elapsed, total_steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="spatial")
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--brightness", type=float, default=1.15)
    p.add_argument("--skip-cpu", action="store_true")
    p.add_argument("--skip-warp", action="store_true")
    args = p.parse_args()

    cfg, benchmark, task, init_states = get_benchmark_cfg(args.suite, args.task_id)

    print(f"Benchmark: {args.suite} task {args.task_id}, {args.n_eval} episodes, {args.max_steps} max steps")
    print(f"Checkpoint: {args.ckpt}\n")

    results = {}

    if not args.skip_cpu:
        print("=== CPU (robosuite, 1 env, sequential) ===")
        cpu_time, cpu_steps = bench_cpu(args, cfg, benchmark, task, init_states)
        cpu_eps = args.n_eval / cpu_time
        cpu_sps = cpu_steps / cpu_time
        print(f"  Wall time: {cpu_time:.1f}s")
        print(f"  Episodes: {args.n_eval}")
        print(f"  Total env-steps: {cpu_steps}")
        print(f"  Episodes/s: {cpu_eps:.2f}")
        print(f"  Env-steps/s: {cpu_sps:.1f}\n")
        results["cpu"] = {"time": cpu_time, "steps": cpu_steps, "eps": cpu_eps, "sps": cpu_sps}

    if not args.skip_warp:
        print("=== Warp (GPU, parallel) ===")
        warp_time, warp_steps = bench_warp(args, cfg, benchmark, task, init_states)
        warp_eps = args.n_eval / warp_time
        warp_sps = warp_steps / warp_time
        print(f"  Wall time: {warp_time:.1f}s")
        print(f"  Episodes: {args.n_eval} (parallel)")
        print(f"  Total env-steps: {warp_steps}")
        print(f"  Episodes/s: {warp_eps:.2f}")
        print(f"  Env-steps/s: {warp_sps:.1f}\n")
        results["warp"] = {"time": warp_time, "steps": warp_steps, "eps": warp_eps, "sps": warp_sps}

    if "cpu" in results and "warp" in results:
        speedup = results["cpu"]["time"] / results["warp"]["time"]
        sps_speedup = results["warp"]["sps"] / results["cpu"]["sps"]
        print("=== Comparison ===")
        print(f"  Wall time speedup: {speedup:.1f}x")
        print(f"  Throughput speedup: {sps_speedup:.1f}x env-steps/s")


if __name__ == "__main__":
    main()