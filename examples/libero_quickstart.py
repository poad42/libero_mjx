#!/usr/bin/env python3
"""Minimal LIBERO-on-Warp example: batched rollout, GPU render, throughput.

Loads one LIBERO task, builds a batched MuJoCo Warp environment, runs a rollout
(zero actions by default, or a BC checkpoint), renders the observation cameras on
the GPU, and prints the success rate and environment-steps per second.

No demo dataset is needed for the zero-action path. A BC checkpoint needs the
LIBERO demo HDF5 for that task, because the policy reads `shape_meta` from it.

Usage:
    python examples/libero_quickstart.py --suite spatial --task-id 0 --n-envs 16
    python examples/libero_quickstart.py --suite spatial --task-id 0 --steps 100 \
        --save-image out.png
    python examples/libero_quickstart.py --suite spatial --task-id 0 --batch-bench
    python examples/libero_quickstart.py --suite object --task-id 0 \
        --checkpoint checkpoints/object_task0.pth --n-envs 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"))
os.environ.setdefault("JAX_PLATFORMS", "rocm")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Import libero_mjx before mujoco. Importing libero_mjx rewrites the mujoco_warp
# render kernel on disk, and that has to happen before mujoco_warp is imported
# (`from mujoco import mjx` pulls it in). If the source is rewritten after the
# module is loaded, warp resolves the nested render megakernel against stale
# line numbers and codegen fails.
import libero_mjx  # noqa: E402,F401

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mujoco import mjx  # noqa: E402

from libero_mjx.envs.libero import LiberoEnv  # noqa: E402
from libero_mjx.envs.base import LiberoState  # noqa: E402
from libero_mjx.render import WarpRenderer  # noqa: E402


def build_state_fn(env: LiberoEnv):
    def make_state(qpos, qvel, rng):
        d = mjx.make_data(
            env._mj_model, impl="warp", naconmax=env._naconmax, njmax=env._njmax
        )
        d = d.replace(qpos=qpos, qvel=qvel)
        d = mjx.forward(env._mjx_model, d)
        info = {
            "rng": rng,
            "step": jnp.array(0, dtype=jnp.int32),
            "gripper_current_action": jnp.zeros(2),
        }
        obs = env._get_obs(d, info)
        metrics = {k: jnp.array(0.0) for k in env._reward_keys()}
        metrics["success"] = jnp.array(0.0, dtype=jnp.float32)
        return LiberoState(d, obs, jnp.array(0.0), jnp.array(0.0), metrics, info)

    return make_state


def init_states(env: LiberoEnv, task_id: int, n_envs: int, seed: int):
    """Return (qpos, qvel) batches, from the LIBERO init file if available."""
    env.load_init_states(task_id)
    nq, nv = env._mj_model.nq, env._mj_model.nv
    states = env._init_states
    if states is None:
        qpos = np.zeros((n_envs, nq), dtype=np.float32)
        qpos[:, :9] = [0.0, 0.0067, -0.1919, -0.0099, -2.4326, -0.0399, 2.1935, 0.0208, -0.0208]
        qvel = np.zeros((n_envs, nv), dtype=np.float32)
        print("[quickstart] no init file found, using the default home pose")
    else:
        states = np.asarray(states)
        idx = np.random.default_rng(seed).integers(0, states.shape[0], size=n_envs)
        qpos = states[idx, 1:1 + nq].astype(np.float32)
        qvel = states[idx, 1 + nq:1 + nq + nv].astype(np.float32)
        print(f"[quickstart] {states.shape[0]} init states from the LIBERO init file")
    return jnp.array(qpos), jnp.array(qvel)


def load_policy(checkpoint: str, suite: str, task_id: int, n_envs: int, arm_qposadr):
    """Load a LIBERO BC transformer checkpoint and return a batched action fn."""
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import yaml

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.utils import (
        control_seed, get_task_embs, safe_device, torch_load_model,
    )

    suite_to_benchmark = {
        "spatial": "LIBERO_SPATIAL", "object": "LIBERO_OBJECT", "goal": "LIBERO_GOAL",
        "scene10": "LIBERO_10", "scene90": "LIBERO_90",
    }
    config_dir = os.path.join(
        os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"), "libero/configs"
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            "seed=42", f"benchmark_name={suite_to_benchmark[suite]}",
            "policy=bc_transformer_policy", "lifelong=single_task",
            f"data.task_order_index={task_id}", "eval.eval=false", "use_wandb=false",
        ])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))
    control_seed(42)
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    benchmark = get_benchmark(suite_to_benchmark[suite])(cfg.data.task_order_index)

    demo_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(task_id))
    if not os.path.exists(demo_path):
        sys.exit(f"demo dataset not found: {demo_path}\n"
                 f"fetch it with scripts/download_datasets.py --suite {suite}")
    _, shape_meta = get_dataset(
        dataset_path=demo_path, obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True, seq_len=cfg.data.seq_len,
    )
    cfg.shape_meta = shape_meta
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    benchmark.set_task_embs(get_task_embs(cfg, descriptions))
    task_emb = benchmark.get_task_emb(task_id)

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), "cuda")
    sd, _, prev = torch_load_model(checkpoint, map_location="cuda")
    algo.policy.load_state_dict(sd)
    algo.policy.previous_mask = prev
    algo.policy.eval()
    policy = algo.policy

    def get_action(obs_images, qpos_t):
        data = {"obs": {
            "agentview_rgb": obs_images["agentview_rgb"],
            "eye_in_hand_rgb": obs_images["eye_in_hand_rgb"],
            "joint_states": qpos_t[:, arm_qposadr].to("cuda"),
            "gripper_states": qpos_t[:, 7:9].to("cuda"),
        }, "task_emb": task_emb.unsqueeze(0).repeat(n_envs, 1).to("cuda")}
        with torch.no_grad():
            d = policy.preprocess_input(data, train_mode=False)
            x = policy.spatial_encode(d)
            policy.latent_queue.append(x)
            if len(policy.latent_queue) > policy.max_seq_len:
                policy.latent_queue.pop(0)
            x = policy.temporal_encode(torch.cat(policy.latent_queue, dim=1))
            dist = policy.policy_head(x[:, -1])
        a = dist.sample().detach().view(-1, 7)
        return torch.nan_to_num(a, nan=0.0).clamp(-1.0, 1.0)

    return get_action, policy


def run(args):
    env = LiberoEnv(
        suite=args.suite, task_id=args.task_id, impl="warp",
        n_envs=args.n_envs, optimize_physics=False,
    )
    qpos, qvel = init_states(env, args.task_id, args.n_envs, args.seed)
    make_state = build_state_fn(env)
    vstep = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(make_state))(
        qpos, qvel, jax.random.split(jax.random.PRNGKey(args.seed), args.n_envs)
    )
    jax.block_until_ready(state.data.qpos)

    renderer = WarpRenderer(
        env._mj_model, n_envs=args.n_envs, img_h=args.img_h, img_w=args.img_w,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    print(f"[quickstart] {args.suite} task {args.task_id}, {args.n_envs} envs, "
          f"{args.img_h}x{args.img_w} cameras, impl=warp")

    policy_fn = None
    if args.checkpoint:
        policy_fn, _ = load_policy(
            args.checkpoint, args.suite, args.task_id, args.n_envs,
            env._robot_arm_qposadr,
        )
        print(f"[quickstart] BC checkpoint: {args.checkpoint}")

    import torch

    def act(images):
        if policy_fn is None:
            return jnp.zeros((args.n_envs, 7))
        qpos_t = torch.utils.dlpack.from_dlpack(state.data.qpos.__dlpack__())
        return jnp.from_dlpack(policy_fn(images, qpos_t))

    # The first render and step compile the Warp and XLA kernels. Warm up so the
    # throughput below is steady state, not compilation.
    for _ in range(args.warmup):
        state = vstep(state, act(renderer.render(state_data=state.data)))
    jax.block_until_ready(state.data.qpos)

    t0 = time.time()
    step = 0
    for step in range(1, args.steps + 1):
        images = renderer.render(state_data=state.data)
        state = vstep(state, act(images))
        if step % args.log_every == 0 or step == args.steps:
            jax.block_until_ready(state.data.qpos)
            elapsed = time.time() - t0
            succ = float(np.asarray(state.metrics["success"]).mean())
            print(f"  step {step:4d}  {step * args.n_envs / elapsed:8.1f} env-steps/s  "
                  f"success={succ:.2f}")

    jax.block_until_ready(state.metrics["success"])
    elapsed = time.time() - t0
    success = np.asarray(state.metrics["success"])
    print(f"[quickstart] {args.n_envs} envs x {step} steps in {elapsed:.1f}s "
          f"= {step * args.n_envs / elapsed:.1f} env-steps/s")
    print(f"[quickstart] success: {int((success > 0.5).sum())}/{args.n_envs} "
          f"= {(success > 0.5).mean():.0%}")

    if args.save_image:
        from PIL import Image
        imgs = renderer.render(state_data=state.data)
        tiles = []
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            tiles.append(np.asarray(imgs[key][0].cpu()))
        strip = np.concatenate(tiles, axis=1)
        Image.fromarray(strip).save(args.save_image)
        print(f"[quickstart] wrote {args.save_image} (agentview | eye-in-hand)")
    return success


def bench(args):
    print(f"[quickstart] batch sweep, {args.suite} task {args.task_id}")
    print(f"{'n_envs':>8}  {'env-steps/s':>12}  {'ms/step':>8}")
    for n in args.bench_envs:
        env = LiberoEnv(suite=args.suite, task_id=args.task_id, impl="warp",
                        n_envs=n, optimize_physics=False)
        qpos, qvel = init_states(env, args.task_id, n, args.seed)
        make_state = build_state_fn(env)
        vstep = jax.jit(jax.vmap(env.step))
        state = jax.jit(jax.vmap(make_state))(
            qpos, qvel, jax.random.split(jax.random.PRNGKey(args.seed), n)
        )
        for _ in range(5):
            state = vstep(state, jnp.zeros((n, 7)))
        jax.block_until_ready(state.data.qpos)
        t0 = time.time()
        for _ in range(args.bench_steps):
            state = vstep(state, jnp.zeros((n, 7)))
        jax.block_until_ready(state.data.qpos)
        dt = (time.time() - t0) / args.bench_steps
        print(f"{n:>8}  {n / dt:>12.1f}  {dt * 1e3:>8.2f}")


def main():
    p = argparse.ArgumentParser(description="Minimal LIBERO-on-Warp example")
    p.add_argument("--suite", default="spatial",
                   choices=["spatial", "object", "goal", "scene10", "scene90"])
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-h", type=int, default=128)
    p.add_argument("--img-w", type=int, default=128)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5,
                   help="Untimed steps before the throughput measurement.")
    p.add_argument("--checkpoint", default=None,
                   help="BC checkpoint (.pth). Needs the task demo HDF5.")
    p.add_argument("--save-image", default=None,
                   help="Write one agentview | eye-in-hand frame to this PNG.")
    p.add_argument("--batch-bench", action="store_true",
                   help="Sweep n_envs and report physics-only env-steps/s.")
    p.add_argument("--bench-envs", type=int, nargs="+", default=[16, 64, 256])
    p.add_argument("--bench-steps", type=int, default=20)
    args = p.parse_args()

    if args.batch_bench:
        bench(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
