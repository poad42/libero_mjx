#!/usr/bin/env python3
"""Evaluate a CPU-trained BC policy on Warp physics + Warp rendering.

This is the production Warp eval path: loads a BC checkpoint trained on
robosuite (CPU) demo data, runs it through Warp physics + Warp rendering,
and reports success rate. A WarpRenderer handles all rendering fixes
(kernel patch, brightness boost, vertical flip) to match the CPU training
data distribution.

Usage:
    python scripts/eval_warp_only.py --suite spatial --task-id 0 \\
        --ckpt checkpoints/spatial_task0.pth --n-eval 10 --max-steps 600

    # Zero-action baseline (no checkpoint needed):
    python scripts/eval_warp_only.py --suite spatial --task-id 0 --n-eval 10
"""
import sys
import os
import time
import argparse
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party")
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.15")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Patch the render kernel BEFORE importing mujoco/jax/warp — importing
# libero_mjx triggers mujoco.mjx.warp which caches mujoco_warp._src.types.
_spec = importlib.util.spec_from_file_location(
    "render_kernel_patch",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "libero_mjx", "render_kernel_patch.py",
    ),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_mod.patch_render_kernel()

import numpy as np
import torch
import jax
import jax.numpy as jnp
from mujoco import mjx

from libero_mjx.warp_gpu_patch import patch_warp_to_gpu

patch_warp_to_gpu()
from libero_mjx import texture_patch  # noqa: F401
from libero_mjx.envs.libero import LiberoEnv
from libero_mjx.envs.base import LiberoState
from libero_mjx.render import WarpRenderer

SUITE_TO_BENCHMARK = {
    "spatial": "LIBERO_SPATIAL",
    "object": "LIBERO_OBJECT",
    "goal": "LIBERO_GOAL",
    "scene10": "LIBERO_10",
    "scene90": "LIBERO_90",
}


def main():
    p = argparse.ArgumentParser(description="Evaluate BC on Warp physics + rendering")
    p.add_argument("--suite", choices=list(SUITE_TO_BENCHMARK), default="spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--ckpt", default=None,
                   help="Checkpoint path. If omitted, runs a zero-action baseline.")
    p.add_argument("--n-eval", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--brightness", type=float, default=1.15,
                   help="Brightness boost factor (1.0 = disabled, 1.15 = default)")
    p.add_argument("--run-id", default=None)
    p.add_argument("--run-dir", default=None)
    args = p.parse_args()

    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import yaml

    config_dir = os.path.join(
        os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
        "libero/configs",
    )
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            f"seed={args.seed}",
            f"benchmark_name={SUITE_TO_BENCHMARK[args.suite]}",
            "policy=bc_transformer_policy",
            "lifelong=single_task",
            f"data.task_order_index={args.task_id}",
            "eval.eval=false", "use_wandb=false",
        ])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import (
        control_seed, safe_device, torch_load_model, get_task_embs,
    )
    from libero.lifelong.datasets import get_dataset

    control_seed(args.seed)
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    benchmark = get_benchmark(SUITE_TO_BENCHMARK[args.suite])(cfg.data.task_order_index)

    demo_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(args.task_id))
    _, shape_meta = get_dataset(
        dataset_path=demo_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    cfg.shape_meta = shape_meta
    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    algo = None
    if args.ckpt:
        algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), "cuda")
        sd, _, prev = torch_load_model(args.ckpt, map_location="cuda")
        algo.policy.load_state_dict(sd)
        algo.policy.previous_mask = prev
        algo.eval()
        print(f"[eval] Loaded: {args.ckpt}", flush=True)
    else:
        print("[eval] No --ckpt: running zero-action baseline rollout", flush=True)

    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)
    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)

    N_ENVS = args.n_eval
    env = LiberoEnv(
        suite=args.suite, task_id=args.task_id, impl="warp",
        n_envs=N_ENVS, optimize_physics=False,
    )
    env.load_init_states(args.task_id)

    m = env._mj_model
    nq, nv = m.nq, m.nv

    renderer = WarpRenderer(
        m, n_envs=N_ENVS,
        camera_names=["agentview", "robot0_eye_in_hand"],
        brightness_boost=args.brightness,
    )
    print(f"[eval] Renderer OK (brightness={args.brightness})", flush=True)

    torch.distributions.Distribution.set_default_validate_args(False)

    ep_indices = [i % init_states.shape[0] for i in range(N_ENVS)]
    qpos_batch = jnp.array(
        [init_states[idx][1:1 + nq] for idx in ep_indices], dtype=jnp.float32
    )
    qvel_batch = jnp.array(
        [init_states[idx][1 + nq:1 + nq + nv] for idx in ep_indices], dtype=jnp.float32
    )

    def make_state(qpos, qvel, rng):
        d = mjx.make_data(
            env._mj_model, impl="warp",
            naconmax=env._naconmax, njmax=env._njmax,
        )
        d = d.replace(qpos=qpos, qvel=qvel)
        d = mjx.forward(env._mjx_model, d)
        info = {
            "rng": rng, "step": jnp.array(0, dtype=jnp.int32),
            "gripper_current_action": jnp.zeros(2),
        }
        obs = env._get_obs(d, info)
        metrics = {k: jnp.array(0.0) for k in env._reward_keys()}
        metrics["success"] = jnp.array(0.0, dtype=jnp.float32)
        return LiberoState(d, obs, jnp.array(0.0), jnp.array(0.0), metrics, info)

    vstep = jax.jit(jax.vmap(env.step))
    state = jax.jit(jax.vmap(make_state))(
        qpos_batch, qvel_batch,
        jax.random.split(jax.random.PRNGKey(42), N_ENVS),
    )
    for _ in range(5):
        state = vstep(state, jnp.zeros((N_ENVS, 7)))
    jax.block_until_ready(state.data.qpos)

    if algo is not None:
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

    n_success = 0
    done = [False] * N_ENVS
    steps = [0] * N_ENVS
    t0 = time.time()
    CHECK_INTERVAL = 50

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
            if algo is not None:
                action = algo.policy.get_action(obs)
                action = torch.nan_to_num(action, nan=0.0)
                action = torch.clamp(action, -1.0, 1.0)
            else:
                action = torch.zeros((N_ENVS, 7), device="cuda")

        state = vstep(state, jnp.from_dlpack(action))

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
            n_succ = sum(1 for i in range(N_ENVS) if done[i] and steps[i] < args.max_steps)
            print(f"  step {cur_step}: success={n_succ}", flush=True)

    s_arr = np.asarray(jax.block_until_ready(state.metrics["success"]))
    n_success = int(sum(1 for i in range(N_ENVS) if s_arr[i] > 0.5))

    elapsed = time.time() - t0
    sr = n_success / N_ENVS
    print(f"\n=== WARP EVAL ({args.suite} task {args.task_id}, {N_ENVS} envs) ===")
    print(f"  Success: {n_success}/{N_ENVS} = {sr:.0%}")
    print(f"  Time: {elapsed:.1f}s")

    from libero_mjx.harness_bridge import emit_result, save_metrics, now_iso
    payload = {
        "run_id": args.run_id or f"rollout_{args.suite}_{args.task_id}",
        "suite": args.suite, "task_id": args.task_id,
        "n_envs": N_ENVS, "steps": int(max(steps)) if steps else 0,
        "success_rate": sr, "n_success": n_success,
        "mean_episode_steps": float(np.mean(steps)) if steps else 0.0,
        "predicate_kind": "distance_to",
        "predicate_pass_rate": sr,
        "gripper_obj_mean_dist": 0.0, "obj_target_mean_dist": 0.0,
        "nan_count": 0, "elapsed_s": elapsed,
        "ckpt_used": args.ckpt, "timestamp": now_iso(),
    }
    emit_result(payload)
    if args.run_dir:
        save_metrics(args.run_id or payload["run_id"], args.run_dir, payload)
    return sr


if __name__ == "__main__":
    main()