#!/usr/bin/env python3
"""Evaluate a BC policy on LIBERO via robosuite OffScreenRenderEnv.

This is the validated eval path: same rendering and physics as training data.
No Warp, no JAX — just torch + robosuite + mujoco.

Usage:
    python scripts/eval_bc.py --suite spatial --task-id 0 \
        --ckpt checkpoints/spatial_task0.pth --n-eval 20 --max-steps 600
"""
import sys, os, time, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch

# Patch robosuite for non-spatial suites (NullMount base fix)
from libero_mjx.robosuite_patch import patch_robosuite
patch_robosuite()

SUITE_TO_BENCHMARK = {
    "spatial": "LIBERO_SPATIAL",
    "object": "LIBERO_OBJECT",
    "goal": "LIBERO_GOAL",
    "scene10": "LIBERO_10",
    "scene90": "LIBERO_90",
}


def main():
    p = argparse.ArgumentParser(description="Evaluate BC on LIBERO via robosuite")
    p.add_argument("--suite", choices=list(SUITE_TO_BENCHMARK), default="spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-eval", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if args.device != "auto":
        device = args.device

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    benchmark_name = SUITE_TO_BENCHMARK[args.suite]

    sys.path.insert(0, os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"))
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from easydict import EasyDict
    import yaml

    config_dir = os.path.join(os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
                              "libero/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=[
            f"seed={args.seed}", f"benchmark_name={benchmark_name}",
            "policy=bc_transformer_policy", "lifelong=single_task",
            f"data.task_order_index={args.task_id}",
            "train.num_workers=4", "train.n_epochs=50",
            "train.batch_size=32", "eval.use_mp=false",
            "eval.num_procs=1", f"eval.n_eval={args.n_eval}",
            f"eval.max_steps={args.max_steps}",
            "eval.eval=false", "use_wandb=false",
        ])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import control_seed, safe_device, torch_load_model, get_task_embs
    from libero.lifelong.datasets import get_dataset
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv

    control_seed(args.seed)
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(benchmark_name)(cfg.data.task_order_index)

    demo_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(args.task_id))
    if not os.path.exists(demo_path):
        print(f"ERROR: Demo dataset not found: {demo_path}")
        sys.exit(1)

    _, shape_meta = get_dataset(
        dataset_path=demo_path,
        obs_modality=cfg.data.obs.modality, initialize_obs_utils=True, seq_len=cfg.data.seq_len,
    )
    cfg.shape_meta = shape_meta

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), device)
    sd, saved_cfg, prev = torch_load_model(args.ckpt, map_location=device)
    algo.policy.load_state_dict(sd)
    algo.policy.previous_mask = prev if prev is not None else None
    algo.eval()
    print(f"[eval] Loaded BC from {args.ckpt}")

    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)
    init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    init_states = torch.load(init_states_path, weights_only=False)

    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": cfg.data.img_h, "camera_widths": cfg.data.img_w,
    }
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
    env.reset()
    env.seed(args.seed)

    successes = []
    t0 = time.time()

    for ep in range(args.n_eval):
        env.reset()
        idx = ep % init_states.shape[0]
        algo.reset()
        obs = env.set_init_state(init_states[idx:idx + 1])
        dummy = np.zeros((1, 7))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)

        done = False
        for step in range(args.max_steps):
            data = raw_obs_to_tensor_obs(obs, task_emb, cfg)
            with torch.no_grad():
                action = algo.policy.get_action(data)
            obs, _, done_arr, _ = env.step(action[:1])
            done = bool(done_arr[0])
            if done:
                break

        successes.append(int(done))
        print(f"  ep {ep + 1}/{args.n_eval}: success={int(done)} steps={step + 1}", flush=True)

    env.close()
    sr = sum(successes) / len(successes)
    print(f"\n=== BC eval ({args.suite} task {args.task_id}, {args.n_eval} eps) ===")
    print(f"  Success rate: {sr:.2%} ({sum(successes)}/{len(successes)})")
    print(f"  Time: {time.time() - t0:.1f}s")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"suite": args.suite, "task_id": args.task_id,
                       "success_rate": sr, "successes": successes,
                       "n_eval": args.n_eval, "seed": args.seed}, f, indent=2)
        print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()