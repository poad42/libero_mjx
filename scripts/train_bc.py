#!/usr/bin/env python3
"""Train a BC transformer policy on any LIBERO suite.

Training only needs demo data (HDF5) + torch. No robosuite, no Warp, no eval.
Eval is done separately via eval_bc.py (robosuite) or train_and_eval.py (warp).

Usage:
    python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 \
        --batch-size 32 --save checkpoints/spatial_task0.pth
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler

SUITE_TO_BENCHMARK = {
    "spatial": "LIBERO_SPATIAL",
    "object": "LIBERO_OBJECT",
    "goal": "LIBERO_GOAL",
    "scene10": "LIBERO_10",
    "scene90": "LIBERO_90",
}


def main():
    p = argparse.ArgumentParser(description="Train BC on any LIBERO suite")
    p.add_argument("--suite", choices=list(SUITE_TO_BENCHMARK), default="spatial")
    p.add_argument("--task-id", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save", type=str, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
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
            "train.num_workers=4", f"train.n_epochs={args.epochs}",
            f"train.batch_size={args.batch_size}",
            "eval.use_mp=false", "eval.num_procs=1",
            "eval.eval=false", "use_wandb=false",
        ])
    cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))

    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.lifelong.algos import get_algo_class
    from libero.lifelong.utils import control_seed, safe_device, get_task_embs
    from libero.lifelong.datasets import get_dataset, SequenceVLDataset

    control_seed(args.seed)
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(benchmark_name)(cfg.data.task_order_index)

    demo_path = os.path.join(cfg.folder, benchmark.get_task_demonstration(args.task_id))
    if not os.path.exists(demo_path):
        print(f"ERROR: Demo dataset not found: {demo_path}")
        print("Only libero_spatial has demo data available.")
        sys.exit(1)

    task_dataset_raw, shape_meta = get_dataset(
        dataset_path=demo_path,
        obs_modality=cfg.data.obs.modality, initialize_obs_utils=True, seq_len=cfg.data.seq_len,
    )
    cfg.shape_meta = shape_meta

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task_dataset = SequenceVLDataset(task_dataset_raw, task_embs[args.task_id])
    algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), device)

    train_loader = DataLoader(
        task_dataset, batch_size=args.batch_size,
        sampler=RandomSampler(task_dataset),
        num_workers=4, drop_last=True, persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(algo.policy.parameters(), lr=args.lr, weight_decay=0.0)
    grad_clip = getattr(cfg.train, "grad_clip", 1.0)

    print(f"[train] suite={args.suite} task={args.task_id} epochs={args.epochs} batch={args.batch_size}")
    print(f"[train] demo: {demo_path}")
    print(f"[train] samples={len(task_dataset)} batches={len(train_loader)} device={device}")

    best_loss = float("inf")
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        algo.policy.train()
        total_loss = 0.0
        total_n = 0
        t0 = time.time()

        for batch in train_loader:
            data = algo.map_tensor_to_device(batch)
            loss = algo.policy.compute_loss(data)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(algo.policy.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(batch)
            total_n += len(batch)

        avg_loss = total_loss / max(total_n, 1)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:03d} | loss={avg_loss:.6f} | {elapsed:.1f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            from libero.lifelong.utils import torch_save_model
            torch_save_model(algo.policy, args.save, cfg=cfg)

    print(f"[train] Done. best_loss={best_loss:.6f} saved={args.save}")


if __name__ == "__main__":
    main()