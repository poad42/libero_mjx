#!/usr/bin/env python3
"""Train a BC transformer policy on any LIBERO suite with domain randomization.

Training only needs demo data (HDF5) + torch. No robosuite, no Warp, no eval.
Eval is done separately via eval_bc.py (robosuite) or eval_warp_only.py (Warp).

Domain randomization: brightness/contrast/noise/blur on images + optional
state noise on low-dim obs. Designed to bridge CPU→Warp sim-to-real gap.

Usage:
    python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 \
        --batch-size 32 --save checkpoints/spatial_task0.pth --augment
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

SUITE_TO_BENCHMARK = {
    "spatial": "LIBERO_SPATIAL",
    "object": "LIBERO_OBJECT",
    "goal": "LIBERO_GOAL",
    "scene10": "LIBERO_10",
    "scene90": "LIBERO_90",
}


def augment_images(img_batch, strength=1.0):
    """Apply domain randomization to image batch.

    Args:
        img_batch: (B, T, H, W, C) uint8 tensor on GPU (channels-last)
        strength: overall augmentation strength multiplier (0=none, 1=moderate, 2=heavy)

    Augmentations:
        - Brightness: ±20% * strength
        - Contrast: ±15% * strength
        - Gaussian noise: std=5 * strength
        - Channel jitter: per-channel offset ±10 * strength
        - Gaussian blur: kernel=3, prob=0.1 * strength
    """
    if strength <= 0:
        return img_batch

    orig_dtype = img_batch.dtype
    img = img_batch.float()  # (B, T, H, W, C)

    # Brightness: multiply by random factor per sample
    b_factor = torch.empty(img.shape[0], 1, 1, 1, 1, device=img.device).uniform_(
        1.0 - 0.2 * strength, 1.0 + 0.2 * strength
    )
    img = img * b_factor

    # Contrast: scale around per-image mean
    dims = (1, 2, 3, 4)  # T, H, W, C
    mean = img.mean(dim=dims, keepdim=True)
    c_factor = torch.empty(img.shape[0], 1, 1, 1, 1, device=img.device).uniform_(
        1.0 - 0.15 * strength, 1.0 + 0.15 * strength
    )
    img = (img - mean) * c_factor + mean

    # Gaussian noise
    noise_std = 5.0 * strength
    img = img + torch.randn_like(img) * noise_std

    # Channel jitter (per-channel brightness offset)
    c_offset = torch.empty(1, 1, 1, 1, img.shape[-1], device=img.device).uniform_(
        -10 * strength, 10 * strength
    )
    img = img + c_offset

    # Gaussian blur (probabilistic, per-sample)
    if np.random.random() < 0.1 * strength:
        k = 3
        B, T, H, W, C = img.shape
        # avg_pool2d needs (N, C, H, W) — reshape
        img = img.permute(0, 1, 4, 2, 3).contiguous()  # (B, T, C, H, W)
        img = img.view(B * T, C, H, W)
        img = F.avg_pool2d(img, kernel_size=k, stride=1, padding=k // 2)
        img = img.view(B, T, C, H, W).permute(0, 1, 3, 4, 2).contiguous()  # back to (B, T, H, W, C)

    return img.clamp(0, 255).to(orig_dtype)


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
    p.add_argument("--augment", action="store_true", help="Enable domain randomization augmentation")
    p.add_argument("--aug-strength", type=float, default=1.0, help="Augmentation strength (0=none, 1=moderate, 2=heavy)")
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
    if args.augment:
        print(f"[train] augmentation: ON (strength={args.aug_strength})")

    best_loss = float("inf")
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        algo.policy.train()
        total_loss = 0.0
        total_n = 0
        t0 = time.time()

        for batch in train_loader:
            data = algo.map_tensor_to_device(batch)

            # Domain randomization augmentation
            if args.augment:
                for key in ("agentview_rgb", "eye_in_hand_rgb"):
                    if key in data.get("obs", {}):
                        data["obs"][key] = augment_images(data["obs"][key], strength=args.aug_strength)
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