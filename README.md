# LIBERO-MJX: All 130 LIBERO Tasks on MuJoCo Warp

All 5 LIBERO manipulation benchmark suites ported to MuJoCo Warp for GPU-parallel simulation.  
130 tasks across spatial, object, goal, scene10, and scene90 suites.

## What is this?

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) is a lifelong robot learning benchmark with 130 manipulation tasks built on robosuite (CPU MuJoCo). This repo ports all 130 task environments to [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) (GPU-parallel MuJoCo via JAX/Warp), enabling thousands of parallel environment instances on a single GPU.

## Suites

| Suite | Tasks | Scene Type | Example Goal |
|-------|-------|-----------|--------------|
| spatial | 10 | Tabletop | Pick up the black bowl and place it on the plate |
| object | 10 | Floor | Put the alphabet soup in the basket |
| goal | 10 | Kitchen | Open the middle drawer of the cabinet |
| scene10 | 10 | Kitchen/Living room | Multi-step tasks (e.g. turn on stove, put moka pot on it) |
| scene90 | 90 | Kitchen/Living room | Multi-step tasks with diverse objects |

All 131 task XMLs are included in `libero_mjx/assets/xml/`.

## Installation

```bash
pip install -e .
# For BC training/eval with robosuite:
pip install -e ".[torch,libero]"
# For GPU rendering:
pip install -e ".[render]"
```

### Requirements

- Python >= 3.10
- MuJoCo >= 3.1.0
- JAX >= 0.4.28 (with CUDA or ROCm backend)
- Warp >= 1.0
- A CUDA or ROCm GPU

**NVIDIA (CUDA):** Install JAX with `pip install jax[cuda12]`  
**AMD (ROCm):** Install JAX with `pip install jax-rocm7-pjrt` and ensure `JAX_PLATFORMS=` (auto-detect)

## Quick Start

### Warp environment (GPU-parallel)

```python
import jax
import jax.numpy as jp
from libero_mjx.envs.libero import LiberoEnv

env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=256)
state = env.reset(jax.random.PRNGKey(0))

# Batched step (256 envs in parallel)
action = jp.zeros((256, 7))
state = env.step(state, action)
print(state.metrics["success"])  # (256,) array
```

### Train BC policy

```bash
python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 \
    --batch-size 32 --save checkpoints/spatial_task0.pth
```

### Evaluate BC policy (robosuite)

```bash
python scripts/eval_bc.py --suite spatial --task-id 0 \
    --ckpt checkpoints/spatial_task0.pth --n-eval 20 --max-steps 600
```

## Architecture

```
libero_mjx/
  __init__.py              # Package exports + auto GPU patch
  warp_gpu_patch.py        # ROCm/CUDA FFI + device detection patches
  robosuite_patch.py       # Fixes robot_base_factory for non-spatial suites
  envs/
    base.py                # LiberoMjxEnv: batched reset/step, state save/restore
    libero.py              # LiberoEnv: unified env for all 5 suites
    spatial.py             # Legacy spatial-only env
  controllers/
    osc.py                 # OSC Cartesian impedance controller (JAX)
  predicates/
    spatial.py             # Success predicates (distance, on, in_region, is_open, ...)
  obs/
    __init__.py            # State-only observation builder
  render.py                # Batched GPU rendering via mujoco_warp (DLPack)
  assets/
    xml/                   # 131 task XMLs (extracted from robosuite)

scripts/
  train_bc.py              # Train BC transformer on LIBERO demo data
  eval_bc.py               # Evaluate BC via robosuite OffScreenRenderEnv
  extract_all_xmls.py      # Extract task XMLs from robosuite (already done)

tests/
  test_all_suites.py       # Smoke test: all 5 suites load + step
  test_env_smoke.py        # Basic env reset/step
  test_osc.py              # OSC controller tests
  test_predicates.py       # Predicate tests
  test_spatial.py          # Spatial suite integration test
  test_parallel.py         # Parallelism / vmap tests
```

## BC Training & Evaluation

### Training

Training uses LIBERO's standard BC transformer architecture (ResNet image encoder + temporal transformer + GMM policy head) and demo data (HDF5). Training runs entirely on PyTorch/GPU — no robosuite or Warp needed.

```bash
# Train on any suite (requires demo data downloaded)
python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 --save ckpt.pth

# Other suites
python scripts/train_bc.py --suite object --task-id 0 --epochs 50 --save ckpt.pth
python scripts/train_bc.py --suite goal --task-id 0 --epochs 50 --save ckpt.pth
python scripts/train_bc.py --suite scene10 --task-id 0 --epochs 50 --save ckpt.pth
python scripts/train_bc.py --suite scene90 --task-id 0 --epochs 50 --save ckpt.pth
```

### Evaluation

Evaluation uses robosuite's `OffScreenRenderEnv` for vision-based rendering (same as training data distribution):

```bash
python scripts/eval_bc.py --suite spatial --task-id 0 \
    --ckpt ckpt.pth --n-eval 20 --max-steps 600
```

### Downloading Demo Data

```python
from libero.libero.utils.download_utils import libero_dataset_download
libero_dataset_download(datasets="libero_spatial", use_huggingface=True)
# Also: libero_object, libero_goal, libero_10, libero_90
```

## Results

BC transformer trained 50 epochs, evaluated 20 episodes per task (robosuite eval):

### Spatial (10 tasks)
| Task | Success Rate |
|------|-------------|
| 0 | 50% |
| 1 | 75% |
| 2 | 0% |
| 3 | 30% |
| 4 | 25% |
| 5 | 45% |
| 6 | 45% |
| 7 | 60% |
| 8 | 95% |
| 9 | 90% |
| **Avg** | **51.5%** |

### Object (10 tasks)
| Task | Success Rate |
|------|-------------|
| 0 | 60% |
| 1 | 25% |
| 2 | 85% |
| 3 | 55% |
| 4 | 85% |
| 5 | 80% |
| 6 | 60% |
| 7 | 40% |
| 8 | 35% |
| 9 | 40% |
| **Avg** | **56.5%** |

### Goal, Scene10, Scene90
Results pending — training in progress.

## GPU Rendering (Experimental)

The repo includes a batched GPU render context (`libero_mjx/render.py`) using mujoco_warp's render API with lbvh acceleration structure. This enables zero-copy DLPack interop between Warp (render), PyTorch (policy), and JAX (physics) — all on GPU without host round-trips.

```python
from libero_mjx.render import RenderContext

ctx = RenderContext(env.mj_model, nworld=4, img_h=128, img_w=128,
                    camera_names=["agentview", "robot0_eye_in_hand"])
images = ctx.render(mw_data)  # (4, 2, 128, 128, 3) uint8
```

**Note:** GPU rendering requires additional patches for ROCm (cubql→lbvh BVH constructor). See `warp_gpu_patch.py`.

## Patches

### `warp_gpu_patch.py`
- Fixes JAX FFI platform registration for ROCm (warp registers `ROCM` but XLA looks up `rocm`)
- Patches device detection in MJX io module to recognize ROCm GPUs
- Patches BVH construction to use `lbvh` instead of `cubql` (HIP doesn't support cubql)
- Injects `GraphMode` enum for mujoco-mjx compatibility

### `robosuite_patch.py`
- Fixes `robot_base_factory` returning strings instead of classes for non-Panda robot bases (affects object/goal/scene10/scene90 suites)
- Falls back to `NullMount` for unknown base names

## Citation

If you use this work, please cite both LIBERO and MuJoCo Warp:

```bibtex
@inproceedings{wang2024libero,
  title={LIBERO: Lifelong Robot Learning},
  author={Wang, Haoyu and Wang, Junlin and Mayne, Matthew and Bao, Cheng and Li, Zichen and Ma, Wenlong and Konidaris, George},
  booktitle={ICLR 2024},
}

@misc{mujoco_warp,
  title={MuJoCo Warp: GPU-parallel simulation for MuJoCo},
  author={DeepMind},
  url={https://github.com/google-deepmind/mujoco_warp},
}
```

## License

MIT