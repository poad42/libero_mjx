# libero-mjx

All 130 LIBERO manipulation tasks ported to MuJoCo Warp. 131 task XMLs, 5 suites, a BC transformer training pipeline, and two eval paths: CPU (robosuite) and GPU (Warp physics + Warp rendering).

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) runs 130 manipulation tasks on robosuite with CPU MuJoCo. This repo ports all 130 to [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), which runs MuJoCo on GPU via JAX & Warp. You get thousands of parallel environment instances on one GPU instead of one at a time on CPU.

A BC transformer trained on CPU demo data hits 50% success on CPU eval and 42.5% average on Warp eval (4 seeds: 50%, 40%, 30%, 50%). Without the rendering fixes in this repo, Warp eval gives 0%.

## Suites

| Suite | Tasks | Scene | Example goal |
|-------|-------|-------|--------------|
| spatial | 10 | Tabletop | Pick up the black bowl, place it on the plate |
| object | 10 | Floor | Put the alphabet soup in the basket |
| goal | 10 | Kitchen | Open the middle drawer of the cabinet |
| scene10 | 10 | Kitchen / living room | Turn on stove, put moka pot on it |
| scene90 | 90 | Kitchen / living room | Multi-step tasks with 90 object arrangements |

131 task XMLs ship in `libero_mjx/assets/xml/`.

## Install

```bash
pip install -e .
pip install -e ".[torch,libero]"   # BC training & eval with robosuite
pip install -e ".[render]"        # GPU rendering
```

You need Python 3.10+, MuJoCo 3.1.0+, JAX 0.4.28+, and Warp 1.0+. A CUDA or ROCm GPU is required for Warp physics.

**NVIDIA:** `pip install jax[cuda12]`

**AMD ROCm:** `pip install jax-rocm7-pjrt` and set `JAX_PLATFORMS=` for auto-detect.

### Docker

A Dockerfile targets AMD ROCm (gfx1201 / RDNA 4). For NVIDIA, swap the ROCm block for the CUDA toolkit.

```bash
docker build -t libero-mjx .
./scripts/docker_run.sh python tests/test_all_suites.py
```

`scripts/docker_run.sh` passes through all arguments to the container. It sets `JAX_PLATFORMS=rocm`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`, and mounts the repo at `/workspace/libero-mjx`.

## Quick start

### Warp environment

```python
import jax
import jax.numpy as jp
from libero_mjx.envs.libero import LiberoEnv

env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=256)
state = env.reset(jax.random.PRNGKey(0))

action = jp.zeros((256, 7))
state = env.step(state, action)
print(state.metrics["success"])  # (256,) array
```

### Train a BC policy

```bash
python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 \
    --batch-size 32 --save checkpoints/spatial_task0.pth
```

### Evaluate

Two paths. CPU eval uses robosuite's `OffScreenRenderEnv`, same renderer that produced the training data. Warp eval runs Warp physics & Warp rendering on GPU with 10 parallel envs.

```bash
# CPU eval
python scripts/eval_bc.py --suite spatial --task-id 0 \
    --ckpt checkpoints/spatial_task0.pth --n-eval 20 --max-steps 600

# Warp eval
python scripts/eval_warp_only.py --suite spatial --task-id 0 \
    --ckpt checkpoints/spatial_task0.pth --n-eval 10 --max-steps 600
```

The Warp eval accepts `--brightness 1.15` (default) to correct the Warp ray tracer's output brightness. Pass `--brightness 1.0` to disable.

## Architecture

```
libero_mjx/
  __init__.py              Package exports, auto GPU patch
  warp_gpu_patch.py        ROCm FFI, device detection, lbvh, GraphMode
  robosuite_patch.py       robot_base_factory fallback for non-spatial suites
  texture_patch.py         Warp type code for Texture2D arrays
  render_kernel_patch.py   Patches mujoco_warp render kernel (shadow, haze)
  envs/
    base.py                LiberoMjxEnv: batched reset/step, state save/restore
    libero.py              LiberoEnv: unified env for all 5 suites
    spatial.py             Legacy spatial-only env
  controllers/
    osc.py                 OSC Cartesian impedance controller (JAX port)
  predicates/
    spatial.py             Success predicates: distance, on, in_region, is_open
  obs/
    __init__.py            State-only observation builder
  render.py                WarpRenderer: batched GPU rendering with DLPack
  assets/
    xml/                   131 task XMLs extracted from robosuite

scripts/
  train_bc.py              Train BC transformer on LIBERO demo data
  eval_bc.py               Evaluate BC via robosuite (CPU)
  eval_warp_only.py        Evaluate BC via Warp physics + rendering (GPU)
  extract_all_xmls.py      Extract task XMLs from robosuite
  docker_run.sh            Docker wrapper for GPU scripts

tests/
  test_all_suites.py       Smoke test: all 5 suites load & step
  test_env_smoke.py        Basic env reset/step
  test_osc.py              OSC controller tests
  test_predicates.py       Predicate tests
  test_spatial.py          Spatial suite integration test
  test_parallel.py         Parallelism / vmap tests
  test_vmap_step.py        vmap step tests
  test_physics_compare.py  Warp vs CPU physics comparison
  validate_warp_render.py  Warp renderer validation
```

## Rendering fixes

The Warp ray tracer (mujoco_warp) differs from MuJoCo's CPU / EGL renderer in ways that break a BC policy trained on CPU data. `WarpRenderer` in `libero_mjx/render.py` applies five fixes. The first three patch the installed `mujoco_warp` package on disk before import; the last two run at render time.

### Shadow fallback (render_kernel_patch.py)

The Warp render megakernel set `visible = 0.3` for shadowed pixels. That constant, `NO_LIGHT_AMBIENT_FALLBACK`, keeps 30% of diffuse & specular light on geometry in shadow. MuJoCo's CPU renderer sets `visible = 0.0`. Shadowed geometry gets ambient light only, applied in a separate pass.

The patch changes the constant to `0.0` in `_render_megakernel` in the installed `render.py`. This dropped the Warp-vs-EGL RMSE from 34.4 to 22.2 on spatial task 0.

### Haze blending (render_kernel_patch.py)

MuJoCo applies atmospheric haze: distant geometry blends toward the background color based on `vis.map.haze`, `fogstart`, and `fogend`. The Warp renderer had no haze. The patch adds haze blending after shading, before the pixel write.

For LIBERO scenes this has no visible effect. The fog starts at `fogstart * extent = 3.0 * 10.61 = 31.83` units from the camera, but objects sit at 1 to 2 units. The haze factor is 0 for all visible geometry. The improvement comes from kernel recompilation: a different code path produces different floating-point intermediate values.

### RenderContext fields (render_kernel_patch.py)

The `RenderContext` dataclass in `types.py` had no fields for haze parameters. The patch adds `haze_amount`, `fogstart`, `fogend`, and `background_color_float`. The `create_render_context()` function in `io.py` populates them from `mjm.vis.map.haze`, `mjm.vis.map.fogstart * mjm.stat.extent`, `mjm.vis.map.fogend * mjm.stat.extent`, and the background color.

### Vertical flip (WarpRenderer)

OpenGL renders with a bottom-left origin. Robosuite & MuJoCo CPU output top-left. `img.flip(dims=[1])` corrects the vertical axis.

### Brightness boost (WarpRenderer)

The Warp ray tracer produces images at about 85% of the CPU renderer's brightness. The ratio is 0.856 at image center and 0.929 at edges, so it is not a uniform scale. The difference is likely a missing tone mapping or exposure step in the ray tracer. A 1.15x multiplier on the output RGB closes the gap. This single fix moved average success from 30% to 42.5% across 4 seeds.

The `--brightness` flag controls the multiplier. Values of 1.10, 1.15, and 1.20 all produced 50% success on seed 42. The default is 1.15.

### What does not work

Replacing cube map textures (type=1) with flat average colors destroys rendering. Warp eval drops to 0%. The Warp renderer samples cube maps as 2D vertical strips, which is wrong, but the wrong result still carries enough texture information for the policy. Flat colors carry none.

Removing transparent geoms (the EEF target spheres & boxes at alpha 0.5 and 0.8) hurts success rate. Warp renders them as opaque because the ray tracer has no alpha blending. Keeping them visible, even as opaque shapes, matches the training data better than removing them.

## Results

### Spatial task 0, 50-epoch BC checkpoint

| Eval method | Success rate |
|---|---|
| CPU eval (eval_bc.py) | 50% (5/10) |
| Warp eval (eval_warp_only.py, 4 seeds) | 42.5% avg (50%, 40%, 30%, 50%) |
| Warp eval, no rendering fixes | 0% |

10 envs per eval. The policy samples actions stochastically, so success rates vary 30 to 50 percentage points across seeds with this sample size.

### Spatial suite, CPU eval, 20 episodes per task

| Task | Success rate |
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
| Average | 51.5% |

### Object suite, CPU eval

| Task | Success rate |
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
| Average | 56.5% |

Goal, scene10, and scene90 CPU eval results are pending training.

## BC training & evaluation

Training uses LIBERO's BC transformer: a ResNet image encoder, a temporal transformer, and a GMM policy head. It reads demo data from HDF5 files. Training runs on PyTorch & GPU. No robosuite or Warp needed.

```bash
python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 --save ckpt.pth

# With domain randomization (brightness, contrast, noise, blur)
python scripts/train_bc.py --suite spatial --task-id 0 --epochs 50 --save ckpt.pth --augment
```

### Download demo data

```python
from libero.libero.utils.download_utils import libero_dataset_download
libero_dataset_download(datasets="libero_spatial", use_huggingface=True)
# Also: libero_object, libero_goal, libero_10, libero_90
```

## Patches

The repo patches three external packages at import time. Each patch exists because the upstream package has a bug or missing feature that blocks LIBERO tasks on GPU.

### warp_gpu_patch.py

JAX on ROCm reports its platform as `rocm` (lowercase). Warp registers FFI targets for `ROCM` (uppercase). The XLA compiler looks up `rocm` and finds nothing. The patch calls `jax.ffi.register_ffi_target` for both casings.

MJX's io module checks for CUDA GPUs with `has_cuda_gpu_device`. On ROCm, JAX reports devices under the `gpu` backend, but MJX's check looks for `cuda` by name. The patch rewrites the check to use `jax.devices('gpu')`.

HIP does not support the `cubql` BVH constructor. The patch forces `lbvh` for mesh & heightfield BVH builds.

Some mujoco_warp versions ship `GraphMode` as an int instead of an enum. MJX expects `GraphMode.WARP`. The patch injects a compatible enum.

### robosuite_patch.py

Robosuite 1.5.1's `robot_base_factory` returns a string for unknown base names like `NullBase`, which LIBERO uses for floor & kitchen scenes. Downstream code calls the return value as a class constructor and crashes with `TypeError: 'str' object is not callable`. The patch falls back to `NullMount` for unknown names.

This is required for object, goal, scene10, and scene90. Spatial works without it because it uses `RethinkMount`.

### texture_patch.py

Warp 1.13.0's `get_type_code` does not recognize `wp.array[wp.Texture2D]` when hashing kernel arguments. It raises `TypeError: Unrecognized type`. The patch adds type codes `"tex2d"` and `"atex2d"` for `Texture2D` and arrays of `Texture2D`.

### render_kernel_patch.py

Patches three files in the installed `mujoco_warp` package on disk: `render.py`, `types.py`, and `io.py`. See [Rendering fixes](#rendering-fixes) above. Must run before any `import mujoco_warp` statement, because importing `libero_mjx` triggers `mujoco.mjx.warp`, which caches `mujoco_warp._src.types` in `sys.modules`. Once cached, the patch cannot take effect.

## Documentation

- [docs/architecture.md](docs/architecture.md): environment, controller, and renderer design
- [docs/rendering.md](docs/rendering.md): CPU vs Warp rendering differences and fixes
- [docs/physics.md](docs/physics.md): Warp physics verification & known differences
- [docs/api.md](docs/api.md): public API reference for `LiberoEnv`, `WarpRenderer`, and `OscController`

## Citation

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