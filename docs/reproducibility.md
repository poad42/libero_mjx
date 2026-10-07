# Reproducibility

This page records what the container is built from, why each source was chosen,
and how to reproduce a build.

## Pin table

Everything comes from a public, versioned source.

| Component | Source | Pin |
|---|---|---|
| Base image | `ghcr.io/rocm/therock_build_manylinux_x86_64:latest` | public TheRock manylinux image |
| ROCm | pip `rocm[libraries,devel,device-gfx1201]` | `https://stable.repo.amd.com/rocm/whl-next/` (10.1.0a…) |
| PyTorch | `torch[device-gfx1201]`, `torchvision[device-gfx1201]` | `2.13.0`, `0.28.0` |
| JAX | `jax-rocm10-plugin`, `jax-rocm10-pjrt` + PyPI `jax` | `0.11.1` |
| MuJoCo | PyPI `mujoco`, `mujoco-mjx`, `mujoco-warp` | `3.13.0` |
| Warp | `github.com/cu-basil/warp` @ `amd-integration-halo` | `3fee69ec…`, `1.17.0+rocm.0` |
| LIBERO | `github.com/cu-basil/LIBERO` | `f6266995…` |

The ROCm runtime, the BLAS kernels, torch and JAX all come from the same
aggregate index, so they are versioned together. See
[ROCm/TheRock `RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
for the index layout and the supported-version matrix.

### Base image

`ghcr.io/rocm/therock_build_manylinux_x86_64` ships no system ROCm, so the only
ROCm is the pip SDK. There is no `/opt/rocm` clang for a host C++ compile to pick
up, and no `C_INCLUDE_PATH` pointing at one, so warp's host build and its hiprtc
kernel builds use a single compiler. The one include-path accommodation that
remains is scoped to run time; see the `C_INCLUDE_PATH` block in the `Dockerfile`.

### Warp

`cu-basil/warp` is a public fork of `AMD-Ecosystem/warp`. The
`amd-integration-halo` branch selects wave32 for RDNA and wave64 for CDNA via
`__GFX9__`, so one branch covers `gfx1201` (RX 9070 XT, RDNA 4) and the CDNA
parts. The pinned commit carries the LLVM 23 `warp-clang` build fixes and the
nested-definition source-extraction fix, so the build applies no patch on top.

> CDNA note: if you target a `gfx9xx` part and `amd-integration-halo` misbehaves,
> build with `WARP_BRANCH=amd-integration`. `scripts/docker_build.sh` selects
> that automatically for `gfx9*` targets.

### LIBERO

The LIBERO port targets robosuite 1.5.x, torch 2.x and gymnasium. Upstream
`Lifelong-Robot-Learning/LIBERO` is written against robosuite 1.4.0, so the build
uses the public fork `cu-basil/LIBERO`, which carries the compatibility edits
(`SingleArmEnv` -> `ManipulationEnv`, composite controller `body_parts`,
`weights_only=False`, gymnasium). It is byte-identical to the tree the checked-in
task XMLs and `libero_mjx/robosuite_patch.py` were developed against.
`scripts/setup_assets.sh` clones the same repo and commit on a host.

## Build

```bash
./scripts/docker_build.sh                          # AMD, gfx1201 (default)
GFX_TARGET=gfx942 ./scripts/docker_build.sh        # CDNA
ACCELERATOR=cuda ./scripts/docker_build.sh         # NVIDIA
```

Every version is a build arg. The full list is in the `Dockerfile` header; the
common ones are:

| Arg | Default | Meaning |
|---|---|---|
| `BASE_IMAGE` | TheRock manylinux | base image (swap for CUDA) |
| `ACCELERATOR` | `rocm` | `rocm` or `cuda` |
| `GFX_TARGET` | `gfx1201` | AMD GPU target |
| `ROCM_INDEX` | stable TheRock index | pip index for ROCm/torch/jax |
| `TORCH_VERSION` | `2.13.0` | torch |
| `TORCHVISION_VERSION` | `0.28.0` | torchvision |
| `JAX_VERSION` | `0.11.1` | jax / plugin / pjrt |
| `MUJOCO_VERSION` | `3.13.0` | mujoco + mjx + mujoco-warp |
| `WARP_REPO` | `cu-basil/warp` | warp source |
| `WARP_BRANCH` | `amd-integration-halo` | warp branch |
| `WARP_COMMIT` | `3fee69ec…` | warp commit |
| `WARP_VERSION` | `1.17.0+rocm.0` | warp version label |
| `LIBERO_REPO` | `cu-basil/LIBERO` | LIBERO source |
| `LIBERO_COMMIT` | `f6266995…` | LIBERO commit |

Build logs are written to `docker/logs/`.

## Rendering notes

The render-kernel patch rewrites `mujoco_warp` source files on disk, so it must
run before `mujoco_warp` is imported. `libero_mjx/__init__.py` does this; see
[rendering.md](rendering.md#patch-ordering).

CPU (robosuite) rendering uses `MUJOCO_GL=osmesa`: the AlmaLinux base ships
Mesa 23.1, which predates gfx1201 EGL device support. The GPU Warp renderer does
not use `MUJOCO_GL`.

## Verify

At build time the image runs `docker/verify_stack.py --build`: imports, versions
and asset roots. At run time it runs the same script without `--build`, which
adds the GPU checks and a reset+step smoke test:

```bash
./scripts/docker_run.sh python /opt/verify_stack.py
```

Expected output ends with `ALL CHECKS PASSED`.

## What is still not bit-exact

- The TheRock index and `cu-basil/warp` branch are moving targets. Pin the
  index URL to a dated channel; `WARP_COMMIT` is already pinned to a commit.
- The base image tag is `:latest`. For a frozen build, replace it with a digest:
  `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:<digest>`.
- The PyPI dependency versions of the LIBERO stack (robosuite, robomimic, hydra,
  transformers, …) are only major-pinned. Add `==` pins in the Dockerfile for a
  fully frozen environment.
