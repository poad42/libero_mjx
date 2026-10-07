# Reproducibility

This page records what the container is built from, why each source was chosen,
and how to reproduce a build byte-for-byte as far as the upstream indexes allow.

## What was wrong before

The earlier image was not reproducible and was pinned to AMD in a way that
could not be swapped:

- ROCm came from a **floating TheRock nightly tarball** fetched from
  `raw.githubusercontent.com/ROCm/TheRock/main/...` and pinned only to a date
  string (`7.14.0a20260605`). Nightly tarballs are pruned and the install script
  tracks `main`.
- `warp` was a **private fork** build that could not be published.
- `torch` and `torchvision` were **built locally from source** (`file:///pytorch/...`),
  so no one else could obtain the same wheels.
- The working image was produced by an interactive `docker commit` (a
  `sleep infinity` layer), not by the checked-in Dockerfile. The Dockerfile
  built a *different* image (`libero-mjx`) than the one the run script used
  (`libero-mjx:patched`).
- The NVIDIA path existed only as comments.

## What it is now

Everything comes from a public, versioned source.

| Component | Source | Pin |
|---|---|---|
| Base image | `ghcr.io/rocm/therock_build_manylinux_x86_64:latest` | public TheRock manylinux image |
| ROCm | pip `rocm[libraries,devel,device-gfx1201]` | `https://stable.repo.amd.com/rocm/whl-next/` (10.1.0a…) |
| PyTorch | `torch[device-gfx1201]`, `torchvision[device-gfx1201]` | `2.13.0`, `0.28.0` |
| JAX | `jax-rocm10-plugin`, `jax-rocm10-pjrt` + PyPI `jax` | `0.11.1` |
| MuJoCo | PyPI `mujoco`, `mujoco-mjx`, `mujoco-warp` | `3.13.0` |
| Warp | `github.com/AMD-Ecosystem/warp` @ `amd-integration-halo` | `1.17.0+rocm.0` + `docker/patch_warp_nested_kernel.py` |
| LIBERO | `github.com/Lifelong-Robot-Learning/LIBERO` | commit `8f1084e3…` + `docker/patches/libero_robosuite15.patch` |

The ROCm runtime, the BLAS kernels, torch and JAX all come from the same
aggregate index, so they are versioned together. See
[ROCm/TheRock `RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
for the index layout and the supported-version matrix.

### Why the TheRock manylinux base

It ships no system ROCm, so the only ROCm is the pip SDK. That removes two
fragile workarounds the old Ubuntu + tarball chain needed: there is no
`/opt/rocm` clang for a host C++ compile to pick up, and no `C_INCLUDE_PATH`
export pointing at one. The remaining include-path accommodation is scoped to
run time (warp's hiprtc invocation), not the host build.

### Why the public warp branch

`AMD-Ecosystem/warp` is the public AMD port. The `amd-integration-halo` branch
selects wave32 for RDNA and wave64 for CDNA via `__GFX9__`, so one branch covers
`gfx1201` (RX 9070 XT, RDNA 4) and the CDNA parts. The older private fork existed
only because the previous public branch was CDNA-only; it is no longer needed.

> CDNA note: if you target a `gfx9xx` part and `amd-integration-halo` misbehaves,
> build with `WARP_BRANCH=amd-integration`. `scripts/docker_build.sh` selects
> that automatically for `gfx9*` targets.

## The LIBERO compatibility patch

The LIBERO port targets robosuite 1.5.x, torch 2.x and gymnasium. Upstream
LIBERO is written against robosuite 1.4.0 (`SingleArmEnv`, `default_mount`, a
gym `seed()`), so a small set of edits is required. Rather than depend on an
unpublished fork, the edits are checked in as
`docker/patches/libero_robosuite15.patch` (18 files, ~500 lines) and applied to
the pinned upstream commit during the build:

```
git clone LIBERO && git checkout 8f1084e3… && git apply docker/patches/libero_robosuite15.patch
```

The patch covers: `SingleArmEnv` -> `ManipulationEnv`, `default_mount` ->
`default_base` with a `NullBase` fallback, the composite `body_parts`
controller config, the `arms` class attribute, `torch.load(weights_only=False)`,
and `gym` -> `gymnasium`. It is the same tree the checked-in task XMLs and
`libero_mjx/robosuite_patch.py` were developed against.

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
| `WARP_BRANCH` | `amd-integration-halo` | warp branch |
| `WARP_VERSION` | `1.17.0+rocm.0` | warp version label |
| `LIBERO_COMMIT` | `8f1084e3…` | LIBERO checkout (patch applied on top) |

Build logs are written to `docker/logs/`.

## Rendering notes

The render megakernel patch must be applied before `mujoco_warp` is imported.
`libero_mjx/__init__.py` does this; see
[rendering.md](rendering.md#patch-ordering-the-nested-render-megakernel-and-stale-source-lines).
The image also applies `docker/patch_warp_nested_kernel.py` to the warp source,
which makes warp's source-extraction fallback validate the function it found.

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

- The TheRock index and `AMD-Ecosystem/warp` branch are moving targets. Pin the
  index URL to a dated channel and `WARP_BRANCH` to a commit for a frozen build.
- The base image tag is `:latest`. For a frozen build, replace it with a digest:
  `ghcr.io/rocm/therock_build_manylinux_x86_64@sha256:<digest>`.
- The PyPI dependency versions of the LIBERO stack (robosuite, robomimic, hydra,
  transformers, …) are only major-pinned. Add `==` pins in the Dockerfile for a
  fully frozen environment.
