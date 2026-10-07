#!/usr/bin/env bash
# Build the reproducible libero-mjx image.
#
#   ./scripts/docker_build.sh                       # AMD ROCm, gfx1201 (default)
#   GFX_TARGET=gfx942 ./scripts/docker_build.sh     # another AMD target
#   ACCELERATOR=cuda ./scripts/docker_build.sh      # NVIDIA (CUDA base image)
#
# Every version is a build arg with a pinned default; see the Dockerfile header.
# Logs go to docker/logs/.
set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

ACCELERATOR="${ACCELERATOR:-rocm}"
GFX_TARGET="${GFX_TARGET:-gfx1201}"

# Warp's public AMD port has two branches: `amd-integration-halo` is the RDNA
# (wave32, gfx10xx/gfx11xx/gfx12xx) successor; CDNA (wave64, gfx9xx) still needs
# the older `amd-integration` branch. Pick a default from the target, but let
# WARP_BRANCH / WARP_VERSION override.
case "$GFX_TARGET" in
  gfx9*)  WARP_BRANCH_DEFAULT=amd-integration;      WARP_VERSION_DEFAULT=1.13.0+rocm.0 ;;
  *)      WARP_BRANCH_DEFAULT=amd-integration-halo; WARP_VERSION_DEFAULT=1.17.0+rocm.0 ;;
esac
WARP_BRANCH="${WARP_BRANCH:-$WARP_BRANCH_DEFAULT}"
WARP_VERSION="${WARP_VERSION:-$WARP_VERSION_DEFAULT}"

if [ "$ACCELERATOR" = "rocm" ]; then
  IMAGE="${LIBERO_MJX_IMAGE:-libero-mjx:${GFX_TARGET}}"
  BASE_IMAGE="${BASE_IMAGE:-ghcr.io/rocm/therock_build_manylinux_x86_64:latest}"
else
  IMAGE="${LIBERO_MJX_IMAGE:-libero-mjx:cuda}"
  BASE_IMAGE="${BASE_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}"
fi

mkdir -p docker/logs
LOG="docker/logs/build-$(date +%Y%m%d-%H%M%S).log"

echo "building ${IMAGE} (accelerator=${ACCELERATOR}, gfx=${GFX_TARGET})"
echo "log: ${LOG}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "ACCELERATOR=${ACCELERATOR}" \
  --build-arg "GFX_TARGET=${GFX_TARGET}" \
  --build-arg "ROCM_INDEX=${ROCM_INDEX:-https://stable.repo.amd.com/rocm/whl-next/}" \
  --build-arg "TORCH_VERSION=${TORCH_VERSION:-2.13.0}" \
  --build-arg "TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.28.0}" \
  --build-arg "JAX_VERSION=${JAX_VERSION:-0.11.1}" \
  --build-arg "MUJOCO_VERSION=${MUJOCO_VERSION:-3.13.0}" \
  --build-arg "WARP_REPO=${WARP_REPO:-https://github.com/cu-basil/warp.git}" \
  --build-arg "WARP_BRANCH=${WARP_BRANCH}" \
  --build-arg "WARP_COMMIT=${WARP_COMMIT:-3fee69ec6329f8d73087e5e91f2f167fb3599a06}" \
  --build-arg "WARP_VERSION=${WARP_VERSION}" \
  --build-arg "LIBERO_REPO=${LIBERO_REPO:-https://github.com/cu-basil/LIBERO.git}" \
  --build-arg "LIBERO_COMMIT=${LIBERO_COMMIT:-f626699538dbc0e58509a93e469e52e9238c2dc6}" \
  -t "${IMAGE}" \
  -f Dockerfile . 2>&1 | tee "${LOG}"

echo "done: ${IMAGE}"
