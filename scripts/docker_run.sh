#!/usr/bin/env bash
# Run any command inside the libero-mjx container on the local GPU.
#
#   ./scripts/docker_run.sh python tests/test_all_suites.py
#   ./scripts/docker_run.sh python scripts/eval_warp_only.py --suite spatial --task-id 0
#   ./scripts/docker_run.sh bash
#
# Mounts (all optional, all overridable by env):
#   REPO_PATH        repo root                 -> /workspace/libero-mjx
#   LIBERO_BASIL_PATH  host LIBERO checkout    -> /workspace/libero_basil
#                      (if unset, the copy baked into the image is used)
#   LIBERO_DATASETS  host demo datasets        -> /workspace/libero_basil/libero/datasets
#   ROBOSUITE_ASSETS host robosuite assets     -> /opt/robosuite_assets
#
# The image is vendor-agnostic; set LIBERO_MJX_IMAGE to pick a tag.
set -euo pipefail

IMAGE="${LIBERO_MJX_IMAGE:-libero-mjx:gfx1201}"
REPO="${REPO_PATH:-$(cd "$(dirname "$0")/.." && pwd)}"

ACCEL_ARGS=()
if [ -e /dev/kfd ]; then
  ACCEL_ARGS+=(--device=/dev/kfd --device=/dev/dri --group-add=video)
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  ACCEL_ARGS+=(--gpus all)
fi

MOUNTS=(-v "$REPO:/workspace/libero-mjx")
if [ -n "${LIBERO_BASIL_PATH:-}" ] && [ -d "${LIBERO_BASIL_PATH}" ]; then
  MOUNTS+=(-v "${LIBERO_BASIL_PATH}:/workspace/libero_basil")
fi
if [ -n "${LIBERO_DATASETS:-}" ] && [ -d "${LIBERO_DATASETS}" ]; then
  MOUNTS+=(-v "${LIBERO_DATASETS}:/workspace/libero_basil/libero/datasets")
fi
if [ -n "${ROBOSUITE_ASSETS:-}" ] && [ -d "${ROBOSUITE_ASSETS}" ]; then
  MOUNTS+=(-v "${ROBOSUITE_ASSETS}:/opt/robosuite_assets")
fi

docker run --rm \
  "${ACCEL_ARGS[@]}" \
  --shm-size 8g \
  "${MOUNTS[@]}" \
  -v "libero-mjx-warp-cache:/opt/warp_cache" \
  -e HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}" \
  -e MUJOCO_GL="${MUJOCO_GL:-osmesa}" \
  -e JAX_PLATFORMS="${JAX_PLATFORMS:-rocm}" \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.15}" \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e HSA_ENABLE_COREDUMP=0 \
  -e REPO_PATH=/workspace/libero-mjx \
  -e LIBERO_BASIL_PATH=/workspace/libero_basil \
  -w /workspace/libero-mjx \
  "$IMAGE" \
  "$@"
