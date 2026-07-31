#!/usr/bin/env bash
# Wrapper to run any script inside the libero-mjx Docker container on the
# local RDNA 4 GPU (gfx1201 / RX 9070 XT). Passes through all args.
#
# Usage:
#   ./scripts/docker_run.sh python tests/validate_warp_render.py
#   ./scripts/docker_run.sh python scripts/eval_warp_only.py --suite spatial --task-id 0
#   ./scripts/docker_run.sh bash

set -euo pipefail

IMAGE="${LIBERO_MJX_IMAGE:-libero-mjx:patched}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LIBERO_BASIL="${LIBERO_BASIL_PATH:-$HOME/workspace/libero_basil}"

docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add=video \
  --shm-size 8g \
  -v "$REPO:/workspace/libero-mjx" \
  -v "$LIBERO_BASIL:/workspace/libero_basil" \
  -v "libero-mjx-warp-cache:/root/.cache/warp" \
  -e JAX_PLATFORMS=rocm \
  -e C_INCLUDE_PATH=/opt/rocm/lib/llvm/lib/clang/23/include \
  -e CPLUS_INCLUDE_PATH=/opt/rocm/lib/llvm/lib/clang/23/include \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 \
  -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
  -e REPO_PATH=/workspace/libero-mjx \
  -e LIBERO_BASIL_PATH=/workspace/libero_basil \
  -w /workspace/libero-mjx \
  "$IMAGE" \
  "$@"