# Dockerfile for LIBERO-MJX: GPU-parallel LIBERO tasks on MuJoCo Warp
# Supports AMD ROCm GPUs (gfx1201/RDNA4). For NVIDIA, see comments below.
#
# Build:  docker build -t libero-mjx .
# Run:    docker run --rm --device=/dev/kfd --device=/dev/dri --group-add=video \
#           --shm-size 8g -v $(pwd):/workspace/libero-mjx \
#           libero-mjx python tests/test_all_suites.py

FROM ubuntu:24.04

ARG ROCM_VERSION=7.14.0a20260605
ARG AMDGPU_FAMILY=gfx120X-all

# ---- System deps ----
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    software-properties-common curl wget git build-essential cmake ninja-build \
    python3 python3-pip python3-venv python3-dev \
    && add-apt-repository -y ppa:kisak/kisak-mesa \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libegl1 libosmesa6 libgl1-mesa-dri ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- ROCm ----
# For NVIDIA: replace this block with NVIDIA CUDA toolkit
#   RUN apt-get install -y nvidia-cuda-toolkit
RUN curl -sSL https://raw.githubusercontent.com/ROCm/TheRock/main/dockerfiles/install_rocm_tarball.sh \
    | bash -s -- ${ROCM_VERSION} ${AMDGPU_FAMILY} nightlies

ENV ROCM_PATH=/opt/rocm
ENV LD_LIBRARY_PATH=/opt/rocm/lib
ENV PATH=/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Include clang headers for warp kernel compilation
ENV C_INCLUDE_PATH=/opt/rocm/lib/llvm/lib/clang/23/include
ENV CPLUS_INCLUDE_PATH=/opt/rocm/lib/llvm/lib/clang/23/include

# ---- Python venv ----
RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/lib/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ---- JAX (ROCm) ----
# IMPORTANT: install jax+jaxlib+plugin together so they don't get overridden later.
# For NVIDIA: pip install jax[cuda12]
RUN pip install --extra-index-url=https://rocm.nightlies.amd.com/v2/gfx120X-all \
    jax==0.10.2 jaxlib==0.10.2 jax-rocm7-plugin==0.10.2 jax-rocm7-pjrt==0.10.2

# ---- PyTorch (ROCm) ----
# IMPORTANT: install torch BEFORE other deps so they don't pull in a CPU/CUDA build.
# For NVIDIA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.0

# ---- MuJoCo + Warp + JAX ecosystem ----
# These are pure Python / have ROCm-compatible builds, safe to pip install.
RUN pip install \
    mujoco==3.10.0 mujoco-mjx==3.10.0 \
    warp-lang \
    flax ml_collections etils \
    numpy scipy absl-py tqdm lxml pyopengl \
    brax mediapy tensorboardX

# ---- LIBERO deps ----
# robosuite/robomimic are CPU-only, no GPU conflict.
RUN pip install \
    robosuite robomimic \
    hydra-core omegaconf easydict \
    transformers tokenizers sentencepiece \
    huggingface_hub pillow imageio imageio-ffmpeg egl_probe

ENV MUJOCO_GL=egl
# mujoco_warp is bundled inside mujoco-mjx as third_party
ENV PYTHONPATH=/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party

WORKDIR /workspace/libero-mjx