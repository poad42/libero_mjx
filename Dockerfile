# Reproducible, vendor-parameterised image for libero-mjx.
#
# The default (and tested) path is AMD ROCm on a public TheRock base image.
# Nothing here is pinned to a private fork or a floating nightly tarball:
#
#   base image   ghcr.io/rocm/therock_build_manylinux_x86_64   (public)
#   ROCm         pip `rocm[libraries,devel,device-<gfx>]`      (public index)
#   torch        TheRock multi-arch wheels with device extras  (public index)
#   JAX          TheRock ROCm wheels                           (public index)
#   warp         AMD-Ecosystem/warp @ amd-integration-halo     (public branch)
#   mujoco       PyPI, pinned
#
# NVIDIA is selected with ACCELERATOR=cuda and a CUDA base image; the ROCm-only
# blocks are guarded, so one file covers both vendors.
#
# Build (AMD, gfx1201):
#   docker build -t libero-mjx:gfx1201 .
#   ./scripts/docker_build.sh                 # wrapper, writes a log
#
# Build (NVIDIA):
#   docker build --build-arg ACCELERATOR=cuda \
#                --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 \
#                -t libero-mjx:cuda .
#
# Run:
#   ./scripts/docker_run.sh python tests/test_all_suites.py

ARG BASE_IMAGE=ghcr.io/rocm/therock_build_manylinux_x86_64:latest
FROM ${BASE_IMAGE}

ARG ACCELERATOR=rocm
ARG GFX_TARGET=gfx1201
ARG ROCM_INDEX=https://stable.repo.amd.com/rocm/whl-next/
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
ARG JAX_VERSION=0.11.1
ARG MUJOCO_VERSION=3.13.0
# AMD's public warp port has two branches. `amd-integration-halo` is the RDNA
# (wave32) successor used for gfx10xx/gfx11xx/gfx12xx; CDNA (wave64, gfx9xx)
# still needs the older `amd-integration` branch. scripts/docker_build.sh picks
# the branch from GFX_TARGET and passes both as build args.
ARG WARP_REPO=https://github.com/AMD-Ecosystem/warp.git
ARG WARP_BRANCH=amd-integration-halo
ARG WARP_VERSION=1.17.0+rocm.0
# LIBERO master is pinned; the repository also needs a small set of robosuite
# 1.5.x / torch 2.x compatibility patches that are not in upstream. They are
# applied from docker/patches/ after checkout.
ARG LIBERO_REPO=https://github.com/Lifelong-Robot-Learning/LIBERO.git
ARG LIBERO_COMMIT=8f1084e3132a39270c3a13ebe37270a43ece2a01
ARG PIP_INDEX=https://pypi.org/simple

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV DEBIAN_FRONTEND=noninteractive

# --------------------------------------------------------------------------
# System runtime libraries
# --------------------------------------------------------------------------
# MuJoCo needs EGL/GL at runtime (MUJOCO_GL=egl). Warp needs libstdc++ and an
# OpenMP runtime for its host build. The manylinux base is deliberately bare.
RUN if [ -f /etc/almalinux-release ] || [ -f /etc/redhat-release ]; then \
      yum install -y epel-release >/dev/null 2>&1 || true; \
      yum install -y \
        mesa-libEGL mesa-libGL mesa-libgbm mesa-dri-drivers mesa-libOSMesa \
        libglvnd-glx libglvnd-egl \
        libstdc++ libgomp which file tar gzip xz \
        && yum clean all && rm -rf /var/cache/yum; \
    elif [ -f /etc/debian_version ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libglvnd0 libglu1-mesa libosmesa6 libstdc++6 \
        libgomp1 which file tar gzip xz-utils ca-certificates \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# --------------------------------------------------------------------------
# Accelerator runtime
# --------------------------------------------------------------------------
# ROCm comes from the pip SDK on the multi-arch index, so the image carries no
# system /opt/rocm and there is no clang in the host include path to shadow
# glibc. The device wheel carries the per-arch BLAS kernels that jax and torch
# need; `rocm-sdk init` links them into the devel tree where hipBLASLt looks.
RUN if [ "${ACCELERATOR}" = "rocm" ]; then \
      pip install --upgrade pip setuptools wheel \
      && pip install --pre --index-url "${ROCM_INDEX}" \
           "rocm[libraries,devel,device-${GFX_TARGET}]" \
      && rocm-sdk init \
      && rocm-sdk version \
      && test -n "$(find / -name "TensileLibrary_lazy_${GFX_TARGET}.dat" -print -quit)" \
      && echo "rocm ${GFX_TARGET} BLAS kernels present"; \
    else \
      pip install --upgrade pip setuptools wheel; \
    fi

# The pip SDK lives under a deterministic path; bake it and assert it so a base
# image bump fails loudly instead of silently misconfiguring hipcc.
ARG ROCM_SDK_ROOT=/opt/_internal/cpython-3.12.10/lib/python3.12/site-packages/_rocm_sdk_devel
RUN if [ "${ACCELERATOR}" = "rocm" ]; then \
      test "$(rocm-sdk path --root)" = "${ROCM_SDK_ROOT}" \
      && test -x "${ROCM_SDK_ROOT}/bin/hipcc" \
      && test -d "${ROCM_SDK_ROOT}/lib/llvm" \
      && echo "rocm sdk verified at ${ROCM_SDK_ROOT}"; \
    fi

ENV ROCM_PATH=${ROCM_SDK_ROOT}
ENV HIP_PATH=${ROCM_SDK_ROOT}
ENV HIP_LLVM_PATH=${ROCM_SDK_ROOT}/lib/llvm

# --------------------------------------------------------------------------
# MuJoCo + MJX + mujoco-warp
# --------------------------------------------------------------------------
# mujoco-mjx 3.13 no longer vendors mujoco_warp; --no-deps protects the ROCm
# warp build below from being replaced by a PyPI wheel.
RUN pip install --no-cache-dir --index-url "${PIP_INDEX}" \
      "mujoco==${MUJOCO_VERSION}" "mujoco-mjx==${MUJOCO_VERSION}" \
 && pip install --no-cache-dir --no-deps --index-url "${PIP_INDEX}" \
      "mujoco-warp==${MUJOCO_VERSION}"

# --------------------------------------------------------------------------
# PyTorch (BC policy, renderer tensors)
# --------------------------------------------------------------------------
# On ROCm the device extra pulls the matching `amd-torch-device-<gfx>` wheel and
# its `rocm-sdk-device-*` dependency. On CUDA the plain PyPI wheels are used.
RUN if [ "${ACCELERATOR}" = "rocm" ]; then \
      pip install --no-cache-dir --index-url "${ROCM_INDEX}" \
        "torch[device-${GFX_TARGET}]==${TORCH_VERSION}" \
        "torchvision[device-${GFX_TARGET}]==${TORCHVISION_VERSION}"; \
    else \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"; \
    fi

# --------------------------------------------------------------------------
# Warp, built from the public AMD branch (ROCm) or PyPI wheel (CUDA)
# --------------------------------------------------------------------------
COPY docker/patch_warp_llvm23.py /tmp/patch_warp_llvm23.py
COPY docker/patch_warp_nested_kernel.py /tmp/patch_warp_nested_kernel.py
COPY docker/patches/ /tmp/patches/
RUN if [ "${ACCELERATOR}" = "rocm" ]; then \
      pip install --no-cache-dir numpy setuptools packaging wheel \
      && git clone --depth 1 --branch "${WARP_BRANCH}" "${WARP_REPO}" /opt/warp_src \
      && printf '%s\n' "${WARP_VERSION}" > /opt/warp_src/VERSION.md \
      && sed -i "s/^version: str = .*/version: str = \"${WARP_VERSION}\"/" /opt/warp_src/warp/config.py \
      && python /tmp/patch_warp_llvm23.py /opt/warp_src \
      && python /tmp/patch_warp_nested_kernel.py /opt/warp_src \
      && cd /opt/warp_src \
      && ( HIP_ARCH=${GFX_TARGET} WARP_GLIBCXX_USE_CXX11_ABI=1 ./build_amd.sh --llvm-path="${HIP_LLVM_PATH}" \
           || HIP_ARCH=${GFX_TARGET} ./build_amd.sh --no-standalone ) \
      && pip install -e /opt/warp_src \
      && python -c "import warp; warp.init(); print('warp', warp.__version__); print('devices', [str(d) for d in warp.get_devices()])"; \
    else \
      pip install --no-cache-dir "warp-lang==${WARP_VERSION%%+*}" \
      && python -c "import warp; print('warp', warp.__version__)"; \
    fi

# MJX 3.13 imports warp as mujoco.mjx.third_party.warp, so the built tree has to
# be visible there too. When mjx does not vendor warp, this is a no-op.
RUN python -c "import os,pathlib,shutil,mujoco; t=pathlib.Path(mujoco.__file__).parent/'mjx'/'third_party'/'warp'; (t.unlink() if (t.is_symlink() or t.is_file()) else (shutil.rmtree(t) if t.is_dir() else None)); os.symlink('/opt/warp_src/warp', t) if os.path.isdir('/opt/warp_src/warp') else None; print('mjx warp link:', t, os.path.islink(t) and os.readlink(t) or 'none')"

# --------------------------------------------------------------------------
# LIBERO + robosuite assets
# --------------------------------------------------------------------------
# The task XMLs reference assets by absolute path. We install LIBERO at the
# canonical /workspace/libero_basil so the checked-in XMLs resolve unchanged,
# and expose the robosuite assets at /opt/robosuite_assets. `LIBERO_ASSETS_ROOT`
# and `ROBOSUITE_ASSETS_ROOT` let a different layout be used without rewriting
# the XMLs (see libero_mjx/envs/base.py).
RUN git clone "${LIBERO_REPO}" /workspace/libero_basil \
 && git -C /workspace/libero_basil checkout --quiet "${LIBERO_COMMIT}" \
 && git -C /workspace/libero_basil apply -p1 /tmp/patches/libero_robosuite15.patch \
 && test -d /workspace/libero_basil/libero/libero/assets \
 && echo "libero assets: $(du -sh /workspace/libero_basil/libero/libero/assets | cut -f1)"


# --------------------------------------------------------------------------
# LIBERO / BC Python dependencies
# --------------------------------------------------------------------------
# robomimic 0.4.0 is only published on GitHub (PyPI stops at 0.3.0), so pin the
# tag. robosuite and bddl match the versions the task XMLs and the robosuite
# patch were built against.
RUN pip install --no-cache-dir --index-url "${PIP_INDEX}" \
      "robosuite==1.5.1" "bddl==3.6.0" \
      "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0" \
      hydra-core omegaconf easydict cloudpickle einops gymnasium h5py \
      imageio imageio-ffmpeg matplotlib termcolor thop tqdm \
      "transformers" tokenizers sentencepiece huggingface_hub \
      pillow lxml pyopengl egl_probe wandb \
      absl-py etils flax ml_collections mediapy tensorboardX

# Robosuite assets live in the installed package; expose the canonical path the
# XMLs use. A symlink keeps a single copy.
RUN RS_ASSETS="$(python -c 'import robosuite, os; print(os.path.join(os.path.dirname(robosuite.__file__), "models", "assets"))')" \
 && ln -sfn "${RS_ASSETS}" /opt/robosuite_assets \
 && test -f /opt/robosuite_assets/base.xml \
 && echo "robosuite assets: ${RS_ASSETS}"

# --------------------------------------------------------------------------
# JAX, installed LAST
# --------------------------------------------------------------------------
# Every package above may pull a CPU jax/jaxlib from PyPI; installing the ROCm
# set last is what makes it stick. On ROCm 10 the plugin package is
# `jax-rocm10-plugin`; the version suffix `+rocm10.1.0` is ignored for equality.
RUN if [ "${ACCELERATOR}" = "rocm" ]; then \
      pip install --no-cache-dir --index-url "${ROCM_INDEX}" \
        "jax-rocm10-plugin==${JAX_VERSION}" "jax-rocm10-pjrt==${JAX_VERSION}" \
      && pip install --no-cache-dir --index-url "${PIP_INDEX}" "jax==${JAX_VERSION}"; \
    else \
      pip install --no-cache-dir --index-url https://storage.googleapis.com/jax-releases/cuda12 \
        "jax[cuda12]==${JAX_VERSION}"; \
    fi

RUN python -c "import jax_plugins.xla_rocm10 as p; print('rocm plugin:', p.__file__)" 2>/dev/null \
 || python -c "import jax; print('jax backend:', jax.default_backend())"

# --------------------------------------------------------------------------
# Runtime include accommodation for warp's hiprtc invocation
# --------------------------------------------------------------------------
# Warp's HIP kernel builds pass an include list that assumes a system ROCm at
# /opt/rocm and a Debian gcc path. Neither exists here, so `float.h` is not
# found and every kernel fails. Two runtime-only accommodations: /opt/rocm is
# symlinked to the pip SDK, and the clang resource + gcc headers go on the
# implicit include path through stable symlinks.
RUN if [ "${ACCELERATOR}" = "rocm" ]; then \
      ln -sfn "${ROCM_SDK_ROOT}" /opt/rocm \
      && test -f /opt/rocm/include/hip/hip_runtime.h \
      && ln -sfn "$(amdclang -print-resource-dir)/include" /usr/local/include/clang-resource \
      && ln -sfn "$(gcc -print-file-name=include)" /usr/local/include/gcc-include \
      && ls -l /opt/rocm /usr/local/include/clang-resource /usr/local/include/gcc-include; \
    fi

ENV C_INCLUDE_PATH=/usr/local/include/clang-resource:/usr/local/include/gcc-include
ENV CPLUS_INCLUDE_PATH=/usr/local/include/clang-resource:/usr/local/include/gcc-include

# --------------------------------------------------------------------------
# Runtime environment
# --------------------------------------------------------------------------
# LIBERO prompts on first import when ~/.libero/config.yaml is missing. Write it
# up front so `import libero` is non-interactive, and create the datasets dir it
# points at. Placed here rather than next to the clone so it does not invalidate
# the cached ROCm/torch/warp layers on a rebuild.
RUN mkdir -p /root/.libero /workspace/libero_basil/libero/datasets \
 && python -c "import yaml; root='/workspace/libero_basil/libero/libero'; yaml.dump({'benchmark_root': root, 'bddl_files': root + '/bddl_files', 'init_states': root + '/init_files', 'datasets': '/workspace/libero_basil/libero/datasets', 'assets': root + '/assets'}, open('/root/.libero/config.yaml', 'w'))" \
 && cat /root/.libero/config.yaml

# The AlmaLinux base ships Mesa 23.1, which predates gfx1201 support in the EGL
# device platform, so hardware EGL is unavailable. OSMesa (software) is the
# portable fallback for MuJoCo's CPU renderer; the GPU Warp renderer does not use
# MUJOCO_GL at all. Override to egl on a host with a newer Mesa.
ENV MUJOCO_GL=osmesa
ENV XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV WARP_CACHE_PATH=/opt/warp_cache
ENV LIBERO_BASIL_PATH=/workspace/libero_basil
# The LIBERO package is a checkout, not a pip install, so its parent directory
# has to be importable too.
ENV PYTHONPATH=/workspace/libero-mjx:/workspace/libero_basil
# HIP_VISIBLE_DEVICES=0 hides the unsupported iGPU (gfx1036) that ROCm otherwise
# enumerates first on some workstations. Override at run time for a cluster.
ENV HIP_VISIBLE_DEVICES=0

# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
COPY docker/verify_stack.py /opt/verify_stack.py
RUN python /opt/verify_stack.py --build

WORKDIR /workspace/libero-mjx
CMD ["bash"]
