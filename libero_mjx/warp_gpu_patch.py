"""Monkey-patch MJX to recognize ROCm GPUs as valid Warp devices.

Must be imported BEFORE warp or mujoco_warp, because the FFI
registration happens at import time.

Usage:
    import libero_mjx  # auto-patches FFI + GPU detection
    # Now warp/mujoco_warp will work on ROCm
"""

# Patch FFI registration FIRST, before any other imports
_ffi_patched = False

def _patch_ffi_registration():
    """Patch jax.ffi.register_ffi_target to also register for lowercase 'rocm'."""
    global _ffi_patched
    if _ffi_patched:
        return
    _ffi_patched = True

    import jax.ffi
    _orig_reg = jax.ffi.register_ffi_target
    def _patched_reg(name, fn, platform="cpu", api_version=1, **kwargs):
        _orig_reg(name, fn, platform, api_version, **kwargs)
        # On ROCm, the XLA backend reports platform as "ROCM" (canonical "rocm").
        # Warp registers for "CUDA" and "ROCM" (uppercase).
        # Also register for lowercase variants to match XLA's canonical name.
        if platform.upper() in ("ROCM", "CUDA"):
            for p in ["rocm", "ROCM", "gpu", "GPU"]:
                try:
                    _orig_reg(name, fn, p, api_version, **kwargs)
                except Exception:
                    pass
    jax.ffi.register_ffi_target = _patched_reg


# Patch immediately at import time
_patch_ffi_registration()

import jax
from jax._src import xla_bridge as _xb


def _has_gpu_device() -> bool:
    """Check for CUDA or ROCm GPU backend."""
    backends = _xb.backends()
    return 'cuda' in backends or 'rocm' in backends


def _is_gpu_device(device: jax.Device) -> bool:
    """Check if device is a CUDA or ROCm GPU."""
    if not _has_gpu_device():
        return False
    gpu_devices = jax.devices('gpu') if jax.devices('gpu') else []
    return device in gpu_devices


def _gpu_devices():
    """Return GPU devices (cuda or rocm)."""
    try:
        return jax.devices('gpu')
    except RuntimeError:
        return []


def patch_warp_to_gpu():
    """Patch MJX io module to recognize ROCm GPUs for Warp implementation."""
    # Must patch FFI registration before warp is imported
    _patch_ffi_registration()

    from mujoco.mjx._src import io

    # Patch the device detection functions
    io.has_cuda_gpu_device = _has_gpu_device
    io._is_cuda_gpu_device = _is_gpu_device

    # Patch _resolve_device to return GPU for Warp
    def _patched_resolve_device(impl):
        from mujoco.mjx._src import types
        impl = types.Impl(impl)
        if impl == types.Impl.JAX:
            return jax.devices()[0]
        if impl == types.Impl.CPP:
            return jax.devices('cpu')[0]
        if impl == types.Impl.WARP:
            gpus = _gpu_devices()
            if gpus:
                return gpus[0]
            return jax.devices('cpu')[0]
        raise ValueError(f'Unsupported implementation: {impl}')

    io._resolve_device = _patched_resolve_device

    # Patch _check_impl_device_compatibility to accept ROCm
    def _patched_check(impl, device):
        from mujoco.mjx._src import types
        impl = types.Impl(impl)
        if impl == types.Impl.WARP:
            is_gpu = _is_gpu_device(device)
            is_cpu = device.platform == 'cpu'
            if not (is_gpu or is_cpu):
                raise AssertionError(
                    'Warp implementation requires a GPU or CPU device, got '
                    f'{device}.'
                )
        is_cpu = device.platform == 'cpu'
        if impl == types.Impl.CPP:
            if not is_cpu:
                raise AssertionError(
                    f'C implementation requires a CPU device, got {device}.'
                )

    io._check_impl_device_compatibility = _patched_check

    # Also patch cubql → lbvh for render context on HIP
    import importlib
    try:
        bvh_mod = importlib.import_module("mujoco.mjx.third_party.mujoco_warp._src.bvh")
    except ImportError:
        try:
            bvh_mod = importlib.import_module("mujoco_warp._src.bvh")
        except ImportError:
            return  # no bvh module available

    _orig_build = bvh_mod.build_mesh_bvh
    def _patched_build(mjm, mid, constructor="sah", leaf_size=2):
        return _orig_build(mjm, mid, constructor="lbvh", leaf_size=leaf_size)
    bvh_mod.build_mesh_bvh = _patched_build

    if hasattr(bvh_mod, "build_hfield_bvh"):
        _orig_hfield = bvh_mod.build_hfield_bvh
        def _patched_hfield(mjm, mid, constructor="sah", leaf_size=2):
            return _orig_hfield(mjm, mid, constructor="lbvh", leaf_size=leaf_size)
        bvh_mod.build_hfield_bvh = _patched_hfield

    # Patch GraphMode for mujoco-mjx compatibility
    # mjx io.py calls getattr(mjxw.types.GraphMode, 'WARP') but some
    # mujoco_warp versions have GraphMode as an int instead of an enum
    try:
        import mujoco_warp._src.types as mw_types
        gm = getattr(mw_types, 'GraphMode', None)
        if gm is None or not hasattr(gm, 'WARP'):
            class _GraphMode:
                WARP = 0
                WARP_OPTIMIZE = 1
                RECORD = 2
            mw_types.GraphMode = _GraphMode
            # Also patch in the mjx io module's reference
            from mujoco.mjx._src import io as mjx_io
            if hasattr(mjx_io, 'mjxw'):
                mjx_io.mjxw.types.GraphMode = _GraphMode
    except (ImportError, AttributeError):
        pass