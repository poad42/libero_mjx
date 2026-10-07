"""Work around a Warp kernel-hash drift for nested ``module="unique"`` kernels.

The problem
-----------
Warp names a ``@wp.kernel(module="unique")`` module after a hash of its
contents (``<kernel key>_<module hash[:8]>``). The name is frozen when the
kernel is declared, and ``user_modules`` caches the compiled module under it.

For a nested kernel (a ``@wp.kernel`` defined inside a builder function, as
``mujoco_warp`` does for its CCD collision kernels), the kernel hash is not
stable across the module's ``block_dim`` variants. Warp hashes a kernel's
resolved ``wp.static`` expressions, and ``adj.build()`` repopulates those when
the module is compiled for a second ``block_dim``. A module can therefore be
built and cached under the name derived from hash H1 while the live kernel
object's hash later drifts to H2. ``ModuleExec.get_kernel_hooks`` then looks up
``<key>_H2_cuda_kernel_forward`` in a module whose metadata only knows
``<key>_H1_cuda_kernel_forward``, and raises::

    KeyError: '..._cuda_kernel_forward_smem_bytes'

(or, before that, a CUDA "named symbol not found" error). ``mujoco_warp`` hits
this because ``_ccd_grid_size`` calls ``wp.get_suggested_block_size`` (which
loads the module at the default ``block_dim`` of 256) and then launches the
same kernel with ``block_dim=m.block_dim.convex_ccd`` (64), creating both
variants. Whether the drift fires depends on hashing order, so the failure is
intermittent.

The fix
-------
Wrap ``ModuleExec.get_kernel_hooks`` so that, when the kernel's current mangled
name is absent from the loaded module's metadata, the name the module was
actually built with is recovered from the metadata and used for the lookup.
The correction is exact: only a single ``<kernel key>_<8 hex chars>`` match is
accepted, so a module that legitimately contains several kernels under one key
is left untouched (and fails loudly as before). The kernel's cached name is
restored after the call so other ``block_dim`` variants still see their own.

The wrapper is a no-op on stacks where the hash does not drift, and it uses
only ``ModuleExec.meta`` / ``Kernel.key``, which are stable across the Warp
1.13 and 1.17 releases used here.
"""
from __future__ import annotations

import re

_FORWARD_SUFFIX = "_cuda_kernel_forward_smem_bytes"

# Number of times a stale kernel name was remapped (diagnostics only).
REMAP_COUNT = 0


def _resolve_built_name(meta, key: str) -> str | None:
    """Return the mangled name this module was built with, or None if ambiguous."""
    prefix = key + "_"
    pattern = re.compile(
        re.escape(prefix) + r"[0-9a-f]{8}" + re.escape(_FORWARD_SUFFIX) + r"$"
    )
    candidates = [entry for entry in meta if pattern.match(entry)]
    if len(candidates) != 1:
        return None
    return candidates[0][: -len(_FORWARD_SUFFIX)]


def patch_stale_kernel_names() -> bool:
    """Install the ``get_kernel_hooks`` wrapper. Idempotent."""
    import warp._src.context as ctx

    module_exec_cls = getattr(ctx, "ModuleExec", None)
    if module_exec_cls is None or getattr(module_exec_cls, "_libero_mjx_stale_name_patch", False):
        return False

    original = module_exec_cls.get_kernel_hooks

    def get_kernel_hooks(self, kernel):
        current_name = kernel.get_mangled_name()
        had_cached_name = hasattr(kernel, "_mangled_name")
        saved_cached_name = getattr(kernel, "_mangled_name", None)

        built_name = None
        if current_name + _FORWARD_SUFFIX not in self.meta:
            built_name = _resolve_built_name(self.meta, kernel.key)

        if built_name is not None:
            global REMAP_COUNT
            REMAP_COUNT += 1
            if had_cached_name:
                kernel._mangled_name = built_name
            kernel.get_mangled_name = lambda: built_name

        try:
            return original(self, kernel)
        finally:
            if built_name is not None:
                if had_cached_name:
                    kernel._mangled_name = saved_cached_name
                try:
                    del kernel.get_mangled_name
                except AttributeError:
                    pass

    module_exec_cls.get_kernel_hooks = get_kernel_hooks
    module_exec_cls._libero_mjx_stale_name_patch = True
    return True


if __name__ == "__main__":
    print(patch_stale_kernel_names())
