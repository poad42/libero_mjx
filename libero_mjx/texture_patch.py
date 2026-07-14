"""Patch warp's get_type_code to handle Texture2D array types.

The issue: warp 1.13.0's get_type_code doesn't recognize wp.array[wp.Texture2D]
when hashing kernel arguments, causing:
  TypeError: Unrecognized type '<class 'warp._src.types.array'>'

This patch adds a type code for Texture2D arrays so kernel hashing succeeds.
Must be imported before mujoco_warp render functions are called.
"""
import warp._src.types as wpt

_orig_get_type_code = wpt.get_type_code


def _patched_get_type_code(arg_type) -> str:
    try:
        return _orig_get_type_code(arg_type)
    except TypeError:
        from warp._src.texture import Texture2D
        if isinstance(arg_type, type) and issubclass(arg_type, Texture2D):
            return "tex2d"
        if wpt.is_array(arg_type):
            dtype = getattr(arg_type, "dtype", None)
            if dtype is not None and isinstance(dtype, type):
                if issubclass(dtype, Texture2D):
                    return "atex2d"
        raise


wpt.get_type_code = _patched_get_type_code

import warp._src.context as wpc
wpc.get_type_code = _patched_get_type_code


def patch():
    """Apply the texture type code patch."""
    pass  # already applied at import time