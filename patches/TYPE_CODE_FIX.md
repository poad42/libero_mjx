# Warp Kernel Hash Fix for Texture2D Array Types

## Problem

Warp 1.13.0's `get_type_code` function in `warp/_src/types.py` doesn't
recognize `wp.array[wp.Texture2D]` when hashing kernel arguments, causing:

```
TypeError: Unrecognized type '<class 'warp._src.types.array'>'
```

This prevents any kernel using `wp.array[wp.Texture2D]` from compiling,
even if the texture sampling itself works.

## Root Cause

The `get_type_code` function handles scalars, vectors, matrices, and arrays
with known dtypes (float, int, etc.), but `Texture2D` as an array dtype falls
through to the `raise TypeError` fallback.

## Fix

Monkey-patch `get_type_code` before importing warp kernels:

```python
import warp._src.types as wpt

_orig = wpt.get_type_code

def _patched_get_type_code(arg_type) -> str:
    try:
        return _orig(arg_type)
    except TypeError:
        from warp._src.texture import Texture2D
        if isinstance(arg_type, type) and issubclass(arg_type, Texture2D):
            return "tex2d"
        if wpt.is_array(arg_type):
            dtype = getattr(arg_type, "dtype", None)
            if dtype is not None and isinstance(dtype, type):
                if issubclass(dtype, Texture2D):
                    return "atex2d"
            if arg_type is wpt.array:
                return "a?"
        raise

wpt.get_type_code = _patched_get_type_code
```

This is already applied in `libero_mjx/texture_patch.py` and imported
by `libero_mjx/__init__.py`.