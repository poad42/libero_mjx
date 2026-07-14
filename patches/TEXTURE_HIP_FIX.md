# Warp Texture Sampling Fix for HIP/ROCm

## Problem

Warp 1.13.0's `texture.h` stubs out `texture_sample` on HIP devices with
`return 0.0f` instead of calling the actual `tex2D`/`tex3D` HIP intrinsics.
This causes all GPU-rendered textures to appear black on AMD ROCm GPUs.

## Root Cause

In `warp/native/texture.h`, each `sample_2d`/`sample_3d` function has three
code paths:

```cpp
#if defined(__CUDA_ARCH__)
    return tex2D<float>(wp_texture_object(tex.tex), u, v);  // CUDA: works
#elif defined(__HIP_DEVICE_COMPILE__)
    return 0.0f;  // HIP: STUBBED — returns black!
#else
    // CPU: software sampling, works
#endif
```

HIP has `tex2D<T>(hipTextureObject_t, float, float)` available (same API as
CUDA), but the warp code never calls it.

## Fix

Replace each `#elif defined(__HIP_DEVICE_COMPILE__) return 0.0f;` block with
the corresponding `tex2D`/`tex3D` call, mirroring the CUDA path:

```cpp
// float sample_2d
#elif defined(__HIP_DEVICE_COMPILE__)
    return tex2D<float>(wp_texture_object(tex.tex), u, v);

// float sample_3d
#elif defined(__HIP_DEVICE_COMPILE__)
    return tex3D<float>(wp_texture_object(tex.tex), u, v, w);

// vec2f sample_2d
#elif defined(__HIP_DEVICE_COMPILE__)
    float2 val = tex2D<float2>(wp_texture_object(tex.tex), u, v);
    return vec2f(val.x, val.y);

// vec2f sample_3d
#elif defined(__HIP_DEVICE_COMPILE__)
    float2 val = tex3D<float2>(wp_texture_object(tex.tex), u, v, w);
    return vec2f(val.x, val.y);

// vec4f sample_2d
#elif defined(__HIP_DEVICE_COMPILE__)
    float4 val = tex2D<float4>(wp_texture_object(tex.tex), u, v);
    return vec4f(val.x, val.y, val.z, val.w);

// vec4f sample_3d
#elif defined(__HIP_DEVICE_COMPILE__)
    float4 val = tex3D<float4>(wp_texture_object(tex.tex), u, v, w);
    return vec4f(val.x, val.y, val.z, val.w);
```

## Verification

Before fix: Warp render mean=19.4 (mostly black), CPU render mean=122.5
After fix:  Warp render mean=116.5 (matches CPU), R=125 G=118 B=107 vs CPU R=132 G=124 B=112

## How to Apply

```bash
# After installing warp, patch the texture.h header:
python -c "
import warp, os
path = os.path.join(os.path.dirname(warp.__file__), 'native', 'texture.h')
print(f'Patch target: {path}')
"
# Then apply the diff or manually replace the HIP stub blocks.
# Clear the warp kernel cache after patching:
rm -rf ~/.cache/warp
```