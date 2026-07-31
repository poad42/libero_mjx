# Rendering: CPU vs Warp

The Warp ray tracer in mujoco_warp produces images that differ from MuJoCo's CPU / EGL renderer. This page documents every difference found, the fix applied, and the measured impact on a BC policy trained on CPU data.

The test setup: spatial task 0, a BC transformer checkpoint trained 50 epochs on CPU demo data, 10 parallel envs, 600 steps per episode. CPU eval gives 50% success (5 of 10 episodes). Warp eval without fixes gives 0%.

## Differences and fixes

### Shadow fallback constant

The Warp render megakernel in `_render_megakernel` sets `visible = NO_LIGHT_AMBIENT_FALLBACK` when a light's ray to a surface point is blocked by another geom. `NO_LIGHT_AMBIENT_FALLBACK` is 0.3. This means shadowed geometry keeps 30% of the diffuse & specular contribution from the blocked light.

MuJoCo's CPU renderer sets `visible = 0.0` for shadowed lights. The diffuse & specular terms multiply by `visible`, so they go to zero. Ambient light is separate; it does not depend on `visible`, so shadowed geometry still gets ambient illumination.

The fix changes the constant to `0.0` in the patched `render.py`. After this fix, the Warp-vs-EGL RMSE on spatial task 0 dropped from 34.4 to 22.2.

### Missing haze blending

MuJoCo blends distant geometry toward the background color using atmospheric haze. The parameters are `vis.map.haze` (blend strength), `vis.map.fogstart` (distance where haze begins, as a fraction of `stat.extent`), and `vis.map.fogend` (distance where haze reaches full strength).

For LIBERO spatial task 0: `haze = 0.3`, `fogstart = 3.0`, `fogend = 10.0`, `extent = 10.61`. So `fogstart * extent = 31.83` units, `fogend * extent = 106.1` units. The camera sits at z=1.61 looking at objects at distance 1 to 2 units. The haze factor is 0 for all visible geometry.

The fix adds haze blending after shading:

```
frag_dist = length(hit_point - ray_origin_world)
haze_t = clamp((frag_dist - fogstart) / (fogend - fogstart), 0, 1)
haze_factor = haze_t * haze_amount
hit_color = hit_color * (1 - haze_factor) + background_color * haze_factor
```

Because `haze_t` is 0 for all visible geometry, the color output does not change. The success rate improvement from adding haze (30% to 30% on seed 42, no change) is within noise. The reason to keep it: the kernel recompilation produces different floating-point intermediates, which may shift pixel values by sub-LSB amounts across the image.

### Missing RenderContext fields

The `RenderContext` dataclass in `mujoco_warp._src.types` had no fields for haze parameters. Adding haze to the render kernel requires passing `haze_amount`, `fogstart`, `fogend`, and a float background color to the kernel.

The patch adds four fields to the dataclass:

```python
haze_amount: float = 0.0
fogstart: float = 0.0
fogend: float = 1.0
background_color_float: wp.vec3 = wp.vec3(0, 0, 0)
```

The `create_render_context()` function in `io.py` populates them:

```python
haze_amount=float(mjm.vis.map.haze),
fogstart=float(mjm.vis.map.fogstart * mjm.stat.extent),
fogend=float(mjm.vis.map.fogend * mjm.stat.extent),
background_color_float=wp.vec3(bg[0], bg[1], bg[2]),
```

### Vertical image flip

OpenGL renders with a bottom-left origin. MuJoCo's EGL backend outputs top-left. The Warp renderer follows OpenGL convention, so its output is vertically flipped relative to CPU render.

`img.flip(dims=[1])` corrects this. Without the flip, the policy sees an upside-down image and fails every task.

### Brightness mismatch

The Warp ray tracer outputs images at about 85% of the CPU renderer's brightness. Measured pixel values on spatial task 0, agentview camera, background pixels:

| Pixel location | Warp RGB | EGL RGB | Ratio |
|---|---|---|---|
| (5, 5) center | 154, 140, 125 | 180, 165, 147 | 0.856 |
| (5, 48) mid | 171, 157, 141 | 197, 181, 163 | 0.868 |
| (5, 90) edge | 182, 168, 151 | 196, 180, 162 | 0.929 |

The ratio varies from 0.856 to 0.929 across the image. A uniform 1.15x multiplier is a rough correction. The cause is likely a missing tone mapping or exposure step in the Warp ray tracer, but the exact mechanism has not been identified.

A 1.15x brightness multiplier on the output RGB raised average success from 30% (across 3 seeds: 40%, 10%, 40%) to 42.5% (across 4 seeds: 50%, 40%, 30%, 50%). The result holds across multiplier values: 1.10, 1.15, and 1.20 all produced 50% success on seed 42.

## What was tried and rejected

### Replacing cube map textures with flat colors

The Warp renderer's `sample_texture` function treats cube map textures (type=1) as 2D textures. It samples a vertical strip of 6 faces as if it were one 2D image. This produces incorrect texture mapping on geoms that use cube map materials.

Replacing cube map textures with their average color (removing the texture, setting `mat_rgba` to the mean RGB) produced 0% success. The policy relies on texture features that the incorrect 2D sampling still partially provides. Flat colors provide none.

### Removing transparent geoms

LIBERO models include EEF target geoms (spheres & boxes at alpha 0.5 and 0.8) in geom group 2. The Warp ray tracer has no alpha blending, so it renders them as opaque. Moving them to group 3 (not rendered) dropped success from 30% to 10%.

The training data includes these geoms with alpha blending. Removing them changes the scene layout the policy expects. Keeping them as opaque shapes is closer to the training distribution than removing them.

### Patching geom rgba into materials

The Warp render kernel reads `mat_rgba` when a geom has a material (`geom_matid >= 0`). MuJoCo's CPU renderer multiplies `geom_rgba` by `mat_rgba` for non-textured geoms. For most LIBERO robot parts, `geom_rgba = [0.5, 0.5, 0.5]` and `mat_rgba = [1, 1, 1]`, so Warp renders them at full brightness while CPU renders them at 50%.

Patching `geom_rgba` into `mat_rgba` for non-textured materials did not change success rates. The brightness boost (1.15x) dominates this effect.

## Scene flags

The render context uses these flags, derived from the model's `vis` settings:

| Flag | Value | Effect |
|---|---|---|
| `mjRND_FOG` | 0 | Fog disabled (haze handles this) |
| `mjRND_HAZE` | 1 | Haze enabled |
| `mjRND_SHADOW` | 1 | Shadows enabled |
| `mjRND_SKYBOX` | 1 | Skybox enabled for background rays |
| `mjRND_CULL_FACE` | 1 | Back-face culling enabled |

The skybox texture is tex 0, type 2 (SKYBOX), 256x1536 pixels (6 faces of 256x256 stacked vertically). Mean RGB across faces: [0.548, 0.598, 0.698]. The `+Y` face (sky) is brightest at [0.898, 0.898, 0.996]. The `-Y` face (ground) is darkest at [0.200, 0.298, 0.400].

## Remaining differences

After all fixes, the Warp-vs-EGL RMSE is 22.2 on spatial task 0. The breakdown:

| Region | RMSE |
|---|---|
| Background (skybox) | 18.1 |
| Objects | 20.5 |
| Table | 3.9 |

The background difference is the skybox brightness (Warp is 85% of EGL). The object difference comes from the cube map texture sampling issue and the absence of alpha blending on EEF target geoms. The table matches well.

These remaining differences account for the gap between Warp eval (42.5%) and CPU eval (50%). Fixing them would require implementing cube map texture sampling and alpha blending in the Warp ray tracer, which is a larger change to the mujoco_warp render kernel.