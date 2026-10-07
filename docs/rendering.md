# Rendering: CPU vs Warp

The Warp ray tracer in mujoco_warp produces images that differ from MuJoCo's CPU / EGL renderer. This page documents every difference found, the fix applied, and the measured impact on a BC policy trained on CPU data.

The test setup: spatial task 0, a BC transformer checkpoint trained 50 epochs on CPU demo data, 10 parallel envs, 600 steps per episode. CPU eval gives 50% success (5 of 10 episodes). Warp eval without fixes gives 0%.

![CPU agentview](docs/images/cpu_agentview.png)
![Warp agentview](docs/images/warp_agentview.png)

CPU (EGL) and Warp (ray trace) agentview renders of spatial task 0, init state 0. Same scene, same camera, all fixes applied.

## Differences and fixes

### Cube-map material textures

The Warp `sample_texture` treats every texture as 2D: it computes a `(u, v)`
pair and samples once. A cube texture is stored as six faces, so a cube material
samples a horizontal line across the strip instead of the intended image. In
LIBERO the table (`tex-table`, wood grain) and the stove top render flat.

MuJoCo's classic renderer (`render_gl3.c` / `render_context.c`) has two cube
cases:

- **6:1 strip** (`tex_width != tex_height`): six faces stacked vertically.
  `sample_skybox` already does this mapping for the skybox; the fix applies the
  same face mapping to material textures. GL samples with the object-space
  direction `(x, y, z)` scaled by `geom_size` when `texuniform` is set.
- **square** (`tex_width == tex_height`): MuJoCo uploads the single image to all
  six faces, and the classic renderer projects object space **planarly**. The
  fix samples the single image with the object-space `xy`.

`render_cube_patch.py` adds a per-texture `tex_cube_face_inv` array to
`RenderContext` (`0` = 2D, `> 0` = 6:1 strip, `< 0` = square), a
`sample_cube_texture` function, and the kernel plumbing. 2D textures are
untouched. The patch is applied automatically by `libero_mjx`.

Measured on spatial task 0, 256x256, brightness 1.15:

| Camera | RMSE before | RMSE after |
|---|---|---|
| agentview | 24.16 | 21.45 |
| eye-in-hand | 29.92 | 20.38 |

The eye-in-hand camera is dominated by the table, so the cube-map fix is a 32%
reduction there. The agentview background is still skybox-dominated, so its
reduction is smaller.

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

Before the cube-map fix, a workaround was to replace cube textures with their
average color. That produced 0% success: the policy relies on texture features
that even the incorrect 2D sampling partially provides, and flat colors provide
none. The correct fix (see above) samples the cube map properly and keeps the
texture.

### Removing transparent geoms

LIBERO models include EEF target geoms (spheres & boxes at alpha 0.5 and 0.8) in geom group 2. The Warp ray tracer has no alpha blending, so it renders them as opaque. Moving them to group 3 (not rendered) dropped success from 30% to 10%.

The training data includes these geoms with alpha blending. Removing them changes the scene layout the policy expects. Keeping them as opaque shapes is closer to the training distribution than removing them.

### Patching geom rgba into materials

The Warp render kernel reads `mat_rgba` when a geom has a material (`geom_matid >= 0`). MuJoCo's CPU renderer multiplies `geom_rgba` by `mat_rgba` for non-textured geoms. For most LIBERO robot parts, `geom_rgba = [0.5, 0.5, 0.5]` and `mat_rgba = [1, 1, 1]`, so Warp renders them at full brightness while CPU renders them at 50%.

Patching `geom_rgba` into `mat_rgba` for non-textured materials did not change success rates. The brightness boost (1.15x) dominates this effect.

## Patch ordering: the nested render megakernel and stale source lines

If the render megakernel fails to compile with

```
warp._src.codegen.WarpCodegenTypeError: '_build_megakernel__locals___render_megakernel':
Warp kernels cannot return values.
```

the cause is patch ordering, not the warp branch. `mujoco_warp` defines
`_render_megakernel` **inside** `_build_megakernel`, and warp resolves a
function's source from its code object's `co_firstlineno` against the file on
disk **at the time the kernel is built**. If the file is rewritten after the
module was imported, the line number is stale:

- `libero_mjx` used to call `patch_warp_to_gpu()` first. That imports
  `mujoco_warp._src.render`, compiling the original file.
- `patch_render_kernel()` then rewrote `render.py` on disk. Every function below
  the insertion point shifted, and `_render_megakernel`'s recorded line now
  pointed at `compute_lighting`.
- At render time warp sliced the rewritten file at the stale line, got
  `compute_lighting`, and `ModuleBuilder.build_kernel` rejected the enclosing
  function's `return _render_megakernel` as a kernel value return.

The fix is in `libero_mjx/__init__.py`: **rewrite the source first, then import
`mujoco_warp`**. `patch_render_kernel()` now runs before `patch_warp_to_gpu()`.
With that ordering the render megakernel compiles on the public stack (warp
`1.17.0+rocm.0` @ `amd-integration-halo`, `mujoco_warp` 3.13, gfx1201).

There is also a latent warp bug here: the fast source-extraction path validates
that `tree.body[0].name == code.co_name`, but the `inspect.getsourcelines`
fallback does not. `docker/patch_warp_nested_kernel.py` makes the fallback
validate and recover by name. It is applied at image build time as defence in
depth; the ordering fix alone is sufficient.

## CPU rendering: EGL vs OSMesa

The reproducible image is built on the TheRock manylinux base (AlmaLinux 8),
whose Mesa is 23.1. That predates gfx1201 support in the EGL device platform, so
hardware EGL is unavailable: `MUJOCO_GL=egl` fails with "EGL driver does not
support the PLATFORM_DEVICE extension". The image installs `mesa-dri-drivers` and
`mesa-libOSMesa` and sets `MUJOCO_GL=osmesa`. Plain MuJoCo renders in software
that way, but **robosuite 1.5.1 still segfaults** when it creates its offscreen
context on this Mesa, in every combination tried (EGL/OSMesa, hardware/llvmpipe).

Consequences:

- The GPU Warp renderer does not use `MUJOCO_GL` and works normally. This is the
  supported render path in the container.
- The CPU reference path (`scripts/render_comparison.py`) and the CPU eval
  (`scripts/eval_bc.py`) need a Mesa with gfx1201 EGL support (24.1+). The
  comparison images in this repository were generated on an Ubuntu 24.04 image
  with a newer Mesa; regenerate them on such a host.
- On a host with a newer Mesa, set `MUJOCO_GL=egl` and the CPU paths work.

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

After all fixes, the Warp-vs-EGL RMSE is 21.5 (agentview) and 20.4
(eye-in-hand) at 256x256 on spatial task 0, down from 24.2 and 29.9 before the
cube-map fix. The remaining gap is:

| Region | RMSE |
|---|---|
| Background (skybox) | 18.1 |
| Objects | 20.5 |
| Table | 3.9 before the cube fix, lower after |

The background difference is the skybox brightness (Warp is ~85% of EGL). The
object difference is the absence of alpha blending on the EEF target geoms, and
the residual brightness mismatch. The table now matches well.

These remaining differences account for the gap between Warp eval (42.5%) and
CPU eval (50%). Alpha blending and a principled exposure correction would need
changes to the mujoco_warp render kernel.