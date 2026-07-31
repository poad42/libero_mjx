"""Patch the mujoco_warp render kernel to match MuJoCo's CPU renderer.

Key differences fixed:
1. Shadow fallback: Warp uses visible=0.3 for shadowed pixels, MuJoCo uses 0.0
2. Haze: MuJoCo applies atmospheric haze (vis.map.haze) blending distant
   objects with the skybox color. Warp renderer doesn't apply haze at all.

This module patches installed mujoco_warp files on disk before mujoco_warp is
imported. Must be called BEFORE any `import mujoco_warp` statement.

Usage:
    from libero_mjx.render_kernel_patch import patch_render_kernel
    patch_render_kernel()  # call before importing mujoco_warp
"""
import os

BASE = "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party/mujoco_warp/_src"
RENDER_PY = f"{BASE}/render.py"
TYPES_PY = f"{BASE}/types.py"
IO_PY = f"{BASE}/io.py"

PATCH_MARKER = "# PATCHED_BY_LIBERO_MJX"


def _read(path):
    with open(path, "r") as f:
        return f.read()


def _write(path, src):
    with open(path, "w") as f:
        f.write(src)


def _backup(path):
    bak = path + ".orig"
    if not os.path.exists(bak):
        _write(bak, _read(path))


def _already_patched(src):
    return PATCH_MARKER in src


def _patch_render_py():
    src = _read(RENDER_PY)
    if _already_patched(src):
        return
    _backup(RENDER_PY)

    patches = []

    old_shadow = (
        "      if shadow_geom_id != -1:\n"
        "        visible = NO_LIGHT_AMBIENT_FALLBACK"
    )
    new_shadow = (
        "      if shadow_geom_id != -1:\n"
        "        visible = 0.0"
    )
    if old_shadow not in src:
        raise ValueError("Cannot find shadow fallback pattern")
    patches.append((old_shadow, new_shadow))

    old_haze = (
        "    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))\n"
        "    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))\n"
        "\n"
        "    rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32("
    )
    new_haze = (
        "    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))\n"
        "    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))\n"
        "\n"
        "    if wp.static(rc.haze_amount > 0.0):\n"
        "      frag_dist = wp.length(hit_point - ray_origin_world)\n"
        "      haze_t = wp.clamp((frag_dist - wp.static(rc.fogstart)) / wp.static(rc.fogend - rc.fogstart), 0.0, 1.0)\n"
        "      haze_factor = haze_t * wp.static(rc.haze_amount)\n"
        "      bg = wp.static(rc.background_color_float)\n"
        "      hit_color = hit_color * (1.0 - haze_factor) + bg * haze_factor\n"
        "\n"
        "    rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32("
    )
    if old_haze not in src:
        raise ValueError("Cannot find haze insertion point")
    patches.append((old_haze, new_haze))

    for old, new in patches:
        src = src.replace(old, new, 1)

    src = src.rstrip() + f"\n{PATCH_MARKER}\n"
    _write(RENDER_PY, src)
    print(f"[render_kernel_patch] Patched render.py: shadow=0.0, +haze")


def _patch_types_py():
    src = _read(TYPES_PY)
    if _already_patched(src):
        return
    _backup(TYPES_PY)

    old = '  geom_ray_types: tuple = ()'
    new = (
        '  geom_ray_types: tuple = ()\n'
        '  haze_amount: float = 0.0\n'
        '  fogstart: float = 0.0\n'
        '  fogend: float = 1.0\n'
        '  background_color_float: dataclasses.field = dataclasses.field('
        'default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))\n'
        f'  {PATCH_MARKER}'
    )
    if old not in src:
        raise ValueError("Cannot find geom_ray_types in types.py")
    src = src.replace(old, new, 1)
    _write(TYPES_PY, src)
    print(f"[render_kernel_patch] Patched types.py: +haze_amount, fogstart, fogend, background_color_float")


def _patch_io_py():
    src = _read(IO_PY)
    if _already_patched(src):
        return
    _backup(IO_PY)

    old = (
        '    light_attenuation_is_default=light_attenuation_is_default,\n'
        '    has_spot_lights=has_spot_lights,\n'
        '  )'
    )
    new = (
        '    light_attenuation_is_default=light_attenuation_is_default,\n'
        '    has_spot_lights=has_spot_lights,\n'
        '    haze_amount=float(mjm.vis.map.haze),\n'
        '    fogstart=float(mjm.vis.map.fogstart * mjm.stat.extent),\n'
        '    fogend=float(mjm.vis.map.fogend * mjm.stat.extent),\n'
        '    background_color_float=wp.vec3(\n'
        '      background_color[0], background_color[1], background_color[2]\n'
        '    ),\n'
        f'    {PATCH_MARKER}\n'
        '  )'
    )
    if old not in src:
        raise ValueError("Cannot find RenderContext construction in io.py")
    src = src.replace(old, new, 1)
    _write(IO_PY, src)
    print(f"[render_kernel_patch] Patched io.py: +haze/fog params to RenderContext")


def patch_render_kernel():
    for path in (RENDER_PY, TYPES_PY, IO_PY):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cannot find {path}")
    _patch_types_py()
    _patch_io_py()
    _patch_render_py()
    print("[render_kernel_patch] All patches applied successfully")