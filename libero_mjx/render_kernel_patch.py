"""Patch the mujoco_warp render kernel to match MuJoCo's CPU renderer.

Three classes of difference are handled, and which ones apply depends on the
installed mujoco_warp version:

1. Shadow fallback. Older mujoco_warp hard-codes ``visible = 0.3`` for shadowed
   pixels where MuJoCo uses ``0.0``. Newer versions expose the value as the
   ``shadow_light_fraction`` argument of ``create_render_context``, so no source
   patch is needed (``WarpRenderer`` passes ``0.0``).
2. Haze. Older versions have no atmospheric haze and no ``RenderContext``
   fields for it; this module adds both. Newer versions dropped haze entirely.
3. Cube-map material textures. Every version samples a cube texture as if it
   were 2D, so geoms that use a cube material render flat. This is fixed by
   ``render_cube_patch`` for all versions.

The module edits the installed ``mujoco_warp`` files on disk, before
``mujoco_warp`` is imported. Call :func:`patch_render_kernel` first:

    from libero_mjx.render_kernel_patch import patch_render_kernel
    patch_render_kernel()

``libero_mjx/__init__.py`` and the eval scripts do this automatically.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    from libero_mjx import render_cube_patch as _cube_mod
    from libero_mjx.render_cube_patch import patch_cube_textures
except ImportError:  # loaded by path (scripts do this before sys.path is set up)
    import importlib.util as _ilu

    _spec = _ilu.spec_from_file_location(
        "render_cube_patch", Path(__file__).with_name("render_cube_patch.py")
    )
    _cube_mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_cube_mod)
    patch_cube_textures = _cube_mod.patch_cube_textures

MARKER_SHADOW = "# PATCHED_BY_LIBERO_MJX_SHADOW"
MARKER_HAZE = "# PATCHED_BY_LIBERO_MJX_HAZE"
MARKER_PLANE = "# PATCHED_BY_LIBERO_MJX_PLANE"


def _find_src() -> Path:
    return _cube_mod._find_src()


def _has_shadow_light_fraction(src: Path) -> bool:
    """True when create_render_context already exposes shadow_light_fraction."""
    for name in ("render_util.py", "io.py"):
        p = src / name
        if p.exists() and "shadow_light_fraction" in p.read_text():
            return True
    return False


def _read(path: Path) -> str:
    return path.read_text()


def _write(path: Path, text: str) -> None:
    bak = Path(str(path) + ".orig")
    if not bak.exists():
        bak.write_text(path.read_text())
    path.write_text(text)


MARKER_BOUNDS = "# PATCHED_BY_LIBERO_MJX_MESH_BOUNDS"


def _patch_mesh_bounds(src: Path) -> bool:
    """Use 3.10's mesh bounding half-size.

    3.13: half = max(|pmin|, |pmax|)
    3.10: half = 0.5 * (pmax - pmin)
    For an off-centre mesh these differ, and `half` feeds the geom bounds used by
    the broadphase.
    """
    bvh_py = src / "bvh.py"
    if not bvh_py.exists():
        return False
    s = _read(bvh_py)
    if MARKER_BOUNDS in s:
        return False
    old = "  half = np.maximum(np.abs(pmin), np.abs(pmax))\n"
    if old not in s:
        return False
    _write(bvh_py, s.replace(old, "  half = 0.5 * (pmax - pmin)\n", 1) + "\n" + MARKER_BOUNDS + "\n")
    return True


MARKER_UV = "# PATCHED_BY_LIBERO_MJX_FACE_UV"


def _patch_incomplete_face_uv(src: Path) -> bool:
    """Port upstream PR #1742: incomplete face UVs must not read adjacent UV data.

    MuJoCo imports an omitted OBJ face corner as mesh_facetexcoord == -1 even when
    mesh_texcoordadr >= 0. 3.13 checks only the mesh-level offset, so the sentinel
    is added to the offset and an unrelated UV is interpolated.  Check all three
    indices and fall back to the OBJECT_PLANE projection when any is missing.
    """
    render_py = src / "render.py"
    if not render_py.exists():
        return False
    s = _read(render_py)
    if MARKER_UV in s:
        return False
    old = """      coords = mesh_facetexcoord[face_adr]
      uv0 = mesh_texcoord[texcoord_offset + coords[0]]
      uv1 = mesh_texcoord[texcoord_offset + coords[1]]
      uv2 = mesh_texcoord[texcoord_offset + coords[2]]
      uv = uv0 * bary_u + uv1 * bary_v + uv2 * (1.0 - bary_u - bary_v)"""
    new_body = """      coords = mesh_facetexcoord[face_adr]
      if coords[0] >= 0 and coords[1] >= 0 and coords[2] >= 0:
        uv0 = mesh_texcoord[texcoord_offset + coords[0]]
        uv1 = mesh_texcoord[texcoord_offset + coords[1]]
        uv2 = mesh_texcoord[texcoord_offset + coords[2]]
        uv = uv0 * bary_u + uv1 * bary_v + uv2 * (1.0 - bary_u - bary_v)"""
    if old not in s:
        return False
    _write(render_py, s.replace(old, new_body, 1) + "\n" + MARKER_UV + "\n")
    return True


MARKER_MARCH = "# PATCHED_BY_LIBERO_MJX_NO_MARCH"


def _patch_mesh_march(src: Path) -> bool:
    """Use 3.10's single mesh query instead of 3.13's backface-cull marching loop.

    3.13 marches past culled back-facing hits to the next surface (up to 8
    steps). On the LIBERO pedestal and cabinet meshes that produces hits where
    3.10 reports none, which changes 10 pixels of the agentview image.
    """
    ray_py = src / "ray.py"
    if not ray_py.exists():
        return False
    s = _read(ray_py)
    if MARKER_MARCH in s:
        return False
    old = """  offset = float(0.0)
  for _ in range(8):
    hit = wp.mesh_query_ray(mesh_bvh, lpnt + lvec * offset, lvec, max_t - offset, t, u, v, sign, n, f)
    if not hit:
      return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
    if (not cull_backfaces) or wp.dot(lvec, n) < 0.0:
      normal = mat @ n
      normal = wp.normalize(normal)
      return offset + t, normal, u, v, f, mesh_geom_id
    offset += t * (1.0 + RAY_TOL_REL) + RAY_TOL_ABS

  return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1"""
    new_body = """  hit = wp.mesh_query_ray(mesh_bvh, lpnt, lvec, max_t, t, u, v, sign, n, f)
  if not hit:
    return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1
  if (not cull_backfaces) or wp.dot(lvec, n) < 0.0:
    normal = mat @ n
    normal = wp.normalize(normal)
    return t, normal, u, v, f, mesh_geom_id
  return -1.0, wp.vec3(0.0, 0.0, 0.0), 0.0, 0.0, -1, -1"""
    if old not in s:
        return False
    _write(ray_py, s.replace(old, new_body, 1) + "\n" + MARKER_MARCH + "\n")
    return True


def _patch_plane_texgen(render_py: Path) -> bool:
    """Use the pre-3.13 plane texgen so the table and stove textures match.

    mujoco_warp 3.13 applies MuJoCo's OBJECT_PLANE texgen (0.5 scale, -0.5
    offset) to plane geoms. 3.10 sampled the plane-local coordinates directly.
    The difference shifts the wood grain on the table and the stove top, which
    is visible to a BC policy trained on the 3.10 renderer.
    """
    src = _read(render_py)
    if MARKER_PLANE in src:
        return False
    old = (
        "    uv = wp.vec2(0.5 * local[0], -0.5 * local[1])\n"
        "    offset = wp.vec2(-0.5, -0.5)\n"
    )
    if old not in src:
        return False
    new = (
        "    uv = wp.vec2(local[0], local[1])\n"
        "    offset = wp.vec2(0.0, 0.0)\n"
        f"    {MARKER_PLANE}\n"
    )
    _write(render_py, src.replace(old, new, 1))
    return True


def _patch_legacy_shadow(render_py: Path) -> bool:
    src = _read(render_py)
    if MARKER_SHADOW in src:
        return False
    old = (
        "      if shadow_geom_id != -1:\n"
        "        visible = NO_LIGHT_AMBIENT_FALLBACK"
    )
    if old not in src:
        return False
    new = (
        "      if shadow_geom_id != -1:\n"
        "        visible = 0.0"
    )
    _write(render_py, src.replace(old, new, 1).rstrip() + f"\n{MARKER_SHADOW}\n")
    return True


def _patch_legacy_haze(src: Path) -> bool:
    render_py = src / "render.py"
    types_py = src / "types.py"
    io_py = src / "io.py"
    if not io_py.exists():
        return False
    if MARKER_HAZE in _read(render_py):
        return False

    old_haze = (
        "    hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))\n"
        "    hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))\n"
        "\n"
        "    rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32("
    )
    if old_haze not in _read(render_py):
        return False

    for p, old, new in (
        (types_py,
         "  geom_ray_types: tuple = ()",
         "  geom_ray_types: tuple = ()\n"
         "  haze_amount: float = 0.0\n"
         "  fogstart: float = 0.0\n"
         "  fogend: float = 1.0\n"
         "  background_color_float: dataclasses.field = dataclasses.field("
         "default_factory=lambda: wp.vec3(0.0, 0.0, 0.0))\n"
         f"  {MARKER_HAZE}"),
        (io_py,
         "    light_attenuation_is_default=light_attenuation_is_default,\n"
         "    has_spot_lights=has_spot_lights,\n"
         "  )",
         "    light_attenuation_is_default=light_attenuation_is_default,\n"
         "    has_spot_lights=has_spot_lights,\n"
         "    haze_amount=float(mjm.vis.map.haze),\n"
         "    fogstart=float(mjm.vis.map.fogstart * mjm.stat.extent),\n"
         "    fogend=float(mjm.vis.map.fogend * mjm.stat.extent),\n"
         "    background_color_float=wp.vec3(\n"
         "      background_color[0], background_color[1], background_color[2]\n"
         "    ),\n"
         f"    {MARKER_HAZE}\n"
         "  )"),
    ):
        text = _read(p)
        if old not in text:
            raise ValueError(f"Cannot find pattern in {p.name}")
        _write(p, text.replace(old, new, 1))

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
    text = _read(render_py)
    _write(render_py, text.replace(old_haze, new_haze, 1).rstrip() + f"\n{MARKER_HAZE}\n")
    return True


def patch_render_kernel() -> None:
    src = _find_src()
    for name in ("render.py", "types.py"):
        if not (src / name).exists():
            raise FileNotFoundError(src / name)

    if _has_shadow_light_fraction(src):
        print("[render_kernel_patch] shadow_light_fraction is a native parameter; "
              "no source patch needed")
    else:
        shadow = _patch_legacy_shadow(src / "render.py")
        haze = _patch_legacy_haze(src)
        print(f"[render_kernel_patch] legacy patches: shadow={shadow} haze={haze}")

    patch_cube_textures()
    plane = _patch_plane_texgen(src / "render.py")
    print(f"[render_kernel_patch] plane texgen: {plane}")
    march = _patch_mesh_march(src)
    print(f"[render_kernel_patch] mesh march: {march}")
    face_uv = _patch_incomplete_face_uv(src)
    print(f"[render_kernel_patch] incomplete face UV: {face_uv}")
    bounds = _patch_mesh_bounds(src)
    print(f"[render_kernel_patch] mesh bounds: {bounds}")
    print("[render_kernel_patch] done")


if __name__ == "__main__":
    patch_render_kernel()
