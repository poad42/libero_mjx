"""Fix cube-map material textures in the mujoco_warp ray tracer.

The bug
-------
MuJoCo supports two texture types: 2D and cube. A cube texture is stored as a
vertical strip of six faces. Warp's ``sample_texture`` treats *every* texture as
2D: it computes a ``(u, v)`` pair and calls ``wp.texture_sample`` once. For a
cube texture that samples a horizontal line across the six-face strip, so any
geom that uses a cube material renders as a smear instead of the intended
texture. ``sample_skybox`` already does the face mapping correctly, but it is
only wired to the skybox.

In LIBERO this is visible on the table (``tex-table``, wood grain) and the stove
top: the CPU renderer shows wood grain and concentric burner rings, the Warp
renderer shows flat colour.

The fix
-------
Reproduce MuJoCo's ``render_gl3.c`` cube texgen. For a regular (non-skybox)
cube texture the texgen planes are the object-space coordinates scaled by the
geom size when ``texuniform`` is set::

    S = size[0] * obj_x     T = size[1] * obj_y     R = size[2] * obj_z

GL then samples the cube map with the direction ``(S, T, R)``. So:

1. Compute the object-space hit point ``local = geom_xmat^T @ (hit - geom_xpos)``.
2. Map ``local`` to a cube face and ``(s, t)`` in OpenGL face order
   (+X, -X, +Y, -Y, +Z, -Z), the same order the texture strip uses.
3. Sample the strip at ``v = (face + t) / 6``.

A per-texture ``tex_cube_face_inv`` array (``1/width`` for cube textures, ``0``
otherwise) is added to ``RenderContext``; a non-positive value keeps the existing
2D path, so 2D textures are untouched.

``texuniform``/``geom_size`` scaling is not applied yet: LIBERO's cube materials
do not set ``texuniform``, so the direction is the object-space point.

The patch edits the installed ``mujoco_warp`` sources in place before import and
is idempotent. Call :func:`patch_cube_textures` before ``import mujoco_warp``
(``libero_mjx.render_kernel_patch`` does this for you).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

MARKER = "# PATCHED_BY_LIBERO_MJX_CUBE"

# RenderContext field. mujoco_warp <= 3.10 has skybox_face_width as a scalar;
# 3.13 has it as a per-camera array.
TYPES_PAIRS = [
    (
        '  skybox_face_width: array("*", int)\n  headlight_active: bool\n',
        '  skybox_face_width: array("*", int)\n'
        '  tex_cube_face_inv: array("*", float)\n'
        '  headlight_active: bool\n',
    ),
    (
        "  skybox_face_width: int\n  headlight_active: bool\n",
        "  skybox_face_width: int\n"
        '  tex_cube_face_inv: array("*", float)\n'
        "  headlight_active: bool\n",
    ),
]

# RenderContext construction. 3.13 builds it in render_util.py, <= 3.10 in io.py.
BUILD_EXPR = """    tex_cube_face_inv=wp.array(
      _tex_cube_face_inv(mjm),
      dtype=float,
    ),
"""

# Encodes how to sample each texture:
#   0.0   not a cube texture, use the 2D path
#   > 0.0 6:1 face strip, face height is 1/value, sample at (face + t) / 6
#   < 0.0 square cube texture (MuJoCo repeats one face on all six), sample at t
HELPER_FUNC = '''

def _tex_cube_face_inv(mjm: mujoco.MjModel) -> np.ndarray:
  """Per-texture cube-map sampling mode for the render kernel."""
  if mjm.ntex == 0:
    return np.zeros(0, dtype=float)
  is_cube = np.asarray(mjm.tex_type) == int(mujoco.mjtTexture.mjTEXTURE_CUBE)
  w = np.asarray(mjm.tex_width, dtype=float)
  h = np.asarray(mjm.tex_height, dtype=float)
  square = is_cube & (w == h)
  strip = is_cube & (w != h)
  return np.where(square, -1.0, np.where(strip, 1.0 / np.maximum(w, 1.0), 0.0))
'''
UTIL_PAIRS = [
    (
        "    skybox_face_width=wp.array(skybox_face_width_np, dtype=int),\n",
        "    skybox_face_width=wp.array(skybox_face_width_np, dtype=int),\n" + BUILD_EXPR,
    ),
    (
        "    skybox_face_width=skybox_face_width,\n",
        "    skybox_face_width=skybox_face_width,\n" + BUILD_EXPR,
    ),
]

SAMPLE_FUNC = '''@wp.func
def sample_cube_texture(
  # In:
  tex: wp.Texture2D,
  face_width_inv: float,
  direction: wp.vec3,
) -> wp.vec3:
  """Sample a 6-face vertical strip with a direction, OpenGL cube-map order."""
  rx = direction[0]
  ry = direction[1]
  rz = direction[2]
  arx = wp.abs(rx)
  ary = wp.abs(ry)
  arz = wp.abs(rz)

  face = int(0)
  sc = float(0.0)
  tc = float(0.0)
  ma = float(1.0)

  if arx >= ary and arx >= arz:
    ma = arx
    if rx > 0.0:
      face = 0
      sc = -rz
      tc = -ry
    else:
      face = 1
      sc = rz
      tc = -ry
  elif ary >= arz:
    ma = ary
    if ry > 0.0:
      face = 2
      sc = rx
      tc = rz
    else:
      face = 3
      sc = rx
      tc = -rz
  else:
    ma = arz
    if rz > 0.0:
      face = 4
      sc = rx
      tc = -ry
    else:
      face = 5
      sc = -rx
      tc = -ry

  s = (math.safe_div(sc, ma) + 1.0) * 0.5
  t = (math.safe_div(tc, ma) + 1.0) * 0.5
  v = float(0.0)
  if face_width_inv > 0.0:
    # 6:1 strip of six faces: keep the linear filter from bleeding between
    # adjacent faces, then pick the face row.
    t_min = 0.5 * face_width_inv
    t = wp.clamp(t, t_min, 1.0 - t_min)
    v = (float(face) + t) * wp.static(1.0 / 6.0)
  else:
    # Square cube texture: MuJoCo repeats the single face on all six sides. The
    # classic renderer projects object space planarly for these, so sample the
    # single image with the object-space xy instead of the cube face mapping.
    s = direction[0] - wp.floor(direction[0])
    v = direction[1] - wp.floor(direction[1])
  color = wp.texture_sample(tex, wp.vec2(s, v), dtype=wp.vec4)
  return wp.vec3(color[0], color[1], color[2])


'''

SAMPLE_DECORATED_OLD = """@wp.func
def sample_texture(
  # Model:
  geom_type: wp.array[int],
  mesh_faceadr: wp.array[int],
  # In:
  geom_id: int,
  tex_repeat: wp.vec2,
"""

SAMPLE_SIG_OLD = """def sample_texture(
  # Model:
  geom_type: wp.array[int],
  mesh_faceadr: wp.array[int],
  # In:
  geom_id: int,
  tex_repeat: wp.vec2,
"""
SAMPLE_SIG_NEW = """def sample_texture(
  # Model:
  geom_type: wp.array[int],
  mesh_faceadr: wp.array[int],
  tex_cube_face_inv: wp.array[float],
  # In:
  geom_id: int,
  tex_id: int,
  tex_repeat: wp.vec2,
"""

# <= 3.10 has no texuniform offset; 3.13 adds one.
SAMPLE_BODY_PAIRS = [
    (
        "  uv = wp.vec2(0.0, 0.0)\n  offset = wp.vec2(0.0, 0.0)\n\n"
        "  if geom_type[geom_id] == GeomType.PLANE:\n",
        "  if tex_cube_face_inv[tex_id] != 0.0:\n"
        "    # MuJoCo render_gl3.c cube texgen: direction is the object-space point.\n"
        "    local = wp.transpose(rot) @ (hit_point - pos)\n"
        "    return sample_cube_texture(tex, tex_cube_face_inv[tex_id], local)\n\n"
        "  uv = wp.vec2(0.0, 0.0)\n  offset = wp.vec2(0.0, 0.0)\n\n"
        "  if geom_type[geom_id] == GeomType.PLANE:\n",
    ),
    (
        "  uv = wp.vec2(0.0, 0.0)\n\n  if geom_type[geom_id] == GeomType.PLANE:\n",
        "  if tex_cube_face_inv[tex_id] != 0.0:\n"
        "    # MuJoCo render_gl3.c cube texgen: direction is the object-space point.\n"
        "    local = wp.transpose(rot) @ (hit_point - pos)\n"
        "    return sample_cube_texture(tex, tex_cube_face_inv[tex_id], local)\n\n"
        "  uv = wp.vec2(0.0, 0.0)\n\n  if geom_type[geom_id] == GeomType.PLANE:\n",
    ),
]

KERNEL_SIG_PAIRS = [
    (
        "    textures: wp.array[wp.Texture2D],\n    splat_position: wp.array[wp.vec3],\n",
        "    textures: wp.array[wp.Texture2D],\n"
        "    tex_cube_face_inv: wp.array[float],\n"
        "    splat_position: wp.array[wp.vec3],\n",
    ),
    (
        "    textures: wp.array[wp.Texture2D],\n    # Out:\n    rgb_out: wp.array2d[wp.uint32],\n",
        "    textures: wp.array[wp.Texture2D],\n"
        "    tex_cube_face_inv: wp.array[float],\n"
        "    # Out:\n    rgb_out: wp.array2d[wp.uint32],\n",
    ),
]

CALL_OLD = """            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              geom_id,
              mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],
              textures[tex_id],
"""
CALL_NEW = """            tex_color = sample_texture(
              geom_type,
              mesh_faceadr,
              tex_cube_face_inv,
              geom_id,
              tex_id,
              mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],
              textures[tex_id],
"""

LAUNCH_PAIRS = [
    (
        "        rc.textures,\n        rc.splat_position,\n",
        "        rc.textures,\n        rc.tex_cube_face_inv,\n        rc.splat_position,\n",
    ),
    (
        "      rc.textures,\n    ],\n",
        "      rc.textures,\n      rc.tex_cube_face_inv,\n    ],\n",
    ),
]


def _find_src() -> Path:
    spec = importlib.util.find_spec("mujoco_warp")
    if spec is not None and spec.submodule_search_locations:
        src = Path(list(spec.submodule_search_locations)[0]) / "_src"
        if src.is_dir():
            return src
    # Older mujoco-mjx vendors mujoco_warp under mjx/third_party, which is not on
    # sys.path until the caller adds it.
    try:
        import mujoco

        src = Path(mujoco.__file__).parent / "mjx" / "third_party" / "mujoco_warp" / "_src"
        if src.is_dir():
            return src
    except Exception:  # noqa: BLE001
        pass
    fallback = Path(
        "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party/mujoco_warp/_src"
    )
    if fallback.is_dir():
        return fallback
    raise ModuleNotFoundError("mujoco_warp is not installed")


def _read(path: Path) -> str:
    return path.read_text()


def _write(path: Path, text: str) -> None:
    bak = Path(str(path) + ".orig")
    if not bak.exists():
        bak.write_text(path.read_text())
    path.write_text(text)


def _apply_pairs(text: str, pairs) -> tuple[str, bool]:
    for old, new in pairs:
        if old in text:
            return text.replace(old, new, 1), True
    return text, False


def patch_cube_textures(verbose: bool = True) -> bool:
    """Apply the cube-map fix to the installed mujoco_warp. Idempotent."""
    src = _find_src()
    types_py = src / "types.py"
    render_py = src / "render.py"
    build_py = None
    for candidate in (src / "render_util.py", src / "io.py"):
        if candidate.exists() and _apply_pairs(candidate.read_text(), UTIL_PAIRS)[1]:
            build_py = candidate
            break
    if build_py is None:
        raise ValueError("RenderContext construction anchor not found in render_util.py or io.py")
    for p in (types_py, render_py, build_py):
        if not p.exists():
            raise FileNotFoundError(p)

    changed = 0

    text = _read(types_py)
    if MARKER not in text:
        text, ok = _apply_pairs(text, TYPES_PAIRS)
        if not ok:
            raise ValueError("types.py: RenderContext field anchor not found")
        _write(types_py, text.rstrip() + f"\n{MARKER}\n")
        changed += 1

    text = _read(build_py)
    if MARKER not in text:
        text, ok = _apply_pairs(text, UTIL_PAIRS)
        if not ok:
            raise ValueError(f"{build_py.name}: RenderContext construction anchor not found")
        _write(build_py, text.rstrip() + "\n" + HELPER_FUNC + f"\n{MARKER}\n")
        changed += 1

    text = _read(render_py)
    if MARKER not in text:
        if SAMPLE_DECORATED_OLD not in text:
            raise ValueError("render.py: decorated sample_texture not found")
        text = text.replace(SAMPLE_DECORATED_OLD, SAMPLE_FUNC + SAMPLE_DECORATED_OLD, 1)
        text, ok = _apply_pairs(text, [(SAMPLE_SIG_OLD, SAMPLE_SIG_NEW)])
        if not ok:
            raise ValueError("render.py: sample_texture signature not found")
        text, ok = _apply_pairs(text, SAMPLE_BODY_PAIRS)
        if not ok:
            raise ValueError("render.py: sample_texture body anchor not found")
        text, ok = _apply_pairs(text, KERNEL_SIG_PAIRS)
        if not ok:
            raise ValueError("render.py: megakernel texture arg not found")
        if CALL_OLD not in text:
            raise ValueError("render.py: sample_texture call site not found")
        text = text.replace(CALL_OLD, CALL_NEW, 1)
        text, ok = _apply_pairs(text, LAUNCH_PAIRS)
        if not ok:
            raise ValueError("render.py: render() launch inputs not found")
        _write(render_py, text.rstrip() + f"\n{MARKER}\n")
        changed += 1

    if verbose:
        print(f"[render_cube_patch] files changed: {changed}")
    return changed > 0


if __name__ == "__main__":
    patch_cube_textures()
