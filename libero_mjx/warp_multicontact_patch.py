"""Optional single-contact override for the box-box multicontact path.

The patch replaces ``collision_convex.multicontact`` with a function that
returns a single EPA witness pair. It was written on the assumption that
mujoco_warp 3.10.0 produced one contact per box-box pair. Measurement on
libero_spatial task 0 (10 envs, 4 seeded random actions, 2026-11-13) shows the
opposite:

  step   patched 3.13.0   native 3.13.0   3.10.0
  1      42               52              42
  12     17               53              47
  13     792              1961            1539
  16     749              1837            1506
  20     455              1168            1167

Both 3.10.0 and native 3.13.0 use a multi-witness box-box path; the patch
removes contacts the 3.10.0 reference keeps. It is therefore off by default.
Set LIBERO_MJX_SINGLE_CONTACT=1 to re-enable it. The patch is a no-op on
mujoco_warp 3.10.0 (the `third_party` build has no `multicontact`).
"""
from __future__ import annotations

import warp as wp

from mujoco_warp._src.collision_core import Geom
from mujoco_warp._src.types import mat43


@wp.func
def multicontact_single(
    polygon: wp.array[wp.vec3],
    clipped: wp.array[wp.vec3],
    plane_normal: wp.array[wp.vec3],
    plane_dist: wp.array[float],
    idx1: wp.array[int],
    idx2: wp.array[int],
    n1: wp.array[wp.vec3],
    n2: wp.array[wp.vec3],
    endvert: wp.array[wp.vec3],
    face1: wp.array[wp.vec3],
    face2: wp.array[wp.vec3],
    epa_vert: wp.array[wp.vec3],
    epa_vert_index: wp.array[int],
    epa_face: int,
    x1: wp.vec3,
    x2: wp.vec3,
    geom1: Geom,
    geom2: Geom,
    geomtype1: int,
    geomtype2: int,
):
    witness1 = mat43()
    witness2 = mat43()
    dists = wp.vec4()
    witness1[0] = x1
    witness2[0] = x2
    return 1, witness1, witness2, dists


def patch_single_contact() -> bool:
    try:
        from mujoco_warp._src import collision_convex
        from mujoco_warp._src import collision_gjk
    except ImportError:
        return False

    if "third_party" in getattr(collision_gjk, "__file__", ""):
        return False
    if not hasattr(collision_convex, "multicontact"):
        return False
    if getattr(collision_convex, "_libero_mjx_single_contact", False):
        return False

    collision_gjk.multicontact = multicontact_single
    collision_convex.multicontact = multicontact_single
    collision_convex._libero_mjx_single_contact = True
    return True


if __name__ == "__main__":
    print(patch_single_contact())
