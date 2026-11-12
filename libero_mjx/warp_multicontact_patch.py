"""Restore the single-contact box-box path of mujoco_warp 3.10.0.

mujoco_warp 3.13.0 always runs the box-box multicontact recovery, independent of
the MULTICCD disable flag. That produces two contact points per box-box pair
where 3.10.0 produced one, which changes nacon, nefc, qfrc_constraint, qacc,
qpos and qvel after one step.

This module replaces ``collision_convex.multicontact`` with a function that
returns the single EPA witness pair. The kernel builders read the name from the
module namespace at build time, so the patch must be applied before the first
``mjwarp.forward`` / ``mjwarp.step``.

The patch is a no-op on mujoco_warp 3.10.0, which has no ``multicontact``.
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
