"""Observation builder for LIBERO MJX port.

Matches LIBERO's obs spec from configs/data/default.yaml:
  - low_dim: gripper_states (robot0_gripper_qpos), joint_states (robot0_joint_pos)
  - rgb: agentview_rgb, eye_in_hand_rgb (state-only for now; vision later)

State-only obs is sufficient for counterfactual label collection.
"""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jp
import mujoco
from mujoco import mjx


def build_obs(
    mj_model: mujoco.MjModel,
    data: mjx.Data,
    keys: Sequence[str] = ("gripper_states", "joint_states"),
    arm_joint_prefix: str = "joint",
    finger_joint_names: Sequence[str] = ("finger_joint1", "finger_joint2"),
) -> jp.ndarray:
    """Build a flat observation vector from MJX data.

    Args:
      mj_model: MjModel (for index lookups).
      data: MJX Data (batched or single).
      keys: ordered list of obs keys to include.
      arm_joint_prefix: prefix for arm joints (e.g. "joint" or "robot0_joint").
      finger_joint_names: names of gripper finger joints.
    Returns:
      [..., obs_dim] concatenated observation.
    """
    parts = []
    for key in keys:
        if key == "joint_states":
            arm_joints = [f"{arm_joint_prefix}{i}" for i in range(1, 8)]
            idx = [mj_model.jnt_qposadr[mj_model.joint(j).id] for j in arm_joints]
            parts.append(data.qpos[..., idx])
        elif key == "gripper_states":
            idx = [mj_model.jnt_qposadr[mj_model.joint(j).id] for j in finger_joint_names]
            parts.append(data.qpos[..., idx])
        elif key == "ee_states":
            site = mj_model.site("gripper").id if mj_model.site("gripper") else mj_model.site(0).id
            parts.append(data.site_xpos[site])
            parts.append(data.site_xmat[site].reshape(*data.site_xpos[site].shape[:-1], 9))
        elif key == "object_states":
            for obj in _free_objects(mj_model):
                parts.append(data.xpos[obj])
                parts.append(data.xmat[obj].reshape(*data.xpos[obj].shape[:-1], 9))
        elif key == "agentview_rgb" or key == "eye_in_hand_rgb":
            raise NotImplementedError(
                f"{key} requires rendering; not implemented in state-only mode"
            )
        else:
            raise KeyError(f"Unknown obs key: {key}")
    return jp.concatenate(parts, axis=-1) if parts else jp.array([])


def _free_objects(mj_model: mujoco.MjModel) -> list[int]:
    """Return body IDs of objects with free joints."""
    obj_ids = []
    for b in range(1, mj_model.nbody):
        jnt = mj_model.body(b).jntadr[0]
        if jnt >= 0 and mj_model.jnt_type[jnt] == mujoco.mjtJoint.mjJNT_FREE:
            obj_ids.append(b)
    return obj_ids


def obs_dim(mj_model: mujoco.MjModel, keys: Sequence[str]) -> int:
    """Compute total observation dimension for given keys."""
    dim = 0
    for key in keys:
        if key == "joint_states":
            dim += 7
        elif key == "gripper_states":
            dim += 2
        elif key == "ee_states":
            dim += 12
        elif key == "object_states":
            dim += 12 * len(_free_objects(mj_model))
    return dim