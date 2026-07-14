"""Operational Space Controller (JAX port of robosuite OSC_POSE).

Converts a Cartesian delta end-effector action (6-DoF: dx,dy,dz,dax,day,daz)
into joint torques using the operational-space formulation:

    tau = J^T * Lambda * [F_pos; Tau_ori] + gravity + nullspace

where Lambda = (J M^-1 J^T)^-1.

This matches robosuite's controllers/parts/arm/osc.py with:
  - impedance_mode='fixed', input_type='delta', input_ref_frame='world'
  - uncouple_pos_ori=True
  - kp=150, damping_ratio=1 (critically damped)
  - output_max=0.05 (position), 0.5 (rotation)
"""

from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx


class OscController:
    """Batched OSC_POSE controller for MJX (JAX)."""

    def __init__(
        self,
        jacobian_site_id: int,
        jacobian_site_body_id: int,
        joint_qposadr: np.ndarray,
        joint_dofadr: np.ndarray,
        arm_actuator_idx: np.ndarray,
        nu: int,
        kp: float = 150.0,
        damping_ratio: float = 1.0,
        output_max: float = 0.05,
        output_max_ori: float = 0.5,
        rest_qpos: Optional[np.ndarray] = None,
        diag_mass: Optional[np.ndarray] = None,
        full_mass_arm: Optional[np.ndarray] = None,
    ):
        self._site_id = jacobian_site_id
        self._site_body_id = jacobian_site_body_id
        self._qposadr = jp.array(joint_qposadr)
        self._dofadr = jp.array(joint_dofadr)
        self._arm_act_idx = jp.array(arm_actuator_idx)
        self._nu = nu
        self._kp = jp.array(kp, dtype=float)
        self._kd = 2.0 * jp.sqrt(jp.array(kp, dtype=float)) * damping_ratio
        self._out_max = jp.array([output_max] * 3 + [output_max_ori] * 3)
        self._out_min = -self._out_max
        self._rest_qpos = jp.array(rest_qpos if rest_qpos is not None else np.zeros(7))
        self._diag_mass = diag_mass if diag_mass is not None else np.ones(43)
        self._full_mass_arm = jp.array(full_mass_arm) if full_mass_arm is not None else None
        self._model: Optional[mjx.Model] = None

    @classmethod
    def from_model(
        cls,
        mj_model: mujoco.MjModel,
        site_name: str = "gripper",
        arm_joint_prefix: str = "joint",
        kp: float = 150.0,
        damping_ratio: float = 1.0,
        output_max: float = 0.05,
        rest_qpos: Optional[np.ndarray] = None,
    ) -> "OscController":
        site = mj_model.site(site_name)
        site_id = site.id
        site_body_id = int(np.asarray(site.bodyid).flat[0])
        arm_joints = [f"{arm_joint_prefix}{i}" for i in range(1, 8)]
        qposadr = np.array([mj_model.jnt_qposadr[mj_model.joint(j).id] for j in arm_joints])
        dofadr = np.array([mj_model.jnt_dofadr[mj_model.joint(j).id] for j in arm_joints])
        # Arm actuators = first 7 (joint torque actuators for arm joints 1-7)
        arm_act = np.arange(7)
        if rest_qpos is None:
            rest_qpos = np.array([
                mj_model.qpos0[mj_model.joint(j).qposadr[0]]
                for j in arm_joints
            ])
        # Diagonal mass from model (fallback for broken Warp sparse reconstruction)
        diag_mass = np.array(mj_model.dof_M0)
        # Full arm mass matrix from CPU (constant for fixed-base robot)
        mjd = mujoco.MjData(mj_model)
        mujoco.mj_forward(mj_model, mjd)
        M_full = np.zeros((mj_model.nv, mj_model.nv))
        mujoco.mj_fullM(mj_model, mjd, M_full)
        arm_dof = np.array([mj_model.jnt_dofadr[mj_model.joint(j).id] for j in arm_joints])
        full_mass_arm = M_full[np.ix_(arm_dof, arm_dof)]
        return cls(
            jacobian_site_id=site_id,
            jacobian_site_body_id=site_body_id,
            joint_qposadr=qposadr,
            joint_dofadr=dofadr,
            arm_actuator_idx=arm_act,
            nu=mj_model.nu,
            kp=kp,
            damping_ratio=damping_ratio,
            output_max=output_max,
            rest_qpos=rest_qpos,
            diag_mass=diag_mass,
            full_mass_arm=full_mass_arm,
        )

    def set_model(self, model: mjx.Model):
        self._model = model

    def compute_torques(self, data: mjx.Data, delta_action: jax.Array) -> jax.Array:
        """Compute arm joint torques from a 6-DoF delta EE action.

        Args:
          data: MJX Data (batched or single).
          delta_action: [..., 6] delta in EE pose (clipped to output range).
        Returns:
          [..., nu] full ctrl vector with zeros for non-arm actuators.
        """
        delta = jp.clip(delta_action * self._out_max, self._out_min, self._out_max)

        # Current EE pose
        ee_pos = data.site_xpos[self._site_id]  # (..., 3)
        ee_mat = data.site_xmat[self._site_id]  # (..., 9) flattened

        # Desired pose = current + delta
        desired_pos = ee_pos + delta[..., :3]
        delta_rotvec = delta[..., 3:6]
        rot_err = self._axisangle_to_rotmat(delta_rotvec)  # (..., 3, 3)
        ee_mat_33 = ee_mat.reshape(*ee_pos.shape[:-1], 3, 3)
        desired_mat = jp.einsum("...ij,...jk->...ik", rot_err, ee_mat_33)

        # Position/orientation error
        pos_err = desired_pos - ee_pos  # (..., 3)
        ori_err = self._orientation_error(desired_mat, ee_mat_33)  # (..., 3)

        # Velocity error (zero base velocity assumption for tabletop)
        ee_vel = self._get_site_vel(data)  # (..., 6)
        vel_pos_err = -ee_vel[..., :3]
        vel_ori_err = -ee_vel[..., 3:6]

        # Desired force/torque
        desired_force = pos_err * self._kp + vel_pos_err * self._kd  # (..., 3)
        desired_torque = ori_err * self._kp + vel_ori_err * self._kd  # (..., 3)

        # Jacobians and mass matrix
        J_full, J_pos, J_ori, mass = self._get_jacobians_mass(data)
        # J_pos: (..., 3, n_arm), J_ori: (..., 3, n_arm), J_full: (..., 6, n_arm)
        # mass: (..., n_arm, n_arm)
        mass_inv = jp.linalg.inv(mass)  # (..., n_arm, n_arm)

        # Lambda matrices (uncoupled)
        lambda_pos_inv = J_pos @ mass_inv @ jp.swapaxes(J_pos, -1, -2)  # (..., 3, 3)
        lambda_ori_inv = J_ori @ mass_inv @ jp.swapaxes(J_ori, -1, -2)  # (..., 3, 3)
        lambda_pos = self._pinv(lambda_pos_inv)  # (..., 3, 3)
        lambda_ori = self._pinv(lambda_ori_inv)  # (..., 3, 3)

        # Decoupled force/torque: Lambda @ F
        decoupled_force = lambda_pos @ desired_force[..., :, None]  # (..., 3, 1)
        decoupled_torque = lambda_ori @ desired_torque[..., :, None]  # (..., 3, 1)
        wrench = jp.concatenate([decoupled_force, decoupled_torque], axis=-2)  # (..., 6, 1)
        wrench = wrench[..., 0]  # (..., 6)

        # tau = J^T * wrench + gravity comp
        torques = jp.swapaxes(J_full, -1, -2) @ wrench[..., :, None]  # (..., n_arm, 1)
        grav_comp = self._gravity_compensation(data)  # (..., n_arm)
        arm_torques = torques[..., 0] + grav_comp  # (..., n_arm)

        # Nullspace torques (PD to initial pose)
        nullspace = self._nullspace_torques(J_full, mass, mass_inv, data)
        arm_torques = arm_torques + nullspace

        # Place arm torques into full ctrl vector
        batch_shape = delta_action.shape[:-1]
        ctrl = jp.zeros((*batch_shape, self._nu))
        ctrl = ctrl.at[..., self._arm_act_idx].set(arm_torques)
        return ctrl

    def _get_site_vel(self, data: mjx.Data) -> jax.Array:
        """Approximate site spatial velocity via J^T @ qvel."""
        jacp, jacr = self._site_jacobian(data)
        # mjx.jac returns (..., nv, 3); velocity = J^T @ qvel = (..., 3, nv) @ (..., nv, 1)
        qvel = data.qvel
        lin_vel = jp.swapaxes(jacp, -1, -2) @ qvel[..., :, None]  # (..., 3, 1)
        ang_vel = jp.swapaxes(jacr, -1, -2) @ qvel[..., :, None]  # (..., 3, 1)
        return jp.concatenate([lin_vel[..., 0], ang_vel[..., 0]], axis=-1)  # (..., 6)

    def _site_jacobian(self, data: mjx.Data) -> Tuple[jax.Array, jax.Array]:
        """Compute translational and rotational Jacobians for the gripper site."""
        site_pos = data.site_xpos[self._site_id]  # (..., 3)
        jacp, jacr = mjx.jac(self._model, data, site_pos, self._site_body_id)
        # jacp, jacr: (..., nv, 3)
        return jacp, jacr

    def _get_jacobians_mass(self, data: mjx.Data) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        jacp, jacr = self._site_jacobian(data)
        # mjx.jac returns (..., nv, 3); arm_dof selects arm rows → (..., n_arm, 3)
        # Transpose to (..., 3, n_arm) to match robosuite J_pos/J_ori convention
        arm_dof = self._dofadr
        J_pos = jp.swapaxes(jacp[..., arm_dof, :], -1, -2)  # (..., 3, n_arm)
        J_ori = jp.swapaxes(jacr[..., arm_dof, :], -1, -2)  # (..., 3, n_arm)
        J_full = jp.concatenate([J_pos, J_ori], axis=-2)  # (..., 6, n_arm)
        mass_full = self._mass_matrix(data)  # (..., nv, nv)
        # Extract arm DOF submatrix (..., n_arm, n_arm)
        mass = mass_full[..., arm_dof, :][..., :, arm_dof]
        return J_full, J_pos, J_ori, mass

    def _mass_matrix(self, data: mjx.Data) -> jax.Array:
        """Full mass matrix. Uses precomputed arm submatrix from CPU."""
        if self._full_mass_arm is not None:
            # Use precomputed full arm mass matrix (constant for fixed-base)
            batch_shape = data.qpos.shape[:-1]
            if batch_shape:
                M = jp.tile(self._full_mass_arm, (*batch_shape, 1, 1))
            else:
                M = self._full_mass_arm
            return M
        # Fallback: diagonal mass matrix from model dof weights
        M_diag = jp.array(self._diag_mass, dtype=data.qpos.dtype)
        nv = self._model.nv
        batch_shape = data.qpos.shape[:-1]
        if batch_shape:
            M = jp.tile(jp.diag(M_diag), (*batch_shape, 1, 1))
        else:
            M = jp.diag(M_diag)
        return M

    def _gravity_compensation(self, data: mjx.Data) -> jax.Array:
        qfrc_bias = data.qfrc_bias
        arm_dof = self._dofadr
        return qfrc_bias[..., arm_dof]

    def _nullspace_torques(
        self, J_full: jax.Array, mass: jax.Array, mass_inv: jax.Array, data: mjx.Data
    ) -> jax.Array:
        joint_kp = 10.0
        joint_kv = 2.0 * jp.sqrt(joint_kp)
        arm_dof = self._dofadr
        arm_qpos = data.qpos[..., self._qposadr]  # (..., n_arm)
        arm_qvel = data.qvel[..., arm_dof]  # (..., n_arm)
        lambda_full_inv = J_full @ mass_inv @ jp.swapaxes(J_full, -1, -2)  # (..., 6, 6)
        lambda_full = self._pinv(lambda_full_inv)  # (..., 6, 6)
        Jbar = mass_inv @ jp.swapaxes(J_full, -1, -2) @ lambda_full  # (..., n_arm, 6)
        n_arm = J_full.shape[-1]
        I = jp.tile(jp.eye(n_arm), (*J_full.shape[:-2], 1, 1))  # (..., n_arm, n_arm)
        nullspace_mat = I - Jbar @ J_full  # (..., n_arm, n_arm)
        pose_torque = mass @ (joint_kp * (self._rest_qpos - arm_qpos) - joint_kv * arm_qvel)[..., :, None]
        ns_torque = jp.swapaxes(nullspace_mat, -1, -2) @ pose_torque
        return ns_torque[..., 0]  # (..., n_arm)

    @staticmethod
    def _axisangle_to_rotmat(rotvec: jax.Array) -> jax.Array:
        """Convert axis-angle (..., 3) to rotation matrix (..., 3, 3)."""
        theta = jp.linalg.norm(rotvec, axis=-1)  # (...,)
        # Avoid division by zero
        safe_theta = jp.where(theta > 1e-8, theta, jp.array(1.0))
        axis = rotvec / safe_theta[..., None]  # (..., 3)
        # Skew-symmetric matrix K
        ax = axis[..., 0]
        ay = axis[..., 1]
        az = axis[..., 2]
        zeros = jp.zeros_like(ax)
        K = jp.stack([
            jp.stack([zeros, -az, ay], axis=-1),
            jp.stack([az, zeros, -ax], axis=-1),
            jp.stack([-ay, ax, zeros], axis=-1),
        ], axis=-2)  # (..., 3, 3)
        I = jp.tile(jp.eye(3), (*rotvec.shape[:-1], 1, 1))
        t = theta[..., None, None]  # (..., 1, 1)
        R = I + jp.sin(t) * K + (1 - jp.cos(t)) * (K @ K)
        return jp.where(theta[..., None, None] > 1e-8, R, I)

    @staticmethod
    def _orientation_error(desired: jax.Array, current: jax.Array) -> jax.Array:
        """Orientation error as axis-angle vector. Both (..., 3, 3)."""
        rc1 = current[..., :, 0]
        rc2 = current[..., :, 1]
        rc3 = current[..., :, 2]
        rd1 = desired[..., :, 0]
        rd2 = desired[..., :, 1]
        rd3 = desired[..., :, 2]
        return 0.5 * (jp.cross(rc1, rd1) + jp.cross(rc2, rd2) + jp.cross(rc3, rd3))

    @staticmethod
    def _pinv(mat: jax.Array, rcond: float = 1e-6) -> jax.Array:
        return jp.linalg.pinv(mat)

    @staticmethod
    def _inv_via_cholesky(mat: jax.Array) -> jax.Array:
        """Matrix inverse via Cholesky decomposition (avoids hipsolver).

        For SPD matrices: inv(A) = inv(L) @ inv(L)^T where A = L @ L^T.
        Falls back to a regularized version if not SPD.
        """
        n = mat.shape[-1]
        # Regularize to ensure SPD
        reg = 1e-6 * jp.eye(n, dtype=mat.dtype)
        A = mat + reg
        try:
            L = jp.linalg.cholesky(A)  # A = L @ L^T
            # inv(A) = inv(L^T) @ inv(L) = solve(L^T, I)^T @ solve(L, I)
            I = jp.tile(jp.eye(n, dtype=mat.dtype), (*mat.shape[:-2], 1, 1))
            inv_L = jp.linalg.solve_triangular(L, I, lower=True)
            inv_A = jp.swapaxes(inv_L, -1, -2) @ inv_L
            return inv_A
        except Exception:
            # Fallback: pseudo-inverse via SVD-free approach
            return jp.linalg.pinv(A)

    @staticmethod
    def _pinv_via_cholesky(mat: jax.Array) -> jax.Array:
        """Pseudo-inverse for small matrices via normal equations.

        For a 3x3 matrix M: pinv(M) = inv(M^T M + reg*I) @ M^T.
        Uses Cholesky to avoid hipsolver.
        """
        MtM = jp.swapaxes(mat, -1, -2) @ mat
        inv_MtM = OscController._inv_via_cholesky(MtM)
        return inv_MtM @ jp.swapaxes(mat, -1, -2)