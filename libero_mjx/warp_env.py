"""Pure Warp + Torch environment for LIBERO eval (no JAX).

Replaces the JAX-based env.step with a direct Warp physics + Torch OSC
controller loop. Eliminates JAX kernel launch overhead, the main bottleneck
identified by profiling (406 ms / 25 substeps, 75% of step time).

The OSC controller runs in Torch eager mode on small matrices (7x7, 3x3, 6x6).
The physics runs via mujoco_warp.step(). Jacobians come from mujoco_warp.jac().
Zero-copy interop via wp.to_torch() and wp.from_dlpack().

The mass matrix is densified from the Warp sparse M field. The M field is
read-only via DLPack on ROCm (the read-only flag is unsupported). The fix is
to .clone() first (creates a writable PyTorch-owned copy). For this tiny
matrix (19x19, 153 non-zeros) the densification is done on CPU: the DtoH+HtoD
transfer (~600 bytes) costs less than 8 GPU kernel launches. The GPU-only
path (clone + GPU scatter) works correctly but is ~7 ms slower per step.

Supports the spatial suite (task 0). Other suites require porting their
predicates and obs construction.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import warp as wp
import mujoco
try:
    import mujoco_warp as mjwarp
except ImportError:
    import sys
    sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party")
    import mujoco_warp as mjwarp


class WarpOscController:
    """Operational Space Controller in Torch (port of JAX OscController).

    Computes arm joint torques from a 6-DoF Cartesian delta EE action.
    The state-dependent mass matrix is densified from the Warp M field
    once per control step and reused for all substeps.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        wm: mjwarp.Model,
        site_name: str = "gripper0_right_grip_site",
        arm_joint_prefix: str = "robot0_joint",
        kp: float = 150.0,
        damping_ratio: float = 1.0,
        output_max: float = 0.05,
        output_max_ori: float = 0.5,
        rest_qpos: Optional[np.ndarray] = None,
    ):
        self._wm = wm
        site = mj_model.site(site_name)
        self.site_id = site.id
        self.site_body_id = int(np.asarray(site.bodyid).flat[0])
        arm_joints = [f"{arm_joint_prefix}{i}" for i in range(1, 8)]
        self.qposadr = np.array([mj_model.jnt_qposadr[mj_model.joint(j).id] for j in arm_joints])
        self.dofadr = np.array([mj_model.jnt_dofadr[mj_model.joint(j).id] for j in arm_joints])
        self.arm_act_idx = np.arange(7)
        self.nu = mj_model.nu
        self.kp = float(kp)
        self.kd = 2.0 * np.sqrt(kp) * damping_ratio
        self.out_max = torch.tensor([output_max] * 3 + [output_max_ori] * 3, device="cuda")
        self.out_min = -self.out_max
        if rest_qpos is None:
            rest_qpos = np.array([mj_model.qpos0[mj_model.joint(j).qposadr[0]] for j in arm_joints])
        self.rest_qpos = torch.tensor(rest_qpos, dtype=torch.float32, device="cuda")

        self._qposadr_t = torch.tensor(self.qposadr, dtype=torch.long, device="cuda")
        self._dofadr_t = torch.tensor(self.dofadr, dtype=torch.long, device="cuda")
        self._arm_act_t = torch.tensor(self.arm_act_idx, dtype=torch.long, device="cuda")
        self._n_arm = 7

        nv = mj_model.nv
        rownnz = np.array(mj_model.M_rownnz)
        colind = np.array(mj_model.M_colind)
        self._mass_all_rows = np.repeat(np.arange(nv), rownnz)
        self._mass_all_cols = colind
        self._nv = nv

    def compute_mass_matrix(self, wd: mjwarp.Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Densify the Warp sparse mass matrix into the arm 7x7 submatrix.

        The M field is read-only via DLPack (ROCm doesn't support the flag),
        so .clone() is used first. For this tiny matrix (19x19, 153 non-zeros)
        the CPU densification is faster than GPU: the DtoH+HtoD transfer
        (~600 bytes) costs less than 8 GPU kernel launches (zeros, scatter,
        tril, transpose, add, index, inv). The GPU alternative
        (wp.to_torch(wd.M).clone().squeeze(1) then GPU scatter) works
        correctly but adds ~7 ms per control step on ROCm.
        """
        M_np = wp.to_torch(wd.M).clone().cpu().numpy().squeeze(1)
        N = M_np.shape[0]
        mat_np = np.zeros((N, self._nv, self._nv), dtype=np.float32)
        mat_np[:, self._mass_all_rows, self._mass_all_cols] = M_np
        mat_np = mat_np + np.tril(mat_np, -1).transpose(0, 2, 1)
        mass_arm = torch.tensor(mat_np[:, self.dofadr][:, :, self.dofadr], device="cuda")
        return mass_arm, torch.linalg.inv(mass_arm)

    def compute_torques(
        self,
        wd: mjwarp.Data,
        desired_pos: torch.Tensor,
        desired_mat: torch.Tensor,
        jacp_wp: wp.array,
        jacr_wp: wp.array,
        mass_arm: torch.Tensor,
        mass_inv: torch.Tensor,
    ) -> torch.Tensor:
        point = wd.site_xpos[:, self.site_id]
        body = self._body_wp
        mjwarp.jac(self._wm, wd, jacp_wp, jacr_wp, point, body)

        ee_pos = wp.to_torch(wd.site_xpos)[:, self.site_id].clone()
        ee_mat = wp.to_torch(wd.site_xmat)[:, self.site_id].clone()

        jacp = wp.to_torch(jacp_wp).transpose(-1, -2).contiguous()
        jacr = wp.to_torch(jacr_wp).transpose(-1, -2).contiguous()
        qvel = wp.to_torch(wd.qvel)
        qpos = wp.to_torch(wd.qpos)

        lin_vel = jacp.transpose(-1, -2) @ qvel[..., :, None]
        ang_vel = jacr.transpose(-1, -2) @ qvel[..., :, None]
        ee_vel = torch.cat([lin_vel[..., 0], ang_vel[..., 0]], dim=-1)

        pos_err = desired_pos - ee_pos
        ori_err = self._orientation_error(desired_mat, ee_mat)

        vel_pos_err = -ee_vel[:, :3]
        vel_ori_err = -ee_vel[:, 3:6]

        desired_force = pos_err * self.kp + vel_pos_err * self.kd
        desired_torque = ori_err * self.kp + vel_ori_err * self.kd

        arm_dof = self._dofadr_t
        J_pos = jacp[:, arm_dof, :].transpose(-1, -2)
        J_ori = jacr[:, arm_dof, :].transpose(-1, -2)
        J_full = torch.cat([J_pos, J_ori], dim=-2)

        lambda_pos_inv = J_pos @ mass_inv @ J_pos.transpose(-1, -2)
        lambda_ori_inv = J_ori @ mass_inv @ J_ori.transpose(-1, -2)
        lambda_pos = torch.linalg.pinv(lambda_pos_inv)
        lambda_ori = torch.linalg.pinv(lambda_ori_inv)

        decoupled_force = lambda_pos @ desired_force[:, :, None]
        decoupled_torque = lambda_ori @ desired_torque[:, :, None]
        wrench = torch.cat([decoupled_force, decoupled_torque], dim=-2)[..., 0]

        torques = J_full.transpose(-1, -2) @ wrench[:, :, None]
        grav_comp = wp.to_torch(wd.qfrc_bias)[:, arm_dof].clone()
        arm_torques = torques[..., 0] + grav_comp

        arm_qpos = qpos[:, self._qposadr_t]
        arm_qvel = qvel[:, self._dofadr_t]
        nullspace = self._nullspace_torques(J_full, mass_inv, mass_arm, arm_qpos, arm_qvel)
        arm_torques = arm_torques + nullspace

        N = arm_torques.shape[0]
        ctrl = torch.zeros(N, self.nu, device="cuda", dtype=torch.float32)
        ctrl[:, self._arm_act_t] = arm_torques
        return ctrl

    def set_body_wp(self, n_envs: int):
        self._body_wp = wp.array([self.site_body_id] * n_envs, dtype=wp.int32)

    def _nullspace_torques(
        self, J_full: torch.Tensor, mass_inv: torch.Tensor, mass: torch.Tensor,
        arm_qpos: torch.Tensor, arm_qvel: torch.Tensor,
    ) -> torch.Tensor:
        joint_kp = 10.0
        joint_kv = 2.0 * np.sqrt(joint_kp)
        lambda_full_inv = J_full @ mass_inv @ J_full.transpose(-1, -2)
        lambda_full = torch.linalg.pinv(lambda_full_inv)
        Jbar = mass_inv @ J_full.transpose(-1, -2) @ lambda_full
        I = torch.eye(self._n_arm, device="cuda").expand(J_full.shape[0], -1, -1)
        nullspace_mat = I - Jbar @ J_full
        pose_torque = mass @ (joint_kp * (self.rest_qpos - arm_qpos) - joint_kv * arm_qvel)[:, :, None]
        ns_torque = nullspace_mat.transpose(-1, -2) @ pose_torque
        return ns_torque[..., 0]

    @staticmethod
    def _orientation_error(desired: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        rc1, rc2, rc3 = current[..., :, 0], current[..., :, 1], current[..., :, 2]
        rd1, rd2, rd3 = desired[..., :, 0], desired[..., :, 1], desired[..., :, 2]
        return 0.5 * (torch.cross(rc1, rd1, dim=-1) + torch.cross(rc2, rd2, dim=-1) + torch.cross(rc3, rd3, dim=-1))

    @staticmethod
    def axisangle_to_rotmat(rotvec: torch.Tensor) -> torch.Tensor:
        theta = torch.linalg.norm(rotvec, dim=-1)
        safe_theta = torch.where(theta > 1e-8, theta, torch.ones_like(theta))
        axis = rotvec / safe_theta[..., None]
        ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
        zeros = torch.zeros_like(ax)
        K = torch.stack([
            torch.stack([zeros, -az, ay], dim=-1),
            torch.stack([az, zeros, -ax], dim=-1),
            torch.stack([-ay, ax, zeros], dim=-1),
        ], dim=-2)
        I = torch.eye(3, device="cuda").expand(rotvec.shape[0], 3, 3)
        t = theta[..., None, None]
        R = I + torch.sin(t) * K + (1 - torch.cos(t)) * (K @ K)
        return torch.where(theta[..., None, None] > 1e-8, R, I)


class WarpEnv:
    """Pure Warp + Torch environment for LIBERO spatial eval.

    No JAX in the physics step. The OSC controller runs in Torch, the physics
    in Warp, and the mass matrix is densified from the Warp M field via CPU.

    Args:
        mj_model: MuJoCo MjModel (CPU) for index lookups.
        n_envs: Number of parallel worlds.
        naconmax: Max contacts across all worlds.
        njmax: Max constraints per world.
        sim_dt: Physics timestep (0.002 = 25 substeps, 0.01 = 5 substeps).
        ctrl_dt: Control timestep (0.05 = 20 Hz).
        episode_length: Max steps per episode.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        n_envs: int = 10,
        naconmax: int = 8192,
        njmax: int = 1024,
        sim_dt: float = 0.002,
        ctrl_dt: float = 0.05,
        episode_length: int = 600,
    ):
        self.n_envs = n_envs
        self.sim_dt = sim_dt
        self.ctrl_dt = ctrl_dt
        self.n_substeps = int(round(ctrl_dt / sim_dt))
        self.episode_length = episode_length
        self.mj_model = mj_model
        self.nq = mj_model.nq
        self.nv = mj_model.nv

        mj_model.opt.timestep = sim_dt
        with wp.ScopedDevice("cuda:0"):
            self.wm = mjwarp.put_model(mj_model)
            self.wd = mjwarp.make_data(mj_model, nworld=n_envs, naconmax=naconmax, njmax=njmax)
            self._jacp_wp = wp.zeros((n_envs, 3, self.nv), dtype=wp.float32)
            self._jacr_wp = wp.zeros((n_envs, 3, self.nv), dtype=wp.float32)

        self._first_step = True

        self.osc = WarpOscController(mj_model, self.wm)
        self.osc.set_body_wp(n_envs)

        m = mj_model
        self.bowl_body = m.body("akita_black_bowl_1_main").id
        self.plate_body = m.body("plate_1_main").id
        self.gripper_ctrl_idx = [7, 8]

        arm_joints = [f"robot0_joint{i}" for i in range(1, 8)]
        self.arm_qposadr = torch.tensor(
            [m.jnt_qposadr[m.joint(j).id] for j in arm_joints], dtype=torch.long, device="cuda"
        )
        finger_joints = ["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]
        self.finger_qposadr = torch.tensor(
            [m.jnt_qposadr[m.joint(j).id] for j in finger_joints], dtype=torch.long, device="cuda"
        )

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set qpos/qvel for all envs and run forward kinematics."""
        qpos_t = torch.tensor(qpos, dtype=torch.float32, device="cuda")
        qvel_t = torch.tensor(qvel, dtype=torch.float32, device="cuda")
        self.wd.qpos = wp.from_dlpack(torch.utils.dlpack.to_dlpack(qpos_t))
        self.wd.qvel = wp.from_dlpack(torch.utils.dlpack.to_dlpack(qvel_t))
        mjwarp.forward(self.wm, self.wd)

    def step(self, action: torch.Tensor, gripper_cur: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one control step: OSC + physics substeps.

        Args:
            action: (N, 7) raw action (6 arm + 1 gripper), range [-1, 1].
            gripper_cur: (N, 2) current gripper accumulator state.
        Returns:
            success: (N,) float tensor (1.0 = success, 0.0 = not).
            gripper_cur: (N, 2) updated gripper accumulator.
        """
        action = torch.clamp(action, -1.0, 1.0)
        arm_action = action[:, :6]
        gripper_action = action[:, 6:7]

        gripper_speed = 0.2
        gripper_sign = torch.sign(gripper_action[:, 0])
        delta_g = torch.stack([-gripper_sign, gripper_sign], dim=-1) * gripper_speed
        gripper_cur = torch.clamp(gripper_cur + delta_g, -1.0, 1.0)
        finger_target = 0.02 * torch.stack([1.0 + gripper_cur[:, 0], gripper_cur[:, 1] - 1.0], dim=-1)

        if self._first_step:
            mjwarp.forward(self.wm, self.wd)
            self._first_step = False

        ee_pos = wp.to_torch(self.wd.site_xpos)[:, self.osc.site_id].clone()
        ee_mat = wp.to_torch(self.wd.site_xmat)[:, self.osc.site_id].clone()

        delta = torch.clamp(arm_action * self.osc.out_max, self.osc.out_min, self.osc.out_max)
        desired_pos = ee_pos + delta[:, :3]
        rot_err = self.osc.axisangle_to_rotmat(delta[:, 3:6])
        desired_mat = rot_err @ ee_mat

        mass_arm, mass_inv = self.osc.compute_mass_matrix(self.wd)

        for _ in range(self.n_substeps):
            ctrl = self.osc.compute_torques(
                self.wd, desired_pos, desired_mat,
                self._jacp_wp, self._jacr_wp,
                mass_arm, mass_inv,
            )
            ctrl[:, self.gripper_ctrl_idx] = finger_target
            self.wd.ctrl = wp.from_dlpack(torch.utils.dlpack.to_dlpack(ctrl.contiguous()))
            mjwarp.step(self.wm, self.wd)

        success = self._eval_success()
        return success, gripper_cur

    def _eval_success(self) -> torch.Tensor:
        bowl_pos = wp.to_torch(self.wd.xpos)[:, self.bowl_body]
        plate_pos = wp.to_torch(self.wd.xpos)[:, self.plate_body]
        return (torch.norm(bowl_pos - plate_pos, dim=-1) < 0.08).float()

    def get_obs(self) -> Dict[str, torch.Tensor]:
        qpos_t = wp.to_torch(self.wd.qpos)
        return {
            "joint_states": qpos_t[:, self.arm_qposadr],
            "gripper_states": qpos_t[:, self.finger_qposadr],
        }

    def get_qpos(self) -> torch.Tensor:
        return wp.to_torch(self.wd.qpos)