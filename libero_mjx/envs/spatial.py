"""LIBERO_SPATIAL task suite (10 tasks) ported to MuJoCo Warp.

All tasks share the same goal: pick up the black bowl (akita_black_bowl_1)
and place it on the plate (plate_1). The only difference is the initial
position of the bowl.

The MJCF XMLs are extracted directly from the robosuite LIBERO env, ensuring
exact physics fidelity. Init states are loaded from LIBERO's .init files
(numpy arrays of shape [N, 92]: 1 dummy + 48 qpos + 43 qvel).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import jax
import jax.numpy as jp
import numpy as np
from etils import epath
import mujoco
from mujoco import mjx

from libero_mjx.envs.base import LiberoMjxEnv, LiberoState
from libero_mjx.envs import register_env
from libero_mjx.predicates.spatial import distance_to, on_top_of


_BDDL_DIR = "libero_spatial"
_XML_DIR = epath.Path(__file__).parent.parent / "assets" / "xml"
# Init states: check common locations, allow override via LIBERO_INIT_DIR env var
_INIT_DIR = os.environ.get(
    "LIBERO_INIT_DIR",
    "/workspace/libero_basil/libero/libero/init_files/libero_spatial",
)
_INIT_DIR = epath.Path(_INIT_DIR)

TASK_NAMES = [
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
]


class LiberoSpatialBase(LiberoMjxEnv):
    """Base for LIBERO_SPATIAL tasks. Shares body/joint indices."""

    def __init__(self, naconmax: int = 8192, optimize_physics: bool = True, **kwargs):
        # LIBERO scenes have many contacts (275 geoms) — need large buffers.
        # njmax: max constraints per world (seen up to ~750 in practice)
        # naconmax: max contacts across all worlds (per-env ~110)
        # Default 8192 = enough for ~64 envs; scale up for more
        kwargs.setdefault("njmax", 1024)
        super().__init__(naconmax=naconmax, **kwargs)

        if optimize_physics:
            # Optimize physics for GPU-parallel simulation:
            # - Remove fluid model (air resistance) — not supported by Warp with implicit
            # - Use implicitfast integrator (like PandaPickCube)
            # - 5 solver iterations (vs robosuite's 100)
            # - timestep=0.005 (vs 0.002) for fewer substeps
            # - Pyramidal friction cone (vs elliptic) for Warp compatibility
            import mujoco
            from mujoco import mjx
            self._mj_model.opt.density = 0.0
            self._mj_model.opt.viscosity = 0.0
            self._mj_model.opt.cone = 0  # pyramidal
            self._mj_model.opt.integrator = 3  # implicitfast
            self._mj_model.opt.timestep = 0.005
            self._mj_model.opt.iterations = 5
            self._mj_model.opt.ls_iterations = 8
            self._mjx_model = mjx.put_model(self._mj_model, impl=self._impl)
            self._osc.set_model(self._mjx_model)
            self._sim_dt = 0.005
            self._ctrl_dt = 0.02  # 4 substeps (vs 25 with robosuite defaults)

    def _get_assets(self) -> Dict[str, bytes]:
        # No assets needed — meshes are loaded from absolute paths in the XML
        return {}

    def _post_init(self) -> None:
        m = self._mj_model
        # Robot arm joints (robot0_joint1-7)
        self._arm_joints = [f"robot0_joint{i}" for i in range(1, 8)]
        self._robot_arm_qposadr = np.array([
            m.jnt_qposadr[m.joint(j).id] for j in self._arm_joints
        ])
        self._robot_arm_dofadr = np.array([
            m.jnt_dofadr[m.joint(j).id] for j in self._arm_joints
        ])

        # Gripper
        self._gripper_site = m.site("gripper0_right_grip_site").id
        self._gripper_site_body = int(np.asarray(m.site("gripper0_right_grip_site").bodyid).flat[0])
        self._gripper_ctrl_idx = np.array([7, 8])  # actuators 7,8 = gripper

        # Objects of interest
        self._bowl_body = m.body("akita_black_bowl_1_main").id
        self._plate_body = m.body("plate_1_main").id
        self._bowl_qposadr = m.jnt_qposadr[m.body("akita_black_bowl_1_main").jntadr[0]]

        # Init qpos from keyframe (if exists) or default
        self._init_q = np.zeros(m.nq, dtype=float)
        self._init_q[:9] = [0.0, 0.0067, -0.1919, -0.0099, -2.4326, -0.0399, 2.1935, 0.0208, -0.0208]

        # Init state tensor (loaded from file)
        self._init_states: Optional[np.ndarray] = None

    def load_init_states(self, task_id: int) -> np.ndarray:
        """Load init states for this task from LIBERO .init file."""
        task_name = TASK_NAMES[task_id]
        init_path = _INIT_DIR / f"{task_name}.init"
        import zipfile, pickle
        with zipfile.ZipFile(str(init_path)) as z:
            with z.open(z.namelist()[0]) as f:
                states = pickle.load(f)
        # Convert to JAX array for efficient vmappable indexing
        self._init_states = jp.array(states)
        return self._init_states

    def reset(self, rng: jax.Array) -> LiberoState:
        """Reset with init state from file if available, else default."""
        if self._init_states is not None:
            # Pick a random init state from the file
            rng, idx_rng = jax.random.split(rng)
            idx = jax.random.randint(idx_rng, (), 0, self._init_states.shape[0])
            init_state = self._init_states[idx]
            # State format: [dummy(1)] + [qpos(48)] + [qvel(43)]
            nq = self._mj_model.nq
            qpos = init_state[1:1 + nq]
            qvel = init_state[1 + nq:1 + nq + self._mj_model.nv]
        else:
            qpos = jp.array(self._init_q, dtype=float)
            qvel = jp.zeros(self._mj_model.nv, dtype=float)

        data = mjx.make_data(
            self._mj_model,
            impl=self._impl,
            naconmax=self._naconmax,
            njmax=self._njmax,
        )
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        info = self._init_info(rng)
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {k: jp.array(0.0) for k in self._reward_keys()}
        metrics["success"] = jp.array(0.0, dtype=float)
        return LiberoState(data, obs, reward, done, metrics, info)

    def _get_predicate(self):
        """Success: bowl on plate (distance < threshold)."""
        plate_body = self._mj_model.body("plate_1_main").id
        return distance_to(self._bowl_body, plate_body, dist=0.08)

    def _get_reward(self, data, info):
        bowl_pos = data.xpos[self._bowl_body]
        plate_pos = data.xpos[self._mj_model.body("plate_1_main").id]
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_bowl = 1 - jp.tanh(5 * jp.linalg.norm(bowl_pos - gripper_pos))
        bowl_plate = 1 - jp.tanh(5 * jp.linalg.norm(bowl_pos - plate_pos))
        return {"gripper_bowl": gripper_bowl, "bowl_plate": bowl_plate}

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        from libero_mjx.obs import build_obs
        return build_obs(
            self._mj_model, data, self._obs_keys(),
            arm_joint_prefix="robot0_joint",
            finger_joint_names=("gripper0_right_finger_joint1", "gripper0_right_finger_joint2"),
        )

    def _make_osc(self):
        from libero_mjx.controllers.osc import OscController
        rest_arm = np.array([self._init_q[i] for i in self._robot_arm_qposadr])
        osc = OscController.from_model(
            self._mj_model,
            site_name="gripper0_right_grip_site",
            arm_joint_prefix="robot0_joint",
            kp=getattr(self._config, "osc_kp", 150.0),
            damping_ratio=getattr(self._config, "osc_damping_ratio", 1.0),
            output_max=getattr(self._config, "osc_output_max", 0.05),
            rest_qpos=rest_arm,
        )
        osc.set_model(self._mjx_model)
        return osc


def _make_task_class(task_id: int):
    """Create a task-specific env class for LIBERO_SPATIAL task task_id."""

    @register_env(f"libero_spatial_{task_id}")
    class _LiberoSpatialTask(LiberoSpatialBase):
        def __init__(self, **kwargs):
            xml_path = _XML_DIR / f"libero_spatial_task{task_id}.xml"
            super().__init__(xml_path=xml_path, **kwargs)
            self.load_init_states(task_id)

    _LiberoSpatialTask.__name__ = f"LiberoSpatialTask{task_id}"
    return _LiberoSpatialTask


# Register all 10 tasks
for _tid in range(10):
    _make_task_class(_tid)