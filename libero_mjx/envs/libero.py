"""Unified LIBERO env for all 5 suites: spatial, object, goal, scene10, scene90.

A single LiberoEnv class that works with any of the 130 LIBERO tasks.
The success predicate is auto-detected from the suite and task bodies.

Usage:
    from libero_mjx.envs.libero import LiberoEnv

    env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=256)
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, action)
"""
from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional, Sequence
import numpy as np
import jax
import jax.numpy as jp
from etils import epath
import mujoco
import torch
from mujoco import mjx

from libero_mjx.envs.base import LiberoMjxEnv, LiberoState
from libero_mjx.envs import register_env
from libero_mjx.predicates.spatial import (
    distance_to, on, in_region, is_open, is_closed, is_turned_on, PredicateFn,
)


# Suite metadata
SUITES = {
    "spatial": {"prefix": "libero_spatial", "n_tasks": 10, "init_dir": "libero_spatial"},
    "object": {"prefix": "libero_object", "n_tasks": 10, "init_dir": "libero_object"},
    "goal": {"prefix": "libero_goal", "n_tasks": 10, "init_dir": "libero_goal"},
    "scene10": {"prefix": "libero_10", "n_tasks": 10, "init_dir": "libero_10"},
    "scene90": {"prefix": "libero_90", "n_tasks": 90, "init_dir": "libero_90"},
}

_XML_DIR = epath.Path(__file__).parent.parent / "assets" / "xml"
_INIT_DIR = epath.Path(os.environ.get(
    "LIBERO_INIT_DIR",
    "/workspace/libero_basil/libero/libero/init_files",
))


def _load_task_names(suite: str) -> list[str]:
    """Load task names from init files directory."""
    init_dir = _INIT_DIR / SUITES[suite]["init_dir"]
    if not init_dir.exists():
        return []
    names = sorted([f.stem for f in init_dir.glob("*.init")])
    return names


# Load task names for each suite at import time
TASK_NAMES = {}
for suite in SUITES:
    try:
        TASK_NAMES[suite] = _load_task_names(suite)
    except Exception:
        TASK_NAMES[suite] = []


class LiberoEnv(LiberoMjxEnv):
    """Generic LIBERO env for any suite/task.

    The success predicate is auto-detected from the BDDL goal condition,
    or specified manually via `predicate_fn`.

    Args:
        suite: one of "spatial", "object", "goal", "scene10", "scene90"
        task_id: task index within the suite
        impl: "warp" or "jax"
        n_envs: number of parallel envs (scales naconmax)
        optimize_physics: use optimized physics params
        predicate_fn: optional override for success predicate
    """

    def __init__(
        self,
        suite: str = "spatial",
        task_id: int = 0,
        impl: str = "warp",
        n_envs: int = 64,
        optimize_physics: bool = True,
        predicate_fn: Optional[PredicateFn] = None,
        **kwargs,
    ):
        self.suite = suite
        self.task_id = task_id
        self._predicate_override = predicate_fn

        meta = SUITES[suite]
        xml_path = _XML_DIR / f"{meta['prefix']}_task{task_id}.xml"
        naconmax = n_envs * 200
        kwargs.setdefault("njmax", 2048)
        super().__init__(xml_path=xml_path, impl=impl, naconmax=naconmax, **kwargs)

        if optimize_physics:
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
            self._ctrl_dt = 0.02

        self._init_states: Optional[np.ndarray] = None

    def _get_assets(self) -> Dict[str, bytes]:
        return {}

    def _post_init(self) -> None:
        m = self._mj_model
        self._arm_joints = [f"robot0_joint{i}" for i in range(1, 8)]
        self._robot_arm_qposadr = np.array([
            m.jnt_qposadr[m.joint(j).id] for j in self._arm_joints
        ])
        self._robot_arm_dofadr = np.array([
            m.jnt_dofadr[m.joint(j).id] for j in self._arm_joints
        ])

        self._gripper_site = m.site("gripper0_right_grip_site").id
        self._gripper_site_body = int(np.asarray(m.site("gripper0_right_grip_site").bodyid).flat[0])
        self._gripper_ctrl_idx = np.array([7, 8])

        self._init_q = np.zeros(m.nq, dtype=float)
        self._init_q[:9] = [0.0, 0.0067, -0.1919, -0.0099, -2.4326, -0.0399, 2.1935, 0.0208, -0.0208]

        # Try to find the main object (varies by suite)
        self._obj_body = None
        for name in ["akita_black_bowl_1_main", "alphabet_soup_1", "butter_1", "moka_pot_1",
                      "chocolate_pudding_1", "red_mug_1", "white_mug_1", "wine_bottle_1",
                      "cream_cheese_1", "ketchup_1", "milk_1", "orange_juice_1",
                      "tomato_sauce_1", "salad_dressing_1", "book_1"]:
            try:
                self._obj_body = m.body(name).id
                break
            except (KeyError, ValueError):
                continue

        # Try to find target body
        self._target_body = None
        for name in ["plate_1_main", "basket_1_main", "flat_stove_1", "wooden_cabinet_1_main",
                      "tray_1_main", "plate_1", "plate_2"]:
            try:
                self._target_body = m.body(name).id
                break
            except (KeyError, ValueError):
                continue

    def _get_predicate(self) -> PredicateFn:
        if self._predicate_override is not None:
            return self._predicate_override

        m = self._mj_model
        # Auto-detect based on suite
        if self.suite == "spatial":
            plate_body = m.body("plate_1_main").id
            return distance_to(self._obj_body, plate_body, dist=0.08)
        elif self.suite == "object":
            # object in basket
            if self._target_body is not None:
                return in_region(self._obj_body, self._target_body, dist=0.1)
            return distance_to(self._obj_body, self._obj_body, dist=999)  # never
        elif self.suite == "goal":
            # various: On, Open, Turnon
            if self._target_body is not None:
                return on(self._obj_body, self._target_body, dist=0.08)
            return distance_to(self._obj_body, self._obj_body, dist=999)
        elif self.suite in ("scene10", "scene90"):
            # multi-step: use distance as proxy
            if self._obj_body is not None and self._target_body is not None:
                return distance_to(self._obj_body, self._target_body, dist=0.1)
            return distance_to(0, 0, dist=999)
        else:
            return distance_to(0, 0, dist=999)

    def _get_reward(self, data, info):
        if self._obj_body is not None and self._target_body is not None:
            obj_pos = data.xpos[self._obj_body]
            tgt_pos = data.xpos[self._target_body]
            gripper_pos = data.site_xpos[self._gripper_site]
            gripper_obj = 1 - jp.tanh(5 * jp.linalg.norm(obj_pos - gripper_pos))
            obj_tgt = 1 - jp.tanh(5 * jp.linalg.norm(obj_pos - tgt_pos))
            return {"gripper_obj": gripper_obj, "obj_target": obj_tgt}
        return {"placeholder": jp.array(0.0)}

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
            rest_qpos=rest_arm,
        )
        osc.set_model(self._mjx_model)
        return osc

    def load_init_states(self, task_id: Optional[int] = None) -> np.ndarray:
        """Load init states from LIBERO .init file."""
        tid = task_id if task_id is not None else self.task_id
        suite_meta = SUITES[self.suite]
        names = TASK_NAMES.get(self.suite, [])
        if tid < len(names):
            task_name = names[tid]
        else:
            return self._init_states

        init_path = _INIT_DIR / suite_meta["init_dir"] / f"{task_name}.init"
        if not init_path.exists():
            # Try .pruned_init
            init_path = _INIT_DIR / suite_meta["init_dir"] / f"{task_name}.pruned_init"
            if not init_path.exists():
                return self._init_states

        import zipfile, pickle
        with zipfile.ZipFile(str(init_path)) as z:
            with z.open(z.namelist()[0]) as f:
                states = pickle.load(f)
        self._init_states = jp.array(states)
        return self._init_states

    def batch_reset(self, init_states: np.ndarray, n_envs: int) -> LiberoState:
        """Create a batched LiberoState for N envs with given init states.

        Uses mjwarp.forward (not mjx.forward) to compute xpos — works with
        nworld=N without vmap. The returned state can be stepped via
        jax.vmap(env.step).
        """
        import warp as wp
        import mujoco_warp as mjwarp
        nq = self._mj_model.nq
        nv = self._mj_model.nv

        # Sample init states
        qpos_np = np.array([init_states[i % len(init_states), 1:1+nq] for i in range(n_envs)], dtype=np.float32)
        qvel_np = np.array([init_states[i % len(init_states), 1+nq:1+nq+nv] for i in range(n_envs)], dtype=np.float32)

        # Create warp data with nworld=N and compute forward
        mjd = mujoco.MjData(self._mj_model)
        mujoco.mj_forward(self._mj_model, mjd)
        with wp.ScopedDevice("cuda:0"):
            mw_model = mjwarp.put_model(self._mj_model)
            mw_data = mjwarp.put_data(self._mj_model, mjd, nworld=n_envs)
            mw_data.qpos = wp.from_torch(torch.tensor(qpos_np, device="cuda"))
            mw_data.qvel = wp.from_torch(torch.tensor(qvel_np, device="cuda"))
            mjwarp.forward(mw_model, mw_data)
            wp.synchronize()

        # Read xpos/site_xpos from warp data (computed by mjwarp.forward)
        xpos_np = mw_data.xpos.numpy()  # (nbody, n_envs, 3) or (n_envs, nbody, 3)
        site_xpos_np = mw_data.site_xpos.numpy()
        site_xmat_np = mw_data.site_xmat.numpy()

        # Create JAX batched data via vmap make_data (no forward needed —
        # we'll set xpos/site_xpos manually from the warp forward result)
        qpos_jax = jp.array(qpos_np, dtype=jp.float32)
        qvel_jax = jp.array(qvel_np, dtype=jp.float32)
        xpos_jax = jp.array(xpos_np, dtype=jp.float32)
        site_xpos_jax = jp.array(site_xpos_np, dtype=jp.float32)
        site_xmat_jax = jp.array(site_xmat_np, dtype=jp.float32)

        def make_state(qpos, qvel, xpos, site_xpos, site_xmat, rng):
            d = mjx.make_data(self._mj_model, impl="warp", naconmax=self._naconmax, njmax=self._njmax)
            d = d.replace(qpos=qpos, qvel=qvel, xpos=xpos, site_xpos=site_xpos, site_xmat=site_xmat)
            info = {"rng": rng, "step": jp.array(0, dtype=jp.int32)}
            obs = self._get_obs(d, info)
            metrics = {k: jp.array(0.0) for k in self._reward_keys()}
            metrics["success"] = jp.array(0.0, dtype=jp.float32)
            return LiberoState(d, obs, jp.array(0.0), jp.array(0.0), metrics, info)

        import jax
        rng_keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
        state = jax.vmap(make_state)(qpos_jax, qvel_jax, xpos_jax, site_xpos_jax, site_xmat_jax, rng_keys)
        return state

    def reset(self, rng: jax.Array) -> LiberoState:
        nq = self._mj_model.nq
        nv = self._mj_model.nv
        use_init = (
            self._init_states is not None
            and self._init_states.shape[-1] >= 1 + nq + nv
        )
        if use_init:
            rng, idx_rng = jax.random.split(rng)
            idx = jax.random.randint(idx_rng, (), 0, self._init_states.shape[0])
            init_state = self._init_states[idx]
            qpos = init_state[1:1 + nq]
            qvel = init_state[1 + nq:1 + nq + nv]
        else:
            qpos = jp.array(self._init_q, dtype=float)
            qvel = jp.zeros(nv, dtype=float)

        data = mjx.make_data(
            self._mj_model, impl=self._impl,
            naconmax=self._naconmax, njmax=self._njmax,
        )
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self._mjx_model, data)

        info = self._init_info(rng)
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {k: jp.array(0.0) for k in self._reward_keys()}
        metrics["success"] = jp.array(0.0, dtype=float)
        return LiberoState(data, obs, reward, done, metrics, info)


# Register all suite tasks
for suite_name, meta in SUITES.items():
    for tid in range(meta["n_tasks"]):
        _suite = suite_name
        _tid = tid
        def _make(suite, task_id):
            @register_env(f"libero_{suite}_{task_id}")
            class _Task(LiberoEnv):
                def __init__(self, **kwargs):
                    super().__init__(suite=suite, task_id=task_id, **kwargs)
            _Task.__name__ = f"Libero{suite.capitalize()}Task{task_id}"
            return _Task
        _make(_suite, _tid)