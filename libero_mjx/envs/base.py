"""LiberoMjxEnv: base environment class for LIBERO tasks on MuJoCo Warp.

Provides batched reset/step, state save/restore (tensor copy, no subprocess),
and success predicate evaluation.  Built on the mujoco_playground MjxEnv
pattern but adapted for LIBERO's action space (OSC Cartesian delta EE).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Optional, Tuple, Union

from etils import epath
import jax
import jax.numpy as jp
from flax import struct
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from libero_mjx.controllers.osc import OscController
from libero_mjx.obs import build_obs
from libero_mjx.predicates.spatial import PredicateFn


class LiberoState(struct.PyTreeNode):
    """Environment state (vmappable)."""

    data: mjx.Data
    obs: jax.Array
    reward: jax.Array
    done: jax.Array
    metrics: Dict[str, jax.Array]
    info: Dict[str, Any]


class LiberoMjxEnv(abc.ABC):
    """Base class for GPU-parallel LIBERO environments via MuJoCo Warp."""

    def __init__(
        self,
        xml_path: Union[str, epath.Path],
        config: Optional[config_dict.ConfigDict] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        impl: str = "warp",
        naconmax: int = 12 * 8192,
        njmax: int = 44,
    ):
        self._config = config.lock() if config else config_dict.ConfigDict()
        if config_overrides:
            self._config.update_from_flattened_dict(config_overrides)

        self._ctrl_dt = getattr(self._config, "ctrl_dt", 0.05)
        self._sim_dt = getattr(self._config, "sim_dt", 0.002)
        self._impl = impl
        self._naconmax = naconmax
        self._njmax = njmax

        xml_path = epath.Path(xml_path) if not isinstance(xml_path, epath.Path) else xml_path
        self._xml_path = xml_path.as_posix()
        xml = xml_path.read_text()
        self._model_assets = self._get_assets()
        mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
        mj_model.opt.timestep = self.sim_dt
        self._mj_model = mj_model
        self._mjx_model = mjx.put_model(mj_model, impl=impl)

        self._action_scale = getattr(self._config, "action_scale", 0.05)
        self._episode_length = getattr(self._config, "episode_length", 600)
        self._post_init()
        self._osc = self._make_osc()

    # -- subclass hooks ------------------------------------------------

    @abc.abstractmethod
    def _post_init(self) -> None:
        """Set up body/site/joint indices after model load."""

    @abc.abstractmethod
    def _get_assets(self) -> Dict[str, bytes]:
        """Return mesh/xml asset dict for MjModel.from_xml_string."""

    @abc.abstractmethod
    def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, jax.Array]:
        """Per-step reward components (dict of scalars)."""

    @abc.abstractmethod
    def _get_predicate(self) -> PredicateFn:
        """Return the success predicate for this task."""

    def _make_osc(self) -> OscController:
        osc = OscController.from_model(
            self._mj_model,
            kp=getattr(self._config, "osc_kp", 150.0),
            damping_ratio=getattr(self._config, "osc_damping_ratio", 1.0),
            output_max=getattr(self._config, "osc_output_max", 0.05),
        )
        osc.set_model(self._mjx_model)
        return osc

    # -- reset / step --------------------------------------------------

    def reset(self, rng: jax.Array) -> LiberoState:
        init_q = jp.array(self._init_q, dtype=float)
        data = mjx.make_data(
            self._mj_model,
            impl=self._impl,
            naconmax=self._naconmax,
            njmax=self._njmax,
        )
        data = data.replace(qpos=init_q, qvel=jp.zeros(self._mjx_model.nv, dtype=float))
        data = self._reset_extra(data, rng)
        # Populate kinematics (site_xpos, xmat, jacobians) for OSC + obs
        data = mjx.forward(self._mjx_model, data)
        info = self._init_info(rng)
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        metrics = {k: jp.array(0.0) for k in self._reward_keys()}
        metrics["success"] = jp.array(0.0, dtype=float)
        return LiberoState(data, obs, reward, done, metrics, info)

    def _reset_extra(self, data: mjx.Data, rng: jax.Array) -> mjx.Data:
        return data

    def _init_info(self, rng: jax.Array) -> Dict[str, Any]:
        return {"rng": rng, "step": jp.array(0, dtype=int)}

    def step(self, state: LiberoState, action: jax.Array) -> LiberoState:
        action = jp.clip(action, -1.0, 1.0)
        arm_action = action[..., :6]
        gripper_action = action[..., 6:7]
        ctrl = self._osc.compute_torques(state.data, arm_action)

        # Gripper: map [-1,1] to finger target [0.0, 0.04] (open=0.04)
        finger_target = (gripper_action + 1.0) * 0.02
        finger_idx = self._gripper_ctrl_idx
        ctrl = ctrl.at[..., finger_idx].set(finger_target)

        def single_step(data, _):
            data = data.replace(ctrl=ctrl)
            data = mjx.step(self._mjx_model, data)
            return data, None

        data = jax.lax.scan(single_step, state.data, (), self.n_substeps)[0]
        info = {**state.info, "step": state.info["step"] + 1}
        raw_rewards = self._get_reward(data, info)
        reward = jp.clip(sum(raw_rewards.values()), -1e4, 1e4)
        success = self._get_predicate()(data).astype(float)
        done = jp.where(
            (success > 0.5) | (info["step"] >= self._episode_length),
            jp.array(1.0),
            jp.array(0.0),
        )
        metrics = {**state.metrics, **raw_rewards, "success": success}
        obs = self._get_obs(data, info)
        return LiberoState(data, obs, reward, done, metrics, info)

    # -- state save/restore --------------------------------------------

    def get_sim_state(self, state: LiberoState) -> Dict[str, jax.Array]:
        return {
            "qpos": state.data.qpos,
            "qvel": state.data.qvel,
            "ctrl": state.data.ctrl,
            "act": state.data.act,
            "mocap_pos": state.data.mocap_pos,
            "mocap_quat": state.data.mocap_quat,
        }

    def set_sim_state(self, state: LiberoState, sim_state: Dict[str, jax.Array]) -> LiberoState:
        data = state.data
        for k in ("qpos", "qvel", "ctrl", "act", "mocap_pos", "mocap_quat"):
            if k in sim_state:
                data = data.replace(**{k: sim_state[k]})
        # Recompute kinematics from restored qpos/qvel
        data = mjx.forward(self._mjx_model, data)
        return state.replace(data=data)

    # -- obs ------------------------------------------------------------

    def _get_obs(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
        return build_obs(self._mj_model, data, self._obs_keys())

    def _obs_keys(self) -> list[str]:
        return ["gripper_states", "joint_states"]

    # -- helpers --------------------------------------------------------

    def _reward_keys(self) -> list[str]:
        return list(self._get_reward(
            mjx.make_data(self._mj_model, impl=self._impl),
            {"step": jp.array(0)},
        ).keys())

    @property
    def dt(self) -> float:
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        return self._sim_dt

    @property
    def n_substeps(self) -> int:
        return int(round(self.dt / self.sim_dt))

    @property
    def action_size(self) -> int:
        return 7

    @property
    def observation_size(self) -> int:
        abstract = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        return abstract.obs.shape[-1]

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def osc(self) -> OscController:
        return self._osc