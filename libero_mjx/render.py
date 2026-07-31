"""Batched GPU rendering via mujoco_warp with CPU-matched output.

WarpRenderer encapsulates everything needed to render LIBERO observations on
GPU via mujoco_warp so that they match robosuite's CPU (EGL) rendering closely
enough for a BC policy trained on CPU data to succeed:

  1. Render kernel patch (shadow + haze) via ``render_kernel_patch``
  2. BVH constructor patch (cubql -> lbvh) for HIP/ROCm
  3. Vertical image flip (OpenGL bottom-left origin -> top-left)
  4. Brightness boost (Warp ray tracer is ~85% as bright as CPU renderer)

Usage::

    from libero_mjx.render import WarpRenderer

    renderer = WarpRenderer(
        env.mj_model, n_envs=10,
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    images = renderer.render(mw_model, mw_data)
    # images["agentview_rgb"]:        (N, H, W, 3) uint8 torch on cuda
    # images["robot0_eye_in_hand_rgb"]: (N, H, W, 3) uint8 torch on cuda
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import warp as wp
import mujoco
try:
    import mujoco_warp as mjwarp
except ImportError:
    import sys
    sys.path.insert(
        0, "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party"
    )
    import mujoco_warp as mjwarp
from mujoco_warp._src.bvh import refit_scene_bvh


_IMG_H = 128
_IMG_W = 128

_CAMERA_KEY_MAP = {
    "agentview": "agentview_rgb",
    "robot0_eye_in_hand": "eye_in_hand_rgb",
}


def _patch_bvh():
    """Patch cubql -> lbvh for HIP/ROCm (cubql is unsupported on HIP)."""
    from mujoco_warp._src import io as mwio
    from mujoco_warp._src import bvh as mwbvh
    _orig_mesh = mwbvh.build_mesh_bvh
    _orig_hfield = mwbvh.build_hfield_bvh

    def _patched_mesh(mjm, mid, constructor="sah", leaf_size=2):
        return _orig_mesh(mjm, mid, constructor="lbvh", leaf_size=leaf_size)

    def _patched_hfield(mjm, mid, constructor="sah", leaf_size=2):
        return _orig_hfield(mjm, mid, constructor="lbvh", leaf_size=leaf_size)

    mwbvh.build_mesh_bvh = _patched_mesh
    mwbvh.build_hfield_bvh = _patched_hfield
    mwio.bvh.build_mesh_bvh = _patched_mesh
    mwio.bvh.build_hfield_bvh = _patched_hfield


class WarpRenderer:
    """Batched GPU renderer using mujoco_warp with CPU-matched output.

    Args:
        mj_model: MuJoCo MjModel (CPU) — used for camera/material lookups.
        n_envs: Number of parallel worlds to render.
        img_h: Image height in pixels.
        img_w: Image width in pixels.
        camera_names: Camera names to render (must exist in the model).
        brightness_boost: Multiplier applied to rendered images to match
            CPU renderer brightness (Warp is ~85% as bright). 1.15 is a good
            default; 1.0 disables the boost.
        enabled_geom_groups: Geom groups to render. [1, 2] matches robosuite
            (group 0 = collision geoms, disabled).
        use_textures: Whether to use textures in rendering.
        use_shadows: Whether to render shadows.
        use_skybox: Whether to render the skybox for background.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        n_envs: int = 1,
        img_h: int = _IMG_H,
        img_w: int = _IMG_W,
        camera_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
        brightness_boost: float = 1.15,
        enabled_geom_groups: Sequence[int] = (1, 2),
        use_textures: bool = True,
        use_shadows: bool = True,
        use_skybox: bool = True,
    ):
        self.n_envs = n_envs
        self.img_h = img_h
        self.img_w = img_w
        self.camera_names = list(camera_names)
        self.brightness_boost = brightness_boost
        self._per_cam = img_h * img_w

        self.cam_ids = sorted([mj_model.camera(name).id for name in self.camera_names])
        self.cam_active = [False] * mj_model.ncam
        for cid in self.cam_ids:
            self.cam_active[cid] = True

        _patch_bvh()

        with wp.ScopedDevice("cuda:0"):
            self.mw_model = mjwarp.put_model(mj_model)
            mjd = mujoco.MjData(mj_model)
            mujoco.mj_forward(mj_model, mjd)
            self.mw_data = mjwarp.put_data(mj_model, mjd, nworld=n_envs)
            self.ctx = mjwarp.create_render_context(
                mjm=mj_model,
                nworld=n_envs,
                cam_res=[(img_w, img_h)] * len(self.camera_names),
                render_rgb=[True] * len(self.camera_names),
                render_depth=[False] * len(self.camera_names),
                cam_active=self.cam_active,
                use_textures=use_textures,
                use_shadows=use_shadows,
                use_ambient_lighting=True,
                render_skybox=use_skybox,
                enabled_geom_groups=list(enabled_geom_groups),
            )

        rgb_adr = self.ctx.rgb_adr.numpy()
        self._cam_offsets = {name: int(rgb_adr[i]) for i, name in enumerate(
            sorted(self.camera_names, key=lambda n: mj_model.camera(n).id)
        )}

    def _sync_from_jax(self, state_data):
        """Copy qpos/qvel from a JAX mjx.Data into the warp data buffer."""
        qpos_t = torch.utils.dlpack.from_dlpack(state_data.qpos.__dlpack__())
        qvel_t = torch.utils.dlpack.from_dlpack(state_data.qvel.__dlpack__())
        self.mw_data.qpos = wp.from_torch(qpos_t)
        self.mw_data.qvel = wp.from_torch(qvel_t)

    def render(self, state_data=None, mw_model=None, mw_data=None) -> Dict[str, torch.Tensor]:
        """Render RGB images for all cameras.

        Either pass a JAX ``state.data`` (mjx.Data) via ``state_data``, or
        pass explicit ``mw_model``/``mw_data`` warp objects. Returns a dict
        mapping camera observation keys to (N, H, W, 3) uint8 torch tensors
        on cuda, with vertical flip and brightness boost applied.

        Args:
            state_data: JAX mjx.Data (from LiberoState.data). The qpos/qvel
                are copied into the internal warp data buffer via DLPack.
            mw_model: Explicit warp model (overrides internal one).
            mw_data: Explicit warp data (overrides internal one).

        Returns:
            Dict mapping camera names (via _CAMERA_KEY_MAP) to image tensors.
        """
        if mw_model is None:
            mw_model = self.mw_model
        if mw_data is None:
            mw_data = self.mw_data
            if state_data is not None:
                self._sync_from_jax(state_data)

        mjwarp.forward(mw_model, mw_data)
        refit_scene_bvh(mw_model, mw_data, self.ctx)
        mjwarp.render(mw_model, mw_data, self.ctx)

        rgb = wp.to_torch(self.ctx.rgb_data).to(torch.int32)
        result = {}
        for cam_name in self.camera_names:
            offset = self._cam_offsets[cam_name]
            raw = rgb[:, offset:offset + self._per_cam].reshape(
                self.n_envs, self.img_h, self.img_w
            )
            img = torch.stack(
                [(raw >> 16) & 0xFF, (raw >> 8) & 0xFF, raw & 0xFF], dim=-1
            ).to(torch.uint8)
            img = img.flip(dims=[1])  # vertical flip: OpenGL bottom-left -> top-left
            if self.brightness_boost != 1.0:
                img = torch.clamp(img.float() * self.brightness_boost, 0, 255).to(torch.uint8)
            key = _CAMERA_KEY_MAP.get(cam_name, f"{cam_name}_rgb")
            result[key] = img.to("cuda")
        return result