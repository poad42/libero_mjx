"""Warp render context for batched GPU rendering via mujoco_warp.

Provides zero-copy RGB observation rendering compatible with LIBERO's BC policy.
Images are rendered on GPU via mjwarp.render and exported via DLPack to PyTorch
without any numpy/host round-trips.

Usage:
    from libero_mjx.render import RenderContext

    ctx = RenderContext(mj_model, nworld=4, img_h=128, img_w=128,
                         camera_names=["agentview", "robot0_eye_in_hand"])
    rgb = ctx.render(mw_data)  # (N, n_cam, H, W, 3) uint8 on GPU
    # Or via DLPack:
    torch_imgs = ctx.render_torch(mw_data)  # (N, n_cam, H, W, 3) torch uint8
"""
from __future__ import annotations

import numpy as np
import warp as wp
import mujoco
import mujoco_warp as mjwarp
from typing import Sequence, Optional, Tuple


class RenderContext:
    """Batched GPU render context using mujoco_warp.

    Wraps mjwarp.create_render_context with lbvh patch for HIP/ROCm.
    Renders RGB images for multiple cameras in parallel across N worlds.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        nworld: int = 1,
        img_h: int = 128,
        img_w: int = 128,
        camera_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
        use_textures: bool = True,
        use_shadows: bool = False,
    ):
        self.nworld = nworld
        self.img_h = img_h
        self.img_w = img_w
        self.camera_names = list(camera_names)

        # Resolve camera IDs
        self.cam_ids = sorted([mj_model.camera(name).id for name in camera_names])
        self.cam_active = [False] * mj_model.ncam
        for cid in self.cam_ids:
            self.cam_active[cid] = True

        # Patch cubql → lbvh for HIP (needed before create_render_context)
        from mujoco_warp._src import io as mwio
        from mujoco_warp._src import bvh as mwbvh
        _orig_mesh = mwbvh.build_mesh_bvh
        _orig_hfield = mwbvh.build_hfield_bvh
        def _patched_mesh(mjm, mid, constructor="sah", leaf_size=2):
            return _orig_mesh(mjm, mid, constructor="lbvh", leaf_size=leaf_size)
        def _patched_hfield(mjm, mid, constructor="sah", leaf_size=2):
            return _orig_hfield(mjm, mid, constructor="lbvh", leaf_size=leaf_size)
        mwio.bvh.build_mesh_bvh = _patched_mesh
        mwio.bvh.build_hfield_bvh = _patched_hfield

        with wp.ScopedDevice("cuda:0"):
            self.mw_model = mjwarp.put_model(mj_model)
            self.ctx = mjwarp.create_render_context(
                mjm=mj_model,
                nworld=nworld,
                cam_res=[(img_w, img_h)] * len(camera_names),
                render_rgb=[True] * len(camera_names),
                render_depth=[False] * len(camera_names),
                cam_active=self.cam_active,
                use_textures=use_textures,
                use_shadows=use_shadows,
            )

    def render(self, mw_data) -> np.ndarray:
        """Render RGB images. Returns (N, n_cam, H, W, 3) uint8 numpy array.

        Args:
            mw_data: mujoco_warp Data (from mjwarp.put_data or mjx.Data._impl)
        Returns:
            (nworld, n_cam, H, W, 3) uint8 numpy array on host
        """
        mjwarp.render(self.mw_model, mw_data, self.ctx)
        wp.synchronize()

        # Extract RGB from render context buffer
        # rgb_data is (nworld, n_cam * H * W) uint32
        rgb_flat = self.ctx.rgb_data.numpy() if hasattr(self.ctx.rgb_data, 'numpy') else np.array(self.ctx.rgb_data)

        n_cam = len(self.camera_names)
        per_cam = self.img_h * self.img_w
        images = []
        for cam_idx in range(n_cam):
            raw = rgb_flat[:, cam_idx * per_cam : (cam_idx + 1) * per_cam]
            raw = raw.reshape(self.nworld, self.img_h, self.img_w)
            r = (raw & 0xFF).astype(np.uint8)
            g = ((raw >> 8) & 0xFF).astype(np.uint8)
            b = ((raw >> 16) & 0xFF).astype(np.uint8)
            img = np.stack([r, g, b], axis=-1)  # (N, H, W, 3)
            images.append(img)

        return np.stack(images, axis=1)  # (N, n_cam, H, W, 3)

    def render_torch(self, mw_data, device="cuda"):
        """Render and return as PyTorch tensor via DLPack (zero-copy).

        Args:
            mw_data: mujoco_warp Data
            device: torch device for output
        Returns:
            (nworld, n_cam, H, W, 3) uint8 torch tensor
        """
        import torch
        rgb_np = self.render(mw_data)
        return torch.from_numpy(rgb_np.copy()).to(device)