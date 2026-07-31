#!/usr/bin/env python3
"""Validate mujoco_warp renderer on local RDNA 4 GPU (gfx1201).

Loads a LIBERO spatial task XML, creates a Warp render context,
renders RGB + depth for agentview and eye-in-hand cameras, and saves
images + stats for visual validation.

No LIBERO benchmark data needed — just the XML model.
"""
import sys, os

REPO = os.environ.get("REPO_PATH", "/workspace/libero-mjx")
sys.path.insert(0, REPO)
sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party")
os.environ.setdefault("JAX_PLATFORMS", "rocm")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import mujoco
import warp as wp
import jax
import mujoco_warp as mjwarp

print(f"JAX version:   {jax.__version__}")
print(f"JAX backend:   {jax.default_backend()}")
print(f"JAX devices:   {jax.devices()}")
print(f"Warp version:  {wp.config.version}")
print(f"Warp devices:  {wp.get_devices()}")

from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()

from mujoco_warp._src import io as mwio
from mujoco_warp._src import bvh as mwbvh
from mujoco_warp._src.bvh import refit_scene_bvh

_o_mesh = mwbvh.build_mesh_bvh
_o_hfield = mwbvh.build_hfield_bvh
def _pm(mjm, mid, constructor="sah", leaf_size=2):
    return _o_mesh(mjm, mid, constructor="lbvh", leaf_size=leaf_size)
def _ph(mjm, mid, constructor="sah", leaf_size=2):
    return _o_hfield(mjm, mid, constructor="lbvh", leaf_size=leaf_size)
mwbvh.build_mesh_bvh = _pm
mwbvh.build_hfield_bvh = _ph
mwio.bvh.build_mesh_bvh = _pm
mwio.bvh.build_hfield_bvh = _ph

from libero_mjx import texture_patch

IMG_H, IMG_W = 128, 128
xml_path = os.path.join(REPO, "libero_mjx/assets/xml/libero_spatial_task0.xml")
out_dir = os.path.join(REPO, "debug_imgs")
os.makedirs(out_dir, exist_ok=True)

print(f"\nLoading model: {xml_path}")
m = mujoco.MjModel.from_xml_path(xml_path)
mjd = mujoco.MjData(m)
mujoco.mj_forward(m, mjd)
print(f"  ngeom={m.ngeom} nmat={m.nmat} ntex={m.ntex} ncam={m.ncam} nbody={m.nbody}")

av_cam = m.camera("agentview").id
eye_cam = m.camera("robot0_eye_in_hand").id
cam_active = [False] * m.ncam
cam_active[av_cam] = True
cam_active[eye_cam] = True
active_cam_ids = sorted([av_cam, eye_cam])
AV_IDX = active_cam_ids.index(av_cam)
EYE_IDX = active_cam_ids.index(eye_cam)
print(f"  agentview cam id={av_cam}, eye_in_hand cam id={eye_cam}")

for i in range(m.ngeom):
    if m.geom_rgba[i][3] < 0.99:
        m.geom_group[i] = 3

for mid in range(m.nmat):
    texid = m.mat_texid[mid, 1]
    if texid >= 0 and m.tex_type[texid] == 1:
        tex_adr = m.tex_adr[texid]
        tw, th, nc = m.tex_width[texid], m.tex_height[texid], m.tex_nchannel[texid]
        face_size = tw * th * nc
        if tex_adr + face_size <= len(m.tex_data):
            face_data = m.tex_data[tex_adr:tex_adr + face_size]
            if nc >= 3:
                ar, ag, ab = (face_data[0::nc].mean() / 255.0,
                              face_data[1::nc].mean() / 255.0,
                              face_data[2::nc].mean() / 255.0)
            else:
                ar = ag = ab = face_data.mean() / 255.0
            boost = 1.13
            m.mat_texid[mid, :] = -1
            m.mat_rgba[mid] = [min(ar*boost, 1), min(ag*boost, 1), min(ab*boost, 1), 1.0]

for mid in range(m.nmat):
    if m.mat_specular[mid] > 0.3 and m.mat_shininess[mid] > 0.9:
        m.mat_emission[mid] = max(m.mat_emission[mid], 0.5)

name = ''
for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
    if 'burner' in name.lower() and 'collision' not in name.lower():
        m.geom_matid[i] = -1
        m.geom_rgba[i] = [0.35, 0.35, 0.35, 1.0]

print("\n--- CPU reference render (MuJoCo EGL) ---")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
renderer = mujoco.Renderer(m, height=IMG_H, width=IMG_W)
renderer.update_scene(mjd, camera=av_cam)
cpu_av = renderer.render()
renderer.update_scene(mjd, camera=eye_cam)
cpu_eye = renderer.render()
print(f"  CPU agentview:  shape={cpu_av.shape} mean={cpu_av.mean():.1f} std={cpu_av.std():.1f} min={cpu_av.min()} max={cpu_av.max()}")
print(f"  CPU eye:        shape={cpu_eye.shape} mean={cpu_eye.mean():.1f} std={cpu_eye.std():.1f}")

from PIL import Image
Image.fromarray(cpu_av).save(os.path.join(out_dir, "validate_cpu_av.png"))
Image.fromarray(cpu_eye).save(os.path.join(out_dir, "validate_cpu_eye.png"))
print(f"  Saved CPU images to {out_dir}/validate_cpu_*.png")

print("\n--- Warp GPU render (mujoco_warp on gfx1201) ---")
N_ENVS = 4
with wp.ScopedDevice("cuda:0"):
    mw_model = mjwarp.put_model(m)
    mw_data = mjwarp.put_data(m, mjd, nworld=N_ENVS)
    render_ctx = mjwarp.create_render_context(
        mjm=m, nworld=N_ENVS,
        cam_res=[(IMG_W, IMG_H), (IMG_W, IMG_H)],
        render_rgb=[True, True],
        render_depth=[True, True],
        cam_active=cam_active,
        use_textures=True,
        use_shadows=True,
        use_ambient_lighting=True,
        render_skybox=True,
        enabled_geom_groups=[1, 2],
    )
    print(f"  Render context created (nworld={N_ENVS})")

    import time
    t0 = time.time()
    for _ in range(3):
        mjwarp.render(mw_model, mw_data, render_ctx)
        wp.synchronize()
    t_render = (time.time() - t0) / 3
    print(f"  Render time: {t_render*1000:.1f} ms ({N_ENVS} envs x 2 cams = {N_ENVS*2} frames)")

    rgb = wp.to_torch(render_ctx.rgb_data).to(torch.int32) if False else render_ctx.rgb_data.numpy()
    depth = render_ctx.depth_data.numpy()
    rgb_adr = render_ctx.rgb_adr.numpy()
    depth_adr = render_ctx.depth_adr.numpy()

    AV_OFF = int(rgb_adr[AV_IDX])
    EYE_OFF = int(rgb_adr[EYE_IDX])
    per_cam = IMG_H * IMG_W

    warp_av_imgs = []
    warp_eye_imgs = []
    for w in range(min(N_ENVS, 2)):
        av_raw = rgb[w, AV_OFF:AV_OFF+per_cam].reshape(IMG_H, IMG_W).astype(np.uint32)
        eye_raw = rgb[w, EYE_OFF:EYE_OFF+per_cam].reshape(IMG_H, IMG_W).astype(np.uint32)
        av_img = np.stack([(av_raw >> 16) & 0xFF, (av_raw >> 8) & 0xFF, av_raw & 0xFF], axis=-1).astype(np.uint8)
        eye_img = np.stack([(eye_raw >> 16) & 0xFF, (eye_raw >> 8) & 0xFF, eye_raw & 0xFF], axis=-1).astype(np.uint8)
        warp_av_imgs.append(av_img)
        warp_eye_imgs.append(eye_img)
        print(f"  Warp env{w} agentview: mean={av_img.mean():.1f} std={av_img.std():.1f} min={av_img.min()} max={av_img.max()}")
        print(f"  Warp env{w} eye:       mean={eye_img.mean():.1f} std={eye_img.std():.1f}")

        Image.fromarray(av_img).save(os.path.join(out_dir, f"validate_warp_av_env{w}.png"))
        Image.fromarray(eye_img).save(os.path.join(out_dir, f"validate_warp_eye_env{w}.png"))

    d_av_off = int(depth_adr[AV_IDX])
    d_eye_off = int(depth_adr[EYE_IDX])
    for w in range(min(N_ENVS, 2)):
        av_depth = depth[w, d_av_off:d_av_off+per_cam].reshape(IMG_H, IMG_W)
        eye_depth = depth[w, d_eye_off:d_eye_off+per_cam].reshape(IMG_H, IMG_W)
        print(f"  Warp env{w} depth av:  min={av_depth.min():.3f} max={av_depth.max():.3f} mean={av_depth.mean():.3f}")
        print(f"  Warp env{w} depth eye: min={eye_depth.min():.3f} max={eye_depth.max():.3f} mean={eye_depth.mean():.3f}")

print("\n--- Render comparison (CPU vs Warp) ---")
warp_av = warp_av_imgs[0]
warp_eye = warp_eye_imgs[0]

for label, cpu_img, warp_img in [("agentview", cpu_av, warp_av), ("eye_in_hand", cpu_eye, warp_eye)]:
    diff = np.abs(cpu_img.astype(float) - warp_img.astype(float))
    diff_flip = np.abs(cpu_img[::-1].astype(float) - warp_img.astype(float))
    rmse = np.sqrt((diff ** 2).mean())
    rmse_flip = np.sqrt((diff_flip ** 2).mean())
    print(f"  {label}: RMSE={rmse:.1f} (no flip), RMSE={rmse_flip:.1f} (cpu flip vs warp)")
    print(f"    CPU  mean={cpu_img.mean():.1f}  Warp mean={warp_img.mean():.1f}")

    side_by_side = np.concatenate([cpu_img, warp_img], axis=1)
    Image.fromarray(side_by_side).save(os.path.join(out_dir, f"validate_compare_{label}.png"))

print(f"\nAll images saved to {out_dir}/")
print("=== Warp render validation PASSED ===")