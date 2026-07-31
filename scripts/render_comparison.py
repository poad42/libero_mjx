"""Generate CPU vs Warp render comparison images for docs."""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw

import importlib.util
spec = importlib.util.spec_from_file_location(
    "render_kernel_patch",
    os.path.join(os.path.dirname(__file__), "..", "libero_mjx", "render_kernel_patch.py"),
)
_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_mod)
_mod.patch_render_kernel()

import jax.numpy as jp
import mujoco
import mujoco.mjx.warp as mjwarp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()

from libero_mjx.envs.libero import LiberoEnv
from libero_mjx.render import WarpRenderer
from libero_mjx.robosuite_patch import patch_robosuite
patch_robosuite()

SUITE = "spatial"
TASK_ID = 0
N_ENVS = 1
IMG_H = 128
IMG_W = 128
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "images")


def render_cpu():
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    config_dir = os.path.join(os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
                            "libero/configs")
    benchmark_name = {
        "spatial": "libero_spatial",
        "object": "libero_object",
        "goal": "libero_goal",
        "scene10": "libero_10",
        "scene90": "libero_90",
    }[SUITE]

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"seed=0",
                f"benchmark_name={benchmark_name}",
                "policy=bc_transformer_policy",
                "lifelong=single_task",
                f"data.task_order_index={TASK_ID}",
            ],
        )
    cfg.folder = cfg.folder or get_libero_path("datasets")
    cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")

    benchmark = get_benchmark(benchmark_name)(cfg.data.task_order_index)
    task = benchmark.get_task(TASK_ID)
    init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    init_states = __import__("torch").load(init_states_path, weights_only=False)

    env_args = {
        "bddl_file_name": os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file),
        "camera_heights": IMG_H,
        "camera_widths": IMG_W,
    }
    env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args)])
    env.reset()
    env.seed(0)
    obs = env.set_init_state(init_states[0:1])
    dummy = np.zeros((1, 7))
    for _ in range(5):
        obs, _, _, _ = env.step(dummy)

    o = obs[0]
    av = o["agentview_image"].copy()
    eye = o["robot0_eye_in_hand_image"].copy()
    env.close()
    return av, eye


def render_warp():
    env = LiberoEnv(suite=SUITE, task_id=TASK_ID, impl="warp", n_envs=N_ENVS, optimize_physics=False)
    env.load_init_states(TASK_ID)

    import torch as torch_mod
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from hydra import initialize_config_dir, compose

    benchmark_name = {
        "spatial": "libero_spatial", "object": "libero_object",
        "goal": "libero_goal", "scene10": "libero_10", "scene90": "libero_90",
    }[SUITE]
    config_dir = os.path.join(os.environ.get("LIBERO_BASIL_PATH", "/workspace/libero_basil"),
                            "libero/configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["seed=0", f"benchmark_name={benchmark_name}", "policy=bc_transformer_policy", "lifelong=single_task", f"data.task_order_index={TASK_ID}"])
    cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
    benchmark = get_benchmark(benchmark_name)(TASK_ID)
    task = benchmark.get_task(TASK_ID)
    init_states_path = os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file)
    init_states = torch_mod.load(init_states_path, weights_only=False)

    m = env._mj_model
    nq, nv = m.nq, m.nv
    qpos = jp.array([init_states[0][1:1 + nq]], dtype=jp.float32)
    qvel = jp.array([init_states[0][1 + nq:1 + nq + nv]], dtype=jp.float32)

    from mujoco import mjx
    d = mjx.make_data(m, impl="warp", naconmax=env._naconmax, njmax=env._njmax)
    d = d.replace(qpos=qpos, qvel=qvel)
    d = mjx.forward(env._mjx_model, d)

    renderer = WarpRenderer(
        m, n_envs=N_ENVS,
        img_h=IMG_H, img_w=IMG_W,
        camera_names=("agentview", "robot0_eye_in_hand"),
        brightness_boost=1.15,
    )
    images = renderer.render(state_data=d)
    av = np.array(images["agentview_rgb"][0].cpu())
    eye = np.array(images["eye_in_hand_rgb"][0].cpu())
    return av, eye


def side_by_side(left, right, labels=("CPU (EGL)", "Warp (ray trace)"), title=""):
    h, w = left.shape[:2]
    pad = 4
    label_h = 16
    total_w = w * 2 + pad * 3
    total_h = h + label_h * 2 + pad
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255

    for i, (img, label) in enumerate([(left, labels[0]), (right, labels[1])]):
        x = pad + i * (w + pad)
        y = label_h
        canvas[y:y + h, x:x + w] = img

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    for i, label in enumerate(labels):
        x = pad + i * (w + pad)
        draw.text((x + 2, 2), label, fill=0)
    if title:
        draw.text((pad, total_h - label_h + 2), title, fill=0)
    return np.array(pil)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Rendering CPU...")
    cpu_av, cpu_eye = render_cpu()

    print("Rendering Warp...")
    warp_av, warp_eye = render_warp()

    print("Saving images...")

    Image.fromarray(cpu_av).save(os.path.join(OUT_DIR, "cpu_agentview.png"))
    Image.fromarray(warp_av).save(os.path.join(OUT_DIR, "warp_agentview.png"))
    Image.fromarray(cpu_eye).save(os.path.join(OUT_DIR, "cpu_eye_in_hand.png"))
    Image.fromarray(warp_eye).save(os.path.join(OUT_DIR, "warp_eye_in_hand.png"))

    av_compare = side_by_side(cpu_av, warp_av, title="agentview")
    Image.fromarray(av_compare).save(os.path.join(OUT_DIR, "compare_agentview.png"))

    eye_compare = side_by_side(cpu_eye, warp_eye, title="robot0_eye_in_hand")
    Image.fromarray(eye_compare).save(os.path.join(OUT_DIR, "compare_eye_in_hand.png"))

    print(f"Saved to {OUT_DIR}/")
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()