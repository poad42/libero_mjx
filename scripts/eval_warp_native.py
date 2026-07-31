"""Full eval with Warp native env (no JAX). 600 steps, 10 envs."""
import os, sys, time, numpy as np, torch, importlib.util

spec = importlib.util.spec_from_file_location("rkp", "libero_mjx/render_kernel_patch.py")
_m = importlib.util.module_from_spec(spec); spec.loader.exec_module(_m); _m.patch_render_kernel()

try:
    import mujoco_warp as mjwarp
except ImportError:
    sys.path.insert(0, "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party")
    import mujoco_warp as mjwarp

import warp as wp
sys.path.insert(0, ".")
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu; patch_warp_to_gpu()
import libero_mjx.texture_patch
from libero_mjx.warp_env import WarpEnv
from libero_mjx.render import WarpRenderer
from libero_mjx.robosuite_patch import patch_robosuite; patch_robosuite()

from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from easydict import EasyDict
import yaml
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import torch_load_model, safe_device, get_task_embs, control_seed
from libero.lifelong.algos import get_algo_class
from libero.lifelong.datasets import get_dataset

N = 10; STEPS = 600
SIM_DT = 0.01; CTRL_DT = 0.05
control_seed(42)
torch.distributions.Distribution.set_default_validate_args(False)
cd = os.path.join(os.environ["LIBERO_BASIL_PATH"], "libero/configs")
with initialize_config_dir(version_base=None, config_dir=cd):
    cfg = compose(config_name="config", overrides=["seed=42","benchmark_name=libero_spatial",
        "policy=bc_transformer_policy","lifelong=single_task","data.task_order_index=0",
        "train.num_workers=4","train.n_epochs=50","train.batch_size=32",
        "eval.use_mp=false","eval.num_procs=1","eval.eval=false","use_wandb=false"])
cfg = EasyDict(yaml.safe_load(OmegaConf.to_yaml(cfg)))
cfg.folder = cfg.folder or get_libero_path("datasets")
cfg.bddl_folder = cfg.bddl_folder or get_libero_path("bddl_files")
cfg.init_states_folder = cfg.init_states_folder or get_libero_path("init_states")
bm = get_benchmark("libero_spatial")(0)
task = bm.get_task(0)
init_states = torch.load(os.path.join(cfg.init_states_folder, task.problem_folder, task.init_states_file), weights_only=False)
demo = os.path.join(cfg.folder, bm.get_task_demonstration(0))
_, shape_meta = get_dataset(dataset_path=demo, obs_modality=cfg.data.obs.modality, initialize_obs_utils=True, seq_len=cfg.data.seq_len)
cfg.shape_meta = shape_meta
descs = [bm.get_task(i).language for i in range(bm.n_tasks)]
embs = get_task_embs(cfg, descs); bm.set_task_embs(embs)
task_emb = bm.get_task_emb(0)
algo = safe_device(get_algo_class(cfg.lifelong.algo)(1, cfg), "cuda")
sd,_,pv = torch_load_model("checkpoints/task0_model_50ep.pth", map_location="cuda")
algo.policy.load_state_dict(sd); algo.policy.previous_mask=pv; algo.eval()

from libero_mjx.envs.libero import LiberoEnv
jax_env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=N, optimize_physics=False)
jax_env.load_init_states(0)
mj_model = jax_env._mj_model

env = WarpEnv(mj_model, n_envs=N, naconmax=jax_env._naconmax, njmax=jax_env._njmax,
              sim_dt=SIM_DT, ctrl_dt=CTRL_DT)

ep_idx = [i % init_states.shape[0] for i in range(N)]
nq = mj_model.nq; nv = mj_model.nv
qp = np.array([init_states[i][1:1+nq] for i in ep_idx], dtype=np.float32)
qv = np.array([init_states[i][1+nq:1+nq+nv] for i in ep_idx], dtype=np.float32)
env.set_state(qp, qv)

renderer = WarpRenderer(mj_model, n_envs=N, brightness_boost=1.15)

_p = algo.policy
def _ga(data):
    _p.eval()
    with torch.no_grad():
        d = _p.preprocess_input(data, train_mode=False)
        x = _p.spatial_encode(d); _p.latent_queue.append(x)
        if len(_p.latent_queue) > _p.max_seq_len: _p.latent_queue.pop(0)
        x = torch.cat(_p.latent_queue, dim=1); x = _p.temporal_encode(x)
        dist = _p.policy_head(x[:, -1])
    return dist.sample().detach().view(-1, 7)
_p.get_action = _ga; _p.reset()

done = [False]*N; steps_arr = [0]*N
gripper_cur = torch.zeros(N, 2, device="cuda")
t0 = time.time()
for s in range(STEPS):
    images = renderer.render(mw_model=env.wm, mw_data=env.wd)
    obs_dict = env.get_obs()
    obs = {"obs": {"agentview_rgb": images["agentview_rgb"], "eye_in_hand_rgb": images["eye_in_hand_rgb"],
        "joint_states": obs_dict["joint_states"], "gripper_states": obs_dict["gripper_states"]},
        "task_emb": task_emb.unsqueeze(0).repeat(N, 1).to("cuda")}
    action = _p.get_action(obs)
    action = torch.nan_to_num(action, nan=0.0)
    success, gripper_cur = env.step(action, gripper_cur)
    for i in range(N):
        if not done[i]: steps_arr[i] += 1
    if (s+1) % 50 == 0:
        for i in range(N):
            if not done[i] and (success[i] > 0.5 or steps_arr[i] >= STEPS): done[i] = True
        ns = sum(1 for i in range(N) if done[i] and steps_arr[i] < STEPS)
        print(f"  step {s+1}: success={ns}", flush=True)
        if all(done): break

n_succ = sum(1 for i in range(N) if success[i] > 0.5)
print(f"\nSuccess: {n_succ}/{N} = {n_succ/N:.0%}")
print(f"Time: {time.time()-t0:.1f}s")