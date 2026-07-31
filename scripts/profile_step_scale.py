"""Profile mjwarp.step at scale with configurable constraint buffer sizes."""
import os, sys, time, argparse, numpy as np, torch, importlib.util

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
from libero_mjx.robosuite_patch import patch_robosuite; patch_robosuite()
from libero_mjx.envs.libero import LiberoEnv

parser = argparse.ArgumentParser()
parser.add_argument("--n-envs", type=int, default=2048)
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--warmup", type=int, default=5)
parser.add_argument("--naconmax", type=int, default=1024)
parser.add_argument("--njmax", type=int, default=512)
args = parser.parse_args()

N = args.n_envs
SIM_DT = 0.01; CTRL_DT = 0.05

jax_env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=10, optimize_physics=False)
jax_env.load_init_states(0)
mj_model = jax_env._mj_model

env = WarpEnv(mj_model, n_envs=N, naconmax=args.naconmax, njmax=args.njmax,
              sim_dt=SIM_DT, ctrl_dt=CTRL_DT)

init_states = jax_env._init_states
ep_idx = [i % init_states.shape[0] for i in range(N)]
nq = mj_model.nq; nv = mj_model.nv
qp = np.array([init_states[i][1:1+nq] for i in ep_idx], dtype=np.float32)
qv = np.array([init_states[i][1+nq:1+nq+nv] for i in ep_idx], dtype=np.float32)
env.set_state(qp, qv)

action = torch.zeros(N, 7, device="cuda")
action[:, 6] = 1.0
gripper_cur = torch.zeros(N, 2, device="cuda")

print(f"N={N}, naconmax={args.naconmax}, njmax={args.njmax}, warmup={args.warmup}, steps={args.steps}")
print(f"VRAM after init: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB free")

for s in range(args.warmup):
    success, gripper_cur = env.step(action, gripper_cur)
wp.synchronize()
print(f"VRAM after warmup: {torch.cuda.mem_get_info(0)[0]/1e9:.2f} GB free")

t0 = time.time()
for s in range(args.steps):
    success, gripper_cur = env.step(action, gripper_cur)
wp.synchronize()
step_total = time.time() - t0
step_ms = step_total / args.steps * 1000
per_substep = step_ms / (CTRL_DT / SIM_DT)
env_steps_s = N * args.steps / step_total

print(f"Step: {step_ms:.1f}ms/ctrl_step ({per_substep:.1f}ms/substep)")
print(f"Throughput: {env_steps_s:.0f} env-steps/s")