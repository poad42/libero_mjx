"""Test: does mjx.step in vmap correctly compute xpos after steps?"""
import sys, os, time
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.2")
sys.path.insert(0, "/workspace/libero-mjx-port")
sys.path.insert(0, "/workspace/libero_basil")

import numpy as np, torch, jax, jax.numpy as jnp
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()
from libero_mjx.envs.libero import LiberoEnv
from libero_mjx.envs.base import LiberoState
from mujoco import mjx

env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=4, optimize_physics=True)
env.load_init_states(0)
nq, nv = env._mj_model.nq, env._mj_model.nv
init_states = torch.load("/workspace/libero_basil/libero/libero/init_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.init", weights_only=False)
N = 4
qpos_batch = jnp.array([init_states[i][1:1+nq] for i in range(N)], dtype=jnp.float32)
qvel_batch = jnp.array([init_states[i][1+nq:1+nq+nv] for i in range(N)], dtype=jnp.float32)

def make_state(qpos, qvel, rng):
    d = mjx.make_data(env._mj_model, impl="warp", naconmax=env._naconmax, njmax=env._njmax)
    d = d.replace(qpos=qpos, qvel=qvel)
    info = {"rng": rng, "step": jnp.array(0, dtype=jnp.int32)}
    obs = env._get_obs(d, info)
    metrics = {k: jnp.array(0.0) for k in env._reward_keys()}
    metrics["success"] = jnp.array(0.0, dtype=jnp.float32)
    return LiberoState(d, obs, jnp.array(0.0), jnp.array(0.0), metrics, info)

t0 = time.time()
print("Compiling vmap make_state...", flush=True)
state = jax.jit(jax.vmap(make_state))(qpos_batch, qvel_batch, jax.random.split(jax.random.PRNGKey(42), N))
jax.block_until_ready(state.data.qpos)
print(f"  make_state compiled in {time.time()-t0:.1f}s", flush=True)

print("Compiling vmap env.step...", flush=True)
t0 = time.time()
vstep = jax.jit(jax.vmap(env.step))
state = vstep(state, jnp.zeros((N, 7)))
jax.block_until_ready(state.data.qpos)
print(f"  first step compiled in {time.time()-t0:.1f}s", flush=True)

ob = env._obj_body; tb = env._target_body
obj = np.asarray(state.data.xpos)[:, ob, :3]
tgt = np.asarray(state.data.xpos)[:, tb, :3]
print(f"After 1 step: obj[0]={obj[0]} tgt[0]={tgt[0]} d={np.linalg.norm(obj[0]-tgt[0]):.3f}", flush=True)
print(f"  all d: {[f'{np.linalg.norm(obj[i]-tgt[i]):.3f}' for i in range(N)]}", flush=True)

for _ in range(4):
    state = vstep(state, jnp.zeros((N, 7)))
jax.block_until_ready(state.data.qpos)
obj = np.asarray(state.data.xpos)[:, ob, :3]
tgt = np.asarray(state.data.xpos)[:, tb, :3]
print(f"After 5 steps: obj[0]={obj[0]} d={np.linalg.norm(obj[0]-tgt[0]):.3f}", flush=True)
print(f"  nan={np.any(np.isnan(np.asarray(state.data.qpos)))}", flush=True)

# Step with actual action
for i in range(100):
    state = vstep(state, jnp.array([[0.3, 0.2, -0.3, 0, 0, 0, 0.5]] * N))
    if i == 99:
        jax.block_until_ready(state.data.qpos)
ee_s = env._gripper_site
ee = np.asarray(state.data.site_xpos)[:, ee_s, :3]
obj = np.asarray(state.data.xpos)[:, ob, :3]
print(f"After 100 action steps: EE[0]={ee[0]} obj[0]={obj[0]}", flush=True)
print(f"  nan={np.any(np.isnan(np.asarray(state.data.qpos)))}", flush=True)
print("DONE", flush=True)