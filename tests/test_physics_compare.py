"""Compare warp physics vs CPU MuJoCo for the same action sequence.

Steps the same init state with the same ctrl on both warp and CPU,
comparing qpos/EE/obj after each step to find exactly where they diverge.
No JAX impl needed — just CPU mujoco vs warp mujoco.
"""
import sys, os, time
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")
sys.path.insert(0, "/workspace/libero-mjx-port")
sys.path.insert(0, "/workspace/libero_basil")

import numpy as np
import torch
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()
from libero_mjx.envs.libero import LiberoEnv
from libero_mjx.envs.base import LiberoState

env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=1, optimize_physics=False)
env.load_init_states(0)
nq, nv = env._mj_model.nq, env._mj_model.nv
m = env._mj_model

init_states = torch.load(
    "/workspace/libero_basil/libero/libero/init_files/libero_spatial/"
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.init",
    weights_only=False,
)
qpos0 = np.array(init_states[0, 1:1+nq], dtype=np.float32)
qvel0 = np.array(init_states[0, 1+nq:1+nq+nv], dtype=np.float32)

ee_site = m.site("gripper0_right_grip_site").id
obj_body = env._obj_body

# ---- CPU: step with zero ctrl 5 times ----
mjd_cpu = mujoco.MjData(m)
mjd_cpu.qpos[:] = qpos0
mjd_cpu.qvel[:] = qvel0
mujoco.mj_forward(m, mjd_cpu)
for _ in range(5):
    mjd_cpu.ctrl[:] = 0
    mujoco.mj_step(m, mjd_cpu)
ee_cpu = mjd_cpu.site_xpos[ee_site].copy()
obj_cpu = mjd_cpu.xpos[obj_body].copy()
print(f"CPU 5 zero: EE={ee_cpu} obj={obj_cpu}", flush=True)

# ---- Warp: step with zero actions 5 times (env.step includes OSC but zero action = zero delta) ----
def make_state(qpos, qvel, rng):
    d = mjx.make_data(m, impl="warp", naconmax=env._naconmax, njmax=env._njmax)
    d = d.replace(qpos=qpos, qvel=qvel)
    d = mjx.forward(env._mjx_model, d)  # populate site_xpos/xpos for OSC + obs
    info = {"rng": rng, "step": jnp.array(0, dtype=jnp.int32), "gripper_current_action": jnp.zeros(2)}
    obs = env._get_obs(d, info)
    metrics = {k: jnp.array(0.0) for k in env._reward_keys()}
    metrics["success"] = jnp.array(0.0, dtype=jnp.float32)
    return LiberoState(d, obs, jnp.array(0.0), jnp.array(0.0), metrics, info)

jit_step = jax.jit(env.step)
state = jax.jit(make_state)(jnp.array(qpos0), jnp.array(qvel0), jax.random.PRNGKey(0))
for _ in range(5):
    state = jit_step(state, jnp.zeros(7))
jax.block_until_ready(state.data.qpos)
ee_warp = np.asarray(state.data.site_xpos[ee_site])
obj_warp = np.asarray(state.data.xpos[obj_body])
print(f"Warp 5 zero: EE={ee_warp} obj={obj_warp}", flush=True)
print(f"  EE delta={np.linalg.norm(ee_cpu - ee_warp):.6f} obj delta={np.linalg.norm(obj_cpu - obj_warp):.6f}", flush=True)

# ---- Now compare with a FIXED ctrl (not OSC — direct ctrl) ----
# Set ctrl to a known fixed value and compare CPU vs warp
fixed_ctrl = np.zeros(m.nu, dtype=np.float32)
fixed_ctrl[:7] = [5.0, -10.0, 2.0, 5.0, 0.5, -1.0, 0.0]  # some torques
fixed_ctrl[7:9] = 0.02  # gripper

print(f"\n=== Fixed ctrl test: {fixed_ctrl[:7]} ===", flush=True)

# CPU
for i in range(10):
    mjd_cpu.ctrl[:] = fixed_ctrl
    mujoco.mj_step(m, mjd_cpu)
    if i % 2 == 0:
        ee = mjd_cpu.site_xpos[ee_site].copy()
        obj = mjd_cpu.xpos[obj_body].copy()
        print(f"CPU step {i+1}: EE={ee} obj={obj}", flush=True)

# Warp — need to set ctrl directly (bypass OSC)
# Create fresh state
state2 = jax.jit(make_state)(jnp.array(qpos0), jnp.array(qvel0), jax.random.PRNGKey(0))
# Step 5 zero first
for _ in range(5):
    state2 = jit_step(state2, jnp.zeros(7))
jax.block_until_ready(state2.data.qpos)

# Now set ctrl directly on the warp data and step
# We need to bypass env.step's OSC and set ctrl manually
# Use mjx.step directly with custom ctrl
def step_with_ctrl(state, ctrl):
    data = state.data.replace(ctrl=ctrl)
    data = mjx.step(env._mjx_model, data)
    # Update info step
    info = {**state.info, "step": state.info["step"] + 1}
    obs = env._get_obs(data, info)
    return state.replace(data=data, obs=obs, info=info)

jit_step_ctrl = jax.jit(step_with_ctrl)
ctrl_jax = jnp.array(fixed_ctrl)
for i in range(10):
    state2 = jit_step_ctrl(state2, ctrl_jax)
    jax.block_until_ready(state2.data.qpos)
    if i % 2 == 0:
        ee = np.asarray(state2.data.site_xpos[ee_site])
        obj = np.asarray(state2.data.xpos[obj_body])
        print(f"Warp step {i+1}: EE={ee} obj={obj}", flush=True)

# Final comparison
ee_cpu = mjd_cpu.site_xpos[ee_site].copy()
obj_cpu = mjd_cpu.xpos[obj_body].copy()
ee_warp = np.asarray(jax.block_until_ready(state2.data.site_xpos[ee_site]))
obj_warp = np.asarray(state2.data.xpos[obj_body])
print(f"\n=== After 10 fixed ctrl steps ===")
print(f"EE:  CPU={ee_cpu} Warp={ee_warp} delta={np.linalg.norm(ee_cpu - ee_warp):.6f}")
print(f"Obj: CPU={obj_cpu} Warp={obj_warp} delta={np.linalg.norm(obj_cpu - obj_warp):.6f}")
qpos_warp = np.asarray(state2.data.qpos)
print(f"qpos nan: CPU={np.any(np.isnan(mjd_cpu.qpos))} Warp={np.any(np.isnan(qpos_warp))}")
print(f"qpos CPU[:7]: {mjd_cpu.qpos[:7]}")
print(f"qpos Warp[:7]: {qpos_warp[:7]}")
print("DONE", flush=True)