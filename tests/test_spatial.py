"""Test: LIBERO_SPATIAL task env loads and steps with Warp."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jp
import numpy as np


def test_spatial_task_smoke():
    """Load task1 (table center), reset with init state, step."""
    from libero_mjx.envs.spatial import LiberoSpatialBase
    from etils import epath

    xml = epath.Path(__file__).parent.parent / "libero_mjx" / "assets" / "xml" / "libero_spatial_task1.xml"
    env = LiberoSpatialBase(xml_path=xml, impl="warp")
    env.load_init_states(1)
    print(f"[test_spatial] task1 loaded, action_size={env.action_size}")

    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)
    print(f"[test_spatial] reset OK obs.shape={state.obs.shape}")

    # Check that bowl and plate are at reasonable positions
    bowl_pos = np.asarray(state.data.xpos[env._bowl_body])
    plate_pos = np.asarray(state.data.xpos[env._plate_body])
    print(f"[test_spatial] bowl_pos={bowl_pos.round(3)} plate_pos={plate_pos.round(3)}")

    # Step with zero action
    action = jp.zeros(env.action_size)
    state = env.step(state, action)
    print(f"[test_spatial] step OK reward={float(state.reward):.4f}")

    # Step with a small upward EE command
    action_up = jp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])  # z+ + gripper open
    state = env.step(state, action_up)
    print(f"[test_spatial] step up OK reward={float(state.reward):.4f}")

    # Batch (vmap) test with 4 envs
    print("[test_spatial] testing vmap with 4 envs...")
    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(rng, 4)
    states = v_reset(keys)
    actions = jp.zeros((4, env.action_size))
    states = v_step(states, actions)
    jax.block_until_ready(states.obs)
    print(f"[test_spatial] vmap OK obs.shape={states.obs.shape}")
    print("[test_spatial] PASS")


if __name__ == "__main__":
    test_spatial_task_smoke()