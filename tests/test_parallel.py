"""Test: 1024 parallel envs — verify batched reset/step + throughput."""

import sys
import os
import time

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mujoco_playground._src.manipulation.franka_emika_panda import pick


def test_parallel_1024():
    """Run 1024 envs in parallel via vmap, measure throughput."""
    env = pick.PandaPickCube()
    N = 1024
    WARMUP = 5
    MEASURE = 200

    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(42), N)
    print(f"[test_parallel] resetting {N} envs...")
    states = v_reset(keys)
    jax.block_until_ready(states.obs)

    actions = jp.zeros((N, env.action_size))
    print(f"[test_parallel] warmup {WARMUP} steps...")
    for _ in range(WARMUP):
        states = v_step(states, actions)
    jax.block_until_ready(states.obs)

    print(f"[test_parallel] measuring {MEASURE} steps...")
    t0 = time.time()
    for _ in range(MEASURE):
        states = v_step(states, actions)
    jax.block_until_ready(states.obs)
    elapsed = time.time() - t0
    total = N * MEASURE
    rate = total / elapsed
    print(f"[test_parallel] {total} steps in {elapsed:.3f}s -> {rate:.0f} steps/sec")
    assert rate > 1000, f"Throughput {rate:.0f} < 1000 steps/sec threshold"

    # State save/restore in batch
    saved = states.data.qpos.copy()
    states2 = v_reset(keys)
    states2 = states2.replace(data=states2.data.replace(qpos=saved))
    states2 = v_step(states2, actions)
    jax.block_until_ready(states2.obs)
    match = jp.allclose(states2.data.qpos, states.data.qpos, atol=1e-5)
    print(f"[test_parallel] batch state restore match: {bool(match)}")
    assert bool(match)
    print("[test_parallel] PASS")


if __name__ == "__main__":
    test_parallel_1024()