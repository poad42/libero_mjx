"""Test all 5 LIBERO suites load, step, and evaluate predicates on Warp."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

import numpy as np
import jax
import jax.numpy as jp
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()
from libero_mjx.envs.libero import LiberoEnv, SUITES


def test_suite(suite, task_id=0, n_envs=4):
    """Test that a suite loads, resets, steps, and checks predicate."""
    meta = SUITES[suite]
    print(f"\n=== {suite} ({meta['n_tasks']} tasks) ===")
    t0 = time.time()

    env = LiberoEnv(suite=suite, task_id=task_id, impl="warp", n_envs=n_envs)
    print(f"  env OK: nq={env._mj_model.nq}, nv={env._mj_model.nv}, "
          f"nbody={env._mj_model.nbody}, ngeom={env._mj_model.ngeom}")

    states = env.load_init_states(task_id)
    if states is not None:
        print(f"  init states: {np.asarray(states).shape}")

    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, jp.zeros(7))
    success = bool(jp.any(state.metrics["success"] > 0.5))
    elapsed = time.time() - t0
    print(f"  step OK: success={success} ({elapsed:.1f}s)")
    return True


def test_all_suites():
    """Test all 5 suites."""
    results = {}
    for suite in ["spatial", "object", "goal", "scene10", "scene90"]:
        try:
            test_suite(suite)
            results[suite] = "PASS"
        except Exception as e:
            print(f"  FAILED: {e}")
            results[suite] = f"FAIL: {e}"

    print(f"\n=== SUMMARY ===")
    for suite, status in results.items():
        print(f"  {suite:10s}: {status}")

    ok = sum(1 for s in results.values() if s == "PASS")
    print(f"\n{ok}/{len(results)} suites passed")
    assert ok == len(results), f"{len(results) - ok} suites failed"


if __name__ == "__main__":
    test_all_suites()