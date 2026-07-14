"""LIBERO MJX: All 130 LIBERO manipulation tasks ported to MuJoCo Warp.

GPU-parallel simulation of all 5 LIBERO suites:
  - spatial (10 tasks)
  - object (10 tasks)
  - goal (10 tasks)
  - scene10 (10 tasks)
  - scene90 (90 tasks)

Each task XML is extracted from the original robosuite/LIBERO benchmark
and reimplemented as a MuJoCo MJX model with a JAX-based OSC controller.

Core modules:
  - envs: Unified LiberoEnv for all 5 suites + base class
  - controllers: OSC controller (JAX port of robosuite OSC_POSE)
  - predicates: Success predicates (distance, on, in_region, is_open, etc.)
  - render: Batched GPU rendering via mujoco_warp with DLPack export
  - warp_gpu_patch: ROCm/CUDA compatibility patches
  - robosuite_patch: Fixes for loading all 5 suites from robosuite

Usage:
    from libero_mjx.envs.libero import LiberoEnv

    env = LiberoEnv(suite="spatial", task_id=0, impl="warp", n_envs=256)
    state = env.reset(jax.random.PRNGKey(0))
    state = env.step(state, action)

For BC training and evaluation, see scripts/train_bc.py and scripts/eval_bc.py.
"""

# Patch Warp GPU detection + texture type code before importing anything else
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()
from libero_mjx.texture_patch import patch as patch_texture  # noqa: F401

from libero_mjx.envs.base import LiberoMjxEnv, LiberoState
from libero_mjx.envs import register_env, load_env, available_envs
from libero_mjx.envs.libero import LiberoEnv, SUITES, TASK_NAMES
from libero_mjx.controllers.osc import OscController
from libero_mjx.predicates.spatial import (
    on_top_of, inside, in_contact, distance_to,
    on, in_region, is_open, is_closed, is_turned_on, PredicateFn,
)

__version__ = "1.0.0"
__all__ = [
    "LiberoMjxEnv", "LiberoState", "LiberoEnv",
    "register_env", "load_env", "available_envs",
    "SUITES", "TASK_NAMES",
    "OscController",
    "on_top_of", "inside", "in_contact", "distance_to",
    "on", "in_region", "is_open", "is_closed", "is_turned_on", "PredicateFn",
]