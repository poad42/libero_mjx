# API reference

## LiberoEnv

```python
from libero_mjx.envs.libero import LiberoEnv

env = LiberoEnv(
    suite="spatial",      # spatial, object, goal, scene10, scene90
    task_id=0,            # int, task index within the suite
    impl="warp",          # "warp" (GPU) or "jax" (CPU JIT)
    n_envs=64,            # number of parallel envs (scales naconmax)
    optimize_physics=True, # apply robosuite solver settings
    predicate_fn=None,    # optional override for the success predicate
)
```

### Methods

#### `reset(rng) -> LiberoState`

Creates a fresh state. If `load_init_states` was called, samples a random init state from the loaded set. Otherwise uses the home position.

```python
state = env.reset(jax.random.PRNGKey(0))
```

#### `step(state, action) -> LiberoState`

Applies a 7-dim action. First 6 dims: Cartesian delta EE command (dx, dy, dz, droll, dpitch, dyaw), clipped to [-0.05, 0.05] for position and [-0.5, 0.5] for rotation. Dim 7: gripper command (-1 to open, +1 to close, 0 to hold).

```python
state = env.step(state, jp.zeros(7))
```

Batch with `jax.vmap`:

```python
vstep = jax.jit(jax.vmap(env.step))
state = vstep(state, jp.zeros((256, 7)))
```

#### `load_init_states(task_id=None) -> jax.Array`

Loads init states from the LIBERO `.init` file for the given task. Updates the OSC `rest_qpos` to match the first init state. Returns a `(N, 1 + nq + nv)` JAX array.

```python
states = env.load_init_states(0)
```

#### `get_sim_state(state) -> dict`

Returns a dict of qpos, qvel, ctrl, act, mocap_pos, mocap_quat for checkpointing.

#### `set_sim_state(state, sim_state) -> LiberoState`

Restores from a `get_sim_state` dict. Calls `mjx.forward` to recompute kinematics.

### Properties

| Property | Type | Description |
|---|---|---|
| `dt` | float | Control timestep (0.02s with optimization, 0.05s default) |
| `sim_dt` | float | Physics timestep (0.005s with optimization, 0.002s default) |
| `n_substeps` | int | Physics steps per control step (4 with optimization, 25 default) |
| `action_size` | int | 7 |
| `observation_size` | int | State obs dimension (9: 7 arm joints + 2 gripper) |
| `mj_model` | `mujoco.MjModel` | CPU model |
| `mjx_model` | `mjx.Model` | MJX model (Warp or JAX backend) |

### Class attributes

| Attribute | Type | Description |
|---|---|---|
| `SUITES` | dict | Suite metadata: prefix, n_tasks, init_dir |
| `TASK_NAMES` | dict | Task names per suite, loaded at import time |

## WarpRenderer

```python
from libero_mjx.render import WarpRenderer

renderer = WarpRenderer(
    mj_model,                      # mujoco.MjModel
    n_envs=10,                     # number of parallel worlds
    img_h=128,                     # image height
    img_w=128,                     # image width
    camera_names=("agentview", "robot0_eye_in_hand"),
    brightness_boost=1.15,         # brightness multiplier (1.0 = off)
    enabled_geom_groups=(1, 2),   # geom groups to render
    use_textures=True,
    use_shadows=True,
    use_skybox=True,
)
```

### Methods

#### `render(state_data=None, mw_model=None, mw_data=None) -> dict`

Renders RGB images for all cameras. Returns a dict mapping camera observation keys to `(N, H, W, 3)` uint8 torch tensors on cuda.

Pass a JAX `state.data` (mjx.Data) to copy qpos & qvel into the internal warp buffer:

```python
images = renderer.render(state_data=state.data)
# images["agentview_rgb"]: (10, 128, 128, 3) uint8 on cuda
# images["eye_in_hand_rgb"]: (10, 128, 128, 3) uint8 on cuda
```

Or pass explicit warp model & data objects:

```python
images = renderer.render(mw_model=model, mw_data=data)
```

The returned images have the vertical flip and brightness boost already applied.

## OscController

```python
from libero_mjx.controllers.osc import OscController

osc = OscController.from_model(
    mj_model,
    site_name="gripper0_right_grip_site",
    arm_joint_prefix="robot0_joint",
    kp=150.0,
    damping_ratio=1.0,
    output_max=0.05,
    rest_qpos=None,  # defaults to model's qpos0
)
osc.set_model(mjx_model)
```

### Methods

#### `compute_torques(data, delta_action) -> jax.Array`

Computes arm torques from a 6-dim delta EE action. Returns a `(nu,)` torque vector with zeros for non-arm actuators.

#### `compute_torques_to_goal(data, desired_pos, desired_mat) -> jax.Array`

Computes arm torques tracking a fixed goal pose. Called per physics substep. `desired_pos` is `(3,)`, `desired_mat` is `(3, 3)`.

## Predicates

```python
from libero_mjx.predicates.spatial import (
    distance_to, on, in_region, is_open, is_closed, is_turned_on,
    on_top_of, inside, in_contact, PredicateFn,
)
```

### Constructors

Each returns a `PredicateFn` callable that takes an `mjx.Data` and returns a boolean array.

```python
# Object within 0.08 units of target body
pred = distance_to(obj_body_id, target_body_id, dist=0.08)

# Object on top of target (above + near in XY)
pred = on(obj_body_id, target_body_id, dist=0.08)

# Object within 0.1 units of target (for "in basket" goals)
pred = in_region(obj_body_id, target_body_id, dist=0.1)

# Drawer joint open past 0.15 radians
pred = is_open(joint_qposadr, threshold=0.15)

# Stove actuator ctrl above 0.5
pred = is_turned_on(actuator_id, threshold=0.5)
```

### Custom predicates

Subclass `PredicateFn` and implement `__call__(data) -> jax.Array`:

```python
class MyPredicate(PredicateFn):
    def __call__(self, data):
        return data.xpos[self._obj][..., 2] > 0.5
```

Pass to `LiberoEnv` via `predicate_fn`:

```python
env = LiberoEnv(suite="spatial", task_id=0, predicate_fn=MyPredicate())
```

## render_kernel_patch

```python
from libero_mjx.render_kernel_patch import patch_render_kernel
patch_render_kernel()  # call before any import mujoco_warp
```

Patches three files in the installed `mujoco_warp` package on disk:

- `render.py`: shadow fallback constant (0.3 to 0.0), haze blending code
- `types.py`: `haze_amount`, `fogstart`, `fogend`, `background_color_float` fields on `RenderContext`
- `io.py`: populates the new fields in `create_render_context()`

Idempotent. Checks for a `# PATCHED_BY_LIBERO_MJX` marker. Backs up originals to `.orig` on first run.

## warp_gpu_patch

```python
from libero_mjx.warp_gpu_patch import patch_warp_to_gpu
patch_warp_to_gpu()
```

Patches JAX FFI registration for ROCm, MJX device detection, BVH construction (lbvh), and `GraphMode` enum. Called automatically by `libero_mjx.__init__`.

## robosuite_patch

```python
from libero_mjx.robosuite_patch import patch_robosuite
patch_robosuite()
```

Patches `robot_base_factory` to fall back to `NullMount` for unknown base names. Required for object, goal, scene10, and scene90 suites.