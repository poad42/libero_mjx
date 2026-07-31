# Architecture

## Environment stack

The environment stack runs physics on GPU and renders on GPU. Three libraries share the same GPU memory through DLPack zero-copy transfers. No host round-trips happen during stepping or rendering.

### Data flow per step

1. The JAX `LiberoState.data` (an `mjx.Data` with `impl="warp"`) holds qpos & qvel on GPU.
2. `WarpRenderer.render()` copies qpos & qvel into a `mujoco_warp` Data buffer via DLPack. No copy to host.
3. `mjwarp.forward()` recomputes kinematics (xpos, xmat, site_xpos).
4. `refit_scene_bvh` updates the bounding volume hierarchy for the new positions.
5. `mjwarp.render()` traces primary & shadow rays, writes packed RGB uint32 to a buffer.
6. `WarpRenderer` unpacks the buffer to `(N, H, W, 3)` uint8 torch tensors on cuda, flips vertically, and applies the brightness multiplier.
7. The BC policy takes the torch tensors as observation, produces an action tensor.
8. `jax.vmap(env.step)` applies the action through the OSC controller and steps physics.

## Interop: three separate graphs, not one

The eval loop uses three GPU libraries that do not share a computation graph. JAX runs physics (JIT-compiled). Warp runs rendering (ray tracing kernels). Torch runs the policy (eager mode). The CPU thread orchestrates the loop and exchanges GPU memory pointers via DLPack. No bytes cross the PCIe bus during stepping or rendering, but `wp.synchronize()` and `jax.block_until_ready()` create hard barriers between the three.

```
JAX state.data (mjx.Data, impl="warp")
  |
  +-- qpos.__dlpack__() -> torch.from_dlpack        [zero-copy]
  |   |
  |   +-> wp.from_torch(qpos_t)                     [zero-copy]
  |   |   Warp: mjwarp.forward()
  |   |   Warp: mjwarp.render()
  |   |   wp.to_torch(ctx.rgb_data)                 [zero-copy]
  |   |   Torch: unpack, flip, brightness boost
  |   |
  |   +-> qpos_t[:, arm_slice]                      [joint/gripper obs]
  |
  +-- Torch policy.get_action(images + obs)
  |
  +-- action -> jnp.from_dlpack(action)             [zero-copy back to JAX]
      JAX: vstep(state, action)
```

The `WarpRenderer.render()` method does not call `wp.synchronize()`. Warp launches on the same CUDA stream execute in order. The `wp.to_torch()` call creates the dependency for the consumer (the policy). The eval loop batches success checks every 50 steps instead of calling `jax.block_until_ready()` every step.

The policy's `get_action` is monkey-patched to return a GPU tensor. The original LIBERO implementation calls `.cpu().numpy()`, causing a GPU-to-host-to-GPU round-trip per step. The patched version returns `dist.sample().detach().view(-1, 7)` directly.

A unified graph (fusing physics, rendering, and policy into one JIT trace) would eliminate the barriers. That requires porting the OSC controller from JAX to Warp or Torch, and porting the BC policy from Torch to JAX. Both are nontrivial.

## Performance

### Per-phase profiling

Spatial task 0, 10 envs, 50 steps, AMD RX 9070 XT (gfx1201, 16 GiB):

| Phase | 25 substeps (sim_dt=0.002) | 5 substeps (sim_dt=0.01) |
|-------|------|------|
| render | 75.2 ms | 94.0 ms |
| obs (DLPack) | 0.3 ms | 0.3 ms |
| policy | 7.4 ms | 7.8 ms |
| step (physics + OSC) | 405.8 ms | 141.2 ms |
| **total** | **541.4 ms** | **277.3 ms** |
| **env-steps/s** | **18.5** | **36.1** |

The physics step dominates at 75% of total time with 25 substeps. Each substep runs the JAX OSC controller (Jacobians, mass matrix inverse, nullspace projection) plus `mjx.step`. With 10 envs, each kernel is small and launch overhead dominates. Reducing substeps from 25 to 5 cuts the step time by 2.9x.

The step time decomposes as: `fixed_overhead + n_substeps * per_substep_cost`. From the two measurements: fixed overhead is 75 ms, per-substep cost is 13 ms.

### CPU vs Warp benchmark

Spatial task 0, 50-epoch BC checkpoint, 10 episodes, 600 max steps:

| Path | Envs | Wall time | Env-steps/s | Success |
|------|------|-----------|-------------|---------|
| CPU (robosuite, EGL) | 1 | 92.6s | 38.4 | 50% |
| Warp (JAX), 25 substeps | 10 | 317.2s | 18.9 | 50% |
| Warp (JAX), 5 substeps | 10 | 146.6s | 40.9 | 70% |
| Warp native (no JAX), 5 substeps | 10 | 116.1s | 51.7 | 60% avg |

The JAX-based Warp eval (25 substeps) is 0.5x CPU throughput. The JAX physics step (25 substeps of OSC + mjx.step) takes 406 ms per control step, while the CPU eval runs robosuite on CPU and the policy on GPU in parallel.

The Warp native env (libero_mjx/warp_env.py) eliminates JAX from the physics step. The OSC controller runs in Torch, the physics in mujoco_warp.step(), and the mass matrix is densified from the Warp M field via CPU. With 5 substeps, the Warp native achieves 51.7 env-steps/s (1.35x CPU) with 60% average success across 3 seeds (60%, 50%, 70%).

Per-phase profiling (10 envs, 5 substeps, sim_dt=0.01):

| Phase | JAX-based | Warp native |
|-------|-----------|-------------|
| render | 94.0 ms | 65.6 ms |
| obs (DLPack) | 0.3 ms | 0.3 ms |
| policy | 7.8 ms | 7.3 ms |
| step (physics + OSC) | 141.2 ms | 84.1 ms |
| **total** | **277.3 ms** | **213.9 ms** |
| **env-steps/s** | **36.1** | **46.7** |

The Warp native step time is 40% faster (84.1 ms vs 141.2 ms) by eliminating JAX kernel launch overhead. The OSC controller runs as Torch eager operations on small matrices (7x7, 3x3, 6x6), and the physics runs as direct mujoco_warp.step() calls. No JAX JIT, no XLA dispatch, no jax.vmap overhead.

### Scaling with batch size

| Envs | Wall time | Total env-steps | Env-steps/s |
|------|-----------|-----------------|-------------|
| 10 | 146.6s | 6000 | 40.9 |
| 50 | 452.1s | 30000 | 66.4 |
| 100 | OOM | - | - |

The 16 GiB GPU runs out of memory at 100 envs with 2 cameras at 128x128. The render context allocates per-env buffers for the ray tracing output.

### What limits throughput

The bottleneck is the physics step: 406 ms with 25 substeps (JAX), 84 ms with 5 substeps (Warp native). Each substep runs the OSC controller (Jacobians, mass matrix, matrix inverse, nullspace projection) plus one mjwarp.step() call. With 10 envs, each kernel processes 10 elements and the GPU is underutilized.

The Warp native env (libero_mjx/warp_env.py) addresses this by porting the OSC controller from JAX to Torch and running physics via direct mujoco_warp.step() calls. This eliminates JAX JIT compilation, XLA dispatch, and jax.vmap overhead. The step time drops from 141 ms (JAX, 5 substeps) to 84 ms (Warp native, 5 substeps), a 40% improvement.

Further improvements:
1. **Increase batch size**: more envs amortizes per-step overhead. 50 envs gives 66.4 env-steps/s (1.7x CPU). 100 envs OOMs on 16 GiB.
2. **Reduce substeps**: sim_dt=0.01 gives 5 substeps instead of 25. The physics is less accurate but the policy tolerates it, and success rises from 50% to 60-70%.
3. **GPU-side mass matrix densification**: the M field is read-only via DLPack on ROCm, forcing a CPU round-trip for densification. A GPU-native densification (e.g. via a Warp kernel) would eliminate the 0.5 ms per control step CPU overhead.

### Memory layout

`LiberoState` is a `flax.struct.PyTreeNode`. It holds:

- `data: mjx.Data` with `impl="warp"`. The warp backend stores arrays on GPU. JAX treats them as opaque through DLPack.
- `obs: jax.Array` (state-only: joint positions & gripper positions, 9 dims)
- `reward, done: jax.Array` (scalars)
- `metrics: Dict[str, jax.Array]` (per-step reward components & success flag)
- `info: Dict[str, Any]` (rng, step counter, gripper current action for incremental control)

`jax.vmap` batches the state. `jax.jit` compiles the step function. The OSC controller runs inside the JIT boundary; it computes Jacobians, mass matrix densification, and torque output from JAX primitives.

## LiberoMjxEnv (base.py)

The base class handles model loading, reset, step, and state save/restore.

### Model loading

`__init__` reads the task XML from `libero_mjx/assets/xml/`, calls `mujoco.MjModel.from_xml_string`, sets the timestep, and calls `mjx.put_model` with the chosen implementation (`"warp"` or `"jax"`). Subclasses implement `_post_init` to resolve body, site, and joint indices by name.

### Reset

`reset(rng)` creates a fresh `mjx.Data` via `mjx.make_data`, sets qpos from the init state or the default home position, zeros qvel, and calls `mjx.forward` to populate kinematics. If `load_init_states` was called, reset samples a random init state from the loaded set.

### Step

`step(state, action)` does five things:

1. Clips the 7-dim action to [-1, 1]. First 6 dims are the Cartesian delta EE command. Dim 7 is the gripper command.
2. Computes the gripper finger target using robosuite's incremental format. `gripper_current_action` accumulates at 0.2 per step, clipped to [-1, 1]. The finger ctrl targets are `[0.02 * (1 + cur_0), 0.02 * (cur_1 - 1)]`.
3. Computes the OSC desired pose: current EE pose plus the clipped delta. The rotation delta is an axis-angle vector converted to a rotation matrix and left-multiplied with the current orientation.
4. Runs `n_substeps` physics substeps via `jax.lax.scan`. Each substep recomputes OSC torques tracking the fixed goal, sets ctrl, and calls `mjx.step`.
5. Evaluates the success predicate, checks episode termination, and builds the observation.

### State save/restore

`get_sim_state` returns a dict of qpos, qvel, ctrl, act, mocap_pos, mocap_quat. `set_sim_state` writes them back and calls `mjx.forward` to recompute kinematics. This is tensor copy, no subprocess.

## LiberoEnv (libero.py)

The unified env for all 5 suites. It extends `LiberoMjxEnv` with:

- Auto-detection of the object body & target body by trying a list of known names.
- Auto-detection of the success predicate based on the suite name (spatial uses `distance_to`, object uses `in_region`, goal uses `on`, scene10/scene90 use `distance_to` as a proxy).
- `load_init_states` reads LIBERO `.init` files (zip + pickle) and updates the OSC rest_qpos to match the first init state's arm joints.
- `batch_reset` creates a batched state for N envs using `mjwarp.forward` to compute xpos without JAX vmap.

### Physics optimization

`optimize_physics=True` (default) sets:

- `density = 0.0`, `viscosity = 0.0` (no air resistance)
- `cone = 0` (pyramidal friction cone)
- `integrator = 3` (implicitfast)
- `timestep = 0.005` (was 0.002)
- `iterations = 5`, `ls_iterations = 8`

This matches robosuite's LIBERO config. The eval script passes `optimize_physics=False` to preserve the default MuJoCo solver settings, which match the CPU eval more closely.

## OscController (controllers/osc.py)

A JAX port of robosuite's `OSC_POSE` controller with `input_type='delta'`, `impedance_mode='fixed'`, and `uncouple_pos_ori=True`.

### Torque computation

`compute_torques_to_goal(data, desired_pos, desired_mat)` computes:

1. Position & orientation error: `pos_err = desired_pos - ee_pos`, `ori_err = 0.5 * cross(rc_i, rd_i)` summed over 3 axes.
2. Velocity error from the site Jacobian: `ee_vel = J^T @ qvel`.
3. Desired force & torque: `F = kp * pos_err + kd * vel_err`.
4. Operational-space mass: `Lambda = (J M^-1 J^T)^-1`. Computed separately for position & rotation (uncoupled).
5. Decoupled wrench: `wrench = Lambda @ F`.
6. Joint torques: `tau = J^T @ wrench + qfrc_bias` (gravity compensation from Warp's qfrc_bias, verified to match CPU within 1e-5).
7. Nullspace torques: PD to rest pose, projected through the nullspace matrix `I - Jbar @ J`.

### Mass matrix

Warp returns the mass matrix in sparse format `(nworld, nM)` where `nM` is the number of non-zero entries. The controller densifies it using the model's `M_rownnz`, `M_rowadr`, and `M_colind` arrays, then copies the lower triangle to the upper.

### Parameters

Default values match robosuite's LIBERO config: `kp=150`, `damping_ratio=1.0` (critically damped), `output_max=0.05` for position, `output_max_ori=0.5` for rotation. Nullspace uses `joint_kp=10`, `joint_kv=2*sqrt(10)`.

## WarpRenderer (render.py)

### Initialization

The constructor takes an `MjModel`, a list of camera names, image size, and `n_envs`. It:

1. Resolves camera IDs & builds the `cam_active` mask.
2. Patches BVH construction to use `lbvh` (for HIP).
3. Creates the `mujoco_warp` model & data with `nworld=n_envs`.
4. Calls `mjwarp.create_render_context` with textures, shadows, skybox, and ambient lighting enabled. `enabled_geom_groups=[1, 2]` matches robosuite (group 0 = collision geoms, disabled).
5. Records the RGB buffer offset for each camera.

### Render call

`render(state_data)` copies qpos & qvel from the JAX state into the warp data buffer, runs `mjwarp.forward`, `refit_scene_bvh`, and `mjwarp.render`, then unpacks the packed uint32 buffer into separate R, G, B channels, flips vertically, and applies the brightness multiplier.

The output is a dict mapping camera names to `(N, H, W, 3)` uint8 torch tensors on cuda. Zero host copies from render to policy input.

## Patches

### Import order

The patch modules must run before the libraries they patch are imported. The import chain in `eval_warp_only.py`:

1. `importlib.util.spec_from_file_location` loads `render_kernel_patch.py` as a standalone module, before any `import mujoco` or `import jax`.
2. `_mod.patch_render_kernel()` patches `render.py`, `types.py`, and `io.py` on disk.
3. `from libero_mjx.warp_gpu_patch import patch_warp_to_gpu` imports jax & patches FFI registration.
4. `patch_warp_to_gpu()` patches MJX io module's device detection.
5. `from libero_mjx import texture_patch` patches warp's type code function.
6. `from libero_mjx.envs.libero import LiberoEnv` triggers `libero_mjx.__init__`, which calls `patch_warp_to_gpu()` again (idempotent) and imports the env modules.
7. `from libero_mjx.render import WarpRenderer` imports `mujoco_warp` and creates the render context.

Steps 1-2 must happen before step 7. If `mujoco_warp._src.types` gets cached in `sys.modules` before the patch, the `RenderContext` dataclass will not have the new fields, and the render context creation will fail or ignore haze parameters.

### Idempotency

Each patch checks for a `# PATCHED_BY_LIBERO_MJX` marker in the target file. If present, the patch skips. Backups go to `.orig` files on first application. This makes the patches safe to run multiple times across processes.