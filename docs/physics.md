# Physics: Warp vs CPU

Warp physics (mujoco_warp) matches CPU MuJoCo for single steps with zero control input. Over 600 steps with a learned policy, small differences accumulate. This page documents what was verified and what differs.

## Single-step verification

A test script steps both CPU MuJoCo (`mujoco.MjData`) and Warp MuJoCo (`mujoco_warp`) from the same init state with `ctrl = 0`. After one step:

| Field | Max diff (Warp vs CPU) |
|---|---|
| qvel | 0.000036 |
| qpos | 0.000036 |

The diff is at float32 precision. Warp physics is correct for a single step.

## Contact allocation

LIBERO models have up to 181 geoms (scene90). The default `njmax` of 44 is too small. `naconmax` (active constraint max) also needs scaling with the number of parallel envs.

`LiberoEnv` sets:

```python
naconmax = n_envs * 1024
njmax = 4096
```

With 10 envs, `naconmax = 10240`, `njmax = 4096`. The `nefc` (number of equality, friction, and contact constraints) can reach 6741 on scene90 tasks. Increasing `njmax` further requires more GPU memory but does not change physics results.

## Physics optimization flags

`LiberoEnv` accepts `optimize_physics` (default `True`). When enabled, it sets:

| Parameter | Default | Optimized | Source |
|---|---|---|---|
| `opt.density` | 0.0 | 0.0 | Matches robosuite |
| `opt.viscosity` | 0.0 | 0.0 | Matches robosuite |
| `opt.cone` | 1 (elliptic) | 0 (pyramidal) | Robosuite uses pyramidal |
| `opt.integrator` | 0 (Euler) | 3 (implicitfast) | Robosuite uses implicitfast |
| `opt.timestep` | 0.002 | 0.005 | Robosuite uses 0.005 |
| `opt.iterations` | 100 | 5 | Robosuite uses 5 |
| `opt.ls_iterations` | 50 | 8 | Robosuite uses 8 |
| `ctrl_dt` | 0.05 | 0.02 | Robosuite action frequency |

The eval script (`eval_warp_only.py`) passes `optimize_physics=False` to match the CPU eval's solver settings. The training script does not touch physics.

## Gripper control format

Robosuite's `PandaGripper.format_action` is incremental and sign-based. The gripper action (dim 7 of the 7-dim action vector) controls finger position through an accumulator:

```
current_action += [-1, +1] * speed * sign(gripper_action)
current_action = clip(current_action, -1, +1)
finger_ctrl = [0.02 * (1 + current_action[0]), 0.02 * (current_action[1] - 1)]
```

With `speed = 0.2` per step:

- `gripper_action = +1` (close): `current_action` moves toward `[-1, +1]`, `finger_ctrl` moves toward `[0, 0]`
- `gripper_action = -1` (open): `current_action` moves toward `[+1, -1]`, `finger_ctrl` moves toward `[0.04, -0.04]`
- `gripper_action = 0`: `current_action` stays, fingers hold position

The `gripper_current_action` accumulator lives in `state.info` and persists across steps. This is in `base.py` step().

## OSC rest_qpos

The OSC nullspace controller drives arm joints toward `rest_qpos`. The default is the model's home position `[0, 0.0067, -0.1919, -0.0099, -2.4326, -0.0399, 2.1935]`.

`load_init_states` updates `rest_qpos` to match the first init state's arm joint positions. This matters because LIBERO init states vary the arm pose. Using the home position as rest_qpos when the arm starts elsewhere produces nullspace torques that fight the init state, causing a warmup drift of up to 0.01 radians.

After the fix, the warmup diff (5 zero-action steps from init state) is 0.000.

## Mass matrix densification

Warp returns the mass matrix in sparse format. The `OscController._densify_mass` method reconstructs the dense matrix:

```python
mat = zeros(batch, nv, nv)
mat[all_rows, all_cols] = qM_sparse
mat = mat + tril(mat, -1).T  # symmetrize
```

`all_rows` and `all_cols` come from the model's `M_rownnz`, `M_rowadr`, and `M_colind` arrays. This is computed at JIT trace time from the model metadata and is constant across steps.

## Gravity compensation

The controller uses `data.qfrc_bias` for gravity compensation. Warp computes `qfrc_bias` accurately. Verification: set the robot to a known pose, run `mjwarp.forward`, read `qfrc_bias`, compare to CPU `mj_forward`. Max diff across all arm DOFs: less than 1e-5.

## Known differences over long horizons

The single-step diff is 0.000036 in qvel. Over 600 steps with a policy, differences accumulate. The success rate gap between Warp eval (42.5%) and CPU eval (50%) is about 7.5 percentage points. Some of that gap is rendering (the remaining RMSE of 22.2), and some is physics drift. The two have not been isolated.

The integrator difference is the most likely source of drift. CPU eval uses MuJoCo's default Euler integrator. Warp eval uses the same solver settings (`optimize_physics=False`), but Warp's `mjwarp.step` may not produce bit-identical results to `mujoco.mj_step` due to different constraint solver implementations on GPU.

## nefc overflow

On scene90, `nefc` (number of equality, friction, and contact constraints) can exceed the default `njmax`. The error message is `nefc overflow - please increase njmax to N`. Increasing `njmax` in `LiberoEnv.__init__` fixes this. The env still loads and steps; the overflow only affects constraint solving accuracy for that step.