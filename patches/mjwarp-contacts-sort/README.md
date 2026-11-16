# Backporting main's canonical contact sort into mujoco_warp 4.0.x

## Why

mujoco_warp 4.0.1 ships `DeterminismType` (`CONTACTS | CONSTRAINT | ATOMICS`) but
the contact-sort hook is a stub:

    def _sort_contacts(m: Model, d: Data):
      """Deterministic contact sort."""
      # TODO(team): Implementation contact sorting.
      pass

The implementation exists only on main (`_src/sort.py`, "Device-side canonical
permutations for contacts and constraints"). Without it, collision atomics leave
the contact list in arrival order, which varies run to run, so the solver input
varies and runs are not reproducible.

`ATOMICS` does not fix this: it enables warp's `RUN_TO_RUN` only for the
restructured non-collision kernels (tendon actuator force, qfrc_actuator, solver,
contact force, sleep), and the narrowphase keeps its atomics. Measured at 16
worlds, 30 steps, step-9 qpos hash: mode none 836252 vs 836254 across two runs,
and ATOMICS with the HIP header fix still 836252 vs 836254.

## What this patch does

`apply.py <site-packages>/mujoco_warp` copies `sort.py` (from main) into
`_src/` and replaces the stub body with `sort.contacts(d)`.

Field compatibility was checked against 4.0.1: every contact field the sort
touches exists (`adhesion, dim, dist, efc_address, elem, flex, frame, friction,
geom, geomcollisionid, includemargin, pos, solimp, solref, solreffriction,
surfacevel, type, vert, worldid`), as do `d.nacon, naconmax, nefc, nf, njmax,
njmax_nnz, nworld, dof_frictionadr, efc`, `m.is_sparse`, `m.eq_type`,
`warp_util.cache_kernel` and `wp.utils.array_scan`.

## Measured result (simgraph-clean:mj4, gfx1201, 16 worlds, 30 steps)

| config | run1 step9 | run2 step9 | verdict |
|---|---|---|---|
| none | 836252 | 836254 | not deterministic |
| ATOMICS + HIP header fix | 836252 | 836254 | not deterministic |
| CONTACTS + this patch | 836254 | 836254 | deterministic, nacon 1344 both |
| ALL + this patch + header fix | 836252 | 836252 | deterministic state, nacon 1344 vs 1341 |

CONTACTS is the config to use. ALL also gives a bit-identical state but the
reported contact count still varies by a few contacts that do not affect the
state, because ATOMICS changes the summation path and the narrowphase still
allocates slots nondeterministically.

## Deploy

The patch is applied at container start because the package lives in the image:

    python /out/mjwarp-contacts-sort/apply.py /opt/python/cp312-cp312/lib/python3.12/site-packages/mujoco_warp

Then set `m.opt.deterministic = mujoco_warp.DeterminismType.CONTACTS` on the
mujoco_warp model (for the LIBERO path: `wenv.wm.opt.deterministic`).
