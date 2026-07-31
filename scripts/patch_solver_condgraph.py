#!/usr/bin/env python3
"""Patch mujoco_warp solver.py to pass max_iters to capture_while calls.

This enables the HIP on-device fast-path (hipStreamWaitValue32 CP gate)
which eliminates per-iteration host D2H + stream sync in the Newton solver
loop, giving ~10% physics step speedup on ROCm/RDNA4.

The patch is idempotent — safe to run multiple times.
"""
import sys

SOLVER_PATH = "/opt/venv/lib/python3.12/site-packages/mujoco/mjx/third_party/mujoco_warp/_src/solver.py"

def main():
    with open(SOLVER_PATH) as f:
        content = f.read()

    changed = False

    # Patch 1: single-line capture_while (monolithic solver)
    old1 = "wp.capture_while(nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, nsolving=nsolving)"
    new1 = "wp.capture_while(nsolving, while_body=_solver_iteration, m=m, d=d, ctx=ctx, nsolving=nsolving, max_iters=m.opt.iterations)"
    if old1 in content:
        content = content.replace(old1, new1)
        changed = True
        print("  patched monolithic solver capture_while")

    # Patch 2: multi-line capture_while (island solver)
    old2 = "      nsolving=nsolving,\n    )"
    new2 = "      nsolving=nsolving,\n      max_iters=m.opt.iterations,\n    )"
    if old2 in content:
        content = content.replace(old2, new2)
        changed = True
        print("  patched island solver capture_while")

    if changed:
        with open(SOLVER_PATH, "w") as f:
            f.write(content)
        print("Done — solver patched for HIP on-device conditional fast-path")
    else:
        print("Already patched (or pattern not found) — no changes")

if __name__ == "__main__":
    main()