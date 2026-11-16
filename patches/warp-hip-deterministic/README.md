# warp 1.17.0+rocm.0: deterministic lowering is dead code on HIP, and the fix

## The defect

`warp/native/deterministic.h` gates the entire deterministic lowering on
`#ifdef __CUDA_ARCH__`. The ROCm fork compiles user kernels for gfx targets via
hipRTC, which defines `__HIP_DEVICE_COMPILE__` but never `__CUDA_ARCH__`. Every
other device gate in the fork (crt.h, builtin.h) tests both arms;
deterministic.h tests only the CUDA arm. Result: on gfx every deterministic
atomic lowers to the plain fallback `cpu_expr`, while the Python launcher still
runs the two-pass count-scan-write. Both passes then execute the real atomic and
the real stores, so every consumed-return counter doubles per launch and outputs
are written twice.

Measured (probe `det_counter2.py`, gfx1201, one launch of 32 reservations):

    flag off: counter 32, slots 0-31 written once
    flag on:  counter 64, slots 0-31 AND 32-63 written

Warp's own test suite masks this by construction:
`warp/tests/deterministic/common.py` filters `if not device.is_hip`, so 25 of 27
counter tests skip on AMD with "No suitable devices to run the test."

## The fix (deterministic.h in this directory)

1. All five lowering gates change from `#ifdef __CUDA_ARCH__` to
   `#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)`.
2. `is_global_store_target` gains a HIP arm. CUDA queries the address space with
   inline PTX (`isspacep.global`); HIP has no `__isGlobal`, and the first attempt
   (`!__builtin_amdgcn_is_shared(ptr)`) misclassified private locals as global,
   which suppressed phase-0 stores to local scratch and made counters
   undercount (nacon 0). The correct arm is
   `return !__builtin_amdgcn_is_shared(ptr) && !__builtin_amdgcn_is_private(ptr);`

With this header bind-mounted over the container's copy and the HIP filter
removed from common.py, the fork's own `test_deterministic_counter.py` runs 29 of
29 on gfx1201 (previously 25 skipped, and the remainder untested).

Deploy: bind-mount the file over `/opt/warp_src/warp/native/deterministic.h` and
point `WARP_CACHE_DIR` at a fresh directory (the module hash does not cover
header changes, so a warm cache silently keeps the broken lowering).

## Remaining limitation: mujoco_warp's ccd kernel violates the two-pass contract

The deterministic docs require "both runs make the same atomic_add() calls".
mujoco_warp 3.13's `ccd_kernel` violates this: it reserves `ccdid =
wp.atomic_add(nccd, geomgeomid, 1)`, then uses `ccdid` to index scratch it
writes and reads back (`epa_vert_in[ccdid]` ...). In phase 0 stores are
suppressed and every thread gets slot 0, so `dist` is computed from stale
scratch and the kernel exits before `contact_write`. The nacon target records
zero reservations, the scan skips it (record_count 0), the writeback skips it,
and phase 1's unmatched call gets `record_slots = -1` and returns slot 0 with
no error. nacon stays 0 forever, contacts vanish, the solve NaNs.

Reproduced independently of mujoco with `ccd_pattern.py` (kernel with a guard on
a counter read, a data-dependent consumed atomic, slot-indexed scratch read
back, and a second consumed atomic in a helper): clean scratch gives identical
det/nondet results; stale scratch silently drops the second counter while the
first still counts. The failure is device-independent; CUDA would behave the
same way. Fixing it needs a mujoco_warp kernel restructure, not a warp change.

Until then, the eval keeps `wp.config.deterministic` gated to warp < 1.14
(`fast/eval_fast.py`), where the attribute does not exist and the flag is a
no-op.
