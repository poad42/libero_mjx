import sys

import warp as wp

MODE = sys.argv[1] if len(sys.argv) > 1 else "det"
STALE = (len(sys.argv) > 2 and sys.argv[2] == "stale")
if MODE == "det":
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN
    wp.config.deterministic_max_records = 64
print("MODE", MODE, "stale", STALE, flush=True)


@wp.func
def contact_write(counter: wp.array(dtype=int), out: wp.array(dtype=float), val: float) -> int:
    cid = wp.atomic_add(counter, 0, 1)
    if cid < out.shape[0]:
        out[cid] = val
    return cid


@wp.func
def ccd_scratch(nccd: wp.array(dtype=int), scratch: wp.array(dtype=float), dest: int, val: float) -> float:
    ccdid = wp.atomic_add(nccd, dest, 1)
    if ccdid >= 8:
        return 0.0
    return scratch[ccdid] + val


@wp.kernel(module="unique", enable_backward=False)
def ccd_like(
    ncol: wp.array(dtype=int),
    dests: wp.array(dtype=int),
    vals: wp.array(dtype=float),
    scratch: wp.array(dtype=float),
    nccd: wp.array(dtype=int),
    counter: wp.array(dtype=int),
    out: wp.array(dtype=float),
):
    tid = wp.tid()
    if tid >= ncol[0]:
        return
    needs = vals[tid] > 0.0
    r = 0.0
    if needs:
        r = ccd_scratch(nccd, scratch, dests[tid], vals[tid])
    if r > 0.5:
        contact_write(counter, out, r)


import numpy as np

dev = "cuda:0"
n = 64
ncol_np = np.zeros(1, dtype=np.int32)
ncol_np[0] = 3
dests_np = np.full(n, 5, dtype=np.int32)
vals_np = np.where(np.arange(n) < 3, 1.0, -1.0).astype(np.float32)
scratch_np = np.full(8, -1.0e30 if STALE else 0.0, dtype=np.float32)
ncol = wp.array(ncol_np, dtype=wp.int32, device=dev)
dests = wp.array(dests_np, dtype=wp.int32, device=dev)
vals = wp.array(vals_np, dtype=wp.float32, device=dev)
scratch = wp.array(scratch_np, dtype=wp.float32, device=dev)
nccd = wp.zeros(66, dtype=wp.int32, device=dev)
counter = wp.zeros(1, dtype=wp.int32, device=dev)
out = wp.zeros(8, dtype=wp.float32, device=dev)
wp.launch(ccd_like, dim=n, inputs=[ncol, dests, vals, scratch, nccd, counter, out], device=dev)
print("counter %d nccd[5] %d out0 %.3f" % (int(counter.numpy()[0]), int(nccd.numpy()[5]), float(out.numpy()[0])), flush=True)
