import sys
import warp as wp

mode = sys.argv[1]
if mode == "det":
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN


@wp.kernel
def reserve(counter: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32), nres: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    if tid % 2 == 0:
        slot = wp.atomic_add(counter, 0, 1)
        out[slot] = wp.float(tid)
        wp.atomic_add(nres, 0, 1)


dev = "cuda:0"
c = wp.zeros(1, dtype=wp.int32, device=dev)
nres = wp.zeros(1, dtype=wp.int32, device=dev)
out = wp.zeros(128, dtype=wp.float32, device=dev)
wp.launch(reserve, dim=64, inputs=[c, out, nres], device=dev)
o = out.numpy()
print("counter %d nres %d" % (int(c.numpy()[0]), int(nres.numpy()[0])), flush=True)
written = [i for i in range(128) if o[i] != 0.0]
print("written slots %d: %s" % (len(written), written[:70]), flush=True)
print("slot values:", [int(o[i]) for i in written[:10]], flush=True)
