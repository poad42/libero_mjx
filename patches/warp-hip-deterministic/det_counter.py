import sys
import warp as wp

mode = sys.argv[1]
if mode == "det":
    wp.config.deterministic = wp.DeterministicMode.RUN_TO_RUN


@wp.kernel
def reserve(counter: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    if tid % 2 == 0:
        slot = wp.atomic_add(counter, 0, 1)
        if slot < out.shape[0]:
            out[slot] = wp.float(tid)
        else:
            out[0] = -1.0


dev = "cuda:0"
c = wp.zeros(1, dtype=wp.int32, device=dev)
out = wp.zeros(64, dtype=wp.float32, device=dev)
expected = 0
for i in range(5):
    wp.launch(reserve, dim=64, inputs=[c, out], device=dev)
    v = int(c.numpy()[0])
    expected += 32
    print("launch %d counter %d expected %d %s" % (i, v, expected, "OK" if v == expected else "WRONG"), flush=True)
o = out.numpy()
print("slots used:", int((o >= 0).sum()) if o[0] != -1.0 else "OVERFLOW", flush=True)
print("first 8 out:", o[:8].tolist(), flush=True)
