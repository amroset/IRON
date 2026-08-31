#!/usr/bin/env python3
"""Hardware trace of the headline layer: split the 'sustained' cascade term.

The kernel already carries event0()/event1() markers around the compute body
(aie_kernels/aie2p/conv2d_1x1.cc:105,246). Enabling the trace unit captures
INSTR_EVENT_0/1 plus the stall and DMA-port counters on one core, which gives:

  * cycles between event0 and event1  -> pure microkernel time (no DMA, no dispatch)
  * MEMORY_STALL / LOCK_STALL         -> exposed data movement
  * INSTR_VECTOR                      -> vector-issue occupancy

Run:  python trace_kernel.py
"""
import json
import pathlib

import numpy as np

from iron.common import AIEContext
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.trace import parse_trace
import aie.utils as aie_utils

HERE = pathlib.Path(__file__).parent
H, C_IN, C_OUT = 7, 768, 3072
TM, TK, TN = 16, 64, 128
COLS = int(__import__("os").environ.get("CONV_COLS", "8"))
TRACE_BYTES = 262144


def main():
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_IN, C_out=C_OUT)
    op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_IN, C_out=C_OUT,
                   tile_m=TM, tile_k=TK, tile_n=TN, num_aie_columns=COLS,
                   use_scalar=False, prio_accuracy=True,
                   emulate_bf16_mmul_with_bfp16=False, swap_mn=None,
                   trace_size=TRACE_BYTES, context=ctx)
    print(f"building {op.name}", flush=True)
    op.compile()
    f = op.get_callable()
    rows, cols = op.input_operands(g["w"], g["x"])

    args, ob, it = [], None, iter([rows, cols])
    for s in op.get_arg_spec():
        if s.direction == "in":
            args.append(XRTTensor.from_torch(next(it)))
        else:
            ob = XRTTensor(s.shape, dtype=s.dtype)
            args.append(ob)
    print(f"C buffer {ob.shape}  (M*N = {op.M * op.N}, trace tail = "
          f"{TRACE_BYTES // 2} elems)", flush=True)

    res = f(*args)
    print(f"npu_time {res.npu_time / 1e3:.1f} us", flush=True)

    raw = ob.to_torch().flatten().numpy()
    tail = raw[op.M * op.N:]
    words = np.frombuffer(tail.tobytes(), dtype=np.uint32)
    nz = int((words != 0).sum())
    print(f"trace words {len(words)}  nonzero {nz}", flush=True)
    (HERE / "trace_raw.txt").write_text("\n".join(f"{int(w):08x}" for w in words))

    mlir_path = next(iter(sorted(pathlib.Path(ctx.base_dir).glob(f"{op.name}.mlir"))), None)
    if mlir_path is None:
        print("no MLIR found; raw trace written for offline parsing", flush=True)
        return
    events = parse_trace(words, mlir_path.read_text())
    (HERE / "trace_events.json").write_text(json.dumps(events, indent=1))
    print(f"parsed {len(events)} trace events -> trace_events.json", flush=True)

    names = {}
    for e in events:
        names[e.get("name", "?")] = names.get(e.get("name", "?"), 0) + 1
    for k, v in sorted(names.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {k:>28}  {v}", flush=True)

    aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    main()
