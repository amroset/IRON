#!/usr/bin/env python3
"""Throughput vs AIE column count, plain and swapped, for the headline layer.

Tile held at 16/64/128 throughout so the only variable is the column count.
The swap needs >= 4 columns (a_col_maj's row distribution), so at 1-2 columns
the auto-heuristic declines and only the plain mapping exists.
"""
import json, pathlib, statistics
from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

HERE = pathlib.Path(__file__).parent
H, C_IN, C_OUT = 7, 768, 3072
TM, TK, TN = 16, 64, 128
WARMUP, SAMPLES = 10, 50

def bench(cols, swap_mn):
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_IN, C_out=C_OUT)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_IN, C_out=C_OUT, tile_m=TM,
                       tile_k=TK, tile_n=TN, num_aie_columns=cols, use_scalar=False,
                       prio_accuracy=True, emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=swap_mn, context=ctx)
        op.compile(); f = op.get_callable()
        rows, cs = op.input_operands(g["w"], g["x"])
        out = op.pad_output(g["y"]).flatten()
        args, ob, it = [], None, iter([rows, cs])
        for s in op.get_arg_spec():
            args.append(XRTTensor.from_torch(next(it)) if s.direction == "in"
                        else (ob := XRTTensor(s.shape, dtype=s.dtype)))
        f(*args)
        if verify_buffer(ob.to_torch(), "C", out, rel_tol=0.005, abs_tol=0.005):
            print("      WRONG", flush=True); return None
        for _ in range(WARMUP): f(*args)
        sec = statistics.median([f(*args).npu_time/1e3 for _ in range(SAMPLES)])*1e-6
        return {"gf": 2.0*C_IN*C_OUT*H*H/sec/1e9, "sec": sec, "swap": bool(op.swap),
                "cores": 4*cols, "pad": (op.M*op.K*op.N)/(C_IN*C_OUT*H*H)}
    except Exception as e:
        print(f"      FAIL {type(e).__name__}: {str(e)[:60]}", flush=True); return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

res = {}
for cols in (1, 2, 4, 8):
    print(f"\n=== {cols} column(s), {4*cols} cores ===", flush=True)
    row = {}
    for tag, sm in (("plain", False), ("auto", None)):
        r = bench(cols, sm)
        if r is None: continue
        row[tag] = r
        print(f"  {tag:>5}  {r['gf']:7.0f} GF/s  swap={r['swap']}  pad={r['pad']:.2f}x", flush=True)
    res[str(cols)] = row
    (HERE/"col_scaling.json").write_text(json.dumps(res, indent=2))
print("\nwrote col_scaling.json", flush=True)
