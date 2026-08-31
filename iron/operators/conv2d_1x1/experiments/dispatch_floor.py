#!/usr/bin/env python3
"""What is actually inside the fixed cost per call?

convnext_budget quotes it as the intercept of `latency = a + b*work`, which is
an extrapolation: no measured point sits at zero work. This measures it
directly instead, by dispatching a conv that does almost no work.

C_in = C_out = 64, 4x4 pixels, tile 16/64/64. Padded MACs = 64*64*(64*cols), and
the array retires 128*cols MAC/cycle, so the compute is 2048 cycles = 1.1 us at
ANY column count. Whatever `npu_time` reports beyond that is the per-call cost,
with no fitting involved.

Sweeping the column count then splits it. Everything that has to be programmed,
started and drained once per column scales with cols; everything else does not.
That does not name the parts, but it does bound them.

Run:  python dispatch_floor.py
"""
import json
import pathlib
import statistics
import sys

import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference

HERE = pathlib.Path(__file__).parent
FREQ = 1.8e9
WARMUP, SAMPLES = 10, 50
H = W = 4
C_IN = C_OUT = 64
TILE = (16, 64, 64)


def bench(cols):
    tm, tk, tn = TILE
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=W, C_in=C_IN, C_out=C_OUT)
        op = Conv2d1x1(batch=1, H=H, W=W, C_in=C_IN, C_out=C_OUT,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=cols,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=False, context=ctx)
        op.compile()
        fn = op.get_callable()
        rows, cs = op.input_operands(g["w"], g["x"])
        out = op.pad_output(g["y"]).flatten()
        args, ob, it = [], None, iter([rows, cs])
        for s in op.get_arg_spec():
            if s.direction == "in":
                args.append(XRTTensor.from_torch(next(it)))
            else:
                ob = XRTTensor(s.shape, dtype=s.dtype)
                args.append(ob)
        fn(*args)
        if verify_buffer(ob.to_torch(), "C", out, rel_tol=0.005, abs_tol=0.005):
            print("      WRONG", flush=True)
            return None
        for _ in range(WARMUP):
            fn(*args)
        sec = statistics.median(
            [fn(*args).npu_time / 1e3 for _ in range(SAMPLES)]) * 1e-6
        P = op.M * op.K * op.N
        peak_mac_per_cyc = 32 * 4 * cols
        compute_us = P / peak_mac_per_cyc / FREQ * 1e6
        return {"cols": cols, "cores": 4 * cols, "sec": sec,
                "us": sec * 1e6, "padded_macs": P,
                "compute_us": compute_us,
                "overhead_us": sec * 1e6 - compute_us}
    except Exception as exc:
        print(f"      FAIL {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    runs = []
    for cols in (1, 2, 4, 8):
        print(f"\n=== {cols} column(s), {4 * cols} cores ===", flush=True)
        r = bench(cols)
        if r is None:
            continue
        runs.append(r)
        print(f"  npu_time {r['us']:7.1f} us   compute {r['compute_us']:5.2f} us"
              f"   -> per-call cost {r['overhead_us']:7.1f} us", flush=True)
        (HERE / "dispatch_floor.json").write_text(json.dumps(runs, indent=2))

    if len(runs) >= 2:
        by = {r["cols"]: r["overhead_us"] for r in runs}
        print("\n--- per-call cost vs columns (compute is 1.1 us throughout) ---")
        base = by.get(1)
        for r in runs:
            rel = f"  {r['overhead_us'] / base:5.2f}x of 1 column" if base else ""
            print(f"  {r['cols']} col  {r['overhead_us']:7.1f} us{rel}")
        if 1 in by and 8 in by:
            fixed = (8 * by[1] - by[8]) / 7      # cols-independent part
            per_col = (by[8] - by[1]) / 7        # slope per extra column
            print(f"\n  linear split: {fixed:.1f} us that does not depend on the "
                  f"column count, plus {per_col:.1f} us per column")
            print("  (a two-point split, so treat it as a shape, not a precise "
                  "decomposition)")
    (HERE / "dispatch_floor.json").write_text(json.dumps(runs, indent=2))
    print("\nwrote dispatch_floor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
