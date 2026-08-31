#!/usr/bin/env python3
"""Cycle budget for the FINAL operator on the headline layer.

Layer: ConvNeXt-T pw_up, C_in=768 -> C_out=3072 @ 7x7, swap engaged,
autotuned tile 16/64/128 (tile_cache.json), 8 AIE columns.

Builds an exact multiplicative cascade:

    of_peak = padding x sustained x floor_dilution

which is an identity, not a fit:
    padding  = U / P                        (mapping arithmetic, exact)
    sustained= 1 / (1024 * b * f)           (b = slope of latency vs padded MACs)
    floor    = b*P / (a + b*P)              (a = intercept, the launch floor)

`sustained` still lumps microkernel efficiency, per-call overhead and
multi-core contention together. The cols sweep splits contention off; the
microkernel term needs the hardware trace window, which this design does not
yet wire up (design.py accepts trace_size and ignores it).

Run:  python final_budget.py
"""
import json
import os
import pathlib
import statistics

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

HERE = pathlib.Path(__file__).parent
H, C_IN, C_OUT = 7, 768, 3072
TM, TK, TN = 16, 64, 128          # tuned tile for this shape, cols=8
FREQ = 1.8e9
WARMUP, SAMPLES = 10, 50          # match the reference deck's protocol

# Column quantum is TN*cols; sweep C_out in exact multiples so the tiling stays
# exact and only the WORK changes -- the controlled-sweep design.
WORK_SWEEP = [1024, 2048, 3072, 4096, 5120, 6144]


def bench(C_out, cols, samples=SAMPLES):
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_IN, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_IN, C_out=C_out,
                       tile_m=TM, tile_k=TK, tile_n=TN, num_aie_columns=cols,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=None, context=ctx)
        op.compile()
        f = op.get_callable()
        rows, colsb = op.input_operands(g["w"], g["x"])
        out = op.pad_output(g["y"]).flatten()
        args, ob, it = [], None, iter([rows, colsb])
        for s in op.get_arg_spec():
            if s.direction == "in":
                args.append(XRTTensor.from_torch(next(it)))
            else:
                ob = XRTTensor(s.shape, dtype=s.dtype)
                args.append(ob)
        f(*args)
        if verify_buffer(ob.to_torch(), "C", out, rel_tol=0.005, abs_tol=0.005):
            print("    NUMERICALLY WRONG", flush=True)
            return None
        for _ in range(WARMUP):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(samples)]
        sec = statistics.median(lat) * 1e-6
        U = C_IN * C_out * H * H
        P = op.M * op.K * op.N
        return {"C_out": C_out, "cols": cols, "sec": sec, "U": U, "P": P,
                "swap": bool(op.swap), "gf": 2.0 * U / sec / 1e9}
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def fit(points):
    """Least squares  sec = a + b*P.  Returns (a, b, R^2)."""
    P = [p["P"] for p in points]; T = [p["sec"] for p in points]
    n = len(P); mp = sum(P) / n; mt = sum(T) / n
    b = sum((x - mp) * (y - mt) for x, y in zip(P, T)) / sum((x - mp) ** 2 for x in P)
    a = mt - b * mp
    ss = sum((y - mt) ** 2 for y in T)
    rs = sum((y - (a + b * x)) ** 2 for x, y in zip(P, T))
    return a, b, 1 - rs / ss


def main():
    out = {}
    for cols in (8, 4):
        peak_mac_per_cyc = 32 * 4 * cols          # 32 MAC/cyc x 4 rows x cols
        print(f"\n===== cols={cols}  ({4 * cols} cores) =====", flush=True)
        pts = []
        for C_out in WORK_SWEEP:
            r = bench(C_out, cols)
            if r is None:
                continue
            pts.append(r)
            print(f"  C_out={C_out:>5}  {r['sec'] * 1e6:8.1f} us  {r['gf']:7.0f} GF/s  "
                  f"pad={r['P'] / r['U']:.3f}x  swap={r['swap']}", flush=True)
        if len(pts) < 3:
            continue
        a, b, r2 = fit(pts)
        sustained = 1.0 / (peak_mac_per_cyc * b * FREQ)
        head = next((p for p in pts if p["C_out"] == C_OUT), None)
        rec = {"points": pts, "a_sec": a, "b_sec_per_mac": b, "r2": r2,
               "sustained": sustained, "peak_mac_per_cyc": peak_mac_per_cyc}
        if head:
            padding = head["U"] / head["P"]
            floor = (b * head["P"]) / (a + b * head["P"])
            of_peak = head["U"] / (head["sec"] * peak_mac_per_cyc * FREQ)
            rec.update({"headline": head, "padding": padding, "floor": floor,
                        "of_peak": of_peak,
                        "cascade": padding * sustained * floor})
            print(f"\n  fit: a={a * 1e6:.1f} us   b={b:.3e} s/MAC   R^2={r2:.4f}")
            print(f"  padding   {padding:.4f}")
            print(f"  sustained {sustained:.4f}   (microkernel x per-call x contention)")
            print(f"  floor     {floor:.4f}")
            print(f"  cascade   {padding * sustained * floor * 100:.2f}%  vs measured "
                  f"{of_peak * 100:.2f}%")
        out[f"cols{cols}"] = rec
    if "cols8" in out and "cols4" in out:
        c = out["cols8"]["sustained"] / out["cols4"]["sustained"]
        out["contention_32_vs_16_cores"] = c
        print(f"\ncontention (32 vs 16 cores): {c:.4f}")
    (HERE / "final_budget.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'final_budget.json'}", flush=True)


if __name__ == "__main__":
    main()
