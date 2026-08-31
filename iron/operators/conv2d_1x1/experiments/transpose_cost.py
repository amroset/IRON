#!/usr/bin/env python3
"""Does the swap's in-kernel transpose explain the 'kernel + buffers' term?

Same layer, same tile 16/64/128, same 8 columns -- only the mapping differs, so
the only change inside the microkernel is that all three operands become
col-major and every A/B load and C load+store carries an aie::transpose.

`sustained` is measured per PADDED MAC, so the two mappings' very different
padding cancels out and the slopes are directly comparable:

    sustained(plain) - sustained(swap)  =  the transpose cost

Plain caps at C_out=4096: rows carry C_out there, and C_out/(tile_m*4) must stay
inside the 64 block-descriptor limit.

Run:  python transpose_cost.py
"""
import json
import pathlib
import statistics

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

HERE = pathlib.Path(__file__).parent
H, C_IN = 7, 768
TM, TK, TN = 16, 64, 128
COLS = 8
FREQ = 1.8e9
PEAK_MAC_PER_CYC = 32 * 4 * COLS
WARMUP, SAMPLES = 10, 50
SWEEP = [1024, 2048, 3072, 4096]          # BD limit: C_out/(TM*4) <= 64


def bench(C_out, swap_mn):
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_IN, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_IN, C_out=C_out,
                       tile_m=TM, tile_k=TK, tile_n=TN, num_aie_columns=COLS,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=swap_mn, context=ctx)
        if bool(op.swap) != bool(swap_mn):
            print(f"    swap not honoured (got {op.swap})", flush=True)
            return None
        op.compile()
        f = op.get_callable()
        rows, cols = op.input_operands(g["w"], g["x"])
        out = op.pad_output(g["y"]).flatten()
        args, ob, it = [], None, iter([rows, cols])
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
        lat = [f(*args).npu_time / 1e3 for _ in range(SAMPLES)]
        sec = statistics.median(lat) * 1e-6
        return {"C_out": C_out, "sec": sec, "P": op.M * op.K * op.N,
                "U": C_IN * C_out * H * H, "swap": bool(op.swap)}
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def fit(points):
    P = [p["P"] for p in points]; T = [p["sec"] for p in points]
    n = len(P); mp = sum(P) / n; mt = sum(T) / n
    b = sum((x - mp) * (y - mt) for x, y in zip(P, T)) / sum((x - mp) ** 2 for x in P)
    a = mt - b * mp
    ss = sum((y - mt) ** 2 for y in T)
    rs = sum((y - (a + b * x)) ** 2 for x, y in zip(P, T))
    return a, b, 1 - rs / ss


def main():
    out = {}
    for tag, swap_mn in (("plain", False), ("swap", True)):
        print(f"\n===== {tag} (swap_mn={swap_mn}) =====", flush=True)
        pts = []
        for C_out in SWEEP:
            r = bench(C_out, swap_mn)
            if r is None:
                continue
            pts.append(r)
            print(f"  C_out={C_out:>5}  {r['sec']*1e6:9.1f} us  "
                  f"P={r['P']/1e6:8.1f}M  pad={r['P']/r['U']:6.2f}x", flush=True)
        if len(pts) < 3:
            continue
        a, b, r2 = fit(pts)
        sus = 1.0 / (PEAK_MAC_PER_CYC * b * FREQ)
        out[tag] = {"points": pts, "a_sec": a, "b": b, "r2": r2, "sustained": sus,
                    "mac_per_cyc_per_core": 1 / (b * FREQ) / 32}
        print(f"  fit a={a*1e6:6.1f}us  R2={r2:.4f}  sustained={sus:.4f}  "
              f"({1/(b*FREQ)/32:.2f} MAC/cyc/core)", flush=True)

    if "plain" in out and "swap" in out:
        p, s = out["plain"]["sustained"], out["swap"]["sustained"]
        out["transpose_cost"] = s / p
        print(f"\nsustained  plain {p:.4f}   swap {s:.4f}")
        print(f"swap/plain = {s/p:.4f}   -> transposes cost {(1-s/p)*100:.1f}% of the rate")
    (HERE / "transpose_cost.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {HERE / 'transpose_cost.json'}", flush=True)


if __name__ == "__main__":
    main()
