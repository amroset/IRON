#!/usr/bin/env python3
"""A/B: true-bf16 mmul (emulate=False) vs block-float mmul (emulate=True).

The whole project runs with emulate_bf16_mmul_with_bfp16=False, which sends
aie::mmul<...,bf16> down the FPU emulation (8x mac_elem_32 per logical mmul,
32 MAC/cyc/core -> the 3.69 TF/s ceiling). Setting the flag converts operands to
bfp16ebs8 (block float: one shared exponent per 8 values) and issues ONE real
matrix instruction (mac_8x8_8x8T_conf) per logical mmul instead.

Measures both halves of the trade:
    speed    - median NPU time, same protocol as the reference deck (10/50)
    accuracy - max/mean relative error vs the torch golden, the suite's own
               0.005 rel/abs verdict, and how far the two paths differ

IMPORTANT constraint found while building this: the block-float path forces
(r,s,t)=(8,8,8), so MMUL::size_C becomes 64 f32 elements, and the col-major
in-register transposes the cols->M swap relies on (aie::transpose on a 64-wide
32-bit vector) are NOT implemented in aie_api -- transpose_bits_impl<32,T,Elems>
only defines shuffle_modes for Elems in {4, 8}. So swapped layers cannot compile
against block float at all. The swap-engaged config is included here to record
that, and the headline layer is ALSO run with swap forced off so the two
datapaths can be compared on the same shape.

Run:  /scratch/amrosetti/ironenv/bin/python bfp16_ab.py
"""
import json
import pathlib
import statistics

import torch

from iron.common import AIEContext
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

HERE = pathlib.Path(__file__).parent
COLS = 8
WARMUP, SAMPLES = 10, 50
TOL = 0.005

# (H, C_in, C_out, tm, tk, tn, swap_mn, label)
# Tiles satisfy both paths: emulate=False -> (4,8,8) needs m%8,k%8,n%16;
#                           emulate=True  -> (8,8,8) needs m%16,k%8,n%16.
CONFIGS = [
    (28, 128,  128,  16, 64, 64, None,  "MobileNet mid 28x28 128->128   (plain)"),
    (28, 512,  512,  16, 64, 64, None,  "wide 28x28 512->512            (plain)"),
    (14, 512,  256,  16, 64, 64, None,  "ResNet 14x14 512->256          (plain)"),
    (7,  768,  3072, 16, 64, 64, False, "ConvNeXt pw_up, swap FORCED OFF (plain)"),
    (7,  768,  3072, 16, 64, 64, None,  "ConvNeXt pw_up, swap AUTO      (swap ON)"),
]


def bench(H, C_in, C_out, tm, tk, tn, swap_mn, emulate):
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=COLS,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=emulate,
                       swap_mn=swap_mn, context=ctx)
        op.compile()
        f = op.get_callable()
        rows, colsb = op.input_operands(g["w"], g["x"])

        args, ob, it = [], None, iter([rows, colsb])
        for s in op.get_arg_spec():
            if s.direction == "in":
                args.append(XRTTensor.from_torch(next(it)))
            else:
                ob = XRTTensor(s.shape, dtype=s.dtype)
                args.append(ob)
        f(*args)

        # accuracy on the LOGICAL region only; unpad_output -> [C_out, H*W]
        got = op.unpad_output(ob.to_torch()).to(torch.float64)
        ref = g["y"].reshape(C_out, H * H).to(torch.float64)
        err = (got - ref).abs()
        rel = err / ref.abs().clamp_min(1e-12)
        ok = bool(((err <= TOL) | (rel <= TOL)).all())

        for _ in range(WARMUP):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(SAMPLES)]
        sec = statistics.median(lat) * 1e-6

        U = C_in * C_out * H * H
        return {"sec": sec, "gf": 2.0 * U / sec / 1e9, "swap": bool(op.swap),
                "passes_tol": ok, "max_abs": float(err.max()),
                "max_rel": float(rel.max()), "mean_rel": float(rel.mean()),
                "out": got}
    except Exception as e:
        msg = str(e)
        kind = "COMPILE ERROR" if "Command failed" in msg else type(e).__name__
        print(f"      -> {kind}: {msg[:100]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    out = []
    for (H, C_in, C_out, tm, tk, tn, swap_mn, label) in CONFIGS:
        print(f"\n=== {label}", flush=True)
        print(f"    {C_in}->{C_out} @ {H}x{H}, tile {tm}/{tk}/{tn}, cols={COLS}", flush=True)
        rec = {"label": label, "H": H, "C_in": C_in, "C_out": C_out,
               "tile": [tm, tk, tn], "swap_mn": swap_mn}
        res = {}
        for emulate in (False, True):
            tag = "bfp16 block-float " if emulate else "true bf16 (current)"
            r = bench(H, C_in, C_out, tm, tk, tn, swap_mn, emulate)
            res["bfp16" if emulate else "bf16"] = r
            if r:
                print(f"    {tag} {r['sec']*1e6:8.1f} us {r['gf']:7.0f} GF/s  "
                      f"swap={r['swap']}  tol={'PASS' if r['passes_tol'] else 'FAIL'}  "
                      f"max_rel={r['max_rel']:.2e}  mean_rel={r['mean_rel']:.2e}",
                      flush=True)
        a, b = res.get("bf16"), res.get("bfp16")
        for k, r in res.items():
            if r:
                rec[k] = {kk: vv for kk, vv in r.items() if kk != "out"}
        if a and b:
            rec["speedup"] = a["sec"] / b["sec"]
            d = (a["out"] - b["out"]).abs()
            rec["paths_max_absdiff"] = float(d.max())
            rec["paths_mean_absdiff"] = float(d.mean())
            print(f"    --> block-float is {rec['speedup']:.2f}x faster; "
                  f"accuracy {'PASSES' if b['passes_tol'] else 'FAILS'} 0.005 tol "
                  f"(max_rel {b['max_rel']:.2e} vs bf16's {a['max_rel']:.2e})",
                  flush=True)
        elif a and not b:
            rec["bfp16_unavailable"] = True
            print("    --> block-float UNAVAILABLE for this config", flush=True)
        out.append(rec)

    (HERE / "bfp16_ab.json").write_text(json.dumps(out, indent=1))

    print("\n\n=========================== SUMMARY ===========================", flush=True)
    print(f"{'config':<42}{'bf16':>9}{'bfp16':>9}{'speedup':>9}{'bfp16 max_rel':>15}{'tol':>6}")
    for r in out:
        a, b = r.get("bf16"), r.get("bfp16")
        if a and b:
            print(f"{r['label'][:42]:<42}{a['gf']:8.0f}G{b['gf']:8.0f}G"
                  f"{r['speedup']:8.2f}x{b['max_rel']:15.2e}"
                  f"{'PASS' if b['passes_tol'] else 'FAIL':>6}")
        elif a:
            print(f"{r['label'][:42]:<42}{a['gf']:8.0f}G{'n/a':>9}{'--':>9}"
                  f"{'block float will not compile':>15}")
        else:
            print(f"{r['label'][:42]:<42}   (both failed)")


if __name__ == "__main__":
    main()
