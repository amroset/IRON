#!/usr/bin/env python3
"""What IS the launch floor, and is it ours or the framework's?

final_budget.py fits `latency = a + b * padded_MACs` over a work sweep and calls
the intercept `a` the launch floor. That is only meaningful if `a` really is
work-independent, and only interesting if we know whether it is a property of
our swapped mapping or of every dispatch through this design.

Note the sweep varies C_out only, so EVERY quantity that scales with the problem
-- padded MACs, and A/B/C DRAM traffic alike -- scales linearly with it. The
intercept therefore excludes all of them by construction: it is the part of a
dispatch that does not depend on how much work the dispatch carries.

Three configurations, same fit, so the intercepts are comparable:

  swap    the headline layer (768->3072 @7x7), cols->M swap engaged
  plain   a wide layer (512->C_out @28x28) the heuristic leaves on the plain
          mapping -- same design.py, same runtime sequence, no swap
  plain4  the same wide layer on 4 columns, to see whether the floor tracks the
          number of shim DMAs that have to be programmed

Run:  python launch_floor.py
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

CONFIGS = {
    # tag:            (H, C_in, tile,         cols, C_out sweep, force_swap)
    "swap":           (7, 768, (16, 64, 128), 8, [1024, 2048, 3072, 4096, 5120, 6144], None),
    # THE CONTROL: same tiny layer, same work scale, swap forced OFF. If the
    # intercept survives here it cannot be something the swap introduces.
    "same_layer_plain": (7, 768, (16, 64, 64), 8, [1024, 2048, 3072, 4096, 5120, 6144], False),
    # A wide layer, where the work per dispatch is 5-20x larger.
    "wide_plain":     (28, 512, (32, 64, 64), 8, [512, 1024, 1536, 2048, 2560, 3072], None),
}


def bench(H, C_in, C_out, tile, cols, force_swap=None):
    tm, tk, tn = tile
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=cols,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=force_swap, context=ctx)
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
        return {"C_out": C_out, "sec": sec, "swap": bool(op.swap),
                "P": op.M * op.K * op.N, "U": C_in * C_out * H * H,
                "gf": 2.0 * C_in * C_out * H * H / sec / 1e9}
    except Exception as exc:
        print(f"      FAIL {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def fit(points):
    P = [p["P"] for p in points]
    T = [p["sec"] for p in points]
    n = len(P)
    mp, mt = sum(P) / n, sum(T) / n
    b = (sum((x - mp) * (y - mt) for x, y in zip(P, T))
         / sum((x - mp) ** 2 for x in P))
    a = mt - b * mp
    ss = sum((y - mt) ** 2 for y in T)
    rs = sum((y - (a + b * x)) ** 2 for x, y in zip(P, T))
    return a, b, 1 - rs / ss


def main():
    out = {}
    for tag, (H, C_in, tile, cols, sweep, force_swap) in CONFIGS.items():
        print(f"\n===== {tag}: {C_in}->C_out @{H}x{H}, tile "
              f"{tile[0]}/{tile[1]}/{tile[2]}, {cols} cols =====", flush=True)
        pts = []
        for C_out in sweep:
            r = bench(H, C_in, C_out, tile, cols, force_swap)
            if r is None:
                continue
            pts.append(r)
            print(f"  C_out={C_out:>5}  {r['sec'] * 1e6:8.1f} us  "
                  f"{r['gf']:7.0f} GF/s  swap={r['swap']}", flush=True)
        if len(pts) < 3:
            continue
        a, b, r2 = fit(pts)
        out[tag] = {"H": H, "C_in": C_in, "tile": list(tile), "cols": cols,
                    "points": pts, "a_sec": a, "b_sec_per_mac": b, "r2": r2,
                    "a_us": a * 1e6, "swap": pts[0]["swap"]}
        print(f"  fit: intercept a = {a * 1e6:.1f} us   R2 = {r2:.4f}", flush=True)
        (HERE / "launch_floor.json").write_text(json.dumps(out, indent=2))

    print("\n--- launch floor, work-independent intercept ---")
    for tag, r in out.items():
        print(f"  {tag:<8} {r['cols']} cols  swap={str(r['swap']):<5}  "
              f"a = {r['a_us']:6.1f} us   R2 = {r['r2']:.4f}")
    if "swap" in out and "same_layer_plain" in out:
        print(f"\nControl: the SAME layer with the swap off has an intercept of "
              f"{out['same_layer_plain']['a_us']:.0f} us, against "
              f"{out['swap']['a_us']:.0f} us with the swap on.")
    if "wide_plain" in out:
        w = out["wide_plain"]
        print(f"On the wide layer the linear model fits poorly (R2 = {w['r2']:.3f}) "
              f"and the intercept comes out {w['a_us']:.0f} us -- unphysical, i.e. "
              f"NOT RESOLVABLE there: the floor is a few percent of a "
              f"{min(p['sec'] for p in w['points']) * 1e6:.0f}-{max(p['sec'] for p in w['points']) * 1e6:.0f} us "
              f"dispatch and disappears into the scatter. Only report a floor "
              f"where the fit supports one.")
    (HERE / "launch_floor.json").write_text(json.dumps(out, indent=2))
    print("\nwrote launch_floor.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
