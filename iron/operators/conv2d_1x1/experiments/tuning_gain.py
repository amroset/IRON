#!/usr/bin/env python3
"""What tile autotuning buys, per layer, on top of the swap.

For every real-model layer, measures the production mapping (swap_mn=None) twice
in the SAME session -- once on the default tile 16/64/64 and once on the tuned
tile from tile_cache.json -- so the comparison is not exposed to run-to-run
drift between separate sweeps.

Records tile_m for each, because the README's mechanism is that tuning can only
help where the row axis is large enough to absorb a bigger m; on the 7x7 layers
the swap puts ~49 pixels on the rows and pins m=16.

Run:  CONV_COLS=8 python tuning_gain.py
"""
import json
import os
import pathlib
import statistics
import sys

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from vector_vs_scalar import SHAPES
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

HERE = pathlib.Path(__file__).parent
COLS = int(os.environ.get("CONV_COLS", "8"))
DEFAULT_TILE = (16, 64, 64)
WARMUP, SAMPLES = 10, 50
CACHE = json.loads((HERE.parent / "tile_cache.json").read_text())


def bench(H, C_in, C_out, tile):
    tm, tk, tn = tile
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=COLS,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=None, context=ctx)
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
            print("      NUMERICALLY WRONG", flush=True)
            return None
        for _ in range(WARMUP):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(SAMPLES)]
        sec = statistics.median(lat) * 1e-6
        return {"gf": 2.0 * C_in * C_out * H * H / sec / 1e9, "sec": sec,
                "tile": list(tile), "swap": bool(op.swap),
                "pad": (op.M * op.K * op.N) / (C_in * C_out * H * H)}
    except Exception as e:
        print(f"      FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    out_path = HERE / "tuning_gain.json"
    results = {}
    for (H, C_in, C_out, model, role) in SHAPES:
        N = H * H
        key = f"{C_in}/{C_out}/{N}"
        tuned = tuple(CACHE.get(f"{C_in}_{C_out}_{N}_{COLS}", DEFAULT_TILE))
        print(f"\n=== {key}  {model} {role}   default {DEFAULT_TILE} -> tuned {tuned} ===",
              flush=True)
        row = {"model": model, "role": role, "H": H, "C_in": C_in, "C_out": C_out, "N": N}
        for tag, tile in (("default", DEFAULT_TILE), ("tuned", tuned)):
            r = bench(H, C_in, C_out, tile)
            if r is None:
                continue
            row[tag] = r
            print(f"  {tag:>8} {str(tile):>14}  {r['gf']:7.0f} GF/s  swap={r['swap']}",
                  flush=True)
        if "default" in row and "tuned" in row:
            row["gain"] = row["tuned"]["gf"] / row["default"]["gf"]
            print(f"  -> {row['gain']:.3f}x", flush=True)
        results[key] = row
        out_path.write_text(json.dumps(results, indent=2))

    gains = [(v["gain"], k, v) for k, v in results.items() if "gain" in v]
    if gains:
        gs = sorted(g[0] for g in gains)
        med = gs[len(gs) // 2]
        print(f"\n{len(gs)} layers · median {med:.3f}x · max {max(gs):.3f}x", flush=True)
        big = [g for g in gains if g[2]["tuned"]["tile"][0] > 16]
        pin = [g for g in gains if g[2]["tuned"]["tile"][0] == 16]
        for name, grp in (("m grew past 16", big), ("m pinned at 16", pin)):
            if grp:
                avg = sum(g[0] for g in grp) / len(grp)
                print(f"  {name:>16}: n={len(grp):>2}  mean {avg:.3f}x", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
