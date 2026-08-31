#!/usr/bin/env python3
"""Production-config sweep + cycle budget.

Measures the COMPLETE operator (swap_mn=None, i.e. the auto-heuristic decides)
for all 22 real-model shapes, then decomposes the elapsed array-cycles into
three buckets that sum exactly to the measured total:

    useful   = U / 1024              MACs the answer actually needs
    padding  = (P - U) / 1024        MACs spent multiplying zeros
    residual = A - P / 1024          cycles issuing no MAC at all

with  U = useful MACs, P = padded MACs, A = t x 1.8 GHz array-cycles,
and 1024 MAC/array-cycle = 32 cores x 32 MAC/cyc.

`residual` is a measured leftover, not an attribution: it lumps DMA stalls,
accumulator load/store, in-kernel transposes, loop and dispatch overhead. No
hardware counters here can split it further -- see the README note.

Run:  CONV_COLS=8 python cycle_budget.py
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

COLS = int(os.environ.get("CONV_COLS", "8"))
TM, TK, TN = 16, 64, 64
NUM_ROWS = 4
FREQ = 1.8e9
MAC_PER_ARRAY_CYCLE = 32 * 32  # 32 cores x 32 MAC/cyc
WARMUP, SAMPLES = 10, 50
HERE = pathlib.Path(__file__).parent


def bench(H, C_in, C_out, swap_mn):
    """Build+verify+time one config. Returns (gf, seconds, op) or (None, None, None)."""
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=TM, tile_k=TK, tile_n=TN, num_aie_columns=COLS,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=swap_mn, context=ctx)
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
            return None, None, None
        for _ in range(WARMUP):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(SAMPLES)]
        sec = statistics.median(lat) * 1e-6
        gf = 2.0 * C_out * C_in * (H * H) / sec / 1e9
        return gf, sec, op
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {str(e)[:70]}", flush=True)
        return None, None, None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def budget(op, C_in, C_out, N, sec):
    """Split elapsed array-cycles into useful / padding / residual."""
    U = C_in * C_out * N
    P = op.M * op.K * op.N          # padded, array-facing dims
    A = sec * FREQ                  # elapsed array-cycles
    c_useful = U / MAC_PER_ARRAY_CYCLE
    c_padded = P / MAC_PER_ARRAY_CYCLE
    return {
        "useful_cyc": c_useful,
        "padding_cyc": c_padded - c_useful,
        "residual_cyc": A - c_padded,
        "total_cyc": A,
        "pad_overhead": P / U,
        "mac_issue_eff": c_padded / A,     # fraction of cycles actually issuing MACs
        "of_peak": c_useful / A,           # useful MACs as fraction of datapath
    }


def main():
    out_path = HERE / os.environ.get("OUT", "cycle_budget.json")
    results = {}
    for (H, C_in, C_out, model, role) in SHAPES:
        N = H * H
        key = f"{C_in}/{C_out}/{N}"
        print(f"\n=== {key}  {model} {role} ===", flush=True)
        row = {"model": model, "role": role, "H": H, "C_in": C_in, "C_out": C_out, "N": N}
        for tag, swap_mn in (("plain", False), ("prod", None)):
            gf, sec, op = bench(H, C_in, C_out, swap_mn)
            if gf is None:
                continue
            b = budget(op, C_in, C_out, N, sec)
            row[tag] = {"gf": gf, "sec": sec, "swap": bool(op.swap), **b}
            print(f"  {tag:>5} swap={str(bool(op.swap)):>5}  {gf:7.0f} GF/s  "
                  f"pad={b['pad_overhead']:5.2f}x  issue={b['mac_issue_eff']*100:4.1f}%  "
                  f"peak={b['of_peak']*100:4.1f}%", flush=True)
        results[key] = row
        out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
