#!/usr/bin/env python3
"""Head-to-head: does shrinking tile_n (finer column quantum, plain mapping)
substitute for the cols->M swap on starved layers?

Configs per shape, all at cols=8:
  plain_n64  swap_mn=False 16/64/64   the starved baseline
  plain_n16  swap_mn=False 16/64/16   the untested hypothesis (finer columns)
  swap_n64   swap_mn=True  16/64/64   production
  swap_n16   swap_mn=True  16/64/16   completeness

Measurement path copied from autotune.py (same op settings as test.py):
build -> verify vs torch golden -> warmup -> median of SAMPLES npu_time.
"""
import json
import os
import statistics
import sys
import traceback

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

COLS = int(os.environ.get("CONV_COLS", "8"))
WARMUP = int(os.environ.get("CONV_WARMUP", "10"))
SAMPLES = int(os.environ.get("CONV_SAMPLES", "50"))
BD_MAX = 64
NUM_ROWS = 4

# (label, swap_mn, tm, tk, tn)
CONFIGS = [
    ("plain_n64", False, 16, 64, 64),
    ("plain_n16", False, 16, 64, 16),
    ("swap_n64", True, 16, 64, 64),
    ("swap_n16", True, 16, 64, 16),
]

# Starved (tall) layers: every 7x7 in the suite, the 14x14 tall ones, plus the
# N=400 control that the auto-heuristic correctly leaves on the plain mapping.
SHAPES = [
    (7, 768, 3072),   # ConvNeXt-Tiny pw_up   -- the headline case
    (7, 1024, 2048),  # ResNeXt-50 expand
    (7, 512, 2048),   # ResNet-50 deep expand
    (7, 2048, 512),   # ResNet-50 deep reduce
    (7, 3072, 768),   # ConvNeXt-Tiny pw_down
    (7, 1152, 192),   # EfficientNet-B0 projection
    (14, 384, 1536),  # ConvNeXt-Tiny pw_up
    (14, 112, 672),   # MobileNetV3 expand
    (14, 80, 480),    # EfficientNet-B0 expand (unaligned)
    (14, 1024, 512),  # DenseNet-121 transition
    (20, 1024, 512),  # YOLOv5l SPPF -- N=400 control, swap should NOT win
]


def _ceil(v, m):
    return ((v + m - 1) // m) * m


def _runnable(H, C_out, swap, tm, tn, cols):
    """BD-limit check for the FORCED mapping (autotune's version assumes auto)."""
    N = H * H
    rr, cc = (N, C_out) if swap else (C_out, N)
    return (_ceil(cc, tn * cols) // (tn * cols) <= BD_MAX and
            _ceil(rr, tm * NUM_ROWS) // (tm * NUM_ROWS) <= BD_MAX)


def occupancy(H, C_out, swap, tm, tn, cols):
    """Predicted useful-core fraction: real elements / padded grid footprint."""
    N = H * H
    rr, cc = (N, C_out) if swap else (C_out, N)
    return (rr / _ceil(rr, tm * NUM_ROWS)) * (cc / _ceil(cc, tn * cols))


def bench(H, C_in, C_out, swap, tm, tk, tn):
    """Returns (gf, note). gf is None when the config cannot run / is wrong."""
    if not _runnable(H, C_out, swap, tm, tn, COLS):
        return None, "BD-limit"
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=COLS,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=swap, context=ctx)
        if bool(op.swap) != bool(swap):
            return None, f"swap-not-honored(got {op.swap})"
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
            return None, "NUMERICALLY WRONG"
        for _ in range(WARMUP):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(SAMPLES)]
        med = statistics.median(lat)
        gf = 2.0 * C_out * C_in * (H * H) / (med * 1e-6) / 1e9
        return gf, f"{med:.1f}us"
    except Exception as e:
        return None, f"FAIL {type(e).__name__}: {str(e)[:80]}"
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    shapes = SHAPES
    if len(sys.argv) == 4:
        shapes = [tuple(int(x) for x in sys.argv[1:4])]
    out_path = os.environ.get("OUT", "results.json")
    all_results = {}
    for (H, C_in, C_out) in shapes:
        N = H * H
        key = f"{C_in}/{C_out}/{N}"
        print(f"\n=== {key}  cols={COLS} ===", flush=True)
        row = {}
        for (label, swap, tm, tk, tn) in CONFIGS:
            occ = occupancy(H, C_out, swap, tm, tn, COLS)
            gf, note = bench(H, C_in, C_out, swap, tm, tk, tn)
            row[label] = {"gf": gf, "note": note, "occ": occ,
                          "tile": f"{tm}/{tk}/{tn}", "swap": swap}
            shown = f"{gf:7.0f} GF/s" if gf is not None else f"{'--':>7}     "
            print(f"  {label:>10} {tm}/{tk}/{tn:<3} occ={occ*100:5.1f}%  "
                  f"{shown}  {note}", flush=True)
        all_results[key] = row
        b, p = row["swap_n64"]["gf"], row["plain_n16"]["gf"]
        if b and p:
            print(f"  -> swap_n64 / plain_n16 = {b / p:.2f}x", flush=True)
        with open(out_path, "w") as fh:
            json.dump(all_results, fh, indent=2)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
