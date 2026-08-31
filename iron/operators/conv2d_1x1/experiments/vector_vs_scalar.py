#!/usr/bin/env python3
"""Vectorized aie::mmul vs the scalar oracle, per layer.

The README quotes an aggregate ("average 125x, 33-254x") whose per-shape numbers
were never saved. This re-measures them so the claim has data behind it.

Both configurations use the PLAIN mapping (swap_mn=False) so the comparison
isolates vectorization alone -- matching how the README's figure was framed.
Scalar needs prio_accuracy=True to stay inside tolerance (README gotcha).

Run:  CONV_COLS=8 OUT=vec_vs_scalar.json python vector_vs_scalar.py
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

COLS = int(os.environ.get("CONV_COLS", "8"))
TM, TK, TN = 16, 64, 64
BD_MAX = 64
NUM_ROWS = 4

# Scalar is ~100x slower, so it gets fewer samples; the median is still stable.
SAMPLES = {False: (10, 50), True: (10, 50)}  # use_scalar -> (warmup, samples)

# The full 22-shape real-model suite (matches autotune.DEFAULT_SHAPES), with the
# model/role each shape comes from -- used verbatim as the plot's bar labels.
SHAPES = [
    (56,   64,  256, "ResNet-50",    "expand"),
    (14,   80,  480, "EfficientNet", "expand"),
    ( 7, 1024, 2048, "ResNeXt-50",   "expand"),
    (14,  512,  128, "DenseNet-121", "dense1x1"),
    (56,   64,   64, "ResNet-50",    "reduce"),
    ( 7,  512, 2048, "ResNet-50",    "deep exp"),
    ( 7, 2048,  512, "ResNet-50",    "deep red"),
    (28,  128,  512, "ResNet-50",    "mid exp"),
    (28,  512,  512, "ResNeXt-50",   "reduce"),
    (56,   96,  384, "ConvNeXt-T",   "pw_up"),
    ( 7,  768, 3072, "ConvNeXt-T",   "pw_up"),
    ( 7, 3072,  768, "ConvNeXt-T",   "pw_down"),
    (14,  384, 1536, "ConvNeXt-T",   "pw_up"),
    ( 7, 1152,  192, "EfficientNet", "project"),
    (28,   40,  240, "EfficientNet", "expand"),
    (14,   96,  576, "MobileNetV2",  "expand"),
    (28,   32,  192, "MobileNetV2",  "expand"),
    (14,  112,  672, "MobileNetV3",  "expand"),
    (28,  256,  128, "DenseNet-121", "dense1x1"),
    (14, 1024,  512, "DenseNet-121", "transition"),
    (20, 1024,  512, "YOLOv5l",      "SPPF"),
    (80,  128,   64, "YOLOv5l",      "P3"),
]


def _ceil(v, m):
    return ((v + m - 1) // m) * m


def _runnable(H, C_out):
    """BD-limit check for the plain mapping."""
    N = H * H
    return (_ceil(N, TN * COLS) // (TN * COLS) <= BD_MAX and
            _ceil(C_out, TM * NUM_ROWS) // (TM * NUM_ROWS) <= BD_MAX)


def bench(H, C_in, C_out, use_scalar):
    if not _runnable(H, C_out):
        return None, "BD-limit"
    warmup, samples = SAMPLES[use_scalar]
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=TM, tile_k=TK, tile_n=TN, num_aie_columns=COLS,
                       use_scalar=use_scalar, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=False, context=ctx)
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
        for _ in range(warmup):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(samples)]
        med = statistics.median(lat)
        return 2.0 * C_out * C_in * (H * H) / (med * 1e-6) / 1e9, f"{med:.0f}us"
    except Exception as e:
        return None, f"FAIL {type(e).__name__}: {str(e)[:70]}"
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    out_path = pathlib.Path(__file__).parent / os.environ.get("OUT", "vec_vs_scalar.json")
    results = {}
    for (H, C_in, C_out, model, role) in SHAPES:
        key = f"{C_in}/{C_out}/{H * H}"
        print(f"\n=== {key}  {model} {role} ===", flush=True)
        row = {"model": model, "role": role, "H": H, "C_in": C_in, "C_out": C_out}
        for use_scalar, tag in ((True, "scalar"), (False, "vector")):
            gf, note = bench(H, C_in, C_out, use_scalar)
            row[tag] = gf
            row[tag + "_note"] = note
            shown = f"{gf:8.2f} GF/s" if gf is not None else f"{'--':>8}     "
            print(f"  {tag:>7}  {shown}  {note}", flush=True)
        if row.get("scalar") and row.get("vector"):
            row["speedup"] = row["vector"] / row["scalar"]
            print(f"  -> {row['speedup']:.0f}x", flush=True)
        results[key] = row
        out_path.write_text(json.dumps(results, indent=2))
    ups = [r["speedup"] for r in results.values() if "speedup" in r]
    if ups:
        print(f"\n{len(ups)} shapes · mean {sum(ups)/len(ups):.0f}x · "
              f"range {min(ups):.0f}-{max(ups):.0f}x", flush=True)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
