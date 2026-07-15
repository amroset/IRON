#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Empirical tile-size autotuner for the pointwise (1x1) conv operator.
#
# An ANALYTICAL tile-shape model was unreliable on this hardware (it picked tiles
# that regressed real layers, and some tiles regress catastrophically in ways no
# static model predicts). So we tune EMPIRICALLY: for each (shape, cols) sweep a
# small candidate set on-device, verify correctness, measure median throughput,
# and cache the fastest tile in tile_cache.json. test.py's get_optimal_tile()
# reads that cache (set CONV_TUNE=1 to use it); the selection lives in the host,
# not in op.py -- mirroring softmax's get_optimal_* convention.
#
# Run:  CONV_COLS=8 python autotune.py            # tune the default shape set
#       CONV_COLS=8 python autotune.py 56 64 256  # tune one shape (H C_in C_out)

import json
import os
import pathlib
import statistics
import sys

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from iron.operators.conv2d_1x1_opt.test import TILE_CANDIDATES, _TILE_CACHE
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import aie.utils as aie_utils

COLS = int(os.environ.get("CONV_COLS", "8"))
WARMUP = int(os.environ.get("CONV_WARMUP", "8"))
SAMPLES = int(os.environ.get("CONV_SAMPLES", "20"))
BD_MAX = 64

# The full real-model suite (H, C_in, C_out) -- matches test.py's shape set, so
# the cache (at cols=8) covers every shape the suite runs.
DEFAULT_SHAPES = [
    # default_shapes
    (56, 64, 256), (14, 80, 480), (7, 1024, 2048), (14, 512, 128),
    # extensive_shapes
    (56, 64, 64), (7, 512, 2048), (7, 2048, 512), (28, 128, 512),
    (28, 512, 512), (56, 96, 384), (7, 768, 3072), (7, 3072, 768),
    (14, 384, 1536), (7, 1152, 192), (28, 40, 240), (14, 96, 576),
    (28, 32, 192), (14, 112, 672), (28, 256, 128), (14, 1024, 512),
    (20, 1024, 512), (80, 128, 64),
]


def _ceil(v, m):
    return ((v + m - 1) // m) * m


def _runnable(H, C_out, tn, tm, cols):
    N = H * H
    swap = C_out > N and cols >= 4
    rr, cc = (N, C_out) if swap else (C_out, N)
    return (_ceil(cc, tn * cols) // (tn * cols) <= BD_MAX and
            _ceil(rr, tm * 4) // (tm * 4) <= BD_MAX)


def bench_tile(H, C_in, C_out, tm, tk, tn):
    """Build+verify+time one tile. Returns median GF/s, or None if it can't run."""
    if not _runnable(H, C_out, tn, tm, COLS):
        return None
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=COLS,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False, context=ctx)
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
            return None  # numerically wrong -> reject this tile
        for _ in range(WARMUP):
            f(*args)
        lat = [f(*args).npu_time / 1e3 for _ in range(SAMPLES)]
        return 2.0 * C_out * C_in * (H * H) / (statistics.median(lat) * 1e-6) / 1e9
    except Exception:
        return None  # compile/L1/runtime failure -> skip
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def tune(shapes):
    cache = {}
    if _TILE_CACHE.exists():
        try:
            cache = json.loads(_TILE_CACHE.read_text())
        except ValueError:
            cache = {}
    for (H, C_in, C_out) in shapes:
        N = H * H
        print(f"\n=== {C_in}/{C_out}/{N}  cols={COLS} ===")
        results = []
        for (tm, tk, tn) in TILE_CANDIDATES:
            gf = bench_tile(H, C_in, C_out, tm, tk, tn)
            tag = f"{tm}/{tk}/{tn}"
            if gf is None:
                print(f"  {tag:>10}   skip")
                continue
            results.append(((tm, tk, tn), gf))
            print(f"  {tag:>10}   {gf:7.0f} GF/s")
        if not results:
            continue
        best_tile, best_gf = max(results, key=lambda r: r[1])
        base = next((g for t, g in results if t == (16, 64, 64)), None)
        gain = f"  ({best_gf / base:.2f}x vs 16/64/64)" if base else ""
        print(f"  -> {best_tile[0]}/{best_tile[1]}/{best_tile[2]} @ {best_gf:.0f} GF/s{gain}")
        cache[f"{C_in}_{C_out}_{N}_{COLS}"] = list(best_tile)
    _TILE_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {_TILE_CACHE}  ({len(cache)} entries)")


if __name__ == "__main__":
    if len(sys.argv) == 4:
        H, C_in, C_out = (int(x) for x in sys.argv[1:4])
        tune([(H, C_in, C_out)])
    else:
        tune(DEFAULT_SHAPES)
