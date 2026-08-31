#!/usr/bin/env python3
"""Roofline measurement: where every real layer lands in (intensity, throughput).

For each of the 22 real-model shapes we measure the COMPLETE operator in its
production configuration (autotuned tile, swap decided by the heuristic) and,
for the layers the heuristic swaps, the plain baseline as well -- so the plot
can draw the swap as an actual displacement rather than two unrelated dots.

The two coordinates:

  y = useful GFLOP/s        2*C_in*C_out*pixels / measured latency.  Identical
                            to every other throughput number in the deck:
                            padding is overhead, never credit.

  x = useful FLOP / DRAM byte
                            Useful flops over the bytes traffic_model.py says
                            actually cross the DRAM boundary for the padded,
                            tiled problem -- including operand re-fetch, which
                            is the whole story here. NOT the tensor footprint:
                            the footprint would hide the very effect the swap
                            fixes, since the plain mapping re-reads B once per
                            row-block pass (48x for the headline layer).

Both numerator conventions are "useful", so a point's position is directly
comparable to the bars on the throughput slides.

Anchors measured alongside the real layers:
  * a large square-ish GEMM (2048/2048/9216), where the array is genuinely full
    -- this is the design's own ceiling, and it should sit near the roofs.
  * the headline layer at 1/2/4/8 columns, to show the scaling trajectory.

Run:  CONV_COLS=8 python roofline.py
"""
import json
import os
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
sys.path.insert(0, str(HERE))
from traffic_model import layer_point  # noqa: E402
from vector_vs_scalar import SHAPES  # noqa: E402

COLS = int(os.environ.get("CONV_COLS", "8"))
WARMUP, SAMPLES = 10, 50
# Median of 50 is robust to jitter WITHIN a session, but not to a bad session:
# re-measuring one config across sessions showed 984-1062 GF/s typical spread
# with one 884 GF/s outlier, while the large GEMM held 2220-2231. Short kernels
# sit close to the launch floor, so host-side contention moves them. Hence
# REPEATS whole sessions (compile, load, warm, time) and take the median of
# those medians -- an outlier session can no longer set a point's position.
REPEATS = int(os.environ.get("CONV_REPEATS", "3"))
DEFAULT_TILE = (16, 64, 64)
_TILE_CACHE = HERE.parent / "tile_cache.json"


def optimal_tile(C_in, C_out, pixels, cols):
    """Same tuned-tile lookup test.py uses, so we measure what actually ships."""
    try:
        cache = json.loads(_TILE_CACHE.read_text())
    except (FileNotFoundError, ValueError):
        return DEFAULT_TILE
    return tuple(cache.get(f"{C_in}_{C_out}_{pixels}_{cols}", DEFAULT_TILE))


def bench(H, C_in, C_out, swap_mn, tile, cols=None):
    """Build, verify against torch, time. Returns a roofline point or None."""
    cols = COLS if cols is None else cols
    tm, tk, tn = tile
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        g = generate_golden_reference(batch=1, H=H, W=H, C_in=C_in, C_out=C_out)
        op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                       tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=cols,
                       use_scalar=False, prio_accuracy=True,
                       emulate_bf16_mmul_with_bfp16=False,
                       swap_mn=swap_mn, context=ctx)
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
            print("      WRONG RESULT", flush=True)
            return None
        for _ in range(WARMUP):
            fn(*args)
        sec = statistics.median(
            [fn(*args).npu_time / 1e3 for _ in range(SAMPLES)]) * 1e-6

        pt = layer_point(op, C_in, C_out, H * H, sec)
        pt.update({
            "sec": sec, "swap": bool(op.swap), "tile": f"{tm}/{tk}/{tn}",
            "cols": cols, "cores": 4 * cols,
            "M": op.M, "K": op.K, "N": op.N,
            "pad_overhead": op.pad_overhead,
        })
        return pt
    except Exception as exc:
        print(f"      FAIL {type(exc).__name__}: {str(exc)[:80]}", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def bench_rep(*args, repeats=None, **kw):
    """`bench` over several independent sessions; keep the median-throughput one.

    Returns that session's point with the observed spread attached, so the
    figure can state its own measurement error instead of implying the points
    are exact.
    """
    repeats = REPEATS if repeats is None else repeats
    runs = [r for r in (bench(*args, **kw) for _ in range(repeats)) if r]
    if not runs:
        return None
    runs.sort(key=lambda r: r["gf"])
    pt = dict(runs[len(runs) // 2])
    pt["gf_runs"] = [r["gf"] for r in runs]
    pt["gf_spread"] = (runs[-1]["gf"] / runs[0]["gf"]) if runs[0]["gf"] else 1.0
    return pt


def main():
    out = {"cols": COLS, "repeats": REPEATS, "layers": {}, "anchors": {}}

    # ---- the 22 real-model shapes ------------------------------------------
    for H, C_in, C_out, model, role in SHAPES:
        pixels = H * H
        key = f"{C_in}/{C_out}/{pixels}"
        tile = optimal_tile(C_in, C_out, pixels, COLS)
        print(f"\n=== {key}  {model} {role}  tile {tile[0]}/{tile[1]}/{tile[2]} ===",
              flush=True)
        row = {"model": model, "role": role, "H": H,
               "C_in": C_in, "C_out": C_out, "pixels": pixels}

        prod = bench_rep(H, C_in, C_out, None, tile)  # heuristic decides = production
        if prod is None:
            continue
        row["prod"] = prod
        print(f"  prod   {prod['gf']:7.0f} GF/s  AI {prod['ai']:7.1f} F/B  "
              f"{prod['gbps']:6.1f} GB/s  swap={prod['swap']}  "
              f"pad={prod['pad_overhead']:.2f}x  "
              f"A x{prod['A_refetch']} B x{prod['B_refetch']}  "
              f"spread {prod['gf_spread']:.2f}x", flush=True)

        # Only meaningful where production actually swaps: otherwise "plain"
        # IS production and a second point would just duplicate it.
        #
        # The baseline runs at the DEFAULT tile, not at production's tuned tile.
        # That tile was tuned FOR the swapped mapping, and forcing it onto the
        # plain mapping is a strawman: for 768/3072/49 the tuned n=128 pads N to
        # 1024 instead of 512, doubling pad overhead to 20.9x and making plain
        # look 2x worse than it is. 16/64/64 is the design default and the
        # baseline the swap slides already quote ("both n=64").
        if prod["swap"]:
            plain = bench_rep(H, C_in, C_out, False, DEFAULT_TILE)
            if plain is not None:
                row["plain"] = plain
                row["swap_gain"] = prod["gf"] / plain["gf"]
                print(f"  plain  {plain['gf']:7.0f} GF/s  AI {plain['ai']:7.1f} F/B  "
                      f"{plain['gbps']:6.1f} GB/s  "
                      f"pad={plain['pad_overhead']:.2f}x  "
                      f"A x{plain['A_refetch']} B x{plain['B_refetch']}", flush=True)

        out["layers"][key] = row
        (HERE / "roofline.json").write_text(json.dumps(out, indent=2))

    # ---- anchor: a large GEMM, where the array is genuinely full ------------
    # 9216 pixels = 96x96. Not a real conv layer -- it is the design's own
    # ceiling, included so the plot shows what "full array" looks like and how
    # far the small real layers sit below it.
    print("\n=== anchor: 2048/2048/9216 (96x96), full array ===", flush=True)
    big = bench_rep(96, 2048, 2048, None, DEFAULT_TILE)
    if big is not None:
        out["anchors"]["big_gemm"] = big
        print(f"  {big['gf']:7.0f} GF/s  AI {big['ai']:7.1f} F/B  "
              f"{big['gbps']:6.1f} GB/s  swap={big['swap']}", flush=True)

    # ---- anchor: headline layer vs column count ----------------------------
    # The swap needs >= 4 columns, so 1 and 2 columns are plain by construction.
    scaling = {}
    for cols in (1, 2, 4, 8):
        print(f"\n=== anchor: 768/3072/49 at {cols} column(s) ===", flush=True)
        pt = bench_rep(7, 768, 3072, None, (16, 64, 128), cols=cols)
        if pt is not None:
            scaling[str(cols)] = pt
            print(f"  {pt['gf']:7.0f} GF/s  AI {pt['ai']:7.1f} F/B  "
                  f"{pt['gbps']:6.1f} GB/s  swap={pt['swap']}", flush=True)
    out["anchors"]["col_scaling"] = scaling

    (HERE / "roofline.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote roofline.json  ({len(out['layers'])} layers)", flush=True)

    peak_gbps = max(
        (p["gbps"] for r in out["layers"].values()
         for p in (r.get("prod"), r.get("plain")) if p),
        default=0.0)
    print(f"highest DRAM rate any layer reached: {peak_gbps:.1f} GB/s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
