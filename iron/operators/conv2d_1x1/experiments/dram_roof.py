#!/usr/bin/env python3
"""Measure the DRAM roof for the roofline plot.

A roofline is only as honest as its roofs. The compute roof we can derive
(2 x 32 MAC/cyc x 32 cores x 1.8 GHz = 3.69 TF/s), but the memory roof must be
MEASURED -- a spec LPDDR5x number describes the SoC's memory controller, not
what the NPU's shim DMAs can actually pull through 8 columns.

So we use `mem_copy`, which exists precisely for this ("designed to use every
column's shimDMA in-out pairs to fully saturate DDR bandwidth"). It streams
DRAM -> core -> DRAM, so each element crosses the boundary twice, and the
bytes/s we report is read+write traffic -- the same convention traffic_model.py
uses to count the conv's own bytes. Roof and points are therefore measured on
the same ruler.

We sweep cores x channels x transfer size and take the max sustained rate. The
sweep matters: small transfers are latency-bound and would understate the roof,
and the core/channel split changes how many shim DMAs run concurrently.
"""
import json
import pathlib
import statistics
import sys
import traceback

import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.common import AIEContext
from iron.common.test_utils import verify_buffer
from iron.operators.mem_copy.op import MemCopy
from iron.operators.mem_copy.reference import generate_golden_reference

HERE = pathlib.Path(__file__).parent
WARMUP, SAMPLES = 10, 50
BYTES_PER_ELEM = 2  # bf16

# (num_cores, num_channels) -- 8 columns x up to 2 shim channels each.
CORE_CHAN = [(8, 1), (8, 2), (16, 2)]
# Elements per core. Large enough that the launch floor stops dominating;
# 8192 is mem_copy's internal line size, so these are whole numbers of lines.
# The top sizes exist to pin the asymptote: at 0.3 MB the dispatch overhead is
# most of the measurement, and reading a "roof" off that point would understate
# the hardware by an order of magnitude.
PER_CORE = [8192, 65536, 262144, 1048576, 2097152, 4194304]


def bench(num_cores, num_channels, tile_size):
    size = num_cores * tile_size
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        golden = generate_golden_reference(input_length=size)
        op = MemCopy(size=size, num_cores=num_cores, num_channels=num_channels,
                     bypass=False, tile_size=tile_size, context=ctx)
        op.compile()
        fn = op.get_callable()

        args, out_buf = [], None
        it = iter([golden["input"]])
        for spec in op.get_arg_spec():
            if spec.direction == "in":
                args.append(XRTTensor.from_torch(next(it)))
            else:
                out_buf = XRTTensor(spec.shape, dtype=spec.dtype)
                args.append(out_buf)

        fn(*args)
        if verify_buffer(out_buf.to_torch(), "output", golden["output"],
                         rel_tol=0.01, abs_tol=1e-6):
            print("      WRONG RESULT", flush=True)
            return None

        for _ in range(WARMUP):
            fn(*args)
        sec = statistics.median(
            [fn(*args).npu_time / 1e3 for _ in range(SAMPLES)]) * 1e-6

        # Each element is read from DRAM and written back: 2 crossings.
        gbps = 2 * size * BYTES_PER_ELEM / sec / 1e9
        return {"cores": num_cores, "channels": num_channels,
                "tile_size": tile_size, "size": size, "sec": sec,
                "MB": 2 * size * BYTES_PER_ELEM / 1e6, "gbps": gbps}
    except Exception as exc:
        print(f"      FAIL {type(exc).__name__}: {str(exc)[:90]}", flush=True)
        traceback.print_exc(limit=2)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    runs = []
    for cores, chans in CORE_CHAN:
        for per_core in PER_CORE:
            print(f"\n=== {cores} cores, {chans} chan, {per_core} elem/core "
                  f"({2 * cores * per_core * BYTES_PER_ELEM / 1e6:.1f} MB moved) ===",
                  flush=True)
            r = bench(cores, chans, per_core)
            if r is None:
                continue
            runs.append(r)
            print(f"  {r['gbps']:7.1f} GB/s   ({r['sec'] * 1e6:.1f} us)", flush=True)
            (HERE / "dram_roof.json").write_text(json.dumps(
                {"runs": runs, "roof_gbps": max(x["gbps"] for x in runs)}, indent=2))

    if not runs:
        print("\nno successful runs", flush=True)
        return 1

    best = max(runs, key=lambda r: r["gbps"])

    # The single fastest point still carries a share of the launch floor, so it
    # UNDERSTATES the streaming rate. Recover the asymptote the same way the
    # cycle-budget slide does: fit latency = a + b*bytes over the large
    # transfers and read the roof off the slope. `a` is the launch floor, 1/b
    # is the bandwidth once the pipe is full.
    fits = {}
    for cores, chans in CORE_CHAN:
        pts = [r for r in runs if r["cores"] == cores and r["channels"] == chans
               and r["size"] * 2 * BYTES_PER_ELEM >= 8e6]
        if len(pts) < 2:
            continue
        xs = [p["size"] * 2 * BYTES_PER_ELEM for p in pts]
        ys = [p["sec"] for p in pts]
        nn = len(xs)
        mx, my = sum(xs) / nn, sum(ys) / nn
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        a = my - b * mx
        ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - my) ** 2 for y in ys)
        fits[f"{cores}c{chans}ch"] = {
            "cores": cores, "channels": chans, "n_points": nn,
            "launch_floor_us": a * 1e6, "asymptotic_gbps": 1e-9 / b,
            "r2": 1 - ss_res / ss_tot if ss_tot else 1.0,
        }
        print(f"  fit {cores:2d}c/{chans}ch: floor {a * 1e6:6.1f} us, "
              f"asymptote {1e-9 / b:6.1f} GB/s, R2 {1 - ss_res / ss_tot:.4f}", flush=True)

    roof = max((f["asymptotic_gbps"] for f in fits.values()), default=best["gbps"])
    out = {
        "runs": runs,
        "fits": fits,
        "best_single_point_gbps": best["gbps"],
        "best_single_point_config": {k: best[k] for k in ("cores", "channels", "tile_size", "MB")},
        "roof_gbps": roof,
        "note": ("read+write DRAM traffic through the shim DMAs, same byte "
                 "convention as traffic_model.py. Roof is the asymptotic slope "
                 "of latency = a + b*bytes over transfers >= 8 MB, i.e. the rate "
                 "with the launch floor removed; the best single measured point "
                 "is reported alongside as the conservative alternative."),
    }
    (HERE / "dram_roof.json").write_text(json.dumps(out, indent=2))
    print(f"\nbest single point : {best['gbps']:.1f} GB/s "
          f"({best['cores']} cores, {best['channels']} chan, {best['MB']:.0f} MB)")
    print(f"DRAM roof (fitted): {roof:.1f} GB/s", flush=True)
    print("wrote dram_roof.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
