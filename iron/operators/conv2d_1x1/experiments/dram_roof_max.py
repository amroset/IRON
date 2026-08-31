#!/usr/bin/env python3
"""A defensible upper bound on the DRAM roof.

dram_roof.py put the roof at 67.1 GB/s, but that value sat at the CORNER of its
sweep: most cores, most "channels", largest transfer, all at once. A maximum on
the boundary is not bracketed, so it is a lower bound on the roof, not the roof.

Two things were wrong with the old sweep, and the second one matters:

  1. It stopped at 16 cores. The array has 32.
  2. `num_channels` is not a channel count. mem_copy passes it straight to
     SequentialPlacer(cores_per_col), so it is cores PER COLUMN. The old (8, 2)
     point therefore ran 8 cores over FOUR columns, using half the shim DMAs,
     which is why it came out no faster than (8, 1) on eight. Nothing in the old
     sweep ever used more than 8 columns AND more than 2 cores per column.

So this sweep holds all 8 columns and walks cores per column 1 -> 4, i.e. 8, 16,
24 and 32 cores, at transfer sizes big enough that the per-call floor is under
1% of the measurement. Each point is repeated across independent sessions,
because the old data had unexplained ~25% dropouts at single sizes and one of
them ended up inside the fitted asymptote.

`bypass` runs the same transfers shim -> memtile -> shim with no core in the
path at all. If the core were the limit, bypass would be faster; if it is not,
bypass and the core path agree and the number is a DMA measurement.

Rates are reported raw AND with the dispatch floor removed. null_dispatch.py
measured that floor directly (~52 us for a dispatch that does nothing), so it can
be subtracted rather than fitted -- which is what the old 70.7 GB/s "asymptote"
was trying to do, with a fit that returned an unphysical 217 us intercept.

Run:  python dram_roof_max.py
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
BYTES_PER_ELEM = 2
WARMUP, SAMPLES, SESSIONS = 5, 30, 3

# Dispatch floor with nothing in the design, measured by null_dispatch.py.
FLOOR_SEC = 52e-6

# (num_cores, cores_per_col). All of these keep every one of the 8 columns busy.
CONFIGS = [(8, 1), (16, 2), (24, 3), (32, 4)]
PER_CORE = [2097152, 4194304, 8388608]


def bench(num_cores, cores_per_col, tile_size, bypass):
    size = num_cores * tile_size
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        golden = generate_golden_reference(input_length=size)
        op = MemCopy(size=size, num_cores=num_cores, num_channels=cores_per_col,
                     bypass=bypass, tile_size=tile_size, context=ctx)
        op.compile()
        fn = op.get_callable()

        args, out_buf, it = [], None, iter([golden["input"]])
        for spec in op.get_arg_spec():
            if spec.direction == "in":
                args.append(XRTTensor.from_torch(next(it)))
            else:
                out_buf = XRTTensor(spec.shape, dtype=spec.dtype)
                args.append(out_buf)

        fn(*args)
        if verify_buffer(out_buf.to_torch(), "output", golden["output"],
                         rel_tol=0.01, abs_tol=1e-6):
            print("        WRONG RESULT", flush=True)
            return None

        for _ in range(WARMUP):
            fn(*args)
        sec = statistics.median(
            [fn(*args).npu_time / 1e3 for _ in range(SAMPLES)]) * 1e-6

        # Each element crosses the DRAM boundary twice: read, then write back.
        gbytes = 2 * size * BYTES_PER_ELEM / 1e9
        return {"cores": num_cores, "cores_per_col": cores_per_col,
                "columns": num_cores // cores_per_col, "tile_size": tile_size,
                "bypass": bypass, "MB": gbytes * 1e3, "sec": sec,
                "gbps": gbytes / sec,
                "gbps_floor_removed": gbytes / (sec - FLOOR_SEC)}
    except Exception as exc:
        print(f"        FAIL {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        traceback.print_exc(file=sys.stdout)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def point(num_cores, cores_per_col, tile_size, bypass):
    """Repeat across independent sessions and keep the best and the median."""
    runs = [r for r in (bench(num_cores, cores_per_col, tile_size, bypass)
                        for _ in range(SESSIONS)) if r]
    if not runs:
        return None
    rates = [r["gbps"] for r in runs]
    out = dict(runs[0])
    out.update({"gbps": statistics.median(rates), "gbps_best": max(rates),
                "gbps_worst": min(rates), "n_sessions": len(rates),
                "gbps_floor_removed": statistics.median(
                    [r["gbps_floor_removed"] for r in runs])})
    tag = "bypass" if bypass else "core  "
    print(f"  {out['cores']:2d} cores /{out['columns']:2d} cols  {tag}"
          f"  {out['MB']:7.1f} MB   {out['gbps']:6.2f} GB/s"
          f"   (best {out['gbps_best']:6.2f}, worst {out['gbps_worst']:6.2f})"
          f"   floor-removed {out['gbps_floor_removed']:6.2f}", flush=True)
    return out


def main():
    rows = []
    print("=== core path: all 8 columns, 8 -> 32 cores ===", flush=True)
    for cores, cpc in CONFIGS:
        for ts in PER_CORE:
            r = point(cores, cpc, ts, bypass=False)
            if r:
                rows.append(r)
                (HERE / "dram_roof_max.json").write_text(json.dumps(rows, indent=2))

    print("\n=== bypass: no core in the path ===", flush=True)
    for cores, cpc in ((8, 1), (32, 4)):
        for ts in PER_CORE[1:]:
            r = point(cores, cpc, ts, bypass=True)
            if r:
                rows.append(r)
                (HERE / "dram_roof_max.json").write_text(json.dumps(rows, indent=2))

    if not rows:
        print("nothing measured")
        return 1

    best = max(rows, key=lambda r: r["gbps_best"])
    bmed = max(rows, key=lambda r: r["gbps"])
    print("\n=== upper bound ===")
    print(f"  best single session: {best['gbps_best']:.2f} GB/s "
          f"({best['cores']} cores, {best['columns']} cols, {best['MB']:.0f} MB, "
          f"{'bypass' if best['bypass'] else 'core path'})")
    print(f"  best median-of-{bmed['n_sessions']}: {bmed['gbps']:.2f} GB/s "
          f"({bmed['cores']} cores, {bmed['columns']} cols, {bmed['MB']:.0f} MB, "
          f"{'bypass' if bmed['bypass'] else 'core path'})")
    print(f"  same point with the {FLOOR_SEC * 1e6:.0f} us dispatch floor "
          f"removed: {bmed['gbps_floor_removed']:.2f} GB/s")

    core = [r for r in rows if not r["bypass"]]
    byp = [r for r in rows if r["bypass"]]
    if core and byp:
        c, b = max(r["gbps"] for r in core), max(r["gbps"] for r in byp)
        print(f"\n  core path {c:.2f} vs bypass {b:.2f} GB/s: "
              f"{'the core is NOT the limit' if b <= c * 1.03 else 'the CORE was limiting the old number'}")

    print(f"\n  dram_roof.py published 67.07 GB/s from a corner of its sweep.")
    print("\nwrote dram_roof_max.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
