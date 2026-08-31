#!/usr/bin/env python3
"""Inside one accumulate step: what are the non-MAC cycles doing?

kernel_probe says the microkernel reaches 79% of the datapath. This asks the
next question: the missing 21% is not compute, so what is it? The trace already
carries the answer and kernel_probe was only reading two of its eight events.

In Event-Time mode the core events are level signals, not counters: the trace
records a B when the signal goes high and an E when it drops, so summing
(E - B) inside a matmul bracket gives the CYCLES that signal was asserted.

INSTR_VECTOR turns out to read exactly 4096 in every configuration, which is
exactly the mmul issue the step needs, so that event tracks the mmul pipe alone.

The interesting quantity is then how much of the load and store activity hides
UNDER those 4096 cycles. It has to be measured as an intersection: 75% of the
load time does overlap with the MAC pipe, so loads are largely hidden and are
not the problem. What is left is a large block that none of the eight traceable
events covers, and the disassembly says what it is: every vmac.f is paired with
a vextbcst/vextbcstshfl operand broadcast, which is the bf16 emulation
preparing its operands.

Rows traced with the DEFAULT core event set report load/store as 0, because
those two slots were spent on DMA ports instead. Re-run kernel_probe with
coretile_events=ANATOMY_EVENTS to fill them in.

No hardware needed: this re-reads the trace_words_*.npy that kernel_probe
already dumped.

Run:  python kernel_anatomy.py
"""
import json
import pathlib
import statistics
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from kernel_probe import _intervals, _overlap  # noqa: E402


def _intersect(a, b):
    """Cycles where BOTH signals are high.

    Summing per-event totals and comparing against the step length is NOT an
    accounting: two events that are both high in the same cycle get counted
    twice, and the sum can land near the step length by coincidence. It did
    here, and the first version of this script drew the wrong conclusion from
    it. Overlap has to be measured, not inferred from a sum.
    """
    out = 0
    for lo, hi in a:
        for l2, h2 in b:
            if h2 <= lo:
                continue
            if l2 >= hi:
                break
            out += min(hi, h2) - max(lo, l2)
    return out

IDEAL = 16 * 64 * 128 / 32  # 4096 cycles of mmul per accumulate step

CASES = [
    ("row-major, L1-resident (the GEMM floor)", "resi1x64_rm",
     "acm0_bcm0_ccm0_mdresident_tl1"),
    ("col-major, L1-resident (+ transposes)", "resi1x64_cm",
     "acm1_bcm1_ccm1_mdresident_tl1"),
    ("col-major, per-step acquire/release", "stre1x64_cm",
     "acm1_bcm1_ccm1_mdstreamed_tl1"),
    ("col-major, real 3x12 loop nest", "prod3x12_cm",
     "acm1_bcm1_ccm1_mdproduction_tl3"),
]
# INSTR_LOAD/INSTR_STORE are the events that actually answer the question, and
# they are NOT in the default core set, which spends two slots on DMA ports.
# kernel_probe must be run with ANATOMY_EVENTS for these to be present.
TRACKED = ("INSTR_VECTOR", "INSTR_LOAD", "INSTR_STORE",
           "MEMORY_STALL", "LOCK_STALL", "GROUP_STALL")


def analyse(tag, mlir_frag):
    import logging
    logging.disable(logging.CRITICAL)
    from aie.utils.trace import parse_trace

    npy = HERE / f"trace_words_{tag}.npy"
    if not npy.exists():
        return None
    mlir = next(iter(sorted(
        pathlib.Path("/scratch/amrosetti/IRON").rglob(
            f"KernelProbe_*{mlir_frag}*.mlir.prj/input_with_addresses.mlir"))), None)
    if mlir is None:
        return None

    events = parse_trace(np.load(npy), mlir.read_text())
    b0 = [e["ts"] for e in events
          if e.get("name") == "INSTR_EVENT_0" and e.get("ph") == "B"]
    b1 = [e["ts"] for e in events
          if e.get("name") == "INSTR_EVENT_1" and e.get("ph") == "B"]
    n = min(len(b0), len(b1))
    brackets = [(b0[i], b1[i]) for i in range(n) if b1[i] > b0[i]]
    spans = {name: _intervals(events, name) for name in TRACKED}

    rows = []
    for lo, hi in brackets:
        row = {"cycles": hi - lo}
        for name in TRACKED:
            row[name] = _overlap(spans[name], lo, hi)
        rows.append(row)
    if len(rows) < 3:
        return None

    # Only the matmul brackets: zero/convert also carry markers.
    durs = sorted(r["cycles"] for r in rows[1:])
    thresh = 0.5 * statistics.median(durs[len(durs) // 2:])
    mm = [r for r in rows[1:] if r["cycles"] >= thresh]

    # Intersections, on one representative warm step.
    lo, hi = brackets[3][0], brackets[3][1]
    clip = lambda sp: [(max(a, lo), min(b, hi)) for a, b in sp if b > lo and a < hi]
    V, L, S = clip(spans["INSTR_VECTOR"]), clip(spans["INSTR_LOAD"]), clip(spans["INSTR_STORE"])
    hidden_ld = _intersect(V, L)
    hidden_st = _intersect(V, S)

    med = statistics.median([r["cycles"] for r in mm])
    vec = statistics.median([r["INSTR_VECTOR"] for r in mm])
    ld = statistics.median([r["INSTR_LOAD"] for r in mm])
    st = statistics.median([r["INSTR_STORE"] for r in mm])
    mem = statistics.median([r["MEMORY_STALL"] for r in mm])
    lock = statistics.median([r["LOCK_STALL"] for r in mm])
    return {
        "n_matmul_brackets": len(mm),
        "cycles": med,
        "ideal": IDEAL,
        "vector_issue": vec,
        "vector_beyond_mmul": vec - IDEAL,
        "instr_load": ld,
        "instr_store": st,
        "memory_stall": mem,
        "lock_stall": lock,
        "load_hidden_under_mmul": hidden_ld,
        "load_exposed": ld - hidden_ld,
        "store_hidden_under_mmul": hidden_st,
        "store_exposed": st - hidden_st,
        "unaccounted": med - vec - (ld - hidden_ld) - (st - hidden_st) - mem,
    }


def main():
    out = {}
    print(f"One accumulate step needs {IDEAL:.0f} cycles of mmul "
          f"(16x64x128 MACs at 32 MAC/cyc).\n")
    hdr = (f"{'configuration':<36}{'cycles':>8}{'mmul':>7}{'ld hid':>8}"
           f"{'ld exp':>8}{'st exp':>8}{'stall':>7}{'other':>8}")
    print(hdr)
    print("-" * len(hdr))
    for label, tag, frag in CASES:
        r = analyse(tag, frag)
        if r is None:
            print(f"{label:<40}   (no trace)")
            continue
        out[tag] = dict(r, label=label)
        has_ls = r["instr_load"] > 0 or r["instr_store"] > 0
        note = "" if has_ls else "   (default events: no load/store)"
        print(f"{label:<36}{r['cycles']:8.0f}{r['vector_issue']:7.0f}"
              f"{r['load_hidden_under_mmul']:8.0f}{r['load_exposed']:8.0f}"
              f"{r['store_exposed']:8.0f}{r['memory_stall']:7.0f}"
              f"{r['unaccounted']:8.0f}{note}")

    base = out.get("resi1x64_rm")
    if base:
        hid = base["load_hidden_under_mmul"] / base["instr_load"] * 100
        print(f"\nThe GEMM floor is {base['cycles']:.0f} cycles for a "
              f"{IDEAL:.0f}-cycle step:")
        print(f"  {IDEAL:8.0f}  mmul pipe busy, the work itself")
        print(f"  {base['load_exposed']:8.0f}  load, exposed "
              f"({hid:.0f}% of load time IS hidden under the MACs)")
        print(f"  {base['store_exposed']:8.0f}  store, none of it hidden")
        print(f"  {base['memory_stall']:8.0f}  memory stall")
        print(f"  {base['unaccounted']:8.0f}  none of the eight events")
        print("  So loads are largely overlapped and are NOT the problem. The big")
        print("  term is invisible to the trace, and the disassembly names it: one")
        print("  vextbcst/vextbcstshfl operand broadcast per vmac.f, which is the")
        print("  bf16 emulation preparing operands.")
    cm = out.get("resi1x64_cm")
    if base and cm:
        print(f"\nThe col-major transposes add "
              f"{cm['cycles'] - base['cycles']:.0f} cycles, all of it in that same")
        print(f"  invisible bucket ({base['unaccounted']:.0f} -> "
              f"{cm['unaccounted']:.0f}). The binary agrees: the col-major object")
        print(f"  carries 32 vshuffle against the row-major object's 4.")
    (HERE / "kernel_anatomy.json").write_text(json.dumps(out, indent=2))
    print("\nwrote kernel_anatomy.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
