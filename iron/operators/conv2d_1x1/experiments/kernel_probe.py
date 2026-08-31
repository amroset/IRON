#!/usr/bin/env python3
"""Measure the microkernel's own cost, tightly, with a hardware trace.

Runs kernel_probe_design on ONE core and reads the event0/event1 markers the
kernel already carries. Each bracket is one `A[m,k] x B[k,n] -> acc` step, so

    MAC/cycle = m*k*n / cycles_per_bracket

against a datapath ceiling of 32 MAC/cycle (256 per logical 4x8x8 mmul, bf16
emulated 8:1). That ratio is the kernel's utilisation, measured rather than
assumed, and it is the term the cycle budget previously had to leave as an
unattributed residual.

Reports both modes (see kernel_probe_design):
    resident  -- L1-resident operands, no DMA in the timed region
    streamed  -- the real acquire/release cadence

Run:  python kernel_probe.py
"""
import json
import pathlib
import statistics
import sys

import numpy as np
import torch
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.common import (
    AIEContext,
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.device_utils import get_kernel_dir

from dataclasses import dataclass, field
from typing import ClassVar, Dict

HERE = pathlib.Path(__file__).parent

# The headline layer's shipping configuration: ConvNeXt-T pw_up 768->3072 @7x7,
# cols->M swap engaged, autotuned tile 16/64/128, so all three operands are
# col-major and the kernel does its in-register transposes.
TILE_M, TILE_K, TILE_N = 16, 64, 128
A_CM, B_CM, C_CM = 1, 1, 1
STEPS = 64          # brackets per run; K/k is 12 in production, 64 gives a
                    # steadier median without overflowing the trace buffer
TRACE_BYTES = 262144
# The default core event set spends two of its eight slots on DMA ports. For the
# anatomy question ("the MAC pipe is idle, doing what?") the useful pair is
# INSTR_LOAD/INSTR_STORE; pass this to KernelProbe(coretile_events=...).
ANATOMY_EVENTS = ("INSTR_EVENT_0", "INSTR_EVENT_1", "INSTR_VECTOR", "INSTR_LOAD",
                  "INSTR_STORE", "MEMORY_STALL", "LOCK_STALL", "GROUP_STALL")
MAC_PER_CYC = 32    # per core: 4x8x8 = 256 MACs, bf16 emulated 8:1
FREQ = 1.8e9


@dataclass
class KernelProbe(MLIROperator):
    """One core, the real kernel object, a tile-shaped workload."""

    m: int
    k: int
    n: int
    steps: int
    a_col_maj: int = 1
    b_col_maj: int = 1
    c_col_maj: int = 1
    mode: str = "resident"
    tiles: int = 1
    trace_size: int = 0
    coretile_events: tuple = ()
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "steps": "st",
        "a_col_maj": "acm",
        "b_col_maj": "bcm",
        "c_col_maj": "ccm",
        "mode": "md",
        "tiles": "tl",
        "trace_size": "tr",
        "coretile_events": "ev",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    @property
    def n_tiles(self):
        """Operand tiles the runtime must feed."""
        return 1 if self.mode == "resident" else self.eff_tiles * self.steps

    @property
    def eff_tiles(self):
        return self.tiles if self.mode == "production" else 1

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                HERE / "kernel_probe_design.py",
                "my_kernel_probe",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "m": self.m, "k": self.k, "n": self.n,
                    "steps": self.steps,
                    "a_col_maj": self.a_col_maj,
                    "b_col_maj": self.b_col_maj,
                    "c_col_maj": self.c_col_maj,
                    "mode": self.mode,
                    "tiles": self.tiles,
                    "trace_size": self.trace_size,
                    "kernel_object": self._kernel_object_name(),
                    "coretile_events": list(self.coretile_events) or None,
                },
            ),
        )

    def _kernel_object_name(self):
        # Distinct object per flag combination, so the probe never silently
        # reuses an object built for a different tile or layout.
        return (f"probe_gemm_{self.m}x{self.k}x{self.n}"
                f"_a{self.a_col_maj}b{self.b_col_maj}c{self.c_col_maj}.o")

    def get_kernel_artifacts(self):
        flags = [f"-DDIM_M={self.m}", f"-DDIM_K={self.k}", f"-DDIM_N={self.n}",
                 "-Dbf16_f32_ONLY"]
        if self.a_col_maj:
            flags.append("-DA_COL_MAJ")
        if self.b_col_maj:
            flags.append("-DB_COL_MAJ")
        if self.c_col_maj:
            flags.append("-DC_COL_MAJ")
        base = self.context.base_dir
        kdir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                extra_flags=flags,
                dependencies=[
                    SourceArtifact(base / "aie_kernels" / kdir / "conv2d_1x1.cc")
                ],
            ),
            KernelObjectArtifact(
                "convert_copy.o",
                [SourceArtifact(base / "aie_kernels" / "generic" / "convert_copy.cc")],
            ),
        ]

    def get_arg_spec(self):
        spec = [
            AIERuntimeArgSpec("in", (self.n_tiles * self.m * self.k,)),
            AIERuntimeArgSpec("in", (self.n_tiles * self.k * self.n,)),
            AIERuntimeArgSpec("out", (self.eff_tiles * self.m * self.n,)),
        ]
        if self.trace_size:
            # Trace lands in its own buffer (ddr_id=3), not in C's tail.
            spec.append(AIERuntimeArgSpec("out", (self.trace_size // 4,),
                                          dtype=np.uint32))
        return spec


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# L1 tile layout
#
# The kernel does not see a plain [m,k] matrix. It sees a sequence of r*s
# sub-blocks, and the col-major flags change BOTH the order of those blocks and
# the storage inside each one -- a col-major operand is block-interleaved, not
# simply transposed. Reading these off the pointer arithmetic in
# conv2d_1x1.cc (matmul_vectorized_2x2_mmul) rather than guessing:
#
#   A row-maj  buf[(z*colA + i)*r*s + rr*s + ss] = A[z*r+rr, i*s+ss]
#   A col-maj  buf[(i*rowA + z)*r*s + ss*r + rr] = A[z*r+rr, i*s+ss]
#   B row-maj  buf[(i*colB + j)*s*t + ss*t + tt] = B[i*s+ss, j*t+tt]
#   B col-maj  buf[(j*colA + i)*s*t + tt*s + ss] = B[i*s+ss, j*t+tt]
#   C row-maj  buf[(z*colB + j)*r*t + rr*t + tt] = C[z*r+rr, j*t+tt]
#   C col-maj  buf[(j*rowA + z)*r*t + tt*r + rr] = C[z*r+rr, j*t+tt]
#
# with rowA=m/r, colA=k/s, colB=n/t. The row-major runs are the control: they
# have the simple layout and a trivial golden, so if they verify, the harness
# is sound and a col-major failure is the packing, not the rig.
R, S, T = 4, 8, 8


def pack_a(A, col_maj):
    m, k = A.shape
    blk = A.reshape(m // R, R, k // S, S).transpose(0, 2, 1, 3)  # rowA,colA,r,s
    return (blk.transpose(1, 0, 3, 2) if col_maj else blk).reshape(-1)


def pack_b(B, col_maj):
    k, n = B.shape
    blk = B.reshape(k // S, S, n // T, T).transpose(0, 2, 1, 3)  # colA,colB,s,t
    return (blk.transpose(1, 0, 3, 2) if col_maj else blk).reshape(-1)


def unpack_c(buf, m, n, col_maj):
    rowA, colB = m // R, n // T
    if col_maj:
        blk = buf.reshape(colB, rowA, T, R).transpose(1, 0, 3, 2)
    else:
        blk = buf.reshape(rowA, colB, R, T)
    return blk.transpose(0, 2, 1, 3).reshape(m, n)


STALL_EVENTS = ("MEMORY_STALL", "LOCK_STALL", "STREAM_STALL")


def _intervals(events, name):
    """[start, end) spans for one event, from its Chrome-format B/E pairs."""
    spans, open_ts = [], None
    for e in events:
        if e.get("name") != name:
            continue
        if e.get("ph") == "B":
            open_ts = e["ts"]
        elif e.get("ph") == "E" and open_ts is not None:
            spans.append((open_ts, e["ts"]))
            open_ts = None
    return spans


def _overlap(spans, lo, hi):
    return sum(max(0, min(hi, b) - max(lo, a)) for a, b in spans)


def parse_brackets(words, mlir_text):
    """Per-step cycle counts, plus the stall time inside each step.

    parse_trace returns Chrome Trace Event Format, so every hardware event
    appears TWICE -- once as "ph":"B" and once as "ph":"E". Pairing an
    INSTR_EVENT_0 "B" with the next INSTR_EVENT_1 "B" gives one
    `A[m,k] x B[k,n]` step; counting both phases would halve every duration.
    """
    from aie.utils.trace import parse_trace

    events = parse_trace(words, mlir_text)
    b0 = [e["ts"] for e in events
          if e.get("name") == "INSTR_EVENT_0" and e.get("ph") == "B"]
    b1 = [e["ts"] for e in events
          if e.get("name") == "INSTR_EVENT_1" and e.get("ph") == "B"]
    n = min(len(b0), len(b1))
    brackets = [(b0[i], b1[i]) for i in range(n) if b1[i] > b0[i]]

    stalls = {name: _intervals(events, name) for name in STALL_EVENTS}
    per_step = []
    for lo, hi in brackets:
        row = {"cycles": hi - lo, "lo": lo, "hi": hi}
        for name, spans in stalls.items():
            row[name] = _overlap(spans, lo, hi)
        per_step.append(row)
    # Gap between consecutive steps: the loop/branch cost outside the markers.
    gaps = [b0[i + 1] - b1[i] for i in range(n - 1) if b0[i + 1] > b1[i]]
    return per_step, gaps, events


def run(mode, col_maj, steps=STEPS, tiles=1, trace=True):
    m, k, n = TILE_M, TILE_K, TILE_N
    cm = 1 if col_maj else 0
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        op = KernelProbe(m=m, k=k, n=n, steps=steps, a_col_maj=cm,
                         b_col_maj=cm, c_col_maj=cm, mode=mode, tiles=tiles,
                         trace_size=TRACE_BYTES if trace else 0, context=ctx)
        print(f"  building {op.name}", flush=True)
        op.compile()
        fn = op.get_callable()

        # Logical operands first, then pack into the layout the kernel expects.
        rng = np.random.default_rng(0)
        n_ops = op.n_tiles
        A_log = [rng.standard_normal((m, k)).astype(np.float32) for _ in range(n_ops)]
        B_log = [rng.standard_normal((k, n)).astype(np.float32) for _ in range(n_ops)]
        a = np.concatenate([pack_a(x, col_maj) for x in A_log])
        b = np.concatenate([pack_b(x, col_maj) for x in B_log])

        args, out_buf, trace_buf = [], None, None
        for i, spec in enumerate(op.get_arg_spec()):
            if spec.direction == "in":
                host = a if i == 0 else b
                args.append(XRTTensor.from_torch(
                    torch.from_numpy(host).to(torch.bfloat16)))
            else:
                buf = XRTTensor(spec.shape, dtype=spec.dtype)
                args.append(buf)
                if out_buf is None:
                    out_buf = buf
                else:
                    trace_buf = buf

        res = fn(*args)
        raw = out_buf.to_torch()

        # --- correctness ------------------------------------------------------
        def bf(x):
            return torch.from_numpy(x).to(torch.bfloat16).to(torch.float32)

        rel = 0.0
        for tl in range(op.eff_tiles):
            flat = raw.flatten()[tl * m * n:(tl + 1) * m * n].to(torch.float32).numpy()
            got = torch.from_numpy(unpack_c(flat, m, n, col_maj))
            if mode == "resident":
                want = (bf(A_log[0]) @ bf(B_log[0])) * steps
            else:
                want = sum(bf(A_log[tl * steps + i]) @ bf(B_log[tl * steps + i])
                           for i in range(steps))
            denom = want.abs().max().clamp(min=1e-6)
            rel = max(rel, ((got - want).abs().max() / denom).item())
        ok = bool(rel < 0.02)
        print(f"  correctness: max rel err {rel:.4f}  ->  {'OK' if ok else 'WRONG'}",
              flush=True)
        if not ok:
            return None

        out = {"mode": mode, "col_maj": col_maj, "steps": steps,
               "tiles": op.eff_tiles, "m": m, "k": k, "n": n,
               "npu_time_us": res.npu_time / 1e3, "correct": ok}

        # --- the trace --------------------------------------------------------
        if trace:
            words = trace_buf.to_torch().flatten().view(torch.int32).numpy()
            words = np.frombuffer(words.tobytes(), dtype=np.uint32)
            # Dump raw before parsing: the parser is fragile and re-running
            # costs a rebuild + a hardware dispatch, so keep the evidence.
            tag0 = f"{mode[:4]}{op.eff_tiles}x{steps}_{'cm' if col_maj else 'rm'}"
            np.save(HERE / f"trace_words_{tag0}.npy", words)
            nz = int((words != 0).sum())
            print(f"  trace: {len(words)} words, {nz} nonzero", flush=True)
            # parse_trace needs the LOWERED module: it reads the concrete
            # NpuWrite32Op register writes that configure the trace units, and
            # those only exist after the trace lowering pipeline. Handing it the
            # design-level .mlir yields "Defined tiles in design are at: []" and
            # a bare sys.exit(1).
            prj = pathlib.Path(ctx.base_dir).rglob(
                f"{op.name}.mlir.prj/input_with_addresses.mlir")
            mlir = next(iter(sorted(prj)), None)
            if mlir is None or nz == 0:
                print("  no parseable trace "
                      f"({'lowered MLIR not found' if mlir is None else 'buffer empty'})",
                      flush=True)
                return out
            per_step, gaps, events = parse_brackets(words, mlir.read_text())
            (HERE / f"kernel_probe_events_{tag0}.json").write_text(
                json.dumps(events[:4000], indent=1, default=str))
            if per_step:
                # Drop the first step: it pays the i-cache cold miss.
                warm = per_step[1:] if len(per_step) > 1 else per_step
                cyc = [w["cycles"] for w in warm]
                med = statistics.median(cyc)
                macs = m * k * n
                ideal = macs / MAC_PER_CYC
                # Not every bracket is a matmul. zero_vectorized and
                # convert_copy_f32_to_bf16 carry event markers of their own, so
                # a production tile traces as 12 long brackets plus two short
                # ones. Split them by duration before doing any arithmetic --
                # counting the short ones as steps is what made the first
                # version report a per-step cost BELOW the bracket median.
                durs = sorted(w["cycles"] for w in warm)
                upper = durs[len(durs) // 2:]
                thresh = 0.5 * statistics.median(upper)
                mm = [w for w in warm if w["cycles"] >= thresh]
                short = [w["cycles"] for w in warm if w["cycles"] < thresh]

                # Wall cost of one accumulate step = the period between the
                # starts of successive matmuls, which carries the gaps and, once
                # per tile, the zero/convert pair.
                if len(mm) > 1:
                    wall = (mm[-1]["lo"] - mm[0]["lo"]) / (len(mm) - 1)
                else:
                    wall = med
                out.update({
                    "n_matmul_brackets": len(mm),
                    "short_brackets": sorted(set(short)),
                    "cycles_per_step_wall": wall,
                    "utilisation_wall": ideal / wall,
                    "n_steps_traced": len(per_step),
                    "cycles_median": med,
                    "cycles_min": min(cyc),
                    "cycles_max": max(cyc),
                    "inter_step_gap_median": statistics.median(gaps) if gaps else 0,
                    "macs_per_step": macs,
                    "mac_per_cycle": macs / med,
                    "utilisation": ideal / med,
                    "ideal_cycles": ideal,
                    "stall_cycles": {
                        name: statistics.median([w[name] for w in warm])
                        for name in STALL_EVENTS
                    },
                })
                st = out["stall_cycles"]
                print(f"  steps {len(per_step)}  median {med:.0f} cyc "
                      f"(ideal {ideal:.0f})  {macs / med:.2f} MAC/cyc "
                      f"= {ideal / med * 100:.1f}% of the datapath", flush=True)
                print(f"    wall: {out['cycles_per_step_wall']:.0f} cyc/step "
                      f"= {out['utilisation_wall'] * 100:.1f}% of the datapath"
                      + (f"   short brackets (zero/convert): "
                         f"{out['short_brackets']}" if out['short_brackets'] else ""),
                      flush=True)
                print(f"    stalls/step: " + "  ".join(
                    f"{k.replace('_STALL', '').lower()} {v:.0f}"
                    for k, v in st.items())
                    + f"   inter-step gap {out['inter_step_gap_median']:.0f}",
                    flush=True)
            else:
                print("  no event0/event1 brackets found", flush=True)
        return out
    except Exception as exc:
        print(f"  FAIL {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        import traceback
        traceback.print_exc(limit=3)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


# Production loop shape for the headline layer, from the design's own arithmetic:
#   row-tiles/core = M/(m*4) = 64/64  = 1
#   col-tiles/core = N/(n*8) = 3072/1024 = 3      -> 3 output tiles per core
#   steps/tile     = K/k     = 768/64 = 12
PROD_TILES, PROD_STEPS = 3, 12


def main():
    results = {}

    def go(mode, col_maj, steps=STEPS, tiles=1):
        tag = f"{mode}_{'colmaj' if col_maj else 'rowmaj'}"
        if mode == "production":
            tag += f"_{tiles}x{steps}"
        print(f"\n=== {tag}: {TILE_M}x{TILE_K}x{TILE_N} ===", flush=True)
        r = run(mode, col_maj, steps=steps, tiles=tiles)
        if r:
            results[tag] = r
        (HERE / "kernel_probe.json").write_text(json.dumps(results, indent=2))

    # Row-major first: it is the control. Its golden is a plain A@B, so if it
    # verifies, the harness is right and any col-major failure is the packing.
    for col_maj in (False, True):
        go("resident", col_maj)
        go("streamed", col_maj)
    # And the real loop nest, at the real tile/step counts.
    go("production", True, steps=PROD_STEPS, tiles=PROD_TILES)

    def cyc(tag):
        return results.get(tag, {}).get("cycles_median")

    def per_step(tag):
        """Wall cycles per accumulate step, INCLUDING the gaps between steps.

        cycles_median counts only what is inside event0..event1. The per-tile
        zero()/convert() and the loop branches live in the gaps, so the honest
        cost of a loop shape is span / number-of-steps.
        """
        r = results.get(tag)
        return r.get("cycles_per_step_wall") if r else None

    ideal = TILE_M * TILE_K * TILE_N / MAC_PER_CYC
    split = {"ideal_cycles": ideal}
    for label, tag in (("gemm_floor", "resident_rowmaj"),
                       ("colmaj", "resident_colmaj"),
                       ("streamed", "streamed_colmaj"),
                       ("production", f"production_colmaj_{PROD_TILES}x{PROD_STEPS}")):
        c = cyc(tag)
        if c:
            split[f"{label}_cycles"] = c
            split[f"{label}_utilisation"] = ideal / c
        ps = per_step(tag)
        if ps:
            split[f"{label}_cycles_per_step_wall"] = ps
            split[f"{label}_utilisation_wall"] = ideal / ps
    split["note"] = (
        "One core. An event0->event1 bracket is one A[m,k]xB[k,n] accumulate "
        "step; ideal = m*k*n/32. `cycles_median` is inside the bracket only. "
        "`..._wall` is the period between successive matmul starts, so it also "
        "carries the loop branches and, in production mode, the per-tile "
        "zero()/convert() pair -- that is the number comparable to the array's "
        "steady state."
    )
    results["split"] = split

    print(f"\n{'':<34}{'in-bracket':>12}{'wall/step':>13}")
    print(f"{'ideal (32 MAC/cyc)':<34}{ideal:12.0f}{'':>13}")
    for label, tag in (("GEMM microkernel floor", "resident_rowmaj"),
                       ("+ col-major transposes", "resident_colmaj"),
                       ("+ per-step acquire/release", "streamed_colmaj"),
                       (f"+ real loop nest ({PROD_TILES} tiles x {PROD_STEPS})",
                        f"production_colmaj_{PROD_TILES}x{PROD_STEPS}")):
        c, ps = cyc(tag), per_step(tag)
        if c:
            g = f"{ps:13.0f}" if ps else " " * 13
            print(f"{label:<34}{c:12.0f}{g}")
    (HERE / "kernel_probe.json").write_text(json.dumps(results, indent=2))
    print("\nwrote kernel_probe.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
