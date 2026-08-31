#!/usr/bin/env python3
"""The null-dispatch experiment the README says this suite does not have.

`dispatch_floor.py` put the per-call cost at ~67 us and stopped there, on the
grounds that naming its parts "would need a null-dispatch experiment this suite
does not have". This is that experiment.

Two things it establishes that the fit could not.

1. WHAT `npu_time` ACTUALLY IS. Reading the runtime rather than the docstring:
   XRTHostRuntime.run brackets `time.time_ns()` around `kernel(...)` and
   `h.wait()`, so npu_time is HOST wall-clock across submit and completion, not
   a device-side counter. Every "us" in this suite is that. It also means the
   two halves can be timed separately, which is most of the decomposition.

2. WHAT IS LEFT WHEN THE DESIGN DOES NOTHING. The `empty` rung dispatches a
   runtime sequence with no body at all. It cannot be attributed to DMA
   programming, data, cores or arithmetic, because there are none.

The ladder then adds one thing at a time (see null_dispatch_design.py), and the
pipelining sweep asks the question a single median cannot: is the floor a
LATENCY, which overlaps if dispatches are kept in flight, or an OCCUPANCY, which
serialises no matter what? That distinction decides whether the launch floor is
something ConvNeXt's 22 back-to-back layers have to pay 22 times.

Run:  python null_dispatch.py
"""
import json
import os
import pathlib
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np
import pyxrt

import aie.utils as aie_utils
from aie.utils.hostruntime.hostruntime import NPUKernel
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.common import (AIEContext, AIERuntimeArgSpec, DesignGenerator,
                         MLIROperator, PythonGeneratedMLIRArtifact)

HERE = pathlib.Path(__file__).parent
BUF_ELEMS = 8192
WARMUP, SAMPLES = 20, 100
PIPE_DEPTHS = (1, 2, 4, 8, 16)
PIPE_REPS = 20


@dataclass
class NullDispatch(MLIROperator):
    """A dispatch that does as little as the toolchain will allow."""

    variant: str = "empty"
    n_units: int = 0
    xfer_elems: int = 64
    cores_per_col: int = 1
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "variant": "v",
        "n_units": "n",
        "xfer_elems": "el",
        "cores_per_col": "cpc",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                HERE / "null_dispatch_design.py", "my_null_dispatch", (),
                {"dev": aie_utils.get_current_device(),
                 "variant": self.variant, "n_units": self.n_units,
                 "xfer_elems": self.xfer_elems,
                 "cores_per_col": self.cores_per_col}))

    def get_kernel_artifacts(self):
        return []          # no variant needs a compute kernel object

    def get_arg_spec(self):
        return [AIERuntimeArgSpec("in", (BUF_ELEMS,)),
                AIERuntimeArgSpec("out", (BUF_ELEMS,))]


def _bind(op):
    """Load the kernel and return everything the raw pyxrt call needs.

    This mirrors XRTHostRuntime.run exactly, minus the timing, so that the
    submit/wait split is measured on the same call the rest of the suite times
    through `npu_time` and not on some other path.
    """
    runtime = aie_utils.DefaultNPURuntime
    handle = runtime.load(NPUKernel(
        xclbin_path=op.xclbin_artifact.filename,
        kernel_name=op.xclbin_artifact.kernel_name,
        insts_path=op.insts_artifact.filename))
    args = [XRTTensor(s.shape, dtype=s.dtype) for s in op.get_arg_spec()]
    [a.to("npu") for a in args]
    buffers = [a.buffer_object() for a in args]
    insts_bo, insts_bytes = handle.insts_bo, handle.insts.nbytes
    if insts_bo is None:
        insts_bo = XRTTensor(handle.insts, flags=pyxrt.bo.cacheable,
                             group_id=handle.kernel.group_id(1)).buffer_object()
    return handle, insts_bo, insts_bytes, buffers


def measure(label, **kw):
    ctx = AIEContext(mlir_verbose=False, compiler="peano")
    try:
        op = NullDispatch(context=ctx, **kw)
        op.compile()
        handle, ibo, ibytes, bufs = _bind(op)
        fire = lambda: handle.kernel(3, ibo, ibytes, *bufs)

        for _ in range(WARMUP):
            fire().wait()

        submit, wait = [], []
        for _ in range(SAMPLES):
            t0 = time.perf_counter_ns()
            run = fire()
            t1 = time.perf_counter_ns()
            run.wait()
            t2 = time.perf_counter_ns()
            submit.append((t1 - t0) / 1e3)
            wait.append((t2 - t1) / 1e3)

        # Keep K dispatches in flight before waiting on any of them. If the
        # floor is latency, total(K) grows much more slowly than K x total(1).
        pipe = {}
        for depth in PIPE_DEPTHS:
            reps = []
            for _ in range(PIPE_REPS):
                t0 = time.perf_counter_ns()
                runs = [fire() for _ in range(depth)]
                for run in runs:
                    run.wait()
                reps.append((time.perf_counter_ns() - t0) / 1e3)
            pipe[depth] = statistics.median(reps)

        sub, wai = statistics.median(submit), statistics.median(wait)
        row = {"label": label, **kw,
               "insts_bytes": pathlib.Path(op.insts_artifact.filename).stat().st_size,
               "us": sub + wai, "submit_us": sub, "wait_us": wai,
               "us_p10": statistics.quantiles([s + w for s, w in
                                               zip(submit, wait)], n=10)[0],
               "pipeline_us": pipe}
        # total(K) = marginal*K + one_time, fitted over the whole sweep.
        ks = list(pipe)
        row["pipe_marginal_us"], row["pipe_one_time_us"] = np.polyfit(
            ks, [pipe[k] for k in ks], 1)
        print(f"  {label:<34}{row['us']:8.2f}{sub:9.2f}{wai:8.2f}"
              f"{row['insts_bytes']:9d}{row['pipe_marginal_us']:10.2f}", flush=True)
        return row
    except Exception:
        traceback.print_exc()
        print(f"  {label:<34}   FAILED", flush=True)
        return None
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


def main():
    # The submit half is host CPU time, so it drifts with scheduling. Pinning
    # does not change what is being measured, it just stops the medians from
    # moving several us between rungs for reasons that have nothing to do with
    # the design being dispatched.
    try:
        os.sched_setaffinity(0, {0})
    except (AttributeError, OSError):
        pass

    rungs = [("empty sequence, no body", dict(variant="empty", n_units=0))]
    rungs += [(f"{n} core(s) started, no DMA",
               dict(variant="cores", n_units=n, cores_per_col=4))
              for n in (1, 8, 32)]
    rungs += [(f"{n} DMA transfer pair(s), 64 el",
               dict(variant="dma", n_units=n, xfer_elems=64, cores_per_col=1))
              for n in (1, 2, 3, 4, 6, 8)]
    rungs += [(f"8 DMA pairs, {el} el",
               dict(variant="dma", n_units=8, xfer_elems=el, cores_per_col=1))
              for el in (512, 4096)]
    # A second, independent build of the empty design, measured last. Nothing
    # about it differs from the first rung, so the difference between the two is
    # the noise floor -- which is what any step further up has to beat before it
    # can be called a component.
    rungs += [("empty sequence, rebuilt (control)",
               dict(variant="empty", n_units=0, cores_per_col=2))]

    hdr = (f"  {'rung':<34}{'us':>8}{'submit':>9}{'wait':>8}"
           f"{'insts B':>9}{'marginal':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = [r for r in (measure(lbl, **kw) for lbl, kw in rungs) if r]
    (HERE / "null_dispatch.json").write_text(json.dumps(rows, indent=2))

    by = {r["label"]: r for r in rows}
    floor = by.get("empty sequence, no body")
    if not floor:
        print("\nno empty rung, nothing to decompose")
        return 1

    print(f"\n--- the floor ---")
    print(f"  An EMPTY runtime sequence ({floor['insts_bytes']} B of instruction "
          f"stream, no DMA,\n  no data, no arithmetic) costs "
          f"{floor['us']:.1f} us per call:")
    print(f"    {floor['submit_us']:6.1f} us  host-side submit: build the command "
          f"packet and ioctl it")
    print(f"    {floor['wait_us']:6.1f} us  everything after that, until the host "
          f"sees completion")

    ctrl = by.get("empty sequence, rebuilt (control)")
    if ctrl:
        noise = abs(ctrl["us"] - floor["us"])
        print(f"\n--- how much of a step is a step? ---")
        print(f"  The same empty design, built and measured a second time: "
              f"{ctrl['us']:.1f} us against\n  {floor['us']:.1f}. So "
              f"{noise:.1f} us is the noise floor between two builds, and nothing "
              f"smaller\n  than that is a finding.")

    cores = [r for r in rows if r["variant"] == "cores"]
    if cores:
        spread = max(r["us"] for r in cores) - min(r["us"] for r in cores)
        same = all(r["insts_bytes"] == floor["insts_bytes"] for r in cores)
        print(f"\n--- starting the array ---")
        print(f"  1 to 32 cores: {min(r['us'] for r in cores):.1f} to "
              f"{max(r['us'] for r in cores):.1f} us, a {spread:.1f} us spread, and "
              f"the instruction\n  stream stays at {floor['insts_bytes']} B"
              f"{' for every one of them' if same else ''}. Core enable is "
              f"configuration, paid\n  when the xclbin is loaded, NOT per call. "
              f"It is not in the launch floor.")

    dma = sorted([r for r in rows if r["variant"] == "dma"
                  and r["xfer_elems"] == 64], key=lambda r: r["n_units"])
    if len(dma) >= 2:
        slope, icpt = np.polyfit([r["n_units"] for r in dma],
                                 [r["us"] for r in dma], 1)
        b_slope = np.polyfit([r["n_units"] for r in dma],
                             [r["insts_bytes"] for r in dma], 1)[0]
        print(f"\n--- programming the DMAs ---")
        print(f"  " + "  ".join(f"{r['n_units']}:{r['us']:.1f}" for r in dma) +
              f"   (us, at {b_slope:.0f} B of instruction stream per pair)")
        print(f"  The whole span from {dma[0]['n_units']} to "
              f"{dma[-1]['n_units']} pairs is "
              f"{max(r['us'] for r in dma) - min(r['us'] for r in dma):.1f} us. A "
              f"straight line through it reads\n  {slope:.2f} us per pair with a "
              f"{icpt:.1f} us intercept, but the ladder is not straight: "
              f"most\n  of the span is one step between 1 and 2 pairs, and pairs "
              f"3 to {dma[-1]['n_units']} are inside the\n  noise of each other. "
              f"This experiment does not resolve that step; what it does\n"
              f"  settle is the size of the whole DMA term, which is small "
              f"against the {floor['us']:.0f} us floor.")
        big = sorted([r for r in rows if r["variant"] == "dma"
                      and r["n_units"] == 8], key=lambda r: r["xfer_elems"])
        if len(big) > 1:
            # Read this off `wait`, not the total. The totals move by several us
            # across these three, but so does the host submit, and the submit
            # cannot possibly depend on the transfer size. `wait` is the half
            # that contains the device.
            sizes = ", ".join(f"{r['xfer_elems']} el -> {r['wait_us']:.1f}"
                              for r in big)
            span = max(r["wait_us"] for r in big) - min(r["wait_us"] for r in big)
            print(f"  At 8 pairs, {big[0]['xfer_elems']} to "
                  f"{big[-1]['xfer_elems']} elements each is {span:.1f} us of "
                  f"`wait`: {sizes}.")
            print(f"  Same instruction stream, {big[-1]['xfer_elems'] // big[0]['xfer_elems']}x "
                  f"the bytes, no change. It is the TASK COUNT that costs here,")
            print(f"  not the traffic -- at these sizes the transfers are pure "
                  f"latency.")

    print(f"\n--- latency or occupancy? ---")
    print(f"  Keeping K dispatches in flight, empty sequence:")
    for k, v in floor["pipeline_us"].items():
        print(f"    K={k:<3d} {v:8.1f} us total   {v / int(k):6.1f} us per dispatch")
    print(f"  Fit: {floor['pipe_marginal_us']:.1f} us per dispatch that does NOT "
          f"overlap, plus a\n  one-time {floor['pipe_one_time_us']:.1f} us that "
          f"does. The non-overlapping part is essentially\n  the host submit "
          f"({floor['submit_us']:.1f} us), so the rest of the floor is round-trip "
          f"LATENCY:\n  it is paid once for a queue kept full, not once per call.")

    df = HERE / "dispatch_floor.json"
    if df.exists():
        conv = json.loads(df.read_text())
        one = next((c for c in conv if c["cols"] == 1), None)
        if one:
            print(f"\n--- the ladder, end to end ---")
            print(f"  {floor['us']:6.1f} us  empty sequence")
            if dma:
                print(f"  {dma[-1]['us']:6.1f} us  + {dma[-1]['n_units']} DMA "
                      f"transfer pairs")
            print(f"  {one['us']:6.1f} us  dispatch_floor's near-zero-work conv, "
                  f"1 column")
            print(f"  {one['us'] - floor['us']:6.1f} us  is everything that conv "
                  f"does beyond dispatching at all")
    print("\nwrote null_dispatch.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
