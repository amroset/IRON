#!/usr/bin/env python3
"""A one-core harness around the SAME microkernel the full array runs.

Why this exists: the deck can say what the whole 32-core operator achieves, and
it can say what the datapath could achieve in principle (32 MAC/cyc/core), but
nothing measured sits between the two. The residual in the cycle budget is a
leftover, not an attribution. Tracing the real operator would fix that, except
the trace stream will not route alongside the full array's DMAs -- both the
8-column and 4-column attempts fail with "Unable to find a legal routing"
(trace_kernel.log, trace_kernel_c4.log).

So we shrink the problem instead of the instrumentation. One core, one column,
the identical `matmul_bf16_f32` object built with the identical -DDIM_*/col-major
flags, fed the identical tile shape. The trace routes trivially, and the
event0/event1 markers already in the kernel bracket exactly one
`A[m,k] x B[k,n] -> acc[m,n]` step -- the same step the real design issues K/k
times per output tile.

Two modes, and the difference between them is the point:

  resident    A and B are acquired ONCE, then `steps` matmul calls run back to
              back on L1-resident data. No DMA can stall the measured region, so
              this is the microkernel's own cost -- the floor the GEMM imposes.

  streamed    A and B are acquired and released every iteration, exactly as
              design.py's core_fn does, with the runtime feeding `steps` tiles.
              Same arithmetic, now with the real fifo cadence.

  production  the whole core_fn loop nest: `tiles` output tiles, each one a
              zero(), `steps` accumulate steps, and a convert_copy(). For the
              headline layer a core really does 3 tiles x 12 steps, so the
              prologue/epilogue amortises over 12 steps, not over 64. Running
              the real shape is what turns "per-tile overhead" from a guess
              into a number.

All three compute a checkable answer, so all three are verified against torch.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2, Tile
from aie.iron.placers import SequentialPlacer

# Must match design.py: the microkernel MAC tile for npu2 bf16, non-emulated.
R, S, T = 4, 8, 8


def my_kernel_probe(
    dev,
    m,
    k,
    n,
    steps,
    a_col_maj,
    b_col_maj,
    c_col_maj,
    mode,
    tiles,
    trace_size,
    kernel_object,
    coretile_events=None,
):
    assert mode in ("resident", "streamed", "production")
    if mode != "production":
        tiles = 1
    assert m % R == 0 and k % S == 0 and n % T == 0, "tile must fit the MAC dims"

    dtype_in = bfloat16
    dtype_acc = np.float32
    dtype_out = bfloat16

    A_l1_ty = np.ndarray[(m, k), np.dtype[dtype_in]]
    B_l1_ty = np.ndarray[(k, n), np.dtype[dtype_in]]
    C_l1_ty = np.ndarray[(m, n), np.dtype[dtype_out]]
    C_l1_ty_acc = np.ndarray[(m, n), np.dtype[dtype_acc]]

    # Resident mode needs one tile of each operand; the streaming modes need one
    # per step, so the DRAM buffers grow accordingly.
    n_ops = 1 if mode == "resident" else tiles * steps
    A_ty = np.ndarray[(n_ops * m * k,), np.dtype[dtype_in]]
    B_ty = np.ndarray[(n_ops * k * n,), np.dtype[dtype_in]]

    # The trace gets its OWN sequence tensor (arg 3) rather than riding in C's
    # tail via ddr_id=-1. The tail form is what the full-array design assumes,
    # but it never produced a byte here -- the trace ops lower correctly and the
    # buffer stays all-zero. An explicit buffer at a real arg index is
    # unambiguous: the shim DMA has somewhere to land that the host also owns.
    enable_trace = trace_size > 0
    C_ty = np.ndarray[(tiles * m * n,), np.dtype[dtype_out]]
    TRACE_ty = np.ndarray[(trace_size // 4,), np.dtype[np.uint32]]

    zero_kernel = Kernel("zero_f32", kernel_object, [C_l1_ty_acc])
    matmul_kernel = Kernel(
        "matmul_bf16_f32", kernel_object, [A_l1_ty, B_l1_ty, C_l1_ty_acc]
    )
    convert_kernel = Kernel(
        "convert_copy_f32_to_bf16", "convert_copy.o",
        [C_l1_ty_acc, C_l1_ty, np.int32],
    )

    of_a = ObjectFifo(A_l1_ty, name="probe_a", depth=2)
    of_b = ObjectFifo(B_l1_ty, name="probe_b", depth=2)
    of_c = ObjectFifo(C_l1_ty, name="probe_c", depth=1)

    def core_resident(in_a, in_b, out_c, zero, matmul, convert, acc):
        # Acquire once, OUTSIDE the timed loop: after this point the core never
        # touches a lock again until the very end, so every cycle between
        # event0 and event1 belongs to the microkernel.
        elem_a = in_a.acquire(1)
        elem_b = in_b.acquire(1)
        zero(acc)
        for _ in range_(steps):
            matmul(elem_a, elem_b, acc)
        in_a.release(1)
        in_b.release(1)
        elem_c = out_c.acquire(1)
        convert(acc, elem_c, m * n)
        out_c.release(1)

    def core_streamed(in_a, in_b, out_c, zero, matmul, convert, acc):
        # The cadence design.py actually uses: one acquire/release pair per
        # k-step. Identical arithmetic, so any extra cycles are lock + DMA wait.
        zero(acc)
        for _ in range_(steps):
            elem_a = in_a.acquire(1)
            elem_b = in_b.acquire(1)
            matmul(elem_a, elem_b, acc)
            in_a.release(1)
            in_b.release(1)
        elem_c = out_c.acquire(1)
        convert(acc, elem_c, m * n)
        out_c.release(1)

    def core_production(in_a, in_b, out_c, zero, matmul, convert, acc):
        # design.py's core_fn verbatim, minus the runtime-parameter plumbing:
        # one zero + K/k accumulate steps + one convert, per output tile.
        for _ in range_(tiles):
            zero(acc)
            for _ in range_(steps):
                elem_a = in_a.acquire(1)
                elem_b = in_b.acquire(1)
                matmul(elem_a, elem_b, acc)
                in_a.release(1)
                in_b.release(1)
            elem_c = out_c.acquire(1)
            convert(acc, elem_c, m * n)
            out_c.release(1)

    acc_buffer = Buffer(type=C_l1_ty_acc, name="probe_acc")
    core_fn = {"resident": core_resident, "streamed": core_streamed,
               "production": core_production}[mode]
    worker = Worker(
        core_fn,
        [of_a.cons(), of_b.cons(), of_c.prod(),
         zero_kernel, matmul_kernel, convert_kernel, acc_buffer],
        placement=Tile(0, 2),
        stack_size=0xD00,
    )

    c_rows = tiles * m
    c_tap = TensorAccessPattern((c_rows, n), offset=0, sizes=[1, 1, c_rows, n],
                                strides=[0, 0, n, 1])

    rt = Runtime()
    if enable_trace:
        with rt.sequence(A_ty, B_ty, C_ty, TRACE_ty) as (A, B, C, _TR):
            # The default core event set spends slots on DMA ports we do not
            # need here. Swapping in INSTR_LOAD/INSTR_STORE is what turns "900
            # cycles the MAC pipe was not issuing" into an attribution.
            ev = None
            if coretile_events:
                from aie.utils.trace.events.aie2p import CoreEvent
                ev = [CoreEvent[name] for name in coretile_events]
            rt.enable_trace(trace_size, workers=[worker], ddr_id=3,
                            coretile_events=ev)
            rt.start(worker)
            rt.fill(of_a.prod(), A)
            rt.fill(of_b.prod(), B)
            rt.drain(of_c.cons(), C, tap=c_tap, wait=True)
    else:
        with rt.sequence(A_ty, B_ty, C_ty) as (A, B, C):
            rt.start(worker)
            rt.fill(of_a.prod(), A)
            rt.fill(of_b.prod(), B)
            rt.drain(of_c.cons(), C, tap=c_tap, wait=True)

    return Program(NPU2(), rt).resolve_program(SequentialPlacer())
