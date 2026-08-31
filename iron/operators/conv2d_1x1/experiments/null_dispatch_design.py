#!/usr/bin/env python3
"""Designs that do progressively less, down to nothing at all.

`dispatch_floor.py` measured the per-call cost with a conv that does almost no
arithmetic, and got ~67 us. It could not say what those 67 us ARE, because that
conv still programs shim DMAs, still moves data, still starts cores. This file
strips those away one at a time so the difference between two adjacent rungs
names a component.

  empty   a runtime sequence with two DRAM arguments and NO body. Sixteen bytes
          of instruction stream, no DMA descriptor, no data. The dispatch still
          has to be built, submitted, executed by the firmware and completed.
          Whatever this costs is the irreducible floor.

  cores   `rt.start()` on N workers, still no DMA and no data. The generated
          MLIR carries real `aie.core` bodies, but the runtime sequence stays
          empty: core enable is part of the xclbin configuration, so if this
          matches `empty` then starting the array is paid at LOAD time and is
          not part of the per-call cost at all.

  dma     N shim -> memtile -> shim forwards of `xfer_elems` bf16. No core
          touches the data, so this is DMA task programming and nothing else.
          Sweeping N gives the marginal cost of one more transfer; sweeping
          `xfer_elems` at fixed N separates task count from bytes.

All three take the same two arguments and go through the same compile and
dispatch path as every other operator in the suite, so the rungs are comparable
to each other and to `dispatch_floor.py`'s conv.
"""
import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer

# Both arguments are this many bf16. Big enough that any `xfer_elems` we sweep
# fits inside it, small enough that allocating it costs nothing.
BUF_ELEMS = 8192


def my_null_dispatch(dev, variant, n_units, xfer_elems, cores_per_col):
    buf_ty = np.ndarray[(BUF_ELEMS,), np.dtype[bfloat16]]
    rt = Runtime()

    if variant == "empty":
        with rt.sequence(buf_ty, buf_ty) as (_a, _b):
            pass
        return Program(dev, rt).resolve_program(SequentialPlacer(cores_per_col))

    if variant == "cores":
        # A Worker with an empty body still lowers to an `aie.core` with an
        # infinite loop, so the cores are real and are running; they just have
        # nothing to synchronise with.
        workers = [Worker(lambda: None, []) for _ in range(n_units)]
        with rt.sequence(buf_ty, buf_ty) as (_a, _b):
            rt.start(*workers)
        return Program(dev, rt).resolve_program(SequentialPlacer(cores_per_col))

    if variant == "dma":
        # `.forward()` routes the fifo straight through a memtile, so there is
        # no compute tile and no kernel object anywhere in this design.
        ins = [ObjectFifo(np.ndarray[(xfer_elems,), np.dtype[bfloat16]],
                          name=f"in{i}", depth=2) for i in range(n_units)]
        outs = [ins[i].cons().forward() for i in range(n_units)]
        tap = TensorAccessPattern((1, BUF_ELEMS), 0,
                                  [1, 1, 1, xfer_elems], [0, 0, 0, 1])
        with rt.sequence(buf_ty, buf_ty) as (a_in, b_out):
            tg = rt.task_group()
            for i in range(n_units):
                rt.fill(ins[i].prod(), a_in, tap, task_group=tg)
            for i in range(n_units):
                rt.drain(outs[i].cons(), b_out, tap, wait=True, task_group=tg)
            rt.finish_task_group(tg)
        return Program(dev, rt).resolve_program(SequentialPlacer(cores_per_col))

    raise ValueError(f"unknown variant {variant!r}")
