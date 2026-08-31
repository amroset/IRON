# Pointwise Convolutions on AMD XDNA2

An implementation of the 1×1 (pointwise) convolution operator for the AMD XDNA2 Neural Processing Unit, built on the [IRON](https://github.com/Xilinx/mlir-aie) framework.

---

## Overview

A 1×1 convolution is fundamentally a matrix multiplication, so this kernel is derived from the IRON GEMM implementation. The contribution here is in **how the work is mapped onto the compute grid**, not in the inner kernel itself.

## Target hardware

The XDNA2 NPU exposes:

| Component | Count | Role |
|---|---|---|
| Compute tiles (AIE2 cores) | 32 | VLIW ISA with SIMD datapath |
| Memory tiles | 8 | Manage DMA transfers |
| Shim tiles | 8 | Interface to the host CPU |

## Key contribution

**Adaptive workload distribution.** The mapping of the output tile grid onto the array is chosen at runtime based on the shape of the input layer, swapping the roles of rows and columns rather than fixing them ahead of time.

This is a small orchestration change, but for shapes that are poorly matched to a static mapping it yields a measured speedup of **up to 6.5×** over the static baseline.

## Documentation

Further detail — including the mapping strategy and measured results — is in the accompanying slide deck:

- [`pointwise_convolution.pdf`](./pointwise_convolution.pdf)

---

## Acknowledgements

Completed as part of the course *Systems on Chips for Data Analytics and Machine Learning* at ETH Zürich.

This project is forked from the IRON / `mlir-aie` framework.