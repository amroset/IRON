# Baseline: scalar vs vectorized kernel — 1×1 conv on YOLO / MobileNetV3 shapes

**Date:** 2026-07-10   **Device:** NPU2 (Strix, aie2p, 4 rows × 8 cols)
**Config (held fixed):** tile `m/k/n = 16/64/64`, `prio_accuracy=True` (bf16 in, f32
accumulate, bf16 out), `emulate_bf16_mmul_with_bfp16=False`, `batch=1`.
**Correctness:** every row passes vs the torch `F.conv2d` golden at `rel/abs_tol = 0.005`.
Throughput (GFLOP/s) is on **useful** work `2·C_out·C_in·H·W` (excludes zero-padding).

> **What this baseline is.** The current operator is a **1×1 conv mapped onto plain
> GEMM** — the compute kernel *is* the GEMM `aie::mmul` kernel and the host data
> orchestration *is* the GEMM design, reused verbatim. The only conv-specific logic is
> the dim relabel `M=C_out, K=C_in, N=H·W`, the weights→A / activation→B mapping (so no
> transpose is needed), and host-side zero-padding (mirrors GEMM's `pad_A`/`pad_B`).
> This table is therefore the **vanilla-GEMM reference**; every planned improvement
> (per-shape orchestration, adaptive axis→array mapping, batching) is measured against it.
>
> - **`scalar`** = the plain triple-loop oracle (`use_scalar=True`): correct but not a
>   performance path. It exists to trust the vectorized result against.
> - **`vectorized`** = the `aie::mmul` 2×2-expanded kernel (`use_scalar=False`): the real path.

## Results

| Model | Layer (1×1) | M/K/N | cols | pad | scalar µs | scalar GF | vec µs | vec GF | **vec speedup** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3 | project 64→24 @56² | 24/64/3136 | 1 | 2.67 | 23 599 | 0.41 | 208.0 | 46.3 | **113×** |
| MobileNetV3 | project 64→24 @56² | 24/64/3136 | 4 | 2.83 | 6 277 | 1.53 | 125.5 | 76.8 | **50×** |
| MobileNetV3 | project 64→24 @56² | 24/64/3136 | 8 | 3.05 | 3 423 | 2.81 | 167.2 | 57.6 | **21×** |
| YOLOv5 | P4 256→128 @40² | 128/256/1600 | 1 | 1.00 | 96 143 | 1.09 | 403.4 | 259.9 | **238×** |
| YOLOv5 | P4 256→128 @40² | 128/256/1600 | 4 | 1.12 | 26 778 | 3.92 | 209.6 | 500.4 | **128×** |
| YOLOv5 | P4 256→128 @40² | 128/256/1600 | 8 | 1.28 | 15 326 | 6.84 | 147.7 | **710.1** | **104×** |
| YOLOv5 | P5 512→256 @20² | 256/512/400 | 1 | 1.12 | 107 112 | 0.98 | 426.4 | 245.9 | **251×** |
| YOLOv5 | P5 512→256 @20² | 256/512/400 | 4 | 1.28 | 31 060 | 3.38 | 194.2 | 539.9 | **160×** |
| YOLOv5 | P5 512→256 @20² | 256/512/400 | 8 | 1.28 | 15 339 | 6.84 | 173.2 | 605.3 | **89×** |

Raw data: [`baseline_scalar_vs_vectorized.csv`](baseline_scalar_vs_vectorized.csv).

## Key findings

1. **Vectorized is 21–251× faster than scalar** across every shape and column count —
   two orders of magnitude. Scalar tops out at ~7 GFLOP/s; vectorized reaches **710 GFLOP/s**
   (YOLO P4 @ 8 cols). This is the headline improvement to show.

2. **The speedup *shrinks* as columns increase** (e.g. YOLO P5: 251× → 89×). That is not
   the vectorized path getting worse — it's the *scalar* path getting proportionally
   *better*: scalar is so compute-bound that adding cores scales it near-linearly
   (c1→c8 ≈ 7×), while the already-efficient vectorized path hits bandwidth/overhead
   limits sooner (c1→c8 ≈ 2.5×). The absolute vectorized latency still improves with columns.

3. **More columns help the vectorized path only while pixels fill them.** YOLO P4/P5
   (large N) scale cleanly to 8 columns. But MobileNetV3 64→24 *regresses* at 8 cols
   (167 µs) vs 4 cols (125 µs): it's padding-bound (M=24→64, pad 2.67→3.05×), so extra
   columns add zero-work rather than parallelism. Confirms the per-shape column rule.

## Status

**Baseline captured.** This is the vanilla-GEMM reference for the pointwise-conv operator.
Next improvements (departing from plain GEMM to exploit conv structure) should re-run the
same shapes/config and compare the vectorized column against this table.
