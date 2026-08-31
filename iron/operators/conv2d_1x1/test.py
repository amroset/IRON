#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Real-model test suite for the pointwise (1x1) conv operator (OPT version).
#
# This exercises the *torch.nn data layout* end to end and, unlike the milestone
# suite, draws its shapes from the actual 1x1 (pointwise) convolutions of eight
# real networks: ResNet-50, ResNeXt-50, MobileNetV2, MobileNetV3-Large,
# EfficientNet-B0, ConvNeXt-Tiny, DenseNet-121 and YOLOv5l. The goal is to
# CONFIRM the shape patterns found across those models (see pointwise_patterns.md)
# with real on-device measurements: column-starvation (58% of real shapes have
# N<512), the tall M>N tail (34%), and channel misalignment / padding overhead.
#
# Mapped GEMM dims:  M = C_out,  K = C_in,  N = H*W (pixels).
# The operator pads each mapped dim up to the tiling granularity and zero-pads the
# host buffers (op.py pad_* helpers), so non-tile-aligned shapes run directly.
#
# Column sweeping: NUM_COLS below is the single knob to sweep the AIE column count
# (the measured throughput lever). Shapes whose padded N cannot be tiled within the
# shim-DMA block-descriptor limit at the current NUM_COLS are auto-skipped, so the
# same suite is valid across NUM_COLS in {1, 2, 4, 8}.

import json
import os
import pathlib

import pytest
import aie.utils as aie_utils

from iron.operators.conv2d_1x1_opt.op import Conv2d1x1
from iron.operators.conv2d_1x1_opt.reference import generate_golden_reference
from iron.common.test_utils import run_test

# --- AIE column count for the whole suite. Sweep without editing the file via
#     the CONV_COLS env var, e.g.  CONV_COLS=8 pytest .../test.py  ---
NUM_COLS = int(os.environ.get("CONV_COLS", "1"))

# Proven default tile. Tile-shape tuning is a separate, EMPIRICAL axis (an
# analytical tile model was unreliable on hardware): autotune.py sweeps a small
# candidate set on-device and caches the fastest per (shape, cols) in
# tile_cache.json; get_optimal_tile() reads it. Set CONV_TUNE=1 to use the tuned
# tiles for the suite. The selection lives here in the host/test (like softmax's
# get_optimal_* helpers), NOT in op.py.
TILE_M, TILE_K, TILE_N = 16, 64, 64
BD_MAX = 64  # shim-DMA block-descriptor limit: N/(n*cols) and M/(m*4) must be <= 64

# Candidate tiles for autotuning. Valid: m mult of 8, n mult of 16, k mult of 8;
# n=16 is skipped (consistently slow on hardware); oversized tiles that overflow
# L1 simply fail to build and are skipped by the tuner.
TILE_CANDIDATES = [
    (16, 64, 64), (32, 64, 64), (64, 64, 64),
    (16, 128, 64), (16, 64, 128), (32, 32, 64),
]
_TILE_CACHE = pathlib.Path(__file__).parent / "tile_cache.json"


def get_optimal_tile(C_in, C_out, N, cols, default=(TILE_M, TILE_K, TILE_N)):
    """Empirically-tuned tile for this (shape, cols), or the proven default if it
    is not in tile_cache.json. Populate the cache with autotune.py."""
    try:
        cache = json.loads(_TILE_CACHE.read_text())
    except (FileNotFoundError, ValueError):
        return default
    return tuple(cache.get(f"{C_in}_{C_out}_{N}_{cols}", default))


def get_params():
    dev = aie_utils.get_current_device()
    device_type = dev.resolve().name
    max_cols = dev.cols
    # Real 1x1 conv shapes as (H=W, C_in=K, C_out=M) with their model/role and
    # regime tag. Regimes:  wide (N>M), tall (M>N), and whether channels are
    # 64-aligned (padding). H*W is the pixel count N.
    # fmt: off
    #      H,  C_in, C_out   model / layer                       regime
    default_shapes = [
        ( 56,   64,   256),  # ResNet-50 expand  64->256 @56x56  wide, aligned, big-N control
        ( 14,   80,   480),  # EfficientNet exp  80->480 @14x14  tall, UNaligned, padding stress
        (  7, 1024,  2048),  # ResNeXt-50 expand 1024->2048 @7x7 very tall, aligned, big-M
        ( 14,  512,   128),  # DenseNet dense1x1 512->128 @14x14 fixed-small-M, col-starved
    ]
    extensive_shapes = [
        # ResNet-50 / ResNeXt-50 (powers-of-two: aligned)
        ( 56,   64,    64),  # ResNet reduce   64->64   @56x56  wide square-ish
        (  7,  512,  2048),  # ResNet deep exp 512->2048 @7x7   tall
        (  7, 2048,   512),  # ResNet deep red 2048->512 @7x7   big-K, tall
        ( 28,  128,   512),  # ResNet mid exp  128->512 @28x28  aligned mid
        ( 28,  512,   512),  # ResNeXt reduce  512->512 @28x28
        # ConvNeXt-Tiny (pointwise MLP up/down)
        ( 56,   96,   384),  # ConvNeXt pw_up  96->384  @56x56  wide, UNaligned-K
        (  7,  768,  3072),  # ConvNeXt pw_up  768->3072 @7x7   EXTREME M, tiny N
        (  7, 3072,   768),  # ConvNeXt pw_dn  3072->768 @7x7   big-K, tall
        ( 14,  384,  1536),  # ConvNeXt pw_up  384->1536 @14x14
        # EfficientNet-B0 / MobileNetV2 / V3 (NAS channels: UNaligned)
        (  7, 1152,   192),  # EfficientNet proj 1152->192 @7x7 tall, UNaligned
        ( 28,   40,   240),  # EfficientNet exp  40->240  @28x28 UNaligned
        ( 14,   96,   576),  # MobileNetV2 exp   96->576  @14x14 UNaligned
        ( 28,   32,   192),  # MobileNetV2 exp   32->192  @28x28 UNaligned-K
        ( 14,  112,   672),  # MobileNetV3 exp   112->672 @14x14 UNaligned
        # DenseNet-121 (fixed small M=128, growing K; transitions)
        ( 28,  256,   128),  # DenseNet dense1x1 256->128 @28x28
        ( 14, 1024,   512),  # DenseNet transition 1024->512 @14x14
        # YOLOv5l
        ( 20, 1024,   512),  # YOLO sppf 1024->512 @20x20       big-K, N=400
        ( 80,  128,    64),  # YOLO P3   128->64   @80x80       big-N (needs cols>1)
    ]
    # fmt: on

    def ceil_to(v, mult):
        return ((v + mult - 1) // mult) * mult

    def runnable(H, C_in, C_out, cols):
        N = H * H
        # Mirror op.py's auto cols->M swap: tall layers (C_out>pixels) map C_out
        # onto the columns and pixels onto the rows when cols>=4.
        swap = C_out > N and cols >= 4
        rows_raw, cols_raw = (N, C_out) if swap else (C_out, N)
        Mp = ceil_to(rows_raw, TILE_M * 4)
        Np = ceil_to(cols_raw, TILE_N * cols)
        # shim-DMA BD limit on both streamed axes
        return Np // (TILE_N * cols) <= BD_MAX and Mp // (TILE_M * 4) <= BD_MAX

    cols = min(NUM_COLS, max_cols)
    tune = bool(int(os.environ.get("CONV_TUNE", "0")))  # use tuned tiles?
    params = []

    def add(shape_list, is_extensive):
        for (H, C_in, C_out) in shape_list:
            if device_type == "npu1" and TILE_M < 16:
                continue
            if not runnable(H, C_in, C_out, cols):
                continue  # would overflow the DMA BD range at this NUM_COLS
            tm, tk, tn = (
                get_optimal_tile(C_in, C_out, H * H, cols) if tune
                else (TILE_M, TILE_K, TILE_N)
            )
            marks = [pytest.mark.extensive] if is_extensive else []
            params.append(
                pytest.param(1, H, H, C_in, C_out, tm, tk, tn, cols, marks=marks)
            )

    add(default_shapes, is_extensive=False)
    add(extensive_shapes, is_extensive=True)
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "batch,H,W,C_in,C_out,m,k,n,num_aie_columns",
    get_params(),
)
def test_conv2d_1x1(batch, H, W, C_in, C_out, m, k, n, num_aie_columns, aie_context):
    # torch.nn-native tensors: x [b,C_in,H,W], w [C_out,C_in,1,1], y [b,C_out,H,W].
    golden = generate_golden_reference(batch=batch, H=H, W=W, C_in=C_in, C_out=C_out)
    x = golden["x"]  # NCHW activation
    w = golden["w"]  # OIHW weight
    y = golden["y"]  # NCHW golden output

    operator = Conv2d1x1(
        batch=batch,
        H=H,
        W=W,
        C_in=C_in,
        C_out=C_out,
        tile_m=m,
        tile_k=k,
        tile_n=n,
        num_aie_columns=num_aie_columns,
        use_scalar=False,  # vectorized aie::mmul path (proven, matches GEMM)
        prio_accuracy=True,  # bf16 in, f32 accumulate -> narrow to bf16 (accurate)
        emulate_bf16_mmul_with_bfp16=False,
        context=aie_context,
    )

    # input_operands returns the (rows, cols) buffers in the design's arg order:
    # (weights, activation) in the plain mapping, (activation, weights) under the
    # cols->M swap. Every buffer stays in natural torch storage; pad_* only
    # zero-extends (no-op for aligned shapes) -- no host transpose.
    rows_buf, cols_buf = operator.input_operands(w, x)
    input_buffers = {"A": rows_buf, "B": cols_buf}
    output_buffers = {"C": operator.pad_output(y).flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
    )

    # Throughput on USEFUL work (real conv MACs = 2*C_out*C_in*pixels, excludes
    # padding). pad_overhead exposes the array time spent multiplying zeros.
    gflops = (2.0 * C_out * C_in * (H * W)) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(
        f"cols={num_aie_columns}  swap={operator.swap}  "
        f"design M/K/N = {operator.M}/{operator.K}/{operator.N} "
        f"(conv C_out/C_in/pixels {operator.M_raw}/{operator.K_raw}/{operator.N_raw}, "
        f"pad_overhead {operator.pad_overhead:.2f}x)\n"
    )

    assert not errors, "Test failed"


# --- batch > 1 -------------------------------------------------------------
# A batch is `batch` independent 1x1 convs sharing the same weights. The program
# is per-image, so a batch runs as one dispatch per image (op.image_operands),
# reusing the compiled program. This exercises both mappings: a wide layer (plain)
# and a tall layer (cols->M swap, when cols>=4).
def get_batched_params():
    dev = aie_utils.get_current_device()
    device_type = dev.resolve().name
    cols = min(NUM_COLS, dev.cols)

    # (batch, H, C_in, C_out)
    batched_shapes = [
        (3, 28, 128, 512),   # wide (plain mapping), N=784
        (2,  7, 1024, 2048), # tall (cols->M swap at cols>=4), N=49
    ]

    def ceil_to(v, mult):
        return ((v + mult - 1) // mult) * mult

    def runnable(H, C_out):
        # Both mappings stream N and M; check the BD limit for whichever axis each
        # ends up on (swap decided per-shape, but the padded tile counts are small
        # here for every column count, so a conservative both-ways check suffices).
        N = H * H
        for rows_raw, cols_raw in ((C_out, N), (N, C_out)):
            Mp, Np = ceil_to(rows_raw, TILE_M * 4), ceil_to(cols_raw, TILE_N * cols)
            if Np // (TILE_N * cols) > BD_MAX or Mp // (TILE_M * 4) > BD_MAX:
                return False
        return True

    params = []
    for (batch, H, C_in, C_out) in batched_shapes:
        if device_type == "npu1" and TILE_M < 16:
            continue
        if not runnable(H, C_out):
            continue
        params.append(pytest.param(batch, H, H, C_in, C_out, TILE_M, TILE_K, TILE_N, cols))
    return params


@pytest.mark.parametrize(
    "batch,H,W,C_in,C_out,m,k,n,num_aie_columns",
    get_batched_params(),
)
def test_conv2d_1x1_batched(batch, H, W, C_in, C_out, m, k, n, num_aie_columns, aie_context):
    golden = generate_golden_reference(batch=batch, H=H, W=W, C_in=C_in, C_out=C_out)
    x = golden["x"]  # [batch, C_in, H, W]
    w = golden["w"]  # [C_out, C_in, 1, 1]  (shared across the batch)
    y = golden["y"]  # [batch, C_out, H, W]

    operator = Conv2d1x1(
        batch=batch, H=H, W=W, C_in=C_in, C_out=C_out,
        tile_m=m, tile_k=k, tile_n=n, num_aie_columns=num_aie_columns,
        use_scalar=False, prio_accuracy=True, emulate_bf16_mmul_with_bfp16=False,
        context=aie_context,
    )

    # One dispatch per image; the weights are shared, only the activation/output
    # slice changes. Each image is verified against its own golden slice.
    for b, (rows_buf, cols_buf) in enumerate(operator.image_operands(w, x)):
        input_buffers = {"A": rows_buf, "B": cols_buf}
        output_buffers = {"C": operator.pad_output(y[b]).flatten()}
        errors, _, _ = run_test(
            operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
        )
        assert not errors, f"batch image {b} failed (swap={operator.swap})"

    print(
        f"\nbatch={batch}  cols={num_aie_columns}  swap={operator.swap}  "
        f"{batch} images x {C_out}/{C_in}/{H*W} verified\n"
    )
