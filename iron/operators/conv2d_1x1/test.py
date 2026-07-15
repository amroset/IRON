#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test for the pointwise (1x1) conv operator.
#
# This exercises the *torch.nn data layout* end to end: the reference builds real
# torch.nn.Conv2d-shaped tensors (NCHW activation, OIHW weight, NCHW golden
# output), and we hand the operator the FLATTENED torch tensors with NO host-side
# reshaping. The 1x1 conv is a plain row-major GEMM  Y = W @ X (weights are A,
# activation is B), so the flattened tensors line up with the design's buffers
# directly. A correct result proves the data-layout mapping is right.
#
# Mapped GEMM dims:  M = C_out,  K = C_in,  N = H*W (pixels).
# Milestone scope: single column (num_aie_columns=1) and batch=1. The conv shape
# no longer has to tile evenly -- the operator pads the mapped dims up to the
# tiling granularity and zero-pads the host buffers (see op.py pad_* helpers), so
# these tests exercise the real, non-tile-aligned MobileNetV3 / YOLO 1x1 shapes.

import pytest
import aie.utils as aie_utils

from iron.operators.conv2d_1x1.op import Conv2d1x1
from iron.operators.conv2d_1x1.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    dev = aie_utils.get_current_device()
    device_type = dev.resolve().name
    # Tiles apply to the MAPPED dims: tile_m over C_out, tile_k over C_in,
    # tile_n over pixels (H*W). The operator now PADS each mapped dim up to the
    # tiling granularity (M->tile_m*4, K->tile_k, N->tile_n*num_cols) and zero-
    # pads the host buffers to match, so the conv shape no longer has to tile
    # evenly. These shapes are the real 1x1 (pointwise) convs from MobileNetV3
    # and YOLO -- most of them do NOT tile evenly, which is exactly the point.
    #
    # With fixed tiles m=16, k=64, n=64 and num_cols=1 the padding lands as:
    #   M (=C_out)  padded to a multiple of 64   -> pads when C_out % 64 != 0
    #   K (=C_in)   padded to a multiple of 64   -> pads when C_in  % 64 != 0
    #   N (=H*W)    padded to a multiple of 64   -> pads when H*W   % 64 != 0
    # fmt: off
    #  batch,  H,  W, C_in, C_out,  m,  k,  n     model / 1x1 layer            pads
    default_params = [
        (    1, 40, 40,  256,   128, 16, 64, 64),  # YOLO  P4  256->128 @40x40   none (aligned control)
        (    1, 14, 14,   80,   240, 16, 64, 64),  # MNv3  expand 80->240 @14x14  M,K,N
        (    1, 20, 20,  512,   256, 16, 64, 64),  # YOLO  P5  512->256 @20x20    N only
    ]
    extensive_params = [
        # NOTE: YOLO P3 (128->64 @80x80, N=6400) is intentionally omitted here:
        # with a single column and tile_n=64 it needs N/64=100 column-tiles, but
        # the shim DMA block-descriptor count caps at 64 (N <= 64*tile_n = 4096
        # per column). Large early feature maps require num_aie_columns > 1 --
        # the multi-column throughput regime, tracked as the next milestone.
        (    1, 56, 56,   64,    24, 16, 64, 64),  # MNv3  project 64->24 @56x56  M only
        (    1, 28, 28,   40,   120, 16, 64, 64),  # MNv3  expand 40->120 @28x28  M,K,N
        (    1, 14, 14,  112,   480, 16, 64, 64),  # MNv3  expand 112->480 @14x14 M,K,N
        (    1,  7,  7,  160,   960, 16, 64, 64),  # MNv3  expand 160->960 @7x7    K,N
        (    1,  7,  7,  960,   160, 16, 64, 64),  # MNv3  project 960->160 @7x7   M,N
        (    1, 20, 20, 1024,   512, 16, 64, 64),  # YOLO  SPPF 1024->512 @20x20   N only
    ]
    # fmt: on

    params = []

    def add_params(param_list, is_extensive):
        for p in param_list:
            batch, H, W, C_in, C_out, m, k, n = p
            # AIE2 mm kernel needs m % (4*r) == 0 (r=4 for bf16), so m >= 16.
            if device_type == "npu1" and m < 16:
                continue
            marks = [pytest.mark.extensive] if is_extensive else []
            params.append(pytest.param(*p, marks=marks))

    add_params(default_params, is_extensive=False)
    add_params(extensive_params, is_extensive=True)
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "batch,H,W,C_in,C_out,m,k,n",
    get_params(),
)
def test_conv2d_1x1(batch, H, W, C_in, C_out, m, k, n, aie_context):
    # torch.nn-native tensors: x [b,C_in,H,W], w [C_out,C_in,1,1], y [b,C_out,H,W].
    golden = generate_golden_reference(
        batch=batch, H=H, W=W, C_in=C_in, C_out=C_out
    )
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
        num_aie_columns=1,  # single column for the layout milestone
        use_scalar=False,  # vectorized aie::mmul path (proven, matches GEMM)
        prio_accuracy=True,  # bf16 in, f32 accumulate -> narrow to bf16 (accurate)
        emulate_bf16_mmul_with_bfp16=False,
        context=aie_context,
    )

    # The flattened torch tensors ARE the operator buffers (no layout reshaping):
    #   A = weights    w [C_out,C_in,1,1] -> [C_out, C_in]  = [M, K]
    #   B = activation x [1,C_in,H,W]     -> [C_in, H*W]    = [K, N]
    #   C = output     y [1,C_out,H,W]    -> [C_out, H*W]   = [M, N]
    # When the conv shape doesn't tile evenly, pad_* zero-extends each buffer up
    # to the padded dims; for aligned shapes they are no-ops. The padded output
    # equals the real output zero-extended, so we compare against a padded golden.
    input_buffers = {
        "A": operator.pad_weight(w).flatten(),
        "B": operator.pad_activation(x).flatten(),
    }
    output_buffers = {"C": operator.pad_output(y).flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
    )

    # Throughput on USEFUL work (real conv MACs = 2*C_out*C_in*pixels, excluding
    # padding). pad_overhead shows how much of the array's time went into
    # multiplying zeros -- the efficiency the host orchestration could reclaim.
    gflops = (2.0 * C_out * C_in * (H * W)) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s")
    print(
        f"Padded M/K/N = {operator.M}/{operator.K}/{operator.N} "
        f"(logical {operator.M_raw}/{operator.K_raw}/{operator.N_raw}, "
        f"pad_overhead {operator.pad_overhead:.2f}x)\n"
    )

    assert not errors, "Test failed"
