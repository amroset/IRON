#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test for the pointwise (1x1) conv operator. Because pointwise conv is a GEMM,
# this mirrors test_gemm.py: build the operator, run, compare against an
# independent torch-conv2d golden. Shapes are given in CONV terms
# (batch, H, W, C_in, C_out); M padding is handled inside the operator.

import numpy as np
import pytest
import aie.utils as aie_utils

from iron.operators.conv2d_1x1.op import Conv2d1x1
from iron.operators.conv2d_1x1.reference import generate_golden_reference
from iron.common.test_utils import run_test


def get_params():
    dev = aie_utils.get_current_device()
    max_aie_columns = dev.cols
    device_type = dev.resolve().name
    # fmt: off
    #  batch,  H,  W, C_in, C_out, num_cols, weight_layout, c_col_maj,  m,  k,  n
    regular_params = [
        (    1, 56, 56,   64,   128,        4,        "oihw",    False, 64, 64, 32),
        (    1, 28, 28,  128,   256,        4,        "oihw",    False, 64, 64, 64),
        (    1, 14, 14,  256,   512,        8,        "oihw",    False, 64, 64, 64),
        (    1, 56, 56,   64,   128,        4,          "kn",    False, 64, 64, 32),
        (    4, 32, 32,   64,    64,        4,        "oihw",    False, 64, 64, 16),
        (    1,  8,  8,   16,    16,        1,        "oihw",    False, 16, 16, 16),  # tiny (scalar-friendly)
    ]
    # fmt: on

    params = []
    for p in regular_params:
        (batch, H, W, C_in, C_out, num_cols,
         weight_layout, c_col_maj, m, k, n) = p
        if num_cols > max_aie_columns:
            continue
        if device_type == "npu1" and m < 16:
            continue
        params.append(pytest.param(*p))
    return params


@pytest.mark.metrics(
    Latency=r"Latency \(us\): (?P<value>[\d\.]+)",
    Bandwidth=r"Effective Bandwidth: (?P<value>[\d\.e\+-]+) GB/s",
    Throughput=r"Throughput: (?P<value>[\d\.e\+-]+) GFLOP/s",
)
@pytest.mark.parametrize(
    "batch,H,W,C_in,C_out,num_cols,weight_layout,c_col_maj,m,k,n",
    get_params(),
)
def test_conv2d_1x1(
    batch, H, W, C_in, C_out, num_cols, weight_layout, c_col_maj, m, k, n,
    aie_context,
):
    b_col_maj = weight_layout == "oihw"

    golden_ref = generate_golden_reference(
        batch=batch, H=H, W=W, C_in=C_in, C_out=C_out,
        b_col_maj=b_col_maj, c_col_maj=c_col_maj,
    )

    operator = Conv2d1x1(
        batch=batch, H=H, W=W, C_in=C_in, C_out=C_out,
        tile_m=m, tile_k=k, tile_n=n,
        num_aie_columns=num_cols,
        weight_layout=weight_layout,
        c_col_maj=c_col_maj,
        prio_accuracy=True,
        emulate_bf16_mmul_with_bfp16=False,
        context=aie_context,
    )

    # The operator pads M (pixels) up to a multiple of tile_m*4. Pad the
    # activation and the expected output to match, so shapes line up; the
    # padded rows are zeros in and zeros out.
    A = golden_ref["input"]            # [M_real, C_in]  (torch bf16)
    B = golden_ref["input_b"][0]
    C = golden_ref["output"][0]        # [M_real, C_out] (or transposed)

    A_np = A.contiguous().view(torch_uint16(A)).numpy().view(bf16_np())
    A_pad = operator.pad_activation(A_np)                      # [M, C_in]

    # Pad the golden C the same way for a shape-matched compare.
    C_np = C.contiguous().view(torch_uint16(C)).numpy().view(bf16_np())
    if c_col_maj:
        C_pad = np.zeros((operator.N, operator.M), dtype=C_np.dtype)
        C_pad[:, : operator.M_real] = C_np
    else:
        C_pad = np.zeros((operator.M, operator.N), dtype=C_np.dtype)
        C_pad[: operator.M_real, :] = C_np

    B_np = B.contiguous().view(torch_uint16(B)).numpy().view(bf16_np())

    input_buffers = {"A": A_pad.flatten(), "B": B_np.flatten()}
    output_buffers = {"C": C_pad.flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
    )

    total_N = C_out
    M_flops = operator.M_real  # count real work, not padded
    gflops = (2.0 * M_flops * C_in * total_N) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    assert not errors, "Test failed"


# --- small dtype helpers (bf16 numpy round-trip) ----------------------------
def torch_uint16(t):
    import torch
    return torch.uint16


def bf16_np():
    import ml_dtypes
    return ml_dtypes.bfloat16