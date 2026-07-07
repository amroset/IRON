#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test for the pointwise (1x1) conv operator. Pointwise conv IS a GEMM, so this
# mirrors test_gemm.py's partition_N == 1 path: build reference + operator with
# the SAME layout flags, flatten the golden tensors straight into the buffer
# dicts, and call run_test. M padding is applied ONLY when the operator's padded
# M differs from the real pixel count.

import pytest
import aie.utils as aie_utils
import torch

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
        (    1,  8,  8,  512,   256,        1,        "oihw",    False, 16, 64, 64),  # tiny, GEMM-shaped
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

    # Same layout flags flow to BOTH reference and operator (as GEMM does).
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
        use_scalar=False,      # vectorized aie::mmul path (proven, matches GEMM)
        prio_accuracy=True,    # bf16 in, f32 accumulate -> narrow to bf16 (accurate)
        emulate_bf16_mmul_with_bfp16=False,
        context=aie_context,
    )

    A = golden_ref["input"]            # [M_real, C_in]
    B = golden_ref["input_b"][0]       # [C_out, C_in] (b_col_maj) or [C_in, C_out]
    C = golden_ref["output"][0]        # [M_real, C_out] (c_col_maj -> [C_out, M_real])

    M_real = A.shape[0]

    # Pad ONLY if the operator's (padded) M exceeds the real pixel count.
    if operator.M != M_real:
        A_in = torch.zeros(operator.M, operator.C_in, dtype=A.dtype)
        A_in[:M_real] = A
        if c_col_maj:
            C_ref = torch.zeros(operator.N, operator.M, dtype=C.dtype)
            C_ref[:, :M_real] = C
        else:
            C_ref = torch.zeros(operator.M, operator.N, dtype=C.dtype)
            C_ref[:M_real] = C
    else:
        A_in = A
        C_ref = C

    input_buffers = {"A": A_in.flatten(), "B": B.flatten()}
    output_buffers = {"C": C_ref.flatten()}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.005, abs_tol=0.005
    )

    # Count real work (not padded) for the throughput figure.
    gflops = (2.0 * operator.M_real * C_in * C_out) / (latency_us * 1e-6) / 1e9

    print(f"\nLatency (us): {latency_us:.1f}")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s")
    print(f"Throughput: {gflops:.6e} GFLOP/s\n")

    assert not errors, "Test failed"