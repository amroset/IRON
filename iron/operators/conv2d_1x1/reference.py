# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Golden reference for a POINTWISE (1x1) 2D convolution.
#
# A 1x1 conv is a GEMM over channels: with M = batch*H*W, K = C_in, N = C_out,
#     C[M, N] = A[M, K] @ B[K, N]
# This mirrors the GEMM reference's construction EXACTLY (build B in the [K, N]
# frame, fill the identity there, then apply the b_col_maj / c_col_maj
# transposes afterwards). Doing it in the [K, N] frame is what makes the
# non-square case (C_in != C_out) correct.

import torch
from iron.common.test_utils import torch_dtype_map


def generate_golden_reference(
    batch: int,
    H: int,
    W: int,
    C_in: int,
    C_out: int,
    dtype="bf16",
    seed=42,
    b_col_maj=True,      # weights [C_out, C_in] (framework/OIHW) -> col-major B
    c_col_maj=False,
):
    torch.manual_seed(seed)
    val_range = 4
    dtype_torch = torch_dtype_map[dtype]

    # GEMM mapping
    M = batch * H * W
    K = C_in
    N = C_out

    # ---- Inputs, built in the [M, K] / [K, N] frame (same as GEMM) ----------
    input_a = torch.randn(M, K, dtype=dtype_torch) * val_range
    input_b_full = torch.rand(K, N, dtype=dtype_torch) * val_range   # [K, N]

    DEBUG_IDENTITY = False
    if DEBUG_IDENTITY:
        # Debug: identity B in the [K, N] frame, so output == input channelwise.
        # Built here (not as eye(C_out, C_in)) so it is correct when K != N.
        input_b_full = torch.zeros(K, N, dtype=dtype_torch)
        diag_dim = min(K, N)
        input_b_full[:diag_dim, :diag_dim] = torch.eye(diag_dim, dtype=dtype_torch)

    # ---- Golden output in the [M, N] frame, BEFORE any transpose ------------
    output_full = torch.matmul(input_a.float(), input_b_full.float()).to(dtype_torch)

    # ---- Apply layout transposes exactly like GEMM -------------------------
    if b_col_maj:
        input_b_full = input_b_full.T          # -> [N, K]
    if c_col_maj:
        output_full = output_full.T            # -> [N, M]

    # partition_N == 1: single-element lists, matching the GEMM contract.
    input_b = [input_b_full.contiguous()]
    output = [output_full.contiguous()]

    return {"input": input_a, "input_b": input_b, "output": output}