# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Golden reference for a POINTWISE (1x1) 2D convolution.
#
# Unlike a matmul reference, this uses torch's native conv2d as an INDEPENDENT
# oracle. That way the test validates the whole conv->GEMM mapping (NHWC
# reshape + weight transpose + output reshape), not merely "a matmul equals a
# matmul". The returned buffers are already in the exact layout the hardware
# expects, matching the GEMM reference's {input, input_b, output} contract.

import torch
import torch.nn.functional as F

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

    # --- Native conv inputs (torch uses NCHW) -------------------------------
    act_nchw = torch.randn(batch, C_in, H, W, dtype=dtype_torch) * val_range
    # 1x1 weight: [C_out, C_in, 1, 1]
    weight = torch.rand(C_out, C_in, 1, 1, dtype=dtype_torch) * val_range

    # --- Independent oracle: real conv2d ------------------------------------
    out_nchw = F.conv2d(act_nchw, weight)          # [batch, C_out, H, W]

    # --- Reshape to the hardware's GEMM layout ------------------------------
    # A = activation NHWC -> [M, C_in], M = batch*H*W
    A = act_nchw.permute(0, 2, 3, 1).reshape(batch * H * W, C_in).contiguous()

    # C = output NHWC -> [M, C_out]
    C = out_nchw.permute(0, 2, 3, 1).reshape(batch * H * W, C_out).contiguous()
    if c_col_maj:
        C = C.T.contiguous()

    # B = weights. Squeeze 1x1 spatial dims -> [C_out, C_in].
    #   b_col_maj=True  : hardware reads B as [N, K] = [C_out, C_in]  (as-is)
    #   b_col_maj=False : hardware wants row-major [K, N] = [C_in, C_out]
    W2d = weight.reshape(C_out, C_in).contiguous()      # [C_out, C_in]
    if b_col_maj:
        B = W2d
    else:
        B = W2d.T.contiguous()                          # [C_in, C_out]

    # Match the GEMM reference's list-wrapped structure (partition_N == 1).
    return {"input": A, "input_b": [B], "output": [C]}