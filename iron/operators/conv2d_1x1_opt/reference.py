# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Golden reference for a POINTWISE (1x1) 2D convolution.
#
# This reference is deliberately expressed in the SAME format that torch.nn
# expects, so the test exercises the real layout conversion end-to-end:
#
#   activation x : [batch, C_in, H, W]      (NCHW, torch channels-first)
#   weight     w : [C_out, C_in, 1, 1]      (OIHW, the torch.nn.Conv2d weight)
#   output     y : [batch, C_out, H, W]     (NCHW)  = F.conv2d(x, w)
#
# The GEMM-frame reshaping (NCHW -> [M, C_in], OIHW -> B, [M, C_out] -> NCHW) is
# NOT done here. That is the operator's job (op.py) and is precisely what we want
# to validate. The reference only knows torch.nn tensors.

import torch
import torch.nn.functional as F

from iron.common.test_utils import torch_dtype_map


def generate_golden_reference(
    batch: int,
    H: int,
    W: int,
    C_in: int,
    C_out: int,
    dtype: str = "bf16",
    seed: int = 42,
):
    """Build torch.nn-native inputs and the golden 1x1-conv output.

    Returns a dict with:
        x : [batch, C_in, H, W]   activation (NCHW)
        w : [C_out, C_in, 1, 1]   weight     (OIHW, 1x1 kernel)
        y : [batch, C_out, H, W]  golden output (NCHW) = conv2d(x, w)
    """
    torch.manual_seed(seed)
    val_range = 4
    dt = torch_dtype_map[dtype]

    # torch.nn-native tensors: channels-first activation, OIHW weight.
    x = torch.randn(batch, C_in, H, W, dtype=dt) * val_range
    w = torch.randn(C_out, C_in, 1, 1, dtype=dt) * val_range

    # Golden pointwise conv. Accumulate in f32 (the NPU path accumulates in f32
    # too when prio_accuracy is set) then narrow back to the working dtype.
    y = F.conv2d(x.float(), w.float()).to(dt)  # [batch, C_out, H, W]

    return {"x": x, "w": w, "y": y}
