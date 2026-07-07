# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pointwise (1x1) 2D convolution operator.
#
# A 1x1 conv IS a GEMM over the channel dimension, so this operator does NOT
# re-implement any data movement or MLIR generation: it derives the GEMM
# dimensions from the conv parameters and reuses the existing `my_matmul`
# design (design.py) verbatim.
#
#   act  [batch, H, W, C_in]  --reshape(NHWC)-->  A [M, C_in]   M = batch*H*W
#   wts  [C_out, C_in]        (framework 1x1)  -> B (b_col_maj=1) or transpose
#   out  [M, C_out]           --reshape------->  [batch, H, W, C_out]
#
# GEMM mapping:  M = batch*H*W,  K = C_in,  N = C_out.

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
from iron.common.device_utils import get_kernel_dir
import aie.utils as aie_utils


@dataclass
class Conv2d1x1(MLIROperator):
    """AIE-accelerated pointwise (1x1) 2D convolution, mapped onto the GEMM design."""

    # ---- Conv problem shape ------------------------------------------------
    batch: int
    H: int
    W: int
    C_in: int
    C_out: int

    # ---- Tiling (identical meaning to GEMM: m=pixels, k=C_in, n=C_out) ------
    tile_m: int = 64
    tile_k: int = 64
    tile_n: int = 64

    # Framework 1x1 weights are [C_out, C_in] (OIHW collapsed) == GEMM-B read
    # column-major. Use weight_layout="oihw" for that; "kn" if you pre-transpose
    # weights to [C_in, C_out] yourself.
    weight_layout: str = "oihw"
    c_col_maj: bool = False

    num_aie_columns: int = field(default=8)
    emulate_bf16_mmul_with_bfp16: bool = field(default=True, repr=False)
    prio_accuracy: bool = field(default=False, repr=False)
    round_conv_even: bool = field(default=True, repr=False)
    dtype_in: str = field(default="bf16", repr=False)
    dtype_out: str = field(default="bf16", repr=False)
    use_scalar: bool = field(default=False, repr=False)
    separate_c_tiles: bool = field(default=False, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "tile_m": "tm",
        "tile_k": "tk",
        "tile_n": "tn",
        "c_col_maj": "cc",
    }

    def __post_init__(self):
        num_aie_rows = 4

        # ---- Derive GEMM dims from conv params -----------------------------
        self.M_real = self.batch * self.H * self.W  # true pixel count (pre-pad)
        self.K = self.C_in
        self.N = self.C_out

        # weight_layout -> b_col_maj
        if self.weight_layout == "oihw":
            self.b_col_maj = True
        elif self.weight_layout == "kn":
            self.b_col_maj = False
        else:
            raise ValueError(
                f"weight_layout must be 'oihw' or 'kn', got {self.weight_layout}"
            )

        # ---- Tile-size floors (same as GEMM) -------------------------------
        if self.emulate_bf16_mmul_with_bfp16:
            min_tile_m, min_tile_k, min_tile_n = 8, 8, 8
        else:
            min_tile_m, min_tile_k, min_tile_n = 4, 8, 8
        if self.tile_m < min_tile_m:
            raise ValueError(f"tile_m ({self.tile_m}) must be >= {min_tile_m}")
        if self.tile_k < min_tile_k:
            raise ValueError(f"tile_k ({self.tile_k}) must be >= {min_tile_k}")
        if self.tile_n < min_tile_n:
            raise ValueError(f"tile_n ({self.tile_n}) must be >= {min_tile_n}")

        # ---- Divisibility on channels; PAD on pixels -----------------------
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns

        if self.K % min_K != 0:
            raise ValueError(
                f"C_in ({self.C_in}) must be a multiple of tile_k ({self.tile_k})"
            )
        if self.N % min_N != 0:
            raise ValueError(
                f"C_out ({self.C_out}) must be a multiple of "
                f"tile_n*num_aie_columns ({min_N})"
            )
        if self.N < min_N:
            # (unreachable given the check above, but explicit about the pitfall)
            raise ValueError(
                f"C_out ({self.C_out}) < tile_n*num_aie_columns ({min_N}): "
                f"columns would be idle."
            )

        # M = batch*H*W rarely divides min_M -> pad up. self.M is the operator
        # (hardware) dimension; self.M_real is used to slice the output back.
        self.M = ((self.M_real + min_M - 1) // min_M) * min_M

        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        # The base `name` is built only from repr=True dataclass fields, but
        # use_scalar / prio_accuracy / emulate_bf16_mmul_with_bfp16 all change the
        # GENERATED MLIR (scalar vs vectorized kernel symbols, the f32-accumulate
        # path, and the mmul shape). If they are not in the name, toggling one on
        # the same shape silently reuses a stale .mlir/.xclbin from build/ and you
        # get wrong/old results. Fold them in so each flag combo is its own cache
        # entry. (The GEMM operator has the same latent issue; it just never varies
        # these flags per-shape in its test suite.)
        return (
            f"{super().name}"
            f"_us{int(self.use_scalar)}"
            f"_pa{int(self.prio_accuracy)}"
            f"_em{int(self.emulate_bf16_mmul_with_bfp16)}"
        )

    @property
    def _kernel_flags_suffix(self):
        return (
            f"_{int(self.prio_accuracy)}_{int(self.emulate_bf16_mmul_with_bfp16)}"
            f"_{int(self.round_conv_even)}"
        )

    def _kernel_object_name(self):
        return (
            f"conv1x1_{self.tile_m}x{self.tile_k}x{self.tile_n}"
            f"_{int(self.b_col_maj)}_{int(self.c_col_maj)}{self._kernel_flags_suffix}.o"
        )

    def get_mlir_artifact(self):
        # design.py exposes `my_matmul` (the reused GEMM design). Place/copy or
        # import the GEMM design.py into this operator's dir.
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matmul",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "M": self.M,          # padded pixel count
                    "K": self.K,          # C_in
                    "N": self.N,          # C_out
                    "m": self.tile_m,
                    "k": self.tile_k,
                    "n": self.tile_n,
                    "n_aie_cols": self.num_aie_columns,
                    "dtype_in_str": self.dtype_in,
                    "dtype_out_str": self.dtype_out,
                    "b_col_maj": int(self.b_col_maj),
                    "c_col_maj": int(self.c_col_maj),
                    "use_scalar": self.use_scalar,
                    "emulate_bf16_mmul_with_bfp16": self.emulate_bf16_mmul_with_bfp16,
                    "prio_accuracy": self.prio_accuracy,
                    "separate_c_tiles": int(self.separate_c_tiles),
                    "trace_size": 0,
                    "generate_taps": False,
                    "kernel_object": self._kernel_object_name(),
                },
            ),
        )

    def get_kernel_artifacts(self):
        base_dir = self.context.base_dir
        kernel_flags = [
            f"-DDIM_M={self.tile_m}",
            f"-DDIM_K={self.tile_k}",
            f"-DDIM_N={self.tile_n}",
        ]
        if self.prio_accuracy:
            kernel_flags.append("-Dbf16_f32_ONLY")
        else:
            kernel_flags.append("-Dbf16_bf16_ONLY")
        if self.round_conv_even:
            kernel_flags.append("-DROUND_CONV_EVEN")
        if self.emulate_bf16_mmul_with_bfp16:
            kernel_flags.append("-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16")
        if self.b_col_maj:
            kernel_flags.append("-DB_COL_MAJ")
        if self.c_col_maj:
            kernel_flags.append("-DC_COL_MAJ")

        kernel_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._kernel_object_name(),
                extra_flags=kernel_flags,
                dependencies=[
                    # Conv kernel source (keeps matmul_* symbol names).
                    SourceArtifact(
                        base_dir / "aie_kernels" / kernel_dir / "conv2d_1x1.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                "convert_copy.o",
                [
                    SourceArtifact(
                        base_dir / "aie_kernels" / "generic" / "convert_copy.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        # Hardware sees padded M.
        return [
            AIERuntimeArgSpec("in", (self.M, self.K)),  # activation A
            AIERuntimeArgSpec(
                "in", (self.N, self.K) if self.b_col_maj else (self.K, self.N)
            ),  # weights B
            AIERuntimeArgSpec(
                "out", (self.N, self.M) if self.c_col_maj else (self.M, self.N)
            ),  # output C
        ]

    # ---- Host-side data (un)padding helpers --------------------------------
    def pad_activation(self, act_np):
        """[batch, H, W, C_in] (NHWC) or [M_real, C_in] -> padded [M, C_in]."""
        if act_np.ndim == 4:
            b, h, w, c = act_np.shape
            act_np = act_np.reshape(b * h * w, c)
        M_real, C = act_np.shape
        if C != self.C_in:
            raise ValueError(f"C_in mismatch: got {C}, expected {self.C_in}")
        if M_real > self.M:
            raise ValueError(f"pixels ({M_real}) exceed padded M ({self.M})")
        if M_real == self.M:
            return act_np
        out = np.zeros((self.M, self.C_in), dtype=act_np.dtype)
        out[:M_real, :] = act_np
        return out

    def unpad_output(self, C_np):
        """Slice the padded output back to the real pixel count.

        Accepts flat [M, C_out] (or [C_out, M] if c_col_maj) and returns the
        first M_real rows; reshape to [batch, H, W, C_out] downstream as needed.
        """
        if self.c_col_maj:
            return C_np[:, : self.M_real]
        return C_np[: self.M_real, :]