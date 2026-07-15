# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Pointwise (1x1) 2D convolution operator.
#
# A 1x1 conv IS a GEMM over the channel dimension, so this operator does NOT
# re-implement any data movement or MLIR generation: it derives the GEMM
# dimensions from the conv parameters and reuses the whole-array GEMM design
# (design.py) VERBATIM -- no design changes, no layout flags, no host reshaping.
#
# The trick is the mapping. Data arrives in the exact layout torch.nn.Conv2d
# uses (channels-first, NCHW). For a 1x1 conv the output is
#
#     y[c_out, p] = sum_c_in  w[c_out, c_in] * x[c_in, p]        (p = pixel index)
#
# which is a PLAIN row-major GEMM  Y = W @ X, if we let the WEIGHTS be GEMM's A
# and the ACTIVATION be GEMM's B (for batch == 1, where the pixel axis is the
# whole trailing H*W):
#
#   weight     w [C_out, C_in, 1, 1]  flat== [C_out, C_in]  -> A [M, K]  (row-major)
#   activation x [1, C_in, H, W]       flat== [C_in, H*W]    -> B [K, N]  (row-major)
#   output     y [1, C_out, H, W]      flat== [C_out, H*W]   -> C [M, N]  (row-major)
#
# So M = C_out, K = C_in, N = H*W (pixels). All three tensors are contiguous and
# row-major, so the flattened torch tensors are fed straight to the design with
# NO transposes -- which is what keeps the DMA access patterns legal (a strided
# bf16 transpose in the DMA is not 4-byte aligned and the hardware rejects it).

from dataclasses import dataclass, field
from typing import ClassVar, Dict

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

    # ---- Tiling (GEMM tiles over the MAPPED dims: m=C_out, k=C_in, n=pixels) -
    tile_m: int = 64
    tile_k: int = 64
    tile_n: int = 64

    # Inherited GEMM layout knobs. The torch.nn 1x1-conv mapping is a plain
    # row-major GEMM, so these stay False; they are kept only for parity with the
    # GEMM operator.
    b_col_maj: bool = False
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
        "b_col_maj": "bc",
        "c_col_maj": "cc",
    }

    def __post_init__(self):
        num_aie_rows = 4

        # ---- Map conv params -> GEMM dims (see module docstring) -----------
        #   M = C_out (output channels)   -> distributed across the 4 rows
        #   K = C_in  (input channels)    -> the reduction dimension
        #   N = H*W   (pixels)            -> distributed across the columns
        # These are the *logical* (unpadded) problem sizes. The array-facing
        # dims self.M/K/N are derived below by rounding up to the tiling
        # granularity; the host buffers are zero-padded to match (see pad_*).
        self.M_raw = self.C_out
        self.K_raw = self.C_in
        self.N_raw = self.batch * self.H * self.W

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

        # batch > 1 would put the batch axis outside C_in in the flattened
        # activation ([batch, C_in, H*W] != [C_in, batch*H*W]), breaking the
        # direct [K, N] mapping. Single image only, for now.
        if self.batch != 1:
            raise ValueError("Conv2d1x1 currently supports batch == 1 only")

        # ---- Pad the MAPPED dims up to the tiling granularity --------------
        # The shared GEMM design requires each mapped dim to tile evenly:
        #   M a multiple of tile_m*4 (the 4 rows), K a multiple of tile_k,
        #   N a multiple of tile_n*num_columns.
        # Real conv shapes rarely satisfy this (e.g. C_out=40, H*W=49), so we
        # round the array-facing dims UP and zero-pad the host buffers to match
        # (see the pad_* helpers, used by the caller/test). Padding is purely
        # host-side: the design and kernel are unchanged and never see a ragged
        # shape. Zero-padded channels/pixels contribute exactly 0, so the padded
        # result equals the logical result zero-extended -- slice it back with
        # unpad_output(). This mirrors GEMM's pad_A/pad_B approach.
        min_M = self.tile_m * num_aie_rows
        min_K = self.tile_k
        min_N = self.tile_n * self.num_aie_columns

        def _round_up(value, multiple):
            return ((value + multiple - 1) // multiple) * multiple

        self.M = _round_up(self.M_raw, min_M)
        self.K = _round_up(self.K_raw, min_K)
        self.N = _round_up(self.N_raw, min_N)

        MLIROperator.__init__(self, context=self.context)

    # ---- Host-side padding helpers -----------------------------------------
    # A 1x1 conv maps to GEMM as A = weights [M, K], B = activation [K, N],
    # C = output [M, N]. When the logical conv dims don't tile evenly the
    # operator rounds M/K/N up (see __post_init__); these helpers zero-pad the
    # flattened torch.nn tensors to the padded dims the design expects. Keeping
    # the pad here (not in the design) is what lets the GEMM design be reused
    # verbatim.
    def _pad2d(self, t, rows, cols):
        import torch

        r, c = t.shape
        if (r, c) == (rows, cols):
            return t.contiguous()
        out = torch.zeros((rows, cols), dtype=t.dtype)
        out[:r, :c] = t
        return out

    def pad_weight(self, w):
        """weight w [C_out, C_in, 1, 1] (or [C_out, C_in]) -> padded A [M, K]."""
        return self._pad2d(w.reshape(self.M_raw, self.K_raw), self.M, self.K)

    def pad_activation(self, x):
        """activation x [1, C_in, H, W] (or [C_in, H*W]) -> padded B [K, N]."""
        return self._pad2d(x.reshape(self.K_raw, self.N_raw), self.K, self.N)

    def pad_output(self, y):
        """golden output y [1, C_out, H, W] -> padded C [M, N] (zero-extended)."""
        return self._pad2d(y.reshape(self.M_raw, self.N_raw), self.M, self.N)

    def unpad_output(self, c):
        """padded C [M, N] -> logical output [C_out, H*W] (drops the padding)."""
        return c.reshape(self.M, self.N)[: self.M_raw, : self.N_raw]

    @property
    def pad_overhead(self):
        """Padded MACs / useful MACs -- 1.0 means no padding waste."""
        return (self.M * self.K * self.N) / (self.M_raw * self.K_raw * self.N_raw)

    @property
    def name(self) -> str:
        # The base `name` is built only from repr=True dataclass fields, but
        # use_scalar / prio_accuracy / emulate_bf16_mmul_with_bfp16 all change the
        # GENERATED MLIR (scalar vs vectorized kernel symbols, the f32-accumulate
        # path, and the mmul shape). If they are not in the name, toggling one on
        # the same shape silently reuses a stale .mlir/.xclbin from build/ and you
        # get wrong/old results. Fold them in so each flag combo is its own cache
        # entry.
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
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_matmul",
                (),
                {
                    "dev": aie_utils.get_current_device(),
                    "M": self.M,  # C_out
                    "K": self.K,  # C_in
                    "N": self.N,  # H*W (pixels)
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
        # A = weights [C_out, C_in], B = activation [C_in, H*W], C = output
        # [C_out, H*W] -- all row-major, matching the flattened torch tensors.
        return [
            AIERuntimeArgSpec("in", (self.M, self.K)),  # weights A == [C_out, C_in]
            AIERuntimeArgSpec(
                "in", (self.K, self.N) if not self.b_col_maj else (self.N, self.K)
            ),  # activation B == [C_in, H*W]
            AIERuntimeArgSpec(
                "out", (self.M, self.N) if not self.c_col_maj else (self.N, self.M)
            ),  # output C == [C_out, H*W]
        ]
