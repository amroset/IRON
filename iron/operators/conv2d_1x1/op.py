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

    # GEMM layout knobs. For the plain (normal) mapping the 1x1-conv is a
    # row-major GEMM, so these stay False. The cols->M swap (see __post_init__)
    # turns all three on to consume the weights/activation/output as-stored
    # (col-major) without any host transpose -- exactly how GEMM uses b/c_col_maj.
    a_col_maj: bool = False
    b_col_maj: bool = False
    c_col_maj: bool = False

    # cols->M axis swap. None = auto (swap tall layers, C_out > pixels, when the
    # column count can support it); True/False force it on/off. Tall layers map
    # C_out onto the 8-wide column axis instead of the few pixels, filling the
    # array. See OPTIMIZATION.md.
    swap_mn: object = field(default=None)

    num_aie_columns: int = field(default=8)
    emulate_bf16_mmul_with_bfp16: bool = field(default=True, repr=False)
    prio_accuracy: bool = field(default=False, repr=False)
    round_conv_even: bool = field(default=True, repr=False)
    dtype_in: str = field(default="bf16", repr=False)
    dtype_out: str = field(default="bf16", repr=False)
    use_scalar: bool = field(default=False, repr=False)
    separate_c_tiles: bool = field(default=False, repr=False)
    trace_size: int = field(default=0, repr=False)
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
        # These are the *logical* (unpadded) problem sizes, PER IMAGE. The
        # array-facing dims self.M/K/N are derived below by rounding up to the
        # tiling granularity; the host buffers are zero-padded to match (see
        # pad_*). N is one image's pixel count (H*W), NOT batch*H*W: a batch is
        # run as `batch` per-image dispatches of this same program (see below and
        # image_operands()).
        self.M_raw = self.C_out
        self.K_raw = self.C_in
        self.N_raw = self.H * self.W

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

        # batch > 1: the batch axis sits OUTSIDE C_in in the stored activation
        # ([batch, C_in, H*W]), so a batch is NOT one big GEMM ([C_in, batch*H*W]
        # would need the channels outermost). Instead it is `batch` independent
        # 1x1 convs that all share the SAME weights: Y_b = W . X_b. This program
        # is per-image (dims use H*W above); a caller runs it once per image,
        # reusing the compiled program -- see image_operands(). (Fusing the batch
        # into a single dispatch, gemv/transpose-style, is a possible future
        # optimization but interacts with the cols->M swap; per-image dispatch is
        # mapping-agnostic and reuses the design untouched.)
        if self.batch < 1:
            raise ValueError(f"batch ({self.batch}) must be >= 1")

        # ---- cols->M axis swap decision -----------------------------------
        # The array is 4 rows x n_aie_columns cols. The plain mapping puts
        # C_out on the 4 rows and pixels on the columns; for TALL layers
        # (C_out > pixels) that wastes the wide column axis on a few pixels. The
        # swap maps C_out onto the columns and pixels onto the rows. It feeds the
        # activation as the A (rows) operand col-major and the weights/output as
        # B/C col-major -- all as-stored, NO host transpose -- exactly how GEMM
        # uses b/c_col_maj. The col-major A row-distribution only lines up with
        # >= n_aie_rows columns (n_A_tiles_per_shim == 1), so the swap is a
        # multi-column path. See OPTIMIZATION.md.
        if self.swap_mn is not None:
            want_swap = bool(self.swap_mn)
        else:
            # Auto: swap only when the plain mapping leaves the columns STARVED.
            # Columns serve pixels (N) normally, so useful columns are
            # ceil(pixels / tile_n). The swap (columns serve C_out) only pays off
            # when the plain mapping uses at most half the columns AND C_out
            # fills more of them -- otherwise the swap's extra in-kernel
            # transposes cost more than the occupancy they buy. Measured crossover
            # at cols=8: N<=196 gains 1.1-5.6x, N=400 already fills the columns
            # and regresses (0.79x). See OPTIMIZATION.md.
            def _ceil_div(a, b):
                return (a + b - 1) // b

            normal_cols = _ceil_div(self.N_raw, self.tile_n)
            swap_cols = _ceil_div(self.C_out, self.tile_n)
            want_swap = (
                normal_cols * 2 <= self.num_aie_columns and swap_cols > normal_cols
            )
        if want_swap and self.num_aie_columns < num_aie_rows:
            if self.swap_mn:  # explicitly requested but unsupported at this col count
                raise ValueError(
                    "cols->M swap (swap_mn=True) requires num_aie_columns >= "
                    f"{num_aie_rows}; got {self.num_aie_columns}"
                )
            want_swap = False  # auto: fall back to the plain mapping for few columns
        self.swap = want_swap
        if self.swap:
            self.a_col_maj = self.b_col_maj = self.c_col_maj = True

        # ---- Pad each CONV axis up to its tiling granularity ---------------
        # Padding is purely host-side and layout-preserving (GEMM's pad_A/pad_B
        # approach): every buffer keeps its natural torch storage order and is
        # only zero-extended, so the padded result equals the logical result
        # zero-extended (slice back with unpad_output). Which tiling multiple a
        # conv axis rounds up to depends on the mapping:
        #   reduction K_hat = C_in            -> multiple of tile_k     (both modes)
        #   rows   M_hat = C_out | pixels      -> multiple of tile_m * n_aie_rows
        #   cols   N_hat = pixels | C_out      -> multiple of tile_n * n_aie_columns
        def _round_up(value, multiple):
            return ((value + multiple - 1) // multiple) * multiple

        rows_mult = self.tile_m * num_aie_rows
        cols_mult = self.tile_n * self.num_aie_columns
        self.C_in_pad = _round_up(self.C_in, self.tile_k)
        if self.swap:
            self.pix_pad = _round_up(self.N_raw, rows_mult)  # pixels -> rows
            self.cout_pad = _round_up(self.C_out, cols_mult)  # C_out  -> cols
        else:
            self.cout_pad = _round_up(self.C_out, rows_mult)  # C_out  -> rows
            self.pix_pad = _round_up(self.N_raw, cols_mult)  # pixels -> cols

        # Design (array-facing) dims: M = rows, K = reduction, N = cols.
        self.K = self.C_in_pad
        if self.swap:
            self.M, self.N = self.pix_pad, self.cout_pad
        else:
            self.M, self.N = self.cout_pad, self.pix_pad

        MLIROperator.__init__(self, context=self.context)

    # ---- Host-side padding helpers -----------------------------------------
    # Every buffer is zero-padded IN ITS NATURAL torch storage order and never
    # transposed on the host (GEMM's pad_A/pad_B approach). The three conv
    # tensors -- weight [C_out, C_in], activation [C_in, pixels], output
    # [C_out, pixels] -- have the SAME bytes in both mappings; only which design
    # operand slot they occupy (see input_operands) and the col-major flags
    # change. The kernel transposes any col-major operand in-tile, so the DMA
    # never sees an illegal (non-4-byte-aligned) bf16 transpose.
    def _pad2d(self, t, rows, cols):
        import torch

        r, c = t.shape
        if (r, c) == (rows, cols):
            return t.contiguous()
        out = torch.zeros((rows, cols), dtype=t.dtype)
        out[:r, :c] = t
        return out

    def pad_weight(self, w):
        """weight [C_out, C_in, 1, 1] (or [C_out, C_in]) -> [cout_pad, C_in_pad],
        stored as-is. GEMM's A (rows) in the plain mapping, GEMM's B (cols,
        col-major) under the swap -- same bytes either way."""
        return self._pad2d(w.reshape(self.C_out, self.C_in), self.cout_pad, self.C_in_pad)

    def pad_activation(self, x):
        """activation [1, C_in, H, W] (or [C_in, H*W]) -> [C_in_pad, pix_pad],
        stored as-is (NCHW). GEMM's B (cols) plain, GEMM's A (rows, col-major)
        under the swap."""
        return self._pad2d(x.reshape(self.C_in, self.N_raw), self.C_in_pad, self.pix_pad)

    def pad_output(self, y):
        """golden output [1, C_out, H, W] -> [cout_pad, pix_pad], stored as-is
        (NCHW). GEMM's C: row-major plain, col-major under the swap."""
        return self._pad2d(y.reshape(self.C_out, self.N_raw), self.cout_pad, self.pix_pad)

    def unpad_output(self, c):
        """padded output [cout_pad, pix_pad] -> logical [C_out, H*W]. The output
        is stored [C_out, pixels] in BOTH modes, so the slice is mode-independent."""
        return c.reshape(self.cout_pad, self.pix_pad)[: self.C_out, : self.N_raw]

    def input_operands(self, w, x):
        """The (rows, cols) input buffers FLATTENED in the design's arg order
        (first arg = rows operand A, second = cols operand B). The swap sends the
        activation to A and the weights to B; the plain mapping does the reverse.
        `x` is a SINGLE image ([1, C_in, H, W] or [C_in, H*W])."""
        weight = self.pad_weight(w).flatten()
        activation = self.pad_activation(x).flatten()
        return (activation, weight) if self.swap else (weight, activation)

    def image_operands(self, w, x):
        """Yield the per-image (rows, cols) input buffers for a batched activation
        `x` ([batch, C_in, H, W]). The program is per-image and the weights are
        shared across the batch, so a caller runs one dispatch per yielded pair
        (see test.py). For batch == 1 this yields a single pair."""
        x = x.reshape(self.batch, self.C_in, self.N_raw)
        for b in range(self.batch):
            yield self.input_operands(w, x[b])

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
            # only when tracing, so existing artifact names are unchanged
            + (f"_tr{self.trace_size}" if self.trace_size else "")
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
            f"_{int(self.a_col_maj)}_{int(self.b_col_maj)}_{int(self.c_col_maj)}"
            f"{self._kernel_flags_suffix}.o"
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
                    # Design dims: M = rows, K = reduction, N = cols. Plain
                    # mapping M=C_out, N=pixels; swap flips them (M=pixels,
                    # N=C_out). K = C_in either way.
                    "M": self.M,
                    "K": self.K,
                    "N": self.N,
                    "m": self.tile_m,
                    "k": self.tile_k,
                    "n": self.tile_n,
                    "n_aie_cols": self.num_aie_columns,
                    "dtype_in_str": self.dtype_in,
                    "dtype_out_str": self.dtype_out,
                    "a_col_maj": int(self.a_col_maj),
                    "b_col_maj": int(self.b_col_maj),
                    "c_col_maj": int(self.c_col_maj),
                    "use_scalar": self.use_scalar,
                    "emulate_bf16_mmul_with_bfp16": self.emulate_bf16_mmul_with_bfp16,
                    "prio_accuracy": self.prio_accuracy,
                    "separate_c_tiles": int(self.separate_c_tiles),
                    "trace_size": self.trace_size,
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
        if self.a_col_maj:
            kernel_flags.append("-DA_COL_MAJ")
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
        # Slots follow the design's sequence args and input_operands() order:
        #   in 0 = rows operand A, in 1 = cols operand B, out = C.
        # Shapes follow the col-major flags (buffers are fed as-stored; the
        # kernel transposes col-major operands in-tile).
        #   plain: A = weights [C_out,C_in], B = activation [C_in,H*W],
        #          C = output [C_out,H*W]  (all row-major)
        #   swap:  A = activation [C_in,H*W] (a_col_maj), B = weights [C_out,C_in]
        #          (b_col_maj), C = output [C_out,H*W] (c_col_maj)
        return [
            AIERuntimeArgSpec(
                "in", (self.M, self.K) if not self.a_col_maj else (self.K, self.M)
            ),  # rows operand A
            AIERuntimeArgSpec(
                "in", (self.K, self.N) if not self.b_col_maj else (self.N, self.K)
            ),  # cols operand B
            AIERuntimeArgSpec(
                "out", (self.M, self.N) if not self.c_col_maj else (self.N, self.M)
            ),  # output C
        ]
