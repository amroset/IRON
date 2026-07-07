# SPDX-License-Identifier: Apache-2.0
#
# Host orchestration for a POINTWISE (1x1) 2D convolution.
#
# A 1x1 conv is a GEMM over the channel dimension, so this file does NOT
# re-implement any data movement. It derives the GEMM dims from the conv
# parameters, handles the NHWC reshape contract + M padding, and then calls the
# existing `my_matmul` design. The ObjectFIFOs, broadcast/distribute/join,
# ping-pong scheduling and drain logic are reused verbatim.
#
#   act  [batch, H, W, C_in]  --reshape(NHWC)-->  A [M, C_in]     M = batch*H*W
#   wts  [C_out, C_in]  (framework OIHW, 1x1)  -> B via b_col_maj=1 (see below)
#   out  [M, C_out]           --reshape------->  [batch, H, W, C_out]
#
# GEMM mapping:  M = batch*H*W (pixels),  K = C_in (reduction),  N = C_out.

import argparse
from pathlib import Path

# The GEMM design must be importable. If your matmul file is e.g. `matmul.py`,
# this import works as-is; otherwise adjust the module name.
from matmul import my_matmul  # noqa: E402


def ceildiv(a, b):
    return (a + b - 1) // b


def conv2d_1x1(
    dev,
    batch,
    H,
    W,
    C_in,
    C_out,
    m,
    k,
    n,
    n_aie_cols,
    dtype_in_str,
    dtype_out_str,
    weight_layout,       # "oihw" (framework, [C_out, C_in]) or "kn" ([C_in, C_out])
    use_scalar,
    emulate_bf16_mmul_with_bfp16,
    prio_accuracy,
    separate_c_tiles,
    trace_size,
    kernel_object,       # compiled conv kernel object, e.g. "conv2d_1x1.o"
    generate_taps=False,
):
    n_aie_rows = 4

    # ---- 1. Derive GEMM dimensions from conv parameters --------------------
    M = batch * H * W          # pixels  -> GEMM rows
    K = C_in                   # input channels -> reduction
    N = C_out                  # output channels -> GEMM cols

    # ---- 2. Weight layout -> b_col_maj -------------------------------------
    # Framework 1x1 weights are [C_out, C_in] (OIHW collapsed). GEMM-B wants
    # [K, N] = [C_in, C_out]. The [C_out, C_in] layout is exactly B read as
    # [N, K], which is what the b_col_maj path expects -> set b_col_maj = 1.
    # If you pre-transpose weights to [C_in, C_out] yourself, pass "kn".
    if weight_layout == "oihw":
        b_col_maj = 1
    elif weight_layout == "kn":
        b_col_maj = 0
    else:
        raise ValueError(f"weight_layout must be 'oihw' or 'kn', got {weight_layout}")

    c_col_maj = 0  # keep output row-major [M, C_out] (NHWC-friendly)

    # ---- 3. Divisibility checks + M padding --------------------------------
    # These mirror the asserts inside my_matmul, but we surface them here with
    # conv-meaningful messages and fix the one that conv routinely violates (M).
    mem_tile_n = n * n_aie_cols
    mem_tile_m_C = m * n_aie_rows

    assert K % k == 0, (
        f"C_in={C_in} must be divisible by k={k} (reduction tiling). "
        f"Pick k as a divisor of C_in (k=C_in gives K_div_k=1)."
    )
    assert N % mem_tile_n == 0, (
        f"C_out={C_out} must be divisible by n*n_aie_cols={mem_tile_n}. "
        f"Reduce n or n_aie_cols, or pad C_out."
    )

    # M = batch*H*W rarely divides mem_tile_m_C. Pad up; the extra pixel-rows
    # are dummy and sliced off host-side after the run.
    M_padded = ceildiv(M, mem_tile_m_C) * mem_tile_m_C
    if M_padded != M:
        pad_rows = M_padded - M
        print(
            f"[conv2d_1x1] padding M {M} -> {M_padded} "
            f"(+{pad_rows} dummy pixel-rows) to satisfy M % {mem_tile_m_C} == 0. "
            f"Slice out[:{M}] after the run."
        )
    M = M_padded

    # Column-utilization sanity check (tall-skinny-N pitfall).
    if N < mem_tile_n:
        print(
            f"[conv2d_1x1] WARNING: C_out={C_out} < mem_tile_n={mem_tile_n}: "
            f"some AIE columns will be idle. Consider smaller n / n_aie_cols."
        )

    # ---- 4. Hand off to the reused GEMM design -----------------------------
    return my_matmul(
        dev,
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        dtype_in_str,
        dtype_out_str,
        b_col_maj,
        c_col_maj,
        use_scalar,
        emulate_bf16_mmul_with_bfp16,
        prio_accuracy,
        separate_c_tiles,
        trace_size,
        kernel_object,   # -> Kernel names still matmul_scalar_bf16_bf16 / zero_scalar_bf16
        "",
        generate_taps,
    )


def main():
    p = argparse.ArgumentParser(
        prog="AIE Pointwise (1x1) Conv MLIR Design",
        description="Maps a 1x1 convolution onto the GEMM (my_matmul) design.",
    )
    p.add_argument("--dev", type=str, choices=["npu1", "npu2"], default="npu2")
    # Conv problem shape
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("-H", type=int, default=56)
    p.add_argument("-W", type=int, default=56)
    p.add_argument("--c-in", type=int, default=64)
    p.add_argument("--c-out", type=int, default=128)
    # Tiling (same meaning as GEMM: m=pixels tile, k=C_in tile, n=C_out tile)
    p.add_argument("-m", type=int, default=64)
    p.add_argument("-k", type=int, default=64)
    p.add_argument("-n", type=int, default=32)
    p.add_argument("--n-aie-cols", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument("--weight-layout", type=str, choices=["oihw", "kn"], default="oihw")
    p.add_argument("--scalar", type=int, choices=[0, 1], default=1)  # scalar oracle first
    p.add_argument("--dtype_in", type=str, choices=["bf16"], default="bf16")
    p.add_argument("--dtype_out", type=str, choices=["bf16", "f32"], default="bf16")
    p.add_argument("--emulate-bf16-mmul-with-bfp16", action="store_true", default=False)
    p.add_argument("--prio-accuracy", action="store_true", default=False)
    p.add_argument("--separate-c-tiles", type=int, choices=[0, 1], default=0)
    p.add_argument("--trace_size", type=int, default=0)
    p.add_argument("--kernel-object", type=str, default="conv2d_1x1.o")
    p.add_argument("--generate-taps", action="store_true")
    p.add_argument("--output-file-path", "-o", type=str)

    args = p.parse_args()

    module = conv2d_1x1(
        args.dev,
        args.batch,
        args.H,
        args.W,
        args.c_in,
        args.c_out,
        args.m,
        args.k,
        args.n,
        args.n_aie_cols,
        args.dtype_in,
        args.dtype_out,
        args.weight_layout,
        args.scalar,
        args.emulate_bf16_mmul_with_bfp16,
        args.prio_accuracy,
        args.separate_c_tiles,
        args.trace_size,
        args.kernel_object,
        args.generate_taps,
    )

    if args.generate_taps:
        return module
    with open(Path(args.output_file_path), "w") as f:
        f.write(str(module))


if __name__ == "__main__":
    main()