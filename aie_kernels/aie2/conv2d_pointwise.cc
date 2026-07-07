// SPDX-License-Identifier: Apache-2.0
//
// A 1x1 convolution is a matmul over the channel dimension: for each spatial
// position (pixel) the output channel vector is  out[oc] = sum_ic act[ic]*W[ic,oc].
// Reshaping the NHWC activation to [n_pixels, c_in] makes this a plain GEMM:
//
//     out[n_pixels, c_out] = act[n_pixels, c_in] . W[c_in, c_out]
//
// This is the correctness ORACLE for the vectorized aie::mmul version: same
// data distribution, same tiling, same in-place K-reduction. Only the arithmetic
// inside each core differs (plain scalar loop vs. vectorized MAC).
//
// L1 tile contract (one core, one k-block):
//   act : [n_pixels, c_in]      row-major   (GEMM A tile)
//   wts : [c_in,     c_out]     row-major   (GEMM B tile; b_row_maj)
//         or [c_out,  c_in]                  (b_col_maj)
//   out : [n_pixels, c_out]     row-major   (GEMM C tile; c_row_maj)
//
// In-place accumulation: out is read-modify-written with '+='. The host zeroes
// the C accumulator once (see zero.cc) before the reduction, then calls this
// kernel once per k-block; the sum over all C_in slices is built across calls.

#define NOCPP

#include <stdio.h>
#include <stdlib.h>

#include "zero.cc"   // provides zero_scalar<> / zero_vectorized<> for accumulator init

#include <aie_api/aie.hpp>

// -----------------------------------------------------------------------------
// Scalar pointwise-conv (== scalar GEMM over channels)
//
// Accumulates in f32 and narrows to T_out at the store. For bf16 output this
// mirrors the hardware MAC's wide internal accumulation (accauto), so results
// bit-track the vectorized kernel instead of drifting due to per-step rounding.
// -----------------------------------------------------------------------------
template <typename T_in,
          typename T_out,
          int n_pixels,             // rowA : pixels (H*W chunk) in this L1 tile
          int c_in,                 // colA : input channels  (reduction dim)
          int c_out,                // colB : output channels
          bool b_row_maj = true,    // weights [c_in, c_out] vs [c_out, c_in]
          bool c_row_maj = true>
static inline void conv2d_1x1_scalar(T_in *act, T_in *wts, T_out *out)
{
    event0();
    for (int p = 0; p < n_pixels; p++) {
        for (int oc = 0; oc < c_out; oc++) {

            float acc = 0.0f;                       // wide accumulator (oracle fidelity)
            for (int ic = 0; ic < c_in; ic++) {
                float a = (float)act[p * c_in + ic];
                float w;
                if constexpr (b_row_maj) {
                    w = (float)wts[ic * c_out + oc]; // [c_in, c_out]
                } else {
                    w = (float)wts[ic + oc * c_in];  // [c_out, c_in]
                }
                acc += a * w;
            }

            T_out *o_ptr;
            if constexpr (c_row_maj) {
                o_ptr = &out[p * c_out + oc];        // [n_pixels, c_out]
            } else {
                o_ptr = &out[p + oc * n_pixels];     // [c_out, n_pixels]
            }
            *o_ptr += (T_out)acc;                    // in-place: reduction across host calls
        }
    }
    event1();
}

// Weight / output layout selection, matching the host's b_col_maj / c_col_maj args.
#ifdef B_COL_MAJ
constexpr bool is_b_row_maj = false;
#else
constexpr bool is_b_row_maj = true;
#endif

#ifdef C_COL_MAJ
constexpr bool is_c_row_maj = false;
#else
constexpr bool is_c_row_maj = true;
#endif

// -----------------------------------------------------------------------------
// C entry points. Names match what the host resolves against:
//   matmul_scalar_bf16_bf16  <- Kernel(f"matmul{scalar_suffix}_{in}_{out}")
//   zero_scalar_bf16         <- Kernel(f"zero{scalar_suffix}_{out}")
//
// Inner tile sizes come from the host tiling; override with -DDIM_M / -DDIM_K /
// -DDIM_N at compile time. DIM_M = pixels tile, DIM_K = C_in, DIM_N = C_out.
// -----------------------------------------------------------------------------
#ifndef DIM_M
#define DIM_M 64      // pixels per L1 tile
#endif
#ifndef DIM_K
#define DIM_K 64      // C_in
#endif
#ifndef DIM_N
#define DIM_N 64      // C_out
#endif

extern "C" {

// Keep the "matmul" symbol name so this drops straight into the existing host
// graph unchanged (pointwise conv == GEMM). Rename here + on the host together
// if you prefer conv-specific symbols.
void matmul_scalar_bf16_bf16(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out)
{
    conv2d_1x1_scalar<bfloat16, bfloat16, DIM_M, DIM_K, DIM_N, is_b_row_maj, is_c_row_maj>(
        a_in, b_in, c_out);
}

void zero_scalar_bf16(bfloat16 *c_out)
{
    zero_scalar<bfloat16, DIM_M, DIM_N>(c_out);
}

} // extern "C"