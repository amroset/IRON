# Pointwise (1×1) Convolution on the AMD XDNA2 NPU

This document explains, from the ground up, what the `conv2d_1x1` operator does and
**how the work is spread across the NPU**. It assumes no prior knowledge of
convolutions, matrix multiplication on accelerators, or the AMD AIE/XDNA
architecture. If you only remember one thing, remember this:

> **A 1×1 convolution is just a matrix multiplication.** Everything here is about
> chopping that matrix multiplication into small tiles and streaming those tiles
> through a grid of tiny processors so they can all work at once.

---

## 1. What is a 1×1 convolution?

An image (or a feature map inside a neural network) is a stack of **channels**.
Think of a 8×8 image where every pixel, instead of just holding one color, holds
a vector of, say, 512 numbers. Those 512 numbers are the **input channels**
(`C_in`).

A **1×1 convolution** (also called a *pointwise* convolution) does something very
simple: **it looks at one pixel at a time, and mixes that pixel's channels into a
new set of output channels.** It does *not* look at neighboring pixels (that is
what a 3×3 convolution would do). For each pixel independently:

```
output_channels = Weights · input_channels
```

where `Weights` is a `C_out × C_in` table that is the same for every pixel.
So if a pixel has 512 input channels and we want 256 output channels, we multiply
a 256×512 weight matrix by the pixel's 512-long vector to get a 256-long vector.

### Why this is exactly a matrix multiplication

Now line up **all** the pixels as the rows of a big matrix. With a batch of images
of height `H` and width `W`:

```
Number of pixels   M = batch × H × W
Input channels     K = C_in
Output channels    N = C_out
```

- **A** = the activations, shape `[M, K]` — one row per pixel, `K` channels each.
- **B** = the weights, shape `[K, N]` — how each input channel feeds each output channel.
- **C** = the result, shape `[M, N]` — one row per pixel, `N` output channels each.

And the whole convolution is just:

```
C = A × B         (an M×K matrix times a K×N matrix, giving M×N)
```

That is why this operator reuses the machinery of a **GEMM** (GEneral Matrix
Multiply). The only "convolution-specific" work is *interpreting* the image
tensor as the flat pixel matrix — and, as the next section explains, even that is
handled **for free by choosing the right GEMM mapping** (§1b), not by reshaping on
the host.

---

## 1b. The data layout: feeding `torch.nn` tensors directly

A real caller does not hand us a neat `[M, K]` matrix — it hands us the tensors in
the exact layout **`torch.nn.Conv2d`** uses. PyTorch is **channels-first (NCHW)**:

```
activation  x : [batch, C_in,  H, W]     (NCHW)
weight      w : [C_out, C_in, 1, 1]      (OIHW, the 1×1 kernel)
output      y : [batch, C_out, H, W]     (NCHW)   = conv2d(x, w)
```

The naive move is to call the activation "A" and reshape it to `[pixels, C_in]`.
That is **wrong** and was the original layout bug: in NCHW the channel axis is
*outer* (stride `H·W`), but the GEMM's contraction axis `K = C_in` must be the
*inner*, contiguous axis, so the reshape scrambles the data. And you cannot fix it
with a DMA transpose either — a strided `bf16` transpose reads isolated 2-byte
elements that are not 4-byte aligned, which the NPU's DMA hardware rejects.

**The clean fix is to pick the mapping so that nothing needs transposing.** Write
out what a 1×1 conv actually computes (per pixel `p = h·W + w`):

```
y[c_out, p] = Σ_c_in  w[c_out, c_in] · x[c_in, p]
```

That is a **plain row-major matrix multiply `Y = W · X`** — if we let the
**weights be GEMM's A** and the **activation be GEMM's B** (for `batch == 1`, where
the whole trailing `H·W` is the pixel axis). Flattened, all three torch tensors
are already exactly the row-major GEMM buffers:

| GEMM role | tensor | torch.nn layout | flattened | GEMM frame |
|-----------|--------|-----------------|-----------|------------|
| **A** `[M, K]` | weight `w`     | `[C_out, C_in, 1, 1]` | `[C_out, C_in]` | row-major ✅ |
| **B** `[K, N]` | activation `x` | `[1, C_in, H, W]`     | `[C_in, H·W]`   | row-major ✅ |
| **C** `[M, N]` | output `y`     | `[1, C_out, H, W]`    | `[C_out, H·W]`  | row-major ✅ |

So the conv dims map to the GEMM dims as **M = C_out, K = C_in, N = H·W (pixels)**.
No transposes, no `b_col_maj`/`c_col_maj` flags, no host reshaping, and — because
everything stays contiguous and row-major — no DMA alignment problems. The design
is reused **verbatim**; all the conv-specific logic is this relabeling in `op.py`.

The payoff: the host passes the **flattened torch tensors verbatim** (`A =
w.flatten()`, `B = x.flatten()`), and the result comes back already in NCHW.
"Assume the data comes in the format the `torch.nn` operator expects" is satisfied
literally.

> **Current scope.** The direct `[K, N]` mapping needs the pixel axis to be the
> whole trailing dimension, so it requires **`batch == 1`** (with `batch > 1` the
> flattened activation is `[batch, C_in, H·W]`, not `[C_in, batch·H·W]`).
> **Padding is now implemented** (§3.5), so the conv shape no longer has to tile
> evenly — real MobileNetV3 / YOLO shapes run directly. Batched inputs
> (`batch > 1`) and the multi-column throughput regime are the planned next steps.

---

## 2. What does the NPU look like?

An NPU (Neural Processing Unit) is not one big processor. It is a **grid of many
tiny processors**, each with its own small local memory. The AMD XDNA2 NPU (also
called **NPU2**, found in Strix/Strix Halo/Krackan laptops) is laid out as a grid
of **tiles**:

```
        col 0     col 1     col 2    ...    col 7        (8 columns)
      ┌────────┬────────┬────────┬───────┬────────┐
row 5 │  core  │  core  │  core  │  ...  │  core  │  ┐
row 4 │  core  │  core  │  core  │  ...  │  core  │  │  4 rows of
row 3 │  core  │  core  │  core  │  ...  │  core  │  │  COMPUTE cores
row 2 │  core  │  core  │  core  │  ...  │  core  │  ┘  (32 total)
      ├────────┼────────┼────────┼───────┼────────┤
row 1 │  mem   │  mem   │  mem   │  ...  │  mem   │     L2 scratchpad (1 per column)
      ├────────┼────────┼────────┼───────┼────────┤
row 0 │  shim  │  shim  │  shim  │  ...  │  shim  │     door to external DRAM
      └────────┴────────┴────────┴───────┴────────┘
```

There are three kinds of tile:

| Tile        | Row      | Role                                                                 |
|-------------|----------|----------------------------------------------------------------------|
| **Shim**    | row 0    | The doorway between the NPU and the computer's main memory (DRAM).   |
| **Mem**     | row 1    | A medium-sized on-chip scratchpad (one per column). Staging area.    |
| **Compute** | rows 2–5 | The actual processors ("AIE cores"). Each has a tiny private memory. |

So NPU2 has **4 compute rows × 8 compute columns = up to 32 compute cores**, each
able to multiply-and-add numbers in parallel. The whole challenge is: *how do we
feed 32 hungry cores with the right slices of A, B, and C so none of them sit
idle?*

### The three levels of memory (L3 / L2 / L1)

Data lives at three "distances" from a core, and moving it is done by **DMA
engines** (Direct Memory Access — hardware that copies blocks of memory without
bothering the cores):

```
  L3  = DRAM (main memory)        — holds the WHOLE A, B, C.   Big, far, slow.
   │      (via shim tiles, row 0)
   ▼
  L2  = mem-tile SRAM (row 1)     — holds a handful of tiles.  Medium.
   │
   ▼
  L1  = core-local SRAM (rows 2-5)— holds ONE small tile.      Tiny, near, fast.
```

The cores can only compute on data that has been brought all the way down to
**L1**. So the operator constantly streams small tiles L3 → L2 → L1, computes,
and streams results L1 → L2 → L3. These streams are managed by objects called
**ObjectFifos** (think: conveyor belts with a fixed number of slots).

---

## 3. How the work is distributed across the NPU

This is the heart of it. We have three matrices and 32 cores. Here is the plan.

### 3.1 Cut the output into tiles, one per core

Recall the mapping from §1b: **M = C_out (output channels), K = C_in, N = H·W
(pixels)**. The output `C` (`M` output channels × `N` pixels) is cut into small
rectangular **tiles** of size `m × n` (by default `m = 64` channels, `n = 64`
pixels). Each compute core is responsible for producing some of these tiles.

Two directions of the grid map to the two dimensions of the output:

- **Columns of the NPU  ↔  pixels (`N`).**
  Each of the (up to 8) columns owns a different `n`-wide slice of the pixels.
  Column 0 computes pixels 0…n-1, column 1 computes n…2n-1, etc.

- **Rows of the NPU  ↔  output channels (`M`).**
  Each of the 4 rows owns a different `m`-tall slice of the output channels. Row 2
  handles channels 0…m-1, row 3 handles m…2m-1, and so on.

So the core at (row, col) computes the output tile for *its* block of output
channels and *its* block of pixels:

```
                    pixels (N)  ──►
              ┌──────┬──────┬──────┬──────┐
  out-chans│  │ core │ core │ core │ core │   row 2  (chans   0..m-1)
   (M)     │  │ r2c0 │ r2c1 │ r2c2 │ r2c3 │
           ▼  ├──────┼──────┼──────┼──────┤
              │ core │ core │ core │ core │   row 3  (chans   m..2m-1)
              │ r3c0 │ r3c1 │ r3c2 │ r3c3 │
              ├──────┼──────┼──────┼──────┤
              │ ...  │ ...  │ ...  │ ...  │   rows 4, 5 ...
              └──────┴──────┴──────┴──────┘
                col0   col1   col2   col3
              (pix    (pix   (pix    (pix
               0..n)  n..2n) 2n..3n) 3n..4n)
```

### 3.2 Feed A and B by broadcasting + distributing

To compute its tile, a core needs:
- the rows of **A** (weights) for its output channels — all `K = C_in` input
  channels of those output channels, and
- the columns of **B** (activation) for its pixels — all `K = C_in` input channels
  of those pixels.

The DMA engines arrange this cleverly:

- **A (weights)** is **broadcast across columns** and **distributed across rows.**
  Every column (every pixel block) needs the same weights, so A is copied to all
  columns. But each *row* only needs *its* slice of output channels, so the output
  channels are split among the 4 rows.

- **B (activation)** is **broadcast across rows** and **distributed across columns.**
  Every row (every output-channel block) is computed from the same pixels, so B is
  copied to all rows. But each *column* only needs *its* slice of pixels, so the
  pixels are split among the columns.

```
              B slice for   B slice for   B slice for
              cols' pixels  cols' pixels  cols' pixels
                  │             │             │
                  ▼             ▼             ▼
   A rows ──►  ┌──────┐     ┌──────┐     ┌──────┐
   (out-chans  │ core │     │ core │     │ core │   same A (weights) broadcast
    for this ─►│      │     │      │     │      │   along the row,
    row)       └──────┘     └──────┘     └──────┘   same B (activation) broadcast
                                                     down the column
```

This is the classic trick that keeps every core busy: A flows along rows, B flows
along columns, and each core sits at the intersection computing one tile.

### 3.3 Add up over the input channels (the "K reduction")

`K = C_in` can be large, too big to bring into a core's tiny L1 memory at once. So
`K` is chopped into blocks of size `k` (default 64). A core computes its output
tile as a **running sum over these k-blocks**:

```
for each k-block (there are K/k of them):
      load an (m × k) tile of A     (this row's out-channels, these k in-channels)
      load a  (k × n) tile of B     (these k in-channels, this col's pixels)
      C_tile += A_tile × B_tile     (multiply-accumulate INTO the output tile)
```

Before the loop the core **zeros** its output tile; then each k-block adds its
contribution. After `K/k` iterations, the tile holds the complete dot-products.
In code (`design.py`, the `core_fn`) this is exactly:

```
zero(C_tile)
for _ in range(K/k):
    matmul(A_tile, B_tile, C_tile)   # C_tile += A_tile × B_tile
```

### 3.4 When the problem is bigger than the array

The NPU has at most 32 cores, but a real problem may have far more than 32 output
tiles. When that happens, **each core simply processes several tiles one after
another** in a loop (`rtp_n_tiles_per_core` in the code). The array makes one
"pass," then reuses the same cores for the next batch of tiles. Two counters
control this:

- `n_c_col_tiles_per_core = N / (n × num_columns)` — how many pixel-slices each
  column must process in sequence.
- `n_c_row_tiles_per_core = M / (m × 4 rows)` — how many output-channel-slices each
  row must process in sequence.

### 3.5 Divisibility and padding

The array wants each mapped dimension to tile evenly: `M = C_out` a multiple of
`m × 4` (so the 4 rows divide evenly), `K = C_in` a multiple of `k`, and
`N = H·W` a multiple of `n × num_columns`. A real conv shape does not always
satisfy this — MobileNetV3 and YOLO are full of counterexamples (`C_out = 40`,
`H·W = 49`, `H·W = 400`, …).

**Padding handles this, entirely on the host.** `op.py` rounds the array-facing
dims `M/K/N` **up** to the tiling granularity and stores the logical sizes as
`M_raw/K_raw/N_raw`; the `pad_weight` / `pad_activation` / `pad_output` helpers
zero-extend the flattened torch tensors to the padded dims, and `unpad_output`
slices the logical region back out. Because the padded channels/pixels are zero,
they contribute exactly 0 to the result, so the padded output equals the logical
output zero-extended. The **design and compute kernel are unchanged** — they
never see a ragged shape — which is what keeps the GEMM design reused verbatim.
This mirrors GEMM's own `pad_A` / `pad_B`.

The cost is real, though: padding makes the array multiply zeros. The operator
exposes `pad_overhead` (padded MACs ÷ useful MACs) so tests can report it — e.g.
the MobileNetV3 `80→240 @14×14` layer runs at `2.23×` overhead. Reclaiming that
waste (by batching to fill `N`, or remapping which conv axis feeds columns vs
rows) is the motivation for the host-orchestration follow-up. (`batch > 1` and
the multi-column regime are the other planned next steps; see §1b.)

---

## 4. Worked example: the single-core test case

Test parameters:
`batch=1, H=8, W=8, C_in=128, C_out=64`, tiles `m=16, k=64, n=64`, `num_columns=1`.

Translate to GEMM (with the torch.nn tensors fed directly — see §1b):

```
M = C_out =  64 output channels   (a multiple of m×4 = 64, so no padding)
K = C_in  = 128 input channels
N = H×W   = 64 pixels
```

Distribution:

- **1 column × 4 rows = 4 cores are used.**
- The 4 rows split the 64 output channels: core in row 2 does channels 0–15, row 3
  does 16–31, row 4 does 32–47, row 5 does 48–63 (`m = 16` each).
- There is only **1 column** and **N/n = 64/64 = 1** pixel-slice, so each core
  produces a single `16 × 64` output tile (16 output channels × 64 pixels).
- To fill its tile, each core loops over **K/k = 128/64 = 2 k-blocks**,
  accumulating over the input channels.

Everything is contiguous row-major: `A = w.flatten()` (`[64,128]`), `B =
x.flatten()` (`[128,64]`), and the result `C` (`[64,64]`) is already the NCHW
output `y` — no reshaping on either side.

A "full" configuration like `C_out=512, C_in=256, H·W=…, num_columns=8` instead
lights up **all 32 cores at once** (8 columns × 4 rows), each computing one 64×64
tile with a 4-step K reduction — that is where the high throughput (hundreds of
GFLOP/s) comes from. (That multi-column, batched regime is the planned follow-up to
this single-core layout milestone.)

---

## 5. Two compute kernels: vectorized vs scalar

Once a core has an A-tile and a B-tile in L1, *how* does it multiply them? There
are two implementations, selected by the `use_scalar` flag.

### Vectorized kernel (the fast, default path)

Uses the hardware's `aie::mmul` matrix-multiply instruction, which multiplies
small fixed-size sub-blocks in a single shot (SIMD — many multiply-adds per
clock). To feed this instruction, the DMA does **not** deliver tiles as plain
rows and columns; it delivers them **pre-shuffled into little `r×s` / `s×t` /
`r×t` sub-blocks** (the shapes the hardware instruction expects). This is the
right choice for real work: it reaches hundreds of GFLOP/s.

### Scalar kernel (the slow, readable "oracle")

A plain triple `for` loop — `for each pixel, for each output channel, sum over
input channels` — with no special instructions. It is far slower, but it is easy
to read and trust, which makes it a useful **reference for checking correctness**
and a **baseline for performance comparisons.**

The catch (and the source of a long-standing bug in this operator): the scalar
loop expects **plain row-major tiles**, but the vectorized path's DMA delivers the
**shuffled sub-block layout**. Feeding shuffled data to the plain loop produces
garbage. The fix: `design.py` now checks `use_scalar` and streams **plain tiles
for the scalar kernel** and **shuffled tiles for the vectorized kernel**. Each
kernel gets exactly the layout it understands.

One more subtlety: the scalar loop accumulates in **32-bit float** (it casts the
16-bit `bf16` inputs up before multiplying). If it multiplied in `bf16` and summed
512 tiny rounded products, the result would drift a few percent low. Accumulating
in float keeps it faithful to the reference. Keep `prio_accuracy=True` for the
scalar path so the intermediate results are also kept in float across k-blocks.

---

## 6. Where each piece of code lives

| File            | Runs on | Responsibility                                                                 |
|-----------------|---------|--------------------------------------------------------------------------------|
| `op.py`         | Host    | The operator definition. Maps conv params to GEMM dims (M=C_out, K=C_in, N=H·W) so the design consumes the flattened torch.nn tensors directly, picks kernel/compile flags, names the build artifacts. |
| `design.py`     | Host    | The **shared GEMM design, reused verbatim** (identical to `gemm/design.py` apart from the scalar-kernel tiling). Describes the dataflow: tile sizes, which core does what, and the DMA streaming patterns. Generates the MLIR that is compiled to an NPU program. |
| `aie_kernels/aie2p/conv2d_1x1.cc` | NPU core | The actual compute kernel(s): the vectorized `aie::mmul` matmul and the scalar plain-loop oracle. |
| `reference.py`  | Host    | The "golden" answer: builds real `torch.nn`-shaped tensors (NCHW `x`, OIHW `w`) and computes `y = F.conv2d(x, w)` to compare the NPU output against. |
| `test.py`       | Host    | Builds the operator + reference for several shapes and checks they match — passing the **flattened torch tensors verbatim**, with no host reshaping. |

The flow: **`op.py` + `design.py`** together generate a hardware program (an
`.xclbin`), the **kernel `.cc`** is compiled into it, the program runs on the NPU,
and **`test.py`** compares the result against **`reference.py`**.

---

## 7. Gotchas worth knowing

- **Toggling `use_scalar` / `prio_accuracy` / `emulate_bf16_mmul_with_bfp16`.**
  These flags change the *generated program*, but historically they were **not**
  part of the operator's cache name, so switching one would silently reuse a stale
  compiled program from `build/` and give wrong/old results. `op.py` now folds
  these flags into the operator `name`, so each combination gets its own cache
  entry. If you ever suspect a stale build after editing the kernel or design,
  delete the cached artifacts: `rm build/Conv2d1x1_* build/conv1x1_*`.

- **The scalar path needs `prio_accuracy=True`** to stay within tolerance (see §5).

- **Data layout (NCHW).** Inputs are assumed to be in `torch.nn.Conv2d` format:
  NCHW activation, OIHW weight, NCHW output. Because the operator maps the
  **weights to GEMM-A and the activation to GEMM-B** (§1b), the flattened torch
  tensors are already the row-major GEMM buffers, so **no host reshaping and no
  layout flags are needed** (`b_col_maj`/`c_col_maj` stay `False`). Do *not* try
  to "fix" the layout by transposing the activation in the DMA — a strided `bf16`
  transpose is not 4-byte aligned and the NPU DMA rejects it.

- **A 1×1 conv adds nothing to the *compute* over a GEMM.** The only conv-specific
  logic is the dimension relabeling in `op.py`; the design and the compute kernel
  are the shared GEMM ones, not bespoke convolution code.
