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
Multiply). The only "convolution-specific" work is reshaping the image
`[batch, H, W, C_in]` into the flat pixel matrix `[M, C_in]` and back again —
which happens on the host CPU, not on the NPU.

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

The output `C` (`M` pixels × `N` output channels) is cut into small rectangular
**tiles** of size `m × n` (by default `m = 64` pixels, `n = 64` channels). Each
compute core is responsible for producing some of these tiles.

Two directions of the grid map to the two dimensions of the output:

- **Columns of the NPU  ↔  output channels (`N`).**
  Each of the (up to 8) columns owns a different `n`-wide slice of the output
  channels. Column 0 computes channels 0…n-1, column 1 computes n…2n-1, etc.

- **Rows of the NPU  ↔  pixels (`M`).**
  Each of the 4 rows owns a different `m`-tall slice of the pixels. Row 2 handles
  pixels 0…m-1, row 3 handles m…2m-1, and so on.

So the core at (row, col) computes the output tile for *its* block of pixels and
*its* block of channels:

```
                 output channels (N)  ──►
              ┌──────┬──────┬──────┬──────┐
   pixels │   │ core │ core │ core │ core │   row 2  (pixels   0..m-1)
   (M)    │   │ r2c0 │ r2c1 │ r2c2 │ r2c3 │
          ▼   ├──────┼──────┼──────┼──────┤
              │ core │ core │ core │ core │   row 3  (pixels   m..2m-1)
              │ r3c0 │ r3c1 │ r3c2 │ r3c3 │
              ├──────┼──────┼──────┼──────┤
              │ ...  │ ...  │ ...  │ ...  │   rows 4, 5 ...
              └──────┴──────┴──────┴──────┘
                col0   col1   col2   col3
              (chans  (chans (chans  (chans
               0..n)  n..2n) 2n..3n) 3n..4n)
```

### 3.2 Feed A and B by broadcasting + distributing

To compute its tile, a core needs:
- the rows of **A** for its pixels (all `K` input channels of those pixels), and
- the columns of **B** for its output channels (all `K` input channels of those channels).

The DMA engines arrange this cleverly:

- **A (activations)** is **broadcast across columns** and **distributed across rows.**
  Every column needs the same pixels (because every output channel is computed
  from the same input pixel), so A is copied to all columns. But each *row* only
  needs *its* slice of pixels, so the pixels are split among the 4 rows.

- **B (weights)** is **broadcast across rows** and **distributed across columns.**
  Every row (every pixel block) needs the same weights, so B is copied to all
  rows. But each *column* only needs the weights for *its* output channels, so the
  channels are split among the columns.

```
              B slice for   B slice for   B slice for
              cols' chans   cols' chans   cols' chans
                  │             │             │
                  ▼             ▼             ▼
   A rows ──►  ┌──────┐     ┌──────┐     ┌──────┐
   (pixels     │ core │     │ core │     │ core │   same A broadcast
    for this ─►│      │     │      │     │      │   along the row,
    row)       └──────┘     └──────┘     └──────┘   same B broadcast
                                                     down the column
```

This is the classic trick that keeps every core busy: A flows along rows, B flows
along columns, and each core sits at the intersection computing one tile.

### 3.3 Add up over the input channels (the "K reduction")

`K = C_in` can be large (512 in our test), too big to bring into a core's tiny L1
memory at once. So `K` is chopped into blocks of size `k` (default 64). A core
computes its output tile as a **running sum over these k-blocks**:

```
for each k-block (there are K/k of them):
      load an (m × k) tile of A          (this row's pixels, these k channels)
      load a  (k × n) tile of B          (these k channels, this col's out-channels)
      C_tile += A_tile × B_tile          (multiply-accumulate INTO the output tile)
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

- `n_c_col_tiles_per_core = N / (n × num_columns)` — how many channel-slices each
  column must process in sequence.
- `n_c_row_tiles_per_core = M / (m × 4 rows)` — how many pixel-slices each row must
  process in sequence.

### 3.5 Padding the pixels

The array wants the pixel count `M` to be a multiple of `m × 4` (so the 4 rows
divide evenly). The real pixel count `batch × H × W` usually is **not**. So the
host **pads** `M` up to the next multiple with zero rows, runs the hardware on the
padded size, and then **slices the padding back off** the result. (See
`op.py`: `pad_activation` / `unpad_output`, and `M_real` vs `M`.)

---

## 4. Worked example: the tiny test case

Test parameters:
`batch=1, H=8, W=8, C_in=512, C_out=256`, tiles `m=16, k=64, n=64`, `num_columns=1`.

Translate to GEMM:

```
M = 1 × 8 × 8 = 64 pixels      (already a multiple of m×4 = 64, so no padding)
K = C_in  = 512 input channels
N = C_out = 256 output channels
```

Distribution:

- **1 column × 4 rows = 4 cores are used.**
- The 4 rows split the 64 pixels: core in row 2 does pixels 0–15, row 3 does
  16–31, row 4 does 32–47, row 5 does 48–63 (`m = 16` each).
- There is only **1 column** but **N/n = 256/64 = 4** channel-slices, so each core
  processes **4 output tiles in sequence** (channels 0–63, 64–127, 128–191,
  192–255).
- Each output tile is `16 × 64`. To fill it, the core loops over
  **K/k = 512/64 = 8 k-blocks**, accumulating.

Total: 4 rows × 4 channel-slices = 16 output tiles of 16×64, which tile up to the
full 64×256 output. Each tile is an 8-step accumulation. 

A "full" configuration like `C_in=256, C_out=512, num_columns=8` instead lights up
**all 32 cores at once** (8 columns × 4 rows), each computing one 64×64 tile with a
4-step K reduction — that is where the high throughput (hundreds of GFLOP/s)
comes from.

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
| `op.py`         | Host    | The operator definition. Turns conv parameters into GEMM dimensions, pads M, picks kernel/compile flags, names the build artifacts. |
| `design.py`     | Host    | Describes the **dataflow**: the tile sizes, which core does what, and the DMA streaming patterns (plain vs shuffled). Generates the MLIR that is compiled to an NPU program. |
| `aie_kernels/aie2p/conv2d_1x1.cc` | NPU core | The actual compute kernel(s): the vectorized `aie::mmul` matmul and the scalar plain-loop oracle. |
| `reference.py`  | Host    | The "golden" answer computed in plain PyTorch, to compare the NPU output against. |
| `test.py`       | Host    | Builds the operator + reference for several shapes and checks they match. |

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

- **Weight layout.** `weight_layout="oihw"` means the weights arrive as
  `[C_out, C_in]` (the usual framework layout); `"kn"` means you pre-transposed
  them to `[C_in, C_out]`. The kernel and the golden reference are told which one,
  so both agree.

- **A 1×1 conv adds nothing to the *compute* over a GEMM.** All the conv-specific
  logic is host-side reshaping/padding. That is why the compute kernel is the
  shared matmul microkernel, not a bespoke convolution kernel.
