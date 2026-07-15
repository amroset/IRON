# Pointwise (1×1) Convolution on the AMD XDNA2 NPU — `conv2d_1x1_opt`

This is the **optimized** pointwise-convolution operator. It starts from the plain
GEMM mapping (a 1×1 conv *is* a matrix multiply) and adds one hardware-aware
optimization — the **cols→M axis swap** — that turns the array's biggest weakness
on real CNN layers (column starvation) into throughput. This document is the full
picture: what the operator computes, how the work is spread across the NPU, the
optimization, and the measured results.

> **If you remember one thing:** a 1×1 convolution is a matrix multiply, and the
> only question that matters for speed is *how you map its three dimensions onto a
> 4-row × 8-column grid of cores*. Picking that mapping per layer is the whole game.

---

## 1. What a 1×1 convolution is (and why it is a GEMM)

A feature map is a stack of **channels**: every pixel holds a vector of `C_in`
numbers. A **1×1 (pointwise) convolution** looks at one pixel at a time and mixes
its channels into `C_out` new ones with a weight table that is the same for every
pixel:

```
output_channels = Weights · input_channels          (Weights is C_out × C_in)
```

Line up all the pixels and this is exactly a matrix multiply. With `batch = 1`:

| GEMM role | tensor | torch.nn layout | flattened | frame |
|-----------|--------|-----------------|-----------|-------|
| **A** `[M, K]` | weight `w`     | `[C_out, C_in, 1, 1]` | `[C_out, C_in]` | row-major |
| **B** `[K, N]` | activation `x` | `[1, C_in, H, W]`     | `[C_in, H·W]`   | row-major |
| **C** `[M, N]` | output `y`     | `[1, C_out, H, W]`    | `[C_out, H·W]`  | row-major |

So the conv maps to GEMM as **M = C_out, K = C_in, N = H·W (pixels)**, and
`C = A × B`. Crucially, the flattened `torch.nn` tensors are *already* the
row-major GEMM buffers — weights→A, activation→B, output→C — so in the plain
mapping there are **no transposes and no host reshaping**, and the result comes
back in NCHW. (Trying to "fix" a layout with a DMA transpose is illegal anyway: a
strided `bf16` transpose reads 2-byte elements that are not 4-byte aligned, which
the NPU DMA rejects. The escape from that trap for the swap is §4.)

*Scope:* the direct `[K, N]` mapping treats one image at a time. Non-tile-aligned
shapes are handled by padding (§3); the axis swap and multi-column execution are
implemented (§4–§5); **`batch > 1` is supported** by running the per-image program
once per image with shared weights (§7).

---

## 2. The NPU and how a GEMM is spread across it

NPU2 (AMD XDNA2, Strix) is a grid of tiles: **4 compute rows × 8 compute columns =
up to 32 cores**, a row of **mem** tiles (L2 scratchpad, one per column), and a row
of **shim** tiles (the door to DRAM). Data moves L3 (DRAM) → L2 (mem) → L1
(core-local) by DMA engines, streamed through **ObjectFifos** (conveyor belts with
a fixed number of slots). Cores compute only on L1 data.

The output `C` (M×N) is cut into `m × n` tiles and mapped onto the grid:

- **Columns ↔ pixels (N).** Each column owns an `n`-wide slice of pixels.
- **Rows ↔ output channels (M).** Each of the 4 rows owns an `m`-tall slice of C_out.
- **K = C_in is the reduction**, chopped into `k`-blocks; each core accumulates its
  tile over `K/k` multiply-adds (`C_tile += A_tile × B_tile`).

A (weights) is broadcast across columns and distributed across rows; B (activation)
is broadcast across rows and distributed across columns. When there are more output
tiles than cores, each core processes several tiles in sequence
(`n_c_col_tiles_per_core = N/(n·cols)`, `n_c_row_tiles_per_core = M/(m·4)`).

---

## 3. Padding (host-side, layout-preserving)

The array wants each mapped dim to tile evenly: `M` a multiple of `m·4`, `K` of
`k`, `N` of `n·cols`. Real conv shapes rarely comply (`C_out = 40`, `H·W = 49`, …).
`op.py` rounds `M/K/N` **up** and the `pad_weight` / `pad_activation` / `pad_output`
helpers **zero-extend** the flattened tensors in their natural storage order;
`unpad_output` slices the logical region back. Zero-padded channels/pixels
contribute exactly 0, so the padded result equals the logical result zero-extended,
and the **design and kernel never see a ragged shape**. This mirrors GEMM's own
`pad_A`/`pad_B` (same principle, same layout-aware, never a host transpose).

`op.py` exposes `pad_overhead` (padded MACs ÷ useful MACs) so tests report how much
array time is spent multiplying zeros.

---

## 4. The optimization: the cols→M axis swap

### The problem — column starvation
The array is **4 rows × 8 columns**. The plain mapping hard-wires **columns ↔
pixels**. For **tall** layers (many output channels, few pixels — the late stages
of every ResNet/ConvNeXt/EfficientNet) the 8-wide column axis is spent on a handful
of pixels (mostly zero-padding), while the large `C_out` dimension — where the work
is — gets only 4-way row parallelism. Across 8 real models, **34% of all 1×1-conv
layers are tall**, and every 7×7 (N=49) layer actually *slows down* as columns are
added: the columns are pure padding.

### The idea
Map **columns ↔ C_out** (8-wide, where the channels are) and **rows ↔ pixels**
(4-wide, plenty for a small pixel count). Put the wide axis where the work is.

### The escape — feed everything as-stored, transpose in the kernel
Swapping forces the activation into the A (rows) operand as `[pixels, C_in]` — but
it is stored `[C_in, pixels]`, i.e. **column-major**, and a `bf16` DMA transpose is
illegal. The fix mirrors exactly how GEMM already handles `b_col_maj`/`c_col_maj`:
feed each operand **as-stored** and let the **compute kernel transpose the small
`r×s` sub-blocks in-register** (`aie::transpose`). No host transpose anywhere.

| design dim | = conv dim | operand (as fed) | flag |
|---|---|---|---|
| M̂ (rows) | pixels H·W | **A** = activation `[C_in, pixels]` | `a_col_maj` |
| K̂ | C_in | (reduction) | — |
| N̂ (cols) | C_out | **B** = weights `[C_out, C_in]` | `b_col_maj` |
| output | — | **C** = output `[C_out, pixels]` (NCHW) | `c_col_maj` |

`a_col_maj` is the one new piece; it is the exact mirror of `b_col_maj` (col-major
`a_dims`, a col-major `A_tiles` DRAM tiler, and an in-kernel `aie::transpose`),
gated by `-DA_COL_MAJ` so the baseline path is untouched. It requires **≥ 4
columns** (so each shim feeds exactly one A row-tile); the swap is a multi-column
optimization, so this is not a real restriction. All three buffers keep their
natural torch storage — only the operand **slot order** and the col-major flags
change — so `pad_*`/`unpad_output` are mapping-independent.

### When it fires — the auto heuristic
The swap only pays off when the plain mapping actually **starves** the columns —
otherwise its extra in-kernel transposes cost more than the occupancy they buy.
`swap_mn=None` (default) swaps iff

```
ceil(pixels / tile_n) · 2 ≤ num_columns    AND    ceil(C_out / tile_n) > ceil(pixels / tile_n)
```

i.e. the plain mapping leaves at least half the columns idle and C_out fills more
of them. Force it with `swap_mn=True`/`False`.

---

## 5. Results (NPU2, cols=8, tile 16/64/64, bf16 in / f32 accumulate)

**Protocol:** 10 warmup rounds, then **median over 50 timed on-device samples**.
Throughput is on **useful conv MACs** (`2·C_out·C_in·H·W`, padding excluded), so
it is directly comparable across mappings.

### Correctness
All **22/22** real-model shapes (ResNet-50, ResNeXt-50, MobileNetV2/V3,
EfficientNet-B0, ConvNeXt-T, DenseNet-121, YOLOv5l) pass the torch golden reference
(rel/abs tol 5e-3) at cols=8 — tall shapes take the swap, wide shapes stay plain.

### Swap speedup across the real-model suite (opt ÷ baseline)
| C_in / C_out / N | speedup | | C_in / C_out / N | speedup |
|---|---:|---|---|---:|
| 768 / 3072 / 49  | **6.27×** | | 1152 / 192 / 49  | 1.44× |
| 1024 / 2048 / 49 | **4.58×** | | 112 / 672 / 196  | 1.28× |
| 512 / 2048 / 49  | **4.40×** | | 80 / 480 / 196   | 1.22× |
| 2048 / 512 / 49  | **3.28×** | | 1024 / 512 / 196 | 1.19× |
| 3072 / 768 / 49  | **3.23×** | | 96 / 576 / 196   | 1.12× |
| 384 / 1536 / 196 | 1.84×     | | **median (11 swapped)** | **1.84×** |

The 7×7 (N=49) tail — the layers plain GEMM handles **worst** — gains **2.9–6.3×**.
Wide layers keep the plain mapping (auto-heuristic declines), so the operator never
trades throughput away; the one tall layer that already fills its columns
(`1024/512/400`, N=400) is correctly left on the plain mapping.

### Column scaling — why the swap exists
Throughput (GF/s) `baseline / opt` as columns are added:

| C_in/C_out/N | cols=1 | cols=2 | cols=4 | cols=8 |
|---|---|---|---|---|
| 768/3072/49  | 211/213 | 206/207 | 187/**650** | 165/**1074** |
| 1024/2048/49 | 218/218 | 217/214 | 204/**633** | 188/**807**  |
| 384/1536/196 | 224/222 | 382/390 | 549/567     | 488/**915**  |

For the 7×7 layers the **baseline is flat or *falling*** as columns are added
(768/3072/49: 211 → 165 GF/s) — extra columns are pure padding. The swap makes
throughput **climb** (213 → 1074). It engages precisely when a column count starves
the layer (14×14 layers wait until cols=8).

### Scalar vs vectorized kernel (cols=8)
| C_in / C_out / N | scalar GF/s | vectorized GF/s | vec / scalar |
|---|---:|---:|---:|
| 768 / 3072 / 49  | 6.7 | **1042** | **156×** |
| 384 / 1536 / 196 | 6.7 | 853 | 128× |
| 2048 / 512 / 49  | 6.6 | 641 | 97× |
| 64 / 256 / 3136  | 7.5 | 711 | 95× |

The scalar kernel is pinned at ~6.6 GF/s (one MAC at a time); the vectorized
`aie::mmul` kernel reaches 575–1042 GF/s — **median 97×, up to 156×**. Stacked, for
`768/3072/49`: scalar 6.7 → vectorized-plain 165 (**25×**) → vectorized+swap 1042
(**156×**).

---

## 6. The two compute kernels

Selected by `use_scalar`:

- **Vectorized (default):** uses the hardware `aie::mmul` instruction on pre-shuffled
  `r×s`/`s×t` sub-blocks (the layout the DMA delivers). Hundreds of GF/s. Also holds
  the `a/b/c_col_maj` in-register transposes the swap relies on.
- **Scalar (reference):** a plain triple loop on plain row-major tiles — slow but
  easy to trust, used as a correctness oracle and performance baseline. It
  accumulates in 32-bit float, so keep `prio_accuracy=True` for it.

`design.py` streams **plain tiles to the scalar kernel and shuffled tiles to the
vectorized kernel**, so each gets the layout it expects.

---

## 7. Using the operator

```python
op = Conv2d1x1(
    batch=1, H=7, W=7, C_in=1024, C_out=2048,   # conv shape (torch.nn dims)
    tile_m=16, tile_k=64, tile_n=64,            # GEMM tile
    num_aie_columns=8,                          # sweep this yourself
    use_scalar=False, prio_accuracy=True,
    swap_mn=None,                               # None=auto, True/False=force
    context=ctx,
)
rows, cols = op.input_operands(w, x)   # ordered for the mapping (handles the swap)
inputs  = {"A": rows, "B": cols}
outputs = {"C": op.pad_output(y).flatten()}
```

- **`num_aie_columns`** is the main throughput lever — sweep it (`{1,2,4,8}`); the
  swap engages automatically at ≥ 4 columns when a layer is starved.
- **`tile_m/k/n`** is the tile-shape lever (16/64/64 is the proven default; it is
  autotuned per shape — see below).
- **`swap_mn`** forces or disables the axis swap; `input_operands` returns the two
  input buffers already in the design's arg order for the chosen mapping.

**Tile autotuning.** The tile shape is tuned **empirically, not analytically** — an
analytical tile model was unreliable here, and some tiles regress *catastrophically*
(e.g. `32/64/64` → 36 GF/s on `2048/512/49` vs 513 for the default) in ways no
static model predicts. `autotune.py` sweeps a small candidate set **on-device** per
`(shape, cols)`, verifies correctness, and caches the fastest in `tile_cache.json`;
`get_optimal_tile()` reads it and `CONV_TUNE=1` runs the suite on the tuned tiles.
The cache ships tuned for the **full 22-shape suite at cols=8** (any other
`(shape, cols)` falls back to the 16/64/64 default). Gains are modest but real —
**up to 1.41×**, median ~1.11× over 16/64/64 — and the best tile **varies per
shape** (no single tile wins: `16/128/64` helps tall big-K layers, larger
`32/64/64`/`64/64/64` win on aligned mid-size, `16/64/64` stays best for several).
Selection lives in the host (like softmax's `get_optimal_*`), so `op.py` is
unchanged.

**Complementary to the swap.** Tuning helps *least* exactly where the swap helps
*most*, for a mechanical reason: on a tall 7×7 layer the swap moves the ~49 pixels
onto the **row axis**, which pins `m = 16` (a bigger row tile would be pure padding)
— **every N=49 layer keeps `m=16`** in the cache. The tuner's real gains (up to
1.41×) land on the 14×14 / wide layers, where the pixel axis is large enough that a
bigger tile (`m=32–64`) is free and amortizes per-tile overhead — often the wide
layers the swap declines. Occupancy (columns) and tile amortization are orthogonal
axes of the same problem, so the two levers cover disjoint parts of the workload.

```bash
CONV_COLS=8 python autotune.py             # tune the shape set -> tile_cache.json
CONV_COLS=8 CONV_TUNE=1 pytest … test.py   # run the suite on tuned tiles
```

**Batching (`batch > 1`).** A batch is `batch` independent 1×1 convs that share the
same weights (`Y_b = W · X_b`). Because the batch axis is stored *outside* the
channels (`[batch, C_in, H·W]`), a batch is not one big GEMM — it is one **dispatch
per image** of this same per-image program, with the weights reused. `op.py` accepts
`batch > 1` and exposes `image_operands(w, x)`, which yields the `(rows, cols)`
buffers for each image; the caller runs one dispatch per image and concatenates:

```python
op = Conv2d1x1(batch=4, H=7, W=7, C_in=1024, C_out=2048, …)
for b, (rows, cols) in enumerate(op.image_operands(w, x)):      # x: [4, C_in, H, W]
    op_func({"A": rows, "B": cols}, out=op.pad_output(y[b]).flatten())
```

This is mapping-agnostic — it works identically for the plain and swapped mappings.
(A single-dispatch batched design, gemv/transpose-style, is possible but interacts
with the swap's operand flip; per-image dispatch reuses the design untouched.)

The `test.py` suite draws its shapes from 8 real models and sweeps columns via the
`CONV_COLS` env var (`CONV_COLS=8 pytest … test.py`); `test_conv2d_1x1_batched`
covers `batch > 1` on both a wide (plain) and a tall (swapped) layer.

---

## 8. Where each piece lives

| File | Runs on | Responsibility |
|------|---------|----------------|
| `op.py` | Host | Operator definition: conv→GEMM mapping, the cols→M swap decision + auto-heuristic, layout-preserving padding, per-image batching (`image_operands`), kernel/compile flags (`-DA_COL_MAJ` etc.), build-artifact naming. |
| `design.py` | Host | The shared whole-array GEMM design; adds the `a_col_maj` col-major A path (mirror of `b_col_maj`) and the scalar-kernel tiling. Generates the MLIR. |
| `aie_kernels/aie2p/conv2d_1x1.cc` | NPU core | Vectorized `aie::mmul` kernel (with `a/b/c_col_maj` in-register transposes) and the scalar oracle. |
| `reference.py` | Host | Golden answer: builds NCHW `x` / OIHW `w`, computes `y = F.conv2d(x, w)`. |
| `test.py` | Host | Real-model suite; feeds flattened torch tensors via `input_operands`, checks against the reference, reports throughput + `pad_overhead`. Holds the tile-selection helper `get_optimal_tile()`. |
| `autotune.py` | Host | Empirical tile-size sweep; writes the fastest tile per (shape, cols) to `tile_cache.json`. |

---

## 9. Gotchas

- **Build cache & flags.** `use_scalar` / `prio_accuracy` / `emulate_bf16…` /
  `a/b/c_col_maj` all change the generated program and are folded into the operator
  `name` and kernel-object name, so each combination is its own cache entry. If you
  suspect a stale build: `rm build/Conv2d1x1_* build/conv1x1_*`.
- **The scalar path needs `prio_accuracy=True`** to stay within tolerance.
- **Never transpose on the host.** The swap feeds col-major operands as-stored and
  transposes in-kernel; a strided `bf16` DMA transpose is illegal (not 4-byte
  aligned). This is the same rule GEMM follows for `b/c_col_maj`.
- **The swap needs ≥ 4 columns.** At 1–2 columns a tall layer auto-falls-back to the
  plain mapping (the A row-distribution only lines up with ≥ 4 columns).
- **`batch > 1` is one dispatch per image** (shared weights), not a single fused
  GEMM — the per-image `N` is `H·W`, never `batch·H·W`. Iterate with
  `op.image_operands(w, x)`.
- **A 1×1 conv adds no *compute* over a GEMM.** All conv-specific logic is the
  dimension mapping and the swap decision in `op.py`; the compute kernel is the
  shared GEMM one.
