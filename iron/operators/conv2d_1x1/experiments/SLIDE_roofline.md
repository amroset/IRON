# Roofline slide, drop-in copy

Figure: `fig_i_roofline_bare.png` (no internal heading, the PowerPoint title
carries it, same as the swap-results slide). Use `fig_i_roofline.png` if you
want the self-contained version with its own heading, e.g. for the appendix or
the summary doc.

Suggested position: **after the cycle-budget slide, before Conclusions.** The
cycle budget explains the vertical gap for one layer; the roofline generalises
it to all 22 and adds the horizontal axis the deck does not otherwise have.

---

## Title

**Where we land on the roofline**

## Bullets

- Two roofs: **3.69 TF/s** bf16 compute, and **67 GB/s** DRAM, *measured with
  `mem_copy`, not a datasheet number*
- Arithmetic intensity is set by **the mapping, not the layer**: plain re-reads
  the activation **48×**, so the layer sits at 5 FLOP/byte with a ceiling of
  only **340 GF/s**
- The swap moves it **right before it moves it up**, intensity ×8.4, ceiling
  **340 GF/s → 2.87 TF/s**, *then* filling the array cashes it in
- **18 of 22** real layers sit left of the ridge → the diagonal, not the flat
  peak, is what actually caps them
- We land at **8–41%** of the applicable roof; with the array genuinely full
  (N=9216) we reach **60%**, right on the ridge

---

## Speaker notes

Two roofs. The flat one is arithmetic, 32 cores, 32 MACs a cycle, 1.8 GHz.
The diagonal one I measured, with the `mem_copy` microbenchmark, streaming
through all eight shim DMAs: 67 GB/s. I did not want to put a datasheet LPDDR
number on a chart and call it a roof.

The x-axis is the part worth pausing on. Arithmetic intensity here is useful
FLOPs over the bytes we *actually* move, which is not the size of the tensors.
Under the plain mapping we make 48 passes over the rows, and each pass re-reads
the whole activation. So the layer presents itself to DRAM as 5 FLOP per byte,
and at 5 FLOP per byte the roofline says you cannot go faster than 340 GF/s. We
measured 165. So even if the array had been perfectly busy, the plain mapping
was never going to beat about 2× of the baseline. The ceiling was in the way.

That reframes the swap slightly, and I think in our favour. I have been
describing it as an occupancy fix. It is also a traffic fix, and the traffic
fix has to come first. Putting channels on the columns makes M sixty-four
instead of three thousand, so the activation is read once instead of 48 times,
and the layer moves 8× to the right. Its own ceiling goes from 340 GF/s to
2.87 TF/s. *Then* filling the array is worth doing. Raise the ceiling, then fill
it. That is the order, and the arrows on this chart are what it looks like.

Where we land: 8 to 41% of whatever roof applies to each layer. Eighteen of the
22 are left of the ridge, so for most of them the binding constraint is the
diagonal. The black diamond is not a real layer; it is a big GEMM, N=9216,
where the array is genuinely full: 60% of peak, and it lands essentially on the
ridge. That is the design's own ceiling, and the earlier slide showed stock IRON
GEMM hits the same number on the same shape.

One honest caveat, in case it is asked: we are never actually bandwidth-*bound*
in the sense of saturating the pipe. The busiest layer reaches 37 GB/s, 56% of
the roof. The pipe is never full. What the roofline shows is the ceiling the
traffic implies, and the remaining distance to that ceiling is the launch floor
and the padding, which is exactly what the previous slide broke down.

---

## If asked

**"Why is intensity so low even after the swap?"**
Two reasons. We still re-read A once per column-block pass, and the tiles are
small because L1 is 64 KiB, the accumulator dominates the budget. A larger `m`
would cut B's re-fetch further, but `m` also sets padding granularity, and on
49-pixel layers padding costs more than the traffic does.

**"Isn't 67 GB/s low for LPDDR5x?"**
It is what the NPU's shim DMAs sustain, which is the roof that applies to us -
the memory controller is shared with the CPU and iGPU. The fitted asymptote is
70.7 GB/s; I used the best directly-measured sustained value, 67, because it is
the more conservative of the two and it does not depend on a fit.

**"Why measure a roof instead of using the spec?"**
Because the spec number would put the diagonal about 2× higher and make every
one of our points look bandwidth-comfortable, which would be flattering and
wrong.

**"Do the arrows match the 6.45× on the swap slide?"**
Not exactly, and deliberately: the swap slide isolates the mapping with the tile
held at n=64 on both sides. Here the green points are the full shipping
configuration, swap *and* the tuned tile, against the plain default-tile
baseline, so the displacement includes the tuning gain too.
