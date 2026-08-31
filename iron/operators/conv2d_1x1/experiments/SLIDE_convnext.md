# ConvNeXt-T deep-dive slide, drop-in copy

Figure: `fig_j_convnext.png`. Two panels: linear-scale roofline for this one
layer, and the itemised loss chain. It is dense: this is a slide you talk
through, not one you flash.

Suggested position: **replace or follow the cycle-budget slide.** It is the same
argument that slide makes, with the residual term actually measured instead of
left over.

---

## Title

**ConvNeXt-T pw_up: where the other 71% goes**

## Bullets

- On a linear scale the gap is honest: we ship **1077 GF/s** against this
  layer's own ceiling of **2868 GF/s**
- Every term in the gap is now measured. The one that used to be a residual -
  the microkernel, was **traced on a single core**
- The GEMM microkernel itself reaches **79%** of the datapath; the swap's
  in-register transposes cost a further **15%**; the real loop nest runs at
  **20.9 of 32 MAC/cycle = 65%**
- The chain closes to **+0.6%**, and only **2.2%** is left unattributed. One
  core running the real loop explains **97.8%** of the array's steady state
- The two biggest losses, **padding 865 GF/s** and the **fixed per-call cost 722 GF/s** -
  are both because the layer has 49 pixels, not because of the kernel

---

## Speaker notes

Left panel first, and note the axes are linear, not log. On the roofline slide
the log scale was right for showing 22 layers across two decades, but it
flatters us: it visually compresses the exact gap I want to talk about. Here the
distance from the green dot to the roof is the missed performance, to scale.
1077 shipped, 2868 available. That is the 71%.

Right panel is that gap, itemised. Read the colours: dark grey is exact
arithmetic, blue is a hardware trace, amber is a fit, orange is the one term I
did not measure directly.

Padding first, and it is exact, not estimated. The tile quantum on the row axis
is m times four, so 64. The layer has 49 pixels. Fifteen of every 64 rows are
zeros. That is 0.766, and it costs 865 GF/s.

Then the blue terms, which are new. I could not trace the full array: the trace
stream will not route alongside 32 cores' worth of DMAs, and it fails at eight
columns and at four. So I shrank the problem instead of the instrumentation: one
core, one column, the same compiled kernel object, the same 16-by-64-by-128
tile. The kernel already carries event markers around its compute body, and one
bracket is exactly one A-times-B accumulate step, the same step the real design
issues twelve times per output tile. The ideal is 4096 cycles.

Row-major, with both operands already in L1 so no DMA can interfere: 5151
cycles in the bracket, 5176 counting the loop branch. That is 79%, and it is the
GEMM microkernel's own floor, load/MAC issue
balance, plus about 128 cycles of memory stall. Every operator built on this
GEMM inherits it, and it lines up with the 77% the stock GEMM reaches on a big
square shape.

Turn on the column-major flags, that is our swap, the in-register transposes -
and it goes to 6111. So the swap costs 15% of the kernel. I think that is worth
saying out loud: our contribution is not free, and this is the first time we can
price it. It buys back far more in occupancy than it costs here.

Then everything the core does around the matmuls rather than in them, which I
have put in one bar called "feeding the kernel": an operand lock pair every
k-step, plus clearing the f32 accumulator and converting it out once per output
tile. Running the real nest, three tiles of twelve steps, gives 6271 cycles.
zero and convert carry event markers of their own, so the trace prices them
directly: 390 and 1293 cycles once per tile. At one core the data movement is
essentially fully hidden; what shows up is that fixed per-tile work.

Orange is the residual, and I want to be precise about it: it is what is left
over, not an attribution. One core running the real loop nest sits at 6271
cycles per step; the 32-core array sits at 6410. So the single-core model
accounts for 97.8% of the array, and 2.2% is unexplained. Contention for L2 and
DRAM is the obvious candidate, the probe is one core, so it cannot see any -
but I have not measured it, so I am not going to label the bar as contention.

And the fixed cost per call, from the intercept of latency against work, 0.600, 722
GF/s. Concretely: we sweep C_out and fit latency = a + b·work. Everything that
scales with the problem, including all the DRAM traffic, goes into b, so a is
the part of a dispatch that does not depend on how much work it carries, about
85 to 93 microseconds across two independent sweeps, with R-squared above 0.999.
It hurts here precisely because this layer is small: 85 microseconds of fixed
cost against 128 microseconds of work.

Multiply it out: 1084 predicted, 1077 measured. Six tenths of a percent.

The conclusion I would draw is the same one as the rest of the talk, but now
it is quantified rather than asserted: the kernel and its transposes cost 931
GF/s between them, and padding plus the fixed per-call cost together cost 1587. The dominant
losses are both consequences of the layer being tiny. The engine is slowest when
it is empty.

---

## If asked

**"Is the single-core number really representative of the array?"**
To 97.8%, measured, that is the whole point of running the real loop nest
rather than a synthetic one. It is the same kernel object, the same tile, the
same loop structure. What one core cannot show is anything that only exists at
32 cores, and that is exactly the 2.2% I left as a residual.

**"Why did the full-array trace not work?"**
Routing. The trace packets need a path to a shim, and with all eight columns'
DMAs already placed the router cannot find one. Both the 8- and 4-column
attempts fail at compile time.

**"79%, is that bad?"**
It is the shared kernel's steady state, not something our operator introduced.
The stock IRON GEMM measures ~77% at its best on a large shape, so we are seeing
the same ceiling from the other direction. Raising it means changing the
microkernel's load/MAC schedule, which is a different project from ours.

**"Could you reduce the transpose cost?"**
Probably, by having the DMA deliver the operands in the block order the mmul
wants rather than transposing in-register. That trades the 15% against DMA
pattern complexity and a possible burst-length penalty, so it needs measuring,
not guessing.

**"1077 or 1028?"**
1077 is the session the launch-floor fit was built on, so the whole chain is
internally consistent with it; the 3-session median for the same configuration
is 1028. Run-to-run spread on this layer is about 5%, because it is short enough
that host-side noise moves it. Both numbers are on the slide.
