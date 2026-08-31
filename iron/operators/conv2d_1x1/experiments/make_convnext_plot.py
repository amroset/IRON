#!/usr/bin/env python3
"""ConvNeXt-T pw_up on one slide: where it sits, and where the rest went.

Two panels, deliberately on LINEAR axes. The log roofline is right for showing
22 layers across two decades, but wrong for one layer, because it visually
compresses exactly the gap this chart is about. On a linear scale the distance
from the point to the roof is the missed performance, to scale.

  left   the roofline zoomed on this layer: the plain to swap displacement, and
         the vertical gap from the shipping point up to its own DRAM ceiling.
  right  that gap, itemised. Every term is measured; the colour says HOW,
         because "we think it is the kernel" and "we traced the kernel" are
         different claims.

Type is sized for projection, so the text budget is tight: anything the
presenter can say out loud is not on the chart. The prose lives in
SLIDE_convnext.md instead.

Palette checked with the dataviz validator (all-pairs): worst CVD dE 9.1, worst
normal-vision dE 22.9. The two sub-3:1 hues take the documented relief, since
every bar is directly labelled and the legend names every category.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
B = json.loads((HERE / "convnext_budget.json").read_text())

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8a84"
GRID = "#e2e2dd"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# How each term was obtained. This is the point of the colour encoding.
HOW_COLOR = {"exact": INK_2, "traced": S1, "fitted": S4, "not measured": S2}
HOW_LABEL = {
    "exact": "exact arithmetic",
    "traced": "hardware trace, one core",
    "fitted": "fit against work",
    "not measured": "residual, not attributed",
}

# Sized for a projected slide.
FS_TICK = 21
FS_AX = 25
FS_TITLE = 31
FS_SUB = 18
FS_ANNOT = 19
FS_BAR = 20
FS_PANEL = 22

PEAK = B["peak_gf"]
R = B["roofline"]
BW = R["dram_roof_gbps"]

fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(27, 12.2), dpi=200,
    gridspec_kw={"width_ratios": [1.0, 1.85], "wspace": 0.12})
fig.patch.set_facecolor(SURFACE)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)


def screen_angle(ax, p0, p1):
    """Slope of a data segment as drawn, so a label can ride the roof line."""
    import math
    (x0, y0), (x1, y1) = ax.transData.transform([p0, p1])
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


# ===========================================================================
# LEFT: the roofline, linear, one layer
# ===========================================================================
style(axL)
axL.yaxis.grid(True, color=GRID, linewidth=1.0)
XHI = 56.0
axL.set_xlim(0, XHI)
axL.set_ylim(0, 3950)

xs = [0, PEAK / BW, XHI]
ys = [0, PEAK, PEAK]
axL.fill_between(xs, ys, 3950, color=INK, alpha=0.035, linewidth=0)
axL.plot(xs, ys, color=INK, lw=3.2, zorder=6, solid_capstyle="round")

axL.text(XHI * 0.98, PEAK + 70, f"bf16 peak  {PEAK / 1000:.2f} TF/s",
         ha="right", va="bottom", fontsize=FS_ANNOT + 3, color=INK,
         fontweight="600")
axL.text(15.0, BW * 15.0 - 130, f"DRAM  {BW:.0f} GB/s",
         ha="left", va="top", fontsize=FS_ANNOT, color=INK_2,
         rotation=screen_angle(axL, (10, BW * 10), (40, BW * 40)),
         rotation_mode="anchor")

ai_p, gf_p = R["ai_plain"], R["gf_plain"]
ai_s = R["ai_swap"]
# The same throughput the cascade was built on. roofline.json's 3-session median
# is a few percent lower; one number on the left and another on the right would
# invite exactly the question the figure answers, so the spread goes in the sub.
gf_s = B["measured_gf"]
gf_s_median = R["gf_swap"]
ceil_s = R["ceiling_swap_gf"]

axL.annotate("", xy=(ai_s, ceil_s), xytext=(ai_s, gf_s),
             arrowprops=dict(arrowstyle="<->", color=S2, lw=3.0,
                             shrinkA=3, shrinkB=3), zorder=7)
axL.text(ai_s + 1.6, (gf_s + ceil_s) / 2,
         f"missed\n{ceil_s - gf_s:.0f} GF/s", ha="left", va="center",
         fontsize=FS_ANNOT + 4, color=S2, fontweight="600", linespacing=1.35)

axL.annotate("", xy=(ai_s, gf_s), xytext=(ai_p, gf_p),
             arrowprops=dict(arrowstyle="-|>,head_width=0.34,head_length=0.75",
                             color=INK_3, lw=2.2, shrinkA=11, shrinkB=15),
             zorder=4)
axL.text((ai_p + ai_s) / 2 - 2, (gf_p + gf_s) / 2 - 150, "cols→M swap",
         ha="center", va="top", fontsize=FS_ANNOT, color=INK_2, style="italic")

axL.plot([ai_p], [gf_p], "o", ms=19, markerfacecolor=SURFACE,
         markeredgecolor=INK_3, markeredgewidth=2.6, zorder=8)
axL.plot([ai_s], [gf_s], "o", ms=21, color=S3, markeredgecolor=SURFACE,
         markeredgewidth=2.4, zorder=9)

axL.text(ai_p + 1.8, gf_p, f"plain  {gf_p:.0f}", ha="left", va="center",
         fontsize=FS_ANNOT, color=INK_2)
axL.text(ai_s + 1.6, gf_s, f"shipping\n{gf_s:.0f} GF/s", ha="left", va="center",
         fontsize=FS_ANNOT + 2, color=INK, fontweight="600", linespacing=1.35)
# Ceiling label sits left of the arrow head, in the unreachable wedge above the
# diagonal, which is the only empty region at that height.
axL.text(ai_s - 2.0, ceil_s, f"ceiling  {ceil_s:.0f} GF/s", ha="right",
         va="center", fontsize=FS_ANNOT, color=INK_2)

axL.set_xlabel("arithmetic intensity  (FLOP / DRAM byte)",
               fontsize=FS_AX, color=INK_2, labelpad=12)
axL.set_ylabel("throughput  (GF/s)", fontsize=FS_AX, color=INK_2, labelpad=12)
axL.set_yticks([0, 1000, 2000, 3000, PEAK])
axL.set_yticklabels(["0", "1000", "2000", "3000", "3690"])
axL.text(0, 1.035, "Linear axes: the gap is to scale", transform=axL.transAxes,
         fontsize=FS_PANEL, color=INK, fontweight="600")


# ===========================================================================
# RIGHT: the itemised gap
# ===========================================================================
style(axR)
axR.yaxis.grid(True, color=GRID, linewidth=1.0)

SHORT = {
    "padding 49 -> 64 px": "padding\n49→64 px",
    "GEMM microkernel": "GEMM\nkernel",
    "cols->M transposes": "cols→M\ntranspose",
    "feeding the kernel": "feeding\nthe kernel",
    "unattributed": "residual",
    "fixed cost per call": "fixed cost\nper call",
}
chain = B["chain"]
stages = chain[1:]
labels = ["bf16\npeak"] + [SHORT.get(s["stage"], s["stage"]) for s in stages] + \
         ["shipping\nmeasured"]
xs_b = range(len(labels))

axR.bar(0, PEAK, 0.64, color=INK_3, alpha=0.30, linewidth=0, zorder=3)
axR.text(0, PEAK + 60, f"{PEAK:.0f}", ha="center", va="bottom",
         fontsize=FS_BAR, color=INK, fontweight="600")

for i, s in enumerate(stages, start=1):
    lo, hi = s["after_gf"], s["before_gf"]
    color = HOW_COLOR[s["how"]]
    axR.bar(i, hi - lo, 0.64, bottom=lo, color=color, linewidth=0, zorder=3)
    axR.plot([i - 0.5, i + 0.5], [hi, hi], color=INK_3, lw=1.2, ls=(0, (4, 3)),
             zorder=2)
    axR.text(i, hi + 60, f"−{s['lost_gf']:.0f}", ha="center", va="bottom",
             fontsize=FS_BAR, color=color, fontweight="600")
    axR.text(i, lo - 75, f"×{s['factor']:.3f}", ha="center", va="top",
             fontsize=FS_BAR - 4, color=INK_2)

last = len(labels) - 1
meas = B["measured_gf"]
axR.bar(last, meas, 0.64, color=S3, linewidth=0, zorder=3)
axR.text(last, meas + 60, f"{meas:.0f}", ha="center", va="bottom",
         fontsize=FS_BAR + 3, color=INK, fontweight="600")
# Keep inside the bar's own width: white type that overhangs lands on the cream
# surface and disappears.
axR.text(last, meas / 2, f"{meas / PEAK * 100:.0f}%", ha="center", va="center",
         fontsize=FS_BAR + 5, color=SURFACE, fontweight="600")

axR.set_xticks(list(xs_b))
axR.set_xticklabels(labels, fontsize=FS_TICK - 4, color=INK_2)
axR.set_xlim(-0.62, last + 0.8)
axR.set_ylim(0, 3950)
axR.set_yticks([0, 1000, 2000, 3000, PEAK])
axR.set_yticklabels(["0", "1000", "2000", "3000", "3690"])

err = B["closure_error"] * 100
axR.text(0, 1.035, f"Every term measured, and the chain closes to {err:+.1f}%",
         transform=axR.transAxes, fontsize=FS_PANEL, color=INK, fontweight="600")

handles = [Patch(facecolor=HOW_COLOR[h], edgecolor="none", label=HOW_LABEL[h])
           for h in ("exact", "traced", "fitted", "not measured")]
handles.append(Line2D([], [], marker="s", linestyle="none", markersize=16,
                      color=S3, label="what ships"))
leg = axR.legend(handles=handles, frameon=True, facecolor=SURFACE,
                 edgecolor=GRID, framealpha=1.0, fontsize=FS_ANNOT,
                 loc="upper right", bbox_to_anchor=(0.997, 0.972),
                 handletextpad=0.8, labelspacing=0.8, borderpad=0.9)
leg.get_frame().set_linewidth(1.3)
for t in leg.get_texts():
    t.set_color(INK_2)

ku = B["kernel_utilisation"]
axR.text(0.125, 0.30,
         f"One core, real loop nest:\n"
         f"{ku['wall_production']:.0f} cyc for a "
         f"{int(ku['ideal_cycles_per_step'])}-cyc step\n"
         f"= {ku['mac_per_cycle_production']:.1f} of 32 MAC/cyc "
         f"({ku['of_32_mac_per_cycle'] * 100:.0f}%)\n"
         f"The 32-core array sits at "
         f"{ku['array_steady_state_cycles_per_step']:.0f},\n"
         f"so one core explains "
         f"{ku['wall_production'] / ku['array_steady_state_cycles_per_step'] * 100:.1f}%.",
         transform=axR.transAxes, fontsize=FS_ANNOT, color=INK_2,
         ha="left", va="top", linespacing=1.65,
         bbox=dict(boxstyle="round,pad=0.75", facecolor=SURFACE,
                   edgecolor=GRID, linewidth=1.3))

fig.text(0.005, 1.058, "ConvNeXt-T pw_up  768→3072 @7²:  where the other 71% goes",
         fontsize=FS_TITLE, color=INK, fontweight="600", va="bottom")
fig.text(0.005, 1.046,
         "cols→M swap, tile 16/64/128, 8 AIE columns, bf16.  Padding and the "
         "fixed per-call cost are both consequences of the layer having only 49 "
         "pixels, not of the kernel.",
         fontsize=FS_SUB, color=INK_2, va="top")

fig.savefig(HERE / "fig_j_convnext.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print("wrote fig_j_convnext.png")
for s in chain:
    if s["factor"]:
        print(f"  {s['stage']:<22} x{s['factor']:.4f} {s['how']:>13} "
              f"-> {s['after_gf']:7.0f} GF/s")
