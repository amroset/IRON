#!/usr/bin/env python3
"""The roofline slide, split in two and stripped down.

fig_i_roofline.png tried to be one chart and carried two arguments at once: the
roof shape, and where each layer's throughput goes. With 11 swap arrows, four
callouts and a three-line subtitle on top of it, the roof itself stopped reading
as a roof. So it is two figures now, each with one job and nothing else:

  fig_i1_roofline.png   the roof, the ridge, and where the 22 layers sit.
  fig_i2_budget.png     where the peak goes for the headline layer.

Design notes worth keeping:

  * The unreachable region above the roof is SHADED. A roofline reads as a roof
    only when you can see the thing it is a ceiling on; an unfilled polyline just
    reads as two trend lines meeting.
  * The x-range runs well past the ridge so the flat segment is a third of the
    width. In the old chart the flat part was short and jammed under the title,
    which is most of why it did not look like a roofline.
  * ONE swap arrow, on the headline layer, instead of eleven. The swap has its
    own slide; here it only has to be recognisable, not enumerated.
  * The budget bars encode evidence with TEXTURE, not hue. The old five-colour
    scheme failed the palette validator (orange vs amber, normal-vision dE 13.7,
    under the 15 floor). Two hues pass with dE 24.0 and carry the same meaning.

Run:  python make_roofline_split.py
"""
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
R = json.loads((HERE / "roofline.json").read_text())
ROOF = json.loads((HERE / "dram_roof.json").read_text())
CB = json.loads((HERE / "convnext_budget.json").read_text())

# ---- Tokens (shared with make_plots.py so the deck reads as one set) --------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8a84"
GRID = "#e2e2dd"
BLUE, GREEN, ORANGE = "#2a78d6", "#1baf7a", "#eb6834"
FORBIDDEN = "#ebebe6"

# Fonts: this is a projected slide, so everything is a step larger than the
# on-screen figures. Nothing here is below 24pt.
FS_TITLE = 44
FS_AX = 36
FS_TICK = 30
FS_LEGEND = 30
FS_ROOF = 33
FS_NOTE = 28
FS_VAL = 30

PEAK = 3690.0                          # 2 x 32 MAC/cyc x 32 cores x 1.8 GHz
BW = ROOF["best_single_point_gbps"]    # measured with mem_copy, read+write
RIDGE = PEAK / BW
HEADLINE = "768/3072/49"               # the layer the swap was built for


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=1.2)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)


def screen_angle(ax, p0, p1):
    """Slope of a data segment as DRAWN. On log-log it depends on the axes
    aspect, so a hard-coded rotation drifts off the line."""
    (x0, y0), (x1, y1) = ax.transData.transform([p0, p1])
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


# =============================================================================
# Figure 1: the roofline, linear axes, one layer
# =============================================================================
def roofline_one_layer():
    """Linear axes on purpose: the vertical gap is the point of this chart.

    A log-log roofline is the right form for placing 22 shapes at once, but it
    flatters every one of them -- a 2.8x shortfall is a small step on a log
    axis. With one layer there is no need to compress anything, so the axes are
    linear and the distance from the point to the roof above it is to scale.
    That gap is exactly what fig_i2_budget.png then decomposes.
    """
    RL = CB["roofline"]
    XHI, YHI = 80.0, 4150.0

    fig, ax = plt.subplots(figsize=(19, 10.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.set_xlim(0, XHI)
    ax.set_ylim(0, YHI)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_yticks([0, 1000, 2000, 3000, 3690])

    # The roof, and the region it forbids.
    xs = np.linspace(0, XHI, 600)
    roof = np.minimum(PEAK, BW * xs)
    ax.fill_between(xs, roof, YHI, color=FORBIDDEN, zorder=0)
    ax.plot(xs, roof, color=INK, linewidth=5.5, solid_capstyle="round", zorder=6)

    ax.text(67.5, PEAK + 110, "bf16 peak  3.69 TF/s", ha="center", va="bottom",
            fontsize=FS_ROOF, fontweight="bold", color=INK, zorder=7)
    ang = screen_angle(ax, (10.0, BW * 10.0), (30.0, BW * 30.0))
    ax.text(13.0, BW * 13.0 + 90, "DRAM  67 GB/s", rotation=ang,
            rotation_mode="anchor", ha="left", va="bottom", fontsize=FS_ROOF,
            fontweight="bold", color=INK, zorder=7)

    ax.axvline(RIDGE, color=INK_3, linewidth=2.2, linestyle=(0, (6, 5)), zorder=2)
    ax.text(RIDGE + 1.0, 120, "ridge", ha="left", va="bottom",
            fontsize=FS_NOTE, color=INK_2, style="italic", zorder=7)

    ai_p, gf_p = RL["ai_plain"], RL["gf_plain"]
    ai_s, gf_s = RL["ai_swap"], RL["gf_swap"]
    ceil_s = RL["ceiling_swap_gf"]

    # How far below its own ceiling the shipping layer sits.
    ax.plot([ai_s, ai_s], [gf_s, ceil_s], color=ORANGE, linewidth=5.0,
            solid_capstyle="butt", zorder=4)
    ax.text(ai_s + 1.8, (gf_s + ceil_s) / 2, f"{ceil_s - gf_s:.0f} GF/s\nmissing",
            ha="left", va="center", fontsize=FS_NOTE, fontweight="bold",
            color=ORANGE, zorder=7)
    ax.text(ai_s - 1.6, ceil_s + 40, f"its ceiling  {ceil_s:.0f}", ha="right",
            va="bottom", fontsize=FS_NOTE, color=INK_2, zorder=7)

    # The swap, on this one layer.
    ax.annotate("", xy=(ai_s - 1.4, gf_s - 30), xytext=(ai_p + 1.4, gf_p + 8),
                arrowprops=dict(arrowstyle="-|>,head_width=0.42,head_length=0.85",
                                color=INK_2, linewidth=3.2,
                                shrinkA=0, shrinkB=0), zorder=3)
    ax.text((ai_p + ai_s) / 2 - 1, (gf_p + gf_s) / 2 - 150, "cols→M swap",
            ha="center", va="top", fontsize=FS_NOTE, color=INK_2,
            style="italic", zorder=7)

    ax.scatter([ai_p], [gf_p], s=560, facecolors="none", edgecolors=INK_3,
               linewidths=3.4, zorder=5)
    ax.text(ai_p + 2.4, gf_p + 45, f"plain  {gf_p:.0f}", ha="left", va="center",
            fontsize=FS_NOTE, color=INK_2, zorder=7,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))

    ax.scatter([ai_s], [gf_s], s=560, c=GREEN, edgecolors=SURFACE,
               linewidths=2.6, zorder=5)
    # Masked background: this label reaches past the ridge line, and a dashed
    # rule running through the digits is worse than a small opaque box.
    ax.text(ai_s + 1.8, gf_s - 60, f"shipping  {gf_s:.0f}", ha="left", va="top",
            fontsize=FS_NOTE, fontweight="bold", color=INK, zorder=7,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2.0))

    ax.set_xlabel("arithmetic intensity   (FLOP per DRAM byte)",
                  fontsize=FS_AX, color=INK, labelpad=14)
    ax.set_ylabel("throughput   (GF/s)", fontsize=FS_AX, color=INK, labelpad=14)

    fig.tight_layout(pad=1.9)
    fig.savefig(HERE / "fig_i1b_one_layer.png", facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# Figure 1: what the swap did, across every layer it fired on
# =============================================================================
def roofline_swap():
    """What the swap did, on the four layers it did the most for.

    Log axes here, deliberately, unlike the single-layer chart. The quantity on
    show is a RATIO, and on log axes equal ratios are equal distances, so the
    arrows are comparable. The price is that the gap to the roof is compressed,
    which is why fig_i1b_one_layer.png exists and keeps linear axes.

    Points come from swap_points.json, i.e. the SAME n=64 runs fig_b plots, so
    the gains printed here are the 6.45x / 4.58x / 4.08x / 3.27x quoted there.
    The earlier version read roofline.json, whose green points are the shipping
    configuration (swap AND the tuned tile) and which therefore disagreed with
    the swap slide by up to 31%. See swap_points.py for why.
    """
    SP = json.loads((HERE / "swap_points.json").read_text())
    XLO, XHI, YLO, YHI = 3.8, 190.0, 80.0, 5600.0

    fig, ax = plt.subplots(figsize=(19, 10.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
    ax.set_yticks([100, 250, 500, 1000, 2000, 3690])
    ax.set_yticklabels(["100", "250", "500", "1000", "2000", "3690"])

    xs = np.logspace(math.log10(XLO), math.log10(XHI), 600)
    roof = np.minimum(PEAK, BW * xs)
    ax.fill_between(xs, roof, YHI, color=FORBIDDEN, zorder=0)
    ax.plot(xs, roof, color=INK, linewidth=5.5, solid_capstyle="round", zorder=6)

    ax.text(115, math.sqrt(PEAK * YHI), "bf16 peak  3.69 TF/s", ha="center",
            va="center", fontsize=FS_ROOF, fontweight="bold", color=INK, zorder=7)
    ang = screen_angle(ax, (5.0, BW * 5.0), (20.0, BW * 20.0))
    ax.text(6.6, BW * 6.6 * 1.16, "DRAM  67 GB/s", rotation=ang,
            rotation_mode="anchor", ha="left", va="bottom", fontsize=FS_ROOF,
            fontweight="bold", color=INK, zorder=7)

    ax.plot([RIDGE, RIDGE], [YLO, PEAK], color=INK_3, linewidth=2.2,
            linestyle=(0, (6, 5)), zorder=2)
    ax.text(RIDGE * 0.94, YLO * 1.10, "ridge", ha="right", va="bottom",
            fontsize=FS_NOTE, color=INK_2, style="italic", zorder=7)

    # The four after-points sit within a factor of 1.9 in y, so a label beside
    # each one overlaps its neighbour. The labels are a stack to the right at a
    # common x instead, ordered as the points are ordered vertically, with a
    # hairline from each point to its row. Gain first so the green numbers form
    # a column, then the layer's name.
    X_GAIN, X_NAME = 46.0, 70.0   # 70 clears "×6.45" plus its mask
    STACK_Y = {"768/3072/49": 1330.0, "1024/2048/49": 880.0,
               "3072/768/49": 620.0, "512/2048/49": 420.0}

    for r in SP:
        bf, af = r["before"], r["after"]
        ax.annotate("", xy=(af["ai"], af["gf"]), xytext=(bf["ai"], bf["gf"]),
                    arrowprops=dict(
                        arrowstyle="-|>,head_width=0.40,head_length=0.80",
                        color=INK_2, linewidth=3.6,
                        shrinkA=12, shrinkB=13), zorder=3)
        ax.plot([af["ai"] * 1.07, X_GAIN * 0.97],
                [af["gf"], STACK_Y[r["key"]]], color=INK_3, linewidth=1.6,
                zorder=2, solid_capstyle="round")

    ax.scatter([r["before"]["ai"] for r in SP], [r["before"]["gf"] for r in SP],
               s=430, facecolors="none", edgecolors=INK_3, linewidths=3.4,
               zorder=5)
    ax.scatter([r["after"]["ai"] for r in SP], [r["after"]["gf"] for r in SP],
               s=500, c=GREEN, edgecolors=SURFACE, linewidths=2.6, zorder=5)

    # The names, not the channel counts: the point of showing them is to remind
    # the room these are layers out of real networks.
    for r in SP:
        ly = STACK_Y[r["key"]]
        ax.text(X_GAIN, ly, f"×{r['gain']:.2f}", ha="left", va="center",
                fontsize=FS_ROOF, fontweight="bold", color=GREEN, zorder=7,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax.text(X_NAME, ly, f"{r['model']} {r['role']}", ha="left", va="center",
                fontsize=FS_NOTE, color=INK, zorder=7,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.text(X_GAIN, 250, "all four @7²", ha="left", va="center",
            fontsize=FS_NOTE, color=INK_2, style="italic", zorder=7)

    ax.set_xlabel("arithmetic intensity   (FLOP per DRAM byte)",
                  fontsize=FS_AX, color=INK, labelpad=14)
    ax.set_ylabel("throughput   (GF/s)", fontsize=FS_AX, color=INK, labelpad=14)

    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=20,
               markerfacecolor="none", markeredgecolor=INK_3,
               markeredgewidth=3.0, label="before the swap"),
        Line2D([], [], marker="o", linestyle="", markersize=21,
               markerfacecolor=GREEN, markeredgecolor=SURFACE, label="after"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=FS_LEGEND,
                    frameon=True, framealpha=1.0, borderpad=0.9,
                    labelspacing=0.62, handletextpad=0.7)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    for tx in leg.get_texts():
        tx.set_color(INK)

    fig.tight_layout(pad=1.9)
    fig.savefig(HERE / "fig_i1_roofline.png", facecolor=SURFACE)
    plt.close(fig)


# =============================================================================
# Figure 2: the cycle budget
# =============================================================================
def budget():
    chain = CB["chain"]
    start = chain[0]["after_gf"]
    steps = chain[1:]
    measured = CB["measured_gf"]

    # One short word each: eight two-line labels collide at this type size.
    SHORT = {"padding 49 -> 64 px": "padding",
             "GEMM microkernel": "GEMM\nkernel",
             "cols->M transposes": "transposes",
             "feeding the kernel": "feeding",
             "unattributed": "residual",
             "fixed cost per call": "per-call\ncost"}

    fig, ax = plt.subplots(figsize=(19, 10.7), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)

    labels = ["bf16\npeak"] + [SHORT.get(s["stage"], s["stage"]) for s in steps] \
             + ["what\nships"]
    x = np.arange(len(labels))

    # Start and end columns.
    ax.bar(x[0], start, width=0.66, color=INK_3, zorder=3)
    ax.bar(x[-1], measured, width=0.66, color=GREEN, zorder=3)

    top = start
    for i, s in enumerate(steps, start=1):
        drop = s["lost_gf"]
        hatched = s["how"] != "exact" and s["how"] != "traced"
        ax.bar(x[i], drop, bottom=top - drop, width=0.66, color=BLUE,
               hatch="///" if hatched else None,
               edgecolor=SURFACE if hatched else "none",
               linewidth=0.0, zorder=3)
        # Connector to the next bar.
        ax.plot([x[i] - 0.33, x[i] + 0.33 + (0.34 if i < len(steps) else 0.34)],
                [top - drop, top - drop], color=INK_3, linewidth=1.8,
                linestyle=(0, (4, 4)), zorder=2)
        ax.text(x[i], top + 90, f"−{drop:.0f}", ha="center",
                va="bottom", fontsize=FS_VAL, fontweight="bold", color=INK)
        top -= drop

    ax.text(x[0], start + 90, f"{start:.0f}", ha="center",
            va="bottom", fontsize=FS_VAL, fontweight="bold", color=INK)
    ax.text(x[-1], measured + 90, f"{measured:.0f}",
            ha="center", va="bottom", fontsize=FS_VAL, fontweight="bold",
            color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FS_TICK - 4, color=INK)
    ax.set_ylim(0, start * 1.16)
    ax.set_yticks([0, 1000, 2000, 3000, 3690])
    ax.set_ylabel("throughput   (GF/s)", fontsize=FS_AX, color=INK, labelpad=14)

    handles = [Patch(facecolor=BLUE, label="measured"),
               Patch(facecolor=BLUE, hatch="///", edgecolor=SURFACE,
                     label="fitted or residual")]
    leg = ax.legend(handles=handles, loc="upper right", fontsize=FS_LEGEND,
                    frameon=True, framealpha=1.0, borderpad=0.9,
                    labelspacing=0.62)
    leg.get_frame().set_facecolor(SURFACE)
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.tight_layout(pad=1.9)
    fig.savefig(HERE / "fig_i2_budget.png", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    roofline_swap()
    roofline_one_layer()
    budget()
    print("wrote fig_i1_roofline.png, fig_i1b_one_layer.png, fig_i2_budget.png")
