#!/usr/bin/env python3
"""The roofline slide: where the design actually lands.

Reads roofline.json (measured points, 3 sessions each) and dram_roof.json
(measured memory roof) and emits fig_i_roofline.png.

What the chart has to say, in order:
  1. Both roofs are real: 3.69 TF/s is arithmetic, and the DRAM diagonal was
     MEASURED with mem_copy, not taken from an LPDDR5x datasheet.
  2. Real 1x1 convs live LEFT of the ridge. Their ceiling is the diagonal, and
     the diagonal is set by arithmetic intensity -- which the mapping controls.
  3. The cols->M swap moves a layer up AND right: it removes re-fetch traffic,
     so it raises the layer's own ceiling before it raises its throughput.
  4. What is left over -- the gap from each point to the roof above it -- is the
     occupancy/launch-floor story the cycle-budget slide already decomposes.

Style follows make_plots.py (same tokens, same type scale) so this reads as one
deck. Palette checked with the dataviz validator: #2a78d6 vs #1baf7a separate by
dE 23.1 under protan and 24.0 under normal vision on the all-pairs list; the
aqua's 2.74:1 surface contrast takes the documented relief (legend present,
direct labels on the points that carry the argument).
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = pathlib.Path(__file__).parent
R = json.loads((HERE / "roofline.json").read_text())
ROOF = json.loads((HERE / "dram_roof.json").read_text())

# ---- Design tokens (identical to make_plots.py) -----------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8a84"
GRID = "#e2e2dd"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

FS_TICK = 17
FS_AXLABEL = 27
FS_TITLE = 25
FS_SUB = 14
FS_LEGEND = 17
FS_ANNOT = 15
FS_ROOF = 18
FS_PT = 13.5

# ---- The two roofs ----------------------------------------------------------
PEAK = 3690.0          # GF/s: 2 x 32 MAC/cyc x 32 cores x 1.8 GHz
BW = ROOF["best_single_point_gbps"]   # GB/s, measured, read+write
RIDGE = PEAK / BW      # FLOP/byte where the diagonal meets the flat roof
# Stock IRON GEMM's best measured result (4096^3, tile 64/64/64). Not a hardware
# limit -- the framework's own steady-state ceiling, which every operator built
# on this GEMM inherits, ours included.
FRAMEWORK = 2824.0

XLO, XHI = 3.2, 150.0
YLO, YHI = 70.0, 5200.0


def roof_at(ai):
    return min(PEAK, BW * ai)


def screen_angle(ax, p0, p1):
    """Angle of the data segment p0->p1 as actually drawn, in degrees.

    A label riding the DRAM diagonal has to match the slope on SCREEN, which on
    a log-log axes depends on the axes' aspect, not on the data. Hard-coding a
    guess makes the text drift off the line and collide with whatever is above
    it, so the angle is read back from the transform instead.
    """
    import math
    (x0, y0), (x1, y1) = ax.transData.transform([p0, p1])
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


# ---- Assemble the points ----------------------------------------------------
prod_swap, prod_plain, ghosts, arrows = [], [], [], []
for key, L in R["layers"].items():
    p = L["prod"]
    (prod_swap if p["swap"] else prod_plain).append((p["ai"], p["gf"], key))
    if "plain" in L:
        b = L["plain"]
        ghosts.append((b["ai"], b["gf"], key))
        arrows.append((b["ai"], b["gf"], p["ai"], p["gf"], key))

BIG = R["anchors"]["big_gemm"]


def build(bare):
    """Render the figure. `bare` drops the internal title block.

    The deck places these charts under a PowerPoint title and crops the figure's
    own heading off (compare fig_b on the swap-results slide). Emitting the bare
    variant directly means no crop step, and no risk of the crop clipping the
    top gridline.
    """
    fig, ax = plt.subplots(figsize=(21, 11.2), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(XLO, XHI)
    ax.set_ylim(YLO, YHI)

    ax.grid(True, which="major", color=GRID, linewidth=1.0)
    ax.grid(False, which="minor")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)
    ax.set_xticks([4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["4", "8", "16", "32", "64", "128"])
    ax.set_yticks([100, 250, 500, 1000, 2000, 3690])
    ax.set_yticklabels(["100", "250", "500", "1000", "2000", "3690"])

    # ---- The roofs themselves ---------------------------------------------------
    # Everything above the roof is unreachable; a very light wash says so without
    # competing with the marks.
    xs_roof = [XLO * (XHI / XLO) ** (i / 400) for i in range(401)]
    ys_roof = [roof_at(x) for x in xs_roof]
    ax.fill_between(xs_roof, ys_roof, YHI, color=INK, alpha=0.035, zorder=1,
                    linewidth=0)
    ax.plot(xs_roof, ys_roof, color=INK, lw=3.0, zorder=6, solid_capstyle="round")

    # The ridge: where the binding constraint changes hands. Its label goes in the
    # empty lower-right, and carries the reading of the whole chart -- which side of
    # it the real layers are on.
    n_left = sum(1 for p in R["layers"].values() if p["prod"]["ai"] < RIDGE)
    ax.plot([RIDGE, RIDGE], [YLO, PEAK], color=INK_3, lw=1.4, ls=(0, (5, 5)), zorder=3)
    ax.text(RIDGE * 1.06, 86,
            f"ridge: {RIDGE:.0f} FLOP/byte\n"
            f"{n_left} of {len(R['layers'])} real layers sit to the LEFT of it,\n"
            f"so the diagonal, not the flat peak, is their ceiling",
            ha="left", va="bottom", fontsize=FS_ANNOT, color=INK_2, linespacing=1.55)

    # Framework ceiling -- dashed, because it is a threshold, not a grid line.
    ax.plot([RIDGE * 0.80, XHI], [FRAMEWORK, FRAMEWORK], color=INK_3, lw=1.8,
            ls=(0, (6, 4)), zorder=5)
    ax.text(XHI * 0.97, FRAMEWORK * 1.06,
            f"IRON GEMM, best measured  {FRAMEWORK / 1000:.2f} TF/s  (77% of peak)",
            ha="right", va="bottom", fontsize=FS_ANNOT, color=INK_2)

    # Roof labels, each sitting on its own roof, each naming what it constrains.
    ax.text(XHI * 0.97, PEAK * 1.08,
            f"bf16 peak  {PEAK / 1000:.2f} TF/s   (32 cores × 32 MAC/cyc × 1.8 GHz)",
            ha="right", va="bottom", fontsize=FS_ROOF, color=INK, fontweight="600")
    roof_angle = screen_angle(ax, (5.0, BW * 5.0), (20.0, BW * 20.0))
    ax.text(4.30, roof_at(4.30) * 1.13, f"DRAM roof: {BW:.0f} GB/s, measured",
            ha="left", va="bottom", fontsize=FS_ROOF, color=INK, fontweight="600",
            rotation=roof_angle, rotation_mode="anchor")

    # ---- Marks ------------------------------------------------------------------
    # Displacement arrows first, so the marks sit on top of their tails.
    for x0, y0, x1, y1, key in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.62",
                                    color=INK_3, lw=1.7, alpha=0.62,
                                    shrinkA=9, shrinkB=11), zorder=4)

    ax.plot([g[0] for g in ghosts], [g[1] for g in ghosts], "o", markersize=13,
            markerfacecolor=SURFACE, markeredgecolor=INK_3, markeredgewidth=2.0,
            linestyle="none", zorder=7)
    ax.plot([p[0] for p in prod_plain], [p[1] for p in prod_plain], "o",
            markersize=17, color=S1, markeredgecolor=SURFACE, markeredgewidth=2.0,
            linestyle="none", zorder=8)
    ax.plot([p[0] for p in prod_swap], [p[1] for p in prod_swap], "o",
            markersize=17, color=S3, markeredgecolor=SURFACE, markeredgewidth=2.0,
            linestyle="none", zorder=8)
    ax.plot([BIG["ai"]], [BIG["gf"]], "D", markersize=16, color=INK,
            markeredgecolor=SURFACE, markeredgewidth=2.0, linestyle="none", zorder=9)

    # ---- Selective direct labels ------------------------------------------------
    # Only the points that carry the argument get a label: the headline layer at
    # both ends of its arrow, the two fastest layers, the weakest, and the anchor.
    def label(x, y, text, dx=0, dy=0, ha="left", va="center", color=INK_2,
              size=FS_PT, weight="normal"):
        ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=size, color=color, zorder=10,
                    fontweight=weight, linespacing=1.35)


    # The headline layer sits in the densest cluster, so its label goes on a leader
    # line out to clear space rather than on top of its neighbours.
    ax.annotate("ConvNeXt-T pw_up  768→3072 @7²\nthe layer the swap was built for",
                xy=(42.8, 1028), xytext=(26.5, 1620),
                fontsize=FS_PT + 1, color=INK, fontweight="600", linespacing=1.4,
                ha="center", va="bottom", zorder=11,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=1.3,
                                shrinkA=4, shrinkB=13))
    ax.annotate("...and where it started:\nplain mapping, 165 GF/s",
                xy=(5.1, 165), xytext=(4.05, 78),
                fontsize=FS_PT + 1, color=INK_2, linespacing=1.4,
                ha="left", va="bottom", zorder=11,
                arrowprops=dict(arrowstyle="-", color=INK_3, lw=1.3,
                                shrinkA=4, shrinkB=9))
    label(72.7, 1510, "YOLOv5l SPPF\n1024→512 @20²", dx=16, dy=12, color=INK_2)
    label(29.6, 156, "EfficientNet expand 80→480 @14²\nweakest layer, 8% of its roof",
          dx=0, dy=-40, ha="center", color=INK_2)
    label(BIG["ai"], BIG["gf"], "large GEMM, N=9216: the array is genuinely full\n"
          f"{BIG['gf'] / 1000:.2f} TF/s = {BIG['gf'] / PEAK * 100:.0f}% of peak, "
          "landing on the ridge",
          dx=18, dy=2, color=INK, weight="600")

    # ---- Legend -----------------------------------------------------------------
    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=13, color=S3,
               markeredgecolor=SURFACE, markeredgewidth=2.0,
               label="shipping, cols→M swap engaged  (11 layers)"),
        Line2D([], [], marker="o", linestyle="none", markersize=13, color=S1,
               markeredgecolor=SURFACE, markeredgewidth=2.0,
               label="shipping, plain mapping (heuristic declines)  (11)"),
        Line2D([], [], marker="o", linestyle="none", markersize=11,
               markerfacecolor=SURFACE, markeredgecolor=INK_3, markeredgewidth=2.0,
               label="baseline it replaced: plain mapping, default tile"),
        Line2D([], [], marker="D", linestyle="none", markersize=12, color=INK,
               markeredgecolor=SURFACE, markeredgewidth=2.0,
               label="full-array anchor (not a real conv layer)"),
    ]
    # The wedge above the DRAM diagonal and left of the ridge is unreachable, so it
    # is the one large region guaranteed to hold no data. Legend and the mechanism
    # callout live there, both on the surface colour so they read over the wash.
    leg = ax.legend(handles=handles, fontsize=FS_LEGEND, ncol=1,
                    loc="upper left", bbox_to_anchor=(0.012, 0.982),
                    handletextpad=0.9, labelspacing=1.0,
                    frameon=True, facecolor=SURFACE, edgecolor=GRID,
                    framealpha=1.0, borderpad=1.0)
    leg.get_frame().set_linewidth(1.3)
    for t in leg.get_texts():
        t.set_color(INK_2)

    # The mechanism behind every arrow is stated in the header rather than in a
    # floating box: a box large enough to hold it had nowhere to sit that did not
    # occlude either the DRAM roof label or the ridge annotation.
    ax.text(16.4, 415, "the swap", fontsize=FS_ANNOT, color=INK_2, style="italic",
            rotation=screen_angle(ax, (5.1, 165), (42.8, 1028)),
            rotation_mode="anchor", ha="left", va="bottom", zorder=10)

    # ---- Axis labels and titles -------------------------------------------------
    ax.set_xlabel("arithmetic intensity   (useful FLOP per DRAM byte actually moved, log)",
                  fontsize=FS_AXLABEL, color=INK_2, labelpad=14)
    ax.set_ylabel("throughput   (GF/s, useful conv MACs, log)",
                  fontsize=FS_AXLABEL, color=INK_2, labelpad=14)

    # Header lines are wrapped by hand: bbox_inches="tight" grows the CANVAS to fit
    # any overhanging text, so one long line would silently stretch the figure past
    # 16:9 and shrink the plot on the slide.
    if not bare:
        ax.text(0, 1.180, "Where the design sits on the roofline",
                transform=ax.transAxes, fontsize=FS_TITLE, color=INK,
                fontweight="600")
        ax.text(0, 1.108,
                "How to read it: the cols→M swap moves a layer RIGHT before it moves "
                "it UP, cutting B's re-fetch from 48× to 1×\n"
                "lifts that layer's own DRAM ceiling ~8×, and only then does filling "
                "the array cash it in.",
                transform=ax.transAxes, fontsize=FS_SUB + 1.5, color=INK,
                linespacing=1.5)
        ax.text(0, 1.022,
                "22 real 1×1 conv layers · RyzenAI-npu4, 8 columns, bf16 · median of "
                "50 timed runs × 3 sessions, all verified against torch · intensity "
                "counts the bytes actually moved, operand re-fetch included",
                transform=ax.transAxes, fontsize=FS_SUB, color=INK_3)

    suffix = "_bare" if bare else ""
    out = HERE / f"fig_i_roofline{suffix}.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out


for _bare in (False, True):
    print("wrote", build(_bare).name)

# ---- Console summary: the numbers the slide claims --------------------------
prod_all = [(L["prod"]["ai"], L["prod"]["gf"], k) for k, L in R["layers"].items()]
fr = sorted((gf / roof_at(ai) * 100, k) for ai, gf, k in prod_all)
left = sum(1 for ai, _, _ in prod_all if ai < RIDGE)
print(f"  DRAM roof      {BW:.1f} GB/s (measured)   ridge {RIDGE:.1f} FLOP/byte")
print(f"  left of ridge  {left} of {len(fr)} layers -> the diagonal is their binding roof")
print(f"  utilisation    {fr[0][0]:.0f}% ({fr[0][1]}) to {fr[-1][0]:.0f}% ({fr[-1][1]}) "
      f"of the applicable roof")
print(f"  anchor         {BIG['gf']:.0f} GF/s = {BIG['gf'] / PEAK * 100:.0f}% of peak "
      f"at AI {BIG['ai']:.1f}")
peak_bw = max(p["gbps"] for L in R["layers"].values()
              for p in (L.get("prod"), L.get("plain")) if p)
print(f"  busiest layer reached {peak_bw:.1f} GB/s = {peak_bw / BW * 100:.0f}% of the "
      f"DRAM roof -> the pipe is never saturated")
