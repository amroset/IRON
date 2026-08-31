#!/usr/bin/env python3
"""Result plots for the conv2d_1x1_opt talk.

Reads results.json (swap experiment, 11 layers x 4 configs) and
vec_vs_scalar.json (vectorization, 22 layers x 2 paths) and emits three
slide-ready PNGs. Every bar is identified by its actual layer -- shape AND
source model -- never by a positional index.

Type scale is sized for projection: numbers, axes and annotations are large;
the model name under each bar is the one deliberately smaller element, since
it is context rather than the reading itself.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
RESULTS = json.loads((HERE / "results.json").read_text())

# ---- Design tokens (dataviz reference palette, light mode) ------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8a84"
GRID = "#e2e2dd"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# ---- Type scale ------------------------------------------------------------
FS_VALUE = 15    # the numbers on the marks -- most important
FS_TICK = 15     # axis tick numbers
FS_SHAPE = 15    # the layer's shape, i.e. its identity
FS_MODEL = 9.5   # source model -- the deliberate exception
FS_AXLABEL = 30
FS_TITLE = 24
FS_SUB = 13.5
FS_LEGEND = 25
FS_ANNOT = 14

# ---- The layer roster for the swap charts ----------------------------------
# (results.json key, C_in->C_out, spatial, model, role, N, swapped in prod?)
LAYERS = [
    ("768/3072/49",   "768→3072",  "7²",  "ConvNeXt-T",   "pw_up",      49, True),
    ("1024/2048/49",  "1024→2048", "7²",  "ResNeXt-50",   "expand",     49, True),
    ("512/2048/49",   "512→2048",  "7²",  "ResNet-50",    "deep exp",   49, True),
    ("2048/512/49",   "2048→512",  "7²",  "ResNet-50",    "deep red",   49, True),
    ("3072/768/49",   "3072→768",  "7²",  "ConvNeXt-T",   "pw_down",    49, True),
    ("1152/192/49",   "1152→192",  "7²",  "EfficientNet", "project",    49, True),
    ("384/1536/196",  "384→1536",  "14²", "ConvNeXt-T",   "pw_up",     196, True),
    ("112/672/196",   "112→672",   "14²", "MobileNetV3",  "expand",    196, True),
    ("80/480/196",    "80→480",    "14²", "EfficientNet", "expand",    196, True),
    ("1024/512/196",  "1024→512",  "14²", "DenseNet-121", "transition", 196, True),
    ("1024/512/400",  "1024→512",  "20²", "YOLOv5l",      "SPPF",      400, False),
]

CONFIGS = [
    ("plain_n64", "plain, n=64  (baseline)", S1),
    ("plain_n16", "plain, n=16  (finer columns)", S2),
    ("swap_n64",  "swap,  n=64  (production)", S3),
    ("swap_n16",  "swap,  n=16", S4),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)


def two_line_xlabels(ax, chans, spatials, models):
    """Three stacked lines, each with its own size: channel transformation big,
    spatial size medium, source model small. Drawn as text rather than tick
    labels so the sizes can differ, and so long names never force a shrink."""
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([""] * len(chans))
    for i, (ch, sp, mo) in enumerate(zip(chans, spatials, models)):
        ax.text(i, -0.018, ch, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_SHAPE, color=INK, clip_on=False)
        ax.text(i, -0.062, sp, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_SHAPE - 3, color=INK, clip_on=False)
        ax.text(i, -0.104, mo, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_MODEL, color=INK_2, clip_on=False)


def n_bands(ax, y):
    """Annotate the three pixel-count regimes -- the mechanism behind the story."""
    bands, start = [], 0
    for i in range(1, len(LAYERS) + 1):
        if i == len(LAYERS) or LAYERS[i][5] != LAYERS[start][5]:
            bands.append((start, i - 1, LAYERS[start][5]))
            start = i
    for a, b, n in bands:
        ax.plot([a - 0.42, b + 0.42], [y, y], color=INK_3, lw=1.2,
                clip_on=False, zorder=5)
        ax.text((a + b) / 2, y * 1.03, f"N = {n} px", ha="center", va="bottom",
                fontsize=FS_ANNOT, color=INK_2, clip_on=False, zorder=5)


# ---------------------------------------------------------------------------
# Chart B runs at projector sizes and drops the forced bar, so it needs its own
# label helpers: the shared ones hard-code the small type scale that suits the
# dense charts, and packing the model name onto one 9.5pt line is unreadable
# once the figure is on a wall.
# ---------------------------------------------------------------------------
FS_B_BAND = 34    # "N = 49 px" -- the mechanism label, called out to be largest
FS_B_SHAPE = 30   # the channel transformation, i.e. the layer's identity
FS_B_SPAT = 26
FS_B_MODEL = 20
FS_B_VALUE = 32
FS_B_TICK = 28
FS_B_AX = 34
FS_B_TITLE = 38
FS_B_SUB = 22


# Ten slots across the plot width leaves roughly 120pt per label, so a 12
# character name like "EfficientNet" caps the type at about 17pt. Wrapping the
# long ones is what buys the size back; only these need it.
WRAP_MODEL = {"EfficientNet": "Efficient\nNet",
              "MobileNetV3": "MobileNet\nV3"}


def wrap_model(m):
    """Two lines, breaking at the hyphen where there is one.

    Every name then fits in nine characters or fewer, which is what keeps the
    model line at 20pt instead of the 17pt that "ConvNeXt-T" beside
    "ResNeXt-50" would force.
    """
    if m in WRAP_MODEL:
        return WRAP_MODEL[m]
    if "-" in m:
        head, tail = m.rsplit("-", 1)
        return f"{head}\n-{tail}"
    return m


def stacked_xlabels_big(ax, chans, spatials, models, roles):
    """Shape (two lines), spatial size, then model and role as one block.

    The shape is split at the arrow so no line exceeds five characters, which is
    what lets it sit at 30pt in a 120pt slot. Model and role are emitted as a
    SINGLE text object rather than two, so a wrapped name simply makes that
    block taller instead of colliding with the line beneath it.
    """
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels([""] * len(chans))
    for i, (ch, sp, mo, ro) in enumerate(zip(chans, spatials, models, roles)):
        src, dst = ch.split("→")
        ax.text(i, -0.030, f"{src}\n→{dst}", transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_B_SHAPE, color=INK,
                clip_on=False, linespacing=1.15)
        ax.text(i, -0.200, sp, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=FS_B_SPAT, color=INK,
                clip_on=False)
        ax.text(i, -0.272, f"{wrap_model(mo)}\n{ro}",
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=FS_B_MODEL, color=INK_2, clip_on=False,
                linespacing=1.35)


def n_bands_big(ax, y, layers):
    """The pixel-count regimes, over whatever subset of layers is plotted."""
    bands, start = [], 0
    for i in range(1, len(layers) + 1):
        if i == len(layers) or layers[i][5] != layers[start][5]:
            bands.append((start, i - 1, layers[start][5]))
            start = i
    for a, b, n in bands:
        ax.plot([a - 0.42, b + 0.42], [y, y], color=INK_3, lw=1.8,
                clip_on=False, zorder=5)
        ax.text((a + b) / 2, y * 1.02, f"N = {n} px", ha="center", va="bottom",
                fontsize=FS_B_BAND, color=INK_2, clip_on=False, zorder=5,
                fontweight="600")


CHANS_L  = [sh for _, sh, _, _, _, _, _ in LAYERS]
SPATS_L  = [f"@{sp}" for _, _, sp, _, _, _, _ in LAYERS]
MODELS_L = [f"{m} {r}" for _, _, _, m, r, _, _ in LAYERS]
xs = range(len(LAYERS))

# ============================================================================
# Chart A -- absolute throughput, all four configurations
# ============================================================================
fig, ax = plt.subplots(figsize=(21, 10.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
style_axes(ax)

width = 0.205
for ci, (key, label, color) in enumerate(CONFIGS):
    off = (ci - 1.5) * width
    vals = [RESULTS[k][key]["gf"] for k, *_ in LAYERS]
    ax.bar([x + off for x in xs], vals, width * 0.9, label=label,
           color=color, zorder=3, linewidth=0)
    for x, v in zip(xs, vals):
        ax.text(x + off, v + 18, f"{v:.0f}", ha="center", va="bottom",
                fontsize=FS_VALUE - 3, color=INK_2, rotation=90, zorder=4,
                fontweight="600")

two_line_xlabels(ax, CHANS_L, SPATS_L, MODELS_L)
ax.set_ylabel("throughput  (GF/s, useful conv MACs)", fontsize=FS_AXLABEL,
              color=INK_2, labelpad=12)
ax.set_ylim(0, 1640)
ax.set_xlim(-0.62, len(LAYERS) - 0.38)
n_bands(ax, 1500)

ax.text(0, 1.135, "Does a finer column quantum substitute for the swap?",
        transform=ax.transAxes, fontsize=FS_TITLE, color=INK, fontweight="600")
ax.text(0, 1.088, "RyzenAI-npu4 · 8 AIE columns · bf16 · median of 50 timed runs, "
                  "all configurations verified against the torch golden reference",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)

leg = ax.legend(frameon=False, fontsize=FS_LEGEND, ncol=4, loc="upper right",
                bbox_to_anchor=(1.0, 1.075), handlelength=1.5, handleheight=0.9,
                columnspacing=2.4)
for t in leg.get_texts():
    t.set_color(INK_2)
    t.set_family("monospace")

fig.savefig(HERE / "fig_a_throughput.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)


# ============================================================================
# Chart B -- what the swap actually buys, per layer
#
# Only the layers that SHIP with the swap appear. The forced bar that used to
# sit at the right (YOLOv5l SPPF, 0.77x) measured what the swap would cost on a
# layer the heuristic declines, which is a different question from this one and
# was the only thing on the chart needing a legend. With it gone there is a
# single series, so the legend goes too.
# ============================================================================
LAYERS_B = [L for L in LAYERS if L[6]]
ratios_b = [RESULTS[k]["swap_n64"]["gf"] / RESULTS[k]["plain_n64"]["gf"]
            for k, *_ in LAYERS_B]
xs_b = range(len(LAYERS_B))

fig, ax = plt.subplots(figsize=(21, 12.0), dpi=200)
fig.patch.set_facecolor(SURFACE)
style_axes(ax)
ax.tick_params(colors=INK_2, length=0, labelsize=FS_B_TICK)

ax.bar(xs_b, ratios_b, 0.6, color=S1, zorder=3, linewidth=0)
for x, v in zip(xs_b, ratios_b):
    ax.text(x, v + 0.10, f"{v:.2f}×", ha="center", va="bottom",
            fontsize=FS_B_VALUE, color=INK_2, zorder=4, fontweight="600")

ax.axhline(1.0, color=INK_3, lw=2.0, ls=(0, (5, 4)), zorder=2)
ax.text(len(LAYERS_B) - 0.45, 1.0, "  parity", va="center", ha="left",
        fontsize=FS_B_SPAT, color=INK_3, clip_on=False)

stacked_xlabels_big(ax,
                    [sh for _, sh, _, _, _, _, _ in LAYERS_B],
                    [f"@{sp}" for _, _, sp, _, _, _, _ in LAYERS_B],
                    [m for _, _, _, m, _, _, _ in LAYERS_B],
                    [r for _, _, _, _, r, _, _ in LAYERS_B])
ax.set_ylabel("swap ÷ plain baseline  (both n=64)", fontsize=FS_B_AX,
              color=INK_2, labelpad=14)
ax.set_ylim(0, 7.6)
ax.set_xlim(-0.62, len(LAYERS_B) - 0.38)
n_bands_big(ax, 7.05, LAYERS_B)

ax.text(0, 1.150, "What the cols→M swap buys, layer by layer",
        transform=ax.transAxes, fontsize=FS_B_TITLE, color=INK, fontweight="600")
ax.text(0, 1.098, "Gains concentrate where the columns are starved.",
        transform=ax.transAxes, fontsize=FS_B_SUB, color=INK_2)

fig.savefig(HERE / "fig_b_swap_gain.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)


# ============================================================================
# Chart C -- vectorized aie::mmul vs the scalar oracle, all 22 shapes
# Horizontal: 22 layers with two-line names need the room only a y-axis gives.
# ============================================================================
VS = json.loads((HERE / "vec_vs_scalar.json").read_text())
BF16_PEAK = 3690.0  # GF/s = 2 x 32 MAC/cyc x 32 cores x 1.8 GHz

rows = sorted(VS.items(), key=lambda kv: -kv[1]["speedup"])
sp = [v["speedup"] for _, v in rows]
sc = [v["scalar"] for _, v in rows]
ve = [v["vector"] for _, v in rows]
mean_sp = sum(sp) / len(sp)
ys = list(range(len(rows)))


def style_h(ax):
    ax.set_facecolor(SURFACE)
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)


fig, (axL, axR, axP) = plt.subplots(
    1, 3, figsize=(25, 15.5), dpi=200, sharey=True,
    gridspec_kw={"width_ratios": [1, 1.05, 0.62], "wspace": 0.05})
fig.patch.set_facecolor(SURFACE)
for a in (axL, axR, axP):
    style_h(a)

axL.barh(ys, sp, 0.6, color=S1, zorder=3, linewidth=0)
for y, v in zip(ys, sp):
    axL.text(v + 5, y, f"{v:.0f}×", va="center", ha="left",
             fontsize=FS_VALUE, color=INK_2, zorder=4, fontweight="600")
axL.axvline(mean_sp, color=INK_3, lw=1.6, ls=(0, (5, 4)), zorder=2)
axL.text(mean_sp, 1.006, f"mean {mean_sp:.0f}×", ha="center", va="bottom",
         fontsize=FS_ANNOT, color=INK_3, clip_on=False,
         transform=axL.get_xaxis_transform())
axL.set_xlabel("Speedup vectorized ÷ scalar", fontsize=FS_AXLABEL, color=INK_2, labelpad=14)
axL.set_xlim(0, 295)

axL.set_yticks(ys)
axL.set_yticklabels([""] * len(ys))
axL.set_ylim(-0.7, len(rows) - 0.3)
axL.invert_yaxis()
for y, (_, v) in zip(ys, rows):
    axL.text(-0.014, y - 0.17, f"{v['C_in']}→{v['C_out']} @{v['H']}²",
             transform=axL.get_yaxis_transform(), ha="right", va="center",
             fontsize=FS_SHAPE, color=INK, clip_on=False)
    axL.text(-0.014, y + 0.24, f"{v['model']} {v['role']}",
             transform=axL.get_yaxis_transform(), ha="right", va="center",
             fontsize=FS_MODEL, color=INK_2, clip_on=False)

for y, s, v in zip(ys, sc, ve):
    axR.plot([s, v], [y, y], color=GRID, lw=3.0, zorder=2, solid_capstyle="round")
axR.plot(ve, ys, "o", markersize=13, color=S1, zorder=4, linestyle="none",
         label="vectorized  aie::mmul")
axR.plot(sc, ys, "o", markersize=13, color=S2, zorder=4, linestyle="none",
         label="scalar oracle")
axR.axvline(BF16_PEAK, color=INK_3, lw=1.5, ls=(0, (2, 3)), zorder=2)
axR.text(BF16_PEAK, 1.006, "bf16 peak 3.69 TF/s", ha="center", va="bottom",
         fontsize=FS_ANNOT, color=INK_3, clip_on=False,
         transform=axR.get_xaxis_transform())

axR.set_xscale("log")
axR.set_xlim(0.45, 9000)
axR.set_xlabel("throughput  (GF/s, log scale)", fontsize=FS_AXLABEL, color=INK_2, labelpad=14)
axR.set_xticks([1, 10, 100, 1000])
axR.set_xticklabels(["1", "10", "100", "1000"])
axR.xaxis.grid(True, which="major", color=GRID, linewidth=1.0)
axR.xaxis.grid(False, which="minor")

leg = axR.legend(frameon=False, fontsize=FS_LEGEND, loc="lower right",
                 bbox_to_anchor=(0.995, 0.012), handletextpad=0.6, labelspacing=0.9)
for t in leg.get_texts():
    t.set_color(INK_2)

# --- third panel: how much of the vector datapath each layer actually uses ---
# The vector unit retires 32 MAC/cyc/core; peak = that x 32 cores x 1.8 GHz x 2.
# Even the best layer reaches ~36%, and the 7x7 tail sits near 3% -- the array
# is idle, which is a mapping problem rather than a vectorization one.
pct = [v / BF16_PEAK * 100 for v in ve]
axP.barh(ys, pct, 0.6, color=S1, zorder=3, linewidth=0)
for y, p in zip(ys, pct):
    axP.text(p + 0.9, y, f"{p:.0f}%" if p >= 10 else f"{p:.1f}%",
             va="center", ha="left", fontsize=FS_VALUE, color=INK_2,
             zorder=4, fontweight="600")
axP.set_xlim(0, 52)
axP.set_xticks([0, 10, 20, 30, 40])
axP.set_xlabel("% of datapath", fontsize=FS_AXLABEL, color=INK_2, labelpad=14)
axP.text(0.5, 1.006, "vectorized, of 32 MAC/cyc", ha="center", va="bottom",
         fontsize=FS_ANNOT, color=INK_3, clip_on=False, transform=axP.transAxes)

axL.text(0.008, 1.058, "Why vectorize at all: aie::mmul vs the scalar oracle",
         fontsize=FS_TITLE, color=INK, fontweight="600", transform=axL.transAxes)
axL.text(0.008, 1.033, "All 22 real-model shapes · plain mapping in both cases, so the "
                       "comparison isolates vectorization · RyzenAI-npu4, 8 columns, bf16",
         fontsize=FS_SUB, color=INK_2, transform=axL.transAxes)

fig.savefig(HERE / "fig_c_vectorization.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print("wrote fig_a_throughput.png, fig_b_swap_gain.png, fig_c_vectorization.png")
print(f"  chart C: {len(rows)} shapes, mean {mean_sp:.0f}x, "
      f"range {min(sp):.0f}-{max(sp):.0f}x")
