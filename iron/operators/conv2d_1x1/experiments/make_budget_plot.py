#!/usr/bin/env python3
"""Cycle-budget figure: where the array's cycles actually go.

Separate from make_plots.py so the two can be edited independently.
Reads cycle_budget.json (produced by cycle_budget.py).
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
D = json.loads((HERE / "cycle_budget.json").read_text())

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8a84"
GRID = "#e2e2dd"
S1, S2 = "#2a78d6", "#eb6834"
OTHER = "#b5b5ae"          # neutral: "everything else", not a series

FS_VALUE, FS_TICK, FS_AXLABEL = 20, 20, 24
FS_TITLE, FS_SUB, FS_LEGEND, FS_ANNOT = 30, 16, 20, 17


def totals(tag):
    u = pad = res = 0.0
    for v in D.values():
        if tag not in v:
            continue
        b = v[tag]
        u += b["useful_cyc"]; pad += b["padding_cyc"]; res += b["residual_cyc"]
    return u / 1e6, pad / 1e6, res / 1e6


plain, prod = totals("plain"), totals("prod")
labels = ["plain mapping", "with the cols→M swap"]
data = [plain, prod]

fig, ax = plt.subplots(figsize=(17, 10), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.yaxis.grid(True, color=GRID, linewidth=1.0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)

xs = [0, 1]
bottoms = [0.0, 0.0]
segs = [("useful  — MACs the answer needs", S1, 0),
        ("padding — MACs on zeros", S2, 1),
        ("residual — no MAC issued", OTHER, 2)]

for name, color, idx in segs:
    vals = [data[i][idx] for i in (0, 1)]
    ax.bar(xs, vals, 0.5, bottom=bottoms, color=color, zorder=3, linewidth=0)
    for i, v in enumerate(vals):
        share = v / sum(data[i]) * 100
        if share >= 4:
            ax.text(xs[i], bottoms[i] + v / 2, f"{v:.2f}M   {share:.0f}%",
                    ha="center", va="center", fontsize=FS_VALUE,
                    color="white" if color != OTHER else INK, fontweight="600", zorder=5)
    bottoms = [bottoms[i] + vals[i] for i in (0, 1)]

for i in (0, 1):
    ax.text(xs[i], bottoms[i] + 0.35, f"{bottoms[i]:.2f}M cycles",
            ha="center", va="bottom", fontsize=FS_VALUE + 2, color=INK, fontweight="600")

ax.annotate("", xy=(0.72, prod[0] + prod[1] + prod[2] + 0.15),
            xytext=(0.28, plain[0] + plain[1] + plain[2] + 0.15),
            arrowprops=dict(arrowstyle="->", color=INK_3, lw=2.5,
                            connectionstyle="arc3,rad=-0.25"), zorder=6)
ax.text(0.5, 14.6, "2.3× fewer cycles", ha="center", va="bottom",
        fontsize=FS_ANNOT + 2, color=INK_3, fontweight="600")

ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=FS_AXLABEL, color=INK)
ax.set_ylabel("array-cycles, all 22 layers  (millions)", fontsize=FS_AXLABEL,
              color=INK_2, labelpad=14)
ax.set_ylim(0, 19)
ax.set_xlim(-0.55, 1.55)

ax.text(0, 1.115, "Where the cycles go", transform=ax.transAxes,
        fontsize=FS_TITLE, color=INK, fontweight="600")
ax.text(0, 1.065, "One array-cycle = 32 cores × 32 MAC/cyc. Summed over all 22 real-model "
                  "layers, production tiles, RyzenAI-npu4 at 8 columns.",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)
ax.text(0, 1.022, "The swap all but removes the padding bucket. What is left is residual — "
                  "and that is now the whole story.",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)

handles = [Patch(facecolor=c, edgecolor="none", label=n) for n, c, _ in segs]
leg = ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND,
                loc="upper right", bbox_to_anchor=(1.0, 0.99), labelspacing=0.8)
for t in leg.get_texts():
    t.set_color(INK_2)

fig.savefig(HERE / "fig_d_cycle_budget.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print(f"plain: useful {plain[0]:.2f}M  padding {plain[1]:.2f}M  residual {plain[2]:.2f}M")
print(f"prod : useful {prod[0]:.2f}M  padding {prod[1]:.2f}M  residual {prod[2]:.2f}M")
print("wrote fig_d_cycle_budget.png")
