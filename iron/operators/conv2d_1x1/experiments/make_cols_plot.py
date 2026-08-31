#!/usr/bin/env python3
"""Throughput vs AIE column count: the swap is what makes columns worth having."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
D = json.loads((HERE / "col_scaling.json").read_text())

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84", "#e2e2dd"
GREY, GREEN = "#9a9a93", "#1baf7a"
FS_V, FS_T, FS_AX, FS_TI, FS_SUB, FS_LEG, FS_AN = 22, 21, 25, 32, 19, 21, 19

cols = ["1", "2", "4", "8"]
plain = [D[c]["plain"]["gf"] for c in cols]
pads = [D[c]["plain"]["pad"] for c in cols]
swap = [D[c]["auto"]["gf"] if D[c]["auto"]["swap"] else None for c in cols]

fig, ax = plt.subplots(figsize=(17, 10), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.yaxis.grid(True, color=GRID, linewidth=1.0); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK_2, length=0, labelsize=FS_T)

xs = range(len(cols)); w = 0.36
ax.bar([x - w/2 for x in xs], plain, w, color=GREY, zorder=3, linewidth=0)
for x, v, p in zip(xs, plain, pads):
    ax.text(x - w/2, v + 16, f"{v:.0f}", ha="center", va="bottom", fontsize=FS_V,
            color=INK_2, fontweight="600")


for x, v in zip(xs, swap):
    if v is None:
        ax.text(x + w/2, 175, "swap needs\n≥ 4 columns", ha="center", va="bottom",
                fontsize=FS_AN - 2, color=INK_3, linespacing=1.4, style="italic")
    else:
        ax.bar([x + w/2], [v], w, color=GREEN, zorder=3, linewidth=0)
        ax.text(x + w/2, v + 16, f"{v:.0f}", ha="center", va="bottom", fontsize=FS_V,
                color=INK_2, fontweight="600")

# the headline ratio: best plain anywhere -> swap at 8 columns
bi = plain.index(max(plain))
ax.annotate("", xy=(3 + w/2, swap[3] + 60), xytext=(bi - w/2, max(plain) + 60),
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=3.0,
                            connectionstyle="arc3,rad=-0.22"), zorder=6)
ax.text(2.0, swap[3] * 0.93, f"{swap[3]/max(plain):.1f}×", ha="center", va="bottom",
        fontsize=40, color=GREEN, fontweight="700")
ax.text(2.0, swap[3] * 0.86, "vs the best plain result", ha="center", va="top",
        fontsize=FS_AN, color=INK_2)

ax.set_xticks(list(xs))
ax.set_xticklabels([""] * len(cols))
for i, c in enumerate(cols):
    ax.text(i, -0.02, c, transform=ax.get_xaxis_transform(), ha="center", va="top",
            fontsize=FS_AX, color=INK, clip_on=False)
    ax.text(i, -0.075, f"{4*int(c)} cores", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=FS_SUB - 2, color=INK_2, clip_on=False)
    ax.text(i, -0.128, f"plain pads {pads[i]:.1f}×", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=FS_SUB - 2, color=GREY, clip_on=False,
            fontweight="600")
ax.set_xlabel("AIE columns used", fontsize=FS_AX, color=INK_2, labelpad=86)
ax.set_ylabel("throughput  (GF/s)", fontsize=FS_AX, color=INK_2, labelpad=14)
ax.set_ylim(0, 1250); ax.set_xlim(-0.6, len(cols) - 0.4)

leg = ax.legend(handles=[Patch(facecolor=GREY, label="plain mapping"),
                         Patch(facecolor=GREEN, label="cols→M swap")],
                frameon=False, fontsize=FS_LEG, loc="upper left",
                bbox_to_anchor=(0.005, 0.99), labelspacing=0.8)
for t in leg.get_texts(): t.set_color(INK_2)

ax.text(0, 1.10, "The swap is what makes extra columns worth having",
        transform=ax.transAxes, fontsize=FS_TI, color=INK, fontweight="600")
ax.text(0, 1.055, "ConvNeXt-T pw_up 768→3072 @7² · tile 16/64/128 throughout, so only the "
                  "column count changes",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)
ax.text(0, 1.015, "Plain pads the 49 pixels up to tile_n × columns — so every column doubles "
                  "the padding and cancels the core it added.",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)

fig.savefig(HERE / "fig_h_columns.png", facecolor=SURFACE, bbox_inches="tight")
print(f"plain {plain}\nswap  {swap}\nratio {swap[3]/max(plain):.2f}x")
print("wrote fig_h_columns.png")
