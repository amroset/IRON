#!/usr/bin/env python3
"""What tile autotuning buys, per layer, measured against a same-session baseline.

Grouping is by whether the tuned tile_m grew past 16 -- the mechanism -- not by
whether the swap engaged, which the data does not support as an explanation.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
D = json.loads((HERE / "tuning_gain.json").read_text())

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84", "#e2e2dd"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

FS_VALUE, FS_TICK, FS_SHAPE, FS_MODEL = 19, 19, 19, 14
FS_AXLABEL, FS_TITLE, FS_SUB, FS_LEGEND, FS_ANNOT = 24, 32, 19, 20, 19

rows = sorted(((v["gain"], k, v) for k, v in D.items() if "gain" in v), reverse=True)
ys = list(range(len(rows)))
gains = [r[0] for r in rows]
grew = [r[2]["tuned"]["tile"][0] > 16 for r in rows]
# Layers whose "tuned" tile IS the default: both sides are the identical build,
# so their spread is a pure repeat-measurement control -- the noise floor.
ctrl = [r[2]["tuned"]["tile"] == r[2]["default"]["tile"] for r in rows]

c_g = [g for g, c in zip(gains, ctrl) if c]
lo, hi = min(c_g), max(c_g)
g_yes = [g for g, m, c in zip(gains, grew, ctrl) if m and not c]
mean_yes = sum(g_yes) / len(g_yes)
clears = sum(1 for g, m, c in zip(gains, grew, ctrl) if m and not c and g > hi)

fig, ax = plt.subplots(figsize=(20, 14), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.xaxis.grid(True, color=GRID, linewidth=1.0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)

ax.axvspan(lo, hi, color=INK_3, alpha=0.16, zorder=1, linewidth=0)
ax.text((lo + hi) / 2, 1.008, "noise floor",
        ha="center", va="bottom", fontsize=FS_ANNOT, color=INK_2,
        fontweight="600", clip_on=False, transform=ax.get_xaxis_transform())

for y, g, m, c in zip(ys, gains, grew, ctrl):
    if c:                                   # control: nothing was tuned
        ax.barh([y], [g], 0.62, facecolor="none", edgecolor=INK_3,
                hatch="///", linewidth=2.2, zorder=3)
    else:
        ax.barh([y], [g], 0.62, zorder=3, linewidth=0,
                color=S2 if g < 1.0 else (S1 if m else INK_3))
for y, g in zip(ys, gains):
    ax.text(g + 0.006, y, f"{g:.3f}×", va="center", ha="left", fontsize=FS_VALUE,
            color=INK_2, zorder=4, fontweight="600")

ax.axvline(1.0, color=INK_3, lw=1.8, ls=(0, (5, 4)), zorder=2)
ax.axvline(mean_yes, color=S1, lw=2.0, ls=(0, (2, 3)), zorder=2)
ax.text(mean_yes, 1.008, f"mean {mean_yes:.2f}×  m grew", ha="center", va="bottom",
        fontsize=FS_ANNOT, color=S1, fontweight="600", clip_on=False,
        transform=ax.get_xaxis_transform())

ax.set_yticks(ys)
ax.set_yticklabels([""] * len(ys))
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.invert_yaxis()
for y, r in zip(ys, rows):
    v = r[2]
    ax.text(-0.013, y - 0.18, f"{v['C_in']}→{v['C_out']} @{v['H']}²",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=FS_SHAPE, color=INK, clip_on=False)
    ax.text(-0.013, y + 0.25,
            f"{v['model']} {v['role']}   m {v['default']['tile'][0]}→{v['tuned']['tile'][0]}",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=FS_MODEL, color=INK_2, clip_on=False)

ax.set_xlabel("tuned tile ÷ default tile 16/64/64   (both on the production mapping)",
              fontsize=FS_AXLABEL, color=INK_2, labelpad=16)
ax.set_xlim(0.75, 1.42)

handles = [Patch(facecolor=S1, edgecolor="none", label="m grew past 16 — row axis absorbs it"),
           Patch(facecolor=INK_3, edgecolor="none", label="m pinned at 16 — only k or n changed"),
           Patch(facecolor="none", edgecolor=INK_3, hatch="///", linewidth=2.2,
                 label="tile unchanged — repeat of the same build (the control)"),
           Patch(facecolor=S2, edgecolor="none", label="regression — cached tile is stale")]
leg = ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND,
                loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=2,
                labelspacing=0.8, columnspacing=3.0, handletextpad=0.7)
for t in leg.get_texts():
    t.set_color(INK_2)

ax.text(0, 1.185, "A second gain: tuned tile settings", transform=ax.transAxes,
        fontsize=FS_TITLE, color=INK, fontweight="600")
ax.text(0, 1.135, f"22 real-model layers, both tiles measured in the same session.  Four layers "
                  f"kept the default tile, so their spread ({lo:.3f}–{hi:.3f}×) is a pure "
                  f"repeat-measurement control.",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)
ax.text(0, 1.098, f"Only where m grows does tuning clear that floor — {clears} of "
                  f"{len(g_yes)} such layers do, at a mean of {mean_yes:.2f}×.",
        transform=ax.transAxes, fontsize=FS_SUB, color=INK_2)

fig.savefig(HERE / "fig_g_tuning.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)
print(f"noise floor (control, n={len(c_g)}): {lo:.3f}-{hi:.3f}x")
print(f"m grew (real tuning, n={len(g_yes)}): mean {mean_yes:.3f}x, {clears} clear the floor")
print("wrote fig_g_tuning.png")
