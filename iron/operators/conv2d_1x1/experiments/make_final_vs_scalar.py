#!/usr/bin/env python3
"""Complete operator (swap engaged where the heuristic fires) vs the scalar oracle,
all 22 real-model layers.

One panel. The "% of the 3.69 TF/s datapath" panel that used to sit on the right
is gone: it answered a different question, it is what fig_i2_budget.png now
decomposes properly, and carrying it here cost 38% of the width that the labels
and the bars needed more. The deep-dive band and its label are gone with it, the
headline layer having its own slides later.

Horizontal bars, unlike the vertical charts elsewhere in the deck, because 22
categories with long names do not fit under a vertical axis at any readable size.

On sizing the row labels, which is the whole difficulty here. With 22 rows the
label size is capped by the row height, and on a slide the image is scaled to
fit, so neither a taller nor a wider figure changes the size the audience sees:
the cap and the scale factor move together and cancel. The only real lever is
lines per row. Putting the model and the shape on ONE line instead of two frees
the whole row height for a single line, which is what lets the shape go to 33pt
from 24. Widening the figure is what makes room for that longer line, so the two
changes only work together.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
CB = json.loads((HERE / "cycle_budget.json").read_text())
VS = json.loads((HERE / "vec_vs_scalar.json").read_text())
HEADLINE = "768/3072/49"

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84", "#e2e2dd"
S1 = "#2a78d6"

# ---- Type scale ----------------------------------------------------------
# FS_SHAPE and FS_MODEL are capped by the row height: 22 rows over ~13 inches
# of plot area is ~44pt per row, and the two lines plus their separation have to
# live inside it. Everything else is capped only by the width, which the removed
# panel just gave back, so those go considerably larger.
FS_VALUE = 30
FS_SHAPE = 33
FS_MODEL = 23
FS_TICK = 28
FS_AXLABEL = 34
FS_TITLE = 42
FS_SUB = 24
FS_LEGEND = 30

# Axis-fraction width reserved for the shape column, so the model name can be
# right aligned just to its left and every row lines up in two clean columns.
# It has to clear the longest shape ("1024→2048 @14²", ~3.9in at 33pt); the left
# margin above then has to clear that PLUS the longest model name
# ("DenseNet-121 transition", ~4.0in at 23pt) or the names clip off the canvas.
SHAPE_SLOT = 0.262

rows = []
for k, v in CB.items():
    if "prod" not in v or k not in VS:
        continue
    p = v["prod"]
    rows.append((p["gf"] / VS[k]["scalar"], k, v["model"], v["role"], v["H"],
                 v["C_in"], v["C_out"], p["of_peak"], p["swap"]))
rows.sort(reverse=True)
ys = list(range(len(rows)))

# 16:9 on purpose. A slide scales the image to fit, so an image WIDER than 16:9
# gets scaled down by width and hands back the size the bigger type just won,
# while one narrower than 16:9 wastes slide width without gaining anything. At
# exactly 16:9 the chart fills the slide and the type is at its largest.
fig, ax = plt.subplots(figsize=(25.8, 14.5), dpi=200)
# Margins set explicitly rather than left to a tight bbox: the row labels are
# drawn outside the axes, and a tight crop expands the canvas to include them,
# which is what silently pushed the aspect to 2.11 and undid the gain.
fig.subplots_adjust(left=0.352, right=0.985, top=0.90, bottom=0.115)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.xaxis.grid(True, color=GRID, linewidth=1.2)
ax.yaxis.grid(False)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK_2, length=0, labelsize=FS_TICK)

sp = [r[0] for r in rows]
col = [S1 if r[8] else INK_3 for r in rows]

ax.barh(ys, sp, 0.64, color=col, zorder=3, linewidth=0)
for y, v, r in zip(ys, sp, rows):
    ax.text(v + 16, y, f"{v:.0f}×", va="center", ha="left", fontsize=FS_VALUE,
            color=INK if r[1] == HEADLINE else INK_2, zorder=4, fontweight="600")

ax.set_xlabel("complete operator ÷ scalar oracle", fontsize=FS_AXLABEL,
              color=INK_2, labelpad=18)
ax.set_xlim(0, 1520)
ax.set_xticks([0, 200, 400, 600, 800, 1000, 1200, 1400])
ax.set_yticks(ys)
ax.set_yticklabels([""] * len(ys))
ax.set_ylim(-0.7, len(rows) - 0.3)
ax.invert_yaxis()

# One line per row, in two right-aligned columns: model, then shape nearest the
# bar it names. Both sit on the row centre, so the whole row height is available
# to a single line.
for y, r in zip(ys, rows):
    ax.text(-0.015, y, f"{r[5]}→{r[6]} @{r[4]}²",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=FS_SHAPE, color=INK, clip_on=False,
            fontweight="600" if r[1] == HEADLINE else "normal")
    ax.text(-0.015 - SHAPE_SLOT, y, f"{r[2]} {r[3]}",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=FS_MODEL, color=INK_2, clip_on=False)

handles = [Patch(facecolor=S1, edgecolor="none",
                 label="swap engaged by the heuristic"),
           Patch(facecolor=INK_3, edgecolor="none",
                 label="heuristic declines, plain mapping")]
leg = ax.legend(handles=handles, frameon=False, fontsize=FS_LEGEND,
                loc="lower right", bbox_to_anchor=(0.995, 0.012),
                labelspacing=1.0, handletextpad=0.8)
for t in leg.get_texts():
    t.set_color(INK_2)

# Heading in FIGURE coordinates, not axes: the axes start a third of the way in
# to leave room for the row labels, and at 42pt the title does not fit in what
# is left. Anchored to the canvas it has the full width.
fig.text(0.012, 0.988, "The complete operator against the scalar reference",
         fontsize=FS_TITLE, color=INK, fontweight="600", va="top")
fig.text(0.012, 0.943, "All 22 real-model layers · swap where the heuristic "
                       "engages · RyzenAI-npu4, 8 columns, bf16",
         fontsize=FS_SUB, color=INK_2, va="top")

fig.savefig(HERE / "fig_f_final_vs_scalar.png", facecolor=SURFACE)
plt.close(fig)

sw = [r for r in rows if r[8]]
print(f"swapped layers: {len(sw)}/{len(rows)}  "
      f"speedup {min(r[0] for r in sw):.0f}-{max(r[0] for r in sw):.0f}x")
print(f"top: {rows[0][2]} {rows[0][3]}  {rows[0][0]:.0f}x")
print("wrote fig_f_final_vs_scalar.png")
