#!/usr/bin/env python3
"""Final-slide cascade: where the ceiling goes, for the headline layer.

ConvNeXt-T pw_up 768->3072 @7x7, swap engaged, tuned tile 16/64/128, 8 columns.
Reads final_budget.json.
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
B = json.loads((HERE / "final_budget.json").read_text())["cols8"]
# scalar reference for the same layer, so the headline ratio tracks re-runs
_SCALAR = json.loads((HERE / "vec_vs_scalar.json").read_text())["768/3072/49"]["scalar"]

SURFACE, INK, INK_2, INK_3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8a84", "#e2e2dd"
S1, S2, OTHER = "#2a78d6", "#eb6834", "#b5b5ae"

PEAK = 3.69  # TF/s
pad, sus, flo = B["padding"], B["sustained"], B["floor"]
meas = B["headline"]["gf"] / 1000.0

stages = [
    ("nominal\nceiling",            PEAK,                 None,  None),
    ("after padding",               PEAK * pad,           pad,   "49 px → 64 rows"),
    ("after kernel\n+ buffer handling", PEAK * pad * sus,  sus,   "fit slope"),
    ("after launch\nfloor",         PEAK * pad * sus * flo, flo, "fit intercept"),
]
vals = [s[1] for s in stages]
colors = [S1, S1, S1, S2]

fig, ax = plt.subplots(figsize=(19, 10.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
ax.yaxis.grid(True, color=GRID, linewidth=1.0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK_2, length=0, labelsize=21)

xs = range(len(stages))
ax.bar(xs, vals, 0.52, color=colors, zorder=3, linewidth=0)
for x, v in zip(xs, vals):
    # Unit lives on the y-axis label; repeating "TFLOP/s" under every number
    # just crowds it now that the type is larger.
    ax.text(x, v + 0.06, f"{v:.2f}", ha="center", va="bottom",
            fontsize=30, color=INK, fontweight="600")

# multiply arrows + loss shares between stages
for i in range(1, len(stages)):
    factor, name = stages[i][2], stages[i][3]
    loss = (vals[i - 1] - vals[i]) / PEAK * 100
    xm = i - 0.5
    ax.annotate("", xy=(i - 0.28, vals[i] + 0.55), xytext=(i - 0.72, vals[i - 1] + 0.55),
                arrowprops=dict(arrowstyle="->", color=INK_3, lw=2.2), zorder=6)
    ax.text(xm, vals[i - 1] + 0.66, f"×{factor:.3f}", ha="center", va="bottom",
            fontsize=24, color=INK, fontweight="600")


ax.axhline(meas, color=S2, lw=2.2, ls=(0, (6, 4)), zorder=2)
ax.text(len(stages) - 0.42, meas, f"  measured  {meas:.2f} TF/s",
        va="center", ha="left", fontsize=21, color=S2, fontweight="600", clip_on=False)

ax.set_xticks(list(xs))
ax.set_xticklabels([""] * len(stages))
for i, st in enumerate(stages):
    ax.text(i, -0.02, st[0], transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=23, color=INK, linespacing=1.5, clip_on=False)
    if st[3]:
        ax.text(i, -0.145, st[3], transform=ax.get_xaxis_transform(), ha="center",
                va="top", fontsize=17, color=INK_2, clip_on=False)
ax.set_ylabel("throughput  (TFLOP/s)", fontsize=25, color=INK_2, labelpad=14)
ax.set_ylim(0, 5.4)
ax.set_xlim(-0.6, len(stages) - 0.25)

ax.text(0, 1.135, "Where the ceiling goes", transform=ax.transAxes,
        fontsize=32, color=INK, fontweight="600")
ax.text(0, 1.088, f"ConvNeXt-T pw_up  768→3072 @7²  ·  swap engaged, tuned tile 16/64/128, "
                  f"8 columns  ·  {B['headline']['gf'] / _SCALAR:.0f}× the scalar reference",
        transform=ax.transAxes, fontsize=20, color=INK_2)
ax.text(0, 1.038, f"Cascade predicts {vals[-1] / PEAK * 100:.1f}% of peak; measured "
                  f"{meas / PEAK * 100:.1f}%.  Work-size fit R² = {B['r2']:.4f}, "
                  f"launch floor {B['a_sec'] * 1e6:.0f} µs.",
        transform=ax.transAxes, fontsize=20, color=INK_2)

# --- inset: the fit that two of the four factors are read off ---------------
pts = sorted(B["points"], key=lambda r: r["P"])
xs_p = [r["P"] / 1e6 for r in pts]
ys_p = [r["sec"] * 1e6 for r in pts]
a_us, b_us = B["a_sec"] * 1e6, B["b_sec_per_mac"] * 1e6

ins = ax.inset_axes([0.615, 0.60, 0.365, 0.345])
ins.set_facecolor(SURFACE)
for sp in ("top", "right"):
    ins.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ins.spines[sp].set_color(INK_3)
ins.grid(True, color=GRID, linewidth=0.8)
ins.set_axisbelow(True)

x_line = [0] + xs_p
ins.plot(x_line, [a_us + b_us * x * 1e6 for x in x_line], color=S1, lw=2.4, zorder=2)
ins.plot(xs_p, ys_p, "o", markersize=10, color=S1, zorder=4, linestyle="none")
# mark the layer this whole slide is about, and split its latency into the
# two parts the last two bars are read from
head = next(r for r in pts if r["C_out"] == 3072)
xh, yh = head["P"] / 1e6, head["sec"] * 1e6
ins.axhline(a_us, color=S2, lw=1.6, ls=(0, (5, 4)), zorder=2)
ins.annotate("", xy=(xh, yh), xytext=(xh, a_us),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.8), zorder=6)
ins.text(xh - 6, (a_us + yh) / 2, f"b·W\n{yh - a_us:.0f} µs", ha="right", va="center",
         fontsize=14, color=INK, fontweight="600", linespacing=1.3, zorder=6)
ins.plot([xh], [yh], "o", markersize=15, color=S2, zorder=7, linestyle="none")
ins.text(xh + 8, yh + 6, "this layer", ha="left", va="bottom", fontsize=14,
         color=S2, fontweight="600", zorder=7)
ins.plot([0], [a_us], "o", markersize=11, color=S2, zorder=5,
         linestyle="none", clip_on=False)
ins.annotate(f"a = {a_us:.0f} µs\nlaunch floor", xy=(0, a_us), xytext=(30, a_us * 0.30),
             fontsize=15, color=S2, fontweight="600", linespacing=1.4,
             arrowprops=dict(arrowstyle="->", color=S2, lw=1.8))
ins.text(0.97, 0.10, f"slope b → {1 / (B['b_sec_per_mac'] * 1.8e9) / 32:.1f} MAC/cyc/core",
         transform=ins.transAxes, ha="right", va="bottom", fontsize=15,
         color=S1, fontweight="600")
ins.text(0.03, 0.93, f"latency = a + b·W        R² = {B['r2']:.4f}",
         transform=ins.transAxes, ha="left", va="top", fontsize=15, color=INK)
ins.set_xlabel("work  (M padded MACs)", fontsize=14, color=INK_2, labelpad=4)
ins.set_ylabel("latency (µs)", fontsize=14, color=INK_2, labelpad=4)
ins.tick_params(colors=INK_2, length=0, labelsize=13)
ins.set_xlim(0, max(xs_p) * 1.06)
ins.set_ylim(0, max(ys_p) * 1.12)

fig.savefig(HERE / "fig_e_cascade.png", facecolor=SURFACE, bbox_inches="tight")
plt.close(fig)

print(f"padding {pad:.4f}  sustained {sus:.4f}  floor {flo:.4f}")
print(f"cascade {vals[-1]:.3f} TF/s ({vals[-1] / PEAK * 100:.2f}%)  measured {meas:.3f} "
      f"({meas / PEAK * 100:.2f}%)")
for i in range(1, len(stages)):
    print(f"  loss {stages[i][3]:<24} {(vals[i - 1] - vals[i]) / PEAK * 100:5.1f}% of peak")
print("wrote fig_e_cascade.png")
