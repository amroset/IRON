#!/usr/bin/env python3
"""Roofline points for the swap slide, on the SAME measurement fig_b uses.

fig_b quotes the swap at 6.45x, and the roofline used to quote 6.2x for the same
layer. Both were right and they measured different things:

  fig_b       swap vs plain with the tile held at n=64 on both sides, so the
              only difference is the mapping. results.json.
  roofline    the full shipping configuration (swap AND the tuned tile) against
              the plain default tile, so the displacement carried the tuning
              gain too. roofline.json.

For three of the four layers that made the roofline number LARGER (4.78 vs 4.58,
4.41 vs 4.07, 4.28 vs 3.27). For the headline layer it came out smaller anyway,
which is run-to-run noise: the plain baselines are traffic-bound and the traffic
path is where the variance lives (3072/768 differs by 31% between the two
sessions on an identical configuration).

Two numbers for one thing is a bad slide, so this recomputes the roofline points
from the n=64 runs. The arrows and the labels then come from one experiment, and
the chart says what it is actually about: the mapping, tile held constant.

No hardware. The timings are already in results.json; only the arithmetic
intensity has to be recomputed, because the byte model depends on the tile and
these runs use n=64 rather than the tuned tile.

Run:  python swap_points.py
"""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent.parent))

from iron.operators.conv2d_1x1_opt.op import Conv2d1x1          # noqa: E402
from traffic_model import layer_point                            # noqa: E402

TILE = (16, 64, 64)     # held constant on both sides: this is the point
COLS = 8
N_TOP = 4


def point(C_in, C_out, pixels, swap, sec):
    H = int(round(pixels ** 0.5))
    tm, tk, tn = TILE
    op = Conv2d1x1(batch=1, H=H, W=H, C_in=C_in, C_out=C_out,
                   tile_m=tm, tile_k=tk, tile_n=tn, num_aie_columns=COLS,
                   use_scalar=False, prio_accuracy=True,
                   emulate_bf16_mmul_with_bfp16=False, swap_mn=swap)
    return layer_point(op, C_in, C_out, pixels, sec)


def main():
    RS = json.loads((HERE / "results.json").read_text())
    RL = json.loads((HERE / "roofline.json").read_text())["layers"]

    rows = []
    for key, r in RS.items():
        if "swap_n64" not in r or "plain_n64" not in r:
            continue
        if not r["swap_n64"].get("swap"):
            continue
        C_in, C_out, pixels = (int(x) for x in key.split("/"))
        meta = RL.get(key, {})
        before = point(C_in, C_out, pixels, False,
                       float(r["plain_n64"]["note"].rstrip("us")) * 1e-6)
        after = point(C_in, C_out, pixels, True,
                      float(r["swap_n64"]["note"].rstrip("us")) * 1e-6)
        rows.append({"key": key, "model": meta.get("model", "?"),
                     "role": meta.get("role", "?"), "pixels": pixels,
                     "before": before, "after": after,
                     # Taken from the stored GF/s, not recomputed, so the label
                     # is bit-for-bit the number fig_b prints.
                     "gain": r["swap_n64"]["gf"] / r["plain_n64"]["gf"]})

    rows.sort(key=lambda r: -r["gain"])
    top = rows[:N_TOP]
    (HERE / "swap_points.json").write_text(json.dumps(top, indent=2))

    print(f"{'layer':>16}{'model':>24}{'AI before':>11}{'AI after':>10}"
          f"{'GF before':>11}{'GF after':>10}{'gain':>8}")
    for r in top:
        print(f"{r['key']:>16}{r['model'] + ' ' + r['role']:>24}"
              f"{r['before']['ai']:11.2f}{r['after']['ai']:10.2f}"
              f"{r['before']['gf']:11.0f}{r['after']['gf']:10.0f}"
              f"{r['gain']:7.2f}x")
    print(f"\nwrote swap_points.json ({len(top)} of {len(rows)} swapped layers)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
