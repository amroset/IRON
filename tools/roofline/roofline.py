#!/usr/bin/env python3
"""Generate a publication-quality roofline plot.

Usage: python3 tools/roofline/roofline.py --data tools/roofline/data.json --out images/roofline.png

The data file should define a `system` object with `peak_flops_gflops`
and `peak_bandwidth_gbps`, and a `points` list with entries that include
`name`, `intensity` (FLOP/Byte) and `performance_gflops` (achieved).
"""
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_roofline(system, points, out_path, dpi=200):
    peak_flops = float(system["peak_flops_gflops"])  # GFLOP/s
    peak_bw = float(system["peak_bandwidth_gbps"])  # GB/s (GByte/s)

    sns.set(style="whitegrid")
    plt.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

    # arithmetic intensity range
    ai = np.logspace(-2, 3, 500)

    # roofline: min(peak_flops, ai * peak_bw)
    roof = np.minimum(peak_flops, ai * peak_bw)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.loglog(ai, roof, color="#1f77b4", linewidth=3, label="Roofline")

    # plot horizontal line for peak flops and slope line for bandwidth
    ax.hlines(peak_flops, ai[0], ai[-1], colors="#1f77b4", linestyles="dashed", linewidth=1)
    ax.plot(ai, ai * peak_bw, color="#ff7f0e", linewidth=2, alpha=0.8, label="Bandwidth limit")

    # fill area under roof for visual
    ax.fill_between(ai, roof, ai * 0 + ax.get_ylim()[0], color="#1f77b4", alpha=0.06)

    # Plot points
    for p in points:
        name = p.get("name")
        intensity = float(p.get("intensity"))
        perf = float(p.get("performance_gflops"))
        ax.scatter(intensity, perf, s=120, label=name)
        ax.annotate(name, (intensity, perf), xytext=(6, -6), textcoords="offset points")

    # Nicify axes and labels
    ax.set_xlabel("Arithmetic Intensity (FLOP / Byte)")
    ax.set_ylabel("Performance (GFLOP/s)")
    ax.set_title("Roofline Plot")

    ax.grid(which="both", linestyle="--", linewidth=0.4, alpha=0.6)
    ax.legend(loc="lower right")

    # Tight layout and save
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    print(f"Saved roofline plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSON data file")
    parser.add_argument("--out", required=True, help="Output image path (png)")
    args = parser.parse_args()

    with open(args.data, "r") as f:
        cfg = json.load(f)

    system = cfg["system"]
    points = cfg.get("points", [])

    make_roofline(system, points, args.out)


if __name__ == "__main__":
    main()
