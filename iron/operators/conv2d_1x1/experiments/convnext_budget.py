#!/usr/bin/env python3
"""ConvNeXt-T pw_up: the complete, fully-measured loss decomposition.

final_budget.py builds an exact identity for this layer,

    of_peak = padding x sustained x floor

but its own docstring flags the weak term: "`sustained` still lumps microkernel
efficiency, per-call overhead and multi-core contention together ... the
microkernel term needs the hardware trace window, which this design does not
yet wire up." kernel_probe.py now supplies it, from a one-core hardware trace of
the identical kernel object at the identical tile. So `sustained` splits:

    sustained = kernel_rowmaj x transposes x acquire/release x per-tile x residual

Everything but `residual` is traced. `residual` is what is left between one core
running the REAL loop nest (3 output tiles x 12 accumulate steps, with the
zero/convert pair the trace prices directly) and the 32-core array's steady
state. It is a residual and is labelled as one: this experiment does not
separate contention from anything else that only appears at 32 cores.

No hardware needed; this reads the three JSONs and composes them.

Run:  python convnext_budget.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).parent
PEAK = 3690.0  # GF/s, 2 x 32 MAC/cyc x 32 cores x 1.8 GHz

FB = json.loads((HERE / "final_budget.json").read_text())["cols8"]
try:
    LF = json.loads((HERE / "launch_floor.json").read_text())
    LF_A_US = LF["swap"]["a_us"]
except (FileNotFoundError, KeyError):
    LF = None
    LF_A_US = float("nan")
KP = json.loads((HERE / "kernel_probe.json").read_text())
RL = json.loads((HERE / "roofline.json").read_text())
DR = json.loads((HERE / "dram_roof.json").read_text())

# Wall cost per accumulate step: the period between successive matmul starts,
# so it carries the loop branches and (in production mode) the per-tile
# zero/convert pair. That is the quantity comparable to the array's steady
# state; the in-bracket median would silently drop all of it.
ideal = KP["split"]["ideal_cycles"]
c_rm = KP["resident_rowmaj"]["cycles_per_step_wall"]    # GEMM floor, L1-resident
c_cm = KP["resident_colmaj"]["cycles_per_step_wall"]    # + in-register transposes
c_st = KP["streamed_colmaj"]["cycles_per_step_wall"]    # + per-step acquire/release
PROD_KEY = next(k for k in KP if k.startswith("production"))
c_pr = KP[PROD_KEY]["cycles_per_step_wall"]             # + real 3x12 loop nest

# -- the multiplicative chain, each factor <= 1 --------------
f_padding = FB["padding"]                 # 49 -> 64 pixels; exact arithmetic
f_kernel = ideal / c_rm                   # traced
f_transpose = c_rm / c_cm                 # traced
# Operand locks (every k-step) and the accumulator clear/convert (every output
# tile) are one story from the core's point of view: cycles spent around the
# matmuls rather than in them. Reported as one term.
f_fifo = c_cm / c_st                      # traced, kept for the JSON
f_pertile = c_st / c_pr                   # traced, kept for the JSON
f_bookkeeping = c_cm / c_pr               # traced, the two together
f_sustained = FB["sustained"]             # array steady state, from the fit
f_residual = f_sustained / (f_kernel * f_transpose * f_bookkeeping)
f_floor = FB["floor"]                     # launch floor dilution, from the fit

chain = [
    ("nominal bf16 peak", None, None,
     "2 x 32 MAC/cyc x 32 cores x 1.8 GHz"),
    ("padding 49 -> 64 px", f_padding, "exact",
     "the tile quantum is m*4 = 64; 15 of every 64 pixel-rows are zeros"),
    ("GEMM microkernel", f_kernel, "traced",
     f"one core, L1-resident, row-major: {c_rm:.0f} cyc/step vs {ideal:.0f} ideal"),
    ("cols->M transposes", f_transpose, "traced",
     f"same kernel with a/b/c_col_maj: {c_cm:.0f} cyc/step: the swap's own cost"),
    ("feeding the kernel", f_bookkeeping, "traced",
     f"cycles the core spends around the matmuls rather than in them: an operand "
     f"lock pair every k-step, plus clearing the f32 accumulator and converting "
     f"it out once per output tile. Running the real 3-tiles x 12-steps nest "
     f"gives {c_pr:.0f} cyc/step against {c_cm:.0f}. zero_vectorized and "
     f"convert_copy carry their own event markers, so the trace prices them "
     f"directly: {KP[PROD_KEY]['short_brackets']} cycles once per tile"),
    ("unattributed", f_residual, "not measured",
     "the gap between one core running the real loop and the 32-core array's "
     "steady state. Candidates are inter-core contention for L2/DRAM and the "
     "shim-DMA programming the probe does not reproduce, but this experiment "
     "does not separate them: it is a residual, not an attribution"),
    ("fixed cost per call", f_floor, "fitted",
     f"a = {FB['a_sec'] * 1e6:.0f} us, the work-independent intercept of "
     f"latency = a + b*W over a C_out sweep (R2 = {FB['r2']:.4f}); "
     f"launch_floor.py reproduces it at {LF_A_US:.0f} us on an independent "
     f"sweep. Note the model only holds for THIS configuration: on the plain "
     f"mapping the same fit returns an unphysical negative intercept, so no "
     f"floor is quoted there"),
]

rows, running = [], PEAK
for name, factor, how, why in chain:
    before = running
    if factor is not None:
        running *= factor
    rows.append({"stage": name, "factor": factor, "how": how, "why": why,
                 "before_gf": before, "after_gf": running,
                 "lost_gf": before - running})

measured = FB["headline"]["gf"]
predicted = running

hl = RL["layers"]["768/3072/49"]
BW = DR["best_single_point_gbps"]
out = {
    "layer": "ConvNeXt-T pw_up 768->3072 @7x7, swap, tile 16/64/128, 8 cols",
    "peak_gf": PEAK,
    "chain": rows,
    "predicted_gf": predicted,
    "measured_gf": measured,
    "closure_error": predicted / measured - 1.0,
    "factors": {
        "padding": f_padding, "kernel": f_kernel, "transpose": f_transpose,
        "fifo": f_fifo, "per_tile": f_pertile,
        "bookkeeping": f_bookkeeping, "residual": f_residual,
        "floor": f_floor,
        "sustained_from_fit": f_sustained,
        "sustained_reconstructed":
            f_kernel * f_transpose * f_bookkeeping * f_residual,
    },
    "kernel_utilisation": {
        "ideal_cycles_per_step": ideal,
        "wall_rowmaj_resident": c_rm, "wall_colmaj_resident": c_cm,
        "wall_colmaj_streamed": c_st, "wall_production": c_pr,
        "in_bracket_production": KP[PROD_KEY]["cycles_median"],
        "short_brackets_zero_convert": KP[PROD_KEY]["short_brackets"],
        "mac_per_cycle_production": (16 * 64 * 128) / c_pr,
        "of_32_mac_per_cycle": ideal / c_pr,
        "array_steady_state_cycles_per_step": ideal / f_sustained,
        "memory_stall_cycles_per_step":
            KP[PROD_KEY]["stall_cycles"]["MEMORY_STALL"],
    },
    "launch_floor": {
        "a_us_final_budget": FB["a_sec"] * 1e6,
        "r2_final_budget": FB["r2"],
        "a_us_independent_rerun": LF_A_US,
        "plain_mapping_same_layer": (LF or {}).get("same_layer_plain", {}).get("a_us"),
        "plain_mapping_r2": (LF or {}).get("same_layer_plain", {}).get("r2"),
        "note": ("Work-independent intercept of latency = a + b*padded_MACs. "
                 "Two independent sweeps of the shipping configuration agree "
                 "(85 and 93 us, R2 >= 0.999), so the ~8% spread is the "
                 "uncertainty on this term. The SAME layer with the swap forced "
                 "off returns a negative intercept at R2 = 0.98: plain-mapping "
                 "latency is super-linear in C_out (its re-fetch traffic grows "
                 "with the problem), so the linear model does not hold there and "
                 "no floor can be extracted. What sits inside `a`: instruction "
                 "stream, array start, pipeline fill/drain: is NOT separated "
                 "by this experiment."),
    },
    "roofline": {
        "ai_swap": hl["prod"]["ai"], "gf_swap": hl["prod"]["gf"],
        "ai_plain": hl["plain"]["ai"], "gf_plain": hl["plain"]["gf"],
        "dram_roof_gbps": BW,
        "ceiling_swap_gf": min(PEAK, BW * hl["prod"]["ai"]),
        "ceiling_plain_gf": min(PEAK, BW * hl["plain"]["ai"]),
    },
}
(HERE / "convnext_budget.json").write_text(json.dumps(out, indent=2))

w = max(len(r["stage"]) for r in rows)
print(f"ConvNeXt-T pw_up 768->3072 @7x7 :  where 3.69 TF/s goes\n")
for r in rows:
    if r["factor"] is None:
        print(f"  {r['stage']:<{w}}                  {r['after_gf']:7.0f} GF/s")
        continue
    print(f"  {r['stage']:<{w}}  x{r['factor']:.4f} {r['how']:>8}  "
          f"{r['after_gf']:7.0f} GF/s   (-{r['lost_gf']:.0f})")
print(f"\n  {'predicted':<{w}}                  {predicted:7.0f} GF/s")
print(f"  {'measured':<{w}}                  {measured:7.0f} GF/s")
print(f"  closure error: {out['closure_error'] * 100:+.1f}%")
ku = out["kernel_utilisation"]
print(f"\nkernel utilisation, real loop nest on one core: "
      f"{ku['of_32_mac_per_cycle'] * 100:.1f}% of 32 MAC/cyc "
      f"= {ku['mac_per_cycle_production']:.2f} MAC/cyc "
      f"({ku['wall_production']:.0f} cyc/step)")
print(f"the 32-core array's steady state is "
      f"{ku['array_steady_state_cycles_per_step']:.0f} cyc/step, so one core "
      f"running the real loop explains "
      f"{ku['wall_production'] / ku['array_steady_state_cycles_per_step'] * 100:.1f}% "
      f"of it; the rest is the unattributed term")
print("\nwrote convnext_budget.json")
