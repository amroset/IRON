#!/bin/bash
# Re-measure every sweep at 10 warmup / 50 timed iterations, then redraw figures.
set -x
PY=/scratch/amrosetti/ironenv/bin/python
cd /scratch/amrosetti/IRON/iron/operators/conv2d_1x1_opt/experiments
export CONV_COLS=8
$PY swap_vs_smalln.py   > rerun_swap.log      2>&1
$PY cycle_budget.py     > rerun_budget.log    2>&1
$PY tuning_gain.py      > rerun_tuning.log    2>&1
$PY vector_vs_scalar.py > rerun_vecscalar.log 2>&1
$PY final_budget.py     > rerun_final.log     2>&1

# Roofline. traffic_model is a pure check (no hardware) and gates the rest: if
# the byte model no longer agrees with design.py's own taps, every arithmetic
# intensity on the roofline is wrong, so stop rather than redraw a wrong chart.
$PY traffic_model.py    > rerun_traffic.log   2>&1 || { echo TRAFFIC_MODEL_FAILED; exit 1; }
$PY dram_roof.py        > rerun_dramroof.log  2>&1   # measures the memory roof
CONV_REPEATS=3 $PY roofline.py > rerun_roofline.log 2>&1

# One-core hardware trace of the microkernel. Supplies the term final_budget.py
# had to leave lumped inside `sustained`, so it must run BEFORE convnext_budget.
$PY kernel_probe.py     > rerun_kernelprobe.log 2>&1
$PY launch_floor.py     > rerun_launchfloor.log 2>&1   # intercept + swap-off control
$PY dispatch_floor.py   > rerun_dispatchfloor.log 2>&1 # per-call cost at ~zero work
$PY null_dispatch.py    > rerun_nulldispatch.log 2>&1 # what that per-call cost IS
$PY convnext_budget.py  > rerun_convnext.log    2>&1   # pure composition, no NPU

for s in make_plots.py make_final_vs_scalar.py make_tuning_plot.py make_cascade_plot.py make_budget_plot.py make_roofline_plot.py make_roofline_split.py make_convnext_plot.py; do
  $PY $s > rerun_fig_$s.log 2>&1
done
echo ALL_DONE
