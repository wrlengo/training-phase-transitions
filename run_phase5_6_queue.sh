#!/bin/bash
# Sequential training queue for theory-validation round (tasks 1-4).
# Total ~22h on this MPS machine; every run flushes CSV per checkpoint.
# Analyses are attempted after each stage; failure of an analysis never
# blocks the next training stage.
set +e
cd /Users/wangrui/Documents/trae_projects/grokking_lar_experiments
PY=/Users/wangrui/miniconda3/envs/quantum_env/bin/python

echo "##### STAGE A: dense eta scan (8 lrs x 3 seeds x 200k) #####"
$PY -u phase5_theory_scan.py \
  --lrs 0.02 0.025 0.035 0.04 0.06 0.07 0.08 0.15 \
  --seeds 0 1 2 --total_steps 200000 --log_every 1000
$PY -u phase5_theory_analysis.py || echo "STAGE A analysis FAILED"
$PY -u phase5d_critical_dense.py || echo "dense critical fit FAILED"

echo "##### STAGE C: MLP order-parameter search (3 seeds x 100k) #####"
$PY -u phase4b_mlp_order_params.py --seeds 0 1 2 --total_steps 100000 \
  --eval_every 500

echo "##### STAGE B: finite-size scan width 4 (3 lrs x 3 seeds x 200k) #####"
$PY -u phase3_free_energy.py --num_channels 4 \
  --lrs 0.02 0.03 0.05 --seeds 0 1 2 --total_steps 200000 \
  --out_dir phase6_finite_size/c4

echo "##### STAGE B: finite-size scan width 12 (3 lrs x 3 seeds x 200k) #####"
$PY -u phase3_free_energy.py --num_channels 12 \
  --lrs 0.02 0.03 0.05 --seeds 0 1 2 --total_steps 200000 \
  --out_dir phase6_finite_size/c12
$PY -u phase6_fss_analysis.py || echo "FSS analysis FAILED"

echo "##### QUEUE COMPLETE #####"
