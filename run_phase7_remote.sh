#!/bin/bash
# Phase 7: modular-addition free-energy scan on Transformer (GPU 0) and
# MLP (GPU 1), run inside tmux sessions on bio502-134, env deepseek-env.
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepseek-env
cd ~/grokking_lar_experiments

ARCH=$1
GPU=$2
OUT=$3
STEPS=$4

mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU python -u phase7_modular_fe_scan.py \
  --arch "$ARCH" \
  --lrs 1e-4 3e-4 1e-3 2e-3 3e-3 5e-3 1e-2 3e-2 1e-1 \
  --seeds 0 1 2 \
  --total_steps "$STEPS" \
  --out_dir "$OUT" 2>&1 | tee "${OUT}_run.log"
echo "PHASE7 $ARCH DONE: $(date)"
