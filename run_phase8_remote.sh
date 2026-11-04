#!/bin/bash
# Phase 8: weight-decay control for the ConvNet free-energy transition.
# Same protocol as phase3/phase5 (c8, CIFAR-10, AdamW, 200K steps) but at
# different weight decay, to test whether the eta_c~0.035 entropy transition
# is wd-specific (baseline wd=5e-4 is reused, not rerun).
# Usage: bash run_phase8_remote.sh <weight_decay> <gpu> <out_dir>
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepseek-env
cd ~/grokking_lar_experiments

WD=$1
GPU=$2
OUT=$3

LRS="0.001 0.005 0.01 0.02 0.03 0.05 0.07 0.1 0.15"
echo "host $(hostname) wd=$WD GPU=$GPU start $(date)"
mkdir -p "$OUT"
CUDA_VISIBLE_DEVICES=$GPU python -u phase3_free_energy.py \
  --lrs $LRS --seeds 0 1 2 --total_steps 200000 \
  --weight_decay "$WD" --num_channels 8 --log_every 1000 \
  --out_dir "$OUT" 2>&1 | tee "${OUT}_run.log"
echo "PHASE8 wd=$WD DONE: $(date)"
