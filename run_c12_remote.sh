#!/bin/bash
# STAGE B width-12 FSS runs on the remote 2x RTX 3090 box (bio502-134),
# conda env deepseek-env, inside tmux session "larfss".
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepseek-env
cd ~/grokking_lar_experiments
mkdir -p phase6_finite_size
echo "host: $(hostname), python: $(which python), start: $(date)"
python -u phase3_free_energy.py --num_channels 12 \
  --lrs 0.02 0.03 0.05 --seeds 0 1 2 --total_steps 200000 \
  --out_dir phase6_finite_size/c12
echo "C12 ALL DONE: $(date)"
