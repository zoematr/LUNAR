#!/bin/bash
#SBATCH --job-name=diag_steer
#SBATCH --output=slurm_logs/diag_steer_%j.out
#SBATCH --error=slurm_logs/diag_steer_%j.err
#SBATCH --time=00:40:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Proxy-vs-base steering diagnostic on Qwen3-30B (~15 min). LRZ A100.
# Oettingenstr: sbatch -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2 run_diag_steering_slurm.sh

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

# sweep several strengths in ONE job (model loaded once, r_UV computed once)
$PY scripts/diag_steering.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer 36 --coeffs 0.25 0.5 0.75 1.0 --n 10
