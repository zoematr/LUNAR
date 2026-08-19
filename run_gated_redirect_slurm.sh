#!/bin/bash
#SBATCH --job-name=gated_redir
#SBATCH --output=slurm_logs/gated_redir_%j.out
#SBATCH --error=slurm_logs/gated_redir_%j.err
#SBATCH --time=02:00:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Thesis 5.4: content-gated redirection vs ungated r_UV on Qwen3-30B. LRZ A100.
# Oettingenstr: sbatch -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2 run_gated_redirect_slurm.sh

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

$PY scripts/diag_gated_redirect.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer 36 --coeffs 0.75 1.0 --n 40 --n_fit 150
