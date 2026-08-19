#!/bin/bash
#SBATCH --job-name=lin_probe
#SBATCH --output=slurm_logs/lin_probe_%j.out
#SBATCH --error=slurm_logs/lin_probe_%j.err
#SBATCH --time=00:40:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Forget-vs-retain linear separability probe on Qwen3-30B (~15 min). LRZ A100.
# Oettingenstr: sbatch -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2 run_linear_probe_slurm.sh

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

$PY scripts/linear_probe.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer 36 --n 150
