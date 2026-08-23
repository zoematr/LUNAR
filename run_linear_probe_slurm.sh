#!/bin/bash
#SBATCH --job-name=lin_probe
#SBATCH --output=slurm_logs/lin_probe_%j.out
#SBATCH --error=slurm_logs/lin_probe_%j.err
#SBATCH --time=01:30:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Forget-vs-retain linear separability probe on Qwen3-30B. Oettingenstr A6000 (2x48GB).
# NEVER -p minor (V100/sm_70 unsupported). LRZ: override on the sbatch line with
#   sbatch -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1 run_linear_probe_slurm.sh
# MCQ-matched contrast: forget=wmdp_bio_mcq, benign=mmlu_biology, general=general_mcq.
# n=450 balanced (benign 475 / general 480 bind). Activation-only, so large n is cheap.

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

$PY scripts/linear_probe.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer 36 --n 450 --forget_pool 900 \
    --forget dataset/unlearning/wmdp_bio_mcq.json --forget_edge wmdp_bio \
    --benign dataset/unlearning/mmlu_biology.json \
    --general dataset/unlearning/general_mcq.json
