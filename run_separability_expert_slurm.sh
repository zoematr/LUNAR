#!/bin/bash
#SBATCH --job-name=sep_expert
#SBATCH --output=slurm_logs/sep_expert_%j.out
#SBATCH --error=slurm_logs/sep_expert_%j.err
#SBATCH --time=02:00:00
#SBATCH -p major
#SBATCH --qos=major_student
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
# Per-EXPERT separability: inside each (layer, expert), do forget/benign tokens
# still separate? Defends the expert-collapse design choice. Oettingenstr A6000.
# NEVER -p minor. LRZ: override with -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1
# MCQ-matched. All-layer, all-expert. n=250 prompts/set (memory-bound, not balance).

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

$PY scripts/separability_expert.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --forget dataset/unlearning/wmdp_bio_mcq.json \
    --retain dataset/unlearning/mmlu_biology.json \
    --max_samples 250 --min_pts 20
