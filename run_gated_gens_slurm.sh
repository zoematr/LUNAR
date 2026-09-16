#!/bin/bash
#SBATCH --job-name=gated_gens
#SBATCH --output=slurm_logs/gated_gens_%j.out
#SBATCH --error=slurm_logs/gated_gens_%j.err
#SBATCH --time=02:00:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Save gated-model generations for the thesis qualitative chat-box examples.
# NEVER -p minor / V100. LRZ A100: -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1
# Same config as run_gated_redirect_slurm.sh, coeff 1.0 only, with --save_generations.

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

$PY scripts/diag_gated_redirect.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer 32 --coeffs 1.0 --n 80 --n_fit 350 \
    --forget dataset/unlearning/wmdp_bio_mcq.json --forget_edge wmdp_bio \
    --benign dataset/unlearning/mmlu_biology.json \
    --general dataset/unlearning/general_mcq.json \
    --save_generations run_results/gated_generations.json
