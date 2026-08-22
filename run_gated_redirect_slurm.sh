#!/bin/bash
#SBATCH --job-name=gated_redir
#SBATCH --output=slurm_logs/gated_redir_%j.out
#SBATCH --error=slurm_logs/gated_redir_%j.err
#SBATCH --time=04:00:00
#SBATCH -p major
#SBATCH --qos=major_student
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Thesis 5.4: content-gated redirection vs ungated r_UV on Qwen3-30B. Oettingenstr A6000.
# NEVER -p minor. LRZ: override with -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1
# gate FIT is activation-only (n_fit large=350); held-out eval GENERATES refusals per
# split x coeff x condition, so n is kept moderate (80). benign budget: 350+80 < 475.
# MCQ-matched: forget=wmdp_bio_mcq, benign=mmlu_biology, general=general_mcq.

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

# same L* as the MoE run and steering (its fine sweep); fall back to 36 if not run yet.
SWEEP=run_results/completions/Qwen3-30B-A3B/lunar_moe/wmdp_bio_mcq/layer_sweep.json
LAYER=$($PY -c "import json;print(json.load(open('$SWEEP'))['selected_layer'])" 2>/dev/null || echo 36)
echo "using layer L* = $LAYER (from $SWEEP)"

$PY scripts/diag_gated_redirect.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer $LAYER --coeffs 0.75 1.0 --n 80 --n_fit 350 \
    --forget dataset/unlearning/wmdp_bio_mcq.json --forget_edge wmdp_bio \
    --benign dataset/unlearning/mmlu_biology.json \
    --general dataset/unlearning/general_mcq.json
