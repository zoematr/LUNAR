#!/bin/bash
#SBATCH --job-name=diag_steer
#SBATCH --output=slurm_logs/diag_steer_%j.out
#SBATCH --error=slurm_logs/diag_steer_%j.err
#SBATCH --time=06:00:00
#SBATCH -p major
#SBATCH --qos=major_student
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# Proxy-vs-base steering diagnostic (T2/T3) on Qwen3-30B. Oettingenstr A6000 (2x48GB).
# NEVER -p minor. LRZ: override with -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1
# GENERATION-BASED (refusal rate per split x coeff), so n is kept moderate: refusal
# is a proportion, tight even at n=250. Datasets default to the MCQ-matched sets.

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

# use the SAME layer the MoE run selected (its fine sweep), so steering, the
# trained model, and gating are all at one layer. Falls back to 36 if not run yet.
SWEEP=run_results/completions/Qwen3-30B-A3B/lunar_moe/wmdp_bio_mcq/layer_sweep.json
LAYER=$($PY -c "import json;print(json.load(open('$SWEEP'))['selected_layer'])" 2>/dev/null || echo 36)
echo "using layer L* = $LAYER (from $SWEEP)"

# sweep several strengths in ONE job (model loaded once, r_UV computed once)
$PY scripts/diag_steering.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer $LAYER --coeffs 0.25 0.5 0.75 1.0 --n 250 \
    --forget dataset/unlearning/wmdp_bio_mcq.json --forget_edge wmdp_bio \
    --benign dataset/unlearning/mmlu_biology.json \
    --general dataset/unlearning/general_mcq.json
