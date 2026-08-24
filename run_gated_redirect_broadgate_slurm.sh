#!/bin/bash
#SBATCH --job-name=gated_broad
#SBATCH --output=slurm_logs/gated_broad_%j.out
#SBATCH --error=slurm_logs/gated_broad_%j.err
#SBATCH --time=04:00:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
# ABLATION of run_gated_redirect_slurm.sh: SAME training-free gated steering, but
# the gate axis is now fit as forget - 0.5*(benign + general) instead of just
# forget - benign, so the gate "knows" what out-of-domain retain content looks
# like. Tests whether this closes the gap where the original gate only partially
# protected general knowledge (41% refuse at gated_mid, vs 1% for benign).
# General fit/eval use the DISJOINT subject-stratified split (general_mcq_train.json
# for fitting, general_mcq_eval.json for held-out eval; no item overlap, both cover
# all 10 MMLU subjects). No training anywhere in this script -- same frozen base
# model as the original gated run.
# NEVER -p minor. Oettingenstr: -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

# same L* as the trained MoE run / steering / original gated run
SWEEP=run_results/completions/Qwen3-30B-A3B/lunar_moe/wmdp_bio_mcq/layer_sweep.json
LAYER=$($PY -c "import json;print(json.load(open('$SWEEP'))['selected_layer'])" 2>/dev/null || echo 36)
echo "using layer L* = $LAYER (from $SWEEP)"

$PY scripts/diag_gated_redirect.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer $LAYER --coeffs 0.75 1.0 --n 80 --n_fit 350 \
    --forget dataset/unlearning/wmdp_bio_mcq.json --forget_edge wmdp_bio \
    --benign dataset/unlearning/mmlu_biology.json \
    --general dataset/unlearning/general_mcq_eval.json \
    --general_fit dataset/unlearning/general_mcq_train.json \
    --gate_include_general
