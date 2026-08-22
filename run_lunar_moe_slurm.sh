#!/bin/bash
#SBATCH --job-name=lunar_qwen3
#SBATCH --output=slurm_logs/lunar_qwen3_%j.out
#SBATCH --error=slurm_logs/lunar_qwen3_%j.err
#SBATCH --time=06:00:00
#SBATCH -p major
#SBATCH --qos=major_student
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# Qwen3-30B-A3B is ~61GB in bf16. 2x A6000 (48GB, sm_86) = 96GB, fits fully on-GPU.
# Oettingenstr cluster. DO NOT use -p minor: those are V100s (sm_70), which this
# PyTorch build has no CUDA kernels for -> cudaErrorNoKernelImageForDevice.
# LRZ equivalent: -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1 (A100-80GB, one is enough)

set -e
mkdir -p slurm_logs

# Call the venv python DIRECTLY (no `source` — sbatch runs under dash where
# `source` doesn't exist; this avoids the "source: not found" -> missing-deps trap).
PY=~/LUNAR/lunar_venv/bin/python

# one-time: the layer sweep needs sentence-transformers
$PY -c "import sentence_transformers" 2>/dev/null || $PY -m pip install -q sentence-transformers

cd ~/LUNAR

# MCQ-matched end to end: forget=wmdp_bio_mcq, benign eval=mmlu_biology (college+HS,
# 475), general eval=general_mcq (480). forget_edge stays [wmdp_bio] (both files carry
# that edge tag). n_test=200 balanced eval; eval gens capped at 128 tokens (refusal +
# short MCQ answers fit) to keep the run in budget. Completions are saved to
# forget_<L>.json (eval_logs['*']['generated_text']) -> score refusal offline with
# scripts/score_refusal.py, no rerun. save_unlearned_model stays false.
$PY run_lunar_moe.py \
    model_family=Qwen3-30B-A3B \
    base_model_path=Qwen/Qwen3-30B-A3B \
    eval_batch_size=8 \
    data_name=wmdp_bio_mcq \
    'forget_edge=[wmdp_bio]' \
    'coeff_list=[+1.0]' \
    use_different_retain_dataset=true \
    different_retain_set_path=dataset/unlearning/wmdp_bio_retain.json \
    retain_eval_data_path=dataset/unlearning/mmlu_biology.json \
    factual_data_path=dataset/unlearning/general_mcq.json \
    n_train=256 n_val=32 n_test=200 \
    layer_sweep=true \
    'sweep_layers=[30,31,32,33,34,35,36,37,38,39,40,41,42]' \
    sweep_n_forget=128 \
    eval_generation_max_new_tokens=128 \
    max_new_tokens=64 \
    if_eval_factual=true
