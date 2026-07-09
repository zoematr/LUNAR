#!/bin/bash
#SBATCH --job-name=lunar_dense
#SBATCH --output=slurm_logs/lunar_dense_%j.out
#SBATCH --error=slurm_logs/lunar_dense_%j.err
#SBATCH --time=01:00:00
#SBATCH -p major
#SBATCH --qos=major_student
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Dense baseline: Qwen2-7B-Instruct (~15GB bf16) — same family as Qwen3-30B-A3B,
# ungated, fits easily on one A6000 (48GB). Same WMDP forget/retain/eval as the MoE
# run, so the two are directly comparable.
# Oettingenstr cluster (A6000, sm_86). Never -p minor (V100/sm_70 unsupported).
# LRZ equivalent: -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR

# "Standard" LUNAR: hardcoded target layer (layer_modified), no sweep.
# layer 20/28 ~= 71% depth, comparable to the MoE's selected layer 36/48.
$PY run_lunar.py \
    model_family=Qwen2-7B-Instruct \
    base_model_path=Qwen/Qwen2-7B-Instruct \
    eval_batch_size=16 \
    data_name=wmdp_bio \
    'forget_edge=[wmdp_bio]' \
    'layer_modified=[20]' \
    'coeff_list=[+2.0]' \
    use_different_retain_dataset=true \
    different_retain_set_path=dataset/unlearning/wmdp_bio_retain.json \
    retain_eval_data_path=dataset/unlearning/mmlu_college_biology.json \
    max_new_tokens=64 \
    if_eval_factual=true
