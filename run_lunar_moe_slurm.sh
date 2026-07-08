#!/bin/bash
#SBATCH --job-name=lunar_qwen3
#SBATCH --output=slurm_logs/lunar_qwen3_%j.out
#SBATCH --error=slurm_logs/lunar_qwen3_%j.err
#SBATCH --time=02:00:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# Qwen3-30B-A3B is ~61GB in bf16. One A100-80GB (sm_80) fits it fully on-GPU.
# LRZ cluster. DO NOT use the default/V100 partitions: V100 is sm_70, which this
# PyTorch build has no CUDA kernels for -> cudaErrorNoKernelImageForDevice.
# Need A100 (sm_80) / A40-A6000 (sm_86) / H100 (sm_90).
# Oettingenstr equivalent: -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2

set -e
mkdir -p slurm_logs

# Call the venv python DIRECTLY (no `source` — sbatch runs under dash where
# `source` doesn't exist; this avoids the "source: not found" -> missing-deps trap).
PY=~/LUNAR/lunar_venv/bin/python

# one-time: the layer sweep needs sentence-transformers
$PY -c "import sentence_transformers" 2>/dev/null || $PY -m pip install -q sentence-transformers

cd ~/LUNAR

$PY run_lunar_moe.py \
    model_family=Qwen3-30B-A3B \
    base_model_path=Qwen/Qwen3-30B-A3B \
    eval_batch_size=8 \
    data_name=wmdp_bio \
    'forget_edge=[wmdp_bio]' \
    use_different_retain_dataset=true \
    different_retain_set_path=dataset/unlearning/wmdp_bio_retain.json \
    retain_eval_data_path=dataset/unlearning/mmlu_college_biology.json \
    layer_sweep=true \
    sweep_stride=4 \
    sweep_n_forget=64 \
    max_new_tokens=64 \
    if_eval_factual=true
