#!/bin/bash
#SBATCH --job-name=lunar_genretain
#SBATCH --output=slurm_logs/lunar_genretain_%j.out
#SBATCH --error=slurm_logs/lunar_genretain_%j.err
#SBATCH --time=06:00:00
#SBATCH -p mcml-hgx-a100-80x4
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
# ABLATION of run_lunar_moe_slurm.sh (T1): identical in every setting EXCEPT the
# retain-training set, which now includes general-knowledge items alongside biology.
#   T1  retain-train = wmdp_bio_retain.json                    (1300 bio passages only)
#   this retain-train = wmdp_bio_and_general_retain.json       (1300 bio + 280 general MCQ)
# Layer is PINNED to 32 (T1's selected layer, layer_sweep=false) so the two runs
# differ ONLY in retain data, not in layer choice -- a clean, isolated comparison.
# Eval sets: benign = mmlu_biology.json (same as T1). General eval uses the
# SUBJECT-STRATIFIED general_mcq_eval.json (20/subject, all 10 subjects), which
# differs from T1's original unstratified first-200 general_mcq.json rows (which
# only covered 5/10 subjects) -- note this when comparing the "general" numbers
# specifically; forget and benign remain directly comparable to T1.
# Writes to save_folder=lunar_moe_general_retain, so T1's forget_32.json is
# untouched.
# LRZ A100. NEVER -p minor. Oettingenstr: -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
$PY -c "import sentence_transformers" 2>/dev/null || $PY -m pip install -q sentence-transformers
cd ~/LUNAR

$PY run_lunar_moe.py \
    model_family=Qwen3-30B-A3B \
    base_model_path=Qwen/Qwen3-30B-A3B \
    eval_batch_size=4 \
    data_name=wmdp_bio_mcq \
    save_folder=lunar_moe_general_retain \
    'forget_edge=[wmdp_bio]' \
    'coeff_list=[+1.0]' \
    use_different_retain_dataset=true \
    different_retain_set_path=dataset/unlearning/wmdp_bio_and_general_retain.json \
    retain_eval_data_path=dataset/unlearning/mmlu_biology.json \
    factual_data_path=dataset/unlearning/general_mcq_eval.json \
    n_train=256 n_val=32 n_test=200 \
    layer_sweep=false \
    'layer_modified=[32]' \
    eval_generation_max_new_tokens=128 \
    max_new_tokens=64 \
    if_eval_factual=true
