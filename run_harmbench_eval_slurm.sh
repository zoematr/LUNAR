#!/usr/bin/env bash
#
#SBATCH --job-name harmbench-eval
#SBATCH --output=/nfs/data8/matrullo/LUNAR/logs/harmbench_%j.txt
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH -p minor

WORK_DIR=/nfs/data8/matrullo/LUNAR
VENV_DIR=/nfs/data8/matrullo/venv_lunar

source $VENV_DIR/bin/activate
cd $WORK_DIR

python scripts/eval_harmbench_refusal.py \
    --model_family olmoe-1b-7b-instruct \
    --model_path allenai/OLMoE-1B-7B-0924-Instruct \
    --test_set dref \
    --max_samples 50 \
    --max_new_tokens 128
