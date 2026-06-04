#!/usr/bin/env bash
#SBATCH --job-name llama4-scout
#SBATCH --output=/dss/dsshome1/09/ra48tah2/LUNAR/logs/llama4_scout_%j.txt
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --qos=gpu
#SBATCH -p lrz-hgx-a100-80x4

WORK_DIR=/dss/dsshome1/09/ra48tah2/LUNAR
cd $WORK_DIR
source lunar_venv/bin/activate
export PYTHONPATH=$WORK_DIR:$PYTHONPATH

# Download models to node-local storage (faster I/O, no home-dir quota hit).
# Keep HF_TOKEN_PATH pointing at home so gated-repo auth still works.
export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
export HF_TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || echo "")

echo "=== Step 1: Generate responses + cos_sim ==="
python scripts/eval_harmbench_refusal.py \
    --model_family llama4-scout \
    --model_path meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --test_set harmbench

# Free the large model cache before loading the judge
rm -rf /tmp/hf_cache_$SLURM_JOB_ID/hub/models--meta-llama--Llama-4-Scout*

echo "=== Step 2: Score with Llama Guard ==="
python scripts/score_safety.py \
    --results run_results/harmbench_refusal/llama4-scout/results.json

echo "=== Done ==="
