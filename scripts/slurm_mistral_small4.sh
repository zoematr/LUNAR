#!/usr/bin/env bash
#SBATCH --job-name mistral-small4
#SBATCH --output=/dss/dsshome1/09/ra48tah2/LUNAR/logs/mistral_small4_%j.txt
#SBATCH --gres=gpu:4
#SBATCH --time=06:00:00
#SBATCH --qos=mcml
#SBATCH -p mcml-hgx-a100-80x4

set -e  # stop on first error

WORK_DIR=/dss/dsshome1/09/ra48tah2/LUNAR
cd $WORK_DIR
source lunar_venv/bin/activate
export PYTHONPATH=$WORK_DIR:$PYTHONPATH

# HF auth — use absolute path (~ can differ on compute nodes)
export HF_TOKEN=$(cat /dss/dsshome1/09/ra48tah2/.cache/huggingface/token 2>/dev/null || echo "")

# Download models to node-local /tmp if it has enough space, else use home
TMP_FREE=$(df --output=avail /tmp 2>/dev/null | tail -1 | tr -d ' ')
if [ "${TMP_FREE:-0}" -gt 260000000 ]; then
    export HF_HOME=/tmp/hf_cache_$SLURM_JOB_ID
    echo "Using node-local cache: $HF_HOME"
else
    echo "WARNING: /tmp too small (${TMP_FREE}K), using home dir cache"
fi

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== Step 1: Generate responses + cos_sim ==="
python scripts/eval_harmbench_refusal.py \
    --model_family mistral-small-4 \
    --model_path mistralai/Mistral-Small-4-119B-2603 \
    --test_set harmbench

# Free the large model cache before loading the judge
rm -rf ${HF_HOME:-~/.cache/huggingface}/hub/models--mistralai--Mistral-Small*

RESULTS=run_results/harmbench_refusal/mistral-small-4/results.json
if [ -f "$RESULTS" ]; then
    echo "=== Step 2: Score with Llama Guard ==="
    python scripts/score_safety.py --results $RESULTS
else
    echo "ERROR: Step 1 did not produce $RESULTS — skipping scoring"
fi

echo "=== Done $(date) ==="
