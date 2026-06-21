#!/usr/bin/env bash
#
#SBATCH --job-name routing-qwen3
#SBATCH --output=/dss/dsshome1/09/ra48tah2/LUNAR/logs/routing_qwen3_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --qos=mcml
#SBATCH -p mcml-hgx-a100-80x4

# Single-dataset MoE routing analysis (knowledge spread across experts).
# Runs routing_single.py over biology (WMDP-bio) and the MMLU baselines,
# saving one JSON per dataset to run_results/routing_single/<model>/.
#
# Submit with:  sbatch run_routing_slurm.sh
# If the assigned node's driver is too old (cuda check prints False), resubmit
# to an H100 partition:  sbatch -p mcml-hgx-h100-94x4 run_routing_slurm.sh

WORK_DIR=/dss/dsshome1/09/ra48tah2/LUNAR
cd $WORK_DIR
source lunar_venv/bin/activate

export PYTHONPATH=$WORK_DIR:$PYTHONPATH
export HF_TOKEN=$(cat /dss/dsshome1/09/ra48tah2/.cache/huggingface/token 2>/dev/null)
# Reuse the existing home HF cache (default location) so already-downloaded
# models are not re-fetched. Do NOT point this at /tmp — that bypasses the
# cache and re-downloads to a wiped, possibly-tmpfs disk.
export HF_HOME=/dss/dsshome1/09/ra48tah2/.cache/huggingface

mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
echo "cuda available:"; python -c "import torch; print(torch.cuda.is_available())"
echo "GPU state at start:"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

MODEL_FAMILY=Qwen3-30B-A3B
MODEL_PATH=Qwen/Qwen3-30B-A3B

for ds in wmdp_bio general_mmlu_science general_mmlu_humanities factual_data; do
  echo "=== routing: $ds ==="
  python scripts/routing_single.py \
    --model_family $MODEL_FAMILY \
    --model_path $MODEL_PATH \
    --data_path dataset/unlearning/$ds.json \
    --dataset_tag $ds \
    --max_samples 160
done

echo "=== done $(date) ==="
