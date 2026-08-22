#!/usr/bin/env bash
#
#SBATCH --job-name noisefloor-qwen3
#SBATCH --output=logs/noisefloor_qwen3_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH -p major
#SBATCH --qos=major_student
# NEVER -p minor (V100/sm_70 unsupported). LRZ: override on the sbatch line with
#   -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1

# Split-half noise floor for the routing analysis: run wmdp_bio twice on two
# DISJOINT halves (same shuffle seed, different offset), so we can measure how
# much the per-layer routing wanders between two samples of the SAME dataset.
# Compare that to the hazardous-vs-benign signal: only trust layer-level claims
# that exceed this half-to-half variation.
#
# Cluster-agnostic (pass VENV_DIR / MODEL_FAMILY / MODEL_PATH / HF_HOME as env
# vars; partition via -p). LRZ example:
#   VENV_DIR=$HOME/LUNAR/lunar_venv \
#     sbatch -p mcml-hgx-a100-80x4 --qos=mcml run_noisefloor_slurm.sh

WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $WORK_DIR

VENV_DIR=${VENV_DIR:-$HOME/LUNAR/lunar_venv}
source $VENV_DIR/bin/activate

export PYTHONPATH=$WORK_DIR:$PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || cat $HOME/.cache/huggingface/token 2>/dev/null)

mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
echo "cuda available:"; python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

MODEL_FAMILY=${MODEL_FAMILY:-Qwen3-30B-A3B}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
ROUTE_TOKENS=${ROUTE_TOKENS:-content}
HALF=${HALF:-450}          # prompts per half — MATCH the routing comparison n (450)
                           # so the floor reflects wander at the SAME sample size
SEED=${SEED:-0}            # same seed for both -> same shuffle order
DATA=dataset/unlearning/wmdp_bio_mcq.json

# Half A: shuffled prompts [0 : HALF]
echo "=== noise-floor half A (seed=$SEED, [0:$HALF]) ==="
python scripts/routing_single.py \
  --model_family $MODEL_FAMILY --model_path $MODEL_PATH \
  --data_path $DATA --dataset_tag wmdp_bio_A \
  --route_tokens $ROUTE_TOKENS --shuffle --seed $SEED --max_samples $HALF

# Half B: same shuffle, DISJOINT slice [HALF : 2*HALF]
echo "=== noise-floor half B (seed=$SEED, [$HALF:$((2*HALF))]) ==="
python scripts/routing_single.py \
  --model_family $MODEL_FAMILY --model_path $MODEL_PATH \
  --data_path $DATA --dataset_tag wmdp_bio_B \
  --route_tokens $ROUTE_TOKENS --shuffle --seed $SEED --offset $HALF --max_samples $HALF

echo "=== done $(date) ==="
echo "outputs: run_results/routing_single/$MODEL_FAMILY/wmdp_bio_A.$ROUTE_TOKENS.json and wmdp_bio_B.$ROUTE_TOKENS.json"
