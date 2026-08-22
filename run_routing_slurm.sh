#!/usr/bin/env bash
#
#SBATCH --job-name routing-qwen3
#SBATCH --output=logs/routing_qwen3_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH -p major
#SBATCH --qos=major_student
# NEVER -p minor (V100/sm_70 unsupported). LRZ: override on the sbatch line with
#   -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1

# Single-dataset MoE routing analysis (knowledge spread across experts).
# Submit from the repo root:  sbatch run_routing_slurm.sh
#
# Cluster-agnostic: pass VENV_DIR / MODEL_FAMILY / MODEL_PATH / HF_HOME /
# MAX_SAMPLES as env vars in the sbatch line, and the partition via -p.
# Examples:
#   # Oettingenstr (major partition, 2x 48GB GPUs):
#   VENV_DIR=/nfs/data8/matrullo/venv_lunar HF_HOME=/nfs/data8/matrullo/hf_cache \
#     sbatch -p major --qos=major_student --gres=gpu:nvidia_rtx_a6000:2 run_routing_slurm.sh
#   # LRZ (single A100-80GB):
#   VENV_DIR=$HOME/LUNAR/lunar_venv \
#     sbatch -p mcml-hgx-a100-80x4 --qos=mcml run_routing_slurm.sh

WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $WORK_DIR

# --- all cluster-specific values come from the environment (pass them in the
#     sbatch command); the defaults below are just fallbacks. No need to edit. ---
VENV_DIR=${VENV_DIR:-/nfs/data8/matrullo/venv_lunar}
source $VENV_DIR/bin/activate

export PYTHONPATH=$WORK_DIR:$PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || cat $HOME/.cache/huggingface/token 2>/dev/null)

mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
echo "cuda available:"; python -c "import torch; print(torch.cuda.is_available())"
echo "GPU state at start:"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader

MODEL_FAMILY=${MODEL_FAMILY:-Qwen3-30B-A3B}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
MAX_SAMPLES=${MAX_SAMPLES:-450}   # balanced across the three sets (benign 475 / general 480 bind)
# Token positions to record routing for: content (question span only, default),
# all (incl. chat template), or last (last question token). To isolate the
# template-dilution effect, run once with ROUTE_TOKENS=content and once with
# ROUTE_TOKENS=all and diff the results (they save to separate files).
ROUTE_TOKENS=${ROUTE_TOKENS:-content}

# Biology routing baseline (knowledge spread across experts), original datasets:
#   wmdp_bio              = hazardous biology (the forget target), free-text Qs
#   mmlu_college_biology  = benign biology (collateral check)
#   general_mmlu_science  = non-bio science (specificity check)
# ROUTE_TOKENS=content strips the chat template (system prompt, role markers, and
# the empty <think></think> block) so routing is recorded over question content
# only. KNOWN, ACCEPTED CAVEAT: mmlu_college_biology keeps its inline MCQ options
# while wmdp_bio/science do not, so the benign comparison still carries some
# MCQ-format asymmetry -- document this when reporting.
for spec in "wmdp_bio_mcq:dataset/unlearning/wmdp_bio_mcq.json" \
            "mmlu_biology:dataset/unlearning/mmlu_biology.json" \
            "general_mcq:dataset/unlearning/general_mcq.json"; do
  tag=${spec%%:*}; path=${spec#*:}
  echo "=== routing: $tag (route_tokens=$ROUTE_TOKENS) ==="
  python scripts/routing_single.py \
    --model_family $MODEL_FAMILY \
    --model_path $MODEL_PATH \
    --data_path "$path" \
    --dataset_tag "$tag" \
    --route_tokens $ROUTE_TOKENS \
    --max_samples $MAX_SAMPLES
done

echo "=== done $(date) ==="
