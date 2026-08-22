#!/usr/bin/env bash
#
#SBATCH --job-name separability-qwen3
#SBATCH --output=logs/separability_qwen3_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH -p major
#SBATCH --qos=major_student
# NEVER -p minor (V100/sm_70 unsupported). LRZ: override on the sbatch line with
#   -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1

# Activation-space separability (forget vs retain): does the forget set route to
# activations linearly separable from the retain set? PCA + silhouette per layer.
#
# Cluster-agnostic (pass VENV_DIR / MODEL_PATH / HF_HOME / FORGET / RETAIN /
# MAX_SAMPLES as env vars; partition via -p). LRZ example:
#   VENV_DIR=$HOME/LUNAR/lunar_venv \
#     sbatch -p mcml-hgx-a100-80x4 --qos=mcml --time=00:20:00 run_separability_slurm.sh

WORK_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $WORK_DIR

VENV_DIR=${VENV_DIR:-$HOME/LUNAR/lunar_venv}
source $VENV_DIR/bin/activate

export PYTHONPATH=$WORK_DIR:$PYTHONPATH
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_TOKEN=$(cat $HF_HOME/token 2>/dev/null || cat $HOME/.cache/huggingface/token 2>/dev/null)

mkdir -p logs

echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
python -c "import torch; print('cuda', torch.cuda.is_available())"

MODEL_FAMILY=${MODEL_FAMILY:-Qwen3-30B-A3B}
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
FORGET=${FORGET:-dataset/unlearning/wmdp_bio_mcq.json}
RETAIN=${RETAIN:-dataset/unlearning/mmlu_biology.json}
MAX_SAMPLES=${MAX_SAMPLES:-1300}   # silhouette needs no balance -> use ALL per set
                                   # (forget ~1273, retain 475); per-cluster means reported

echo "=== separability: forget=$FORGET  retain=$RETAIN  max_samples=$MAX_SAMPLES ==="
python scripts/separability.py \
  --model_family $MODEL_FAMILY --model_path $MODEL_PATH \
  --forget $FORGET --retain $RETAIN --max_samples $MAX_SAMPLES

echo "=== done $(date) ==="
