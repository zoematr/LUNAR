#!/bin/bash
#SBATCH --job-name=lunar_grid
#SBATCH --output=slurm_logs/lunar_grid_%j.out
#SBATCH --error=slurm_logs/lunar_grid_%j.err
#SBATCH --time=03:00:00
#SBATCH -p major
#SBATCH --qos=major_student
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
# Coefficient (x learning-rate) grid search for dense LUNAR on WMDP-bio.
# Step 1: run LUNAR's layer sweep ONCE (coeff=1.0) to pick the target layer L*.
# Step 2: grid over coeff (x lr) at the FIXED L*, so coeff isn't confounded by layer.
# Step 3: aggregate every cell's forget/retain/factual selectivity into one CSV.
# Oettingenstr A6000 (sm_86). LRZ: -p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1

set -e
mkdir -p slurm_logs
PY=~/LUNAR/lunar_venv/bin/python
cd ~/LUNAR
$PY -c "import sentence_transformers" 2>/dev/null || $PY -m pip install -q sentence-transformers

MODEL=Qwen2-7B-Instruct
MPATH=Qwen/Qwen2-7B-Instruct
DATA=wmdp_bio

# --- grid axes (edit here) ---
COEFFS=(0.0 0.25 0.5 0.75 1.0 1.5 2.0)   # primary axis (repo-added redirection strength)
LRS=(0.01)                               # add 0.005 for the paper's other Qwen2-7B lr

# shared overrides (no brackets here -> safe to word-split; bracket args are quoted inline)
COMMON="model_family=$MODEL base_model_path=$MPATH eval_batch_size=16 data_name=$DATA \
  use_different_retain_dataset=true \
  different_retain_set_path=dataset/unlearning/wmdp_bio_retain.json \
  retain_eval_data_path=dataset/unlearning/mmlu_college_biology.json \
  num_epochs=20 max_new_tokens=64 if_eval_factual=true \
  eval_generation_max_new_tokens=128 eval_generation_max_length=256"
# ^ eval gen capped at 128 tokens: refusal + short WMDP/MMLU answers fit easily, and
#   all cells use the same cap so the selectivity surface is internally comparable.

# --- Step 1: layer selection once (paper coeff=1.0) -> L* ---
echo "=== [grid] Step 1: layer sweep (coeff=1.0) ==="
$PY run_lunar.py $COMMON 'forget_edge=[wmdp_bio]' \
    layer_sweep=true sweep_stride=2 'coeff_list=[1.0]' lr=0.01 \
    save_folder=grid/_sweep
LSTAR=$($PY -c "import json;print(json.load(open('run_results/completions/$MODEL/grid/_sweep/$DATA/layer_sweep.json'))['selected_layer'])")
echo "=== [grid] selected layer L* = $LSTAR ==="

# --- Step 2: coeff x lr grid at fixed L* ---
# don't abort the whole grid on one cell failure (e.g. transient OOM); log and continue.
set +e
FAILED=()
for lr in "${LRS[@]}"; do
  for coeff in "${COEFFS[@]}"; do
    echo "=== [grid] coeff=$coeff lr=$lr layer=$LSTAR ==="
    $PY run_lunar.py $COMMON 'forget_edge=[wmdp_bio]' \
        layer_sweep=false "layer_modified=[$LSTAR]" \
        "coeff_list=[$coeff]" lr=$lr \
        save_folder=grid/c${coeff}_lr${lr} \
      || FAILED+=("c${coeff}_lr${lr}")
  done
done
[ ${#FAILED[@]} -gt 0 ] && echo "=== [grid] WARNING: failed cells: ${FAILED[*]} ==="
set -e

# --- Step 3: aggregate to a single CSV + printed table ---
echo "=== [grid] aggregating ==="
$PY scripts/aggregate_grid.py --model $MODEL --data $DATA \
    --out run_results/completions/$MODEL/grid/selectivity.csv
