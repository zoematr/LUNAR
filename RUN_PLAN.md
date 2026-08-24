# Final run plan (frozen 2026-08-22)

The definitive batch. MCQ end to end, balanced classes, one working cluster default.
Old free-text results are archived in `run_results_archive_freetext/`.

## Datasets (all MCQ, no biology in the general set)
| role | file | n | note |
|------|------|---|------|
| forget | `wmdp_bio_mcq.json` | 1273 | native WMDP MCQ (options embedded, letter answer) |
| benign (bio) | `mmlu_biology.json` | 475 | MMLU college_biology (165) + high_school_biology (310) |
| general (non-bio) | `general_mcq.json` | 480 | 10 non-bio MMLU subjects x 48; specificity/out-of-domain retain |
| training retain | `wmdp_bio_retain.json` | 1300 | LUNAR retain-preservation set |
| r_UV reference | `dataset/splits/harmful.json` | 260 | D_ref for the redirection direction |

Balanced ceiling = 475 (benign). Activation-only diagnostics use n=450; generation
evals use smaller n (refusal is a proportion, tight at a few hundred).

## The 7 runs
| run | result | partition/GPU | n (balanced) | walltime | output |
|-----|--------|---------------|--------------|----------|--------|
| `run_routing_slurm.sh` | F1 routing overlap | major / a6000:2 | 450/dataset | 2h | run_results/routing_single/ |
| `run_noisefloor_slurm.sh` | F1 half-to-half floor | major / a6000:2 | 450/half | 1.5h | run_results/routing_single/ (wmdp_bio_A/B) |
| `run_separability_slurm.sh` | F2 silhouette | major / a6000:2 | ALL (fgt 1273 / bng 475) | 1h | run_results/separability/ |
| `run_linear_probe_slurm.sh` | T4 probe | major / a6000:2 | 450 (pool 900) | 1.5h | run_results/probe/ |
| `run_diag_steering_slurm.sh` | T2/T3 steer + coeff sweep | major / a6000:2 | 250 (generates) | 6h | .out log |
| `run_gated_redirect_slurm.sh` | T5 gated fix | major / a6000:2 | fit 350 / eval 80 | 4h | .out log |
| `run_lunar_moe_slurm.sh` | T1 trained main result | major / a6000:2 | train 256 / test 200 | 4h | run_results/completions/.../wmdp_bio_mcq/forget_<L>.json |

Optional: `run_lunar_dense_slurm.sh` (Qwen2-7B dense reference, same datasets/sizes,
intuition only, never a like-for-like baseline).

## T1 refusal is scored OFFLINE (no rerun)
The MoE run saves generations in `eval_logs[split]["generated_text"]`. After it finishes:
```
python scripts/score_refusal.py run_results/completions/Qwen3-30B-A3B/lunar_moe/wmdp_bio_mcq/forget_<L>.json
```
`<L>` = the layer the sweep selected (printed in the .out; expected 36). Reports
refuse% for forget / benign / general using the is_refusal keyword detector.

## Launch (Oettingenstr A6000, the default)
```
cd ~/LUNAR && git pull
for j in routing noisefloor separability linear_probe diag_steering gated_redirect lunar_moe; do
  sbatch run_${j}_slurm.sh
done
```
LRZ A100 instead: add `-p mcml-hgx-a100-80x4 --qos=mcml --gres=gpu:1` to each `sbatch`.
NEVER `-p minor` (V100/sm_70 has no kernels in this torch build).

## Locked decisions
- Forget format: MCQ everywhere (incl. training). Caveat to state: unlearning on MCQ
  means the model sees options at redirection time (unlearn "pick the option", not
  free-form recall).
- Layer: fixed L*=36 in all diagnostics; the MoE run re-selects via its own sweep and
  will confirm or flag a different L*.
- Checkpoint: NOT saved (save_unlearned_model=false). Generations suffice for offline
  refusal scoring.
