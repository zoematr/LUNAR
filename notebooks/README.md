# Analysis index — read this first

Every analysis below has: a **question**, a **notebook** (open it, Run All), the **result files** it loads, and a **verdict**. Verdicts follow one rule:

> **SOLID** = robust + no confound → you can build on it.
> **CONFOUNDED** = a plausible non-hypothesis explanation isn't ruled out → do the control before interpreting.
> **NOISE-LIMITED** = needs more data / a noise floor before trusting fine detail.
> **PENDING** = experiment not run yet.

Only build on **SOLID**. Full reasoning for each verdict is in [`../DECISION_LOG.md`](../DECISION_LOG.md).

Model: **Qwen3-30B-A3B** (48 layers, 128 experts, top-8). Forget = `wmdp_bio` (hazardous biology). Retain = `mmlu_college_biology` (benign biology).

---

## Analyses, in order

| # | Notebook | Question | Loads | Verdict |
|---|----------|----------|-------|---------|
| — | *(no notebook yet)* | How often does the model produce unsafe output? | `run_results/harmbench_refusal/*/summary.json` | **SOLID** — Qwen3 8% / Llama4-Scout 21.5% unsafe |
| 01 | `01_routing_collateral.ipynb` | Do hazardous & benign biology use the **same experts**? | `run_results/routing_single/Qwen3-30B-A3B/*.content.json` | **SOLID** — yes, alignment 0.84 |
| 02 | `02_separability_residual.ipynb` | Is there a **residual-stream direction** separating forget/retain? | `run_results/separability/Qwen3-30B-A3B/silhouette.json`, `activations.npz` | **CONFOUNDED** — format (free-text vs MCQ) inflates it |
| 03 | `03_separability_expert.ipynb` | Do forget/retain separate **inside individual experts**? | `run_results/separability/Qwen3-30B-A3B/silhouette_expert.json`, `expert_scatter.npz` | **CONFOUNDED** — same format issue |
| 04 | *(pending)* | Is the separability **content or format**? (symmetric-MCQ control) | pending run | **PENDING** — the next experiment |
| — | *(pending)* | Noise floor: how much do per-layer routing numbers wander between two halves? | `wmdp_bio_A.content.json` exists; `_B` not pulled | **PENDING** |

**The one-paragraph state of the project:** hazardous and benign biology route through the same experts (01, SOLID) → unlearning can't remove experts, it must redirect activations while preserving retain. Whether a clean redirection *direction* exists is open (02/03, CONFOUNDED by format). The **format control (04)** is the next experiment and resolves it.

---

## How to rerun

1. **Produce the JSONs** (cluster, GPU): the `scripts/*.py` run via the `run_*_slurm.sh` scripts.
   - routing → `scripts/routing_single.py` / `run_routing_slurm.sh`
   - separability (residual) → `scripts/separability.py` / `run_separability_slurm.sh`
   - separability (per-expert) → `scripts/separability_expert.py`
2. **Pull results** to `run_results/…` (git or scp).
3. **Open the notebook**, Run All. Each notebook is self-contained and pandas-free; it loads only from `run_results/`.

---

## Data inventory (`run_results/`)

| file | produced by | what it is |
|---|---|---|
| `harmbench_refusal/<model>/summary.json` | `eval_harmbench_refusal.py` + `score_safety.py` | % unsafe (Llama-Guard) on 200 HarmBench behaviors |
| `routing_single/<model>/<tag>.content.json` | `routing_single.py` (masked run) | per-layer `freq` + `applied_weight` [48×128], one per dataset |
| `routing_single/<model>/<tag>.json` | `routing_single.py` (old all-token run) | same, before template masking — superseded by `.content` |
| `separability/<model>/silhouette.json` | `separability.py` | per-layer silhouette (residual, last-token) |
| `separability/<model>/activations.npz` | `separability.py` | last-token residual activations for PCA plots |
| `separability/<model>/silhouette_expert.json` | `separability_expert.py` | per-(layer,expert) silhouette [48×128] |
| `separability/<model>/expert_scatter.npz` | `separability_expert.py` | per-token 2D coords + routing + labels, for per-expert scatters |

---

## Scripts (`scripts/` and repo root)

| script | role |
|---|---|
| `scripts/routing_single.py` | routing analysis (→ notebook 01) |
| `scripts/separability.py` | residual separability (→ notebook 02) |
| `scripts/separability_expert.py` | per-expert separability (→ notebook 03) |
| `scripts/eval_harmbench_refusal.py`, `scripts/score_safety.py` | safety / refusal eval |
| `run_lunar.py`, `run_lunar_moe.py` | **the unlearning method itself** (dense / MoE) — not run yet |
| `eval_routing_shift.py` | *next experiment*: routing KL before/after unlearning |
| `eval_causal_ablation.py` | *next experiment*: ablate forget-specific experts, measure knowledge drop |
| `scripts/download_*.py` | build the datasets (reproducibility) |
| `scripts/verify_routing.py`, `scripts/verify_model_wiring.py` | verification utilities |
| `scripts/build_benign_bio.py`, `scripts/build_wmdp_retain.py` | **parked** — alternate benign sets, not currently used (may reuse for the format control) |
| `run_*_slurm.sh` | SLURM submitters for the above |

`archive/` (root) and `notebooks/archive/` hold **superseded** code (old OLMoE routing analysis), kept for reference, not maintained.
