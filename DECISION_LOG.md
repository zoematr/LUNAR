# Decision log

One entry per experiment. A **SOLID** entry is *closed* — don't re-litigate it. Fill the template every time; if you can't write the one-sentence takeaway, the verdict is not SOLID.

Template:
```
## <date> — <short name>
Question:
Prediction + meaning of each outcome:
Baseline / control:
Headline number (± noise floor):
Confounds + status (ruled out / flagged):
Robustness check:
Verdict: SOLID / CONFOUNDED / NOISE-LIMITED / PENDING
One-sentence takeaway:
```

---

## Safety / refusal eval
Question: How often does the model produce unsafe output on harmful prompts?
Baseline / control: keyword-match vs Llama-Guard classifier (disagreement is itself a finding).
Headline number: Qwen3-30B 8% unsafe (9/200), Llama4-Scout 21.5% (75/200). Top category S14 (code-interpreter abuse).
Confounds + status: classifier choice matters (keyword vs Guard) — reported both, so flagged not hidden.
Robustness check: two models, same pipeline.
Verdict: **SOLID**
One-sentence takeaway: MoE models here produce unsafe output at model-dependent rates (Qwen 8%, Llama4 21.5%); keyword vs classifier disagreement is model-dependent.

## Routing / collateral — do hazardous & benign biology share experts?
Question: Do wmdp_bio (hazardous) and mmlu_college_biology (benign) route to the same experts?
Prediction + meaning: high overlap → shared → can't unlearn by removing experts; low → separable → could target experts.
Baseline / control: template-masked rerun (robustness); note format *works against* overlap.
Headline number: alignment 0.84 (benign puts 84% as much weight on hazardous's top experts as on its own); top-8 overlap 0.59; ~52 effective experts/layer for both.
Confounds + status: format (benign has MCQ options) would *lower* overlap, yet overlap is high → confound ruled out (conservative).
Robustness check: content-masked run vs old all-token run — alignment 0.84 → 0.84 (unchanged). (Per-layer rankings DO move → noise-limited at the layer level.)
Verdict: **SOLID** (aggregate) / NOISE-LIMITED (per-layer rankings)
One-sentence takeaway: Hazardous and benign biology are processed by the same experts, so unlearning must redirect activations while preserving retain, not remove experts.

## Separability (residual) — is there a direction separating forget/retain?
Question: At each layer's residual stream, do forget and retain separate (→ is a LUNAR-style redirection direction available)?
Prediction + meaning: high silhouette → direction exists → redirection viable; ~0 → entangled even in representation.
Baseline / control: **symmetric-MCQ format control — NOT YET RUN.**
Headline number: silhouette (PCA-2D) up to 0.56, peak layer 44; full-dim biased low (ignore).
Confounds + status: **format (free-text forget vs MCQ retain) INFLATES separability — NOT ruled out.** Early-layer separation is a format red flag.
Robustness check: —
Verdict: **CONFOUNDED**
One-sentence takeaway: None yet — cannot claim "content is separable" until the format control (04) is run.

## Separability (per-expert) — do forget/retain separate inside experts?
Question: Inside individual experts, do forget/retain tokens separate?
Headline number: mean per-expert silhouette 0.20, max 0.88; strongest early-mid; routing-exclusivity != internal separability.
Confounds + status: same format confound, **worse** here (all-tokens run includes MCQ option tokens = pure format). NOT ruled out.
Verdict: **CONFOUNDED**
One-sentence takeaway: None yet — same format control needed.

## Format control — content or format? (NEXT)
Question: Does forget/retain separability survive when both are the SAME format (symmetric MCQ)?
Prediction + meaning: still separates → boundary is biology content (findings 02/03 become real); collapses → it was format (discard 02/03).
Verdict: **PENDING**

## Noise floor — how much do per-layer routing numbers wander?
Question: Between two disjoint halves of wmdp_bio, how much does per-layer routing differ (→ what counts as signal)?
Status: `wmdp_bio_A.content.json` produced; half B not pulled yet.
Verdict: **PENDING**
