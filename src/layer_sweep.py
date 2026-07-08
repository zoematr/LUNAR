# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
LUNAR Procedure 2 — layer selection via the (s1 - s2) sweep, adapted for MoE.

Faithful to LUNAR §3.2 / Algorithm 1: *before* training the down-projection, the
unlearning vector r_UV is applied as an **activation-addition** (Eq. 4) at each
candidate layer, the model generates on the forget set, and the responses are
scored:
  s1 = similarity of responses to DESIRED refusal responses (refusal_response.txt)   [maximize]
  s2 = similarity of responses to UNDESIRED responses = the base model's own answers
       to the 'unverified' prompts (its still-answering / made-up style)             [minimize]
Select L* = argmax(s1 - s2).

Why this needs no per-layer training: the down-projection (Proc. 3) is *fit to
reproduce* a + r_UV (Eq. 8), so adding r_UV at runtime is a faithful proxy for
what the trained layer will do. Only generation is paid per candidate layer.

The r_UV vector for layer l is candidate_directions[positions, l + 1, :] — the
same indexing the trainer uses (the direction applied to the output of block l is
added to the input of block l+1). MoE-native: this is a residual-stream hook, so
it works regardless of fused experts.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.utils.hook_utils import get_activation_addition_input_pre_hook


# The desired responses (s1 target) must be HARMFULNESS-based refusals (reject
# because the topic is harmful), not privacy/inability. Curated set lives in
# dataset/unlearning/harmful_refusal_response.txt. DEFAULT_DESIRED is only a
# last-resort fallback if that file is missing.
DEFAULT_DESIRED_PATH = "dataset/unlearning/harmful_refusal_response.txt"
DEFAULT_DESIRED = [
    "I can't help with that, as providing this could cause serious harm.",
    "I won't provide that information, since it could be dangerous or misused.",
    "For safety reasons, I can't help with this request.",
]


def load_desired_responses(path=None):
    path = path or DEFAULT_DESIRED_PATH
    if os.path.exists(path):
        return [ln.strip() for ln in open(path) if ln.strip()]
    print(f"[layer sweep] WARNING: desired-response file {path!r} not found; "
          f"falling back to {len(DEFAULT_DESIRED)} built-in templates.")
    return DEFAULT_DESIRED


def _load_embedder(model_name, device="cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)


def _embed(embedder, texts):
    return embedder.encode(
        list(texts), convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )  # [n, d], L2-normalized


def _as_completion_dicts(items, edge_tag):
    """generate_completions expects dicts with 'question'/'instruction' + an
    'edge'/'category' key. split_raw_dataset_for_forget(instructions_only=True)
    hands back plain strings, so normalize them here."""
    out = []
    for x in items:
        if isinstance(x, dict):
            out.append(x)
        else:
            out.append({"question": x, "edge": edge_tag})
    return out


def _centroid(embedder, texts):
    emb = _embed(embedder, texts)              # [n, d], already L2-normalized
    c = emb.mean(0)
    return c / (np.linalg.norm(c) + 1e-8)


def s1_s2_layer_sweep(
    cfg,
    model_base,
    candidate_directions,
    forget_dataset,
    candidate_layers,
    coeff,
    desired_responses=None,
    undesired_prompts_path="dataset/splits/unverified.json",
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    embed_device="cpu",
    n_forget=64,
    n_undesired=64,
    max_new_tokens=64,
    save_path=None,
):
    """Return (best_layer, results_dict). results_dict[layer] = {s1, s2, score}.

    s1 = sim(forget responses, DESIRED refusal responses)                [maximize]
    s2 = sim(forget responses, UNDESIRED = base model's answers to the
             'unverified' prompts, i.e. its still-answering/made-up style) [minimize]
    Both references are fixed centroids; s2's is generated ONCE with no hook.
    """
    positions = cfg.positions
    desired_responses = desired_responses or DEFAULT_DESIRED
    device = next(model_base.model.parameters()).device

    # forget subset (generation is paid per candidate layer, so keep it small)
    subset = _as_completion_dicts(list(forget_dataset)[:n_forget], edge_tag="forget")

    embedder = _load_embedder(embed_model_name, device=embed_device)
    desired_centroid = _centroid(embedder, desired_responses)

    # --- UNDESIRED reference: base model's own answers to the unverified prompts ---
    # (generated once, no redirection hook; represents "still answering / making
    #  things up" — what we want the unlearned forget responses to move AWAY from.)
    with open(undesired_prompts_path) as f:
        undesired_prompts = json.load(f)
    undesired_prompts = _as_completion_dicts(undesired_prompts[:n_undesired], edge_tag="unverified")
    print(f"[layer sweep] generating undesired reference: base model on "
          f"{len(undesired_prompts)} unverified prompts (no hook)")
    undesired_comps = model_base.generate_completions(
        undesired_prompts, fwd_pre_hooks=[], fwd_hooks=[],
        batch_size=cfg.eval_batch_size, max_new_tokens=max_new_tokens,
    )
    undesired_responses = [c["response"] for c in undesired_comps]
    undesired_centroid = _centroid(embedder, undesired_responses)

    tok = getattr(model_base, "tokenizer", None)

    def _tok_len(text):
        if tok is None:
            return len(text.split())
        return len(tok(text, add_special_tokens=False)["input_ids"])

    results = {}
    generations = {}   # per-layer: full prompt/response/answer + length, for offline inspection
    print(f"[layer sweep] candidates={candidate_layers}  coeff={coeff}  n_forget={len(subset)}")
    for l in candidate_layers:
        vec = candidate_directions[positions, l + 1, :].to(device)   # r_UV at layer l
        pre_hook = (
            model_base.model_block_modules[l + 1],
            get_activation_addition_input_pre_hook(vector=vec, coeff=float(coeff)),
        )
        comps = model_base.generate_completions(
            subset,
            fwd_pre_hooks=[pre_hook],
            fwd_hooks=[],
            batch_size=cfg.eval_batch_size,
            max_new_tokens=max_new_tokens,
        )
        resp = [c["response"] for c in comps]

        resp_emb = _embed(embedder, resp)                       # [n, d]
        s1 = float(np.mean(resp_emb @ desired_centroid))        # -> desired refusal, maximize
        s2 = float(np.mean(resp_emb @ undesired_centroid))      # -> undesired (unverified answers), minimize
        score = s1 - s2

        # long-generation diagnostics: how many responses ran to the token budget
        lens = [_tok_len(r) for r in resp]
        n_at_max = int(sum(1 for x in lens if x >= max_new_tokens))
        results[int(l)] = {
            "s1": s1, "s2": s2, "score": score,
            "resp_tokens_mean": float(np.mean(lens)),
            "resp_tokens_max": int(max(lens)),
            "n_at_max_tokens": n_at_max,             # >0 => runaway / truncated generations
            "frac_at_max_tokens": n_at_max / max(1, len(lens)),
        }
        generations[int(l)] = [
            {"prompt": c["prompt"], "response": r,
             "original_answer": c["original_answer"], "resp_tokens": ln}
            for c, r, ln in zip(comps, resp, lens)
        ]
        print(f"  layer {l:3d}: s1={s1:.3f}  s2={s2:.3f}  (s1-s2)={score:.3f}  "
              f"| tok mean {np.mean(lens):.0f} max {max(lens)}  @max {n_at_max}/{len(lens)}")

    best = max(results, key=lambda k: results[k]["score"])
    print(f"\n[layer sweep] selected layer {best}  (s1-s2={results[best]['score']:.3f})")
    if results[best]["n_at_max_tokens"] > 0:
        print(f"[layer sweep] WARNING: {results[best]['n_at_max_tokens']} responses hit the token "
              f"budget on the selected layer -> raise max_new_tokens or inspect generations.")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        json.dump(
            {
                "candidate_layers": [int(l) for l in candidate_layers],
                "coeff": float(coeff),
                "n_forget": len(subset),
                "max_new_tokens": int(max_new_tokens),
                "results": results,
                "selected_layer": int(best),
            },
            open(save_path, "w"),
            indent=2,
        )
        print(f"[layer sweep] saved curve to {save_path}")
        gen_path = str(save_path).replace(".json", "_generations.json")
        json.dump(generations, open(gen_path, "w"), indent=2)
        print(f"[layer sweep] saved generations to {gen_path}")

    return int(best), results
