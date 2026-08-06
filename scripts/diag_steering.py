#!/usr/bin/env python3
"""
diag_steering.py — is the broad over-refusal from the STEERING DIRECTION, or from
the training / expert-swap?

It applies LUNAR's redirection as the clean runtime activation-addition (the PROXY,
exactly like Procedure 2 / the layer sweep) at a chosen layer, and generates on
forget / benign-bio / general prompts — with NO training and NO expert-swap. It also
reports ||coeff * r_UV|| vs the residual-stream norm at that layer (is the push too
large?).

How to read it:
  * proxy ALSO over-refuses benign/general  -> the direction is broadly-refusing.
      The steering is applied correctly; broad over-refusal is a real property.
  * proxy is SELECTIVE (refuses forget, answers benign/general), yet the TRAINED
      model over-refuses -> the broad damage is introduced by the training/expert-
      swap (the average-and-broadcast), not the direction -> "applied badly" there.
  * ||coeff*r_UV|| >> ||activation||  -> the raw (un-normalized) difference-of-means
      is too big; coeff=1 is effectively a huge push regardless of direction.

Usage (GPU node):
  python scripts/diag_steering.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --layer 36 --coeff 1.0 --n 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from src.model_utils.model_loader import load_model
from src.dataset_utils import load_dataset_to_get_direction
from src.generate_directions import generate_candidate_directions
from src.utils.hook_utils import get_activation_addition_input_pre_hook

_REFUSE = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry,|i can'?t|i cannot|i can not|i won'?t|"
    r"i'?m not able|i'?m unable|unfortunately, i|no, i)",
    re.I,
)


def _load_questions(path, n):
    d = json.load(open(path))
    qs = [(x.get("question") or x.get("instruction") or "").strip() for x in d]
    return [q for q in qs if q][:n]


def _layer_act_norm(model_base, prompts, layer, n=8):
    """Mean L2 norm of the residual stream entering block[layer+1], last token."""
    norms = []

    def pre(module, inp):
        a = inp[0] if isinstance(inp, (tuple, list)) else inp
        norms.append(a[:, -1, :].float().norm(dim=-1).mean().item())

    h = model_base.model_block_modules[layer + 1].register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            for p in prompts[:n]:
                enc = model_base.tokenize_instructions_fn(instructions=[p])
                model_base.model(
                    input_ids=enc.input_ids.to(model_base.model.device),
                    attention_mask=enc.attention_mask.to(model_base.model.device),
                )
    finally:
        h.remove()
    return sum(norms) / len(norms) if norms else float("nan")


def _generate(model_base, prompts, layer, coeff, cand_dir, positions, max_new_tokens, batch_size):
    subset = [{"question": p, "edge": "diag"} for p in prompts]
    pre_hooks = []
    if coeff != 0:
        vec = cand_dir[positions, layer + 1, :]
        pre_hooks = [(
            model_base.model_block_modules[layer + 1],
            get_activation_addition_input_pre_hook(vector=vec, coeff=float(coeff)),
        )]
    comps = model_base.generate_completions(
        subset, fwd_pre_hooks=pre_hooks, fwd_hooks=[],
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    return [c["response"] for c in comps]


def _refuse_rate(responses):
    return sum(1 for r in responses if _REFUSE.match(str(r).strip())) / max(1, len(responses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--coeff", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=10, help="prompts per split to generate")
    ap.add_argument("--forget_edge", default="wmdp_bio")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--forget", default="dataset/unlearning/wmdp_bio.json")
    ap.add_argument("--benign", default="dataset/unlearning/mmlu_college_biology.json")
    ap.add_argument("--general", default="dataset/unlearning/factual_data.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.create({
        "positions": -1,
        "forget_edge": [args.forget_edge],
        "use_harmful": True,
        "use_unverified": False,
        "eval_batch_size": args.batch_size,
    })

    print(f"loading {args.model_family} ...")
    model_base = load_model(args.model_family, args.model_path, device)
    positions = cfg.positions

    # --- r_UV, exactly as the training run computes it (mean harmful - mean forget) ---
    data_path = args.forget
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg, data_path, instructions_only=True, use_harmful=True, use_unverified=False
    )
    print(f"computing r_UV from {len(harmful_train)} harmful / {len(forget_train)} forget ...")
    cand_dir = generate_candidate_directions(cfg, model_base, harmful_train, forget_train)

    # --- Check B: magnitude ---
    r_uv = cand_dir[positions, args.layer + 1, :]
    ruv_norm = float(r_uv.float().norm().item())
    act_norm = _layer_act_norm(model_base, forget_train, args.layer, n=8)
    print("\n" + "=" * 68)
    print(f"CHECK B — magnitude at layer {args.layer}")
    print(f"  ||r_UV||               = {ruv_norm:.2f}")
    print(f"  ||coeff * r_UV||       = {args.coeff * ruv_norm:.2f}   (coeff={args.coeff})")
    print(f"  ||activation|| (mean)  = {act_norm:.2f}")
    ratio = (args.coeff * ruv_norm) / act_norm if act_norm else float("nan")
    print(f"  ratio push/activation  = {ratio:.2f}  "
          f"({'>1 => push dominates the activation (too strong)' if ratio > 1 else 'push is a fraction of the activation'})")
    print("=" * 68)

    # --- Check A: proxy (activation-addition) vs base, per split ---
    splits = {"forget (hazardous)": args.forget,
              "benign biology": args.benign,
              "general knowledge": args.general}
    print(f"\nCHECK A — proxy (a + {args.coeff}*r_UV @ layer {args.layer}) vs base, {args.n}/split\n")
    for name, path in splits.items():
        prompts = _load_questions(path, args.n)
        base = _generate(model_base, prompts, args.layer, 0.0, cand_dir, positions,
                         args.max_new_tokens, args.batch_size)
        proxy = _generate(model_base, prompts, args.layer, args.coeff, cand_dir, positions,
                          args.max_new_tokens, args.batch_size)
        print(f"### {name}  |  base refuse {_refuse_rate(base):.0%}  ->  PROXY refuse {_refuse_rate(proxy):.0%}")
        for p, b, pr in list(zip(prompts, base, proxy))[:5]:
            print(f"   Q: {p[:80]}")
            print(f"     base : {str(b)[:100]}")
            print(f"     proxy: {str(pr)[:100]}")
        print()

    print("READ IT:")
    print("  - proxy refuses benign/general too  -> direction is broadly-refusing (real).")
    print("  - proxy selective but trained model wasn't -> training/expert-swap is the culprit.")
    print("  - ratio push/activation >> 1 -> raw r_UV too big; normalize or lower coeff.")


if __name__ == "__main__":
    main()
