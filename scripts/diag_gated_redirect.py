#!/usr/bin/env python3
"""
diag_gated_redirect.py — thesis 5.4: can we recover SELECTIVITY?

The probe showed forget and benign biology are linearly separable, yet LUNAR's
r_UV is orthogonal to that axis (cos~0) and is added UNCONDITIONALLY, so it pushes
benign into refusal as hard as forget. Fix: keep r_UV as the redirection, but GATE
it on the discriminative axis -- only redirect activations that look forget-like.

  gate_dir = mean(forget_train) - mean(benign_train)   (the separable axis)
  redirect = r_UV = mean(harmful) - mean(forget)        (LUNAR's toward-refusal dir)
  at each position: if <a, gate_hat> > thr:  a += coeff * r_UV   else leave a alone

Everything is fit on a TRAIN split and evaluated on HELD-OUT forget/benign/general,
so a thin / topic-only separating direction (which the clustering overlap warned
about) will show up as a leaky gate rather than a false win.

Reports, per coeff:
  base            : no steering (sanity, ~0%)
  ungated r_UV    : LUNAR's broad push (reproduces the over-refusal)
  gated r_UV      : the fix, at a balanced and a forget-lenient threshold
plus gate generalization on held-out (does the gate itself separate f/b?).

Usage (GPU node):
  python scripts/diag_gated_redirect.py --model_family Qwen3-30B-A3B \
      --model_path Qwen/Qwen3-30B-A3B --layer 36 --coeffs 0.75 1.0 --n 40
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model_utils.model_loader import load_model
from src.dataset_utils import load_dataset_to_get_direction
from src.generate_directions import generate_candidate_directions

_REFUSE = re.compile(
    r"^\s*(i'?m sorry|i am sorry|sorry,|i can'?t|i cannot|i can not|i won'?t|"
    r"i'?m not able|i'?m unable|unfortunately, i|no, i)",
    re.I,
)


def _load_questions(path, n):
    d = json.load(open(path))
    qs = [(x.get("question") or x.get("instruction") or "").strip() for x in d]
    return [q for q in qs if q][:n]


def _capture_layer(model_base, prompts, tap):
    """Last-token residual at the INPUT to block `tap` (the steering/gate point)."""
    out = []

    def pre(module, inp):
        a = inp[0] if isinstance(inp, (tuple, list)) else inp
        out.append(a[:, -1, :].float().detach().cpu().numpy())

    h = model_base.model_block_modules[tap].register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            for p in prompts:
                enc = model_base.tokenize_instructions_fn(instructions=[p])
                model_base.model(
                    input_ids=enc.input_ids.to(model_base.model.device),
                    attention_mask=enc.attention_mask.to(model_base.model.device),
                )
    finally:
        h.remove()
    return np.concatenate(out, axis=0)


def get_gated_addition_pre_hook(gate_hat, thr, add_vec, coeff):
    """Add coeff*add_vec at positions whose projection onto gate_hat exceeds thr."""
    def hook_fn(module, inp):
        a = inp[0] if isinstance(inp, tuple) else inp
        g = gate_hat.to(a)
        v = add_vec.to(a)
        proj = a @ g                          # [batch, seq]
        mask = (proj > thr).to(a.dtype).unsqueeze(-1)  # [batch, seq, 1]
        a = a + mask * (coeff * v)
        return (a, *inp[1:]) if isinstance(inp, tuple) else a
    return hook_fn


def get_ungated_addition_pre_hook(add_vec, coeff):
    def hook_fn(module, inp):
        a = inp[0] if isinstance(inp, tuple) else inp
        a = a + coeff * add_vec.to(a)
        return (a, *inp[1:]) if isinstance(inp, tuple) else a
    return hook_fn


def _refuse_rate(responses):
    return sum(1 for r in responses if _REFUSE.match(str(r).strip())) / max(1, len(responses))


def _generate(model_base, prompts, tap, pre_hook, max_new_tokens, batch_size):
    subset = [{"question": p, "edge": "diag"} for p in prompts]
    hooks = [] if pre_hook is None else [(model_base.model_block_modules[tap], pre_hook)]
    comps = model_base.generate_completions(
        subset, fwd_pre_hooks=hooks, fwd_hooks=[],
        batch_size=batch_size, max_new_tokens=max_new_tokens,
    )
    return [c["response"] for c in comps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--coeffs", type=float, nargs="+", default=[0.75, 1.0])
    ap.add_argument("--n", type=int, default=40, help="held-out prompts per split")
    ap.add_argument("--n_fit", type=int, default=150, help="train prompts to fit gate")
    ap.add_argument("--forget_edge", default="wmdp_bio")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--forget", default="dataset/unlearning/wmdp_bio_mcq.json")
    ap.add_argument("--benign", default="dataset/unlearning/mmlu_biology.json")
    ap.add_argument("--general", default="dataset/unlearning/general_mcq_eval.json",
                    help="held-out general eval set (subject-stratified, all 10 MMLU "
                         "subjects, disjoint from --general_fit)")
    ap.add_argument("--general_fit", default="dataset/unlearning/general_mcq_train.json",
                    help="general items used to FIT the gate when --gate_include_general "
                         "is set; must not overlap --general")
    ap.add_argument("--gate_include_general", action="store_true",
                    help="fit the gate as forget - 0.5*(benign+general) instead of "
                         "forget - benign, so the gate axis also knows what "
                         "out-of-domain retain content looks like")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tap = args.layer + 1
    cfg = OmegaConf.create({
        "positions": -1, "forget_edge": [args.forget_edge],
        "use_harmful": True, "use_unverified": False, "eval_batch_size": args.batch_size,
    })

    print(f"loading {args.model_family} ...")
    model_base = load_model(args.model_family, args.model_path, device)

    # --- LUNAR's redirection direction r_UV (same code path as the real run) ---
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg, args.forget, instructions_only=True, use_harmful=True, use_unverified=False
    )
    cand_dir = generate_candidate_directions(cfg, model_base, harmful_train, forget_train)
    r_uv = cand_dir[-1, tap, :].float().to(device)

    # --- discriminative GATE axis, fit on TRAIN forget/benign(/general), held-out for eval ---
    benign_all = _load_questions(args.benign, 10_000)
    general_all = _load_questions(args.general, 10_000)
    # train fit set (disjoint from the held-out eval set taken from the tail)
    nf_fit = min(args.n_fit, len(forget_train) - args.n)
    nb_fit = min(args.n_fit, len(benign_all) - args.n)
    f_fit, b_fit = forget_train[:nf_fit], benign_all[:nb_fit]
    # held-out eval sets (tail, disjoint from fit)
    f_ev = forget_train[-args.n:]
    b_ev = benign_all[-args.n:]
    g_ev = general_all[-args.n:]

    Af = _capture_layer(model_base, f_fit, tap)
    Ab = _capture_layer(model_base, b_fit, tap)

    if args.gate_include_general:
        # Fit general on a SEPARATE file (--general_fit) from the general eval file
        # (--general), so fit and held-out eval never share items. Default points
        # at the subject-stratified general_mcq_train.json / general_mcq_eval.json
        # pair (all 10 MMLU subjects on both sides, no item overlap).
        general_fit_all = _load_questions(args.general_fit, 10_000)
        ng_fit = min(args.n_fit, len(general_fit_all))
        g_fit = general_fit_all[:ng_fit]
        Ag_fit = _capture_layer(model_base, g_fit, tap)
        # domain-balanced: average the two retain-domain means equally, rather than
        # pooling all points together (which would let whichever domain has more
        # fit examples dominate the gate).
        retain_mean = 0.5 * (Ab.mean(0) + Ag_fit.mean(0))
        print(f"fit gate on {len(f_fit)} forget / {len(b_fit)} benign / {len(g_fit)} general "
              f"(gate = forget - 0.5*(benign+general)); "
              f"eval on held-out {len(f_ev)}/{len(b_ev)}/{len(g_ev)}")
    else:
        retain_mean = Ab.mean(0)
        print(f"fit gate on {len(f_fit)} forget / {len(b_fit)} benign "
              f"(gate = forget - benign, general NOT in fit); "
              f"eval on held-out {len(f_ev)}/{len(b_ev)}/{len(g_ev)}")

    gate = Af.mean(0) - retain_mean               # points retain -> forget
    gate_hat_np = gate / (np.linalg.norm(gate) + 1e-8)
    gate_hat = torch.tensor(gate_hat_np, dtype=torch.float32, device=device)

    pf = Af @ gate_hat_np
    pb = Ab @ gate_hat_np
    thr_mid = float(0.5 * (pf.mean() + pb.mean()))          # balanced
    thr_len = float(np.quantile(pf, 0.25))                  # forget-lenient (keep ~75% forget)

    # gate generalization on held-out
    Af_ev = _capture_layer(model_base, f_ev, tap)
    Ab_ev = _capture_layer(model_base, b_ev, tap)
    Ag_ev = _capture_layer(model_base, g_ev, tap)
    def _rates(A, thr):
        p = A @ gate_hat_np
        return float((p > thr).mean())
    print("\n" + "=" * 72)
    print(f"GATE (held-out, tap=block {tap})   thr_mid={thr_mid:.2f}  thr_lenient={thr_len:.2f}")
    print("  fraction PASSING the gate (-> would be redirected):")
    for name, thr in [("mid", thr_mid), ("lenient", thr_len)]:
        print(f"    thr_{name:<8} forget={_rates(Af_ev,thr):.0%}  benign={_rates(Ab_ev,thr):.0%}"
              f"  general={_rates(Ag_ev,thr):.0%}   (want forget HIGH, benign/general LOW)")
    print("=" * 72)

    # --- behavioral eval: refuse% per split, base / ungated / gated ---
    splits = {"forget": f_ev, "benign": b_ev, "general": g_ev}
    base = {k: _refuse_rate(_generate(model_base, v, tap, None,
            args.max_new_tokens, args.batch_size)) for k, v in splits.items()}

    print("\nrefuse% by condition (held-out, n per split):")
    for coeff in args.coeffs:
        ung = get_ungated_addition_pre_hook(r_uv, coeff)
        gmid = get_gated_addition_pre_hook(gate_hat, thr_mid, r_uv, coeff)
        glen = get_gated_addition_pre_hook(gate_hat, thr_len, r_uv, coeff)
        res = {"base": base}
        for label, hook in [("ungated", ung), ("gated_mid", gmid), ("gated_lenient", glen)]:
            res[label] = {k: _refuse_rate(_generate(model_base, v, tap, hook,
                          args.max_new_tokens, args.batch_size)) for k, v in splits.items()}
        print(f"\n  coeff = {coeff}")
        hdr = "  " + "condition".ljust(15) + "".join(s.rjust(9) for s in splits)
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for label in ["base", "ungated", "gated_mid", "gated_lenient"]:
            row = "  " + label.ljust(15) + "".join(f"{res[label][s]:>8.0%} " for s in splits)
            print(row)
    print("\nREAD: gated should keep forget HIGH while benign/general drop vs ungated.")
    print("If gated benign/general stay high, the gate is leaky (thin/topic-only axis).")


if __name__ == "__main__":
    main()
