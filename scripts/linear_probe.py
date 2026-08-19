#!/usr/bin/env python3
"""
linear_probe.py — two questions that feed thesis 4.3 / 5.4:

  Q1 (separability): is forget linearly separable from retain in the residual
      stream? -> is a SELECTIVE redirection direction even possible?
  Q2 (LUNAR's direction): does LUNAR's DEPLOYED direction r_UV distinguish forget
      from benign, and does the best discriminative axis align with r_UV?

Captures last-token residual activations at every block input in ONE forward pass,
then at each layer:
  * cv_acc  : 5-fold CV logistic-regression accuracy, forget vs {benign, general}.
              Both use natural-language questions, so the format confound that
              inflated the earlier silhouette score is controlled. chance = 0.50.
  * ruv_sep : how well a threshold on <a, r_UV_hat> separates forget vs retain
              (IN-SAMPLE threshold; r_UV itself is computed from held-out data).
  * cos_wr  : cos(discriminative axis w, r_UV) in raw activation space.

r_UV is the DEPLOYED direction, computed by the same code path as the real run:
  r_UV = mean(harmful-instruction acts) - mean(forget-prompt acts)   [Eq. 5]
(harmful = dataset/splits/harmful.json, the same reference set the run/diag use.)

How to read it:
  * cv_acc ~ 0.50                 -> forget/retain NOT separable; no selective
                                     direction exists -> deeper-entanglement finding.
  * cv_acc HIGH but ruv_sep ~0.50 -> a selective axis EXISTS, yet r_UV does NOT
                                     distinguish forget from benign -> mechanistic
                                     explanation for the broad over-refusal:
                                     r_UV pushes benign as hard as forget.
  * cv_acc HIGH, ruv_sep HIGH,
    cos_wr low                    -> selective axis exists and is MISALIGNED with
                                     r_UV -> a discriminative-direction redirection
                                     is worth trying (potential positive, 5.4).
  * cv_acc HIGH and cos_wr ~1     -> r_UV already ~ the discriminative axis; a
                                     "better direction" won't help.

Usage (GPU node):
  python scripts/linear_probe.py --model_family Qwen3-30B-A3B \
      --model_path Qwen/Qwen3-30B-A3B --layer 36 --n 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.model_utils.model_loader import load_model
from src.dataset_utils import load_dataset_to_get_direction
from src.generate_directions import generate_candidate_directions


def _load_questions(path, n):
    d = json.load(open(path))
    qs = [(x.get("question") or x.get("instruction") or "").strip() for x in d]
    qs = [q for q in qs if q]
    return qs[:n]


def _capture_all_layers(model_base, prompts, n_blocks):
    """acts[b] = np.array [n_prompts, hidden] of last-token residual at the INPUT to
    block b -- the same tap point generate_directions and the steering hook use."""
    buf = {b: [] for b in range(n_blocks)}
    handles = []

    def mk(b):
        def pre(module, inp):
            a = inp[0] if isinstance(inp, (tuple, list)) else inp
            buf[b].append(a[:, -1, :].float().detach().cpu().numpy())
        return pre

    for b in range(n_blocks):
        handles.append(model_base.model_block_modules[b].register_forward_pre_hook(mk(b)))
    try:
        with torch.no_grad():
            for p in prompts:
                enc = model_base.tokenize_instructions_fn(instructions=[p])
                model_base.model(
                    input_ids=enc.input_ids.to(model_base.model.device),
                    attention_mask=enc.attention_mask.to(model_base.model.device),
                )
    finally:
        for h in handles:
            h.remove()
    return {b: np.concatenate(buf[b], axis=0) for b in range(n_blocks)}


def _threshold_acc(proj, y):
    """Best-threshold (midpoint of class means) accuracy of a 1-D projection.
    IN-SAMPLE; orientation-invariant."""
    thr = 0.5 * (proj[y == 0].mean() + proj[y == 1].mean())
    acc = ((proj > thr).astype(float) == y).mean()
    return float(max(acc, 1 - acc))


def _probe_layer(Xf, Xr, r_uv):
    """forget=Xf (label 0), retain=Xr (label 1), r_uv = deployed direction at layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.concatenate([Xf, Xr], axis=0)
    y = np.concatenate([np.zeros(len(Xf)), np.ones(len(Xr))])

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    k = min(5, len(Xf), len(Xr))
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    cv_acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean())

    # r_UV as a 1-D probe on the SAME forget-vs-retain contrast
    r_hat = r_uv / (np.linalg.norm(r_uv) + 1e-8)
    ruv_sep = _threshold_acc(X @ r_hat, y)

    # angle between the fitted discriminative axis and the deployed r_UV
    clf.fit(X, y)
    w = clf.named_steps["logisticregression"].coef_.ravel()
    scale = clf.named_steps["standardscaler"].scale_
    w_input = w / scale  # standardized-space weights -> raw activation space
    cos = float(np.dot(w_input, r_uv) /
                (np.linalg.norm(w_input) * np.linalg.norm(r_uv) + 1e-8))
    return {"cv_acc": cv_acc, "ruv_sep": ruv_sep, "cos_wr": cos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--layer", type=int, default=36, help="steering layer, for highlighting")
    ap.add_argument("--n", type=int, default=150, help="prompts per split (classes balanced)")
    ap.add_argument("--forget", default="dataset/unlearning/wmdp_bio.json")
    ap.add_argument("--forget_edge", default="wmdp_bio")
    ap.add_argument("--benign", default="dataset/unlearning/mmlu_college_biology.json")
    ap.add_argument("--general", default="dataset/unlearning/factual_data.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.create({
        "positions": -1, "forget_edge": [args.forget_edge],
        "use_harmful": True, "use_unverified": False, "eval_batch_size": 8,
    })

    print(f"loading {args.model_family} ...")
    model_base = load_model(args.model_family, args.model_path, device)
    n_blocks = len(model_base.model_block_modules)

    # --- deployed direction r_UV, same code path as the real run / diag ---
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg, args.forget, instructions_only=True, use_harmful=True, use_unverified=False
    )
    print(f"computing r_UV from {len(harmful_train)} harmful / {len(forget_train)} forget ...")
    cand_dir = generate_candidate_directions(cfg, model_base, harmful_train, forget_train)
    # cand_dir: [n_eoi_pos, n_blocks, d]; last eoi position, matching diag (cfg.positions=-1)
    r_uv = {b: cand_dir[-1, b, :].float().cpu().numpy() for b in range(n_blocks)}

    # --- probe splits: reuse the SAME forget prompts that define r_UV ---
    benign_q = _load_questions(args.benign, 10_000)
    general_q = _load_questions(args.general, 10_000)
    n = min(args.n, len(forget_train), len(benign_q), len(general_q))
    forget_q, benign_q, general_q = forget_train[:n], benign_q[:n], general_q[:n]
    print(f"n per split = {n}  (forget {len(forget_train)}, benign {len(benign_q)}, "
          f"general {len(general_q)})")

    print("capturing activations (forget) ...")
    af = _capture_all_layers(model_base, forget_q, n_blocks)
    print("capturing activations (benign) ...")
    ab = _capture_all_layers(model_base, benign_q, n_blocks)
    print("capturing activations (general) ...")
    ag = _capture_all_layers(model_base, general_q, n_blocks)

    rows = []
    for b in range(n_blocks):
        fb = _probe_layer(af[b], ab[b], r_uv[b])
        fg = _probe_layer(af[b], ag[b], r_uv[b])
        rows.append((b, fb, fg))

    print("\n" + "=" * 82)
    print("LINEAR PROBE — last-token residual, 5-fold CV logistic regression")
    print("  fVb = forget vs benign-bio   |   fVg = forget vs general   (chance = 0.50)")
    print("  ruv_sep = deployed r_UV as 1-D probe on that contrast (in-sample threshold)")
    print("  cos     = cos(discriminative axis, r_UV);  ~1 => r_UV already discriminative")
    print("=" * 82)
    hdr = (f"{'blk':>3} | {'fVb_acc':>7} {'fVb_ruv':>7} {'fVb_cos':>7} | "
           f"{'fVg_acc':>7} {'fVg_ruv':>7} {'fVg_cos':>7}")
    print(hdr)
    print("-" * len(hdr))
    steer_idx = args.layer + 1  # steering taps block_modules[layer+1]
    for b, fb, fg in rows:
        mark = "  <-- steering layer" if b == steer_idx else ""
        print(f"{b:>3} | {fb['cv_acc']:>7.2f} {fb['ruv_sep']:>7.2f} {fb['cos_wr']:>7.2f} | "
              f"{fg['cv_acc']:>7.2f} {fg['ruv_sep']:>7.2f} {fg['cos_wr']:>7.2f}{mark}")

    peak_fb = max(rows, key=lambda r: r[1]["cv_acc"])
    peak_fg = max(rows, key=lambda r: r[2]["cv_acc"])
    print("\nSUMMARY")
    print(f"  peak forget-vs-benign : blk {peak_fb[0]}  acc={peak_fb[1]['cv_acc']:.2f}  "
          f"ruv_sep={peak_fb[1]['ruv_sep']:.2f}  cos={peak_fb[1]['cos_wr']:.2f}")
    print(f"  peak forget-vs-general: blk {peak_fg[0]}  acc={peak_fg[1]['cv_acc']:.2f}  "
          f"ruv_sep={peak_fg[1]['ruv_sep']:.2f}  cos={peak_fg[1]['cos_wr']:.2f}")
    if steer_idx < n_blocks:
        at = next(r for r in rows if r[0] == steer_idx)
        print(f"  @steering blk {at[0]}     : fVb acc={at[1]['cv_acc']:.2f} "
              f"ruv={at[1]['ruv_sep']:.2f} cos={at[1]['cos_wr']:.2f} | "
              f"fVg acc={at[2]['cv_acc']:.2f} ruv={at[2]['ruv_sep']:.2f} cos={at[2]['cos_wr']:.2f}")

    out = {
        "n_per_split": n, "n_blocks": n_blocks, "steer_block": steer_idx,
        "r_uv": "mean(harmful.json) - mean(forget); deployed direction",
        "rows": [{"blk": b, "fVb": fb, "fVg": fg} for b, fb, fg in rows],
    }
    os.makedirs("results/probe", exist_ok=True)
    dst = f"results/probe/linear_probe_{args.model_family}.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nsaved -> {dst}")


if __name__ == "__main__":
    main()
