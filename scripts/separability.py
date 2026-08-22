"""
CASAL-style activation-space separability for forget vs retain.

For every transformer layer, capture the residual-stream activation at the LAST
QUESTION token (the CASAL/LUNAR convention) for each forget prompt and each
retain prompt. Then:
  - compute a per-layer silhouette score = how separable the forget and retain
    clusters are in activation space (higher = cleaner separation), mirroring
    CASAL's known-vs-unknown analysis (Appendix H.4), and
  - save the activations so you can PCA-plot them offline (CASAL Fig. 2D / 19-21
    style), colored by forget vs retain.

This answers: even though forget/retain share experts (routing analysis), is
there a *direction* in representation space that tells them apart? -> whether a
LUNAR-style activation redirection is feasible.

Usage (from repo root):
  python scripts/separability.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --forget dataset/unlearning/wmdp_bio_mcq.json \
    --retain dataset/unlearning/mmlu_college_biology.json \
    --max_samples 160
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from src.model_utils.model_loader import load_model


def _text(d):
    return (d.get("question") or d.get("instruction") or "").strip()


def _template_span(model_base, prompts, max_probe=6):
    """Tokens of the fixed chat template wrapping each prompt (prefix, suffix),
    found as the leading/trailing tokens shared across differently-worded prompts."""
    idlists = []
    for p in prompts[:max_probe]:
        idlists.append(model_base.tokenize_instructions_fn(instructions=[p])["input_ids"][0].tolist())
    if len(idlists) < 2:
        return 0, 0
    ref = idlists[0]
    s_min = e_min = len(ref)
    for other in idlists[1:]:
        n = min(len(ref), len(other))
        s = 0
        while s < n and ref[s] == other[s]:
            s += 1
        e = 0
        while e < n and ref[-1 - e] == other[-1 - e]:
            e += 1
        s_min, e_min = min(s_min, s), min(e_min, e)
    return s_min, e_min


def collect_last_token_acts(model_base, prompts, num_layers, suffix_len):
    """Residual-stream activation at the last question token, per layer.
    Returns array [num_layers, n_prompts, hidden]."""
    store = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out  # [batch, seq, hidden]
            store[layer_idx] = h.detach().float()
        return hook

    for l in range(num_layers):
        hooks.append(model_base.model_block_modules[l].register_forward_hook(make_hook(l)))

    model_base._eval()
    device = next(model_base.model.parameters()).device
    acts = [[] for _ in range(num_layers)]
    try:
        with torch.no_grad():
            for p in tqdm(prompts, desc="activations"):
                enc = model_base.tokenize_instructions_fn(instructions=[p]).to(device)
                seq = enc["input_ids"].shape[1]
                pos = max(seq - suffix_len - 1, 0)          # last question token
                store.clear()
                model_base._forward(enc)
                for l in range(num_layers):
                    acts[l].append(store[l][0, pos].cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
    return np.stack([np.stack(a) for a in acts])           # [L, n, hidden]


def pca(X, k=2):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def silhouette_samples(X, y):
    """Per-point silhouette for 2 labels (CASAL H.4). Euclidean. Returns the s vector.
    Silhouette is defined per point, so cluster sizes need NOT be equal; but the
    overall MEAN is point-weighted, so a larger cluster dominates it. Callers should
    also report per-cluster means so the smaller cluster stays visible."""
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = y == y[i]; same[i] = False
        other = y != y[i]
        a = D[i, same].mean() if same.any() else 0.0
        b = D[i, other].mean() if other.any() else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return s


def _sil_summary(X, y):
    """(overall mean, forget-cluster mean, retain-cluster mean). y: 1=forget, 0=retain."""
    s = silhouette_samples(X, y)
    return (float(s.mean()),
            float(s[y == 1].mean()) if (y == 1).any() else float("nan"),
            float(s[y == 0].mean()) if (y == 0).any() else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--forget", default="dataset/unlearning/wmdp_bio_mcq.json")
    ap.add_argument("--retain", default="dataset/unlearning/mmlu_biology.json")
    ap.add_argument("--max_samples", type=int, default=1300,
                    help="cap PER set; silhouette needs no balance, so default uses all "
                         "(forget ~1273, retain ~475). Per-cluster means are reported.")
    ap.add_argument("--save_path", default="run_results/separability")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    f_prompts = [_text(d) for d in json.load(open(args.forget)) if _text(d)][: args.max_samples]
    r_prompts = [_text(d) for d in json.load(open(args.retain)) if _text(d)][: args.max_samples]
    print(f"forget: {len(f_prompts)}  retain: {len(r_prompts)}")

    model_base = load_model(args.model_family, args.model_path, device)
    num_layers = len(model_base.model_block_modules)
    _, suffix_len = _template_span(model_base, f_prompts + r_prompts)
    print(f"layers={num_layers}  template suffix tokens stripped: {suffix_len}")

    Af = collect_last_token_acts(model_base, f_prompts, num_layers, suffix_len)  # [L,nf,h]
    Ar = collect_last_token_acts(model_base, r_prompts, num_layers, suffix_len)  # [L,nr,h]
    y = np.array([1] * Af.shape[1] + [0] * Ar.shape[1])   # 1=forget, 0=retain

    # overall + per-cluster silhouette, full-dim and PCA-2D, per layer
    sil_full, silF_forget, silF_retain = [], [], []
    sil_2d, sil2_forget, sil2_retain = [], [], []
    for l in range(num_layers):
        X = np.concatenate([Af[l], Ar[l]], axis=0)
        o, f, r = _sil_summary(X, y);            sil_full.append(o); silF_forget.append(f); silF_retain.append(r)
        o, f, r = _sil_summary(pca(X, 2), y);    sil_2d.append(o);   sil2_forget.append(f); sil2_retain.append(r)

    save_dir = Path(args.save_path) / args.model_family
    save_dir.mkdir(parents=True, exist_ok=True)
    # activations for offline PCA plotting (float16 to save space)
    np.savez_compressed(save_dir / "activations.npz",
                        forget=Af.astype(np.float16), retain=Ar.astype(np.float16),
                        labels=y)
    rnd = lambda xs: [round(v, 4) for v in xs]
    json.dump({"model_family": args.model_family,
               "forget": args.forget, "retain": args.retain,
               "n_forget": int(Af.shape[1]), "n_retain": int(Ar.shape[1]),
               "num_layers": num_layers, "suffix_tokens": int(suffix_len),
               "note": "silhouette needs no class balance; overall mean is point-weighted "
                       "(larger cluster dominates), so per-cluster means are reported too.",
               "silhouette_full": rnd(sil_full),
               "silhouette_full_forget": rnd(silF_forget),
               "silhouette_full_retain": rnd(silF_retain),
               "silhouette_pca2": rnd(sil_2d),
               "silhouette_pca2_forget": rnd(sil2_forget),
               "silhouette_pca2_retain": rnd(sil2_retain)},
              open(save_dir / "silhouette.json", "w"), indent=2)

    print(f"\nper-layer silhouette (higher = more separable)  n_forget={Af.shape[1]} "
          f"n_retain={Ar.shape[1]} (unbalanced OK)")
    print(f"{'layer':>5} | {'full':>7} {'full_f':>7} {'full_r':>7} | {'pca2':>7} {'pca2_f':>7} {'pca2_r':>7}")
    for l in range(num_layers):
        print(f"{l:5d} | {sil_full[l]:7.3f} {silF_forget[l]:7.3f} {silF_retain[l]:7.3f} | "
              f"{sil_2d[l]:7.3f} {sil2_forget[l]:7.3f} {sil2_retain[l]:7.3f}")
    bl = int(np.argmax(sil_2d))
    print(f"\nmost separable layer (pca-2d overall): {bl}  (silhouette {sil_2d[bl]:.3f})")
    print(f"mean silhouette: full {np.mean(sil_full):.3f}  pca-2d {np.mean(sil_2d):.3f}")
    print(f"saved activations + scores to {save_dir}")


if __name__ == "__main__":
    main()
