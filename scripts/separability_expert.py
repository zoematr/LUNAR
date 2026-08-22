"""
Per-EXPERT activation separability (CASAL Fig. 19-21 style), forget vs retain.

The routing analysis showed forget and retain share experts. This asks the next
question: *inside* each expert, do the forget and retain tokens still separate,
or are they truly mixed? For every content token we record its hidden state (the
input the MoE routes) and the top-k experts it went to; then for each
(layer, expert) we PCA the tokens routed there and score how separable the
forget/retain clusters are (silhouette).

Outputs (small; drilling done offline in a notebook):
  - silhouette_expert.json : per-(layer, expert) silhouette [L][E]  (the heatmap)
  - expert_scatter.npz     : per-token 2D PCA coords (per layer) + top-k routing +
                             labels, so the notebook can scatter ANY (layer, expert)
                             on demand -- no need to pick layers in advance.

Usage (from repo root, on a GPU node):
  python scripts/separability_expert.py \
    --model_family Qwen3-30B-A3B --model_path Qwen/Qwen3-30B-A3B \
    --forget dataset/unlearning/wmdp_bio.json \
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
from src.model_utils.moe_model_base import resolve_text_config


def _text(d):
    return (d.get("question") or d.get("instruction") or "").strip()


def _template_span(model_base, prompts, max_probe=6):
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


def collect(model_base, prompts, num_layers, top_k, prefix_len, suffix_len, label, bucket):
    """For each content token: its hidden state + top-k experts, per layer.
    Appends into `bucket` (dict of lists keyed by layer) and returns labels list."""
    gate_out, mlp_in = {}, {}
    hooks = []

    def gate_hook(l):
        def h(m, i, o):
            gate_out[l] = (o if isinstance(o, torch.Tensor) else o[0]).detach().float()
        return h

    def mlp_hook(l):
        def h(m, i, o):
            mlp_in[l] = i[0].detach().float()          # input hidden states [b, seq, hidden]
        return h

    for l in range(num_layers):
        hooks.append(model_base._get_router(l).register_forward_hook(gate_hook(l)))
        hooks.append(model_base._get_mlp_modules()[l].register_forward_hook(mlp_hook(l)))

    model_base._eval()
    device = next(model_base.model.parameters()).device
    labels = []
    try:
        with torch.no_grad():
            for p in tqdm(prompts, desc=f"collect ({label})"):
                enc = model_base.tokenize_instructions_fn(instructions=[p]).to(device)
                seq = enc["input_ids"].shape[1]
                lo, hi = prefix_len, seq - suffix_len
                if hi - lo < 1:
                    lo, hi = 0, seq
                gate_out.clear(); mlp_in.clear()
                model_base._forward(enc)
                ntok = hi - lo
                for l in range(num_layers):
                    logits = gate_out[l].reshape(seq, -1)[lo:hi]      # [ntok, experts]
                    hid = mlp_in[l].reshape(seq, -1)[lo:hi]           # [ntok, hidden]
                    topk = torch.topk(logits, k=top_k, dim=-1).indices.cpu().numpy().astype(np.uint8)
                    bucket[l].append((hid.cpu().numpy().astype(np.float16), topk))
                labels.extend([label] * ntok)
    finally:
        for h in hooks:
            h.remove()
    return labels


def pca(X, k=2):
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def silhouette(X, y):
    if len(set(y.tolist())) < 2:
        return float("nan")
    sq = (X ** 2).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = y == y[i]; same[i] = False
        other = y != y[i]
        a = D[i, same].mean() if same.any() else 0.0
        b = D[i, other].mean() if other.any() else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_family", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--forget", default="dataset/unlearning/wmdp_bio_mcq.json")
    ap.add_argument("--retain", default="dataset/unlearning/mmlu_biology.json")
    ap.add_argument("--max_samples", type=int, default=250,
                    help="prompts PER set. Not 'all': this captures per-TOKEN hidden "
                         "states at every layer, so memory (not balance) is the limit.")
    ap.add_argument("--min_pts", type=int, default=20,
                    help="skip an expert at a layer if fewer tokens routed there")
    ap.add_argument("--save_path", default="run_results/separability")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fp = [_text(d) for d in json.load(open(args.forget)) if _text(d)][: args.max_samples]
    rp = [_text(d) for d in json.load(open(args.retain)) if _text(d)][: args.max_samples]
    print(f"forget {len(fp)}  retain {len(rp)}")

    model_base = load_model(args.model_family, args.model_path, device)
    num_layers = len(model_base.model_block_modules)
    num_experts = model_base._get_num_experts()
    top_k = resolve_text_config(model_base.model).num_experts_per_tok
    prefix_len, suffix_len = _template_span(model_base, fp + rp)
    print(f"layers={num_layers} experts={num_experts} top_k={top_k} suffix={suffix_len}")

    bucket = {l: [] for l in range(num_layers)}     # per layer: list of (hidden[ntok,h], topk[ntok,k])
    y = collect(model_base, fp, num_layers, top_k, prefix_len, suffix_len, 1, bucket)  # 1=forget
    y += collect(model_base, rp, num_layers, top_k, prefix_len, suffix_len, 0, bucket)  # 0=retain
    y = np.array(y, dtype=np.uint8)
    Nt = len(y)
    print(f"total content tokens: {Nt}")

    # per layer: stack tokens, global PCA-2D (for plotting), per-expert silhouette
    coords = np.zeros((Nt, num_layers, 2), dtype=np.float16)
    routing = np.zeros((Nt, num_layers, top_k), dtype=np.uint8)
    sil = [[float("nan")] * num_experts for _ in range(num_layers)]
    for l in range(num_layers):
        H = np.concatenate([b[0] for b in bucket[l]], axis=0).astype(np.float32)   # [Nt, hidden]
        R = np.concatenate([b[1] for b in bucket[l]], axis=0)                       # [Nt, k]
        Z = pca(H, 2)
        coords[:, l, :] = Z.astype(np.float16)
        routing[:, l, :] = R
        for e in range(num_experts):
            idx = np.where((R == e).any(axis=1))[0]
            if len(idx) >= args.min_pts:
                sil[l][e] = silhouette(Z[idx], y[idx])   # silhouette in the plotted 2D space
        bucket[l] = None                                  # free memory

    save_dir = Path(args.save_path) / args.model_family
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / "expert_scatter.npz",
                        coords=coords, routing=routing, labels=y)
    json.dump({"model_family": args.model_family, "forget": args.forget, "retain": args.retain,
               "num_layers": num_layers, "num_experts": num_experts, "top_k": top_k,
               "n_tokens": int(Nt), "min_pts": args.min_pts,
               "silhouette_expert": [[round(v, 4) if v == v else None for v in row] for row in sil]},
              open(save_dir / "silhouette_expert.json", "w"), indent=1)

    arr = np.array(sil, dtype=float)
    finite = arr[np.isfinite(arr)]
    print(f"\nper-expert silhouette: {finite.size} (layer,expert) cells scored")
    print(f"  mean {np.nanmean(arr):.3f}   max {np.nanmax(arr):.3f}")
    li, ei = np.unravel_index(np.nanargmax(arr), arr.shape)
    print(f"  most separable expert: layer {li}, expert {ei} = {arr[li, ei]:.3f}")
    print(f"saved to {save_dir}")


if __name__ == "__main__":
    main()
