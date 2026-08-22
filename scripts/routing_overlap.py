#!/usr/bin/env python3
"""
routing_overlap.py — OFFLINE pairwise routing-overlap between domains.

Reads the per-domain routing JSONs written by routing_single.py and computes, for
every pair of domains, the similarity of their expert-usage distributions:
  * cosine similarity   (1 = identical routing, 0 = orthogonal)
  * Jensen-Shannon divergence, base-2  (0 = identical, 1 = disjoint)
both PER LAYER and averaged over layers, on two views:
  * freq            (how OFTEN each expert is selected)
  * applied_weight  (how much OUTPUT MASS each expert carries)

This is the F1 overlap number. High cosine / low JS between hazardous-bio and
benign-bio => they route through the same experts => expert-level unlearning is not
viable and activation redirection is required. Runs on CPU, no GPU, no rerun.

Usage:
  python scripts/routing_overlap.py \
      --dir run_results/routing_single/Qwen3-30B-A3B --route_tokens content \
      --datasets wmdp_bio_mcq mmlu_biology general_mcq
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np


def _load(dir_, tag, route_tokens):
    p = os.path.join(dir_, f"{tag}.{route_tokens}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing routing file: {p}\n"
                         f"(run routing first, or check --route_tokens / --datasets)")
    return json.load(open(p))


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _js(p, q):
    """Jensen-Shannon divergence, base-2, for two non-negative vectors."""
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / (b[mask] + 1e-12))))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _pair_metrics(A, B):
    """A, B: [L][E] arrays. Returns per-layer cosine, per-layer JS, and their means."""
    L = A.shape[0]
    cos = np.array([_cos(A[l], B[l]) for l in range(L)])
    js = np.array([_js(A[l], B[l]) for l in range(L)])
    return cos, js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="run_results/routing_single/<model_family>")
    ap.add_argument("--route_tokens", default="content", choices=["all", "content", "last"])
    ap.add_argument("--datasets", nargs="+",
                    default=["wmdp_bio_mcq", "mmlu_biology", "general_mcq"])
    ap.add_argument("--views", nargs="+", default=["freq", "applied_weight"])
    args = ap.parse_args()

    data = {tag: _load(args.dir, tag, args.route_tokens) for tag in args.datasets}
    L = data[args.datasets[0]]["num_layers"]
    arrs = {view: {tag: np.array(data[tag][view], dtype=np.float64) for tag in args.datasets}
            for view in args.views}

    out = {"route_tokens": args.route_tokens, "datasets": args.datasets,
           "num_layers": L, "n_prompts": {t: data[t]["n_prompts"] for t in args.datasets},
           "pairs": {}}

    for view in args.views:
        print("\n" + "=" * 74)
        print(f"VIEW = {view}   (cosine: 1=identical routing | JS base-2: 0=identical)")
        print("=" * 74)
        for a, b in itertools.combinations(args.datasets, 2):
            cos, js = _pair_metrics(arrs[view][a], arrs[view][b])
            key = f"{a}__vs__{b}"
            out["pairs"].setdefault(key, {})[view] = {
                "cosine_per_layer": cos.round(4).tolist(),
                "js_per_layer": js.round(4).tolist(),
                "cosine_mean": round(float(cos.mean()), 4),
                "js_mean": round(float(js.mean()), 4),
                "cosine_min_layer": [int(cos.argmin()), round(float(cos.min()), 4)],
                "js_max_layer": [int(js.argmax()), round(float(js.max()), 4)],
            }
            print(f"\n  {a}  vs  {b}")
            print(f"    mean cosine = {cos.mean():.4f}   mean JS = {js.mean():.4f}")
            print(f"    most-divergent layer: cosine min {cos.min():.4f} @L{cos.argmin()} | "
                  f"JS max {js.max():.4f} @L{js.argmax()}")

    dst = os.path.join(args.dir, f"routing_overlap.{args.route_tokens}.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"\nsaved per-layer overlap -> {dst}")


if __name__ == "__main__":
    main()
