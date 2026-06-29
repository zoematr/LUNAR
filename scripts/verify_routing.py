"""
Transparent verification of the routing slide numbers, straight from the run's
JSON files. Every reported figure is shown as an explicit operation on the raw
`applied_weight` / `freq` arrays, plus by-hand spot checks you can confirm by
opening the JSON and reading a single cell.

Usage (from repo root, after the run):
  python scripts/verify_routing.py
  python scripts/verify_routing.py --mode content      # default
  python scripts/verify_routing.py --dir run_results/routing_single/Qwen3-30B-A3B
"""

import argparse
import json
import math
from pathlib import Path


def load(d, tag, mode):
    suffix = f".{mode}" if mode else ""
    return json.load(open(Path(d) / f"{tag}{suffix}.json"))


def argsort_desc(row):
    return sorted(range(len(row)), key=lambda i: row[i], reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="run_results/routing_single/Qwen3-30B-A3B")
    ap.add_argument("--mode", default="content", help="'content', 'all', or '' for old files")
    ap.add_argument("--haz", default="wmdp_bio")
    ap.add_argument("--ben", default="mmlu_college_biology")
    args = ap.parse_args()

    H = load(args.dir, args.haz, args.mode)
    B = load(args.dir, args.ben, args.mode)
    HW, BW = H["applied_weight"], B["applied_weight"]   # raw nested lists [L][E]
    HF = H["freq"]
    BF = B["freq"]
    L, E = len(HW), len(HW[0])
    k = H["top_k"]

    print(f"=== files ===")
    print(f"  hazardous: {args.haz}{'.'+args.mode if args.mode else ''}.json  "
          f"(route_tokens={H.get('route_tokens')}, n={H['n_prompts']})")
    print(f"  benign   : {args.ben}{'.'+args.mode if args.mode else ''}.json  "
          f"(route_tokens={B.get('route_tokens')}, n={B['n_prompts']})")
    print(f"  applied_weight shape: {L} layers x {E} experts,  top_k={k}\n")

    # ---- sanity: each layer's applied_weight sums to ~1 (it is a distribution) ----
    s0 = sum(HW[0])
    print(f"[sanity] sum of applied_weight at layer 0 = {s0:.3f}  (should be ~1.0)\n")

    # ---- Bullet 2: concentration = exp(entropy of freq) per layer, averaged ----
    def eff_experts(F):
        vals = []
        for l in range(L):
            ent = -sum(p * math.log(p + 1e-12) for p in F[l])
            vals.append(math.exp(ent))
        return sum(vals) / len(vals)
    print("Bullet 2  effective experts/layer (exp of freq-entropy, of %d):" % E)
    print(f"          hazardous {eff_experts(HF):.1f}   benign {eff_experts(BF):.1f}\n")

    # ---- Bullet 1 + 3: per-layer exposure / self / alignment ----
    align = []
    for l in range(L):
        haz_top = argsort_desc(HW[l])[:k]      # hazardous's top-k experts at layer l
        ben_top = argsort_desc(BW[l])[:k]      # benign's   top-k experts at layer l
        exposure = sum(BW[l][e] for e in haz_top)   # benign mass on hazardous's experts
        ben_self = sum(BW[l][e] for e in ben_top)   # benign mass on its own experts
        align.append(exposure / ben_self)
    mean_align = sum(align) / L
    print(f"Bullet 1  mean alignment = mean_l( sum_benign[hazTop] / sum_benign[benTop] ) = {mean_align:.2f}")
    print(f"Bullet 3a layers with alignment >= 0.70: {sum(a >= 0.70 for a in align)} of {L}")

    lb = min(range(L), key=lambda l: align[l])  # least-entangled layer
    haz_top_lb = set(argsort_desc(HW[lb])[:k])
    ben_top_lb = set(argsort_desc(BW[lb])[:k])
    print(f"Bullet 3b least-entangled layer {lb} (alignment {align[lb]:.2f}): "
          f"{len(haz_top_lb & ben_top_lb)}/{k} top experts shared")

    # ---- Bullet 3c: UOE target = highest single hazardous expert weight, anywhere ----
    la = max(range(L), key=lambda l: max(HW[l]))
    ue = argsort_desc(HW[la])[0]
    print(f"Bullet 3c UOE target = argmax over all cells of hazardous applied_weight "
          f"-> layer {la}, expert {ue}")
    print(f"          hazardous applied_weight[{la}][{ue}] = {HW[la][ue]:.3f}")
    print(f"          benign    applied_weight[{la}][{ue}] = {BW[la][ue]:.3f}"
          f"  ({BW[la][ue]/HW[la][ue]:.0%} as much)\n")

    # ---- BY-HAND spot check: open the JSON and read these exact cells ----
    print("[spot check] open the JSON files and confirm these raw cells:")
    print(f"   wmdp_bio.{args.mode}.json            -> applied_weight[{la}][{ue}] == {HW[la][ue]:.6f}")
    print(f"   mmlu_college_biology.{args.mode}.json -> applied_weight[{la}][{ue}] == {BW[la][ue]:.6f}")


if __name__ == "__main__":
    main()
