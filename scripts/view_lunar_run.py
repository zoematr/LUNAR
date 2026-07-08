#!/usr/bin/env python3
"""
Inspect a finished LUNAR-MoE run WITHOUT rerunning it — reads only the saved JSONs.

Usage:
    python scripts/view_lunar_run.py run_results/completions/<model>/lunar_moe/<data>
    python scripts/view_lunar_run.py <run_dir> --samples 5   # more example generations

Shows, in order:
  1. Layer sweep (Procedure 2): s1/s2/(s1-s2) per candidate layer + the selected layer.
  2. Long-generation health on the sweep: how many responses ran to the token budget.
  3. A few sample sweep generations at the selected layer (read them).
  4. Final unlearning metrics: forget vs retained_edge vs factual (probs / rougeL / ppl).
  5. Long-generation health + sample generations on the final eval, per split.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _bar(x, width=24, lo=None, hi=None):
    """Tiny text bar for a value in [lo, hi]."""
    if lo is None or hi is None:
        return ""
    if hi - lo < 1e-9:
        return ""
    frac = max(0.0, min(1.0, (x - lo) / (hi - lo)))
    return "█" * int(frac * width)


def show_sweep(run_dir, samples):
    sweep = _load(os.path.join(run_dir, "layer_sweep.json"))
    if sweep is None:
        print("· no layer_sweep.json (run used a hardcoded layer_modified, not the sweep)\n")
        return None
    res = {int(k): v for k, v in sweep["results"].items()}
    layers = sorted(res)
    scores = [res[l]["score"] for l in layers]
    lo, hi = min(scores), max(scores)
    sel = sweep["selected_layer"]

    print("=" * 74)
    print(f"LAYER SWEEP (Procedure 2)   n_forget={sweep['n_forget']}  "
          f"coeff={sweep['coeff']}  max_new_tokens={sweep.get('max_new_tokens','?')}")
    print("=" * 74)
    print(f"{'layer':>6} {'s1(refuse)':>11} {'s2(answer)':>11} {'s1-s2':>8} "
          f"{'tok_mean':>9} {'@max':>7}   score")
    for l in layers:
        r = res[l]
        star = "  <-- selected" if l == sel else ""
        atmax = f"{r.get('n_at_max_tokens','?')}"
        print(f"{l:>6} {r['s1']:>11.3f} {r['s2']:>11.3f} {r['score']:>8.3f} "
              f"{r.get('resp_tokens_mean',0):>9.0f} {atmax:>7}   {_bar(r['score'], lo=lo, hi=hi)}{star}")
    print(f"\n  >>> SELECTED LAYER: {sel}   (s1-s2 = {res[sel]['score']:.3f})")

    # long-generation red flag
    flagged = [l for l in layers if res[l].get("n_at_max_tokens", 0) > 0]
    if flagged:
        print(f"  !! long-generation flag: layers {flagged} had responses at the token budget "
              f"(runaway/truncated).")
    else:
        print("  ok: no responses hit the token budget on any candidate layer.")

    # sample generations at the selected layer
    gens = _load(os.path.join(run_dir, "layer_sweep_generations.json"))
    if gens and str(sel) in gens:
        print(f"\n  --- sample generations @ selected layer {sel} "
              f"(showing {samples}) ---")
        for g in gens[str(sel)][:samples]:
            print(f"    Q: {g['prompt'][:110]}")
            print(f"    A[{g['resp_tokens']} tok]: {g['response'][:200]}")
            print(f"    gt: {str(g['original_answer'])[:90]}\n")
    print()
    return sel


def show_final(run_dir, sel, samples, max_new_tokens_guess=64):
    # find the forget_<layers>.json (there may be several; take the newest)
    cands = sorted(glob.glob(os.path.join(run_dir, "forget_*.json")),
                   key=os.path.getmtime, reverse=True)
    if not cands:
        print("· no forget_*.json (final eval not found in this dir)\n")
        return
    logs = _load(cands[0])
    print("=" * 74)
    print(f"FINAL UNLEARNING EVAL   ({os.path.basename(cands[0])})")
    print("=" * 74)

    splits = [k for k in ("forget", "retained_edge", "factual_data") if k in logs]
    metric_keys = ["probs", "rougeL_recall", "rouge1_recall", "perplexity", "mrr", "hit_rate"]
    header = f"{'metric':>16} " + " ".join(f"{s:>16}" for s in splits)
    print(header)
    print("-" * len(header))
    for m in metric_keys:
        row = f"{m:>16} "
        any_present = False
        for s in splits:
            v = logs[s].get(m, None)
            if isinstance(v, (int, float)):
                row += f"{v:>16.4f} "
                any_present = True
            else:
                row += f"{'—':>16} "
        if any_present:
            print(row)
    print("\n  read it as:  forget probs LOW = knowledge removed;  "
          "retained_edge / factual probs HIGH = preserved.")
    print("  (forget=hazardous bio  |  retained_edge=in-file retain  |  factual=general knowledge)\n")

    # long-generation health + samples from generated_text = [input, generated, gt]
    for s in splits:
        gt = logs[s].get("generated_text", [])
        if not gt:
            continue
        lens = [len(str(g[1]).split()) for g in gt]  # word proxy (no tokenizer here)
        long_ct = sum(1 for x in lens if x >= max_new_tokens_guess * 0.9)
        print(f"  [{s}]  n={len(gt)}  resp_words mean={sum(lens)/len(lens):.0f} "
              f"max={max(lens)}  (~{long_ct} near budget)")
        for inp, gen, truth in gt[:samples]:
            print(f"     Q: {str(inp)[:100]}")
            print(f"     A: {str(gen)[:180]}")
            print(f"     gt: {str(truth)[:80]}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="run_results/completions/<model>/lunar_moe/<data>")
    ap.add_argument("--samples", type=int, default=3, help="example generations to print per section")
    args = ap.parse_args()

    if not os.path.isdir(args.run_dir):
        raise SystemExit(f"not a directory: {args.run_dir}")

    print(f"\nRun: {args.run_dir}\n")
    sweep = _load(os.path.join(args.run_dir, "layer_sweep.json"))
    mnt = sweep.get("max_new_tokens", 64) if sweep else 64
    sel = show_sweep(args.run_dir, args.samples)
    show_final(args.run_dir, sel, args.samples, max_new_tokens_guess=mnt)


if __name__ == "__main__":
    main()
