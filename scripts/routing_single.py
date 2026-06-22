"""
Single-dataset MoE routing analysis.

Forward-passes every prompt in ONE dataset and records, per transformer layer:
  - expert selection frequency   (how often each expert is in the top-k)
  - average routing weight        (mean softmax prob per expert, over all tokens)
  - gate entropy                  (mean Shannon entropy of the gate distribution)
  - top experts                   (the most-selected experts per layer)

Decoupled by design: run it once per dataset, then compare the saved JSONs
offline (e.g. in a Colab notebook). No forget/retain coupling.

Output: a single JSON (easy to load with json.load + np.array) at
  run_results/routing_single/<model_family>/<dataset_tag>.json

Usage:
  python scripts/routing_single.py \\
      --model_family Qwen3-30B-A3B \\
      --model_path Qwen/Qwen3-30B-A3B \\
      --data_path dataset/unlearning/wmdp_bio.json \\
      --dataset_tag wmdp_bio \\
      --max_samples 169
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


def collect_routing(model_base, prompts, top_k, num_layers, num_experts):
    """Return per-layer routing stats accumulated over all prompts."""
    counts       = np.zeros((num_layers, num_experts), dtype=np.float64)
    weight_sums  = np.zeros((num_layers, num_experts), dtype=np.float64)
    entropy_sums = np.zeros(num_layers, dtype=np.float64)
    token_counts = np.zeros(num_layers, dtype=np.int64)
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            # Router/gate is an nn.Linear -> logits; some return a tuple.
            logits = out if isinstance(out, torch.Tensor) else out[0]
            logits = logits.detach().float().reshape(-1, logits.shape[-1])  # [tokens, experts]
            probs = torch.softmax(logits, dim=-1)
            topk_idx = torch.topk(probs, k=top_k, dim=-1).indices
            np.add.at(counts[layer_idx], topk_idx.reshape(-1).cpu().numpy(), 1)
            weight_sums[layer_idx] += probs.sum(dim=0).cpu().numpy()
            H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            entropy_sums[layer_idx] += float(H.sum().cpu())
            token_counts[layer_idx] += logits.shape[0]
        return hook_fn

    for layer_idx in range(num_layers):
        hooks.append(model_base._get_router(layer_idx).register_forward_hook(make_hook(layer_idx)))

    model_base._eval()
    device = next(model_base.model.parameters()).device
    try:
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="forward passes"):
                enc = model_base.tokenize_instructions_fn(instructions=[prompt]).to(device)
                model_base._forward(enc)
    finally:
        for h in hooks:
            h.remove()

    return counts, weight_sums, entropy_sums, token_counts


def main():
    parser = argparse.ArgumentParser(description="Single-dataset MoE routing analysis")
    parser.add_argument("--model_family", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True,
                        help="JSON list of entries with a 'question' field")
    parser.add_argument("--dataset_tag", default=None,
                        help="Label for the output file (default: data file stem)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap number of prompts (for balancing across datasets)")
    parser.add_argument("--save_path", default="run_results/routing_single")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else args.device if args.device != "auto" else "cpu")
    print(f"device: {device}")

    # ── load prompts ────────────────────────────────────────────────────
    with open(args.data_path) as f:
        data = json.load(f)
    # Datasets use either "question" (WMDP/MMLU) or "instruction" (harmful/Alpaca).
    def _text(d):
        return (d.get("question") or d.get("instruction") or "").strip()
    prompts = [_text(d) for d in data if _text(d)]
    if args.max_samples:
        prompts = prompts[: args.max_samples]
    tag = args.dataset_tag or Path(args.data_path).stem
    print(f"dataset '{tag}': {len(prompts)} prompts")

    # ── load model ──────────────────────────────────────────────────────
    print(f"Loading {args.model_family} ...")
    model_base = load_model(args.model_family, args.model_path, device)
    num_layers  = len(model_base.model_block_modules)
    num_experts = model_base._get_num_experts()
    top_k       = resolve_text_config(model_base.model).num_experts_per_tok
    print(f"layers={num_layers}  experts={num_experts}  top_k={top_k}")

    # ── collect ─────────────────────────────────────────────────────────
    counts, weight_sums, entropy_sums, token_counts = collect_routing(
        model_base, prompts, top_k, num_layers, num_experts
    )

    # ── normalise ───────────────────────────────────────────────────────
    freq = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)        # selection frequency
    avg_weight = weight_sums / (token_counts[:, None] + 1e-9)         # mean prob per expert
    entropy = entropy_sums / (token_counts + 1e-9)                    # mean gate entropy/layer
    top_experts = [np.argsort(freq[l])[::-1][:10].astype(int).tolist()
                   for l in range(num_layers)]

    out = {
        "model_family": args.model_family,
        "model_path": args.model_path,
        "dataset": tag,
        "data_path": args.data_path,
        "n_prompts": len(prompts),
        "num_layers": num_layers,
        "num_experts": num_experts,
        "top_k": top_k,
        "freq": freq.round(6).tolist(),                  # [num_layers][num_experts]
        "avg_weight": avg_weight.round(6).tolist(),      # [num_layers][num_experts]
        "entropy_per_layer": entropy.round(6).tolist(),  # [num_layers]
        "top_experts_per_layer": top_experts,            # [num_layers][10]
    }

    save_dir = Path(args.save_path) / args.model_family
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {tag} routing summary ===")
    print(f"  mean gate entropy (all layers): {entropy.mean():.4f}")
    print(f"  layer with lowest entropy (most concentrated): "
          f"{int(entropy.argmin())} ({entropy.min():.4f})")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
