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


def _template_span(model_base, prompts, max_probe=6):
    """Empirically count the fixed chat-template tokens that wrap every prompt.

    The chat template (system prompt, role markers, the empty <think></think>
    block) is identical across prompts, so it shows up as the leading and
    trailing tokens shared by differently-worded prompts. Returns
    (prefix_len, suffix_len): the number of template tokens before and after the
    question span. Robust to BPE boundary merging; takes the *minimum* shared
    run across several probe prompts to avoid counting accidentally-shared
    leading/trailing question tokens.
    """
    idlists = []
    for p in prompts[:max_probe]:
        enc = model_base.tokenize_instructions_fn(instructions=[p])
        idlists.append(enc["input_ids"][0].tolist())
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


def collect_routing(model_base, prompts, top_k, num_layers, num_experts,
                    norm_topk_prob=True, token_mode="content",
                    prefix_len=0, suffix_len=0):
    """Return per-layer routing stats accumulated over the selected tokens.

    token_mode controls which token positions are counted:
      "all"     -> every token in the wrapped prompt, incl. the chat template
                   (the original behavior; template tokens dilute the signal).
      "content" -> only the question-span tokens (template stripped).
      "last"    -> only the last question token (CASAL-style single position).
    """
    counts        = np.zeros((num_layers, num_experts), dtype=np.float64)
    weight_sums   = np.zeros((num_layers, num_experts), dtype=np.float64)  # full-softmax prob
    applied_sums  = np.zeros((num_layers, num_experts), dtype=np.float64)  # actual top-k weight
    entropy_sums  = np.zeros(num_layers, dtype=np.float64)
    token_counts  = np.zeros(num_layers, dtype=np.int64)
    # Contiguous row slice [lo:hi] selecting which token positions to count for
    # the current prompt; set before each forward, read by every layer's hook.
    # Using an integer slice (not a device tensor) keeps this correct even when
    # device_map="auto" shards layers across GPUs.
    span = {"lo": 0, "hi": None}
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, inp, out):
            # Router/gate is an nn.Linear -> logits; some return a tuple.
            logits = out if isinstance(out, torch.Tensor) else out[0]
            logits = logits.detach().float().reshape(-1, logits.shape[-1])  # [tokens, experts]
            logits = logits[span["lo"]:span["hi"]]                          # selected positions
            if logits.shape[0] == 0:
                return
            probs = torch.softmax(logits, dim=-1)
            topk_w, topk_idx = torch.topk(probs, k=top_k, dim=-1)           # values + indices
            # --- selection frequency (how often each expert is in the top-k) ---
            np.add.at(counts[layer_idx], topk_idx.reshape(-1).cpu().numpy(), 1)
            # --- full-softmax probability mass (kept for reference; tends to be flat) ---
            weight_sums[layer_idx] += probs.sum(dim=0).cpu().numpy()
            # --- ACTUAL applied weight: the renormalized top-k weight that multiplies
            #     each selected expert's output (norm_topk_prob), scattered back per expert ---
            if norm_topk_prob:
                topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
            applied = torch.zeros_like(probs)
            applied.scatter_(1, topk_idx, topk_w)                           # [tokens, experts]
            applied_sums[layer_idx] += applied.sum(dim=0).cpu().numpy()
            # --- gate entropy (per-token, full softmax) ---
            H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            entropy_sums[layer_idx] += float(H.sum().cpu())
            token_counts[layer_idx] += logits.shape[0]
        return hook_fn

    for layer_idx in range(num_layers):
        hooks.append(model_base._get_router(layer_idx).register_forward_hook(make_hook(layer_idx)))

    model_base._eval()
    device = next(model_base.model.parameters()).device
    n_fallback = 0
    try:
        with torch.no_grad():
            for prompt in tqdm(prompts, desc=f"forward ({token_mode})"):
                enc = model_base.tokenize_instructions_fn(instructions=[prompt]).to(device)
                seq_len = enc["input_ids"].shape[1]
                if token_mode == "all":
                    lo, hi = 0, seq_len
                else:
                    c_lo, c_hi = prefix_len, seq_len - suffix_len
                    if c_hi - c_lo < 1:            # degenerate: prompt shorter than template
                        lo, hi = 0, seq_len        # fall back to all tokens
                        n_fallback += 1
                    elif token_mode == "last":
                        lo, hi = c_hi - 1, c_hi     # last question token only
                    else:                          # "content"
                        lo, hi = c_lo, c_hi
                span["lo"], span["hi"] = lo, hi
                model_base._forward(enc)
    finally:
        for h in hooks:
            h.remove()
    if n_fallback:
        print(f"  [warn] {n_fallback}/{len(prompts)} prompt(s) shorter than the "
              f"template span; counted all their tokens")

    return counts, weight_sums, applied_sums, entropy_sums, token_counts


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
    parser.add_argument("--route_tokens", choices=["all", "content", "last"],
                        default="content",
                        help="Which token positions to record routing for: 'all' "
                             "(incl. chat template), 'content' (question span only), "
                             "or 'last' (last question token, CASAL-style)")
    parser.add_argument("--offset", type=int, default=0,
                        help="Skip the first N prompts before capping (for split-half "
                             "noise-floor runs, e.g. [0:600] vs --offset 600)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle prompts (with --seed) before offset/cap, so "
                             "subsets are not biased by file ordering")
    parser.add_argument("--seed", type=int, default=0)
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
    if args.shuffle:
        import random
        random.Random(args.seed).shuffle(prompts)
    if args.offset:
        prompts = prompts[args.offset:]
    if args.max_samples:
        prompts = prompts[: args.max_samples]
    tag = args.dataset_tag or Path(args.data_path).stem
    print(f"dataset '{tag}': {len(prompts)} prompts "
          f"(mode={args.route_tokens}, offset={args.offset}, shuffle={args.shuffle})")

    # ── load model ──────────────────────────────────────────────────────
    print(f"Loading {args.model_family} ...")
    model_base = load_model(args.model_family, args.model_path, device)
    num_layers  = len(model_base.model_block_modules)
    num_experts = model_base._get_num_experts()
    text_cfg    = resolve_text_config(model_base.model)
    top_k       = text_cfg.num_experts_per_tok
    norm_topk   = getattr(text_cfg, "norm_topk_prob", True)
    print(f"layers={num_layers}  experts={num_experts}  top_k={top_k}  norm_topk_prob={norm_topk}")

    # ── collect ─────────────────────────────────────────────────────────
    prefix_len, suffix_len = _template_span(model_base, prompts)
    print(f"chat-template tokens: prefix={prefix_len}  suffix={suffix_len}  "
          f"-> routing recorded over '{args.route_tokens}' tokens")
    counts, weight_sums, applied_sums, entropy_sums, token_counts = collect_routing(
        model_base, prompts, top_k, num_layers, num_experts,
        norm_topk_prob=norm_topk, token_mode=args.route_tokens,
        prefix_len=prefix_len, suffix_len=suffix_len,
    )

    # ── normalise ───────────────────────────────────────────────────────
    freq = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)         # selection frequency
    avg_weight = weight_sums / (token_counts[:, None] + 1e-9)          # mean full-softmax prob (flat)
    applied_weight = applied_sums / (token_counts[:, None] + 1e-9)     # mean APPLIED top-k weight (impact)
    entropy = entropy_sums / (token_counts + 1e-9)                     # mean gate entropy/layer
    top_experts = [np.argsort(freq[l])[::-1][:10].astype(int).tolist()
                   for l in range(num_layers)]
    # experts that contribute the most output mass per layer (by applied weight)
    top_impact = [np.argsort(applied_weight[l])[::-1][:10].astype(int).tolist()
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
        "norm_topk_prob": bool(norm_topk),
        "route_tokens": args.route_tokens,                   # which positions were counted
        "template_prefix_tokens": int(prefix_len),
        "template_suffix_tokens": int(suffix_len),
        "offset": args.offset,
        "shuffle": bool(args.shuffle),
        "freq": freq.round(6).tolist(),                      # [L][E] selection frequency
        "avg_weight": avg_weight.round(6).tolist(),          # [L][E] full-softmax mean (reference)
        "applied_weight": applied_weight.round(6).tolist(),  # [L][E] actual output weight (impact)
        "entropy_per_layer": entropy.round(6).tolist(),      # [L]
        "top_experts_per_layer": top_experts,                # [L][10] by frequency
        "top_impact_per_layer": top_impact,                  # [L][10] by applied weight
    }

    save_dir = Path(args.save_path) / args.model_family
    save_dir.mkdir(parents=True, exist_ok=True)
    # Encode the token mode (and offset, if any) in the filename so different
    # runs don't silently overwrite each other — e.g. wmdp_bio.content.json vs
    # wmdp_bio.all.json, and split-half wmdp_bio.content.off600.json.
    off = f".off{args.offset}" if args.offset else ""
    out_path = save_dir / f"{tag}.{args.route_tokens}{off}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n=== {tag} routing summary ===")
    print(f"  mean gate entropy (all layers): {entropy.mean():.4f}")
    print(f"  layer with lowest entropy (most concentrated): "
          f"{int(entropy.argmin())} ({entropy.min():.4f})")
    print(f"  Saved to {out_path}")


if __name__ == "__main__":
    main()
