# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Routing frequency, weight, and entropy analysis for OLMoE.

For each transformer layer, tracks three things per expert:
  1. How often each expert is selected (routing frequency)
  2. The average routing weight when selected (contribution weight)
  3. Gate entropy — Shannon entropy of the full softmax over all experts

Outputs:
  - routing_freq_forget.npy        [n_layers, n_experts]  selection frequency
  - routing_freq_retain.npy        [n_layers, n_experts]
  - routing_weight_forget.npy      [n_layers, n_experts]  avg softmax probability per token (all experts)
  - routing_weight_retain.npy      [n_layers, n_experts]
  - routing_contrib_forget.npy     [n_layers, n_experts]  same as routing_weight (kept for compatibility)
  - routing_contrib_retain.npy     [n_layers, n_experts]
  - routing_entropy_forget.npy     [n_layers]             mean gate entropy per layer
  - routing_entropy_retain.npy     [n_layers]
  - routing_stats.json             per-layer top-k experts + overlap + entropy metrics
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from src.dataset_utils import split_raw_dataset_for_forget
from src.model_utils.model_loader import load_model


# ── helpers ─────────────────────────────────────────────────────────────────

def _collect_routing_stats(
    model_base,
    prompts: list[str],
    top_k: int,
    num_layers: int,
    num_experts: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-pass each prompt and accumulate per-layer routing statistics.

    Returns (counts, weight_sums, entropy_sums, token_counts):
      - counts        [num_layers, num_experts]  how many times each expert was selected (top-k)
      - weight_sums   [num_layers, num_experts]  sum of full-softmax probability for every expert across all tokens
      - entropy_sums  [num_layers]               sum of per-token gate entropy
      - token_counts  [num_layers]               number of tokens seen

    All raw (not normalised).
    """
    counts        = np.zeros((num_layers, num_experts), dtype=np.float32)
    weight_sums   = np.zeros((num_layers, num_experts), dtype=np.float32)
    entropy_sums  = np.zeros(num_layers, dtype=np.float64)
    token_counts  = np.zeros(num_layers, dtype=np.int64)
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Both OLMoE and Qwen2MoE gates are nn.Linear — output is a plain logit tensor
            router_logits = output.detach().float() if isinstance(output, torch.Tensor) else output[0].detach().float()

            # Full softmax over all experts — matches what the model actually uses internally
            # (both OLMoE and Qwen1.5-MoE have norm_topk_prob=False, so no renormalization)
            probs = torch.softmax(router_logits, dim=-1)  # [seq_len, num_experts]

            # Selection counts: which experts were in the top-k
            top_k_index = torch.topk(probs, k=top_k, dim=-1).indices
            np.add.at(counts[layer_idx], top_k_index.reshape(-1).cpu().numpy(), 1)

            # Weight sums: accumulate full softmax probability for every expert, every token
            weight_sums[layer_idx] += probs.sum(dim=0).cpu().numpy()

            # Shannon entropy of the full gate distribution
            H = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
            entropy_sums[layer_idx] += float(H.sum().cpu())
            token_counts[layer_idx] += router_logits.shape[0]
        return hook_fn

    for layer_idx in range(num_layers):
        router = model_base._get_router(layer_idx)
        hooks.append(router.register_forward_hook(make_hook(layer_idx)))

    model_base._eval()
    input_device = next(model_base.model.parameters()).device

    try:
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="forward passes", leave=False):
                enc = model_base.tokenize_instructions_fn(instructions=[prompt]).to(input_device)
                model_base._forward(enc)
    finally:
        for h in hooks:
            h.remove()

    return counts, weight_sums, entropy_sums, token_counts


def _overlap_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── main ────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="config", config_name="routing_analysis")
def analyze_routing(cfg: DictConfig):
    cfg_device = cfg.get("device", "cpu")
    if cfg_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg_device)
    print(f"device: {device}")

    model_base = load_model(cfg.model_family, cfg.model_path, device)

    num_layers  = len(model_base.model.model.layers)
    num_experts = model_base._get_num_experts()
    top_k       = model_base.model.config.num_experts_per_tok
    print(f"layers={num_layers}  experts={num_experts}  top_k={top_k}")

    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    forget_prompts, retain_prompts = split_raw_dataset_for_forget(
        cfg,
        data_path,
        model_base,
        forget_edge=cfg.forget_edge,
        instructions_only=True,
        torch_reformat=False,
    )

    if cfg.get("max_samples"):
        forget_prompts = forget_prompts[: cfg.max_samples]
        retain_prompts = retain_prompts[: cfg.max_samples]

    print(f"forget={len(forget_prompts)}  retain={len(retain_prompts)}")

    print("collecting forget-set routing ...")
    counts_forget, weight_sums_forget, entropy_sums_forget, token_counts_forget = _collect_routing_stats(
        model_base, forget_prompts, top_k, num_layers, num_experts, device
    )
    print("collecting retain-set routing ...")
    counts_retain, weight_sums_retain, entropy_sums_retain, token_counts_retain = _collect_routing_stats(
        model_base, retain_prompts, top_k, num_layers, num_experts, device
    )

    # selection frequency: how often each expert is chosen (normalised per token)
    freq_forget = counts_forget / (counts_forget.sum(axis=1, keepdims=True) + 1e-9)
    freq_retain = counts_retain / (counts_retain.sum(axis=1, keepdims=True) + 1e-9)

    # average softmax probability per expert per token (over all experts, all tokens)
    num_tokens_forget = token_counts_forget[:, np.newaxis] + 1e-9  # [num_layers, 1]
    num_tokens_retain = token_counts_retain[:, np.newaxis] + 1e-9
    avg_weight_forget = weight_sums_forget / num_tokens_forget
    avg_weight_retain = weight_sums_retain / num_tokens_retain

    # contribution = avg softmax probability per token (same as avg_weight here,
    # since weight_sums already covers all experts unconditionally)
    contrib_forget = avg_weight_forget
    contrib_retain = avg_weight_retain

    # mean gate entropy per layer: H averaged over all tokens
    mean_entropy_forget = entropy_sums_forget / (token_counts_forget + 1e-9)  # [num_layers]
    mean_entropy_retain = entropy_sums_retain / (token_counts_retain + 1e-9)

    # per-layer stats
    stats = {"model": cfg.model_family, "data": cfg.data_name, "layers": []}
    for i in range(num_layers):
        top5_forget = np.argsort(freq_forget[i])[::-1][:5].tolist()
        top5_retain = np.argsort(freq_retain[i])[::-1][:5].tolist()
        overlap_freq   = _overlap_cosine(freq_forget[i], freq_retain[i])
        overlap_contrib = _overlap_cosine(contrib_forget[i], contrib_retain[i])

        # specificity by frequency
        specificity_freq = freq_forget[i] / (freq_forget[i] + freq_retain[i] + 1e-9)
        top5_specific_freq = np.argsort(specificity_freq)[::-1][:5].tolist()

        # specificity by contribution (freq * weight combined)
        specificity_contrib = contrib_forget[i] / (contrib_forget[i] + contrib_retain[i] + 1e-9)
        top5_specific_contrib = np.argsort(specificity_contrib)[::-1][:5].tolist()

        stats["layers"].append({
            "layer": i,
            "top5_forget":                  top5_forget,
            "top5_retain":                  top5_retain,
            "top5_forget_specific":         top5_specific_freq,
            "top5_forget_specific_contrib": top5_specific_contrib,
            "cosine_overlap":               round(overlap_freq, 4),
            "cosine_overlap_contrib":       round(overlap_contrib, 4),
            "entropy_forget":               round(float(mean_entropy_forget[i]), 4),
            "entropy_retain":               round(float(mean_entropy_retain[i]), 4),
            "entropy_diff":                 round(float(mean_entropy_forget[i] - mean_entropy_retain[i]), 4),
        })

    save_dir = Path(cfg.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "routing_freq_forget.npy",     freq_forget)
    np.save(save_dir / "routing_freq_retain.npy",     freq_retain)
    np.save(save_dir / "routing_weight_forget.npy",   avg_weight_forget)
    np.save(save_dir / "routing_weight_retain.npy",   avg_weight_retain)
    np.save(save_dir / "routing_contrib_forget.npy",  contrib_forget)
    np.save(save_dir / "routing_contrib_retain.npy",  contrib_retain)
    np.save(save_dir / "routing_entropy_forget.npy",  mean_entropy_forget)
    np.save(save_dir / "routing_entropy_retain.npy",  mean_entropy_retain)
    with open(save_dir / "routing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {save_dir}")
    print("\nPer-layer routing stats (forget vs retain):")
    for entry in stats["layers"]:
        print(f"  layer {entry['layer']:2d}"
              f"  overlap_freq={entry['cosine_overlap']:.4f}"
              f"  overlap_contrib={entry['cosine_overlap_contrib']:.4f}"
              f"  H_forget={entry['entropy_forget']:.3f}"
              f"  H_retain={entry['entropy_retain']:.3f}"
              f"  ΔH={entry['entropy_diff']:+.3f}"
              f"  top-specific={entry['top5_forget_specific_contrib']}")


if __name__ == "__main__":
    analyze_routing()
