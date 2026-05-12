# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Routing frequency analysis for OLMoE.

For each transformer layer, counts how often each expert is activated on the
forget-set vs. retain-set.  Outputs:
  - routing_freq_forget.npy   [n_layers, n_experts]  (normalised per token)
  - routing_freq_retain.npy   [n_layers, n_experts]
  - routing_stats.json        per-layer top-k experts + overlap metrics
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

def _collect_routing_counts(
    model_base,
    prompts: list[str],
    top_k: int,
    num_layers: int,
    num_experts: int,
    device: torch.device,
) -> np.ndarray:
    """
    Forward-pass each prompt individually and accumulate expert hit counts.

    Returns counts [num_layers, num_experts] as float32 (raw, not normalised).
    """
    counts = np.zeros((num_layers, num_experts), dtype=np.float32)
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # output: (router_logits, top_k_weights, top_k_indices)
            # use the actual routing decisions rather than recomputing from logits
            top_k_index = output[2].detach()  # [seq_len, top_k]
            flat = top_k_index.reshape(-1).cpu().numpy()
            np.add.at(counts[layer_idx], flat, 1)
        return hook_fn

    for layer_idx in range(num_layers):
        router = model_base._get_router(layer_idx)
        hooks.append(router.register_forward_hook(make_hook(layer_idx)))

    model_base._eval()
    tokenizer = model_base.tokenizer

    try:
        with torch.no_grad():
            for prompt in tqdm(prompts, desc="forward passes", leave=False):
                formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
                enc = tokenizer(formatted, return_tensors="pt").to(device)
                model_base._forward(enc)
    finally:
        for h in hooks:
            h.remove()

    return counts


def _overlap_cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── main ────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="config", config_name="routing_analysis")
def analyze_routing(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    counts_forget = _collect_routing_counts(
        model_base, forget_prompts, top_k, num_layers, num_experts, device
    )
    print("collecting retain-set routing ...")
    counts_retain = _collect_routing_counts(
        model_base, retain_prompts, top_k, num_layers, num_experts, device
    )

    # normalise to per-token frequency
    freq_forget = counts_forget / (counts_forget.sum(axis=1, keepdims=True) + 1e-9)
    freq_retain = counts_retain / (counts_retain.sum(axis=1, keepdims=True) + 1e-9)

    # per-layer stats
    stats = {"model": cfg.model_family, "data": cfg.data_name, "layers": []}
    for i in range(num_layers):
        top5_forget = np.argsort(freq_forget[i])[::-1][:5].tolist()
        top5_retain = np.argsort(freq_retain[i])[::-1][:5].tolist()
        overlap     = _overlap_cosine(freq_forget[i], freq_retain[i])

        # specificity: how much each expert skews toward the forget set
        specificity = freq_forget[i] / (freq_forget[i] + freq_retain[i] + 1e-9)
        top5_specific = np.argsort(specificity)[::-1][:5].tolist()

        stats["layers"].append({
            "layer": i,
            "top5_forget":   top5_forget,
            "top5_retain":   top5_retain,
            "top5_forget_specific": top5_specific,
            "cosine_overlap": round(overlap, 4),
        })

    save_dir = Path(cfg.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "routing_freq_forget.npy", freq_forget)
    np.save(save_dir / "routing_freq_retain.npy", freq_retain)
    with open(save_dir / "routing_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {save_dir}")
    print("\nPer-layer cosine overlap (forget vs retain):")
    for entry in stats["layers"]:
        print(f"  layer {entry['layer']:2d}  overlap={entry['cosine_overlap']:.4f}"
              f"  top-forget-specific={entry['top5_forget_specific']}")


if __name__ == "__main__":
    analyze_routing()
