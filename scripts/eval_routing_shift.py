# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Expert selection shift analysis: KL divergence of routing distributions
before vs after unlearning.

Compares routing frequency vectors on the forget set between a base model
and an unlearned checkpoint.  High KL in a layer means unlearning substantially
re-routed forget-set tokens there — evidence of rerouting rather than erasure
(cf. GRIP, arXiv 2601.16905).

Requires a saved unlearned model checkpoint (set save_unlearned_model: true
in forget_moe.yaml before running the unlearning experiment).

Outputs:
  - routing_shift_kl.npy       [n_layers]   KL(base_freq || unlearned_freq) per layer
  - routing_shift_l1.npy       [n_layers]   L1 distance per layer
  - routing_shift_stats.json   per-layer details

Usage:
  python eval_routing_shift.py \\
    model_path=allenai/OLMoE-1B-7B-0924-Instruct \\
    unlearned_model_path=models_lunar/pistol_sample1/olmoe-1b-7b-instruct/... \\
    data_name=pistol_sample1 \\
    save_path=run_results/routing/olmoe-1b-7b-instruct/pistol_sample1
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


def _collect_freq(
    model_base,
    prompts: list[str],
    num_layers: int,
    num_experts: int,
    top_k: int,
    device: torch.device,
) -> np.ndarray:
    """Returns normalised routing frequency [num_layers, num_experts]."""
    counts = np.zeros((num_layers, num_experts), dtype=np.float32)
    hooks = []

    def make_hook(layer_idx: int):
        def hook_fn(module, input, output):
            # Gate modules differ by architecture:
            #   OLMoE: gate returns (logits, weights, indices)
            #   Qwen2MoE: gate (nn.Linear) returns a plain logit tensor [seq, experts]
            if isinstance(output, torch.Tensor):
                _, top_k_index = torch.topk(output.detach(), k=top_k, dim=-1)
            else:
                top_k_index = output[2].detach()
            flat_idx = top_k_index.reshape(-1).cpu().numpy()
            np.add.at(counts[layer_idx], flat_idx, 1)
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

    freq = counts / (counts.sum(axis=1, keepdims=True) + 1e-9)
    return freq


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) with zero-handling."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / (q[mask] + 1e-9))))


def _l1_dist(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(np.abs(p - q)))


@hydra.main(version_base=None, config_path="config", config_name="routing_analysis")
def eval_routing_shift(cfg: DictConfig):
    unlearned_path = cfg.get("unlearned_model_path")
    if not unlearned_path:
        raise ValueError(
            "unlearned_model_path must be set. "
            "Enable save_unlearned_model: true in forget_moe.yaml before unlearning."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Loading base model ...")
    model_base = load_model(cfg.model_family, cfg.model_path, device)
    num_layers  = len(model_base.model.model.layers)
    num_experts = model_base._get_num_experts()
    top_k       = model_base.model.config.num_experts_per_tok
    print(f"layers={num_layers}  experts={num_experts}  top_k={top_k}")

    print("Loading unlearned model ...")
    model_unlearned = load_model(cfg.model_family, unlearned_path, device)

    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")
    forget_prompts, _ = split_raw_dataset_for_forget(
        cfg,
        data_path,
        model_base,
        forget_edge=cfg.forget_edge,
        instructions_only=True,
        torch_reformat=False,
    )
    if cfg.get("max_samples"):
        forget_prompts = forget_prompts[: cfg.max_samples]
    print(f"forget prompts: {len(forget_prompts)}")

    print("Collecting base routing ...")
    freq_base = _collect_freq(model_base, forget_prompts, num_layers, num_experts, top_k, device)

    print("Collecting unlearned routing ...")
    freq_unlearned = _collect_freq(model_unlearned, forget_prompts, num_layers, num_experts, top_k, device)

    kl_per_layer = np.array([_kl_div(freq_base[i], freq_unlearned[i]) for i in range(num_layers)])
    l1_per_layer = np.array([_l1_dist(freq_base[i], freq_unlearned[i]) for i in range(num_layers)])

    stats = {
        "model":           cfg.model_family,
        "data":            cfg.data_name,
        "unlearned_path":  unlearned_path,
        "layers": [
            {
                "layer":  i,
                "kl_base_vs_unlearned": round(float(kl_per_layer[i]), 6),
                "l1_base_vs_unlearned": round(float(l1_per_layer[i]), 6),
                "top5_base":       np.argsort(freq_base[i])[::-1][:5].tolist(),
                "top5_unlearned":  np.argsort(freq_unlearned[i])[::-1][:5].tolist(),
            }
            for i in range(num_layers)
        ],
    }

    save_dir = Path(cfg.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / "routing_shift_kl.npy", kl_per_layer)
    np.save(save_dir / "routing_shift_l1.npy", l1_per_layer)
    with open(save_dir / "routing_shift_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved to {save_dir}")
    print("\nPer-layer routing shift (KL base → unlearned on forget set):")
    for entry in stats["layers"]:
        print(f"  layer {entry['layer']:2d}  KL={entry['kl_base_vs_unlearned']:.4f}"
              f"  L1={entry['l1_base_vs_unlearned']:.4f}"
              f"  top5_base={entry['top5_base']}"
              f"  top5_unlearned={entry['top5_unlearned']}")

    most_shifted = int(np.argmax(kl_per_layer))
    print(f"\nLayer with highest routing shift: {most_shifted} (KL={kl_per_layer[most_shifted]:.4f})")


if __name__ == "__main__":
    eval_routing_shift()
