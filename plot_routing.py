# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Routing analysis visualizations for MoE models.

Generates three figures per dataset:
  1. 4x4 grid of per-layer expert frequency bar charts (forget vs retain)
  2. Per-layer Jaccard overlap between forget and retain top-k expert sets
  3. Per-layer routing entropy line plot (if entropy files exist)

Usage:
  python plot_routing.py --dataset wmdp_bio --top_k 8
  python plot_routing.py --dataset pistol_sample1 --top_k 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

TOP_K = 8
NUM_EXPERTS = 64
UNIFORM_BASELINE = TOP_K / NUM_EXPERTS  # 0.125 = 12.5%


def load_freq(routing_dir: Path, split: str) -> np.ndarray:
    """Load freq array and convert to per-token fraction (multiply by TOP_K)."""
    freq = np.load(routing_dir / f"routing_freq_{split}.npy")  # [L, E], sums to 1
    return freq * TOP_K  # now each row sums to TOP_K; uniform = TOP_K/NUM_EXPERTS


def jaccard_top_k(freq_f: np.ndarray, freq_r: np.ndarray, k: int) -> np.ndarray:
    """Jaccard similarity of top-k expert sets per layer."""
    num_layers = freq_f.shape[0]
    scores = np.zeros(num_layers)
    for i in range(num_layers):
        top_f = set(np.argsort(freq_f[i])[::-1][:k])
        top_r = set(np.argsort(freq_r[i])[::-1][:k])
        scores[i] = len(top_f & top_r) / len(top_f | top_r)
    return scores


def plot_frequency_grid(freq_f: np.ndarray, freq_r: np.ndarray,
                        dataset: str, save_path: Path):
    """4x4 grid of per-layer expert frequency bar charts."""
    num_layers = freq_f.shape[0]
    num_experts = freq_f.shape[1]
    expert_idx = np.arange(num_experts)

    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 14), sharey=False)
    fig.suptitle(f"Expert activation frequency per layer — {dataset}", fontsize=14, y=1.01)

    bar_width = 0.45

    for i, ax in enumerate(axes.flat):
        if i >= num_layers:
            ax.set_visible(False)
            continue

        ax.bar(expert_idx - bar_width / 2, freq_f[i], width=bar_width,
               color="#4477AA", alpha=0.85, label="forget" if i == 0 else "")
        ax.bar(expert_idx + bar_width / 2, freq_r[i], width=bar_width,
               color="#EE6677", alpha=0.85, label="retain" if i == 0 else "")

        ax.axhline(UNIFORM_BASELINE, color="black", linewidth=0.8,
                   linestyle="--", label="uniform" if i == 0 else "")

        ax.set_title(f"Layer {i}", fontsize=9)
        ax.set_xlim(-1, num_experts)
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.tick_params(labelsize=7)

    # shared legend
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=10, framealpha=0.9)

    fig.text(0.5, -0.01, "Expert index", ha="center", fontsize=11)
    fig.text(-0.01, 0.5, "Fraction of tokens routed to expert",
             va="center", rotation="vertical", fontsize=11)

    plt.tight_layout()
    out = save_path / f"fig_freq_grid_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_jaccard(freq_f: np.ndarray, freq_r: np.ndarray,
                 dataset: str, save_path: Path, top_k: int = TOP_K):
    """Per-layer Jaccard overlap between forget and retain top-k expert sets."""
    layers = np.arange(freq_f.shape[0])

    jaccard_topk  = jaccard_top_k(freq_f, freq_r, top_k)
    jaccard_top16 = jaccard_top_k(freq_f, freq_r, 16)
    jaccard_top32 = jaccard_top_k(freq_f, freq_r, 32)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, jaccard_topk,  marker="o", label=f"top-{top_k}",  color="#4477AA")
    ax.plot(layers, jaccard_top16, marker="s", label="top-16", color="#EE6677")
    ax.plot(layers, jaccard_top32, marker="^", label="top-32", color="#228833")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Jaccard similarity (forget ∩ retain / forget ∪ retain)")
    ax.set_title(f"Forget vs retain routing overlap — {dataset}")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle=":")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = save_path / f"fig_jaccard_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_entropy(routing_dir: Path, dataset: str, save_path: Path):
    """Per-layer mean gate entropy (forget vs retain)."""
    ef = routing_dir / "routing_entropy_forget.npy"
    er = routing_dir / "routing_entropy_retain.npy"
    if not ef.exists():
        print(f"No entropy files found in {routing_dir} — re-run analyze_routing.py first.")
        return

    entropy_f = np.load(ef)
    entropy_r = np.load(er)
    layers = np.arange(len(entropy_f))
    max_entropy = np.log(NUM_EXPERTS)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, entropy_f, marker="o", label="forget",  color="#4477AA")
    ax.plot(layers, entropy_r, marker="s", label="retain",  color="#EE6677")
    ax.axhline(max_entropy, color="grey", linewidth=0.8, linestyle="--",
               label=f"max entropy (log {NUM_EXPERTS} = {max_entropy:.2f})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean gate entropy (nats)")
    ax.set_title(f"Routing entropy per layer — {dataset}")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = save_path / f"fig_entropy_{dataset}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="wmdp_bio")
    parser.add_argument("--model",   default="olmoe-1b-7b-instruct")
    parser.add_argument("--top_k",   type=int, default=TOP_K)
    parser.add_argument("--base_dir", default="run_results/routing")
    args = parser.parse_args()

    routing_dir = Path(args.base_dir) / args.model / args.dataset
    save_path   = routing_dir
    save_path.mkdir(parents=True, exist_ok=True)

    freq_f = load_freq(routing_dir, "forget")
    freq_r = load_freq(routing_dir, "retain")

    plot_frequency_grid(freq_f, freq_r, args.dataset, save_path)
    plot_jaccard(freq_f, freq_r, args.dataset, save_path, top_k=args.top_k)
    plot_entropy(routing_dir, args.dataset, save_path)


if __name__ == "__main__":
    main()
