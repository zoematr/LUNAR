# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Routing analysis visualizations for MoE models.

Generates three figures per dataset:
  1. 4x4 grid of per-layer expert frequency bar charts (forget vs retain)
  2. Per-layer Jaccard overlap between forget and retain top-k expert sets
  3. Per-layer routing entropy line plot (if entropy files exist)

Usage:
  python plot_routing.py --dataset wmdp_bio
  python plot_routing.py --dataset pistol_sample1
  python plot_routing.py --dataset wmdp_bio --top_k 8  # explicit override
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

def load_metadata(routing_dir: Path) -> dict:
    """Read top_k and num_experts saved by analyze_routing.py."""
    stats_path = routing_dir / "routing_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            s = json.load(f)
        return {"top_k": s.get("top_k"), "num_experts": s.get("num_experts")}
    return {}


def load_freq(routing_dir: Path, split: str, top_k: int) -> np.ndarray:
    """Load freq array and convert to per-token fraction (multiply by top_k)."""
    freq = np.load(routing_dir / f"routing_freq_{split}.npy")  # [L, E], sums to 1
    return freq * top_k  # now each row sums to top_k; uniform = top_k/num_experts


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
                        dataset: str, save_path: Path, top_k: int, num_experts: int):
    """4x4 grid of per-layer expert frequency bar charts."""
    num_layers = freq_f.shape[0]
    expert_idx = np.arange(num_experts)
    uniform_baseline = top_k / num_experts

    nrows, ncols = 4, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 14), sharey=False)
    fig.suptitle(f"Expert activation frequency per layer — {dataset}", fontsize=14, y=1.01)

    bar_width = 0.45
    xticks = np.linspace(0, num_experts - 1, 5).astype(int)

    for i, ax in enumerate(axes.flat):
        if i >= num_layers:
            ax.set_visible(False)
            continue

        ax.bar(expert_idx - bar_width / 2, freq_f[i], width=bar_width,
               color="#4477AA", alpha=0.85, label="forget" if i == 0 else "")
        ax.bar(expert_idx + bar_width / 2, freq_r[i], width=bar_width,
               color="#EE6677", alpha=0.85, label="retain" if i == 0 else "")

        ax.axhline(uniform_baseline, color="black", linewidth=0.8,
                   linestyle="--", label="uniform" if i == 0 else "")

        ax.set_title(f"Layer {i}", fontsize=9)
        ax.set_xlim(-1, num_experts)
        ax.set_xticks(xticks)
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
                 dataset: str, save_path: Path, top_k: int = 8):
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


def plot_entropy(routing_dir: Path, dataset: str, save_path: Path, num_experts: int):
    """Per-layer mean gate entropy (forget vs retain)."""
    ef = routing_dir / "routing_entropy_forget.npy"
    er = routing_dir / "routing_entropy_retain.npy"
    if not ef.exists():
        print(f"No entropy files found in {routing_dir} — re-run analyze_routing.py first.")
        return

    entropy_f = np.load(ef)
    entropy_r = np.load(er)
    layers = np.arange(len(entropy_f))
    max_entropy = np.log(num_experts)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, entropy_f, marker="o", label="forget",  color="#4477AA")
    ax.plot(layers, entropy_r, marker="s", label="retain",  color="#EE6677")
    ax.axhline(max_entropy, color="grey", linewidth=0.8, linestyle="--",
               label=f"max entropy (log {num_experts} = {max_entropy:.2f})")

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
    parser.add_argument("--dataset",  default="wmdp_bio")
    parser.add_argument("--model",    default="olmoe-1b-7b-instruct")
    parser.add_argument("--top_k",    type=int, default=None,
                        help="Override top_k from routing_stats.json (default: read from metadata)")
    parser.add_argument("--base_dir", default="run_results/routing")
    args = parser.parse_args()

    routing_dir = Path(args.base_dir) / args.model / args.dataset
    save_path   = routing_dir
    save_path.mkdir(parents=True, exist_ok=True)

    meta = load_metadata(routing_dir)
    top_k       = args.top_k or meta.get("top_k") or 8
    num_experts = meta.get("num_experts") or 64

    if meta.get("top_k") is None:
        print(f"Warning: routing_stats.json missing top_k — using {top_k}. Re-run analyze_routing.py to fix.")

    freq_f = load_freq(routing_dir, "forget", top_k)
    freq_r = load_freq(routing_dir, "retain", top_k)

    plot_frequency_grid(freq_f, freq_r, args.dataset, save_path, top_k=top_k, num_experts=num_experts)
    plot_jaccard(freq_f, freq_r, args.dataset, save_path, top_k=top_k)
    plot_entropy(routing_dir, args.dataset, save_path, num_experts=num_experts)


if __name__ == "__main__":
    main()
