# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Causal expert ablation: measure how much forget-set knowledge drops when
progressively silencing the most forget-specific experts.

Protocol:
  For each layer L and ablation size k in {1, 2, 4, 8}:
    - zero the output of the top-k most forget-specific experts in layer L
    - run inference on the forget set
    - record first-token MCQ accuracy (A/B/C/D) or ROUGE-1 if not MCQ

  An expert is ranked by its specificity score:
      contrib_forget[layer, expert] / (contrib_forget + contrib_retain + 1e-9)
  This requires routing_contrib_forget.npy and routing_contrib_retain.npy from
  analyze_routing.py.

Outputs (per dataset):
  - ablation_results.json   {layer: {k: accuracy}} table
  - ablation_curve.npy      [n_layers, n_ablation_sizes] accuracy matrix

Usage:
  python eval_causal_ablation.py \\
    data_name=pistol_sample1 \\
    routing_dir=run_results/routing/olmoe-1b-7b-instruct/pistol_sample1 \\
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

# ablation sizes (number of experts silenced per layer)
ABLATION_K = [0, 1, 2, 4, 8]


def _run_mcq_accuracy(
    model_base,
    prompts: list[str],
    choices_list: list[list[str]],  # list of [A, B, C, D] options per prompt
    answers: list[str],             # ground-truth letter per prompt
    device: torch.device,
) -> float:
    """First-token probability MCQ accuracy."""
    tokenizer = model_base.tokenizer
    model_base._eval()

    option_ids = {
        letter: tokenizer.encode(letter, add_special_tokens=False)[0]
        for letter in ["A", "B", "C", "D"]
    }

    correct = 0
    with torch.no_grad():
        for prompt, answer in zip(prompts, answers):
            formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
            enc = tokenizer(formatted, return_tensors="pt").to(device)
            out = model_base._forward(enc)
            logits = out.logits[0, -1]  # last token logits
            scores = {k: logits[v].item() for k, v in option_ids.items()}
            pred = max(scores, key=scores.get)
            if pred == answer:
                correct += 1

    return correct / len(prompts) if prompts else 0.0


def _run_generation_rouge(
    model_base,
    prompts: list[str],
    references: list[str],
    device: torch.device,
    max_new_tokens: int = 64,
) -> float:
    """ROUGE-1 F1 for open-ended generation."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
    except ImportError:
        raise ImportError("pip install rouge-score for non-MCQ evaluation")

    tokenizer = model_base.tokenizer
    model_base._eval()
    scores = []

    with torch.no_grad():
        for prompt, ref in tqdm(zip(prompts, references), desc="generating", leave=False):
            formatted = f"<|user|>\n{prompt}\n<|assistant|>\n"
            enc = tokenizer(formatted, return_tensors="pt").to(device)
            gen = model_base.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(gen[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            scores.append(scorer.score(ref, text)["rouge1"].fmeasure)

    return float(np.mean(scores)) if scores else 0.0


def _make_zero_hook(layer_idx: int, expert_indices: list[int], model_base):
    """
    Returns a list of hooks that zero the output of specified experts in a layer.
    Hooks are registered on individual expert modules (e.g. the FFN of each expert).
    """
    hooks = []
    experts = model_base._get_layer_experts(layer_idx)
    for eidx in expert_indices:
        def _zero_hook(module, input, output, _eidx=eidx):
            return torch.zeros_like(output)
        hooks.append(experts[eidx].register_forward_hook(_zero_hook))
    return hooks


@hydra.main(version_base=None, config_path="config", config_name="routing_analysis")
def eval_causal_ablation(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    routing_dir = Path(cfg.get("routing_dir", cfg.save_path))
    contrib_forget = np.load(routing_dir / "routing_contrib_forget.npy")  # [L, E]
    contrib_retain = np.load(routing_dir / "routing_contrib_retain.npy")  # [L, E]
    specificity = contrib_forget / (contrib_forget + contrib_retain + 1e-9)
    # rank experts per layer by forget-specificity (descending)
    ranked_experts = np.argsort(specificity, axis=1)[:, ::-1]  # [L, E] most-specific first

    model_base = load_model(cfg.model_family, cfg.model_path, device)
    num_layers = len(model_base.model.model.layers)

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

    # load ground-truth answers from dataset
    with open(data_path) as f:
        raw_data = json.load(f)

    # detect MCQ vs open-ended
    use_mcq = False
    answers = []
    references = []
    forget_edge = cfg.forget_edge if isinstance(cfg.forget_edge, list) else [cfg.forget_edge]

    forget_items = [
        item for item in raw_data
        if any(e in item.get("edge", "") for e in forget_edge)
    ]
    if cfg.get("max_samples"):
        forget_items = forget_items[: cfg.max_samples]

    if forget_items and "answer" in forget_items[0]:
        use_mcq = forget_items[0]["answer"] in ["A", "B", "C", "D"]

    if use_mcq:
        answers = [item["answer"] for item in forget_items]
        choices_list = [item.get("choices", []) for item in forget_items]
    else:
        references = [item.get("target", item.get("answer", "")) for item in forget_items]

    print(f"forget prompts={len(forget_prompts)}  mode={'MCQ' if use_mcq else 'ROUGE'}")

    results = {}
    ablation_curve = np.zeros((num_layers, len(ABLATION_K)), dtype=np.float32)

    for li, layer_idx in enumerate(tqdm(range(num_layers), desc="layers")):
        layer_results = {}
        for ki, k in enumerate(tqdm(ABLATION_K, desc=f"layer {layer_idx} ablations", leave=False)):
            # install hooks to zero k most forget-specific experts
            hooks = []
            if k > 0:
                experts_to_zero = ranked_experts[layer_idx, :k].tolist()
                hooks = _make_zero_hook(layer_idx, experts_to_zero, model_base)

            try:
                if use_mcq:
                    acc = _run_mcq_accuracy(
                        model_base, forget_prompts, choices_list, answers, device
                    )
                else:
                    acc = _run_generation_rouge(model_base, forget_prompts, references, device)
            finally:
                for h in hooks:
                    h.remove()

            layer_results[k] = round(acc, 4)
            ablation_curve[li, ki] = acc

        results[layer_idx] = layer_results
        print(f"  layer {layer_idx}: {layer_results}")

    save_dir = Path(cfg.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "ablation_results.json", "w") as f:
        json.dump({"model": cfg.model_family, "data": cfg.data_name,
                   "ablation_k": ABLATION_K, "metric": "mcq_acc" if use_mcq else "rouge1",
                   "layers": results}, f, indent=2)
    np.save(save_dir / "ablation_curve.npy", ablation_curve)

    print(f"\nSaved to {save_dir}")

    # find the layer where ablation hurts the most
    if ABLATION_K[0] == 0:
        baseline = ablation_curve[:, 0]
        max_drop_per_layer = baseline - ablation_curve[:, -1]
        best_layer = int(np.argmax(max_drop_per_layer))
        print(f"Layer with biggest knowledge drop at k={ABLATION_K[-1]}: "
              f"{best_layer}  (Δacc={max_drop_per_layer[best_layer]:.3f})")


if __name__ == "__main__":
    eval_causal_ablation()
