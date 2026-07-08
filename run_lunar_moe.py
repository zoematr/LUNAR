# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
Run LUNAR unlearning for Mixture-of-Experts models.
Uses MoE-specific activation collection and EstimatedNet training.
"""

from __future__ import annotations

import json
import logging
import os
from itertools import chain
from pathlib import Path

import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset_utils import (
    load_dataset_to_get_direction,
    prepare_estimated_net_list_moe,
    prepare_trainset_moe,
    split_raw_dataset_for_forget,
)
from src.estimated_net_utils import (
    ActivationDataset_multiple_layers,
    train_multiple_layers,
)
from src.eval_util import custom_evaluate
from src.generate_directions import generate_candidate_directions
from src.model_utils.model_loader import load_model


@hydra.main(version_base=None, config_path="config", config_name="forget_moe")
def run_forget_moe(cfg):
    print(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    positions = cfg.positions
    coeff_list = cfg.coeff_list

    # -----------------------------
    # Load model
    # -----------------------------
    print(f'loading model from {cfg.model_family} at {cfg.model_path}')
    model_base = load_model(cfg.model_family, cfg.model_path, device)
    data_path = os.path.join("dataset/unlearning", f"{cfg.data_name}.json")

    # -----------------------------
    # Generate refusal directions (same as dense — uses block-level activations)
    # -----------------------------
    harmful_train, forget_train = load_dataset_to_get_direction(
        cfg,
        data_path,
        instructions_only=True,
        use_harmful=cfg.use_harmful,
        use_unverified=cfg.use_unverified,
    )
    candidate_directions = generate_candidate_directions(
        cfg, model_base, harmful_train, forget_train
    )

    # -----------------------------
    # Prepare forget/retain split (needed for the layer sweep AND training)
    # -----------------------------
    forget_dataset, retain_dataset = split_raw_dataset_for_forget(
        cfg,
        data_path,
        model_base,
        forget_edge=cfg.forget_edge,
        instructions_only=True,
        torch_reformat=False,
    )
    print(f"forget_dataset: {len(forget_dataset)}")
    print(f"retain_dataset: {len(retain_dataset)}")

    # -----------------------------
    # Procedure 2 (LUNAR §3.2 / Alg. 1): layer selection.
    # If cfg.layer_sweep: apply r_UV as an activation-addition at each candidate
    # layer (no training), generate on the forget set, and pick argmax(s1 - s2).
    # Else: use the hardcoded cfg.layer_modified (original behavior).
    # -----------------------------
    if cfg.get("layer_sweep", False):
        from src.layer_sweep import s1_s2_layer_sweep, load_desired_responses

        num_layers = len(model_base.model_block_modules)
        if cfg.get("sweep_layers", None):
            candidate_layers = [int(l) for l in cfg.sweep_layers]
        else:
            candidate_layers = list(range(0, num_layers - 1, cfg.get("sweep_stride", 4)))
        best_layer, _ = s1_s2_layer_sweep(
            cfg,
            model_base,
            candidate_directions,
            forget_dataset,
            candidate_layers=candidate_layers,
            coeff=float(coeff_list[0]),
            desired_responses=load_desired_responses(cfg.get("desired_responses_path", None)),
            embed_model_name=cfg.get("embed_model_name", "sentence-transformers/all-MiniLM-L6-v2"),
            n_forget=cfg.get("sweep_n_forget", 64),
            max_new_tokens=cfg.max_new_tokens,
            save_path=f"{cfg.save_path}/layer_sweep.json",
        )
        layer_idx_list = [best_layer]
    else:
        layer_idx_list = list(cfg.layer_modified)
    print(f"target layer(s): {layer_idx_list}")

    # direction (r_UV) for the selected layer(s)
    direction = [candidate_directions[positions, l + 1, :] for l in layer_idx_list]

    (
        forget_input_list,
        forget_target_list,
        remain_input_list,
        remain_target_list,
        _,
    ) = prepare_trainset_moe(
        layer_idx_list,
        model_base,
        forget_dataset,
        retain_dataset,
        direction,
        coeff_list,
        device,
    )

    # -----------------------------
    # Initialize one shared EstimatedNet per layer (averaged expert weights)
    # -----------------------------
    estimated_net_list = prepare_estimated_net_list_moe(
        device=device,
        layer_idx_list=layer_idx_list,
        model_base=model_base,
        init_model_list=None,
    )

    train_dataset_forget = ActivationDataset_multiple_layers(
        forget_input_list, forget_target_list
    )
    train_dataset_remain = ActivationDataset_multiple_layers(
        remain_input_list, remain_target_list
    )
    combined_dataset = ConcatDataset([train_dataset_forget, train_dataset_remain])
    train_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

    # -----------------------------
    # Training
    # -----------------------------
    optimizer = optim.AdamW(
        chain(*[model.parameters() for model in estimated_net_list]), lr=cfg.lr
    )
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    updated_estimated_net_list = train_multiple_layers(
        estimated_net_list,
        train_loader,
        optimizer,
        scheduler,
        device=device,
        num_epochs=cfg.num_epochs,
    )

    # -----------------------------
    # Apply learned weight to ALL experts in each target layer
    # -----------------------------
    for i, layer_idx in enumerate(layer_idx_list):
        experts = model_base._get_layer_experts(layer_idx)
        learned_weight = updated_estimated_net_list[i].down_proj.weight.data
        for expert in experts:
            model_base._get_expert_down_proj(expert).weight.data = learned_weight.clone()

    if cfg.save_unlearned_model:
        if not os.path.exists(os.path.dirname(cfg.save_unlearned_model_path)):
            os.makedirs(os.path.dirname(cfg.save_unlearned_model_path))
        print(f"Saving unlearned model to {cfg.save_unlearned_model_path}")
        model_base._save_pretrained(cfg.save_unlearned_model_path)

    # -----------------------------
    # Evaluation
    # -----------------------------
    eval_logs_forget_edge = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="forget_edge",
        output_es_score=cfg.compute_es_score,
    )
    eval_logs_retained_edge = custom_evaluate(
        cfg=cfg,
        data_path=data_path,
        tokenizer=model_base.tokenizer,
        model=model_base,
        eval_target="retained_edge",
        output_es_score=False,
    )
    if cfg.if_eval_factual:
        eval_logs_factual_data = custom_evaluate(
            cfg=cfg,
            data_path=cfg.factual_data_path,
            tokenizer=model_base.tokenizer,
            model=model_base,
            eval_target="factual_data",
            output_es_score=False,
        )
        eval_logs = {
            "forget": eval_logs_forget_edge,
            "retained_edge": eval_logs_retained_edge,
            "factual_data": eval_logs_factual_data,
        }
    else:
        eval_logs = {
            "forget": eval_logs_forget_edge,
            "retained_edge": eval_logs_retained_edge,
        }

    save_str = "_".join([str(layer_idx) for layer_idx in layer_idx_list])
    save_file = f"{cfg.save_path}/forget_{save_str}.json"
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Saving completions to {save_file}")
    with open(save_file, "w") as f:
        json.dump(eval_logs, f, indent=4)


if __name__ == "__main__":
    run_forget_moe()
