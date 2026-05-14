# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import contextlib
import functools
from tqdm import tqdm
from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor


@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs,
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()


def get_activations_fwd_hook(cache):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            output = output[0]
        activation = output.clone().detach()
        cache.append(activation)

    return hook_fn


def get_activations_pre_hook(cache):
    """Hook function to capture input activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input):
        if isinstance(input, tuple):
            input = input[0]
        activation = input.clone().detach()
        cache.append(activation)

    return hook_fn


def get_post_block_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    fwd_hooks = [
        (model.model.layers[layer_idx], get_activations_fwd_hook(cache=activations))
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=[], module_forward_hooks=fwd_hooks):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )
    return activations


def get_pre_down_proj_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    pre_hooks = [
        (
            model.model.layers[layer_idx].mlp.down_proj,
            get_activations_pre_hook(cache=activations),
        )
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])

        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations


def get_pre_post_attention_layernorm_activation(
    model, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    torch.cuda.empty_cache()
    instructions = input_data

    activations = []
    pre_hooks = [
        (
            model.model.layers[layer_idx].post_attention_layernorm,
            get_activations_pre_hook(cache=activations),
        )
    ]

    for i in tqdm(range(0, len(instructions), batch_size)):
        inputs = tokenize_instructions_fn(instructions=instructions[i : i + batch_size])
        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return activations


def get_activations(
    model_base,
    layer_idx,
    forget_dataset,
    retain_dataset,
    batch_size_forget=1,
    batch_size_remain=1,
):

    post_block_activation_forget = get_post_block_activation(
        model=model_base.model,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )

    post_block_activation_remain = get_post_block_activation(
        model=model_base.model,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )

    # get pre post attention layernorm activations
    pre_post_attention_layernorm_activation_forget = (
        get_pre_post_attention_layernorm_activation(
            model=model_base.model,
            input_data=forget_dataset,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size_forget,
        )
    )

    pre_post_attention_layernorm_activation_remain = (
        get_pre_post_attention_layernorm_activation(
            model=model_base.model,
            input_data=retain_dataset,
            tokenize_instructions_fn=model_base.tokenize_instructions_fn,
            layer_idx=layer_idx,
            batch_size=batch_size_remain,
        )
    )

    pre_down_proj_activation_forget = get_pre_down_proj_activation(
        model=model_base.model,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )

    pre_down_proj_activation_remain = get_pre_down_proj_activation(
        model=model_base.model,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )

    return (
        post_block_activation_forget,
        post_block_activation_remain,
        pre_post_attention_layernorm_activation_forget,
        pre_post_attention_layernorm_activation_remain,
        pre_down_proj_activation_forget,
        pre_down_proj_activation_remain,
    )


def get_pertubred_activations_fwd_hook(vector, coeff):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        nonlocal vector
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output  # .clone().detach()

        vector = vector.to(activation)
        activation += coeff * vector

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_skip_fwd_hook():
    """Hook function to to make output equals input so that can skip this layer."""

    def hook_fn(module, input, output):
        # input is a tuple with len 1, but output is a tuple with len 2
        # return input[0]
        return (input[0], *output[1:])

    return hook_fn


def get_reverse_purtubred_activations_fwd_hook(vector, coeff):
    """Hook function to capture output activations of the layer.
    It stores activations for each token in the sequence."""

    def hook_fn(module, input, output):
        nonlocal vector
        if isinstance(output, tuple):
            activation = output[0]
        else:
            activation = output  # .clone().detach()

        vector = vector.to(activation)
        activation -= coeff * vector

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def _get_pre_down_proj_activation_moe_fused(
    model_base, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    """
    Pre-down_proj capture for the fused OlmoeExperts used in newer transformers.

    OlmoeExperts stores all expert weights as stacked 3D tensors, so there are no
    individual nn.Module objects to attach forward hooks to. Instead, we temporarily
    replace the module's forward method with one that captures the intermediate
    activation (act(gate) * up) before the down_proj linear for each active expert.
    """
    import torch.nn.functional as F

    experts_module = model_base.model.model.layers[layer_idx].mlp.experts
    num_experts = experts_module.num_experts
    activations = []

    for i in tqdm(range(0, len(input_data), batch_size)):
        instructions = input_data[i: i + batch_size]
        inputs = tokenize_instructions_fn(instructions=instructions)

        per_expert_captures = [[] for _ in range(num_experts)]

        def capturing_forward(hidden_states, top_k_index, top_k_weights):
            final_hidden_states = torch.zeros_like(hidden_states)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = (expert_mask.sum(dim=(-1, -2)) > 0).nonzero()

            for expert_idx_t in expert_hit:
                expert_idx = expert_idx_t[0]
                if expert_idx >= num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = F.linear(
                    current_state, experts_module.gate_up_proj[expert_idx]
                ).chunk(2, dim=-1)
                pre_down = experts_module.act_fn(gate) * up
                per_expert_captures[expert_idx.item()].append(pre_down.detach())
                out = F.linear(pre_down, experts_module.down_proj[expert_idx])
                out = out * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))

            return final_hidden_states

        original_forward = experts_module.forward
        experts_module.forward = capturing_forward
        try:
            with torch.no_grad():
                model_base.model(
                    input_ids=inputs.input_ids.to(model_base.model.device),
                    attention_mask=inputs.attention_mask.to(model_base.model.device),
                )
        finally:
            experts_module.forward = original_forward

        all_expert_acts = [act for caps in per_expert_captures for act in caps]
        if all_expert_acts:
            combined = torch.cat(all_expert_acts, dim=0)
            activations.append(combined.unsqueeze(0))

    return activations


def get_pre_down_proj_activation_moe(
    model_base, input_data, tokenize_instructions_fn, layer_idx, batch_size
):
    """
    MoE version of get_pre_down_proj_activation.

    In dense models, every token passes through one down_proj. In MoE models, each
    token is routed to k experts, so each expert's down_proj sees a different subset
    of tokens. This function hooks into every expert's down_proj, collects activations,
    and concatenates them across all experts for each forward pass.

    Result per sample: [1, total_expert_tokens, intermediate_size], where
    total_expert_tokens = sum of tokens processed by each active expert (>= seq_len
    because of top-k routing).
    """
    experts = model_base._get_layer_experts(layer_idx)

    # Newer transformers uses a fused OlmoeExperts with stacked weight tensors
    # instead of individual nn.Module objects — hooks can't attach to tensor slices.
    if getattr(experts[0], '_is_fused', False):
        return _get_pre_down_proj_activation_moe_fused(
            model_base, input_data, tokenize_instructions_fn, layer_idx, batch_size
        )

    activations = []

    for i in tqdm(range(0, len(input_data), batch_size)):
        instructions = input_data[i : i + batch_size]
        inputs = tokenize_instructions_fn(instructions=instructions)

        # One cache list per expert — cleared each forward pass
        expert_caches = [[] for _ in experts]
        pre_hooks = [
            (
                model_base._get_expert_down_proj(expert),
                get_activations_pre_hook(cache=expert_caches[j]),
            )
            for j, expert in enumerate(experts)
        ]

        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=[]):
            model_base.model(
                input_ids=inputs.input_ids.to(model_base.model.device),
                attention_mask=inputs.attention_mask.to(model_base.model.device),
            )

        # Concatenate non-empty expert activations
        # expert_caches[j][0] has shape [n_tokens_for_expert_j, intermediate_size]
        expert_acts = [cache[0] for cache in expert_caches if len(cache) > 0]
        if expert_acts:
            combined = torch.cat(expert_acts, dim=0)  # [total_expert_tokens, intermediate_size]
            activations.append(combined.unsqueeze(0))  # [1, total_expert_tokens, intermediate_size]

    return activations


def get_activations_moe(
    model_base,
    layer_idx,
    forget_dataset,
    retain_dataset,
    batch_size_forget=1,
    batch_size_remain=1,
):
    """
    MoE version of get_activations.
    Block-level hooks (post_block, pre_post_attention_layernorm) are reused unchanged.
    Only the pre_down_proj hook is replaced with the MoE-aware version.
    """
    post_block_activation_forget = get_post_block_activation(
        model=model_base.model,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )
    post_block_activation_remain = get_post_block_activation(
        model=model_base.model,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )
    pre_post_attn_forget = get_pre_post_attention_layernorm_activation(
        model=model_base.model,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )
    pre_post_attn_remain = get_pre_post_attention_layernorm_activation(
        model=model_base.model,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )
    pre_down_proj_forget = get_pre_down_proj_activation_moe(
        model_base=model_base,
        input_data=forget_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_forget,
    )
    pre_down_proj_remain = get_pre_down_proj_activation_moe(
        model_base=model_base,
        input_data=retain_dataset,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        layer_idx=layer_idx,
        batch_size=batch_size_remain,
    )

    return (
        post_block_activation_forget,
        post_block_activation_remain,
        pre_post_attn_forget,
        pre_post_attn_remain,
        pre_down_proj_forget,
        pre_down_proj_remain,
    )


def perturb_post_block_activations_forget(
    post_block_activation_forget, direction, coeff=+2.0
):
    """perturb the output_activations of forget data. but only perturb the last token"""

    ### post_block_activation_forget: list of n_sample tensors [1, seq_length, d_model]
    for i in range(len(post_block_activation_forget)):
        post_block_activation_forget[i] += coeff * direction

    return post_block_activation_forget
