# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from src.utils.utils import get_orthogonalized_matrix
from src.model_utils.moe_model_base import MoEModelBase

# OLMoE-1B-7B-0924-Instruct uses the same tokenizer and chat format as OLMo
OLMOE_CHAT_TEMPLATE = """<|user|>
{instruction}
<|assistant|>
"""

OLMOE_REFUSAL_TOKS = [40, 2170]  # ['I', 'As'] — verify with tokenizer if needed


class _FusedExpertProxy:
    """Thin proxy giving a uniform .down_proj.weight view over one expert inside
    the fused OlmoeExperts module (newer transformers), where weights are stored
    as stacked 3D tensors rather than individual nn.Linear modules."""
    _is_fused = True

    class _DownProjProxy:
        def __init__(self, weight):
            self.weight = weight

    def __init__(self, experts_module, idx: int):
        self._experts_module = experts_module
        self._idx = idx
        self.down_proj = self._DownProjProxy(experts_module.down_proj[idx])


def format_instruction_olmoe_chat(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True,
):
    formatted_instruction = OLMOE_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction


def tokenize_instructions_olmoe_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace=True,
):
    if outputs is not None:
        prompts = [
            format_instruction_olmoe_chat(
                instruction=instruction,
                output=output,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_olmoe_chat(
                instruction=instruction,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


def orthogonalize_olmoe_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )
    # OLMoE: orthogonalize o_proj in attention and down_proj of each expert
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        for expert in block.mlp.experts:
            expert.down_proj.weight.data = get_orthogonalized_matrix(
                expert.down_proj.weight.data.T, direction
            ).T


def act_add_olmoe_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    # Apply bias to all experts in the layer
    for expert in model.model.layers[layer - 1].mlp.experts:
        dtype = expert.down_proj.weight.dtype
        device = expert.down_proj.weight.device
        bias = (coeff * direction).to(dtype=dtype, device=device)
        expert.down_proj.bias = torch.nn.Parameter(bias)


class OLMoEModel(MoEModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_olmoe_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            OLMOE_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
        )

    def _get_refusal_toks(self):
        return OLMOE_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block_module.self_attn for block_module in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        # Returns the MoE block (OlmoeSparseMoeBlock), not individual experts
        return torch.nn.ModuleList(
            [block_module.mlp for block_module in self.model_block_modules]
        )

    # --- MoEModelBase methods ---

    def _get_layer_experts(self, layer_idx: int):
        experts = self.model.model.layers[layer_idx].mlp.experts
        if isinstance(experts, torch.nn.ModuleList):
            return list(experts)
        # Newer transformers: OlmoeExperts stores weights as stacked 3D tensors.
        # Return proxy objects so callers get a uniform .down_proj.weight interface.
        return [_FusedExpertProxy(experts, i) for i in range(experts.num_experts)]

    def _get_expert_down_proj(self, expert):
        return expert.down_proj

    def _get_num_experts(self) -> int:
        return self.model.config.num_experts

    def _get_router(self, layer_idx: int):
        return self.model.model.layers[layer_idx].mlp.gate

    # --- Standard ModelBase methods ---

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_olmoe_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_olmoe_weights, direction=direction, coeff=coeff, layer=layer
        )

    def _to(self, device=None, dtype=None):
        # device_map="auto" already handles placement; moving manually conflicts with accelerate hooks
        return self

    def _eval(self):
        self.model.eval()
        return self

    def _forward(self, batch):
        return self.model(**batch)

    def _device(self):
        return next(self.model.parameters()).device.type

    def _generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        max_new_tokens,
        do_sample,
        num_beams,
        num_return_sequences,
        use_cache,
        pad_token_id,
        output_scores,
        return_dict_in_generate,
    ):
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            output_scores=output_scores,
            return_dict_in_generate=return_dict_in_generate,
        )

    def _save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)
