# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from src.utils.utils import get_orthogonalized_matrix
from src.model_utils.moe_model_base import MoEModelBase

# Qwen2-57B-A14B-Instruct uses the same im_start/im_end chat format as Qwen2
QWEN2MOE_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN2MOE_REFUSAL_TOKS = [40, 2121]  # ['I', 'As'] — same Qwen2 tiktoken vocab


def format_instruction_qwen2moe_chat(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True,
):
    formatted = QWEN2MOE_CHAT_TEMPLATE.format(instruction=instruction)
    if not include_trailing_whitespace:
        formatted = formatted.rstrip()
    if output is not None:
        formatted += output
    return formatted


def tokenize_instructions_qwen2moe_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen2moe_chat(
                instruction=instr,
                output=out,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen2moe_chat(
                instruction=instr,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr in instructions
        ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def orthogonalize_qwen2moe_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )
    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        for expert in block.mlp.experts:
            expert.down_proj.weight.data = get_orthogonalized_matrix(
                expert.down_proj.weight.data.T, direction
            ).T
        # shared expert has a larger intermediate dim but same hidden dim — safe to orthogonalize
        block.mlp.shared_expert.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.shared_expert.down_proj.weight.data.T, direction
        ).T


def act_add_qwen2moe_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    for expert in model.model.layers[layer - 1].mlp.experts:
        dtype = expert.down_proj.weight.dtype
        device = expert.down_proj.weight.device
        expert.down_proj.bias = torch.nn.Parameter(
            (coeff * direction).to(dtype=dtype, device=device)
        )
    shared = model.model.layers[layer - 1].mlp.shared_expert
    shared.down_proj.bias = torch.nn.Parameter(
        (coeff * direction).to(dtype=shared.down_proj.weight.dtype, device=shared.down_proj.weight.device)
    )


class Qwen2MoEModel(MoEModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="/tmp/offload_qwen2moe",
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
            tokenize_instructions_qwen2moe_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            QWEN2MOE_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
        )

    def _get_refusal_toks(self):
        return QWEN2MOE_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block.self_attn for block in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block.mlp for block in self.model_block_modules]
        )

    # --- MoEModelBase methods ---

    def _get_layer_experts(self, layer_idx: int):
        # Returns only routed experts — shared_expert has a different intermediate
        # size (num_shared_experts × intermediate_size) so its down_proj shape
        # differs and cannot be swapped with the learned EstimatedNet weight.
        return list(self.model.model.layers[layer_idx].mlp.experts)

    def _get_expert_down_proj(self, expert):
        return expert.down_proj

    def _get_num_experts(self) -> int:
        return self.model.config.num_experts

    def _get_router(self, layer_idx: int):
        return self.model.model.layers[layer_idx].mlp.gate

    # --- Standard ModelBase methods ---

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen2moe_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_qwen2moe_weights, direction=direction, coeff=coeff, layer=layer
        )

    def _to(self, device=None, dtype=None):
        # device_map="auto" handles placement; manual .to() conflicts with accelerate hooks
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
