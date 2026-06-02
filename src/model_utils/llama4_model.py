# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Float

from src.utils.utils import get_orthogonalized_matrix
from src.model_utils.moe_model_base import (
    MoEModelBase,
    resolve_text_model,
    resolve_text_config,
    is_fused_experts,
    fused_experts_as_list,
    orthogonalize_fused_down_proj,
)

# Llama 4 uses a NEW chat format (not the Llama 3 one): the header markers are
# <|header_start|>/<|header_end|> and the turn terminator is <|eot|>.
# <|begin_of_text|> is added automatically by the tokenizer.
LLAMA4_CHAT_TEMPLATE = """<|header_start|>user<|header_end|>

{instruction}<|eot|><|header_start|>assistant<|header_end|>

"""

# Llama 4 uses a different tokenizer than Llama 3 (vocab ~202k). Verified against
# the Llama-4-Scout tokenizer: encode("I", add_special_tokens=False) == [53].
# (The Llama 3 value 40 does NOT apply here: 40 = '<'.)
LLAMA4_REFUSAL_TOKS = [53]  # 'I'


# Llama4's routed experts are NOT a ModuleList of Linear layers.
# `Llama4TextExperts` stores them as fused 3D nn.Parameters:
#     gate_up_proj : [num_experts, hidden, 2*expert_dim]
#     down_proj    : [num_experts, expert_dim(=intermediate), hidden]   (hidden last)
# and the forward is a batched matmul `bmm(x, down_proj)` (no bias term). The
# shared fused-expert adapters in moe_model_base auto-detect this orientation
# (vs Qwen3-MoE's [E, hidden, intermediate]) by matching hidden_size, and expose
# the nn.Linear convention (.weight is [hidden, intermediate]) so LUNAR's expert
# weight-swap (run_lunar_moe.py) and averaging (dataset_utils.py) work unchanged.


def _is_moe_layer(block) -> bool:
    # MoE layers carry a Llama4TextMoe `feed_forward` with `.experts`; dense
    # layers carry a Llama4TextMLP (no `.experts`). Scout is all-MoE, but guard.
    return hasattr(block.feed_forward, "experts")


def format_instruction_llama4_chat(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True,
):
    formatted = LLAMA4_CHAT_TEMPLATE.format(instruction=instruction)
    if not include_trailing_whitespace:
        formatted = formatted.rstrip()
    if output is not None:
        formatted += output
    return formatted


def tokenize_instructions_llama4_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_llama4_chat(
                instruction=instr,
                output=out,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_llama4_chat(
                instruction=instr,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr in instructions
        ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def orthogonalize_llama4_weights(model, direction: Float[Tensor, "d_model"]):
    text_model = resolve_text_model(model)
    hidden_size = resolve_text_config(model).hidden_size
    text_model.embed_tokens.weight.data = get_orthogonalized_matrix(
        text_model.embed_tokens.weight.data, direction
    )
    for block in text_model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        if not _is_moe_layer(block):
            # Dense Llama4TextMLP layer: orthogonalize its single down_proj.
            block.feed_forward.down_proj.weight.data = get_orthogonalized_matrix(
                block.feed_forward.down_proj.weight.data.T, direction
            ).T
            continue
        moe = block.feed_forward
        # Routed experts: fused down_proj, orientation auto-detected.
        orthogonalize_fused_down_proj(moe.experts.down_proj, direction, hidden_size)
        # Shared expert runs on *every* token and writes to the residual stream,
        # so it must be orthogonalized too (cf. qwen2moe_model.py).
        moe.shared_expert.down_proj.weight.data = get_orthogonalized_matrix(
            moe.shared_expert.down_proj.weight.data.T, direction
        ).T


def act_add_llama4_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    # Act-add injects a bias on down_proj. Llama4's routed experts are fused
    # nn.Parameters consumed by a bias-less bmm, so there is no per-expert bias
    # to set and the activation-addition trick cannot be applied to them. Use the
    # EstimatedNet weight-swap path (run_lunar_moe.py) for Llama 4 instead.
    raise NotImplementedError(
        "Activation-addition is not supported for Llama 4's fused experts "
        "(down_proj is a 3D nn.Parameter with a bias-less bmm forward). "
        "Use the MoE weight-swap path (run_lunar_moe.py)."
    )


class Llama4Model(MoEModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="/tmp/offload_llama4",
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
            tokenize_instructions_llama4_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            LLAMA4_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
        )

    def _get_refusal_toks(self):
        return LLAMA4_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return resolve_text_model(self.model).layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block.self_attn for block in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        # feed_forward is the MoE block (or dense MLP) in Llama 4
        return torch.nn.ModuleList(
            [block.feed_forward for block in self.model_block_modules]
        )

    # --- MoEModelBase methods ---

    def _get_layer_experts(self, layer_idx: int):
        block = resolve_text_model(self.model).layers[layer_idx]
        if not _is_moe_layer(block):
            raise ValueError(
                f"Layer {layer_idx} is a dense layer; choose a MoE layer for LUNAR-MoE."
            )
        experts_module = block.feed_forward.experts
        hidden_size = resolve_text_config(self.model).hidden_size
        if is_fused_experts(experts_module):
            return fused_experts_as_list(experts_module, hidden_size)
        return list(experts_module)

    def _get_expert_down_proj(self, expert):
        return expert.down_proj

    def _get_num_experts(self) -> int:
        return resolve_text_config(self.model).num_local_experts

    def _get_router(self, layer_idx: int):
        return resolve_text_model(self.model).layers[layer_idx].feed_forward.router

    # --- Standard ModelBase methods ---

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_llama4_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_llama4_weights, direction=direction, coeff=coeff, layer=layer
        )

    def _to(self, device=None, dtype=None):
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
