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

# Qwen3 MoE uses the same im_start/im_end ChatML format as Qwen2.
#
# Thinking mode: Qwen3 emits a <think>...</think> reasoning block *before* the
# answer when thinking is enabled (the default). For LUNAR that is fatal — the
# first generated/assistant-position token would be "<think>", not the
# refusal/answer token, so the refusal direction and EOI position would be
# measured over the reasoning block instead of the answer.
#
# `apply_chat_template(..., enable_thinking=False)` disables it by seeding the
# assistant turn with an *empty* think block "<think>\n\n</think>\n\n". We bake
# that empty block into the template suffix so that (a) generation starts at the
# answer and (b) `_get_eoi_toks()` captures the empty block, anchoring the
# refusal-token measurement at the first answer position. (Qwen3 shares the
# Qwen2 BPE vocab, so the `/no_think` soft switch is unnecessary here.)
QWEN3MOE_SYSTEM_PROMPT = "You are a helpful assistant."

QWEN3MOE_CHAT_TEMPLATE = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
<think>

</think>

"""

# Qwen3 reuses the Qwen2 BPE vocab, so these IDs carry over from qwen2moe.
# Sanity check once: tokenizer.convert_ids_to_tokens([40, 2121]) -> ['I', 'As'].
QWEN3MOE_REFUSAL_TOKS = [40, 2121]  # ['I', 'As']


def format_instruction_qwen3moe_chat(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True,
):
    formatted = QWEN3MOE_CHAT_TEMPLATE.format(
        system=QWEN3MOE_SYSTEM_PROMPT,
        instruction=instruction,
    )
    if not include_trailing_whitespace:
        formatted = formatted.rstrip()
    if output is not None:
        formatted += output
    return formatted


def tokenize_instructions_qwen3moe_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_qwen3moe_chat(
                instruction=instr,
                output=out,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen3moe_chat(
                instruction=instr,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr in instructions
        ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def orthogonalize_qwen3moe_weights(model, direction: Float[Tensor, "d_model"]):
    text_model = resolve_text_model(model)
    hidden_size = resolve_text_config(model).hidden_size
    text_model.embed_tokens.weight.data = get_orthogonalized_matrix(
        text_model.embed_tokens.weight.data, direction
    )
    for block in text_model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        # Qwen3-30B-A3B has all layers MoE (mlp_only_layers=[]), but guard for
        # variants where some layers are a dense Qwen3MoeMLP (no .experts).
        experts = getattr(block.mlp, "experts", None)
        if experts is None:
            continue
        if is_fused_experts(experts):
            # Newer transformers: Qwen3MoeExperts stores a batched 3D down_proj.
            orthogonalize_fused_down_proj(experts.down_proj, direction, hidden_size)
        else:
            for expert in experts:
                expert.down_proj.weight.data = get_orthogonalized_matrix(
                    expert.down_proj.weight.data.T, direction
                ).T


def act_add_qwen3moe_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    experts = resolve_text_model(model).layers[layer - 1].mlp.experts
    if is_fused_experts(experts):
        # Fused experts have no per-expert bias and a bias-less batched matmul,
        # so activation-addition cannot be applied. Use the weight-swap path.
        raise NotImplementedError(
            "Activation-addition is not supported for fused Qwen3MoeExperts; "
            "use the MoE weight-swap path (run_lunar_moe.py)."
        )
    for expert in experts:
        dtype = expert.down_proj.weight.dtype
        device = expert.down_proj.weight.device
        expert.down_proj.bias = torch.nn.Parameter(
            (coeff * direction).to(dtype=dtype, device=device)
        )


class Qwen3MoEModel(MoEModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="/tmp/offload_qwen3moe",
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
            tokenize_instructions_qwen3moe_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            QWEN3MOE_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False
        )

    def _get_refusal_toks(self):
        return QWEN3MOE_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return resolve_text_model(self.model).layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList(
            [block.self_attn for block in self.model_block_modules]
        )

    def _get_mlp_modules(self):
        return torch.nn.ModuleList(
            [block.mlp for block in self.model_block_modules]
        )

    # --- MoEModelBase methods ---
    # Handles both layouts: older transformers store experts as an nn.ModuleList;
    # newer ones store a fused Qwen3MoeExperts (batched 3D down_proj).

    def _get_layer_experts(self, layer_idx: int):
        experts = resolve_text_model(self.model).layers[layer_idx].mlp.experts
        if is_fused_experts(experts):
            hidden_size = resolve_text_config(self.model).hidden_size
            return fused_experts_as_list(experts, hidden_size)
        return list(experts)

    def _get_expert_down_proj(self, expert):
        return expert.down_proj

    def _get_num_experts(self) -> int:
        return resolve_text_config(self.model).num_experts

    def _get_router(self, layer_idx: int):
        return resolve_text_model(self.model).layers[layer_idx].mlp.gate

    # --- Standard ModelBase methods ---

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_qwen3moe_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_qwen3moe_weights, direction=direction, coeff=coeff, layer=layer
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
