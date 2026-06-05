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
    load_generative_lm,
    is_fused_experts,
    fused_experts_as_list,
    orthogonalize_fused_down_proj,
)

# ---------------------------------------------------------------------------
# Mistral Small 4 (e.g. mistralai/Mistral-Small-4-119B-2603)
#
# IMPORTANT — this is NOT the Mixtral architecture. Per the checkpoint's
# config.json it is:
#   architectures : ["Mistral3ForConditionalGeneration"]   (multimodal wrapper)
#   model_type    : "mistral3"; text_config.model_type : "mistral4"
#   MoE           : DeepSeek-style — n_routed_experts=128, n_shared_experts=1,
#                   num_experts_per_tok=4, first_k_dense_replace=0 (all-MoE),
#                   field names match DeepSeek-V3 (n_routed_experts, n_group,
#                   topk_group, routed_scaling_factor, ...).
#   Attention     : MLA (q_lora_rank=1024, kv_lora_rank=256, qk_*_head_dim, ...).
#                   The output projection self_attn.o_proj still exists and is
#                   the only attention write into the residual stream, so it is
#                   the correct orthogonalization target; the q_a/q_b/kv_a/kv_b
#                   low-rank projections must NOT be touched.
#   Tokenizer     : Tekken (vocab 131072) — NOT the old SentencePiece vocab.
#
# Two hard requirements before this will run:
#   1. transformers >= 5.3 (the checkpoint declares transformers_version
#      "5.3.0.dev0"; model_type "mistral4" is unknown to 4.x and will fail to
#      load). Update requirements.txt accordingly.
#   2. Module names confirmed against the loaded mistral4 model (meta device):
#      block.mlp is Mistral4MoE with .experts / .shared_experts / .gate, and the
#      routed experts are a FUSED Mistral4NaiveMoe (batched 3D down_proj of shape
#      [num_experts, hidden, intermediate]), handled via the shared fused adapter.
# ---------------------------------------------------------------------------

# Tekken chat format, verified against the model's own chat_template.jinja:
#   <s>[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS][INST]...[/INST]
# The [MODEL_SETTINGS] block with reasoning_effort "none" is how this model
# disables its reasoning trace (the analogue of Qwen3's enable_thinking=False) —
# essential for LUNAR, otherwise a [THINK]... block would precede the answer and
# contaminate the first-token refusal measurement. <s> (BOS) is added by the
# tokenizer, so it is not included here. The JSON braces are doubled for .format().
MISTRAL_SMALL4_CHAT_TEMPLATE = (
    '[MODEL_SETTINGS]{{"reasoning_effort": "none"}}[/MODEL_SETTINGS]'
    "[INST]{instruction}[/INST]"
)

# Tekken vocab (131072). Verified against the Mistral-Small-4 tokenizer:
#   encode("I", add_special_tokens=False) == [1073]
# (The old Mistral-7B SentencePiece id 315, and 40, do NOT apply: 40 = '<SPECIAL_40>'.)
MISTRAL_SMALL4_REFUSAL_TOKS = [1073]  # 'I'


def format_instruction_mistral_small4_chat(
    instruction: str,
    output: str = None,
    include_trailing_whitespace: bool = True,
):
    formatted = MISTRAL_SMALL4_CHAT_TEMPLATE.format(instruction=instruction)
    if not include_trailing_whitespace:
        formatted = formatted.rstrip()
    if output is not None:
        formatted += output
    return formatted


def tokenize_instructions_mistral_small4_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    include_trailing_whitespace: bool = True,
):
    if outputs is not None:
        prompts = [
            format_instruction_mistral_small4_chat(
                instruction=instr,
                output=out,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr, out in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_mistral_small4_chat(
                instruction=instr,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instr in instructions
        ]
    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")


def _routed_experts(block):
    """Routed-expert ModuleList for a DeepSeek-style MoE layer, or None if dense."""
    mlp = block.mlp
    return getattr(mlp, "experts", None)


def orthogonalize_mistral_small4_weights(model, direction: Float[Tensor, "d_model"]):
    text_model = resolve_text_model(model)
    hidden_size = resolve_text_config(model).hidden_size
    text_model.embed_tokens.weight.data = get_orthogonalized_matrix(
        text_model.embed_tokens.weight.data, direction
    )
    for block in text_model.layers:
        # MLA: o_proj is the residual write; leave the low-rank q/kv projections.
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(
            block.self_attn.o_proj.weight.data.T, direction
        ).T
        experts = _routed_experts(block)
        if experts is None:
            # Dense layer (first_k_dense_replace): single mlp.down_proj.
            block.mlp.down_proj.weight.data = get_orthogonalized_matrix(
                block.mlp.down_proj.weight.data.T, direction
            ).T
            continue
        if is_fused_experts(experts):
            # Mistral4NaiveMoe stores a batched 3D down_proj.
            orthogonalize_fused_down_proj(experts.down_proj, direction, hidden_size)
        else:
            for expert in experts:
                expert.down_proj.weight.data = get_orthogonalized_matrix(
                    expert.down_proj.weight.data.T, direction
                ).T
        # DeepSeek-style shared expert (n_shared_experts=1) writes on every token.
        block.mlp.shared_experts.down_proj.weight.data = get_orthogonalized_matrix(
            block.mlp.shared_experts.down_proj.weight.data.T, direction
        ).T


def act_add_mistral_small4_weights(
    model, direction: Float[Tensor, "d_model"], coeff, layer
):
    text_model = resolve_text_model(model)
    block = text_model.layers[layer - 1]
    experts = _routed_experts(block)
    if experts is not None and is_fused_experts(experts):
        raise NotImplementedError(
            "Activation-addition is not supported for fused Mistral4NaiveMoe experts; "
            "use the MoE weight-swap path (run_lunar_moe.py)."
        )
    targets = list(experts) if experts is not None else [block.mlp]
    shared = getattr(block.mlp, "shared_experts", None)
    if shared is not None:
        targets.append(shared)
    for module in targets:
        dtype = module.down_proj.weight.dtype
        device = module.down_proj.weight.device
        module.down_proj.bias = torch.nn.Parameter(
            (coeff * direction).to(dtype=dtype, device=device)
        )


class MistralSmall4Model(MoEModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Mistral Small 4 is a multimodal Mistral3ForConditionalGeneration
        # (image-text-to-text); its mistral4 text backbone is NOT registered for
        # AutoModelForCausalLM, so load via the multimodal auto class. The text
        # decoder is then reached through resolve_text_model (model.model.language_model).
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # The checkpoint is FP8-quantized with activation_scheme="static". On H100
        # the fused grouped_mm expert kernel does not support static activation
        # scaling (NotImplementedError); force "dynamic" so activation scales are
        # computed at runtime and the fused MoE path works. Patch both the top-level
        # and the nested text_config quantization_config (dict or object form).
        def _force_dynamic(cfg):
            qc = getattr(cfg, "quantization_config", None)
            if isinstance(qc, dict):
                if qc.get("activation_scheme") == "static":
                    qc["activation_scheme"] = "dynamic"
            elif qc is not None and getattr(qc, "activation_scheme", None) == "static":
                qc.activation_scheme = "dynamic"

        _force_dynamic(config)
        if hasattr(config, "text_config"):
            _force_dynamic(config.text_config)

        model = load_generative_lm(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="/tmp/offload_mistral_small4",
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
            tokenize_instructions_mistral_small4_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True,
        )

    def _get_eoi_toks(self):
        return self.tokenizer.encode(
            MISTRAL_SMALL4_CHAT_TEMPLATE.split("{instruction}")[-1],
            add_special_tokens=False,
        )

    def _get_refusal_toks(self):
        return MISTRAL_SMALL4_REFUSAL_TOKS

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

    # --- MoEModelBase methods (Mistral4MoE: mlp.experts/shared_experts/gate) ---

    def _get_layer_experts(self, layer_idx: int):
        block = resolve_text_model(self.model).layers[layer_idx]
        experts = _routed_experts(block)
        if experts is None:
            raise ValueError(
                f"Layer {layer_idx} is a dense layer; choose a MoE layer for LUNAR-MoE."
            )
        # Routed experts only; the shared expert has a different intermediate
        # size and cannot be swapped with the shared EstimatedNet weight.
        if is_fused_experts(experts):
            hidden_size = resolve_text_config(self.model).hidden_size
            return fused_experts_as_list(experts, hidden_size)
        return list(experts)

    def _get_expert_down_proj(self, expert):
        return expert.down_proj

    def _get_num_experts(self) -> int:
        return resolve_text_config(self.model).n_routed_experts

    def _get_router(self, layer_idx: int):
        return resolve_text_model(self.model).layers[layer_idx].mlp.gate

    # --- Standard ModelBase methods ---

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(
            orthogonalize_mistral_small4_weights, direction=direction
        )

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(
            act_add_mistral_small4_weights,
            direction=direction,
            coeff=coeff,
            layer=layer,
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
