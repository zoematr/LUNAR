# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import abstractmethod
from typing import List
import torch

from src.model_utils.model_base import ModelBase


def resolve_text_model(model):
    """Return the inner decoder stack that exposes ``.layers`` and ``.embed_tokens``.

    Multimodal checkpoints (Llama 4, Mistral Small 4, Qwen3.x-VL...) wrap the
    text transformer one or two levels deep (``model.language_model``,
    ``model.model.language_model``, ...), so the plain ``model.model`` path used
    by text-only models is not enough. We probe the common layouts and fall back
    to a module search.
    """
    base = getattr(model, "model", None)
    candidates = [
        base,                                          # text-only *ForCausalLM
        getattr(base, "language_model", None),         # *ForConditionalGeneration -> Model -> language_model
        getattr(model, "language_model", None),        # some wrappers expose it at the top
        getattr(getattr(model, "language_model", None), "model", None),  # wrapper -> *ForCausalLM -> model
    ]
    for cand in candidates:
        if cand is not None and hasattr(cand, "layers") and hasattr(cand, "embed_tokens"):
            return cand
    for _, module in model.named_modules():
        if hasattr(module, "layers") and hasattr(module, "embed_tokens"):
            return module
    raise AttributeError(
        "Could not locate the text decoder (a module with .layers and .embed_tokens). "
        "Inspect the loaded model structure with print(model)."
    )


def resolve_text_config(model):
    """Return the text sub-config, unwrapping multimodal configs.

    Uses ``config.get_text_config()`` when available (returns ``text_config`` on
    multimodal configs, ``self`` otherwise), falling back to an explicit attr.
    """
    config = model.config
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


class MoEModelBase(ModelBase):
    """
    Abstract base class for Mixture-of-Experts models.
    Extends ModelBase with MoE-specific abstract methods that concrete subclasses
    (e.g. OLMoEModel) must implement to expose expert structure to LUNAR.
    """

    @abstractmethod
    def _get_layer_experts(self, layer_idx: int) -> List[torch.nn.Module]:
        """Returns the list of expert modules for a given transformer layer."""
        pass

    @abstractmethod
    def _get_expert_down_proj(self, expert: torch.nn.Module) -> torch.nn.Linear:
        """Returns the down_proj linear layer of a given expert module."""
        pass

    @abstractmethod
    def _get_num_experts(self) -> int:
        """Returns the total number of experts per layer."""
        pass

    @abstractmethod
    def _get_router(self, layer_idx: int) -> torch.nn.Module:
        """Returns the router (gate) module for a given transformer layer."""
        pass
