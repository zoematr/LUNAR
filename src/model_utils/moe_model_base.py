# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import abstractmethod
from typing import List
import torch

from src.model_utils.model_base import ModelBase
from src.utils.utils import get_orthogonalized_matrix


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


# ---------------------------------------------------------------------------
# Fused experts.
#
# Newer transformers store a layer's routed experts as a single batched module
# (e.g. Llama4TextExperts, Qwen3MoeExperts) with stacked 3D nn.Parameters rather
# than an nn.ModuleList of nn.Linear experts. The down_proj is [E, A, B] where
# one of {A, B} is hidden_size and the other is the (moe) intermediate size, and
# the stored orientation differs across architectures:
#   * Llama 4    : [E, intermediate, hidden]   (hidden last; bmm convention)
#   * Qwen3-MoE  : [E, hidden, intermediate]   (hidden middle; nn.Linear convention)
# The adapters below auto-detect the orientation by matching hidden_size and
# expose the nn.Linear convention (.weight is [hidden, intermediate]) that the
# LUNAR contract (run_lunar_moe.py / dataset_utils.py) reads and writes.
# ---------------------------------------------------------------------------


def is_fused_experts(experts_module) -> bool:
    """True if a layer's experts are a fused batched module, not an nn.ModuleList."""
    if isinstance(experts_module, torch.nn.ModuleList):
        return False
    dp = getattr(experts_module, "down_proj", None)
    return isinstance(dp, torch.Tensor) and dp.dim() == 3


def _hidden_axis(fused_down_proj: torch.Tensor, hidden_size: int) -> int:
    """Return the per-expert axis (0 or 1) of the slice that equals hidden_size."""
    _, a, b = fused_down_proj.shape
    if a == hidden_size:
        return 0  # slice [hidden, intermediate] (nn.Linear convention)
    if b == hidden_size:
        return 1  # slice [intermediate, hidden] (bmm convention)
    raise ValueError(
        f"fused down_proj {tuple(fused_down_proj.shape)} has no axis matching "
        f"hidden_size {hidden_size}"
    )


class _FusedExpertWeight:
    """nn.Linear.weight-like view ([hidden, intermediate]) over expert `idx` of a
    fused 3D down_proj parameter. Reads clone to a [hidden, intermediate] tensor;
    writes accept a [hidden, intermediate] tensor and store it back in-place in
    the parameter's native orientation."""

    def __init__(self, fused_param: torch.nn.Parameter, idx: int, hidden_size: int):
        self._fused = fused_param
        self._idx = idx
        self._hidden_first = _hidden_axis(fused_param, hidden_size) == 0

    def _as_linear(self, slice_2d: torch.Tensor) -> torch.Tensor:
        return slice_2d if self._hidden_first else slice_2d.t()

    @property
    def shape(self) -> torch.Size:
        return self._as_linear(self._fused[self._idx]).shape

    def clone(self) -> torch.Tensor:
        return self._as_linear(self._fused.data[self._idx]).contiguous().clone()

    @property
    def data(self) -> torch.Tensor:
        return self._as_linear(self._fused.data[self._idx])

    @data.setter
    def data(self, value: torch.Tensor):
        stored = value if self._hidden_first else value.t()
        self._fused.data[self._idx] = stored.to(
            dtype=self._fused.dtype, device=self._fused.device
        )


class _FusedDownProj:
    def __init__(self, weight: _FusedExpertWeight):
        self.weight = weight


class FusedExpertProxy:
    """Uniform `.down_proj.weight` view over one expert of a fused experts module."""

    def __init__(self, fused_down_proj: torch.nn.Parameter, idx: int, hidden_size: int):
        self.down_proj = _FusedDownProj(
            _FusedExpertWeight(fused_down_proj, idx, hidden_size)
        )


def fused_experts_as_list(experts_module, hidden_size: int) -> List["FusedExpertProxy"]:
    """Return per-expert proxies for a fused experts module."""
    fused = experts_module.down_proj
    return [FusedExpertProxy(fused, i, hidden_size) for i in range(fused.shape[0])]


def orthogonalize_fused_down_proj(fused_param, direction, hidden_size: int):
    """Project `direction` out of the hidden (output) axis of a fused down_proj,
    in-place, regardless of the parameter's stored orientation."""
    if _hidden_axis(fused_param, hidden_size) == 1:
        # [E, intermediate, hidden] — hidden already last.
        fused_param.data = get_orthogonalized_matrix(fused_param.data, direction)
    else:
        # [E, hidden, intermediate] — move hidden last, orthogonalize, move back.
        fused_param.data = (
            get_orthogonalized_matrix(fused_param.data.transpose(1, 2), direction)
            .transpose(1, 2)
            .contiguous()
        )


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
