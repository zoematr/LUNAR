# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import abstractmethod
from typing import List
import torch

from src.model_utils.model_base import ModelBase


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
