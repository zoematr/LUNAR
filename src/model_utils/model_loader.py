# Copyright (c) Meta Platforms, Inc. and affiliates.

from src.model_utils.llama2_model import Llama2Model
from src.model_utils.llama3_model import Llama3Model
from src.model_utils.llama4_model import Llama4Model
from src.model_utils.mistral_model import MistralModel
from src.model_utils.mistral_small4_model import MistralSmall4Model
from src.model_utils.gemma_model import GemmaModel
from src.model_utils.qwen_model import QwenModel
from src.model_utils.qwen2moe_model import Qwen2MoEModel
from src.model_utils.qwen3moe_model import Qwen3MoEModel
from src.model_utils.qwen3_5moe_model import Qwen3_5MoEModel
from src.model_utils.olmo_model import OLMoModel
from src.model_utils.olmoe_model import OLMoEModel


def load_model(model_family, model_path, device, load_in_4bit=False):
    kwargs = {"load_in_4bit": load_in_4bit} if load_in_4bit else {}
    if model_family == "llama2-7b-chat":
        model_base = Llama2Model(model_path, **kwargs)
    elif model_family == "mistral-7b-instruct":
        model_base = MistralModel(model_path, **kwargs)
    elif model_family == "mistral-small-4":
        model_base = MistralSmall4Model(model_path, **kwargs)
    elif model_family == "llama3-8b-instruct":
        model_base = Llama3Model(model_path, **kwargs)
    elif model_family == "llama4-scout":
        model_base = Llama4Model(model_path, **kwargs)
    elif model_family == "gemma-7b-it":
        model_base = GemmaModel(model_path, **kwargs)
    elif model_family == "Qwen2-7B-Instruct":
        model_base = QwenModel(model_path, **kwargs)
    elif model_family in ("Qwen2-57B-A14B-Instruct", "Qwen1.5-MoE-A2.7B-Chat"):
        model_base = Qwen2MoEModel(model_path, **kwargs)
    elif model_family == "Qwen3-30B-A3B":
        model_base = Qwen3MoEModel(model_path, **kwargs)
    elif model_family == "Qwen3.6-35B-A3B":
        model_base = Qwen3_5MoEModel(model_path, **kwargs)
    elif model_family == "olmo-2-1b-instruct":
        model_base = OLMoModel(model_path, **kwargs)
    elif model_family == "olmoe-1b-7b-instruct":
        model_base = OLMoEModel(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown model family: {model_path}")
    if not load_in_4bit:
        model_base = model_base._to(device)
    return model_base
