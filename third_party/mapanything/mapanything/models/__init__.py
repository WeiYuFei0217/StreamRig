# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0; see ../../LICENSE.
# Modified for StreamRig: trimmed to the model entry points used here (see NOTICE).

"""供 StreamRig 使用的 MapAnything 主干模型入口。"""

from omegaconf import DictConfig, OmegaConf

from mapanything.models.mapanything import MapAnything


def resolve_special_float(value):
    if value == "inf":
        return float("inf")
    if value == "-inf":
        return float("-inf")
    raise ValueError(f"Unknown special float value: {value}")


def init_model(model_str: str, model_config: DictConfig, torch_hub_force_reload=False):
    """从 OmegaConf 配置构建随发布附带的 MapAnything 主干。"""
    if not OmegaConf.has_resolver("special_float"):
        OmegaConf.register_new_resolver("special_float", resolve_special_float)
    model_dict = OmegaConf.to_container(model_config, resolve=True)
    return model_factory(model_str, torch_hub_force_reload=torch_hub_force_reload, **model_dict)


def model_factory(model_str: str, **kwargs):
    if model_str != "mapanything":
        raise ValueError(f"Unknown model: {model_str}. Valid options are: mapanything")
    return MapAnything(**kwargs)


def get_available_models() -> list:
    return ["mapanything"]


__all__ = ["MapAnything", "init_model", "model_factory", "get_available_models"]
