"""CPU 回归：streamrig.config.build_stream_model 把 config 的 model 段完整传给 StreamRigModel。

训练（scripts/train_stream.py）与评测（scripts/eval_stream.py）都经这一入口构建模型；
本测试确认 yaml 中的结构键被透传、缺省值与发布结构一致。

用法：
    python -m pytest tests/test_config.py -q
"""

from __future__ import annotations

import yaml

import streamrig.models
from streamrig.config import build_stream_model


class _SpyModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def to(self, device):
        self.device = device
        return self


def _capture_kwargs(monkeypatch, mcfg):
    monkeypatch.setattr(streamrig.models, "StreamRigModel", _SpyModel)
    return build_stream_model(mcfg, "cpu").kwargs


def test_yaml_model_keys_are_passed_through(monkeypatch):
    mcfg = yaml.safe_load("""
num_latents: 8
num_frames_per_group: 3
resampler_layers: 1
bridge_merged_layers: 4
pose_head_hidden_dim: 256
pose_head_layers: 2
""")
    assert _capture_kwargs(monkeypatch, mcfg) == mcfg


def test_defaults_match_release_architecture(monkeypatch):
    kwargs = _capture_kwargs(monkeypatch, {"num_frames_per_group": 3})
    assert kwargs == {
        "num_latents": 16,
        "num_frames_per_group": 3,
        "resampler_layers": 2,
        "bridge_merged_layers": 8,
        "pose_head_hidden_dim": 512,
        "pose_head_layers": 3,
    }
