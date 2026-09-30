"""
config.py -- YAML 配置 → StreamRigModel 的统一构建入口。

训练（scripts/train_stream.py）与评测（scripts/eval_stream.py）共用
:func:`build_stream_model`，保证同一份 config 在两处构建出同一结构。

示例用法：
    cfg = yaml.safe_load(open("configs/nclt_streamrig.yaml"))
    model = build_stream_model(cfg["model"], device="cuda:0")
"""

from __future__ import annotations


def build_stream_model(mcfg: dict, device=None):
    """按 config 的 model 段构建 StreamRigModel，可选移到 device。"""
    from .models import StreamRigModel

    model = StreamRigModel(
        num_latents=mcfg.get("num_latents", 16),
        num_frames_per_group=mcfg["num_frames_per_group"],
        resampler_layers=mcfg.get("resampler_layers", 2),
        bridge_merged_layers=mcfg.get("bridge_merged_layers", 8),
        pose_head_hidden_dim=mcfg.get("pose_head_hidden_dim", 512),
        pose_head_layers=mcfg.get("pose_head_layers", 3),
    )
    return model.to(device) if device is not None else model
