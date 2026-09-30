"""
infoshare_io.py -- info_sharing 特征缓存的存取工具 (bf16 以 uint16 位存)。

功能:
  1. 约定并生成特征缓存的目录布局, 与训练加载器 `streamrig` 的读法一一对应:

        <root>/scenes/<scene_id>/trajectories/<traj_id>/features/<ts_key>/rig.npy

     每个 `rig.npy` 是一个 rig(同一时刻的 K 个相机)经 frozen backbone 的
     info_sharing 之后的 token 特征, 形状 [K, TOKENS, DIM](NCLT: [5, 256, 768])。

  2. bf16 位存: numpy 没有 bfloat16 dtype, 因此落盘时把 torch.bfloat16 张量按位
     reinterpret 成 uint16 存成 .npy; 读回时再 bitcast 回 bfloat16 并升到 float32。

  3. 原子写: 先写 `<path>.tmp<pid>.npy` 再 os.replace, 避免多进程/断点续跑时
     产生半截文件。

  4. manifest.json / COMPLETE 标记: 记录本次生成的配置(模型、图像管线、形状、
     数量)与全量完成状态。

示例用法:
    import torch
    from data_preprocessing.common.infoshare_io import (
        rig_feature_path, save_rig_features, load_rig_features,
        write_manifest, write_complete_marker,
    )

    feats = torch.randn(5, 256, 768)                       # [K, TOKENS, DIM] fp32
    p = rig_feature_path("/data/nclt_infoshare_test", "nclt", "2012-02-19", "1329674902134")
    save_rig_features(feats, p)
    back = load_rig_features(p)                            # [5, 256, 768] float32
    assert torch.equal(back, feats.to(torch.bfloat16).float())
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping

import numpy as np
import torch

__all__ = [
    "FEATURES_DIRNAME",
    "RIG_FILENAME",
    "MANIFEST_FILENAME",
    "COMPLETE_FILENAME",
    "trajectory_feature_dir",
    "rig_feature_path",
    "save_rig_features",
    "load_rig_features",
    "write_manifest",
    "write_complete_marker",
]

# ---------------------------------------------------------------------------
# 目录/文件名约定(改这里 = 改全流程, 训练加载器按同一约定读)
# ---------------------------------------------------------------------------

FEATURES_DIRNAME = "features"
RIG_FILENAME = "rig.npy"
MANIFEST_FILENAME = "manifest.json"
COMPLETE_FILENAME = "COMPLETE"


# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

def trajectory_feature_dir(root: str, scene_id: str, traj_id: str) -> str:
    """返回某条轨迹的特征目录 `<root>/scenes/<scene>/trajectories/<traj>/features`。"""
    return os.path.join(root, "scenes", scene_id, "trajectories", traj_id, FEATURES_DIRNAME)


def rig_feature_path(root: str, scene_id: str, traj_id: str, ts_key: str) -> str:
    """返回单个 rig 的特征文件路径 `.../features/<ts_key>/rig.npy`。

    ts_key 必须与 step1 轨迹文件(trajectory.tum)里使用的键一致(NCLT = 13 位毫秒字符串),
    否则训练加载器取 `set(listdir(features)) & set(poses)` 会得到空集。
    """
    return os.path.join(trajectory_feature_dir(root, scene_id, traj_id), str(ts_key), RIG_FILENAME)


# ---------------------------------------------------------------------------
# 读写
# ---------------------------------------------------------------------------

def save_rig_features(feats: torch.Tensor, path: str) -> None:
    """把 [K, TOKENS, DIM] 特征以 bf16(uint16 位存)原子写到 `path`。

    参数:
        feats: torch.Tensor, 三维 [K, TOKENS, DIM], 任意浮点 dtype, 任意 device。
        path : 目标 .npy 路径(建议用 rig_feature_path 生成)。
    """
    if feats.dim() != 3:
        raise ValueError(f"期望 [K, TOKENS, DIM] 三维张量, 收到 {tuple(feats.shape)}")
    arr = feats.detach().to(torch.bfloat16).contiguous().view(torch.uint16).cpu().numpy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # np.save 会自动补 .npy 后缀, 因此临时名不带后缀、replace 时再补上
    tmp = f"{path}.tmp{os.getpid()}"
    np.save(tmp, arr)
    os.replace(tmp + ".npy", path)


def load_rig_features(path: str, expected_dim: int | None = None) -> torch.Tensor:
    """读回 [K, TOKENS, DIM] float32 特征(uint16 → bfloat16 → float32)。

    参数:
        path        : rig.npy 路径。
        expected_dim: 非 None 时校验最后一维, 不符立即报错。
    """
    u16 = np.load(path)
    if u16.ndim != 3:
        raise ValueError(f"特征缓存维度应为 3, 实际 {u16.shape}: {path}")
    if expected_dim is not None and u16.shape[-1] != int(expected_dim):
        raise ValueError(f"特征缓存形状 {u16.shape} 与 expected_dim={expected_dim} 不符: {path}")
    return torch.from_numpy(u16.astype(np.uint16)).view(torch.bfloat16).float()


# ---------------------------------------------------------------------------
# manifest / 完成标记
# ---------------------------------------------------------------------------

def write_manifest(root: str, payload: Mapping[str, Any]) -> str:
    """把本次生成的配置写成 `<root>/manifest.json`(自动补 created_at)。"""
    os.makedirs(root, exist_ok=True)
    doc = dict(payload)
    doc.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    path = os.path.join(root, MANIFEST_FILENAME)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def write_complete_marker(root: str, note: str = "") -> str:
    """写 `<root>/COMPLETE` 标记(内容 = 时间戳 + 备注), 表示该缓存已全量生成。"""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, COMPLETE_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        if note:
            f.write(note.rstrip() + "\n")
    return path
