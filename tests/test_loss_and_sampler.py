"""
test_loss_and_sampler.py -- 损失与变长窗口 sampler 的回归测试（CPU，pytest）。

保护目标：
  1. StreamPoseLoss 输出与内联的参考公式数值一致
  2. sqrt_chordal = sqrt(chordal_frob + 1e-8)，且随旋转角单调
  3. 平移按 GT 距离归一化的手算小例（含 eps 下限分支）
  4. VariableNStreamBatchSampler：batch 内 N 一致、范围正确、偏长分布、
     DDP 各 rank N 序列相同/seed 不同、set_epoch 改变序列

用法：
    python -m pytest tests/test_loss_and_sampler.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamrig.losses import StreamPoseLoss
from streamrig.datasets import VariableNStreamBatchSampler


# ---------------------------------------------------------------- 构造工具
def _rand_rot(*shape):
    """随机旋转矩阵 [*, 3, 3]（QR 正交化）。"""
    g = torch.Generator().manual_seed(1234)
    A = torch.randn(*shape, 3, 3, generator=g)
    Q, R = torch.linalg.qr(A)
    det = torch.det(Q)
    Q[..., :, 0] = Q[..., :, 0] * det.unsqueeze(-1)  # 保证 det=+1
    return Q


def _make_pred_batch(B=2, N=12, W=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    pred = {
        "rotation_matrices_intra": _rand_rot(B, W - 1),
        "trans_intra": torch.randn(B, W - 1, 3, generator=g) * 0.1,
        "rotation_matrices_inter": _rand_rot(B, N - 1, W),
        "trans_inter": torch.randn(B, N - 1, W, 3, generator=g) * 2.0,
    }
    calib = torch.eye(4).view(1, 1, 4, 4).repeat(B, W, 1, 1)
    for k in range(1, W):
        calib[:, k, 0, 3] = 0.1 * k
    T_rel = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1).clone()
    Rr = _rand_rot(B, N)
    tr = torch.randn(B, N, 3, generator=g) * 3.0
    T_rel[..., :3, :3] = Rr
    T_rel[..., :3, 3] = tr
    T_rel[:, 0] = torch.eye(4)
    batch = {"rig_calib": calib, "T_rel_gt": T_rel}
    return pred, batch


def _reference_loss(pred, batch, rotation_weight=5.0, translation_weight=1.0,
                    intra_weight=0.5, inter_weight=1.0, loss_clamp=10.0, eps=0.1):
    """StreamPoseLoss 的逐行内联参考实现（sqrt chordal + 平移按距离归一化 + 逐步距均匀平均）。"""
    rig_calib = batch["rig_calib"]
    T_rel_gt = batch["T_rel_gt"]
    R_gt_intra = rig_calib[:, 1:, :3, :3]
    t_gt_intra = rig_calib[:, 1:, :3, 3]
    T_inter = T_rel_gt[:, 1:].unsqueeze(2) @ rig_calib.unsqueeze(1)
    R_gt_inter = T_inter[..., :3, :3]
    t_gt_inter = T_inter[..., :3, 3]

    def sqrt_chordal(Rp, Rg):
        I = torch.eye(3, dtype=Rp.dtype)
        for _ in range(Rp.dim() - 2):
            I = I.unsqueeze(0)
        I = I.expand_as(Rp)
        return torch.sqrt(F.mse_loss(torch.matmul(Rp.transpose(-1, -2), Rg), I) + 1e-8)

    rot_intra = sqrt_chordal(pred["rotation_matrices_intra"], R_gt_intra)
    trans_intra = F.l1_loss(pred["trans_intra"], t_gt_intra)
    n_d = R_gt_inter.shape[1]
    rot_inter = torch.stack([
        sqrt_chordal(pred["rotation_matrices_inter"][:, d], R_gt_inter[:, d]) for d in range(n_d)
    ]).mean()
    denom = t_gt_inter.norm(dim=-1, keepdim=True).clamp(min=eps)
    trans_inter = ((pred["trans_inter"] - t_gt_inter).abs() / denom).mean(dim=(0, 2, 3)).mean()
    total = intra_weight * (rotation_weight * rot_intra + translation_weight * trans_intra) \
        + inter_weight * (rotation_weight * rot_inter + translation_weight * trans_inter)
    return torch.clamp(total, max=loss_clamp)


# ---------------------------------------------------------------- 1. 与参考公式一致
def test_loss_equals_reference():
    pred, batch = _make_pred_batch()
    out = StreamPoseLoss(loss_clamp=0.0)(pred, batch)
    ref = _reference_loss(pred, batch, loss_clamp=1e9)
    assert torch.allclose(out["loss"], ref, rtol=1e-6, atol=0), \
        f"损失与参考公式不一致: {out['loss'].item()} vs {ref.item()}"


# ---------------------------------------------------------------- 2. sqrt_chordal
def test_sqrt_chordal_matches_sqrt_of_frob():
    crit = StreamPoseLoss()
    Rp, Rg = _rand_rot(4, 3), torch.eye(3).expand(4, 3, 3, 3)
    expect = torch.sqrt(StreamPoseLoss._chordal_frob(Rp, Rg) + 1e-8)
    assert torch.allclose(crit._rotation_loss(Rp, Rg), expect, rtol=0, atol=0)


def test_sqrt_chordal_monotone_in_angle():
    from scipy.spatial.transform import Rotation
    crit = StreamPoseLoss()
    I = torch.eye(3).view(1, 3, 3)
    prev = 0.0
    for deg in (1, 5, 20, 60, 120):
        R = torch.from_numpy(Rotation.from_euler("z", deg, degrees=True).as_matrix()).float().view(1, 3, 3)
        cur = crit._rotation_loss(R, I).item()
        assert cur > prev, f"{deg}° 不单调"
        prev = cur


# ---------------------------------------------------------------- 3. 距离归一化
def test_trans_norm_by_dist_hand_case():
    """手算小例：每维 err=0.1，‖t_gt‖=2 → 归一化 L1=0.1/2=0.05；‖t_gt‖=0 → 除 eps=0.1。"""
    B, N, W = 1, 2, 2  # W=2：两相机标定均为 identity，intra GT 为零位姿
    eye = torch.eye(3).view(1, 1, 1, 3, 3)
    pred = {
        "rotation_matrices_intra": torch.eye(3).view(1, 1, 3, 3).clone(),
        "trans_intra": torch.zeros(B, W - 1, 3),
        "rotation_matrices_inter": eye.expand(B, N - 1, W, 3, 3).clone(),
        # 两相机 t_gt 均为 (2,0,0)，预测偏 (0.1,0.1,0.1)
        "trans_inter": torch.tensor([[[[2.1, 0.1, 0.1], [2.1, 0.1, 0.1]]]]),
    }
    calib = torch.eye(4).view(1, 1, 4, 4).repeat(B, W, 1, 1)
    T_rel = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1)
    T_rel[0, 1, 0, 3] = 2.0
    batch = {"rig_calib": calib, "T_rel_gt": T_rel}
    crit = StreamPoseLoss()
    out = crit(pred, batch)
    assert abs(out["trans_loss_inter"].item() - 0.05) < 1e-6

    # eps 分支：t_gt 为 0 时除 eps=0.1 而不是除 0
    T_rel2 = T_rel.clone()
    T_rel2[0, 1, 0, 3] = 0.0
    pred2 = dict(pred)
    pred2["trans_inter"] = torch.tensor([[[[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]]])
    out2 = crit(pred2, {"rig_calib": calib, "T_rel_gt": T_rel2})
    assert abs(out2["trans_loss_inter"].item() - 1.0) < 1e-6  # 0.1/0.1=1.0


# ---------------------------------------------------------------- 4. 变长 N sampler
def test_variable_n_sampler_basic():
    s = VariableNStreamBatchSampler(num_batches=200, batch_size=6, n_min=2, n_max=12, seed=42)
    ns = []
    for batch in s:
        assert len(batch) == 6
        n_set = {n for (n, _) in batch}
        assert len(n_set) == 1, "batch 内 N 不一致"
        n = n_set.pop()
        assert 2 <= n <= 12
        ns.append(n)
    assert len(ns) == 200
    # 偏长：均值应明显高于均匀分布的 7.0（P(N)∝N 的期望 ≈ 8.43）
    assert np.mean(ns) > 7.5, f"N 均值 {np.mean(ns):.2f} 未体现偏长"


def test_variable_n_sampler_ddp_alignment():
    s0 = VariableNStreamBatchSampler(100, 4, 2, 12, seed=42, rank=0, world_size=2)
    s1 = VariableNStreamBatchSampler(100, 4, 2, 12, seed=42, rank=1, world_size=2)
    b0, b1 = list(s0), list(s1)
    n0 = [b[0][0] for b in b0]
    n1 = [b[0][0] for b in b1]
    assert n0 == n1, "各 rank N 序列应相同（batch 形状对齐）"
    seeds0 = {sd for b in b0 for (_, sd) in b}
    seeds1 = {sd for b in b1 for (_, sd) in b}
    assert not (seeds0 & seeds1), "各 rank 窗口 seed 应不同"


def test_variable_n_sampler_set_epoch():
    s = VariableNStreamBatchSampler(50, 4, 2, 12, seed=42)
    e0 = [sd for b in s for (_, sd) in b]
    s.set_epoch(1)
    e1 = [sd for b in s for (_, sd) in b]
    assert e0 != e1, "set_epoch 后采样序列应改变"
    s.set_epoch(0)
    e0b = [sd for b in s for (_, sd) in b]
    assert e0 == e0b, "同 epoch 采样应可复现"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
