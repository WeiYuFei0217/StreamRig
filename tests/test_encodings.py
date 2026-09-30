"""
test_encodings.py -- CausalBridge 组 / 身份编码的回归测试（CPU，pytest）。

保护目标：
  1. 任意窗口长度下前向正常、形状正确
     （binary group embedding + 共享 inter 身份块 + 无时序位置编码，不受窗口长度限制）；
  2. 因果性：扰动未来组不改变历史组输出；
  3. warm-start：同形张量直接拷贝，latent query 数不同时载入前缀，backbone 键跳过；
     键名与模型不匹配时报错。

用法：
    python -m pytest tests/test_encodings.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamrig.models.stream_bridge import StreamRigBridge, StreamPoseHead


# 小模型规格（快）
DIM, HEADS, W, L = 64, 4, 3, 8


def _mk_bridge(**kw):
    args = dict(embed_dim=DIM, num_heads=HEADS, num_merged_layers=2,
                num_latents_per_frame=L, num_frames=W)
    args.update(kw)
    torch.manual_seed(7)
    return StreamRigBridge(**args).eval()


def _mk_head(**kw):
    args = dict(embed_dim=DIM, hidden_dim=32, num_heads=HEADS, num_layers=2, num_frames=W)
    args.update(kw)
    torch.manual_seed(11)
    return StreamPoseHead(**args).eval()


def _run(bridge, head, x):
    tokens, snaps = bridge(x)
    return tokens, head(tokens, anchor_snaps=snaps)


def _latents(N, seed=0):
    torch.manual_seed(seed)
    return torch.randn(2, N, W, L, DIM)


# ---------------------------------------------------------------- 1. 任意窗口长度
@pytest.mark.parametrize("n", [2, 5, 24, 60])
def test_any_window_length(n):
    b = _mk_bridge()
    h = _mk_head()
    x = _latents(n, seed=1)
    with torch.no_grad():
        tok, out = _run(b, h, x)
    assert tok.shape == x.shape
    assert out["rot_inter"].shape == (2, n - 1, W, 6)
    assert out["trans_inter"].shape == (2, n - 1, W, 3)
    assert torch.isfinite(out["rot_inter"]).all() and torch.isfinite(out["trans_inter"]).all()


# ---------------------------------------------------------------- 2. 因果性(扰动未来不改历史)
def test_causality():
    b = _mk_bridge()
    h = _mk_head()
    N, t_pert = 10, 6
    x1 = _latents(N, seed=2)
    x2 = x1.clone()
    x2[:, t_pert:] += 1.0  # 扰动组 t_pert 及之后
    with torch.no_grad():
        _, o1 = _run(b, h, x1)
        _, o2 = _run(b, h, x2)
    # 历史组(inter 索引 0..t_pert-2 对应组 1..t_pert-1)与 intra(组0)必须逐位不变
    assert torch.equal(o1["rot_intra"], o2["rot_intra"])
    assert torch.equal(o1["trans_intra"], o2["trans_intra"])
    assert torch.equal(o1["rot_inter"][:, : t_pert - 1], o2["rot_inter"][:, : t_pert - 1])
    assert torch.equal(o1["trans_inter"][:, : t_pert - 1], o2["trans_inter"][:, : t_pert - 1])
    # 未来组必须确实变化(sanity)
    assert not torch.equal(o1["rot_inter"][:, t_pert - 1 :], o2["rot_inter"][:, t_pert - 1 :])


# ---------------------------------------------------------------- 3. warm-start 加载规则
def test_warmstart_direct_copy_and_latent_prefix(tmp_path):
    from streamrig.utils.warmstart import warmstart_from_checkpoint

    class _Dummy(torch.nn.Module):
        """只暴露 warmstart 需要的最小接口。"""
        def __init__(self):
            super().__init__()
            self.bridge = torch.nn.Module()
            self.bridge.group_embed = torch.nn.Parameter(torch.zeros(2, 1, DIM))
            self.pose_head = torch.nn.Module()
            self.pose_head.frame_identity_embed = torch.nn.Parameter(
                torch.zeros(2 * W - 1, 1, DIM))
            self.resampler = torch.nn.Module()
            self.resampler.latents = torch.nn.Parameter(torch.zeros(16, DIM))
            self.bridge.merged_layers = torch.nn.ModuleList(
                [torch.nn.Linear(2, 2, bias=False) for _ in range(2)])

    torch.manual_seed(9)
    src = {
        "bridge.group_embed": torch.randn(2, 1, DIM),
        "pose_head.frame_identity_embed": torch.randn(2 * W - 1, 1, DIM),
        "resampler.latents": torch.randn(64, DIM),   # latent 数多于模型：载入前缀
        "bridge.merged_layers.0.weight": torch.randn(2, 2),
        "backbone.some.weight": torch.randn(3),
        "extra.only_in_source": torch.randn(1),
    }
    ckpt = tmp_path / "fake_init.pt"
    torch.save({"model_state_dict": src}, ckpt)

    m = _Dummy()
    report = warmstart_from_checkpoint(m, str(ckpt))
    assert torch.equal(m.bridge.group_embed.data, src["bridge.group_embed"])
    assert torch.equal(m.pose_head.frame_identity_embed.data,
                       src["pose_head.frame_identity_embed"])
    assert torch.equal(m.resampler.latents.data, src["resampler.latents"][:16])
    assert report["remapped"] == ["resampler.latents"]
    assert report["skipped_backbone"] == 1
    assert report["num_merged_loaded"] == 1
    assert report["source_only"] == ["extra.only_in_source"]
    assert report["model_not_covered"] == ["bridge.merged_layers.1.weight"]

    # 键名不匹配（如带 DDP 的 module. 前缀）时报错，且不修改模型参数
    bad = tmp_path / "prefixed.pt"
    torch.save({"module." + k: v for k, v in src.items()}, bad)
    m2 = _Dummy()
    before = {k: v.clone() for k, v in m2.state_dict().items()}
    with pytest.raises(ValueError, match="warm-start 失败"):
        warmstart_from_checkpoint(m2, str(bad))
    assert all(torch.equal(before[k], v) for k, v in m2.state_dict().items())
