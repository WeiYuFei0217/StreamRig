"""
test_anchor_snapshot.py -- 锚快照布局回归测试。

覆盖:
  1. 核心等价性:snapshot N=2 前向 == 一对 rig 的双向注意力前向(bridge 与 pose head 都要一致)
     —— 即 (A_1, G_1) 成对双向注意力与成对重定位模型的读出一致,热启动起点不变;
  2. 因果性:扰动未来组不影响 ≤t 的输出;扰动 A_s(s≠t)不影响 G_t;
  3. 任意 N 形状正确，长窗口可前向;
  4. 真实权重:加载初始化权重后 snapshot N=2 与双向 N=2 数值一致(需设置环境变量)。

示例用法:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_anchor_snapshot.py -q
"""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamrig.models.stream_bridge import StreamRigBridge, StreamPoseHead


# 成对重定位模型的初始化权重：用环境变量 STREAMRIG_RIG_INIT_CKPT 指定（结构规格从权重推断）；
# 未设置则跳过真实权重测试。
_RIG_INIT_CKPT = os.environ.get("STREAMRIG_RIG_INIT_CKPT", "")
requires_ckpt = pytest.mark.skipif(
    not os.path.isfile(_RIG_INIT_CKPT),
    reason="未设置 STREAMRIG_RIG_INIT_CKPT",
)

# 小模型规格(快)
DIM, HEADS, W, L = 64, 4, 3, 8


def _mk_bridge(**kw):
    args = dict(embed_dim=DIM, num_heads=HEADS, num_merged_layers=2,
                num_latents_per_frame=L, num_frames=W)
    args.update(kw)
    torch.manual_seed(7)
    return StreamRigBridge(**args).eval()


def _mk_head(**kw):
    args = dict(embed_dim=DIM, hidden_dim=32, num_heads=HEADS, num_layers=2,
                num_frames=W)
    args.update(kw)
    torch.manual_seed(11)
    return StreamPoseHead(**args).eval()


def _latents(N, seed=0):
    torch.manual_seed(seed)
    return torch.randn(2, N, W, L, DIM)


def _bidir_forward(bridge, x):
    """参考实现:同一套 embedding,全部组之间无 mask 的双向注意力(成对读出)。"""
    B, N, Wx, Lx, C = x.shape
    emb = torch.stack([bridge.embed_group(x[:, g], g) for g in range(N)], dim=1)
    merged = emb.reshape(B, N * Wx * Lx, C)
    merged = torch.cat(
        [merged[:, :Lx] + bridge.anchor_embed_merge.expand(B, Lx, C), merged[:, Lx:]], dim=1)
    merged = bridge._run_merged_layers(merged, attn_mask=None)
    return bridge.out_norm(merged).reshape(B, N, Wx, Lx, C)




# ------------------------------------------------- 1. 核心:snapshot N=2 == bidir N=2
def test_snapshot_n2_equals_bidir_n2():
    b_snap = _mk_bridge()

    x = _latents(2)
    with torch.no_grad():
        out_snap, snaps = b_snap(x)
        out_bidir = _bidir_forward(b_snap, x)
    assert out_snap.shape == out_bidir.shape == x.shape
    assert torch.allclose(out_snap, out_bidir, atol=1e-6), \
        f"snapshot N=2 与双向 N=2 不一致, max diff={(out_snap-out_bidir).abs().max()}"
    # A_1 输出 == 双向模式的组 0 输出
    assert torch.allclose(snaps[:, 0], out_bidir[:, 0], atol=1e-6)

    # pose head 同样一致(N=2 时逐目标 ref 都是 A_1 的 cam0 = 双向模式组 0 的 cam0)
    head = _mk_head()
    with torch.no_grad():
        h_snap = head(out_snap, anchor_snaps=snaps)
        h_bidir = head(out_bidir, anchor_snaps=out_bidir[:, :1])
    for k in h_snap:
        assert torch.allclose(h_snap[k], h_bidir[k], atol=1e-6), f"pose head {k} 不一致"


# ------------------------------------------------- 2. 因果性
def test_snapshot_causality():
    b = _mk_bridge()
    x = _latents(6)
    with torch.no_grad():
        out, snaps = b(x)

    # 扰动未来组 s=4(>t) → 组 1..3 的输出与 A_1..A_3 快照逐位不变
    x2 = x.clone()
    x2[:, 4] += 1.0
    with torch.no_grad():
        out2, snaps2 = b(x2)
    assert torch.equal(out[:, 1:4], out2[:, 1:4]), "扰动未来组影响了因果历史输出"
    assert torch.equal(snaps[:, :3], snaps2[:, :3]), "扰动未来组影响了早期锚快照"
    # 被扰动组及其锚快照必须变化(sanity)
    assert not torch.allclose(out[:, 4], out2[:, 4])
    assert not torch.allclose(snaps[:, 3], snaps2[:, 3])

    # 扰动锚组(组 0)→ 一切都该变(锚快照是它的复制)
    x3 = x.clone()
    x3[:, 0] += 1.0
    with torch.no_grad():
        out3, _ = b(x3)
    assert not torch.allclose(out[:, 1], out3[:, 1])


def test_snapshot_head_causality_via_ref():
    """扰动未来组不应改变早期目标的位姿输出(bridge+head 全链路)。"""
    b = _mk_bridge()
    head = _mk_head()
    x = _latents(5)
    x2 = x.clone()
    x2[:, 4] += 1.0
    with torch.no_grad():
        o1, s1 = b(x)
        o2, s2 = b(x2)
        h1 = head(o1, anchor_snaps=s1)
        h2 = head(o2, anchor_snaps=s2)
    # inter 目标 g=1..3(索引 0..2)不受组 4 扰动影响
    assert torch.equal(h1["rot_inter"][:, :3], h2["rot_inter"][:, :3])
    assert torch.equal(h1["trans_inter"][:, :3], h2["trans_inter"][:, :3])
    assert torch.equal(h1["rot_intra"], h2["rot_intra"])


# ------------------------------------------------- 3. 形状 + 长窗口
@pytest.mark.parametrize("N", [3, 8, 12])
def test_snapshot_shapes(N):
    b = _mk_bridge()
    head = _mk_head()
    x = _latents(N)
    with torch.no_grad():
        out, snaps = b(x)
        h = head(out, anchor_snaps=snaps)
    assert out.shape == (2, N, W, L, DIM)
    assert snaps.shape == (2, N - 1, W, L, DIM)
    assert h["rot_inter"].shape == (2, N - 1, W, 6)
    assert h["rot_intra"].shape == (2, W - 1, 6)


def test_snapshot_long_window():
    """长窗口(N=20)能构建 + 前向，形状正确。"""
    b = _mk_bridge()
    head = _mk_head()
    x = _latents(20)
    with torch.no_grad():
        out, snaps = b(x)
        h = head(out, anchor_snaps=snaps)
    assert out.shape == (2, 20, W, L, DIM)
    assert h["rot_inter"].shape == (2, 19, W, 6)


# ------------------------------------------------- 4. 真实权重
class _BareModel(nn.Module):
    """只含 bridge/pose_head 的容器,复用 warmstart_from_checkpoint 的键前缀。"""

    def __init__(self, bridge, pose_head):
        super().__init__()
        self.bridge = bridge
        self.pose_head = pose_head


@requires_ckpt
def test_warmstart_real_weights_n2_equivalence():
    import re

    from streamrig.utils.warmstart import _extract_state_dict, warmstart_from_checkpoint

    # 结构规格从初始化权重推断：相机数 / 通道 / merged 层数 / latent 数 / pose head MLP
    src = _extract_state_dict(torch.load(_RIG_INIT_CKPT, map_location="cpu"))
    W_r, _, C_r = src["bridge.frame_embed"].shape
    n_layers = len({m.group(1) for k in src
                    if (m := re.match(r"bridge\.merged_layers\.(\d+)\.", k))})
    L_r = src["resampler.latents"].shape[0] if "resampler.latents" in src else 16
    n_mlp = sum(1 for k, v in src.items()
                if re.match(r"pose_head\.rot_mlp\.\d+\.weight$", k) and v.dim() == 2)
    hidden = src["pose_head.rotation_head.weight"].shape[1]

    torch.manual_seed(3)
    bridge = StreamRigBridge(
        embed_dim=C_r, num_heads=8, num_merged_layers=n_layers,
        num_latents_per_frame=L_r, num_frames=W_r,
    ).eval()
    head = StreamPoseHead(
        embed_dim=C_r, hidden_dim=hidden, num_heads=8, num_layers=n_mlp + 1, num_frames=W_r,
    ).eval()
    m = _BareModel(bridge, head)
    report = warmstart_from_checkpoint(m, _RIG_INIT_CKPT)
    assert report["num_merged_loaded"] == n_layers and not report["skipped_shape_mismatch"]

    torch.manual_seed(5)
    x = torch.randn(1, 2, W_r, L_r, C_r)
    with torch.no_grad():
        out_s, snaps = m.bridge(x)
        out_b = _bidir_forward(m.bridge, x)
        h_s = m.pose_head(out_s, anchor_snaps=snaps)
        h_b = m.pose_head(out_b, anchor_snaps=out_b[:, :1])
    assert torch.allclose(out_s, out_b, atol=1e-5), \
        f"真实权重 snapshot N=2 与 bidir N=2 不一致, max={(out_s-out_b).abs().max()}"
    for k in h_s:
        assert torch.allclose(h_s[k], h_b[k], atol=1e-5), f"真实权重 pose head {k} 不一致"
