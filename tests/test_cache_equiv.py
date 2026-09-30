"""
test_cache_equiv.py -- 逐 rig KV-cache 增量推理与整窗因果前向的等价性测试（CPU，pytest）。

测试目标：
  1. **增量 ⇔ 整窗**：同权重同输入，逐组位姿输出与 `bridge.forward`（mha + snapshot mask）
     的整窗输出一致 —— 这条顺带护住 `embed_group`/`anchor_block` 与 forward 内联
     embedding 代码的一致性（增量侧走逐组方法、整窗侧走原路径，漂移会立刻被抓到）。
  2. **cache 结构**：每步只定格 G_t 的 W·L 个 token，锚副本 A_t 的 K/V 不占 cache。
  3. **重锚链式复合正确**：重锚后的段 = 以新锚重跑的整窗前向，复合后逐组一致
     （即评测脚本"每个重锚块一次整窗前向"与部署时逐 rig 增量推理等价）。

只用 bridge/pose_head（不碰 MapAnything backbone），几秒跑完。

用法：
    python -m pytest tests/test_cache_equiv.py -q
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamrig.models.modules import RotationUtils
from streamrig.models.stream_bridge import StreamRigBridge, StreamPoseHead
from streamrig.models.stream_cache import StreamCacheRunner

DIM, HEADS = 64, 4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _build(W, L, layers=2):
    """发布结构的 bridge+head（小尺寸）。"""
    torch.manual_seed(7)
    bridge = StreamRigBridge(
        embed_dim=DIM, num_heads=HEADS, num_merged_layers=layers,
        num_latents_per_frame=L, num_frames=W,
    ).eval()
    head = StreamPoseHead(embed_dim=DIM, num_heads=HEADS, num_frames=W).eval()
    return bridge, head


@torch.no_grad()
def _window(bridge, head, latents):
    """整窗一次前向（bridge.forward + pose head），作为增量 cache 的参考。"""
    all_tokens, anchor_snaps = bridge(latents)
    return head(all_tokens, anchor_snaps=anchor_snaps)


@torch.no_grad()
def _run_incremental(bridge, head, latents):
    """逐组增量推理，返回 (runner, 每组 dict 列表)（列表 index 0 对应查询组 1）。"""
    runner = StreamCacheRunner(bridge, head)
    runner.set_anchor(latents[:, 0])
    return runner, [runner.step(latents[:, t]) for t in range(1, latents.shape[1])]


def _cmp_poses(window, incr, tol, tag):
    """整窗输出 vs 增量逐组输出的最大绝对偏差断言。"""
    worst = 0.0
    for t, got in enumerate(incr, start=1):
        for key in ("rot_inter", "trans_inter"):
            ref = window[key][:, t - 1]
            worst = max(worst, (ref - got[key]).abs().max().item())
    for key in ("rot_intra", "trans_intra"):
        worst = max(worst, (window[key] - incr[0][key]).abs().max().item())
    assert worst < tol, f"[{tag}] 增量 cache 与整窗前向不一致：max_abs_err={worst:.3e}"
    return worst


def _to_T(win, idx):
    """从整窗输出取第 idx 个查询组的 4x4 位姿（cam0 = 里程计）。"""
    R = RotationUtils.rotation_6d_to_matrix(win["rot_inter"][:, idx, 0])
    T = torch.eye(4, dtype=R.dtype).unsqueeze(0).repeat(R.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = win["trans_inter"][:, idx, 0]
    return T


# ---------------------------------------------------------------------------
# 1. 增量 ⇔ 整窗 bridge.forward
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N,layers", [(2, 2), (3, 2), (10, 2), (12, 3)])
def test_incremental_matches_bridge_forward(N, layers):
    """逐组 KV-cache 推理必须复现整窗前向的每一组位姿。"""
    torch.manual_seed(12)
    B, W, L = 1, 2, 6
    bridge, head = _build(W, L, layers=layers)
    latents = torch.randn(B, N, W, L, DIM)

    win = _window(bridge, head, latents)
    _, incr = _run_incremental(bridge, head, latents)
    err = _cmp_poses(win, incr, 1e-5, f"N={N} layers={layers}")
    print(f"[cache==bridge.forward N={N}] max_abs_err={err:.3e}")


# ---------------------------------------------------------------------------
# 2. cache 结构：每步只定格 G_t，锚副本不占 cache
# ---------------------------------------------------------------------------
def test_cache_holds_only_query_groups():
    torch.manual_seed(14)
    B, N, W, L = 1, 8, 2, 6
    bridge, head = _build(W, L, layers=2)
    latents = torch.randn(B, N, W, L, DIM)
    runner = StreamCacheRunner(bridge, head, capacity_groups=2)   # 小容量，覆盖扩容路径
    runner.set_anchor(latents[:, 0])
    for t in range(1, N):
        runner.step(latents[:, t])
        assert runner.cache.hist_len == t * W * L, (
            f"t={t}: cache token 数 {runner.cache.hist_len} != 期望 {t * W * L}"
        )
        assert runner.cache.n_groups == t, "cache 里的组数与已喂入组数不符"


# ---------------------------------------------------------------------------
# 3. 重锚：清 cache + 新锚 + 链式复合
# ---------------------------------------------------------------------------
def test_reanchor_matches_segment_windows():
    """在组 r 重锚后，全局位姿 = 段1整窗 ∘ 段2整窗（链式复合）。"""
    torch.manual_seed(16)
    B, N, W, L, r = 1, 13, 2, 6, 6
    bridge, head = _build(W, L, layers=2)
    latents = torch.randn(B, N, W, L, DIM)

    runner = StreamCacheRunner(bridge, head)
    runner.set_anchor(latents[:, 0])
    T_glob = [torch.eye(4).unsqueeze(0)]
    anchor_glob = torch.eye(4).unsqueeze(0)
    for t in range(1, N):
        out = runner.step(latents[:, t])
        T_glob.append(anchor_glob @ out["T"])
        if t == r:
            runner.reanchor()
            assert runner.cache.n_groups == 0, "重锚后 cache 未清空"
            anchor_glob = T_glob[t]

    seg1 = _window(bridge, head, latents[:, : r + 1])
    seg2 = _window(bridge, head, latents[:, r:])
    for t in range(1, N):
        if t <= r:
            ref_T = _to_T(seg1, t - 1)
        else:
            ref_T = _to_T(seg1, r - 1) @ _to_T(seg2, t - r - 1)
        err = (ref_T - T_glob[t]).abs().max().item()
        assert err < 1e-5, f"组 {t} 的链式复合位姿偏差 {err:.3e}"
