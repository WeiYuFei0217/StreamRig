"""
test_extra_layers.py -- 可扩展 bridge merged 层数 + 新层残差恒等零初始化 回归测试。

初始化源未覆盖的 bridge merged 层做残差恒等零初始化
(注意力 out_proj 与 FFN 输出层置零 → 新层 = 恒等映射), 热启动起点保持不变。

覆盖:
  1. num_merged_layers=2 时零初始化函数不动任何东西(无 index>=2 的层),返回 0;
  2. 核心等价性:4 层 bridge(层 0/1 载 2 层权重 + 层 2/3 恒等零初始化)前向
     == 2 层 bridge 前向(同权重)—— 证明新层恒等,热启动起点不变;
  3. 零初始化后新层的注意力/FFN 输出投影确为零、已加载层不动;
  4. 因果性:4 层 bridge 扰动未来组不改历史组输出;
  5. 4/6/8 层都能构建 + 前向 + 反向(梯度有限)。

示例用法:
    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_extra_layers.py -q
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamrig.models.stream_bridge import StreamRigBridge
from streamrig.utils.warmstart import zero_init_extra_merged_layers

DIM, HEADS, W, L = 64, 4, 3, 8


def _mk_bridge(num_merged_layers=2, seed=7, **kw):
    args = dict(embed_dim=DIM, num_heads=HEADS, num_merged_layers=num_merged_layers,
                num_latents_per_frame=L, num_frames=W)
    args.update(kw)
    torch.manual_seed(seed)
    return StreamRigBridge(**args).eval()


def _tokens(bridge, x):
    """bridge 输出 (all_tokens, anchor_snaps)，取 all_tokens。"""
    return bridge(x)[0]


def _latents(N, seed=0):
    torch.manual_seed(seed)
    return torch.randn(2, N, W, L, DIM)


def _wrap(bridge):
    """把 bridge 包成带 .bridge 属性的对象,供 zero_init_extra_merged_layers 使用。"""
    return types.SimpleNamespace(bridge=bridge)


# ------------------------------------------------- 1. 2 层时零初始化不动任何东西
def test_zero_init_noop_when_2layers():
    b = _mk_bridge(num_merged_layers=2)
    before = [p.clone() for p in b.parameters()]
    n = zero_init_extra_merged_layers(_wrap(b), num_loaded=2)
    assert n == 0, f"2 层不应有新层被零初始化,实际 {n}"
    after = list(b.parameters())
    assert all(torch.equal(x, y) for x, y in zip(before, after)), "2 层参数不应被改动"


# ------------------------------------------------- 2. 核心:4 层恒等零初始化 == 2 层前向
def test_4layer_zero_init_equals_2layer():
    b2 = _mk_bridge(num_merged_layers=2, seed=7)
    b4 = _mk_bridge(num_merged_layers=4, seed=9)  # 不同 seed,确保靠 load 对齐而非巧合

    # 把 b2 的全部共享参数(embeddings/out_norm/merged_layers.0/1)拷进 b4
    sd2 = b2.state_dict()
    sd4 = b4.state_dict()
    for k in sd2:
        assert k in sd4, f"键 {k} 不在 4 层 bridge 中"
        sd4[k] = sd2[k].clone()
    b4.load_state_dict(sd4)  # 层 2/3 仍是 b4 的随机初始化

    # 对层 2/3 做残差恒等零初始化
    n = zero_init_extra_merged_layers(_wrap(b4), num_loaded=2)
    assert n == 2

    x = _latents(6, seed=3)
    with torch.no_grad():
        out2 = _tokens(b2, x)
        out4 = _tokens(b4, x)
    assert torch.allclose(out2, out4, atol=1e-5), \
        f"恒等零初始化后 4 层前向应 == 2 层,max diff={(out2 - out4).abs().max().item():.2e}"


# ------------------------------------------------- 3. zero 后新层输出投影确为零
def test_zero_init_makes_out_proj_zero():
    b4 = _mk_bridge(num_merged_layers=4, seed=5)
    zero_init_extra_merged_layers(_wrap(b4), num_loaded=2)
    for i in (2, 3):
        blk = b4.merged_layers[i]
        assert blk.self_attn.out_proj.weight.abs().sum().item() == 0.0
        assert blk.ffn[3].weight.abs().sum().item() == 0.0
    # 层 0/1 不应被动
    for i in (0, 1):
        assert b4.merged_layers[i].self_attn.out_proj.weight.abs().sum().item() > 0


# ------------------------------------------------- 4. 因果性:4 层扰动未来组不改历史输出
def test_causality_preserved_4layers():
    b4 = _mk_bridge(num_merged_layers=4, seed=5)
    zero_init_extra_merged_layers(_wrap(b4), num_loaded=2)
    N = 6
    x = _latents(N, seed=1)
    x2 = x.clone()
    # 扰动最后一组(未来)
    x2[:, -1] += 3.0 * torch.randn_like(x2[:, -1])
    with torch.no_grad():
        o1 = _tokens(b4, x)
        o2 = _tokens(b4, x2)
    # 组 0..N-2 的输出应不受最后一组扰动影响
    diff = (o1[:, :-1] - o2[:, :-1]).abs().max().item()
    assert diff < 1e-5, f"因果性被破坏:历史组输出随未来组变化,max diff={diff:.2e}"


# ------------------------------------------------- 5. 4/6/8 层构建 + 前向 + 反向
def test_468_layers_build_forward_backward():
    for nl in (4, 6, 8):
        b = _mk_bridge(num_merged_layers=nl, seed=13)
        zero_init_extra_merged_layers(_wrap(b), num_loaded=2)
        b.train()
        x = _latents(5, seed=2).requires_grad_(True)
        out = _tokens(b, x)
        assert out.shape == x.shape, f"{nl} 层前向形状不符"
        loss = out.pow(2).mean()
        loss.backward()
        # 至少有一个 merged 层参数拿到有限梯度
        grads = [p.grad for p in b.merged_layers.parameters() if p.grad is not None]
        assert grads, f"{nl} 层反向无梯度"
        assert all(torch.isfinite(g).all() for g in grads), f"{nl} 层梯度含 NaN/Inf"
