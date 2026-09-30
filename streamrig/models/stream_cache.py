"""
stream_cache.py -- 锚快照布局的 **KV-cache 逐 rig 增量推理引擎**（eval only, no_grad）。

功能
----
把训练时"整窗一次前向"的计算改写成流式部署形态：rig 逐个到来，每步只算当前
pair [A_t, G_t]，历史组的 K/V 定格在 cache 里复用。两条规则：

  1. **锚每步重放、永不进 cache**：A_t 是锚组的副本，内容恒定（设锚时算一次），
     每步与 G_t 拼成 pair 参与注意力；A_t 的 K/V 用完即弃 —— snapshot 的 mask 规则
     保证任何后续 query 都不会引用旧 pair 的锚副本，所以丢掉它不损失任何东西。
  2. **G_t 的每层 K/V 存入 cache**：G_t 在第 l 层的输入只依赖 pair ≤ t（严格因果），
     故其 K/V 一经算出即定格，与整窗前向里同一位置的 K/V 同义。

状态上界由重锚块长 N 决定（每次重锚清空 cache）。

cache 布局（每层一条连续 buffer，避免每步 concat 整个历史 = O(t) 拷贝）：

    [ 历史区：已定格的 G_1..G_{t-1}，每组 W·L 个 token | pair 暂存区：本步的 A_t(W·L) + G_t(W·L) ]

数学等价性
----------
每步的注意力等价于"整窗前向里该 pair 那几行"：A_t 的 query 看本 pair 全部 2·W·L 个
token；G_t 的 query 看 [历史 ∪ A_t ∪ G_t]（见 `tests/test_cache_equiv.py`）。

示例用法（固定块长 N 的重锚协议）
--------
    from streamrig.models.stream_cache import StreamCacheRunner

    runner = StreamCacheRunner(model.bridge, model.pose_head)
    runner.set_anchor(latents_0)                # [B, W, L, C] 锚组 resampler 输出
    for t in range(1, n_groups):
        out = runner.step(latents_t)            # dict: T(4x4 相对当前锚) / rot_inter / ...
        if t % (N - 1) == 0:                    # 块末 rig 成为下一块的锚
            runner.reanchor()
"""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .modules import RotationUtils


class StreamKVCache:
    """snapshot 流式推理的逐层 KV cache（单条连续 buffer + 尾部 pair 暂存区）。

    这是全流程里唯一**有意可变**的对象（就地写 buffer 是 KV-cache 的本义：
    已存的值不改，每步只追加，不重算历史）。

    Args:
        num_layers: merged 层数
        num_frames: W（每组相机数）
        latents_per_frame: L（每相机 latent 数）
        capacity_groups: buffer 初始容量（组），不够会自动翻倍
    """

    def __init__(
        self,
        num_layers: int,
        num_frames: int,
        latents_per_frame: int,
        capacity_groups: int = 16,
    ):
        self.num_layers = int(num_layers)
        self.W = int(num_frames)
        self.L = int(latents_per_frame)
        self.T = self.W * self.L                       # 每组 token 数

        self._cap_init = max(2, int(capacity_groups)) * self.T
        self.k_buf: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.v_buf: List[Optional[torch.Tensor]] = [None] * self.num_layers
        self.clear()

    # -- 状态 -----------------------------------------------------------
    def clear(self):
        """清空 cache（重锚时用）。保留已分配的 buffer 以免反复申请显存。"""
        self.hist_len = 0       # 已定格的历史 token 数（不含本步 pair 暂存区）
        self.n_groups = 0       # 已定格的历史组数
        self._pair_open = False

    # -- 每步流程 -------------------------------------------------------
    def begin_pair(self, k_like: torch.Tensor):
        """开始新的一步：确保 buffer 能放下 [历史 + 本步 pair 的 2·T 个 token]。

        k_like: 形如 [B, H, *, Dh] 的张量，用来定 dtype/device/头数（首步分配用）。
        """
        need = self.hist_len + 2 * self.T
        B, H, _, Dh = k_like.shape
        for i in range(self.num_layers):
            buf = self.k_buf[i]
            if buf is not None and buf.shape[2] >= need and buf.dtype == k_like.dtype \
                    and buf.device == k_like.device \
                    and buf.shape[0] == B and buf.shape[1] == H and buf.shape[3] == Dh:
                continue
            cap = max(need, self._cap_init, 0 if buf is None else 2 * buf.shape[2])
            for name in ("k_buf", "v_buf"):
                prev = getattr(self, name)[i]
                new = torch.empty(B, H, cap, Dh, dtype=k_like.dtype, device=k_like.device)
                if prev is not None and self.hist_len > 0:
                    new[:, :, : self.hist_len] = prev[:, :, : self.hist_len]
                getattr(self, name)[i] = new
        self._pair_open = True

    def write_layer(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """写入本步第 layer 层的 pair K/V（[B, H, 2T, Dh]），返回四个**视图**：

            (K_all, V_all)   = [历史 ∪ A_t ∪ G_t]，供 G_t 的 query 用（全可见）
            (K_pair, V_pair) = [A_t ∪ G_t]，供 A_t 的 query 用（只见本 pair）
        """
        assert self._pair_open, "write_layer 前必须先 begin_pair"
        pos, end = self.hist_len, self.hist_len + 2 * self.T
        self.k_buf[layer][:, :, pos:end] = k
        self.v_buf[layer][:, :, pos:end] = v
        return (self.k_buf[layer][:, :, :end], self.v_buf[layer][:, :, :end],
                self.k_buf[layer][:, :, pos:end], self.v_buf[layer][:, :, pos:end])

    def commit(self):
        """本步算完：G_t 的 K/V 转正进历史区，A_t 的丢弃。"""
        assert self._pair_open, "commit 前必须先 begin_pair"
        T, pos = self.T, self.hist_len
        for i in range(self.num_layers):
            # G_t 覆盖 A_t 的槽位 —— 锚副本的 K/V 用完即弃（源/目不重叠，可直接 copy）
            self.k_buf[i][:, :, pos: pos + T] = self.k_buf[i][:, :, pos + T: pos + 2 * T]
            self.v_buf[i][:, :, pos: pos + T] = self.v_buf[i][:, :, pos + T: pos + 2 * T]
        self.hist_len += T
        self.n_groups += 1
        self._pair_open = False


class StreamCacheRunner:
    """用 KV-cache 做逐组增量推理的流式 runner（eval only）。

    只依赖 bridge/pose_head 的**权重与子模块**，不改动它们的任何既有前向路径；
    embedding 注入复用 `StreamRigBridge.embed_group` / `anchor_block`。

    Args:
        bridge: StreamRigBridge
        pose_head: StreamPoseHead
        capacity_groups: cache buffer 初始容量（组）
    """

    def __init__(self, bridge, pose_head, capacity_groups: int = 16):
        self.bridge = bridge
        self.head = pose_head
        self.W = bridge.num_frames
        self.L = bridge.num_latents_per_frame
        self.T = self.W * self.L

        self.cache = StreamKVCache(
            num_layers=len(bridge.merged_layers), num_frames=self.W,
            latents_per_frame=self.L, capacity_groups=capacity_groups,
        )
        self.anchor_blk = None           # [B, T, C]，设锚时算一次、每步重放
        self.pair_idx = 0                # 当前段内已处理的查询组数
        self._last_latents = None        # 最近一步的组 latent（重锚时成为新锚）

    # -- 锚 -------------------------------------------------------------
    @torch.no_grad()
    def set_anchor(self, latents_anchor: torch.Tensor):
        """设（或重设）锚组，并清空 cache。latents_anchor: [B, W, L, C]。"""
        self.anchor_blk = self.bridge.anchor_block(
            self.bridge.embed_group(latents_anchor, 0),
        )
        self.cache.clear()
        self.pair_idx = 0

    @torch.no_grad()
    def reanchor(self, latents_anchor: Optional[torch.Tensor] = None):
        """重锚：把锚迁到最近一步的组（或显式给定的 latent），并清空 cache。

        块末 rig 成为下一块的锚，窗口从这里重新开始。
        """
        new_anchor = self._last_latents if latents_anchor is None else latents_anchor
        assert new_anchor is not None, "reanchor 前至少要跑过一步 step"
        self.set_anchor(new_anchor)

    # -- 每步 -----------------------------------------------------------
    @torch.no_grad()
    def step(self, latents_g: torch.Tensor) -> Dict[str, torch.Tensor]:
        """喂入一个新组，返回它相对**当前锚**的位姿。latents_g: [B, W, L, C]。"""
        assert self.anchor_blk is not None, "step 前必须先 set_anchor"
        B, W, L, C = latents_g.shape
        T = self.T
        self.pair_idx += 1

        g_emb = self.bridge.embed_group(latents_g, self.pair_idx)
        x = torch.cat([self.anchor_blk, g_emb.reshape(B, T, C)], dim=1)   # [B, 2T, C]

        for li, layer in enumerate(self.bridge.merged_layers):
            attn = layer.self_attn
            H = attn.num_heads
            Dh = C // H
            xn = layer.norm1(x)
            qkv = F.linear(xn, attn.in_proj_weight, attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.view(B, 2 * T, H, Dh).transpose(1, 2)
            k = k.view(B, 2 * T, H, Dh).transpose(1, 2)
            v = v.view(B, 2 * T, H, Dh).transpose(1, 2)
            if li == 0:
                self.cache.begin_pair(k)
            K_all, V_all, K_pair, V_pair = self.cache.write_layer(li, k, v)
            # 锚副本行：只见本 pair；查询组行：本 pair + 全部 cache 历史
            out_a = F.scaled_dot_product_attention(q[:, :, :T], K_pair, V_pair)
            out_g = F.scaled_dot_product_attention(q[:, :, T:], K_all, V_all)
            out = torch.cat([out_a, out_g], dim=2).transpose(1, 2).reshape(B, 2 * T, C)
            x = x + attn.out_proj(out)
            x = x + layer.ffn(layer.norm2(x))

        self.cache.commit()

        merged = self.bridge.out_norm(x)
        anchor_snap = merged[:, :T].reshape(B, W, L, C)
        group_tok = merged[:, T:].reshape(B, W, L, C)
        self._last_latents = latents_g
        return pose_for_group(self.head, anchor_snap, group_tok, self.pair_idx)


@torch.no_grad()
def pose_for_group(head, anchor_snap: torch.Tensor, group_tokens: torch.Tensor,
                   group_index: int) -> Dict[str, torch.Tensor]:
    """StreamPoseHead 的**逐组**读出（与整窗 `StreamPoseHead.forward` 数学等价）。

    整窗版一次把 N·W-1 个 pose query 堆进 batch 维；流式下只算当前组的 W 个
    （外加 group_index==1 时那一次 intra，对应整窗布局里最前面的 W-1 个）。
    cross-attn 逐 pose 独立 ⇒ 拆开算与堆一起算等价。

    Args:
        anchor_snap:  [B, W, L, C] 本步锚副本 A_t 的输出
        group_tokens: [B, W, L, C] 本步查询组 G_t 的输出
        group_index:  该组在当前段内的组号（≥1）

    Returns:
        dict: T [B,4,4]（该组相对锚的位姿）、rot_inter/trans_inter [B,W,*]，
              group_index==1 时另含 rot_intra/trans_intra
    """
    B, W, L, C = group_tokens.shape
    include_intra = (group_index == 1)

    targets, refs = [], []
    if include_intra:                       # 整窗布局里 intra 块在最前面，ref = A_1 的 cam0
        for i in range(1, W):
            targets.append(anchor_snap[:, i])
            refs.append(anchor_snap[:, 0])
    for j in range(W):                      # 本组 W 个相机，ref = A_t 的 cam0
        targets.append(group_tokens[:, j])
        refs.append(anchor_snap[:, 0])

    fie = head.identity_embed(W, 1)          # [intra(W-1); inter(W)]
    if not include_intra:
        fie = fie[W - 1:]
    tgt = torch.stack(targets, dim=1) + fie.unsqueeze(0)        # [B, P, L, C]
    P = tgt.shape[1]
    tgt_flat = tgt.reshape(B * P, L, C)

    kv = torch.cat([torch.stack(refs, dim=1).reshape(B * P, L, C), tgt_flat], dim=1)
    kv_normed = head.norm_kv(kv)
    queries = head.pose_queries.unsqueeze(0).expand(B * P, -1, -1)
    attended, _ = head.cross_attn(
        query=head.norm_query(queries), key=kv_normed, value=kv_normed, need_weights=False,
    )
    attended = head.norm_after_attn(queries + attended)

    rot = head.rotation_head(head.rot_mlp(attended[:, 0]))
    trans = head.translation_head(head.trans_mlp(attended[:, 1]))
    rot = rot.reshape(B, P, -1)
    trans = trans.reshape(B, P, 3)

    n_intra = (W - 1) if include_intra else 0
    out = {
        "rot_inter": rot[:, n_intra:],                      # [B, W, rot_dim]
        "trans_inter": trans[:, n_intra:],                  # [B, W, 3]
    }
    if include_intra:
        out["rot_intra"] = rot[:, :n_intra]
        out["trans_intra"] = trans[:, :n_intra]

    R = RotationUtils.rotation_6d_to_matrix(out["rot_inter"][:, 0])
    T4 = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T4[:, :3, :3] = R
    T4[:, :3, 3] = out["trans_inter"][:, 0]
    out["T"] = T4                                           # [B, 4, 4] 该组相对锚
    return out

