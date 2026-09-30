"""
stream_bridge.py -- StreamRig 的 CausalBridge 与 pose head。

结构（与论文 Sec. III-B 对应）：
  - 一个"组"(group) = 流里的一个时刻 = 一个 rig 快照（W 个相机，每相机 L 个 latent）。
  - group 0 = 锚 rig，其 cam0 = 参考相机 A0。
  - 组级因果：组 t 只能注意组 0..t，一次前向即可解出所有组相对 A0 的位姿，
    且每个位姿只用到因果历史 —— 训练整窗并行、推理逐 rig 增量两者等价。
  - 锚快照：每个查询组 t 配一份锚副本 A_t，
    A_t 只与 G_t 成对双向注意，G_t 另可见因果历史 G_1..G_t（论文 Eq. 3）。
  - pose head 预测"每个相机相对 A0 的位姿"：group0 内 cam1..W-1 = 已知标定(intra，辅助)，
    group t(t>=1) 的 cam0 = rig t 相对 rig 0 的里程计位姿 T_0->t(inter)，其余相机为辅助目标。
  - 不加时序位置编码（NoPE）：组 embedding 只区分锚/查询两种角色，窗口长度可超过训练所见。

示例用法：
    bridge = StreamRigBridge(embed_dim=768, num_merged_layers=8,
                             num_latents_per_frame=16, num_frames=5)
    all_tokens, anchor_snaps = bridge(latents)   # latents [B, N, W, L, C]
    head = StreamPoseHead(embed_dim=768, num_frames=5)
    out = head(all_tokens, anchor_snaps=anchor_snaps)   # dict: rot_intra/trans_intra/rot_inter/trans_inter
"""

import torch
import torch.nn as nn
from typing import Dict

from .modules import MergedSelfAttentionBlock


class StreamRigBridge(nn.Module):
    """
    CausalBridge：N 个 rig 组的 merged 自注意力 + 组级因果 / 锚快照可见性。

    数据流：
      1. 逐组加 frame_embed（逐相机，所有组共享）+ anchor_embed（仅 group0-cam0 = A0）
      2. 逐组加 group_embed（二元：group0=anchor 角色，其余=query 角色）
      3. 拼接成 token 序列，在 A0 的 token 上再注入 anchor_embed_merge
      4. 锚快照 mask 下的 merged 自注意力 x num_merged_layers
      5. out_norm，拆回 [B, N, W, L, C]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_merged_layers: int = 8,
        num_latents_per_frame: int = 16,
        num_frames: int = 5,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents_per_frame = num_latents_per_frame
        self.num_frames = num_frames
        self.num_heads = num_heads

        # 逐相机 embedding，所有组共享（论文 Eq. 2 的 e_cam）
        self.frame_embed = nn.Parameter(torch.randn(num_frames, 1, embed_dim) * 0.02)

        # 锚标记：仅 group0-cam0(A0) 使用（论文 Eq. 2 的 e_anchor）
        self.anchor_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # 组 embedding [2]：行0 = anchor 角色、行1 = 全部 query 共享（论文 Eq. 2 的 e_group），
        # 不编码组序，因此窗口长度可超过训练所见。
        self.group_embed = nn.Parameter(torch.randn(2, 1, embed_dim) * 0.02)

        # 合并后在 A0 的 token 上再注入一次锚向量
        self.anchor_embed_merge = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # merged 自注意力层
        self.merged_layers = nn.ModuleList([
            MergedSelfAttentionBlock(embed_dim, num_heads, ff_mult, dropout)
            for _ in range(num_merged_layers)
        ])

        self.out_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.frame_embed, std=0.02)
        nn.init.trunc_normal_(self.anchor_embed, std=0.02)
        nn.init.trunc_normal_(self.group_embed, std=0.02)
        nn.init.trunc_normal_(self.anchor_embed_merge, std=0.02)

    @staticmethod
    def _build_snapshot_mask(
        num_pairs: int, tokens_per_group: int, device: torch.device,
    ) -> torch.Tensor:
        """
        构造锚快照布局的布尔 mask [S, S]（True=屏蔽）。

        槽位布局 [A_1, G_1, A_2, G_2, ..., A_P, G_P]（P=num_pairs=N-1，t=k+1）：
          - A_t(槽 2k)   只可见 {A_t, G_t}       —— 与 G_t 成对双向；
          - G_t(槽 2k+1) 可见 {A_t} ∪ {G_1..G_t} —— 自己的成对锚 + 因果历史 + 自己。
        严格因果：任何 ≤t 的输出不依赖 >t 的组；A_s(s≠t) 与 G_t 互不可见。
        """
        sid = torch.arange(2 * num_pairs, device=device)
        pair = sid // 2                      # 槽位所属 pair 序号 k
        is_q = (sid % 2 == 1)                # 是否为查询组槽位
        q_pair, k_pair = pair.view(-1, 1), pair.view(1, -1)
        q_isq, k_isq = is_q.view(-1, 1), is_q.view(1, -1)
        # 锚副本行：只允许本 pair 的两个槽
        anchor_allow = (~q_isq) & (k_pair == q_pair)
        # 查询组行：允许本 pair 的锚副本，或 pair 序号 ≤ 自己的查询组（因果历史+自己）
        hist_ok = (k_pair <= q_pair)
        query_allow = q_isq & (((~k_isq) & (k_pair == q_pair)) | (k_isq & hist_ok))
        blocked = ~(anchor_allow | query_allow)                            # [2P, 2P]
        return blocked.repeat_interleave(tokens_per_group, dim=0) \
                      .repeat_interleave(tokens_per_group, dim=1)          # [S, S]

    def _run_merged_layers(
        self,
        merged: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """依次跑完 merged 层栈（nn.MultiheadAttention + 布尔 attn_mask，True=屏蔽）。"""
        for layer in self.merged_layers:
            merged = layer(merged, attn_mask=attn_mask)
        return merged

    # ------------------------------------------------------------------
    # 逐 rig 增量推理（KV-cache，见 stream_cache.py）用的逐组 embedding 入口。
    # 与整窗前向的语义一一对应，由 tests/test_cache_equiv.py 的整窗对拍测试保证。
    # ------------------------------------------------------------------

    def embed_group(self, latents_g: torch.Tensor, group_index: int) -> torch.Tensor:
        """给**单个组**的 latent 加上 forward 中的全部 embedding。

        与 `forward` 中「frame_embed → anchor_embed(仅组0-cam0) → group_embed」的加法顺序
        逐条对应，只是把"整块 [B,N,W,L,C] 一次加"改成"一组一次加"（逐元素加法，逐位一致）。

        Args:
            latents_g: [B, W, L, C] 该组 resampler 输出
            group_index: 该组在**当前段**内的组号（0 = 锚组，>=1 = 查询组）

        Returns:
            [B, W, L, C] 已加全部 embedding 的组特征
        """
        B, W, L, C = latents_g.shape
        x = latents_g + self.frame_embed[:W].view(1, W, 1, C)

        if group_index == 0:
            anchor = torch.zeros_like(x)
            anchor[:, 0] = self.anchor_embed.view(1, 1, C).expand(B, L, C)
            x = x + anchor

        g_embed = self.group_embed[0 if group_index == 0 else 1]
        return x + g_embed.view(1, 1, 1, C)

    def anchor_block(self, anchor_embedded: torch.Tensor) -> torch.Tensor:
        """把已加 embedding 的锚组摊平成 snapshot 布局的锚副本块 A_t [B, W*L, C]。

        与 `_forward_snapshot` 里构造 `anchor_blk` 的两行逐条对应（含在 cam0 的
        L 个 token 上注入 anchor_embed_merge）。流式下 A_t 恒定不变、每步重放，
        故只需在设锚（或重锚）时算一次。
        """
        B, W, L, C = anchor_embedded.shape
        blk = anchor_embedded.reshape(B, W * L, C)
        anchor_vec = self.anchor_embed_merge.expand(B, L, C)
        return torch.cat([blk[:, :L] + anchor_vec, blk[:, L:]], dim=1)

    def forward(self, latents: torch.Tensor):
        """
        Args:
            latents: [B, N, W, L, C]  —— N 个 rig 组，每组 W 相机 x L latent

        Returns:
            (all_tokens [B, N, W, L, C], anchor_snaps [B, N-1, W, L, C])，见 `_forward_snapshot`
        """
        B, N, W, L, C = latents.shape

        # === 逐相机 embedding + 锚（仅 group0-cam0） ===
        frame_pe = self.frame_embed[:W].view(1, 1, W, 1, C)
        x = latents + frame_pe  # [B, N, W, L, C]

        anchor = torch.zeros_like(x)
        anchor[:, 0, 0] = self.anchor_embed.view(1, 1, C).expand(B, L, C)
        x = x + anchor

        # === 组 embedding：组 0 用行0(anchor)，组 1..N-1 共享行1(query) ===
        g_embed = torch.cat(
            [self.group_embed[0:1], self.group_embed[1:2].expand(N - 1, -1, -1)], dim=0,
        ).view(1, N, 1, 1, C)
        x = x + g_embed

        return self._forward_snapshot(x)

    def _forward_snapshot(self, x: torch.Tensor):
        """
        锚快照前向。输入 x = 已加全部 embedding 的 [B, N, W, L, C]（组 0 为锚）。

        布局 [A_1, G_1, ..., A_{N-1}, G_{N-1}]：A_t 为锚组 embedded 特征的复制
        （含 anchor_embed_merge 注入），与 G_t 成对双向注意；G_t 另可见因果历史 G_1..t-1。
        N=2 时无任何屏蔽位，等价于一对 rig 的双向注意力（与成对重定位模型的读出方式相同）。

        Returns:
            all_tokens:   [B, N, W, L, C]（index0 = A_1 输出，供 intra；1..N-1 = G_t 输出）
            anchor_snaps: [B, N-1, W, L, C]（A_t 输出，pose head 的逐目标 ref）
        """
        B, N, W, L, C = x.shape
        assert N >= 2, "窗口至少含 2 个 rig（锚组 + 一个查询组）"
        T = W * L
        P = N - 1

        # 锚块 = 组 0 的 embedded 特征 + anchor_embed_merge
        anchor_blk = x[:, 0].reshape(B, T, C)
        anchor_vec = self.anchor_embed_merge.expand(B, L, C)
        anchor_blk = torch.cat([anchor_blk[:, :L] + anchor_vec, anchor_blk[:, L:]], dim=1)

        query_blk = x[:, 1:].reshape(B, P, T, C)
        anchor_rep = anchor_blk.unsqueeze(1).expand(B, P, T, C)
        merged = torch.stack([anchor_rep, query_blk], dim=2).reshape(B, 2 * P * T, C)

        mask = self._build_snapshot_mask(P, T, merged.device)
        if not mask.any():
            mask = None  # N==2：全可见，走无 mask 的 kernel
        merged = self._run_merged_layers(merged, attn_mask=mask)

        merged = self.out_norm(merged)

        out = merged.reshape(B, P, 2, W, L, C)
        anchor_snaps = out[:, :, 0]                                  # [B, P, W, L, C]
        all_tokens = torch.cat([anchor_snaps[:, 0:1], out[:, :, 1]], dim=1)  # [B, N, W, L, C]
        return all_tokens, anchor_snaps


class StreamPoseHead(nn.Module):
    """
    Pose head：预测每个相机相对 A0(group0-cam0) 的位姿。

    输出布局（跳过 A0）：
      - group 0 的 cam1..cam(W-1)：intra，共 W-1 个（= rig 标定，辅助目标）
      - group g(g=1..N-1) 的 cam0..cam(W-1)：inter，共 (N-1)*W 个
        其中 cam0 = rig g 相对 rig 0 的里程计位姿 T_0->g
    总位姿数 = (W-1) + (N-1)*W = N*W - 1。

    每个目标相机由两个可学习 query（旋转 / 平移）cross-attend 到
    [锚快照 A_t 的 cam0 token ⊕ 该相机 token]，再经两个三层 MLP 解码 6D 旋转与 3D 平移。
    frame_identity_embed [2W-1]：前 W-1 行给 intra 目标，后 W 行给每个 inter 组共享，
    因此窗口长度不受限。
    """

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 3,
        num_frames: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.rot_dim = 6   # 旋转固定用 6D 连续表示（旋转矩阵前两列）

        # 共享 pose queries（rotation + translation）
        self.pose_queries = nn.Parameter(torch.randn(2, embed_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.norm_after_attn = nn.LayerNorm(embed_dim)

        def _mlp(in_d, hid_d, n):
            layers = []
            for i in range(n):
                layers += [
                    nn.Linear(in_d if i == 0 else hid_d, hid_d),
                    nn.LayerNorm(hid_d),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            return nn.Sequential(*layers)

        self.rot_mlp = _mlp(embed_dim, hidden_dim, num_layers - 1)
        self.trans_mlp = _mlp(embed_dim, hidden_dim, num_layers - 1)
        self.rotation_head = nn.Linear(hidden_dim, self.rot_dim)
        self.translation_head = nn.Linear(hidden_dim, 3)

        # 位姿身份编码 [2W-1]：intra 块 + 一个被所有 inter 组共享的块
        self.frame_identity_embed = nn.Parameter(
            torch.randn(2 * num_frames - 1, 1, embed_dim) * 0.02,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pose_queries, std=0.02)
        nn.init.trunc_normal_(self.frame_identity_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.rotation_head.weight)
        nn.init.xavier_uniform_(self.translation_head.weight)

        # 6D 偏置初始化为单位旋转
        self.rotation_head.bias.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        nn.init.zeros_(self.translation_head.bias)

    def identity_embed(self, W: int, num_inter_groups: int) -> torch.Tensor:
        """intra 块 [0:W-1] + 共享 inter 块 [W-1:2W-1] 对 num_inter_groups 个组平铺。"""
        assert W == self.num_frames, f"相机数 W={W} 应等于 num_frames={self.num_frames}"
        inter_block = self.frame_identity_embed[W - 1:]
        return torch.cat(
            [self.frame_identity_embed[: W - 1],
             inter_block.repeat(num_inter_groups, 1, 1)], dim=0,
        )

    def forward(
        self, all_tokens: torch.Tensor, anchor_snaps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            all_tokens: [B, N, W, L, C]
            anchor_snaps: [B, N-1, W, L, C] 逐目标锚快照 A_t：
                inter 目标 g 的 ref 用 A_g 的 cam0，intra 目标的 ref 用 A_1 的 cam0。

        Returns:
            dict:
              rot_intra:   [B, W-1, rot_dim]        group0 cam1..W-1
              trans_intra: [B, W-1, 3]
              rot_inter:   [B, N-1, W, rot_dim]     group1..N-1 各 W 相机
              trans_inter: [B, N-1, W, 3]
              （里程计 T_0->t 取 rot_inter[:, t-1, 0] / trans_inter[:, t-1, 0]）
        """
        B, N, W, L, C = all_tokens.shape

        # 收集所有 target（跳过 A0）：group0 cam1..W-1，然后 group1..N-1 全部相机
        target_list, ref_list = [], []
        for i in range(1, W):
            target_list.append(all_tokens[:, 0, i])       # group0 非锚相机
            ref_list.append(anchor_snaps[:, 0, 0])        # intra 的 ref = A_1 cam0
        for g in range(1, N):
            for j in range(W):
                target_list.append(all_tokens[:, g, j])   # group g 全部相机
                ref_list.append(anchor_snaps[:, g - 1, 0])  # inter 的 ref = A_g cam0

        targets = torch.stack(target_list, dim=1)  # [B, P, L, C]
        num_poses = targets.shape[1]               # = N*W - 1

        fie = self.identity_embed(W, N - 1)
        targets = targets + fie.unsqueeze(0)

        targets_flat = targets.reshape(B * num_poses, L, C)

        # 逐目标 ref（A_g 的 cam0），与 target_list 一一对应
        ref_flat = torch.stack(ref_list, dim=1).reshape(B * num_poses, L, C)
        kv = torch.cat([ref_flat, targets_flat], dim=1)  # [B*P, 2L, C]
        kv_normed = self.norm_kv(kv)

        queries = self.pose_queries.unsqueeze(0).expand(B * num_poses, -1, -1)
        queries_normed = self.norm_query(queries)

        attended, _ = self.cross_attn(
            query=queries_normed, key=kv_normed, value=kv_normed, need_weights=False,
        )
        attended = self.norm_after_attn(queries + attended)

        rot_feat = self.rot_mlp(attended[:, 0])
        trans_feat = self.trans_mlp(attended[:, 1])
        rotations = self.rotation_head(rot_feat)
        translations = self.translation_head(trans_feat)

        rotations = rotations.reshape(B, num_poses, -1)
        translations = translations.reshape(B, num_poses, 3)

        # 拆分 intra / inter
        n_intra = W - 1
        rot_intra = rotations[:, :n_intra]                    # [B, W-1, rot_dim]
        trans_intra = translations[:, :n_intra]               # [B, W-1, 3]
        rot_inter = rotations[:, n_intra:].reshape(B, N - 1, W, self.rot_dim)
        trans_inter = translations[:, n_intra:].reshape(B, N - 1, W, 3)

        return {
            "rot_intra": rot_intra,
            "trans_intra": trans_intra,
            "rot_inter": rot_inter,
            "trans_inter": trans_inter,
        }
