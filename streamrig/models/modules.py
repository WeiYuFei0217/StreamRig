"""
modules.py -- StreamRig 的通用构件（rig 组 = group）。

功能：提供 StreamRig 可训练模块（Rig-Resampler / CausalBridge / pose head）复用的基础层。
Resampler 与注意力块的参数命名与成对重定位模型一致，便于直接加载其权重热启动：

1. ``PerceiverResampler``：把每个相机的 patch token 压成固定数量的 latent token；
2. ``MergedSelfAttentionBlock``：CausalBridge 的 merged 自注意力块；
3. ``RotationUtils``：6D 旋转表示 → 旋转矩阵。

示例用法::

    from streamrig.models.modules import PerceiverResampler, RotationUtils

    resampler = PerceiverResampler(embed_dim=768, num_latents=16, num_heads=8, num_layers=2)
    latents = resampler(patch_tokens)           # [B, W, P, C] -> [B, W, num_latents, C]
    R = RotationUtils.rotation_6d_to_matrix(pred_6d)   # [..., 6] -> [..., 3, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    """Perceiver Resampler：以可学习 latent 为 query，对 patch token 做交叉注意力，
    把变长 patch token 压成固定数量的 latent token。"""

    def __init__(
        self,
        embed_dim: int = 768,
        num_latents: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: token 维度
            num_latents: 输出 latent 数
            num_heads: 注意力头数
            num_layers: 交叉注意力层数
            ff_mult: FFN 隐层倍数
            dropout: dropout 概率
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_heads = num_heads

        # 可学习 latent query
        self.latents = nn.Parameter(torch.randn(num_latents, embed_dim) * 0.02)

        # 交叉注意力层
        self.layers = nn.ModuleList([
            PerceiverCrossAttentionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 输出 LayerNorm
        self.out_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.latents, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: patch token，[B, N_patches, C]，或 [B, N_frames, N_patches, C]（逐相机分别压缩）

        Returns:
            [B, num_latents, C]，或 [B, N_frames, num_latents, C]
        """
        N_frames = None
        if x.dim() == 4:
            B_orig, N_frames, N_patches, C = x.shape
            x = x.reshape(B_orig * N_frames, N_patches, C)
        B = x.shape[0]

        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for layer in self.layers:
            latents = layer(latents, x)
        latents = self.out_norm(latents)

        if N_frames is not None:
            latents = latents.reshape(B_orig, N_frames, self.num_latents, -1)
        return latents


class PerceiverCrossAttentionBlock(nn.Module):
    """Perceiver 交叉注意力块：pre-norm 交叉注意力 + FFN（均带残差）。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_context = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        ff_dim = int(embed_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            latents: query token [B, num_latents, C]
            context: key/value token（图像 patch）[B, N, C]

        Returns:
            更新后的 latent [B, num_latents, C]
        """
        # 交叉注意力
        latents_normed = self.norm1(latents)
        context_normed = self.norm_context(context)

        attn_out, _ = self.cross_attn(
            query=latents_normed,
            key=context_normed,
            value=context_normed,
        )
        latents = latents + attn_out

        # FFN
        latents = latents + self.ffn(self.norm2(latents))

        return latents


class MergedSelfAttentionBlock(nn.Module):
    """CausalBridge 的 merged 自注意力块：pre-norm 自注意力 + FFN（均带残差）。"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 自注意力
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = int(embed_dim * ff_mult)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: token [B, N, C]
            attn_mask: 可选布尔掩码 [N, N]（True = 屏蔽，即该 key 对该 query 不可见），
                传给 nn.MultiheadAttention，用于注入锚快照可见性；None = 全连接自注意力。

        Returns:
            更新后的 token [B, N, C]
        """
        # 自注意力
        x_normed = self.norm1(x)
        attn_out, _ = self.self_attn(
            x_normed, x_normed, x_normed, attn_mask=attn_mask, need_weights=False,
        )
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.norm2(x))

        return x


class RotationUtils:
    """旋转表示转换工具（6D → 旋转矩阵）。"""

    @staticmethod
    def rotation_6d_to_matrix(rot_6d: torch.Tensor) -> torch.Tensor:
        """
        6D 旋转表示 → 旋转矩阵。

        6D 表示 = 旋转矩阵的前两列；第三列由 Gram-Schmidt 正交化后叉乘得到。

        Args:
            rot_6d: [B, 6] or [6]

        Returns:
            R: [B, 3, 3] or [3, 3]
        """
        squeeze = False
        if rot_6d.dim() == 1:
            rot_6d = rot_6d.unsqueeze(0)
            squeeze = True

        a1 = rot_6d[:, :3]  # [B, 3]
        a2 = rot_6d[:, 3:]  # [B, 3]

        # 用 F.normalize 做 Gram-Schmidt 正交化；eps 用于数值稳定。
        b1 = F.normalize(a1, dim=-1, eps=1e-6)

        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1, eps=1e-6)

        b3 = torch.cross(b1, b2, dim=-1)

        R = torch.stack([b1, b2, b3], dim=-1)  # [B, 3, 3]

        if squeeze:
            R = R.squeeze(0)

        return R
