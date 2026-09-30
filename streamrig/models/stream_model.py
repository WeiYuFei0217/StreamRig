"""
stream_model.py -- StreamRig 因果流式 rig 里程计主模型。

功能：读取 frozen MapAnything 逐 rig 联合感知（注入内参射线与外参）后的 info-sharing 特征缓存
      （由 data_preprocessing/ 预计算），经可训练的
      [Rig-Resampler (PerceiverResampler) -> CausalBridge (StreamRigBridge) -> StreamPoseHead]，
      一次前向输出窗口内 N 个 rig 相对锚 rig0 参考相机的因果里程计位姿。

示例用法：
    model = StreamRigModel(num_latents=16, num_frames_per_group=5)
    out = model(batch)   # batch["infoshare"]: [B, N, W, 256, 768]
    # out["traj_R"] [B,N,3,3], out["traj_t"] [B,N,3] 即锚系里程计（group0=identity）
"""

from typing import Dict

import torch
import torch.nn as nn

from .modules import PerceiverResampler, RotationUtils
from .stream_bridge import StreamRigBridge, StreamPoseHead


class StreamRigModel(nn.Module):
    """读取 frozen MapAnything info-sharing 特征的可训练因果流式后端。"""

    def __init__(
        self,
        embed_dim: int = 768,
        num_latents: int = 16,
        num_frames_per_group: int = 5,
        resampler_layers: int = 2,
        bridge_merged_layers: int = 8,
        pose_head_hidden_dim: int = 512,
        pose_head_layers: int = 3,
        num_heads: int = 8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_latents = num_latents
        self.num_frames_per_group = num_frames_per_group

        # ImageNet 均值/方差 buffer（发布权重的 state_dict 含这两个键）
        self.register_buffer("_img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1))
        self.register_buffer("_img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1))

        # ===== 可训练因果流式后端（输入为 embed_dim 维 info-sharing 特征）=====
        self.resampler = PerceiverResampler(
            embed_dim=embed_dim, num_latents=num_latents,
            num_heads=num_heads, num_layers=resampler_layers,
        )
        self.bridge = StreamRigBridge(
            embed_dim=embed_dim, num_heads=num_heads,
            num_merged_layers=bridge_merged_layers,
            num_latents_per_frame=num_latents, num_frames=num_frames_per_group,
        )
        self.pose_head = StreamPoseHead(
            embed_dim=embed_dim, hidden_dim=pose_head_hidden_dim,
            num_heads=num_heads, num_layers=pose_head_layers,
            num_frames=num_frames_per_group,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _run_backend(self, latents: torch.Tensor, B: int, N: int, W: int,
                     device: torch.device) -> Dict[str, torch.Tensor]:
        """可训练后端：bridge → pose_head → 6D 转矩阵 → 锚系轨迹。"""
        all_tokens, anchor_snaps = self.bridge(latents)
        head = self.pose_head(all_tokens, anchor_snaps=anchor_snaps)

        # 旋转 6D -> 矩阵
        rd = self.pose_head.rot_dim
        rm_intra = RotationUtils.rotation_6d_to_matrix(
            head["rot_intra"].reshape(-1, rd)
        ).reshape(B, W - 1, 3, 3)
        rm_inter = RotationUtils.rotation_6d_to_matrix(
            head["rot_inter"].reshape(-1, rd)
        ).reshape(B, N - 1, W, 3, 3)

        # 锚系里程计轨迹 T_0->t：group0=identity，group t(t>=1)= inter 的 cam0
        traj_R = torch.eye(3, device=device, dtype=rm_inter.dtype).view(1, 1, 3, 3).expand(B, N, 3, 3).clone()
        traj_t = torch.zeros(B, N, 3, device=device, dtype=rm_inter.dtype)
        traj_R[:, 1:] = rm_inter[:, :, 0]              # [B, N-1, 3, 3]
        traj_t[:, 1:] = head["trans_inter"][:, :, 0]   # [B, N-1, 3]

        return {
            # 结构化预测（供 loss）
            "rot_intra": head["rot_intra"],
            "trans_intra": head["trans_intra"],
            "rotation_matrices_intra": rm_intra,
            "rot_inter": head["rot_inter"],
            "trans_inter": head["trans_inter"],
            "rotation_matrices_inter": rm_inter,
            # 锚系里程计（供评估/轨迹累计）
            "traj_R": traj_R,          # [B, N, 3, 3]
            "traj_t": traj_t,          # [B, N, 3]
        }

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        infoshare = batch["infoshare"]  # [B, N, W, 256, C] 预计算 info-sharing 特征
        B, N, W = infoshare.shape[:3]
        device = infoshare.device

        # 逐组 resample
        latents_list = []
        for g in range(N):
            latents_g = self.resampler(infoshare[:, g])               # [B, W, L, C]
            latents_list.append(latents_g)
        latents = torch.stack(latents_list, dim=1)  # [B, N, W, L, C]

        return self._run_backend(latents, B, N, W, device)
