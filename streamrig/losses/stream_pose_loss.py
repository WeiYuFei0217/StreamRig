"""
stream_pose_loss.py -- StreamRig 因果流式位姿损失（论文 Eq. 4）。

监督目标：每个相机相对锚 A0(=rig0 的 cam0) 的位姿，一次前向全回传。
  - intra（group0 的 cam1..W-1）：GT = 固定 rig 标定 rig_calib[:,1:]，权重 intra_weight
  - inter（group g=1..N-1 的 cam0..W-1）：GT = T_rel_gt[:,g] @ rig_calib，权重 inter_weight
    其中 inter 的 cam0 = rig g 相对 rig0 的里程计位姿 T_0->g（0->1, 0->2, ..., 0->(N-1) 全部监督）。

旋转损失：sqrt_chordal = sqrt(chordal_frob) ∝ sin(θ/2)，小角度时近似线性（论文 Eq. 4 的 RMS chordal 项）。
平移损失：米制 L1；inter 平移按 GT 距离归一化 |t_pred-t_gt| / max(‖t_gt‖, eps)，使不同步距的量级可比。
inter 项逐步距 d 计算后对各 d 均匀平均；总损失按 loss_clamp 上限截断。

batch 需提供：
  rig_calib: [B, W, 4, 4]         固定 rig 标定（cam i 相对 cam0）
  T_rel_gt:  [B, N, 4, 4]         各 rig 的 cam0 相对 rig0 cam0（T_rel_gt[:,0]=I）

示例用法：
    criterion = StreamPoseLoss(intra_weight=0.5, inter_weight=1.0)
    losses = criterion(pred, batch)     # losses["loss"] 回传
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamPoseLoss(nn.Module):
    def __init__(
        self,
        rotation_weight: float = 5.0,
        translation_weight: float = 1.0,
        intra_weight: float = 0.5,
        inter_weight: float = 1.0,
        loss_clamp: float = 10.0,
        trans_norm_eps: float = 0.1,
    ):
        super().__init__()
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
        self.intra_weight = intra_weight
        self.inter_weight = inter_weight
        self.loss_clamp = loss_clamp
        self.trans_norm_eps = trans_norm_eps

    def _gt(self, batch):
        rig_calib = batch["rig_calib"]        # [B, W, 4, 4]
        T_rel_gt = batch["T_rel_gt"]          # [B, N, 4, 4]

        # intra：group0 的 cam1..W-1 相对 A0 = 标定本身
        R_gt_intra = rig_calib[:, 1:, :3, :3]                   # [B, W-1, 3, 3]
        t_gt_intra = rig_calib[:, 1:, :3, 3]                    # [B, W-1, 3]

        # inter：group g(g>=1) 的每相机相对 A0 = T_rel_gt[:,g] @ rig_calib
        T_inter = T_rel_gt[:, 1:].unsqueeze(2) @ rig_calib.unsqueeze(1)  # [B, N-1, W, 4, 4]
        R_gt_inter = T_inter[..., :3, :3]                       # [B, N-1, W, 3, 3]
        t_gt_inter = T_inter[..., :3, 3]                        # [B, N-1, W, 3]
        return R_gt_intra, t_gt_intra, R_gt_inter, t_gt_inter

    def forward(self, pred, batch):
        R_pred_intra = pred["rotation_matrices_intra"]  # [B, W-1, 3, 3]
        t_pred_intra = pred["trans_intra"]              # [B, W-1, 3]
        R_pred_inter = pred["rotation_matrices_inter"]  # [B, N-1, W, 3, 3]
        t_pred_inter = pred["trans_inter"]              # [B, N-1, W, 3]

        R_gt_intra, t_gt_intra, R_gt_inter, t_gt_inter = self._gt(batch)

        rot_intra = self._rotation_loss(R_pred_intra, R_gt_intra)
        trans_intra = F.l1_loss(t_pred_intra, t_gt_intra)
        rot_inter, trans_inter = self._inter_loss_per_d(
            R_pred_inter, R_gt_inter, t_pred_inter, t_gt_inter)

        intra_loss = self.intra_weight * (
            self.rotation_weight * rot_intra + self.translation_weight * trans_intra
        )
        inter_loss = self.inter_weight * (
            self.rotation_weight * rot_inter + self.translation_weight * trans_inter
        )
        total = intra_loss + inter_loss
        if self.loss_clamp > 0:
            total = torch.clamp(total, max=self.loss_clamp)

        return {
            "loss": total,
            "rot_loss_intra": rot_intra, "trans_loss_intra": trans_intra,
            "rot_loss_inter": rot_inter, "trans_loss_inter": trans_inter,
        }

    def _inter_loss_per_d(self, R_pred, R_gt, t_pred, t_gt):
        """逐步距路径：per-d 计算旋转/平移损失后按均匀权重 w_d = 1/n_d 加权求和。

        旋转：每个 d 切片各自过 _rotation_loss（逐 d 取 sqrt 再加权）。
        平移：L1 逐元素，按 GT 距离归一化（max(‖t_gt‖, eps) 防静止对除零）。
        """
        n_d = R_pred.shape[1]
        w = torch.ones(n_d, device=t_pred.device, dtype=t_pred.dtype)
        w = w / w.sum()

        rot_per_d = torch.stack(
            [self._rotation_loss(R_pred[:, d], R_gt[:, d]) for d in range(n_d)])  # [n_d]
        rot_inter = (w * rot_per_d).sum()

        err = (t_pred - t_gt).abs()                                   # [B, n_d, W, 3]
        denom = t_gt.norm(dim=-1, keepdim=True).clamp(min=self.trans_norm_eps)
        err = err / denom
        trans_per_d = err.mean(dim=(0, 2, 3))                          # [n_d]
        trans_inter = (w * trans_per_d).sum()
        return rot_inter, trans_inter

    def _rotation_loss(self, R_pred, R_gt):
        """sqrt_chordal：sqrt(MSE(R_pred^T R_gt, I)) ∝ sin(θ/2)，角度近线性。"""
        return torch.sqrt(self._chordal_frob(R_pred, R_gt) + 1e-8)

    @staticmethod
    def _chordal_frob(R_pred, R_gt):
        I = torch.eye(3, device=R_pred.device, dtype=R_pred.dtype)
        for _ in range(R_pred.dim() - 2):
            I = I.unsqueeze(0)
        I = I.expand_as(R_pred)
        R_diff = torch.matmul(R_pred.transpose(-1, -2), R_gt)
        return F.mse_loss(R_diff, I)
