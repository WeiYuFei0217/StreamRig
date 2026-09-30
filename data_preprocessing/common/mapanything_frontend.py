"""
mapanything_frontend.py -- 预处理共用的冻结 MapAnything 前端：离线加载 + 单 rig info_sharing 感知。

功能:
  1. pin_cached_dinov2_main_branch: 构建 MapAnything 期间把 DINOv2 的 torch-hub 仓库固定为 main 分支。
  2. load_frozen_mapanything: 加载 frozen MapAnything(eval, 参数不求梯度)。
  3. perceive_rig_from_encodings: 给定一个 rig 的 fp32 DINOv2 编码与 rig 内外参, 注入射线方向 +
     相机位姿 + 公制尺度标志, 跑 info_sharing, 返回 [B, K, 256, C] 的 fp32 特征
     (KITTI-360 step2 缓存即由此生成)。

依赖: torch, mapanything 及其 uniception 依赖(在函数内按需 import, 不影响本包其他模块)。

示例用法:
    from data_preprocessing.common.mapanything_frontend import (
        load_frozen_mapanything, perceive_rig_from_encodings,
    )
    backbone = load_frozen_mapanything("/path/to/map-anything-model", "cuda:0")
    # enc: [1, K, 1024, 16, 16] fp32 DINOv2 编码; extr: [1, K, 4, 4]; intr: [1, K, 3, 3]
    feats = perceive_rig_from_encodings(backbone, extr, intr, enc)[0]   # [K, 256, 768]
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import torch


@contextmanager
def pin_cached_dinov2_main_branch():
    """构建 MapAnything 期间把 DINOv2 的 torch-hub 仓库固定为 main 分支（本地已有缓存时可离线加载）。"""
    original_load = torch.hub.load

    def load_with_pinned_ref(repo_or_dir, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            repo_or_dir = "facebookresearch/dinov2:main"
        return original_load(repo_or_dir, *args, **kwargs)

    torch.hub.load = load_with_pinned_ref
    try:
        yield
    finally:
        torch.hub.load = original_load


def load_frozen_mapanything(model_path: str, device: str):
    """加载 frozen MapAnything(HuggingFace from_pretrained 格式), 返回 eval 模式的模型。"""
    from mapanything.models import MapAnything

    model_path_obj = Path(model_path).expanduser()
    with pin_cached_dinov2_main_branch():
        if model_path_obj.exists():
            backbone = MapAnything.from_pretrained(str(model_path_obj))
        else:
            backbone = MapAnything.from_pretrained(model_path)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone.to(device).eval()


def extrinsics_to_quat_trans(extrinsics: torch.Tensor):
    """[B, K, 4, 4] rig 外参 -> (四元数 [B, K, 4], 平移 [B, K, 3])。"""
    from mapanything.utils.geometry import rotation_matrix_to_quaternion

    R = extrinsics[:, :, :3, :3]
    t = extrinsics[:, :, :3, 3]
    quats = rotation_matrix_to_quaternion(R)
    return quats, t


def build_views(images_norm, quats, trans, ray_dirs) -> List[Dict[str, Any]]:
    """组装 MapAnything 的逐 view 输入(射线方向 + 相机位姿 + 公制尺度标志)。"""
    B, W = images_norm.shape[:2]
    device = images_norm.device
    is_metric = torch.ones(B, dtype=torch.bool, device=device)
    true_shape = torch.tensor(
        [[images_norm.shape[3], images_norm.shape[4]]], device=device,
    ).expand(B, -1).contiguous()
    norm_type = ["dinov2"] * B
    views = []
    for i in range(W):
        rd = ray_dirs[i] if isinstance(ray_dirs, list) else ray_dirs
        views.append({
            "img": images_norm[:, i],
            "ray_directions_cam": rd,
            "camera_pose_quats": quats[:, i],
            "camera_pose_trans": trans[:, i],
            "is_metric_scale": is_metric,
            "data_norm_type": norm_type,
            "true_shape": true_shape,
        })
    return views


def extract_features(backbone, views: List[Dict[str, Any]],
                     cached_enc_feats: torch.Tensor) -> torch.Tensor:
    """DINOv2 编码 -> 注入几何输入 -> info_sharing -> [B, W, N_patches, C]。"""
    from uniception.models.info_sharing.cross_attention_transformer import (
        MultiViewTransformerInput,
    )

    enc_feats = list(cached_enc_feats.unbind(dim=1))
    batch_size_per_view = views[0]["img"].shape[0]
    with torch.autocast("cuda", enabled=False):
        enc_feats = backbone._encode_and_fuse_optional_geometric_inputs(
            views, enc_feats,
        )
    input_scale_token = (
        backbone.scale_token.unsqueeze(0).unsqueeze(-1)
        .repeat(batch_size_per_view, 1, 1)
    )
    info_input = MultiViewTransformerInput(
        features=enc_feats, additional_input_tokens=input_scale_token,
    )
    info_result = backbone.info_sharing(info_input)
    features_list = (
        info_result[0].features if isinstance(info_result, tuple)
        else info_result.features
    )

    features_per_view = []
    for view_feat in features_list:
        B, C, H_p, W_p = view_feat.shape
        features_per_view.append(view_feat.permute(0, 2, 3, 1).reshape(B, H_p * W_p, C))
    return torch.stack(features_per_view, dim=1)


def perceive_rig_from_encodings(backbone, extrinsics_g, intrinsics_g, enc_feats_g,
                                image_size: int = 224) -> torch.Tensor:
    """单 rig 感知(全程 fp32) -> info_sharing 特征 [B, W, N_patches, C]（已 detach）。

    enc_feats_g: [B, W, C_enc, H_p, W_p] fp32 DINOv2 编码; extrinsics_g: [B, W, 4, 4];
    intrinsics_g: [B, W, 3, 3]。图像内容在编码之后不再参与计算, 只用其形状生成射线与
    true_shape, 因此这里构造同形状的零张量作为 view 的 img。
    """
    from mapanything.utils.geometry import get_rays_in_camera_frame

    B_, W_ = enc_feats_g.shape[:2]
    images_g = torch.zeros(
        B_, W_, 3, image_size, image_size,
        device=enc_feats_g.device, dtype=torch.float32,
    )
    H, W_img = images_g.shape[3], images_g.shape[4]
    quats, trans = extrinsics_to_quat_trans(extrinsics_g)
    ray_dirs = [
        get_rays_in_camera_frame(
            intrinsics=intrinsics_g[:, i], height=H, width=W_img,
            normalize_to_unit_sphere=True,
        )[1] for i in range(W_)
    ]
    views = build_views(images_g, quats, trans, ray_dirs)

    with torch.no_grad():
        feats = extract_features(backbone, views, enc_feats_g)
    return feats.detach().float()
