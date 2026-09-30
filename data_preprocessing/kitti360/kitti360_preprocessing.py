"""kitti360_preprocessing.py -- KITTI-360 透视相机的图像预处理几何（isotropic_square_crop）。

透视相机先在原始整流图上按主点居中裁成方形，再等比例缩放到目标尺寸；
图像变换与 K 的更新由同一份元数据驱动（step1 生成元数据、step2 按元数据处理图像）。

示例用法::

    from data_preprocessing.kitti360.kitti360_preprocessing import (
        apply_perspective_transform, perspective_transform,
    )
    K224, meta = perspective_transform(source_K, 1408, 376, 224)
    image224 = apply_perspective_transform(image_bgr, meta)
"""

from __future__ import annotations

import cv2
import numpy as np

from streamrig.datasets.formats import CANONICAL_PERSPECTIVE_PREPROCESSING


def fov_xy_deg(K: np.ndarray, width: int, height: int) -> list[float]:
    """按有效主点计算非对称针孔图像的水平/垂直总 FOV。"""

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    fov_x = np.degrees(np.arctan(cx / fx) + np.arctan((width - 1 - cx) / fx))
    fov_y = np.degrees(np.arctan(cy / fy) + np.arctan((height - 1 - cy) / fy))
    return [float(fov_x), float(fov_y)]


def perspective_transform(
    source_K: np.ndarray,
    source_width: int,
    source_height: int,
    image_size: int,
) -> tuple[np.ndarray, dict]:
    """计算方形裁剪 + 缩放后的针孔内参与可复用元数据。"""

    width, height, target = int(source_width), int(source_height), int(image_size)
    source_K = np.asarray(source_K, dtype=np.float64).reshape(3, 3)

    side = min(width, height)
    left = int(np.clip(round(float(source_K[0, 2]) - side / 2.0), 0, width - side))
    top = int(np.clip(round(float(source_K[1, 2]) - side / 2.0), 0, height - side))
    crop_xywh = [left, top, side, side]
    scale_x = scale_y = target / float(side)
    output_K = source_K.copy()
    output_K[0, 2] -= left
    output_K[1, 2] -= top
    output_K[:2, :] *= scale_x

    metadata = {
        "kind": CANONICAL_PERSPECTIVE_PREPROCESSING,
        "source_size": [width, height],
        "target_size": [target, target],
        "crop_xywh": crop_xywh,
        "scale_xy": [float(scale_x), float(scale_y)],
        "intrinsics": output_K.reshape(-1).tolist(),
        "fov_xy_deg": fov_xy_deg(output_K, target, target),
    }
    return output_K.astype(np.float32), metadata


def apply_perspective_transform(image_bgr: np.ndarray, metadata: dict) -> np.ndarray:
    """按 ``perspective_transform`` 的元数据处理一张 BGR 图像。"""

    if metadata.get("kind") != CANONICAL_PERSPECTIVE_PREPROCESSING:
        raise ValueError(f"非法 perspective 元数据: {metadata.get('kind')!r}")
    height, width = image_bgr.shape[:2]
    if [width, height] != list(metadata["source_size"]):
        raise ValueError(
            f"图像尺寸 {width}x{height} 与标定 {metadata['source_size']} 不一致"
        )
    left, top, crop_width, crop_height = map(int, metadata["crop_xywh"])
    crop = image_bgr[top : top + crop_height, left : left + crop_width]
    target_width, target_height = map(int, metadata["target_size"])
    return cv2.resize(
        crop, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
