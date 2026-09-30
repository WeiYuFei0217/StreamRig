"""
formats.py -- KITTI-360 预处理产物的格式标识（manifest / rig_config / 特征缓存）与共用常量。

预处理脚本写入 ``*_FORMAT``；加载器通过 :func:`check_format` 校验。校验只看标识的
通用后缀（如 ``-kitti360-v2``），不限定前缀；特征形状另由加载器按 manifest 的
feature_shape 校验（见 kitti360_dataset.py）。

示例用法：
    from streamrig.datasets.formats import KITTI360_PROCESSED_FORMAT, check_format
    check_format(manifest.get("format"), KITTI360_PROCESSED_FORMAT, "processed manifest")
"""

from __future__ import annotations

# 透视相机图像预处理方式（主点居中方形裁剪 + 等比例缩放），写入各产物并由加载器校验
CANONICAL_PERSPECTIVE_PREPROCESSING = "isotropic_square_crop"

KITTI360_PROCESSED_FORMAT = "streamrig-kitti360-v2"
KITTI360_RIG_FORMAT = "streamrig-kitti360-rig-v2"
KITTI360_INFOSHARE_FORMAT = "streamrig-kitti360-infoshare-v2"

# 每种产物接受的格式后缀
_ACCEPTED_SUFFIXES = {
    KITTI360_PROCESSED_FORMAT: ("-kitti360-v2",),
    KITTI360_RIG_FORMAT: ("-kitti360-rig-v2",),
    KITTI360_INFOSHARE_FORMAT: ("-kitti360-infoshare-v2", "-kitti360-infoshare-v1"),
}


def check_format(value, expected: str, what: str) -> None:
    """校验格式标识。缺失（None）时放行；存在但后缀不匹配时报错。"""
    if value is None:
        return
    if not (isinstance(value, str) and value.endswith(_ACCEPTED_SUFFIXES[expected])):
        raise ValueError(f"{what} 的 format={value!r} 不受支持，应为 {expected!r}")
