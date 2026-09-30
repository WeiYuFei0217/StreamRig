"""
data_preprocessing.common -- 各数据集预处理流程共用的小工具集合。

本包**不依赖** streamrig 训练代码。

子模块:
  - infoshare_io: info_sharing 特征缓存的 bf16(uint16 位存) 读写与目录布局约定(只依赖 numpy / torch)。
  - mapanything_frontend: frozen MapAnything 的离线加载与单 rig info_sharing 感知(另需 mapanything)。

示例用法:
    from data_preprocessing.common.infoshare_io import (
        rig_feature_path, save_rig_features, load_rig_features,
    )
"""

__all__ = ["infoshare_io", "mapanything_frontend"]
