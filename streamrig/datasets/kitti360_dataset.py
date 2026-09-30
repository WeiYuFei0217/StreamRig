"""
kitti360_dataset.py -- KITTI-360 四相机 StreamRig 数据集。

输入是 KITTI-360 预处理脚本（见 ``data_preprocessing/kitti360/``）生成的轻量索引与
info-sharing 特征缓存（与 NCLT 同布局的
``scenes/kitti360/trajectories/<seq>/features/<frame>/rig.npy``，[4,256,768]）：

* rig 固定为四目：透视双目 image_00/01（官方 ``data_rect``，主点居中方形裁剪后缩放到
  224×224）+ 双侧鱼眼 image_02/03（MEI 模型重映射成 90° 方形虚拟针孔图）；
  缓存与每个 view 的 K 使用同一 canonical 变换。
* GT 使用 ``T_world_pose @ T_pose_cam``，参考系是窗口首帧的 rectified image_00，
  与 NCLT 相同，``T_rel_gt[0] = I``（锚系绝对位姿）。

示例用法::

    ds = Kitti360StreamDataset(processed_root="/path/to/kitti360_streamrig_metadata",
                               infoshare_root="/path/to/kitti360_infoshare_bf16",
                               split="train", num_rigs=12)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from .formats import (
    CANONICAL_PERSPECTIVE_PREPROCESSING,
    KITTI360_INFOSHARE_FORMAT,
    KITTI360_PROCESSED_FORMAT,
    KITTI360_RIG_FORMAT,
    check_format,
)

# 四目 rig：image_00/01 透视双目 + image_02/03 鱼眼
CAMERA_IDS = (0, 1, 2, 3)
INFOSHARE_SHAPE = [len(CAMERA_IDS), 256, 768]


def _split_valid_runs(ts_ns, gap_threshold_ms, num_rigs_min, min_run_span_ms):
    """按真实相机时间切分连续片段，返回半开区间 ``[(start, end), ...]``。"""

    gap_ns = int(round(float(gap_threshold_ms) * 1_000_000))
    min_span_ns = int(round(float(min_run_span_ms) * 1_000_000))

    def valid(start, end):
        return (
            end - start >= num_rigs_min
            and int(ts_ns[end - 1] - ts_ns[start]) >= min_span_ns
        )

    runs = []
    start = 0
    for idx in range(1, len(ts_ns)):
        if int(ts_ns[idx] - ts_ns[idx - 1]) > gap_ns:
            if valid(start, idx):
                runs.append((start, idx))
            start = idx
    if len(ts_ns) and valid(start, len(ts_ns)):
        runs.append((start, len(ts_ns)))
    return runs


class Kitti360StreamDataset(torch.utils.data.Dataset):
    """KITTI-360 四相机流式窗口数据集。

    从 ``split``（``processed_root/splits/<split>.txt``）的序列中随机采窗（相邻 rig 帧间隔逐对
    均匀采自 [stride_min, stride_max]）；评测脚本直接读取 seqs / runs 做完整序列推理。接口与
    :class:`StreamOdomDataset` 对齐，因而可直接复用现有 collate、loss 和模型。
    """

    def __init__(
        self,
        processed_root: str,
        infoshare_root: str,
        split: str = "train",
        num_rigs: int = 12,
        stride_min: int = 1,
        stride_max: int = 6,
        gap_threshold_ms: float = 500.0,
        min_run_span_ms: float = 30_000.0,
        samples_per_epoch: int = 48_000,
        image_size: int = 224,
    ):
        super().__init__()
        if not infoshare_root:
            raise ValueError("Kitti360StreamDataset 需要 infoshare_root")

        self.processed_root = Path(processed_root)
        manifest_path = self.processed_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"缺 KITTI-360 manifest: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        check_format(self.manifest.get("format"), KITTI360_PROCESSED_FORMAT, str(manifest_path))
        canonical = self.manifest.get("perspective_preprocessing")
        if canonical != CANONICAL_PERSPECTIVE_PREPROCESSING:
            raise ValueError(
                "KITTI-360 processed_root 的透视相机预处理不是 isotropic_square_crop："
                f"perspective_preprocessing={canonical!r}；请重跑 data_preprocessing/kitti360/ 下的预处理脚本"
            )
        self.split = split
        self.N = int(num_rigs)
        self.K = len(CAMERA_IDS)
        self.stride_min = int(stride_min)
        self.stride_max = int(stride_max)
        self.samples_per_epoch = int(samples_per_epoch)
        self.image_size = int(image_size)
        self.infoshare_root = Path(infoshare_root)

        complete_path = self.infoshare_root / "COMPLETE"
        if not complete_path.is_file():
            raise RuntimeError(
                f"KITTI-360 info_sharing 缓存未通过全量验收（缺 COMPLETE）: "
                f"{self.infoshare_root}"
            )
        cache_manifest_path = self.infoshare_root / "manifest.json"
        if not cache_manifest_path.is_file():
            raise FileNotFoundError(f"缺 KITTI-360 info_sharing manifest: {cache_manifest_path}")
        self.infoshare_manifest = json.loads(
            cache_manifest_path.read_text(encoding="utf-8")
        )
        check_format(
            self.infoshare_manifest.get("format"), KITTI360_INFOSHARE_FORMAT,
            str(cache_manifest_path),
        )
        if self.infoshare_manifest.get("perspective_preprocessing") != canonical:
            raise ValueError("info_sharing 缓存与 processed_root 的图像预处理方式不一致")
        rig_digest = hashlib.sha256(
            (self.processed_root / "rig_config.json").read_bytes()
        ).hexdigest()
        if self.infoshare_manifest.get("source_rig_config_sha256") != rig_digest:
            raise ValueError("info_sharing 缓存与当前 rig_config 的内外参指纹不一致")
        # 校验特征形状 [4, 256, 768]
        feature_shape = self.infoshare_manifest.get("feature_shape")
        if feature_shape != INFOSHARE_SHAPE:
            raise ValueError(
                f"info_sharing manifest 的 feature_shape={feature_shape}，应为 {INFOSHARE_SHAPE}"
            )

        if self.image_size != int(self.manifest["image_size"]):
            raise ValueError(
                f"image_size={self.image_size} 与预处理 manifest={self.manifest['image_size']} 不一致"
            )

        rig_path = self.processed_root / "rig_config.json"
        self.rig_config = json.loads(rig_path.read_text(encoding="utf-8"))
        check_format(self.rig_config.get("format"), KITTI360_RIG_FORMAT, str(rig_path))
        if self.rig_config.get("perspective_preprocessing") != canonical:
            raise ValueError("manifest 与 rig_config 的 perspective_preprocessing 不一致")
        all_body_T_cams = [
            np.asarray(cam["body_T_cam"], dtype=np.float64).reshape(4, 4)
            for cam in self.rig_config["cameras"]
        ]
        self.body_T_cams = [all_body_T_cams[c] for c in CAMERA_IDS]
        ref_inv = np.linalg.inv(self.body_T_cams[0])
        # rig 标定：各相机相对 image_00
        self.rig_calib = np.stack(
            [ref_inv @ transform for transform in self.body_T_cams], axis=0
        ).astype(np.float32)

        split_path = self.processed_root / "splits" / f"{self.split}.txt"
        if not split_path.is_file():
            raise FileNotFoundError(f"未知 KITTI-360 split: {split_path}")
        seq_names = [
            line.strip() for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.seqs = []
        for seq_name in seq_names:
            meta_path = self.processed_root / "sequences" / seq_name / "frames.npz"
            meta = np.load(meta_path)
            frame_ids = meta["frame_ids"].astype(np.int64)
            timestamps_ns = meta["timestamps_ns"].astype(np.int64)
            world_T_pose = meta["world_T_pose"].astype(np.float64)
            if timestamps_ns.ndim != 2 or timestamps_ns.shape[1] != 4:
                raise ValueError(f"{meta_path}: timestamps_ns 应为 [L,4]")
            runs = _split_valid_runs(
                timestamps_ns[:, 0], gap_threshold_ms, self.N, min_run_span_ms
            )
            if runs:
                self.seqs.append({
                    "scene": "kitti360",
                    "traj": seq_name,
                    "frame_ids": frame_ids,
                    "timestamps_ns": timestamps_ns,
                    "world_T_pose": world_T_pose,
                    "runs": runs,
                })

        min_span = (self.N - 1) * self.stride_min
        self._run_flat = []
        weights = []
        for seq_idx, seq in enumerate(self.seqs):
            for run_start, run_end in seq["runs"]:
                if run_end - run_start >= min_span + 1:
                    self._run_flat.append((seq_idx, run_start, run_end))
                    weights.append(run_end - run_start)
        if not self._run_flat:
            raise RuntimeError(
                f"KITTI-360 split={self.split} 没有能容纳 N={self.N} 的连续片段"
            )
        weight_arr = np.asarray(weights, dtype=np.float64)
        self._run_probs = weight_arr / weight_arr.sum()

        print(
            f"[Kitti360StreamDataset] split={self.split}, {len(self.seqs)} 序列, "
            f"{len(self._run_flat)} run, N={self.N}, K={self.K}",
            flush=True,
        )

    def __len__(self):
        return self.samples_per_epoch

    def _load_infoshare(self, seq_name: str, frame_key: str):
        path = (
            self.infoshare_root / "scenes" / "kitti360" / "trajectories"
            / seq_name / "features" / frame_key / "rig.npy"
        )
        if not path.is_file():
            raise FileNotFoundError(f"缺 KITTI-360 info_sharing: {path}")
        packed = np.load(path)
        tensor = torch.from_numpy(packed.astype(np.uint16)).view(torch.bfloat16).float()
        expected_shape = tuple(INFOSHARE_SHAPE)
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"{path}: 缓存 shape 应为 {expected_shape}，收到 {tuple(tensor.shape)}")
        return tensor

    def _build_sample(self, seq, indices):
        frame_ids = seq["frame_ids"][indices]
        frame_keys = [f"{int(frame_id):010d}" for frame_id in frame_ids]
        seq_name = seq["traj"]

        ref_body_T_cam = self.body_T_cams[0]
        world_T_cam = [
            seq["world_T_pose"][idx] @ ref_body_T_cam for idx in indices
        ]
        anchor_inv = np.linalg.inv(world_T_cam[0])
        relative = np.stack([anchor_inv @ pose for pose in world_T_cam], axis=0).astype(np.float32)

        sample = {
            "rig_calib": torch.from_numpy(self.rig_calib.copy()),
            "T_rel_gt": torch.from_numpy(relative),
            "scene_id": "kitti360",
            "traj_id": seq_name,
            "ts_list": frame_keys,
            "infoshare": torch.stack([
                self._load_infoshare(seq_name, key) for key in frame_keys
            ], dim=0),
        }
        return sample

    def __getitem__(self, index):
        # index = (N, seed) 元组（由 VariableNStreamBatchSampler 产生）
        n, sample_seed = index
        return self._sample_train_window(int(n), np.random.default_rng(int(sample_seed)))

    def _sample_train_window(self, n, rng):
        run_idx = int(rng.choice(len(self._run_flat), p=self._run_probs))
        seq_idx, run_start, run_end = self._run_flat[run_idx]
        run_length = run_end - run_start

        for _ in range(20):
            strides = rng.integers(self.stride_min, self.stride_max + 1, size=n - 1)
            span = int(strides.sum())
            if span <= run_length - 1:
                break
        else:
            strides = np.full(n - 1, self.stride_min, dtype=np.int64)
            span = int(strides.sum())

        anchor = run_start + int(rng.integers(0, run_length - span))
        indices = [anchor]
        for stride in strides:
            indices.append(indices[-1] + int(stride))
        return self._build_sample(self.seqs[seq_idx], indices)
