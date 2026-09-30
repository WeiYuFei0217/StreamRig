"""
step1_prepare_kitti360.py -- 把官方 KITTI-360 整理成 StreamRig 可直接读取的轻量索引。

功能：本脚本 **不复制原图**（原始 2D 包约 480 GiB），只生成约 31 MiB 的元数据：

* 四相机 rig_config：透视双目按主点水平居中的等比例方形裁剪（isotropic_square_crop）
  对应的 224 域内参，以及四个相机的 ``body_T_cam`` 外参；
* 官方 MEI 鱼眼 → 90° 虚拟针孔的固定 remap（离线算好存 ``rectify_maps/``）；
* 每条序列的帧号、四相机时间戳、``world_T_pose``、官方 ``world_T_cam0``；
* 按序列互不重叠的 train/test split 与 TUM 轨迹。

输出布局（``--out-root``）::

    manifest.json                       # 全局设置 + 每序列统计
    rig_config.json                     # 四相机 K / body_T_cam / 图像变换元数据
    rectify_maps/image_02.npz           # 鱼眼→虚拟针孔 remap（image_03 同理）
    splits/{train,test}.txt
    sequences/<drive>/frames.npz        # frame_ids/timestamps_ns/world_T_pose/official_world_T_cam0
    sequences/<drive>/frame_ids.txt
    sequences/<drive>/trajectory.tum            # world_T_pose（GPS/IMU body 系）
    sequences/<drive>/trajectory_cam0_rect.tum  # world_T_pose @ body_T_cam0

示例用法::

    conda activate streamrig
    python data_preprocessing/kitti360/step1_prepare_kitti360.py \
      --raw-root /path/to/KITTI-360 \
      --out-root /path/to/kitti360_streamrig_metadata
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# 允许不安装包直接运行：把仓库根加入 sys.path（本文件在 data_preprocessing/kitti360/ 下）
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing.kitti360.kitti360_preprocessing import perspective_transform  # noqa: E402
from streamrig.datasets.formats import (  # noqa: E402
    CANONICAL_PERSPECTIVE_PREPROCESSING,
    KITTI360_PROCESSED_FORMAT,
    KITTI360_RIG_FORMAT,
)


# StreamRig 使用的按序列互不重叠的划分（KITTI-360 未发布官方 odometry 划分）
DEFAULT_TRAIN = (0, 2, 3, 4, 5, 6, 7)
DEFAULT_TEST = (9, 10)

# 写入 rig_config 的透视预处理说明（保留完整垂直视野，按主点水平居中方形裁剪，再等比例缩放）。
# 缓存 manifest 记录 rig_config.json 的 SHA-256，加载器按该指纹校验，因此 info_sharing 缓存
# 须与生成它的 processed_root 配套使用。
PERSPECTIVE_CROP_REASON = (
    "canonical 方案：保留完整垂直视野，按主点水平居中方形裁剪，"
    "再等比例缩放；图像与 K 使用同一变换"
)


def as_homogeneous(values):
    """把 12/16 个数解释成 4x4 的 SE(3)。"""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 12:
        array = array.reshape(3, 4)
        return np.vstack([array, [0.0, 0.0, 0.0, 1.0]])
    if array.size == 16:
        return array.reshape(4, 4)
    raise ValueError(f"SE(3) 需要 12/16 个数，收到 {array.size}")


def parse_named_calibration(path):
    """解析 ``key: v1 v2 ...`` 形式的官方标定文本。"""

    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        fields = raw.split()
        try:
            values[key.strip()] = np.asarray([float(x) for x in fields], dtype=np.float64)
        except ValueError:
            continue
    return values


def read_mei_yaml(path):
    """读官方鱼眼 MEI 标定 yaml（首行是 OpenCV 的 ``%YAML`` 指令，需跳过）。"""

    text = Path(path).read_text(encoding="utf-8")
    text = "\n".join(text.splitlines()[1:])
    return yaml.safe_load(text)


def build_mei_to_pinhole_map(mei, image_size, fov_deg):
    """按 KITTI-360 官方 CameraFisheye 投影生成目标像素到源像素的反向映射。

    官方 helper 使用 unified/MEI 的 xi + k1/k2，未使用 yaml 中很小的 p1/p2；这里
    保持与官方工具逐式一致，避免 OpenCV fisheye 四参数模型近似。
    """

    size = int(image_size)
    focal = size / (2.0 * np.tan(np.deg2rad(float(fov_deg)) / 2.0))
    center = (size - 1.0) / 2.0
    u, v = np.meshgrid(np.arange(size, dtype=np.float64), np.arange(size, dtype=np.float64))
    rays = np.stack([(u - center) / focal, (v - center) / focal, np.ones_like(u)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    xi = float(mei["mirror_parameters"]["xi"])
    denom = rays[..., 2] + xi
    x = rays[..., 0] / denom
    y = rays[..., 1] / denom
    radius2 = x * x + y * y
    k1 = float(mei["distortion_parameters"]["k1"])
    k2 = float(mei["distortion_parameters"]["k2"])
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
    x *= radial
    y *= radial

    projection = mei["projection_parameters"]
    map_x = float(projection["gamma1"]) * x + float(projection["u0"])
    map_y = float(projection["gamma2"]) * y + float(projection["v0"])
    width = int(mei["image_width"])
    height = int(mei["image_height"])
    valid = (
        np.isfinite(map_x) & np.isfinite(map_y)
        & (map_x >= 0.0) & (map_x <= width - 1.0)
        & (map_y >= 0.0) & (map_y <= height - 1.0)
    )
    return map_x.astype(np.float32), map_y.astype(np.float32), valid, focal, center


def parse_timestamps_ns(path):
    """官方 ``timestamps.txt`` → int64 纳秒数组。"""

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return np.asarray([np.datetime64(line.strip(), "ns").astype(np.int64) for line in lines])


def parse_pose_file(path, matrix_columns):
    """官方 ``poses.txt`` / ``cam0_to_world.txt`` → (frame_ids, [L,4,4])。"""

    data = np.loadtxt(path, dtype=np.float64)
    data = np.atleast_2d(data)
    frame_ids = data[:, 0].astype(np.int64)
    transforms = np.stack([as_homogeneous(row[1:1 + matrix_columns]) for row in data])
    if len(np.unique(frame_ids)) != len(frame_ids):
        raise ValueError(f"{path}: frame id 重复")
    return frame_ids, transforms


def standard_png_frame_ids(path):
    """目录内 ``0000000000.png`` 形式文件的 frame id 集合。"""

    pattern = re.compile(r"^[0-9]{10}\.png$")
    return {
        int(name[:-4]) for name in os.listdir(path)
        if pattern.match(name)
    }


def write_tum(path, timestamps_ns, transforms):
    """以 TUM 格式落盘轨迹，便于用 evo 等工具直接可视化/对齐。"""

    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp_ns, transform in zip(timestamps_ns, transforms):
            quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
            trans = transform[:3, 3]
            handle.write(
                f"{int(timestamp_ns) / 1e9:.9f} "
                f"{trans[0]:.10f} {trans[1]:.10f} {trans[2]:.10f} "
                f"{quat[0]:.10f} {quat[1]:.10f} {quat[2]:.10f} {quat[3]:.10f}\n"
            )


def sequence_name(sequence_id):
    return f"2013_05_28_drive_{int(sequence_id):04d}_sync"


def matrix_to_list(matrix):
    return np.asarray(matrix, dtype=np.float64).reshape(-1).tolist()


def prepare_calibration(raw_root, out_root, image_size, fisheye_fov_deg):
    """生成 rig_config.json + 鱼眼 remap；返回 (body_T_cams, intrinsics, rig_config)。"""

    calibration_root = raw_root / "calibration"
    cam_to_pose_raw = parse_named_calibration(calibration_root / "calib_cam_to_pose.txt")
    perspective = parse_named_calibration(calibration_root / "perspective.txt")

    body_T_cams = []
    intrinsics = []
    camera_records = []
    for cam_id in range(4):
        key = f"image_{cam_id:02d}"
        body_T_cam = as_homogeneous(cam_to_pose_raw[key])
        if cam_id < 2:
            # 透视相机：官方 body_T_cam 是未整流相机系，需要右乘 inv(R_rect) 变到整流相机系
            rectification = np.eye(4, dtype=np.float64)
            rectification[:3, :3] = perspective[f"R_rect_{cam_id:02d}"].reshape(3, 3)
            body_T_cam = body_T_cam @ np.linalg.inv(rectification)
            projection = perspective[f"P_rect_{cam_id:02d}"].reshape(3, 4)
            source_width, source_height = perspective[f"S_rect_{cam_id:02d}"].astype(int)
            source_K = projection[:3, :3].copy()
            K, preprocessing = perspective_transform(
                source_K,
                source_width,
                source_height,
                image_size,
            )
            source_kind = "rectified_perspective"
            preprocessing["reason"] = PERSPECTIVE_CROP_REASON
        else:
            # 鱼眼相机：官方 MEI 模型离线生成到 90° 虚拟针孔的 remap
            mei = read_mei_yaml(calibration_root / f"image_{cam_id:02d}.yaml")
            map_x, map_y, valid, focal, center = build_mei_to_pinhole_map(
                mei, image_size, fisheye_fov_deg
            )
            maps_root = out_root / "rectify_maps"
            maps_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                maps_root / f"image_{cam_id:02d}.npz",
                map_x=map_x,
                map_y=map_y,
                valid_mask=valid,
                fov_deg=np.float64(fisheye_fov_deg),
            )
            K = np.asarray([
                [focal, 0.0, center],
                [0.0, focal, center],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            source_kind = "mei_fisheye_to_virtual_pinhole"
            preprocessing = {
                "kind": "official_mei_remap",
                "source_size": [int(mei["image_width"]), int(mei["image_height"])],
                "target_size": [image_size, image_size],
                "virtual_fov_deg": float(fisheye_fov_deg),
                "valid_fraction": float(valid.mean()),
                "official_projection_terms": ["xi", "k1", "k2", "gamma1", "gamma2", "u0", "v0"],
            }
        body_T_cams.append(body_T_cam)
        intrinsics.append(K)
        camera_records.append({
            "id": cam_id,
            "name": key,
            "source_kind": source_kind,
            **({"source_intrinsics": matrix_to_list(source_K)} if cam_id < 2 else {}),
            "intrinsics": matrix_to_list(K),
            "body_T_cam": matrix_to_list(body_T_cam),
            "preprocessing": preprocessing,
        })

    # rig 参考系固定为整流后的 image_00；训练/缓存注入的是 cam0_T_cam
    reference_inv = np.linalg.inv(body_T_cams[0])
    extrinsics = [reference_inv @ transform for transform in body_T_cams]
    rig_config = {
        "format": KITTI360_RIG_FORMAT,
        "dataset": "KITTI-360",
        "body_frame": "GPS_IMU_pose",
        "reference_camera": "image_00_rectified",
        "body_T_cam_precomputed": True,
        "width": image_size,
        "height": image_size,
        "perspective_preprocessing": CANONICAL_PERSPECTIVE_PREPROCESSING,
        "cameras": camera_records,
        "extrinsics_cam0_reference": [matrix_to_list(transform) for transform in extrinsics],
    }
    (out_root / "rig_config.json").write_text(
        json.dumps(rig_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return body_T_cams, intrinsics, rig_config


def prepare_sequence(raw_root, out_root, seq_name, body_T_cam0):
    """整理单条 drive：严格取四相机 PNG ∩ poses ∩ cam0_to_world ∩ 时间戳可索引范围。"""

    image_root = raw_root / "data_2d_raw" / seq_name
    pose_root = raw_root / "data_poses" / seq_name
    pose_frame_ids, world_T_pose = parse_pose_file(pose_root / "poses.txt", 12)
    cam0_frame_ids, official_world_T_cam0 = parse_pose_file(
        pose_root / "cam0_to_world.txt", 16
    )
    pose_lookup = {int(frame): transform for frame, transform in zip(pose_frame_ids, world_T_pose)}
    cam0_lookup = {
        int(frame): transform for frame, transform in zip(cam0_frame_ids, official_world_T_cam0)
    }

    image_sets = []
    timestamps = []
    image_counts = []
    for cam_id in range(4):
        kind = "data_rect" if cam_id < 2 else "data_rgb"
        directory = image_root / f"image_{cam_id:02d}" / kind
        frame_set = standard_png_frame_ids(directory)
        image_sets.append(frame_set)
        image_counts.append(len(frame_set))
        timestamps.append(parse_timestamps_ns(image_root / f"image_{cam_id:02d}" / "timestamps.txt"))

    common = set(pose_lookup)
    for frame_set in image_sets:
        common &= frame_set
    common &= set(cam0_lookup)
    common = {
        frame for frame in common
        if all(frame < len(cam_timestamps) for cam_timestamps in timestamps)
    }
    frame_ids = np.asarray(sorted(common), dtype=np.int64)
    if not len(frame_ids):
        raise RuntimeError(f"{seq_name}: 四相机 ∩ poses ∩ cam0_to_world 为空")
    world_T_pose_kept = np.stack([pose_lookup[int(frame)] for frame in frame_ids])
    official_cam0_kept = np.stack([cam0_lookup[int(frame)] for frame in frame_ids])
    timestamps_kept = np.stack([
        np.asarray([timestamps[cam_id][frame] for frame in frame_ids], dtype=np.int64)
        for cam_id in range(4)
    ], axis=1)

    seq_out = out_root / "sequences" / seq_name
    seq_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        seq_out / "frames.npz",
        frame_ids=frame_ids,
        timestamps_ns=timestamps_kept,
        world_T_pose=world_T_pose_kept,
        official_world_T_cam0=official_cam0_kept,
    )
    write_tum(seq_out / "trajectory.tum", timestamps_kept[:, 0], world_T_pose_kept)
    computed_world_T_cam0 = world_T_pose_kept @ body_T_cam0
    write_tum(seq_out / "trajectory_cam0_rect.tum", timestamps_kept[:, 0], computed_world_T_cam0)
    (seq_out / "frame_ids.txt").write_text(
        "\n".join(f"{int(frame):010d}" for frame in frame_ids) + "\n", encoding="utf-8"
    )

    # 自检：用 world_T_pose @ body_T_cam0 复原官方 cam0_to_world，残差应在毫米/1e-5° 量级
    delta = np.linalg.inv(official_cam0_kept) @ computed_world_T_cam0
    translation_error = np.linalg.norm(delta[:, :3, 3], axis=1)
    rotation_error = Rotation.from_matrix(delta[:, :3, :3]).magnitude()
    sync_offsets_ms = (timestamps_kept - timestamps_kept[:, :1]) / 1e6
    return {
        "sequence": seq_name,
        "pose_frames": int(len(pose_frame_ids)),
        "camera_image_counts": image_counts,
        "kept_four_camera_pose_frames": int(len(frame_ids)),
        "dropped_pose_frames": int(len(pose_frame_ids) - len(frame_ids)),
        "frame_min": int(frame_ids[0]),
        "frame_max": int(frame_ids[-1]),
        "max_camera_sync_offset_ms": float(np.abs(sync_offsets_ms).max()),
        "cam0_reconstruction_translation_max_m": float(translation_error.max()),
        "cam0_reconstruction_rotation_max_deg": float(np.degrees(rotation_error.max())),
    }


def write_split(out_root, split_name, sequence_ids):
    """落盘 ``splits/<split>.txt``（每行一条 drive 名），返回序列名列表。"""

    names = [sequence_name(sequence_id) for sequence_id in sequence_ids]
    path = out_root / "splits" / f"{split_name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names


def main():
    parser = argparse.ArgumentParser(
        description="KITTI-360 → StreamRig 轻量索引（不复制原图）"
    )
    parser.add_argument("--raw-root", required=True,
                        help="官方 KITTI-360 根目录（含 calibration/data_2d_raw/data_poses）")
    parser.add_argument("--out-root", required=True, help="轻量索引输出目录（约 31 MiB）")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fisheye-fov-deg", type=float, default=90.0)
    parser.add_argument("--train-seqs", type=int, nargs="+", default=list(DEFAULT_TRAIN))
    parser.add_argument("--test-seqs", type=int, nargs="+", default=list(DEFAULT_TEST))
    args = parser.parse_args()

    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not (raw_root / "calibration" / "calib_cam_to_pose.txt").is_file():
        raise FileNotFoundError(f"不是完整 KITTI-360 根目录: {raw_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    split_ids = {
        "train": tuple(args.train_seqs),
        "test": tuple(args.test_seqs),
    }
    flattened = [seq for ids in split_ids.values() for seq in ids]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"train/test split 存在重叠序列: {split_ids}")

    body_T_cams, _, rig_config = prepare_calibration(
        raw_root, out_root, args.image_size, args.fisheye_fov_deg
    )
    split_names = {
        name: write_split(out_root, name, ids) for name, ids in split_ids.items()
    }

    all_sequences = [name for names in split_names.values() for name in names]
    sequence_stats = [
        prepare_sequence(raw_root, out_root, seq_name, body_T_cams[0])
        for seq_name in all_sequences
    ]

    manifest = {
        "format": KITTI360_PROCESSED_FORMAT,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        # step2 默认从这里取原图根；换机器时用 step2 的 --raw-root 覆盖即可
        "raw_root": str(raw_root),
        "image_size": int(args.image_size),
        "fisheye_virtual_fov_deg": float(args.fisheye_fov_deg),
        "perspective_preprocessing": CANONICAL_PERSPECTIVE_PREPROCESSING,
        "splits": split_names,
        "camera_order": ["image_00_rect", "image_01_rect", "image_02_virtual90", "image_03_virtual90"],
        "pose_convention": "world_T_pose; sample GT = inv(world_T_cam0_at_anchor) @ world_T_cam0_at_t",
        "sequence_stats": sequence_stats,
        "total_kept_frames": int(sum(item["kept_four_camera_pose_frames"] for item in sequence_stats)),
        "rig_config": "rig_config.json",
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
