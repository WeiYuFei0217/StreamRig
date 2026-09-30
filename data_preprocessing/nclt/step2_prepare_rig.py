#!/usr/bin/env python3
"""
step2_prepare_rig.py -- 由 NCLT 真值位姿 + step1 图像时间戳生成 StreamRig 的 rig 元数据。

功能 (生成训练/评测读取的两类元数据):
  1) step1 轨迹: 对每个 session, 取 Cam1 的全部图像时间戳(stride=1), 在官方
     groundtruth_<date>.csv 上做 **平移线性插值 + 旋转 Slerp**, 得到 body 系
     world_T_body, 按 TUM 格式写成
         <out-root>/step1_rig_trajectories_<split>/scenes/nclt/trajectories/<date>/trajectory.tum
     时间戳键 = 16 位微秒**截断**成 13 位毫秒 (str(ts_us)[:13]), 写进 TUM 时再 /1000 变秒。
     该 13 位毫秒字符串就是全流程的 rig 主键 (step3 特征目录名与之一致)。
  2) step2 rig 配置:
         <out-root>/step2_rig5_configs_<split>/scenes/nclt/rig_config.json
     每个相机给出 body_T_cam = inv(T_camNormal_body) 与训练用的 224x224 FOV90 内参
     K = [[112,0,112],[0,112,112],[0,0,1]]; 同时记下 1068 原图的焦距 focal_length_1068,
     step3 用它算 FOV90 的 center crop 尺寸。

依赖: numpy, scipy (不需要 torch / opencv / mapanything)。

train/test 划分 (常量 TRAIN_SESSIONS / TEST_SESSIONS):
  train = 2012-01-08, 2012-02-02, 2012-02-04, 2012-03-17, 2012-05-26, 2012-10-28,
          2012-11-17, 2013-04-05   (8 个 session)
  test  = 2012-02-19, 2012-08-20   (2 个 session)

示例用法:
    python step2_prepare_rig.py \
        --centered-root /data/nclt_centered \
        --out-root      /data/nclt_meta \
        --split test

    # groundtruth csv 不在 centered-root 下时 (step1 的软链失败), 显式指过去:
    python step2_prepare_rig.py --centered-root /data/nclt_centered \
        --gt-root /data/nclt_raw --out-root /data/nclt_meta --split train

输出布局:
    <out-root>/step1_rig_trajectories_<split>/scenes/nclt/trajectories/<date>/trajectory.tum
    <out-root>/step1_rig_trajectories_<split>/scenes/nclt/trajectories/<date>/traj_meta.json
    <out-root>/step2_rig5_configs_<split>/scenes/nclt/rig_config.json
    <out-root>/step2_rig5_configs_<split>/scenes/nclt/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SCENE_ID = "nclt"
CAMERA_IDS = [1, 2, 3, 4, 5]
CAPTURE_HZ = 5.0

TRAIN_SESSIONS = [
    "2012-01-08", "2012-02-02", "2012-02-04", "2012-03-17",
    "2012-05-26", "2012-10-28", "2012-11-17", "2013-04-05",
]
TEST_SESSIONS = ["2012-02-19", "2012-08-20"]
SPLITS = {"train": TRAIN_SESSIONS, "test": TEST_SESSIONS}

# rig 训练用的固定内参: 224x224, FOV=90° -> f = 112
RIG_IMAGE_SIZE = 224
RIG_HFOV_DEG = 90.0
RIG_K = np.array([[112.0, 0.0, 112.0], [0.0, 112.0, 112.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# 位姿
# ---------------------------------------------------------------------------

def rpy_to_se3(x, y, z, roll, pitch, yaw) -> np.ndarray:
    """NCLT groundtruth 的 (x,y,z,r,p,h)[弧度] -> 4x4 world_T_body (官方约定)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cp * cy, -cp * sy, sp, x],
        [sr * sp * cy + cr * sy, -sr * sp * sy + cr * cy, -sr * cp, y],
        [-cr * sp * cy + sr * sy, cr * sp * sy + sr * cy, cr * cp, z],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def load_gt_poses(csv_path: str):
    """读 groundtruth_<date>.csv -> (ts_us int64 [N], world_T_body float64 [N,4,4])，丢 NaN 行。"""
    raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64)
    raw = raw[~np.any(np.isnan(raw), axis=1)]
    ts_us = raw[:, 0].astype(np.int64)
    poses = np.zeros((len(raw), 4, 4), dtype=np.float64)
    for i in range(len(raw)):
        poses[i] = rpy_to_se3(raw[i, 1], raw[i, 2], raw[i, 3], raw[i, 4], raw[i, 5], raw[i, 6])
    return ts_us, poses


def interpolate_pose(gt_ts: np.ndarray, gt_poses: np.ndarray, query_ts: int) -> np.ndarray:
    """单时间戳插值: 平移线性 + 旋转 Slerp; 越界取端点。"""
    idx = int(np.searchsorted(gt_ts, query_ts))
    if idx <= 0:
        return gt_poses[0]
    if idx >= len(gt_ts):
        return gt_poses[-1]
    t0, t1 = gt_ts[idx - 1], gt_ts[idx]
    alpha = float(query_ts - t0) / max(float(t1 - t0), 1)
    p0, p1 = gt_poses[idx - 1], gt_poses[idx]
    t_interp = (1 - alpha) * p0[:3, 3] + alpha * p1[:3, 3]
    slerp = Slerp([0, 1], Rotation.concatenate(
        [Rotation.from_matrix(p0[:3, :3]), Rotation.from_matrix(p1[:3, :3])]))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = slerp([alpha])[0].as_matrix()
    T[:3, 3] = t_interp
    return T


# ---------------------------------------------------------------------------
# 输入定位
# ---------------------------------------------------------------------------

def image_timestamps_us(centered_root: str, session: str, cam_id: int = 1) -> np.ndarray:
    """某相机 lb3_centered 目录下的全量图像时间戳(微秒, 升序)。"""
    cam_dir = os.path.join(centered_root, session, "lb3_centered", f"Cam{cam_id}")
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"找不到 {cam_dir} (先跑 step1_undistort_center_crop.py)")
    ts = sorted(int(n[:-4]) for n in os.listdir(cam_dir) if n.endswith(".jpg"))
    return np.array(ts, dtype=np.int64)


def find_gt_csv(session: str, centered_root: str, gt_root: str | None) -> str:
    """按 <gt-root>/<date>/ -> <centered-root>/<date>/ -> <gt-root>/ 的顺序找 groundtruth csv。"""
    name = f"groundtruth_{session}.csv"
    cands = []
    if gt_root:
        cands += [os.path.join(gt_root, session, name), os.path.join(gt_root, name)]
    cands += [os.path.join(centered_root, session, name), os.path.join(centered_root, name)]
    for p in cands:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"找不到 {name}; 试过: {cands}")


def load_centered_meta(centered_root: str):
    """读 step1 落盘的 image_meta_centered.pkl -> (K (5,3,3), T_camNormal_body (5,4,4))。"""
    pkl = os.path.join(centered_root, "image_meta_centered.pkl")
    if not os.path.isfile(pkl):
        raise FileNotFoundError(f"找不到 {pkl} (先跑 step1_undistort_center_crop.py)")
    with open(pkl, "rb") as f:
        meta = pickle.load(f)
    return np.array(meta["K"], dtype=np.float64), np.array(meta["T"], dtype=np.float64)


# ---------------------------------------------------------------------------
# 生成
# ---------------------------------------------------------------------------

def generate_trajectories(sessions, centered_root, gt_root, out_root, split) -> dict:
    """逐 session 写 trajectory.tum + traj_meta.json; 返回 {date: 帧数}。"""
    counts = {}
    for session in sessions:
        print(f"  step1 {session} ...", flush=True)
        out_dir = os.path.join(out_root, f"step1_rig_trajectories_{split}",
                               "scenes", SCENE_ID, "trajectories", session)
        os.makedirs(out_dir, exist_ok=True)

        gt_ts, gt_poses = load_gt_poses(find_gt_csv(session, centered_root, gt_root))
        img_ts = image_timestamps_us(centered_root, session, cam_id=1)
        img_ts = img_ts[(img_ts >= gt_ts[0]) & (img_ts <= gt_ts[-1])]   # 只留 GT 覆盖区间

        tmp = os.path.join(out_dir, f"trajectory.tum.tmp{os.getpid()}")
        with open(tmp, "w") as f:
            f.write(f"# NCLT Rig body trajectory {session} (全量 stride=1)\n")
            f.write("# format: timestamp_sec tx ty tz qx qy qz qw\n")
            for ts_us in img_ts:
                T = interpolate_pose(gt_ts, gt_poses, ts_us)
                pos = T[:3, 3]
                q = Rotation.from_matrix(T[:3, :3]).as_quat()      # (x, y, z, w)
                # 用毫秒整数 /1000 写入, 避免 float64 直接吃 16 位微秒丢精度
                ts_sec = int(str(int(ts_us))[:13]) / 1000.0
                f.write(f"{ts_sec:.3f} {pos[0]:.9f} {pos[1]:.9f} {pos[2]:.9f} "
                        f"{q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f}\n")
        os.replace(tmp, os.path.join(out_dir, "trajectory.tum"))

        with open(os.path.join(out_dir, "traj_meta.json"), "w") as f:
            json.dump({"scene_id": SCENE_ID, "date": session, "num_frames": int(len(img_ts)),
                       "num_cameras": len(CAMERA_IDS), "capture_hz": CAPTURE_HZ}, f, indent=2)

        counts[session] = int(len(img_ts))
        print(f"    {len(img_ts):,} frames", flush=True)
    return counts


def generate_rig_config(sessions, centered_root, out_root, split) -> None:
    """写 rig_config.json + manifest.json。"""
    out_dir = os.path.join(out_root, f"step2_rig5_configs_{split}", "scenes", SCENE_ID)
    os.makedirs(out_dir, exist_ok=True)
    K_all, T_camNormal_body = load_centered_meta(centered_root)

    cameras = []
    for i, cam_id in enumerate(CAMERA_IDS):
        cameras.append({
            "name": f"Cam{cam_id}",
            "index": i,
            "nclt_cam_id": cam_id,
            # body_T_cam = inv(T_camNormal_body); 训练端取 inv(body_T_cam[0]) @ body_T_cam[i] 当 rig 外参
            "body_T_cam": np.linalg.inv(T_camNormal_body[i]).tolist(),
            "intrinsics": RIG_K.tolist(),
            # 1068 原图的焦距, step3 靠它算 FOV90 的 center crop 尺寸 round(2f)
            "focal_length_1068": float(K_all[i, 0, 0]),
        })

    with open(os.path.join(out_dir, "rig_config.json"), "w") as f:
        json.dump({"num_cameras": len(CAMERA_IDS),
                   "dataset_source": "nclt", "hfov_deg": RIG_HFOV_DEG,
                   "width": RIG_IMAGE_SIZE, "height": RIG_IMAGE_SIZE,
                   "cameras": cameras}, f, indent=2)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump({"scene_id": SCENE_ID, "selected_trajectories": list(sessions),
                   "num_cameras": len(CAMERA_IDS)}, f, indent=2)
    print("  step2: rig_config + manifest written")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NCLT rig 元数据生成 (step1 轨迹 + step2 rig 配置)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--centered-root", required=True,
                    help="step1 输出根 (含 image_meta_centered.pkl 与 <date>/lb3_centered/)")
    ap.add_argument("--out-root", required=True, help="元数据输出根")
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--gt-root", default="", help="groundtruth csv 所在根; 留空则在 centered-root 下找")
    ap.add_argument("--sessions", nargs="+", default=None, help="覆盖 split 的默认 session 列表")
    args = ap.parse_args()

    sessions = args.sessions or SPLITS[args.split]
    print(f"=== NCLT step2: rig metadata ({args.split}) ===")
    print(f"  sessions: {sessions}")

    counts = generate_trajectories(sessions, args.centered_root, args.gt_root or None,
                                   args.out_root, args.split)
    generate_rig_config(sessions, args.centered_root, args.out_root, args.split)

    total = sum(counts.values())
    print(f"\n=== 完成: {len(sessions)} sessions, {total:,} rigs ===")
    print(f"  step1_root = {args.out_root}/step1_rig_trajectories_{args.split}")
    print(f"  step2_root = {args.out_root}/step2_rig5_configs_{args.split}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
