#!/usr/bin/env python3
"""k360_drive_metrics.py -- KITTI-360 drive 级指标后处理（valid run 联合池化 + 完整 drive ATE）。

指标定义
--------
KITTI-360 test 划分的两条 drive（0009 / 0010）里，只有**四目图像 ∩ 官方位姿**都齐全的帧
才可用；这些帧把一条 drive 切成若干条连续 **valid run**（0009 六条、0010 四条，共 10 条）。
评测在每条 valid run 上独立做流式推理，于是：

* **相对误差 t_rel / r_rel**：10 条 run 的 100--800 m 子段**放在一起按段计数联合池化**，
  不是先按 run 求平均再平均。
* **ATE**：在完整 drive 上做 SE3 对齐（run 间以真值相对位姿衔接）；两条 drive 按 GT 路径
  长度加权平均。

GT 与帧号来源
-------------

* 官方 GT：``--processed-root``（``data_preprocessing/kitti360/step1_prepare_kitti360.py``
  的产物，``sequences/<seq>/frames.npz`` 里的 ``frame_ids`` + ``official_world_T_cam0``）；
  也可以用 ``--data-poses-root`` 直接读官方 ``data_poses/<seq>/cam0_to_world.txt``。
* 每条 run 的采样帧号：评测 NPZ 自带 ``source_frame_ids``，无需任何额外查表。

示例用法
--------
    python scripts/k360_drive_metrics.py \
        --eval-dir <out>/<exp>/test \
        --processed-root /path/to/kitti360_streamrig_metadata \
        --out-dir <out>/drive_metrics
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitti_metrics import compute_ate, compute_kitti_per_length  # noqa: E402

K360_LENGTHS: tuple[int, ...] = (100, 200, 300, 400, 500, 600, 700, 800)
STEP_SIZE = 10
DRIVES: tuple[str, ...] = ("0009", "0010")
SEQ_TEMPLATE = "2013_05_28_drive_{drive}_sync"
# test 划分的 valid run 条数（10 条：0009 六条 + 0010 四条）
EXPECTED_RUNS = {"0009": 6, "0010": 4}
# 归档轨迹与 GT 的一致性容差（米 / 无量纲矩阵元素）
GT_TOLERANCE = 3e-3

_NPZ_PATTERN = re.compile(
    r"^kitti360_kitti360_2013_05_28_drive_(?P<drive>\d{4})_sync_r(?P<rig>\d+)_k(?P<k>\d+)\.npz$")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _as_poses(value: Any, label: str) -> np.ndarray:
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or len(poses) < 2:
        raise ValueError(f"{label}: 期望 [L,4,4] 且 L>=2，实际 {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError(f"{label}: 位姿含 NaN/Inf")
    expected_bottom = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), poses[:, 3].shape)
    if not np.allclose(poses[:, 3], expected_bottom, atol=2e-5, rtol=0.0):
        raise ValueError(f"{label}: 齐次矩阵最后一行不是 [0,0,0,1]")
    return poses


def anchor_first(poses: np.ndarray) -> np.ndarray:
    """首帧锚定：T[i] <- inv(T[0]) @ T[i]（只消 gauge，不拟尺度）。"""
    return np.linalg.inv(poses[0])[None] @ poses


def _path_length(poses: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1).sum())


def _weighted(items: Sequence[tuple[float, float]]) -> float:
    total = float(sum(weight for _, weight in items))
    if total <= 0:
        raise ValueError("聚合权重非正")
    return float(sum(value * weight for value, weight in items) / total)


# ---------------------------------------------------------------------------
# 官方 GT 读入
# ---------------------------------------------------------------------------
def _parse_cam0_to_world(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """解析官方 ``data_poses/<seq>/cam0_to_world.txt``：每行 = frame_id + 16 个数（4x4）。"""
    frame_ids, poses = [], []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 17:
                raise ValueError(f"{path}: 期望每行 1+16 个字段，实际 {len(parts)}")
            frame_ids.append(int(parts[0]))
            poses.append(np.asarray(parts[1:], dtype=np.float64).reshape(4, 4))
    order = np.argsort(np.asarray(frame_ids, dtype=np.int64))
    return (np.asarray(frame_ids, dtype=np.int64)[order],
            np.stack(poses)[order])


def load_official_drive(drive: str, processed_root: str | None,
                        data_poses_root: str | None) -> dict[str, np.ndarray]:
    """载入一条 drive 的官方 cam0 世界位姿参考表（frame_id -> world_T_cam0）。"""
    seq = SEQ_TEMPLATE.format(drive=drive)
    if processed_root:
        path = Path(processed_root) / "sequences" / seq / "frames.npz"
        if not path.is_file():
            raise SystemExit(f"缺少预处理索引: {path}")
        with np.load(path, allow_pickle=False) as data:
            missing = {"frame_ids", "official_world_T_cam0"}.difference(data.files)
            if missing:
                raise ValueError(f"{path}: 缺少键 {sorted(missing)}")
            frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
            world = _as_poses(data["official_world_T_cam0"], f"{path}:official_world_T_cam0")
        source = str(path)
    else:
        path = Path(data_poses_root) / seq / "cam0_to_world.txt"
        if not path.is_file():
            raise SystemExit(f"缺少官方位姿文件: {path}")
        frame_ids, world = _parse_cam0_to_world(path)
        world = _as_poses(world, f"{path}:cam0_to_world")
        source = str(path)
    if len(frame_ids) != len(world) or np.any(np.diff(frame_ids) <= 0):
        raise ValueError(f"{source}: frame_ids 非严格递增或与位姿数不符")
    return {"frame_ids": frame_ids, "world_T_cam0": world, "source": source}


def _lookup_world(reference: Mapping[str, Any], source_frame_ids: np.ndarray) -> np.ndarray:
    """按帧号在官方 GT 表里查 world_T_cam0；任一帧缺失即报错（不做插值）。"""
    frame_ids = np.asarray(reference["frame_ids"], dtype=np.int64)
    wanted = np.asarray(source_frame_ids, dtype=np.int64)
    indices = np.searchsorted(frame_ids, wanted)
    in_range = indices < len(frame_ids)
    matched = np.zeros(len(indices), dtype=bool)
    matched[in_range] = frame_ids[indices[in_range]] == wanted[in_range]
    if not np.all(matched):
        raise ValueError(f"官方 GT 缺少帧 {wanted[~matched][:8].tolist()}")
    return np.asarray(reference["world_T_cam0"], dtype=np.float64)[indices]


# ---------------------------------------------------------------------------
# 评测 NPZ 读入
# ---------------------------------------------------------------------------
def load_runs(eval_dir: str, k: int) -> list[dict[str, Any]]:
    """从评测目录读出指定 interval-k 的 10 条 valid run。"""
    runs = []
    for path in sorted(glob.glob(os.path.join(eval_dir, f"*_k{k}.npz"))):
        match = _NPZ_PATTERN.match(os.path.basename(path))
        if not match or int(match.group("k")) != k:
            continue
        with np.load(path, allow_pickle=False) as data:
            if "source_frame_ids" not in data.files:
                raise SystemExit(
                    f"{path}: 缺少 source_frame_ids —— 请用本仓库的 eval_stream.py "
                    "重新评测（--save-poses 会写入该字段）")
            source_ids = np.asarray(data["source_frame_ids"], dtype=np.int64)
            runs.append({
                "unit": f"{match.group('drive')}/f{int(source_ids[0])}",
                "drive": match.group("drive"),
                "start_frame": int(source_ids[0]),
                "T_gt": np.asarray(data["T_gt"], dtype=np.float64),
                "T_est": np.asarray(data["T_est"], dtype=np.float64),
                "source_frame_ids": source_ids,
                "source": path,
            })
    return runs


# ---------------------------------------------------------------------------
# 核心：run 拼 drive、指标
# ---------------------------------------------------------------------------
def stitch_drive_runs(runs: Sequence[Mapping[str, Any]],
                      official_reference: Mapping[str, Any],
                      gt_tolerance: float = GT_TOLERANCE) -> dict[str, Any]:
    """把同一条 drive 的各 run 按时间顺序以官方 GT 相对位姿衔接成一条完整 drive 轨迹。

    对第 i 条 run（GT 起点 ``G_start``，上一条 run 的终点 ``E_prev``/``G_prev``），其估计
    起点定为 ``E_prev @ inv(G_prev) @ G_start``；run 内部的预测形状**逐位不变**。
    """
    ordered = sorted(runs, key=lambda item: int(item["start_frame"]))
    if not ordered:
        raise ValueError("drive 没有可拼接的 run")
    drive = str(ordered[0]["drive"])
    if any(str(run["drive"]) != drive for run in ordered):
        raise ValueError("stitch_drive_runs 收到了多条 drive 的 run")

    first_world = _lookup_world(official_reference, ordered[0]["source_frame_ids"][:1])[0]
    drive_anchor_inv = np.linalg.inv(first_world)

    gt_parts, est_parts, source_parts = [], [], []
    offsets, run_records = [0], []
    previous_est = previous_gt = None

    for run in ordered:
        source_ids = np.asarray(run["source_frame_ids"], dtype=np.int64)
        local_gt = anchor_first(_as_poses(run["T_gt"], f"{run['unit']}:T_gt"))
        local_est = anchor_first(_as_poses(run["T_est"], f"{run['unit']}:T_est"))
        if not (len(source_ids) == len(local_gt) == len(local_est)):
            raise ValueError(
                f"{run['unit']}: 帧号/GT/估计长度为 "
                f"{len(source_ids)}/{len(local_gt)}/{len(local_est)}")
        world = _lookup_world(official_reference, source_ids)
        global_gt = drive_anchor_inv[None] @ world
        # 校验：评测 NPZ 中的 GT 必须与官方 cam0 GT 一致
        official_local_gt = np.linalg.inv(world[0])[None] @ world
        gt_max_abs = float(np.max(np.abs(official_local_gt - local_gt)))
        if gt_max_abs > gt_tolerance:
            raise ValueError(
                f"{run['unit']}: 评测 NPZ 里的 T_gt 与官方 cam0 GT 不一致 "
                f"(max_abs={gt_max_abs:.6g} > {gt_tolerance})")

        estimated_anchor = (global_gt[0] if previous_est is None
                            else previous_est @ np.linalg.inv(previous_gt) @ global_gt[0])
        global_est = estimated_anchor[None] @ local_est

        gt_parts.append(global_gt)
        est_parts.append(global_est)
        source_parts.append(source_ids)
        offsets.append(offsets[-1] + len(source_ids))
        run_records.append({
            "unit": run["unit"],
            "start_frame": int(source_ids[0]),
            "end_frame": int(source_ids[-1]),
            "pose_count": int(len(source_ids)),
            "archived_gt_max_abs_error": gt_max_abs,
            "source": str(run.get("source", "")),
        })
        previous_est, previous_gt = global_est[-1], global_gt[-1]

    T_gt = np.concatenate(gt_parts, axis=0)
    T_est = np.concatenate(est_parts, axis=0)
    return {
        "drive": drive,
        "T_gt": T_gt,
        "T_est": T_est,
        "source_frame_ids": np.concatenate(source_parts),
        "run_offsets": np.asarray(offsets, dtype=np.int64),
        "run_records": run_records,
        "pose_count": int(len(T_gt)),
        "gt_path_length_m": _path_length(T_gt),
    }


def _run_kitti(run: Mapping[str, Any], lengths: Sequence[int]) -> dict[str, Any]:
    T_gt = anchor_first(_as_poses(run["T_gt"], f"{run['unit']}:T_gt"))
    T_est = anchor_first(_as_poses(run["T_est"], f"{run['unit']}:T_est"))
    result = compute_kitti_per_length(list(T_gt), list(T_est), list(lengths), STEP_SIZE)
    overall = result["overall"]
    return {
        "t_rel_percent": float(overall["t_rel_percent"]),
        "r_rel_deg_per_m": float(overall["r_rel_deg_per_m"]),
        "segment_count": int(overall["count"]),
        "per_length": {str(length): result[length] for length in lengths},
    }


def _pool_kitti(rows: Sequence[Mapping[str, Any]], lengths: Sequence[int]) -> dict[str, Any]:
    """10 条 run 的子段按段计数联合池化。"""
    def pool(cells, key: str, count_key: str = "segment_count") -> float:
        pairs = [(float(cell[key]), int(cell[count_key])) for cell in cells
                 if int(cell[count_key]) > 0]
        if not pairs:
            raise ValueError(f"{key} 没有有效 KITTI 子段")
        return float(sum(value * count for value, count in pairs)
                     / sum(count for _, count in pairs))

    out: dict[str, Any] = {
        "t_rel_percent": pool(rows, "t_rel_percent"),
        "r_rel_deg_per_m": pool(rows, "r_rel_deg_per_m"),
        "segment_count": int(sum(int(row["segment_count"]) for row in rows)),
        "per_length": {},
    }
    for length in lengths:
        cells = [row["per_length"][str(length)] for row in rows]
        count = int(sum(int(cell["count"]) for cell in cells))
        out["per_length"][str(length)] = {
            "t_rel_percent": pool(cells, "t_rel_percent", "count") if count else None,
            "r_rel_deg_per_m": pool(cells, "r_rel_deg_per_m", "count") if count else None,
            "count": count,
        }
    return out


def aggregate_k360_drives(runs: Sequence[Mapping[str, Any]],
                          official_references: Mapping[str, Mapping[str, Any]],
                          lengths: Sequence[int] = K360_LENGTHS) -> dict[str, Any]:
    """两条 drive 的聚合（t_rel/r_rel 按段池化，ATE 按 GT 路径长加权）。"""
    by_drive = {drive: [run for run in runs if str(run["drive"]) == drive] for drive in DRIVES}
    missing = [drive for drive, items in by_drive.items() if not items]
    if missing:
        raise ValueError(f"缺少 drive: {missing}")
    for drive, items in by_drive.items():
        if len(items) != EXPECTED_RUNS[drive]:
            raise ValueError(
                f"drive {drive} 应有 {EXPECTED_RUNS[drive]} 条 valid run，实际 {len(items)} 条")

    stitched = {drive: stitch_drive_runs(items, official_references[drive])
                for drive, items in by_drive.items()}

    run_metrics = []
    for run in runs:
        row = _run_kitti(run, lengths)
        row.update(unit=run["unit"], drive=str(run["drive"]))
        run_metrics.append(row)
    pooled = _pool_kitti(run_metrics, lengths)

    drive_rows: dict[str, dict[str, Any]] = {}
    for drive in DRIVES:
        selected = stitched[drive]
        ate = compute_ate(list(selected["T_gt"]), list(selected["T_est"]))
        drive_rows[drive] = {
            "pose_count": selected["pose_count"],
            "n_runs": len(by_drive[drive]),
            "gt_path_length_m": selected["gt_path_length_m"],
            "ate_se3_m": float(ate["se3"]),
            "ate_sim3_m": float(ate["sim3"]),
            "run_records": selected["run_records"],
        }
    ate_se3 = _weighted([(drive_rows[drive]["ate_se3_m"],
                               drive_rows[drive]["gt_path_length_m"]) for drive in DRIVES])
    return {
        "protocol": {
            "version": "streamrig-release-k360-drive-v1",
            "kitti": "valid runs; 100..800 m; step=10; segments pooled over all runs",
            "ate": ("complete-drive ATE-SE3; arithmetic mean weighted by the stitched "
                    "complete-drive GT path length"),
        },
        "kitti": pooled,
        "ate_se3_m": ate_se3,
        "drives": drive_rows,
        "run_metrics": run_metrics,
        "_stitched": stitched,
    }


def archive_aggregate(result: Mapping[str, Any], method_id: str, out_dir: str | Path) -> dict[str, str]:
    """保存用于报告 ATE 的那两条完整 drive 轨迹（供复算）。"""
    method_dir = Path(out_dir) / "trajectories" / method_id
    method_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for drive in DRIVES:
        selected = result["_stitched"][drive]
        path = method_dir / f"drive_{drive}.npz"
        np.savez_compressed(
            path,
            T_est=np.asarray(selected["T_est"], dtype=np.float32),
            T_gt=np.asarray(selected["T_gt"], dtype=np.float32),
            source_frame_ids=np.asarray(selected["source_frame_ids"], dtype=np.int64),
            run_offsets=np.asarray(selected["run_offsets"], dtype=np.int64),
            drive=drive,
            gt_path_length_m=float(selected["gt_path_length_m"]),
            protocol_version=result["protocol"]["version"],
        )
        paths[drive] = str(path.resolve())
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval-dir", required=True,
                        help="eval_stream.py 的 --out-dir/<exp-name>/test")
    parser.add_argument("--processed-root", default=None,
                        help="step1_prepare_kitti360.py 的产物根目录（推荐）")
    parser.add_argument("--data-poses-root", default=None,
                        help="官方 KITTI-360 data_poses 根目录（processed-root 的替代）")
    parser.add_argument("--out-dir", required=True, help="输出目录（summary.json / metrics.csv / trajectories/）")
    parser.add_argument("--k", type=int, required=True,
                        help="interval-k（重锚间隔）")
    parser.add_argument("--method-id", default="streamrig_k360", help="输出轨迹子目录名")
    args = parser.parse_args()

    if not args.processed_root and not args.data_poses_root:
        parser.error("必须给 --processed-root 或 --data-poses-root 之一")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    official = {drive: load_official_drive(drive, args.processed_root, args.data_poses_root)
                for drive in DRIVES}
    runs = load_runs(args.eval_dir, args.k)
    total_expected = sum(EXPECTED_RUNS.values())
    if len(runs) != total_expected:
        raise SystemExit(
            f"{args.eval_dir}: k={args.k} 只找到 {len(runs)}/{total_expected} 条 valid run NPZ")

    result = aggregate_k360_drives(runs, official)
    archives = archive_aggregate(result, args.method_id, out_dir)

    summary = {key: value for key, value in result.items() if not key.startswith("_")}
    summary.update({
        "method_id": args.method_id,
        "k": args.k,
        "N": args.k + 1,
        "eval_dir": str(Path(args.eval_dir).resolve()),
        "official_gt_sources": {drive: official[drive]["source"] for drive in DRIVES},
        "trajectory_archives": archives,
    })
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    row = {
        "method_id": args.method_id,
        "N": args.k + 1,
        "t_rel_percent": result["kitti"]["t_rel_percent"],
        "r_rel_deg_per_m": result["kitti"]["r_rel_deg_per_m"],
        "r_rel_deg_per_100m": result["kitti"]["r_rel_deg_per_m"] * 100.0,
        "ate_m": result["ate_se3_m"],
        "segment_count": result["kitti"]["segment_count"],
        "ate_0009_m": result["drives"]["0009"]["ate_se3_m"],
        "ate_0010_m": result["drives"]["0010"]["ate_se3_m"],
        "gt_length_0009_m": result["drives"]["0009"]["gt_path_length_m"],
        "gt_length_0010_m": result["drives"]["0010"]["gt_path_length_m"],
    }
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    print(f"[{args.method_id}] N={args.k + 1} "
          f"t={row['t_rel_percent']:.6f}% "
          f"r={row['r_rel_deg_per_100m']:.6f} deg/100m "
          f"ATE={row['ate_m']:.6f} m seg={row['segment_count']}")
    print(f"   drive 0009 ATE-SE3={row['ate_0009_m']:.6f} m "
          f"| drive 0010 ATE-SE3={row['ate_0010_m']:.6f} m")
    print(f"[done] -> {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
