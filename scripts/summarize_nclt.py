#!/usr/bin/env python3
"""summarize_nclt.py -- 把 NCLT 整轨位姿 NPZ 汇总成 100--800 m 相对误差与整日期 ATE。

指标定义
--------
* 评测序列 = test 划分的 2012-02-19 与 2012-08-20 两条**完整**日期，各自是一个独立评测单位；
* 每条日期内部先按 KITTI 相对误差定义在 100/200/.../800 m 段长上做**按段计数加权**的
  overall 池化（段起点步长 step_size=10 帧）；
* 两条日期之间**等权平均**（macro mean）；
* 按时间戳空洞（相邻帧时间差 > 1000 ms）分段，段间以真值相对位姿衔接（见 :func:`restitch`）；
* ATE 在整条日期上计算：SE3 = 只做刚体对齐，Sim3 = 额外拟合一个全局尺度。

输出 ``<eval_dir>/metrics_100_800m.json``。

示例用法
--------
    python scripts/summarize_nclt.py --eval-dir <out>/<exp> --intervals <k>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitti_metrics import compute_ate, compute_kitti_per_length  # noqa: E402

SEQUENCES = ("2012-02-19", "2012-08-20")
LENGTHS_M = (100, 200, 300, 400, 500, 600, 700, 800)
STEP_SIZE = 10
# 认定为"时间戳空洞"的阈值（毫秒）
GAP_MS = 1000


def restitch(estimate: np.ndarray, ground_truth: np.ndarray,
             timestamps: np.ndarray, gap_ms: int = GAP_MS) -> tuple[np.ndarray, int]:
    """按时间戳空洞分段，段间以 GT 相对位姿衔接预测轨迹（无空洞时原样返回）。

    参数
    ----
    estimate:     [L,4,4] 模型预测的锚系绝对位姿
    ground_truth: [L,4,4] 对应的 GT 锚系绝对位姿
    timestamps:   [L] 毫秒时间戳（严格递增）
    gap_ms:       判定空洞的时间差阈值

    返回 ``(stitched, n_segments)``。段内位姿逐位不变，只有段的整体位姿被刚体平移/旋转到
    由 GT 相对位姿确定的起点上。
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.int64)
    if estimate.shape != ground_truth.shape or estimate.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"形状不一致: est={estimate.shape} gt={ground_truth.shape} ts={timestamps.shape}")
    gaps = np.where(np.diff(timestamps) > gap_ms)[0]
    bounds = [0, *[int(index) + 1 for index in gaps], len(estimate)]
    segments = list(zip(bounds[:-1], bounds[1:]))
    stitched = estimate.copy()
    for index in range(1, len(segments)):
        previous_end = segments[index - 1][1] - 1
        current_start = segments[index][0]
        # GT 给出的"跨空洞相对位姿"
        gt_gap = np.linalg.inv(ground_truth[previous_end]) @ ground_truth[current_start]
        desired = stitched[previous_end] @ gt_gap
        shift = desired @ np.linalg.inv(estimate[current_start])
        begin, end = segments[index]
        stitched[begin:end] = shift[None] @ estimate[begin:end]
    return stitched, len(segments)


def sequence_metrics(T_gt: np.ndarray, T_est: np.ndarray) -> dict:
    """单条完整日期的 KITTI 相对误差 + ATE。"""
    per_length = compute_kitti_per_length(
        list(T_gt), list(T_est), list(LENGTHS_M), STEP_SIZE)
    overall = per_length["overall"]
    ate = compute_ate(list(T_gt), list(T_est))
    return {
        "t_rel_percent": overall["t_rel_percent"],
        "r_rel_deg_per_m": overall["r_rel_deg_per_m"],
        "ate_se3_m": ate["se3"],
        "ate_sim3_m": ate["sim3"],
        "segment_count": overall["count"],
        "per_length": {str(length): per_length[length] for length in LENGTHS_M},
    }


def load_sequence(eval_dir: str, sequence: str, k: int, gap_ms: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    """读一条日期的位姿 NPZ，按时间戳空洞分段衔接。"""
    path = os.path.join(eval_dir, f"nclt_nclt_{sequence}_r0_k{k}.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"缺少位姿 NPZ: {path}")
    with np.load(path, allow_pickle=False) as data:
        T_est = np.asarray(data["T_est"], dtype=np.float64)
        T_gt = np.asarray(data["T_gt"], dtype=np.float64)
        timestamps = np.asarray([int(value) for value in data["ts"]], dtype=np.int64)
    T_est, segments = restitch(T_est, T_gt, timestamps, gap_ms)
    return T_gt, T_est, segments, int(T_est.shape[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--eval-dir", required=True,
                        help="eval_stream.py 的 --out-dir/<exp-name>")
    parser.add_argument("--output", default=None, help="默认 <eval-dir>/metrics_100_800m.json")
    parser.add_argument("--stride", type=int, default=3, help="评测帧间隔（仅写入输出元信息）")
    parser.add_argument("--intervals", type=int, nargs="+", required=True,
                        help="interval-k 列表")
    parser.add_argument("--gap-ms", type=int, default=GAP_MS, help="判定时间戳空洞的阈值（毫秒）")
    args = parser.parse_args()

    summary_path = os.path.join(args.eval_dir, "summary.json")
    checkpoint, config = None, None
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as handle:
            summary = json.load(handle)
        checkpoint, config = summary.get("ckpt"), summary.get("config")
        present = {row["traj"] for row in summary.get("per_seq", [])}
        missing = set(SEQUENCES) - present
        if missing:
            raise SystemExit(f"summary.json 缺少 NCLT 序列: {sorted(missing)}")

    grid, per_sequence_grid = {}, {}
    for k in args.intervals:
        N = k + 1
        rows = []
        for sequence in SEQUENCES:
            T_gt, T_est, segments, n_frames = load_sequence(
                args.eval_dir, sequence, k, args.gap_ms)
            rows.append({
                "traj": sequence,
                "N": N,
                "k": k,
                "n_frames": n_frames,
                "gt_gap_segments": segments,
                **sequence_metrics(T_gt, T_est),
            })
        per_sequence_grid[str(N)] = rows
        grid[str(N)] = {
            "N": N,
            "k": k,
            "t_rel_percent": mean(row["t_rel_percent"] for row in rows),
            "r_rel_deg_per_m": mean(row["r_rel_deg_per_m"] for row in rows),
            "ate_se3_m": mean(row["ate_se3_m"] for row in rows),
            "ate_sim3_m": mean(row["ate_sim3_m"] for row in rows),
            "n_sequences": len(rows),
        }

    artifact = {
        "dataset": "NCLT test: 2012-02-19 + 2012-08-20",
        "stride": args.stride,
        "lengths_m": list(LENGTHS_M),
        "step_size_frames": STEP_SIZE,
        "aggregation": "segments pooled within each complete date, then equal-weight mean over the two dates",
        "gap_protocol": (f"segments split at timestamp gaps > {args.gap_ms} ms are joined "
                         "with the ground-truth relative pose"),
        "checkpoint": checkpoint,
        "config": config,
        "intervals": list(args.intervals),
        "N": [k + 1 for k in args.intervals],
        "grid": grid,
        "per_sequence": per_sequence_grid,
    }
    output = args.output or os.path.join(args.eval_dir, "metrics_100_800m.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, indent=2, ensure_ascii=False)
    for N, row in grid.items():
        print(f"[summarize] N={N:>3} t_rel={row['t_rel_percent']:.4f}% "
              f"r_rel={row['r_rel_deg_per_m'] * 100:.4f} deg/100m "
              f"ATE-SE3={row['ate_se3_m']:.4f} m")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
