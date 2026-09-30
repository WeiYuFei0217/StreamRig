"""
kitti_metrics.py -- 标准 KITTI 里程计相对误差 + ATE(SE3/Sim3 对齐) 的实现。

本模块提供统一评估的两类指标：
  1. KITTI 相对误差（滑窗子段，标准 KITTI odometry devkit 算法的 numpy 移植）：
     对每个段长 L（米），从所有可行起点（每 step_size 帧一个）出发，沿 GT 轨迹累计路径长度
     首次超过 L 的子段，计算 est 与 gt 的相对位姿误差；
        t_rel = mean(平移误差 / L) * 100      单位：百分比 (%)
        r_rel = mean(旋转误差[rad] / L) * 180/pi   单位：deg/m
     KITTI 官方段长为 [100,200,...,800]m、step_size=10 帧。这里段长可配置。
  2. ATE（绝对轨迹误差，RMSE，单位米）：用 evo 的 Umeyama 对齐
        SE3  : correct_scale=False（保留公制尺度，反映真实米制误差）
        Sim3 : correct_scale=True （相似变换含尺度对齐，SLAM/VO 常用做法）

实现：KITTI 相对误差按 KITTI devkit 的 C++ 算法用 numpy 实现；ATE 对齐用 evo 的 Umeyama 实现。

示例：
    from kitti_metrics import compute_kitti_per_length, compute_ate
    kitti = compute_kitti_per_length(poses_gt, poses_est, lengths=[100,200,...,800])
    ate = compute_ate(poses_gt, poses_est)   # {"se3": rmse_m, "sim3": rmse_m}
"""

import numpy as np


# ----------------------------------------------------------------------
# KITTI 相对误差（标准 devkit 算法）
# ----------------------------------------------------------------------
def _trajectory_distances(poses):
    """累计路径长度：dist[i] = sum_{j<i} ||t_j - t_{j-1}||。poses: [L,4,4]。"""
    dist = [0.0]
    for i in range(1, len(poses)):
        d = np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3])
        dist.append(dist[-1] + d)
    return dist


def _last_frame_from_segment_length(dist, first_frame, length):
    """从 first_frame 起，找到第一个使累计路径 > length 的帧索引；找不到返回 -1。"""
    for i in range(first_frame, len(dist)):
        if dist[i] > dist[first_frame] + length:
            return i
    return -1


def _rotation_error(pose_error):
    """4x4 相对误差矩阵的旋转误差（弧度）。"""
    R = pose_error[:3, :3]
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    cos = max(min((trace - 1.0) / 2.0, 1.0), -1.0)
    return float(np.arccos(cos))


def _translation_error(pose_error):
    """4x4 相对误差矩阵的平移误差（米）。"""
    return float(np.linalg.norm(pose_error[:3, 3]))


def calc_sequence_errors(poses_gt, poses_est, lengths, step_size=10):
    """KITTI 逐子段误差。返回 list[(first_frame, r_err_per_m, t_err_ratio, length)]。
    r_err_per_m: 旋转误差[rad]/length；t_err_ratio: 平移误差[m]/length。"""
    errors = []
    dist = _trajectory_distances(poses_gt)
    for first_frame in range(0, len(poses_gt), step_size):
        for length in lengths:
            last_frame = _last_frame_from_segment_length(dist, first_frame, length)
            if last_frame == -1 or last_frame >= len(poses_est):
                continue
            # 相对位姿（est 与 gt 各自 first->last），再求误差
            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
            pose_delta_est = np.linalg.inv(poses_est[first_frame]) @ poses_est[last_frame]
            pose_error = np.linalg.inv(pose_delta_est) @ pose_delta_gt
            r_err = _rotation_error(pose_error)
            t_err = _translation_error(pose_error)
            errors.append((first_frame, r_err / length, t_err / length, length))
    return errors


def compute_kitti_per_length(poses_gt, poses_est, lengths, step_size=10):
    """按段长汇总 KITTI 指标。
    返回 {length: {"t_rel_percent": .., "r_rel_deg_per_m": .., "count": ..}}，
    以及 "overall": 所有段长的子段合并平均（KITTI 官方定义）。"""
    poses_gt = [np.asarray(p, dtype=np.float64) for p in poses_gt]
    poses_est = [np.asarray(p, dtype=np.float64) for p in poses_est]
    errors = calc_sequence_errors(poses_gt, poses_est, lengths, step_size)
    out = {}
    for length in lengths:
        seg = [(r, t) for (_, r, t, L) in errors if L == length]
        if seg:
            r_arr = np.array([r for r, _ in seg])
            t_arr = np.array([t for _, t in seg])
            out[length] = {
                "t_rel_percent": float(t_arr.mean() * 100.0),
                "r_rel_deg_per_m": float(np.degrees(r_arr.mean())),
                "count": len(seg),
            }
        else:
            out[length] = {"t_rel_percent": None, "r_rel_deg_per_m": None, "count": 0}
    # KITTI 官方 overall：所有子段（不分长度）合并平均
    if errors:
        r_all = np.array([r for (_, r, _, _) in errors])
        t_all = np.array([t for (_, _, t, _) in errors])
        out["overall"] = {
            "t_rel_percent": float(t_all.mean() * 100.0),
            "r_rel_deg_per_m": float(np.degrees(r_all.mean())),
            "count": len(errors),
        }
    else:
        out["overall"] = {"t_rel_percent": None, "r_rel_deg_per_m": None, "count": 0}
    return out


# ----------------------------------------------------------------------
# ATE（evo Umeyama 对齐，SE3 / Sim3）
# ----------------------------------------------------------------------
def compute_ate(poses_gt, poses_est):
    """ATE RMSE（米），SE3 与 Sim3 两种对齐。返回 {"se3": .., "sim3": ..}。"""
    from evo.core.trajectory import PosePath3D
    from evo.core import metrics

    poses_gt = [np.asarray(p, dtype=np.float64) for p in poses_gt]
    poses_est = [np.asarray(p, dtype=np.float64) for p in poses_est]

    from evo.core.geometry import GeometryException

    def _ate(correct_scale):
        traj_ref = PosePath3D(poses_se3=poses_gt)
        traj_est = PosePath3D(poses_se3=poses_est)
        try:
            traj_est.align(traj_ref, correct_scale=correct_scale)
        except GeometryException:
            # 退化情形(如近直线轨迹导致 Umeyama 秩亏):改用首帧位姿对齐,不做旋转/尺度拟合。
            T_fix = poses_gt[0] @ np.linalg.inv(poses_est[0])
            traj_est = PosePath3D(poses_se3=[T_fix @ p for p in poses_est])
        ape = metrics.APE(metrics.PoseRelation.translation_part)
        ape.process_data((traj_ref, traj_est))
        return float(ape.get_statistic(metrics.StatisticsType.rmse))

    return {"se3": _ate(False), "sim3": _ate(True)}
