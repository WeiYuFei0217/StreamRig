"""
eval_stream.py -- 完整序列流式评测（KITTI 相对误差 + ATE）。

对一个 ckpt，在未参与训练的 test 序列上做完整流式评估：
  - 数据：NCLT test = 2012-02-19 / 2012-08-20；KITTI-360 test = drive 0009 / 0010。
    两个数据集缺省 stride=3。
  - interval-k 拼接：模型窗口内输出锚系 T_0->t；拼长轨迹时每 k 帧重锚一次（重锚块 N=k+1）。
    每个重锚块做一次因果整窗前向；由于组级因果 mask，这与逐 rig KV 缓存增量推理
    数学等价（见 tests/test_cache_equiv.py）。每个 k 得一条完整估计轨迹。
  - 指标（每 seq × k）：
      KITTI 相对误差（滑窗子段，段长 [100,200,..,800] 米，每段长 t_rel[%] / r_rel[deg/m]）
      ATE = SE3 对齐 RMSE(米) + Sim3 对齐 RMSE(米)。
  - 落盘：out-dir/<expname>/<dataset>_<scene>_<traj>_r<rs>_k<k>.json（各段长+ATE）
          + out-dir/<expname>/summary.json（NCLT）或
            out-dir/<expname>/test/summary.json（KITTI-360）。

示例（NCLT：config 的 step1_root / step2_root / infoshare_root 指向 test 数据）：
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_stream.py \
        --config configs/nclt_streamrig.yaml \
        --ckpt release_weights/NCLT-StreamRig.pth \
        --exp-name streamrig_nclt --out-dir outputs/eval --intervals 23
    # 快速检查（限制每序列帧数）：加 --max-len 200
    # 一键评测与指标汇总见 scripts/eval_nclt.sh / scripts/eval_kitti360.sh
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

# 仓库根 = 本脚本所在 scripts/ 的上一级；加入 sys.path，使得直接 `python scripts/xxx.py`
# 运行时能 import 到同仓库的 streamrig 包（无需先 pip install）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from streamrig.config import build_stream_model  # noqa: E402
from streamrig.datasets import Kitti360StreamDataset, StreamOdomDataset  # noqa: E402
from streamrig.datasets.stream_odom_dataset import INFOSHARE_DIM  # noqa: E402
from kitti_metrics import compute_kitti_per_length, compute_ate  # noqa: E402

# KITTI 相对误差的子段长度（米）
LENGTHS_M = [100, 200, 300, 400, 500, 600, 700, 800]
# KITTI-360 test 划分应包含的 drive（与 data_preprocessing/kitti360/step1 的划分一致）
K360_SPLIT_SEQUENCES = {
    "test": (
        "2013_05_28_drive_0009_sync",
        "2013_05_28_drive_0010_sync",
    ),
}


def build_T(R, t):
    T = torch.eye(4, device=R.device, dtype=R.dtype).repeat(R.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def infer_dataset_kind(requested, dcfg):
    """解析数据集类型；auto 时按 config 的 ``dataset_type`` 判断（缺省为 NCLT）。"""
    if requested != "auto":
        return requested
    return "kitti360" if dcfg.get("dataset_type") == "kitti360" else "nclt"


def check_k360_split_sequences(ds, split):
    """检查 KITTI-360 split 实际包含的序列与预期一致。"""
    expected = list(K360_SPLIT_SEQUENCES[split])
    actual = [seq["traj"] for seq in ds.seqs]
    if actual != expected:
        raise ValueError(
            f"KITTI-360 split={split} 应含 {expected}，实际为 {actual}"
        )


def _is_kitti360(ds):
    return isinstance(ds, Kitti360StreamDataset)


def _load_sequence_features(ds, scene_id, traj_id, ts_list):
    """载入一段序列的 info-sharing 特征 [L,K,256,C]；缺帧直接报错。"""
    L, K = len(ts_list), ds.K
    if _is_kitti360(ds):
        # K360 loader 的 key 是十位 frame-id；缺帧直接报错。
        return torch.stack([ds._load_infoshare(traj_id, key) for key in ts_list], dim=0)

    # 整条序列的特征预载到内存（float32）
    feats = torch.zeros(L, K, 256, INFOSHARE_DIM, dtype=torch.float32)
    for ti, ts in enumerate(ts_list):
        feats[ti] = ds._load_infoshare(scene_id, traj_id, ts)
    return feats


def load_full_sequence(ds, si, rs, re, stride, max_len):
    """载入一条完整评测序列（不像训练那样只取一个窗口）。
    返回 feats[L,K,256,C]、T_gt[L,4,4]（锚系）, meta=(scene, traj, L, ts_list, idxs)。"""
    seq = ds.seqs[si]
    scene_id, traj_id = seq["scene"], seq["traj"]
    idxs = list(range(rs, re, stride))
    if max_len > 0:
        idxs = idxs[:max_len]
    if _is_kitti360(ds):
        ts_list = [f"{int(seq['frame_ids'][i]):010d}" for i in idxs]
        bT0 = ds.body_T_cams[0]
        world_T_cam = seq["world_T_pose"][idxs] @ bT0
        T_w_ref0_inv = np.linalg.inv(world_T_cam[0])
        T_gt = (T_w_ref0_inv[None] @ world_T_cam).astype(np.float32)
    else:
        poses = seq["poses"]
        ts_list = [seq["ts"][i] for i in idxs]
        body_T_cams, _ = ds._get_calib(scene_id)
        bT0 = body_T_cams[0]
        T_w_ref0_inv = np.linalg.inv(poses[ts_list[0]] @ bT0)
        T_gt = np.stack([T_w_ref0_inv @ (poses[t] @ bT0) for t in ts_list], 0).astype(np.float32)
    L = len(ts_list)

    feats = _load_sequence_features(ds, scene_id, traj_id, ts_list)
    return feats, torch.from_numpy(T_gt), (scene_id, traj_id, L, ts_list, idxs)


@torch.no_grad()
def predict_window(model, feats_win, device):
    """一窗前向 → 窗内各帧相对局部锚(帧0)的位姿 [Nw,4,4]（T[0]=I）。"""
    batch = {"infoshare": feats_win.unsqueeze(0).to(device).float()}
    out = model(batch)
    return build_T(out["traj_R"][0], out["traj_t"][0])


@torch.no_grad()
def accumulate_interval_k(model, feats, k, device, w_max=12):
    """interval-k 累计：每 k 帧重锚，窗口 [a,a+k]（N=k+1≤w_max），逐窗前向并复合相对位姿。"""
    L = feats.shape[0]
    assert k >= 1, f"interval-k 必须 ≥1，收到 {k}"
    assert k + 1 <= w_max, f"k+1={k+1} 超过窗口上限 {w_max}"
    T_glob = [None] * L
    T_glob[0] = torch.eye(4, device=device)
    a = 0
    while a < L - 1:
        end = min(a + k, L - 1)
        Twin = predict_window(model, feats[a:end + 1], device)  # rel to a
        for j in range(1, end - a + 1):
            T_glob[a + j] = T_glob[a] @ Twin[j]
        a = end
    return torch.stack(T_glob, 0)  # [L,4,4]


def build_eval_dataset(dcfg, ds_kind, split, gap_threshold_ms):
    """构建完整序列评测 dataset（NCLT 的 step1/step2/infoshare 根须指向 test 数据）。"""
    if ds_kind == "kitti360":
        return Kitti360StreamDataset(
            processed_root=dcfg["processed_root"],
            infoshare_root=dcfg["infoshare_root"],
            split=split,
            num_rigs=dcfg["num_rigs"],
            stride_min=dcfg.get("stride_min", 1),
            stride_max=dcfg.get("stride_max", 6),
            gap_threshold_ms=gap_threshold_ms,
            min_run_span_ms=dcfg.get("min_run_span_ms", 30000),
            samples_per_epoch=dcfg.get("samples_per_epoch", 48000),
            image_size=dcfg.get("image_size", 224),
        )

    return StreamOdomDataset(
        step1_root=dcfg["step1_root"], step2_root=dcfg["step2_root"],
        infoshare_root=dcfg["infoshare_root"],
        num_rigs=dcfg["num_rigs"], num_cameras=dcfg["num_cameras"],
        gap_threshold_ms=gap_threshold_ms,
        min_run_span_ms=dcfg.get("min_run_span_ms", 30000),
    )


def trajectory_time_metadata(ds, si, idxs, ts_list):
    """NPZ 时间/帧号元数据（NCLT：毫秒时间戳；KITTI-360：纳秒时间戳与官方帧号）。"""
    if not _is_kitti360(ds):
        return {
            "anchor_ts": str(ts_list[0]),
            "ts": np.array([str(t) for t in ts_list]),
        }
    seq = ds.seqs[si]
    frame_ids = seq["frame_ids"][idxs].astype(np.int64)
    timestamps_ns = seq["timestamps_ns"][idxs, 0].astype(np.int64)
    return {
        "anchor_ts": str(int(timestamps_ns[0])),
        "ts": timestamps_ns.astype(str),
        "timestamps_ns": timestamps_ns,
        "source_frame_ids": frame_ids,
    }


def main():
    ap = argparse.ArgumentParser(description="StreamRig 完整序列流式评测（KITTI 相对误差 + ATE）")
    ap.add_argument("--config", required=True, help="模型/数据配置 yaml")
    ap.add_argument("--ckpt", required=True, help="模型权重（发布权重或训练 checkpoint）")
    ap.add_argument("--exp-name", required=True, help="输出子目录名")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--dataset", default="auto", choices=["auto", "nclt", "kitti360"],
                    help="auto = 按 config 的 data.dataset_type 判断")
    ap.add_argument(
        "--processed-root",
        default=None,
        help="覆盖 config 的 KITTI-360 预处理元数据根目录（实际值写入 summary.json）",
    )
    ap.add_argument(
        "--infoshare-root",
        default=None,
        help="覆盖 config 的 info-sharing 缓存根（实际值写入 summary.json）",
    )
    ap.add_argument("--split", choices=["test"], default="test",
                    help="KITTI-360 评测划分（test = drive 0009 + 0010）；NCLT 的评测数据由 config 路径决定")
    ap.add_argument("--stride", type=int, default=-1,
                    help="评测帧间隔（-1 = 默认 3）")
    ap.add_argument("--max-len", type=int, default=-1, help="每序列最大帧数（-1 = 完整序列）")
    ap.add_argument("--intervals", type=int, nargs="+", required=True,
                    help="interval-k 列表（重锚间隔，窗口 N=k+1）")
    ap.add_argument("--w-max", type=int, default=-1,
                    help="重锚窗口上限 N，用于校验 intervals（-1 = 按数据集默认）")
    ap.add_argument("--step-size", type=int, default=10, help="KITTI 子段起点步长(帧)")
    ap.add_argument("--gap-threshold-ms", type=int, default=-1,
                    help="覆盖 config 的 gap_threshold_ms（-1 = 用 config）；设为极大值（如 100000000）"
                         "时每条序列整条评测，不切分")
    ap.add_argument("--save-poses", action=argparse.BooleanOptionalAction, default=True,
                    help="逐 seq×k 保存预测/真值轨迹 npz（T_est/T_gt 锚系位姿 [L,4,4] + 元数据），"
                         "供汇总脚本计算指标。默认开启。")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg, dcfg = cfg["model"], cfg["data"]
    configured_processed_root = dcfg.get("processed_root")
    configured_infoshare_root = dcfg.get("infoshare_root")
    if args.processed_root:
        dcfg["processed_root"] = args.processed_root
    if args.infoshare_root:
        dcfg["infoshare_root"] = args.infoshare_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp32 矩阵乘使用 TF32（发布模型的训练与评测均在此设置下进行）
    torch.backends.cuda.matmul.allow_tf32 = True

    ds_kind = infer_dataset_kind(args.dataset, dcfg)
    split = args.split
    stride = args.stride if args.stride > 0 else 3
    lengths = list(LENGTHS_M)

    split_text = f" split={split}" if ds_kind == "kitti360" else ""
    print(f"[eval] exp={args.exp_name} dataset={ds_kind}{split_text} stride={stride} "
          f"device={device} 段长={lengths}", flush=True)

    # 模型
    model = build_stream_model(mcfg, device)
    ck = torch.load(args.ckpt, map_location="cpu")
    res = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    if res.missing_keys or res.unexpected_keys:
        raise ValueError(f"Checkpoint/config mismatch: missing={res.missing_keys}, "
                         f"unexpected={res.unexpected_keys}")
    print("[eval] ckpt 加载: 键完全匹配", flush=True)
    model.eval()

    # 完整序列 dataset；K360 的评测 drive 由 split 指定。
    default_gap_ms = 500 if ds_kind == "kitti360" else 400
    _gap_ms = (args.gap_threshold_ms if args.gap_threshold_ms > 0
               else dcfg.get("gap_threshold_ms", default_gap_ms))
    ds = build_eval_dataset(dcfg, ds_kind, split, _gap_ms)
    if ds_kind == "kitti360":
        check_k360_split_sequences(ds, split)
    print(f"[eval] gap_threshold_ms={_gap_ms}", flush=True)
    w_max = args.w_max if args.w_max > 0 else (48 if ds_kind == "kitti360" else dcfg["num_rigs"])
    if (len(set(args.intervals)) != len(args.intervals)
            or any(k < 1 or k + 1 > w_max for k in args.intervals)):
        raise SystemExit(
            f"[eval] intervals 必须是 [1,{w_max - 1}] 内的无重复整数，收到 {args.intervals}")
    print(f"[eval] w_max={w_max} (训练窗 num_rigs={dcfg['num_rigs']}), intervals={args.intervals}", flush=True)

    # 枚举所有 (si, rs, re) 评测 run
    runs = []
    for si, s in enumerate(ds.seqs):
        for (rs, re) in s["runs"]:
            if (re - rs) >= 2 * stride:   # 至少 2 帧
                runs.append((re - rs, si, rs, re))
    runs.sort(reverse=True)   # 按长度降序
    print(f"[eval] 共 {len(runs)} 条 {split} run 待评估(完整轨迹,按 GT 距离切段)", flush=True)

    # K360 结果写在 split 子目录下。
    out_exp = (os.path.join(args.out_dir, args.exp_name, split)
               if ds_kind == "kitti360" else os.path.join(args.out_dir, args.exp_name))
    os.makedirs(out_exp, exist_ok=True)
    summary_args = dict(vars(args))
    if ds_kind != "kitti360":
        # --split 只对 K360 有意义。
        summary_args.pop("split", None)
    summary = {"exp": args.exp_name, "dataset": ds_kind, "stride": stride,
               "ckpt": args.ckpt, "config": args.config, "args": summary_args,
               "configured_processed_root": configured_processed_root,
               "effective_processed_root": dcfg.get("processed_root"),
               "configured_infoshare_root": configured_infoshare_root,
               "effective_infoshare_root": dcfg.get("infoshare_root"),
               "gap_threshold_ms": _gap_ms, "step_size": args.step_size,
               "lengths": lengths, "intervals": args.intervals,
               "per_seq": []}
    if ds_kind == "kitti360":
        summary.update({
            "split": split,
            "effective_w_max": w_max,
            "k360_sequences": list(K360_SPLIT_SEQUENCES[split]),
        })
    output_context = {"split": split} if ds_kind == "kitti360" else {}

    for (rlen, si, rs, re) in runs:
        feats, T_gt, meta = load_full_sequence(ds, si, rs, re, stride, args.max_len)
        scene_id, traj_id, L, ts_list, idxs = meta
        time_metadata = trajectory_time_metadata(ds, si, idxs, ts_list)
        T_gt_np = T_gt.numpy().astype(np.float64)
        print(f"  [{scene_id}/{traj_id}] L={L} 帧", flush=True)
        seq_rec = {"scene": scene_id, "traj": traj_id, "L": L, "k": {}}
        for k in args.intervals:
            try:
                T_est = accumulate_interval_k(
                    model, feats, k, device, w_max=w_max)
            except (RuntimeError, torch.cuda.OutOfMemoryError, AssertionError, TimeoutError) as e:
                # 某个 k 前向失败时记录 skipped 并继续评测其余 k
                msg = str(e)[:120]
                print(f"    [k={k} N={k+1} 跳过] {type(e).__name__}: {msg}", flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                skipped = {"skipped": True, "reason": f"{type(e).__name__}: {msg}"}
                seq_rec["k"][str(k)] = skipped
                fn = f"{ds_kind}_{scene_id}_{traj_id}_r{rs}_k{k}.json".replace("/", "_")
                with open(os.path.join(out_exp, fn), "w") as f:
                    json.dump({"scene": scene_id, "traj": traj_id, "rs": rs,
                               "L": L, "k": k, **output_context, **skipped}, f, indent=1)
                continue
            T_est_np = T_est.cpu().numpy().astype(np.float64)
            kitti = compute_kitti_per_length(T_gt_np, T_est_np, lengths, args.step_size)
            try:
                ate = compute_ate(T_gt_np, T_est_np)
            except Exception as e:
                ate = {"se3": None, "sim3": None, "err": str(e)}
            rec = {"kitti": kitti, "ate": ate}
            seq_rec["k"][str(k)] = rec
            # 逐 seq×k 落盘。文件名带起始帧 rs，区分同 scene/traj 的多条 run。
            fn = f"{ds_kind}_{scene_id}_{traj_id}_r{rs}_k{k}.json".replace("/", "_")
            with open(os.path.join(out_exp, fn), "w") as f:
                json.dump({"scene": scene_id, "traj": traj_id, "rs": rs, "L": L,
                           "k": k, **output_context, **rec}, f, indent=1)
            # 保存预测/真值轨迹（锚系位姿 [L,4,4]，float32），供 summarize_nclt.py /
            # k360_drive_metrics.py 汇总指标。
            if args.save_poses:
                pose_fn = f"{ds_kind}_{scene_id}_{traj_id}_r{rs}_k{k}.npz".replace("/", "_")
                np.savez_compressed(
                    os.path.join(out_exp, pose_fn),
                    T_est=T_est_np.astype(np.float32), T_gt=T_gt_np.astype(np.float32),
                    scene=str(scene_id), traj=str(traj_id), rs=int(rs), L=int(L), k=int(k),
                    stride=int(stride), **output_context, **time_metadata,
                )
        summary["per_seq"].append(seq_rec)

    with open(os.path.join(out_exp, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[eval] 逐 run 结果 -> {out_exp}（报告指标由 summarize_nclt.py / "
          f"k360_drive_metrics.py 汇总）", flush=True)


if __name__ == "__main__":
    main()
