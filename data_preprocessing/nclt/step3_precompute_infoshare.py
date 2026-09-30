#!/usr/bin/env python3
"""
step3_precompute_infoshare.py -- 用 frozen MapAnything 预计算 NCLT 每个 rig 的 info_sharing 特征缓存。

功能:
  对 step2 枚举出的每一个 rig(同一毫秒的 5 台相机), 跑完整的 frozen 感知前端:
      1068x1068 原图 -> cv2 图像管线 -> DINOv2 编码(fp32)
      -> 注入 rig 几何(射线方向 + 相机位姿) -> info_sharing(24 层 alternating-attention)
  得到 [K, 256, 768] 的 token 特征, 以 bf16(uint16 位存)落盘:
      <out-root>/scenes/nclt/trajectories/<date>/features/<ts_ms>/rig.npy
  训练/评测直接读盘, 不再在线运行 backbone。
  冻结前端对同一 rig 的输出是确定的, 因此可以预先缓存。

图像管线:
  cv2.imread(BGR) -> cv2.rotate(ROTATE_90_CLOCKWISE) -> 按该相机焦距做 FOV90 center crop
  (crop = round(2 * focal_length_1068)) -> cv2.resize(224, INTER_AREA) -> BGR2RGB
  -> ToTensor 得到 [0,1] 的图像, 由 backbone 内部按 data_norm_type="dinov2" 归一化。
  输入为 step1 的 1068x1068 JPEG。

整条链路走 fp32(不开 autocast), 只在落盘时转为 bf16。

多卡/多进程分片: 起 `--world` 个进程(每进程占一张卡), 每个进程处理全局 rig 序号
  index % world == rank 的那些; 断点续跑靠"输出文件已存在则跳过"。
  每个 rank 结束时写 `.rank<r>_of_<w>.done`, 最后一个完成的 rank 负责写 `COMPLETE`。

依赖: torch, opencv-python, torchvision, numpy, mapanything(>=1.0) 及其 uniception 依赖。
      不依赖 streamrig 训练包。

示例用法:
    # 单卡, test 划分全量
    CUDA_VISIBLE_DEVICES=0 python step3_precompute_infoshare.py \
        --centered-root /data/nclt_centered \
        --step1-root /data/nclt_meta/step1_rig_trajectories_test \
        --step2-root /data/nclt_meta/step2_rig5_configs_test \
        --model-path /path/to/map-anything-model \
        --out-root /data/nclt_infoshare_test --split test --selfcheck

    # 3 卡分片跑训练集(每卡一个进程)
    for r in 0 1 2; do
      CUDA_VISIBLE_DEVICES=$r python step3_precompute_infoshare.py \
          --centered-root /data/nclt_centered \
          --step1-root /data/nclt_meta/step1_rig_trajectories_train \
          --step2-root /data/nclt_meta/step2_rig5_configs_train \
          --model-path /path/to/map-anything-model \
          --out-root /data/nclt_infoshare_train --split train --rank $r --world 3 &
    done; wait

输出布局:
    <out-root>/scenes/nclt/trajectories/<date>/features/<ts_ms>/rig.npy   # [5,256,768] bf16(uint16)
    <out-root>/manifest.json                                             # 生成配置
    <out-root>/COMPLETE                                                  # 全量完成标记
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import cv2
import numpy as np
import torch
from torchvision import transforms

# 允许 `python step3_precompute_infoshare.py` 直接跑(把 data_preprocessing 的父目录塞进 sys.path)
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from data_preprocessing.common.infoshare_io import (  # noqa: E402
    load_rig_features, rig_feature_path, save_rig_features,
    write_complete_marker, write_manifest,
)
from data_preprocessing.common.mapanything_frontend import load_frozen_mapanything  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

SCENE_ID = "nclt"
CAMERA_IDS = [1, 2, 3, 4, 5]
IMAGE_SIZE = 224
PATCH_SIZE = 14
NUM_TOKENS = (IMAGE_SIZE // PATCH_SIZE) ** 2      # 16*16 = 256
TARGET_HFOV_DEG = 90.0                            # center crop 的目标水平 FOV

TRAIN_SESSIONS = [
    "2012-01-08", "2012-02-02", "2012-02-04", "2012-03-17",
    "2012-05-26", "2012-10-28", "2012-11-17", "2013-04-05",
]
TEST_SESSIONS = ["2012-02-19", "2012-08-20"]
SPLITS = {"train": TRAIN_SESSIONS, "test": TEST_SESSIONS}

_TO_TENSOR = transforms.ToTensor()


# ---------------------------------------------------------------------------
# MapAnything 加载
# ---------------------------------------------------------------------------

def build_backbone(model_path: str, device: str):
    """加载 frozen MapAnything, 返回 (model, info_sharing 维度)。"""
    model = load_frozen_mapanything(model_path, device)
    return model, int(model.info_sharing.dim)


def _geometric_input_config_is_deterministic(model) -> tuple[bool, dict]:
    """检查 geometric_input dropout 概率是否已退化为确定(推理期应当如此)。"""
    cfg = dict(getattr(model, "geometric_input_config", {}) or {})
    ok = (float(cfg.get("overall_prob", 0.0)) >= 1.0
          and float(cfg.get("dropout_prob", 1.0)) <= 0.0
          and float(cfg.get("ray_dirs_prob", 0.0)) >= 1.0
          and float(cfg.get("cam_prob", 0.0)) >= 1.0)
    return ok, cfg


# ---------------------------------------------------------------------------
# 图像管线
# ---------------------------------------------------------------------------

def fov_crop_size(focal_length_1068: float, hfov_deg: float = TARGET_HFOV_DEG) -> int:
    """1068 原图上裁出指定水平 FOV 所需的边长: round(2 * f * tan(hfov/2))。"""
    return int(round(2.0 * focal_length_1068 * math.tan(math.radians(hfov_deg / 2.0))))


def load_and_preprocess(img_path: str, crop_size: int) -> np.ndarray:
    """1068 原图 -> 顺时针转 90° -> center crop -> resize 224 (INTER_AREA) -> RGB uint8。"""
    img = cv2.imread(img_path)
    if img is None:
        raise OSError(f"图像读取失败: {img_path}")
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    off = (img.shape[0] - crop_size) // 2
    crop = img[off:off + crop_size, off:off + crop_size]
    resized = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_rig_images(paths: list[str], crop_sizes: list[int]) -> torch.Tensor:
    """读一个 rig 的 K 张图 -> [1, K, 3, 224, 224] float32, 值域 [0,1] 未做外部归一化。"""
    imgs = [_TO_TENSOR(load_and_preprocess(p, cs)) for p, cs in zip(paths, crop_sizes)]
    return torch.stack(imgs).unsqueeze(0)


# ---------------------------------------------------------------------------
# 几何 / 前向
# ---------------------------------------------------------------------------

def load_rig_calibration(step2_root: str):
    """读 rig_config.json -> (extrinsics [K,4,4] f32, intrinsics [K,3,3] f32, focal_1068 [K])。

    extrinsics = inv(body_T_cam[0]) @ body_T_cam[i] (以 0 号相机为参考系, 与训练加载器相同)。
    """
    path = os.path.join(step2_root, "scenes", SCENE_ID, "rig_config.json")
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cams = cfg["cameras"]
    body_T_cam = [np.array(c["body_T_cam"], dtype=np.float64).reshape(4, 4) for c in cams]
    ref_inv = np.linalg.inv(body_T_cam[0])
    extr = np.stack([ref_inv @ m for m in body_T_cam], 0).astype(np.float32)
    intr = np.stack([np.array(c["intrinsics"], dtype=np.float32) for c in cams], 0)
    focal = [float(c["focal_length_1068"]) for c in cams]
    return extr, intr, focal


@torch.no_grad()
def perceive_rig(model, imgs: torch.Tensor, extr: torch.Tensor, intr: torch.Tensor) -> torch.Tensor:
    """一个 rig 的完整 frozen 前向 -> [K, 256, C] fp32。

    imgs: [1,K,3,224,224] raw [0,1]; extr: [1,K,4,4]; intr: [1,K,3,3]。
    """
    from mapanything.utils.geometry import (
        get_rays_in_camera_frame, rotation_matrix_to_quaternion,
    )
    from uniception.models.info_sharing.cross_attention_transformer import (
        MultiViewTransformerInput,
    )

    device = imgs.device
    B, K = imgs.shape[0], imgs.shape[1]

    # 1) DINOv2 编码(fp32, 不开 autocast); views 只需 img + data_norm_type
    enc_views = [{"img": imgs[:, i], "data_norm_type": ["dinov2"] * B} for i in range(K)]
    enc = model._encode_n_views(enc_views)
    if isinstance(enc, (tuple, list)) and len(enc) and not torch.is_tensor(enc[0]):
        enc = enc[0]                                    # 返回值可能再包一层 tuple
    enc_list = []
    for f in enc:
        while isinstance(f, (tuple, list)):
            f = f[0]
        enc_list.append(f if f.dim() == 4 else f.unsqueeze(0))     # [B,C,H_p,W_p]

    # 2) 组几何 views: 射线方向 + 相机位姿(rig 外参) + 公制尺度标志
    quats = rotation_matrix_to_quaternion(extr[:, :, :3, :3])      # [B,K,4]
    trans = extr[:, :, :3, 3]                                      # [B,K,3]
    ray_dirs = [get_rays_in_camera_frame(
        intrinsics=intr[:, i], height=IMAGE_SIZE, width=IMAGE_SIZE,
        normalize_to_unit_sphere=True)[1] for i in range(K)]
    true_shape = torch.tensor([[IMAGE_SIZE, IMAGE_SIZE]], device=device).expand(B, -1).contiguous()
    is_metric = torch.ones(B, dtype=torch.bool, device=device)
    views = [{
        "img": imgs[:, i],
        "ray_directions_cam": ray_dirs[i],
        "camera_pose_quats": quats[:, i],
        "camera_pose_trans": trans[:, i],
        "is_metric_scale": is_metric,
        "data_norm_type": ["dinov2"] * B,
        "true_shape": true_shape,
    } for i in range(K)]

    # 3) 注入几何 + info_sharing(全程 fp32)
    with torch.autocast("cuda", enabled=False):
        fused = model._encode_and_fuse_optional_geometric_inputs(views, enc_list)
    scale_token = model.scale_token.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1)
    info_out = model.info_sharing(
        MultiViewTransformerInput(features=list(fused), additional_input_tokens=scale_token))
    feats_list = info_out[0].features if isinstance(info_out, tuple) else info_out.features

    # 4) 每个 view [B,C,H_p,W_p] -> [B, H_p*W_p, C], 堆成 [B,K,256,C]
    per_view = [v.permute(0, 2, 3, 1).reshape(v.shape[0], -1, v.shape[1]) for v in feats_list]
    return torch.stack(per_view, dim=1)[0].float()                  # [K,256,C]


# ---------------------------------------------------------------------------
# rig 枚举
# ---------------------------------------------------------------------------

def read_tum_ts_keys(tum_path: str) -> list[str]:
    """读 trajectory.tum, 返回 13 位毫秒键列表(与训练加载器 _load_tum_ms 的键格式相同)。"""
    keys = []
    with open(tum_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            keys.append(f"{int(round(float(parts[0]) * 1000)):010d}")
    return keys


def build_ms_index(centered_root: str, session: str, cam_id: int) -> dict[str, str]:
    """一次 listdir 建 {13 位毫秒前缀: 完整文件名} 索引。

    NCLT 图像文件名是 16 位微秒; 每个相机目录只 listdir 一次, 按 13 位毫秒前缀建表查找。
    """
    cam_dir = os.path.join(centered_root, session, "lb3_centered", f"Cam{cam_id}")
    if not os.path.isdir(cam_dir):
        raise FileNotFoundError(f"找不到 {cam_dir}")
    index = {}
    for name in os.listdir(cam_dir):
        if name.endswith(".jpg"):
            index.setdefault(name[:13], name)
    return index


def enumerate_rigs(centered_root: str, step1_root: str, sessions: list[str] | None):
    """枚举 (session, ts_key, [5 个相机的图像路径]); 只保留 5 台相机都有图的时刻。"""
    traj_root = os.path.join(step1_root, "scenes", SCENE_ID, "trajectories")
    if not os.path.isdir(traj_root):
        raise FileNotFoundError(f"找不到 {traj_root} (先跑 step2_prepare_rig.py)")
    found = sorted(os.listdir(traj_root))
    use = [s for s in found if sessions is None or s in sessions]

    rigs, n_skip, used = [], 0, []
    for session in use:
        tum = os.path.join(traj_root, session, "trajectory.tum")
        if not os.path.isfile(tum):
            continue
        try:
            idx = {c: build_ms_index(centered_root, session, c) for c in CAMERA_IDS}
        except FileNotFoundError as e:
            # 该 session 的 step1 图像不完整时跳过
            print(f"[WARN] 跳过 session {session}: {e}", flush=True)
            continue
        for ts in read_tum_ts_keys(tum):
            names = [idx[c].get(ts) for c in CAMERA_IDS]
            if any(n is None for n in names):
                n_skip += 1
                continue
            rigs.append((session, ts, [
                os.path.join(centered_root, session, "lb3_centered", f"Cam{c}", n)
                for c, n in zip(CAMERA_IDS, names)]))
        used.append(session)
    return rigs, used, n_skip


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _rank_marker(out_root: str, rank: int, world: int) -> str:
    return os.path.join(out_root, f".rank{rank}_of_{world}.done")


def _mark_complete_if_ready(out_root, rigs, rank, world, K, dim):
    """所有 rank 结束且每个应有 rig 文件存在后，才发布 COMPLETE。"""
    marker = _rank_marker(out_root, rank, world)
    open(marker, "w").close()
    if not all(os.path.exists(_rank_marker(out_root, r, world)) for r in range(world)):
        return False
    missing = [f"{session}/{ts}" for session, ts, _ in rigs
               if not os.path.isfile(rig_feature_path(out_root, SCENE_ID, session, ts))]
    if missing:
        raise RuntimeError(f"缓存缺少 {len(missing)} 个 rig（前 5 个: {missing[:5]}）；不写 COMPLETE")
    write_complete_marker(out_root, f"{len(rigs)} rigs, K={K}, dim={dim}, world={world}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="NCLT info_sharing 特征缓存预计算 (frozen MapAnything)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--centered-root", required=True, help="step1 输出根(1068x1068 jpg)")
    ap.add_argument("--step1-root", required=True, help="step2 的 step1_rig_trajectories_<split> 目录")
    ap.add_argument("--step2-root", required=True, help="step2 的 step2_rig5_configs_<split> 目录")
    ap.add_argument("--model-path", required=True, help="MapAnything 权重目录(from_pretrained)")
    ap.add_argument("--out-root", required=True, help="特征缓存输出根")
    ap.add_argument("--split", default="auto", choices=["train", "test", "auto"],
                    help="auto = step1-root 下找到的全部 session")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理本 rank 的前 N 个 rig(用于快速检查)")
    ap.add_argument("--selfcheck", action="store_true", help="每个 rig 落盘后读回校验(期望误差恒为 0)")
    ap.add_argument("--overwrite", action="store_true", help="已存在也重算(默认跳过=断点续跑)")
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    sessions = None if args.split == "auto" else SPLITS[args.split]
    rigs, used_sessions, n_skip = enumerate_rigs(args.centered_root, args.step1_root, sessions)
    total = len(rigs)
    mine = [r for i, r in enumerate(rigs) if i % args.world == args.rank]
    if args.limit > 0:
        mine = mine[: args.limit]
    print(f"[rank {args.rank}/{args.world}] sessions={used_sessions} 全局 rig {total:,}"
          f"(跳过缺图 {n_skip:,}), 本 rank {len(mine):,}", flush=True)
    if args.limit <= 0:
        os.makedirs(args.out_root, exist_ok=True)
        for stale in (_rank_marker(args.out_root, args.rank, args.world),
                      os.path.join(args.out_root, "COMPLETE")):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass
    if total == 0:
        print("[ERROR] 没有枚举到任何 rig, 检查 --centered-root / --step1-root", flush=True)
        return 1

    extr_np, intr_np, focal = load_rig_calibration(args.step2_root)
    K = extr_np.shape[0]
    crop_sizes = [fov_crop_size(f) for f in focal]
    print(f"  K={K}, FOV{TARGET_HFOV_DEG:g} crop sizes = {crop_sizes} (1068 原图上)", flush=True)

    model, dim = build_backbone(args.model_path, device)
    det_ok, gcfg = _geometric_input_config_is_deterministic(model)
    print(f"  info_sharing dim = {dim}; geometric_input_config={gcfg} "
          f"{'(确定性 OK)' if det_ok else '(警告: geometric_input 概率不是推理值, 输出不确定)'}", flush=True)

    extr = torch.from_numpy(extr_np).unsqueeze(0).to(device)
    intr = torch.from_numpy(intr_np).unsqueeze(0).to(device)

    done = skipped = failed = 0
    max_err = 0.0
    for session, ts, paths in mine:
        out_path = rig_feature_path(args.out_root, SCENE_ID, session, ts)
        if not args.overwrite and os.path.exists(out_path):
            skipped += 1
            continue
        try:
            imgs = load_rig_images(paths, crop_sizes).to(device)
            feats = perceive_rig(model, imgs, extr, intr)              # [K,256,C]
        except (OSError, RuntimeError) as e:
            failed += 1
            print(f"[rank {args.rank}] 跳过 {session}/{ts}: {e}", flush=True)
            continue
        save_rig_features(feats, out_path)
        if args.selfcheck:
            back = load_rig_features(out_path, expected_dim=dim).to(device)
            max_err = max(max_err, (back - feats.to(torch.bfloat16).float()).abs().max().item())
        done += 1
        if done % args.log_every == 0:
            print(f"[rank {args.rank}] 进度 {done + skipped + failed:,}/{len(mine):,}"
                  f"（{done:,} 存 / {skipped:,} 跳 / {failed} 失败）", flush=True)

    print(f"[rank {args.rank}] 完成: 新存 {done:,}, 跳 {skipped:,}, 失败 {failed}", flush=True)
    if args.selfcheck:
        print(f"[rank {args.rank}] selfcheck max_abs_err = {max_err:.2e} (应为 0)", flush=True)

    if failed:
        print(f"[ERROR] 本 rank 有 {failed} 个 rig 预计算失败；不写完成标记，请修复后重跑", flush=True)
        return 1

    if args.limit > 0:
        print("[INFO] --limit 模式, 不写 manifest / COMPLETE", flush=True)
        return 0

    if args.rank == 0:
        write_manifest(args.out_root, {
            "dataset": "nclt", "scene_id": SCENE_ID, "split": args.split,
            "sessions": used_sessions, "num_rigs_expected": total,
            "num_cameras": K, "num_tokens": NUM_TOKENS, "feature_dim": dim,
            "dtype": "bfloat16 (stored as uint16 bitcast in .npy)",
            "layout": "scenes/<scene>/trajectories/<traj>/features/<ts_ms>/rig.npy",
            "model_path": os.path.abspath(args.model_path),
            "image_pipeline": ("imread(BGR) -> rotate90CW -> FOV90 center crop(round(2f)) "
                               "-> resize224(INTER_AREA) -> BGR2RGB -> ToTensor[0,1]; "
                               "no external normalization; fp32 forward"),
            "fov_crop_sizes_1068": crop_sizes,
            "geometric_input_config": {k: float(v) for k, v in gcfg.items()
                                       if isinstance(v, (int, float))},
        })

    if _mark_complete_if_ready(args.out_root, rigs, args.rank, args.world, K, dim):
        print(f"[rank {args.rank}] 所有 rank 完成, 已写 COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
