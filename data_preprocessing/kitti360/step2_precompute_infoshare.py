"""
step2_precompute_infoshare.py -- 预计算 KITTI-360 rig 的 MapAnything info_sharing 缓存。

功能：把 step1 索引里的每个 rig（四相机同一帧）一次性跑完 frozen 感知前端，把 rig 融合后的
``info_sharing`` 特征落盘，训练/评测直接读盘，省掉每步在线跑 backbone 的开销。

计算链路（与 NCLT 缓存相同，全程 fp32）::

    square-crop / MEI-remap 后的 RGB[0,1]
      → frozen DINOv2 encoder（fp32，backbone 内部按 data_norm_type="dinov2" 归一化）
      → 注入每相机的目标 K 射线 + cam0_T_cam rig 外参（公制尺度为真）
      → MapAnything info_sharing
      → [K,256,768] 先转 bf16 再 bitcast 成 numpy uint16 存盘（1.5 MiB/rig，K=4）

输出布局（``--out-root``）::

    manifest.json                      # 特征 shape/源 manifest 与 rig_config 的 SHA-256
    COMPLETE                           # 只有 --verify-only 全量验收通过才写，训练加载器强制要求
    scenes/kitti360/trajectories/<drive>/features/<frame10>/rig.npy

车辆时间轨迹**不**注入单 rig 感知层（它是训练标签 T_rel_gt）。

示例用法（单卡，只处理少量 rig 做检查）::

    conda activate streamrig
    python data_preprocessing/kitti360/step2_precompute_infoshare.py \
      --processed-root /path/to/kitti360_streamrig_metadata \
      --raw-root /path/to/KITTI-360 \
      --out-root /path/to/kitti360_infoshare_bf16 \
      --model-path /path/to/map-anything-model \
      --splits test --limit 64 --selfcheck

四卡分片跑全量（每进程只看一张卡），完成后做一次全量验收并写 COMPLETE::

    for rank in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$rank python data_preprocessing/kitti360/step2_precompute_infoshare.py \
        --processed-root ... --raw-root ... --out-root ... --model-path ... \
        --rank $rank --world 4 --selfcheck &
    done; wait
    python data_preprocessing/kitti360/step2_precompute_infoshare.py \
      --processed-root ... --out-root ... --verify-only --verify-samples 128
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# 允许不安装包直接运行：把仓库根加入 sys.path（本文件在 data_preprocessing/kitti360/ 下）
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocessing.common.infoshare_io import (  # noqa: E402
    load_rig_features,
    save_rig_features,
)
from data_preprocessing.common.mapanything_frontend import (  # noqa: E402
    load_frozen_mapanything,
    perceive_rig_from_encodings,
)
from data_preprocessing.kitti360.kitti360_preprocessing import (  # noqa: E402
    apply_perspective_transform,
)
from streamrig.datasets.formats import (  # noqa: E402
    CANONICAL_PERSPECTIVE_PREPROCESSING,
    KITTI360_INFOSHARE_FORMAT,
    KITTI360_PROCESSED_FORMAT,
    KITTI360_RIG_FORMAT,
    check_format,
)


CACHE_FORMAT = KITTI360_INFOSHARE_FORMAT
CAMERA_IDS = (0, 1, 2, 3)   # image_00/01 透视双目 + image_02/03 鱼眼
IMAGE_SIZE = 224
PATCH_TOKENS = 256          # 224/14 = 16 → 16×16 个 patch
FEATURE_DIM = 768           # MapAnything info_sharing 通道数
CACHE_SHAPE = (len(CAMERA_IDS), PATCH_TOKENS, FEATURE_DIM)


def encode_dino_fp32(backbone, images_raw01: torch.Tensor) -> torch.Tensor:
    """从 raw[0,1] 图像用 ``_encode_n_views`` 算 fp32 DINOv2 特征 → ``[B,K,C,H,W]``。

    不加 autocast，整条 fp32；``_encode_n_views`` 可能返回 ``(features, ...)`` 或
    K 个 ``[1,C,H,W]``，这里统一取出张量。
    """

    num_views = images_raw01.shape[1]
    views = [
        {"img": images_raw01[:, index], "data_norm_type": ["dinov2"]}
        for index in range(num_views)
    ]
    feature_list = backbone._encode_n_views(views)
    if (
        isinstance(feature_list, (tuple, list))
        and len(feature_list)
        and not torch.is_tensor(feature_list[0])
    ):
        feature_list = feature_list[0]
    tensors = []
    for item in feature_list:
        while isinstance(item, (tuple, list)):
            item = item[0]
        tensors.append(item if item.dim() == 4 else item.unsqueeze(0))
    return torch.stack(tensors, dim=1)


# --------------------------------------------------------------------------------------
# 枚举 / 路径 / 校验
# --------------------------------------------------------------------------------------
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_path(out_root: Path, sequence: str, frame_id: int) -> Path:
    return (
        Path(out_root) / "scenes" / "kitti360" / "trajectories" / sequence
        / "features" / f"{int(frame_id):010d}" / "rig.npy"
    )


def enumerate_rigs(processed_root: Path, splits: tuple[str, ...]) -> list[tuple[str, int]]:
    """按 split 顺序枚举 step1 保留下来的全部 rig（序列内按 frame id 升序）。"""

    manifest = json.loads((processed_root / "manifest.json").read_text(encoding="utf-8"))
    check_format(manifest.get("format"), KITTI360_PROCESSED_FORMAT, "processed manifest")
    if manifest.get("perspective_preprocessing") != CANONICAL_PERSPECTIVE_PREPROCESSING:
        raise ValueError("processed_root 的透视相机预处理不是 isotropic_square_crop")
    sequences = []
    seen = set()
    for split in splits:
        if split not in manifest["splits"]:
            raise ValueError(f"manifest 不含 split={split!r}")
        for sequence in manifest["splits"][split]:
            if sequence not in seen:
                sequences.append(sequence)
                seen.add(sequence)
    rigs = []
    for sequence in sequences:
        meta = np.load(processed_root / "sequences" / sequence / "frames.npz")
        rigs.extend((sequence, int(frame_id)) for frame_id in meta["frame_ids"])
    return rigs


class Kitti360RigImages:
    """按 step1 的 rig_config 读取四目图像并做 canonical 变换，同时给出 rig 内外参。

    透视相机 image_00/01 按 rig_config 记录的主点居中方形裁剪 + 缩放；鱼眼 image_02/03 用
    step1 生成的 MEI 重映射表转成 90° 方形虚拟针孔图。外参以 image_00 为参考系。
    """

    def __init__(self, processed_root: Path, raw_root: Path):
        self.raw_root = Path(raw_root)
        rig_path = Path(processed_root) / "rig_config.json"
        rig_config = json.loads(rig_path.read_text(encoding="utf-8"))
        check_format(rig_config.get("format"), KITTI360_RIG_FORMAT, str(rig_path))
        if rig_config.get("perspective_preprocessing") != CANONICAL_PERSPECTIVE_PREPROCESSING:
            raise ValueError("rig_config 的 perspective_preprocessing 不是 isotropic_square_crop")
        all_body_T_cams = [
            np.asarray(cam["body_T_cam"], dtype=np.float64).reshape(4, 4)
            for cam in rig_config["cameras"]
        ]
        all_intrinsics = [
            np.asarray(cam["intrinsics"], dtype=np.float32).reshape(3, 3)
            for cam in rig_config["cameras"]
        ]
        body_T_cams = [all_body_T_cams[c] for c in CAMERA_IDS]
        ref_inv = np.linalg.inv(body_T_cams[0])
        self.extrinsics = np.stack(
            [ref_inv @ transform for transform in body_T_cams], axis=0
        ).astype(np.float32)
        self.intrinsics = np.stack(
            [all_intrinsics[c] for c in CAMERA_IDS], axis=0
        ).astype(np.float32)
        self._perspective_preprocessing = {
            int(cam["id"]): cam["preprocessing"]
            for cam in rig_config["cameras"] if int(cam["id"]) < 2
        }
        self._fisheye_maps = {}
        for cam_id in CAMERA_IDS:
            if cam_id < 2:
                continue
            maps = np.load(Path(processed_root) / "rectify_maps" / f"image_{cam_id:02d}.npz")
            self._fisheye_maps[cam_id] = (
                maps["map_x"].astype(np.float32),
                maps["map_y"].astype(np.float32),
            )

    def _raw_image_path(self, seq_name, frame_id, cam_id):
        image_kind = "data_rect" if cam_id < 2 else "data_rgb"
        return (
            self.raw_root / "data_2d_raw" / seq_name / f"image_{cam_id:02d}"
            / image_kind / f"{int(frame_id):010d}.png"
        )

    def load_camera_image(self, seq_name: str, frame_id: int, cam_id: int) -> np.ndarray:
        """读取一个相机的 canonical RGB uint8 图像，返回 ``[H,W,3]``。"""

        path = self._raw_image_path(seq_name, frame_id, cam_id)
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"读不到 KITTI-360 图像: {path}")
        if cam_id < 2:
            processed = apply_perspective_transform(
                image_bgr, self._perspective_preprocessing[cam_id]
            )
        else:
            map_x, map_y = self._fisheye_maps[cam_id]
            processed = cv2.remap(
                image_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
        return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    def load_rig_images(self, seq_name: str, frame_id: int) -> np.ndarray:
        """返回四目 ``[K,H,W,3]`` RGB uint8。"""

        return np.stack([
            self.load_camera_image(seq_name, frame_id, cam_id)
            for cam_id in CAMERA_IDS
        ], axis=0)


@torch.no_grad()
def perceive_rig(
    backbone,
    rig: Kitti360RigImages,
    sequence: str,
    frame_id: int,
    device: str,
) -> torch.Tensor:
    """对单个 rig 跑 fp32 感知，返回 ``[K,256,768]`` float32。"""

    images = rig.load_rig_images(sequence, frame_id)
    images_t = (
        torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).float().div_(255.0)
        .to(device)
    )
    extrinsics = torch.from_numpy(rig.extrinsics).unsqueeze(0).to(device)
    intrinsics = torch.from_numpy(rig.intrinsics).unsqueeze(0).to(device)
    encoded = encode_dino_fp32(backbone, images_t).float()
    features = perceive_rig_from_encodings(
        backbone, extrinsics, intrinsics, encoded, image_size=IMAGE_SIZE,
    )[0]
    if tuple(features.shape) != CACHE_SHAPE or not torch.isfinite(features).all():
        raise RuntimeError(
            f"{sequence}/{frame_id:010d}: info_sharing 非法 {tuple(features.shape)}"
        )
    return features


def write_root_manifest(
    out_root: Path,
    processed_root: Path,
    raw_root: Path,
    model_path: str,
    splits: tuple[str, ...],
    total_rigs: int,
) -> None:
    """写缓存根 manifest；加载器会用其中的 SHA-256 指纹拒绝错标定缓存。"""

    payload = {
        "format": CACHE_FORMAT,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_processed_root": str(processed_root),
        "source_raw_root": str(raw_root),
        "source_manifest_sha256": file_sha256(processed_root / "manifest.json"),
        "source_rig_config_sha256": file_sha256(processed_root / "rig_config.json"),
        "perspective_preprocessing": CANONICAL_PERSPECTIVE_PREPROCESSING,
        "model_path": str(Path(model_path).resolve()),
        "splits": list(splits),
        "total_rigs": int(total_rigs),
        "feature_layer": "MapAnything.info_sharing",
        "feature_shape": list(CACHE_SHAPE),
        "storage": "bfloat16 bitcast as numpy uint16",
        "geometry": {
            "intrinsics": "per-view K after canonical image preprocessing",
            "extrinsics": "per-view selected-reference_T_cam rig calibration",
            "metric_scale": True,
            "temporal_gt_pose_in_feature": False,
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    tmp = out_root / f"manifest.json.tmp{os.getpid()}"
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, out_root / "manifest.json")


def verify_cache(
    out_root: Path,
    rigs: list[tuple[str, int]],
    samples: int,
    write_complete: bool = True,
) -> None:
    """全量缺失检查 + 等距抽样 dtype/shape/finite 检查；通过后写 COMPLETE。"""

    manifest_path = out_root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"缓存缺 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_shape = CACHE_SHAPE
    if manifest.get("feature_shape") != list(expected_shape):
        raise RuntimeError(
            f"manifest feature_shape={manifest.get('feature_shape')}，应为 {list(expected_shape)}"
        )
    missing = [item for item in rigs if not cache_path(out_root, *item).is_file()]
    if missing:
        preview = ", ".join(f"{seq}/{frame:010d}" for seq, frame in missing[:5])
        raise RuntimeError(f"缓存缺 {len(missing)}/{len(rigs)} 个 rig，示例: {preview}")
    picks = np.linspace(0, len(rigs) - 1, min(samples, len(rigs)), dtype=int)
    for index in picks:
        path = cache_path(out_root, *rigs[int(index)])
        raw = np.load(path, mmap_mode="r")
        if raw.dtype != np.uint16 or tuple(raw.shape) != expected_shape:
            raise RuntimeError(f"{path}: dtype/shape={raw.dtype}/{raw.shape}")
        feature = load_rig_features(str(path))
        if not torch.isfinite(feature).all():
            raise RuntimeError(f"{path}: 出现 NaN/Inf")
    print(
        f"[verify] {len(rigs)} 个文件齐全；抽查 {len(picks)} 个 dtype/shape/finite 通过",
        flush=True,
    )
    if write_complete:
        (out_root / "COMPLETE").write_text(
            datetime.now().astimezone().isoformat(timespec="seconds") + "\n", encoding="utf-8"
        )
        print(f"[verify] 已写 {out_root / 'COMPLETE'}（训练加载器要求存在此标记）", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KITTI-360 MapAnything info_sharing 缓存预计算"
    )
    parser.add_argument("--processed-root", required=True, help="step1 输出的轻量索引根目录")
    parser.add_argument("--raw-root", default=None,
                        help="官方 KITTI-360 根目录；缺省时用 processed manifest 里的 raw_root")
    parser.add_argument("--out-root", required=True, help="缓存输出根目录")
    parser.add_argument("--model-path", default=None,
                        help="MapAnything 权重目录（HuggingFace from_pretrained 格式；--verify-only 时不需要）")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--rank", type=int, default=0, help="分片 id（多卡时每进程一张卡）")
    parser.add_argument("--world", type=int, default=1, help="分片总数")
    parser.add_argument("--limit", type=int, default=0, help=">0 时只处理本 rank 的前 N 个 rig（用于快速检查）")
    parser.add_argument("--selfcheck", action="store_true",
                        help="每写一个文件就读回，验证 bf16 位存逐位一致")
    parser.add_argument("--verify-only", action="store_true",
                        help="不计算，只做全量验收；通过后写 COMPLETE")
    parser.add_argument("--verify-samples", type=int, default=32)
    parser.add_argument("--no-write-complete", action="store_true",
                        help="--verify-only 通过后不写 COMPLETE")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_root = Path(args.processed_root).resolve()
    manifest = json.loads((processed_root / "manifest.json").read_text(encoding="utf-8"))
    raw_root = Path(args.raw_root or manifest["raw_root"]).resolve()
    out_root = Path(args.out_root).resolve()
    splits = tuple(args.splits)
    if args.world < 1 or not 0 <= args.rank < args.world:
        raise ValueError(f"非法 rank/world={args.rank}/{args.world}")

    rigs = enumerate_rigs(processed_root, splits)
    if args.verify_only:
        verify_cache(
            out_root, rigs, args.verify_samples,
            write_complete=not args.no_write_complete,
        )
        return
    if not args.model_path:
        raise SystemExit("预计算需要 --model-path（MapAnything 权重目录）")
    mine = [item for index, item in enumerate(rigs) if index % args.world == args.rank]
    if args.limit > 0:
        mine = mine[: args.limit]

    if args.rank == 0:
        write_root_manifest(
            out_root, processed_root, raw_root, args.model_path, splits, len(rigs),
        )

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    rig = Kitti360RigImages(processed_root, raw_root)
    backbone = load_frozen_mapanything(args.model_path, device)
    print(
        f"[rank {args.rank}/{args.world}] total={len(rigs)}, mine={len(mine)}, "
        f"cameras={CAMERA_IDS}, canonical={manifest['perspective_preprocessing']}, "
        f"device={device}",
        flush=True,
    )

    done = skipped = 0
    for sequence, frame_id in mine:
        path = cache_path(out_root, sequence, frame_id)
        if path.is_file():
            skipped += 1
            continue
        try:
            features = perceive_rig(backbone, rig, sequence, frame_id, device)
        except (FileNotFoundError, OSError) as error:
            print(f"[rank {args.rank}] 跳过 {sequence}/{frame_id:010d}: {error}", flush=True)
            continue
        save_rig_features(features, str(path))
        if args.selfcheck:
            restored = load_rig_features(str(path)).to(device)
            if not torch.equal(restored, features.to(torch.bfloat16).float()):
                raise RuntimeError(f"{path}: bf16 位存读回不逐位一致")
        done += 1
        if done % args.log_every == 0:
            print(f"[rank {args.rank}] 进度 {done + skipped}/{len(mine)}（新存 {done}，已有 {skipped}）",
                  flush=True)

    print(f"[rank {args.rank}] 完成：新存 {done}，已有 {skipped}，分配 {len(mine)}", flush=True)


if __name__ == "__main__":
    main()
