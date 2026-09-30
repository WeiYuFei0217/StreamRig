#!/usr/bin/env python3
"""
step1_undistort_center_crop.py -- NCLT Ladybug3 图像去畸变 + 主点居中方裁 1068x1068。

功能:
  对 step0 下载解包得到的 NCLT 原始鱼眼图 (<raw-root>/<date>/lb3/Cam{1..5}/<ts_us>.tiff):
    1) 用官方 U2D 查表 (cv2.remap, INTER_LINEAR) 去畸变, 得到 1616x1232 无畸变图;
    2) 以该相机去畸变后的主点为中心裁出 1068x1068 正方形 (crop_half = ceil(f_max*tan(52.5°)) = 534),
       **不做任何缩放** —— 后续 step3 会在 1068 原图上再做 FOV90 center crop + resize 224;
    3) 存成 jpg (OpenCV 默认质量 95), 文件名沿用原始 16 位微秒时间戳;
    4) 同时推导并落盘 centered 内外参 (image_meta_centered.pkl / intrinsics_centered.txt /
       calibration_centered.json)。

  只处理 Cam1-Cam5 (Cam0 朝天, 对里程计无用)。**不含深度/velodyne 分支** —— StreamRig 不需要深度。

内外参推导 (全部来自官方标定文件):
  - 焦距 f: 直接取 K_cam{n}.csv 的 K[0,0] (U2D 去畸变不改变焦距)。
  - 去畸变图中的主点:
        cx_undist = K_orig[0, 2]
        cy_undist = K_orig[1, 2] - PP_ROW_OFFSET  (主点行向偏移 PP_ROW_OFFSET = 2.34375 px)
  - 裁切原点: row_start = round(cy_undist) - 534, col_start = round(cx_undist) - 534;
    裁后主点 cx_new = cx_undist - col_start, cy_new = cy_undist - row_start (都落在 [533.5, 534.5))。
  - 外参 T_camNormal_body: 按官方 devkit 约定由 x_lb3_c{n}.csv 推出
        T_camNormal_body = ssc(x_camNormal_cam) @ inv(ssc(x_lb3_c)) @ inv(ssc(x_body_lb3))
    其中 x_body_lb3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50] (官方常量),
         x_camNormal_cam = [0,0,0, 0,0,90] (把相机坐标系转到"图像顺时针转 90°后"的朝向)。

依赖: numpy, opencv-python (不需要 torch / mapanything)。

示例用法:
    # 全部 10 个 session, 10 个并行 worker
    python step1_undistort_center_crop.py \
        --raw-root   /data/nclt_raw \
        --calib-root /data/nclt_raw/calib \
        --out-root   /data/nclt_centered \
        --split all --workers 10

    # 只跑 test 划分的两个 session
    python step1_undistort_center_crop.py \
        --raw-root /data/nclt_raw --calib-root /data/nclt_raw/calib \
        --out-root /data/nclt_centered --split test --workers 10

    # 只推导并打印/落盘标定, 不处理图像
    python step1_undistort_center_crop.py \
        --calib-root /data/nclt_raw/calib --out-root /data/nclt_centered --calib-only

输出布局:
    <out-root>/image_meta_centered.pkl        # {"K": (5,3,3), "T": (5,4,4)}  T = T_camNormal_body
    <out-root>/intrinsics_centered.txt        # 人读版
    <out-root>/calibration_centered.json      # 人读版 + 裁切原点等信息
    <out-root>/<date>/lb3_centered/Cam{1..5}/<ts_us>.jpg      # 1068x1068
    <out-root>/<date>/groundtruth_<date>.csv  # 指向 raw-root 的软链(供 step2 就近读取)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 常量 (修改会改变预处理结果)
# ---------------------------------------------------------------------------

CAMERA_IDS = [1, 2, 3, 4, 5]          # Cam0 朝天, 不用
TARGET_FOV_DEG = 105.0                # 裁切尺寸的设计 FOV(按最大焦距 Cam1 折算)
CROP_HALF = 534                       # ceil(409.72 * tan(52.5°))
CROP_SIZE = CROP_HALF * 2             # 1068

# 去畸变图主点行坐标相对官方 K 的偏移 (px)
PP_ROW_OFFSET = 2.34375

# 官方 devkit 常量
X_BODY_LB3 = [0.035, 0.002, -1.23, -179.93, -0.23, 0.50]
X_CAMNORMAL_CAM = [0.0, 0.0, 0.0, 0.0, 0.0, 90.0]

TRAIN_SESSIONS = [
    "2012-01-08", "2012-02-02", "2012-02-04", "2012-03-17",
    "2012-05-26", "2012-10-28", "2012-11-17", "2013-04-05",
]
TEST_SESSIONS = ["2012-02-19", "2012-08-20"]

_SESSION_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# 位姿工具 (官方 devkit ssc_to_homo)
# ---------------------------------------------------------------------------

def ssc_to_homo(ssc) -> np.ndarray:
    """6-DOF (x, y, z, roll_deg, pitch_deg, heading_deg) -> 4x4 齐次变换 (官方约定)。"""
    sr, cr = np.sin(np.pi / 180.0 * ssc[3]), np.cos(np.pi / 180.0 * ssc[3])
    sp, cp = np.sin(np.pi / 180.0 * ssc[4]), np.cos(np.pi / 180.0 * ssc[4])
    sh, ch = np.sin(np.pi / 180.0 * ssc[5]), np.cos(np.pi / 180.0 * ssc[5])
    H = np.zeros((4, 4), dtype=np.float64)
    H[0, 0] = ch * cp
    H[0, 1] = -sh * cr + ch * sp * sr
    H[0, 2] = sh * sr + ch * sp * cr
    H[1, 0] = sh * cp
    H[1, 1] = ch * cr + sh * sp * sr
    H[1, 2] = -ch * sr + sh * sp * cr
    H[2, 0] = -sp
    H[2, 1] = cp * sr
    H[2, 2] = cp * cr
    H[0, 3], H[1, 3], H[2, 3] = ssc[0], ssc[1], ssc[2]
    H[3, 3] = 1.0
    return H


def camnormal_T_body(x_lb3_c) -> np.ndarray:
    """由官方 x_lb3_c{n}.csv 推出 T_camNormal_body (相机"正立"坐标系 <- body 坐标系)。"""
    T_lb3_c = ssc_to_homo(x_lb3_c)
    T_body_lb3 = ssc_to_homo(X_BODY_LB3)
    T_camNormal_cam = ssc_to_homo(X_CAMNORMAL_CAM)
    T_c_body = np.linalg.inv(T_lb3_c) @ np.linalg.inv(T_body_lb3)
    return T_camNormal_cam @ T_c_body


# ---------------------------------------------------------------------------
# U2D 去畸变查表
# ---------------------------------------------------------------------------

class Undistort:
    """官方 U2D_Cam{n}_1616X1232.txt 查表去畸变 (与 NCLT devkit undistort.py 一致)。"""

    def __init__(self, fin: str):
        with open(fin, "r") as f:
            header = f.readline().rstrip()
            chunks = re.sub(r"[^0-9,]", "", header).split(",")
            self.W, self.H = int(chunks[0]), int(chunks[1])
            self.mapu = np.zeros((self.H, self.W), dtype=np.float32)
            self.mapv = np.zeros((self.H, self.W), dtype=np.float32)
            for line in f.readlines():
                c = line.rstrip().split(" ")
                r, col = int(c[0]), int(c[1])
                self.mapu[r, col] = float(c[3])
                self.mapv[r, col] = float(c[2])
        mask = np.ones(self.mapu.shape, dtype=np.uint8)
        mask = cv2.remap(mask, self.mapu, self.mapv, cv2.INTER_LINEAR)
        self.mask = cv2.erode(mask, np.ones((30, 30), np.uint8), iterations=1)

    def undistort(self, img: np.ndarray) -> np.ndarray:
        return cv2.remap(img, self.mapu, self.mapv, cv2.INTER_LINEAR)

    def valid_bbox(self):
        rows, cols = np.where(self.mask > 0)
        return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


# ---------------------------------------------------------------------------
# 标定推导
# ---------------------------------------------------------------------------

def derive_calibration(calib_root: str, check_bbox: bool = True) -> list[dict]:
    """由官方 cam_params/ + U2D/ 推出 5 个相机的 centered 内外参。

    参数:
        calib_root: 含 cam_params/ 与 U2D/ 的目录 (step0 下载产物)。
        check_bbox: True = 加载 U2D 查表检查裁切框是否落在有效区内。
    返回: 每相机一个 dict, 按 CAMERA_IDS 顺序。
    """
    cam_params_dir = os.path.join(calib_root, "cam_params")
    u2d_dir = os.path.join(calib_root, "U2D")
    for d in (cam_params_dir, u2d_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"标定目录不存在: {d} (先跑 step0_download.sh --calib-only)")

    cam_info = []
    for idx, cam_id in enumerate(CAMERA_IDS):
        K_orig = np.genfromtxt(os.path.join(cam_params_dir, f"K_cam{cam_id}.csv"), delimiter=",")
        f_len = float(K_orig[0, 0])
        cx_u = float(K_orig[0, 2])
        cy_u = float(K_orig[1, 2]) - PP_ROW_OFFSET

        row_start = int(round(cy_u)) - CROP_HALF
        col_start = int(round(cx_u)) - CROP_HALF
        cx_new = cx_u - col_start
        cy_new = cy_u - row_start
        fov = 2.0 * math.degrees(math.atan(CROP_HALF / f_len))

        fits = None
        if check_bbox:
            u2d_path = os.path.join(u2d_dir, f"U2D_Cam{cam_id}_1616X1232.txt")
            print(f"  Cam{cam_id}: 加载 U2D 查表检查有效区 ...", end=" ", flush=True)
            bbox = Undistort(u2d_path).valid_bbox()
            fits = bool(row_start >= bbox[0] and row_start + CROP_SIZE <= bbox[1]
                        and col_start >= bbox[2] and col_start + CROP_SIZE <= bbox[3])
            print(f"f={f_len:.2f} FOV={fov:.2f}° {'OK' if fits else 'OUT-OF-VALID-AREA!'}")
        else:
            print(f"  Cam{cam_id}: f={f_len:.2f} FOV={fov:.2f}° "
                  f"crop_origin=(row={row_start}, col={col_start})")

        x_lb3_c = np.genfromtxt(os.path.join(cam_params_dir, f"x_lb3_c{cam_id}.csv"), delimiter=",")
        cam_info.append(dict(
            cam_id=cam_id, cam_idx=idx, f=f_len,
            cx_undist=cx_u, cy_undist=cy_u,
            row_start=row_start, col_start=col_start,
            cx=cx_new, cy=cy_new, fov_deg=fov, fits_valid_area=fits,
            K=np.array([[f_len, 0.0, cx_new], [0.0, f_len, cy_new], [0.0, 0.0, 1.0]]),
            T_camNormal_body=camnormal_T_body(x_lb3_c),
        ))
    return cam_info


def save_calibration(cam_info: list[dict], out_root: str) -> None:
    """落盘 image_meta_centered.pkl / intrinsics_centered.txt / calibration_centered.json。"""
    os.makedirs(out_root, exist_ok=True)
    K_all = np.stack([c["K"] for c in cam_info], 0)                    # (5,3,3)
    T_all = np.stack([c["T_camNormal_body"] for c in cam_info], 0)     # (5,4,4)

    pkl_path = os.path.join(out_root, "image_meta_centered.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump({"K": K_all.tolist(), "T": T_all.tolist()}, f)

    txt_path = os.path.join(out_root, "intrinsics_centered.txt")
    with open(txt_path, "w") as f:
        f.write(f"# NCLT centered intrinsics  FOV~{TARGET_FOV_DEG} deg  "
                f"crop={CROP_SIZE}x{CROP_SIZE}  undistorted then cropped\n")
        f.write("# cam  f          cx         cy         fov\n")
        for c in cam_info:
            f.write(f"Cam{c['cam_id']}  {c['f']:.6f}  {c['cx']:.6f}  {c['cy']:.6f}  {c['fov_deg']:.2f}\n")

    json_path = os.path.join(out_root, "calibration_centered.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "crop_size": CROP_SIZE,
            "crop_half": CROP_HALF,
            "cameras": [{
                "name": f"Cam{c['cam_id']}", "index": c["cam_idx"],
                "focal_length_px": c["f"], "cx": c["cx"], "cy": c["cy"],
                "fov_deg": c["fov_deg"],
                "undistorted_crop_origin": {"row": c["row_start"], "col": c["col_start"]},
                "K": c["K"].tolist(),
                "T_camNormal_body": c["T_camNormal_body"].tolist(),
            } for c in cam_info],
        }, f, indent=2)

    print(f"  saved {pkl_path}\n  saved {txt_path}\n  saved {json_path}")


# ---------------------------------------------------------------------------
# 图像处理
# ---------------------------------------------------------------------------

def _process_one_camera(task: dict) -> dict:
    """处理 (session, camera) 一个组合; 断点续传(已存在的 jpg 跳过)。子进程入口。"""
    session, cam = task["session"], task["cam"]
    raw_dir = os.path.join(task["raw_root"], session, "lb3", f"Cam{cam['cam_id']}")
    out_dir = os.path.join(task["out_root"], session, "lb3_centered", f"Cam{cam['cam_id']}")
    if not os.path.isdir(raw_dir):
        return dict(session=session, cam_id=cam["cam_id"], status="missing_raw",
                    n_done=0, n_total=0)

    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(raw_dir) if n.endswith(".tiff"))
    existing = set(os.listdir(out_dir))
    pending = [n for n in names if n[:-5] + ".jpg" not in existing]
    if not pending:
        return dict(session=session, cam_id=cam["cam_id"], status="already_done",
                    n_done=0, n_total=len(names))

    undist = Undistort(os.path.join(task["u2d_dir"], f"U2D_Cam{cam['cam_id']}_1616X1232.txt"))
    rs, cs = cam["row_start"], cam["col_start"]
    jpg_params = [cv2.IMWRITE_JPEG_QUALITY, int(task["jpeg_quality"])]

    n_done, n_bad = 0, 0
    for name in pending:
        img = cv2.imread(os.path.join(raw_dir, name))
        if img is None:
            n_bad += 1
            continue
        crop = undist.undistort(img)[rs:rs + CROP_SIZE, cs:cs + CROP_SIZE]
        if crop.shape[0] != CROP_SIZE or crop.shape[1] != CROP_SIZE:
            n_bad += 1
            continue
        # 先 imencode 再原子替换: 直接 imwrite 到 .tmp 后缀会让 OpenCV 找不到编码器
        ok, buf = cv2.imencode(".jpg", crop, jpg_params)
        if not ok:
            n_bad += 1
            continue
        tmp = os.path.join(out_dir, f".{name[:-5]}.{os.getpid()}.tmp.jpg")
        with open(tmp, "wb") as fh:
            fh.write(buf.tobytes())
        os.replace(tmp, os.path.join(out_dir, name[:-5] + ".jpg"))
        n_done += 1
    return dict(session=session, cam_id=cam["cam_id"], status="ok" if n_bad == 0 else f"ok({n_bad} bad)",
                n_done=n_done, n_total=len(names))


def link_ground_truth(raw_root: str, out_root: str, session: str) -> None:
    """把 raw-root 下的 groundtruth csv 软链到 out-root/<session>/ , 供 step2 就近读取。"""
    src = os.path.join(raw_root, session, f"groundtruth_{session}.csv")
    dst_dir = os.path.join(out_root, session)
    dst = os.path.join(dst_dir, f"groundtruth_{session}.csv")
    if not os.path.isfile(src) or os.path.exists(dst):
        return
    os.makedirs(dst_dir, exist_ok=True)
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError as e:          # 跨文件系统/权限问题时不致命, step2 可用 --gt-root 指过去
        print(f"  [WARN] {session}: groundtruth 软链失败 ({e}); step2 请用 --gt-root {raw_root}")


def discover_sessions(raw_root: str) -> list[str]:
    """扫描 raw-root 下所有含 lb3/Cam1 的日期目录。"""
    out = []
    for name in sorted(os.listdir(raw_root)):
        if _SESSION_RE.match(name) and os.path.isdir(os.path.join(raw_root, name, "lb3", "Cam1")):
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="NCLT 去畸变 + 主点居中 1068x1068 方裁 (Cam1-5, 无深度分支)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--raw-root", default="", help="step0 产物根: <raw-root>/<date>/lb3/Cam{0..5}/*.tiff")
    ap.add_argument("--calib-root", required=True, help="含 cam_params/ 与 U2D/ 的目录")
    ap.add_argument("--out-root", required=True, help="输出根: <out-root>/<date>/lb3_centered/Cam{1..5}/*.jpg")
    ap.add_argument("--split", default="all", choices=["train", "test", "all", "auto"],
                    help="auto = 扫描 raw-root 里所有可处理的日期")
    ap.add_argument("--sessions", nargs="+", default=None, help="显式指定日期, 覆盖 --split")
    ap.add_argument("--workers", type=int, default=5, help="并行进程数 (任务粒度 = session x camera)")
    ap.add_argument("--jpeg-quality", type=int, default=95,
                    help="JPEG 质量(95 = OpenCV 默认, 发布数据使用该值)")
    ap.add_argument("--calib-only", action="store_true", help="只推导并落盘标定, 不处理图像")
    ap.add_argument("--no-bbox-check", action="store_true", help="跳过 U2D 有效区检查")
    args = ap.parse_args()

    print("=== NCLT step1: undistort + principal-point-centered 1068x1068 crop ===")
    print(f"  calib-root : {args.calib_root}")
    print(f"  out-root   : {args.out_root}")

    cam_info = derive_calibration(args.calib_root, check_bbox=not args.no_bbox_check)
    save_calibration(cam_info, args.out_root)
    if args.calib_only:
        print("--calib-only 指定, 结束。")
        return 0

    if not args.raw_root:
        ap.error("处理图像需要 --raw-root")
    if args.sessions:
        sessions = list(args.sessions)
    elif args.split == "train":
        sessions = list(TRAIN_SESSIONS)
    elif args.split == "test":
        sessions = list(TEST_SESSIONS)
    elif args.split == "all":
        sessions = TRAIN_SESSIONS + TEST_SESSIONS
    else:
        sessions = discover_sessions(args.raw_root)
    print(f"  sessions   : {len(sessions)} -> {sessions}")

    # 任务粒度 = (session, camera), 共 len(sessions)*5 个, 便于负载均衡
    u2d_dir = os.path.join(args.calib_root, "U2D")
    tasks = [dict(session=s, cam=dict(cam_id=c["cam_id"], row_start=c["row_start"], col_start=c["col_start"]),
                  raw_root=args.raw_root, out_root=args.out_root, u2d_dir=u2d_dir,
                  jpeg_quality=args.jpeg_quality)
             for s in sessions for c in cam_info]

    for s in sessions:
        link_ground_truth(args.raw_root, args.out_root, s)

    n_img, n_fail = 0, 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_process_one_camera, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as e:                      # noqa: BLE001 - 单个相机失败不拖垮整批
                t = futs[fut]
                n_fail += 1
                print(f"  [{i}/{len(tasks)}] [ERROR] {t['session']}/Cam{t['cam']['cam_id']}: {e}", flush=True)
                continue
            n_img += r["n_done"]
            if r["status"] == "missing_raw":
                n_fail += 1
            print(f"  [{i}/{len(tasks)}] {r['session']}/Cam{r['cam_id']}: {r['status']}, "
                  f"新处理 {r['n_done']}/{r['n_total']}", flush=True)

    print(f"\n=== 完成: 新处理 {n_img:,} 张, 失败/缺失任务 {n_fail} ===")
    print(f"输出: {args.out_root}/<date>/lb3_centered/Cam{{1..5}}/<ts_us>.jpg  ({CROP_SIZE}x{CROP_SIZE})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
