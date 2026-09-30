#!/usr/bin/env bash
# ============================================================================
# eval_kitti360.sh -- KITTI-360 一键评测（流式评测 → drive 级聚合 → 打印指标）。
#
# 评测设置：
#   test 划分 drive 0009 + 0010 共 10 条 valid run，stride 3，
#   t_rel/r_rel 在 10 条 run 上按段计数联合池化；
#   ATE 在完整 drive 上计算（run 间以真值相对位姿衔接），两条 drive 按 GT 路径长加权。
#
# 用法：
#   bash scripts/eval_kitti360.sh \
#       --processed-root  /path/to/kitti360_streamrig_metadata \
#       --infoshare-root  /path/to/kitti360_infoshare_bf16 \
#       --out-dir         outputs/eval_kitti360
#
# 可选参数：
#   --config <yaml>    默认 configs/kitti360_streamrig.yaml
#   --ckpt <pth>       默认 release_weights/KITTI360-StreamRig.pth
#   --exp-name <name>  默认 streamrig_kitti360
#   --k <int>          interval-k（重锚间隔），默认 2
#   --gpu <id>         默认 0
#   --data-poses-root <dir>  用官方 data_poses/ 代替 processed_root 提供 ATE 用的 GT
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="$REPO_ROOT/configs/kitti360_streamrig.yaml"
CKPT="$REPO_ROOT/release_weights/KITTI360-StreamRig.pth"
EXP_NAME="streamrig_kitti360"
K=2
GPU=0
PROCESSED_ROOT=""; INFOSHARE_ROOT=""; OUT_DIR=""; DATA_POSES_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)           CONFIG="$2"; shift 2 ;;
    --ckpt)             CKPT="$2"; shift 2 ;;
    --processed-root)   PROCESSED_ROOT="$2"; shift 2 ;;
    --infoshare-root)   INFOSHARE_ROOT="$2"; shift 2 ;;
    --out-dir)          OUT_DIR="$2"; shift 2 ;;
    --exp-name)         EXP_NAME="$2"; shift 2 ;;
    --k)                K="$2"; shift 2 ;;
    --gpu)              GPU="$2"; shift 2 ;;
    --data-poses-root)  DATA_POSES_ROOT="$2"; shift 2 ;;
    -h|--help)          sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

for pair in "PROCESSED_ROOT:--processed-root" "INFOSHARE_ROOT:--infoshare-root" \
            "OUT_DIR:--out-dir"; do
  var="${pair%%:*}"; flag="${pair##*:}"
  if [[ -z "${!var}" ]]; then echo "缺少必填参数 $flag" >&2; exit 2; fi
done

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
EVAL_DIR="$OUT_DIR/$EXP_NAME/test"
DRIVE_DIR="$OUT_DIR/drive_metrics"
RESOLVED_CONFIG="$OUT_DIR/resolved_kitti360_config.yaml"
W_MAX=$((K + 1))

# --- 1) 解析 config 占位符 ------------------------------------------------
python - "$CONFIG" "$RESOLVED_CONFIG" "$PROCESSED_ROOT" "$INFOSHARE_ROOT" <<'PY'
import sys, yaml
src, dst, processed, infoshare = sys.argv[1:5]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["model"].pop("warmstart_ckpt", None)          # 评测不需要 warm-start 权重
data = cfg["data"]
data["processed_root"] = processed
data["infoshare_root"] = infoshare
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
print(f"[eval_kitti360] resolved config -> {dst}")
PY

# --- 2) 流式评测（test 划分 = drive 0009 + 0010） ----------------------------
echo "[eval_kitti360] step 1/3  streaming evaluation (stride 3) on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" python "$REPO_ROOT/scripts/eval_stream.py" \
  --config "$RESOLVED_CONFIG" --ckpt "$CKPT" \
  --exp-name "$EXP_NAME" --out-dir "$OUT_DIR" \
  --dataset kitti360 --split test --stride 3 --step-size 10 \
  --w-max "$W_MAX" --intervals "$K" \
  --processed-root "$PROCESSED_ROOT" --infoshare-root "$INFOSHARE_ROOT" --save-poses

# --- 3) drive 级聚合 -----------------------------------------------------
echo "[eval_kitti360] step 2/3  drive-level aggregation"
GT_ARGS=(--processed-root "$PROCESSED_ROOT")
if [[ -n "$DATA_POSES_ROOT" ]]; then GT_ARGS=(--data-poses-root "$DATA_POSES_ROOT"); fi
python "$REPO_ROOT/scripts/k360_drive_metrics.py" \
  --eval-dir "$EVAL_DIR" --out-dir "$DRIVE_DIR" --k "$K" \
  --method-id "$EXP_NAME" "${GT_ARGS[@]}"

# --- 4) 打印指标 ----------------------------------------------------------
echo "[eval_kitti360] step 3/3  metrics"
python - "$DRIVE_DIR/summary.json" "$K" <<'PY'
import json, sys

path, k = sys.argv[1], int(sys.argv[2])
summary = json.load(open(path, encoding="utf-8"))
rows = [
    ("t_rel [%]",        summary["kitti"]["t_rel_percent"]),
    ("r_rel [deg/100m]", summary["kitti"]["r_rel_deg_per_m"] * 100),
    ("ATE-SE3 [m]",      summary["ate_se3_m"]),
    ("ATE 0009 [m]",     summary["drives"]["0009"]["ate_se3_m"]),
    ("ATE 0010 [m]",     summary["drives"]["0010"]["ate_se3_m"]),
]
print()
print("=" * 56)
print("KITTI-360  stride 3, drives 0009 + 0010")
print("=" * 56)
for name, value in rows:
    print(f"{name:<20} {float(value):>16.4f}")
print(f"{'segments':<20} {int(summary['kitti']['segment_count']):>16d}")
print("=" * 56)
PY

echo "[eval_kitti360] done -> $EVAL_DIR  |  $DRIVE_DIR"
