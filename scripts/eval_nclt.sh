#!/usr/bin/env bash
# ============================================================================
# eval_nclt.sh -- NCLT 一键评测（流式评测 → 100–800 m 汇总 → 打印指标）。
#
# 评测设置：
#   test 划分的两条完整日期 2012-02-19 / 2012-08-20，stride 3，
#   100–800 m KITTI 相对误差 + 整日期 ATE，两日期等权平均；
#   按时间戳空洞分段，段间以真值相对位姿衔接。
#
# 用法：
#   bash scripts/eval_nclt.sh \
#       --step1-root   /path/to/nclt_meta/step1_rig_trajectories_test \
#       --step2-root   /path/to/nclt_meta/step2_rig5_configs_test \
#       --infoshare-root /path/to/nclt_infoshare_test \
#       --out-dir      outputs/eval_nclt
#
# 可选参数：
#   --config <yaml>    默认 configs/nclt_streamrig.yaml
#   --ckpt <pth>       默认 release_weights/NCLT-StreamRig.pth
#   --exp-name <name>  默认 streamrig_nclt
#   --k <int>          interval-k（重锚间隔），默认 23
#   --gpu <id>         默认 0（单卡串行；NCLT 单条序列特征预载约 35 GB 内存）
# ============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG="$REPO_ROOT/configs/nclt_streamrig.yaml"
CKPT="$REPO_ROOT/release_weights/NCLT-StreamRig.pth"
EXP_NAME="streamrig_nclt"
K=23
GPU=0
STEP1_ROOT=""; STEP2_ROOT=""; INFOSHARE_ROOT=""; OUT_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)         CONFIG="$2"; shift 2 ;;
    --ckpt)           CKPT="$2"; shift 2 ;;
    --step1-root)     STEP1_ROOT="$2"; shift 2 ;;
    --step2-root)     STEP2_ROOT="$2"; shift 2 ;;
    --infoshare-root) INFOSHARE_ROOT="$2"; shift 2 ;;
    --out-dir)        OUT_DIR="$2"; shift 2 ;;
    --exp-name)       EXP_NAME="$2"; shift 2 ;;
    --k)              K="$2"; shift 2 ;;
    --gpu)            GPU="$2"; shift 2 ;;
    -h|--help)        sed -n '2,23p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

for pair in "STEP1_ROOT:--step1-root" "STEP2_ROOT:--step2-root" \
            "INFOSHARE_ROOT:--infoshare-root" "OUT_DIR:--out-dir"; do
  var="${pair%%:*}"; flag="${pair##*:}"
  if [[ -z "${!var}" ]]; then echo "缺少必填参数 $flag" >&2; exit 2; fi
done

mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
EVAL_DIR="$OUT_DIR/$EXP_NAME"
RESOLVED_CONFIG="$OUT_DIR/resolved_nclt_config.yaml"
W_MAX=$((K + 1))

# --- 1) 把发布 config 里的 /path/to 占位符替换成本机真实路径 ---------------
python - "$CONFIG" "$RESOLVED_CONFIG" "$STEP1_ROOT" "$STEP2_ROOT" "$INFOSHARE_ROOT" <<'PY'
import sys, yaml
src, dst, step1, step2, infoshare = sys.argv[1:6]
cfg = yaml.safe_load(open(src, encoding="utf-8"))
cfg["model"].pop("warmstart_ckpt", None)          # 评测不需要 warm-start 权重
data = cfg["data"]
data["step1_root"] = step1
data["step2_root"] = step2
data["infoshare_root"] = infoshare
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
print(f"[eval_nclt] resolved config -> {dst}")
PY

# --- 2) 流式评测（--gap-threshold-ms 极大 = 每个日期整条评测） -------------
echo "[eval_nclt] step 1/3  streaming evaluation (stride 3) on GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" python "$REPO_ROOT/scripts/eval_stream.py" \
  --config "$RESOLVED_CONFIG" --ckpt "$CKPT" \
  --exp-name "$EXP_NAME" --out-dir "$OUT_DIR" \
  --dataset nclt --stride 3 --w-max "$W_MAX" --intervals "$K" \
  --step-size 10 --gap-threshold-ms 100000000 --save-poses

# --- 3) 100–800 m 汇总 -----------------------------------------------------
echo "[eval_nclt] step 2/3  100-800 m summary"
python "$REPO_ROOT/scripts/summarize_nclt.py" \
  --eval-dir "$EVAL_DIR" --stride 3 --intervals "$K"

# --- 4) 打印指标 ----------------------------------------------------------
echo "[eval_nclt] step 3/3  metrics"
python - "$EVAL_DIR/metrics_100_800m.json" "$K" <<'PY'
import json, sys

path, k = sys.argv[1], int(sys.argv[2])
grid = json.load(open(path, encoding="utf-8"))
row = grid["grid"][str(k + 1)]
rows = [("mean", row)] + [(item["traj"], item) for item in grid["per_sequence"][str(k + 1)]]
print()
print("=" * 72)
print("NCLT  stride 3, 100-800 m, two dates weighted equally")
print("=" * 72)
print(f"{'sequence':<12} {'t_rel [%]':>14} {'r_rel [deg/100m]':>18} {'ATE-SE3 [m]':>14}")
for label, item in rows:
    print(f"{label:<12} {float(item['t_rel_percent']):>14.4f} "
          f"{float(item['r_rel_deg_per_m']) * 100:>18.4f} {float(item['ate_se3_m']):>14.4f}")
print("=" * 72)
PY

echo "[eval_nclt] done -> $EVAL_DIR"
