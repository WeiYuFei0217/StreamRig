#!/usr/bin/env bash
#
# step0_download.sh -- 下载 StreamRig 所需的 NCLT 原始数据与一次性标定文件。
#
# 功能:
#   1) 10 个 session 的 Ladybug3 图像包 images/<date>_lb3.tar.gz (约 1.0 TB 合计)
#   2) 10 个 session 的真值位姿 ground_truth/groundtruth_<date>.csv (约 1.1 GB 合计)
#   3) 一次性共享标定: ladybug3_calib/cam_params.zip (K_cam*.csv + x_lb3_c*.csv)
#                     ladybug3_calib/U2D_ALL_1616X1232.tar.gz (去畸变查表, 151 MiB)
#   velodyne / hokuyo / sensor_data / covariance 本方法都不需要, 不下载。
#
# 特性: aria2c 16 连接并行(没装则自动回退 wget); 全部支持断点续传;
#       已完成条目记在 <out-root>/.download_done.txt, 重跑自动跳过;
#       ctrl+C 随时停止, 重跑继续。
#
# 用法:
#   bash step0_download.sh --out-root /data/nclt_raw                 # 全量(train+test)
#   bash step0_download.sh --out-root /data/nclt_raw --split test    # 只下 test 两个 session
#   bash step0_download.sh --out-root /data/nclt_raw --calib-only    # 只下标定文件
#   bash step0_download.sh --out-root /data/nclt_raw --no-extract    # 下完不解包
#   bash step0_download.sh --out-root /data/nclt_raw --delete-archive # 解包后删除 tar.gz
#   bash step0_download.sh --out-root /data/nclt_raw --sessions 2012-02-19 2012-08-20
#
# 下载后的目录布局 (解包后):
#   <out-root>/calib/cam_params/{K_cam0..5.csv, x_lb3_c0..5.csv}
#   <out-root>/calib/U2D/U2D_Cam{0..5}_1616X1232.txt
#   <out-root>/<date>/lb3/Cam{0..5}/<ts_us>.tiff
#   <out-root>/<date>/groundtruth_<date>.csv
#
# 磁盘: 图像 tar.gz 约 1.0 TB, 解包后 tiff 约 1.0 TB(解包后可删 tar.gz);
#       如果磁盘吃紧, 用 --split test 先跑通 test 两个 session(约 205 GB tar.gz)。

set -uo pipefail

BASE_URL="https://s3.us-east-2.amazonaws.com/nclt.perl.engin.umich.edu"
CALIB_URL="${BASE_URL}/ladybug3_calib"
CONNECTIONS=16

# ---- StreamRig 官方划分(与 step2_prepare_rig.py 的常量保持一致) ----
TRAIN_SESSIONS=(2012-01-08 2012-02-02 2012-02-04 2012-03-17 2012-05-26 2012-10-28 2012-11-17 2013-04-05)
TEST_SESSIONS=(2012-02-19 2012-08-20)

OUT_ROOT=""
SPLIT="all"
CALIB_ONLY=false
EXTRACT=true
KEEP_ARCHIVE=true
USER_SESSIONS=()

usage() { sed -n '2,31p' "$0"; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-root)   OUT_ROOT="$2"; shift 2 ;;
        --split)      SPLIT="$2"; shift 2 ;;
        --sessions)   shift; while [[ $# -gt 0 && "$1" != --* ]]; do USER_SESSIONS+=("$1"); shift; done ;;
        --calib-only) CALIB_ONLY=true; shift ;;
        --no-extract) EXTRACT=false; shift ;;
        --delete-archive) KEEP_ARCHIVE=false; shift ;;
        -h|--help)    usage 0 ;;
        *) echo "未知参数: $1"; usage 1 ;;
    esac
done

[[ -z "$OUT_ROOT" ]] && { echo "必须指定 --out-root"; usage 1; }
mkdir -p "$OUT_ROOT"
DONE_LOG="${OUT_ROOT}/.download_done.txt"
touch "$DONE_LOG"

# ---- 选 session ----
if [[ ${#USER_SESSIONS[@]} -gt 0 ]]; then
    SESSIONS=("${USER_SESSIONS[@]}")
else
    case "$SPLIT" in
        train) SESSIONS=("${TRAIN_SESSIONS[@]}") ;;
        test)  SESSIONS=("${TEST_SESSIONS[@]}") ;;
        all)   SESSIONS=("${TRAIN_SESSIONS[@]}" "${TEST_SESSIONS[@]}") ;;
        *) echo "--split 只能是 train / test / all"; exit 1 ;;
    esac
fi

# ---- 下载器: aria2c 优先, 否则 wget ----
if command -v aria2c >/dev/null 2>&1; then
    DOWNLOADER="aria2c"
elif command -v wget >/dev/null 2>&1; then
    DOWNLOADER="wget"
    echo "[WARN] 未找到 aria2c, 回退 wget 单连接(建议 apt install aria2); 下载日志: ${OUT_ROOT}/.wget.log"
else
    echo "[ERROR] 既没有 aria2c 也没有 wget, 无法下载"; exit 1
fi

fetch() {  # fetch <url> <output_path> <label>
    local url="$1" output="$2" label="$3"
    if grep -qxF "$label" "$DONE_LOG" 2>/dev/null; then
        echo "[SKIP] $label"; return 0
    fi
    mkdir -p "$(dirname "$output")"
    echo ""; echo "[DOWN] $label"; echo "       $url"
    local ok=false
    if [[ "$DOWNLOADER" == "aria2c" ]]; then
        aria2c -x "$CONNECTIONS" -s "$CONNECTIONS" -k 20M -c \
               --file-allocation=none --summary-interval=30 \
               --auto-file-renaming=false --allow-overwrite=true \
               -d "$(dirname "$output")" -o "$(basename "$output")" "$url" && ok=true
    else
        # 日志写到 <out-root>/.wget.log（不在当前目录生成 wget-log）
        wget --continue -a "${OUT_ROOT}/.wget.log" -O "$output" "$url" && ok=true
    fi
    if $ok; then
        echo "[DONE] $label ($(du -h "$output" | cut -f1))"
        echo "$label" >> "$DONE_LOG"; return 0
    fi
    echo "[FAIL] $label -- 重新运行本脚本可续传"; return 1
}

echo "=========================================="
echo "  NCLT download for StreamRig"
echo "  out-root : $OUT_ROOT"
echo "  split    : $SPLIT  (${#SESSIONS[@]} sessions)"
echo "  sessions : ${SESSIONS[*]}"
echo "  tool     : $DOWNLOADER"
echo "  started  : $(date)"
echo "=========================================="

# ---------------- Phase 0: 一次性标定 ----------------
echo ""; echo "--- Phase 0: calibration (cam_params + U2D maps) ---"
CALIB_DIR="${OUT_ROOT}/calib"
mkdir -p "$CALIB_DIR"
fetch "${CALIB_URL}/cam_params.zip"            "${CALIB_DIR}/cam_params.zip"  "calib/cam_params" || exit 1
fetch "${CALIB_URL}/U2D_ALL_1616X1232.tar.gz"  "${CALIB_DIR}/U2D_ALL.tar.gz"  "calib/U2D"        || exit 1

if $EXTRACT; then
    if [[ ! -f "${CALIB_DIR}/cam_params/K_cam1.csv" ]]; then
        # cam_params.zip 顶层可能带也可能不带 cam_params/ 目录, 统一整理到 calib/cam_params/
        rm -rf "${CALIB_DIR}/_cp"; mkdir -p "${CALIB_DIR}/_cp"
        unzip -oq "${CALIB_DIR}/cam_params.zip" -d "${CALIB_DIR}/_cp"
        mkdir -p "${CALIB_DIR}/cam_params"
        find "${CALIB_DIR}/_cp" -name '*.csv' -exec mv -f {} "${CALIB_DIR}/cam_params/" \;
        rm -rf "${CALIB_DIR}/_cp"
        echo "  解包 cam_params -> ${CALIB_DIR}/cam_params"
    fi
    if [[ ! -f "${CALIB_DIR}/U2D/U2D_Cam1_1616X1232.txt" ]]; then
        rm -rf "${CALIB_DIR}/_u2d"; mkdir -p "${CALIB_DIR}/_u2d"
        tar xzf "${CALIB_DIR}/U2D_ALL.tar.gz" -C "${CALIB_DIR}/_u2d"
        mkdir -p "${CALIB_DIR}/U2D"
        find "${CALIB_DIR}/_u2d" -name 'U2D_Cam*_1616X1232.txt' -exec mv -f {} "${CALIB_DIR}/U2D/" \;
        rm -rf "${CALIB_DIR}/_u2d"
        echo "  解包 U2D -> ${CALIB_DIR}/U2D"
    fi
fi

$CALIB_ONLY && { echo ""; echo "--calib-only 指定, 结束。"; exit 0; }

# ---------------- Phase 1: ground truth (小文件, 先下) ----------------
echo ""; echo "--- Phase 1: ground truth poses (~110 MB each) ---"
for s in "${SESSIONS[@]}"; do
    fetch "${BASE_URL}/ground_truth/groundtruth_${s}.csv" \
          "${OUT_ROOT}/${s}/groundtruth_${s}.csv" "${s}/groundtruth" || exit 1
done

# ---------------- Phase 2: Ladybug3 图像 (逐个下, 集中带宽) ----------------
echo ""; echo "--- Phase 2: Ladybug3 images (~90-115 GB each, 逐个下载) ---"
for s in "${SESSIONS[@]}"; do
    ARCHIVE="${OUT_ROOT}/${s}/${s}_lb3.tar.gz"
    fetch "${BASE_URL}/images/${s}_lb3.tar.gz" "$ARCHIVE" "${s}/lb3" || exit 1

    if $EXTRACT; then
        if [[ -d "${OUT_ROOT}/${s}/lb3/Cam1" ]]; then
            echo "  ${s}: lb3/ 已解包, 跳过"
        else
            echo "  ${s}: 解包中 (tar 内路径形如 ${s}/lb3/Cam0/*.tiff) ..."
            tar xzf "$ARCHIVE" -C "$OUT_ROOT" || { echo "[FAIL] 解包 ${s}"; exit 1; }
            echo "  ${s}: 解包完成"
        fi
        if ! $KEEP_ARCHIVE; then
            rm -f "$ARCHIVE"; echo "  ${s}: 已删除 tar.gz (--delete-archive)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "  全部完成 $(date)"
echo "  下一步: python step1_undistort_center_crop.py \\"
echo "            --raw-root ${OUT_ROOT} --calib-root ${OUT_ROOT}/calib \\"
echo "            --out-root <centered_root> --split ${SPLIT}"
echo "=========================================="
