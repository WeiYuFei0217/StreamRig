#!/usr/bin/env bash
# ======================================================================================
# step0_download.sh -- 下载 StreamRig 在 KITTI-360 上训练/评测所需的官方数据包（可续传、可并行）。
#
# 功能：只下载四相机 rig 里程计需要的部分，**不下载**深度/语义/点云/3D 标注：
#   * 9 条 drive 的 image_00/01（透视 rectified，data_rect，1408x376）
#   * 9 条 drive 的 image_02/03（鱼眼原图，data_rgb，1400x1400）
#   * data_timestamps_perspective.zip / data_timestamps_fisheye.zip
#   * calibration.zip（相机内外参 + 鱼眼 MEI 标定）
#   * data_poses.zip（poses.txt 与 cam0_to_world.txt）
#
# 官方 ZIP 合计约 477 GiB，解压后标准 PNG 约 480 GiB（ZIP 里 PNG 基本不可再压缩，
# 下载过程中 zip 与解压结果会同时占盘，因此建议至少准备 520 GiB 可用空间：脚本每
# 下完一个包就校验 → 解压 → 删除该 zip，峰值额外占用只有单个最大包（约 34 GiB）。
#
# 每个包下载/解压成功后会写 ``<root>/.download_complete/<zip 名>`` 标记；重跑脚本会
# 跳过已完成的包，中断的下载由 ``wget -c`` 断点续传。
#
# 示例用法：
#   bash step0_download.sh --root /path/to/kitti360            # 全量下载（默认 4 并行）
#   bash step0_download.sh --root /path/to/kitti360 --jobs 8   # 8 并行
#   bash step0_download.sh --root /path/to/kitti360 --list     # 只打印清单与体积，不下载
#   bash step0_download.sh --root /path/to/kitti360 --drives 0009,0010   # 只下评测两条
#
# 下载完成后的目录（``--root`` 下的 KITTI-360 即 step1 的 --raw-root）：
#   <root>/KITTI-360/calibration/{calib_cam_to_pose.txt,perspective.txt,image_02.yaml,image_03.yaml}
#   <root>/KITTI-360/data_poses/<drive>/{poses.txt,cam0_to_world.txt}
#   <root>/KITTI-360/data_2d_raw/<drive>/image_0{0,1}/{data_rect/*.png,timestamps.txt}
#   <root>/KITTI-360/data_2d_raw/<drive>/image_0{2,3}/{data_rgb/*.png,timestamps.txt}
#
# 注意：KITTI-360 采用 CC BY-NC-SA 4.0 许可，下载前请先在官网注册并接受许可条款：
#   https://www.cvlibs.net/datasets/kitti-360/
# ======================================================================================
set -uo pipefail

S3_2D="https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/data_2d_raw"
S3_CALIB="https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip"
S3_POSES="https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip"

ALL_DRIVES=(0000 0002 0003 0004 0005 0006 0007 0009 0010)
ALL_CAMERAS=(00 01 02 03)

ROOT=""
JOBS=4
LIST_ONLY=0
DRIVES_ARG=""
CAMERAS_ARG=""

usage() {
    sed -n '2,33p' "$0"
    exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --root) ROOT="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --drives) DRIVES_ARG="$2"; shift 2 ;;
        --cameras) CAMERAS_ARG="$2"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "未知参数: $1" >&2; usage 1 ;;
    esac
done

if [[ -z "$ROOT" ]]; then
    echo "必须提供 --root <下载目录>" >&2
    usage 1
fi

# 逗号分隔的子集筛选（默认全量）
IFS=',' read -r -a DRIVES <<< "${DRIVES_ARG:-$(IFS=,; echo "${ALL_DRIVES[*]}")}"
IFS=',' read -r -a CAMERAS <<< "${CAMERAS_ARG:-$(IFS=,; echo "${ALL_CAMERAS[*]}")}"

DATA_ROOT="$ROOT/KITTI-360"
MARKER_DIR="$DATA_ROOT/.download_complete"

# 官方包体积（GiB，来自 S3 对象大小，用于清单打印）
declare -A ZIP_GIB=(
    [2013_05_28_drive_0000_sync_image_00.zip]=9.00  [2013_05_28_drive_0000_sync_image_01.zip]=8.96
    [2013_05_28_drive_0000_sync_image_02.zip]=25.33 [2013_05_28_drive_0000_sync_image_03.zip]=25.27
    [2013_05_28_drive_0002_sync_image_00.zip]=11.97 [2013_05_28_drive_0002_sync_image_01.zip]=11.92
    [2013_05_28_drive_0002_sync_image_02.zip]=33.31 [2013_05_28_drive_0002_sync_image_03.zip]=33.56
    [2013_05_28_drive_0003_sync_image_00.zip]=0.82  [2013_05_28_drive_0003_sync_image_01.zip]=0.81
    [2013_05_28_drive_0003_sync_image_02.zip]=2.30  [2013_05_28_drive_0003_sync_image_03.zip]=2.42
    [2013_05_28_drive_0004_sync_image_00.zip]=9.23  [2013_05_28_drive_0004_sync_image_01.zip]=9.18
    [2013_05_28_drive_0004_sync_image_02.zip]=26.86 [2013_05_28_drive_0004_sync_image_03.zip]=27.17
    [2013_05_28_drive_0005_sync_image_00.zip]=5.41  [2013_05_28_drive_0005_sync_image_01.zip]=5.37
    [2013_05_28_drive_0005_sync_image_02.zip]=15.71 [2013_05_28_drive_0005_sync_image_03.zip]=15.68
    [2013_05_28_drive_0006_sync_image_00.zip]=7.95  [2013_05_28_drive_0006_sync_image_01.zip]=7.89
    [2013_05_28_drive_0006_sync_image_02.zip]=22.92 [2013_05_28_drive_0006_sync_image_03.zip]=22.94
    [2013_05_28_drive_0007_sync_image_00.zip]=2.76  [2013_05_28_drive_0007_sync_image_01.zip]=2.75
    [2013_05_28_drive_0007_sync_image_02.zip]=7.78  [2013_05_28_drive_0007_sync_image_03.zip]=8.07
    [2013_05_28_drive_0009_sync_image_00.zip]=11.30 [2013_05_28_drive_0009_sync_image_01.zip]=11.20
    [2013_05_28_drive_0009_sync_image_02.zip]=33.03 [2013_05_28_drive_0009_sync_image_03.zip]=33.09
    [2013_05_28_drive_0010_sync_image_00.zip]=3.11  [2013_05_28_drive_0010_sync_image_01.zip]=3.08
    [2013_05_28_drive_0010_sync_image_02.zip]=9.34  [2013_05_28_drive_0010_sync_image_03.zip]=9.28
    [data_timestamps_perspective.zip]=0.001 [data_timestamps_fisheye.zip]=0.001
)

# 组装本次要下载的 2D 包清单
zips=()
for drive in "${DRIVES[@]}"; do
    for camera in "${CAMERAS[@]}"; do
        zips+=("2013_05_28_drive_${drive}_sync_image_${camera}.zip")
    done
done
zips+=(data_timestamps_perspective.zip data_timestamps_fisheye.zip)

total_gib=0
for zip_name in "${zips[@]}"; do
    total_gib=$(awk -v a="$total_gib" -v b="${ZIP_GIB[$zip_name]:-0}" 'BEGIN{printf "%.2f", a+b}')
done

echo "===================== KITTI-360 下载清单 ====================="
echo "目标目录 : $DATA_ROOT"
echo "drives   : ${DRIVES[*]}"
echo "cameras  : ${CAMERAS[*]}  (00/01=透视 data_rect, 02/03=鱼眼 data_rgb)"
printf '%-52s %8s\n' "包名" "GiB"
for zip_name in "${zips[@]}"; do
    printf '%-52s %8s\n' "$zip_name" "${ZIP_GIB[$zip_name]:-?}"
done
printf '%-52s %8s\n' "calibration.zip" "0.00"
printf '%-52s %8s\n' "data_poses.zip" "0.01"
echo "-------------------------------------------------------------"
printf '%-52s %8s\n' "2D 包合计" "$total_gib"
echo "解压后 PNG 约 480 GiB（全量）；建议预留 520 GiB 以上可用空间。"
echo "============================================================="

if [[ "$LIST_ONLY" -eq 1 ]]; then
    exit 0
fi

mkdir -p "$DATA_ROOT/data_2d_raw" "$MARKER_DIR" || exit 1

download_one() {
    # $1 = zip 名（位于 data_2d_raw 前缀下）
    local zip_name="$1"
    if [[ -f "$MARKER_DIR/$zip_name" ]]; then
        printf '[skip] %s（已完成）\n' "$zip_name"
        return 0
    fi
    printf '[start] %s\n' "$zip_name"
    cd "$DATA_ROOT" || return 1
    # -c 断点续传；--tries=0 + retry-connrefused 应对长时间大文件传输
    wget -c --retry-connrefused --waitretry=5 --read-timeout=60 \
        --timeout=60 --tries=0 --progress=dot:giga "$S3_2D/$zip_name" || return 1
    unzip -tq "$zip_name" || { printf '[bad-zip] %s\n' "$zip_name"; return 1; }
    unzip -oq -d data_2d_raw "$zip_name" || return 1
    rm -f -- "$zip_name"
    touch "$MARKER_DIR/$zip_name"
    printf '[done] %s\n' "$zip_name"
}
export -f download_one
export DATA_ROOT MARKER_DIR S3_2D

# 标定与位姿（很小，串行下）
# 注意两个包的内部布局不同：calibration.zip 自带 calibration/ 顶层目录，
# data_poses.zip 的成员直接是 <drive>/poses.txt，必须解到 data_poses/ 之下。
cd "$ROOT" || exit 1
for spec in "calibration.zip|$S3_CALIB|$DATA_ROOT" "data_poses.zip|$S3_POSES|$DATA_ROOT/data_poses"; do
    IFS='|' read -r zip_name url dest <<< "$spec"
    if [[ -f "$MARKER_DIR/$zip_name" ]]; then
        printf '[skip] %s（已完成）\n' "$zip_name"
        continue
    fi
    wget -c --retry-connrefused --waitretry=5 --timeout=60 --tries=0 "$url" || exit 1
    unzip -tq "$zip_name" || exit 1
    mkdir -p "$dest" || exit 1
    unzip -oq -d "$dest" "$zip_name" || exit 1
    rm -f -- "$zip_name"
    touch "$MARKER_DIR/$zip_name"
    printf '[done] %s → %s\n' "$zip_name" "$dest"
done

# 2D 图像包并行下载（每个包独立续传 + 独立完成标记）
printf '%s\n' "${zips[@]}" \
    | xargs -r -n 1 -P "$JOBS" bash -c 'download_one "$1"' _
status=$?

if [[ "$status" -ne 0 ]]; then
    echo "[失败] 有包未成功；重跑本脚本会跳过已完成的包并续传未完成的包" >&2
    exit "$status"
fi

echo "[完成] $(date --iso-8601=seconds)"
echo "下一步：python step1_prepare_kitti360.py --raw-root $DATA_ROOT --out-root <metadata 目录>"
