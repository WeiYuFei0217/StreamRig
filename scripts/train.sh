#!/bin/bash
# train.sh -- StreamRig 多卡 DDP 训练启动器。
#
# 功能：按给定 GPU 列表拉起 torchrun + scripts/train_stream.py，日志同时输出到控制台并追加到
# <output-dir>/train.log。相对路径按调用时的当前目录解析。本脚本不激活 conda，请先激活
# Python 环境（如 `conda activate streamrig`）。建议在 tmux 等持久会话中运行。
#
# 用法:
#   bash scripts/train.sh <gpus逗号分隔> <config> <output-dir> <master-port> [resume-ckpt] [weights-only]
# 例:
#   bash scripts/train.sh 0,1,2,3 configs/nclt_streamrig.yaml outputs/nclt_streamrig 29601
#
# 第 5 个参数可选：ckpt 路径，透传 train_stream.py 的 --resume
#   （恢复 model/optimizer/scheduler/epoch/step，跳过 warm-start），用于中断续训。
# 第 6 个参数非空 = 追加 --resume-weights-only（只加载权重，step/lr 从 0 开始）。
set -euo pipefail
usage() {
  echo "Usage: bash scripts/train.sh <gpus> <config> <output-dir> <master-port> [resume-ckpt] [weights-only]"
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ $# -lt 4 || $# -gt 6 ]]; then
  usage >&2
  exit 2
fi
GPUS="$1"; PORT="$4"; WONLY="${6:-}"
# 路径先按调用目录转成绝对路径，再切到仓库根
CFG="$(realpath -m -- "$2")"
OUT="$(realpath -m -- "$3")"
RESUME="${5:+$(realpath -m -- "$5")}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
IFS=',' read -r -a GPU_IDS <<< "$GPUS"
NGPU=${#GPU_IDS[@]}
if (( NGPU == 0 )); then
  echo "GPU list cannot be empty" >&2
  exit 2
fi
EXTRA=()
[[ -n "$RESUME" ]] && EXTRA+=(--resume "$RESUME")
[[ -n "$WONLY" ]] && EXTRA+=(--resume-weights-only)
mkdir -p "$OUT"
echo "[launch] gpus=$GPUS ngpu=$NGPU cfg=$CFG out=$OUT port=$PORT resume=${RESUME:-none} $(date '+%F %T')" \
  | tee -a "$OUT/train.log"
CUDA_VISIBLE_DEVICES="$GPUS" PYTORCH_ALLOC_CONF=expandable_segments:True \
  torchrun --nproc_per_node="$NGPU" --master-port="$PORT" \
  scripts/train_stream.py --config "$CFG" --output-dir "$OUT" ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 \
  | tee -a "$OUT/train.log"
