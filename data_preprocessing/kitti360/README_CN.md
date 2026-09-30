# KITTI-360 数据预处理

[English](README.md) · [首页](../../README_CN.md)

相机组由校正后的双目 `image_00/01` 与两个鱼眼相机 `image_02/03` 组成。训练 drive：0000、0002–0007；
评测 drive：0009、0010。下载前请在 [KITTI-360 官网](https://www.cvlibs.net/datasets/kitti-360/)注册并接受许可条款。
以下命令在本目录执行。

```bash
# 0. 下载透视与鱼眼图像、时间戳、标定和位姿（约 480 GiB）
bash step0_download.sh --root /data/kitti360

# 1. 相机组元数据（约 31 MiB）
python step1_prepare_kitti360.py --raw-root /data/kitti360/KITTI-360 --out-root /data/kitti360/streamrig_metadata

# 2. 冻结前端特征（每张 GPU 一个进程），完成后验收缓存
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python step2_precompute_infoshare.py \
    --processed-root /data/kitti360/streamrig_metadata --raw-root /data/kitti360/KITTI-360 \
    --out-root /data/kitti360/infoshare_bf16 --model-path /data/map-anything-model \
    --rank $r --world 4 &
done; wait
python step2_precompute_infoshare.py --processed-root /data/kitti360/streamrig_metadata \
  --out-root /data/kitti360/infoshare_bf16 --verify-only
```

配置项：

```yaml
data:
  processed_root: /data/kitti360/streamrig_metadata
  infoshare_root: /data/kitti360/infoshare_bf16
```
