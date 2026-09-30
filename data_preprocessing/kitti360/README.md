# KITTI-360 preprocessing

[中文](README_CN.md) · [Home](../../README.md)

The rig consists of the rectified stereo pair `image_00/01` and the two fisheyes `image_02/03`.
Training drives: 0000, 0002–0007; evaluation drives: 0009, 0010. Register on the
[KITTI-360 website](https://www.cvlibs.net/datasets/kitti-360/) and accept its license before
downloading. Run the commands below from this directory.

```bash
# 0. download perspective and fisheye images, timestamps, calibration and poses (~480 GiB)
bash step0_download.sh --root /data/kitti360

# 1. rig metadata (~31 MiB)
python step1_prepare_kitti360.py --raw-root /data/kitti360/KITTI-360 --out-root /data/kitti360/streamrig_metadata

# 2. frozen front-end features (one process per GPU), then verify the cache
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python step2_precompute_infoshare.py \
    --processed-root /data/kitti360/streamrig_metadata --raw-root /data/kitti360/KITTI-360 \
    --out-root /data/kitti360/infoshare_bf16 --model-path /data/map-anything-model \
    --rank $r --world 4 &
done; wait
python step2_precompute_infoshare.py --processed-root /data/kitti360/streamrig_metadata \
  --out-root /data/kitti360/infoshare_bf16 --verify-only
```

Config entries:

```yaml
data:
  processed_root: /data/kitti360/streamrig_metadata
  infoshare_root: /data/kitti360/infoshare_bf16
```
