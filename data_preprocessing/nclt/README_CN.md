# NCLT 数据预处理

[English](README.md) · [首页](../../README_CN.md)

使用 NCLT Ladybug3 的五个侧向相机（`Cam1`–`Cam5`）。训练日期：
2012-01-08、2012-02-02、2012-02-04、2012-03-17、2012-05-26、2012-10-28、2012-11-17、2013-04-05；
评测日期：2012-02-19、2012-08-20。以下命令在本目录执行。

```bash
# 0. 下载图像、真值与标定（约 1 TB；--split test 只下载评测日期）
bash step0_download.sh --out-root /data/nclt_raw

# 1. 去畸变与裁剪（约 330 GiB，步骤 3 完成后可删除）
python step1_undistort_center_crop.py --raw-root /data/nclt_raw --calib-root /data/nclt_raw/calib \
    --out-root /data/nclt_centered

# 2. 轨迹与相机组标定
for s in train test; do
  python step2_prepare_rig.py --centered-root /data/nclt_centered --out-root /data/nclt_meta --split $s
done

# 3. 冻结前端特征（每张 GPU 一个进程；训练集换成 train 路径再跑一次）
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python step3_precompute_infoshare.py \
    --centered-root /data/nclt_centered \
    --step1-root /data/nclt_meta/step1_rig_trajectories_test \
    --step2-root /data/nclt_meta/step2_rig5_configs_test \
    --model-path /data/map-anything-model \
    --out-root /data/nclt_infoshare_test --rank $r --world 4 &
done; wait
```

配置项：

```yaml
data:
  step1_root: /data/nclt_meta/step1_rig_trajectories_train
  step2_root: /data/nclt_meta/step2_rig5_configs_train
  infoshare_root: /data/nclt_infoshare_train
```
