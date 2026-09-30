# NCLT preprocessing

[中文](README_CN.md) · [Home](../../README.md)

Uses the five side cameras (`Cam1`–`Cam5`) of the NCLT Ladybug3. Training sessions:
2012-01-08, 2012-02-02, 2012-02-04, 2012-03-17, 2012-05-26, 2012-10-28, 2012-11-17, 2013-04-05;
evaluation sessions: 2012-02-19, 2012-08-20. Run the commands below from this directory.

```bash
# 0. download images, ground truth and calibration (~1 TB; --split test for the evaluation sessions only)
bash step0_download.sh --out-root /data/nclt_raw

# 1. undistort and crop the images (~330 GiB, can be deleted after step 3)
python step1_undistort_center_crop.py --raw-root /data/nclt_raw --calib-root /data/nclt_raw/calib \
    --out-root /data/nclt_centered

# 2. trajectories and rig calibration
for s in train test; do
  python step2_prepare_rig.py --centered-root /data/nclt_centered --out-root /data/nclt_meta --split $s
done

# 3. frozen front-end features (one process per GPU; repeat with the train roots)
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python step3_precompute_infoshare.py \
    --centered-root /data/nclt_centered \
    --step1-root /data/nclt_meta/step1_rig_trajectories_test \
    --step2-root /data/nclt_meta/step2_rig5_configs_test \
    --model-path /data/map-anything-model \
    --out-root /data/nclt_infoshare_test --rank $r --world 4 &
done; wait
```

Config entries:

```yaml
data:
  step1_root: /data/nclt_meta/step1_rig_trajectories_train
  step2_root: /data/nclt_meta/step2_rig5_configs_train
  infoshare_root: /data/nclt_infoshare_train
```
