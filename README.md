# StreamRig: Exploiting Intra-Rig Geometry for Streaming Multi-Camera Odometry

<p align="center">
  <a href="https://weiyufei0217.github.io/StreamRig_Anonymous/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-2EA44F?style=for-the-badge" alt="Project Page"></a>
</p>

[中文](README_CN.md)

StreamRig is a causal multi-camera visual odometry method. A frozen MapAnything front-end
perceives each rig together with its calibration, a Rig-Resampler compresses every camera into
a few latent tokens, a CausalBridge attends over the compressed rig history, and a pose head
predicts each rig's metric pose relative to a periodically re-anchored reference.

<p align="center"><img src="assets/figs/streamrig-overview.jpg" width="100%" alt="StreamRig architecture"/></p>

## Installation

```bash
conda create -n streamrig python=3.12 -y
conda activate streamrig
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e ./third_party/mapanything
pip install -e .
```

Download the MapAnything backbone (DINOv2-large) as described in the
[G2G installation guide](https://github.com/WeiYuFei0217/G2G#2-install-mapanything-backbone).

## Data

Each dataset is converted into rig metadata and a cache of frozen front-end features:

- [NCLT](data_preprocessing/nclt/README.md) (5 cameras, ~470 GiB feature cache)
- [KITTI-360](data_preprocessing/kitti360/README.md) (4 cameras, ~104 GiB feature cache)

## Checkpoints

Download from [Hugging Face](https://huggingface.co/feixue22/StreamRig) or
[Baidu Netdisk](https://pan.baidu.com/s/1sPUkvda6HD85fm3q_KPkAw?pwd=8888) (code `8888`)
into `release_weights/`, then check them with `cd release_weights && sha256sum -c SHA256SUMS`.

| File | Dataset |
|---|---|
| `NCLT-StreamRig.pth` | NCLT |
| `KITTI360-StreamRig.pth` | KITTI-360 |

## Training

Fill in the `/path/to/...` entries of the [config](configs/). `model.warmstart_ckpt` is a
relocalization model trained with [G2G](https://github.com/WeiYuFei0217/G2G#relocalization-task-1).

```bash
bash scripts/train.sh 0,1,2,3 configs/nclt_streamrig.yaml outputs/nclt_streamrig 29601
bash scripts/train.sh 0,1,2,3 configs/kitti360_streamrig.yaml outputs/kitti360_streamrig 29611
```

## Evaluation

```bash
bash scripts/eval_nclt.sh \
  --step1-root /data/nclt_meta/step1_rig_trajectories_test \
  --step2-root /data/nclt_meta/step2_rig5_configs_test \
  --infoshare-root /data/nclt_infoshare_test \
  --out-dir outputs/eval_nclt

bash scripts/eval_kitti360.sh \
  --processed-root /data/kitti360/streamrig_metadata \
  --infoshare-root /data/kitti360/infoshare_bf16 \
  --out-dir outputs/eval_kitti360
```

Both scripts use the released weights by default; add `--ckpt <final.pt or exported .pth>` to
evaluate your own model.

| Dataset | Sequences | t_rel (%) | r_rel (°/100 m) | ATE (m) |
|---|---|---|---|---|
| NCLT | 2012-02-19, 2012-08-20 | 2.77 | 1.39 | 28.4 |
| KITTI-360 | 0009, 0010 | 2.59 | 0.98 | 63.7 |

Metrics follow the KITTI odometry protocol on 100–800 m segments at stride 3, and ATE uses SE(3)
alignment on complete sequences. Where a recording is interrupted, the pieces on either side are
joined with the ground-truth relative pose before the metrics are computed.
NCLT evaluation needs about 35 GB of host memory.

## Acknowledgements and license

StreamRig builds on [MapAnything](https://github.com/facebookresearch/map-anything),
[DINOv2](https://github.com/facebookresearch/dinov2) and G2G. The code is released under
[CC BY-NC 4.0](LICENSE); the bundled MapAnything code keeps its
[Apache 2.0 license](third_party/mapanything/LICENSE).
