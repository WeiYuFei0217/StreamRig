"""
stream_odom_dataset.py -- StreamRig 流式里程计数据集（NCLT 五目 rig 的 step1/step2 布局）。

一个样本 = 同一轨迹内 N 个 rig 时刻（里程计窗口），每 rig 用全部 K 相机。
  - 特征：读预计算的 MapAnything info-sharing 缓存 rig.npy（uint16 bit-cast bfloat16, [K,256,C]）。
  - 采样：随机 valid run + 随机锚点；窗口内相邻 rig 的帧间隔逐对独立地从
    [stride_min, stride_max] 均匀采样。评测脚本直接读取 seqs / runs 做完整序列推理。
  - 缓存根须含全量生成完成后写入的 COMPLETE 标记；缺 rig 文件直接报错。

变长 N（配合 VariableNStreamBatchSampler）：__getitem__ 接受 (N, seed) 元组 index，
每 batch 内 N 一致、batch 间 N 按 P(N) ∝ N^length_bias_power 采样。

其他约定：
  - valid-run 切分：相邻原生帧时距 > gap_threshold_ms 处切开，窗口不跨传感器空洞；
    短于 num_rigs 帧或跨度不足 min_run_span_ms 的 run 被丢弃。
  - GT 锚系位姿 T_rel_gt[i] = inv(T_w_ref0) @ poses[ts_i] @ body_T_cam0（T_rel_gt[0]=I），
    里程计 T_0->t 即 T_rel_gt[t]。

磁盘布局：step1_root/scenes/nclt/trajectories/{date}/trajectory.tum，
infoshare_root/scenes/nclt/trajectories/{date}/features/{ts}/rig.npy，
rig_config 在 step2_root/scenes/nclt/rig_config.json（每相机 body_T_cam 外参与内参）。

输出字段（供 StreamRigModel + StreamPoseLoss）：
  infoshare:  [N, K, 256, 768] float32  特征
  rig_calib:  [K, 4, 4]  rig 标定（各相机相对 cam0）
  T_rel_gt:   [N, 4, 4]  锚系位姿（=里程计，T_rel_gt[0]=I）
  scene_id / traj_id / ts_list

示例用法：
    ds = StreamOdomDataset(step1_root, step2_root, "/path/to/nclt_infoshare_train",
                           num_rigs=48, num_cameras=5,
                           stride_min=1, stride_max=6, samples_per_epoch=57600)
    loader = DataLoader(ds, batch_size=2, collate_fn=stream_collate_fn, num_workers=4)
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.utils.data
from scipy.spatial.transform import Rotation

# info-sharing 特征通道数（MapAnything）
INFOSHARE_DIM = 768


def stream_collate_fn(batch: list[dict]) -> dict:
    """张量 stack，字符串/列表保持 list。"""
    out = {}
    for key in batch[0]:
        vals = [b[key] for b in batch]
        out[key] = torch.stack(vals) if isinstance(vals[0], torch.Tensor) else vals
    return out


def _load_tum_ms(path: str) -> dict[str, np.ndarray]:
    """TUM 轨迹 → {毫秒键: T_world_body(4,4)}。"""
    poses: dict[str, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            ts_key = f"{int(round(float(parts[0]) * 1000)):010d}"
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = Rotation.from_quat([float(x) for x in parts[4:8]]).as_matrix()
            mat[:3, 3] = [float(x) for x in parts[1:4]]
            poses[ts_key] = mat
    return poses


def _split_valid_runs(ts_int, gap_threshold_ms, num_rigs_min, min_run_span_ms):
    """升序 ms 时间戳 → valid run [(s,e)]，相邻间隔 > gap 处切开；段须够长。"""
    def _ok(s, e):
        if e - s < num_rigs_min:
            return False
        return int(ts_int[e - 1] - ts_int[s]) >= min_run_span_ms

    runs, run_start, n = [], 0, len(ts_int)
    for i in range(1, n):
        if ts_int[i] - ts_int[i - 1] > gap_threshold_ms:
            if _ok(run_start, i):
                runs.append((run_start, i))
            run_start = i
    if n and _ok(run_start, n):
        runs.append((run_start, n))
    return runs


class StreamOdomDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        step1_root: str,
        step2_root: str,
        infoshare_root: str,                # info-sharing 特征缓存根（与 step1/step2 同一划分）
        num_rigs: int = 12,
        num_cameras: int = 5,
        stride_min: int = 1,
        stride_max: int = 6,
        gap_threshold_ms: int = 400,
        min_run_span_ms: int = 30000,
        samples_per_epoch: int = 100000,
    ):
        super().__init__()
        self.step2_root = step2_root
        if not infoshare_root:
            raise ValueError("StreamOdomDataset 需要 infoshare_root")
        if not os.path.isfile(os.path.join(infoshare_root, "COMPLETE")):
            raise RuntimeError(
                f"NCLT info-sharing 缓存不完整（缺 COMPLETE 标记）: {infoshare_root}；"
                "请用 data_preprocessing/nclt/step3_precompute_infoshare.py 全量生成")
        self.infoshare_root = infoshare_root
        self.N = num_rigs
        self.K = num_cameras
        self.stride_min = stride_min
        self.stride_max = stride_max
        self.samples_per_epoch = samples_per_epoch

        self._rig_configs: dict[str, dict] = {}   # scene_id -> rig_config
        self._calib_cache: dict[str, tuple] = {}  # scene_id -> (body_T_cams, rig_calib)

        # ---- 枚举 (scene, traj) -> valid runs（以特征缓存中存在的 rig 为准）----
        feat_scenes = os.path.join(infoshare_root, "scenes")
        step1_scenes = os.path.join(step1_root, "scenes")
        scene_ids = sorted(os.listdir(feat_scenes)) if os.path.isdir(feat_scenes) else []

        self.seqs: list[dict] = []  # {scene, traj, ts(list[str]), poses, runs}
        for scene_id in scene_ids:
            rig_path = os.path.join(step2_root, "scenes", scene_id, "rig_config.json")
            if not os.path.isfile(rig_path):
                continue
            traj_dir = os.path.join(feat_scenes, scene_id, "trajectories")
            if not os.path.isdir(traj_dir):
                continue
            for traj_id in sorted(os.listdir(traj_dir)):
                feat_dir = os.path.join(traj_dir, traj_id, "features")
                tum_path = os.path.join(
                    step1_scenes, scene_id, "trajectories", traj_id, "trajectory.tum")
                if not os.path.isdir(feat_dir) or not os.path.isfile(tum_path):
                    continue
                poses = _load_tum_ms(tum_path)
                ts = sorted(set(os.listdir(feat_dir)) & set(poses.keys()))
                if len(ts) < num_rigs:
                    continue
                ts_int = np.array([int(t) for t in ts])
                runs = _split_valid_runs(ts_int, gap_threshold_ms, num_rigs, min_run_span_ms)
                if not runs:
                    continue
                self.seqs.append({
                    "scene": scene_id, "traj": traj_id,
                    "ts": ts, "poses": poses, "runs": runs,
                })
                if scene_id not in self._rig_configs:
                    with open(rig_path, "r", encoding="utf-8") as f:
                        self._rig_configs[scene_id] = json.load(f)
        if not self.seqs:
            raise RuntimeError(
                f"没有找到可用序列：step1_root={step1_root}, step2_root={step2_root}, "
                f"infoshare_root={infoshare_root}（三者须对应同一划分）")

        # 采样权重 ∝ run 长度；能放下 N 个 rig（min stride）的 run 才可用
        min_span = (num_rigs - 1) * stride_min
        self._run_flat, weights = [], []
        for si, s in enumerate(self.seqs):
            for (rs, re) in s["runs"]:
                if (re - rs) >= min_span + 1:
                    self._run_flat.append((si, rs, re))
                    weights.append(re - rs)
        if not self._run_flat:
            raise RuntimeError(f"没有能容纳 N={num_rigs} 的连续片段: {step1_root}")
        w = np.array(weights, dtype=np.float64)
        self._run_probs = w / w.sum()

        print(f"[StreamOdomDataset] {len(self.seqs)} 序列, {len(self._run_flat)} 可用run, "
              f"N={num_rigs}, K={num_cameras}, stride∈[{stride_min},{stride_max}]", flush=True)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_body_T_cam(rig_config, cam_idx):
        cam = rig_config["cameras"][cam_idx]
        return np.array(cam["body_T_cam"], dtype=np.float64).reshape(4, 4)

    def _get_calib(self, scene_id):
        """返回 (body_T_cams[K], rig_calib[K,4,4])，按场景缓存；rig_calib 为各相机相对 cam0 的外参。"""
        if scene_id in self._calib_cache:
            return self._calib_cache[scene_id]
        rig_config = self._rig_configs[scene_id]
        body_T_cams = [self._get_body_T_cam(rig_config, c) for c in range(self.K)]
        T_ref_inv = np.linalg.inv(body_T_cams[0])
        rig_calib = np.stack(
            [T_ref_inv @ body_T_cams[i] for i in range(self.K)], 0).astype(np.float32)
        self._calib_cache[scene_id] = (body_T_cams, rig_calib)
        return self._calib_cache[scene_id]

    def _load_infoshare(self, scene_id, traj_id, ts_key):
        """读预存 rig 融合特征缓存 [K,256,768](uint16 bitcast bf16 → float32)。
        缓存格式: features/{ts}/rig.npy（每 rig 一个 rig.npy 在 ts 子目录）。
        文件缺失或通道维不符时立即报错。"""
        path = os.path.join(self.infoshare_root, "scenes", scene_id, "trajectories",
                            traj_id, "features", ts_key, "rig.npy")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺 NCLT info-sharing 缓存: {path}")
        u16 = np.load(path)                      # [K,256,C] uint16
        if u16.ndim != 3 or u16.shape[-1] != INFOSHARE_DIM:
            raise ValueError(f"infoshare 缓存形状 {u16.shape} 与通道数 {INFOSHARE_DIM} 不符: {path}")
        return torch.from_numpy(u16.astype(np.uint16)).view(torch.bfloat16).float()

    def _build_sample(self, seq, idxs):
        scene_id, traj_id = seq["scene"], seq["traj"]
        ts_all, poses = seq["ts"], seq["poses"]
        ts_list = [ts_all[i] for i in idxs]
        N, K = len(idxs), self.K  # N 取窗口实际长度（支持变长 N）

        body_T_cams, rig_calib = self._get_calib(scene_id)

        # 锚系绝对位姿 GT（=里程计 T_0->t）
        bT0 = body_T_cams[0]
        T_w_ref0_inv = np.linalg.inv(poses[ts_list[0]] @ bT0)
        T_abs = np.stack(
            [T_w_ref0_inv @ (poses[t] @ bT0) for t in ts_list], 0).astype(np.float32)

        # 特征: 预存 info_sharing [K,256,C]
        infoshare = torch.zeros(N, K, 256, INFOSHARE_DIM, dtype=torch.float32)
        for ti, ts in enumerate(ts_list):
            infoshare[ti] = self._load_infoshare(scene_id, traj_id, ts)  # [K,256,C]

        sample = {
            "rig_calib": torch.from_numpy(rig_calib.copy()),
            "T_rel_gt": torch.from_numpy(T_abs),
            "scene_id": scene_id,
            "traj_id": traj_id,
            "ts_list": ts_list,
            "infoshare": infoshare,
        }
        return sample

    # ------------------------------------------------------------------
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        # index = (N, seed) 元组（由 VariableNStreamBatchSampler 产生）
        n, seed = index
        return self._sample_train_window(int(n), np.random.default_rng(int(seed)))

    def _sample_train_window(self, n, rng):
        """随机 run + 随机逐对 stride 采一个长度 n 的训练窗口。"""
        ri = int(rng.choice(len(self._run_flat), p=self._run_probs))
        si, rs, re = self._run_flat[ri]
        seq = self.seqs[si]
        L = re - rs  # 已保证 L >= (num_rigs-1)*stride_min + 1，n <= num_rigs 必能放下

        for _ in range(20):
            # 相邻 rig 的帧间隔逐对独立、均匀采自 [stride_min, stride_max]
            strides = rng.integers(self.stride_min, self.stride_max + 1, size=n - 1)
            span = int(strides.sum())
            if span <= L - 1:
                break
        else:
            strides = np.full(n - 1, self.stride_min, dtype=np.int64)
            span = int(strides.sum())
        anchor = rs + int(rng.integers(0, L - span))
        idxs = [anchor]
        cur = anchor
        for s in strides:
            cur += int(s)
            idxs.append(cur)
        assert len(idxs) == n and idxs[-1] < re
        return self._build_sample(seq, idxs)


class VariableNStreamBatchSampler(torch.utils.data.Sampler):
    """变长 N batch sampler：每个 batch 内 N 一致，batch 间 N 从偏长分布采样。

    yield 长度 batch_size 的 [(N, seed), ...]，由 StreamOdomDataset.__getitem__ 消费。
    P(N) ∝ N^length_bias_power（默认 1.0，即 P(N) ∝ N）。
    DDP 约定：N 序列各 rank 相同（batch 形状对齐、负载均衡），窗口 seed 各 rank 不同。
    每 epoch 需调 set_epoch 以更新采样序列（种子混入 epoch）。
    """

    def __init__(self, num_batches, batch_size, n_min, n_max,
                 length_bias_power=1.0, seed=42, rank=0, world_size=1):
        self.num_batches = num_batches
        self.batch_size = batch_size
        self.N_choices = np.arange(n_min, n_max + 1)
        w = self.N_choices.astype(np.float64) ** length_bias_power
        self.N_probs = w / w.sum()
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def __iter__(self):
        rng_n = np.random.default_rng([self.seed, self._epoch, 7777])        # 各 rank 相同
        rng_w = np.random.default_rng([self.seed, self._epoch, self.rank])   # 各 rank 不同
        for _ in range(self.num_batches):
            n = int(rng_n.choice(self.N_choices, p=self.N_probs))
            seeds = [int(rng_w.integers(0, 2 ** 31 - 1))
                     for _ in range(self.batch_size)]
            yield [(n, seed) for seed in seeds]

    def __len__(self):
        return self.num_batches
