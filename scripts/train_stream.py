"""
train_stream.py -- StreamRig 因果流式训练（DDP，从初始化权重热启动）。

关键设计：
  - 一次前向喂一个 N 个 rig 的窗口，组级因果 / 锚快照 mask 使每组只看历史，
    同时监督 T_0->1..T_0->(N-1) 全部相对位姿（N 由变长窗口 sampler 逐 batch 抽取）。
  - 冻结 MapAnything 前端的输出预先算成 info-sharing 缓存，训练只读缓存，只训 resampler/bridge/pose_head。
  - warm-start 自 config 的 model.warmstart_ckpt；初始化源未覆盖的 bridge merged 层
    做残差恒等零初始化。

用法（torchrun 4 卡，与发布配置的 batch 设置一致；推荐用 scripts/train.sh 包装）：
    torchrun --nproc_per_node=4 --master-port=29601 scripts/train_stream.py \
        --config configs/nclt_streamrig.yaml --output-dir outputs/nclt_streamrig
"""

import argparse
import math
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys
# 仓库根 = 本脚本所在 scripts/ 的上一级；加入 sys.path，使得直接 `python scripts/xxx.py`
# 运行时能 import 到同仓库的 streamrig 包（无需先 pip install）。
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from streamrig.config import build_stream_model  # noqa: E402
from streamrig.losses import StreamPoseLoss  # noqa: E402
from streamrig.datasets import (  # noqa: E402
    Kitti360StreamDataset,
    StreamOdomDataset,
    VariableNStreamBatchSampler,
    stream_collate_fn,
)
from streamrig.utils import warmstart_from_checkpoint, zero_init_extra_merged_layers  # noqa: E402


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a, **k):
    if is_main():
        print(*a, **k, flush=True)


def setup_ddp(timeout_min=60):
    if "RANK" in os.environ:
        dist.init_process_group("nccl", timeout=timedelta(minutes=timeout_min))
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_world_size()
    return 0, 1


def resolve_batch_size(tcfg, world):
    """校验 batch 合约并返回每卡 batch：batch_size × world_size = global_batch_size，
    global_batch_size × accum_steps = effective_batch_size。"""
    accum = int(tcfg.get("accum_steps", 1))
    local_bs = int(tcfg["batch_size"])
    if local_bs < 1 or world < 1 or accum < 1:
        raise ValueError("train.batch_size、world_size、train.accum_steps 须为正整数")
    global_bs = local_bs * world
    expected_global = int(tcfg.get("global_batch_size", global_bs))
    if global_bs != expected_global:
        raise ValueError(
            f"train.batch_size({local_bs}) × world_size({world}) = {global_bs}，"
            f"不等于 train.global_batch_size({expected_global})；请按 GPU 数调整 train.batch_size")
    expected_eff = int(tcfg.get("effective_batch_size", global_bs * accum))
    if global_bs * accum != expected_eff:
        raise ValueError(
            f"global batch({global_bs}) × train.accum_steps({accum}) = {global_bs * accum}，"
            f"不等于 train.effective_batch_size({expected_eff})；请调整 train.accum_steps")
    return local_bs


def build_dataset(dcfg):
    """构建训练集（只用训练序列）。"""
    if dcfg.get("dataset_type") == "kitti360":
        return Kitti360StreamDataset(
            processed_root=dcfg["processed_root"],
            infoshare_root=dcfg["infoshare_root"],
            split=dcfg.get("train_split", "train"),
            num_rigs=dcfg["num_rigs"],
            stride_min=dcfg.get("stride_min", 1),
            stride_max=dcfg.get("stride_max", 6),
            gap_threshold_ms=dcfg.get("gap_threshold_ms", 500),
            min_run_span_ms=dcfg.get("min_run_span_ms", 30000),
            samples_per_epoch=dcfg.get("samples_per_epoch", 48000),
            image_size=dcfg.get("image_size", 224),
        )
    return StreamOdomDataset(
        step1_root=dcfg["step1_root"], step2_root=dcfg["step2_root"],
        infoshare_root=dcfg["infoshare_root"],
        num_rigs=dcfg["num_rigs"], num_cameras=dcfg["num_cameras"],
        stride_min=dcfg.get("stride_min", 1), stride_max=dcfg.get("stride_max", 6),
        gap_threshold_ms=dcfg.get("gap_threshold_ms", 400), min_run_span_ms=dcfg.get("min_run_span_ms", 30000),
        samples_per_epoch=dcfg.get("samples_per_epoch", 100000),
    )


def save_ckpt(path, raw_model, optimizer, scheduler, step, epoch, cfg):
    torch.save({
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step, "epoch": epoch, "config": cfg,
    }, path)


def main():
    ap = argparse.ArgumentParser(description="StreamRig 因果流式训练（DDP，建议经 scripts/train.sh 启动）")
    ap.add_argument("--config", required=True, help="训练配置 yaml（见 configs/）")
    ap.add_argument("--output-dir", required=True, help="输出目录（train.log / checkpoint）")
    ap.add_argument("--resume", default=None,
                    help="从训练 checkpoint 续训（恢复 model/optimizer/scheduler/epoch/step）")
    ap.add_argument("--resume-weights-only", action="store_true",
                    help="只加载权重，step/lr/optimizer/epoch 从 0 开始")
    ap.add_argument("--nccl-timeout", type=int, default=60, help="DDP 进程组超时（分钟）")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]

    local_rank, world = setup_ddp(args.nccl_timeout)
    device = f"cuda:{local_rank}"
    # fp32 矩阵乘使用 TF32（发布模型的训练与评测均在此设置下进行）
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.output_dir, exist_ok=True)
    log(f"[train] world_size={world}, output={args.output_dir}")
    log(f"[train] config={args.config}")

    # 先校验 batch 合约，避免读取大数据后才发现 GPU 数配置错误。
    bs = resolve_batch_size(tcfg, world)
    global_bs = bs * world
    rank = dist.get_rank() if world > 1 else 0

    # ---- 数据 ----
    train_ds = build_dataset(dcfg)
    nw = tcfg.get("num_workers", 8)

    # 变长 N 窗口：每个 batch 抽一个窗口长度 N∈[n_min, n_max]，batch sampler 直接负责 DDP 分工
    n_min, n_max = dcfg.get("n_min", 2), dcfg.get("n_max", dcfg["num_rigs"])
    assert n_max <= dcfg["num_rigs"], "n_max 不能超过 num_rigs（run 可用性按 num_rigs 过滤）"
    num_batches = dcfg.get("samples_per_epoch", 100000) // global_bs
    train_batch_sampler = VariableNStreamBatchSampler(
        num_batches=num_batches, batch_size=bs, n_min=n_min, n_max=n_max,
        length_bias_power=dcfg.get("length_bias_power", 1.0),
        seed=dcfg.get("seed", 42), rank=rank, world_size=world,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_batch_sampler,
                              collate_fn=stream_collate_fn, num_workers=nw,
                              pin_memory=True, persistent_workers=nw > 0)
    log(f"[data] 变长 N∈[{n_min},{n_max}] 偏长采样, {num_batches} batch/epoch/rank")

    # ---- 模型 + warm-start ----
    model = build_stream_model(mcfg, device)

    if args.resume is None and mcfg.get("warmstart_ckpt"):
        report = warmstart_from_checkpoint(model, mcfg["warmstart_ckpt"])
        # 键列表较长时只打印数量
        log("[warmstart] " + str({k: (f"<{len(v)} keys>" if isinstance(v, list) and len(v) > 4 else v)
                                   for k, v in report.items()}))
        # bridge_merged_layers 多于初始化源加载的层数时，把多出的层做残差恒等零初始化
        # （新层初始 = 恒等映射，保持热启动起点）。放在 warmstart 之后，避免被覆盖。
        n_loaded = report["num_merged_loaded"]
        n_zeroed = zero_init_extra_merged_layers(model, n_loaded)
        if n_zeroed:
            log(f"[warmstart] 已恒等零初始化 {n_zeroed} 个新增 merged 层"
                f"（索引 {n_loaded}..{n_loaded + n_zeroed - 1}）")
        else:
            log("[warmstart] 初始化源已覆盖全部 merged 层，无需零初始化")

    raw_model = model
    if world > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    trainable = list(raw_model.parameters())
    criterion = StreamPoseLoss(
        rotation_weight=tcfg.get("rotation_weight", 5.0),
        translation_weight=tcfg.get("translation_weight", 1.0),
        intra_weight=tcfg.get("intra_weight", 0.5),
        inter_weight=tcfg.get("inter_weight", 1.0),
        loss_clamp=tcfg.get("loss_clamp", 10.0),
    )
    log(f"[loss] loss_clamp={criterion.loss_clamp}")
    optimizer = AdamW(trainable, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.01))

    # 梯度累积：gstep/scheduler/save_every 一律按 optimizer step 计
    #（accum_steps=1 时 1 iter = 1 optimizer step）
    accum = tcfg.get("accum_steps", 1)
    epochs = tcfg["epochs"]
    iters_per_epoch = len(train_loader)
    steps_per_epoch = iters_per_epoch // accum
    total_steps = epochs * steps_per_epoch
    warmup = tcfg.get("warmup_steps", 500)

    def lr_lambda(s):
        if s < warmup:
            return s / max(1, warmup)
        prog = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * prog)) * (1 - tcfg.get("min_lr_ratio", 0.05)) + tcfg.get("min_lr_ratio", 0.05)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    grad_clip = tcfg.get("grad_clip", 1.0)

    start_epoch, gstep = 0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        raw_model.load_state_dict(ck.get("model_state_dict", ck), strict=True)   # 权重总是加载
        if args.resume_weights_only:
            log(f"[resume-weights-only] 只加载权重 from {args.resume}; "
                f"step/lr/optimizer/epoch 从 0 开始")
        else:
            if "optimizer_state_dict" not in ck:
                raise ValueError("Released files contain weights only; use --resume-weights-only")
            optimizer.load_state_dict(ck["optimizer_state_dict"])
            scheduler.load_state_dict(ck["scheduler_state_dict"])
            start_epoch, gstep = ck["epoch"], ck["step"]
            # 恰在 epoch 边界时从下一个 epoch 开始，否则从 checkpoint 所在 epoch 的开头重跑
            if gstep > 0 and gstep % steps_per_epoch == 0 and start_epoch * steps_per_epoch < gstep:
                start_epoch += 1
            log(f"[resume] from {args.resume} @ epoch{start_epoch} step{gstep}")

    log(f"[train] iters/epoch={iters_per_epoch}, accum={accum}, "
        f"optim steps/epoch={steps_per_epoch}, total_optim_steps={total_steps}, "
        f"eff_batch={global_bs * accum}, "
        f"trainable={sum(p.numel() for p in trainable):,}")

    log_every = tcfg.get("log_every", 50)
    save_every = tcfg.get("save_every_steps", 5000)
    model.train()

    for epoch in range(start_epoch, epochs):
        train_batch_sampler.set_epoch(epoch)
        t0 = time.time()
        run_loss, run_iters = 0.0, 0
        optimizer.zero_grad(set_to_none=True)  # 丢弃上个 epoch 可能残留的未 step 梯度
        for it, batch in enumerate(train_loader):
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            is_boundary = ((it + 1) % accum == 0)
            out = model(batch)
            losses = criterion(out, batch)
            loss = losses["loss"] / accum
            # 非边界 iter 跳过 DDP 梯度同步（累积期只累本地梯度）
            if world > 1 and accum > 1 and not is_boundary:
                with model.no_sync():
                    loss.backward()
            else:
                loss.backward()
            run_loss += losses["loss"].item()
            run_iters += 1
            if not is_boundary:
                continue

            clip_grad_norm_(trainable, grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            gstep += 1

            if gstep % log_every == 0:
                dt = time.time() - t0
                sps = (it + 1) / dt
                log(f"e{epoch} s{gstep} loss={run_loss/max(run_iters,1):.4f} "
                    f"rot={losses['rot_loss_inter'].item():.4f} trans={losses['trans_loss_inter'].item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} {sps:.2f}it/s")
                run_loss, run_iters = 0.0, 0

            if gstep % save_every == 0 and is_main():
                save_ckpt(os.path.join(args.output_dir, "last.pt"), raw_model, optimizer, scheduler, gstep, epoch, cfg)

        if is_main():
            save_ckpt(os.path.join(args.output_dir, f"epoch_{epoch:03d}.pt"), raw_model, optimizer, scheduler, gstep, epoch, cfg)
        log(f"[train] epoch {epoch} done in {(time.time()-t0)/60:.1f}min")

    if is_main():
        save_ckpt(os.path.join(args.output_dir, "final.pt"), raw_model, optimizer, scheduler, gstep, epochs, cfg)
    log("[train] 完成")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
