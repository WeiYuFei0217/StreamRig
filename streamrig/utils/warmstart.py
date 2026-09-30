"""
warmstart.py -- 从预训练的重定位模型权重热启动 StreamRig 的可训练模块。

初始化源为预训练的重定位模型（来源见 README 的 Training）：同名同形张量直接加载；
resampler 的 latent query 数不同时载入前缀；初始化源未覆盖的 bridge merged 层由
`zero_init_extra_merged_layers` 做残差恒等零初始化；backbone.* 参数跳过。

示例用法：
    from streamrig.utils.warmstart import warmstart_from_checkpoint, zero_init_extra_merged_layers
    report = warmstart_from_checkpoint(model, "/path/to/reloc_checkpoint.pth")
    zero_init_extra_merged_layers(model, report["num_merged_loaded"])
"""

import re
from typing import Dict

import torch


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # 可能本身就是 state_dict
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise ValueError("无法从 checkpoint 解析出 state_dict")


def warmstart_from_checkpoint(model, ckpt_path: str) -> Dict:
    """把预训练的非 backbone 权重加载进 StreamRigModel，返回加载报告。

    报告字段：
      direct_loaded           按名按形直接加载的张量数
      remapped                做了前缀截取的键（resampler.latents）
      skipped_backbone        跳过的 backbone.* 张量数
      skipped_shape_mismatch  两侧都有但形状不符、未加载的键
      source_only             初始化源有、模型没有的键
      model_not_covered       模型有、初始化源未覆盖的键（保持模型初始化）
      num_merged_loaded       从初始化源加载的 bridge merged 层数

    直接加载 0 个张量或未加载任何 merged 层时报错（通常是键名前缀不符或文件不是
    重定位模型），此时模型参数不被修改。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = _extract_state_dict(ckpt)

    tgt = model.state_dict()
    new_state = {}
    remapped = []
    skipped_backbone = 0
    skipped_shape = []
    source_only = []
    direct = 0

    for k, v in src.items():
        if k.startswith("backbone."):
            skipped_backbone += 1
            continue
        if k not in tgt:
            source_only.append(k)
            continue

        if k == "resampler.latents" and tgt[k].shape != v.shape:
            # latent query 数不同：载入前缀，多出的（若有）保持初始化
            tv = tgt[k].clone()
            n = min(v.shape[0], tv.shape[0])
            tv[:n] = v[:n]
            new_state[k] = tv
            remapped.append(k)
            continue

        # 其余：形状匹配则直接加载；不匹配则跳过并记录
        if tgt[k].shape == v.shape:
            new_state[k] = v
            direct += 1
        else:
            skipped_shape.append(k)

    # 统计从 ckpt 加载的 merged 层数（其余层由 zero_init_extra_merged_layers 初始化）
    merged_ids = set()
    for k in new_state:
        m = re.match(r"bridge\.merged_layers\.(\d+)\.", k)
        if m:
            merged_ids.add(int(m.group(1)))
    num_merged_loaded = (max(merged_ids) + 1) if merged_ids else 0

    report = {
        "direct_loaded": direct,
        "remapped": remapped,
        "skipped_backbone": skipped_backbone,
        "skipped_shape_mismatch": skipped_shape,
        "source_only": source_only,
        "model_not_covered": [k for k in tgt if k not in new_state],
        "num_merged_loaded": num_merged_loaded,
    }
    if direct == 0 or num_merged_loaded == 0:
        raise ValueError(
            f"warm-start 失败：{ckpt_path} 与模型不匹配（direct_loaded={direct}, "
            f"num_merged_loaded={num_merged_loaded}）；源有模型无的键示例: {source_only[:5]}")
    if merged_ids != set(range(num_merged_loaded)):
        raise ValueError(f"warm-start 失败：加载到的 merged 层不连续: {sorted(merged_ids)}")

    model.load_state_dict({**tgt, **new_state}, strict=True)
    return report


def zero_init_extra_merged_layers(model, num_loaded: int) -> int:
    """把 bridge 中索引 >= num_loaded 的 merged 层做残差恒等零初始化。

    初始化源只有 num_loaded 层 merged 自注意力时，把每个新层的两条残差分支输出投影置零：
      - self_attn.out_proj.weight / bias  （x = x + attn_out 分支）
      - ffn[3].weight / bias               （x = x + ffn(...) 分支，ffn 的第二个 Linear）
    置零后新层 = 恒等映射，热启动首步前向 == 纯 num_loaded 层前向，
    新层在训练中从恒等慢慢长出（residual identity 初始化）。

    Args:
        model: StreamRigModel（未 DDP 包裹时调用）。
        num_loaded: 已从 ckpt 加载的 merged 层数（取 warmstart 报告的 num_merged_loaded）。
    Returns:
        被零初始化的层数。
    """
    merged = model.bridge.merged_layers
    n_zeroed = 0
    with torch.no_grad():
        for i in range(num_loaded, len(merged)):
            blk = merged[i]
            # 注意力输出投影（nn.MultiheadAttention.out_proj）
            blk.self_attn.out_proj.weight.zero_()
            if blk.self_attn.out_proj.bias is not None:
                blk.self_attn.out_proj.bias.zero_()
            # FFN 输出投影：Sequential(Linear, GELU, Dropout, Linear, Dropout) 的第二个 Linear = 索引 3
            ffn_out = blk.ffn[3]
            ffn_out.weight.zero_()
            if ffn_out.bias is not None:
                ffn_out.bias.zero_()
            n_zeroed += 1
    return n_zeroed
