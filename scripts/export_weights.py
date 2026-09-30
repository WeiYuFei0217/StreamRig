"""导出仅含非 backbone 模型张量的发布文件。

示例：python scripts/export_weights.py outputs/run/final.pt release_weights/model.pth
训练断点保留在 outputs；导出文件不包含 config、epoch、step 或优化器状态。
"""
import argparse
from pathlib import Path
import torch


def extract_weights(checkpoint):
    """只接受已知模型容器或纯张量字典，拒绝混入训练元数据。"""
    state = checkpoint
    for key in ("model_state_dict", "state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    if not isinstance(state, dict) or not state:
        raise ValueError("Expected a non-empty model state dictionary")
    if any(not isinstance(k, str) or not isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError("Model state must contain only named tensors")
    result = {k: v.detach().cpu().clone() for k, v in state.items() if not k.startswith("backbone.")}
    if not result:
        raise ValueError("No non-backbone model tensors found")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.destination.resolve() or args.destination.exists():
        parser.error("Use a new destination; existing files are never overwritten")
    state = extract_weights(torch.load(args.source, map_location="cpu", weights_only=True))
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.destination)
    print(f"Exported {len(state)} model tensors to {args.destination}")


if __name__ == "__main__":
    main()
