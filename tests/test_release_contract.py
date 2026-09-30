"""发布合约回归：batch 误配须失败，导出不能带训练状态。

用法：pytest tests/test_release_contract.py -q。
"""
import pytest
import torch
from export_weights import extract_weights
from train_stream import resolve_batch_size


def test_batch_contract_rejects_changed_world_and_accumulation():
    cfg = dict(batch_size=2, global_batch_size=16, effective_batch_size=16, accum_steps=1)
    assert resolve_batch_size(cfg, 8) * 8 == 16
    with pytest.raises(ValueError):
        resolve_batch_size(cfg, 4)
    with pytest.raises(ValueError):
        resolve_batch_size(dict(cfg, accum_steps=2), 8)
    cfg.update(batch_size=4)
    assert resolve_batch_size(cfg, 4) * 4 == 16


def test_export_roundtrip_has_no_training_metadata(tmp_path):
    tensor = torch.arange(6).reshape(2, 3)
    original = dict(model_state_dict={'pose.weight': tensor, 'backbone.weight': tensor},
                    optimizer_state_dict={'state': {}}, epoch=9, step=40000,
                    config={'train': {'batch_size': 6}})
    path = tmp_path / 'weights.pth'
    torch.save(extract_weights(original), path)
    result = torch.load(path, weights_only=True)
    assert set(result) == {'pose.weight'}
    assert torch.equal(result['pose.weight'], tensor)
    with pytest.raises(ValueError):
        extract_weights({'pose.weight': tensor, 'epoch': 9})
