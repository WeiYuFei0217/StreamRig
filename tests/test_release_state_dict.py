"""按两份发布配置构建模型，state_dict 的键名、顺序与形状须与发布权重的快照一致（CPU）。

快照 tests/release_state_dict_shapes.json 记录发布权重 NCLT-StreamRig.pth /
KITTI360-StreamRig.pth 的全部键（各 167 个）及形状；模型结构改动若使发布权重无法严格加载，
本测试失败。

用法：
    python -m pytest tests/test_release_state_dict.py -q
"""

import json
from pathlib import Path

import pytest
import yaml

from streamrig.config import build_stream_model

PROJECT = Path(__file__).resolve().parents[1]
SNAPSHOT = json.loads((Path(__file__).with_name("release_state_dict_shapes.json")).read_text())
CONFIGS = {
    "nclt": PROJECT / "configs" / "nclt_streamrig.yaml",
    "kitti360": PROJECT / "configs" / "kitti360_streamrig.yaml",
}


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_release_config_state_dict_matches_snapshot(name):
    cfg = yaml.safe_load(CONFIGS[name].read_text(encoding="utf-8"))
    model = build_stream_model(cfg["model"])
    got = [(key, list(value.shape)) for key, value in model.state_dict().items()]
    expected = list(SNAPSHOT[name].items())
    assert len(expected) == 167
    missing = sorted(set(SNAPSHOT[name]) - {key for key, _ in got})
    unexpected = sorted({key for key, _ in got} - set(SNAPSHOT[name]))
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    assert got == expected
