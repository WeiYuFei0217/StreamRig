"""验证残缺特征缓存不能被标记为完整。

用法：python -m pytest tests/test_nclt_cache_completion.py -q
"""

import pytest

from data_preprocessing.common.infoshare_io import rig_feature_path
from data_preprocessing.nclt.step3_precompute_infoshare import _mark_complete_if_ready


def test_missing_rig_rejects_complete_marker(tmp_path):
    rigs = [("2012-02-19", "1000000000000", []),
            ("2012-02-19", "1000000000001", [])]
    existing = rig_feature_path(str(tmp_path), "nclt", rigs[0][0], rigs[0][1])
    from pathlib import Path
    Path(existing).parent.mkdir(parents=True)
    Path(existing).write_bytes(b"rig")

    with pytest.raises(RuntimeError, match="缺少 1 个 rig"):
        _mark_complete_if_ready(str(tmp_path), rigs, 0, 1, 5, 768)
    assert not (tmp_path / "COMPLETE").exists()

    missing = rig_feature_path(str(tmp_path), "nclt", rigs[1][0], rigs[1][1])
    Path(missing).parent.mkdir(parents=True, exist_ok=True)
    Path(missing).write_bytes(b"rig")
    assert _mark_complete_if_ready(str(tmp_path), rigs, 0, 1, 5, 768)
    assert (tmp_path / "COMPLETE").exists()
