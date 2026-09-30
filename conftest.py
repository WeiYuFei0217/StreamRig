"""pytest 全局配置：把仓库根加进 sys.path，使测试无需 pip install 即可 import streamrig。

同时把 scripts/ 加进 sys.path，供直接 import 训练/评测脚本的回归测试使用。

示例用法::

    pytest -q                 # 运行全部 CPU 测试
    pytest tests/test_cache_equiv.py -q
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
for _p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
