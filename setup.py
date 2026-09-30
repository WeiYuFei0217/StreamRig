"""StreamRig 安装脚本。

功能：把 `streamrig` 包（Rig-Resampler / Causal Bridge / Stream Pose Head 及其数据集与损失）
装进当前 Python 环境。冻结的 MapAnything backbone 单独 vendored 在
`third_party/mapanything/`，请另外安装（见 README 安装步骤）。

示例用法::

    pip install -e .
    pip install -e ./third_party/mapanything
"""

from setuptools import setup, find_packages

setup(
    name="streamrig",
    version="1.0.0",
    description=(
        "StreamRig: Exploiting Intra-Rig Geometry for Streaming Multi-Camera Odometry"
    ),
    packages=find_packages(include=["streamrig", "streamrig.*"]),
    python_requires=">=3.10",
)
