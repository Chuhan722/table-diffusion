"""测试会话级环境准备。

cuBLAS 在 torch.use_deterministic_algorithms(True) 下要求进程级
CUBLAS_WORKSPACE_CONFIG（须在 CUDA 上下文初始化前设置），否则相关
CUDA 测试直接 RuntimeError。conftest 在任何测试模块导入 torch 之前
执行，因此在这里 setdefault 是唯一可靠的进程内注入点；已有外部值
时不覆盖。
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "reference"))

FOUR_STATES = [(0, 0), (0, 1), (1, 0), (1, 1)]
FOUR_FEATURES = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
)
FOUR_TARGET = np.array([3.0, 1.0, 1.0])
FOUR_WEIGHTS = np.ones(3)
FOUR_IDS = np.array([0, 1, 2, 3])
FOUR_GAIN_MATRIX = np.array(
    [
        [0.0, -1.5, 0.5, -1.5],
        [0.5, 0.0, 1.0, 0.0],
        [-1.5, -3.0, 0.0, -2.0],
        [-1.5, -2.0, 0.0, 0.0],
    ]
)


@pytest.fixture
def four_record():
    from resevo.state import make_workload

    workload = make_workload(FOUR_FEATURES, FOUR_TARGET, FOUR_WEIGHTS)
    return workload, FOUR_IDS.copy()


def build_four_record_blocks():
    """四条记录例子的规范化块、增量与增益，多个新核测试共用。"""
    from resevo.candidates import full_single_row_supports, normalize_block
    from resevo.gain import block_deltas, block_gains
    from resevo.state import make_workload, table_residual

    workload = make_workload(FOUR_FEATURES, FOUR_TARGET, FOUR_WEIGHTS)
    ids = FOUR_IDS.copy()
    residual = table_residual(workload, ids)
    blocks = []
    for spec in full_single_row_supports(4, 4):
        nb = normalize_block(spec, ids, 4)
        d = block_deltas(workload.features, nb.outcomes[0], nb.outcomes)
        g = block_gains(d, residual, workload.weights)
        blocks.append((nb, d, g))
    return workload, ids, blocks
