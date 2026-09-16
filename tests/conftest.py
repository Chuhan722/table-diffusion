import sys
from pathlib import Path

import numpy as np
import pytest

# 让 tests 零安装即可 import resevo 与 tests/reference 下的对拍参考实现
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "reference"))

# 四条记录小表例子，状态顺序 00,01,10,11，查询 A=1、B=1、A且B
FOUR_STATES = [(0, 0), (0, 1), (1, 0), (1, 1)]
FOUR_FEATURES = np.array(
    [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
)
FOUR_TARGET = np.array([3.0, 1.0, 1.0])
FOUR_WEIGHTS = np.ones(3)
FOUR_IDS = np.array([0, 1, 2, 3])

# 文档增益矩阵锚点，行=源状态，列=目标状态
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
    """四条记录例子的规范化块、增量与增益，多个模块测试共用。"""
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
