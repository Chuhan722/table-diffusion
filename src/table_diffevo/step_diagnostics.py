"""查询空间中整代更新步幅的逐行精确分解。

本模块只提供生成后诊断，不参与扩散采样、接受、早停或 checkpoint 选择。给定
当前合成表与一个 proposal，它先计算每条记录对全部查询计数的变化，再把平方
workload loss 的二次项精确拆成逐行自身项和跨行交叉项。
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from table_diffevo.queries import eval_query_mask


def compute_row_query_deltas(
    current: pd.DataFrame,
    proposal: pd.DataFrame,
    queries: List[Dict[str, Any]],
) -> np.ndarray:
    """返回每条记录对每个查询计数的变化矩阵。

    第 ``i, j`` 个元素为查询 `j` 在 proposal 第 `i` 行上的指示值减去
    current 第 `i` 行上的指示值，因此只可能为 `-1`、`0` 或 `1`。
    """
    if not isinstance(current, pd.DataFrame) or not isinstance(
        proposal, pd.DataFrame
    ):
        raise ValueError("current 和 proposal 必须是 pandas DataFrame")
    if len(current) != len(proposal):
        raise ValueError(
            f"current/proposal 行数不一致：{len(current)} vs {len(proposal)}"
        )
    if list(current.columns) != list(proposal.columns):
        raise ValueError("current/proposal 列名或列顺序不一致")
    if not isinstance(queries, list):
        raise ValueError("queries 必须是列表")

    before = current.reset_index(drop=True)
    after = proposal.reset_index(drop=True)
    deltas = np.empty((len(before), len(queries)), dtype=np.int8)
    for query_index, query in enumerate(queries):
        before_mask = np.asarray(
            eval_query_mask(before, query), dtype=np.int8
        )
        after_mask = np.asarray(
            eval_query_mask(after, query), dtype=np.int8
        )
        deltas[:, query_index] = after_mask - before_mask
    return deltas


def decompose_query_step(
    row_query_deltas: np.ndarray,
    count_residual: np.ndarray,
) -> Dict[str, Any]:
    """精确分解平方 workload loss 的一阶收益与二次步幅。

    `row_query_deltas` 的每一行是单条记录的查询计数变化 `δq_i`，
    `count_residual` 是旧状态的计数残差 `target - q`。返回满足

    `net_gain = linear_gain - self_penalty - cross_penalty`

    且 `self_penalty + cross_penalty = 0.5 * ||sum_i δq_i||²`。
    """
    raw_deltas = np.asarray(row_query_deltas)
    if raw_deltas.ndim != 2 or raw_deltas.dtype.kind not in "iuf":
        raise ValueError("row_query_deltas 必须是有限数值二维数组")
    deltas = raw_deltas.astype(float, copy=False)
    if not np.all(np.isfinite(deltas)):
        raise ValueError("row_query_deltas 必须是有限数值二维数组")

    raw_residual = np.asarray(count_residual)
    if (
        raw_residual.shape != (deltas.shape[1],)
        or raw_residual.dtype.kind not in "iuf"
    ):
        raise ValueError(
            "count_residual 必须是长度与查询数一致的有限数值一维数组"
        )
    residual = raw_residual.astype(float, copy=False)
    if not np.all(np.isfinite(residual)):
        raise ValueError(
            "count_residual 必须是长度与查询数一致的有限数值一维数组"
        )

    delta_q = deltas.sum(axis=0, dtype=float)
    linear_gain = float(np.dot(residual, delta_q))
    self_penalty = float(0.5 * np.einsum("ij,ij->", deltas, deltas))
    quadratic_penalty = float(0.5 * np.dot(delta_q, delta_q))
    cross_penalty = float(quadratic_penalty - self_penalty)
    net_gain = float(linear_gain - quadratic_penalty)
    return {
        "delta_q": delta_q,
        "linear_gain": linear_gain,
        "self_penalty": self_penalty,
        "cross_penalty": cross_penalty,
        "quadratic_penalty": quadratic_penalty,
        "net_gain": net_gain,
    }
