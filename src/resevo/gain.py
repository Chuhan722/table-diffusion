"""查询增量与转移增益。

对块 B 与后继元组 u，查询增量 d_B(u)=a(u)-a(x_B)，块内贡献求和。
转移增益 G_B(u)=e^T W d_B(u)-(1/2)||d_B(u)||_W^2，
它精确等于把该块换成 u 后的损失下降 L(S)-L(S_{B->u})，
带二次项，只看线性方向项会误判。
运算顺序与参考实现一致，便于对拍逐位核对。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .state import Workload, loss_from_residual, table_loss


def block_deltas(
    features: NDArray[np.float64],
    source: tuple[int, ...],
    outcomes: tuple[tuple[int, ...], ...],
) -> NDArray[np.float64]:
    """全部后继的查询增量矩阵，形状 后继数×查询数，索引 0 为源元组增量全 0。"""
    num_queries = features.shape[1]
    deltas = np.zeros((len(outcomes), num_queries), dtype=np.float64)
    source_sum = features[list(source)].sum(axis=0)
    for j, outcome in enumerate(outcomes):
        if j == 0:
            continue  # 源元组增量恒为 0
        deltas[j] = features[list(outcome)].sum(axis=0) - source_sum
    return deltas


def block_gains(
    deltas: NDArray[np.float64],
    residual: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """转移增益 G=d@(We)-(1/2)sum(d*d*W)，源元组增益恒为 0。"""
    return deltas @ (weights * residual) - np.sum(deltas * deltas * weights, axis=1) / 2


def gain_by_replay(
    workload: Workload,
    state_ids: NDArray[np.int64],
    rows: tuple[int, ...],
    outcome: tuple[int, ...],
) -> float:
    """暴力重放版增益，复制整表替换后重算损失，与公式版互相独立作对拍标准。"""
    replaced = state_ids.copy()
    replaced[list(rows)] = outcome
    return table_loss(workload, state_ids) - table_loss(workload, replaced)


def loss_after_step(
    workload: Workload,
    residual: NDArray[np.float64],
    delta: NDArray[np.float64],
) -> float:
    """按增量更新后的损失 L(S')=(1/2)||e-d||_W^2，测试交叉验证用。"""
    return loss_from_residual(workload, residual - delta)
