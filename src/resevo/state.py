"""表状态与账本。

固定约定，计数单位。状态用整数编号指向贡献矩阵的行，
贡献矩阵 features 形状为 状态数 M 乘 查询数 Q，
整表答案 q(S)=sum_i a(x_i)，残差 e=y-q，损失 L=(1/2)e^T W e，W 为正对角权重。
本模块内所有运算顺序与参考实现保持一致，便于对拍逐位核对。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Workload:
    """查询负载，贡献矩阵、目标与权重的只读打包。"""

    features: NDArray[np.float64]  # M×Q，每个状态对每个查询的贡献
    target: NDArray[np.float64]  # Q
    weights: NDArray[np.float64]  # Q，严格正对角

    @property
    def num_states(self) -> int:
        return self.features.shape[0]

    @property
    def num_queries(self) -> int:
        return self.features.shape[1]


def make_workload(
    features: ArrayLike,
    target: ArrayLike,
    weights: ArrayLike,
    features_prevalidated: bool = False,
) -> Workload:
    """校验并构造查询负载，坏输入在这里一次性拒绝。

    features_prevalidated 为真时跳过贡献矩阵的全量有限性扫描，
    只允许在每行注册时已单独校验过的来源使用，形状校验照常执行。
    """
    a = np.asarray(features, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 1 or a.shape[1] < 1:
        raise ValueError("features 必须是非空的 状态×查询 矩阵")
    if y.shape != (a.shape[1],) or w.shape != (a.shape[1],):
        raise ValueError("target 与 weights 必须每个查询一个值")
    if not features_prevalidated and not np.isfinite(a).all():
        raise ValueError("所有数值输入必须有限")
    if not (np.isfinite(y).all() and np.isfinite(w).all()):
        raise ValueError("所有数值输入必须有限")
    if np.any(w <= 0):
        raise ValueError("weights 必须严格正")
    a.setflags(write=False)
    y.setflags(write=False)
    w.setflags(write=False)
    return Workload(a, y, w)


def check_state_ids(workload: Workload, state_ids: ArrayLike) -> NDArray[np.int64]:
    """校验行状态编号向量并返回 int64 副本。"""
    raw = np.asarray(state_ids)
    if raw.ndim != 1 or raw.size == 0 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("state_ids 必须是非空一维整数向量")
    s = raw.astype(np.int64, copy=True)
    if np.any(s < 0) or np.any(s >= workload.num_states):
        raise ValueError("状态编号必须落在贡献矩阵行数范围内")
    return s


def table_answers(workload: Workload, state_ids: NDArray[np.int64]) -> NDArray[np.float64]:
    """整表答案 q(S)=sum_i a(x_i)。"""
    return workload.features[state_ids].sum(axis=0)


def table_residual(workload: Workload, state_ids: NDArray[np.int64]) -> NDArray[np.float64]:
    """残差 e=y-q，一轮内所有块共用同一个残差。"""
    return workload.target - table_answers(workload, state_ids)


def loss_from_residual(workload: Workload, residual: NDArray[np.float64]) -> float:
    """损失 L=(1/2)e^T W e，运算顺序与参考实现一致。"""
    e = residual
    return float(np.dot(e * workload.weights, e) / 2)


def table_loss(workload: Workload, state_ids: NDArray[np.int64]) -> float:
    """整表损失，先算残差再算二次型。"""
    return loss_from_residual(workload, table_residual(workload, state_ids))


def features_from_predicates(states: list, predicates: list) -> NDArray[np.float64]:
    """由状态列表与查询谓词列表构造 0/1 贡献矩阵。

    最简占位，初版只支持每个谓词对单个状态返回 0 或 1 的计数查询，
    真实 schema 的字段依赖索引等复杂构造留待规模阶段。
    """
    a = np.zeros((len(states), len(predicates)), dtype=np.float64)
    for i, s in enumerate(states):
        for j, p in enumerate(predicates):
            v = float(p(s))
            if v not in (0.0, 1.0):
                raise ValueError("初版只支持 0/1 计数查询贡献")
            a[i, j] = v
    return a
