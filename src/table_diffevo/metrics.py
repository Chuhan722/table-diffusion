"""
实验度量计算模块

为接受规则对照实验和 α 调度实验提供统一的度量计算接口。
所有计算与 evolution.py 和 objective.py 保持完全一致。
"""
import numpy as np
from typing import Tuple
from .objective import compute_residual, compute_loss


def compute_normalized_l1(
    target: np.ndarray,
    current: np.ndarray,
    n_records: int
) -> float:
    """
    计算归一化 L1 误差（与 evolution.py:904 完全一致）。

    公式：normalized_l1 = mean(|target - current|) / n_records

    Parameters
    ----------
    target : np.ndarray, shape (m,)
        目标计数向量
    current : np.ndarray, shape (m,)
        当前合成表的查询答案
    n_records : int
        记录总数 N

    Returns
    -------
    float
        归一化 L1 误差，范围 [0, 1]

    Examples
    --------
    >>> target = np.array([180, 95, 42])
    >>> current = np.array([170, 100, 42])
    >>> compute_normalized_l1(target, current, 300)
    0.016666666666666666  # (10 + 5 + 0) / 3 / 300
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)

    if target.shape != current.shape:
        raise ValueError(
            f"target 与 current 形状不一致: {target.shape} vs {current.shape}"
        )

    if n_records <= 0:
        raise ValueError(f"n_records 必须为正数，收到: {n_records}")

    abs_errors = np.abs(target - current)
    return float(np.mean(abs_errors) / n_records)


def compute_squared_loss(
    target: np.ndarray,
    current: np.ndarray
) -> float:
    """
    计算平方 loss Q（wrapper for objective.compute_loss）。

    在无噪声、无权重情况下：Q = ½ Σ(target - current)²

    Parameters
    ----------
    target : np.ndarray, shape (m,)
        目标计数向量
    current : np.ndarray, shape (m,)
        当前合成表的查询答案

    Returns
    -------
    float
        平方 loss Q ≥ 0
    """
    return compute_loss(target, current, sigma=None, kappa=1.0, weights=None)


def compute_all_metrics(
    target: np.ndarray,
    current: np.ndarray,
    n_records: int
) -> Tuple[float, float, np.ndarray]:
    """
    一次性计算所有度量（normalized_l1, squared_loss, residual）。

    避免重复计算 target - current。

    Parameters
    ----------
    target : np.ndarray, shape (m,)
        目标计数向量
    current : np.ndarray, shape (m,)
        当前合成表的查询答案
    n_records : int
        记录总数 N

    Returns
    -------
    normalized_l1 : float
        归一化 L1 误差
    squared_loss : float
        平方 loss Q
    residual : np.ndarray, shape (m,)
        比例残差向量 ε
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)

    if target.shape != current.shape:
        raise ValueError(
            f"target 与 current 形状不一致: {target.shape} vs {current.shape}"
        )

    if n_records <= 0:
        raise ValueError(f"n_records 必须为正数，收到: {n_records}")

    # 计算原始差值（只算一次）
    raw_diff = target - current

    # normalized L1
    normalized_l1 = float(np.mean(np.abs(raw_diff)) / n_records)

    # squared loss Q
    squared_loss = float(0.5 * np.sum(raw_diff ** 2))

    # residual（比例残差）
    residual = raw_diff / n_records

    return normalized_l1, squared_loss, residual
