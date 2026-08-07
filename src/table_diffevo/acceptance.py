"""
接受规则模块

实现 Issue #33 预注册的接受规则：
- A0：平方 loss 主判
- A1：归一化 L1 主判 + Q 平局判

所有规则在候选覆盖前计算差值，确保四象限统计正确。
"""
import numpy as np
from typing import Tuple
from .metrics import compute_normalized_l1, compute_squared_loss


def check_acceptance(
    rule: str,
    target: np.ndarray,
    current: np.ndarray,
    candidate: np.ndarray,
    n_records: int,
    eps_L1: float = 0.0,
    eps_Q: float = 0.0,
) -> Tuple[bool, float, float]:
    """
    检查候选是否接受（统一接口）。

    Parameters
    ----------
    rule : str
        接受规则，'A0' 或 'A1'
    target : np.ndarray
        目标计数向量
    current : np.ndarray
        当前状态的查询答案
    candidate : np.ndarray
        候选状态的查询答案
    n_records : int
        记录总数 N
    eps_L1 : float, default=0.0
        归一化 L1 改善阈值（A1 使用）
    eps_Q : float, default=0.0
        平方 loss 改善阈值

    Returns
    -------
    accept : bool
        是否接受候选
    delta_L1 : float
        L1_candidate - L1_current（负值表示改善）
    delta_Q : float
        Q_candidate - Q_current（负值表示改善）

    Raises
    ------
    ValueError
        如果 rule 不是 'A0' 或 'A1'
    """
    # 计算差值（在覆盖前）
    L1_current = compute_normalized_l1(target, current, n_records)
    L1_candidate = compute_normalized_l1(target, candidate, n_records)
    Q_current = compute_squared_loss(target, current)
    Q_candidate = compute_squared_loss(target, candidate)

    delta_L1 = L1_candidate - L1_current
    delta_Q = Q_candidate - Q_current

    # 根据规则判断
    if rule == "A0":
        accept = _check_A0(delta_Q, eps_Q)
    elif rule == "A1":
        accept = _check_A1(delta_L1, delta_Q, eps_L1, eps_Q)
    else:
        raise ValueError(f"未知接受规则: {rule}，仅支持 'A0' 或 'A1'")

    return accept, delta_L1, delta_Q


def _check_A0(delta_Q: float, eps_Q: float) -> bool:
    """
    A0 规则：平方 loss 主判。

    接受条件：Q_candidate < Q_current - eps_Q

    Parameters
    ----------
    delta_Q : float
        Q_candidate - Q_current
    eps_Q : float
        改善阈值

    Returns
    -------
    bool
        是否接受
    """
    return delta_Q < -eps_Q


def _check_A1(
    delta_L1: float,
    delta_Q: float,
    eps_L1: float,
    eps_Q: float
) -> bool:
    """
    A1 规则：归一化 L1 主判 + Q 平局判。

    接受条件：
    1. 若 L1_candidate < L1_current - eps_L1 → 接受
    2. 若 |L1_candidate - L1_current| <= eps_L1 → 用 Q 裁决
       - Q_candidate < Q_current - eps_Q → 接受
       - 否则 → 拒绝
    3. 若 L1_candidate > L1_current + eps_L1 → 拒绝

    Parameters
    ----------
    delta_L1 : float
        L1_candidate - L1_current
    delta_Q : float
        Q_candidate - Q_current
    eps_L1 : float
        L1 改善阈值
    eps_Q : float
        Q 改善阈值

    Returns
    -------
    bool
        是否接受
    """
    if delta_L1 < -eps_L1:
        # L1 明显改善 → 接受
        return True
    elif abs(delta_L1) <= eps_L1:
        # L1 打平 → 用 Q 裁决
        return delta_Q < -eps_Q
    else:
        # L1 恶化 → 拒绝
        return False
