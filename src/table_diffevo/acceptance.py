"""接受规则模块。

实现 A0/A1/A2 接受规则，用于判断是否接受候选提案。
"""

from enum import Enum
from typing import Tuple
import numpy as np

from .metrics import compute_normalized_l1, compute_squared_loss


class AcceptanceRule(Enum):
    """接受规则枚举。"""
    A0 = "A0"  # 平方 loss Q 主判
    A1 = "A1"  # 归一化 L1 主判 + Q 平局判


def check_acceptance(
    rule: str,
    target: np.ndarray,
    current_q: np.ndarray,
    candidate_q: np.ndarray,
    n_records: int,
    eps_L1: float = 1e-5,
    eps_Q: float = 0.0
) -> Tuple[bool, float, float]:
    """检查是否接受候选提案。

    Args:
        rule: 接受规则，"A0" 或 "A1"
        target: 目标计数向量
        current_q: 当前表的查询答案
        candidate_q: 候选表的查询答案
        n_records: 记录总数
        eps_L1: L1 容差（用于 A1）
        eps_Q: Q 容差（用于 A0 和 A1）

    Returns:
        accept: 是否接受候选
        delta_L1: L1 变化量（candidate - current，负数表示改善）
        delta_Q: Q 变化量（candidate - current，负数表示改善）

    Raises:
        ValueError: 未知的接受规则
    """
    # 计算度量
    L1_current = compute_normalized_l1(target, current_q, n_records)
    L1_candidate = compute_normalized_l1(target, candidate_q, n_records)
    Q_current = compute_squared_loss(target, current_q)
    Q_candidate = compute_squared_loss(target, candidate_q)

    delta_L1 = L1_candidate - L1_current
    delta_Q = Q_candidate - Q_current

    # 接受判断
    if rule == "A0":
        # A0: Q 改善即接受
        accept = (Q_candidate <= Q_current + eps_Q)

    elif rule == "A1":
        # A1: L1 主判 + Q 平局判
        if L1_candidate < L1_current - eps_L1:
            # L1 明显改善
            accept = True
        elif abs(delta_L1) <= eps_L1:
            # L1 打平（在 eps_L1 容差内），用 Q 裁决
            accept = (Q_candidate <= Q_current + eps_Q)
        else:
            # L1 恶化
            accept = False

    elif rule == "A2":
        raise NotImplementedError("A2 规则暂未实现")

    else:
        raise ValueError(f"未知的接受规则: {rule}")

    return accept, delta_L1, delta_Q
