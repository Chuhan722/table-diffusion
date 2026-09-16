"""熵校准，把增益变成方向分布。

在参考分布 R 之上解 min sum KL(P||R) s.t. sum P*G >= delta 的指数倾斜解，
P_B(u;beta)=R_B(u)exp(beta*G_B(u))/Z_B(beta)，所有块共享同一个 beta。
平均增益 D(beta) 关于 beta 单调不减，其导数为各块增益方差之和。
M=sum_B max_u G_B(u) 为最大可达增益总和，delta=alpha*M。
若 D(0)>=delta 取 beta=0，否则先翻倍括根再二分求唯一根。
整个数值步骤只反复评价概率与平均增益，不抽样任何表。
括根与二分的次数、端点选取和 log-sum-exp 顺序与参考实现完全一致，便于对拍逐位核对。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp


@dataclass
class TiltResult:
    """熵校准结果。frozen 为真表示本次支持没有正的一阶方向，整体保持不动。"""

    beta: float
    probabilities: list[NDArray[np.float64]]  # 每块的方向分布 P，与块 outcomes 对齐
    direction_gain: float  # D(beta)=sum P*G
    max_gain_sum: float  # M
    required_gain: float  # delta=alpha*M
    frozen: bool


def tilted_distributions(
    gains: list[NDArray[np.float64]],
    references: list[NDArray[np.float64]],
    beta: float,
) -> tuple[list[NDArray[np.float64]], float]:
    """给定 beta 的各块指数倾斜分布与平均增益，log-sum-exp 归一化。"""
    ps = []
    total = 0.0
    for g, R in zip(gains, references):
        # 先减去块内最大增益再乘 beta，避免正向上溢，运算顺序与参考一致
        logits = np.log(R) + beta * (g - np.max(g))
        P = np.exp(logits - logsumexp(logits))
        ps.append(P)
        total += float(P @ g)
    return ps, total


def calibrate_beta(
    gains: list[NDArray[np.float64]],
    references: list[NDArray[np.float64]],
    old_loss: float,
    alpha: float = 0.5,
    numerical_tol: float = 1e-12,
) -> TiltResult:
    """求共享 beta 与方向分布。

    old_loss 只用来给冻结判据定尺度，判据与参考实现一致，
    M 不超过 numerical_tol*max(1,old_loss) 时视为没有正方向，返回冻结结果。
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须落在 (0,1)")
    if not np.isfinite(numerical_tol) or numerical_tol < 0:
        raise ValueError("numerical_tol 必须有限且非负")
    max_gain_sum = float(sum(max(float(np.max(g)), 0.0) for g in gains))
    requirement = alpha * max_gain_sum
    scale = max(1.0, old_loss)

    if max_gain_sum <= numerical_tol * scale:
        # 没有正的一阶方向，本轮整体保持不动，明确报告停滞而不是静默收敛
        frozen_ps = []
        for R in references:
            P = np.zeros_like(R)
            P[0] = 1.0
            frozen_ps.append(P)
        return TiltResult(0.0, frozen_ps, 0.0, max_gain_sum, requirement, True)

    ps, D = tilted_distributions(gains, references, 0.0)
    if D >= requirement:
        beta = 0.0
    else:
        lo = 0.0
        hi = 1.0 / max(max_gain_sum, 1e-300)
        for _ in range(100):
            ps, D = tilted_distributions(gains, references, hi)
            if D >= requirement:
                break
            hi *= 2.0
        else:
            raise FloatingPointError("无法括住 beta 根，检查数值条件")
        for _ in range(80):
            mid = (lo + hi) / 2.0
            _, D_mid = tilted_distributions(gains, references, mid)
            if D_mid >= requirement:
                hi = mid
            else:
                lo = mid
        beta = hi  # 取满足约束的上端点
        ps, D = tilted_distributions(gains, references, beta)
    if D <= 0:
        raise FloatingPointError("方向增益必须严格为正")
    return TiltResult(beta, ps, D, max_gain_sum, requirement, False)
