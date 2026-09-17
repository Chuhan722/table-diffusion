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
class FlatTiltResult:
    """拼接布局的熵校准结果，probabilities 与拼接后的支持项对齐。"""

    beta: float
    probabilities: NDArray[np.float64]  # T，全部块的支持项拼接
    direction_gain: float
    max_gain_sum: float
    required_gain: float
    frozen: bool


def calibrate_beta_flat(
    gains_flat: NDArray[np.float64],
    references_flat: NDArray[np.float64],
    segment_offsets: NDArray[np.int64],
    old_loss: float,
    alpha: float = 0.5,
    numerical_tol: float = 1e-12,
    multiplicities=None,
) -> FlatTiltResult:
    """calibrate_beta 的拼接矢量化版，数学定义与逐块版完全一致。

    所有块的增益与参考质量拼成一维数组，segment_offsets 给出块边界，
    每块的源项固定在段首，分段 log-sum-exp 用 reduceat 一次算完，
    括根与二分的判据结构与逐块版相同，浮点求和顺序不同数值容差内一致。
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须落在 (0,1)")
    if not np.isfinite(numerical_tol) or numerical_tol < 0:
        raise ValueError("numerical_tol 必须有限且非负")
    offsets = np.asarray(segment_offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size < 2 or offsets[0] != 0:
        raise ValueError("段边界必须从 0 开始且至少一段")
    if np.any(np.diff(offsets) < 1) or offsets[-1] != gains_flat.shape[0]:
        raise ValueError("每段至少含源项且边界须覆盖全部支持项")
    num_groups = offsets.size - 1
    seg_starts = offsets[:-1]
    lengths = np.diff(offsets)
    seg_ids = np.repeat(np.arange(num_groups), lengths)

    if multiplicities is not None:
        mult = np.asarray(multiplicities, dtype=np.float64)
        if mult.shape != (num_groups,) or np.any(mult < 1) or not np.isfinite(mult).all():
            raise ValueError("multiplicities 必须与段数等长且每项至少为 1")
    else:
        mult = np.ones(num_groups, dtype=np.float64)

    gmax = np.maximum.reduceat(gains_flat, seg_starts)
    max_gain_sum = float(mult @ np.maximum(gmax, 0.0))
    requirement = alpha * max_gain_sum
    scale = max(1.0, old_loss)

    if max_gain_sum <= numerical_tol * scale:
        frozen = np.zeros_like(references_flat)
        frozen[seg_starts] = 1.0
        return FlatTiltResult(0.0, frozen, 0.0, max_gain_sum, requirement, True)

    log_ref = np.log(references_flat)
    centered = gains_flat - gmax[seg_ids]

    def evaluate(beta: float):
        logits = log_ref + beta * centered
        m = np.maximum.reduceat(logits, seg_starts)
        z = np.exp(logits - m[seg_ids])
        norm = np.add.reduceat(z, seg_starts)
        P = z / norm[seg_ids]
        per_group = np.add.reduceat(P * gains_flat, seg_starts)
        return P, float(mult @ per_group)

    ps, D = evaluate(0.0)
    if D >= requirement:
        beta = 0.0
    else:
        lo = 0.0
        hi = 1.0 / max(max_gain_sum, 1e-300)
        for _ in range(100):
            ps, D = evaluate(hi)
            if D >= requirement:
                break
            hi *= 2.0
        else:
            raise FloatingPointError("无法括住 beta 根，检查数值条件")
        for _ in range(80):
            mid = (lo + hi) / 2.0
            if mid <= lo or mid >= hi:
                break  # 区间已到浮点分辨极限，继续二分不再改变端点
            _, D_mid = evaluate(mid)
            if D_mid >= requirement:
                hi = mid
            else:
                lo = mid
            if hi - lo <= 1e-13 * hi:
                break  # 相对宽度到达双精度水平，再分下去只动末两位
        beta = hi  # 取满足约束的上端点
        ps, D = evaluate(beta)
    if D <= 0:
        raise FloatingPointError("方向增益必须严格为正")
    return FlatTiltResult(beta, ps, D, max_gain_sum, requirement, False)


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
    multiplicities=None,
) -> tuple[list[NDArray[np.float64]], float]:
    """给定 beta 的各块指数倾斜分布与平均增益，log-sum-exp 归一化。

    multiplicities 给出每个块的重数，c 个相同块共享同一份 P，
    平均增益按重数加权 D=sum_g c_g*(P_g@g_g)，None 走原逐块路径逐位不变。
    """
    ps = []
    total = 0.0
    for idx, (g, R) in enumerate(zip(gains, references)):
        # 先减去块内最大增益再乘 beta，避免正向上溢，运算顺序与参考一致
        logits = np.log(R) + beta * (g - np.max(g))
        P = np.exp(logits - logsumexp(logits))
        ps.append(P)
        if multiplicities is None:
            total += float(P @ g)
        else:
            total += float(multiplicities[idx]) * float(P @ g)
    return ps, total


def calibrate_beta(
    gains: list[NDArray[np.float64]],
    references: list[NDArray[np.float64]],
    old_loss: float,
    alpha: float = 0.5,
    numerical_tol: float = 1e-12,
    multiplicities=None,
) -> TiltResult:
    """求共享 beta 与方向分布。

    old_loss 只用来给冻结判据定尺度，判据与参考实现一致，
    M 不超过 numerical_tol*max(1,old_loss) 时视为没有正方向，返回冻结结果。
    multiplicities 给出每个块的重数，M 与 D 都按重数加权，
    等价于把每块展开成重数份相同块，None 走原路径逐位不变。
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须落在 (0,1)")
    if not np.isfinite(numerical_tol) or numerical_tol < 0:
        raise ValueError("numerical_tol 必须有限且非负")
    if multiplicities is not None:
        mult = np.asarray(multiplicities, dtype=np.float64)
        if mult.shape != (len(gains),) or np.any(mult < 1) or not np.isfinite(mult).all():
            raise ValueError("multiplicities 必须与块数等长且每项至少为 1")
        max_gain_sum = float(
            sum(float(c) * max(float(np.max(g)), 0.0) for c, g in zip(mult, gains))
        )
    else:
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

    ps, D = tilted_distributions(gains, references, 0.0, multiplicities)
    if D >= requirement:
        beta = 0.0
    else:
        lo = 0.0
        hi = 1.0 / max(max_gain_sum, 1e-300)
        for _ in range(100):
            ps, D = tilted_distributions(gains, references, hi, multiplicities)
            if D >= requirement:
                break
            hi *= 2.0
        else:
            raise FloatingPointError("无法括住 beta 根，检查数值条件")
        for _ in range(80):
            mid = (lo + hi) / 2.0
            _, D_mid = tilted_distributions(gains, references, mid, multiplicities)
            if D_mid >= requirement:
                hi = mid
            else:
                lo = mid
        beta = hi  # 取满足约束的上端点
        ps, D = tilted_distributions(gains, references, beta, multiplicities)
    if D <= 0:
        raise FloatingPointError("方向增益必须严格为正")
    return TiltResult(beta, ps, D, max_gain_sum, requirement, False)
