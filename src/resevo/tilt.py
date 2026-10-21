"""熵校准，把增益变成方向分布。

在参考分布 R 之上解 min sum KL(P||R) s.t. sum P*G >= delta 的指数倾斜解，
P_B(u;beta)=R_B(u)exp(beta*G_B(u))/Z_B(beta)，所有块共享同一个 beta。
平均增益 D(beta) 关于 beta 单调不减，其导数为各块增益方差之和。
M=sum_B max_u G_B(u) 为最大可达增益总和，delta=alpha*M。
若 D(0)>=delta 取 beta=0，否则先翻倍括根再求唯一根。
整个数值步骤只反复评价概率与平均增益，不抽样任何表。
逐块版 calibrate_beta 的括根与二分次数、端点选取和 log-sum-exp 顺序
与参考实现完全一致便于逐位对拍，拼接版 calibrate_beta_flat 走布伦特法，
同一个方程同容差 1e-13，只是逼近路线更省全量扫描。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import logsumexp

from .gain import validate_numerical_tol


def validate_tilt_parameters(alpha: float, numerical_tol: float) -> None:
    if not (0 < alpha < 1):
        raise ValueError("alpha 必须落在 (0,1)")
    validate_numerical_tol(numerical_tol)


def calibration_requirement(max_gain_sum: float, alpha: float, old_loss: float) -> float:
    """旧损失仅检查是否可表示，不参与方向存在性或精度阈值。"""
    if not np.isfinite(old_loss) or not np.isfinite(max_gain_sum):
        raise FloatingPointError("numerical_indeterminate: 损失或最大增益和非有限")
    if max_gain_sum < 0:
        raise FloatingPointError("numerical_indeterminate: 最大增益和不能为负")
    requirement = alpha * max_gain_sum
    if max_gain_sum > 0 and (requirement <= 0 or not np.isfinite(requirement)):
        raise FloatingPointError("numerical_indeterminate: 正收益要求无法以 float64 表示")
    return requirement


def validate_tilt_result(direction_gain, requirement, max_gain_sum, numerical_tol):
    if not np.isfinite(direction_gain) or direction_gain <= 0:
        raise FloatingPointError("numerical_indeterminate: 校准方向增益须为正的有限数")
    if direction_gain < requirement and (requirement - direction_gain) / max_gain_sum > numerical_tol:
        raise FloatingPointError("numerical_indeterminate: 校准误差超过局部增益相对容差")


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
    beta_hint: float | None = None,
) -> FlatTiltResult:
    """calibrate_beta 的拼接矢量化版，数学定义与逐块版完全一致。

    所有块的增益与参考质量拼成一维数组，segment_offsets 给出块边界，
    每块的源项固定在段首，分段归一化用 reduceat 一次算完，
    段内增益先减段最大故指数恒不上溢，无需再逐次求段最大移位，
    求根用布伦特法，括根区间保证收缩且超线性收敛，
    停止判据为相对容差 1e-13 与逐块二分版同精度，返回达标一侧的端点，
    根与逐块版在容差内一致，末几位小数可能不同，
    beta_hint 给出括根起点，典型取上一轮的根，省掉从 1/M 起的翻倍串。
    """
    validate_tilt_parameters(alpha, numerical_tol)
    if not np.isfinite(gains_flat).all():
        raise FloatingPointError("numerical_indeterminate: 输入增益非有限")
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
    requirement = calibration_requirement(max_gain_sum, alpha, old_loss)

    if max_gain_sum == 0.0:
        if np.any(gmax > 0):
            raise FloatingPointError("numerical_indeterminate: 正增益在聚合中丢失")
        frozen = np.zeros_like(references_flat)
        frozen[seg_starts] = 1.0
        return FlatTiltResult(0.0, frozen, 0.0, max_gain_sum, requirement, True)

    # 段内先减段最大，centered 全体非正，exp 无上溢之忧，天然免去逐次移位
    centered = gains_flat - gmax[seg_ids]

    def evaluate(beta: float):
        if not np.isfinite(beta):
            raise FloatingPointError("numerical_indeterminate: beta 超出有限表示范围")
        zw = references_flat * np.exp(beta * centered)
        norm = np.add.reduceat(zw, seg_starts)
        P = zw / norm[seg_ids]
        per_group = np.add.reduceat(P * gains_flat, seg_starts)
        return P, float(mult @ per_group)

    ps, D = evaluate(0.0)
    if D >= requirement:
        beta = 0.0
    else:
        # 括根，有热启动点就从它出发折半或翻倍，没有则从 1/M 翻倍
        lo = 0.0
        hi = 0.0
        probe = 1.0 / max(max_gain_sum, 1e-300)
        if beta_hint is not None and np.isfinite(beta_hint) and beta_hint > 0.0:
            probe = float(beta_hint)
        ps, D = evaluate(probe)
        if D >= requirement:
            hi = probe
            for _ in range(100):
                probe /= 2.0
                ps, D = evaluate(probe)
                if D < requirement:
                    lo = probe
                    break
                hi = probe
            else:
                raise FloatingPointError("无法括住 beta 根，检查数值条件")
        else:
            lo = probe
            for _ in range(100):
                probe *= 2.0
                ps, D = evaluate(probe)
                if D >= requirement:
                    hi = probe
                    break
                lo = probe
            else:
                raise FloatingPointError("无法括住 beta 根，检查数值条件")
        # 布伦特法求根，区间保证收缩且超线性收敛，容差与纯二分版同为相对 1e-13
        beta = float(
            brentq(
                lambda b: evaluate(b)[1] - requirement,
                lo,
                hi,
                xtol=1e-300,
                rtol=1e-13,
                maxiter=200,
            )
        )
        ps, D = evaluate(beta)
        # 根可能落在约束下侧，向上微推到达标一侧，与二分版取上端点的语义一致
        for _ in range(8):
            if D >= requirement:
                break
            beta = beta * (1.0 + 4e-13) if beta > 0 else hi * 1e-16
            ps, D = evaluate(beta)
    validate_tilt_result(D, requirement, max_gain_sum, numerical_tol)
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
    if not np.isfinite(beta):
        raise FloatingPointError("numerical_indeterminate: beta 超出有限表示范围")
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

    给定的增益须已由构造层处理局部消减；只有 M=0 才返回冻结结果。
    old_loss 保留用于兼容和有限性检查，不能改变方向存在性。
    numerical_tol 用于校准误差相对 M 的检查，不是物理收益停止门槛。
    multiplicities 给出每个块的重数，M 与 D 都按重数加权，
    等价于把每块展开成重数份相同块，None 走原路径逐位不变。
    """
    validate_tilt_parameters(alpha, numerical_tol)
    if any(not np.isfinite(g).all() for g in gains):
        raise FloatingPointError("numerical_indeterminate: 输入增益非有限")
    maxima = [max(float(np.max(g)), 0.0) for g in gains]
    if multiplicities is not None:
        mult = np.asarray(multiplicities, dtype=np.float64)
        if mult.shape != (len(gains),) or np.any(mult < 1) or not np.isfinite(mult).all():
            raise ValueError("multiplicities 必须与块数等长且每项至少为 1")
        max_gain_sum = float(
            sum(float(c) * maximum for c, maximum in zip(mult, maxima))
        )
    else:
        max_gain_sum = float(sum(maxima))
    requirement = calibration_requirement(max_gain_sum, alpha, old_loss)

    if max_gain_sum == 0.0:
        if any(maximum > 0 for maximum in maxima):
            raise FloatingPointError("numerical_indeterminate: 正增益在聚合中丢失")
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
    validate_tilt_result(D, requirement, max_gain_sum, numerical_tol)
    return TiltResult(beta, ps, D, max_gain_sum, requirement, False)
