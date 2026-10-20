"""整代矩与解析步长。

从方向分布得到速率 r_B(u)=P_B(u) 且保持项速率为 0，
整代矩 b_B=sum r，v_B=sum r*d，跨行交互 C=(1/2)[||sum v||_W^2-sum||v_B||_W^2]，
C 可正可负，不允许先截成零。
解析步长在 D>0 时 h*=min(h_feas, D/(2C))（C>0）否则 h_feas=1/max b，
可选阻尼 gamma 与期望改行数预算 B_max 都必须在抽样前决定。
运算顺序与参考实现一致，便于对拍逐位核对。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rates_from_probabilities(
    probabilities: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """速率向量，非保持项取方向概率，保持项恒为 0。"""
    rates = []
    for P in probabilities:
        r = P.copy()
        r[0] = 0.0
        rates.append(r)
    return rates


def block_leave_rates(rates: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """各块离开速率 b_B=sum_u r_B(u)。"""
    return np.array([float(r.sum()) for r in rates])


def block_drifts(
    rates: list[NDArray[np.float64]],
    deltas: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """各块漂移 v_B=sum_u r_B(u) d_B(u)。"""
    return [r @ d for r, d in zip(rates, deltas)]


def cross_interaction(
    drifts: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
    multiplicities=None,
) -> float:
    """跨行交互 C，快公式，不遍历块对，可为负。

    multiplicities 给出每块重数 c，等价于把每块展开成 c 份相同块，
    总漂移为 sum c*v，块自交叉扣除项为 sum c*||v||_W^2，None 走原路径逐位不变。
    """
    v = np.stack(drifts)
    if multiplicities is None:
        total = v.sum(axis=0)
        return float((np.dot(total * weights, total) - np.sum(v * v * weights)) / 2)
    c = np.asarray(multiplicities, dtype=np.float64)[:, None]
    total = (c * v).sum(axis=0)
    return float((np.dot(total * weights, total) - np.sum(c * v * v * weights)) / 2)


def cross_interaction_slow(
    drifts: list[NDArray[np.float64]],
    weights: NDArray[np.float64],
) -> float:
    """跨行交互 C 的慢速双重求和版本 sum_{B<B'} v_B^T W v_B'，测试标准。"""
    total = 0.0
    for i in range(len(drifts)):
        for j in range(i + 1, len(drifts)):
            total += float(np.dot(drifts[i] * weights, drifts[j]))
    return total


def analytic_step(
    max_leave_rate: float,
    direction_gain: float,
    interaction: float,
    damping: float = 1.0,
) -> float:
    """解析步长 h=gamma*min(h_feas, D/(2C))，C<=0 时只受可行上限约束。"""
    if not (0 < damping <= 1):
        raise ValueError("damping 必须落在 (0,1]")
    if max_leave_rate <= 0 or not np.isfinite(direction_gain) or not np.isfinite(interaction):
        raise FloatingPointError("速率矩非法或非有限")
    feasible = 1.0 / max_leave_rate
    optimal = min(feasible, direction_gain / (2 * interaction)) if interaction > 0 else feasible
    return damping * optimal


def expected_changed_rows_at_unit_step(
    rates: list[NDArray[np.float64]],
    outcomes: list[tuple[tuple[int, ...], ...]],
    multiplicities=None,
) -> float:
    """单位步长下的期望改变行数 H=sum_B sum_u r_B(u) c_B(u)。

    c_B(u) 为后继元组里与源元组不同的真实行数，源元组固定在索引 0。
    multiplicities 给出每块重数，重数份相同块的期望改行数按重数累加。
    """
    total = 0.0
    for idx, (r, outs) in enumerate(zip(rates, outcomes)):
        source = outs[0]
        if multiplicities is None:
            # 原路径逐项累加，运算顺序保持逐位不变
            for k in range(1, len(outs)):
                changed = sum(1 for a, b in zip(outs[k], source) if a != b)
                total += float(r[k]) * changed
            continue
        block_total = 0.0
        for k in range(1, len(outs)):
            changed = sum(1 for a, b in zip(outs[k], source) if a != b)
            block_total += float(r[k]) * changed
        total += float(multiplicities[idx]) * block_total
    return total


def apply_row_budget(step: float, unit_expected_rows: float, max_expected_rows: float) -> float:
    """期望改行数预算，h=min(h, B_max/H)，在抽样前决定，不看抽样结果。"""
    if max_expected_rows <= 0:
        raise ValueError("B_max 必须为正")
    if unit_expected_rows <= 0:
        return step
    return min(step, max_expected_rows / unit_expected_rows)
