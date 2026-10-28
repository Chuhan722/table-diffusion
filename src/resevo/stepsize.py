"""整代矩与解析步长。

从方向分布得到速率 r_B(u)=P_B(u) 且保持项速率为 0，
整代矩 b_B=sum r，v_B=sum r*d，跨行交互 C=(1/2)[||sum v||_W^2-sum||v_B||_W^2]，
C 可正可负，不允许先截成零。
解析步长在 D>0 时 h*=min(h_feas, D/(2C))（C>0）否则 h_feas=1/max b，
可选阻尼 gamma 与期望改行数预算 B_max 都必须在抽样前决定。
步长限额与概率归一化保留原运算顺序；预测账本使用等价的稳定表达式。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_step_parameters(
    damping: float,
    max_expected_rows: float | None = None,
) -> None:
    """在冻结或提前停止之前验证外部步长参数。"""
    if not np.isfinite(damping) or not (0 < damping <= 1):
        raise ValueError("damping 必须为有限数且落在 (0,1]")
    if max_expected_rows is not None and (
        not np.isfinite(max_expected_rows) or max_expected_rows <= 0
    ):
        raise ValueError("max_expected_rows 必须为正的有限数或 None")


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
    *,
    unit_expected_rows: float | None = None,
    max_expected_rows: float | None = None,
) -> float:
    """解析步长，先取可行/曲率/改行预算上限，再阻尼并验证最终表示。"""
    validate_step_parameters(damping, max_expected_rows)
    if (
        not np.isfinite(max_leave_rate) or max_leave_rate <= 0
        or not np.isfinite(direction_gain) or direction_gain <= 0
        or not np.isfinite(interaction)
    ):
        raise FloatingPointError("numerical_indeterminate: 速率矩非法或非有限")
    if unit_expected_rows is not None and (
        not np.isfinite(unit_expected_rows) or unit_expected_rows <= 0
    ):
        raise FloatingPointError("numerical_indeterminate: 正方向的单位期望改行数须正且有限")
    if max_expected_rows is not None and unit_expected_rows is None:
        raise ValueError("改行预算需要 unit_expected_rows")
    feasible = 1.0 / max_leave_rate
    optimal = min(feasible, direction_gain / (2 * interaction)) if interaction > 0 else feasible
    if max_expected_rows is not None:
        optimal = min(optimal, max_expected_rows / unit_expected_rows)
    step = damping * optimal
    _validate_positive_step(step)
    return step


def _validate_positive_step(step: float) -> None:
    if not np.isfinite(step) or step <= 0:
        raise FloatingPointError("numerical_indeterminate: 正步长无法以非零有限 float64 表示")


def final_probabilities(
    rates: NDArray[np.float64],
    step: float,
    segment_offsets: NDArray[np.int64] | None = None,
) -> tuple[NDArray[np.float64], bool]:
    """按原求和/清理顺序构造最终概率，并返回是否保留了非保持质量。

    单块保留原 sum，拼接布局保留原 reduceat；不改为从舍入后的保持项反推质量。
    """
    _validate_positive_step(step)
    if rates.ndim != 1 or rates.size == 0:
        raise ValueError("速率须为非空一维向量")
    if segment_offsets is None:
        starts = np.array([0], dtype=np.int64)
    else:
        offsets = np.asarray(segment_offsets)
        if (
            offsets.ndim != 1 or offsets.size < 2
            or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != rates.size
            or np.any(np.diff(offsets) < 1)
        ):
            raise ValueError("概率段边界须覆盖全部非空段")
        starts = offsets[:-1]
    if np.any(rates[starts] != 0.0):
        raise FloatingPointError("numerical_indeterminate: 保持项速率必须为零")
    probabilities = step * rates
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise FloatingPointError("numerical_indeterminate: 最终概率非有限或为负")
    if segment_offsets is None:
        mass = float(probabilities[1:].sum())
        stay = 1.0 - mass
        if not np.isfinite(mass) or stay < -1e-12:
            raise FloatingPointError("numerical_indeterminate: 步长违反随机矩阵约束")
        probabilities[0] = max(stay, 0.0)
        if stay < 0.0:
            probabilities /= probabilities.sum()
        return probabilities, mass > 0.0
    mass = np.add.reduceat(probabilities, starts)
    stay = 1.0 - mass
    if not np.isfinite(mass).all() or np.any(stay < -1e-12):
        raise FloatingPointError("numerical_indeterminate: 步长违反随机矩阵约束")
    probabilities[starts] = np.maximum(stay, 0.0)
    for b in np.flatnonzero(stay < 0.0):
        lo, hi = int(offsets[b]), int(offsets[b + 1])
        probabilities[lo:hi] /= probabilities[lo:hi].sum()
    return probabilities, bool(np.any(mass > 0.0))


def expected_improvement(step: float, direction_gain: float, interaction: float) -> float:
    """同一 hD-h²C，用 h(D-hC) 避免先平方巨大/微小步长。"""
    _validate_positive_step(step)
    if not np.isfinite(direction_gain) or direction_gain <= 0 or not np.isfinite(interaction):
        raise FloatingPointError("numerical_indeterminate: 方向增益或交互项非法")
    improvement = step * (direction_gain - step * interaction)
    if not np.isfinite(improvement) or improvement < 0:
        raise FloatingPointError("numerical_indeterminate: 预测下降量无法可靠表示")
    return improvement


def kernel_statistics(
    old_loss: float,
    direction_gain: float,
    interaction: float,
    step: float,
    unit_expected_rows: float,
    has_movement: bool,
) -> tuple[float, float, float]:
    """验证活跃核的可发布账本；真实冻结在构造器中另行返回。"""
    if not has_movement:
        raise FloatingPointError("numerical_indeterminate: 正方向的最终非保持质量全部下溢")
    if not np.isfinite(old_loss) or old_loss < 0 or not np.isfinite(unit_expected_rows) or unit_expected_rows <= 0:
        raise FloatingPointError("numerical_indeterminate: 损失或期望改行数非法")
    expected_loss = old_loss - expected_improvement(step, direction_gain, interaction)
    # 先将较大的因子减半，避免 hD 的中间上溢或先减半微小因子的下溢。
    half_gain = (max(step, direction_gain) * 0.5) * min(step, direction_gain)
    upper = old_loss - half_gain
    expected_rows = step * unit_expected_rows
    if not all(np.isfinite(value) for value in (expected_loss, upper, expected_rows)) or expected_rows <= 0:
        raise FloatingPointError("numerical_indeterminate: 最终预测账本非有限或改行数下溢")
    return expected_loss, upper, expected_rows


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
    if max_expected_rows is None:
        raise ValueError("max_expected_rows 必须为正的有限数")
    validate_step_parameters(1.0, max_expected_rows)
    if not np.isfinite(unit_expected_rows) or unit_expected_rows < 0 or np.isnan(step) or step < 0:
        raise FloatingPointError("numerical_indeterminate: 改行预算的步长或单位改行数非法")
    if unit_expected_rows == 0:
        if not np.isfinite(step):
            raise FloatingPointError("numerical_indeterminate: 无改行时步长仍须有限")
        return step
    bounded = min(step, max_expected_rows / unit_expected_rows)
    _validate_positive_step(bounded)
    return bounded
