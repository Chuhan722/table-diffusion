"""查询增量与转移增益。

对块 B 与后继元组 u，查询增量 d_B(u)=a(u)-a(x_B)，块内贡献求和。
转移增益 G_B(u)=e^T W d_B(u)-(1/2)||d_B(u)||_W^2，
它精确等于把该块换成 u 后的损失下降 L(S)-L(S_{B->u})，
带二次项，只看线性方向项会误判。
正常计数路径保留参考实现的运算顺序；可能消减的局部项另作精确复核。
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .state import Workload, loss_from_residual, table_loss


def validate_numerical_tol(numerical_tol: float) -> None:
    if not np.isfinite(numerical_tol) or numerical_tol < 0:
        raise ValueError("numerical_tol 必须有限且非负")


def integral_values(*arrays) -> bool:
    """有界临时内存检查整数量，供精确的整数/半整数快速路径使用。"""
    for values in arrays:
        for chunk in np.nditer(
            values, flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"], order="K", buffersize=65536,
        ):
            if not np.all(chunk == np.rint(chunk)):
                return False
    return True


def gain_recheck_mask(
    gains, magnitudes, terms, numerical_tol: float, *,
    integral: bool = False, xp=np,
):
    """按局部算术规模筛选需复核的项，不把容差当作物理收益门槛。"""
    validate_numerical_tol(numerical_tol)
    if not bool(xp.all(xp.isfinite(gains))) or not bool(xp.all(xp.isfinite(magnitudes))):
        raise FloatingPointError("numerical_indeterminate: 局部增益或运算规模非有限")
    if bool(xp.any(magnitudes < 0)):
        raise FloatingPointError("numerical_indeterminate: 局部运算规模不能为负")
    if integral and bool(xp.all(magnitudes <= 2.0**50)):
        return xp.zeros_like(gains, dtype=xp.bool_)
    operations = 8.0 * (xp.asarray(terms, dtype=xp.float64) + 1.0)
    ku = operations * (np.finfo(np.float64).eps / 2.0)
    if bool(xp.any(ku >= 0.5)):
        raise FloatingPointError("numerical_indeterminate: 无法建立有效的局部舍入界")
    gamma = ku / (1.0 - ku)
    relative = xp.maximum(numerical_tol, gamma / (1.0 - gamma))
    threshold = xp.maximum(
        relative * magnitudes, operations * np.nextafter(0.0, 1.0),
    )
    mask = (xp.asarray(terms) > 0) & (xp.abs(gains) <= threshold)
    if integral:
        # 该范围内整数乘加及最终半整数相减可精确表示，不把精确零当成数值不确定。
        mask &= magnitudes > 2.0**50
    return mask


def _fraction(value, cache):
    value = float(value)
    if not np.isfinite(value):
        raise FloatingPointError("numerical_indeterminate: 精确复核输入非有限")
    if value not in cache:
        cache[value] = Fraction.from_float(value)
    return cache[value]


def _rounded_gain(gain: Fraction) -> float:
    try:
        result = float(gain)
    except OverflowError as exc:
        raise FloatingPointError("numerical_indeterminate: 局部增益超出 float64 范围") from exc
    if not np.isfinite(result) or (gain != 0 and result == 0.0):
        raise FloatingPointError("numerical_indeterminate: 局部增益无法以非零有限 float64 表示")
    return result


class ExactGainOracle:
    """按需从目标与旧表答案复核，不能从已舍入的 residual 反推出答案。"""

    def __init__(self, weights, residual_at, fractions):
        self.weights = weights
        self.residual_at = residual_at
        self.fractions = fractions
        self.residuals = {}

    @classmethod
    def from_workload(cls, workload: Workload, state_ids, multiplicities=None):
        if multiplicities is None:
            states, multiplicities = np.unique(state_ids, return_counts=True)
        else:
            states = np.asarray(state_ids)
        if not integral_values(multiplicities) or np.any(np.asarray(multiplicities) < 0):
            raise ValueError("旧表记录重数必须为非负整数")
        counts = [int(count) for count in multiplicities]
        fractions = {}

        def residual_at(query):
            column = workload.features[states, query]
            if np.all((column == 0) | (column == 1)):
                answer = Fraction(sum(counts[i] for i in np.flatnonzero(column)))
            else:
                answer = sum(
                    (_fraction(value, fractions) * count for value, count in zip(column, counts)),
                    Fraction(0),
                )
            return _fraction(workload.target[query], fractions) - answer

        return cls(workload.weights, residual_at, fractions)

    @classmethod
    def from_answers(cls, target, weights, answers):
        fractions = {}

        def residual_at(query):
            return _fraction(target[query], fractions) - _fraction(answers[query], fractions)

        return cls(weights, residual_at, fractions)

    def _term(self, query, delta):
        if delta == 0:
            return Fraction(0)
        if query not in self.residuals:
            self.residuals[query] = self.residual_at(query)
        return _fraction(self.weights[query], self.fractions) * delta * (self.residuals[query] - delta / 2)

    def sparse_gain(self, deltas, indices):
        gain = Fraction(0)
        for query, delta in zip(indices, deltas):
            gain += self._term(int(query), _fraction(delta, self.fractions))
        return _rounded_gain(gain)

    def block_gain(self, features, source, outcome):
        if source == outcome:
            return 0.0
        before, after = features[list(source)], features[list(outcome)]
        gain = Fraction(0)
        for query in np.flatnonzero(np.any(before != after, axis=0)):
            delta = sum(
                (_fraction(b, self.fractions) - _fraction(a, self.fractions)
                 for a, b in zip(before[:, query], after[:, query])),
                Fraction(0),
            )
            gain += self._term(int(query), delta)
        return _rounded_gain(gain)


def exact_local_gain(deltas, residual, weights, cache=None) -> float:
    """对给定 float64 局部输入作有理数复核，不读取原表或替代收益定义。"""
    d, e, w = (np.asarray(value, dtype=np.float64) for value in (deltas, residual, weights))
    if d.ndim != 1 or d.shape != e.shape or d.shape != w.shape:
        raise ValueError("局部复核的增量、残差和权重须为等长一维向量")
    if not all(np.isfinite(value).all() for value in (d, e, w)):
        raise FloatingPointError("numerical_indeterminate: 局部复核输入非有限")
    fractions = {} if cache is None else cache

    def exact(value):
        return _fraction(value, fractions)

    gain = Fraction(0)
    for j in np.flatnonzero(d):
        delta = exact(d[j])
        gain += exact(w[j]) * delta * (exact(e[j]) - delta / 2)
    return _rounded_gain(gain)


def exact_sparse_path_gains(paths, parts, oracle, hs_values=None, hs_cols=None):
    """复核选中的稀疏路径；parts 为 (路径号, indptr, 题号, 增量) 的 CPU 小片段。"""
    rows = {}
    for path_ids, indptr, indices, data in parts:
        for row, path in enumerate(path_ids):
            lo, hi = int(indptr[row]), int(indptr[row + 1])
            if hi > lo:
                rows.setdefault(int(path), []).append((indices[lo:hi], data[lo:hi]))
    out = np.empty(len(paths), dtype=np.float64)
    for row, path in enumerate(paths):
        slices = list(rows.get(int(path), ()))
        if hs_values is not None:
            live = np.flatnonzero(hs_values[row])
            if live.size:
                slices.append((hs_cols[live], hs_values[row, live]))
        if not slices:
            out[row] = 0.0
            continue
        indices = np.concatenate([item[0] for item in slices])
        values = np.concatenate([item[1] for item in slices])
        if np.unique(indices).size != indices.size:
            raise ValueError("局部复核路径的查询增量必须已经规范化合并")
        out[row] = oracle.sparse_gain(values, indices)
    return out


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
    *,
    numerical_tol: float = 1e-12,
    exact_gain=None,
    integer_inputs: bool | None = None,
) -> NDArray[np.float64]:
    """转移增益，保留原浮点路径；仅可能发生消减的项作局部精确复核。"""
    we = weights * residual
    linear = deltas @ we
    quadratic = np.sum(deltas * deltas * weights, axis=1) / 2
    gains = linear - quadratic
    magnitudes = np.abs(deltas) @ np.abs(we) + quadratic
    check = gain_recheck_mask(
        gains, magnitudes, np.count_nonzero(deltas, axis=1), numerical_tol,
        integral=integral_values(weights, residual, deltas) if integer_inputs is None else integer_inputs,
    )
    cache = {}
    for row in np.flatnonzero(check):
        gains[row] = (
            exact_gain(int(row)) if exact_gain is not None
            else exact_local_gain(deltas[row], residual, weights, cache)
        )
    return gains


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
