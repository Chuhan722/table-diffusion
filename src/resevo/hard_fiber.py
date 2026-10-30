"""可选硬纤维后处理：保持完整指定边缘和全部一阶的惰性对称交换。

有序可行表的均匀测度平稳，直方图质量与 N!/product(n!) 成正比。
单次与四行原子双交换都保留；约束不满足时留在原状态。
固定更新预算不构成全纤维连通、充分混合、熵最优或真实质量改善的证明。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

HARD_FIBER_KERNEL = "lazy-symmetric-exchanges-v1"


@dataclass(frozen=True)
class HardFiberDiagnostics:
    requested_steps: int
    executed_steps: int
    lazy_steps: int
    single_proposals: int
    paired_proposals: int
    rejected_constraints: int
    identity_proposals: int
    relabel_proposals: int
    histogram_moves: int
    moved_cells: int
    constraint_checks: int
    preserved_sets: tuple[tuple[int, ...], ...]
    stop_reason: str
    kernel_contract: str = HARD_FIBER_KERNEL
    backend: str = "cpu"
    target_measure: str = "uniform_labeled_tables"
    invariants_verified: bool = True
    mixing_certified: bool = False


@dataclass
class HardFiberResult:
    codes: NDArray[np.int64]
    diagnostics: HardFiberDiagnostics


def _validated_input(X, doms, sets, steps, rng):
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("硬纤维更新步数必须为非负整数")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("硬纤维后处理需要独立的 numpy.random.Generator")
    values = np.asarray(X)
    if (
        values.ndim != 2 or not all(values.shape)
        or not np.issubdtype(values.dtype, np.integer)
    ):
        raise ValueError("硬纤维输入必须为非空二维整数编码表")
    domains = np.asarray(doms, dtype=object)
    if (
        domains.ndim != 1 or len(domains) != values.shape[1]
        or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d <= 0 for d in domains)
    ):
        raise ValueError("公开值域大小必须为与表字段数一致的正整数向量")
    domains = tuple(map(int, domains))
    limit = np.iinfo(np.int64).max
    if max(domains) > limit or (values.dtype.kind == "u" and int(values.max()) > limit):
        raise ValueError("公开值域和编码必须可由 int64 表示")
    original = values.astype(np.int64, copy=True)
    if any(np.any(original[:, i] < 0) or np.any(original[:, i] >= int(d)) for i, d in enumerate(domains)):
        raise ValueError("硬纤维编码超出公开值域")
    if not isinstance(sets, (list, tuple)):
        raise ValueError("硬纤维约束须为字段索引集合的列表")
    kept = {(i,) for i in range(values.shape[1])}
    for fields in sets:
        raw = np.asarray(fields, dtype=object)
        if raw.ndim != 1:
            raise ValueError("每个硬纤维约束须为一维字段索引")
        if not raw.size:
            continue  # 常量计数由行数守恒，不需要空字段边缘。
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)) for i in raw):
            raise ValueError("硬纤维字段索引必须为整数")
        indices = tuple(sorted(set(map(int, raw))))
        if indices[0] < 0 or indices[-1] >= values.shape[1]:
            raise ValueError("硬纤维字段索引越界")
        kept.add(indices)
    return original, tuple(sorted(kept))


def _marginal_counts(rows, fields):
    return Counter(tuple(int(row[j]) for j in fields) for row in rows)


def _apply_exchange(current, row_ids, fields, kept, by_field):
    """对一个合法形状的提议做原子校验；供实际循环及精确核审计共用。"""
    before = current[row_ids].copy()
    after = before.copy()
    for pair, field in enumerate(fields):
        a, b = 2 * pair, 2 * pair + 1
        after[a, field], after[b, field] = before[b, field], before[a, field]
    moved = int(np.count_nonzero(after != before))
    if moved == 0:
        return "identity_proposals", 0, 0
    affected = sorted({i for field in fields for i in by_field[field]})
    checked = 0
    for i in affected:
        checked += 1
        if _marginal_counts(before, kept[i]) != _marginal_counts(after, kept[i]):
            return "rejected_constraints", 0, checked
    all_fields = tuple(range(current.shape[1]))
    outcome = (
        "relabel_proposals"
        if _marginal_counts(before, all_fields) == _marginal_counts(after, all_fields)
        else "histogram_moves"
    )
    current[row_ids] = after
    return outcome, moved, checked


def _verify_marginals(original, current, kept):
    for fields in kept:
        if _marginal_counts(original, fields) != _marginal_counts(current, fields):
            raise RuntimeError(f"硬纤维完整边缘复核失败：{fields}")


def _step(current, kept, by_field, rng):
    kind = int(rng.integers(4))
    if kind < 2:
        return "lazy_steps", None, 0, 0
    n, width = current.shape
    paired = n >= 4 and kind == 3
    row_ids = rng.choice(n, size=4 if paired else 2, replace=False)
    fields = tuple(int(rng.integers(width)) for _ in range(2 if paired else 1))
    outcome, moved, checked = _apply_exchange(current, row_ids, fields, kept, by_field)
    return "paired_proposals" if paired else "single_proposals", outcome, moved, checked


def hard_fiber_sample(
    X: NDArray[np.int64], doms: list[int], sets: list[tuple[int, ...]],
    steps: int, rng: np.random.Generator,
) -> HardFiberResult:
    """返回独立持有的终点编码表，不修改输入；锚点只取传入的合成表。

    全部一阶及 sets 的完整边缘逐格保持，不读取真实表或带噪目标。
    N>=4 时，每步以1/2保持、1/4单次交换、1/4四行原子双交换；
    行位置和字段均匀选择、独立于当前值，双交换的字段可相同。
    N=2/3 时为1/2保持、1/2单次交换；N=1明确返回 no_row_pairs。
    每个动作都是自身的逆，不满足约束时作自环；不按损失接受或提早停止。

    单次交换保留原条件列洗牌的可达支撑；双交换可跨越部分局部障碍，
    但不是任意约束的Markov基。没有连通性或有限预算混合保证。
    steps=0仍校验输入但不消耗rng；正预算不保证产生非平凡变化。
    整数元组稀疏计数避免分配值域乘积大小的张量。
    """
    original, kept = _validated_input(X, doms, sets, steps, rng)
    current = original.copy()
    n, width = current.shape
    by_field = tuple(
        tuple(i for i, fields in enumerate(kept) if len(fields) > 1 and j in fields)
        for j in range(width)
    )
    counts = dict(
        executed_steps=0, lazy_steps=0, single_proposals=0, paired_proposals=0,
        rejected_constraints=0, identity_proposals=0, relabel_proposals=0,
        histogram_moves=0, moved_cells=0, constraint_checks=0,
    )
    if steps == 0:
        reason = "disabled"
    elif n == 1:
        reason = "no_row_pairs"
    else:
        reason = "step_limit"
        for _ in range(int(steps)):
            counts["executed_steps"] += 1
            family, outcome, moved, checked = _step(current, kept, by_field, rng)
            counts[family] += 1
            if outcome is not None:
                counts[outcome] += 1
            counts["moved_cells"] += moved
            counts["constraint_checks"] += checked
    _verify_marginals(original, current, kept)
    return HardFiberResult(current, HardFiberDiagnostics(
        requested_steps=int(steps), preserved_sets=kept, stop_reason=reason, **counts,
    ))
