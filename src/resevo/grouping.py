"""重复记录压缩，相同状态的行共享菜单增益与概率计算。

按文档规模化建议第六条，当若干相同记录具有相同支持和参考分布时，
候选、增益、倾斜概率只算一次，但抽样绝不能用一个共同开关让它们一起变，
必须按多项分布抽取各后继的数量，这与组内每行独立按同一核抽样精确同分布，
方差与跨行相关性都不变。

数学对应关系，组 g 含 c_g 行，每行是一个独立单行块，块结构完全相同，
M 与 D 按重数加权，跨行交互 C 的总漂移为 sum c_g*v_g，
自交叉扣除项为 sum c_g*||v_g||_W^2，期望改行数按重数累加，
期望损失恒等式 E[L']=L-hD+h^2C 在加权矩下逐字成立。

卡住时的配对板块按行级工作，本模块在冻结重试轮回退到行级菜单与行级核，
慢一轮不影响正确性，平时轮次全部走组路径。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidates import BlockSupport, NormalizedBlock, normalize_block, validate_partition
from .dataset import StateRegistry
from .editspace import (
    EditBudget,
    _explore_edits,
    _joint_edits,
    _single_field_edits,
    generate_edit_supports,
    tuples_from_ids,
)
from .engine import BlockKernel, EvolveResult, RoundRecord, build_kernel, sample_next
from .gain import block_deltas, block_gains
from .pairing import PairingBudget, build_paired_supports
from .state import Workload, loss_from_residual
from .stepsize import (
    analytic_step,
    block_drifts,
    block_leave_rates,
    cross_interaction,
    expected_changed_rows_at_unit_step,
    rates_from_probabilities,
)
from .tilt import calibrate_beta


@dataclass(frozen=True)
class GroupedTable:
    """按状态编号分组后的表，行列表按原行索引升序，组按状态编号升序。"""

    unique_ids: NDArray[np.int64]  # G，每组的状态编号
    counts: NDArray[np.int64]  # G，每组的行数
    row_lists: tuple[tuple[int, ...], ...]  # 每组的原行索引

    @property
    def num_groups(self) -> int:
        return len(self.unique_ids)

    @property
    def num_rows(self) -> int:
        return int(self.counts.sum())


def group_state_ids(state_ids) -> GroupedTable:
    """把状态编号向量分组，相同状态的行归入同一组。"""
    s = np.asarray(state_ids)
    if s.ndim != 1 or s.size == 0 or not np.issubdtype(s.dtype, np.integer):
        raise ValueError("state_ids 必须是非空一维整数向量")
    uniq, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    lists: list[list[int]] = [[] for _ in range(len(uniq))]
    for i, g in enumerate(inverse):
        lists[int(g)].append(i)
    return GroupedTable(
        uniq.astype(np.int64),
        counts.astype(np.int64),
        tuple(tuple(rows) for rows in lists),
    )


def _weighted_donor_tuple(
    grouped: GroupedTable,
    registry: StateRegistry,
    cumulative: NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    """按重数加权抽供体状态，与在展开表上均匀抽一行精确同分布。"""
    r = int(rng.integers(int(cumulative[-1])))
    g = int(np.searchsorted(cumulative, r, side="right"))
    return registry.state_tuple(int(grouped.unique_ids[g]))


def _group_donor_copies(
    current: tuple[str, ...],
    grouped: GroupedTable,
    registry: StateRegistry,
    cumulative: NDArray[np.int64],
    budget: EditBudget,
    rng: np.random.Generator,
) -> list[tuple[str, ...]]:
    """供体复制的组版本，供体行按重数加权抽取等价于均匀抽行。"""
    schema = registry.schema
    results = []
    for _ in range(budget.donor_copies):
        donor = _weighted_donor_tuple(grouped, registry, cumulative, rng)
        k = int(rng.integers(1, 3))  # 每次复制 1 到 2 个字段
        fields = rng.permutation(schema.num_fields)[:k]
        edited = list(current)
        for j in fields:
            edited[int(j)] = donor[int(j)]
        results.append(tuple(edited))
    return results


def generate_group_edit_supports(
    grouped: GroupedTable,
    registry: StateRegistry,
    rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
) -> list[BlockSupport]:
    """为每个组生成一份编辑菜单，组内所有行共享。

    rows 里放的是组索引不是行索引，规范化时配合 unique_ids 取源状态。
    残差与目标不参与生成，供体抽样按重数加权，其余来源与行级版同预算。
    """
    if budget is None:
        budget = EditBudget()
    joint_sets = [fs for fs in (joint_field_sets or []) if len(fs) >= 2]
    schema = registry.schema
    cumulative = np.cumsum(grouped.counts)
    supports = []
    for g in range(grouped.num_groups):
        current = registry.state_tuple(int(grouped.unique_ids[g]))
        paths: list[tuple[str, ...]] = []
        paths += _single_field_edits(current, schema, budget, rng)
        paths += _group_donor_copies(current, grouped, registry, cumulative, budget, rng)
        paths += _joint_edits(current, schema, joint_sets, budget, rng)
        paths += _explore_edits(current, schema, budget, rng)
        assert len(paths) <= budget.max_nonstay_paths()
        outcomes = tuple((registry.register_edit(int(grouped.unique_ids[g]), p),) for p in paths)
        mobility = (1.0,) * len(outcomes)
        supports.append(BlockSupport((g,), outcomes, mobility))
    return supports


@dataclass
class GroupKernelResult:
    """组核构造输出，blocks 与组一一对应，全部整代矩已按重数加权。"""

    blocks: list[BlockKernel]  # rows 里放组索引
    grouped: GroupedTable
    old_loss: float
    beta: float
    direction_gain: float
    interaction: float
    step: float
    expected_loss: float
    analytic_upper_bound: float
    expected_changed_rows: float
    max_gain_sum: float
    required_gain: float
    status: str


def grouped_residual(workload: Workload, grouped: GroupedTable) -> NDArray[np.float64]:
    """残差 e=y-sum_g c_g a(u_g)，与逐行求和精确同值仅浮点顺序不同。"""
    answers = grouped.counts.astype(np.float64) @ workload.features[grouped.unique_ids]
    return workload.target - answers


def build_group_kernel(
    workload: Workload,
    grouped: GroupedTable,
    supports: list[BlockSupport],
    stay_probability: float = 0.9,
    alpha: float = 0.5,
    damping: float = 1.0,
    max_expected_rows: float | None = None,
    numerical_tol: float = 1e-12,
) -> GroupKernelResult:
    """构造一轮的组核，每组算一份增益与概率，矩按重数加权。"""
    validate_partition(supports, grouped.num_groups)
    if np.any(grouped.unique_ids < 0) or np.any(grouped.unique_ids >= workload.num_states):
        raise ValueError("组状态编号必须落在贡献矩阵行数范围内")

    residual = grouped_residual(workload, grouped)
    old_loss = loss_from_residual(workload, residual)

    normalized: list[NormalizedBlock] = []
    deltas_list: list[NDArray[np.float64]] = []
    gains_list: list[NDArray[np.float64]] = []
    for spec in supports:
        nb = normalize_block(spec, grouped.unique_ids, workload.num_states, stay_probability)
        d = block_deltas(workload.features, nb.outcomes[0], nb.outcomes)
        g = block_gains(d, residual, workload.weights)
        normalized.append(nb)
        deltas_list.append(d)
        gains_list.append(g)

    tilt = calibrate_beta(
        gains_list,
        [nb.reference for nb in normalized],
        old_loss,
        alpha=alpha,
        numerical_tol=numerical_tol,
        multiplicities=grouped.counts,
    )

    if tilt.frozen:
        blocks = []
        for nb, d, g, P in zip(normalized, deltas_list, gains_list, tilt.probabilities):
            blocks.append(BlockKernel(nb.rows, nb.outcomes, nb.reference, d, g, np.zeros_like(P), P.copy()))
        return GroupKernelResult(
            blocks, grouped, old_loss, 0.0, 0.0, 0.0, 0.0,
            old_loss, old_loss, 0.0, tilt.max_gain_sum, tilt.required_gain,
            "no_positive_direction",
        )

    rates = rates_from_probabilities(tilt.probabilities)
    leave = block_leave_rates(rates)
    drifts = block_drifts(rates, deltas_list)
    interaction = cross_interaction(drifts, workload.weights, multiplicities=grouped.counts)
    h_star = analytic_step(float(leave.max()), tilt.direction_gain, interaction)
    unit_rows = expected_changed_rows_at_unit_step(
        rates, [nb.outcomes for nb in normalized], multiplicities=grouped.counts
    )
    if max_expected_rows is not None and unit_rows > 0:
        step = damping * min(h_star, max_expected_rows / unit_rows)
    else:
        step = damping * h_star

    blocks = []
    for nb, d, g, r in zip(normalized, deltas_list, gains_list, rates):
        probabilities = step * r
        probabilities[0] = 1.0 - float(probabilities[1:].sum())
        # 只做舍入清理，不是按目标的接受或修改
        if probabilities[0] < -1e-12:
            raise FloatingPointError("步长违反随机矩阵约束")
        if probabilities[0] < 0:
            probabilities[0] = 0.0
            probabilities /= probabilities.sum()
        blocks.append(BlockKernel(nb.rows, nb.outcomes, nb.reference, d, g, r, probabilities))

    expected_loss = old_loss - step * tilt.direction_gain + step * step * interaction
    upper = old_loss - step * tilt.direction_gain / 2
    return GroupKernelResult(
        blocks, grouped, old_loss, tilt.beta, tilt.direction_gain, interaction,
        step, expected_loss, upper, step * unit_rows,
        tilt.max_gain_sum, tilt.required_gain, "ok",
    )


def sample_group_next(
    result: GroupKernelResult,
    state_ids,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """按组核抽下一代，每组按多项分布抽各后继的数量。

    组内 c 行独立按同一单行核抽样，其计数向量精确服从多项分布，
    因此这里不是近似而是同分布的等价实现，行按组内升序切段分配，
    表是多重集，把哪个计数分给哪个具体行不影响任何查询答案。
    """
    s = np.asarray(state_ids)
    out = s.astype(np.int64, copy=True)
    grouped = result.grouped
    for block in result.blocks:
        g = block.rows[0]
        rows = grouped.row_lists[g]
        counts = rng.multinomial(len(rows), block.probabilities)
        pos = 0
        for j, c in enumerate(counts):
            if c == 0:
                continue
            u = int(block.outcomes[j][0])
            for i in rows[pos : pos + c]:
                out[i] = u
            pos += c
    return out


@dataclass
class RoundPlan:
    """提供器每轮返回的计划，组路径或行级回退路径二选一。"""

    mode: str  # "grouped" 或 "rows"
    workload: Workload
    supports: list[BlockSupport]
    grouped: GroupedTable | None = None


def make_grouped_provider(
    registry: StateRegistry,
    target,
    weights,
    menu_rng: np.random.Generator,
    budget: EditBudget | None = None,
    joint_field_sets: list[tuple[int, ...]] | None = None,
    pairing_budget: PairingBudget | None = None,
    pairing: bool = False,
):
    """组路径候选提供器，平时按组出菜单，冻结重试轮可回退行级配对。"""

    def provider(state_ids, round_index: int, frozen_streak: int = 0) -> RoundPlan:
        if pairing and frozen_streak > 0:
            table = tuples_from_ids(registry, state_ids)
            singles = generate_edit_supports(
                table, registry.schema, registry, menu_rng, budget, joint_field_sets
            )
            menu = build_paired_supports(
                state_ids, singles, registry, target, weights, menu_rng, pairing_budget
            )
            return RoundPlan("rows", menu.workload, menu.supports, None)
        grouped = group_state_ids(state_ids)
        supports = generate_group_edit_supports(
            grouped, registry, menu_rng, budget, joint_field_sets
        )
        workload = registry.build_workload(target, weights)
        return RoundPlan("grouped", workload, supports, grouped)

    return provider


def evolve_grouped(
    state_ids,
    num_rounds: int,
    rng: np.random.Generator,
    plan_provider,
    stay_probability: float = 0.9,
    alpha: float = 0.5,
    damping: float = 1.0,
    max_expected_rows: float | None = None,
    max_frozen_retries: int = 0,
) -> EvolveResult:
    """组路径多轮循环，表的真相始终是长 N 的状态编号向量。

    每轮重新分组，冻结与重试语义与行级 evolve 完全一致，
    行级回退轮走原 build_kernel 与 sample_next，数学主线不变。
    """
    if num_rounds < 1:
        raise ValueError("轮数必须为正")
    if max_frozen_retries < 0:
        raise ValueError("冻结重试次数不能为负")
    current = np.asarray(state_ids).astype(np.int64, copy=True)
    records: list[RoundRecord] = []
    frozen_streak = 0
    for k in range(num_rounds):
        plan = plan_provider(current, k, frozen_streak)
        if plan.mode == "rows":
            result = build_kernel(
                plan.workload, current, plan.supports,
                stay_probability, alpha, damping, max_expected_rows,
            )
        elif plan.mode == "grouped":
            result = build_group_kernel(
                plan.workload, plan.grouped, plan.supports,
                stay_probability, alpha, damping, max_expected_rows,
            )
        else:
            raise ValueError(f"未知计划模式 {plan.mode}")
        records.append(
            RoundRecord(
                k, result.old_loss, result.beta, result.direction_gain,
                result.interaction, result.step, result.expected_loss, result.status,
            )
        )
        if result.status == "no_positive_direction":
            frozen_streak += 1
            if frozen_streak > max_frozen_retries:
                return EvolveResult(current, records, "no_positive_direction")
            continue
        frozen_streak = 0
        if plan.mode == "rows":
            current = sample_next(result, current, rng)
        else:
            current = sample_group_next(result, current, rng)
    return EvolveResult(current, records, "round_limit")
