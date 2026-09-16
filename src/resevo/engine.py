"""引擎组装，核构造、一次性抽样与多轮循环。

一轮的完整流程，取当前残差，构造合法支持与参考分布，算增量与增益，
熵校准得方向分布 P，速率化并算整代矩与解析步长 h，得到最终核 K。
方向分布 P 不是抽样分布，最终抽样只认 K，每块只抽一次，
不抽后变异，不按新表损失接受拒绝回退，非法状态直接报实现错误。
期望损失满足精确恒等式 E[L']=L-hD+h^2C，解析实数上界为 L-hD/2。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidates import (
    BlockSupport,
    NormalizedBlock,
    full_single_row_supports,
    normalize_block,
    validate_partition,
)
from .gain import block_deltas, block_gains
from .state import Workload, check_state_ids, loss_from_residual, table_residual
from .stepsize import (
    analytic_step,
    block_drifts,
    block_leave_rates,
    cross_interaction,
    expected_changed_rows_at_unit_step,
    rates_from_probabilities,
)
from .tilt import calibrate_beta


@dataclass
class BlockKernel:
    """一个块的最终转移核，outcomes 索引 0 恒为源元组。"""

    rows: tuple[int, ...]
    outcomes: tuple[tuple[int, ...], ...]
    reference: NDArray[np.float64]
    deltas: NDArray[np.float64]
    gains: NDArray[np.float64]
    rates: NDArray[np.float64]
    probabilities: NDArray[np.float64]


@dataclass
class KernelResult:
    """一轮核构造的全部输出，行级边缘只是解释输出，不是抽样输入。"""

    blocks: list[BlockKernel]
    row_marginals: NDArray[np.float64]  # N×M
    old_loss: float
    beta: float
    direction_gain: float  # D
    interaction: float  # C
    step: float  # h
    expected_loss: float  # L-hD+h^2C
    analytic_upper_bound: float  # L-hD/2，实数意义上界
    expected_changed_rows: float  # h*H
    max_gain_sum: float  # M
    required_gain: float  # delta=alpha*M
    status: str  # "ok" 或 "no_positive_direction"


def build_kernel(
    workload: Workload,
    state_ids,
    supports: list[BlockSupport] | None = None,
    stay_probability: float = 0.9,
    alpha: float = 0.5,
    damping: float = 1.0,
    max_expected_rows: float | None = None,
    numerical_tol: float = 1e-12,
) -> KernelResult:
    """构造一轮的最终核，所有块共用同一个旧残差，不抽样任何表。"""
    s = check_state_ids(workload, state_ids)
    n = len(s)
    m = workload.num_states
    if supports is None:
        # 第一版决策，候选集直接用全集单行块
        supports = full_single_row_supports(n, m)
    validate_partition(supports, n)

    residual = table_residual(workload, s)
    old_loss = loss_from_residual(workload, residual)

    normalized: list[NormalizedBlock] = []
    deltas_list: list[NDArray[np.float64]] = []
    gains_list: list[NDArray[np.float64]] = []
    for spec in supports:
        nb = normalize_block(spec, s, m, stay_probability)
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
    )

    if tilt.frozen:
        # 本次支持没有正的一阶方向，整体保持不动并明确报告停滞
        blocks = []
        for nb, d, g, P in zip(normalized, deltas_list, gains_list, tilt.probabilities):
            blocks.append(BlockKernel(nb.rows, nb.outcomes, nb.reference, d, g, np.zeros_like(P), P.copy()))
        marginals = _row_marginals(blocks, n, m)
        return KernelResult(
            blocks, marginals, old_loss, 0.0, 0.0, 0.0, 0.0,
            old_loss, old_loss, 0.0, tilt.max_gain_sum, tilt.required_gain,
            "no_positive_direction",
        )

    rates = rates_from_probabilities(tilt.probabilities)
    leave = block_leave_rates(rates)
    drifts = block_drifts(rates, deltas_list)
    interaction = cross_interaction(drifts, workload.weights)
    h_star = analytic_step(float(leave.max()), tilt.direction_gain, interaction)
    unit_rows = expected_changed_rows_at_unit_step(rates, [nb.outcomes for nb in normalized])
    if max_expected_rows is not None and unit_rows > 0:
        # 期望改行数预算按文档 h=gamma*min(h*, B_max/H)，在抽样前决定
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

    marginals = _row_marginals(blocks, n, m)
    expected_loss = old_loss - step * tilt.direction_gain + step * step * interaction
    upper = old_loss - step * tilt.direction_gain / 2
    return KernelResult(
        blocks, marginals, old_loss, tilt.beta, tilt.direction_gain, interaction,
        step, expected_loss, upper, step * unit_rows,
        tilt.max_gain_sum, tilt.required_gain, "ok",
    )


def _row_marginals(blocks: list[BlockKernel], n: int, m: int) -> NDArray[np.float64]:
    """行级边缘 K_i(x)，只用于解释与单行块抽样等价核对。"""
    marginals = np.zeros((n, m), dtype=np.float64)
    for block in blocks:
        for outcome, p in zip(block.outcomes, block.probabilities):
            for i, x in zip(block.rows, outcome):
                marginals[i, x] += p
    return marginals


def two_stage_view(block: BlockKernel) -> tuple[float, NDArray[np.float64]]:
    """两阶段分解，rho=离开概率，T=离开后的条件分布，b=0 时只保持。

    这是同一分布的另一种写法，不允许再乘一次按适应度选行的参与率。
    """
    rho = float(block.probabilities[1:].sum())
    leave = float(block.rates.sum())
    if leave == 0.0:
        return 0.0, np.zeros_like(block.rates)
    return rho, block.rates / leave


def attribute_marginal(
    result: KernelResult,
    row: int,
    attribute_of_state,
    value,
) -> float:
    """属性边缘 Pr(A'=a)=sum_{u:A(u)=a} K_row(u)，从同一个核取边缘导出。"""
    total = 0.0
    for x in range(result.row_marginals.shape[1]):
        if attribute_of_state(x) == value:
            total += float(result.row_marginals[row, x])
    return total


def sample_next(
    result: KernelResult,
    state_ids,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """按最终核抽下一代，每块只抽一次，把整个后继元组写入新表。

    多行块绝不能按行级边缘独立抽样，那会破坏联合分布。
    """
    s = np.asarray(state_ids)
    out = s.astype(np.int64, copy=True)
    for block in result.blocks:
        j = int(rng.choice(len(block.outcomes), p=block.probabilities))
        outcome = block.outcomes[j]
        out[list(block.rows)] = outcome
    return out


@dataclass
class RoundRecord:
    """一轮的日志，全部量在抽样前算好。"""

    round_index: int
    old_loss: float
    beta: float
    direction_gain: float
    interaction: float
    step: float
    expected_loss: float
    status: str


@dataclass
class EvolveResult:
    """多轮演化输出。"""

    state_ids: NDArray[np.int64]
    records: list[RoundRecord]
    stop_reason: str  # "no_positive_direction" 或 "round_limit"


def evolve(
    workload: Workload | None,
    state_ids,
    num_rounds: int,
    rng: np.random.Generator,
    supports: list[BlockSupport] | None = None,
    stay_probability: float = 0.9,
    alpha: float = 0.5,
    damping: float = 1.0,
    max_expected_rows: float | None = None,
    support_provider=None,
    max_frozen_retries: int = 0,
) -> EvolveResult:
    """多轮循环，轮内共享旧残差，轮间才更新残差。

    两种候选模式，固定 supports 一份用到底，或者传 support_provider，
    每轮把当前表轮号与连续冻结次数交给提供器，返回本轮的负载与菜单，
    提供器模式下负载允许只增不改地扩状态，旧编号贡献行必须保持前缀一致。
    冻结不再必然立即停，随机菜单本轮无正增益不是全局证书，
    连续冻结超过 max_frozen_retries 次才停，0 保持旧行为冻结即停。
    最简占位，停止只区分冻结与轮数上限，数值不确定等状态留待后续版本。
    """
    if num_rounds < 1:
        raise ValueError("轮数必须为正")
    if support_provider is not None and supports is not None:
        raise ValueError("固定候选与候选提供器只能二选一")
    if support_provider is None and workload is None:
        raise ValueError("固定候选模式必须给定负载")
    if max_frozen_retries < 0:
        raise ValueError("冻结重试次数不能为负")
    if support_provider is None:
        current = check_state_ids(workload, state_ids)
    else:
        current = np.asarray(state_ids).astype(np.int64, copy=True)
    records: list[RoundRecord] = []
    frozen_streak = 0
    for k in range(num_rounds):
        if support_provider is not None:
            round_workload, round_supports = support_provider(current, k, frozen_streak)
        else:
            round_workload, round_supports = workload, supports
        result = build_kernel(
            round_workload, current, round_supports,
            stay_probability, alpha, damping, max_expected_rows,
        )
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
            continue  # 冻结轮整表保持不动，提供器下一轮刷新菜单再试
        frozen_streak = 0
        current = sample_next(result, current, rng)
    return EvolveResult(current, records, "round_limit")
