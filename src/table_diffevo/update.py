"""
向参考记录靠近一步

对每条当前记录 x_i 和它抽到的参考记录 z_i*，本轮做三件事（完整方案第 7 节）：

1. 可能保持不变（记录参与概率 ρ_t）
2. 可能复制参考记录的一部分属性块（属性块复制概率 η_t）
3. 以很小概率发生随机变异（变异概率 μ_t）

## 记录参与（7.2，ρ_t）

先抽 U_i ~ Bernoulli(ρ_t)：
- U_i = 0：整条记录保持不变
- U_i = 1：进入属性块复制过程

ρ_t 控制一轮中大约多少比例的记录有机会变化。

## 属性块复制（7.3，η_t）

对每个可修改属性块 g：
- 与参考记录相同 → 直接保持
- 与参考记录不同 → 以概率 η_t 复制参考记录该块，否则保持原值

可选的残差驱动扩散会在 Bernoulli 对数几率上连续加入实际单块转移的方向量：

    logit(p_copy) = logit(η_t) + strength * direction

方向为零或 strength=0 时精确保持 η_t；负方向概率降低但在有限数值下不被硬置零。
它改变随机转移核，不使用 ``direction > 0`` 资格筛选。

逐块靠近，不是一步整行复制。

## 变异（7.4，μ_t）

每条参与更新的记录最多变异一个块：
1. 以概率 μ_t 决定是否变异
2. 随机选一个块
3. 从该块的合法先验分布抽一个值

**玩具阶段简化（已与设计确认）：**
- 合法先验用 schema 合法值上的均匀分布（类别块）/ 范围内均匀整数（数值块）
- 暂不做合法性检查与回退（7.5），单字段值域天然合法，跨字段约束留待后续

## 职责边界

本模块只负责"给定当前记录和已对齐的参考记录，靠近一步"。
- donors 已按行对齐：donors.iloc[i] 是 current.iloc[i] 的参考记录
  （从候选池按抽样索引取 donor 的逻辑在上游，见 sampling.sample_donors）
- ρ、η、μ 随轮次的衰减调度由主循环负责，本函数只接收当前轮的标量值
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd
from table_diffevo.schema import Schema
from table_diffevo.directional_diffusion import (
    DEFAULT_DIRECTION_LOGIT_CLIP,
    tilted_copy_probabilities,
    validate_direction_logit_clip,
)


@dataclass(frozen=True)
class MutationEvent:
    """一次已经抽好的行—属性变异。"""

    row_index: int
    attribute: str
    value: Any


@dataclass(frozen=True)
class UpdateRandomPlan:
    """一次更新中与最终复制核无关的共同随机方案。

    ``initial_copy_mask`` 是现行独立 B 核抽出的初始复制开关。调用方可以
    原样应用它，也可以让另一个无门控复制核软调这些开关后，把最终开关传给
    :func:`apply_update_random_plan`。突变事件已经在这里抽好，因此更换复制核
    不会改变主随机流或复制后的突变。
    """

    participate: np.ndarray
    initial_copy_mask: np.ndarray
    mutation_events: tuple[MutationEvent, ...]


def sample_update_random_plan(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    rho: float = 0.1,
    eta: float = 0.5,
    mu: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    copy_direction_scores: Optional[np.ndarray] = None,
    copy_direction_strength: float = 0.0,
    direction_logit_clip: Optional[float] = DEFAULT_DIRECTION_LOGIT_CLIP,
    participate: Optional[np.ndarray] = None,
    copy_probability_bounds: Optional[tuple] = None,
) -> UpdateRandomPlan:
    """按现行顺序抽取参与行、初始复制开关和突变事件。

    本函数只抽取随机方案，不生成下一张表。随机数消费顺序严格保持为：参与
    行、逐属性复制开关、突变行、逐突变行的属性和值。

    participate 提供时跳过内部参与行抽签，直接使用外部掩码；调用方必须已在
    本函数原本的随机流槽位（即传入的 rng 的当前位置之前、供体均匀数之后）
    用同一 rng 以 ``rng.random(n_records) < rho`` 抽出该掩码，才能保证与
    历史路径逐位一致。后续复制开关与突变的随机消费不变（全长随机带）。

    copy_probability_bounds 提供 ``(lo, hi)`` 时，把倾斜后的逐 (行, 属性)
    复制概率硬夹到 ``[lo, hi]``（保证任何方向永远保留反方向概率，机制不会
    退化成确定性筛选）。要求 ``0 ≤ lo ≤ eta ≤ hi ≤ 1``（中性分数的概率
    必须留在带内，保持"分数为零 = 历史 η"不变量）。只允许与倾斜路径联用
    （copy_direction_scores 非 None 且 strength > 0），否则 fail-closed。
    """

    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho 必须在 [0, 1]，得到 {rho}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta 必须在 [0, 1]，得到 {eta}")
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu 必须在 [0, 1]，得到 {mu}")

    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )

    if (
        isinstance(copy_direction_strength, (bool, np.bool_))
        or not isinstance(
            copy_direction_strength,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(copy_direction_strength)
        or copy_direction_strength < 0.0
    ):
        raise ValueError(
            "copy_direction_strength 必须是非负有限数值，"
            f"得到 {copy_direction_strength!r}"
        )
    copy_direction_strength = float(copy_direction_strength)
    direction_logit_clip = validate_direction_logit_clip(
        direction_logit_clip
    )

    if copy_probability_bounds is not None:
        if (
            not isinstance(copy_probability_bounds, (tuple, list))
            or len(copy_probability_bounds) != 2
        ):
            raise ValueError(
                "copy_probability_bounds 必须是 (lo, hi) 二元组，"
                f"得到 {copy_probability_bounds!r}"
            )
        bounds_lo, bounds_hi = copy_probability_bounds
        for name, value in (("lo", bounds_lo), ("hi", bounds_hi)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value, (int, float, np.integer, np.floating)
                )
                or not np.isfinite(value)
            ):
                raise ValueError(
                    f"copy_probability_bounds 的 {name} 必须是有限数值，"
                    f"得到 {value!r}"
                )
        bounds_lo = float(bounds_lo)
        bounds_hi = float(bounds_hi)
        if not (0.0 <= bounds_lo <= eta <= bounds_hi <= 1.0):
            raise ValueError(
                "copy_probability_bounds 必须满足 0 ≤ lo ≤ eta ≤ hi ≤ 1"
                "（中性分数概率须留在带内），"
                f"得到 lo={bounds_lo}, hi={bounds_hi}, eta={eta}"
            )
        if copy_direction_scores is None or copy_direction_strength == 0.0:
            raise ValueError(
                "copy_probability_bounds 只允许与倾斜路径联用"
                "（需要 copy_direction_scores 且 strength > 0）"
            )
    else:
        bounds_lo = bounds_hi = None

    if rng is None:
        rng = np.random.default_rng()

    n_records = len(current)
    attr_names = schema.attribute_names()
    if copy_direction_scores is None:
        if copy_direction_strength != 0.0:
            raise ValueError(
                "copy_direction_strength 非零时必须提供 copy_direction_scores"
            )
        direction_scores = None
    else:
        direction_scores = np.asarray(copy_direction_scores)
        expected_shape = (n_records, len(attr_names))
        if direction_scores.shape != expected_shape:
            raise ValueError(
                "copy_direction_scores 必须是 shape (N, A) 的二维数组，"
                f"得到 {direction_scores.shape}，期望 {expected_shape}"
            )
        if direction_scores.dtype.kind not in "iuf":
            raise ValueError("copy_direction_scores 必须是数值数组")
        direction_scores = direction_scores.astype(float, copy=False)
        if not np.all(np.isfinite(direction_scores)):
            raise ValueError("copy_direction_scores 必须全部为有限数值")

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)

    # 保持历史随机数表达式和消费顺序，不按参与行数量缩短随机带。
    # 外部掩码提供时不再抽参与签（调用方已在原槽位抽过，见 docstring）。
    if participate is None:
        participate = rng.random(n_records) < rho
    else:
        participate = np.asarray(participate)
        if (
            participate.shape != (n_records,)
            or participate.dtype.kind != "b"
        ):
            raise ValueError(
                "participate 必须是与 current 行数一致的布尔向量，"
                f"得到 shape {participate.shape}"
            )
    initial_copy_mask = np.zeros(
        (n_records, len(attr_names)), dtype=bool
    )
    for attr_idx, attr in enumerate(attr_names):
        current_values = current_reset[attr].to_numpy()
        donor_values = donors_reset[attr].to_numpy()
        differ = current_values != donor_values
        if direction_scores is None or copy_direction_strength == 0.0:
            copy_roll = rng.random(n_records) < eta
        else:
            copy_probability = tilted_copy_probabilities(
                eta,
                direction_scores[:, attr_idx],
                copy_direction_strength,
                logit_clip=direction_logit_clip,
            )
            if bounds_lo is not None:
                # 硬夹带：倾斜再猛也保留反方向概率（见 docstring）
                copy_probability = np.clip(
                    copy_probability, bounds_lo, bounds_hi
                )
            copy_roll = rng.random(n_records) < copy_probability
        initial_copy_mask[:, attr_idx] = participate & differ & copy_roll

    mutate_mask = participate & (rng.random(n_records) < mu)
    mutation_events = []
    for row_index in np.nonzero(mutate_mask)[0]:
        attribute = _sample_mutation_block(schema, rng)
        value = _sample_legal_value(schema.get_block(attribute), rng)
        mutation_events.append(
            MutationEvent(int(row_index), attribute, value)
        )

    return UpdateRandomPlan(
        participate=participate,
        initial_copy_mask=initial_copy_mask,
        mutation_events=tuple(mutation_events),
    )


def apply_update_random_plan(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    plan: UpdateRandomPlan,
    *,
    final_copy_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """应用已抽好的复制开关与突变，生成唯一下一张表。"""

    if not isinstance(plan, UpdateRandomPlan):
        raise ValueError("plan 必须是 UpdateRandomPlan")
    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )

    n_records = len(current)
    attr_names = schema.attribute_names()
    expected_shape = (n_records, len(attr_names))
    participate = np.asarray(plan.participate)
    if participate.shape != (n_records,) or participate.dtype.kind != "b":
        raise ValueError(
            "plan.participate 必须是与 current 行数一致的布尔向量"
        )
    selected_mask = np.asarray(
        plan.initial_copy_mask
        if final_copy_mask is None
        else final_copy_mask
    )
    if selected_mask.shape != expected_shape or selected_mask.dtype.kind != "b":
        raise ValueError(
            "复制开关必须是 shape (N, A) 的布尔数组，"
            f"得到 shape {selected_mask.shape}"
        )
    if np.any(selected_mask & ~participate[:, None]):
        raise ValueError("最终复制开关不能启用未参与行")

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)
    next_table = current_reset.copy()
    for attr_idx, attr in enumerate(attr_names):
        copy_mask = selected_mask[:, attr_idx]
        if copy_mask.any():
            donor_values = donors_reset[attr].to_numpy()
            new_values = next_table[attr].to_numpy().copy()
            new_values[copy_mask] = donor_values[copy_mask]
            next_table[attr] = new_values

    _apply_planned_mutations_in_place(next_table, attr_names, plan)
    return next_table


def apply_planned_mutations(
    copy_table: pd.DataFrame,
    schema: Schema,
    plan: UpdateRandomPlan,
) -> pd.DataFrame:
    """在已经物化的复制表上应用同一份预抽突变。"""

    if not isinstance(plan, UpdateRandomPlan):
        raise ValueError("plan 必须是 UpdateRandomPlan")
    next_table = copy_table.reset_index(drop=True).copy()
    _apply_planned_mutations_in_place(
        next_table, schema.attribute_names(), plan
    )
    return next_table


def _apply_planned_mutations_in_place(
    next_table: pd.DataFrame,
    attr_names: list[str],
    plan: UpdateRandomPlan,
) -> None:
    """验证并原位应用突变；只由已复制输入的公开包装函数调用。"""

    n_records = len(next_table)
    for event in plan.mutation_events:
        if not isinstance(event, MutationEvent):
            raise ValueError("plan.mutation_events 必须只包含 MutationEvent")
        if not 0 <= event.row_index < n_records:
            raise ValueError("变异事件的行号超出 current 范围")
        if event.attribute not in attr_names:
            raise ValueError("变异事件包含 schema 之外的属性")
        next_table.at[event.row_index, event.attribute] = event.value


def evolve_step(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    rho: float = 0.1,
    eta: float = 0.5,
    mu: float = 0.01,
    rng: Optional[np.random.Generator] = None,
    copy_direction_scores: Optional[np.ndarray] = None,
    copy_direction_strength: float = 0.0,
    direction_logit_clip: Optional[float] = DEFAULT_DIRECTION_LOGIT_CLIP,
    return_diagnostics: bool = False,
    participate: Optional[np.ndarray] = None,
    copy_probability_bounds: Optional[tuple] = None,
    value_gains: Optional[Any] = None,
    value_domains: Optional[dict] = None,
    value_guidance_strength: float = 0.0,
    value_guidance_drop_donor: bool = False,
) -> Any:
    """
    全表同步向参考记录靠近一步，生成下一代 S_{t+1}。

    Parameters
    ----------
    current : pd.DataFrame, shape (N, n_attributes)
        当前记录表 S_t
    donors : pd.DataFrame, shape (N, n_attributes)
        已按行对齐的参考记录：donors.iloc[i] 是 current.iloc[i] 的参考记录
    schema : Schema
        属性 schema 定义
    rho : float, default 0.1
        记录参与概率 ρ_t，一轮中大约多少比例的记录有机会变化
    eta : float, default 0.5
        属性块复制概率 η_t，不同的块以此概率复制参考记录
    mu : float, default 0.01
        变异概率 μ_t，参与更新的记录以此概率变异一个块
    rng : np.random.Generator or None
        随机数生成器。推荐显式传入 np.random.default_rng(seed) 保证复现
    copy_direction_scores : np.ndarray or None, shape (N, A), default None
        每条记录、每个属性块的实际单块复制方向量。None 表示使用历史固定 η。
        提供时只连续倾斜复制概率，不执行正负阈值筛选。
    copy_direction_strength : float, default 0.0
        非负有限方向强度。0 精确退化到历史固定 η 路径；正值越大，复制概率对
        方向量越敏感。有限强度下负方向仍保留非零复制概率。
    direction_logit_clip : float or None, default 30
        残差方向 Bernoulli logit 的显式数值护栏。默认30保持历史语义；None
        显式关闭护栏。
    return_diagnostics : bool, default False
        True 时返回 ``(next_table, diagnostics)``，其中只含不参与决策的公开
        转移工作量。默认 False 保持历史 DataFrame 返回值。
    participate : np.ndarray or None, default None
        预抽好的参与行掩码（bool, shape (N,)）。提供时跳过内部参与签，
        随机流约束见 sample_update_random_plan 文档。None 保持历史行为。
    copy_probability_bounds : tuple or None, default None
        (lo, hi) 复制概率硬夹带，只允许与倾斜路径联用，语义与校验见
        sample_update_random_plan 文档。None 保持历史行为。
    value_gains : dict, callable or None, default None
        属性 → (N, V_a) 逐格增益矩阵（ValueGainComputer.compute 的返回），
        或按需回调 ``fn(rows) -> dict{attr: (K, V_a)}``（只算参与行子集，
        生产路径；参与行为空时不调用）。非 None 时整轮改走值引导核
        （sample_value_guided_plan），不再抽复制开关与突变事件；与倾斜
        路径（copy_direction_scores / copy_probability_bounds）互斥，
        fail-closed。
    value_domains : dict or None, default None
        属性 → 候选值列表，与 value_gains 配套提供。
    value_guidance_strength : float, default 0.0
        值引导强度 λ_t。0 时新核仍生效（分布退化为 base），臂内随机流自洽。
    value_guidance_drop_donor : bool, default False
        True 时 base 分布去掉供体分量（η 质量并给自值）——供体价值对照臂。

    Returns
    -------
    pd.DataFrame, shape (N, n_attributes)
        下一代记录表 S_{t+1}（新对象，不修改输入）

    Raises
    ------
    ValueError
        current 与 donors 形状不一致、概率参数越界

    Notes
    -----
    **复现性（铁律 5）：** 使用固定种子的 rng 保证结果可复现。

    **全表同步（铁律）：** 所有记录基于同一份输入同步生成下一状态。

    Examples
    --------
    >>> from table_diffevo.sampling import compute_sampling_probs, sample_donors
    >>> from table_diffevo.distance import pairwise_block_distance
    >>> from table_diffevo.schema import load_schema
    >>>
    >>> schema = load_schema("configs/schema.yaml")
    >>> probs = compute_sampling_probs(fitness, distances)
    >>> rng = np.random.default_rng(42)
    >>> donor_idx = sample_donors(probs, rng)
    >>> donors = current.iloc[donor_idx].reset_index(drop=True)
    >>> next_table = evolve_step(current, donors, schema, rng=rng)
    """
    if not isinstance(return_diagnostics, (bool, np.bool_)):
        raise ValueError("return_diagnostics 必须是布尔值")
    if (value_gains is None) != (value_domains is None):
        raise ValueError(
            "value_gains 与 value_domains 必须同时提供或同时缺省"
        )
    if value_gains is not None:
        if copy_direction_scores is not None or copy_probability_bounds is not None:
            raise ValueError(
                "值引导核与倾斜路径（copy_direction_scores / "
                "copy_probability_bounds）互斥，fail-closed"
            )
        plan_vg = sample_value_guided_plan(
            current,
            donors,
            schema,
            rho=rho,
            eta=eta,
            mu=mu,
            rng=rng,
            value_gains=value_gains,
            value_domains=value_domains,
            guidance_strength=value_guidance_strength,
            drop_donor=value_guidance_drop_donor,
            participate=participate,
        )
        next_table = apply_value_guided_plan(current, schema, plan_vg)
        if return_diagnostics:
            return next_table, {
                "participating_rows": int(plan_vg.participate.sum()),
                "mutated_rows": 0,
                "value_guided_changed_cells": int(plan_vg.changed_mask.sum()),
            }
        return next_table
    if value_guidance_strength != 0.0 or value_guidance_drop_donor:
        raise ValueError(
            "value_guidance_strength/drop_donor 只在提供 value_gains 时有效"
        )
    plan = sample_update_random_plan(
        current,
        donors,
        schema,
        rho=rho,
        eta=eta,
        mu=mu,
        rng=rng,
        copy_direction_scores=copy_direction_scores,
        copy_direction_strength=copy_direction_strength,
        direction_logit_clip=direction_logit_clip,
        participate=participate,
        copy_probability_bounds=copy_probability_bounds,
    )
    next_table = apply_update_random_plan(current, donors, schema, plan)

    if return_diagnostics:
        return next_table, {
            "participating_rows": int(plan.participate.sum()),
            "mutated_rows": int(len(plan.mutation_events)),
        }
    return next_table


def value_guided_probabilities(
    cur_idx: np.ndarray,
    don_idx: np.ndarray,
    gains: np.ndarray,
    *,
    eta: float,
    mu_cell: float,
    guidance_strength: float,
    drop_donor: bool = False,
) -> np.ndarray:
    """构造值引导核的逐行值分布 (N, V)，行和恒为 1。

    base = (1−μc)·[η·δ_donor + (1−η)·δ_self] + μc·uniform；
    drop_donor 时 η 质量并给 δ_self。λ>0 时按
    p ∝ base·exp(λ·(gain − 行内最大)) 重加权（数值稳定），λ=0 逐位返回 base。

    独立成纯函数：生产采样与等价测试共用同一实现，λ=0 的"分布退化为
    base"断言无需跑采样。
    """
    n_records, v_count = gains.shape
    base = np.full((n_records, v_count), mu_cell / v_count, dtype=float)
    rows = np.arange(n_records)
    if drop_donor:
        base[rows, cur_idx] += 1.0 - mu_cell
    else:
        base[rows, cur_idx] += (1.0 - mu_cell) * (1.0 - eta)
        base[rows, don_idx] += (1.0 - mu_cell) * eta
    if guidance_strength > 0.0:
        shifted = guidance_strength * (
            gains - gains.max(axis=1, keepdims=True)
        )
        p = base * np.exp(shifted)
        total = p.sum(axis=1, keepdims=True)
        if not np.all(total > 0.0):
            raise ValueError(
                f"存在全零概率行（mu_cell={mu_cell}, λ={guidance_strength}）"
            )
        return p / total
    return base


@dataclass(frozen=True)
class ValueGuidedPlan:
    """残差引导值分布核（value guidance kernel）的一次抽样结果。

    new_columns 是全长新列（非参与行保持原值），apply 时整列写回。
    changed_mask (N, A) 只作诊断，不参与决策。
    """

    participate: np.ndarray
    new_columns: dict[str, np.ndarray]
    changed_mask: np.ndarray


def sample_value_guided_plan(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    *,
    rho: float,
    eta: float,
    mu: float,
    rng: np.random.Generator,
    value_gains,
    value_domains: dict,
    guidance_strength: float = 0.0,
    drop_donor: bool = False,
    participate: Optional[np.ndarray] = None,
) -> ValueGuidedPlan:
    """残差引导的逐格值分布抽样：把"定新值"从 η 硬币换成全值域抽签。

    对每条参与行 i、每个属性 a，新值从整个值域按

        p(v) ∝ base(v) · exp(λ · gain_a[i, v])

    抽取，其中 λ = guidance_strength，gain 来自
    :class:`~table_diffevo.vectorized_eval.ValueGainComputer`（欠账为正）。

    base 是旧核语义的分布化改写（λ=0 时与旧核每格边缘分布一致）：

        base = (1−μc)·[η·δ_donor + (1−η)·δ_self] + μc·uniform,  μc = μ/A

    - μc = μ/A 对齐旧核"每行变异率 μ、变异行随机挑 1 个属性"的每格变异率；
    - drop_donor=True 时 η 质量并给 δ_self（供体价值对照臂），
      即 base = (1−μc)·δ_self + μc·uniform；
    - 变异不再单独抽事件：均匀分量已并入 base，这是新核语义。

    value_gains 两态（参与行切片优化）：
    - dict：属性 → 全表 (N, V_a) 增益矩阵，内部按参与行切片；
    - callable：``fn(rows) -> dict{attr: (K, V_a)}`` 只算参与行子集
      （生产路径——gain 逐行独立，参与行为空时根本不调用）。

    随机流合同：participate 槽位与旧核一致（None 时 ``rng.random(N) < rho``，
    外部传入时须由调用方在同槽位抽出——lottery 管线复用）；之后每属性消费
    一条参与行长度 (K,) 的均匀带用于值抽签；K=0 时整轮不再消费任何随机数。
    新核与旧核的随机流不同——整臂自洽，不承诺与旧核逐位对齐。

    fail-closed：表/供体中出现值域外的值、gains 形状不符、λ 非法均抛错。
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho 必须在 [0, 1]，得到 {rho}")
    if not (0.0 <= eta <= 1.0):
        raise ValueError(f"eta 必须在 [0, 1]，得到 {eta}")
    if not (0.0 <= mu <= 1.0):
        raise ValueError(f"mu 必须在 [0, 1]，得到 {mu}")
    if (
        isinstance(guidance_strength, (bool, np.bool_))
        or not isinstance(
            guidance_strength, (int, float, np.integer, np.floating)
        )
        or not np.isfinite(guidance_strength)
        or guidance_strength < 0.0
    ):
        raise ValueError(
            f"guidance_strength 必须是非负有限数值，得到 {guidance_strength!r}"
        )
    guidance_strength = float(guidance_strength)
    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )
    if rng is None:
        raise ValueError("value guidance 核必须显式传入 rng")

    n_records = len(current)
    attr_names = schema.attribute_names()
    n_attrs = len(attr_names)

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)

    # 参与行：槽位与旧核一致
    if participate is None:
        participate = rng.random(n_records) < rho
    else:
        participate = np.asarray(participate)
        if participate.shape != (n_records,) or participate.dtype.kind != "b":
            raise ValueError(
                "participate 必须是与 current 行数一致的布尔向量，"
                f"得到 shape {participate.shape}"
            )

    mu_cell = mu / n_attrs  # 每格变异率（对齐旧核 μ×1/A）

    rows = np.flatnonzero(participate)
    k_rows = len(rows)
    changed_mask = np.zeros((n_records, n_attrs), dtype=bool)

    # 参与行为空：零成本短路（不取 gain、不消费属性随机带）
    if k_rows == 0:
        return ValueGuidedPlan(
            participate=participate,
            new_columns={
                attr: current_reset[attr].to_numpy().copy()
                for attr in attr_names
            },
            changed_mask=changed_mask,
        )

    # gain 两态：callable 只算参与行子集；dict 为全表矩阵、内部切片
    if callable(value_gains):
        gains_by_attr = value_gains(rows)
        expected_rows = k_rows
    else:
        gains_by_attr = value_gains
        expected_rows = n_records

    new_columns: dict[str, np.ndarray] = {}

    for attr_idx, attr in enumerate(attr_names):
        if attr not in value_domains or attr not in gains_by_attr:
            raise ValueError(f"value_gains/value_domains 缺少属性 {attr!r}")
        domain = list(value_domains[attr])
        v_count = len(domain)
        if v_count < 1:
            raise ValueError(f"属性 {attr!r} 的值域为空")
        gains = np.asarray(gains_by_attr[attr], dtype=float)
        if gains.shape != (expected_rows, v_count):
            raise ValueError(
                f"属性 {attr!r} 的 gains 形状 {gains.shape} 不符，"
                f"期望 {(expected_rows, v_count)}"
            )
        if not np.all(np.isfinite(gains)):
            raise ValueError(f"属性 {attr!r} 的 gains 含非有限值")
        gains_sub = gains if expected_rows == k_rows else gains[rows]

        # 值→下标：排序 + searchsorted 全向量化（fail-closed：映射后回读
        # 必须与原值逐位相等，值域外的表值在这里报错）；只映射参与行
        cur_vals = current_reset[attr].to_numpy()
        cur_sub = cur_vals[rows]
        don_sub = donors_reset[attr].to_numpy()[rows]
        domain_arr = np.asarray(domain)
        sorter = np.argsort(domain_arr, kind="stable")
        sorted_domain = domain_arr[sorter]

        def _to_idx(vals, which):
            slot = np.searchsorted(sorted_domain, vals)
            np.clip(slot, 0, v_count - 1, out=slot)
            idx = sorter[slot]
            if not np.array_equal(domain_arr[idx], vals):
                bad = vals[domain_arr[idx] != vals][:3]
                raise ValueError(
                    f"属性 {attr!r} 的{which}表值 {bad!r} 不在值域 "
                    f"{domain!r} 内"
                )
            return idx

        cur_idx = _to_idx(cur_sub, "当前")
        don_idx = _to_idx(don_sub, "供体")

        # base 分布 + guidance 重加权（纯函数，行和恒 1），只算参与行
        p = value_guided_probabilities(
            cur_idx,
            don_idx,
            gains_sub,
            eta=eta,
            mu_cell=mu_cell,
            guidance_strength=guidance_strength,
            drop_donor=drop_donor,
        )
        cum = np.cumsum(p, axis=1)
        total = cum[:, -1]

        # 参与行长度随机带，逐行 CDF 抽签
        u = rng.random(k_rows) * total
        new_idx = (cum < u[:, None]).sum(axis=1)
        np.clip(new_idx, 0, v_count - 1, out=new_idx)

        new_vals = cur_vals.copy()
        new_vals[rows] = domain_arr[new_idx]
        new_columns[attr] = new_vals
        changed_mask[rows, attr_idx] = new_vals[rows] != cur_sub

    return ValueGuidedPlan(
        participate=participate,
        new_columns=new_columns,
        changed_mask=changed_mask,
    )


def apply_value_guided_plan(
    current: pd.DataFrame,
    schema: Schema,
    plan: ValueGuidedPlan,
) -> pd.DataFrame:
    """把值引导抽样结果写成下一张表（新对象，不修改输入）。"""
    if not isinstance(plan, ValueGuidedPlan):
        raise ValueError("plan 必须是 ValueGuidedPlan")
    n_records = len(current)
    attr_names = schema.attribute_names()
    participate = np.asarray(plan.participate)
    if participate.shape != (n_records,) or participate.dtype.kind != "b":
        raise ValueError("plan.participate 必须是与 current 行数一致的布尔向量")
    next_table = current.reset_index(drop=True).copy()
    for attr in attr_names:
        if attr not in plan.new_columns:
            raise ValueError(f"plan.new_columns 缺少属性 {attr!r}")
        col = np.asarray(plan.new_columns[attr])
        if col.shape != (n_records,):
            raise ValueError(
                f"plan.new_columns[{attr!r}] 形状 {col.shape} 不符"
            )
        next_table[attr] = col
    return next_table


def _sample_mutation_block(schema: Schema, rng: np.random.Generator) -> str:
    """随机选择一个可修改属性块的名字。"""
    names = schema.attribute_names()
    idx = rng.integers(0, len(names))
    return names[idx]


def _sample_legal_value(block, rng: np.random.Generator):
    """
    从块的合法先验分布抽一个值（玩具阶段：均匀分布）。

    - 类别块：合法取值集合上的均匀抽样
    - 数值块：[min, max] 范围内的均匀整数（含端点）
    """
    if block.is_numeric():
        low, high = block.range
        # 范围内均匀整数，含端点
        return int(rng.integers(int(low), int(high) + 1))
    else:
        idx = rng.integers(0, len(block.values))
        return block.values[idx]
