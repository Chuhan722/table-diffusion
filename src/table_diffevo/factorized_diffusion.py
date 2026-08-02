"""至多低阶因子的联合 mask 能量与随机扫描 Gibbs 扩散。

固定 recipient、donor 和残差场后，每个合取查询只依赖它涉及的活跃复制 bit。
本模块把这些局部布尔函数聚合成稀疏因子，并用随机坐标 Gibbs 更新近似采样联合
mask 分布。它不执行正收益门槛、方向 argmax、top-k 或整代 proposal 接受检查。
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import tilted_copy_probabilities
from table_diffevo.joint_diffusion import enumerate_copy_masks
from table_diffevo.queries import eval_condition
from table_diffevo.schema import Schema


@dataclass(frozen=True)
class MaskEnergyFactor:
    """一个只依赖少量活跃 mask bit 的中心化能量表。"""

    scope: Tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class SparseMaskEnergy:
    """一条 recipient-donor 对的稀疏联合 mask 能量。"""

    active_attribute_indices: np.ndarray
    active_attributes: Tuple[str, ...]
    factors: Tuple[MaskEnergyFactor, ...]
    factors_by_variable: Tuple[Tuple[int, ...], ...]
    n_queries: int
    n_active_queries: int
    max_active_query_order: int

    @property
    def n_active_attributes(self) -> int:
        return len(self.active_attributes)


def _validate_nonnegative_integer(value: int, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} 必须是非负整数，得到 {value!r}")
    return int(value)


def _validate_open_probability(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"{name} 必须是 (0, 1) 内的有限数值，得到 {value!r}")
    return float(value)


def _validate_strength(value: float) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"strength 必须是非负有限数值，得到 {value!r}")
    return float(value)


def _validate_numeric_vector(
    values: np.ndarray,
    expected_length: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (expected_length,):
        raise ValueError(
            f"{name} 必须是长度 {expected_length} 的一维数组，"
            f"得到 shape {raw.shape}"
        )
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} 必须是有限数值数组")
    result = raw.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是有限数值数组")
    return result


def _validate_mask(mask: np.ndarray, width: int, name: str = "mask") -> np.ndarray:
    raw = np.asarray(mask)
    if raw.shape != (width,):
        raise ValueError(
            f"{name} 必须是长度 {width} 的一维 0/1 数组，得到 {raw.shape}"
        )
    if raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} 必须只包含 0/1 或布尔值")
    if not np.all(np.isfinite(raw)) or np.any((raw != 0) & (raw != 1)):
        raise ValueError(f"{name} 必须只包含 0/1 或布尔值")
    return raw.astype(bool, copy=False)


def _validate_masks(masks: np.ndarray, width: int) -> np.ndarray:
    raw = np.asarray(masks)
    if raw.ndim != 2 or raw.shape[1] != width:
        raise ValueError(
            f"masks 必须是 shape (n, {width}) 的二维 0/1 数组，"
            f"得到 {raw.shape}"
        )
    if raw.dtype.kind not in "biuf":
        raise ValueError("masks 必须只包含 0/1 或布尔值")
    if not np.all(np.isfinite(raw)) or np.any((raw != 0) & (raw != 1)):
        raise ValueError("masks 必须只包含 0/1 或布尔值")
    return raw.astype(bool, copy=False)


def _validate_logit_clip(logit_clip: Optional[float]) -> Optional[float]:
    if logit_clip is None:
        return None
    if (
        isinstance(logit_clip, (bool, np.bool_))
        or not isinstance(logit_clip, (int, float, np.integer, np.floating))
        or not np.isfinite(logit_clip)
        or logit_clip <= 0.0
    ):
        raise ValueError(
            f"logit_clip 必须是正有限数值或 None，得到 {logit_clip!r}"
        )
    return float(logit_clip)


def _stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def build_sparse_mask_energy(
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    residual: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    max_factor_order: int = 3,
) -> SparseMaskEnergy:
    """按查询构造一条 recipient-donor 对的中心化稀疏 mask 能量。

    ``recipient`` 与 ``donor`` 都必须恰有一行。对每个查询，只枚举该查询涉及且
    donor 值不同的局部属性；查询在空局部 mask 下的贡献被减去，因此模型能量
    满足 ``U(empty)=0``，并等于完整 hybrid 相对 recipient 的方向势能。
    """
    if len(recipient) != 1 or len(donor) != 1:
        raise ValueError("recipient 和 donor 都必须恰好包含一行")
    maximum_order = _validate_nonnegative_integer(
        max_factor_order, "max_factor_order"
    )
    if maximum_order > 8:
        raise ValueError("max_factor_order 不得超过绝对护栏 8")

    attr_names = schema.attribute_names()
    missing_recipient = [
        name for name in attr_names if name not in recipient.columns
    ]
    missing_donor = [name for name in attr_names if name not in donor.columns]
    if missing_recipient or missing_donor:
        raise ValueError(
            "recipient/donor 缺少 schema 属性列："
            f"recipient={missing_recipient}, donor={missing_donor}"
        )
    residual_values = _validate_numeric_vector(
        residual, len(queries), "residual"
    )
    if weights is None:
        weight_values = np.ones(len(queries), dtype=float)
    else:
        weight_values = _validate_numeric_vector(
            weights, len(queries), "weights"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_residual = residual_values * weight_values
    if not np.all(np.isfinite(weighted_residual)):
        raise ValueError("residual * weights 超出 float64 可表示范围")

    recipient_reset = recipient.reset_index(drop=True)
    donor_reset = donor.reset_index(drop=True)
    recipient_values = recipient_reset.loc[0, attr_names].to_numpy()
    donor_values = donor_reset.loc[0, attr_names].to_numpy()
    active_attribute_indices = np.flatnonzero(
        recipient_values != donor_values
    ).astype(np.intp, copy=False)
    active_attributes = tuple(
        attr_names[index] for index in active_attribute_indices
    )
    active_position = {
        attr: position for position, attr in enumerate(active_attributes)
    }
    schema_attributes = set(attr_names)

    pair = pd.concat(
        [recipient_reset.iloc[[0]], donor_reset.iloc[[0]]],
        ignore_index=True,
    )
    aggregated = {}
    n_active_queries = 0
    max_active_query_order = 0
    for query_index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"queries[{query_index}] 必须是字典")
        conditions = query.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError(
                f"queries[{query_index}].conditions 必须是列表"
            )
        query_attributes = []
        for condition_index, condition in enumerate(conditions):
            if not isinstance(condition, dict):
                raise ValueError(
                    f"queries[{query_index}].conditions[{condition_index}] "
                    "必须是字典"
                )
            attr = condition.get("attribute")
            if attr not in schema_attributes:
                raise ValueError(
                    f"queries[{query_index}] 包含未知属性 {attr!r}"
                )
            if attr not in query_attributes:
                query_attributes.append(attr)
        scope = tuple(sorted(
            active_position[attr]
            for attr in query_attributes
            if attr in active_position
        ))
        order = len(scope)
        if order == 0:
            continue
        if order > maximum_order:
            raise ValueError(
                f"queries[{query_index}] 的活跃因子阶数 {order} "
                f"超过 max_factor_order={maximum_order}"
            )
        n_active_queries += 1
        max_active_query_order = max(max_active_query_order, order)

        local_masks = enumerate_copy_masks(
            order, max_active_attributes=maximum_order
        )
        local_scope_position = {
            active_index: position
            for position, active_index in enumerate(scope)
        }
        query_mask = np.ones(len(local_masks), dtype=bool)
        for condition in conditions:
            attr = condition["attribute"]
            truth = eval_condition(pair, condition).to_numpy(dtype=bool)
            active_index = active_position.get(attr)
            if active_index is None:
                query_mask &= truth[0]
            else:
                selected = local_masks[
                    :, local_scope_position[active_index]
                ]
                query_mask &= np.where(selected, truth[1], truth[0])
        values = query_mask.astype(float) * weighted_residual[query_index]
        values -= values[0]
        if scope in aggregated:
            aggregated[scope] += values
        else:
            aggregated[scope] = values

    factors = []
    for scope in sorted(aggregated, key=lambda item: (len(item), item)):
        values = np.asarray(aggregated[scope], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("聚合后的因子能量超出 float64 可表示范围")
        values[0] = 0.0
        if np.any(values != 0.0):
            factors.append(MaskEnergyFactor(scope=scope, values=values))
    factors_tuple = tuple(factors)
    adjacency = [[] for _ in active_attributes]
    for factor_index, factor in enumerate(factors_tuple):
        for variable in factor.scope:
            adjacency[variable].append(factor_index)

    return SparseMaskEnergy(
        active_attribute_indices=active_attribute_indices,
        active_attributes=active_attributes,
        factors=factors_tuple,
        factors_by_variable=tuple(tuple(values) for values in adjacency),
        n_queries=len(queries),
        n_active_queries=n_active_queries,
        max_active_query_order=max_active_query_order,
    )


def evaluate_sparse_mask_energies(
    model: SparseMaskEnergy,
    masks: np.ndarray,
) -> np.ndarray:
    """评价任意一批完整 mask 的中心化方向能量。"""
    checked = _validate_masks(masks, model.n_active_attributes)
    energies = np.zeros(len(checked), dtype=float)
    for factor in model.factors:
        scope = np.asarray(factor.scope, dtype=np.intp)
        powers = 1 << np.arange(len(scope), dtype=np.intp)
        indices = checked[:, scope].astype(np.intp) @ powers
        energies += factor.values[indices]
    return energies


def evaluate_sparse_mask_energy(
    model: SparseMaskEnergy,
    mask: np.ndarray,
) -> float:
    """评价一个完整 mask 的中心化方向能量。"""
    checked = _validate_mask(mask, model.n_active_attributes)
    return float(evaluate_sparse_mask_energies(model, checked[None, :])[0])


def sparse_single_directions(model: SparseMaskEnergy) -> np.ndarray:
    """返回模型中每个活跃属性相对空 mask 的单块方向。"""
    n_active = model.n_active_attributes
    if n_active == 0:
        return np.zeros(0, dtype=float)
    one_hot = np.eye(n_active, dtype=bool)
    return evaluate_sparse_mask_energies(model, one_hot)


def conditional_energy_difference(
    model: SparseMaskEnergy,
    mask: np.ndarray,
    variable: int,
) -> float:
    """返回固定其他 bit 时 ``U(M_g=1)-U(M_g=0)``。"""
    checked = _validate_mask(mask, model.n_active_attributes)
    if (
        isinstance(variable, (bool, np.bool_))
        or not isinstance(variable, (int, np.integer))
        or not 0 <= variable < model.n_active_attributes
    ):
        raise ValueError(
            f"variable 必须在 [0, {model.n_active_attributes}) 内，"
            f"得到 {variable!r}"
        )
    variable = int(variable)
    return _conditional_energy_difference_unchecked(model, checked, variable)


def _conditional_energy_difference_unchecked(
    model: SparseMaskEnergy,
    mask: np.ndarray,
    variable: int,
) -> float:
    """已校验输入上的条件能量差，供 Gibbs 内循环使用。"""
    difference = 0.0
    for factor_index in model.factors_by_variable[variable]:
        factor = model.factors[factor_index]
        local_position = factor.scope.index(variable)
        lower_index = 0
        for position, scoped_variable in enumerate(factor.scope):
            if scoped_variable != variable and mask[scoped_variable]:
                lower_index |= 1 << position
        upper_index = lower_index | (1 << local_position)
        difference += (
            factor.values[upper_index] - factor.values[lower_index]
        )
    return float(difference)


def _conditional_copy_probability_unchecked(
    model: SparseMaskEnergy,
    mask: np.ndarray,
    variable: int,
    base_logit: float,
    strength: float,
    logit_clip: Optional[float],
) -> float:
    difference = _conditional_energy_difference_unchecked(
        model, mask, variable
    )
    logit = base_logit + strength * difference
    if not np.isfinite(logit):
        raise ValueError("条件 logit 超出 float64 可表示范围")
    if logit_clip is not None:
        logit = float(np.clip(logit, -logit_clip, logit_clip))
    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exponential = float(np.exp(logit))
    return exponential / (1.0 + exponential)


def conditional_copy_probability(
    model: SparseMaskEnergy,
    mask: np.ndarray,
    variable: int,
    eta: float,
    strength: float,
    *,
    logit_clip: Optional[float] = None,
) -> float:
    """返回联合 Gibbs 核的单 bit 精确条件复制概率。"""
    baseline = _validate_open_probability(eta, "eta")
    beta = _validate_strength(strength)
    clip = _validate_logit_clip(logit_clip)
    checked = _validate_mask(mask, model.n_active_attributes)
    if (
        isinstance(variable, (bool, np.bool_))
        or not isinstance(variable, (int, np.integer))
        or not 0 <= variable < model.n_active_attributes
    ):
        raise ValueError(
            f"variable 必须在 [0, {model.n_active_attributes}) 内，"
            f"得到 {variable!r}"
        )
    base_logit = float(np.log(baseline) - np.log1p(-baseline))
    return _conditional_copy_probability_unchecked(
        model,
        checked,
        int(variable),
        base_logit,
        beta,
        clip,
    )


def random_scan_gibbs_mask(
    model: SparseMaskEnergy,
    initial_mask: np.ndarray,
    eta: float,
    strength: float,
    n_steps: int,
    rng: np.random.Generator,
    *,
    logit_clip: Optional[float] = None,
) -> np.ndarray:
    """从显式初始 mask 做随机坐标、带放回的 Gibbs 微步。

    每个微步均匀选择一个活跃 bit，并按其完整条件分布重采样。``n_steps=0``
    精确返回初始 mask 且不消耗 RNG；空活跃集合也不消耗 RNG。
    """
    mask = _validate_mask(
        initial_mask, model.n_active_attributes, "initial_mask"
    ).copy()
    baseline = _validate_open_probability(eta, "eta")
    beta = _validate_strength(strength)
    steps = _validate_nonnegative_integer(n_steps, "n_steps")
    clip = _validate_logit_clip(logit_clip)
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng 必须是 np.random.Generator")
    if steps == 0 or model.n_active_attributes == 0:
        return mask

    base_logit = float(np.log(baseline) - np.log1p(-baseline))
    for _ in range(steps):
        variable = int(rng.integers(0, model.n_active_attributes))
        probability = _conditional_copy_probability_unchecked(
            model,
            mask,
            variable,
            base_logit,
            beta,
            clip,
        )
        mask[variable] = rng.random() < probability
    return mask


def propagate_random_scan_distribution(
    model: SparseMaskEnergy,
    initial_probabilities: np.ndarray,
    eta: float,
    strength: float,
    n_steps: int,
    *,
    max_active_attributes: int = 12,
    logit_clip: Optional[float] = None,
) -> np.ndarray:
    """在小状态空间精确传播随机扫描 Gibbs 分布，用于混合诊断。

    该函数枚举全部 mask，复杂度仍为 ``O(n_steps*k*2^k)``，不得用于宽表生产
    路径。它与 :func:`random_scan_gibbs_mask` 使用相同的随机坐标带放回语义。
    """
    baseline = _validate_open_probability(eta, "eta")
    beta = _validate_strength(strength)
    steps = _validate_nonnegative_integer(n_steps, "n_steps")
    clip = _validate_logit_clip(logit_clip)
    masks = enumerate_copy_masks(
        model.n_active_attributes,
        max_active_attributes=max_active_attributes,
    )
    raw_probabilities = np.asarray(initial_probabilities)
    if raw_probabilities.shape != (len(masks),):
        raise ValueError(
            "initial_probabilities 必须与完整 mask 状态数一致，"
            f"得到 {raw_probabilities.shape}，期望 ({len(masks)},)"
        )
    if raw_probabilities.dtype.kind not in "iuf":
        raise ValueError("initial_probabilities 必须是非负有限数值数组")
    probabilities = raw_probabilities.astype(float, copy=True)
    with np.errstate(over="ignore", invalid="ignore"):
        total_probability = float(probabilities.sum())
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.isfinite(total_probability)
        or total_probability <= 0.0
    ):
        raise ValueError("initial_probabilities 必须是非负有限且总和为正")
    probabilities /= total_probability
    if steps == 0 or model.n_active_attributes == 0:
        return probabilities

    energies = evaluate_sparse_mask_energies(model, masks)
    base_logit = np.log(baseline) - np.log1p(-baseline)
    pair_data = []
    state_indices = np.arange(len(masks), dtype=np.intp)
    for variable in range(model.n_active_attributes):
        lower = state_indices[~masks[:, variable]]
        upper = lower | (1 << variable)
        logits = base_logit + beta * (energies[upper] - energies[lower])
        if not np.all(np.isfinite(logits)):
            raise ValueError("条件 logit 超出 float64 可表示范围")
        if clip is not None:
            logits = np.clip(logits, -clip, clip)
        pair_data.append((lower, upper, _stable_sigmoid(logits)))

    inverse_width = 1.0 / model.n_active_attributes
    for _ in range(steps):
        next_probabilities = np.zeros_like(probabilities)
        for lower, upper, copy_probability in pair_data:
            pair_mass = probabilities[lower] + probabilities[upper]
            next_probabilities[lower] += (
                pair_mass * (1.0 - copy_probability) * inverse_width
            )
            next_probabilities[upper] += (
                pair_mass * copy_probability * inverse_width
            )
        probabilities = next_probabilities
        probabilities /= probabilities.sum()
    return probabilities


def evolve_step_factorized_gibbs(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    residual: np.ndarray,
    *,
    rho: float = 0.1,
    eta: float = 0.5,
    mu: float = 0.01,
    copy_direction_scores: np.ndarray,
    copy_direction_strength: float,
    n_sweeps: int,
    rng: np.random.Generator,
    gibbs_rng: Optional[np.random.Generator] = None,
    weights: Optional[np.ndarray] = None,
    max_factor_order: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """执行一轮“独立定向初值 + 低阶因子 Gibbs”同步更新。

    主 ``rng`` 的抽取顺序与 :func:`table_diffevo.update.evolve_step` 保持一致：
    participation、逐属性独立复制初值和 mutation 都从该流抽取。额外 Gibbs 微步只
    使用独立的 ``gibbs_rng``，因此增加 sweep 不会错位后续 donor、复制或变异随机
    流。``n_sweeps=0`` 不构造因子、不消费 ``gibbs_rng``，并精确退化到现有定向
    ``evolve_step``。

    返回的诊断只记录公开生成过程的工作量和墙钟，不评价 loss，也不参与更新决策。
    """
    for value, name in ((rho, "rho"), (eta, "eta"), (mu, "mu")):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"{name} 必须是 [0, 1] 内的有限数值，得到 {value!r}"
            )
    rho = float(rho)
    eta = float(eta)
    mu = float(mu)
    strength = _validate_strength(copy_direction_strength)
    sweeps = _validate_nonnegative_integer(n_sweeps, "n_sweeps")
    maximum_order = _validate_nonnegative_integer(
        max_factor_order, "max_factor_order"
    )
    if maximum_order > 8:
        raise ValueError("max_factor_order 不得超过绝对护栏 8")
    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng 必须是 np.random.Generator")
    if sweeps > 0:
        _validate_open_probability(eta, "eta")
        if not isinstance(gibbs_rng, np.random.Generator):
            raise ValueError(
                "n_sweeps 非零时 gibbs_rng 必须是 np.random.Generator"
            )
        residual_values = _validate_numeric_vector(
            residual, len(queries), "residual"
        )
        weight_values = (
            None
            if weights is None
            else _validate_numeric_vector(weights, len(queries), "weights")
        )
    else:
        # 0 sweep 不使用查询或残差，保留与既有 evolve_step 相同的最小依赖边界。
        residual_values = residual
        weight_values = weights

    attr_names = schema.attribute_names()
    n_records = len(current)
    raw_scores = np.asarray(copy_direction_scores)
    expected_shape = (n_records, len(attr_names))
    if raw_scores.shape != expected_shape:
        raise ValueError(
            "copy_direction_scores 必须是 shape (N, A) 的二维数组，"
            f"得到 {raw_scores.shape}，期望 {expected_shape}"
        )
    if raw_scores.dtype.kind not in "iuf":
        raise ValueError("copy_direction_scores 必须是有限数值数组")
    direction_scores = raw_scores.astype(float, copy=False)
    if not np.all(np.isfinite(direction_scores)):
        raise ValueError("copy_direction_scores 必须是有限数值数组")

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)
    proposal = current_reset.copy()
    participate = rng.random(n_records) < rho
    differs = np.zeros((n_records, len(attr_names)), dtype=bool)
    copy_masks = np.zeros_like(differs)

    # 保留既有 evolve_step 的“每个属性抽 N 个 roll”顺序，0 sweep 可逐随机位对拍。
    for attr_index, attr in enumerate(attr_names):
        current_values = current_reset[attr].to_numpy()
        donor_values = donors_reset[attr].to_numpy()
        differs[:, attr_index] = current_values != donor_values
        if strength == 0.0:
            copy_probabilities = eta
        else:
            copy_probabilities = tilted_copy_probabilities(
                eta,
                direction_scores[:, attr_index],
                strength,
            )
        copy_masks[:, attr_index] = (
            rng.random(n_records) < copy_probabilities
        )
    copy_masks &= differs

    factor_build_elapsed = 0.0
    gibbs_sample_elapsed = 0.0
    active_gibbs_rows = 0
    active_blocks = 0
    factor_count = 0
    factor_table_entries = 0
    gibbs_microsteps = 0
    if sweeps > 0:
        for row_index in np.flatnonzero(participate):
            active_indices = np.flatnonzero(differs[row_index])
            n_active = len(active_indices)
            if n_active == 0:
                continue
            build_start = time.perf_counter()
            model = build_sparse_mask_energy(
                current_reset.iloc[[row_index]],
                donors_reset.iloc[[row_index]],
                schema,
                queries,
                residual_values,
                weights=weight_values,
                max_factor_order=maximum_order,
            )
            factor_build_elapsed += time.perf_counter() - build_start
            if not np.array_equal(
                model.active_attribute_indices, active_indices
            ):
                raise RuntimeError("因子模型的活跃属性顺序与更新 mask 不一致")
            sample_start = time.perf_counter()
            copy_masks[row_index, active_indices] = random_scan_gibbs_mask(
                model,
                copy_masks[row_index, active_indices],
                eta,
                strength,
                sweeps * n_active,
                gibbs_rng,
            )
            gibbs_sample_elapsed += time.perf_counter() - sample_start
            active_gibbs_rows += 1
            active_blocks += n_active
            factor_count += len(model.factors)
            factor_table_entries += sum(
                len(factor.values) for factor in model.factors
            )
            gibbs_microsteps += sweeps * n_active

    for attr_index, attr in enumerate(attr_names):
        selected = participate & copy_masks[:, attr_index]
        if np.any(selected):
            new_values = proposal[attr].to_numpy().copy()
            donor_values = donors_reset[attr].to_numpy()
            new_values[selected] = donor_values[selected]
            proposal[attr] = new_values

    # 与 evolve_step 相同：每条参与记录至多变异一个属性，且发生在复制之后。
    mutate_rows = np.flatnonzero(
        participate & (rng.random(n_records) < mu)
    )
    for row_index in mutate_rows:
        block_index = int(rng.integers(0, len(attr_names)))
        block = schema.get_block(attr_names[block_index])
        if block.is_numeric():
            low, high = block.range
            value = int(rng.integers(int(low), int(high) + 1))
        else:
            value_index = int(rng.integers(0, len(block.values)))
            value = block.values[value_index]
        proposal.at[row_index, block.name] = value

    diagnostics = {
        "participating_rows": int(participate.sum()),
        "active_gibbs_rows": active_gibbs_rows,
        "active_blocks": active_blocks,
        "factor_count": factor_count,
        "factor_table_entries": factor_table_entries,
        "gibbs_microsteps": gibbs_microsteps,
        "factor_build_elapsed_sec": factor_build_elapsed,
        "gibbs_sample_elapsed_sec": gibbs_sample_elapsed,
    }
    return proposal, diagnostics
