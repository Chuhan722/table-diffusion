"""整代曲率感知的有限温度 Gibbs 扩散研究原型。

固定当前合成表、donor、参与行和旧残差后，本模块把所有复制 bit 视为一个整代
mask。能量同时包含残差一阶项与平方 workload 的完整查询空间二次项。更新只使用
单 bit Gibbs 条件概率，不执行收益门槛、argmax、top-k 或 generation acceptance。

该模块目前只服务冻结研究脚本，不接入默认生成器。
"""

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import tilted_copy_probabilities
from table_diffevo.factorized_diffusion import (
    SparseMaskEnergy,
    build_sparse_mask_energy,
    conditional_energy_difference,
)
from table_diffevo.joint_diffusion import enumerate_copy_masks
from table_diffevo.queries import eval_condition
from table_diffevo.schema import Schema


@dataclass(frozen=True)
class QueryDeltaFactor:
    """一条查询相对 recipient 的局部 0/1 mask 变化表。"""

    query_index: int
    scope: Tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class SparseQueryDelta:
    """一条 recipient-donor 对的稀疏查询变化模型。"""

    active_attribute_indices: np.ndarray
    active_attributes: Tuple[str, ...]
    factors: Tuple[QueryDeltaFactor, ...]
    factors_by_variable: Tuple[Tuple[int, ...], ...]
    n_queries: int
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


def _validate_nonnegative_finite(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0.0
    ):
        raise ValueError(f"{name} 必须是非负有限数值，得到 {value!r}")
    return float(value)


def _validate_open_probability(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0.0 < value < 1.0
    ):
        raise ValueError(f"{name} 必须是 (0, 1) 内的有限数值，得到 {value!r}")
    return float(value)


def _validate_unit_probability(value: float, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} 必须是 [0, 1] 内的有限数值，得到 {value!r}")
    return float(value)


def _validate_numeric_vector(
    values: np.ndarray,
    expected_length: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (expected_length,) or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} 必须是长度 {expected_length} 的有限数值一维数组"
        )
    result = raw.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"{name} 必须是长度 {expected_length} 的有限数值一维数组"
        )
    return result


def _validate_mask(mask: np.ndarray, width: int, name: str = "mask") -> np.ndarray:
    raw = np.asarray(mask)
    if raw.shape != (width,) or raw.dtype.kind not in "biuf":
        raise ValueError(
            f"{name} 必须是长度 {width} 的一维 0/1 数组，得到 {raw.shape}"
        )
    if not np.all(np.isfinite(raw)) or np.any((raw != 0) & (raw != 1)):
        raise ValueError(f"{name} 必须只包含 0/1 或布尔值")
    return raw.astype(bool, copy=False)


def _stable_sigmoid_scalar(logit: float) -> float:
    if not np.isfinite(logit):
        raise ValueError("条件 logit 超出 float64 可表示范围")
    if logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-logit)))
    exponential = float(np.exp(logit))
    return exponential / (1.0 + exponential)


def _bernoulli_entropy_scalar(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return float(
        -probability * np.log(probability)
        - (1.0 - probability) * np.log1p(-probability)
    )


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    payload = (
        str(array.dtype).encode()
        + repr(array.shape).encode()
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def build_sparse_query_delta(
    recipient: pd.DataFrame,
    donor: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    *,
    max_factor_order: int = 3,
) -> SparseQueryDelta:
    """构造每条查询相对 recipient 的稀疏局部变化因子。

    每个因子值都是完整 query indicator 在相应局部 mask 下的值减去空 mask
    的值，因此只可能为 ``-1``、``0`` 或 ``1``。查询只依赖自身涉及的活跃属性，
    不枚举 recipient 与 donor 的全部属性组合。
    """
    if len(recipient) != 1 or len(donor) != 1:
        raise ValueError("recipient 和 donor 都必须恰好包含一行")
    if not isinstance(queries, list):
        raise ValueError("queries 必须是列表")
    maximum_order = _validate_nonnegative_integer(
        max_factor_order, "max_factor_order"
    )
    if maximum_order > 8:
        raise ValueError("max_factor_order 不得超过绝对护栏 8")

    attr_names = schema.attribute_names()
    missing_recipient = [
        name for name in attr_names if name not in recipient.columns
    ]
    missing_donor = [
        name for name in attr_names if name not in donor.columns
    ]
    if missing_recipient or missing_donor:
        raise ValueError(
            "recipient/donor 缺少 schema 属性列："
            f"recipient={missing_recipient}, donor={missing_donor}"
        )

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

    factors = []
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
        values = query_mask.astype(np.int8) - np.int8(query_mask[0])
        values[0] = 0
        if np.any(values != 0):
            factors.append(QueryDeltaFactor(
                query_index=query_index,
                scope=scope,
                values=values,
            ))

    factors_tuple = tuple(factors)
    adjacency = [[] for _ in active_attributes]
    for factor_index, factor in enumerate(factors_tuple):
        for variable in factor.scope:
            adjacency[variable].append(factor_index)
    return SparseQueryDelta(
        active_attribute_indices=active_attribute_indices,
        active_attributes=active_attributes,
        factors=factors_tuple,
        factors_by_variable=tuple(tuple(items) for items in adjacency),
        n_queries=len(queries),
        max_active_query_order=max_active_query_order,
    )


def evaluate_sparse_query_delta(
    model: SparseQueryDelta,
    mask: np.ndarray,
) -> np.ndarray:
    """评价一个完整行 mask 对所有查询计数的变化。"""
    checked = _validate_mask(mask, model.n_active_attributes)
    delta = np.zeros(model.n_queries, dtype=np.int16)
    for factor in model.factors:
        local_index = 0
        for position, variable in enumerate(factor.scope):
            if checked[variable]:
                local_index |= 1 << position
        delta[factor.query_index] += int(factor.values[local_index])
    return delta


def conditional_query_delta_difference(
    model: SparseQueryDelta,
    mask: np.ndarray,
    variable: int,
) -> np.ndarray:
    """返回固定其他 bit 时 ``δq(M_b=1)-δq(M_b=0)``。"""
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
    difference = np.zeros(model.n_queries, dtype=np.int16)
    for factor_index in model.factors_by_variable[variable]:
        factor = model.factors[factor_index]
        local_position = factor.scope.index(variable)
        lower_index = 0
        for position, scoped_variable in enumerate(factor.scope):
            if scoped_variable != variable and checked[scoped_variable]:
                lower_index |= 1 << position
        upper_index = lower_index | (1 << local_position)
        difference[factor.query_index] += int(
            factor.values[upper_index] - factor.values[lower_index]
        )
    return difference


def generation_curvature_energy(
    proportional_residual: np.ndarray,
    total_query_delta: np.ndarray,
    n_records: int,
    curvature_weight: float,
) -> float:
    """返回 ``<e,D> - gamma/(2N)||D||²``。"""
    if (
        isinstance(n_records, (bool, np.bool_))
        or not isinstance(n_records, (int, np.integer))
        or n_records <= 0
    ):
        raise ValueError(f"n_records 必须是正整数，得到 {n_records!r}")
    gamma = _validate_nonnegative_finite(
        curvature_weight, "curvature_weight"
    )
    raw_delta = np.asarray(total_query_delta)
    if raw_delta.ndim != 1 or raw_delta.dtype.kind not in "iuf":
        raise ValueError("total_query_delta 必须是有限数值一维数组")
    delta = raw_delta.astype(float, copy=False)
    if not np.all(np.isfinite(delta)):
        raise ValueError("total_query_delta 必须是有限数值一维数组")
    residual = _validate_numeric_vector(
        proportional_residual, len(delta), "proportional_residual"
    )
    linear = float(np.dot(residual, delta))
    quadratic = float(0.5 * np.dot(delta, delta) / int(n_records))
    energy = linear - gamma * quadratic
    if not np.isfinite(energy):
        raise ValueError("整代曲率能量超出 float64 可表示范围")
    return float(energy)


def conditional_generation_energy_difference(
    linear_model: SparseMaskEnergy,
    query_model: SparseQueryDelta,
    mask: np.ndarray,
    variable: int,
    total_query_delta: np.ndarray,
    n_records: int,
    curvature_weight: float,
) -> Dict[str, Any]:
    """计算一个 bit 从 0 到 1 的精确整代条件能量差。"""
    if (
        linear_model.n_active_attributes
        != query_model.n_active_attributes
        or not np.array_equal(
            linear_model.active_attribute_indices,
            query_model.active_attribute_indices,
        )
    ):
        raise ValueError("linear/query 因子的活跃属性不一致")
    checked = _validate_mask(mask, query_model.n_active_attributes)
    if (
        isinstance(n_records, (bool, np.bool_))
        or not isinstance(n_records, (int, np.integer))
        or n_records <= 0
    ):
        raise ValueError(f"n_records 必须是正整数，得到 {n_records!r}")
    gamma = _validate_nonnegative_finite(
        curvature_weight, "curvature_weight"
    )
    total = _validate_numeric_vector(
        total_query_delta,
        query_model.n_queries,
        "total_query_delta",
    )
    query_difference = conditional_query_delta_difference(
        query_model, checked, variable
    ).astype(float, copy=False)
    linear_difference = conditional_energy_difference(
        linear_model, checked, variable
    )
    if checked[int(variable)]:
        delta_when_zero = total - query_difference
    else:
        delta_when_zero = total
    self_and_cross = float(
        np.dot(delta_when_zero, query_difference)
        + 0.5 * np.dot(query_difference, query_difference)
    )
    curvature_difference = -gamma * self_and_cross / int(n_records)
    energy_difference = float(linear_difference + curvature_difference)
    if not np.isfinite(energy_difference):
        raise ValueError("条件曲率能量差超出 float64 可表示范围")
    return {
        "energy_difference": energy_difference,
        "linear_difference": float(linear_difference),
        "curvature_difference": float(curvature_difference),
        "query_delta_difference": query_difference,
        "delta_when_zero": delta_when_zero,
    }


def conditional_generation_copy_probability(
    linear_model: SparseMaskEnergy,
    query_model: SparseQueryDelta,
    mask: np.ndarray,
    variable: int,
    total_query_delta: np.ndarray,
    n_records: int,
    curvature_weight: float,
    eta: float,
    strength: float,
) -> Dict[str, Any]:
    """返回整代曲率 Gibbs 的单 bit 条件概率及分解。"""
    baseline = _validate_open_probability(eta, "eta")
    beta = _validate_nonnegative_finite(strength, "strength")
    result = conditional_generation_energy_difference(
        linear_model,
        query_model,
        mask,
        variable,
        total_query_delta,
        n_records,
        curvature_weight,
    )
    base_logit = float(np.log(baseline) - np.log1p(-baseline))
    logit = base_logit + beta * result["energy_difference"]
    probability = _stable_sigmoid_scalar(logit)
    return {
        **result,
        "logit": float(logit),
        "probability": probability,
    }


def evolve_step_generation_curvature_gibbs(
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
    curvature_weight: float,
    rng: np.random.Generator,
    gibbs_rng: Optional[np.random.Generator] = None,
    max_factor_order: int = 3,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """执行一轮整代曲率感知的有限步 Gibbs 复制更新。

    主 ``rng`` 与现有更新保持 participation、独立初始复制 mask 和 mutation 的抽取
    顺序。额外微步只使用 ``gibbs_rng``。当 ``curvature_weight=0`` 时，逐行模型、
    坐标顺序和条件概率与现有 ``evolve_step_factorized_gibbs`` 相同；查询变化模型
    只提供二次项所需的公开查询变化，不消费随机数。

    曲率能量只描述 donor-copy mask。非零 ``mu`` 的变异仍发生在 Gibbs 之后，因此
    正式的精确 loss 恒等式实验固定 ``mu=0``。
    """
    rho = _validate_unit_probability(rho, "rho")
    eta = _validate_unit_probability(eta, "eta")
    mu = _validate_unit_probability(mu, "mu")
    strength = _validate_nonnegative_finite(
        copy_direction_strength, "copy_direction_strength"
    )
    sweeps = _validate_nonnegative_integer(n_sweeps, "n_sweeps")
    gamma = _validate_nonnegative_finite(
        curvature_weight, "curvature_weight"
    )
    maximum_order = _validate_nonnegative_integer(
        max_factor_order, "max_factor_order"
    )
    if maximum_order > 8:
        raise ValueError("max_factor_order 不得超过绝对护栏 8")
    if not isinstance(current, pd.DataFrame) or not isinstance(
        donors, pd.DataFrame
    ):
        raise ValueError("current 和 donors 必须是 pandas DataFrame")
    if len(current) == 0:
        raise ValueError("current 必须至少包含一行")
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
    else:
        residual_values = residual

    attr_names = schema.attribute_names()
    n_records = len(current)
    raw_scores = np.asarray(copy_direction_scores)
    expected_shape = (n_records, len(attr_names))
    if raw_scores.shape != expected_shape or raw_scores.dtype.kind not in "iuf":
        raise ValueError(
            "copy_direction_scores 必须是有限数值二维数组，"
            f"得到 {raw_scores.shape}，期望 {expected_shape}"
        )
    direction_scores = raw_scores.astype(float, copy=False)
    if not np.all(np.isfinite(direction_scores)):
        raise ValueError("copy_direction_scores 必须是有限数值二维数组")

    current_reset = current.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)
    proposal = current_reset.copy()
    participate = rng.random(n_records) < rho
    differs = np.zeros((n_records, len(attr_names)), dtype=bool)
    copy_masks = np.zeros_like(differs)

    # 与现有 evolve_step_factorized_gibbs 完全相同的主 RNG 顺序。
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
    initial_effective_copy_masks = copy_masks & participate[:, None]

    factor_build_start = time.perf_counter()
    row_models = {}
    total_query_delta = np.zeros(len(queries), dtype=float)
    linear_factor_count = 0
    linear_factor_table_entries = 0
    query_factor_count = 0
    query_factor_table_entries = 0
    active_blocks = 0
    if sweeps > 0:
        for row_index in np.flatnonzero(participate):
            active_indices = np.flatnonzero(differs[row_index])
            if len(active_indices) == 0:
                continue
            recipient = current_reset.iloc[[row_index]]
            donor = donors_reset.iloc[[row_index]]
            linear_model = build_sparse_mask_energy(
                recipient,
                donor,
                schema,
                queries,
                residual_values,
                max_factor_order=maximum_order,
            )
            query_model = build_sparse_query_delta(
                recipient,
                donor,
                schema,
                queries,
                max_factor_order=maximum_order,
            )
            if (
                not np.array_equal(
                    linear_model.active_attribute_indices, active_indices
                )
                or not np.array_equal(
                    query_model.active_attribute_indices, active_indices
                )
            ):
                raise RuntimeError("曲率因子的活跃属性顺序与更新 mask 不一致")
            local_mask = copy_masks[row_index, active_indices]
            row_delta = evaluate_sparse_query_delta(
                query_model, local_mask
            ).astype(float, copy=False)
            total_query_delta += row_delta
            row_models[row_index] = (linear_model, query_model, active_indices)
            active_blocks += len(active_indices)
            linear_factor_count += len(linear_model.factors)
            linear_factor_table_entries += sum(
                len(factor.values) for factor in linear_model.factors
            )
            query_factor_count += len(query_model.factors)
            query_factor_table_entries += sum(
                len(factor.values) for factor in query_model.factors
            )
    factor_build_elapsed = (
        time.perf_counter() - factor_build_start if sweeps > 0 else 0.0
    )
    initial_query_delta = total_query_delta.copy()

    sample_start = time.perf_counter()
    conditional_probability_min = None
    conditional_probability_max = None
    conditional_entropy_sum = 0.0
    conditional_probability_count = 0
    linear_query_consistency_max_error = 0.0
    gibbs_microsteps = 0
    if sweeps > 0:
        base_logit = float(np.log(eta) - np.log1p(-eta))
        for row_index in np.flatnonzero(participate):
            models = row_models.get(int(row_index))
            if models is None:
                continue
            linear_model, query_model, active_indices = models
            local_mask = copy_masks[row_index, active_indices]
            n_active = len(active_indices)
            for _ in range(sweeps * n_active):
                variable = int(gibbs_rng.integers(0, n_active))
                conditional = conditional_generation_energy_difference(
                    linear_model,
                    query_model,
                    local_mask,
                    variable,
                    total_query_delta,
                    n_records,
                    gamma,
                )
                query_linear = float(np.dot(
                    residual_values,
                    conditional["query_delta_difference"],
                ))
                linear_query_consistency_max_error = max(
                    linear_query_consistency_max_error,
                    abs(query_linear - conditional["linear_difference"]),
                )
                logit = (
                    base_logit
                    + strength * conditional["energy_difference"]
                )
                probability = _stable_sigmoid_scalar(logit)
                conditional_probability_min = (
                    probability
                    if conditional_probability_min is None
                    else min(conditional_probability_min, probability)
                )
                conditional_probability_max = (
                    probability
                    if conditional_probability_max is None
                    else max(conditional_probability_max, probability)
                )
                conditional_entropy_sum += _bernoulli_entropy_scalar(
                    probability
                )
                conditional_probability_count += 1

                old_value = bool(local_mask[variable])
                new_value = bool(gibbs_rng.random() < probability)
                if new_value != old_value:
                    sign = 1.0 if new_value else -1.0
                    total_query_delta += (
                        sign * conditional["query_delta_difference"]
                    )
                    local_mask[variable] = new_value
                gibbs_microsteps += 1
            copy_masks[row_index, active_indices] = local_mask
    gibbs_sample_elapsed = (
        time.perf_counter() - sample_start if sweeps > 0 else 0.0
    )

    for attr_index, attr in enumerate(attr_names):
        selected = participate & copy_masks[:, attr_index]
        if np.any(selected):
            new_values = proposal[attr].to_numpy().copy()
            donor_values = donors_reset[attr].to_numpy()
            new_values[selected] = donor_values[selected]
            proposal[attr] = new_values

    # 与现有更新相同：复制后，每条参与记录至多变异一个属性。
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

    initial_linear = (
        float(np.dot(residual_values, initial_query_delta))
        if sweeps > 0 else 0.0
    )
    final_linear = (
        float(np.dot(residual_values, total_query_delta))
        if sweeps > 0 else 0.0
    )
    initial_quadratic = (
        float(0.5 * np.dot(initial_query_delta, initial_query_delta) / n_records)
        if sweeps > 0 else 0.0
    )
    final_quadratic = (
        float(0.5 * np.dot(total_query_delta, total_query_delta) / n_records)
        if sweeps > 0 else 0.0
    )
    diagnostics = {
        "participating_rows": int(participate.sum()),
        "active_gibbs_rows": len(row_models),
        "active_blocks": int(active_blocks),
        "factor_count": int(linear_factor_count),
        "factor_table_entries": int(linear_factor_table_entries),
        "query_factor_count": int(query_factor_count),
        "query_factor_table_entries": int(query_factor_table_entries),
        "gibbs_microsteps": int(gibbs_microsteps),
        "factor_build_elapsed_sec": float(factor_build_elapsed),
        "gibbs_sample_elapsed_sec": float(gibbs_sample_elapsed),
        "curvature_weight": gamma,
        "initial_copy_mask_sha256": _array_sha256(
            initial_effective_copy_masks
        ),
        "final_copy_mask_sha256": _array_sha256(
            copy_masks & participate[:, None]
        ),
        "initial_query_delta": initial_query_delta.tolist(),
        "final_query_delta": total_query_delta.tolist(),
        "initial_linear_energy": initial_linear,
        "final_linear_energy": final_linear,
        "initial_quadratic_energy": initial_quadratic,
        "final_quadratic_energy": final_quadratic,
        "initial_generation_energy": (
            initial_linear - gamma * initial_quadratic
        ),
        "final_generation_energy": (
            final_linear - gamma * final_quadratic
        ),
        "conditional_probability_count": conditional_probability_count,
        "conditional_probability_min": conditional_probability_min,
        "conditional_probability_max": conditional_probability_max,
        "conditional_entropy_mean": (
            conditional_entropy_sum / conditional_probability_count
            if conditional_probability_count else None
        ),
        "all_conditionals_bidirectional": (
            conditional_probability_count == 0
            or (
                conditional_probability_min > 0.0
                and conditional_probability_max < 1.0
            )
        ),
        "linear_query_consistency_max_error": float(
            linear_query_consistency_max_error
        ),
    }
    return proposal, diagnostics
