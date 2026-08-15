"""至多低阶因子的联合 mask 能量与随机扫描 Gibbs 扩散。

固定 recipient、donor 和残差场后，每个合取查询只依赖它涉及的活跃复制 bit。
本模块把这些局部布尔函数聚合成稀疏因子，并用随机坐标 Gibbs 更新近似采样联合
mask 分布。它不执行正收益门槛、方向 argmax、top-k 或整代 proposal 接受检查。
"""

import copy
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import (
    DEFAULT_DIRECTION_LOGIT_CLIP,
    tilted_copy_probabilities,
    validate_direction_logit_clip,
)
from table_diffevo.joint_diffusion import enumerate_copy_masks
from table_diffevo.queries import _coerce_to_column_type, eval_condition
from table_diffevo.schema import Schema


DEFAULT_LOGIT_CLIP = 30.0


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


@dataclass(frozen=True)
class _CompiledMaskCondition:
    """一个已解析且可在整批 recipient-donor 上评价的查询条件。"""

    attribute_index: int
    attribute: str
    operator: str
    value: Any = None
    lower: Any = None
    upper: Any = None


@dataclass(frozen=True)
class _CompiledMaskQuery:
    """一个查询的条件索引和去重属性索引，均保持原始查询语义。"""

    condition_indices: Tuple[int, ...]
    attribute_indices: Tuple[int, ...]


@dataclass(frozen=True)
class CompiledMaskWorkload:
    """只包含公开 schema/workload 静态结构的显式编译对象。

    该对象不缓存 target、residual、recipient、donor 或查询结果。调用方负责它的
    生命周期，并可在多轮更新间显式复用；更新入口会拒绝 schema、查询或最高阶数
    不一致的对象。
    """

    attribute_names: Tuple[str, ...]
    max_factor_order: int
    n_queries: int
    n_unique_conditions: int
    _query_signature: Tuple[Any, ...]
    _conditions: Tuple[_CompiledMaskCondition, ...]
    _queries: Tuple[_CompiledMaskQuery, ...]
    _local_masks: Tuple[np.ndarray, ...]


@dataclass(frozen=True)
class _PreparedMaskEnergyBatch:
    """一次更新中批量评价后的动态条件真值与残差。"""

    compiled_workload: CompiledMaskWorkload
    active_attribute_indices: Tuple[np.ndarray, ...]
    recipient_condition_truth: np.ndarray
    donor_condition_truth: np.ndarray
    weighted_residual: np.ndarray


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


def _freeze_signature_value(value: Any) -> Any:
    """把查询操作数转成稳定、可比较且保留类型的结构签名。"""
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if np.isnan(value):
            return ("float", "nan")
        if np.isposinf(value):
            return ("float", "+inf")
        if np.isneginf(value):
            return ("float", "-inf")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (list, tuple)):
        return (
            type(value).__name__,
            tuple(_freeze_signature_value(item) for item in value),
        )
    if isinstance(value, dict):
        items = [
            (
                _freeze_signature_value(key),
                _freeze_signature_value(item),
            )
            for key, item in value.items()
        ]
        return ("dict", tuple(sorted(items, key=repr)))
    return (
        type(value).__module__,
        type(value).__qualname__,
        repr(value),
    )


def _condition_structure(
    condition: Dict[str, Any],
    *,
    query_index: int,
    condition_index: int,
    attribute_position: Dict[str, int],
) -> Tuple[int, str, str, Any, Any, Any, Tuple[Any, ...]]:
    """校验并解析一个受支持条件，同时生成兼容性签名。"""
    if not isinstance(condition, dict):
        raise ValueError(
            f"queries[{query_index}].conditions[{condition_index}] "
            "必须是字典"
        )
    attr = condition.get("attribute")
    try:
        attr_index = attribute_position[attr]
    except (KeyError, TypeError):
        raise ValueError(
            f"queries[{query_index}] 包含未知属性 {attr!r}"
        ) from None
    operator = condition.get("operator")
    if operator in ("==", ">="):
        if "value" not in condition:
            raise ValueError(
                f"queries[{query_index}].conditions[{condition_index}] "
                f"的 {operator!r} 算子缺少 value"
            )
        value = condition["value"]
        lower = None
        upper = None
        operands = (_freeze_signature_value(value),)
    elif operator == "between":
        missing = [
            name for name in ("lower", "upper") if name not in condition
        ]
        if missing:
            raise ValueError(
                f"queries[{query_index}].conditions[{condition_index}] "
                f"的 'between' 算子缺少 {missing}"
            )
        value = None
        lower = condition["lower"]
        upper = condition["upper"]
        operands = (
            _freeze_signature_value(lower),
            _freeze_signature_value(upper),
        )
    else:
        raise ValueError(
            f"queries[{query_index}].conditions[{condition_index}] "
            f"包含不支持的操作符 {operator!r}"
        )
    signature = (attr_index, operator, *operands)
    return attr_index, attr, operator, value, lower, upper, signature


def _mask_workload_signature(
    schema: Schema,
    queries: List[Dict[str, Any]],
    maximum_order: int,
) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    """生成编译对象的 schema/workload 兼容性签名。"""
    attr_names = tuple(schema.attribute_names())
    if len(set(attr_names)) != len(attr_names):
        raise ValueError("schema 属性名不得重复")
    attribute_position = {
        attr: index for index, attr in enumerate(attr_names)
    }
    query_signatures = []
    for query_index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"queries[{query_index}] 必须是字典")
        conditions = query.get("conditions")
        if not isinstance(conditions, list):
            raise ValueError(
                f"queries[{query_index}].conditions 必须是列表"
            )
        signatures = []
        for condition_index, condition in enumerate(conditions):
            *_, signature = _condition_structure(
                condition,
                query_index=query_index,
                condition_index=condition_index,
                attribute_position=attribute_position,
            )
            signatures.append(signature)
        query_signatures.append(tuple(signatures))
    return attr_names, (maximum_order, tuple(query_signatures))


def compile_mask_workload(
    schema: Schema,
    queries: List[Dict[str, Any]],
    *,
    max_factor_order: int = 3,
) -> CompiledMaskWorkload:
    """预编译只依赖公开 schema 和固定查询 workload 的静态结构。"""
    maximum_order = _validate_nonnegative_integer(
        max_factor_order, "max_factor_order"
    )
    if maximum_order > 8:
        raise ValueError("max_factor_order 不得超过绝对护栏 8")
    attr_names, query_signature = _mask_workload_signature(
        schema, queries, maximum_order
    )
    attribute_position = {
        attr: index for index, attr in enumerate(attr_names)
    }
    compiled_conditions = []
    condition_lookup = {}
    compiled_queries = []
    for query_index, query in enumerate(queries):
        condition_indices = []
        query_attributes = []
        for condition_index, condition in enumerate(query["conditions"]):
            (
                attr_index,
                attr,
                operator,
                value,
                lower,
                upper,
                condition_signature,
            ) = _condition_structure(
                condition,
                query_index=query_index,
                condition_index=condition_index,
                attribute_position=attribute_position,
            )
            compiled_index = condition_lookup.get(condition_signature)
            if compiled_index is None:
                compiled_index = len(compiled_conditions)
                condition_lookup[condition_signature] = compiled_index
                compiled_conditions.append(_CompiledMaskCondition(
                    attribute_index=attr_index,
                    attribute=attr,
                    operator=operator,
                    value=copy.deepcopy(value),
                    lower=copy.deepcopy(lower),
                    upper=copy.deepcopy(upper),
                ))
            condition_indices.append(compiled_index)
            if attr_index not in query_attributes:
                query_attributes.append(attr_index)
        compiled_queries.append(_CompiledMaskQuery(
            condition_indices=tuple(condition_indices),
            attribute_indices=tuple(query_attributes),
        ))

    local_masks = []
    for order in range(maximum_order + 1):
        masks = enumerate_copy_masks(
            order, max_active_attributes=maximum_order
        )
        masks.setflags(write=False)
        local_masks.append(masks)
    return CompiledMaskWorkload(
        attribute_names=attr_names,
        max_factor_order=maximum_order,
        n_queries=len(queries),
        n_unique_conditions=len(compiled_conditions),
        _query_signature=query_signature,
        _conditions=tuple(compiled_conditions),
        _queries=tuple(compiled_queries),
        _local_masks=tuple(local_masks),
    )


def _validate_compiled_workload_match(
    compiled_workload: CompiledMaskWorkload,
    schema: Schema,
    queries: List[Dict[str, Any]],
    maximum_order: int,
) -> None:
    if not isinstance(compiled_workload, CompiledMaskWorkload):
        raise ValueError(
            "compiled_workload 必须由 compile_mask_workload 创建"
        )
    attr_names, query_signature = _mask_workload_signature(
        schema, queries, maximum_order
    )
    if (
        compiled_workload.attribute_names != attr_names
        or compiled_workload.max_factor_order != maximum_order
        or compiled_workload._query_signature != query_signature
    ):
        raise ValueError(
            "compiled_workload 与当前 schema、queries 或 "
            "max_factor_order 不匹配"
        )


def _evaluate_compiled_condition(
    pair: pd.DataFrame,
    condition: _CompiledMaskCondition,
) -> np.ndarray:
    """用与 ``eval_condition`` 相同的类型对齐语义批量评价一个条件。"""
    column = pair[condition.attribute]
    if condition.operator == "==":
        value = _coerce_to_column_type(column, condition.value)
        truth = column == value
    elif condition.operator == ">=":
        truth = column >= condition.value
    else:
        truth = (
            (column >= condition.lower)
            & (column <= condition.upper)
        )
    return truth.to_numpy(dtype=bool)


def _prepare_sparse_mask_energy_batch(
    recipients: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    compiled_workload: CompiledMaskWorkload,
    residual: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> _PreparedMaskEnergyBatch:
    """一次批量评价动态条件真值，不保留输入表或跨轮状态。"""
    if not isinstance(compiled_workload, CompiledMaskWorkload):
        raise ValueError(
            "compiled_workload 必须由 compile_mask_workload 创建"
        )
    if len(recipients) != len(donors):
        raise ValueError(
            f"recipients 行数 ({len(recipients)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )
    attr_names = tuple(schema.attribute_names())
    if compiled_workload.attribute_names != attr_names:
        raise ValueError("compiled_workload 与当前 schema 不匹配")
    missing_recipients = [
        name for name in attr_names if name not in recipients.columns
    ]
    missing_donors = [
        name for name in attr_names if name not in donors.columns
    ]
    if missing_recipients or missing_donors:
        raise ValueError(
            "recipients/donors 缺少 schema 属性列："
            f"recipients={missing_recipients}, donors={missing_donors}"
        )

    residual_values = _validate_numeric_vector(
        residual, compiled_workload.n_queries, "residual"
    )
    if weights is None:
        weight_values = np.ones(compiled_workload.n_queries, dtype=float)
    else:
        weight_values = _validate_numeric_vector(
            weights, compiled_workload.n_queries, "weights"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        weighted_residual = residual_values * weight_values
    if not np.all(np.isfinite(weighted_residual)):
        raise ValueError("residual * weights 超出 float64 可表示范围")

    recipients_reset = recipients.reset_index(drop=True)
    donors_reset = donors.reset_index(drop=True)
    selected_columns = list(attr_names)
    recipient_values = recipients_reset.loc[:, selected_columns].to_numpy()
    donor_values = donors_reset.loc[:, selected_columns].to_numpy()
    differs = recipient_values != donor_values
    active_attribute_indices = tuple(
        np.flatnonzero(differs[row_index]).astype(np.intp, copy=False)
        for row_index in range(len(recipients_reset))
    )

    n_rows = len(recipients_reset)
    n_conditions = compiled_workload.n_unique_conditions
    condition_truth = np.empty((2 * n_rows, n_conditions), dtype=bool)
    if n_rows > 0 and n_conditions > 0:
        pair = pd.concat(
            [recipients_reset, donors_reset], ignore_index=True
        )
        for condition_index, condition in enumerate(
            compiled_workload._conditions
        ):
            condition_truth[:, condition_index] = (
                _evaluate_compiled_condition(pair, condition)
            )
    condition_truth.setflags(write=False)
    weighted_residual.setflags(write=False)
    for indices in active_attribute_indices:
        indices.setflags(write=False)
    return _PreparedMaskEnergyBatch(
        compiled_workload=compiled_workload,
        active_attribute_indices=active_attribute_indices,
        recipient_condition_truth=condition_truth[:n_rows],
        donor_condition_truth=condition_truth[n_rows:],
        weighted_residual=weighted_residual,
    )


def _build_sparse_mask_energy_from_batch(
    prepared: _PreparedMaskEnergyBatch,
    row_index: int,
) -> SparseMaskEnergy:
    """从已批量评价的条件真值构造一条记录的稀疏能量。"""
    compiled = prepared.compiled_workload
    active_attribute_indices = prepared.active_attribute_indices[row_index]
    active_attributes = tuple(
        compiled.attribute_names[index]
        for index in active_attribute_indices
    )
    active_position = {
        int(attribute_index): position
        for position, attribute_index in enumerate(active_attribute_indices)
    }
    aggregated = {}
    n_active_queries = 0
    max_active_query_order = 0
    for query_index, query in enumerate(compiled._queries):
        scope = tuple(sorted(
            active_position[attribute_index]
            for attribute_index in query.attribute_indices
            if attribute_index in active_position
        ))
        order = len(scope)
        if order == 0:
            continue
        if order > compiled.max_factor_order:
            raise ValueError(
                f"queries[{query_index}] 的活跃因子阶数 {order} "
                f"超过 max_factor_order={compiled.max_factor_order}"
            )
        n_active_queries += 1
        max_active_query_order = max(max_active_query_order, order)

        local_masks = compiled._local_masks[order]
        local_scope_position = {
            active_index: position
            for position, active_index in enumerate(scope)
        }
        query_mask = np.ones(len(local_masks), dtype=bool)
        for condition_index in query.condition_indices:
            condition = compiled._conditions[condition_index]
            active_index = active_position.get(condition.attribute_index)
            if active_index is None:
                query_mask &= prepared.recipient_condition_truth[
                    row_index, condition_index
                ]
            else:
                selected = local_masks[
                    :, local_scope_position[active_index]
                ]
                query_mask &= np.where(
                    selected,
                    prepared.donor_condition_truth[
                        row_index, condition_index
                    ],
                    prepared.recipient_condition_truth[
                        row_index, condition_index
                    ],
                )
        values = (
            query_mask.astype(float)
            * prepared.weighted_residual[query_index]
        )
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
        n_queries=compiled.n_queries,
        n_active_queries=n_active_queries,
        max_active_query_order=max_active_query_order,
    )


def build_sparse_mask_energies_batch(
    recipients: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    compiled_workload: CompiledMaskWorkload,
    residual: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
) -> Tuple[SparseMaskEnergy, ...]:
    """用显式编译对象批量评价条件并构造多条稀疏 mask 能量。"""
    prepared = _prepare_sparse_mask_energy_batch(
        recipients,
        donors,
        schema,
        compiled_workload,
        residual,
        weights=weights,
    )
    return tuple(
        _build_sparse_mask_energy_from_batch(prepared, row_index)
        for row_index in range(len(recipients))
    )


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
        with np.errstate(over="ignore", invalid="ignore"):
            energies += factor.values[indices]
    if not np.all(np.isfinite(energies)):
        raise ValueError("稀疏 mask 能量超出 float64 可表示范围")
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
    difference = _conditional_energy_difference_unchecked(
        model, checked, variable
    )
    if not np.isfinite(difference):
        raise ValueError("条件能量差超出 float64 可表示范围")
    return difference


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
        with np.errstate(over="ignore", invalid="ignore"):
            difference += (
                factor.values[upper_index] - factor.values[lower_index]
            )
    return float(difference)


def _conditional_copy_probability_unchecked(
    model: SparseMaskEnergy,
    mask: np.ndarray,
    variable: int,
    baseline_probability: float,
    base_logit: float,
    strength: float,
    logit_clip: Optional[float],
    condition_observer: Optional[
        Callable[[int, float, float], None]
    ] = None,
) -> float:
    difference = _conditional_energy_difference_unchecked(
        model, mask, variable
    )
    if strength == 0.0 or difference == 0.0:
        raw_logit = base_logit
        probability = baseline_probability
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            raw_logit = base_logit + strength * difference
        if np.isnan(raw_logit):
            raise ValueError("条件 logit 无法由有限输入稳定计算")
        if logit_clip is None:
            if not np.isfinite(raw_logit):
                raise ValueError("条件 logit 超出 float64 可表示范围")
            effective_logit = raw_logit
        else:
            effective_logit = float(np.clip(
                raw_logit, -logit_clip, logit_clip
            ))
        if effective_logit >= 0.0:
            probability = float(
                1.0 / (1.0 + np.exp(-effective_logit))
            )
        else:
            exponential = float(np.exp(effective_logit))
            probability = exponential / (1.0 + exponential)
    if condition_observer is not None:
        condition_observer(
            int(variable), float(raw_logit), float(probability)
        )
    return probability


def conditional_copy_probability(
    model: SparseMaskEnergy,
    mask: np.ndarray,
    variable: int,
    eta: float,
    strength: float,
    *,
    logit_clip: Optional[float] = DEFAULT_LOGIT_CLIP,
) -> float:
    """返回联合 Gibbs 核的单 bit 条件复制概率。

    默认沿用现有方向核的 ``[-30, 30]`` 数值护栏，使有限输入在 float64 中仍保留
    双向支持。显式传入 ``None`` 可关闭护栏，用于理论目标的精确条件式；极端
    logit 此时可能舍入到 0/1，超出 float64 时会明确报错。
    """
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
        baseline,
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
    logit_clip: Optional[float] = DEFAULT_LOGIT_CLIP,
    condition_observer: Optional[
        Callable[[int, float, float], None]
    ] = None,
) -> np.ndarray:
    """从显式初始 mask 做随机坐标、带放回的 Gibbs 微步。

    每个微步均匀选择一个活跃 bit，并按其完整条件分布重采样。``n_steps=0``
    精确返回初始 mask 且不消耗 RNG；空活跃集合也不消耗 RNG。
    默认把条件 logit 截到 ``[-30, 30]``，避免有限输入在 float64 中丢失双向支持；
    ``logit_clip=None`` 只用于显式请求未截断的理论条件式。
    可选 ``condition_observer`` 在每个实际微步收到变量索引、截断前 logit 和实际
    采样概率；它只用于诊断，采样器不会为它额外消耗随机数。
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
    if condition_observer is not None and not callable(condition_observer):
        raise ValueError("condition_observer 必须可调用或为 None")
    if steps == 0 or model.n_active_attributes == 0:
        return mask

    base_logit = float(np.log(baseline) - np.log1p(-baseline))
    for _ in range(steps):
        variable = int(rng.integers(0, model.n_active_attributes))
        probability = _conditional_copy_probability_unchecked(
            model,
            mask,
            variable,
            baseline,
            base_logit,
            beta,
            clip,
            condition_observer,
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
    logit_clip: Optional[float] = DEFAULT_LOGIT_CLIP,
) -> np.ndarray:
    """在小状态空间精确传播随机扫描 Gibbs 分布，用于混合诊断。

    该函数枚举全部 mask，复杂度仍为 ``O(n_steps*k*2^k)``，不得用于宽表生产
    路径。它精确传播由 ``logit_clip`` 定义的随机坐标转移核，并与
    :func:`random_scan_gibbs_mask` 使用相同语义；关闭护栏且 logit 可表示时才是
    未截断理论 Gibbs 核的精确传播。
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
        with np.errstate(over="ignore", invalid="ignore"):
            differences = energies[upper] - energies[lower]
            logits = base_logit + beta * (
                differences
            )
        if np.any(np.isnan(logits)):
            raise ValueError("条件 logit 无法由有限输入稳定计算")
        if clip is None:
            if not np.all(np.isfinite(logits)):
                raise ValueError("条件 logit 超出 float64 可表示范围")
        else:
            logits = np.clip(logits, -clip, clip)
        copy_probabilities = _stable_sigmoid(logits)
        copy_probabilities[(beta == 0.0) | (differences == 0.0)] = baseline
        pair_data.append((lower, upper, copy_probabilities))

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
    gibbs_logit_clip: Optional[float] = DEFAULT_LOGIT_CLIP,
    compiled_workload: Optional[CompiledMaskWorkload] = None,
    direction_logit_clip: Optional[float] = DEFAULT_DIRECTION_LOGIT_CLIP,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """执行一轮“独立定向初值 + 低阶因子 Gibbs”同步更新。

    主 ``rng`` 的抽取顺序与 :func:`table_diffevo.update.evolve_step` 保持一致：
    participation、逐属性独立复制初值和 mutation 都从该流抽取。额外 Gibbs 微步只
    使用独立的 ``gibbs_rng``，因此增加 sweep 不会错位后续 donor、复制或变异随机
    流。``n_sweeps=0`` 不构造因子、不消费 ``gibbs_rng``，并精确退化到现有定向
    ``evolve_step``。非零 sweep 默认使用与现有方向核一致的 ``[-30, 30]`` 条件
    logit 数值护栏；可通过 ``gibbs_logit_clip=None`` 显式关闭。显式传入由
    :func:`compile_mask_workload` 创建的 ``compiled_workload`` 时，只对参与且
    recipient/donor 不同的记录批量评价查询条件；不传则保留旧逐行构造路径。

    返回的诊断只记录公开生成过程的工作量、实际条件 logit、概率和墙钟，不评价
    loss，也不参与更新决策。``direction_logit_clip`` 控制共同的独立定向初始
    mask，``gibbs_logit_clip`` 只控制后续 factor 条件 Gibbs；两者分别显式记录。
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
    direction_clip = validate_direction_logit_clip(direction_logit_clip)
    sweeps = _validate_nonnegative_integer(n_sweeps, "n_sweeps")
    maximum_order = _validate_nonnegative_integer(
        max_factor_order, "max_factor_order"
    )
    if maximum_order > 8:
        raise ValueError("max_factor_order 不得超过绝对护栏 8")
    compiled_validation_elapsed = 0.0
    if len(current) != len(donors):
        raise ValueError(
            f"current 行数 ({len(current)}) 与 donors 行数 "
            f"({len(donors)}) 不一致"
        )
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng 必须是 np.random.Generator")
    if sweeps > 0:
        _validate_open_probability(eta, "eta")
        clip = _validate_logit_clip(gibbs_logit_clip)
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
        if compiled_workload is not None:
            validation_start = time.perf_counter()
            _validate_compiled_workload_match(
                compiled_workload,
                schema,
                queries,
                maximum_order,
            )
            compiled_validation_elapsed = (
                time.perf_counter() - validation_start
            )
    else:
        # 0 sweep 不使用查询或残差，保留与既有 evolve_step 相同的最小依赖边界。
        residual_values = residual
        weight_values = weights
        clip = gibbs_logit_clip

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
                logit_clip=direction_clip,
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
    factor_model_builds = 0
    condition_evaluation_batches = 0
    conditional_logit_diagnostics = {
        "condition_count": 0,
        "raw_logit_min": None,
        "raw_logit_max": None,
        "raw_logit_abs_max": 0.0,
        "raw_logit_abs_max_condition": None,
        "logit_clip": float(clip) if clip is not None else None,
        "clip_hit_count": 0,
        "clip_hit_conditions": [],
        "conditional_probability_min": None,
        "conditional_probability_max": None,
        "minimum_binary_outcome_probability": None,
        "conditional_entropy_sum": 0.0,
        "all_finite": True,
        "all_conditionals_bidirectional": True,
    }
    if sweeps > 0:
        gibbs_rows = np.flatnonzero(
            participate & np.any(differs, axis=1)
        )
        if compiled_workload is not None and len(gibbs_rows) > 0:
            build_start = time.perf_counter()
            prepared = _prepare_sparse_mask_energy_batch(
                current_reset.iloc[gibbs_rows],
                donors_reset.iloc[gibbs_rows],
                schema,
                compiled_workload,
                residual_values,
                weights=weight_values,
            )
            factor_build_elapsed += time.perf_counter() - build_start
            condition_evaluation_batches = 1
        else:
            prepared = None

        for batch_index, row_index in enumerate(gibbs_rows):
            active_indices = np.flatnonzero(differs[row_index])
            n_active = len(active_indices)
            build_start = time.perf_counter()
            if prepared is None:
                model = build_sparse_mask_energy(
                    current_reset.iloc[[row_index]],
                    donors_reset.iloc[[row_index]],
                    schema,
                    queries,
                    residual_values,
                    weights=weight_values,
                    max_factor_order=maximum_order,
                )
            else:
                model = _build_sparse_mask_energy_from_batch(
                    prepared, batch_index
                )
            factor_build_elapsed += time.perf_counter() - build_start
            factor_model_builds += 1
            if not np.array_equal(
                model.active_attribute_indices, active_indices
            ):
                raise RuntimeError("因子模型的活跃属性顺序与更新 mask 不一致")

            def observe_condition(variable, raw_logit, probability):
                diagnostics = conditional_logit_diagnostics
                if not np.isfinite(raw_logit) or not np.isfinite(probability):
                    diagnostics["all_finite"] = False
                    raise ValueError("实际 Gibbs 条件 logit/概率不是有限值")
                attribute_index = int(active_indices[variable])
                context = {
                    "microstep_within_round": int(
                        diagnostics["condition_count"]
                    ),
                    "row": int(row_index),
                    "local_variable": int(variable),
                    "attribute_index": attribute_index,
                    "attribute": attr_names[attribute_index],
                    "raw_logit": float(raw_logit),
                }
                absolute = abs(raw_logit)
                if (
                    diagnostics["raw_logit_abs_max_condition"] is None
                    or absolute > diagnostics["raw_logit_abs_max"]
                ):
                    diagnostics["raw_logit_abs_max"] = float(absolute)
                    diagnostics["raw_logit_abs_max_condition"] = context
                diagnostics["condition_count"] += 1
                diagnostics["raw_logit_min"] = (
                    float(raw_logit)
                    if diagnostics["raw_logit_min"] is None
                    else min(diagnostics["raw_logit_min"], raw_logit)
                )
                diagnostics["raw_logit_max"] = (
                    float(raw_logit)
                    if diagnostics["raw_logit_max"] is None
                    else max(diagnostics["raw_logit_max"], raw_logit)
                )
                diagnostics["conditional_probability_min"] = (
                    float(probability)
                    if diagnostics["conditional_probability_min"] is None
                    else min(
                        diagnostics["conditional_probability_min"],
                        probability,
                    )
                )
                diagnostics["conditional_probability_max"] = (
                    float(probability)
                    if diagnostics["conditional_probability_max"] is None
                    else max(
                        diagnostics["conditional_probability_max"],
                        probability,
                    )
                )
                minimum_outcome = min(probability, 1.0 - probability)
                diagnostics["minimum_binary_outcome_probability"] = (
                    float(minimum_outcome)
                    if diagnostics[
                        "minimum_binary_outcome_probability"
                    ] is None
                    else min(
                        diagnostics[
                            "minimum_binary_outcome_probability"
                        ],
                        minimum_outcome,
                    )
                )
                if 0.0 < probability < 1.0:
                    diagnostics["conditional_entropy_sum"] += float(
                        -probability * np.log(probability)
                        - (1.0 - probability) * np.log1p(-probability)
                    )
                else:
                    diagnostics["all_conditionals_bidirectional"] = False
                if clip is not None and absolute >= clip:
                    diagnostics["clip_hit_count"] += 1
                    diagnostics["clip_hit_conditions"].append(context)

            sample_start = time.perf_counter()
            copy_masks[row_index, active_indices] = random_scan_gibbs_mask(
                model,
                copy_masks[row_index, active_indices],
                eta,
                strength,
                sweeps * n_active,
                gibbs_rng,
                logit_clip=clip,
                condition_observer=observe_condition,
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
        "direction_logit_clip": direction_clip,
        "gibbs_logit_clip": clip,
        "active_gibbs_rows": active_gibbs_rows,
        "active_blocks": active_blocks,
        "factor_count": factor_count,
        "factor_table_entries": factor_table_entries,
        "gibbs_microsteps": gibbs_microsteps,
        "factor_builder": (
            "not_used"
            if sweeps == 0
            else (
                "compiled_batch"
                if compiled_workload is not None
                else "legacy_rowwise"
            )
        ),
        "factor_model_builds": factor_model_builds,
        "condition_evaluation_batches": condition_evaluation_batches,
        "factor_conditional_logit_diagnostics": {
            **conditional_logit_diagnostics,
            "clip_hit_rate": (
                float(
                    conditional_logit_diagnostics["clip_hit_count"]
                    / conditional_logit_diagnostics["condition_count"]
                )
                if conditional_logit_diagnostics["condition_count"]
                else 0.0
            ),
            "raw_logit_strictly_inside_clip": bool(
                clip is None
                or conditional_logit_diagnostics["condition_count"] == 0
                or conditional_logit_diagnostics["raw_logit_abs_max"]
                < clip
            ),
            "conditional_entropy_mean": (
                float(
                    conditional_logit_diagnostics[
                        "conditional_entropy_sum"
                    ]
                    / conditional_logit_diagnostics["condition_count"]
                )
                if conditional_logit_diagnostics["condition_count"]
                else None
            ),
        },
        "compiled_unique_conditions": (
            compiled_workload.n_unique_conditions
            if sweeps > 0 and compiled_workload is not None
            else 0
        ),
        "compiled_validation_elapsed_sec": compiled_validation_elapsed,
        "factor_build_elapsed_sec": factor_build_elapsed,
        "gibbs_sample_elapsed_sec": gibbs_sample_elapsed,
    }
    return proposal, diagnostics
