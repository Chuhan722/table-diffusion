"""剩余查询缺口感知的全局绝对误差 Gibbs（吉布斯）复制核。

本模块只在已经固定的当前表、供体、参与行和初始复制开关上重抽复制开关。
它没有候选接受、拒绝、重试、回滚或赢家选择；完成固定数量的微步后，最终
开关表会被一次性物化成唯一复制表。

目标函数是全部已测查询的相对绝对误差：

``E = mean_j(abs(target_j - count_j) / max(target_j, floor))``。

每个条件微步精确维护同一行内多属性的合取作用，以及多行先求总查询计数再
计算误差的共同作用。实现固定使用 NumPy 双精度浮点数。
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from table_diffevo.queries import eval_condition, evaluate_table
from table_diffevo.schema import Schema


DEFAULT_GAP_L1_FLOOR = 8.0
DEFAULT_GAP_L1_ETA = 0.5
DEFAULT_GAP_L1_STRENGTH = 2.0
DEFAULT_GAP_L1_SWEEPS = 8
DEFAULT_GAP_L1_LOGIT_CLIP = 30.0
TRACE_FORMAT = "issue53_gap_l1_microstep_trace_le_v1"


@dataclass(frozen=True)
class _CompiledGapCondition:
    attribute_index: int
    condition: Dict[str, Any]


@dataclass(frozen=True)
class CompiledGapL1Workload:
    """只缓存模式和有序查询的静态结构，不缓存表、目标或结果。"""

    attribute_names: Tuple[str, ...]
    n_queries: int
    conditions: Tuple[_CompiledGapCondition, ...]
    query_condition_indices: Tuple[Tuple[int, ...], ...]
    query_indices_by_attribute: Tuple[np.ndarray, ...]
    condition_indices_by_attribute_query: Tuple[
        Tuple[Tuple[int, ...], ...], ...
    ]
    signature: Tuple[Any, ...]


@dataclass
class _GapPlan:
    compiled: CompiledGapL1Workload
    active_rows: np.ndarray
    active_coordinates: np.ndarray
    row_lookup: np.ndarray
    current_condition_truth: np.ndarray
    donor_condition_truth: np.ndarray
    current_row_indicators: np.ndarray
    row_indicators: np.ndarray
    failure_counts: np.ndarray
    current_attribute_failures: Tuple[Tuple[np.ndarray, ...], ...]
    donor_attribute_failures: Tuple[Tuple[np.ndarray, ...], ...]
    current_counts: np.ndarray
    plan_counts: np.ndarray
    target: np.ndarray
    denominators: np.ndarray
    error_terms: np.ndarray
    error_sum: float
    mask: np.ndarray


def _require_finite_vector(
    values: Any,
    expected_length: int,
    name: str,
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (expected_length,) or raw.dtype.kind not in "iuf":
        raise ValueError(
            f"{name} 必须是长度 {expected_length} 的有限数值向量"
        )
    result = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} 必须是长度 {expected_length} 的有限数值向量")
    return result


def _require_positive_finite(value: Any, name: str) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} 必须是正有限数值")
    return float(value)


def _require_open_probability(value: Any, name: str) -> float:
    result = _require_positive_finite(value, name)
    if result >= 1.0:
        raise ValueError(f"{name} 必须严格位于 (0, 1)")
    return result


def _require_nonnegative_integer(value: Any, name: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise ValueError(f"{name} 必须是非负整数")
    return int(value)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_freeze_value(x) for x in value))
    return (type(value).__module__, type(value).__qualname__, repr(value))


def _query_signature(
    schema: Schema,
    queries: List[Dict[str, Any]],
) -> Tuple[Tuple[str, ...], Tuple[Any, ...]]:
    attributes = tuple(schema.attribute_names())
    if len(set(attributes)) != len(attributes):
        raise ValueError("schema 属性名不得重复")
    positions = {name: index for index, name in enumerate(attributes)}
    query_signatures = []
    for query_index, query in enumerate(queries):
        if not isinstance(query, dict) or not isinstance(
            query.get("conditions"), list
        ):
            raise ValueError(f"queries[{query_index}] 的 conditions 必须是列表")
        condition_signatures = []
        for condition_index, condition in enumerate(query["conditions"]):
            if not isinstance(condition, dict):
                raise ValueError(
                    f"queries[{query_index}].conditions[{condition_index}] "
                    "必须是字典"
                )
            attribute = condition.get("attribute")
            if attribute not in positions:
                raise ValueError(
                    f"queries[{query_index}] 包含未知属性 {attribute!r}"
                )
            operator = condition.get("operator")
            if operator in ("==", ">="):
                if "value" not in condition:
                    raise ValueError(f"{operator} 条件缺少 value")
                operands = (_freeze_value(condition["value"]),)
            elif operator == "between":
                if "lower" not in condition or "upper" not in condition:
                    raise ValueError("between 条件缺少 lower/upper")
                operands = (
                    _freeze_value(condition["lower"]),
                    _freeze_value(condition["upper"]),
                )
            else:
                raise ValueError(f"不支持的操作符：{operator!r}")
            condition_signatures.append(
                (positions[attribute], operator, *operands)
            )
        query_signatures.append(tuple(condition_signatures))
    return attributes, tuple(query_signatures)


def compile_gap_l1_workload(
    schema: Schema,
    queries: List[Dict[str, Any]],
) -> CompiledGapL1Workload:
    """预编译查询到属性的静态邻接关系。"""

    attributes, signature = _query_signature(schema, queries)
    positions = {name: index for index, name in enumerate(attributes)}
    conditions: list[_CompiledGapCondition] = []
    query_condition_indices: list[Tuple[int, ...]] = []
    per_attribute_query_conditions: list[dict[int, list[int]]] = [
        {} for _ in attributes
    ]
    for query_index, query in enumerate(queries):
        indices = []
        for condition in query["conditions"]:
            condition_index = len(conditions)
            attribute_index = positions[condition["attribute"]]
            conditions.append(_CompiledGapCondition(
                attribute_index=attribute_index,
                condition=dict(condition),
            ))
            indices.append(condition_index)
            per_attribute_query_conditions[attribute_index].setdefault(
                query_index, []
            ).append(condition_index)
        query_condition_indices.append(tuple(indices))

    query_indices_by_attribute: list[np.ndarray] = []
    condition_indices_by_attribute_query = []
    for mapping in per_attribute_query_conditions:
        query_indices = np.asarray(sorted(mapping), dtype=np.intp)
        query_indices.setflags(write=False)
        query_indices_by_attribute.append(query_indices)
        condition_indices_by_attribute_query.append(tuple(
            tuple(mapping[int(query_index)]) for query_index in query_indices
        ))
    return CompiledGapL1Workload(
        attribute_names=attributes,
        n_queries=len(queries),
        conditions=tuple(conditions),
        query_condition_indices=tuple(query_condition_indices),
        query_indices_by_attribute=tuple(query_indices_by_attribute),
        condition_indices_by_attribute_query=tuple(
            condition_indices_by_attribute_query
        ),
        signature=signature,
    )


def _validate_compiled(
    compiled: CompiledGapL1Workload,
    schema: Schema,
    queries: List[Dict[str, Any]],
) -> None:
    if not isinstance(compiled, CompiledGapL1Workload):
        raise ValueError("compiled_workload 必须由 compile_gap_l1_workload 创建")
    attributes, signature = _query_signature(schema, queries)
    if compiled.attribute_names != attributes or compiled.signature != signature:
        raise ValueError("compiled_workload 与当前 schema/queries 不匹配")


def normalized_gap_l1_error(
    query_counts: Any,
    target: Any,
    *,
    floor: float = DEFAULT_GAP_L1_FLOOR,
) -> float:
    """计算协议定义的相对绝对查询误差。"""

    floor_value = _require_positive_finite(floor, "floor")
    raw_counts = np.asarray(query_counts)
    raw_target = np.asarray(target)
    if raw_counts.ndim != 1 or raw_counts.dtype.kind not in "iuf":
        raise ValueError("query_counts 必须是一维有限数值向量")
    counts = raw_counts.astype(np.float64, copy=False)
    targets = _require_finite_vector(raw_target, len(counts), "target")
    if not np.all(np.isfinite(counts)):
        raise ValueError("query_counts 必须是一维有限数值向量")
    if len(counts) == 0:
        raise ValueError("至少需要一个查询")
    denominators = np.maximum(targets, floor_value)
    if np.any(denominators <= 0.0):
        raise ValueError("max(target, floor) 必须为正")
    return float(np.mean(np.abs(targets - counts) / denominators))


def _evaluate_conditions(
    frame: pd.DataFrame,
    compiled: CompiledGapL1Workload,
) -> np.ndarray:
    truth = np.empty((len(frame), len(compiled.conditions)), dtype=bool)
    for condition_index, condition in enumerate(compiled.conditions):
        truth[:, condition_index] = eval_condition(
            frame, condition.condition
        ).to_numpy(dtype=bool)
    return truth


def _prepare_plan(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    participate: Any,
    initial_mask: Any,
    *,
    floor: float,
    compiled_workload: Optional[CompiledGapL1Workload],
) -> _GapPlan:
    if len(current) != len(donors):
        raise ValueError("current 与 donors 行数必须一致")
    compiled = (
        compile_gap_l1_workload(schema, queries)
        if compiled_workload is None
        else compiled_workload
    )
    _validate_compiled(compiled, schema, queries)
    n_rows = len(current)
    n_attributes = len(compiled.attribute_names)
    current_reset = current.reset_index(drop=True)
    donor_reset = donors.reset_index(drop=True)
    missing = [
        name
        for name in compiled.attribute_names
        if name not in current_reset or name not in donor_reset
    ]
    if missing:
        raise ValueError(f"current/donors 缺少属性：{missing}")

    raw_participate = np.asarray(participate)
    if raw_participate.shape != (n_rows,) or raw_participate.dtype.kind not in "biuf":
        raise ValueError("participate 必须是与表等长的 0/1 向量")
    if np.any((raw_participate != 0) & (raw_participate != 1)):
        raise ValueError("participate 必须是与表等长的 0/1 向量")
    participate_bool = raw_participate.astype(bool, copy=False)

    raw_mask = np.asarray(initial_mask)
    if raw_mask.shape != (n_rows, n_attributes) or raw_mask.dtype.kind not in "biuf":
        raise ValueError(
            f"initial_mask 必须是 shape ({n_rows}, {n_attributes}) 的 0/1 数组"
        )
    if np.any((raw_mask != 0) & (raw_mask != 1)):
        raise ValueError("initial_mask 必须只包含 0/1")
    mask = raw_mask.astype(bool, copy=True)

    selected_columns = list(compiled.attribute_names)
    current_values = current_reset.loc[:, selected_columns].to_numpy()
    donor_values = donor_reset.loc[:, selected_columns].to_numpy()
    differs = current_values != donor_values
    active = participate_bool[:, None] & differs
    if np.any(mask & ~active):
        raise ValueError("initial_mask 在非活跃开关上必须为 0")
    active_coordinates = np.argwhere(active).astype(np.intp, copy=False)
    active_rows = np.flatnonzero(np.any(active, axis=1)).astype(
        np.intp, copy=False
    )
    row_lookup = np.full(n_rows, -1, dtype=np.intp)
    row_lookup[active_rows] = np.arange(len(active_rows), dtype=np.intp)

    counts = _require_finite_vector(
        current_counts, compiled.n_queries, "current_counts"
    )
    if compiled.n_queries == 0:
        raise ValueError("至少需要一个查询")
    targets = _require_finite_vector(target, compiled.n_queries, "target")
    floor_value = _require_positive_finite(floor, "floor")
    denominators = np.maximum(targets, floor_value)
    if np.any(denominators <= 0.0):
        raise ValueError("max(target, floor) 必须为正")

    if len(active_rows):
        pair = pd.concat(
            [current_reset.iloc[active_rows], donor_reset.iloc[active_rows]],
            ignore_index=True,
        )
        pair_truth = _evaluate_conditions(pair, compiled)
        current_truth = pair_truth[: len(active_rows)]
        donor_truth = pair_truth[len(active_rows) :]
    else:
        current_truth = np.empty((0, len(compiled.conditions)), dtype=bool)
        donor_truth = current_truth.copy()

    n_active_rows = len(active_rows)
    failure_counts = np.zeros(
        (n_active_rows, compiled.n_queries), dtype=np.int16
    )
    current_attribute_failures: list[Tuple[np.ndarray, ...]] = []
    donor_attribute_failures: list[Tuple[np.ndarray, ...]] = []
    for attribute_index in range(n_attributes):
        current_by_query = []
        donor_by_query = []
        for condition_indices in compiled.condition_indices_by_attribute_query[
            attribute_index
        ]:
            indices = np.asarray(condition_indices, dtype=np.intp)
            current_by_query.append(
                np.sum(~current_truth[:, indices], axis=1, dtype=np.int16)
            )
            donor_by_query.append(
                np.sum(~donor_truth[:, indices], axis=1, dtype=np.int16)
            )
        current_attribute_failures.append(tuple(current_by_query))
        donor_attribute_failures.append(tuple(donor_by_query))

    for query_index, condition_indices in enumerate(
        compiled.query_condition_indices
    ):
        if not condition_indices:
            continue
        indices = np.asarray(condition_indices, dtype=np.intp)
        failure_counts[:, query_index] = np.sum(
            ~current_truth[:, indices], axis=1, dtype=np.int16
        )
    current_row_indicators = failure_counts == 0

    for local_row, row_index in enumerate(active_rows):
        selected_attributes = np.flatnonzero(mask[row_index])
        for attribute_index_raw in selected_attributes:
            attribute_index = int(attribute_index_raw)
            query_indices = compiled.query_indices_by_attribute[attribute_index]
            if len(query_indices) == 0:
                continue
            current_failures = np.asarray([
                values[local_row]
                for values in current_attribute_failures[attribute_index]
            ], dtype=np.int16)
            donor_failures = np.asarray([
                values[local_row]
                for values in donor_attribute_failures[attribute_index]
            ], dtype=np.int16)
            failure_counts[local_row, query_indices] += (
                donor_failures - current_failures
            )

    row_indicators = failure_counts == 0
    plan_counts = counts.copy()
    if n_active_rows:
        plan_counts += np.sum(
            row_indicators.astype(np.int64)
            - current_row_indicators.astype(np.int64),
            axis=0,
            dtype=np.int64,
        )
    error_terms = np.abs(targets - plan_counts) / denominators
    error_sum = float(np.sum(error_terms, dtype=np.float64))
    return _GapPlan(
        compiled=compiled,
        active_rows=active_rows,
        active_coordinates=active_coordinates,
        row_lookup=row_lookup,
        current_condition_truth=current_truth,
        donor_condition_truth=donor_truth,
        current_row_indicators=current_row_indicators,
        row_indicators=row_indicators,
        failure_counts=failure_counts,
        current_attribute_failures=tuple(current_attribute_failures),
        donor_attribute_failures=tuple(donor_attribute_failures),
        current_counts=counts,
        plan_counts=plan_counts,
        target=targets,
        denominators=denominators,
        error_terms=error_terms,
        error_sum=error_sum,
        mask=mask,
    )


def _attribute_failures(
    values: Tuple[np.ndarray, ...],
    local_row: int,
) -> np.ndarray:
    return np.asarray(
        [per_query[local_row] for per_query in values], dtype=np.int16
    )


def _condition_pair(
    plan: _GapPlan,
    row_index: int,
    attribute_index: int,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 E0/E1 及两侧局部状态，不改变计划。"""

    local_row = int(plan.row_lookup[row_index])
    if local_row < 0:
        raise ValueError("待评价坐标不在活跃参与行中")
    query_indices = plan.compiled.query_indices_by_attribute[attribute_index]
    if len(query_indices) == 0:
        empty_failures = np.zeros(0, dtype=np.int16)
        empty_indicators = np.zeros(0, dtype=bool)
        value = plan.error_sum / plan.compiled.n_queries
        return (
            value,
            value,
            empty_failures,
            empty_failures,
            empty_indicators,
            empty_indicators,
        )
    current_failures = _attribute_failures(
        plan.current_attribute_failures[attribute_index], local_row
    )
    donor_failures = _attribute_failures(
        plan.donor_attribute_failures[attribute_index], local_row
    )
    old_failures = (
        donor_failures if plan.mask[row_index, attribute_index]
        else current_failures
    )
    base_failures = plan.failure_counts[local_row, query_indices] - old_failures
    failures0 = base_failures + current_failures
    failures1 = base_failures + donor_failures
    indicators0 = failures0 == 0
    indicators1 = failures1 == 0
    old_indicators = plan.row_indicators[local_row, query_indices]
    counts0 = (
        plan.plan_counts[query_indices]
        + indicators0.astype(np.int8)
        - old_indicators.astype(np.int8)
    )
    counts1 = (
        plan.plan_counts[query_indices]
        + indicators1.astype(np.int8)
        - old_indicators.astype(np.int8)
    )
    terms0 = (
        np.abs(plan.target[query_indices] - counts0)
        / plan.denominators[query_indices]
    )
    terms1 = (
        np.abs(plan.target[query_indices] - counts1)
        / plan.denominators[query_indices]
    )
    old_terms = plan.error_terms[query_indices]
    sum0 = plan.error_sum - float(np.sum(old_terms)) + float(np.sum(terms0))
    sum1 = plan.error_sum - float(np.sum(old_terms)) + float(np.sum(terms1))
    return (
        sum0 / plan.compiled.n_queries,
        sum1 / plan.compiled.n_queries,
        failures0,
        failures1,
        indicators0,
        indicators1,
    )


def _set_coordinate(
    plan: _GapPlan,
    row_index: int,
    attribute_index: int,
    selected: bool,
    failures: np.ndarray,
    indicators: np.ndarray,
) -> None:
    old = bool(plan.mask[row_index, attribute_index])
    if selected == old:
        return
    local_row = int(plan.row_lookup[row_index])
    query_indices = plan.compiled.query_indices_by_attribute[attribute_index]
    if len(query_indices):
        old_indicators = plan.row_indicators[local_row, query_indices]
        plan.plan_counts[query_indices] += (
            indicators.astype(np.int8) - old_indicators.astype(np.int8)
        )
        plan.failure_counts[local_row, query_indices] = failures
        plan.row_indicators[local_row, query_indices] = indicators
        new_terms = (
            np.abs(plan.target[query_indices] - plan.plan_counts[query_indices])
            / plan.denominators[query_indices]
        )
        plan.error_sum += float(
            np.sum(new_terms, dtype=np.float64)
            - np.sum(plan.error_terms[query_indices], dtype=np.float64)
        )
        plan.error_terms[query_indices] = new_terms
    plan.mask[row_index, attribute_index] = selected


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + np.exp(-value)))
    exponential = float(np.exp(value))
    return exponential / (1.0 + exponential)


def gap_l1_conditional_probability(
    score: float,
    reference_scale: float,
    *,
    eta: float = DEFAULT_GAP_L1_ETA,
    strength: float = DEFAULT_GAP_L1_STRENGTH,
    logit_clip: float = DEFAULT_GAP_L1_LOGIT_CLIP,
) -> Tuple[float, float, float, bool]:
    """把 ``E0-E1`` 转成软条件概率。

    返回实际概率、截断前对数几率、截断后对数几率和是否真正发生截断。
    """

    if not np.isfinite(score):
        raise ValueError("score 必须是有限数值")
    scale = _require_positive_finite(reference_scale, "reference_scale")
    baseline = _require_open_probability(eta, "eta")
    beta = _require_positive_finite(strength, "strength")
    clip = _require_positive_finite(logit_clip, "logit_clip")
    base_logit = float(np.log(baseline) - np.log1p(-baseline))
    raw_logit = float(base_logit + beta * float(score) / scale)
    if not np.isfinite(raw_logit):
        raise ValueError("条件对数几率不是有限值")
    effective_logit = float(np.clip(raw_logit, -clip, clip))
    probability = _stable_sigmoid(effective_logit)
    if not 0.0 < probability < 1.0:
        raise RuntimeError("有限条件概率丢失双向严格正支持")
    return probability, raw_logit, effective_logit, raw_logit != effective_logit


def _distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "q25": None,
            "median": None,
            "q75": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _probability_bin(probability: float) -> str:
    if probability < 0.001:
        return "open_0_0p001"
    if probability < 0.01:
        return "closed_0p001_open_0p01"
    if probability <= 0.99:
        return "closed_0p01_0p99"
    if probability <= 0.999:
        return "open_0p99_closed_0p999"
    return "open_0p999_1"


def _update_trace(
    digest: Any,
    *,
    step: int,
    row_index: int,
    attribute_index: int,
    e0: float,
    e1: float,
    score: float,
    normalized_score: float,
    raw_logit: float,
    probability: float,
    random_roll: float,
    before: bool,
    after: bool,
    clipped: bool,
) -> None:
    digest.update(struct.pack(
        "<qqqddddddd???",
        step,
        row_index,
        attribute_index,
        e0,
        e1,
        score,
        normalized_score,
        raw_logit,
        probability,
        random_roll,
        before,
        after,
        clipped,
    ))


def _materialize_copy_table(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    attribute_names: Sequence[str],
    mask: np.ndarray,
) -> pd.DataFrame:
    result = current.reset_index(drop=True).copy(deep=True)
    donor_reset = donors.reset_index(drop=True)
    for attribute_index, attribute in enumerate(attribute_names):
        selected = mask[:, attribute_index]
        if np.any(selected):
            values = result[attribute].to_numpy().copy()
            values[selected] = donor_reset[attribute].to_numpy()[selected]
            result[attribute] = values
    return result


def isolated_gap_l1_scores(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    floor: float = DEFAULT_GAP_L1_FLOOR,
    compiled_workload: Optional[CompiledGapL1Workload] = None,
    exact_target_numerators: Optional[Any] = None,
    exact_target_denominator: Optional[int] = None,
) -> Dict[str, Any]:
    """计算其他开关全为 0 时所有不同值行—属性的孤立分数。

    定尺调用方可同时提供目标的共同整数分母表示。此时先用独立整数公共尺度
    判定数学上精确为零的分数，再把这些位置显式设为 ``0.0``；非零分数仍按
    冻结的 NumPy 双精度路径取值。这避免理论抵消被浮点尾差误收进 RMS。
    """

    attributes = schema.attribute_names()
    plan = _prepare_plan(
        current,
        donors,
        schema,
        queries,
        target,
        current_counts,
        np.ones(len(current), dtype=bool),
        np.zeros((len(current), len(attributes)), dtype=bool),
        floor=floor,
        compiled_workload=compiled_workload,
    )
    exact_zero_spec = None
    if (exact_target_numerators is None) != (
        exact_target_denominator is None
    ):
        raise ValueError(
            "exact_target_numerators 与 exact_target_denominator 必须同时提供"
        )
    if exact_target_numerators is not None:
        if not float(floor).is_integer():
            raise ValueError("精确目标零分数判定要求整数 floor")
        raw_numerators = np.asarray(exact_target_numerators)
        if (
            raw_numerators.shape != (len(queries),)
            or raw_numerators.dtype.kind not in "iu"
            or raw_numerators.dtype.kind == "b"
            or isinstance(exact_target_denominator, (bool, np.bool_))
            or not isinstance(
                exact_target_denominator, (int, np.integer)
            )
            or exact_target_denominator <= 0
        ):
            raise ValueError("精确目标必须是整数向量和正整数共同分母")
        numerator_values = [int(value) for value in raw_numerators]
        denominator_value = int(exact_target_denominator)
        raw_denominators = [
            max(
                numerator,
                int(floor) * denominator_value,
            )
            for numerator in numerator_values
        ]
        divisors = [
            math.gcd(
                math.gcd(abs(numerator), denominator_value),
                denominator,
            )
            for numerator, denominator in zip(
                numerator_values, raw_denominators
            )
        ]
        reduced_denominators = [
            denominator // divisor
            for denominator, divisor in zip(raw_denominators, divisors)
        ]
        common_multiple = math.lcm(*reduced_denominators)
        exact_zero_spec = (
            numerator_values,
            denominator_value,
            divisors,
            [
                common_multiple // denominator
                for denominator in reduced_denominators
            ],
        )

    coordinates: list[Tuple[int, int]] = []
    scores: list[float] = []
    for row_index_raw, attribute_index_raw in plan.active_coordinates:
        row_index = int(row_index_raw)
        attribute_index = int(attribute_index_raw)
        e0, e1, _, _, indicators0, indicators1 = _condition_pair(
            plan, row_index, attribute_index
        )
        score = float(e0 - e1)
        if exact_zero_spec is not None:
            query_indices = plan.compiled.query_indices_by_attribute[
                attribute_index
            ]
            local_row = int(plan.row_lookup[row_index])
            old_indicators = plan.row_indicators[local_row, query_indices]
            counts0 = (
                plan.plan_counts[query_indices]
                + indicators0.astype(np.int8)
                - old_indicators.astype(np.int8)
            ).astype(np.int64)
            counts1 = (
                plan.plan_counts[query_indices]
                + indicators1.astype(np.int8)
                - old_indicators.astype(np.int8)
            ).astype(np.int64)
            numerators, exact_denominator, divisors, weights = exact_zero_spec
            exact_units = 0
            for local_index, query_index_raw in enumerate(query_indices):
                query_index = int(query_index_raw)
                numerator = numerators[query_index]
                divisor = divisors[query_index]
                weight = weights[query_index]
                residual0 = abs(
                    numerator - int(counts0[local_index]) * exact_denominator
                )
                residual1 = abs(
                    numerator - int(counts1[local_index]) * exact_denominator
                )
                exact_units += (
                    residual0 // divisor - residual1 // divisor
                ) * weight
            if exact_units == 0:
                score = 0.0
        coordinates.append((row_index, attribute_index))
        scores.append(score)
    return {
        "coordinates": np.asarray(coordinates, dtype=np.intp).reshape(-1, 2),
        "scores": np.asarray(scores, dtype=np.float64),
        "current_error": float(plan.error_sum / plan.compiled.n_queries),
    }


def evaluate_gap_l1_condition(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participate: Any,
    mask: Any,
    row_index: int,
    attribute_index: int,
    reference_scale: float,
    eta: float = DEFAULT_GAP_L1_ETA,
    strength: float = DEFAULT_GAP_L1_STRENGTH,
    floor: float = DEFAULT_GAP_L1_FLOOR,
    logit_clip: float = DEFAULT_GAP_L1_LOGIT_CLIP,
    compiled_workload: Optional[CompiledGapL1Workload] = None,
) -> Dict[str, Any]:
    """只读评价一个活跃开关的当前完整上下文条件式。"""

    if (
        isinstance(row_index, (bool, np.bool_))
        or not isinstance(row_index, (int, np.integer))
        or not 0 <= row_index < len(current)
    ):
        raise ValueError("row_index 超出表范围")
    attribute_count = len(schema.attribute_names())
    if (
        isinstance(attribute_index, (bool, np.bool_))
        or not isinstance(attribute_index, (int, np.integer))
        or not 0 <= attribute_index < attribute_count
    ):
        raise ValueError("attribute_index 超出属性范围")
    plan = _prepare_plan(
        current,
        donors,
        schema,
        queries,
        target,
        current_counts,
        participate,
        mask,
        floor=floor,
        compiled_workload=compiled_workload,
    )
    coordinates = {tuple(map(int, pair)) for pair in plan.active_coordinates}
    coordinate = (int(row_index), int(attribute_index))
    if coordinate not in coordinates:
        raise ValueError("指定开关不是参与且 current != donor 的活跃开关")
    e0, e1, *_ = _condition_pair(plan, *coordinate)
    score = float(e0 - e1)
    probability, raw_logit, logit, clipped = gap_l1_conditional_probability(
        score,
        reference_scale,
        eta=eta,
        strength=strength,
        logit_clip=logit_clip,
    )
    return {
        "e0": e0,
        "e1": e1,
        "score": score,
        "normalized_score": float(score / reference_scale),
        "raw_logit": raw_logit,
        "logit": logit,
        "probability": probability,
        "clipped": clipped,
    }


def stable_nonzero_rms(values: Any) -> Tuple[float, Dict[str, Any]]:
    """排除精确零后，用先缩放再平方的稳定算法计算均方根。"""

    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError("values 必须是一维有限数值向量")
    scores = raw.astype(np.float64, copy=False)
    if not np.all(np.isfinite(scores)):
        raise ValueError("values 必须是一维有限数值向量")
    nonzero = scores[scores != 0.0]
    zero_count = int(len(scores) - len(nonzero))
    if len(nonzero) == 0:
        return 0.0, {
            "total_count": int(len(scores)),
            "nonzero_count": 0,
            "zero_count": zero_count,
            "absolute_min": None,
            "absolute_q25": None,
            "absolute_median": None,
            "absolute_q75": None,
            "rms": 0.0,
            "absolute_max": None,
            "absolute_max_over_rms": None,
        }
    absolute = np.abs(nonzero)
    maximum = float(np.max(absolute))
    rms = float(maximum * np.sqrt(np.mean((nonzero / maximum) ** 2)))
    if not np.isfinite(rms) or rms <= 0.0:
        raise RuntimeError("非零孤立分数未得到正有限参考尺度")
    return rms, {
        "total_count": int(len(scores)),
        "nonzero_count": int(len(nonzero)),
        "zero_count": zero_count,
        "absolute_min": float(np.min(absolute)),
        "absolute_q25": float(np.quantile(absolute, 0.25)),
        "absolute_median": float(np.quantile(absolute, 0.5)),
        "absolute_q75": float(np.quantile(absolute, 0.75)),
        "rms": rms,
        "absolute_max": maximum,
        "absolute_max_over_rms": float(maximum / rms),
    }


def evolve_step_gap_l1_global(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participate: Any,
    initial_mask: Any,
    reference_scale: float,
    rng: np.random.Generator,
    n_sweeps: int = DEFAULT_GAP_L1_SWEEPS,
    eta: float = DEFAULT_GAP_L1_ETA,
    strength: float = DEFAULT_GAP_L1_STRENGTH,
    floor: float = DEFAULT_GAP_L1_FLOOR,
    logit_clip: float = DEFAULT_GAP_L1_LOGIT_CLIP,
    compiled_workload: Optional[CompiledGapL1Workload] = None,
    verify_full_recount: bool = True,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """执行全局剩余缺口绝对误差的固定随机扫描。

    ``n_sweeps * K`` 个微步在全部活跃行—属性开关中有放回抽坐标。函数不
    抽参与行、不生成初始开关，也不执行突变或任何生成后门控。
    """

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng 必须是 np.random.Generator")
    scale = _require_positive_finite(reference_scale, "reference_scale")
    sweeps = _require_nonnegative_integer(n_sweeps, "n_sweeps")
    started = time.perf_counter()
    plan = _prepare_plan(
        current,
        donors,
        schema,
        queries,
        target,
        current_counts,
        participate,
        initial_mask,
        floor=floor,
        compiled_workload=compiled_workload,
    )
    prepared_elapsed = time.perf_counter() - started
    k = len(plan.active_coordinates)
    microsteps = sweeps * k
    scores: list[float] = []
    normalized_scores: list[float] = []
    raw_logits: list[float] = []
    probabilities: list[float] = []
    entropies: list[float] = []
    probability_bins = {
        "open_0_0p001": 0,
        "closed_0p001_open_0p01": 0,
        "closed_0p01_0p99": 0,
        "open_0p99_closed_0p999": 0,
        "open_0p999_1": 0,
    }
    clip_hits = 0
    trace = hashlib.sha256()
    scan_started = time.perf_counter()
    for step in range(microsteps):
        coordinate_index = int(rng.integers(0, k))
        row_index = int(plan.active_coordinates[coordinate_index, 0])
        attribute_index = int(plan.active_coordinates[coordinate_index, 1])
        before = bool(plan.mask[row_index, attribute_index])
        (
            e0,
            e1,
            failures0,
            failures1,
            indicators0,
            indicators1,
        ) = _condition_pair(plan, row_index, attribute_index)
        score = float(e0 - e1)
        normalized = float(score / scale)
        probability, raw_logit, _, clipped = gap_l1_conditional_probability(
            score,
            scale,
            eta=eta,
            strength=strength,
            logit_clip=logit_clip,
        )
        random_roll = float(rng.random())
        after = bool(random_roll < probability)
        _set_coordinate(
            plan,
            row_index,
            attribute_index,
            after,
            failures1 if after else failures0,
            indicators1 if after else indicators0,
        )
        _update_trace(
            trace,
            step=step,
            row_index=row_index,
            attribute_index=attribute_index,
            e0=e0,
            e1=e1,
            score=score,
            normalized_score=normalized,
            raw_logit=raw_logit,
            probability=probability,
            random_roll=random_roll,
            before=before,
            after=after,
            clipped=clipped,
        )
        scores.append(score)
        normalized_scores.append(normalized)
        raw_logits.append(raw_logit)
        probabilities.append(probability)
        entropy = float(
            -probability * np.log(probability)
            - (1.0 - probability) * np.log1p(-probability)
        )
        entropies.append(entropy)
        probability_bins[_probability_bin(probability)] += 1
        clip_hits += int(clipped)
    scan_elapsed = time.perf_counter() - scan_started

    materialize_started = time.perf_counter()
    copy_table = _materialize_copy_table(
        current, donors, plan.compiled.attribute_names, plan.mask
    )
    materialize_elapsed = time.perf_counter() - materialize_started
    recount_elapsed = 0.0
    if verify_full_recount:
        recount_started = time.perf_counter()
        recounted = evaluate_table(copy_table, queries).astype(np.float64)
        recount_elapsed = time.perf_counter() - recount_started
        if not np.array_equal(recounted, plan.plan_counts):
            raise RuntimeError("增量查询计数与最终复制表完整复算不一致")

    minimum_outcome = (
        min(min(probability, 1.0 - probability) for probability in probabilities)
        if probabilities
        else None
    )
    diagnostics = {
        "kernel": "gap_l1_global_random_scan",
        "no_gate": True,
        "n_sweeps": sweeps,
        "active_switches_k": int(k),
        "gibbs_microsteps": int(microsteps),
        "conditional_error_evaluations": int(2 * microsteps),
        "query_indicator_increment_updates": int(microsteps),
        "reference_scale": scale,
        "trace_format": TRACE_FORMAT,
        "microstep_trace_sha256": trace.hexdigest(),
        "final_query_counts": plan.plan_counts.astype(np.int64).tolist(),
        "final_on_switches": int(np.sum(plan.mask)),
        "clip_hit_count": int(clip_hits),
        "nonfinite_condition_count": 0,
        "exact_zero_or_one_probability_count": 0,
        "minimum_binary_outcome_probability": (
            float(minimum_outcome) if minimum_outcome is not None else None
        ),
        "probability_bins": probability_bins,
        "score_distribution": _distribution(scores),
        "normalized_score_distribution": _distribution(normalized_scores),
        "raw_logit_distribution": _distribution(raw_logits),
        "probability_distribution": _distribution(probabilities),
        "binary_entropy_distribution": _distribution(entropies),
        "near_deterministic_count": int(sum(
            probability < 0.01 or probability > 0.99
            for probability in probabilities
        )),
        "prepared_elapsed_sec_diagnostic_only": prepared_elapsed,
        "scan_elapsed_sec_diagnostic_only": scan_elapsed,
        "materialize_elapsed_sec_diagnostic_only": materialize_elapsed,
        "full_recount_elapsed_sec_diagnostic_only": recount_elapsed,
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    if microsteps == 0 and trace.hexdigest() != hashlib.sha256(b"").hexdigest():
        raise RuntimeError("空扫描 trace 身份失败")
    return copy_table, plan.mask.copy(), diagnostics
