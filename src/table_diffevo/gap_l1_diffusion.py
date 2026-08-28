"""剩余查询缺口感知的全局绝对误差 Gibbs（吉布斯）复制核。

本模块只在已经固定的当前表、供体、参与行和初始复制开关上重抽复制开关。
它没有候选接受、拒绝、重试、回滚或赢家选择；完成固定数量的微步后，最终
开关表会被一次性物化成唯一复制表。

目标函数是全部已测查询的相对绝对误差：

``E = mean_j(abs(target_j - count_j) / max(target_j, floor))``。

每个条件微步精确维护同一行内多属性的合取作用，以及多行先求总查询计数再
计算误差的共同作用。历史端点使用 NumPy 双精度浮点数；可选 CUDA 后端复用
同一输入、随机流和输出契约，并固定使用显卡双精度浮点数。
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from table_diffevo.queries import eval_condition, evaluate_table
from table_diffevo.schema import Schema
from table_diffevo.vectorized_eval import (
    evaluate_conditions_vectorized,
)


DEFAULT_GAP_L1_FLOOR = 8.0
DEFAULT_GAP_L1_ETA = 0.5
DEFAULT_GAP_L1_STRENGTH = 2.0
DEFAULT_GAP_L1_SWEEPS = 8
DEFAULT_GAP_L1_LOGIT_CLIP = 30.0
TRACE_FORMAT = "issue53_gap_l1_microstep_trace_le_v1"
BATCH_EXECUTION_FORMAT = "issue53_gap_l1_batched_cuda_float64_v2"


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


@dataclass(frozen=True)
class _GapInputStructure:
    compiled: CompiledGapL1Workload
    current: pd.DataFrame
    donors: pd.DataFrame
    active_rows: np.ndarray
    active_coordinates: np.ndarray
    row_lookup: np.ndarray
    mask: np.ndarray
    counts: np.ndarray
    targets: np.ndarray
    denominators: np.ndarray


@dataclass
class _CudaGapPlan:
    torch: Any
    device: Any
    compiled: CompiledGapL1Workload
    active_rows: np.ndarray
    active_coordinates: np.ndarray
    row_lookup: np.ndarray
    query_indices_by_attribute: Tuple[Any, ...]
    current_row_indicators: Any
    row_indicators: Any
    failure_counts: Any
    current_attribute_failures: Tuple[Any, ...]
    donor_attribute_failures: Tuple[Any, ...]
    current_counts: Any
    plan_counts: Any
    target: Any
    denominators: Any
    error_terms: Any
    error_sum: Any
    mask: Any


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


def _prepare_input_structure(
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
) -> _GapInputStructure:
    """验证公共输入并构造与数值后端无关的稀疏活跃结构。"""

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
    if (
        raw_participate.shape != (n_rows,)
        or raw_participate.dtype.kind not in "biuf"
    ):
        raise ValueError("participate 必须是与表等长的 0/1 向量")
    if np.any((raw_participate != 0) & (raw_participate != 1)):
        raise ValueError("participate 必须是与表等长的 0/1 向量")
    participate_bool = raw_participate.astype(bool, copy=False)

    raw_mask = np.asarray(initial_mask)
    if (
        raw_mask.shape != (n_rows, n_attributes)
        or raw_mask.dtype.kind not in "biuf"
    ):
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
    return _GapInputStructure(
        compiled=compiled,
        current=current_reset,
        donors=donor_reset,
        active_rows=active_rows,
        active_coordinates=active_coordinates,
        row_lookup=row_lookup,
        mask=mask,
        counts=counts,
        targets=targets,
        denominators=denominators,
    )


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
    structure = _prepare_input_structure(
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
    compiled = structure.compiled
    current_reset = structure.current
    donor_reset = structure.donors
    active_coordinates = structure.active_coordinates
    active_rows = structure.active_rows
    row_lookup = structure.row_lookup
    mask = structure.mask
    counts = structure.counts
    targets = structure.targets
    denominators = structure.denominators
    n_attributes = len(compiled.attribute_names)

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


def _require_cuda_backend() -> Tuple[Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("缺口核 CUDA 后端需要 PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("请求缺口核 CUDA 后端但 CUDA 不可用")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("缺口核 CUDA 后端要求开启确定性算法")
    return torch, torch.device("cuda")


def _cuda_attribute_failures(
    truth: Any,
    condition_groups: Tuple[Tuple[int, ...], ...],
    *,
    torch: Any,
    device: Any,
) -> Any:
    n_rows = int(truth.shape[0])
    n_queries = len(condition_groups)
    result = torch.zeros(
        (n_rows, n_queries), dtype=torch.int32, device=device
    )
    if n_rows == 0 or n_queries == 0:
        return result
    for local_query, indices in enumerate(condition_groups):
        condition_t = torch.as_tensor(
            indices, dtype=torch.long, device=device
        )
        result[:, local_query] = (~truth[:, condition_t]).sum(
            dim=1, dtype=torch.int32
        )
    return result


def _prepare_cuda_plan(
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
) -> _CudaGapPlan:
    """在 CUDA 上建立缺口条件状态，不调用 NumPy 条件算术。"""

    torch, device = _require_cuda_backend()
    structure = _prepare_input_structure(
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
    compiled = structure.compiled
    n_active_rows = len(structure.active_rows)
    n_queries = compiled.n_queries

    if n_active_rows:
        pair = pd.concat(
            [
                structure.current.iloc[structure.active_rows],
                structure.donors.iloc[structure.active_rows],
            ],
            ignore_index=True,
        )
        pair_truth = evaluate_conditions_vectorized(
            pair,
            [item.condition for item in compiled.conditions],
            schema,
            device="cuda",
            return_tensor=True,
            float64=True,
        )
        current_truth = pair_truth[:n_active_rows]
        donor_truth = pair_truth[n_active_rows:]
    else:
        current_truth = torch.empty(
            (0, len(compiled.conditions)), dtype=torch.bool, device=device
        )
        donor_truth = current_truth.clone()

    failure_counts = torch.zeros(
        (n_active_rows, n_queries), dtype=torch.int32, device=device
    )
    if n_active_rows and compiled.conditions:
        for query_index, indices in enumerate(
            compiled.query_condition_indices
        ):
            if not indices:
                continue
            condition_t = torch.as_tensor(
                indices, dtype=torch.long, device=device
            )
            failure_counts[:, query_index] = (
                ~current_truth[:, condition_t]
            ).sum(dim=1, dtype=torch.int32)
    current_row_indicators = failure_counts == 0

    query_indices_by_attribute = tuple(
        torch.tensor(indices, dtype=torch.long, device=device)
        for indices in compiled.query_indices_by_attribute
    )
    current_attribute_failures = tuple(
        _cuda_attribute_failures(
            current_truth,
            compiled.condition_indices_by_attribute_query[attribute_index],
            torch=torch,
            device=device,
        )
        for attribute_index in range(len(compiled.attribute_names))
    )
    donor_attribute_failures = tuple(
        _cuda_attribute_failures(
            donor_truth,
            compiled.condition_indices_by_attribute_query[attribute_index],
            torch=torch,
            device=device,
        )
        for attribute_index in range(len(compiled.attribute_names))
    )

    mask = torch.as_tensor(
        structure.mask, dtype=torch.bool, device=device
    ).clone()
    if n_active_rows:
        active_rows_t = torch.as_tensor(
            structure.active_rows, dtype=torch.long, device=device
        )
        active_mask = mask.index_select(0, active_rows_t)
        for attribute_index, query_indices in enumerate(
            query_indices_by_attribute
        ):
            if query_indices.numel() == 0:
                continue
            selected = active_mask[:, attribute_index].to(torch.int32)
            delta = (
                donor_attribute_failures[attribute_index]
                - current_attribute_failures[attribute_index]
            )
            old = failure_counts.index_select(1, query_indices)
            failure_counts[:, query_indices] = (
                old + selected.unsqueeze(1) * delta
            )

    row_indicators = failure_counts == 0
    if not np.all(structure.counts == np.rint(structure.counts)):
        raise ValueError("CUDA 缺口核要求 current_counts 为整数计数")
    current_counts_t = torch.as_tensor(
        np.rint(structure.counts).astype(np.int64),
        dtype=torch.int64,
        device=device,
    )
    plan_counts = current_counts_t.clone()
    if n_active_rows:
        plan_counts += (
            row_indicators.to(torch.int64)
            - current_row_indicators.to(torch.int64)
        ).sum(dim=0, dtype=torch.int64)
    target_t = torch.as_tensor(
        structure.targets, dtype=torch.float64, device=device
    )
    denominators_t = torch.as_tensor(
        structure.denominators, dtype=torch.float64, device=device
    )
    error_terms = (
        torch.abs(target_t - plan_counts.to(torch.float64)) / denominators_t
    )
    error_sum = error_terms.sum(dtype=torch.float64)
    torch.cuda.synchronize(device)
    return _CudaGapPlan(
        torch=torch,
        device=device,
        compiled=compiled,
        active_rows=structure.active_rows,
        active_coordinates=structure.active_coordinates,
        row_lookup=structure.row_lookup,
        query_indices_by_attribute=query_indices_by_attribute,
        current_row_indicators=current_row_indicators,
        row_indicators=row_indicators,
        failure_counts=failure_counts,
        current_attribute_failures=current_attribute_failures,
        donor_attribute_failures=donor_attribute_failures,
        current_counts=current_counts_t,
        plan_counts=plan_counts,
        target=target_t,
        denominators=denominators_t,
        error_terms=error_terms,
        error_sum=error_sum,
        mask=mask,
    )


def _condition_pair_cuda(
    plan: _CudaGapPlan,
    row_index: Any,
    attribute_index: int,
    *,
    local_row: Optional[int] = None,
) -> Tuple[Any, Any, Any, Any, Any, Any]:
    """CUDA 版 E0/E1，只读评价当前完整临时组合。"""

    torch = plan.torch
    if local_row is None:
        local_row = int(plan.row_lookup[int(row_index)])
    if local_row < 0:
        raise ValueError("待评价坐标不在活跃参与行中")
    query_indices = plan.query_indices_by_attribute[attribute_index]
    if query_indices.numel() == 0:
        empty_failures = torch.empty(
            0, dtype=torch.int32, device=plan.device
        )
        empty_indicators = torch.empty(
            0, dtype=torch.bool, device=plan.device
        )
        value = plan.error_sum / plan.compiled.n_queries
        return (
            value,
            value,
            empty_failures,
            empty_failures,
            empty_indicators,
            empty_indicators,
        )
    current_failures = plan.current_attribute_failures[
        attribute_index
    ][local_row]
    donor_failures = plan.donor_attribute_failures[
        attribute_index
    ][local_row]
    old_failures = torch.where(
        plan.mask[row_index, attribute_index],
        donor_failures,
        current_failures,
    )
    base_failures = (
        plan.failure_counts[local_row, query_indices] - old_failures
    )
    failures0 = base_failures + current_failures
    failures1 = base_failures + donor_failures
    indicators0 = failures0 == 0
    indicators1 = failures1 == 0
    old_indicators = plan.row_indicators[local_row, query_indices]
    counts0 = (
        plan.plan_counts[query_indices]
        + indicators0.to(torch.int64)
        - old_indicators.to(torch.int64)
    )
    counts1 = (
        plan.plan_counts[query_indices]
        + indicators1.to(torch.int64)
        - old_indicators.to(torch.int64)
    )
    terms0 = (
        torch.abs(plan.target[query_indices] - counts0.to(torch.float64))
        / plan.denominators[query_indices]
    )
    terms1 = (
        torch.abs(plan.target[query_indices] - counts1.to(torch.float64))
        / plan.denominators[query_indices]
    )
    old_terms = plan.error_terms[query_indices]
    sum0 = plan.error_sum - old_terms.sum(dtype=torch.float64) + terms0.sum(
        dtype=torch.float64
    )
    sum1 = plan.error_sum - old_terms.sum(dtype=torch.float64) + terms1.sum(
        dtype=torch.float64
    )
    return (
        sum0 / plan.compiled.n_queries,
        sum1 / plan.compiled.n_queries,
        failures0,
        failures1,
        indicators0,
        indicators1,
    )


def _set_coordinate_cuda(
    plan: _CudaGapPlan,
    row_index: Any,
    attribute_index: int,
    selected: Any,
    failures: Any,
    indicators: Any,
    *,
    local_row: Optional[int] = None,
) -> None:
    torch = plan.torch
    if local_row is None:
        local_row = int(plan.row_lookup[int(row_index)])
    query_indices = plan.query_indices_by_attribute[attribute_index]
    old_selected = plan.mask[row_index, attribute_index]
    changed = selected != old_selected
    if query_indices.numel():
        old_indicators = plan.row_indicators[local_row, query_indices]
        old_counts = plan.plan_counts[query_indices]
        candidate_counts = (
            old_counts
            + indicators.to(torch.int64)
            - old_indicators.to(torch.int64)
        )
        candidate_terms = (
            torch.abs(
                plan.target[query_indices]
                - candidate_counts.to(torch.float64)
            )
            / plan.denominators[query_indices]
        )
        candidate_error_sum = (
            plan.error_sum
            - plan.error_terms[query_indices].sum(dtype=torch.float64)
            + candidate_terms.sum(dtype=torch.float64)
        )
        plan.plan_counts[query_indices] = torch.where(
            changed, candidate_counts, old_counts
        )
        old_failures = plan.failure_counts[local_row, query_indices]
        plan.failure_counts[local_row, query_indices] = torch.where(
            changed, failures, old_failures
        )
        plan.row_indicators[local_row, query_indices] = torch.where(
            changed, indicators, old_indicators
        )
        plan.error_terms[query_indices] = torch.where(
            changed, candidate_terms, plan.error_terms[query_indices]
        )
        plan.error_sum = torch.where(
            changed, candidate_error_sum, plan.error_sum
        )
    plan.mask[row_index, attribute_index] = selected


def _validate_cuda_batch_inputs(
    donor_tables: Sequence[Any],
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    rngs: Sequence[Any],
) -> int:
    """在使用显卡前验证逐地址批量输入和随机流。"""

    lengths = (
        len(donor_tables),
        len(participates),
        len(initial_masks),
        len(rngs),
    )
    if lengths[0] == 0 or any(value != lengths[0] for value in lengths[1:]):
        raise ValueError("逐地址输入与 rngs 必须是同长度非空序列")
    if any(not isinstance(rng, np.random.Generator) for rng in rngs):
        raise ValueError("rngs 中每一项都必须是 np.random.Generator")
    return lengths[0]


def _cuda_padded_query_layout(
    compiled: CompiledGapL1Workload,
    *,
    torch: Any,
    device: Any,
) -> Tuple[Any, Any, int]:
    """按属性填充查询索引，并为填充位置建立互不重复的哨兵。"""

    real_widths = [
        len(indices) for indices in compiled.query_indices_by_attribute
    ]
    padded_width = max([1] + real_widths)
    n_attributes = len(compiled.attribute_names)
    n_queries = compiled.n_queries
    indices = np.empty((n_attributes, padded_width), dtype=np.int64)
    valid = np.zeros((n_attributes, padded_width), dtype=bool)
    sentinels = np.arange(
        n_queries, n_queries + padded_width, dtype=np.int64
    )
    for attribute_index, query_indices in enumerate(
        compiled.query_indices_by_attribute
    ):
        width = len(query_indices)
        if width:
            indices[attribute_index, :width] = query_indices
            valid[attribute_index, :width] = True
        indices[attribute_index, width:] = sentinels[: padded_width - width]
    return (
        torch.as_tensor(indices, dtype=torch.long, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
        padded_width,
    )


def _gap_l1_padded_batched_microstep_cuda(
    *,
    failure_counts: Any,
    row_indicators: Any,
    plan_counts: Any,
    error_terms: Any,
    error_sum: Any,
    masks: Any,
    current_failures: Any,
    donor_failures: Any,
    query_indices_by_attribute: Any,
    query_valid_by_attribute: Any,
    target: Any,
    denominators: Any,
    query_count: int,
    scale: Any,
    base_logit: Any,
    strength: Any,
    clip: Any,
    torch: Any,
    batch_indices: Any,
    coordinates: Any,
    local_rows: Any,
    rolls: Any,
    step_valid: Any,
) -> Tuple[Any, Any]:
    """同时推进多个地址各自的一个严格有序微步。"""

    row_indices = coordinates[:, 0]
    attribute_indices = coordinates[:, 1]
    query_indices = query_indices_by_attribute.index_select(
        0, attribute_indices
    )
    query_valid = query_valid_by_attribute.index_select(
        0, attribute_indices
    )
    current = current_failures[
        batch_indices, attribute_indices, local_rows
    ]
    donor = donor_failures[
        batch_indices, attribute_indices, local_rows
    ]
    old_selected = masks[
        batch_indices, row_indices, attribute_indices
    ]
    old_failures = torch.where(
        old_selected.unsqueeze(1), donor, current
    )

    full_failure_rows = failure_counts[batch_indices, local_rows]
    full_indicator_rows = row_indicators[batch_indices, local_rows]
    failure_rows = full_failure_rows.gather(1, query_indices)
    indicator_rows = full_indicator_rows.gather(1, query_indices)
    count_rows = plan_counts.gather(1, query_indices)
    error_rows = error_terms.gather(1, query_indices)
    base_failures = failure_rows - old_failures
    failures0 = base_failures + current
    failures1 = base_failures + donor
    indicators0 = failures0 == 0
    indicators1 = failures1 == 0
    counts0 = (
        count_rows
        + indicators0.to(torch.int64)
        - indicator_rows.to(torch.int64)
    )
    counts1 = (
        count_rows
        + indicators1.to(torch.int64)
        - indicator_rows.to(torch.int64)
    )
    target_rows = target[query_indices]
    denominator_rows = denominators[query_indices]
    terms0 = (
        torch.abs(target_rows - counts0.to(torch.float64))
        / denominator_rows
    )
    terms1 = (
        torch.abs(target_rows - counts1.to(torch.float64))
        / denominator_rows
    )

    old_sum = torch.where(query_valid, error_rows, 0.0).sum(
        dim=1, dtype=torch.float64
    )
    sum0 = (
        error_sum
        - old_sum
        + torch.where(query_valid, terms0, 0.0).sum(
            dim=1, dtype=torch.float64
        )
    )
    sum1 = (
        error_sum
        - old_sum
        + torch.where(query_valid, terms1, 0.0).sum(
            dim=1, dtype=torch.float64
        )
    )
    e0 = sum0 / query_count
    e1 = sum1 / query_count
    score = e0 - e1
    normalized = score / scale
    raw_logit = base_logit + strength * normalized
    effective_logit = raw_logit.clamp(min=-clip, max=clip)
    probabilities = effective_logit.sigmoid()
    clipped = raw_logit != effective_logit
    proposed = rolls < probabilities
    selected = torch.where(step_valid, proposed, old_selected)
    changed = step_valid & (selected != old_selected)

    selected_failures = torch.where(
        selected.unsqueeze(1), failures1, failures0
    )
    selected_indicators = torch.where(
        selected.unsqueeze(1), indicators1, indicators0
    )
    selected_counts = torch.where(
        selected.unsqueeze(1), counts1, counts0
    )
    selected_terms = torch.where(
        selected.unsqueeze(1), terms1, terms0
    )
    selected_sum = torch.where(selected, sum1, sum0)
    update = changed.unsqueeze(1) & query_valid

    updated_failure_rows = full_failure_rows.scatter(
        1,
        query_indices,
        torch.where(update, selected_failures, failure_rows),
    )
    updated_indicator_rows = full_indicator_rows.scatter(
        1,
        query_indices,
        torch.where(update, selected_indicators, indicator_rows),
    )
    failure_counts[batch_indices, local_rows] = updated_failure_rows
    row_indicators[batch_indices, local_rows] = updated_indicator_rows
    plan_counts.scatter_(
        1,
        query_indices,
        torch.where(update, selected_counts, count_rows),
    )
    error_terms.scatter_(
        1,
        query_indices,
        torch.where(update, selected_terms, error_rows),
    )
    error_sum.copy_(torch.where(changed, selected_sum, error_sum))
    masks[batch_indices, row_indices, attribute_indices] = selected

    floats = torch.stack((
        e0,
        e1,
        score,
        normalized,
        raw_logit,
        probabilities,
        rolls,
    ), dim=1)
    floats = torch.where(
        step_valid.unsqueeze(1), floats, torch.zeros_like(floats)
    )
    booleans = torch.stack(
        (old_selected, selected, clipped), dim=1
    )
    booleans &= step_valid.unsqueeze(1)
    return floats, booleans


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


def _cuda_probability_parameters(
    plan: _CudaGapPlan,
    reference_scale: float,
    *,
    eta: float,
    strength: float,
    logit_clip: float,
) -> Tuple[float, Any, Any, Any, Any]:
    """验证公共参数，并在显卡上建立双精度条件概率常量。"""

    scale = _require_positive_finite(reference_scale, "reference_scale")
    baseline = _require_open_probability(eta, "eta")
    beta = _require_positive_finite(strength, "strength")
    clip = _require_positive_finite(logit_clip, "logit_clip")
    torch = plan.torch
    scale_t = torch.tensor(
        scale, dtype=torch.float64, device=plan.device
    )
    baseline_t = torch.tensor(
        baseline, dtype=torch.float64, device=plan.device
    )
    base_logit_t = torch.log(baseline_t) - torch.log1p(-baseline_t)
    beta_t = torch.tensor(beta, dtype=torch.float64, device=plan.device)
    clip_t = torch.tensor(clip, dtype=torch.float64, device=plan.device)
    return scale, scale_t, base_logit_t, beta_t, clip_t


def _conditional_probability_cuda(
    score: Any,
    scale: Any,
    base_logit: Any,
    strength: Any,
    logit_clip: Any,
) -> Tuple[Any, Any, Any, Any]:
    raw_logit = base_logit + strength * score / scale
    effective_logit = raw_logit.clamp(min=-logit_clip, max=logit_clip)
    probability = effective_logit.sigmoid()
    return (
        probability,
        raw_logit,
        effective_logit,
        raw_logit != effective_logit,
    )


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


def _full_query_recount_cuda_compiled(
    frame: pd.DataFrame,
    schema: Schema,
    compiled: CompiledGapL1Workload,
    *,
    torch: Any,
    device: Any,
) -> np.ndarray:
    """用现有 CUDA 条件评价器完整复算全部合取查询。"""

    truth = evaluate_conditions_vectorized(
        frame,
        [item.condition for item in compiled.conditions],
        schema,
        device="cuda",
        return_tensor=True,
        float64=True,
    )
    counts = torch.empty(
        compiled.n_queries,
        dtype=torch.int64,
        device=device,
    )
    for query_index, indices in enumerate(
        compiled.query_condition_indices
    ):
        if indices:
            condition_t = torch.as_tensor(
                indices, dtype=torch.long, device=device
            )
            matches = truth[:, condition_t].all(dim=1)
            counts[query_index] = matches.sum(dtype=torch.int64)
        else:
            counts[query_index] = len(frame)
    result = np.asarray(
        counts.detach().cpu().numpy(), dtype=np.int64
    )
    torch.cuda.synchronize(device)
    return result


def _full_query_recount_cuda(
    frame: pd.DataFrame,
    schema: Schema,
    plan: _CudaGapPlan,
) -> np.ndarray:
    return _full_query_recount_cuda_compiled(
        frame,
        schema,
        plan.compiled,
        torch=plan.torch,
        device=plan.device,
    )


def _build_scan_diagnostics(
    *,
    backend: Optional[str],
    sweeps: int,
    k: int,
    scale: float,
    trace: Any,
    final_query_counts: np.ndarray,
    final_mask: np.ndarray,
    scores: Sequence[float],
    normalized_scores: Sequence[float],
    raw_logits: Sequence[float],
    probabilities: Sequence[float],
    entropies: Sequence[float],
    probability_bins: Mapping[str, int],
    clip_hits: int,
    prepared_elapsed: float,
    scan_elapsed: float,
    materialize_elapsed: float,
    recount_elapsed: float,
    total_elapsed: float,
) -> Dict[str, Any]:
    microsteps = sweeps * k
    minimum_outcome = (
        min(min(probability, 1.0 - probability) for probability in probabilities)
        if probabilities
        else None
    )
    digest = trace.hexdigest()
    if microsteps == 0 and digest != hashlib.sha256(b"").hexdigest():
        raise RuntimeError("空扫描 trace 身份失败")
    result = {
        "kernel": "gap_l1_global_random_scan",
        "no_gate": True,
        "n_sweeps": sweeps,
        "active_switches_k": int(k),
        "gibbs_microsteps": int(microsteps),
        "conditional_error_evaluations": int(2 * microsteps),
        "query_indicator_increment_updates": int(microsteps),
        "reference_scale": scale,
        "trace_format": TRACE_FORMAT,
        "microstep_trace_sha256": digest,
        "final_query_counts": np.asarray(
            final_query_counts, dtype=np.int64
        ).tolist(),
        "final_on_switches": int(np.sum(final_mask)),
        "clip_hit_count": int(clip_hits),
        "nonfinite_condition_count": 0,
        "exact_zero_or_one_probability_count": 0,
        "minimum_binary_outcome_probability": (
            float(minimum_outcome) if minimum_outcome is not None else None
        ),
        "probability_bins": dict(probability_bins),
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
        "elapsed_sec_diagnostic_only": total_elapsed,
    }
    if backend is not None:
        result["backend"] = backend
    return result


def _build_exact_zero_spec(
    queries: Sequence[Mapping[str, Any]],
    floor: float,
    exact_target_numerators: Optional[Any],
    exact_target_denominator: Optional[int],
) -> Optional[Tuple[List[int], int, List[int], List[int]]]:
    if (exact_target_numerators is None) != (
        exact_target_denominator is None
    ):
        raise ValueError(
            "exact_target_numerators 与 exact_target_denominator 必须同时提供"
        )
    if exact_target_numerators is None:
        return None
    if not float(floor).is_integer():
        raise ValueError("精确目标零分数判定要求整数 floor")
    raw_numerators = np.asarray(exact_target_numerators)
    if (
        raw_numerators.shape != (len(queries),)
        or raw_numerators.dtype.kind not in "iu"
        or raw_numerators.dtype.kind == "b"
        or isinstance(exact_target_denominator, (bool, np.bool_))
        or not isinstance(exact_target_denominator, (int, np.integer))
        or exact_target_denominator <= 0
    ):
        raise ValueError("精确目标必须是整数向量和正整数共同分母")
    numerator_values = [int(value) for value in raw_numerators]
    denominator_value = int(exact_target_denominator)
    raw_denominators = [
        max(numerator, int(floor) * denominator_value)
        for numerator in numerator_values
    ]
    divisors = [
        math.gcd(
            math.gcd(abs(numerator), denominator_value), denominator
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
    return (
        numerator_values,
        denominator_value,
        divisors,
        [
            common_multiple // denominator
            for denominator in reduced_denominators
        ],
    )


def _exact_score_is_zero(
    query_indices: Sequence[int],
    counts0: Sequence[int],
    counts1: Sequence[int],
    exact_zero_spec: Tuple[List[int], int, List[int], List[int]],
) -> bool:
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
    return exact_units == 0


def _isolated_gap_l1_scores_cuda(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    floor: float,
    compiled_workload: Optional[CompiledGapL1Workload],
    exact_target_numerators: Optional[Any],
    exact_target_denominator: Optional[int],
) -> Dict[str, Any]:
    """显卡批量计算全零上下文中的全部孤立分数。"""

    attributes = schema.attribute_names()
    plan = _prepare_cuda_plan(
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
    exact_zero_spec = _build_exact_zero_spec(
        queries,
        floor,
        exact_target_numerators,
        exact_target_denominator,
    )
    torch = plan.torch
    n_rows = len(current)
    score_matrix = torch.zeros(
        (n_rows, len(attributes)), dtype=torch.float64, device=plan.device
    )
    exact_payloads = []
    for attribute_index, query_indices in enumerate(
        plan.query_indices_by_attribute
    ):
        coordinate_mask = (
            plan.active_coordinates[:, 1] == attribute_index
        )
        rows = plan.active_coordinates[coordinate_mask, 0]
        if len(rows) == 0:
            continue
        local_rows = plan.row_lookup[rows]
        rows_t = torch.as_tensor(rows, dtype=torch.long, device=plan.device)
        local_rows_t = torch.as_tensor(
            local_rows, dtype=torch.long, device=plan.device
        )
        if query_indices.numel() == 0:
            continue
        current_failures = plan.current_attribute_failures[
            attribute_index
        ].index_select(0, local_rows_t)
        donor_failures = plan.donor_attribute_failures[
            attribute_index
        ].index_select(0, local_rows_t)
        row_failures = plan.failure_counts.index_select(
            0, local_rows_t
        ).index_select(1, query_indices)
        failures1 = row_failures - current_failures + donor_failures
        indicators0 = plan.row_indicators.index_select(
            0, local_rows_t
        ).index_select(1, query_indices)
        indicators1 = failures1 == 0
        base_counts = plan.plan_counts.index_select(0, query_indices)
        counts1 = (
            base_counts.unsqueeze(0)
            + indicators1.to(torch.int64)
            - indicators0.to(torch.int64)
        )
        old_terms = plan.error_terms.index_select(0, query_indices)
        terms1 = (
            torch.abs(
                plan.target.index_select(0, query_indices).unsqueeze(0)
                - counts1.to(torch.float64)
            )
            / plan.denominators.index_select(0, query_indices).unsqueeze(0)
        )
        scores = (
            old_terms.unsqueeze(0).sum(dim=1, dtype=torch.float64)
            - terms1.sum(dim=1, dtype=torch.float64)
        ) / plan.compiled.n_queries
        score_matrix[rows_t, attribute_index] = scores
        if exact_zero_spec is not None:
            exact_payloads.append((
                rows,
                attribute_index,
                np.asarray(
                    query_indices.detach().cpu().numpy(), dtype=np.int64
                ),
                np.asarray(
                    base_counts.detach().cpu().numpy(), dtype=np.int64
                ),
                np.asarray(counts1.detach().cpu().numpy(), dtype=np.int64),
            ))

    plan.torch.cuda.synchronize(plan.device)
    coordinates = np.asarray(
        plan.active_coordinates, dtype=np.intp
    ).reshape(-1, 2)
    scores_np = np.asarray(
        score_matrix[
            torch.as_tensor(
                coordinates[:, 0], dtype=torch.long, device=plan.device
            ),
            torch.as_tensor(
                coordinates[:, 1], dtype=torch.long, device=plan.device
            ),
        ].detach().cpu().numpy(),
        dtype=np.float64,
    )
    if exact_zero_spec is not None:
        coordinate_positions = {
            (int(row), int(attribute)): index
            for index, (row, attribute) in enumerate(coordinates)
        }
        for rows, attribute_index, query_indices, counts0, counts1 in exact_payloads:
            for local_index, row in enumerate(rows):
                if _exact_score_is_zero(
                    query_indices,
                    counts0,
                    counts1[local_index],
                    exact_zero_spec,
                ):
                    scores_np[coordinate_positions[
                        (int(row), int(attribute_index))
                    ]] = 0.0
    return {
        "coordinates": coordinates,
        "scores": scores_np,
        "current_error": float(
            (plan.error_sum / plan.compiled.n_queries).detach().cpu().item()
        ),
        "backend": "torch_cuda_float64",
    }


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
    device: str = "numpy",
) -> Dict[str, Any]:
    """计算其他开关全为 0 时所有不同值行—属性的孤立分数。

    定尺调用方可同时提供目标的共同整数分母表示。此时先用独立整数公共尺度
    判定数学上精确为零的分数，再把这些位置显式设为 ``0.0``；非零分数仍按
    冻结的 NumPy 双精度路径取值。这避免理论抵消被浮点尾差误收进 RMS。
    """

    if device == "cuda":
        return _isolated_gap_l1_scores_cuda(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            floor=floor,
            compiled_workload=compiled_workload,
            exact_target_numerators=exact_target_numerators,
            exact_target_denominator=exact_target_denominator,
        )
    if device != "numpy":
        raise ValueError("缺口孤立分数 device 只支持 'numpy' 或 'cuda'")
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
    exact_zero_spec = _build_exact_zero_spec(
        queries,
        floor,
        exact_target_numerators,
        exact_target_denominator,
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
            if _exact_score_is_zero(
                query_indices,
                counts0,
                counts1,
                exact_zero_spec,
            ):
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
    device: str = "numpy",
) -> Dict[str, Any]:
    """只读评价一个活跃开关的当前完整上下文条件式。"""

    if device not in ("numpy", "cuda"):
        raise ValueError("缺口条件评价 device 只支持 'numpy' 或 'cuda'")

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
    prepare = _prepare_cuda_plan if device == "cuda" else _prepare_plan
    plan = prepare(
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
    if device == "cuda":
        e0_t, e1_t, *_ = _condition_pair_cuda(plan, *coordinate)
        score_t = e0_t - e1_t
        (
            _,
            scale_t,
            base_logit_t,
            strength_t,
            clip_t,
        ) = _cuda_probability_parameters(
            plan,
            reference_scale,
            eta=eta,
            strength=strength,
            logit_clip=logit_clip,
        )
        probability_t, raw_logit_t, logit_t, clipped_t = (
            _conditional_probability_cuda(
                score_t,
                scale_t,
                base_logit_t,
                strength_t,
                clip_t,
            )
        )
        values = plan.torch.stack((
            e0_t,
            e1_t,
            score_t,
            score_t / scale_t,
            raw_logit_t,
            logit_t,
            probability_t,
        )).detach().cpu().numpy()
        clipped = bool(clipped_t.detach().cpu().item())
        plan.torch.cuda.synchronize(plan.device)
        if not np.all(np.isfinite(values)):
            raise RuntimeError("CUDA 缺口条件评价产生非有限数值")
        probability = float(values[6])
        if not 0.0 < probability < 1.0:
            raise RuntimeError("有限条件概率丢失双向严格正支持")
        return {
            "e0": float(values[0]),
            "e1": float(values[1]),
            "score": float(values[2]),
            "normalized_score": float(values[3]),
            "raw_logit": float(values[4]),
            "logit": float(values[5]),
            "probability": probability,
            "clipped": clipped,
            "backend": "torch_cuda_float64",
        }

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


def _evolve_step_gap_l1_global_cuda(
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
    n_sweeps: int,
    eta: float,
    strength: float,
    floor: float,
    logit_clip: float,
    compiled_workload: Optional[CompiledGapL1Workload],
    verify_full_recount: bool,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """CUDA 双精度后端；随机带仍由冻结的 NumPy 流形成。"""

    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng 必须是 np.random.Generator")
    sweeps = _require_nonnegative_integer(n_sweeps, "n_sweeps")
    started = time.perf_counter()
    plan = _prepare_cuda_plan(
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
    (
        scale,
        scale_t,
        base_logit_t,
        strength_t,
        clip_t,
    ) = _cuda_probability_parameters(
        plan,
        reference_scale,
        eta=eta,
        strength=strength,
        logit_clip=logit_clip,
    )
    plan.torch.cuda.synchronize(plan.device)
    prepared_elapsed = time.perf_counter() - started
    torch = plan.torch
    k = len(plan.active_coordinates)
    microsteps = sweeps * k

    # 必须逐微步交替消费“坐标、均匀随机数”；坐标留在主机，随机带只读移交显卡。
    coordinate_indices = np.empty(microsteps, dtype=np.int64)
    random_rolls = np.empty(microsteps, dtype=np.float64)
    scan_started = time.perf_counter()
    for step in range(microsteps):
        coordinate_indices[step] = rng.integers(0, k)
        random_rolls[step] = rng.random()
    if microsteps:
        coordinate_tape = np.asarray(
            plan.active_coordinates[coordinate_indices], dtype=np.int64
        )
        local_row_tape = plan.row_lookup[coordinate_tape[:, 0]]
    else:
        coordinate_tape = np.empty((0, 2), dtype=np.int64)
        local_row_tape = np.empty(0, dtype=np.intp)
    random_rolls_t = torch.as_tensor(
        random_rolls, dtype=torch.float64, device=plan.device
    )

    e0_values = torch.empty(
        microsteps, dtype=torch.float64, device=plan.device
    )
    e1_values = torch.empty_like(e0_values)
    score_values = torch.empty_like(e0_values)
    normalized_values = torch.empty_like(e0_values)
    raw_logit_values = torch.empty_like(e0_values)
    probability_values = torch.empty_like(e0_values)
    before_values = torch.empty(
        microsteps, dtype=torch.bool, device=plan.device
    )
    after_values = torch.empty_like(before_values)
    clipped_values = torch.empty_like(before_values)

    for step in range(microsteps):
        row_index = int(coordinate_tape[step, 0])
        attribute_index = int(coordinate_tape[step, 1])
        local_row = int(local_row_tape[step])
        before = plan.mask[row_index, attribute_index].clone()
        (
            e0,
            e1,
            failures0,
            failures1,
            indicators0,
            indicators1,
        ) = _condition_pair_cuda(
            plan,
            row_index,
            attribute_index,
            local_row=local_row,
        )
        score = e0 - e1
        normalized = score / scale_t
        probability, raw_logit, _, clipped = (
            _conditional_probability_cuda(
                score,
                scale_t,
                base_logit_t,
                strength_t,
                clip_t,
            )
        )
        after = random_rolls_t[step] < probability
        selected_failures = torch.where(after, failures1, failures0)
        selected_indicators = torch.where(after, indicators1, indicators0)
        _set_coordinate_cuda(
            plan,
            row_index,
            attribute_index,
            after,
            selected_failures,
            selected_indicators,
            local_row=local_row,
        )
        e0_values[step] = e0
        e1_values[step] = e1
        score_values[step] = score
        normalized_values[step] = normalized
        raw_logit_values[step] = raw_logit
        probability_values[step] = probability
        before_values[step] = before
        after_values[step] = after
        clipped_values[step] = clipped

    one_minus_probability = 1.0 - probability_values
    entropy_values = -(
        probability_values * probability_values.log()
        + one_minus_probability * (-probability_values).log1p()
    )
    float_values = torch.stack((
        e0_values,
        e1_values,
        score_values,
        normalized_values,
        raw_logit_values,
        probability_values,
        random_rolls_t,
        entropy_values,
    ), dim=1).detach().cpu().numpy()
    bool_values = torch.stack((
        before_values,
        after_values,
        clipped_values,
    ), dim=1).detach().cpu().numpy()
    final_query_counts = np.asarray(
        plan.plan_counts.detach().cpu().numpy(), dtype=np.int64
    )
    final_mask = np.asarray(
        plan.mask.detach().cpu().numpy(), dtype=bool
    )
    torch.cuda.synchronize(plan.device)

    if not np.all(np.isfinite(float_values)):
        raise RuntimeError("CUDA 缺口扫描产生非有限条件数值")
    probabilities = np.asarray(float_values[:, 5], dtype=np.float64)
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise RuntimeError("CUDA 缺口扫描产生精确 0/1 条件概率")
    switches = np.asarray(bool_values[:, 1], dtype=bool)
    expected_switches = random_rolls < probabilities
    if not np.array_equal(switches, expected_switches):
        raise RuntimeError("CUDA 抽样开关与只读随机带不一致")

    trace = hashlib.sha256()
    probability_bins = {
        "open_0_0p001": 0,
        "closed_0p001_open_0p01": 0,
        "closed_0p01_0p99": 0,
        "open_0p99_closed_0p999": 0,
        "open_0p999_1": 0,
    }
    for step in range(microsteps):
        row_index = int(coordinate_tape[step, 0])
        attribute_index = int(coordinate_tape[step, 1])
        e0, e1, score, normalized, raw_logit, probability, roll, _ = (
            map(float, float_values[step])
        )
        before, after, clipped = map(bool, bool_values[step])
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
            random_roll=roll,
            before=before,
            after=after,
            clipped=clipped,
        )
        probability_bins[_probability_bin(probability)] += 1
    scan_elapsed = time.perf_counter() - scan_started

    materialize_started = time.perf_counter()
    copy_table = _materialize_copy_table(
        current, donors, plan.compiled.attribute_names, final_mask
    )
    materialize_elapsed = time.perf_counter() - materialize_started
    recount_elapsed = 0.0
    if verify_full_recount:
        recount_started = time.perf_counter()
        recounted = _full_query_recount_cuda(copy_table, schema, plan)
        recount_elapsed = time.perf_counter() - recount_started
        if not np.array_equal(recounted, final_query_counts):
            raise RuntimeError("增量查询计数与 CUDA 最终完整复算不一致")

    diagnostics = _build_scan_diagnostics(
        backend="torch_cuda_float64",
        sweeps=sweeps,
        k=k,
        scale=scale,
        trace=trace,
        final_query_counts=final_query_counts,
        final_mask=final_mask,
        scores=float_values[:, 2].tolist(),
        normalized_scores=float_values[:, 3].tolist(),
        raw_logits=float_values[:, 4].tolist(),
        probabilities=probabilities.tolist(),
        entropies=float_values[:, 7].tolist(),
        probability_bins=probability_bins,
        clip_hits=int(np.sum(bool_values[:, 2], dtype=np.int64)),
        prepared_elapsed=prepared_elapsed,
        scan_elapsed=scan_elapsed,
        materialize_elapsed=materialize_elapsed,
        recount_elapsed=recount_elapsed,
        total_elapsed=time.perf_counter() - started,
    )
    return copy_table, final_mask.copy(), diagnostics


def _evolve_step_gap_l1_global_cuda_batched(
    current: pd.DataFrame,
    donor_tables: Sequence[pd.DataFrame],
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    reference_scale: float,
    rngs: Sequence[np.random.Generator],
    n_sweeps: int,
    eta: float,
    strength: float,
    floor: float,
    logit_clip: float,
    compiled_workload: Optional[CompiledGapL1Workload],
    verify_full_recount: bool,
) -> Tuple[
    Tuple[Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]], ...],
    Dict[str, Any],
]:
    """不同地址的 CUDA 双精度批量执行及内部对拍轨迹。"""

    batch_size = _validate_cuda_batch_inputs(
        donor_tables, participates, initial_masks, rngs
    )
    sweeps = _require_nonnegative_integer(n_sweeps, "n_sweeps")
    compiled = (
        compile_gap_l1_workload(schema, queries)
        if compiled_workload is None
        else compiled_workload
    )
    started = time.perf_counter()
    plans = [
        _prepare_cuda_plan(
            current,
            donor_tables[index],
            schema,
            queries,
            target,
            current_counts,
            participates[index],
            initial_masks[index],
            floor=floor,
            compiled_workload=compiled,
        )
        for index in range(batch_size)
    ]
    first = plans[0]
    torch = first.torch
    device = first.device
    n_attributes = len(compiled.attribute_names)
    n_queries = compiled.n_queries
    active_row_counts = [len(plan.active_rows) for plan in plans]
    maximum_active_rows = max(active_row_counts)
    active_switches = [len(plan.active_coordinates) for plan in plans]
    microsteps_by_address = [sweeps * value for value in active_switches]
    maximum_microsteps = max(microsteps_by_address)
    (
        query_indices_by_attribute,
        query_valid_by_attribute,
        padded_query_width,
    ) = _cuda_padded_query_layout(
        compiled, torch=torch, device=device
    )
    extended_query_count = n_queries + padded_query_width

    failure_counts = torch.zeros(
        (batch_size, maximum_active_rows, extended_query_count),
        dtype=torch.int32,
        device=device,
    )
    row_indicators = torch.ones(
        (batch_size, maximum_active_rows, extended_query_count),
        dtype=torch.bool,
        device=device,
    )
    plan_counts = torch.zeros(
        (batch_size, extended_query_count),
        dtype=torch.int64,
        device=device,
    )
    error_terms = torch.zeros(
        (batch_size, extended_query_count),
        dtype=torch.float64,
        device=device,
    )
    error_sum = torch.stack([plan.error_sum for plan in plans], dim=0)
    masks = torch.stack([plan.mask for plan in plans], dim=0)
    current_failures = torch.zeros(
        (
            batch_size,
            n_attributes,
            maximum_active_rows,
            padded_query_width,
        ),
        dtype=torch.int32,
        device=device,
    )
    donor_failures = torch.zeros_like(current_failures)
    for batch_index, plan in enumerate(plans):
        active_rows = active_row_counts[batch_index]
        if active_rows:
            failure_counts[
                batch_index, :active_rows, :n_queries
            ] = plan.failure_counts
            row_indicators[
                batch_index, :active_rows, :n_queries
            ] = plan.row_indicators
        plan_counts[batch_index, :n_queries] = plan.plan_counts
        error_terms[batch_index, :n_queries] = plan.error_terms
        for attribute_index, query_indices in enumerate(
            compiled.query_indices_by_attribute
        ):
            width = len(query_indices)
            if active_rows and width:
                current_failures[
                    batch_index,
                    attribute_index,
                    :active_rows,
                    :width,
                ] = plan.current_attribute_failures[attribute_index]
                donor_failures[
                    batch_index,
                    attribute_index,
                    :active_rows,
                    :width,
                ] = plan.donor_attribute_failures[attribute_index]

    target_t = torch.zeros(
        extended_query_count, dtype=torch.float64, device=device
    )
    target_t[:n_queries] = first.target
    denominators_t = torch.ones_like(target_t)
    denominators_t[:n_queries] = first.denominators
    (
        scale_value,
        scale_t,
        base_logit_t,
        strength_t,
        clip_t,
    ) = _cuda_probability_parameters(
        first,
        reference_scale,
        eta=eta,
        strength=strength,
        logit_clip=logit_clip,
    )

    coordinate_tapes = np.zeros(
        (batch_size, maximum_microsteps, 2), dtype=np.int64
    )
    local_row_tapes = np.zeros(
        (batch_size, maximum_microsteps), dtype=np.int64
    )
    roll_tapes = np.zeros(
        (batch_size, maximum_microsteps), dtype=np.float64
    )
    valid_step_mask = np.zeros(
        (batch_size, maximum_microsteps), dtype=bool
    )
    for batch_index, (plan, rng) in enumerate(zip(plans, rngs)):
        microsteps = microsteps_by_address[batch_index]
        active_count = active_switches[batch_index]
        if microsteps == 0:
            continue
        coordinate_indices = np.empty(microsteps, dtype=np.int64)
        rolls = np.empty(microsteps, dtype=np.float64)
        for step in range(microsteps):
            coordinate_indices[step] = rng.integers(0, active_count)
            rolls[step] = rng.random()
        coordinates = np.asarray(
            plan.active_coordinates[coordinate_indices], dtype=np.int64
        )
        coordinate_tapes[batch_index, :microsteps] = coordinates
        local_row_tapes[batch_index, :microsteps] = (
            plan.row_lookup[coordinates[:, 0]]
        )
        roll_tapes[batch_index, :microsteps] = rolls
        valid_step_mask[batch_index, :microsteps] = True

    coordinate_tapes_t = torch.as_tensor(
        coordinate_tapes, dtype=torch.long, device=device
    )
    local_row_tapes_t = torch.as_tensor(
        local_row_tapes, dtype=torch.long, device=device
    )
    roll_tapes_t = torch.as_tensor(
        roll_tapes, dtype=torch.float64, device=device
    )
    valid_step_mask_t = torch.as_tensor(
        valid_step_mask, dtype=torch.bool, device=device
    )
    float_values_t = torch.zeros(
        (batch_size, maximum_microsteps, 7),
        dtype=torch.float64,
        device=device,
    )
    bool_values_t = torch.zeros(
        (batch_size, maximum_microsteps, 3),
        dtype=torch.bool,
        device=device,
    )
    batch_indices = torch.arange(
        batch_size, dtype=torch.long, device=device
    )
    del plan
    del first
    del plans
    torch.cuda.synchronize(device)
    prepared_elapsed = time.perf_counter() - started

    scan_started = time.perf_counter()
    for step in range(maximum_microsteps):
        floats, booleans = _gap_l1_padded_batched_microstep_cuda(
            failure_counts=failure_counts,
            row_indicators=row_indicators,
            plan_counts=plan_counts,
            error_terms=error_terms,
            error_sum=error_sum,
            masks=masks,
            current_failures=current_failures,
            donor_failures=donor_failures,
            query_indices_by_attribute=query_indices_by_attribute,
            query_valid_by_attribute=query_valid_by_attribute,
            target=target_t,
            denominators=denominators_t,
            query_count=n_queries,
            scale=scale_t,
            base_logit=base_logit_t,
            strength=strength_t,
            clip=clip_t,
            torch=torch,
            batch_indices=batch_indices,
            coordinates=coordinate_tapes_t[:, step],
            local_rows=local_row_tapes_t[:, step],
            rolls=roll_tapes_t[:, step],
            step_valid=valid_step_mask_t[:, step],
        )
        float_values_t[:, step] = floats
        bool_values_t[:, step] = booleans

    safe_probabilities_t = torch.where(
        valid_step_mask_t,
        float_values_t[:, :, 5],
        torch.full_like(float_values_t[:, :, 5], 0.5),
    )
    one_minus_probability_t = 1.0 - safe_probabilities_t
    entropy_values_t = -(
        safe_probabilities_t * safe_probabilities_t.log()
        + one_minus_probability_t * (-safe_probabilities_t).log1p()
    )
    entropy_values_t = torch.where(
        valid_step_mask_t,
        entropy_values_t,
        torch.zeros_like(entropy_values_t),
    )
    padded_float_values = np.asarray(
        float_values_t.detach().cpu().numpy(), dtype=np.float64
    )
    padded_bool_values = np.asarray(
        bool_values_t.detach().cpu().numpy(), dtype=bool
    )
    padded_entropy_values = np.asarray(
        entropy_values_t.detach().cpu().numpy(), dtype=np.float64
    )
    final_counts = np.asarray(
        plan_counts[:, :n_queries].detach().cpu().numpy(), dtype=np.int64
    )
    final_masks = np.asarray(
        masks.detach().cpu().numpy(), dtype=bool
    )
    torch.cuda.synchronize(device)
    scan_elapsed = time.perf_counter() - scan_started

    valid_float_values = padded_float_values[valid_step_mask]
    if not np.all(np.isfinite(valid_float_values)):
        raise RuntimeError("CUDA 批量缺口扫描产生非有限条件数值")
    probabilities = padded_float_values[:, :, 5][valid_step_mask]
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise RuntimeError("CUDA 批量缺口扫描产生精确 0/1 条件概率")
    if not np.array_equal(
        padded_bool_values[:, :, 1][valid_step_mask],
        roll_tapes[valid_step_mask] < probabilities,
    ):
        raise RuntimeError("CUDA 批量抽样开关与只读随机带不一致")
    if np.any(padded_float_values[~valid_step_mask] != 0.0) or np.any(
        padded_bool_values[~valid_step_mask]
    ):
        raise RuntimeError("CUDA 批量填充微步意外产生输出")

    post_started = time.perf_counter()
    address_rows = []
    coordinates_by_address = []
    rolls_by_address = []
    floats_by_address = []
    booleans_by_address = []
    for batch_index, microsteps in enumerate(microsteps_by_address):
        coordinates = coordinate_tapes[batch_index, :microsteps].copy()
        rolls = roll_tapes[batch_index, :microsteps].copy()
        float_values = padded_float_values[batch_index, :microsteps].copy()
        bool_values = padded_bool_values[batch_index, :microsteps].copy()
        entropy_values = padded_entropy_values[
            batch_index, :microsteps
        ].copy()
        trace = hashlib.sha256()
        probability_bins = {
            "open_0_0p001": 0,
            "closed_0p001_open_0p01": 0,
            "closed_0p01_0p99": 0,
            "open_0p99_closed_0p999": 0,
            "open_0p999_1": 0,
        }
        for step in range(microsteps):
            row_index, attribute_index = map(int, coordinates[step])
            e0, e1, score, normalized, raw_logit, probability, roll = map(
                float, float_values[step]
            )
            before, after, clipped = map(bool, bool_values[step])
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
                random_roll=roll,
                before=before,
                after=after,
                clipped=clipped,
            )
            probability_bins[_probability_bin(probability)] += 1

        materialize_started = time.perf_counter()
        copy_table = _materialize_copy_table(
            current,
            donor_tables[batch_index],
            compiled.attribute_names,
            final_masks[batch_index],
        )
        materialize_elapsed = time.perf_counter() - materialize_started
        recount_elapsed = 0.0
        if verify_full_recount:
            recount_started = time.perf_counter()
            recounted = _full_query_recount_cuda_compiled(
                copy_table,
                schema,
                compiled,
                torch=torch,
                device=device,
            )
            recount_elapsed = time.perf_counter() - recount_started
            if not np.array_equal(recounted, final_counts[batch_index]):
                raise RuntimeError(
                    "批量增量查询计数与 CUDA 最终完整复算不一致"
                )
        address_rows.append({
            "table": copy_table,
            "trace": trace,
            "probability_bins": probability_bins,
            "entropy_values": entropy_values,
            "materialize_elapsed": materialize_elapsed,
            "recount_elapsed": recount_elapsed,
        })
        coordinates_by_address.append(coordinates)
        rolls_by_address.append(rolls)
        floats_by_address.append(float_values)
        booleans_by_address.append(bool_values)
    post_elapsed = time.perf_counter() - post_started
    total_elapsed = time.perf_counter() - started
    padding_microsteps = int(
        batch_size * maximum_microsteps - sum(microsteps_by_address)
    )

    results = []
    for batch_index, row in enumerate(address_rows):
        float_values = floats_by_address[batch_index]
        bool_values = booleans_by_address[batch_index]
        diagnostics = _build_scan_diagnostics(
            backend="torch_cuda_float64_batched",
            sweeps=sweeps,
            k=active_switches[batch_index],
            scale=scale_value,
            trace=row["trace"],
            final_query_counts=final_counts[batch_index],
            final_mask=final_masks[batch_index],
            scores=float_values[:, 2].tolist(),
            normalized_scores=float_values[:, 3].tolist(),
            raw_logits=float_values[:, 4].tolist(),
            probabilities=float_values[:, 5].tolist(),
            entropies=row["entropy_values"].tolist(),
            probability_bins=row["probability_bins"],
            clip_hits=int(np.sum(bool_values[:, 2], dtype=np.int64)),
            prepared_elapsed=prepared_elapsed,
            scan_elapsed=scan_elapsed,
            materialize_elapsed=row["materialize_elapsed"],
            recount_elapsed=row["recount_elapsed"],
            total_elapsed=total_elapsed,
        )
        diagnostics["batch_execution"] = {
            "format": BATCH_EXECUTION_FORMAT,
            "batch_size": int(batch_size),
            "batch_index": int(batch_index),
            "maximum_padded_microsteps": int(maximum_microsteps),
            "padding_microsteps_across_batch": padding_microsteps,
            "strict_internal_order_preserved": True,
            "timing_scope": "shared_batch_not_additive",
            "shared_prepare_elapsed_sec_diagnostic_only": prepared_elapsed,
            "shared_scan_elapsed_sec_diagnostic_only": scan_elapsed,
            "shared_post_elapsed_sec_diagnostic_only": post_elapsed,
            "shared_elapsed_sec_diagnostic_only": total_elapsed,
        }
        results.append((
            row["table"],
            final_masks[batch_index].copy(),
            diagnostics,
        ))

    debug = {
        "backend": "torch_cuda_float64_batched",
        "batch_execution_format": BATCH_EXECUTION_FORMAT,
        "batch_size": batch_size,
        "active_switches_k_by_address": tuple(map(int, active_switches)),
        "microsteps_by_address": tuple(map(int, microsteps_by_address)),
        "maximum_padded_microsteps": int(maximum_microsteps),
        "padding_microsteps": padding_microsteps,
        "valid_step_mask": valid_step_mask,
        "reference_scale": scale_value,
        "coordinates": tuple(coordinates_by_address),
        "random_rolls": tuple(rolls_by_address),
        "float_values": tuple(floats_by_address),
        "bool_values": tuple(booleans_by_address),
        "timings": {
            "prepare_elapsed_sec": prepared_elapsed,
            "scan_elapsed_sec": scan_elapsed,
            "post_elapsed_sec": post_elapsed,
            "total_elapsed_sec": total_elapsed,
        },
    }
    return tuple(results), debug


def evolve_step_gap_l1_global_batched(
    current: pd.DataFrame,
    donor_tables: Sequence[pd.DataFrame],
    schema: Schema,
    queries: List[Dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    reference_scale: float,
    rngs: Sequence[np.random.Generator],
    n_sweeps: int = DEFAULT_GAP_L1_SWEEPS,
    eta: float = DEFAULT_GAP_L1_ETA,
    strength: float = DEFAULT_GAP_L1_STRENGTH,
    floor: float = DEFAULT_GAP_L1_FLOOR,
    logit_clip: float = DEFAULT_GAP_L1_LOGIT_CLIP,
    compiled_workload: Optional[CompiledGapL1Workload] = None,
    verify_full_recount: bool = True,
    device: str = "cuda",
) -> Tuple[Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]], ...]:
    """同时执行同一状态内多个不同地址的无门控缺口扫描。

    批量化只改变显卡算术的调度方式。每个地址仍使用自己的供体、参与行、
    初始开关和随机流，并独立执行严格有序的 ``n_sweeps * K`` 个微步；函数
    不执行接受、拒绝、候选筛选、回滚或赢家选择。
    """

    if device != "cuda":
        raise ValueError("批量缺口扫描 device 只支持 'cuda'")
    results, _ = _evolve_step_gap_l1_global_cuda_batched(
        current,
        donor_tables,
        schema,
        queries,
        target,
        current_counts,
        participates=participates,
        initial_masks=initial_masks,
        reference_scale=reference_scale,
        rngs=rngs,
        n_sweeps=n_sweeps,
        eta=eta,
        strength=strength,
        floor=floor,
        logit_clip=logit_clip,
        compiled_workload=compiled_workload,
        verify_full_recount=verify_full_recount,
    )
    return results


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
    device: str = "numpy",
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, Any]]:
    """执行全局剩余缺口绝对误差的固定随机扫描。

    ``n_sweeps * K`` 个微步在全部活跃行—属性开关中有放回抽坐标。函数不
    抽参与行、不生成初始开关，也不执行突变或任何生成后门控。
    """

    if device == "cuda":
        return _evolve_step_gap_l1_global_cuda(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            participate=participate,
            initial_mask=initial_mask,
            reference_scale=reference_scale,
            rng=rng,
            n_sweeps=n_sweeps,
            eta=eta,
            strength=strength,
            floor=floor,
            logit_clip=logit_clip,
            compiled_workload=compiled_workload,
            verify_full_recount=verify_full_recount,
        )
    if device != "numpy":
        raise ValueError("缺口扫描 device 只支持 'numpy' 或 'cuda'")

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

    diagnostics = _build_scan_diagnostics(
        backend=None,
        sweeps=sweeps,
        k=k,
        scale=scale,
        trace=trace,
        final_query_counts=plan.plan_counts,
        final_mask=plan.mask,
        scores=scores,
        normalized_scores=normalized_scores,
        raw_logits=raw_logits,
        probabilities=probabilities,
        entropies=entropies,
        probability_bins=probability_bins,
        clip_hits=clip_hits,
        prepared_elapsed=prepared_elapsed,
        scan_elapsed=scan_elapsed,
        materialize_elapsed=materialize_elapsed,
        recount_elapsed=recount_elapsed,
        total_elapsed=time.perf_counter() - started,
    )
    return copy_table, plan.mask.copy(), diagnostics
