"""第 6B-1B 阶段独立 CUDA（显卡）缺口算术。

本模块不导入 ``table_diffevo.gap_l1_diffusion`` 或其条件评价辅助函数。
条件编码、增量状态、条件概率、随机重放和完整查询复算均在这里独立实现。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


TRACE_STRUCT = "<qqqddddddd???"


@dataclass(frozen=True)
class _Workload:
    attributes: tuple[str, ...]
    conditions: tuple[dict[str, Any], ...]
    query_condition_indices: tuple[tuple[int, ...], ...]
    query_indices_by_attribute: tuple[np.ndarray, ...]
    condition_groups_by_attribute: tuple[
        tuple[tuple[int, ...], ...], ...
    ]


@dataclass
class _Plan:
    torch: Any
    device: Any
    workload: _Workload
    active_rows: np.ndarray
    coordinates: np.ndarray
    row_lookup: np.ndarray
    query_indices_by_attribute: tuple[Any, ...]
    current_attribute_failures: tuple[Any, ...]
    donor_attribute_failures: tuple[Any, ...]
    current_indicators: Any
    failures: Any
    indicators: Any
    counts: Any
    target: Any
    denominators: Any
    error_terms: Any
    error_sum: Any
    mask: Any


def _require_cuda() -> tuple[Any, Any]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("独立缺口重放需要 PyTorch") from error
    if not torch.cuda.is_available():
        raise RuntimeError("独立缺口重放请求 CUDA，但显卡不可用")
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("独立缺口重放要求开启确定性算法")
    return torch, torch.device("cuda")


def _rng_sha256(rng: np.random.Generator) -> str:
    value = json.loads(json.dumps(rng.bit_generator.state))
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compile_workload(
    schema: Any,
    queries: Sequence[Mapping[str, Any]],
) -> _Workload:
    attributes = tuple(schema.attribute_names())
    positions = {name: index for index, name in enumerate(attributes)}
    if len(positions) != len(attributes):
        raise ValueError("独立 CUDA 查询属性名不得重复")
    conditions = []
    query_condition_indices = []
    per_attribute: list[dict[int, list[int]]] = [
        {} for _ in attributes
    ]
    for query_index, query in enumerate(queries):
        indices = []
        for condition in query["conditions"]:
            attribute = condition["attribute"]
            if attribute not in positions:
                raise ValueError(f"独立 CUDA 查询包含未知属性：{attribute}")
            if condition["operator"] not in {"==", ">=", "between"}:
                raise ValueError("独立 CUDA 查询包含不支持的操作符")
            condition_index = len(conditions)
            conditions.append(dict(condition))
            indices.append(condition_index)
            per_attribute[positions[attribute]].setdefault(
                query_index, []
            ).append(condition_index)
        query_condition_indices.append(tuple(indices))
    query_indices_by_attribute = []
    condition_groups_by_attribute = []
    for mapping in per_attribute:
        query_indices = np.asarray(sorted(mapping), dtype=np.int64)
        query_indices_by_attribute.append(query_indices)
        condition_groups_by_attribute.append(tuple(
            tuple(mapping[int(query_index)])
            for query_index in query_indices
        ))
    return _Workload(
        attributes=attributes,
        conditions=tuple(conditions),
        query_condition_indices=tuple(query_condition_indices),
        query_indices_by_attribute=tuple(query_indices_by_attribute),
        condition_groups_by_attribute=tuple(
            condition_groups_by_attribute
        ),
    )


def _encoded_columns(
    frame: pd.DataFrame,
    attributes: Sequence[str],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, int]]]:
    columns = {}
    category_maps = {}
    for attribute in attributes:
        column = frame[attribute]
        if pd.api.types.is_numeric_dtype(column):
            columns[attribute] = column.to_numpy(dtype=np.float64)
            continue
        mapping: dict[str, int] = {}
        encoded = np.empty(len(frame), dtype=np.float64)
        for row, raw_value in enumerate(column.to_numpy()):
            value = str(raw_value)
            if value not in mapping:
                mapping[value] = len(mapping)
            encoded[row] = mapping[value]
        columns[attribute] = encoded
        category_maps[attribute] = mapping
    return columns, category_maps


def _condition_truth_cuda(
    frame: pd.DataFrame,
    schema: Any,
    conditions: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    device: Any,
) -> Any:
    attributes = tuple(schema.attribute_names())
    columns, category_maps = _encoded_columns(frame, attributes)
    truth = torch.empty(
        (len(frame), len(conditions)), dtype=torch.bool, device=device
    )
    column_tensors = {
        attribute: torch.tensor(values, dtype=torch.float64, device=device)
        for attribute, values in columns.items()
    }
    for condition_index, condition in enumerate(conditions):
        attribute = condition["attribute"]
        operator = condition["operator"]
        values = column_tensors[attribute]
        if attribute in category_maps:
            if operator != "==":
                raise ValueError("字符串类别属性只支持 ==")
            comparison = category_maps[attribute].get(
                str(condition["value"]), -1
            )
            truth[:, condition_index] = values == float(comparison)
        elif operator == "==":
            truth[:, condition_index] = values == float(condition["value"])
        elif operator == ">=":
            truth[:, condition_index] = values >= float(condition["value"])
        elif operator == "between":
            truth[:, condition_index] = (
                (values >= float(condition["lower"]))
                & (values <= float(condition["upper"]))
            )
        else:
            raise ValueError("独立 CUDA 查询包含不支持的操作符")
    return truth


def _attribute_failure_matrix(
    truth: Any,
    groups: Sequence[Sequence[int]],
    *,
    torch: Any,
    device: Any,
) -> Any:
    result = torch.zeros(
        (truth.shape[0], len(groups)), dtype=torch.int32, device=device
    )
    for local_query, indices in enumerate(groups):
        condition_t = torch.tensor(
            indices, dtype=torch.long, device=device
        )
        result[:, local_query] = (~truth[:, condition_t]).sum(
            dim=1, dtype=torch.int32
        )
    return result


def _prepare(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    queries: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    current_counts: Sequence[int],
    participate: Sequence[bool],
    initial_mask: np.ndarray,
    *,
    floor: float,
) -> _Plan:
    torch, device = _require_cuda()
    workload = _compile_workload(schema, queries)
    if len(current) != len(donors):
        raise ValueError("独立 CUDA current/donors 行数不一致")
    current = current.reset_index(drop=True)
    donors = donors.reset_index(drop=True)
    n_rows = len(current)
    n_attributes = len(workload.attributes)
    participate_values = np.asarray(participate)
    mask_values = np.asarray(initial_mask)
    if participate_values.shape != (n_rows,):
        raise ValueError("独立 CUDA participate 形状不一致")
    if mask_values.shape != (n_rows, n_attributes):
        raise ValueError("独立 CUDA initial_mask 形状不一致")
    participate_bool = participate_values.astype(bool, copy=False)
    mask_bool = mask_values.astype(bool, copy=True)
    current_values = current.loc[:, workload.attributes].to_numpy()
    donor_values = donors.loc[:, workload.attributes].to_numpy()
    active = participate_bool[:, None] & (current_values != donor_values)
    if np.any(mask_bool & ~active):
        raise RuntimeError("独立 CUDA 初始开关越出活跃域")
    coordinates = np.argwhere(active).astype(np.int64, copy=False)
    active_rows = np.flatnonzero(np.any(active, axis=1)).astype(
        np.int64, copy=False
    )
    row_lookup = np.full(n_rows, -1, dtype=np.int64)
    row_lookup[active_rows] = np.arange(len(active_rows), dtype=np.int64)

    if len(active_rows):
        pair = pd.concat(
            [current.iloc[active_rows], donors.iloc[active_rows]],
            ignore_index=True,
        )
        truth = _condition_truth_cuda(
            pair,
            schema,
            workload.conditions,
            torch=torch,
            device=device,
        )
        current_truth = truth[: len(active_rows)]
        donor_truth = truth[len(active_rows) :]
    else:
        current_truth = torch.empty(
            (0, len(workload.conditions)), dtype=torch.bool, device=device
        )
        donor_truth = current_truth.clone()

    n_queries = len(queries)
    failures = torch.zeros(
        (len(active_rows), n_queries), dtype=torch.int32, device=device
    )
    for query_index, indices in enumerate(
        workload.query_condition_indices
    ):
        if not indices:
            continue
        condition_t = torch.tensor(
            indices, dtype=torch.long, device=device
        )
        failures[:, query_index] = (~current_truth[:, condition_t]).sum(
            dim=1, dtype=torch.int32
        )
    current_indicators = failures == 0
    query_indices_by_attribute = tuple(
        torch.tensor(indices, dtype=torch.long, device=device)
        for indices in workload.query_indices_by_attribute
    )
    current_attribute_failures = tuple(
        _attribute_failure_matrix(
            current_truth,
            workload.condition_groups_by_attribute[attribute],
            torch=torch,
            device=device,
        )
        for attribute in range(n_attributes)
    )
    donor_attribute_failures = tuple(
        _attribute_failure_matrix(
            donor_truth,
            workload.condition_groups_by_attribute[attribute],
            torch=torch,
            device=device,
        )
        for attribute in range(n_attributes)
    )
    mask = torch.tensor(mask_bool, dtype=torch.bool, device=device)
    if len(active_rows):
        rows_t = torch.tensor(active_rows, dtype=torch.long, device=device)
        active_mask = mask.index_select(0, rows_t)
        for attribute, query_indices in enumerate(
            query_indices_by_attribute
        ):
            if query_indices.numel() == 0:
                continue
            delta = (
                donor_attribute_failures[attribute]
                - current_attribute_failures[attribute]
            )
            selected = active_mask[:, attribute].to(torch.int32)
            failures[:, query_indices] = (
                failures.index_select(1, query_indices)
                + selected.unsqueeze(1) * delta
            )
    indicators = failures == 0

    raw_counts = np.asarray(current_counts)
    if raw_counts.shape != (n_queries,) or not np.all(
        raw_counts == np.rint(raw_counts)
    ):
        raise ValueError("独立 CUDA 查询计数必须是整数向量")
    counts = torch.tensor(
        np.rint(raw_counts).astype(np.int64),
        dtype=torch.int64,
        device=device,
    )
    if len(active_rows):
        counts += (
            indicators.to(torch.int64)
            - current_indicators.to(torch.int64)
        ).sum(dim=0, dtype=torch.int64)
    target_t = torch.tensor(target, dtype=torch.float64, device=device)
    if target_t.shape != (n_queries,):
        raise ValueError("独立 CUDA 目标向量形状不一致")
    denominators = torch.maximum(
        target_t,
        torch.full_like(target_t, float(floor)),
    )
    error_terms = (
        torch.abs(target_t - counts.to(torch.float64)) / denominators
    )
    return _Plan(
        torch=torch,
        device=device,
        workload=workload,
        active_rows=active_rows,
        coordinates=coordinates,
        row_lookup=row_lookup,
        query_indices_by_attribute=query_indices_by_attribute,
        current_attribute_failures=current_attribute_failures,
        donor_attribute_failures=donor_attribute_failures,
        current_indicators=current_indicators,
        failures=failures,
        indicators=indicators,
        counts=counts,
        target=target_t,
        denominators=denominators,
        error_terms=error_terms,
        error_sum=error_terms.sum(dtype=torch.float64),
        mask=mask,
    )


def _condition(
    plan: _Plan,
    row: Any,
    attribute: int,
    local_row: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    torch = plan.torch
    query_indices = plan.query_indices_by_attribute[attribute]
    if query_indices.numel() == 0:
        value = plan.error_sum / len(plan.workload.query_condition_indices)
        empty_failure = torch.empty(
            0, dtype=torch.int32, device=plan.device
        )
        empty_indicator = torch.empty(
            0, dtype=torch.bool, device=plan.device
        )
        return (
            value,
            value,
            empty_failure,
            empty_failure,
            empty_indicator,
            empty_indicator,
        )
    current_failures = plan.current_attribute_failures[attribute][local_row]
    donor_failures = plan.donor_attribute_failures[attribute][local_row]
    previous_failures = torch.where(
        plan.mask[row, attribute], donor_failures, current_failures
    )
    base_failures = (
        plan.failures[local_row, query_indices] - previous_failures
    )
    failures0 = base_failures + current_failures
    failures1 = base_failures + donor_failures
    indicators0 = failures0 == 0
    indicators1 = failures1 == 0
    previous_indicators = plan.indicators[local_row, query_indices]
    counts0 = (
        plan.counts[query_indices]
        + indicators0.to(torch.int64)
        - previous_indicators.to(torch.int64)
    )
    counts1 = (
        plan.counts[query_indices]
        + indicators1.to(torch.int64)
        - previous_indicators.to(torch.int64)
    )
    terms0 = (
        torch.abs(plan.target[query_indices] - counts0.to(torch.float64))
        / plan.denominators[query_indices]
    )
    terms1 = (
        torch.abs(plan.target[query_indices] - counts1.to(torch.float64))
        / plan.denominators[query_indices]
    )
    previous_terms = plan.error_terms[query_indices]
    sum0 = (
        plan.error_sum
        - previous_terms.sum(dtype=torch.float64)
        + terms0.sum(dtype=torch.float64)
    )
    sum1 = (
        plan.error_sum
        - previous_terms.sum(dtype=torch.float64)
        + terms1.sum(dtype=torch.float64)
    )
    n_queries = len(plan.workload.query_condition_indices)
    return (
        sum0 / n_queries,
        sum1 / n_queries,
        failures0,
        failures1,
        indicators0,
        indicators1,
    )


def _set(
    plan: _Plan,
    row: Any,
    attribute: int,
    local_row: int,
    selected: Any,
    failures: Any,
    indicators: Any,
) -> None:
    torch = plan.torch
    query_indices = plan.query_indices_by_attribute[attribute]
    changed = selected != plan.mask[row, attribute]
    if query_indices.numel():
        previous_indicators = plan.indicators[local_row, query_indices]
        previous_counts = plan.counts[query_indices]
        candidate_counts = (
            previous_counts
            + indicators.to(torch.int64)
            - previous_indicators.to(torch.int64)
        )
        candidate_terms = (
            torch.abs(
                plan.target[query_indices]
                - candidate_counts.to(torch.float64)
            )
            / plan.denominators[query_indices]
        )
        candidate_sum = (
            plan.error_sum
            - plan.error_terms[query_indices].sum(dtype=torch.float64)
            + candidate_terms.sum(dtype=torch.float64)
        )
        plan.counts[query_indices] = torch.where(
            changed, candidate_counts, previous_counts
        )
        plan.failures[local_row, query_indices] = torch.where(
            changed, failures, plan.failures[local_row, query_indices]
        )
        plan.indicators[local_row, query_indices] = torch.where(
            changed, indicators, previous_indicators
        )
        plan.error_terms[query_indices] = torch.where(
            changed, candidate_terms, plan.error_terms[query_indices]
        )
        plan.error_sum = torch.where(changed, candidate_sum, plan.error_sum)
    plan.mask[row, attribute] = selected


def _materialize(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    attributes: Sequence[str],
    mask: np.ndarray,
) -> pd.DataFrame:
    result = current.reset_index(drop=True).copy(deep=True)
    donor_values = donors.reset_index(drop=True)
    for attribute_index, attribute in enumerate(attributes):
        selected = mask[:, attribute_index]
        if np.any(selected):
            values = result[attribute].to_numpy().copy()
            values[selected] = donor_values[attribute].to_numpy()[selected]
            result[attribute] = values
    return result


def _full_recount(
    frame: pd.DataFrame,
    schema: Any,
    plan: _Plan,
) -> np.ndarray:
    torch = plan.torch
    truth = _condition_truth_cuda(
        frame,
        schema,
        plan.workload.conditions,
        torch=torch,
        device=plan.device,
    )
    counts = torch.empty(
        len(plan.workload.query_condition_indices),
        dtype=torch.int64,
        device=plan.device,
    )
    for query_index, indices in enumerate(
        plan.workload.query_condition_indices
    ):
        if indices:
            condition_t = torch.tensor(
                indices, dtype=torch.long, device=plan.device
            )
            counts[query_index] = truth[:, condition_t].all(dim=1).sum(
                dtype=torch.int64
            )
        else:
            counts[query_index] = len(frame)
    return np.asarray(counts.detach().cpu().numpy(), dtype=np.int64)


def query_counts_cuda(
    frame: pd.DataFrame,
    schema: Any,
    queries: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """独立地在显卡上完整计算一组合取查询计数。"""

    torch, device = _require_cuda()
    workload = _compile_workload(schema, queries)
    truth = _condition_truth_cuda(
        frame,
        schema,
        workload.conditions,
        torch=torch,
        device=device,
    )
    counts = torch.empty(len(queries), dtype=torch.int64, device=device)
    for query_index, indices in enumerate(
        workload.query_condition_indices
    ):
        if indices:
            condition_t = torch.tensor(indices, dtype=torch.long, device=device)
            counts[query_index] = truth[:, condition_t].all(dim=1).sum(
                dtype=torch.int64
            )
        else:
            counts[query_index] = len(frame)
    result = np.asarray(counts.detach().cpu().numpy(), dtype=np.int64)
    torch.cuda.synchronize(device)
    return result


def replay_gap_l1_cuda(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    queries: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    current_counts: Sequence[int],
    participate: Sequence[bool],
    initial_mask: np.ndarray,
    *,
    reference_scale: float,
    seed: int,
    n_sweeps: int,
    eta: float,
    strength: float,
    floor: float,
    logit_clip: float,
    capture_trace: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """用与生产核分离的 CUDA 算术逐微步重放一个地址。"""

    plan = _prepare(
        current,
        donors,
        schema,
        queries,
        target,
        current_counts,
        participate,
        initial_mask,
        floor=floor,
    )
    torch = plan.torch
    if not isinstance(n_sweeps, int) or isinstance(n_sweeps, bool) or n_sweeps < 0:
        raise ValueError("独立 CUDA n_sweeps 必须是非负整数")
    if not 0.0 < float(eta) < 1.0:
        raise ValueError("独立 CUDA eta 必须在 (0,1)")
    if reference_scale <= 0.0 or strength <= 0.0 or logit_clip <= 0.0:
        raise ValueError("独立 CUDA 概率尺度参数必须为正")
    scale_t = torch.tensor(
        reference_scale, dtype=torch.float64, device=plan.device
    )
    eta_t = torch.tensor(eta, dtype=torch.float64, device=plan.device)
    base_logit_t = torch.log(eta_t) - torch.log1p(-eta_t)
    strength_t = torch.tensor(
        strength, dtype=torch.float64, device=plan.device
    )
    clip_t = torch.tensor(
        logit_clip, dtype=torch.float64, device=plan.device
    )

    rng = np.random.default_rng(seed)
    initial_rng_sha = _rng_sha256(rng)
    k = len(plan.coordinates)
    steps = n_sweeps * k
    coordinate_indices = np.empty(steps, dtype=np.int64)
    rolls = np.empty(steps, dtype=np.float64)
    for step in range(steps):
        coordinate_indices[step] = rng.integers(0, k)
        rolls[step] = rng.random()
    if steps:
        coordinates = np.asarray(
            plan.coordinates[coordinate_indices], dtype=np.int64
        )
        local_rows = plan.row_lookup[coordinates[:, 0]]
    else:
        coordinates = np.empty((0, 2), dtype=np.int64)
        local_rows = np.empty(0, dtype=np.int64)
    coordinates_t = torch.tensor(
        coordinates, dtype=torch.long, device=plan.device
    )
    rolls_t = torch.tensor(rolls, dtype=torch.float64, device=plan.device)
    floats = torch.empty((steps, 6), dtype=torch.float64, device=plan.device)
    booleans = torch.empty((steps, 3), dtype=torch.bool, device=plan.device)
    for step in range(steps):
        row_t = coordinates_t[step, 0]
        attribute = int(coordinates[step, 1])
        local_row = int(local_rows[step])
        before = plan.mask[row_t, attribute].clone()
        e0, e1, f0, f1, i0, i1 = _condition(
            plan, row_t, attribute, local_row
        )
        score = e0 - e1
        normalized = score / scale_t
        raw_logit = base_logit_t + strength_t * score / scale_t
        effective = raw_logit.clamp(min=-clip_t, max=clip_t)
        probability = effective.sigmoid()
        clipped = raw_logit != effective
        after = rolls_t[step] < probability
        _set(
            plan,
            row_t,
            attribute,
            local_row,
            after,
            torch.where(after, f1, f0),
            torch.where(after, i1, i0),
        )
        floats[step] = torch.stack((
            e0, e1, score, normalized, raw_logit, probability
        ))
        booleans[step] = torch.stack((before, after, clipped))

    float_values = np.asarray(
        floats.detach().cpu().numpy(), dtype=np.float64
    )
    bool_values = np.asarray(booleans.detach().cpu().numpy(), dtype=bool)
    final_mask = np.asarray(plan.mask.detach().cpu().numpy(), dtype=bool)
    incremental_counts = np.asarray(
        plan.counts.detach().cpu().numpy(), dtype=np.int64
    )
    torch.cuda.synchronize(plan.device)
    if not np.all(np.isfinite(float_values)):
        raise RuntimeError("独立 CUDA 重放出现非有限条件数值")
    if np.any((float_values[:, 5] <= 0.0) | (float_values[:, 5] >= 1.0)):
        raise RuntimeError("独立 CUDA 重放出现单向条件概率")
    if not np.array_equal(bool_values[:, 1], rolls < float_values[:, 5]):
        raise RuntimeError("独立 CUDA 开关与随机带不一致")

    trace = hashlib.sha256()
    trace_records = []
    for step in range(steps):
        row, attribute = map(int, coordinates[step])
        e0, e1, score, normalized, raw_logit, probability = map(
            float, float_values[step]
        )
        before, after, clipped = map(bool, bool_values[step])
        roll = float(rolls[step])
        trace.update(struct.pack(
            TRACE_STRUCT,
            step,
            row,
            attribute,
            e0,
            e1,
            score,
            normalized,
            raw_logit,
            probability,
            roll,
            before,
            after,
            clipped,
        ))
        if capture_trace:
            trace_records.append({
                "step": step,
                "row_index": row,
                "attribute_index": attribute,
                "e0": e0,
                "e1": e1,
                "score": score,
                "normalized_score": normalized,
                "raw_logit": raw_logit,
                "probability": probability,
                "random_roll": roll,
                "before": before,
                "after": after,
                "clipped": clipped,
            })
    table = _materialize(
        current, donors, plan.workload.attributes, final_mask
    )
    recounted = _full_recount(table, schema, plan)
    torch.cuda.synchronize(plan.device)
    if not np.array_equal(recounted, incremental_counts):
        raise RuntimeError("独立 CUDA 增量计数与完整显卡复算不一致")
    diagnostics = {
        "initial_rng_sha256": initial_rng_sha,
        "endpoint_rng_sha256": _rng_sha256(rng),
        "trace_sha256": trace.hexdigest(),
        "active_switches_k": int(k),
        "microsteps": int(steps),
        "clip_hits": int(np.sum(bool_values[:, 2], dtype=np.int64)),
        "final_query_counts": recounted,
        "backend": "independent_torch_cuda_float64",
    }
    if capture_trace:
        diagnostics["trace_records"] = trace_records
    return table, final_mask, diagnostics


def _exact_zero_spec(
    target_numerators: Sequence[int],
    target_denominator: int,
    floor: float,
) -> tuple[list[int], list[int], list[int]]:
    numerators = [int(value) for value in target_numerators]
    raw_denominators = [
        max(value, int(floor) * int(target_denominator))
        for value in numerators
    ]
    divisors = [
        math.gcd(
            math.gcd(abs(value), int(target_denominator)), denominator
        )
        for value, denominator in zip(numerators, raw_denominators)
    ]
    reduced = [
        denominator // divisor
        for denominator, divisor in zip(raw_denominators, divisors)
    ]
    common = math.lcm(*reduced)
    return numerators, divisors, [
        common // denominator for denominator in reduced
    ]


def isolated_gap_l1_scores_cuda(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    queries: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    current_counts: Sequence[int],
    *,
    floor: float,
    exact_target_numerators: Sequence[int],
    exact_target_denominator: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """独立显卡批量复算全零上下文的孤立分数。"""

    attributes = tuple(schema.attribute_names())
    plan = _prepare(
        current,
        donors,
        schema,
        queries,
        target,
        current_counts,
        np.ones(len(current), dtype=bool),
        np.zeros((len(current), len(attributes)), dtype=bool),
        floor=floor,
    )
    torch = plan.torch
    score_matrix = torch.zeros(
        (len(current), len(attributes)),
        dtype=torch.float64,
        device=plan.device,
    )
    exact_payloads = []
    for attribute, query_indices in enumerate(
        plan.query_indices_by_attribute
    ):
        rows = plan.coordinates[
            plan.coordinates[:, 1] == attribute, 0
        ]
        if len(rows) == 0 or query_indices.numel() == 0:
            continue
        local_rows = plan.row_lookup[rows]
        rows_t = torch.tensor(rows, dtype=torch.long, device=plan.device)
        local_rows_t = torch.tensor(
            local_rows, dtype=torch.long, device=plan.device
        )
        current_failures = plan.current_attribute_failures[
            attribute
        ].index_select(0, local_rows_t)
        donor_failures = plan.donor_attribute_failures[
            attribute
        ].index_select(0, local_rows_t)
        row_failures = plan.failures.index_select(
            0, local_rows_t
        ).index_select(1, query_indices)
        failures1 = row_failures - current_failures + donor_failures
        indicators0 = plan.indicators.index_select(
            0, local_rows_t
        ).index_select(1, query_indices)
        indicators1 = failures1 == 0
        counts0 = plan.counts.index_select(0, query_indices)
        counts1 = (
            counts0.unsqueeze(0)
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
        ) / len(queries)
        score_matrix[rows_t, attribute] = scores
        exact_payloads.append((
            rows,
            attribute,
            np.asarray(query_indices.detach().cpu().numpy(), dtype=np.int64),
            np.asarray(counts0.detach().cpu().numpy(), dtype=np.int64),
            np.asarray(counts1.detach().cpu().numpy(), dtype=np.int64),
        ))
    coordinates = np.asarray(plan.coordinates, dtype=np.int64).reshape(-1, 2)
    if len(coordinates):
        scores = np.asarray(
            score_matrix[
                torch.tensor(
                    coordinates[:, 0], dtype=torch.long, device=plan.device
                ),
                torch.tensor(
                    coordinates[:, 1], dtype=torch.long, device=plan.device
                ),
            ].detach().cpu().numpy(),
            dtype=np.float64,
        )
    else:
        scores = np.empty(0, dtype=np.float64)
    torch.cuda.synchronize(plan.device)

    numerators, divisors, weights = _exact_zero_spec(
        exact_target_numerators, exact_target_denominator, floor
    )
    positions = {
        (int(row), int(attribute)): index
        for index, (row, attribute) in enumerate(coordinates)
    }
    for rows, attribute, query_indices, counts0, counts1 in exact_payloads:
        for local_index, row in enumerate(rows):
            units = 0
            for offset, query_raw in enumerate(query_indices):
                query = int(query_raw)
                residual0 = abs(
                    numerators[query]
                    - int(counts0[offset]) * exact_target_denominator
                )
                residual1 = abs(
                    numerators[query]
                    - int(counts1[local_index, offset])
                    * exact_target_denominator
                )
                units += (
                    residual0 // divisors[query]
                    - residual1 // divisors[query]
                ) * weights[query]
            if units == 0:
                scores[positions[(int(row), int(attribute))]] = 0.0
    current_error = float(
        (plan.error_sum / len(queries)).detach().cpu().item()
    )
    return coordinates, scores, current_error
