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


def _validate_independent_batch_inputs(
    donor_tables: Sequence[Any],
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    seeds: Sequence[int],
    *,
    n_sweeps: int,
    reference_scale: float,
    eta: float,
    strength: float,
    floor: float,
    logit_clip: float,
) -> int:
    lengths = (
        len(donor_tables),
        len(participates),
        len(initial_masks),
        len(seeds),
    )
    if lengths[0] == 0 or any(value != lengths[0] for value in lengths[1:]):
        raise ValueError("独立批量输入与 seeds 必须是同长度非空序列")
    if any(
        not isinstance(seed, (int, np.integer)) or isinstance(seed, bool)
        for seed in seeds
    ):
        raise ValueError("独立批量 seeds 中每一项都必须是整数")
    if (
        not isinstance(n_sweeps, int)
        or isinstance(n_sweeps, bool)
        or n_sweeps < 0
    ):
        raise ValueError("独立批量 n_sweeps 必须是非负整数")
    if not np.isfinite(eta) or not 0.0 < float(eta) < 1.0:
        raise ValueError("独立批量 eta 必须是有限的开区间概率")
    for name, value in (
        ("reference_scale", reference_scale),
        ("strength", strength),
        ("floor", floor),
        ("logit_clip", logit_clip),
    ):
        if not np.isfinite(value) or float(value) <= 0.0:
            raise ValueError(f"独立批量 {name} 必须是有限正数")
    return lengths[0]


def _independent_padded_query_layout(
    workload: _Workload,
    *,
    torch: Any,
    device: Any,
) -> tuple[Any, Any, int]:
    """独立建立按属性填充的查询索引和互异哨兵。"""

    widths = [len(indices) for indices in workload.query_indices_by_attribute]
    padded_width = max([1] + widths)
    n_attributes = len(workload.attributes)
    n_queries = len(workload.query_condition_indices)
    indices = np.empty((n_attributes, padded_width), dtype=np.int64)
    valid = np.zeros((n_attributes, padded_width), dtype=bool)
    sentinels = np.arange(
        n_queries, n_queries + padded_width, dtype=np.int64
    )
    for attribute, query_indices in enumerate(
        workload.query_indices_by_attribute
    ):
        width = len(query_indices)
        if width:
            indices[attribute, :width] = query_indices
            valid[attribute, :width] = True
        indices[attribute, width:] = sentinels[: padded_width - width]
    return (
        torch.as_tensor(indices, dtype=torch.long, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device),
        padded_width,
    )


def _advance_independent_batch_step(
    *,
    failure_counts: Any,
    row_indicators: Any,
    counts: Any,
    error_terms: Any,
    error_sum: Any,
    masks: Any,
    current_failures: Any,
    donor_failures: Any,
    padded_query_indices: Any,
    padded_query_valid: Any,
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
) -> tuple[Any, Any]:
    """不调用生产微步代码，独立推进每个地址的一个有序微步。"""

    row_indices = coordinates[:, 0]
    attribute_indices = coordinates[:, 1]
    query_indices = padded_query_indices.index_select(0, attribute_indices)
    query_valid = padded_query_valid.index_select(0, attribute_indices)
    current = current_failures[
        batch_indices, attribute_indices, local_rows
    ]
    donor = donor_failures[
        batch_indices, attribute_indices, local_rows
    ]
    before = masks[batch_indices, row_indices, attribute_indices]
    previous_attribute_failures = torch.where(
        before.unsqueeze(1), donor, current
    )

    full_failure_rows = failure_counts[batch_indices, local_rows]
    full_indicator_rows = row_indicators[batch_indices, local_rows]
    failure_rows = full_failure_rows.gather(1, query_indices)
    indicator_rows = full_indicator_rows.gather(1, query_indices)
    count_rows = counts.gather(1, query_indices)
    old_error_rows = error_terms.gather(1, query_indices)
    base_failures = failure_rows - previous_attribute_failures
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

    old_sum = torch.where(query_valid, old_error_rows, 0.0).sum(
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
    after = torch.where(step_valid, proposed, before)
    changed = step_valid & (after != before)

    selected_failures = torch.where(
        after.unsqueeze(1), failures1, failures0
    )
    selected_indicators = torch.where(
        after.unsqueeze(1), indicators1, indicators0
    )
    selected_counts = torch.where(
        after.unsqueeze(1), counts1, counts0
    )
    selected_terms = torch.where(after.unsqueeze(1), terms1, terms0)
    selected_sum = torch.where(after, sum1, sum0)
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
    counts.scatter_(
        1,
        query_indices,
        torch.where(update, selected_counts, count_rows),
    )
    error_terms.scatter_(
        1,
        query_indices,
        torch.where(update, selected_terms, old_error_rows),
    )
    error_sum.copy_(torch.where(changed, selected_sum, error_sum))
    masks[batch_indices, row_indices, attribute_indices] = after

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
    booleans = torch.stack((before, after, clipped), dim=1)
    booleans &= step_valid.unsqueeze(1)
    return floats, booleans


def replay_gap_l1_batched_cuda(
    current: pd.DataFrame,
    donor_tables: Sequence[pd.DataFrame],
    schema: Any,
    queries: Sequence[Mapping[str, Any]],
    target: Sequence[float],
    current_counts: Sequence[int],
    participates: Sequence[Sequence[bool]],
    initial_masks: Sequence[np.ndarray],
    *,
    reference_scale: float,
    seeds: Sequence[int],
    n_sweeps: int,
    eta: float,
    strength: float,
    floor: float,
    logit_clip: float,
    capture_trace: bool = False,
) -> dict[str, Any]:
    """独立重放不同地址的批量显卡算术，不调用生产批量核。"""

    batch_size = _validate_independent_batch_inputs(
        donor_tables,
        participates,
        initial_masks,
        seeds,
        n_sweeps=n_sweeps,
        reference_scale=reference_scale,
        eta=eta,
        strength=strength,
        floor=floor,
        logit_clip=logit_clip,
    )
    plans = [
        _prepare(
            current,
            donor_tables[index],
            schema,
            queries,
            target,
            current_counts,
            participates[index],
            initial_masks[index],
            floor=floor,
        )
        for index in range(batch_size)
    ]
    first = plans[0]
    torch = first.torch
    device = first.device
    n_attributes = len(first.workload.attributes)
    n_queries = len(first.workload.query_condition_indices)
    active_row_counts = [len(plan.active_rows) for plan in plans]
    maximum_active_rows = max(active_row_counts)
    active_switches = [len(plan.coordinates) for plan in plans]
    microsteps_by_address = [
        n_sweeps * active_count for active_count in active_switches
    ]
    maximum_microsteps = max(microsteps_by_address)
    (
        padded_query_indices,
        padded_query_valid,
        padded_query_width,
    ) = _independent_padded_query_layout(
        first.workload, torch=torch, device=device
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
    counts = torch.zeros(
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
            ] = plan.failures
            row_indicators[
                batch_index, :active_rows, :n_queries
            ] = plan.indicators
        counts[batch_index, :n_queries] = plan.counts
        error_terms[batch_index, :n_queries] = plan.error_terms
        for attribute, query_indices in enumerate(
            plan.workload.query_indices_by_attribute
        ):
            width = len(query_indices)
            if active_rows and width:
                current_failures[
                    batch_index,
                    attribute,
                    :active_rows,
                    :width,
                ] = plan.current_attribute_failures[attribute]
                donor_failures[
                    batch_index,
                    attribute,
                    :active_rows,
                    :width,
                ] = plan.donor_attribute_failures[attribute]

    target_t = torch.zeros(
        extended_query_count, dtype=torch.float64, device=device
    )
    target_t[:n_queries] = first.target
    denominators_t = torch.ones_like(target_t)
    denominators_t[:n_queries] = first.denominators
    scale_t = torch.tensor(
        float(reference_scale), dtype=torch.float64, device=device
    )
    eta_t = torch.tensor(float(eta), dtype=torch.float64, device=device)
    base_logit_t = torch.log(eta_t) - torch.log1p(-eta_t)
    strength_t = torch.tensor(
        float(strength), dtype=torch.float64, device=device
    )
    clip_t = torch.tensor(
        float(logit_clip), dtype=torch.float64, device=device
    )

    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    initial_rng_hashes = tuple(_rng_sha256(rng) for rng in rngs)
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
            plan.coordinates[coordinate_indices], dtype=np.int64
        )
        coordinate_tapes[batch_index, :microsteps] = coordinates
        local_row_tapes[batch_index, :microsteps] = (
            plan.row_lookup[coordinates[:, 0]]
        )
        roll_tapes[batch_index, :microsteps] = rolls
        valid_step_mask[batch_index, :microsteps] = True
    endpoint_rng_hashes = tuple(_rng_sha256(rng) for rng in rngs)

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
    for step in range(maximum_microsteps):
        floats, booleans = _advance_independent_batch_step(
            failure_counts=failure_counts,
            row_indicators=row_indicators,
            counts=counts,
            error_terms=error_terms,
            error_sum=error_sum,
            masks=masks,
            current_failures=current_failures,
            donor_failures=donor_failures,
            padded_query_indices=padded_query_indices,
            padded_query_valid=padded_query_valid,
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

    padded_float_values = np.asarray(
        float_values_t.detach().cpu().numpy(), dtype=np.float64
    )
    padded_bool_values = np.asarray(
        bool_values_t.detach().cpu().numpy(), dtype=bool
    )
    final_counts = np.asarray(
        counts[:, :n_queries].detach().cpu().numpy(), dtype=np.int64
    )
    final_masks = np.asarray(masks.detach().cpu().numpy(), dtype=bool)
    torch.cuda.synchronize(device)
    valid_float_values = padded_float_values[valid_step_mask]
    if not np.all(np.isfinite(valid_float_values)):
        raise RuntimeError("独立批量重放出现非有限条件数值")
    probabilities = padded_float_values[:, :, 5][valid_step_mask]
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise RuntimeError("独立批量重放出现单向条件概率")
    if not np.array_equal(
        padded_bool_values[:, :, 1][valid_step_mask],
        roll_tapes[valid_step_mask] < probabilities,
    ):
        raise RuntimeError("独立批量开关与随机带不一致")
    if np.any(padded_float_values[~valid_step_mask] != 0.0) or np.any(
        padded_bool_values[~valid_step_mask]
    ):
        raise RuntimeError("独立批量填充微步意外产生输出")

    tables = []
    traces = []
    trace_records_by_address = []
    diagnostics_by_address = []
    coordinates_by_address = []
    rolls_by_address = []
    floats_by_address = []
    booleans_by_address = []
    for batch_index, (plan, donor_table) in enumerate(
        zip(plans, donor_tables)
    ):
        microsteps = microsteps_by_address[batch_index]
        coordinates = coordinate_tapes[batch_index, :microsteps].copy()
        rolls = roll_tapes[batch_index, :microsteps].copy()
        float_values = padded_float_values[
            batch_index, :microsteps
        ].copy()
        bool_values = padded_bool_values[batch_index, :microsteps].copy()
        trace = hashlib.sha256()
        trace_records = []
        for step in range(microsteps):
            row, attribute = map(int, coordinates[step])
            e0, e1, score, normalized, raw_logit, probability, roll = map(
                float, float_values[step]
            )
            before, after, clipped = map(bool, bool_values[step])
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
            current,
            donor_table,
            plan.workload.attributes,
            final_masks[batch_index],
        )
        recounted = _full_recount(table, schema, plan)
        if not np.array_equal(recounted, final_counts[batch_index]):
            raise RuntimeError("独立批量增量计数与完整显卡复算不一致")
        tables.append(table)
        traces.append(trace.hexdigest())
        trace_records_by_address.append(trace_records)
        diagnostics_by_address.append({
            "initial_rng_sha256": initial_rng_hashes[batch_index],
            "endpoint_rng_sha256": endpoint_rng_hashes[batch_index],
            "trace_sha256": trace.hexdigest(),
            "active_switches_k": int(active_switches[batch_index]),
            "microsteps": int(microsteps),
            "clip_hits": int(
                np.sum(bool_values[:, 2], dtype=np.int64)
            ),
            "final_query_counts": recounted,
            "backend": "independent_torch_cuda_float64_batched",
            "no_gate": True,
        })
        coordinates_by_address.append(coordinates)
        rolls_by_address.append(rolls)
        floats_by_address.append(float_values)
        booleans_by_address.append(bool_values)
    torch.cuda.synchronize(device)

    result = {
        "backend": "independent_torch_cuda_float64_batched",
        "batch_execution_format": (
            "issue53_gap_l1_independent_batched_audit_v1"
        ),
        "batch_size": int(batch_size),
        "active_switches_k_by_address": tuple(map(int, active_switches)),
        "microsteps_by_address": tuple(map(int, microsteps_by_address)),
        "maximum_padded_microsteps": int(maximum_microsteps),
        "padding_microsteps": int(
            batch_size * maximum_microsteps - sum(microsteps_by_address)
        ),
        "valid_step_mask": valid_step_mask,
        "reference_scale": float(reference_scale),
        "tables": tuple(tables),
        "masks": final_masks,
        "final_query_counts": final_counts,
        "trace_sha256": tuple(traces),
        "coordinates": tuple(coordinates_by_address),
        "random_rolls": tuple(rolls_by_address),
        "float_values": tuple(floats_by_address),
        "bool_values": tuple(booleans_by_address),
        "diagnostics": tuple(diagnostics_by_address),
        "no_gate": True,
        "acceptance_rejection_or_selection_performed": False,
        "production_kernel_called": False,
        "formal_pipeline_enabled": False,
    }
    if capture_trace:
        result["trace_records"] = tuple(trace_records_by_address)
    return result


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
