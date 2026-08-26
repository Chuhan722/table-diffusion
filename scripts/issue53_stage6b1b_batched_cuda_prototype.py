"""第 6B-1B 阶段多地址 CUDA 批量原型。

本模块只用于同尺寸人工负载的性能与一致性试验。它不被正式采集、结构审计、
独立算术审计或公共生成器导入，也不定义新的方法参数。首版原型故意限定为：
一批地址共享 current、donors、participate 和活跃坐标，只允许初始开关与随机带
不同。这个边界足以测量把逐地址 Python 循环改成批量张量更新后的核心扫描收益，
但不冒充已经完成的生产实现。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from table_diffevo import gap_l1_diffusion as production


PROTOTYPE_BACKEND = "artificial_batched_torch_cuda_float64"


def _batched_microstep(
    *,
    failure_counts: Any,
    row_indicators: Any,
    plan_counts: Any,
    error_terms: Any,
    error_sum: Any,
    masks: Any,
    current_failures: Any,
    donor_failures: Any,
    affected_by_attribute: Any,
    target: Any,
    denominators: Any,
    scale: Any,
    base_logit: Any,
    strength: Any,
    clip: Any,
    batch_indices: Any,
    coordinates: Any,
    local_rows: Any,
    rolls: Any,
) -> tuple[Any, Any]:
    """并行推进一批地址各自的一个严格有序微步。"""

    if failure_counts.__class__.__module__.split(".")[0] != "torch":
        raise TypeError("批量原型状态必须是 PyTorch 张量")

    row_indices = coordinates[:, 0]
    attribute_indices = coordinates[:, 1]
    affected = affected_by_attribute.index_select(0, attribute_indices)
    current = current_failures[
        attribute_indices, local_rows
    ]
    donor = donor_failures[
        attribute_indices, local_rows
    ]
    old_selected = masks[
        batch_indices, row_indices, attribute_indices
    ]
    old_failures = torch.where(
        old_selected.unsqueeze(1), donor, current
    )

    failure_rows = failure_counts[batch_indices, local_rows]
    indicator_rows = row_indicators[batch_indices, local_rows]
    base_failures = failure_rows - old_failures
    failures0 = base_failures + current
    failures1 = base_failures + donor
    indicators0 = failures0 == 0
    indicators1 = failures1 == 0
    counts0 = (
        plan_counts
        + indicators0.to(torch.int64)
        - indicator_rows.to(torch.int64)
    )
    counts1 = (
        plan_counts
        + indicators1.to(torch.int64)
        - indicator_rows.to(torch.int64)
    )
    terms0 = (
        torch.abs(target.unsqueeze(0) - counts0.to(torch.float64))
        / denominators.unsqueeze(0)
    )
    terms1 = (
        torch.abs(target.unsqueeze(0) - counts1.to(torch.float64))
        / denominators.unsqueeze(0)
    )

    old_sum = torch.where(
        affected, error_terms, 0.0
    ).sum(dim=1, dtype=torch.float64)
    sum0 = (
        error_sum
        - old_sum
        + torch.where(affected, terms0, 0.0).sum(
            dim=1, dtype=torch.float64
        )
    )
    sum1 = (
        error_sum
        - old_sum
        + torch.where(affected, terms1, 0.0).sum(
            dim=1, dtype=torch.float64
        )
    )
    query_count = target.numel()
    e0 = sum0 / query_count
    e1 = sum1 / query_count
    score = e0 - e1
    normalized = score / scale
    raw_logit = base_logit + strength * normalized
    effective_logit = raw_logit.clamp(min=-clip, max=clip)
    probabilities = effective_logit.sigmoid()
    clipped = raw_logit != effective_logit
    selected = rolls < probabilities
    changed = selected != old_selected

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
    update = changed.unsqueeze(1) & affected

    failure_counts[batch_indices, local_rows] = torch.where(
        update, selected_failures, failure_rows
    )
    row_indicators[batch_indices, local_rows] = torch.where(
        update, selected_indicators, indicator_rows
    )
    plan_counts.copy_(torch.where(update, selected_counts, plan_counts))
    error_terms.copy_(torch.where(update, selected_terms, error_terms))
    error_sum.copy_(torch.where(changed, selected_sum, error_sum))
    masks[batch_indices, row_indices, attribute_indices] = selected

    floats = torch.stack(
        (
            e0,
            e1,
            score,
            normalized,
            raw_logit,
            probabilities,
            rolls,
        ),
        dim=1,
    )
    booleans = torch.stack(
        (old_selected, selected, clipped), dim=1
    )
    return floats, booleans


def _build_static_failure_tensors(plan: Any) -> tuple[Any, Any, Any]:
    torch = plan.torch
    attribute_count = len(plan.compiled.attribute_names)
    active_row_count = len(plan.active_rows)
    query_count = plan.compiled.n_queries
    current = torch.zeros(
        (attribute_count, active_row_count, query_count),
        dtype=torch.int32,
        device=plan.device,
    )
    donor = torch.zeros_like(current)
    affected = torch.zeros(
        (attribute_count, query_count),
        dtype=torch.bool,
        device=plan.device,
    )
    for attribute_index, query_indices in enumerate(
        plan.query_indices_by_attribute
    ):
        if query_indices.numel() == 0:
            continue
        current[attribute_index, :, query_indices] = (
            plan.current_attribute_failures[attribute_index]
        )
        donor[attribute_index, :, query_indices] = (
            plan.donor_attribute_failures[attribute_index]
        )
        affected[attribute_index, query_indices] = True
    return current, donor, affected


def run_same_workload_batched_prototype(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    queries: list[dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participate: Any,
    initial_masks: Sequence[Any],
    reference_scale: float,
    seeds: Sequence[int],
    n_sweeps: int = 8,
    eta: float = 0.5,
    strength: float = 2.0,
    floor: float = 8.0,
    logit_clip: float = 30.0,
) -> dict[str, Any]:
    """批量运行共享静态工作量、不同初值与随机带的人工地址。

    返回逐地址表、开关、增量查询计数、微步轨迹和分段耗时。调用方必须用
    现有单地址生产实现逐地址对拍；本函数本身不具有生产资格。
    """

    if len(initial_masks) == 0 or len(initial_masks) != len(seeds):
        raise ValueError("initial_masks 与 seeds 必须是同长度非空序列")
    batch_size = len(initial_masks)
    compiled = production.compile_gap_l1_workload(schema, queries)
    started = time.perf_counter()
    plans = [
        production._prepare_cuda_plan(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            participate,
            initial_mask,
            floor=floor,
            compiled_workload=compiled,
        )
        for initial_mask in initial_masks
    ]
    first = plans[0]
    for plan in plans[1:]:
        if not np.array_equal(
            plan.active_coordinates, first.active_coordinates
        ) or not np.array_equal(plan.active_rows, first.active_rows):
            raise ValueError(
                "首版批量原型要求所有地址共享活跃行与活跃坐标"
            )

    torch = first.torch
    current_failures, donor_failures, affected = (
        _build_static_failure_tensors(first)
    )
    failure_counts = torch.stack(
        [plan.failure_counts for plan in plans], dim=0
    )
    row_indicators = torch.stack(
        [plan.row_indicators for plan in plans], dim=0
    )
    plan_counts = torch.stack([plan.plan_counts for plan in plans], dim=0)
    error_terms = torch.stack([plan.error_terms for plan in plans], dim=0)
    error_sum = torch.stack([plan.error_sum for plan in plans], dim=0)
    masks = torch.stack([plan.mask for plan in plans], dim=0)
    target_t = first.target
    denominators_t = first.denominators
    (
        scale_value,
        scale_t,
        base_logit_t,
        strength_t,
        clip_t,
    ) = production._cuda_probability_parameters(
        first,
        reference_scale,
        eta=eta,
        strength=strength,
        logit_clip=logit_clip,
    )

    sweeps = production._require_nonnegative_integer(n_sweeps, "n_sweeps")
    active_coordinates = np.asarray(
        first.active_coordinates, dtype=np.int64
    )
    active_count = len(active_coordinates)
    microsteps = sweeps * active_count
    coordinate_tapes = np.empty(
        (batch_size, microsteps, 2), dtype=np.int64
    )
    local_row_tapes = np.empty(
        (batch_size, microsteps), dtype=np.int64
    )
    roll_tapes = np.empty(
        (batch_size, microsteps), dtype=np.float64
    )
    for batch_index, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        coordinate_indices = np.empty(microsteps, dtype=np.int64)
        rolls = np.empty(microsteps, dtype=np.float64)
        for step in range(microsteps):
            coordinate_indices[step] = rng.integers(0, active_count)
            rolls[step] = rng.random()
        coordinates = active_coordinates[coordinate_indices]
        coordinate_tapes[batch_index] = coordinates
        local_row_tapes[batch_index] = first.row_lookup[coordinates[:, 0]]
        roll_tapes[batch_index] = rolls

    coordinate_tapes_t = torch.as_tensor(
        coordinate_tapes, dtype=torch.long, device=first.device
    )
    local_row_tapes_t = torch.as_tensor(
        local_row_tapes, dtype=torch.long, device=first.device
    )
    roll_tapes_t = torch.as_tensor(
        roll_tapes, dtype=torch.float64, device=first.device
    )
    float_values_t = torch.empty(
        (batch_size, microsteps, 7),
        dtype=torch.float64,
        device=first.device,
    )
    bool_values_t = torch.empty(
        (batch_size, microsteps, 3),
        dtype=torch.bool,
        device=first.device,
    )
    batch_indices = torch.arange(
        batch_size, dtype=torch.long, device=first.device
    )
    torch.cuda.synchronize(first.device)
    prepared_elapsed = time.perf_counter() - started

    scan_started = time.perf_counter()
    for step in range(microsteps):
        floats, booleans = _batched_microstep(
            failure_counts=failure_counts,
            row_indicators=row_indicators,
            plan_counts=plan_counts,
            error_terms=error_terms,
            error_sum=error_sum,
            masks=masks,
            current_failures=current_failures,
            donor_failures=donor_failures,
            affected_by_attribute=affected,
            target=target_t,
            denominators=denominators_t,
            scale=scale_t,
            base_logit=base_logit_t,
            strength=strength_t,
            clip=clip_t,
            batch_indices=batch_indices,
            coordinates=coordinate_tapes_t[:, step],
            local_rows=local_row_tapes_t[:, step],
            rolls=roll_tapes_t[:, step],
        )
        float_values_t[:, step] = floats
        bool_values_t[:, step] = booleans
    float_values = np.asarray(
        float_values_t.detach().cpu().numpy(), dtype=np.float64
    )
    bool_values = np.asarray(
        bool_values_t.detach().cpu().numpy(), dtype=bool
    )
    final_counts = np.asarray(
        plan_counts.detach().cpu().numpy(), dtype=np.int64
    )
    final_masks = np.asarray(
        masks.detach().cpu().numpy(), dtype=bool
    )
    torch.cuda.synchronize(first.device)
    scan_elapsed = time.perf_counter() - scan_started

    if not np.all(np.isfinite(float_values)):
        raise RuntimeError("批量原型产生非有限微步数值")
    probabilities = float_values[:, :, 5]
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise RuntimeError("批量原型产生精确 0/1 条件概率")
    if not np.array_equal(
        bool_values[:, :, 1], roll_tapes < probabilities
    ):
        raise RuntimeError("批量原型开关与冻结随机带不一致")

    post_started = time.perf_counter()
    trace_hashes = []
    tables = []
    for batch_index in range(batch_size):
        digest = hashlib.sha256()
        for step in range(microsteps):
            row_index, attribute_index = map(
                int, coordinate_tapes[batch_index, step]
            )
            e0, e1, score, normalized, raw_logit, probability, roll = map(
                float, float_values[batch_index, step]
            )
            before, after, clipped = map(
                bool, bool_values[batch_index, step]
            )
            production._update_trace(
                digest,
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
        trace_hashes.append(digest.hexdigest())
        tables.append(production._materialize_copy_table(
            current,
            donors,
            first.compiled.attribute_names,
            final_masks[batch_index],
        ))
    post_elapsed = time.perf_counter() - post_started

    return {
        "backend": PROTOTYPE_BACKEND,
        "batch_size": batch_size,
        "active_switches_k": active_count,
        "microsteps_per_address": microsteps,
        "reference_scale": scale_value,
        "tables": tuple(tables),
        "masks": final_masks,
        "final_query_counts": final_counts,
        "trace_sha256": tuple(trace_hashes),
        "coordinates": coordinate_tapes,
        "float_values": float_values,
        "bool_values": bool_values,
        "timings": {
            "prepare_elapsed_sec": prepared_elapsed,
            "scan_elapsed_sec": scan_elapsed,
            "post_elapsed_sec": post_elapsed,
            "total_elapsed_sec": (
                prepared_elapsed + scan_elapsed + post_elapsed
            ),
        },
        "formal_state_or_address_read": False,
        "production_pipeline_enabled": False,
    }
