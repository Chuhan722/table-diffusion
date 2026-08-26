"""第 6B-1B 阶段多地址 CUDA（显卡计算）批量人工原型。

本模块只用于人工负载的性能与一致性试验。它不被正式采集、结构审计、
独立算术审计或公共生成器导入，也不定义新的方法参数。当前原型允许同一
状态内的地址具有不同 donors（供体表）、participate（参与行）、活跃坐标、
K、初始开关和随机带；不同长度使用填充与有效位掩码对齐，每个地址内部
仍严格保持原来的坐标抽取顺序和固定扫描次数。
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from table_diffevo import gap_l1_diffusion as production


PROTOTYPE_BACKEND = "artificial_variable_batched_torch_cuda_float64_v2"


def _padded_batched_microstep(
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
    batch_indices: Any,
    coordinates: Any,
    local_rows: Any,
    rolls: Any,
    step_valid: Any,
) -> tuple[Any, Any]:
    """并行推进多个地址各自的一个严格有序微步。

    已经走完的短地址仍占据批量中的填充位置，但 ``step_valid=False`` 会使其
    保持原状态。查询列表也按属性填充；填充查询使用互不重复的哨兵位置，
    避免原地散射时出现重复索引。
    """

    if failure_counts.__class__.__module__.split(".")[0] != "torch":
        raise TypeError("批量原型状态必须是 PyTorch 张量")

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
    floats = torch.where(
        step_valid.unsqueeze(1), floats, torch.zeros_like(floats)
    )
    booleans = torch.stack(
        (old_selected, selected, clipped), dim=1
    )
    booleans &= step_valid.unsqueeze(1)
    return floats, booleans


def _validate_batch_lengths(
    donor_tables: Sequence[Any],
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    seeds: Sequence[int],
) -> int:
    lengths = (
        len(donor_tables),
        len(participates),
        len(initial_masks),
        len(seeds),
    )
    if lengths[0] == 0 or any(value != lengths[0] for value in lengths[1:]):
        raise ValueError("各地址输入与 seeds 必须是同长度非空序列")
    return lengths[0]


def _padded_query_layout(
    compiled: Any,
    *,
    torch_module: Any,
    device: Any,
) -> tuple[Any, Any, int]:
    """建立按属性排列的查询索引，并为短列表添加独立哨兵。"""

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
        torch_module.as_tensor(indices, dtype=torch.long, device=device),
        torch_module.as_tensor(valid, dtype=torch.bool, device=device),
        padded_width,
    )


def run_variable_workloads_batched_prototype(
    current: pd.DataFrame,
    donor_tables: Sequence[pd.DataFrame],
    schema: Any,
    queries: list[dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    reference_scale: float,
    seeds: Sequence[int],
    n_sweeps: int = 8,
    eta: float = 0.5,
    strength: float = 2.0,
    floor: float = 8.0,
    logit_clip: float = 30.0,
) -> dict[str, Any]:
    """批量运行同一状态内结构不同的多个人工地址。

    current（当前表）、查询、目标和当前查询计数由同一状态共享；每个地址
    分别提供供体、参与行、初始开关与随机种子。返回逐地址表、开关、查询
    计数和微步轨迹，供调用方与现有单地址实现逐项对拍。
    """

    batch_size = _validate_batch_lengths(
        donor_tables, participates, initial_masks, seeds
    )
    sweeps = production._require_nonnegative_integer(n_sweeps, "n_sweeps")
    compiled = production.compile_gap_l1_workload(schema, queries)
    started = time.perf_counter()
    plans = [
        production._prepare_cuda_plan(
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
    torch_module = first.torch
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
    ) = _padded_query_layout(
        compiled, torch_module=torch_module, device=device
    )
    extended_query_count = n_queries + padded_query_width

    failure_counts = torch_module.zeros(
        (batch_size, maximum_active_rows, extended_query_count),
        dtype=torch.int32,
        device=device,
    )
    row_indicators = torch_module.ones(
        (batch_size, maximum_active_rows, extended_query_count),
        dtype=torch.bool,
        device=device,
    )
    plan_counts = torch_module.zeros(
        (batch_size, extended_query_count),
        dtype=torch.int64,
        device=device,
    )
    error_terms = torch_module.zeros(
        (batch_size, extended_query_count),
        dtype=torch.float64,
        device=device,
    )
    error_sum = torch_module.stack(
        [plan.error_sum for plan in plans], dim=0
    )
    masks = torch_module.stack([plan.mask for plan in plans], dim=0)
    current_failures = torch_module.zeros(
        (
            batch_size,
            n_attributes,
            maximum_active_rows,
            padded_query_width,
        ),
        dtype=torch.int32,
        device=device,
    )
    donor_failures = torch_module.zeros_like(current_failures)
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

    target_t = torch_module.zeros(
        extended_query_count, dtype=torch.float64, device=device
    )
    target_t[:n_queries] = first.target
    denominators_t = torch_module.ones_like(target_t)
    denominators_t[:n_queries] = first.denominators
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
    for batch_index, (plan, seed) in enumerate(zip(plans, seeds)):
        microsteps = microsteps_by_address[batch_index]
        active_count = active_switches[batch_index]
        if microsteps == 0:
            continue
        rng = np.random.default_rng(int(seed))
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

    coordinate_tapes_t = torch_module.as_tensor(
        coordinate_tapes, dtype=torch.long, device=device
    )
    local_row_tapes_t = torch_module.as_tensor(
        local_row_tapes, dtype=torch.long, device=device
    )
    roll_tapes_t = torch_module.as_tensor(
        roll_tapes, dtype=torch.float64, device=device
    )
    valid_step_mask_t = torch_module.as_tensor(
        valid_step_mask, dtype=torch.bool, device=device
    )
    float_values_t = torch_module.zeros(
        (batch_size, maximum_microsteps, 7),
        dtype=torch.float64,
        device=device,
    )
    bool_values_t = torch_module.zeros(
        (batch_size, maximum_microsteps, 3),
        dtype=torch.bool,
        device=device,
    )
    batch_indices = torch_module.arange(
        batch_size, dtype=torch.long, device=device
    )
    del plan
    del first
    del plans
    torch_module.cuda.synchronize(device)
    prepared_elapsed = time.perf_counter() - started

    scan_started = time.perf_counter()
    for step in range(maximum_microsteps):
        floats, booleans = _padded_batched_microstep(
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
        plan_counts[:, :n_queries].detach().cpu().numpy(), dtype=np.int64
    )
    final_masks = np.asarray(
        masks.detach().cpu().numpy(), dtype=bool
    )
    torch_module.cuda.synchronize(device)
    scan_elapsed = time.perf_counter() - scan_started

    valid_float_values = padded_float_values[valid_step_mask]
    if not np.all(np.isfinite(valid_float_values)):
        raise RuntimeError("批量原型产生非有限微步数值")
    probabilities = padded_float_values[:, :, 5][valid_step_mask]
    if np.any((probabilities <= 0.0) | (probabilities >= 1.0)):
        raise RuntimeError("批量原型产生精确 0/1 条件概率")
    if not np.array_equal(
        padded_bool_values[:, :, 1][valid_step_mask],
        roll_tapes[valid_step_mask] < probabilities,
    ):
        raise RuntimeError("批量原型开关与冻结随机带不一致")
    if np.any(padded_float_values[~valid_step_mask] != 0.0) or np.any(
        padded_bool_values[~valid_step_mask]
    ):
        raise RuntimeError("批量原型的填充微步意外产生输出")

    post_started = time.perf_counter()
    trace_hashes = []
    tables = []
    coordinates_by_address = []
    rolls_by_address = []
    floats_by_address = []
    booleans_by_address = []
    for batch_index, microsteps in enumerate(microsteps_by_address):
        coordinates = coordinate_tapes[batch_index, :microsteps].copy()
        rolls = roll_tapes[batch_index, :microsteps].copy()
        float_values = padded_float_values[batch_index, :microsteps].copy()
        bool_values = padded_bool_values[batch_index, :microsteps].copy()
        digest = hashlib.sha256()
        for step in range(microsteps):
            row_index, attribute_index = map(int, coordinates[step])
            e0, e1, score, normalized, raw_logit, probability, roll = map(
                float, float_values[step]
            )
            before, after, clipped = map(bool, bool_values[step])
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
        coordinates_by_address.append(coordinates)
        rolls_by_address.append(rolls)
        floats_by_address.append(float_values)
        booleans_by_address.append(bool_values)
        tables.append(production._materialize_copy_table(
            current,
            donor_tables[batch_index],
            compiled.attribute_names,
            final_masks[batch_index],
        ))
    post_elapsed = time.perf_counter() - post_started

    return {
        "backend": PROTOTYPE_BACKEND,
        "batch_size": batch_size,
        "active_switches_k_by_address": tuple(map(int, active_switches)),
        "microsteps_by_address": tuple(map(int, microsteps_by_address)),
        "maximum_padded_microsteps": int(maximum_microsteps),
        "padding_microsteps": int(
            batch_size * maximum_microsteps - sum(microsteps_by_address)
        ),
        "valid_step_mask": valid_step_mask,
        "reference_scale": scale_value,
        "tables": tuple(tables),
        "masks": final_masks,
        "final_query_counts": final_counts,
        "trace_sha256": tuple(trace_hashes),
        "coordinates": tuple(coordinates_by_address),
        "random_rolls": tuple(rolls_by_address),
        "float_values": tuple(floats_by_address),
        "bool_values": tuple(booleans_by_address),
        "timings": {
            "prepare_elapsed_sec": prepared_elapsed,
            "scan_elapsed_sec": scan_elapsed,
            "post_elapsed_sec": post_elapsed,
            "total_elapsed_sec": (
                prepared_elapsed + scan_elapsed + post_elapsed
            ),
        },
        "different_address_structures_enabled": True,
        "strict_internal_order_preserved": True,
        "formal_state_or_address_read": False,
        "production_pipeline_enabled": False,
    }


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
    """兼容首版同结构人工基准的薄封装。"""

    batch_size = len(initial_masks)
    result = run_variable_workloads_batched_prototype(
        current,
        [donors] * batch_size,
        schema,
        queries,
        target,
        current_counts,
        participates=[participate] * batch_size,
        initial_masks=initial_masks,
        reference_scale=reference_scale,
        seeds=seeds,
        n_sweeps=n_sweeps,
        eta=eta,
        strength=strength,
        floor=floor,
        logit_clip=logit_clip,
    )
    active_switches = result["active_switches_k_by_address"]
    microsteps = result["microsteps_by_address"]
    if len(set(active_switches)) != 1 or len(set(microsteps)) != 1:
        raise RuntimeError("同结构兼容封装得到不同的地址长度")
    result["active_switches_k"] = active_switches[0]
    result["microsteps_per_address"] = microsteps[0]
    result["coordinates"] = np.stack(result["coordinates"], axis=0)
    result["random_rolls"] = np.stack(result["random_rolls"], axis=0)
    result["float_values"] = np.stack(result["float_values"], axis=0)
    result["bool_values"] = np.stack(result["bool_values"], axis=0)
    return result
