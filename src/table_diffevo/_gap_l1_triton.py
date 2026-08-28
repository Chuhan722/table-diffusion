"""剩余缺口核单微步的确定性 Triton 点式计算与状态写回。"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl


BLOCK_SIZE = 256


@triton.jit
def _prepare_kernel(
    e0_ptr,
    e1_ptr,
    scale_ptr,
    base_ptr,
    strength_ptr,
    clip_ptr,
    rolls_ptr,
    failures0_ptr,
    failures1_ptr,
    indicators0_ptr,
    indicators1_ptr,
    indicator_row_ptr,
    plan_counts_ptr,
    target_ptr,
    denominators_ptr,
    query_indices_ptr,
    state_mask_row_ptr,
    attribute_index,
    e0_values_ptr,
    e1_values_ptr,
    score_values_ptr,
    normalized_values_ptr,
    raw_values_ptr,
    probability_values_ptr,
    before_values_ptr,
    after_values_ptr,
    clipped_values_ptr,
    candidate_counts_ptr,
    candidate_terms_ptr,
    selected_failures_ptr,
    selected_indicators_ptr,
    step,
    width,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < width
    e0 = tl.load(e0_ptr)
    e1 = tl.load(e1_ptr)
    scale = tl.load(scale_ptr)
    base = tl.load(base_ptr)
    strength = tl.load(strength_ptr)
    clip = tl.load(clip_ptr)
    roll = tl.load(rolls_ptr + step)
    before = tl.load(state_mask_row_ptr + attribute_index).to(tl.int1)
    score = e0 - e1
    normalized = score / scale
    raw = base + (strength * score) / scale
    effective = tl.minimum(tl.maximum(raw, -clip), clip)
    probability = 1.0 / (1.0 + tl.exp(-effective))
    clipped = raw != effective
    selected = roll < probability
    first = offsets == 0
    tl.store(e0_values_ptr + step + offsets, e0, mask=first)
    tl.store(e1_values_ptr + step + offsets, e1, mask=first)
    tl.store(score_values_ptr + step + offsets, score, mask=first)
    tl.store(normalized_values_ptr + step + offsets, normalized, mask=first)
    tl.store(raw_values_ptr + step + offsets, raw, mask=first)
    tl.store(probability_values_ptr + step + offsets, probability, mask=first)
    tl.store(before_values_ptr + step + offsets, before, mask=first)
    tl.store(after_values_ptr + step + offsets, selected, mask=first)
    tl.store(clipped_values_ptr + step + offsets, clipped, mask=first)

    query_index = tl.load(query_indices_ptr + offsets, mask=valid, other=0)
    failures0 = tl.load(failures0_ptr + offsets, mask=valid, other=0)
    failures1 = tl.load(failures1_ptr + offsets, mask=valid, other=0)
    indicators0 = tl.load(indicators0_ptr + offsets, mask=valid, other=0).to(tl.int1)
    indicators1 = tl.load(indicators1_ptr + offsets, mask=valid, other=0).to(tl.int1)
    selected_failures = tl.where(selected, failures1, failures0)
    selected_indicators = tl.where(selected, indicators1, indicators0)
    old_indicators = tl.load(indicator_row_ptr + query_index, mask=valid, other=0).to(
        tl.int1
    )
    old_counts = tl.load(plan_counts_ptr + query_index, mask=valid, other=0)
    candidate_counts = (
        old_counts + selected_indicators.to(tl.int64) - old_indicators.to(tl.int64)
    )
    target = tl.load(target_ptr + query_index, mask=valid, other=0.0)
    denominator = tl.load(denominators_ptr + query_index, mask=valid, other=1.0)
    candidate_terms = tl.abs(target - candidate_counts.to(tl.float64)) / denominator
    tl.store(candidate_counts_ptr + offsets, candidate_counts, mask=valid)
    tl.store(candidate_terms_ptr + offsets, candidate_terms, mask=valid)
    tl.store(selected_failures_ptr + offsets, selected_failures, mask=valid)
    tl.store(selected_indicators_ptr + offsets, selected_indicators, mask=valid)


@triton.jit
def _commit_kernel(
    candidate_counts_ptr,
    candidate_terms_ptr,
    selected_failures_ptr,
    selected_indicators_ptr,
    dense_positions_ptr,
    dense_membership_ptr,
    plan_counts_ptr,
    failure_row_ptr,
    indicator_row_ptr,
    error_terms_ptr,
    candidate_error_sum_ptr,
    error_sum_ptr,
    state_mask_row_ptr,
    attribute_index,
    before_values_ptr,
    after_values_ptr,
    step,
    n_queries,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_queries
    before = tl.load(before_values_ptr + step).to(tl.int1)
    selected = tl.load(after_values_ptr + step).to(tl.int1)
    changed = selected != before
    member = tl.load(dense_membership_ptr + offsets, mask=valid, other=0).to(tl.int1)
    update = valid & member & changed
    position = tl.load(dense_positions_ptr + offsets, mask=valid, other=0)
    counts = tl.load(candidate_counts_ptr + position, mask=update, other=0)
    terms = tl.load(candidate_terms_ptr + position, mask=update, other=0.0)
    failures = tl.load(selected_failures_ptr + position, mask=update, other=0)
    indicators = tl.load(selected_indicators_ptr + position, mask=update, other=0).to(
        tl.int1
    )
    tl.store(plan_counts_ptr + offsets, counts, mask=update)
    tl.store(error_terms_ptr + offsets, terms, mask=update)
    tl.store(failure_row_ptr + offsets, failures, mask=update)
    tl.store(indicator_row_ptr + offsets, indicators, mask=update)
    first = offsets == 0
    candidate_error_sum = tl.load(candidate_error_sum_ptr)
    tl.store(
        error_sum_ptr + offsets,
        candidate_error_sum,
        mask=first & changed,
    )
    tl.store(
        state_mask_row_ptr + attribute_index + offsets,
        selected,
        mask=first,
    )


def launch_prepare(*arguments: Any, width: int) -> None:
    """提交一个候选计算 kernel；所有局部输出由调用者预分配。"""

    _prepare_kernel[(triton.cdiv(width, BLOCK_SIZE),)](
        *arguments,
        width,
        BLOCK=BLOCK_SIZE,
    )


def launch_commit(*arguments: Any, n_queries: int) -> None:
    """提交一个唯一位置状态写回 kernel。"""

    _commit_kernel[(triton.cdiv(n_queries, BLOCK_SIZE),)](
        *arguments,
        n_queries,
        BLOCK=BLOCK_SIZE,
    )
