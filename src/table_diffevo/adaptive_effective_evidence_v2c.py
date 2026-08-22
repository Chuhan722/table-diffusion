"""Research-only three-scale effective-round evidence for Issue #53 V2c.

This module applies the unchanged V2 overlapping-batch calculation at the
pre-registered ``b``, ``2b``, and ``4b`` batch lengths.  A checkpoint has
adaptive numerical evidence only when its three scales are compatible and
the immediately preceding checkpoint was also compatible.

The state is deliberately revocable.  This module does not assess
stationarity, convergence, quality, or a generator stopping decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt, sqrt
from typing import Sequence

import numpy as np

from table_diffevo.effective_evidence import (
    V2EffectiveRoundEvidence,
    compute_v2_effective_round_evidence_for_batch,
)


V2C_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION = (
    "issue53-v2c-three-scale-adaptive-effective-evidence-research-v1"
)
V2C_ADAPTIVE_CHECKPOINTS: tuple[int, ...] = tuple(
    range(256, 2048 + 1, 128)
)
V2C_SCALE_RATIO_LIMIT = 1.25
V2C_RESOURCE_ROUND_CAP = 2048


@dataclass(frozen=True)
class V2CAdaptiveCheckpointEvidence:
    """Three-scale evidence and current revocable state at one checkpoint."""

    actual_round_count: int
    b1_batch_round_count: int
    b2_batch_round_count: int
    b3_batch_round_count: int
    single_round_variance: float | None
    b1_long_run_variance: float | None
    b2_long_run_variance: float | None
    b3_long_run_variance: float | None
    b1_raw_correlation_inflation: float | None
    b2_raw_correlation_inflation: float | None
    b3_raw_correlation_inflation: float | None
    b1_conservative_correlation_inflation: float | None
    b2_conservative_correlation_inflation: float | None
    b3_conservative_correlation_inflation: float | None
    scale_ratio: float | None
    official_correlation_inflation: float | None
    official_long_run_variance: float | None
    effective_round_count: float | None
    mcse: float | None
    b1_numerically_estimable: bool
    b2_numerically_estimable: bool
    b3_numerically_estimable: bool
    b1_reason: str | None
    b2_reason: str | None
    b3_reason: str | None
    previous_three_scale_compatible: bool
    three_scale_compatible: bool
    adaptive_numerically_estimable: bool
    reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=(
            V2C_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        init=False,
    )


@dataclass(frozen=True)
class V2CAdaptiveTrajectorySummary:
    """Pure sequence summary for the 15 frozen compatibility decisions."""

    checkpoint_adaptive_numerically_estimable: tuple[bool, ...]
    first_adaptive_numerically_estimable_round: int | None
    resource_round_count: int
    post_first_three_scale_incompatible_checkpoint_count: int
    post_first_has_three_scale_incompatibility: bool
    current_three_scale_compatible: bool
    adaptive_numerically_estimable: bool
    no_ready_reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=(
            V2C_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        init=False,
    )


@dataclass(frozen=True)
class V2CAdaptiveTrajectoryEvidence:
    """All checkpoint diagnostics for one complete 2048-round trajectory."""

    checkpoint_evidence: tuple[V2CAdaptiveCheckpointEvidence, ...]
    first_adaptive_numerically_estimable_round: int | None
    resource_round_count: int
    post_first_three_scale_incompatible_checkpoint_count: int
    post_first_has_three_scale_incompatibility: bool
    current_three_scale_compatible: bool
    adaptive_numerically_estimable: bool
    reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=(
            V2C_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        init=False,
    )


def _materialize_sequence(
    sequence: Sequence[object] | np.ndarray,
    *,
    name: str,
) -> tuple[object, ...]:
    try:
        return tuple(sequence)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc


def _normalize_conservative_inflation(
    value: float,
    *,
    name: str,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a finite real number at least 1")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite real number at least 1"
        ) from exc
    if not np.isfinite(normalized) or normalized < 1.0:
        raise ValueError(f"{name} must be a finite real number at least 1")
    return normalized


def compute_v2c_scale_ratio(
    b1_conservative_correlation_inflation: float,
    b2_conservative_correlation_inflation: float,
    b3_conservative_correlation_inflation: float,
) -> float:
    """Return max/min across the three conservative scale inflations."""

    normalized = (
        _normalize_conservative_inflation(
            b1_conservative_correlation_inflation,
            name="b1_conservative_correlation_inflation",
        ),
        _normalize_conservative_inflation(
            b2_conservative_correlation_inflation,
            name="b2_conservative_correlation_inflation",
        ),
        _normalize_conservative_inflation(
            b3_conservative_correlation_inflation,
            name="b3_conservative_correlation_inflation",
        ),
    )
    return max(normalized) / min(normalized)


def v2c_scale_ratio_is_acceptable(
    b1_conservative_correlation_inflation: float,
    b2_conservative_correlation_inflation: float,
    b3_conservative_correlation_inflation: float,
) -> bool:
    """Apply the pre-registered inclusive three-scale ratio boundary."""

    return compute_v2c_scale_ratio(
        b1_conservative_correlation_inflation,
        b2_conservative_correlation_inflation,
        b3_conservative_correlation_inflation,
    ) <= V2C_SCALE_RATIO_LIMIT


def _checkpoint_evidence_from_scales(
    *,
    b1: V2EffectiveRoundEvidence,
    b2: V2EffectiveRoundEvidence,
    b3: V2EffectiveRoundEvidence,
    previous_three_scale_compatible: bool,
    scale_ratio: float | None,
    official_correlation_inflation: float | None,
    official_long_run_variance: float | None,
    effective_round_count: float | None,
    mcse: float | None,
    three_scale_compatible: bool,
    adaptive_numerically_estimable: bool,
    reason: str | None,
) -> V2CAdaptiveCheckpointEvidence:
    single_round_variance = next(
        (
            evidence.single_round_variance
            for evidence in (b1, b2, b3)
            if evidence.single_round_variance is not None
        ),
        None,
    )
    return V2CAdaptiveCheckpointEvidence(
        actual_round_count=b1.actual_round_count,
        b1_batch_round_count=b1.batch_round_count,
        b2_batch_round_count=b2.batch_round_count,
        b3_batch_round_count=b3.batch_round_count,
        single_round_variance=single_round_variance,
        b1_long_run_variance=b1.long_run_variance,
        b2_long_run_variance=b2.long_run_variance,
        b3_long_run_variance=b3.long_run_variance,
        b1_raw_correlation_inflation=b1.raw_correlation_inflation,
        b2_raw_correlation_inflation=b2.raw_correlation_inflation,
        b3_raw_correlation_inflation=b3.raw_correlation_inflation,
        b1_conservative_correlation_inflation=(
            b1.conservative_correlation_inflation
        ),
        b2_conservative_correlation_inflation=(
            b2.conservative_correlation_inflation
        ),
        b3_conservative_correlation_inflation=(
            b3.conservative_correlation_inflation
        ),
        scale_ratio=scale_ratio,
        official_correlation_inflation=official_correlation_inflation,
        official_long_run_variance=official_long_run_variance,
        effective_round_count=effective_round_count,
        mcse=mcse,
        b1_numerically_estimable=b1.numerically_estimable,
        b2_numerically_estimable=b2.numerically_estimable,
        b3_numerically_estimable=b3.numerically_estimable,
        b1_reason=b1.reason,
        b2_reason=b2.reason,
        b3_reason=b3.reason,
        previous_three_scale_compatible=(
            previous_three_scale_compatible
        ),
        three_scale_compatible=three_scale_compatible,
        adaptive_numerically_estimable=adaptive_numerically_estimable,
        reason=reason,
    )


def _combine_v2c_scales(
    *,
    b1: V2EffectiveRoundEvidence,
    b2: V2EffectiveRoundEvidence,
    b3: V2EffectiveRoundEvidence,
    previous_three_scale_compatible: bool,
) -> V2CAdaptiveCheckpointEvidence:
    """Combine three already-computed V2 scales under the frozen rule."""

    if not isinstance(previous_three_scale_compatible, (bool, np.bool_)):
        raise ValueError("previous compatibility must be boolean")
    previous_compatible = bool(previous_three_scale_compatible)

    actual_round_counts = {
        b1.actual_round_count,
        b2.actual_round_count,
        b3.actual_round_count,
    }
    if len(actual_round_counts) != 1:
        raise ValueError("all three scales must use the same round count")
    actual_round_count = b1.actual_round_count
    base_batch_round_count = isqrt(actual_round_count)
    expected_batch_round_counts = (
        base_batch_round_count,
        2 * base_batch_round_count,
        4 * base_batch_round_count,
    )
    if (
        b1.batch_round_count,
        b2.batch_round_count,
        b3.batch_round_count,
    ) != expected_batch_round_counts:
        raise ValueError("scales must use the frozen b, 2b, and 4b lengths")

    if not all(
        evidence.numerically_estimable for evidence in (b1, b2, b3)
    ):
        return _checkpoint_evidence_from_scales(
            b1=b1,
            b2=b2,
            b3=b3,
            previous_three_scale_compatible=previous_compatible,
            scale_ratio=None,
            official_correlation_inflation=None,
            official_long_run_variance=None,
            effective_round_count=None,
            mcse=None,
            three_scale_compatible=False,
            adaptive_numerically_estimable=False,
            reason="core_not_estimable",
        )

    required_values = tuple(
        value
        for evidence in (b1, b2, b3)
        for value in (
            evidence.single_round_variance,
            evidence.long_run_variance,
            evidence.raw_correlation_inflation,
            evidence.conservative_correlation_inflation,
        )
    )
    if any(value is None for value in required_values):
        raise RuntimeError("estimable V2 scale is missing a numerical field")
    if not (
        b1.single_round_variance
        == b2.single_round_variance
        == b3.single_round_variance
    ):
        raise RuntimeError("the three scales disagree on round variance")

    single_round_variance = float(b1.single_round_variance)
    inflations = (
        float(b1.conservative_correlation_inflation),
        float(b2.conservative_correlation_inflation),
        float(b3.conservative_correlation_inflation),
    )
    scale_ratio = compute_v2c_scale_ratio(*inflations)
    official_inflation = max(inflations)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        official_long_run_variance = float(
            single_round_variance * official_inflation
        )
        effective_round_count = float(
            actual_round_count / official_inflation
        )
        mcse = float(sqrt(
            official_long_run_variance / actual_round_count
        ))

    combined_values = (
        scale_ratio,
        official_inflation,
        official_long_run_variance,
        effective_round_count,
        mcse,
    )
    if not np.all(np.isfinite(combined_values)):
        return _checkpoint_evidence_from_scales(
            b1=b1,
            b2=b2,
            b3=b3,
            previous_three_scale_compatible=previous_compatible,
            scale_ratio=None,
            official_correlation_inflation=None,
            official_long_run_variance=None,
            effective_round_count=None,
            mcse=None,
            three_scale_compatible=False,
            adaptive_numerically_estimable=False,
            reason="nonfinite_computation",
        )

    three_scale_compatible = scale_ratio <= V2C_SCALE_RATIO_LIMIT
    adaptive_numerically_estimable = (
        previous_compatible and three_scale_compatible
    )
    if adaptive_numerically_estimable:
        reason = None
    elif three_scale_compatible:
        reason = "awaiting_consecutive_multiscale_evidence"
    else:
        reason = "multiscale_disagreement"

    return _checkpoint_evidence_from_scales(
        b1=b1,
        b2=b2,
        b3=b3,
        previous_three_scale_compatible=previous_compatible,
        scale_ratio=scale_ratio,
        official_correlation_inflation=official_inflation,
        official_long_run_variance=official_long_run_variance,
        effective_round_count=effective_round_count,
        mcse=mcse,
        three_scale_compatible=three_scale_compatible,
        adaptive_numerically_estimable=adaptive_numerically_estimable,
        reason=reason,
    )


def _normalize_previous_compatibility(
    *,
    actual_round_count: int,
    previous_three_scale_compatible: bool | None,
) -> bool:
    if actual_round_count == V2C_ADAPTIVE_CHECKPOINTS[0]:
        if previous_three_scale_compatible is not None:
            raise ValueError(
                "the first V2c checkpoint has no previous checkpoint"
            )
        return False
    if not isinstance(previous_three_scale_compatible, (bool, np.bool_)):
        raise ValueError(
            "later V2c checkpoints require previous compatibility"
        )
    return bool(previous_three_scale_compatible)


def compute_v2c_adaptive_checkpoint_evidence(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    previous_three_scale_compatible: bool | None = None,
) -> V2CAdaptiveCheckpointEvidence:
    """Compute the three scales and current state at one frozen checkpoint.

    The 256-round checkpoint requires ``previous_three_scale_compatible=None``
    and therefore cannot acquire adaptive numerical evidence.  Every later
    checkpoint requires the immediately preceding checkpoint's compatibility
    value.
    """

    materialized_round_indices = _materialize_sequence(
        round_indices,
        name="round_indices",
    )
    materialized_values = _materialize_sequence(values, name="values")
    actual_round_count = len(materialized_round_indices)
    if actual_round_count not in V2C_ADAPTIVE_CHECKPOINTS:
        raise ValueError("round count must be a pre-registered V2c checkpoint")
    previous_compatible = _normalize_previous_compatibility(
        actual_round_count=actual_round_count,
        previous_three_scale_compatible=previous_three_scale_compatible,
    )

    b1_batch_round_count = isqrt(actual_round_count)
    batch_round_counts = (
        b1_batch_round_count,
        2 * b1_batch_round_count,
        4 * b1_batch_round_count,
    )
    scales = tuple(
        compute_v2_effective_round_evidence_for_batch(
            materialized_round_indices,
            materialized_values,
            batch_round_count=batch_round_count,
        )
        for batch_round_count in batch_round_counts
    )
    return _combine_v2c_scales(
        b1=scales[0],
        b2=scales[1],
        b3=scales[2],
        previous_three_scale_compatible=previous_compatible,
    )


def summarize_v2c_adaptive_checkpoint_compatibilities(
    compatibilities: Sequence[bool] | np.ndarray,
) -> V2CAdaptiveTrajectorySummary:
    """Summarize exactly 15 compatibility states without a stop claim."""

    materialized = _materialize_sequence(
        compatibilities,
        name="compatibilities",
    )
    if len(materialized) != len(V2C_ADAPTIVE_CHECKPOINTS):
        raise ValueError(
            "compatibilities must cover all fixed V2c checkpoints"
        )
    if any(
        not isinstance(compatibility, (bool, np.bool_))
        for compatibility in materialized
    ):
        raise ValueError("compatibilities must contain only boolean values")
    normalized = tuple(bool(value) for value in materialized)
    decisions = tuple(
        False if index == 0 else normalized[index - 1] and normalized[index]
        for index in range(len(normalized))
    )
    first_index = next(
        (index for index, decision in enumerate(decisions) if decision),
        None,
    )
    first_round = (
        None
        if first_index is None
        else V2C_ADAPTIVE_CHECKPOINTS[first_index]
    )
    post_first_incompatible_count = (
        0
        if first_index is None
        else sum(
            not compatibility
            for compatibility in normalized[first_index + 1:]
        )
    )
    no_ready_reason = (
        "resource_cap_without_consecutive_multiscale_evidence"
        if first_round is None
        else None
    )
    return V2CAdaptiveTrajectorySummary(
        checkpoint_adaptive_numerically_estimable=decisions,
        first_adaptive_numerically_estimable_round=first_round,
        resource_round_count=(
            first_round
            if first_round is not None
            else V2C_RESOURCE_ROUND_CAP
        ),
        post_first_three_scale_incompatible_checkpoint_count=(
            post_first_incompatible_count
        ),
        post_first_has_three_scale_incompatibility=(
            post_first_incompatible_count > 0
        ),
        current_three_scale_compatible=normalized[-1],
        adaptive_numerically_estimable=decisions[-1],
        no_ready_reason=no_ready_reason,
    )


def compute_v2c_adaptive_trajectory_evidence(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> V2CAdaptiveTrajectoryEvidence:
    """Evaluate every V2c checkpoint on one complete 2048-round trajectory."""

    materialized_round_indices = _materialize_sequence(
        round_indices,
        name="round_indices",
    )
    materialized_values = _materialize_sequence(values, name="values")
    if len(materialized_round_indices) != V2C_RESOURCE_ROUND_CAP:
        raise ValueError("trajectory must contain exactly 2048 round identities")
    if len(materialized_values) != V2C_RESOURCE_ROUND_CAP:
        raise ValueError("trajectory must contain exactly 2048 values")

    checkpoint_evidence: list[V2CAdaptiveCheckpointEvidence] = []
    previous_compatible: bool | None = None
    for checkpoint in V2C_ADAPTIVE_CHECKPOINTS:
        evidence = compute_v2c_adaptive_checkpoint_evidence(
            materialized_round_indices[:checkpoint],
            materialized_values[:checkpoint],
            previous_three_scale_compatible=previous_compatible,
        )
        checkpoint_evidence.append(evidence)
        previous_compatible = evidence.three_scale_compatible

    frozen_checkpoint_evidence = tuple(checkpoint_evidence)
    summary = summarize_v2c_adaptive_checkpoint_compatibilities(
        [
            evidence.three_scale_compatible
            for evidence in frozen_checkpoint_evidence
        ]
    )
    decisions = tuple(
        evidence.adaptive_numerically_estimable
        for evidence in frozen_checkpoint_evidence
    )
    if decisions != summary.checkpoint_adaptive_numerically_estimable:
        raise RuntimeError("checkpoint and trajectory decisions disagree")

    current_reason = (
        summary.no_ready_reason
        if summary.no_ready_reason is not None
        else frozen_checkpoint_evidence[-1].reason
    )
    return V2CAdaptiveTrajectoryEvidence(
        checkpoint_evidence=frozen_checkpoint_evidence,
        first_adaptive_numerically_estimable_round=(
            summary.first_adaptive_numerically_estimable_round
        ),
        resource_round_count=summary.resource_round_count,
        post_first_three_scale_incompatible_checkpoint_count=(
            summary.post_first_three_scale_incompatible_checkpoint_count
        ),
        post_first_has_three_scale_incompatibility=(
            summary.post_first_has_three_scale_incompatibility
        ),
        current_three_scale_compatible=(
            summary.current_three_scale_compatible
        ),
        adaptive_numerically_estimable=(
            summary.adaptive_numerically_estimable
        ),
        reason=current_reason,
    )
