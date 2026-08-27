"""Research-only adaptive effective-round evidence for Issue #53 V2b.

This module compares the existing V2 overlapping-batch estimator at two
pre-registered batch lengths.  It can classify numerical scale agreement at
fixed checkpoints and summarize the first agreeing checkpoint.  It does not
assess stationarity, convergence, quality, or a generator stopping decision.
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


V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION = (
    "issue53-v2b-adaptive-effective-round-evidence-research-v1"
)
V2B_ADAPTIVE_CHECKPOINTS: tuple[int, ...] = tuple(
    range(256, 2048 + 1, 128)
)
V2B_SCALE_RATIO_LIMIT = 1.25
V2B_RESOURCE_ROUND_CAP = 2048


@dataclass(frozen=True)
class V2BAdaptiveCheckpointEvidence:
    """Two-scale numerical evidence at one pre-registered checkpoint."""

    actual_round_count: int
    short_batch_round_count: int
    long_batch_round_count: int
    single_round_variance: float | None
    short_long_run_variance: float | None
    long_long_run_variance: float | None
    short_raw_correlation_inflation: float | None
    long_raw_correlation_inflation: float | None
    short_conservative_correlation_inflation: float | None
    long_conservative_correlation_inflation: float | None
    scale_ratio: float | None
    official_correlation_inflation: float | None
    official_long_run_variance: float | None
    effective_round_count: float | None
    mcse: float | None
    short_numerically_estimable: bool
    long_numerically_estimable: bool
    short_reason: str | None
    long_reason: str | None
    adaptive_numerically_estimable: bool
    reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=(
            V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        init=False,
    )


@dataclass(frozen=True)
class V2BAdaptiveTrajectorySummary:
    """First numerical-agreement checkpoint on the fixed V2b schedule."""

    first_adaptive_numerically_estimable_round: int | None
    resource_round_count: int
    adaptive_numerically_estimable: bool
    reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=(
            V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        init=False,
    )


@dataclass(frozen=True)
class V2BAdaptiveTrajectoryEvidence:
    """All fixed V2b checkpoint diagnostics for one 2048-round trajectory."""

    checkpoint_evidence: tuple[V2BAdaptiveCheckpointEvidence, ...]
    first_adaptive_numerically_estimable_round: int | None
    resource_round_count: int
    adaptive_numerically_estimable: bool
    reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=(
            V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
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


def compute_v2b_scale_ratio(
    short_conservative_correlation_inflation: float,
    long_conservative_correlation_inflation: float,
) -> float:
    """Return the symmetric multiplicative difference between two scales."""

    normalized: list[float] = []
    for name, value in (
        (
            "short_conservative_correlation_inflation",
            short_conservative_correlation_inflation,
        ),
        (
            "long_conservative_correlation_inflation",
            long_conservative_correlation_inflation,
        ),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise ValueError(f"{name} must be a finite real number at least 1")
        try:
            normalized_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a finite real number at least 1"
            ) from exc
        if not np.isfinite(normalized_value) or normalized_value < 1.0:
            raise ValueError(f"{name} must be a finite real number at least 1")
        normalized.append(normalized_value)

    return max(normalized) / min(normalized)


def v2b_scale_ratio_is_acceptable(
    short_conservative_correlation_inflation: float,
    long_conservative_correlation_inflation: float,
) -> bool:
    """Apply the single pre-registered ``scale_ratio <= 1.25`` rule."""

    return compute_v2b_scale_ratio(
        short_conservative_correlation_inflation,
        long_conservative_correlation_inflation,
    ) <= V2B_SCALE_RATIO_LIMIT


def _flatten_not_estimable_scales(
    *,
    short: V2EffectiveRoundEvidence,
    long: V2EffectiveRoundEvidence,
) -> V2BAdaptiveCheckpointEvidence:
    single_round_variance = short.single_round_variance
    if single_round_variance is None:
        single_round_variance = long.single_round_variance

    return V2BAdaptiveCheckpointEvidence(
        actual_round_count=short.actual_round_count,
        short_batch_round_count=short.batch_round_count,
        long_batch_round_count=long.batch_round_count,
        single_round_variance=single_round_variance,
        short_long_run_variance=short.long_run_variance,
        long_long_run_variance=long.long_run_variance,
        short_raw_correlation_inflation=(
            short.raw_correlation_inflation
        ),
        long_raw_correlation_inflation=long.raw_correlation_inflation,
        short_conservative_correlation_inflation=(
            short.conservative_correlation_inflation
        ),
        long_conservative_correlation_inflation=(
            long.conservative_correlation_inflation
        ),
        scale_ratio=None,
        official_correlation_inflation=None,
        official_long_run_variance=None,
        effective_round_count=None,
        mcse=None,
        short_numerically_estimable=short.numerically_estimable,
        long_numerically_estimable=long.numerically_estimable,
        short_reason=short.reason,
        long_reason=long.reason,
        adaptive_numerically_estimable=False,
        reason="core_not_estimable",
    )


def _combine_v2b_scales(
    *,
    short: V2EffectiveRoundEvidence,
    long: V2EffectiveRoundEvidence,
) -> V2BAdaptiveCheckpointEvidence:
    if short.actual_round_count != long.actual_round_count:
        raise ValueError("short and long scales must use the same round count")
    if not short.numerically_estimable or not long.numerically_estimable:
        return _flatten_not_estimable_scales(short=short, long=long)

    required_values = (
        short.single_round_variance,
        long.single_round_variance,
        short.long_run_variance,
        long.long_run_variance,
        short.raw_correlation_inflation,
        long.raw_correlation_inflation,
        short.conservative_correlation_inflation,
        long.conservative_correlation_inflation,
    )
    if any(value is None for value in required_values):
        raise RuntimeError("estimable V2 scale is missing a numerical field")
    if short.single_round_variance != long.single_round_variance:
        raise RuntimeError("short and long scales disagree on round variance")

    single_round_variance = float(short.single_round_variance)
    short_inflation = float(short.conservative_correlation_inflation)
    long_inflation = float(long.conservative_correlation_inflation)
    scale_ratio = compute_v2b_scale_ratio(
        short_inflation,
        long_inflation,
    )
    official_inflation = max(short_inflation, long_inflation)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        official_long_run_variance = float(
            single_round_variance * official_inflation
        )
        effective_round_count = float(
            short.actual_round_count / official_inflation
        )
        mcse = float(sqrt(
            official_long_run_variance / short.actual_round_count
        ))

    official_values = (
        scale_ratio,
        official_inflation,
        official_long_run_variance,
        effective_round_count,
        mcse,
    )
    if not np.all(np.isfinite(official_values)):
        return V2BAdaptiveCheckpointEvidence(
            actual_round_count=short.actual_round_count,
            short_batch_round_count=short.batch_round_count,
            long_batch_round_count=long.batch_round_count,
            single_round_variance=single_round_variance,
            short_long_run_variance=short.long_run_variance,
            long_long_run_variance=long.long_run_variance,
            short_raw_correlation_inflation=(
                short.raw_correlation_inflation
            ),
            long_raw_correlation_inflation=(
                long.raw_correlation_inflation
            ),
            short_conservative_correlation_inflation=short_inflation,
            long_conservative_correlation_inflation=long_inflation,
            scale_ratio=None,
            official_correlation_inflation=None,
            official_long_run_variance=None,
            effective_round_count=None,
            mcse=None,
            short_numerically_estimable=True,
            long_numerically_estimable=True,
            short_reason=None,
            long_reason=None,
            adaptive_numerically_estimable=False,
            reason="nonfinite_computation",
        )

    adaptive_numerically_estimable = (
        scale_ratio <= V2B_SCALE_RATIO_LIMIT
    )
    return V2BAdaptiveCheckpointEvidence(
        actual_round_count=short.actual_round_count,
        short_batch_round_count=short.batch_round_count,
        long_batch_round_count=long.batch_round_count,
        single_round_variance=single_round_variance,
        short_long_run_variance=short.long_run_variance,
        long_long_run_variance=long.long_run_variance,
        short_raw_correlation_inflation=(
            short.raw_correlation_inflation
        ),
        long_raw_correlation_inflation=long.raw_correlation_inflation,
        short_conservative_correlation_inflation=short_inflation,
        long_conservative_correlation_inflation=long_inflation,
        scale_ratio=scale_ratio,
        official_correlation_inflation=official_inflation,
        official_long_run_variance=official_long_run_variance,
        effective_round_count=effective_round_count,
        mcse=mcse,
        short_numerically_estimable=True,
        long_numerically_estimable=True,
        short_reason=None,
        long_reason=None,
        adaptive_numerically_estimable=adaptive_numerically_estimable,
        reason=(
            None
            if adaptive_numerically_estimable
            else "multiscale_disagreement"
        ),
    )


def compute_v2b_adaptive_checkpoint_evidence(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> V2BAdaptiveCheckpointEvidence:
    """Compute two-scale evidence at one fixed V2b checkpoint."""

    materialized_round_indices = _materialize_sequence(
        round_indices,
        name="round_indices",
    )
    materialized_values = _materialize_sequence(values, name="values")
    actual_round_count = len(materialized_round_indices)
    if actual_round_count not in V2B_ADAPTIVE_CHECKPOINTS:
        raise ValueError("round count must be a pre-registered V2b checkpoint")

    short_batch_round_count = isqrt(actual_round_count)
    long_batch_round_count = 2 * short_batch_round_count
    short = compute_v2_effective_round_evidence_for_batch(
        materialized_round_indices,
        materialized_values,
        batch_round_count=short_batch_round_count,
    )
    long = compute_v2_effective_round_evidence_for_batch(
        materialized_round_indices,
        materialized_values,
        batch_round_count=long_batch_round_count,
    )
    return _combine_v2b_scales(short=short, long=long)


def summarize_v2b_adaptive_checkpoint_decisions(
    decisions: Sequence[bool] | np.ndarray,
) -> V2BAdaptiveTrajectorySummary:
    """Summarize exactly 15 fixed checkpoint decisions without a stop claim."""

    materialized_decisions = _materialize_sequence(
        decisions,
        name="decisions",
    )
    if len(materialized_decisions) != len(V2B_ADAPTIVE_CHECKPOINTS):
        raise ValueError("decisions must cover all fixed V2b checkpoints")
    if any(
        not isinstance(decision, (bool, np.bool_))
        for decision in materialized_decisions
    ):
        raise ValueError("decisions must contain only boolean values")

    first_round = next(
        (
            checkpoint
            for checkpoint, decision in zip(
                V2B_ADAPTIVE_CHECKPOINTS,
                materialized_decisions,
            )
            if bool(decision)
        ),
        None,
    )
    return V2BAdaptiveTrajectorySummary(
        first_adaptive_numerically_estimable_round=first_round,
        resource_round_count=(
            first_round
            if first_round is not None
            else V2B_RESOURCE_ROUND_CAP
        ),
        adaptive_numerically_estimable=first_round is not None,
        reason=(
            None
            if first_round is not None
            else "resource_cap_without_multiscale_evidence"
        ),
    )


def compute_v2b_adaptive_trajectory_evidence(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> V2BAdaptiveTrajectoryEvidence:
    """Evaluate all V2b checkpoints on one complete 2048-round trajectory."""

    materialized_round_indices = _materialize_sequence(
        round_indices,
        name="round_indices",
    )
    materialized_values = _materialize_sequence(values, name="values")
    if len(materialized_round_indices) != V2B_RESOURCE_ROUND_CAP:
        raise ValueError("trajectory must contain exactly 2048 round identities")
    if len(materialized_values) != V2B_RESOURCE_ROUND_CAP:
        raise ValueError("trajectory must contain exactly 2048 values")

    checkpoint_evidence = tuple(
        compute_v2b_adaptive_checkpoint_evidence(
            materialized_round_indices[:checkpoint],
            materialized_values[:checkpoint],
        )
        for checkpoint in V2B_ADAPTIVE_CHECKPOINTS
    )
    summary = summarize_v2b_adaptive_checkpoint_decisions(
        [
            evidence.adaptive_numerically_estimable
            for evidence in checkpoint_evidence
        ]
    )
    return V2BAdaptiveTrajectoryEvidence(
        checkpoint_evidence=checkpoint_evidence,
        first_adaptive_numerically_estimable_round=(
            summary.first_adaptive_numerically_estimable_round
        ),
        resource_round_count=summary.resource_round_count,
        adaptive_numerically_estimable=(
            summary.adaptive_numerically_estimable
        ),
        reason=summary.reason,
    )
