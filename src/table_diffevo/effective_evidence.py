"""Research-only effective-round evidence for Issue #53 V2.

This module estimates the information in one contiguous scalar outer-round
trajectory.  It deliberately has no dataset identity, quality threshold,
stationarity classification, convergence state, or stopping decision.

The default candidate uses overlapping batch means with
``batch_round_count = floor(sqrt(actual_round_count))``.  A second research
entry point exposes the same formula at an explicit batch length for V2b
multiscale checks.  Neither entry point enforces an ``insufficient_history``
threshold or makes a stationarity, convergence, quality, or stopping decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isqrt, sqrt
from typing import Sequence

import numpy as np


V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION = (
    "issue53-v2-effective-round-evidence-research-v1"
)


@dataclass(frozen=True)
class V2EffectiveRoundEvidence:
    """Numerical ESS/MCSE evidence without a stationarity interpretation."""

    actual_round_count: int
    batch_round_count: int
    overlapping_batch_count: int
    single_round_variance: float | None
    long_run_variance: float | None
    raw_correlation_inflation: float | None
    conservative_correlation_inflation: float | None
    raw_effective_round_count: float | None
    effective_round_count: float | None
    mcse: float | None
    numerically_estimable: bool
    reason: str | None
    stationarity_not_assessed: bool = field(default=True, init=False)
    contract_version: str = field(
        default=V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION,
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


def _coerce_round_indices(
    round_indices: Sequence[int] | np.ndarray,
) -> tuple[int, ...]:
    materialized = _materialize_sequence(
        round_indices,
        name="round_indices",
    )
    if any(isinstance(value, (bool, np.bool_)) for value in materialized):
        raise ValueError("round_indices must not contain boolean values")

    array = np.asarray(materialized)
    if array.ndim != 1:
        raise ValueError("round_indices must be a one-dimensional sequence")
    if array.size < 2:
        raise ValueError("round_indices must contain at least two rounds")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("round_indices must contain only integers")

    normalized = tuple(int(value) for value in array)
    if normalized[0] < 1:
        raise ValueError("round_indices must contain post-round identities")
    if any(
        current != previous + 1
        for previous, current in zip(normalized, normalized[1:])
    ):
        raise ValueError("round_indices must be strictly contiguous")
    return normalized


def _coerce_values(
    values: Sequence[float] | np.ndarray,
    *,
    expected_count: int,
) -> np.ndarray:
    materialized = _materialize_sequence(values, name="values")
    if any(isinstance(value, (bool, np.bool_)) for value in materialized):
        raise ValueError("values must not contain boolean values")

    array = np.asarray(materialized)
    if array.ndim != 1:
        raise ValueError("values must be a one-dimensional sequence")
    if array.size != expected_count:
        raise ValueError("values and round_indices must have the same length")
    if not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError("values must contain only real numeric values")

    normalized = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("values must contain only finite values")
    return normalized.copy()


def _not_estimable(
    *,
    actual_round_count: int,
    batch_round_count: int,
    overlapping_batch_count: int,
    single_round_variance: float | None,
    long_run_variance: float | None,
    reason: str,
) -> V2EffectiveRoundEvidence:
    return V2EffectiveRoundEvidence(
        actual_round_count=actual_round_count,
        batch_round_count=batch_round_count,
        overlapping_batch_count=overlapping_batch_count,
        single_round_variance=single_round_variance,
        long_run_variance=long_run_variance,
        raw_correlation_inflation=None,
        conservative_correlation_inflation=None,
        raw_effective_round_count=None,
        effective_round_count=None,
        mcse=None,
        numerically_estimable=False,
        reason=reason,
    )


def _compute_v2_effective_round_evidence_with_batch_count(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    batch_round_count: int,
) -> V2EffectiveRoundEvidence:
    """Estimate scalar effective rounds with one explicit batch length."""

    normalized_round_indices = _coerce_round_indices(round_indices)
    normalized_values = _coerce_values(
        values,
        expected_count=len(normalized_round_indices),
    )

    actual_round_count = len(normalized_round_indices)
    if isinstance(batch_round_count, (bool, np.bool_)) or not isinstance(
        batch_round_count,
        (int, np.integer),
    ):
        raise ValueError("batch_round_count must be an integer")
    normalized_batch_round_count = int(batch_round_count)
    if not 1 <= normalized_batch_round_count < actual_round_count:
        raise ValueError(
            "batch_round_count must be at least 1 and smaller than the "
            "round count"
        )

    batch_round_count = normalized_batch_round_count
    overlapping_batch_count = actual_round_count - batch_round_count + 1

    with np.errstate(over="ignore", invalid="ignore"):
        single_round_variance = float(
            np.var(normalized_values, ddof=1)
        )
    if not np.isfinite(single_round_variance):
        return _not_estimable(
            actual_round_count=actual_round_count,
            batch_round_count=batch_round_count,
            overlapping_batch_count=overlapping_batch_count,
            single_round_variance=None,
            long_run_variance=None,
            reason="nonfinite_computation",
        )
    if single_round_variance == 0.0:
        return _not_estimable(
            actual_round_count=actual_round_count,
            batch_round_count=batch_round_count,
            overlapping_batch_count=overlapping_batch_count,
            single_round_variance=0.0,
            long_run_variance=None,
            reason="zero_round_variance",
        )

    with np.errstate(over="ignore", invalid="ignore"):
        overall_mean = float(np.mean(normalized_values))
        centered_values = normalized_values - overall_mean
        cumulative_sum = np.concatenate((
            np.asarray([0.0]),
            np.cumsum(centered_values, dtype=np.float64),
        ))
        batch_mean_deviations = (
            cumulative_sum[batch_round_count:]
            - cumulative_sum[:-batch_round_count]
        ) / batch_round_count
        squared_deviation_sum = float(
            np.sum(np.square(batch_mean_deviations), dtype=np.float64)
        )
        long_run_variance = float(
            (
                actual_round_count
                * batch_round_count
                / (
                    (actual_round_count - batch_round_count)
                    * overlapping_batch_count
                )
            )
            * squared_deviation_sum
        )

    if not np.isfinite(overall_mean) or not np.isfinite(long_run_variance):
        return _not_estimable(
            actual_round_count=actual_round_count,
            batch_round_count=batch_round_count,
            overlapping_batch_count=overlapping_batch_count,
            single_round_variance=single_round_variance,
            long_run_variance=None,
            reason="nonfinite_computation",
        )
    if long_run_variance == 0.0:
        return _not_estimable(
            actual_round_count=actual_round_count,
            batch_round_count=batch_round_count,
            overlapping_batch_count=overlapping_batch_count,
            single_round_variance=single_round_variance,
            long_run_variance=0.0,
            reason="degenerate_long_run_variance",
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_correlation_inflation = float(
            long_run_variance / single_round_variance
        )
        conservative_correlation_inflation = max(
            1.0,
            raw_correlation_inflation,
        )
        raw_effective_round_count = float(
            actual_round_count / raw_correlation_inflation
        )
        effective_round_count = float(
            actual_round_count / conservative_correlation_inflation
        )
        mcse = float(sqrt(
            single_round_variance
            * conservative_correlation_inflation
            / actual_round_count
        ))

    derived_values = (
        raw_correlation_inflation,
        conservative_correlation_inflation,
        raw_effective_round_count,
        effective_round_count,
        mcse,
    )
    if (
        raw_correlation_inflation <= 0.0
        or not np.all(np.isfinite(derived_values))
    ):
        return _not_estimable(
            actual_round_count=actual_round_count,
            batch_round_count=batch_round_count,
            overlapping_batch_count=overlapping_batch_count,
            single_round_variance=single_round_variance,
            long_run_variance=long_run_variance,
            reason="nonfinite_computation",
        )

    return V2EffectiveRoundEvidence(
        actual_round_count=actual_round_count,
        batch_round_count=batch_round_count,
        overlapping_batch_count=overlapping_batch_count,
        single_round_variance=single_round_variance,
        long_run_variance=long_run_variance,
        raw_correlation_inflation=raw_correlation_inflation,
        conservative_correlation_inflation=(
            conservative_correlation_inflation
        ),
        raw_effective_round_count=raw_effective_round_count,
        effective_round_count=effective_round_count,
        mcse=mcse,
        numerically_estimable=True,
        reason=None,
    )


def compute_v2_effective_round_evidence_for_batch(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    batch_round_count: int,
) -> V2EffectiveRoundEvidence:
    """Research-only V2 calculation with a caller-specified batch length.

    This entry point exposes the unchanged V2 overlapping-batch formula for
    multiscale research.  It has no checkpoint, stationarity, convergence, or
    stopping semantics.
    """

    return _compute_v2_effective_round_evidence_with_batch_count(
        round_indices,
        values,
        batch_round_count=batch_round_count,
    )


def compute_v2_effective_round_evidence(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> V2EffectiveRoundEvidence:
    """Estimate scalar effective rounds and MCSE using overlapping batches.

    ``round_indices`` must identify consecutive post-round observations.
    Initial states, missing rounds, duplicate rounds, and reordered rounds are
    rejected.  The calculation is:

    * ``b = floor(sqrt(n))``;
    * estimate long-run variance from every length-``b`` overlapping batch;
    * divide long-run variance by ordinary sample variance to estimate serial
      correlation inflation;
    * cap the official effective round count at the actual round count;
    * calculate official MCSE from the same conservative inflation used by
      the official effective round count.

    The result never assesses stationarity or makes a stopping decision.  This
    research core also does not apply the not-yet-selected minimum-history
    policy.  Exact zero variance and exact zero long-run variance fail closed.
    """

    materialized_round_indices = _materialize_sequence(
        round_indices,
        name="round_indices",
    )
    return _compute_v2_effective_round_evidence_with_batch_count(
        materialized_round_indices,
        values,
        batch_round_count=isqrt(len(materialized_round_indices)),
    )
