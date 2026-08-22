"""Pure query-fit target assessment for one explicitly identified checkpoint.

The current exact-count policy uses an overall query-count RMSE limit of one
record and a per-query absolute-error limit of two records.  A future privacy
layer may derive different public limits from released measurement noise and
pass those limits here.  This module deliberately accepts neither sigma nor a
privacy budget, a reference table, true answers, or an offline L1 metric.

All metrics in one assessment come from the same count-error vector.  Callers
must therefore keep the returned assessment attached to the exact checkpoint
that produced it; combining the loss of one table with the maximum error of a
different table would violate the contract.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

CalibrationSource = Literal[
    "exact_integer_counts",
    "external_noise_calibrated",
]


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite nonnegative number")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return normalized


def _normalize_max_limits(
    value: float | Sequence[float],
) -> float | tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return _finite_nonnegative(value, name="max_abs_count_error_limit")
    if isinstance(value, (str, bytes)):
        raise TypeError(
            "max_abs_count_error_limit must be a number or numeric sequence"
        )
    try:
        items = tuple(value)
    except TypeError as exc:
        raise TypeError(
            "max_abs_count_error_limit must be a number or numeric sequence"
        ) from exc
    if not items:
        raise ValueError("per-query max_abs_count_error_limit cannot be empty")
    return tuple(
        _finite_nonnegative(item, name="per-query max_abs_count_error_limit")
        for item in items
    )


@dataclass(frozen=True)
class QueryFitThresholds:
    """Explicit public limits used to assess one checkpoint."""

    count_rmse_limit: float
    max_abs_count_error_limit: float | tuple[float, ...]
    calibration_source: CalibrationSource

    def __post_init__(self) -> None:
        rmse_limit = _finite_nonnegative(
            self.count_rmse_limit,
            name="count_rmse_limit",
        )
        max_limits = _normalize_max_limits(self.max_abs_count_error_limit)
        if self.calibration_source not in {
            "exact_integer_counts",
            "external_noise_calibrated",
        }:
            raise ValueError(
                "calibration_source must be exact_integer_counts or "
                "external_noise_calibrated"
            )
        object.__setattr__(self, "count_rmse_limit", rmse_limit)
        object.__setattr__(self, "max_abs_count_error_limit", max_limits)

    @classmethod
    def exact_integer_counts(cls) -> QueryFitThresholds:
        """Return the accepted sigma=0 candidate without dataset tuning."""

        return cls(
            count_rmse_limit=1.0,
            max_abs_count_error_limit=2.0,
            calibration_source="exact_integer_counts",
        )

    @classmethod
    def external_noise_calibrated(
        cls,
        *,
        count_rmse_limit: float,
        max_abs_count_error_limit: float | Sequence[float],
    ) -> QueryFitThresholds:
        """Accept externally derived limits without receiving sigma itself."""

        return cls(
            count_rmse_limit=count_rmse_limit,
            max_abs_count_error_limit=max_abs_count_error_limit,
            calibration_source="external_noise_calibrated",
        )

    def per_query_max_limits(self, query_count: int) -> tuple[float, ...]:
        """Expand a uniform limit or validate heterogeneous public limits."""

        if isinstance(query_count, bool) or not isinstance(query_count, int):
            raise TypeError("query_count must be a positive integer")
        if query_count <= 0:
            raise ValueError("query_count must be a positive integer")
        limits = self.max_abs_count_error_limit
        if isinstance(limits, float):
            return (limits,) * query_count
        if len(limits) != query_count:
            raise ValueError(
                "per-query max_abs_count_error_limit length must match "
                "the count-error vector"
            )
        return limits


@dataclass(frozen=True)
class QueryFitAssessment:
    """Immutable fit evidence for exactly one checkpoint."""

    squared_loss: float
    count_rmse: float
    max_abs_count_error: float
    per_query_abs_count_errors: tuple[float, ...]
    rmse_within_limit: bool
    every_query_within_limit: bool
    fit_target_reached: bool
    exact_residual: bool
    calibration_source: CalibrationSource


def _count_error_vector(count_errors: Sequence[float]) -> np.ndarray:
    raw = np.asarray(count_errors)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError("count_errors must be a nonempty one-dimensional vector")
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
        raw.dtype,
        np.number,
    ):
        raise TypeError("count_errors must contain real numeric values")
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("count_errors must contain real numeric values")
    normalized = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(normalized)):
        raise ValueError("count_errors must contain only finite values")
    return normalized


def assess_query_fit(
    count_errors: Sequence[float],
    thresholds: QueryFitThresholds,
) -> QueryFitAssessment:
    """Assess one checkpoint using only its measured-query count errors."""

    if not isinstance(thresholds, QueryFitThresholds):
        raise TypeError("thresholds must be QueryFitThresholds")
    errors = _count_error_vector(count_errors)
    limits = np.asarray(
        thresholds.per_query_max_limits(int(errors.size)),
        dtype=np.float64,
    )
    absolute_errors = np.abs(errors)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_errors = np.square(errors)
    squared_error_sum = math.fsum(float(value) for value in squared_errors)
    squared_loss = 0.5 * squared_error_sum
    count_rmse = math.sqrt(squared_error_sum / int(errors.size))
    if not math.isfinite(squared_loss) or not math.isfinite(count_rmse):
        raise ValueError("count-error metric computation must remain finite")

    max_abs_error = float(np.max(absolute_errors))
    rmse_within = count_rmse <= thresholds.count_rmse_limit
    every_query_within = bool(np.all(absolute_errors <= limits))
    return QueryFitAssessment(
        squared_loss=squared_loss,
        count_rmse=count_rmse,
        max_abs_count_error=max_abs_error,
        per_query_abs_count_errors=tuple(float(value) for value in absolute_errors),
        rmse_within_limit=rmse_within,
        every_query_within_limit=every_query_within,
        fit_target_reached=rmse_within and every_query_within,
        exact_residual=bool(np.all(errors == 0.0)),
        calibration_source=thresholds.calibration_source,
    )
