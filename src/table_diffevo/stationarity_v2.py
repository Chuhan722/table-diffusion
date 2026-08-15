"""Threshold-free evidence primitives for the Issue #53 V2 detector.

This module implements only trajectory summaries and mathematical evidence.
It deliberately does not contain thresholds, convergence states, or stopping
decisions.  Those policy choices belong to a later layer and must be calibrated
separately from these evidence contracts.

The current research candidate uses 100-round subblocks, but the low-level
collector requires the duration explicitly and records it in its output.  A
future production detector must freeze one data-independent rule; it must not
silently select a duration from a dataset name.  The scalar primitive remains
duration-agnostic: it accepts exactly 12 ordered summaries, corresponding to
three blocks of four subblocks each.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Sequence

import numpy as np

from table_diffevo.stationarity import StationarityTrace


STATIONARITY_V2_SCALAR_EVIDENCE_CONTRACT_VERSION = (
    "issue53-stage2-v2-scalar-evidence-v1"
)
STATIONARITY_V2_SUBBLOCK_SUMMARY_CONTRACT_VERSION = (
    "issue53-stage2-v2-subblock-summary-v1"
)
STATIONARITY_V2_SUBBLOCK_COLLECTION_CONTRACT_VERSION = (
    "issue53-stage2-v2-subblock-collection-v1"
)
V2_CURRENT_SUBBLOCK_ROUND_CANDIDATE = 100
V2_CANDIDATE_BLOCK_COUNT = 3
V2_SUBBLOCKS_PER_BLOCK = 4
V2_SCALAR_EVIDENCE_POINT_COUNT = (
    V2_CANDIDATE_BLOCK_COUNT * V2_SUBBLOCKS_PER_BLOCK
)
V2_PAIRWISE_SLOPE_COUNT = (
    V2_SCALAR_EVIDENCE_POINT_COUNT
    * (V2_SCALAR_EVIDENCE_POINT_COUNT - 1)
    // 2
)
MAD_NORMAL_CONSISTENCY_FACTOR = 1.4826


@dataclass(frozen=True)
class V2SubblockSummary:
    """Source-free summaries of one complete, contiguous post-round block."""

    subblock_number: int
    start_round_index: int
    end_round_index: int
    round_count: int
    normalized_query_mean: tuple[float, ...]
    l1_mean: float
    l1_p90_minus_p10: float
    unique_row_rate_mean: float
    normalized_row_entropy_mean: float
    active_round_rate: float
    mean_changed_row_fraction: float
    mean_changed_query_fraction: float
    mean_normalized_query_l1_movement: float
    contract_version: str = field(
        default=STATIONARITY_V2_SUBBLOCK_SUMMARY_CONTRACT_VERSION,
        init=False,
    )


@dataclass(frozen=True)
class V2SubblockCollection:
    """Complete subblocks plus an explicit, never-summarized trailing tail."""

    trace_contract_version: str
    query_identity_sha256: str
    target_identity_sha256: str
    n_records: int
    query_count: int
    post_round_count: int
    subblock_round_count: int
    trailing_post_round_count: int
    subblocks: tuple[V2SubblockSummary, ...]
    contract_version: str = field(
        default=STATIONARITY_V2_SUBBLOCK_COLLECTION_CONTRACT_VERSION,
        init=False,
    )

    @property
    def complete_subblock_count(self) -> int:
        """Number of complete, non-overlapping subblocks（完整小块数）."""

        return len(self.subblocks)


def _validate_subblock_round_count(subblock_round_count: int) -> int:
    if isinstance(subblock_round_count, bool) or not isinstance(
        subblock_round_count, (int, np.integer)
    ):
        raise ValueError("subblock_round_count must be an integer")
    normalized = int(subblock_round_count)
    if normalized < 2:
        raise ValueError("subblock_round_count must be at least 2")
    return normalized


def _summarize_v2_subblock(
    trace: StationarityTrace,
    positions: Sequence[int],
    *,
    subblock_number: int,
) -> V2SubblockSummary:
    """Summarize one already-validated complete subblock."""

    rows = [trace.observations[position] for position in positions]
    round_indices = [int(row["round_index"]) for row in rows]
    expected_round_indices = list(
        range(round_indices[0], round_indices[0] + len(round_indices))
    )
    if round_indices != expected_round_indices:
        raise ValueError("V2 subblock post_round values must be contiguous")

    position_array = np.asarray(positions, dtype=np.int64)
    normalized_answers = (
        trace.measured_query_answers[position_array] / trace.n_records
    )
    l1_values = np.asarray(
        [row["current_normalized_l1"] for row in rows],
        dtype=np.float64,
    )

    return V2SubblockSummary(
        subblock_number=int(subblock_number),
        start_round_index=round_indices[0],
        end_round_index=round_indices[-1],
        round_count=len(rows),
        normalized_query_mean=tuple(
            float(value) for value in np.mean(normalized_answers, axis=0)
        ),
        l1_mean=float(np.mean(l1_values)),
        l1_p90_minus_p10=float(
            np.percentile(l1_values, 90, method="linear")
            - np.percentile(l1_values, 10, method="linear")
        ),
        unique_row_rate_mean=float(
            np.mean([row["unique_row_rate"] for row in rows])
        ),
        normalized_row_entropy_mean=float(
            np.mean([row["normalized_row_entropy"] for row in rows])
        ),
        active_round_rate=float(
            np.mean([row["actual_changed_row_count"] > 0 for row in rows])
        ),
        mean_changed_row_fraction=float(
            np.mean([
                row["actual_changed_row_count"] / trace.n_records
                for row in rows
            ])
        ),
        mean_changed_query_fraction=float(
            np.mean([
                row["actual_changed_query_count"] / trace.query_count
                for row in rows
            ])
        ),
        mean_normalized_query_l1_movement=float(
            np.mean([
                row["normalized_query_l1_movement_mean"] for row in rows
            ])
        ),
    )


def collect_v2_subblock_summaries(
    trace: StationarityTrace,
    *,
    subblock_round_count: int,
) -> V2SubblockCollection:
    """Collect non-overlapping V2 summaries without thresholds or decisions.

    ``subblock_round_count`` is intentionally required: 100 rounds is the
    current hypothesis, not an implicit universal constant.  Every complete
    subblock contains exactly that many consecutive ``post_round`` states and
    the initial state is never included.  An incomplete tail is not averaged;
    its length is returned explicitly for online callers to keep collecting.

    This analysis seam may be used to compare candidate durations on
    development traces.  A production protocol must freeze a single common
    rule before validation and must not dispatch on dataset identity.
    """

    if not isinstance(trace, StationarityTrace):
        raise ValueError("trace must be a StationarityTrace")
    trace.validate()
    normalized_round_count = _validate_subblock_round_count(
        subblock_round_count
    )

    post_round_positions = trace.post_round_positions()
    complete_subblock_count, trailing_post_round_count = divmod(
        len(post_round_positions), normalized_round_count
    )
    subblocks = []
    for subblock_index in range(complete_subblock_count):
        start = subblock_index * normalized_round_count
        end = start + normalized_round_count
        subblocks.append(
            _summarize_v2_subblock(
                trace,
                post_round_positions[start:end],
                subblock_number=subblock_index + 1,
            )
        )

    return V2SubblockCollection(
        trace_contract_version=trace.contract_version,
        query_identity_sha256=trace.query_identity_sha256,
        target_identity_sha256=trace.target_identity_sha256,
        n_records=trace.n_records,
        query_count=trace.query_count,
        post_round_count=trace.post_round_count,
        subblock_round_count=normalized_round_count,
        trailing_post_round_count=trailing_post_round_count,
        subblocks=tuple(subblocks),
    )


@dataclass(frozen=True)
class V2ScalarEvidence:
    """Auditable, threshold-free evidence for one ordered scalar sequence.

    The descriptive fields correspond to the short mathematical names used in
    the detector design:

    * ``reference_level`` is R, the median of the 12 input values.
    * ``direction_change`` is D, the fitted end-to-end change over 11
      subblock intervals.
    * ``residual_scale`` is S, the robust scale of deviations from the fitted
      line.
    * ``trend_strength`` is T = |D| / S.
    * ``outlier_strength`` is O = max(|residual|) / S.

    ``trend_strength`` and ``outlier_strength`` can be infinite only under the
    explicit zero-scale rules documented by :func:`compute_v2_scalar_evidence`.
    This object is a computational result, not yet a persistence format.
    """

    subblock_values: tuple[float, ...]
    pairwise_slopes: tuple[float, ...]
    slope_per_subblock: float
    trend_intercept: float
    residuals: tuple[float, ...]
    residual_center: float
    reference_level: float
    direction_change: float
    residual_scale: float
    trend_strength: float
    maximum_absolute_residual: float
    outlier_strength: float
    zero_scale: bool
    contract_version: str = field(
        default=STATIONARITY_V2_SCALAR_EVIDENCE_CONTRACT_VERSION,
        init=False,
    )

    @property
    def R(self) -> float:
        """Median reference level（中位参考水平）."""

        return self.reference_level

    @property
    def D(self) -> float:
        """Fitted end-to-end directional change（拟合首尾方向变化）."""

        return self.direction_change

    @property
    def S(self) -> float:
        """Robust residual scale（稳健残差尺度）."""

        return self.residual_scale

    @property
    def T(self) -> float:
        """Absolute trend-to-scale ratio（趋势相对噪声强度）."""

        return self.trend_strength

    @property
    def O(self) -> float:
        """Maximum-residual-to-scale ratio（最大异常残差强度）."""

        return self.outlier_strength


def _coerce_subblock_values(
    subblock_values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Validate and copy the fixed-width scalar evidence input."""

    values = np.asarray(subblock_values)
    if values.ndim != 1:
        raise ValueError("subblock_values must be a one-dimensional sequence")
    if values.size != V2_SCALAR_EVIDENCE_POINT_COUNT:
        raise ValueError(
            "subblock_values must contain exactly "
            f"{V2_SCALAR_EVIDENCE_POINT_COUNT} values; got {values.size}"
        )
    if np.issubdtype(values.dtype, np.bool_) or not (
        np.issubdtype(values.dtype, np.integer)
        or np.issubdtype(values.dtype, np.floating)
    ):
        raise ValueError("subblock_values must contain real numeric values")

    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("subblock_values must contain only finite values")
    return values.copy()


def compute_v2_scalar_evidence(
    subblock_values: Sequence[float] | np.ndarray,
) -> V2ScalarEvidence:
    """Compute R/D/S/T/O without making a convergence decision.

    Let the 12 ordered values be ``x[0]`` through ``x[11]``.  The fitted slope
    ``b`` is the median of all 66 pairwise slopes.  The fitted intercept ``a``
    is ``median(x[i] - b*i)``.  The remaining quantities are:

    ``R = median(x)``
    ``D = 11*b``
    ``residual[i] = x[i] - (a + b*i)``
    ``S = 1.4826 * median(abs(residual - median(residual)))``
    ``T = abs(D)/S``
    ``O = max(abs(residual))/S``

    No epsilon is added to S.  If S is exactly zero, T is zero when D is zero
    and positive infinity otherwise.  Under the same condition, O is zero
    when every residual is zero and positive infinity otherwise.  ``zero_scale``
    records this branch explicitly.
    """

    values = _coerce_subblock_values(subblock_values)
    positions = np.arange(V2_SCALAR_EVIDENCE_POINT_COUNT, dtype=np.float64)

    slopes = np.asarray(
        [
            (values[j] - values[i]) / (positions[j] - positions[i])
            for i in range(V2_SCALAR_EVIDENCE_POINT_COUNT - 1)
            for j in range(i + 1, V2_SCALAR_EVIDENCE_POINT_COUNT)
        ],
        dtype=np.float64,
    )
    slope = float(np.median(slopes))
    intercept = float(np.median(values - slope * positions))
    residuals = values - (intercept + slope * positions)
    residual_center = float(np.median(residuals))

    reference_level = float(np.median(values))
    direction_change = float(
        slope * (V2_SCALAR_EVIDENCE_POINT_COUNT - 1)
    )
    residual_scale = float(
        MAD_NORMAL_CONSISTENCY_FACTOR
        * np.median(np.abs(residuals - residual_center))
    )
    maximum_absolute_residual = float(np.max(np.abs(residuals)))

    zero_scale = residual_scale == 0.0
    if zero_scale:
        trend_strength = 0.0 if direction_change == 0.0 else inf
        outlier_strength = (
            0.0 if maximum_absolute_residual == 0.0 else inf
        )
    else:
        trend_strength = abs(direction_change) / residual_scale
        outlier_strength = maximum_absolute_residual / residual_scale

    return V2ScalarEvidence(
        subblock_values=tuple(float(value) for value in values),
        pairwise_slopes=tuple(float(value) for value in slopes),
        slope_per_subblock=slope,
        trend_intercept=intercept,
        residuals=tuple(float(value) for value in residuals),
        residual_center=residual_center,
        reference_level=reference_level,
        direction_change=direction_change,
        residual_scale=residual_scale,
        trend_strength=float(trend_strength),
        maximum_absolute_residual=maximum_absolute_residual,
        outlier_strength=float(outlier_strength),
        zero_scale=zero_scale,
    )
