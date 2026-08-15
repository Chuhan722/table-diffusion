"""Threshold-free scalar evidence for the Issue #53 V2 detector.

This module implements only the mathematical summary of one V2 candidate
window.  It deliberately does not contain thresholds, convergence states, or
stopping decisions.  Those policy choices belong to a later layer and must be
calibrated separately from this evidence contract.

The candidate window contains three 400-round blocks, each represented by four
100-round subblock summaries.  Therefore this primitive accepts exactly 12
ordered scalar values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Sequence

import numpy as np


STATIONARITY_V2_SCALAR_EVIDENCE_CONTRACT_VERSION = (
    "issue53-stage2-v2-scalar-evidence-v1"
)
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
