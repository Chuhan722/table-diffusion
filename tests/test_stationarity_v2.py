from __future__ import annotations

import math

import numpy as np
import pytest

from table_diffevo.stationarity_v2 import (
    MAD_NORMAL_CONSISTENCY_FACTOR,
    STATIONARITY_V2_SCALAR_EVIDENCE_CONTRACT_VERSION,
    V2_PAIRWISE_SLOPE_COUNT,
    V2_SCALAR_EVIDENCE_POINT_COUNT,
    compute_v2_scalar_evidence,
)


def test_constant_sequence_has_zero_direction_scale_and_outlier_evidence() -> None:
    evidence = compute_v2_scalar_evidence([3.5] * 12)

    assert evidence.contract_version == (
        STATIONARITY_V2_SCALAR_EVIDENCE_CONTRACT_VERSION
    )
    assert len(evidence.subblock_values) == V2_SCALAR_EVIDENCE_POINT_COUNT
    assert len(evidence.pairwise_slopes) == V2_PAIRWISE_SLOPE_COUNT == 66
    assert evidence.R == 3.5
    assert evidence.slope_per_subblock == 0.0
    assert evidence.D == 0.0
    assert evidence.S == 0.0
    assert evidence.T == 0.0
    assert evidence.maximum_absolute_residual == 0.0
    assert evidence.O == 0.0
    assert evidence.zero_scale is True


def test_exact_line_preserves_direction_when_residual_scale_is_zero() -> None:
    values = [2.0 + 0.5 * index for index in range(12)]

    evidence = compute_v2_scalar_evidence(values)

    assert evidence.R == pytest.approx(4.75)
    assert evidence.slope_per_subblock == pytest.approx(0.5)
    assert evidence.trend_intercept == pytest.approx(2.0)
    assert evidence.D == pytest.approx(5.5)
    assert evidence.S == 0.0
    assert math.isinf(evidence.T)
    assert evidence.O == 0.0
    assert evidence.zero_scale is True
    assert evidence.residuals == pytest.approx((0.0,) * 12)


def test_single_spike_is_not_mistaken_for_a_trend() -> None:
    values = [0.0] * 12
    values[5] = 10.0

    evidence = compute_v2_scalar_evidence(values)

    assert evidence.R == 0.0
    assert evidence.slope_per_subblock == 0.0
    assert evidence.D == 0.0
    assert evidence.S == 0.0
    assert evidence.T == 0.0
    assert evidence.maximum_absolute_residual == 10.0
    assert math.isinf(evidence.O)
    assert evidence.zero_scale is True


def test_affine_rescaling_preserves_dimensionless_strengths() -> None:
    values = np.asarray(
        [
            2.10,
            2.05,
            2.50,
            2.45,
            2.95,
            2.80,
            3.40,
            3.35,
            3.70,
            3.95,
            4.05,
            4.35,
        ]
    )
    scale = 3.0
    offset = 17.0

    original = compute_v2_scalar_evidence(values)
    transformed = compute_v2_scalar_evidence(offset + scale * values)

    assert original.S > 0.0
    assert transformed.R == pytest.approx(offset + scale * original.R)
    assert transformed.D == pytest.approx(scale * original.D)
    assert transformed.S == pytest.approx(scale * original.S)
    assert transformed.maximum_absolute_residual == pytest.approx(
        scale * original.maximum_absolute_residual
    )
    assert transformed.T == pytest.approx(original.T)
    assert transformed.O == pytest.approx(original.O)
    assert transformed.zero_scale is False


def test_reversing_sequence_reverses_only_the_direction() -> None:
    values = [
        2.10,
        2.05,
        2.50,
        2.45,
        2.95,
        2.80,
        3.40,
        3.35,
        3.70,
        3.95,
        4.05,
        4.35,
    ]

    forward = compute_v2_scalar_evidence(values)
    backward = compute_v2_scalar_evidence(list(reversed(values)))

    assert backward.R == pytest.approx(forward.R)
    assert backward.D == pytest.approx(-forward.D)
    assert backward.S == pytest.approx(forward.S)
    assert backward.T == pytest.approx(forward.T)
    assert backward.O == pytest.approx(forward.O)


def test_slope_residual_and_scale_follow_the_frozen_formulas() -> None:
    values = np.asarray(
        [0.0, 0.3, 0.1, 0.7, 0.4, 1.2, 0.9, 1.5, 1.4, 1.9, 1.7, 2.4]
    )
    expected_slopes = [
        (values[j] - values[i]) / (j - i)
        for i in range(11)
        for j in range(i + 1, 12)
    ]
    expected_slope = float(np.median(expected_slopes))
    expected_intercept = float(
        np.median(values - expected_slope * np.arange(12))
    )
    expected_residuals = values - (
        expected_intercept + expected_slope * np.arange(12)
    )
    expected_residual_center = float(np.median(expected_residuals))
    expected_scale = float(
        MAD_NORMAL_CONSISTENCY_FACTOR
        * np.median(np.abs(expected_residuals - expected_residual_center))
    )

    evidence = compute_v2_scalar_evidence(values)

    assert evidence.pairwise_slopes == pytest.approx(expected_slopes)
    assert evidence.slope_per_subblock == pytest.approx(expected_slope)
    assert evidence.trend_intercept == pytest.approx(expected_intercept)
    assert evidence.residuals == pytest.approx(expected_residuals)
    assert evidence.residual_center == pytest.approx(expected_residual_center)
    assert evidence.D == pytest.approx(11.0 * expected_slope)
    assert evidence.S == pytest.approx(expected_scale)
    assert evidence.T == pytest.approx(abs(evidence.D) / evidence.S)
    assert evidence.O == pytest.approx(
        evidence.maximum_absolute_residual / evidence.S
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.0] * 11, "exactly 12"),
        ([0.0] * 13, "exactly 12"),
        ([[0.0] * 12], "one-dimensional"),
        ([0.0] * 11 + [math.nan], "finite"),
        ([0.0] * 11 + [math.inf], "finite"),
        ([False] * 12, "real numeric"),
        (["0"] * 12, "real numeric"),
    ],
)
def test_invalid_inputs_are_rejected(
    values: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_v2_scalar_evidence(values)  # type: ignore[arg-type]
