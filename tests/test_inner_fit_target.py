"""Deterministic boundary tests for the pure inner fit-target interface."""

import math

import numpy as np
import pytest

from table_diffevo.inner_fit_target import (
    QueryFitThresholds,
    assess_query_fit,
)


def test_exact_count_candidate_accepts_observed_two_count_peak() -> None:
    thresholds = QueryFitThresholds.exact_integer_counts()

    result = assess_query_fit([2, 1, 1, 0, 0, 0], thresholds)

    assert result.squared_loss == 3.0
    assert result.count_rmse == 1.0
    assert result.max_abs_count_error == 2.0
    assert result.rmse_within_limit is True
    assert result.every_query_within_limit is True
    assert result.fit_target_reached is True
    assert result.exact_residual is False
    assert result.calibration_source == "exact_integer_counts"


def test_peak_guard_rejects_large_outlier_hidden_by_rmse() -> None:
    errors = np.zeros(100, dtype=float)
    errors[17] = -10.0

    result = assess_query_fit(
        errors,
        QueryFitThresholds.exact_integer_counts(),
    )

    assert result.count_rmse == 1.0
    assert result.rmse_within_limit is True
    assert result.max_abs_count_error == 10.0
    assert result.every_query_within_limit is False
    assert result.fit_target_reached is False


def test_rmse_and_peak_limits_must_both_pass() -> None:
    result = assess_query_fit(
        [2, -2],
        QueryFitThresholds.exact_integer_counts(),
    )

    assert result.count_rmse == 2.0
    assert result.max_abs_count_error == 2.0
    assert result.rmse_within_limit is False
    assert result.every_query_within_limit is True
    assert result.fit_target_reached is False


def test_exact_residual_is_qualified_and_preserves_all_metrics() -> None:
    result = assess_query_fit(
        [0, -0.0, 0],
        QueryFitThresholds.exact_integer_counts(),
    )

    assert result.squared_loss == 0.0
    assert result.count_rmse == 0.0
    assert result.max_abs_count_error == 0.0
    assert result.per_query_abs_count_errors == (0.0, 0.0, 0.0)
    assert result.fit_target_reached is True
    assert result.exact_residual is True


def test_limits_are_closed_at_boundary_and_fail_immediately_above_it() -> None:
    thresholds = QueryFitThresholds.exact_integer_counts()

    at_boundary = assess_query_fit([2, 1, 1, 0, 0, 0], thresholds)
    above_peak = assess_query_fit(
        [np.nextafter(2.0, math.inf), 0, 0, 0, 0, 0],
        thresholds,
    )
    above_rmse = assess_query_fit(
        [np.nextafter(1.0, math.inf)] * 6,
        thresholds,
    )

    assert at_boundary.fit_target_reached is True
    assert above_peak.rmse_within_limit is True
    assert above_peak.every_query_within_limit is False
    assert above_peak.fit_target_reached is False
    assert above_rmse.rmse_within_limit is False
    assert above_rmse.every_query_within_limit is True
    assert above_rmse.fit_target_reached is False


def test_sign_changes_do_not_change_fit_assessment() -> None:
    thresholds = QueryFitThresholds.exact_integer_counts()

    positive = assess_query_fit([2, 1, 1, 0], thresholds)
    signed = assess_query_fit([-2, 1, -1, 0], thresholds)

    assert positive == signed


def test_noise_interface_accepts_external_heterogeneous_limits_only() -> None:
    thresholds = QueryFitThresholds.external_noise_calibrated(
        count_rmse_limit=3.0,
        max_abs_count_error_limit=(1.0, 2.0, 4.0),
    )

    qualified = assess_query_fit([1.0, -2.0, 3.0], thresholds)
    one_query_failed = assess_query_fit([1.0, -2.5, 3.0], thresholds)

    assert thresholds.calibration_source == "external_noise_calibrated"
    assert thresholds.per_query_max_limits(3) == (1.0, 2.0, 4.0)
    assert qualified.fit_target_reached is True
    assert qualified.calibration_source == "external_noise_calibrated"
    assert one_query_failed.rmse_within_limit is True
    assert one_query_failed.every_query_within_limit is False
    assert one_query_failed.fit_target_reached is False


def test_uniform_noise_limit_expands_without_sigma_input() -> None:
    thresholds = QueryFitThresholds.external_noise_calibrated(
        count_rmse_limit=2.5,
        max_abs_count_error_limit=4.0,
    )

    assert thresholds.per_query_max_limits(3) == (4.0, 4.0, 4.0)
    assert assess_query_fit([2.0, -1.0, 3.0], thresholds).fit_target_reached


@pytest.mark.parametrize(
    "changes,error_type",
    [
        ({"count_rmse_limit": True}, TypeError),
        ({"count_rmse_limit": -1.0}, ValueError),
        ({"count_rmse_limit": math.inf}, ValueError),
        ({"max_abs_count_error_limit": True}, TypeError),
        ({"max_abs_count_error_limit": ()}, ValueError),
        ({"max_abs_count_error_limit": (1.0, math.nan)}, ValueError),
        ({"calibration_source": "unknown"}, ValueError),
    ],
)
def test_invalid_thresholds_fail_closed(changes, error_type) -> None:
    kwargs = {
        "count_rmse_limit": 1.0,
        "max_abs_count_error_limit": 2.0,
        "calibration_source": "exact_integer_counts",
    }
    kwargs.update(changes)

    with pytest.raises(error_type):
        QueryFitThresholds(**kwargs)


@pytest.mark.parametrize(
    "count_errors,error_type",
    [
        ([], ValueError),
        ([[1.0]], ValueError),
        ([True, False], TypeError),
        (["1.0"], TypeError),
        ([1.0 + 1.0j], TypeError),
        ([math.nan], ValueError),
        ([math.inf], ValueError),
        ([1e308, 1e308], ValueError),
    ],
)
def test_invalid_count_errors_fail_closed(count_errors, error_type) -> None:
    with pytest.raises(error_type):
        assess_query_fit(
            count_errors,
            QueryFitThresholds.exact_integer_counts(),
        )


def test_per_query_noise_limit_length_must_match_errors() -> None:
    thresholds = QueryFitThresholds.external_noise_calibrated(
        count_rmse_limit=2.0,
        max_abs_count_error_limit=(1.0, 2.0),
    )

    with pytest.raises(ValueError, match="length must match"):
        assess_query_fit([0.0, 0.0, 0.0], thresholds)


def test_api_rejects_sigma_l1_and_reference_inputs() -> None:
    thresholds = QueryFitThresholds.exact_integer_counts()

    with pytest.raises(TypeError):
        assess_query_fit([0.0], thresholds, sigma=1.0)
    with pytest.raises(TypeError):
        assess_query_fit([0.0], thresholds, normalized_l1=0.0)
    with pytest.raises(TypeError):
        assess_query_fit([0.0], thresholds, reference_table=object())
