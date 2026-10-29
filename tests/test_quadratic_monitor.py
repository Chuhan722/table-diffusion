"""条件平方风险差、有限观察预算与数值/输入边界。"""
from dataclasses import asdict
from decimal import getcontext, localcontext
from fractions import Fraction
import json

import numpy as np
import pytest
from scipy.special import ndtri

from resevo.quadratic_monitor import IndependentGaussianQuadraticMonitor


def _monitor(**overrides):
    kwargs = dict(
        reference=[0, 0], validation=[1.0, 1.0], validation_sigma=[0.1, 0.2],
        weights=[1.0, 1.0], error_probability=0.05, max_looks=3,
    )
    kwargs.update(overrides)
    return IndependentGaussianQuadraticMonitor(**kwargs)


def test_difference_and_known_variance_match_independent_formula():
    rng = np.random.default_rng(20261029)
    reference = rng.integers(0, 20, 7)
    validation = rng.normal(10, 3, 7)
    sigma = np.array([0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0])
    weights = np.array([1.0, 2.0, 0.25, 3.0, 0.5, 1.0, 2.0])
    monitor = _monitor(
        reference=reference, validation=validation, validation_sigma=sigma,
        weights=weights, max_looks=200,
    )
    for look in range(1, 201):
        candidate = rng.integers(0, 20, 7)
        row = monitor.observe(candidate)
        difference = 0.5 * np.sum(weights * ((candidate - validation) ** 2 - (reference - validation) ** 2))
        variance = sum(
            (Fraction.from_float(float(w)) * int(s - r) * Fraction.from_float(float(sd))) ** 2
            for w, s, r, sd in zip(weights, candidate, reference, sigma)
        )
        ideal_bound = difference - monitor.critical_value * np.sqrt(float(variance))
        assert row.look == look and monitor.looks_used == look
        assert row.risk_difference == pytest.approx(difference, rel=1e-12, abs=1e-12)
        assert Fraction.from_float(row.standard_error_upper) ** 2 >= variance
        assert row.lower_bound <= ideal_bound + 1e-11
        assert abs(ideal_bound) > 1e-8
        assert row.worse == (ideal_bound > 0)
        json.dumps(asdict(row), allow_nan=False)


def test_large_total_losses_do_not_erase_difference():
    monitor = _monitor(reference=[1, 1], validation=[1e16, 1e16], validation_sigma=[1.0, 1.0])
    row = monitor.observe([2, 0])
    assert row.risk_difference == 1.0
    assert row.standard_error_upper == pytest.approx(np.sqrt(2.0), rel=1e-15)


@pytest.mark.parametrize("validation, expected_difference, worse", [
    (0.0, 2.0, True), (1.0, 0.0, False), (2.0, -2.0, False),
])
def test_zero_noise_is_not_an_estimated_zero_standard_error(validation, expected_difference, worse):
    monitor = _monitor(reference=[0], validation=[validation], validation_sigma=[0.0], weights=[1.0])
    row = monitor.observe([2])
    assert row.zero_variance and row.standard_error_upper == 0.0
    assert row.risk_difference == expected_difference and row.worse is worse


def test_identical_answers_are_exact_zero_and_still_use_a_look():
    monitor = _monitor(reference=[3, 5], validation=[-1.0, 11.0])
    for look in range(1, 4):
        row = monitor.observe([3, 5])
        assert row.look == look and row.risk_difference == row.lower_bound == 0.0
        assert row.zero_variance and not row.worse
    with pytest.raises(ValueError, match="预算已耗尽"):
        monitor.observe([4, 5])
    assert monitor.looks_used == 3


def test_allocation_is_fixed_and_not_overspent_by_rounding():
    monitor = _monitor(error_probability=0.05, max_looks=7)
    assert Fraction.from_float(monitor.error_probability_per_look) * 7 <= Fraction.from_float(0.05)
    assert monitor.critical_value >= -ndtri(monitor.error_probability_per_look)
    first = monitor.observe([2, 2])
    for _ in range(6):
        again = monitor.observe([2, 2])
        assert again.error_probability_per_look == first.error_probability_per_look
        assert again.critical_value == first.critical_value
        assert again.lower_bound == first.lower_bound
    with pytest.raises(ValueError, match="预算已耗尽"):
        monitor.observe([2, 2])


def test_equality_at_the_critical_boundary_does_not_alarm():
    prototype = _monitor(reference=[0], validation=[0.0], validation_sigma=[1.0], weights=[1.0])
    critical = prototype.critical_value
    for offset, expected in ((0.0, False), (-1e-8, True), (1e-8, False)):
        monitor = _monitor(
            reference=[0], validation=[0.5 - critical + offset],
            validation_sigma=[1.0], weights=[1.0],
        )
        assert monitor.observe([1]).worse is expected


def test_input_ownership_and_candidate_are_not_mutated():
    reference = np.array([1.0, 2.0])
    validation = np.array([-0.5, 3.5])
    sigma = np.array([0.1, 0.2])
    weights = np.array([2.0, 0.5])
    expected = _monitor(reference=reference, validation=validation, validation_sigma=sigma, weights=weights)
    actual = _monitor(reference=reference, validation=validation, validation_sigma=sigma, weights=weights)
    for values in (reference, validation, sigma, weights):
        assert values.flags.writeable
        values[:] = 99
    candidate = np.array([3.0, 1.0])
    before = candidate.copy()
    assert asdict(actual.observe(candidate)) == asdict(expected.observe(candidate))
    np.testing.assert_array_equal(candidate, before)
    assert candidate.flags.writeable


@pytest.mark.parametrize("kwargs", [
    {"reference": []}, {"reference": [[0, 0]]}, {"reference": [0.5, 0]},
    {"reference": [-1, 0]}, {"reference": [2**50 + 1, 0]},
    {"reference": [np.inf, 0]}, {"validation": [np.nan, 0]},
    {"validation": [1]}, {"validation_sigma": [1]}, {"weights": [1]},
    {"validation_sigma": [-1, 1]}, {"validation_sigma": [np.inf, 1]},
    {"weights": [0, 1]}, {"weights": [-1, 1]}, {"weights": [np.nan, 1]},
    {"error_probability": 0.0}, {"error_probability": 0.5},
    {"error_probability": 1.0}, {"error_probability": np.nan},
    {"error_probability": np.inf}, {"max_looks": 0}, {"max_looks": -1},
    {"max_looks": True}, {"max_looks": np.bool_(True)}, {"max_looks": 2.0},
    {"max_looks": 2**64}, {"max_looks": np.uint64(2**63)},
])
def test_invalid_constructor_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _monitor(**kwargs)


@pytest.mark.parametrize("candidate", [[], [[0, 0]], [0], [-1, 0], [1.5, 0], [np.nan, 0], [2**50 + 1, 0]])
def test_invalid_observation_does_not_consume_a_look(candidate):
    monitor = _monitor()
    with pytest.raises(ValueError):
        monitor.observe(candidate)
    assert monitor.looks_used == 0
    assert monitor.observe([2, 2]).look == 1


def test_unrepresentable_per_look_probability_is_rejected():
    with pytest.raises(ValueError, match="不可表示"):
        _monitor(error_probability=np.nextafter(0.0, 1.0), max_looks=2)


def test_variance_does_not_square_large_or_small_scales_in_float64():
    monitor = _monitor(
        reference=[0], validation=[0.5], validation_sigma=[1e-200], weights=[1e200],
    )
    precision = getcontext().prec
    row = monitor.observe([1])
    assert row.risk_difference == 0.0 and not row.zero_variance
    assert row.standard_error_upper == pytest.approx(1.0, rel=1e-15)
    assert getcontext().prec == precision


def test_decimal_context_does_not_change_uncertainty_computation():
    with localcontext() as context:
        context.prec, context.Emax, context.Emin = 2, 10, -10
        monitor = _monitor(reference=[0], validation=[0.5], validation_sigma=[1e100], weights=[1.0])
        row = monitor.observe([1])
        assert row.standard_error_upper == pytest.approx(1e100, rel=1e-15)
        assert (context.prec, context.Emax, context.Emin) == (2, 10, -10)


@pytest.mark.parametrize("weight,sigma", [
    (np.nextafter(0.0, 1.0), np.nextafter(0.0, 1.0)),
    (1e308, 1e308),
    (1.0, 1e308),
])
def test_unrepresentable_uncertainty_is_not_reported_as_noise_free(weight, sigma):
    monitor = _monitor(reference=[0], validation=[0.5], validation_sigma=[sigma], weights=[weight])
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:"):
        monitor.observe([1])
    assert monitor.looks_used == 0


def test_unrepresentable_mean_difference_is_explicit():
    monitor = _monitor(reference=[0], validation=[-1e308], validation_sigma=[1.0], weights=[1e308])
    with pytest.raises(FloatingPointError, match="numerical_indeterminate:"):
        monitor.observe([1])
    assert monitor.looks_used == 0
