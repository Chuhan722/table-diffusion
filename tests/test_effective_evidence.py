"""Deterministic tests for Issue #53 V2 effective-round evidence."""

from dataclasses import asdict

import numpy as np
import pytest

from table_diffevo.effective_evidence import (
    V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION,
    compute_v2_effective_round_evidence,
)


FORBIDDEN_DECISION_FIELDS = {
    "stable",
    "converged",
    "qualified",
    "stop",
    "stop_round",
    "threshold",
    "quality_pass",
}


def _round_indices(count: int, *, start: int = 1) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.int64)


def test_overlapping_batch_formula_is_recomputed_exactly() -> None:
    result = compute_v2_effective_round_evidence(
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
    )

    assert result.actual_round_count == 4
    assert result.batch_round_count == 2
    assert result.overlapping_batch_count == 3
    assert result.single_round_variance == pytest.approx(5.0 / 3.0)
    assert result.long_run_variance == pytest.approx(8.0 / 3.0)
    assert result.raw_correlation_inflation == pytest.approx(8.0 / 5.0)
    assert result.conservative_correlation_inflation == pytest.approx(
        8.0 / 5.0
    )
    assert result.raw_effective_round_count == pytest.approx(2.5)
    assert result.effective_round_count == pytest.approx(2.5)
    assert result.mcse == pytest.approx(np.sqrt(2.0 / 3.0))
    assert result.numerically_estimable is True
    assert result.reason is None
    assert result.stationarity_not_assessed is True
    assert result.contract_version == (
        V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION
    )


@pytest.mark.parametrize(
    "round_indices",
    [
        [1],
        [0, 1],
        [1, 3],
        [1, 1],
        [2, 1],
        [1.0, 2.0],
        [1, True],
        [[1, 2], [3, 4]],
    ],
)
def test_round_identity_contract_rejects_invalid_inputs(
    round_indices,
) -> None:
    values = np.arange(len(round_indices), dtype=np.float64)

    with pytest.raises(ValueError):
        compute_v2_effective_round_evidence(round_indices, values)


@pytest.mark.parametrize(
    "values",
    [
        [1.0],
        [False, True],
        [1.0, True],
        ["1", "2"],
        [[1.0], [2.0]],
        [np.nan, 1.0],
        [np.inf, 1.0],
        [-np.inf, 1.0],
    ],
)
def test_value_contract_rejects_invalid_inputs(values) -> None:
    with pytest.raises(ValueError):
        compute_v2_effective_round_evidence([1, 2], values)


def test_constant_sequence_fails_closed() -> None:
    result = compute_v2_effective_round_evidence(
        _round_indices(32),
        np.ones(32),
    )

    assert result.numerically_estimable is False
    assert result.reason == "zero_round_variance"
    assert result.single_round_variance == 0.0
    assert result.long_run_variance is None
    assert result.raw_correlation_inflation is None
    assert result.effective_round_count is None
    assert result.mcse is None


def test_periodic_batch_coupling_fails_closed() -> None:
    values = np.where(np.arange(256) % 2 == 0, 1.0, -1.0)

    result = compute_v2_effective_round_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.batch_round_count == 16
    assert result.single_round_variance is not None
    assert result.single_round_variance > 0.0
    assert result.long_run_variance == 0.0
    assert result.numerically_estimable is False
    assert result.reason == "degenerate_long_run_variance"
    assert result.effective_round_count is None
    assert result.mcse is None


def test_shift_and_positive_scale_have_the_required_invariance() -> None:
    positions = np.arange(64, dtype=np.float64)
    values = np.sin(0.37 * positions) + 0.01 * positions
    indices = _round_indices(len(values), start=101)

    base = compute_v2_effective_round_evidence(indices, values)
    shifted = compute_v2_effective_round_evidence(indices, values + 13.0)
    scaled = compute_v2_effective_round_evidence(indices, values * 7.0)

    assert base.numerically_estimable is True
    assert shifted.numerically_estimable is True
    assert scaled.numerically_estimable is True
    assert shifted.raw_correlation_inflation == pytest.approx(
        base.raw_correlation_inflation,
        rel=1e-10,
        abs=1e-12,
    )
    assert shifted.effective_round_count == pytest.approx(
        base.effective_round_count,
        rel=1e-10,
        abs=1e-12,
    )
    assert shifted.mcse == pytest.approx(
        base.mcse,
        rel=1e-10,
        abs=1e-12,
    )
    assert scaled.raw_correlation_inflation == pytest.approx(
        base.raw_correlation_inflation,
        rel=1e-10,
        abs=1e-12,
    )
    assert scaled.effective_round_count == pytest.approx(
        base.effective_round_count,
        rel=1e-10,
        abs=1e-12,
    )
    assert scaled.mcse == pytest.approx(
        7.0 * base.mcse,
        rel=1e-10,
        abs=1e-12,
    )


def test_single_spike_stays_finite_and_cannot_create_extra_evidence() -> None:
    values = np.zeros(256, dtype=np.float64)
    values[len(values) // 2] = 1.0

    result = compute_v2_effective_round_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.numerically_estimable is True
    assert result.reason is None
    assert result.effective_round_count is not None
    assert 0.0 < result.effective_round_count <= len(values)
    assert np.all(np.isfinite([
        result.single_round_variance,
        result.long_run_variance,
        result.raw_correlation_inflation,
        result.conservative_correlation_inflation,
        result.raw_effective_round_count,
        result.effective_round_count,
        result.mcse,
    ]))


def test_negative_correlation_raw_ess_can_exceed_n_but_formal_ess_cannot() -> None:
    values = np.where(np.arange(25) % 2 == 0, 1.0, -1.0)

    result = compute_v2_effective_round_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.numerically_estimable is True
    assert result.raw_effective_round_count is not None
    assert result.raw_effective_round_count > len(values)
    assert result.effective_round_count == pytest.approx(len(values))
    assert result.conservative_correlation_inflation == 1.0
    assert result.single_round_variance is not None
    assert result.mcse == pytest.approx(
        np.sqrt(result.single_round_variance / len(values))
    )


def test_trend_is_numerical_evidence_not_a_convergence_decision() -> None:
    result = compute_v2_effective_round_evidence(
        _round_indices(64),
        np.linspace(0.0, 1.0, 64),
    )
    fields = set(asdict(result))

    assert result.numerically_estimable is True
    assert result.stationarity_not_assessed is True
    assert FORBIDDEN_DECISION_FIELDS.isdisjoint(fields)


def test_finite_input_that_overflows_fails_closed() -> None:
    result = compute_v2_effective_round_evidence(
        [1, 2, 3, 4],
        [1e308, -1e308, 1e308, -1e308],
    )

    assert result.numerically_estimable is False
    assert result.reason == "nonfinite_computation"
    assert result.effective_round_count is None
    assert result.mcse is None


def test_inputs_are_not_mutated() -> None:
    round_indices = _round_indices(16)
    values = np.linspace(-1.0, 1.0, 16)
    original_round_indices = round_indices.copy()
    original_values = values.copy()

    compute_v2_effective_round_evidence(round_indices, values)

    np.testing.assert_array_equal(round_indices, original_round_indices)
    np.testing.assert_array_equal(values, original_values)
