"""Deterministic tests for Issue #53 V2b adaptive evidence."""

from dataclasses import asdict
from math import isqrt

import numpy as np
import pytest

from table_diffevo.adaptive_effective_evidence import (
    V2B_ADAPTIVE_CHECKPOINTS,
    V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION,
    V2B_RESOURCE_ROUND_CAP,
    V2B_SCALE_RATIO_LIMIT,
    compute_v2b_adaptive_checkpoint_evidence,
    compute_v2b_adaptive_trajectory_evidence,
    compute_v2b_scale_ratio,
    summarize_v2b_adaptive_checkpoint_decisions,
    v2b_scale_ratio_is_acceptable,
)
from table_diffevo.effective_evidence import (
    compute_v2_effective_round_evidence,
    compute_v2_effective_round_evidence_for_batch,
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


def _positions(count: int = 256) -> np.ndarray:
    return np.arange(count, dtype=np.float64)


def test_explicit_batch_lengths_recompute_two_hand_checked_obm_values() -> None:
    indices = [1, 2, 3, 4]
    values = [1.0, 2.0, 3.0, 4.0]

    short = compute_v2_effective_round_evidence_for_batch(
        indices,
        values,
        batch_round_count=1,
    )
    long = compute_v2_effective_round_evidence_for_batch(
        indices,
        values,
        batch_round_count=2,
    )

    assert short.single_round_variance == pytest.approx(5.0 / 3.0)
    assert short.long_run_variance == pytest.approx(5.0 / 3.0)
    assert long.single_round_variance == pytest.approx(5.0 / 3.0)
    assert long.long_run_variance == pytest.approx(8.0 / 3.0)


@pytest.mark.parametrize("batch_round_count", [0, 4, True, 1.5, "2"])
def test_explicit_batch_length_rejects_invalid_values(
    batch_round_count,
) -> None:
    with pytest.raises(ValueError):
        compute_v2_effective_round_evidence_for_batch(
            [1, 2, 3, 4],
            [1.0, 2.0, 3.0, 4.0],
            batch_round_count=batch_round_count,
        )


def test_explicit_numpy_integer_batch_length_is_accepted() -> None:
    result = compute_v2_effective_round_evidence_for_batch(
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
        batch_round_count=np.int64(2),
    )

    assert result.batch_round_count == 2


@pytest.mark.parametrize(
    "values",
    [
        np.sin(0.17 * _positions()),
        np.linspace(-1.0, 1.0, 256),
        np.ones(256),
    ],
)
def test_original_v2_entry_point_is_exactly_the_explicit_sqrt_batch(
    values: np.ndarray,
) -> None:
    indices = _round_indices(len(values))

    original = compute_v2_effective_round_evidence(indices, values)
    explicit = compute_v2_effective_round_evidence_for_batch(
        indices,
        values,
        batch_round_count=isqrt(len(values)),
    )

    assert original == explicit


def test_scale_ratio_threshold_has_exact_inclusive_boundary() -> None:
    just_above = np.nextafter(V2B_SCALE_RATIO_LIMIT, np.inf)

    assert compute_v2b_scale_ratio(1.0, 1.25) == 1.25
    assert v2b_scale_ratio_is_acceptable(1.0, 1.25) is True
    assert v2b_scale_ratio_is_acceptable(1.25, 1.0) is True
    assert v2b_scale_ratio_is_acceptable(1.0, just_above) is False


@pytest.mark.parametrize(
    "short_inflation,long_inflation",
    [
        (True, 1.0),
        (1.0, False),
        (0.99, 1.0),
        (1.0, np.nan),
        (np.inf, 1.0),
        ("1", 1.0),
    ],
)
def test_scale_ratio_rejects_nonconservative_or_nonreal_inputs(
    short_inflation,
    long_inflation,
) -> None:
    with pytest.raises(ValueError):
        compute_v2b_scale_ratio(short_inflation, long_inflation)


def test_checkpoint_uses_the_two_frozen_batch_lengths() -> None:
    values = np.sin(0.5 * _positions())
    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.actual_round_count == 256
    assert result.short_batch_round_count == 16
    assert result.long_batch_round_count == 32
    assert result.short_numerically_estimable is True
    assert result.long_numerically_estimable is True
    assert result.scale_ratio == pytest.approx(1.0)
    assert result.adaptive_numerically_estimable is True
    assert result.reason is None
    assert result.stationarity_not_assessed is True
    assert result.contract_version == (
        V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
    )


@pytest.mark.parametrize(
    "values,larger_scale",
    [
        (np.linspace(0.0, 1.0, 256), "long"),
        (np.sin(0.15 * _positions()), "short"),
    ],
)
def test_official_uncertainty_is_always_the_larger_scale(
    values: np.ndarray,
    larger_scale: str,
) -> None:
    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(len(values)),
        values,
    )

    short = result.short_conservative_correlation_inflation
    long = result.long_conservative_correlation_inflation
    assert short is not None
    assert long is not None
    assert result.official_correlation_inflation == max(short, long)
    if larger_scale == "short":
        assert short > long
    else:
        assert long > short
    assert result.single_round_variance is not None
    assert result.official_long_run_variance == pytest.approx(
        result.single_round_variance * max(short, long)
    )
    assert result.effective_round_count == pytest.approx(
        len(values) / max(short, long)
    )
    assert result.mcse == pytest.approx(
        np.sqrt(result.official_long_run_variance / len(values))
    )


def test_one_degenerate_scale_makes_the_whole_checkpoint_fail_closed() -> None:
    values = np.where(
        (np.arange(256) // 16) % 2 == 0,
        1.0,
        -1.0,
    )

    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.short_numerically_estimable is True
    assert result.long_numerically_estimable is False
    assert result.long_reason == "degenerate_long_run_variance"
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "core_not_estimable"
    assert result.scale_ratio is None
    assert result.official_correlation_inflation is None
    assert result.effective_round_count is None
    assert result.mcse is None


def test_constant_sequence_makes_both_scales_fail_closed() -> None:
    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(256),
        np.ones(256),
    )

    assert result.short_numerically_estimable is False
    assert result.long_numerically_estimable is False
    assert result.short_reason == "zero_round_variance"
    assert result.long_reason == "zero_round_variance"
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "core_not_estimable"


def test_single_spike_stays_finite_and_cannot_create_extra_evidence() -> None:
    values = np.zeros(256, dtype=np.float64)
    values[len(values) // 2] = 1.0

    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.short_numerically_estimable is True
    assert result.long_numerically_estimable is True
    assert result.effective_round_count is not None
    assert 0.0 < result.effective_round_count <= len(values)
    assert result.mcse is not None
    assert np.all(np.isfinite([
        result.short_long_run_variance,
        result.long_long_run_variance,
        result.scale_ratio,
        result.official_correlation_inflation,
        result.official_long_run_variance,
        result.effective_round_count,
        result.mcse,
    ]))


def test_negative_correlation_floor_applies_to_both_scales_and_mcse() -> None:
    values = np.sin(0.5 * _positions())

    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.short_raw_correlation_inflation is not None
    assert result.long_raw_correlation_inflation is not None
    assert result.short_raw_correlation_inflation < 1.0
    assert result.long_raw_correlation_inflation < 1.0
    assert result.short_conservative_correlation_inflation == 1.0
    assert result.long_conservative_correlation_inflation == 1.0
    assert result.effective_round_count == pytest.approx(len(values))
    assert result.single_round_variance is not None
    assert result.mcse == pytest.approx(
        np.sqrt(result.single_round_variance / len(values))
    )


def test_shift_and_positive_scale_preserve_adaptive_diagnostics() -> None:
    values = np.sin(0.07 * _positions()) + 0.002 * _positions()
    indices = _round_indices(len(values), start=101)

    base = compute_v2b_adaptive_checkpoint_evidence(indices, values)
    shifted = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        values + 13.0,
    )
    scaled = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        values * 7.0,
    )

    for transformed in (shifted, scaled):
        assert transformed.short_conservative_correlation_inflation \
            == pytest.approx(
                base.short_conservative_correlation_inflation,
                rel=1e-10,
                abs=1e-12,
            )
        assert transformed.long_conservative_correlation_inflation \
            == pytest.approx(
                base.long_conservative_correlation_inflation,
                rel=1e-10,
                abs=1e-12,
            )
        assert transformed.scale_ratio == pytest.approx(
            base.scale_ratio,
            rel=1e-10,
            abs=1e-12,
        )
        assert transformed.effective_round_count == pytest.approx(
            base.effective_round_count,
            rel=1e-10,
            abs=1e-12,
        )
        assert transformed.adaptive_numerically_estimable is (
            base.adaptive_numerically_estimable
        )
    assert shifted.mcse == pytest.approx(
        base.mcse,
        rel=1e-10,
        abs=1e-12,
    )
    assert scaled.mcse == pytest.approx(
        7.0 * base.mcse,
        rel=1e-10,
        abs=1e-12,
    )


def test_trend_is_not_mislabelled_as_convergence_or_stopping() -> None:
    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(256),
        np.linspace(0.0, 1.0, 256),
    )
    fields = set(asdict(result))

    assert result.stationarity_not_assessed is True
    assert result.reason == "multiscale_disagreement"
    assert FORBIDDEN_DECISION_FIELDS.isdisjoint(fields)


def test_finite_input_that_overflows_fails_closed() -> None:
    values = np.resize(
        np.asarray([1e308, -1e308], dtype=np.float64),
        256,
    )

    result = compute_v2b_adaptive_checkpoint_evidence(
        _round_indices(len(values)),
        values,
    )

    assert result.adaptive_numerically_estimable is False
    assert result.reason == "core_not_estimable"
    assert result.short_reason == "nonfinite_computation"
    assert result.long_reason == "nonfinite_computation"


@pytest.mark.parametrize(
    "round_indices,values",
    [
        (np.arange(0, 256), np.linspace(0.0, 1.0, 256)),
        (
            np.concatenate([np.arange(1, 128), np.arange(129, 258)]),
            np.linspace(0.0, 1.0, 256),
        ),
        (_round_indices(256), np.resize([False, True], 256)),
        (_round_indices(256), np.full(256, np.nan)),
        (_round_indices(256), np.ones((256, 1))),
    ],
)
def test_checkpoint_rejects_invalid_identity_or_value_inputs(
    round_indices,
    values,
) -> None:
    with pytest.raises(ValueError):
        compute_v2b_adaptive_checkpoint_evidence(round_indices, values)


def test_checkpoint_classifier_rejects_unregistered_round_count() -> None:
    indices = _round_indices(257)
    values = np.linspace(0.0, 1.0, 257)

    with pytest.raises(ValueError, match="pre-registered V2b checkpoint"):
        compute_v2b_adaptive_checkpoint_evidence(indices, values)

    scale_only = compute_v2_effective_round_evidence_for_batch(
        indices,
        values,
        batch_round_count=16,
    )
    assert scale_only.numerically_estimable is True


def test_checkpoint_inputs_are_not_mutated() -> None:
    round_indices = _round_indices(256)
    values = np.sin(0.11 * _positions())
    original_round_indices = round_indices.copy()
    original_values = values.copy()

    compute_v2b_adaptive_checkpoint_evidence(round_indices, values)

    np.testing.assert_array_equal(round_indices, original_round_indices)
    np.testing.assert_array_equal(values, original_values)


def test_first_ready_summary_distinguishes_2048_pass_from_cap_failure() -> None:
    first_at_cap = summarize_v2b_adaptive_checkpoint_decisions(
        [False] * 14 + [True]
    )
    never_ready = summarize_v2b_adaptive_checkpoint_decisions(
        [False] * 15
    )

    assert len(V2B_ADAPTIVE_CHECKPOINTS) == 15
    assert V2B_ADAPTIVE_CHECKPOINTS[-1] == V2B_RESOURCE_ROUND_CAP
    assert first_at_cap.first_adaptive_numerically_estimable_round == 2048
    assert first_at_cap.resource_round_count == 2048
    assert first_at_cap.adaptive_numerically_estimable is True
    assert first_at_cap.reason is None
    assert never_ready.first_adaptive_numerically_estimable_round is None
    assert never_ready.resource_round_count == 2048
    assert never_ready.adaptive_numerically_estimable is False
    assert never_ready.reason == (
        "resource_cap_without_multiscale_evidence"
    )


def test_first_ready_summary_uses_the_earliest_true_checkpoint() -> None:
    summary = summarize_v2b_adaptive_checkpoint_decisions(
        [False, False, True] + [True] * 12
    )

    assert summary.first_adaptive_numerically_estimable_round == 512
    assert summary.resource_round_count == 512
    assert FORBIDDEN_DECISION_FIELDS.isdisjoint(set(asdict(summary)))


@pytest.mark.parametrize(
    "decisions",
    [
        [False] * 14,
        [False] * 16,
        [False] * 14 + [1],
    ],
)
def test_first_ready_summary_rejects_invalid_decision_sequences(
    decisions,
) -> None:
    with pytest.raises(ValueError):
        summarize_v2b_adaptive_checkpoint_decisions(decisions)


def test_complete_trajectory_reports_all_checkpoints_and_first_ready() -> None:
    positions = _positions(V2B_RESOURCE_ROUND_CAP)
    result = compute_v2b_adaptive_trajectory_evidence(
        _round_indices(V2B_RESOURCE_ROUND_CAP),
        np.sin(0.5 * positions),
    )

    assert tuple(
        checkpoint.actual_round_count
        for checkpoint in result.checkpoint_evidence
    ) == V2B_ADAPTIVE_CHECKPOINTS
    assert result.first_adaptive_numerically_estimable_round == 256
    assert result.resource_round_count == 256
    assert result.adaptive_numerically_estimable is True
    assert result.reason is None
    assert result.stationarity_not_assessed is True
    assert FORBIDDEN_DECISION_FIELDS.isdisjoint(set(asdict(result)))


def test_complete_constant_trajectory_reaches_cap_without_evidence() -> None:
    result = compute_v2b_adaptive_trajectory_evidence(
        _round_indices(V2B_RESOURCE_ROUND_CAP),
        np.ones(V2B_RESOURCE_ROUND_CAP),
    )

    assert result.first_adaptive_numerically_estimable_round is None
    assert result.resource_round_count == V2B_RESOURCE_ROUND_CAP
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "resource_cap_without_multiscale_evidence"
    assert all(
        checkpoint.reason == "core_not_estimable"
        for checkpoint in result.checkpoint_evidence
    )


@pytest.mark.parametrize("count", [2047, 2049])
def test_complete_trajectory_requires_exact_resource_cap(count: int) -> None:
    with pytest.raises(ValueError, match="exactly 2048"):
        compute_v2b_adaptive_trajectory_evidence(
            _round_indices(count),
            np.linspace(0.0, 1.0, count),
        )
