"""Deterministic tests for Issue #53 V2c three-scale evidence."""

from dataclasses import asdict, replace
from math import isqrt, sqrt

import numpy as np
import pytest

import table_diffevo.adaptive_effective_evidence_v2c as v2c
from table_diffevo.adaptive_effective_evidence import (
    V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION,
    compute_v2b_adaptive_trajectory_evidence,
)
from table_diffevo.adaptive_effective_evidence_v2c import (
    V2C_ADAPTIVE_CHECKPOINTS,
    V2C_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION,
    V2C_RESOURCE_ROUND_CAP,
    V2C_SCALE_RATIO_LIMIT,
    compute_v2c_adaptive_checkpoint_evidence,
    compute_v2c_adaptive_trajectory_evidence,
    compute_v2c_scale_ratio,
    summarize_v2c_adaptive_checkpoint_compatibilities,
    v2c_scale_ratio_is_acceptable,
)
from table_diffevo.effective_evidence import (
    V2EffectiveRoundEvidence,
    compute_v2_effective_round_evidence_for_batch,
)


FORBIDDEN_DECISION_FIELDS = {
    "confirmed",
    "converged",
    "qualified",
    "quality_pass",
    "stable",
    "stop",
    "stop_round",
    "threshold",
}


def _round_indices(count: int, *, start: int = 1) -> np.ndarray:
    return np.arange(start, start + count, dtype=np.int64)


def _positions(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.float64)


def _synthetic_scale(
    *,
    batch_round_count: int,
    conservative_inflation: float,
    single_round_variance: float = 2.0,
) -> V2EffectiveRoundEvidence:
    actual_round_count = 256
    raw_inflation = conservative_inflation
    long_run_variance = single_round_variance * raw_inflation
    return V2EffectiveRoundEvidence(
        actual_round_count=actual_round_count,
        batch_round_count=batch_round_count,
        overlapping_batch_count=(
            actual_round_count - batch_round_count + 1
        ),
        single_round_variance=single_round_variance,
        long_run_variance=long_run_variance,
        raw_correlation_inflation=raw_inflation,
        conservative_correlation_inflation=conservative_inflation,
        raw_effective_round_count=(actual_round_count / raw_inflation),
        effective_round_count=(
            actual_round_count / conservative_inflation
        ),
        mcse=sqrt(
            single_round_variance
            * conservative_inflation
            / actual_round_count
        ),
        numerically_estimable=True,
        reason=None,
    )


def _synthetic_scales(
    inflations: tuple[float, float, float],
) -> tuple[V2EffectiveRoundEvidence, ...]:
    return tuple(
        _synthetic_scale(
            batch_round_count=batch_round_count,
            conservative_inflation=inflation,
        )
        for batch_round_count, inflation in zip((16, 32, 64), inflations)
    )


def _collect_mapping_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(
            *(_collect_mapping_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, (list, tuple)):
        return set().union(
            *(_collect_mapping_keys(item) for item in value),
            set(),
        )
    return set()


def test_three_explicit_obm_lengths_match_hand_calculation() -> None:
    indices = [1, 2, 3, 4, 5]
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    results = tuple(
        compute_v2_effective_round_evidence_for_batch(
            indices,
            values,
            batch_round_count=batch_round_count,
        )
        for batch_round_count in (1, 2, 3)
    )

    assert tuple(result.single_round_variance for result in results) \
        == pytest.approx((2.5, 2.5, 2.5))
    assert tuple(result.long_run_variance for result in results) \
        == pytest.approx((2.5, 25.0 / 6.0, 5.0))


def test_all_fifteen_checkpoints_have_the_frozen_batch_table() -> None:
    expected = (
        (256, 16, 32, 64),
        (384, 19, 38, 76),
        (512, 22, 44, 88),
        (640, 25, 50, 100),
        (768, 27, 54, 108),
        (896, 29, 58, 116),
        (1024, 32, 64, 128),
        (1152, 33, 66, 132),
        (1280, 35, 70, 140),
        (1408, 37, 74, 148),
        (1536, 39, 78, 156),
        (1664, 40, 80, 160),
        (1792, 42, 84, 168),
        (1920, 43, 86, 172),
        (2048, 45, 90, 180),
    )

    actual = tuple(
        (checkpoint, isqrt(checkpoint), 2 * isqrt(checkpoint),
         4 * isqrt(checkpoint))
        for checkpoint in V2C_ADAPTIVE_CHECKPOINTS
    )
    assert actual == expected


def test_checkpoint_matches_three_independent_v2_scales_and_identities() -> None:
    checkpoint = 512
    positions = _positions(checkpoint)
    values = np.sin(0.07 * positions) + 0.002 * positions
    indices = _round_indices(checkpoint)

    result = compute_v2c_adaptive_checkpoint_evidence(
        indices,
        values,
        previous_three_scale_compatible=False,
    )
    independent = tuple(
        compute_v2_effective_round_evidence_for_batch(
            indices,
            values,
            batch_round_count=batch_round_count,
        )
        for batch_round_count in (22, 44, 88)
    )

    assert all(scale.numerically_estimable for scale in independent)
    assert (
        result.b1_long_run_variance,
        result.b2_long_run_variance,
        result.b3_long_run_variance,
    ) == tuple(scale.long_run_variance for scale in independent)
    assert (
        result.b1_raw_correlation_inflation,
        result.b2_raw_correlation_inflation,
        result.b3_raw_correlation_inflation,
    ) == tuple(scale.raw_correlation_inflation for scale in independent)
    formal_inflations = tuple(
        scale.conservative_correlation_inflation for scale in independent
    )
    assert (
        result.b1_conservative_correlation_inflation,
        result.b2_conservative_correlation_inflation,
        result.b3_conservative_correlation_inflation,
    ) == formal_inflations
    assert result.scale_ratio == pytest.approx(
        max(formal_inflations) / min(formal_inflations)
    )
    assert result.official_correlation_inflation == max(formal_inflations)
    assert result.single_round_variance is not None
    assert result.official_long_run_variance == pytest.approx(
        result.single_round_variance
        * result.official_correlation_inflation
    )
    assert result.effective_round_count == pytest.approx(
        checkpoint / result.official_correlation_inflation
    )
    assert result.mcse == pytest.approx(
        sqrt(result.official_long_run_variance / checkpoint)
    )


def test_three_scale_ratio_has_exact_inclusive_boundary() -> None:
    just_above = np.nextafter(V2C_SCALE_RATIO_LIMIT, np.inf)

    assert compute_v2c_scale_ratio(1.0, 1.25, 1.1) == 1.25
    assert v2c_scale_ratio_is_acceptable(1.0, 1.25, 1.1) is True
    assert v2c_scale_ratio_is_acceptable(1.25, 1.1, 1.0) is True
    assert v2c_scale_ratio_is_acceptable(1.1, 1.0, 1.25) is True
    assert v2c_scale_ratio_is_acceptable(1.0, just_above, 1.1) is False


@pytest.mark.parametrize(
    "inflations",
    [
        (True, 1.0, 1.0),
        (1.0, False, 1.0),
        (1.0, 1.0, True),
        (0.99, 1.0, 1.0),
        (1.0, np.nan, 1.0),
        (1.0, 1.0, np.inf),
        ("1", 1.0, 1.0),
    ],
)
def test_three_scale_ratio_rejects_invalid_inputs(inflations) -> None:
    with pytest.raises(ValueError):
        compute_v2c_scale_ratio(*inflations)


def test_first_checkpoint_can_only_await_a_second_compatible_checkpoint() -> None:
    values = np.sin(0.5 * _positions(256))
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        values,
    )

    assert result.actual_round_count == 256
    assert (
        result.b1_batch_round_count,
        result.b2_batch_round_count,
        result.b3_batch_round_count,
    ) == (16, 32, 64)
    assert result.three_scale_compatible is True
    assert result.previous_three_scale_compatible is False
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "awaiting_consecutive_multiscale_evidence"
    assert result.stationarity_not_assessed is True
    assert result.contract_version == (
        V2C_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
    )


def test_later_compatible_checkpoint_uses_only_immediate_previous_state() -> None:
    values = np.sin(0.5 * _positions(384))

    after_true = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(384),
        values,
        previous_three_scale_compatible=True,
    )
    after_false = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(384),
        values,
        previous_three_scale_compatible=False,
    )

    assert after_true.three_scale_compatible is True
    assert after_true.adaptive_numerically_estimable is True
    assert after_true.reason is None
    assert after_false.three_scale_compatible is True
    assert after_false.adaptive_numerically_estimable is False
    assert after_false.reason == "awaiting_consecutive_multiscale_evidence"


@pytest.mark.parametrize("largest_index", [0, 1, 2])
def test_official_uncertainty_uses_whichever_scale_is_largest(
    largest_index: int,
) -> None:
    inflations = [1.0, 1.0, 1.0]
    inflations[largest_index] = 1.2
    scales = _synthetic_scales(tuple(inflations))

    result = v2c._combine_v2c_scales(
        b1=scales[0],
        b2=scales[1],
        b3=scales[2],
        previous_three_scale_compatible=True,
    )

    assert result.three_scale_compatible is True
    assert result.adaptive_numerically_estimable is True
    assert result.official_correlation_inflation == 1.2
    assert result.official_long_run_variance == pytest.approx(2.4)
    assert result.effective_round_count == pytest.approx(256 / 1.2)
    assert result.mcse == pytest.approx(sqrt(2.4 / 256))


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_any_nonestimable_scale_makes_checkpoint_fail_closed(
    failed_index: int,
) -> None:
    scales = list(_synthetic_scales((1.0, 1.1, 1.2)))
    scales[failed_index] = replace(
        scales[failed_index],
        long_run_variance=0.0,
        raw_correlation_inflation=None,
        conservative_correlation_inflation=None,
        raw_effective_round_count=None,
        effective_round_count=None,
        mcse=None,
        numerically_estimable=False,
        reason="degenerate_long_run_variance",
    )

    result = v2c._combine_v2c_scales(
        b1=scales[0],
        b2=scales[1],
        b3=scales[2],
        previous_three_scale_compatible=True,
    )

    assert result.three_scale_compatible is False
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "core_not_estimable"
    assert result.scale_ratio is None
    assert result.official_correlation_inflation is None
    assert result.effective_round_count is None
    assert result.mcse is None


def test_nonfinite_three_scale_combination_fails_closed() -> None:
    scales = tuple(
        replace(
            scale,
            single_round_variance=1e308,
            long_run_variance=1e308,
        )
        for scale in _synthetic_scales((1.0, 1.0, 2.0))
    )

    result = v2c._combine_v2c_scales(
        b1=scales[0],
        b2=scales[1],
        b3=scales[2],
        previous_three_scale_compatible=True,
    )

    assert all(
        (
            result.b1_numerically_estimable,
            result.b2_numerically_estimable,
            result.b3_numerically_estimable,
        )
    )
    assert result.three_scale_compatible is False
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "nonfinite_computation"
    assert result.scale_ratio is None
    assert result.official_long_run_variance is None


def test_multiscale_disagreement_keeps_finite_official_diagnostics() -> None:
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        np.linspace(0.0, 1.0, 256),
    )

    assert result.scale_ratio is not None
    assert result.scale_ratio > V2C_SCALE_RATIO_LIMIT
    assert result.three_scale_compatible is False
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "multiscale_disagreement"
    assert result.official_correlation_inflation is not None
    assert result.official_long_run_variance is not None
    assert result.effective_round_count is not None
    assert result.mcse is not None


def test_negative_correlation_floor_applies_to_all_three_scales() -> None:
    values = np.sin(0.5 * _positions(256))
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        values,
    )

    raw = (
        result.b1_raw_correlation_inflation,
        result.b2_raw_correlation_inflation,
        result.b3_raw_correlation_inflation,
    )
    conservative = (
        result.b1_conservative_correlation_inflation,
        result.b2_conservative_correlation_inflation,
        result.b3_conservative_correlation_inflation,
    )
    assert all(value is not None and value < 1.0 for value in raw)
    assert conservative == (1.0, 1.0, 1.0)
    assert result.effective_round_count == pytest.approx(256.0)
    assert result.single_round_variance is not None
    assert result.mcse == pytest.approx(
        sqrt(result.single_round_variance / 256)
    )


def test_shift_and_positive_scale_preserve_three_scale_diagnostics() -> None:
    values = np.sin(0.5 * _positions(384))
    indices = _round_indices(384, start=101)

    base = compute_v2c_adaptive_checkpoint_evidence(
        indices,
        values,
        previous_three_scale_compatible=True,
    )
    shifted = compute_v2c_adaptive_checkpoint_evidence(
        indices,
        values + 13.0,
        previous_three_scale_compatible=True,
    )
    scaled = compute_v2c_adaptive_checkpoint_evidence(
        indices,
        values * 7.0,
        previous_three_scale_compatible=True,
    )

    for transformed in (shifted, scaled):
        for field_name in (
            "b1_conservative_correlation_inflation",
            "b2_conservative_correlation_inflation",
            "b3_conservative_correlation_inflation",
            "scale_ratio",
            "effective_round_count",
        ):
            assert getattr(transformed, field_name) == pytest.approx(
                getattr(base, field_name),
                rel=1e-10,
                abs=1e-12,
            )
        assert transformed.three_scale_compatible is (
            base.three_scale_compatible
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


def test_constant_sequence_makes_all_scales_fail_closed() -> None:
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        np.ones(256),
    )

    assert not any((
        result.b1_numerically_estimable,
        result.b2_numerically_estimable,
        result.b3_numerically_estimable,
    ))
    assert (result.b1_reason, result.b2_reason, result.b3_reason) == (
        "zero_round_variance",
        "zero_round_variance",
        "zero_round_variance",
    )
    assert result.three_scale_compatible is False
    assert result.reason == "core_not_estimable"


def test_periodic_batch_coupling_makes_whole_checkpoint_fail_closed() -> None:
    values = np.where(
        (np.arange(256) // 16) % 2 == 0,
        1.0,
        -1.0,
    )
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        values,
    )

    assert not all((
        result.b1_numerically_estimable,
        result.b2_numerically_estimable,
        result.b3_numerically_estimable,
    ))
    assert "degenerate_long_run_variance" in {
        result.b1_reason,
        result.b2_reason,
        result.b3_reason,
    }
    assert result.three_scale_compatible is False
    assert result.reason == "core_not_estimable"


def test_single_spike_stays_finite_and_ess_cannot_exceed_rounds() -> None:
    values = np.zeros(256, dtype=np.float64)
    values[len(values) // 2] = 1.0

    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        values,
    )

    assert all((
        result.b1_numerically_estimable,
        result.b2_numerically_estimable,
        result.b3_numerically_estimable,
    ))
    assert result.effective_round_count is not None
    assert 0.0 < result.effective_round_count <= 256
    assert np.all(np.isfinite([
        result.b1_long_run_variance,
        result.b2_long_run_variance,
        result.b3_long_run_variance,
        result.scale_ratio,
        result.official_correlation_inflation,
        result.official_long_run_variance,
        result.effective_round_count,
        result.mcse,
    ]))


def test_trend_has_no_stationarity_convergence_quality_or_stop_fields() -> None:
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        np.linspace(0.0, 1.0, 256),
    )

    assert result.stationarity_not_assessed is True
    assert FORBIDDEN_DECISION_FIELDS.isdisjoint(
        _collect_mapping_keys(asdict(result))
    )


def test_finite_huge_values_fail_closed_with_scale_reasons() -> None:
    values = np.resize(
        np.asarray([1e308, -1e308], dtype=np.float64),
        256,
    )
    result = compute_v2c_adaptive_checkpoint_evidence(
        _round_indices(256),
        values,
    )

    assert result.three_scale_compatible is False
    assert result.adaptive_numerically_estimable is False
    assert result.reason == "core_not_estimable"
    assert {result.b1_reason, result.b2_reason, result.b3_reason} == {
        "nonfinite_computation"
    }


@pytest.mark.parametrize(
    "round_indices,values",
    [
        (np.arange(0, 256), np.linspace(0.0, 1.0, 256)),
        (
            np.concatenate((np.arange(1, 101), np.arange(102, 258))),
            np.linspace(0.0, 1.0, 256),
        ),
        (
            np.concatenate((np.arange(1, 101), [100], np.arange(102, 257))),
            np.linspace(0.0, 1.0, 256),
        ),
        (
            np.concatenate((np.arange(1, 101), [102, 101],
                            np.arange(103, 257))),
            np.linspace(0.0, 1.0, 256),
        ),
        (np.resize([False, True], 256), np.linspace(0.0, 1.0, 256)),
        (_round_indices(256), np.resize([False, True], 256)),
        (_round_indices(256), np.full(256, np.nan)),
        (_round_indices(256), np.full(256, np.inf)),
        (_round_indices(256), np.ones((256, 1))),
    ],
)
def test_checkpoint_rejects_invalid_identity_or_value_inputs(
    round_indices,
    values,
) -> None:
    with pytest.raises(ValueError):
        compute_v2c_adaptive_checkpoint_evidence(round_indices, values)


def test_checkpoint_rejects_unregistered_round_count_but_v2_stays_general() -> None:
    indices = _round_indices(257)
    values = np.linspace(0.0, 1.0, 257)

    with pytest.raises(ValueError, match="pre-registered V2c checkpoint"):
        compute_v2c_adaptive_checkpoint_evidence(indices, values)

    scale_only = compute_v2_effective_round_evidence_for_batch(
        indices,
        values,
        batch_round_count=16,
    )
    assert scale_only.numerically_estimable is True


def test_checkpoint_requires_exact_previous_state_contract() -> None:
    values_256 = np.sin(0.5 * _positions(256))
    values_384 = np.sin(0.5 * _positions(384))

    with pytest.raises(ValueError, match="has no previous"):
        compute_v2c_adaptive_checkpoint_evidence(
            _round_indices(256),
            values_256,
            previous_three_scale_compatible=False,
        )
    with pytest.raises(ValueError, match="require previous"):
        compute_v2c_adaptive_checkpoint_evidence(
            _round_indices(384),
            values_384,
        )
    with pytest.raises(ValueError, match="require previous"):
        compute_v2c_adaptive_checkpoint_evidence(
            _round_indices(384),
            values_384,
            previous_three_scale_compatible=1,
        )


def test_checkpoint_inputs_are_not_mutated() -> None:
    round_indices = _round_indices(384)
    values = np.sin(0.5 * _positions(384))
    original_round_indices = round_indices.copy()
    original_values = values.copy()

    compute_v2c_adaptive_checkpoint_evidence(
        round_indices,
        values,
        previous_three_scale_compatible=True,
    )

    np.testing.assert_array_equal(round_indices, original_round_indices)
    np.testing.assert_array_equal(values, original_values)


def test_tt_first_becomes_ready_at_second_checkpoint() -> None:
    summary = summarize_v2c_adaptive_checkpoint_compatibilities(
        [True, True] + [False] * 13
    )

    assert summary.checkpoint_adaptive_numerically_estimable[:2] == (
        False,
        True,
    )
    assert summary.first_adaptive_numerically_estimable_round == 384
    assert summary.resource_round_count == 384
    assert summary.no_ready_reason is None


def test_tft_never_becomes_ready() -> None:
    summary = summarize_v2c_adaptive_checkpoint_compatibilities(
        [True, False, True] + [False] * 12
    )

    assert not any(summary.checkpoint_adaptive_numerically_estimable)
    assert summary.first_adaptive_numerically_estimable_round is None
    assert summary.resource_round_count == V2C_RESOURCE_ROUND_CAP
    assert summary.no_ready_reason == (
        "resource_cap_without_consecutive_multiscale_evidence"
    )


def test_tftt_first_becomes_ready_at_fourth_checkpoint() -> None:
    summary = summarize_v2c_adaptive_checkpoint_compatibilities(
        [True, False, True, True] + [False] * 11
    )

    assert summary.checkpoint_adaptive_numerically_estimable[:4] == (
        False,
        False,
        False,
        True,
    )
    assert summary.first_adaptive_numerically_estimable_round == 640
    assert summary.resource_round_count == 640


def test_first_ready_at_2048_is_distinct_from_cap_without_evidence() -> None:
    first_at_cap = summarize_v2c_adaptive_checkpoint_compatibilities(
        [False] * 13 + [True, True]
    )
    isolated_final_true = summarize_v2c_adaptive_checkpoint_compatibilities(
        [False] * 14 + [True]
    )

    assert len(V2C_ADAPTIVE_CHECKPOINTS) == 15
    assert V2C_ADAPTIVE_CHECKPOINTS[-1] == V2C_RESOURCE_ROUND_CAP
    assert first_at_cap.first_adaptive_numerically_estimable_round == 2048
    assert first_at_cap.resource_round_count == 2048
    assert first_at_cap.adaptive_numerically_estimable is True
    assert first_at_cap.no_ready_reason is None
    assert isolated_final_true.first_adaptive_numerically_estimable_round \
        is None
    assert isolated_final_true.resource_round_count == 2048
    assert isolated_final_true.current_three_scale_compatible is True
    assert isolated_final_true.adaptive_numerically_estimable is False
    assert isolated_final_true.no_ready_reason == (
        "resource_cap_without_consecutive_multiscale_evidence"
    )


def test_current_state_revokes_but_first_ready_history_does_not() -> None:
    summary = summarize_v2c_adaptive_checkpoint_compatibilities(
        [True, True, False] + [False] * 12
    )

    assert summary.first_adaptive_numerically_estimable_round == 384
    assert summary.resource_round_count == 384
    assert summary.current_three_scale_compatible is False
    assert summary.adaptive_numerically_estimable is False
    assert summary.post_first_has_three_scale_incompatibility is True
    assert (
        summary.post_first_three_scale_incompatible_checkpoint_count
        == 13
    )
    assert summary.no_ready_reason is None


@pytest.mark.parametrize(
    "compatibilities",
    [
        [False] * 14,
        [False] * 16,
        [False] * 14 + [1],
    ],
)
def test_summary_rejects_invalid_compatibility_sequences(
    compatibilities,
) -> None:
    with pytest.raises(ValueError):
        summarize_v2c_adaptive_checkpoint_compatibilities(compatibilities)


def test_complete_trajectory_reports_all_checkpoints_and_first_ready() -> None:
    positions = _positions(V2C_RESOURCE_ROUND_CAP)
    result = compute_v2c_adaptive_trajectory_evidence(
        _round_indices(V2C_RESOURCE_ROUND_CAP),
        np.sin(0.5 * positions),
    )

    assert tuple(
        checkpoint.actual_round_count
        for checkpoint in result.checkpoint_evidence
    ) == V2C_ADAPTIVE_CHECKPOINTS
    assert all(
        checkpoint.three_scale_compatible
        for checkpoint in result.checkpoint_evidence
    )
    assert result.checkpoint_evidence[0].adaptive_numerically_estimable \
        is False
    assert all(
        checkpoint.adaptive_numerically_estimable
        for checkpoint in result.checkpoint_evidence[1:]
    )
    assert result.first_adaptive_numerically_estimable_round == 384
    assert result.resource_round_count == 384
    assert result.current_three_scale_compatible is True
    assert result.adaptive_numerically_estimable is True
    assert result.reason is None
    assert result.stationarity_not_assessed is True
    assert FORBIDDEN_DECISION_FIELDS.isdisjoint(
        _collect_mapping_keys(asdict(result))
    )


def test_complete_constant_trajectory_reaches_cap_without_evidence() -> None:
    result = compute_v2c_adaptive_trajectory_evidence(
        _round_indices(V2C_RESOURCE_ROUND_CAP),
        np.ones(V2C_RESOURCE_ROUND_CAP),
    )

    assert result.first_adaptive_numerically_estimable_round is None
    assert result.resource_round_count == V2C_RESOURCE_ROUND_CAP
    assert result.current_three_scale_compatible is False
    assert result.adaptive_numerically_estimable is False
    assert result.reason == (
        "resource_cap_without_consecutive_multiscale_evidence"
    )
    assert all(
        checkpoint.reason == "core_not_estimable"
        for checkpoint in result.checkpoint_evidence
    )


@pytest.mark.parametrize("count", [2047, 2049])
def test_complete_trajectory_requires_exact_resource_cap(count: int) -> None:
    with pytest.raises(ValueError, match="exactly 2048"):
        compute_v2c_adaptive_trajectory_evidence(
            _round_indices(count),
            np.linspace(0.0, 1.0, count),
        )


def test_v2b_public_behavior_and_contract_remain_unchanged() -> None:
    positions = _positions(2048)
    result = compute_v2b_adaptive_trajectory_evidence(
        _round_indices(2048),
        np.sin(0.5 * positions),
    )

    assert V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION == (
        "issue53-v2b-adaptive-effective-round-evidence-research-v1"
    )
    assert result.first_adaptive_numerically_estimable_round == 256
    assert result.resource_round_count == 256
