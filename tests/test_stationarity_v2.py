from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from table_diffevo.stationarity import (
    StationarityTrace,
    build_stationarity_observation,
)
from table_diffevo.stationarity_v2 import (
    MAD_NORMAL_CONSISTENCY_FACTOR,
    STATIONARITY_V2_SCALAR_EVIDENCE_CONTRACT_VERSION,
    STATIONARITY_V2_SUBBLOCK_COLLECTION_CONTRACT_VERSION,
    STATIONARITY_V2_SUBBLOCK_SUMMARY_CONTRACT_VERSION,
    V2_CURRENT_SUBBLOCK_ROUND_CANDIDATE,
    V2_PAIRWISE_SLOPE_COUNT,
    V2_SCALAR_EVIDENCE_POINT_COUNT,
    collect_v2_subblock_summaries,
    compute_v2_scalar_evidence,
)


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _rng_hash(index: int) -> str:
    return hashlib.sha256(f"v2-rng-{index}".encode("utf-8")).hexdigest()


def _make_v2_trace(
    post_round_query_vectors: list[list[float]],
    *,
    initial_query_vector: list[float] | None = None,
    n_records: int = 10,
    changed_rows_per_round: list[int] | None = None,
) -> StationarityTrace:
    post_answers = np.asarray(post_round_query_vectors, dtype=np.float64)
    if post_answers.ndim != 2 or post_answers.shape[0] == 0:
        raise ValueError("post_round_query_vectors must be non-empty and 2D")
    query_count = post_answers.shape[1]
    if initial_query_vector is None:
        initial_answers = post_answers[0].copy()
    else:
        initial_answers = np.asarray(
            initial_query_vector, dtype=np.float64
        )
    if initial_answers.shape != (query_count,):
        raise ValueError("initial query vector has the wrong shape")
    if changed_rows_per_round is None:
        changed_rows_per_round = [n_records] * len(post_answers)
    if len(changed_rows_per_round) != len(post_answers):
        raise ValueError("changed row counts have the wrong length")

    target = np.zeros(query_count, dtype=np.float64)
    initial_frame = pd.DataFrame({"x": np.arange(n_records) % 2})
    observations = [
        build_stationarity_observation(
            frame=initial_frame,
            target=target,
            current_query_answers=initial_answers,
            n_records=n_records,
            squared_loss=float(
                0.5 * np.dot(initial_answers, initial_answers)
            ),
            state_index=0,
            round_index=0,
            phase="initial",
            proposal_attempt_count=0,
            proposal_accepted=False,
            applied_attempt_index=0,
            attempted_participating_row_count=0,
            applied_participating_row_count=0,
            actual_changed_row_count=0,
            actual_changed_cell_count=0,
            actual_changed_query_count=0,
            normalized_query_l1_movement_mean=0.0,
            gibbs_microstep_count_attempted=0,
            gibbs_microstep_count_applied=0,
            candidate_evaluation_count_cumulative=0,
            current_table_sha256=_frame_hash(initial_frame),
            primary_rng_state_sha256=_rng_hash(0),
            factorized_gibbs_rng_state_sha256=None,
        )
    ]
    all_answers = [initial_answers]
    previous_frame = initial_frame
    previous_answers = initial_answers

    for round_index, (answers, changed_row_count) in enumerate(
        zip(post_answers, changed_rows_per_round), start=1
    ):
        if not 0 <= changed_row_count <= n_records:
            raise ValueError("changed row count is outside the table")
        current_frame = previous_frame.copy()
        if changed_row_count:
            current_frame.iloc[:changed_row_count, 0] = (
                1 - current_frame.iloc[:changed_row_count, 0]
            )
        changed_cells = current_frame.ne(previous_frame).to_numpy(dtype=bool)
        delta_query = answers - previous_answers
        observations.append(
            build_stationarity_observation(
                frame=current_frame,
                target=target,
                current_query_answers=answers,
                n_records=n_records,
                squared_loss=float(0.5 * np.dot(answers, answers)),
                state_index=round_index,
                round_index=round_index,
                phase="post_round",
                proposal_attempt_count=1,
                proposal_accepted=True,
                applied_attempt_index=1,
                attempted_participating_row_count=n_records,
                applied_participating_row_count=n_records,
                actual_changed_row_count=int(
                    np.any(changed_cells, axis=1).sum()
                ),
                actual_changed_cell_count=int(changed_cells.sum()),
                actual_changed_query_count=int(np.count_nonzero(delta_query)),
                normalized_query_l1_movement_mean=float(
                    np.mean(np.abs(delta_query)) / n_records
                ),
                gibbs_microstep_count_attempted=0,
                gibbs_microstep_count_applied=0,
                candidate_evaluation_count_cumulative=round_index,
                current_table_sha256=_frame_hash(current_frame),
                primary_rng_state_sha256=_rng_hash(round_index),
                factorized_gibbs_rng_state_sha256=None,
            )
        )
        all_answers.append(answers.copy())
        previous_frame = current_frame
        previous_answers = answers

    return StationarityTrace(
        n_records=n_records,
        query_identity_sha256="3" * 64,
        target_identity_sha256="4" * 64,
        observations=observations,
        measured_query_answers=np.stack(all_answers),
        termination_reason="max_rounds",
    )


def test_subblocks_use_only_complete_post_round_groups() -> None:
    post_answers = [
        [1.0 + 0.01 * index, 2.0 + 0.02 * index]
        for index in range(1, 206)
    ]
    trace = _make_v2_trace(
        post_answers,
        initial_query_vector=[9.0, 9.0],
        changed_rows_per_round=[2] * len(post_answers),
    )

    collection = collect_v2_subblock_summaries(
        trace,
        subblock_round_count=V2_CURRENT_SUBBLOCK_ROUND_CANDIDATE,
    )

    assert collection.contract_version == (
        STATIONARITY_V2_SUBBLOCK_COLLECTION_CONTRACT_VERSION
    )
    assert collection.trace_contract_version == trace.contract_version
    assert collection.query_identity_sha256 == trace.query_identity_sha256
    assert collection.target_identity_sha256 == trace.target_identity_sha256
    assert collection.n_records == 10
    assert collection.query_count == 2
    assert collection.post_round_count == 205
    assert collection.subblock_round_count == 100
    assert collection.complete_subblock_count == 2
    assert collection.trailing_post_round_count == 5

    first = collection.subblocks[0]
    assert first.contract_version == (
        STATIONARITY_V2_SUBBLOCK_SUMMARY_CONTRACT_VERSION
    )
    assert first.subblock_number == 1
    assert first.start_round_index == 1
    assert first.end_round_index == 100
    assert first.round_count == 100
    assert first.normalized_query_mean == pytest.approx(
        np.mean(np.asarray(post_answers[:100]), axis=0) / trace.n_records
    )

    first_rows = trace.observations[1:101]
    first_l1 = np.asarray([
        row["current_normalized_l1"] for row in first_rows
    ])
    assert first.l1_mean == pytest.approx(np.mean(first_l1))
    assert first.l1_p90_minus_p10 == pytest.approx(
        np.percentile(first_l1, 90, method="linear")
        - np.percentile(first_l1, 10, method="linear")
    )
    assert first.unique_row_rate_mean == pytest.approx(
        np.mean([row["unique_row_rate"] for row in first_rows])
    )
    assert first.normalized_row_entropy_mean == pytest.approx(
        np.mean([row["normalized_row_entropy"] for row in first_rows])
    )
    assert first.active_round_rate == 1.0
    assert first.mean_changed_row_fraction == pytest.approx(0.2)
    assert first.mean_changed_query_fraction == 1.0
    assert first.mean_normalized_query_l1_movement == pytest.approx(
        np.mean([
            row["normalized_query_l1_movement_mean"] for row in first_rows
        ])
    )

    second = collection.subblocks[1]
    assert second.subblock_number == 2
    assert second.start_round_index == 101
    assert second.end_round_index == 200
    assert second.round_count == 100


def test_subblock_duration_is_explicit_and_not_a_dataset_mapping() -> None:
    post_answers = [[1.0, 2.0]] * 205
    trace = _make_v2_trace(
        post_answers,
        changed_rows_per_round=[0] * len(post_answers),
    )

    fifty = collect_v2_subblock_summaries(
        trace,
        subblock_round_count=50,
    )
    one_hundred = collect_v2_subblock_summaries(
        trace,
        subblock_round_count=100,
    )

    assert V2_CURRENT_SUBBLOCK_ROUND_CANDIDATE == 100
    assert fifty.subblock_round_count == 50
    assert fifty.complete_subblock_count == 4
    assert fifty.trailing_post_round_count == 5
    assert one_hundred.subblock_round_count == 100
    assert one_hundred.complete_subblock_count == 2
    assert one_hundred.trailing_post_round_count == 5
    assert fifty.query_identity_sha256 == one_hundred.query_identity_sha256


def test_incomplete_tail_is_reported_but_never_summarized() -> None:
    trace = _make_v2_trace(
        [[1.0, 2.0]] * 99,
        changed_rows_per_round=[0] * 99,
    )

    collection = collect_v2_subblock_summaries(
        trace,
        subblock_round_count=100,
    )

    assert collection.complete_subblock_count == 0
    assert collection.subblocks == ()
    assert collection.trailing_post_round_count == 99


def test_subblock_movement_uses_applied_state_changes() -> None:
    current = np.asarray([1.0, 2.0])
    post_answers = []
    changed_rows = []
    for index in range(100):
        if index % 2 == 0:
            current = current + np.asarray([0.01, 0.02])
            changed_rows.append(2)
        else:
            changed_rows.append(0)
        post_answers.append(current.tolist())
    trace = _make_v2_trace(
        post_answers,
        initial_query_vector=[1.0, 2.0],
        changed_rows_per_round=changed_rows,
    )

    summary = collect_v2_subblock_summaries(
        trace,
        subblock_round_count=100,
    ).subblocks[0]

    assert summary.active_round_rate == pytest.approx(0.5)
    assert summary.mean_changed_row_fraction == pytest.approx(0.1)
    assert summary.mean_changed_query_fraction == pytest.approx(0.5)
    assert summary.mean_normalized_query_l1_movement == pytest.approx(
        0.00075
    )
    assert not hasattr(summary, "stable")
    assert not hasattr(summary, "converged")
    assert not hasattr(summary, "stalled")


@pytest.mark.parametrize("value", [True, 1, 0, -1, 2.5, "100"])
def test_invalid_subblock_duration_is_rejected(value: object) -> None:
    trace = _make_v2_trace(
        [[1.0, 2.0]] * 2,
        changed_rows_per_round=[0, 0],
    )

    with pytest.raises(ValueError, match="subblock_round_count"):
        collect_v2_subblock_summaries(
            trace,
            subblock_round_count=value,  # type: ignore[arg-type]
        )


def test_subblock_collector_requires_the_existing_trace_contract() -> None:
    with pytest.raises(ValueError, match="StationarityTrace"):
        collect_v2_subblock_summaries(
            object(),  # type: ignore[arg-type]
            subblock_round_count=100,
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
