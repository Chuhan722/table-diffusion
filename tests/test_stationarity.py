import hashlib
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from table_diffevo.stationarity import (
    STATIONARITY_REPLAY_CONTRACT_VERSION,
    STATIONARITY_TRACE_CONTRACT_VERSION,
    StationarityDetectorConfig,
    StationarityTrace,
    build_stationarity_observation,
    load_stationarity_trace,
    ordered_query_identity_sha256,
    replay_stationarity,
    save_stationarity_trace,
    stationarity_row_diversity_metrics,
    target_answer_identity_sha256,
)


def _frame_hash(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _rng_hash(index):
    return hashlib.sha256(f"rng-{index}".encode("utf-8")).hexdigest()


def _make_trace(
    query_vectors,
    *,
    moving,
    termination_reason="max_rounds",
    n_records=4,
    changed_rows_per_round=None,
):
    target = np.zeros(len(query_vectors[0]), dtype=float)
    frame_a = pd.DataFrame({"x": np.arange(n_records) % 2})
    if changed_rows_per_round is None:
        changed_rows_per_round = [
            n_records if moving else 0
        ] * len(query_vectors)
    if len(changed_rows_per_round) != len(query_vectors):
        raise ValueError("changed_rows_per_round 长度必须与 query_vectors 相同")
    initial_q = np.asarray(query_vectors[0], dtype=float)
    observations = [
        build_stationarity_observation(
            frame=frame_a,
            target=target,
            current_query_answers=initial_q,
            n_records=n_records,
            squared_loss=float(0.5 * np.dot(initial_q, initial_q)),
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
            current_table_sha256=_frame_hash(frame_a),
            primary_rng_state_sha256=_rng_hash(0),
            factorized_gibbs_rng_state_sha256=None,
        )
    ]
    answers = [initial_q]
    previous_frame = frame_a
    previous_q = initial_q
    for index, (values, changed_row_count) in enumerate(
        zip(query_vectors, changed_rows_per_round), start=1
    ):
        current_q = np.asarray(values, dtype=float)
        current_frame = previous_frame.copy()
        if not 0 <= changed_row_count <= n_records:
            raise ValueError("changed row count 超出表范围")
        if changed_row_count:
            current_frame.iloc[:changed_row_count, 0] = (
                1 - current_frame.iloc[:changed_row_count, 0]
            )
        changed = current_frame.ne(previous_frame).to_numpy(dtype=bool)
        delta_q = current_q - previous_q
        observations.append(
            build_stationarity_observation(
                frame=current_frame,
                target=target,
                current_query_answers=current_q,
                n_records=n_records,
                squared_loss=float(0.5 * np.dot(current_q, current_q)),
                state_index=index,
                round_index=index,
                phase="post_round",
                proposal_attempt_count=1,
                proposal_accepted=True,
                applied_attempt_index=1,
                attempted_participating_row_count=n_records,
                applied_participating_row_count=n_records,
                actual_changed_row_count=int(
                    np.any(changed, axis=1).sum()
                ),
                actual_changed_cell_count=int(changed.sum()),
                actual_changed_query_count=int(np.count_nonzero(delta_q)),
                normalized_query_l1_movement_mean=float(
                    np.mean(np.abs(delta_q)) / n_records
                ),
                gibbs_microstep_count_attempted=0,
                gibbs_microstep_count_applied=0,
                candidate_evaluation_count_cumulative=index,
                current_table_sha256=_frame_hash(current_frame),
                primary_rng_state_sha256=_rng_hash(index),
                factorized_gibbs_rng_state_sha256=None,
            )
        )
        answers.append(current_q)
        previous_frame = current_frame
        previous_q = current_q
    return StationarityTrace(
        n_records=n_records,
        query_identity_sha256="1" * 64,
        target_identity_sha256="2" * 64,
        observations=observations,
        measured_query_answers=np.stack(answers),
        termination_reason=termination_reason,
    )


def _config(**changes):
    values = {
        "window_size": 2,
        "query_mean_shift_tolerance": 1e-12,
        "query_p95_shift_tolerance": 1e-12,
        "l1_mean_shift_tolerance": 1e-12,
        "l1_p90_minus_p10_shift_tolerance": 1e-12,
        "unique_row_rate_tolerance": 1e-12,
        "normalized_row_entropy_tolerance": 1e-12,
        "minimum_active_round_rate": 0.5,
        "minimum_mean_changed_row_fraction": 0.5,
        "stall_patience_checks": 2,
    }
    values.update(changes)
    return StationarityDetectorConfig(**values)


def test_row_diversity_has_explicit_normalized_entropy():
    unique = pd.DataFrame({"x": [0, 1, 2, 3]})
    collapsed = pd.DataFrame({"x": [0, 0, 0, 0]})

    unique_metrics = stationarity_row_diversity_metrics(unique)
    collapsed_metrics = stationarity_row_diversity_metrics(collapsed)

    assert unique_metrics["unique_row_rate"] == 1.0
    assert unique_metrics["normalized_row_entropy"] == pytest.approx(1.0)
    assert collapsed_metrics["unique_row_rate"] == pytest.approx(0.25)
    assert collapsed_metrics["normalized_row_entropy"] == 0.0


def test_query_and_target_identities_are_ordered_and_deterministic():
    first = {
        "conditions": [{
            "attribute": "x", "operator": "==", "value": 0
        }]
    }
    second = {
        "conditions": [{
            "attribute": "x", "operator": "==", "value": 1
        }]
    }

    assert ordered_query_identity_sha256([first, second]) == (
        ordered_query_identity_sha256([first, second])
    )
    assert ordered_query_identity_sha256([first, second]) != (
        ordered_query_identity_sha256([second, first])
    )
    assert target_answer_identity_sha256([1.0, 2.0]) != (
        target_answer_identity_sha256([2.0, 1.0])
    )


def test_trace_round_trip_is_exact_and_strict(tmp_path):
    trace = _make_trace([[2.0, 1.0]] * 8, moving=True)
    output = tmp_path / "trace"

    paths = save_stationarity_trace(trace, output)
    loaded = load_stationarity_trace(output)

    assert loaded.contract_version == STATIONARITY_TRACE_CONTRACT_VERSION
    assert loaded.observations == trace.observations
    np.testing.assert_array_equal(
        loaded.measured_query_answers,
        trace.measured_query_answers,
    )
    assert len(paths["query_array_sha256"]) == 64
    metadata = json.loads(
        (output / "stationarity_trace.json").read_text(encoding="utf-8")
    )
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    assert metadata["query_array"]["dtype"] == "float64"
    with pytest.raises(FileExistsError):
        save_stationarity_trace(trace, output)


def test_trace_loader_rejects_modified_query_array(tmp_path):
    trace = _make_trace([[2.0, 1.0]] * 4, moving=True)
    output = tmp_path / "trace"
    save_stationarity_trace(trace, output)
    array_path = output / "measured_query_answers.npz"
    array_path.write_bytes(array_path.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="SHA-256"):
        load_stationarity_trace(output)


def test_trace_loader_rejects_metadata_shape_mismatch(tmp_path):
    trace = _make_trace([[2.0, 1.0]] * 4, moving=True)
    output = tmp_path / "trace"
    save_stationarity_trace(trace, output)
    metadata_path = output / "stationarity_trace.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["query_array"]["shape"] = [999, 2]
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="shape"):
        load_stationarity_trace(output)


@pytest.mark.parametrize("location", ["metadata", "query_array", "observation"])
def test_trace_loader_rejects_unknown_fields(tmp_path, location):
    trace = _make_trace([[2.0, 1.0]] * 4, moving=True)
    output = tmp_path / "trace"
    save_stationarity_trace(trace, output)
    metadata_path = output / "stationarity_trace.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if location == "metadata":
        metadata["unknown"] = True
    elif location == "query_array":
        metadata["query_array"]["unknown"] = True
    else:
        metadata["observations"][0]["unknown"] = True
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="未知字段"):
        load_stationarity_trace(output)


def test_dynamic_stability_requires_two_completed_checks():
    trace = _make_trace([[2.0, 1.0]] * 8, moving=True)
    config = _config()

    result = replay_stationarity(trace, config)

    assert result.contract_version == STATIONARITY_REPLAY_CONTRACT_VERSION
    assert result.status == "stationary_qualified"
    assert result.candidate_round_index == 8
    assert len(result.checks) == 2
    assert result.checks[0]["moving_stability_streak"] == 1
    assert result.checks[1]["moving_stability_streak"] == 2
    payload = result.to_dict()
    assert payload["detector_config"] == config.to_dict()
    assert payload["trace"]["query_identity_sha256"] == "1" * 64
    assert payload["trace"]["target_identity_sha256"] == "2" * 64
    assert payload["trace"]["post_round_count"] == 8
    assert len(payload["trace"]["trace_identity_sha256"]) == 64
    json.dumps(payload, allow_nan=False)


def test_detector_evidence_uses_explicit_all_pairwise_formulas():
    trace = _make_trace(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [4.0, 0.0],
            [6.0, 0.0],
        ],
        moving=True,
        termination_reason="in_progress",
        n_records=10,
    )
    config = _config(
        query_mean_shift_tolerance=1.0,
        query_p95_shift_tolerance=1.0,
        l1_mean_shift_tolerance=1.0,
        l1_p90_minus_p10_shift_tolerance=1.0,
        unique_row_rate_tolerance=1.0,
        normalized_row_entropy_tolerance=1.0,
    )

    result = replay_stationarity(trace, config)
    check = result.checks[0]

    assert check["query_mean_shift"] == pytest.approx(0.2)
    assert check["query_p95_shift"] == pytest.approx(0.38)
    assert check["l1_mean_shift"] == pytest.approx(0.2)
    assert check["l1_p90_minus_p10_shift"] == pytest.approx(0.0)
    assert check["window_l1_means"] == pytest.approx([0.05, 0.15, 0.25])
    assert check["window_l1_p90_minus_p10"] == pytest.approx(
        [0.08, 0.08, 0.08]
    )
    assert check["window_l1_p95"] == pytest.approx(
        [0.095, 0.195, 0.295]
    )
    assert check["window_active_round_rates"] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert check["window_mean_changed_row_fractions"] == pytest.approx(
        [1.0, 1.0, 1.0]
    )
    assert check["minimum_observed_active_round_rate"] == 1.0
    assert check["minimum_observed_mean_changed_row_fraction"] == 1.0


def test_replay_is_exactly_deterministic():
    trace = _make_trace([[2.0, 1.0]] * 8, moving=True)
    config = _config()

    assert replay_stationarity(trace, config).to_dict() == (
        replay_stationarity(trace, config).to_dict()
    )


def test_frozen_flat_trace_is_stalled_not_stationary():
    trace = _make_trace([[2.0, 1.0]] * 8, moving=False)

    result = replay_stationarity(trace, _config())

    assert result.status == "stalled"
    assert result.candidate_round_index == 8
    assert all(
        check["movement_sufficient"] is False for check in result.checks
    )


def test_active_but_microscopic_row_movement_is_stalled():
    trace = _make_trace(
        [[500.0, 250.0]] * 8,
        moving=True,
        n_records=1_000,
        changed_rows_per_round=[1] * 8,
    )
    result = replay_stationarity(
        trace,
        _config(
            minimum_active_round_rate=0.5,
            minimum_mean_changed_row_fraction=0.01,
            unique_row_rate_tolerance=1.0,
            normalized_row_entropy_tolerance=1.0,
        ),
    )

    assert result.status == "stalled"
    assert all(
        check["minimum_observed_active_round_rate"] == 1.0
        for check in result.checks
    )
    assert all(
        check["minimum_observed_mean_changed_row_fraction"]
        == pytest.approx(0.001)
        for check in result.checks
    )
    assert all(
        check["movement_sufficient"] is False for check in result.checks
    )


def test_earlier_movement_cannot_hide_a_frozen_recent_window():
    trace = _make_trace(
        [[2.0, 1.0]] * 8,
        moving=True,
        changed_rows_per_round=[4, 4, 4, 4, 0, 0, 0, 0],
    )
    result = replay_stationarity(
        trace,
        _config(
            unique_row_rate_tolerance=1.0,
            normalized_row_entropy_tolerance=1.0,
        ),
    )

    assert result.status == "stalled"
    assert result.checks[0]["window_active_round_rates"] == [1.0, 1.0, 0.0]
    assert result.checks[1]["window_active_round_rates"] == [1.0, 0.0, 0.0]
    assert all(
        check["movement_sufficient"] is False for check in result.checks
    )


def test_high_but_stable_l1_can_still_qualify_stationarity():
    trace = _make_trace([[4.0, 4.0]] * 8, moving=True)

    result = replay_stationarity(trace, _config())

    assert trace.observations[-1]["current_normalized_l1"] == 1.0
    assert result.status == "stationary_qualified"


def test_one_low_movement_check_is_only_insufficient_movement():
    trace = _make_trace(
        [[2.0, 1.0]] * 6,
        moving=False,
        termination_reason="in_progress",
    )

    result = replay_stationarity(trace, _config())

    assert result.status == "insufficient_movement"
    assert result.candidate_round_index is None


def test_persistent_drift_reaches_horizon_without_qualification():
    query_vectors = [[0.5 * index, 0.0] for index in range(8)]
    trace = _make_trace(query_vectors, moving=True)

    result = replay_stationarity(trace, _config())

    assert result.status == "horizon_reached"
    assert result.candidate_round_index is None
    assert all(check["stable"] is False for check in result.checks)


def test_partial_drifting_trace_is_running_after_first_check():
    query_vectors = [[0.5 * index, 0.0] for index in range(6)]
    trace = _make_trace(
        query_vectors,
        moving=True,
        termination_reason="in_progress",
    )

    result = replay_stationarity(trace, _config())

    assert result.status == "running"
    assert len(result.checks) == 1
    assert result.checks[0]["stable"] is False


def test_late_stability_does_not_use_the_earlier_drifting_block():
    query_vectors = [
        [0.0, 0.0],
        [4.0, 0.0],
        *([[2.0, 1.0]] * 8),
    ]
    trace = _make_trace(query_vectors, moving=True)

    result = replay_stationarity(trace, _config())

    assert result.status == "stationary_qualified"
    assert result.candidate_round_index == 10
    assert len(result.checks) == 3


def test_short_trace_is_collecting():
    trace = _make_trace(
        [[2.0, 1.0]] * 5,
        moving=True,
        termination_reason="in_progress",
    )

    result = replay_stationarity(trace, _config())

    assert result.status == "collecting"
    assert result.checks == []


@pytest.mark.parametrize(
    "termination_reason", ["exact_residual", "self_cooling_ratio"]
)
def test_terminal_short_trace_is_not_reported_as_collecting(
    termination_reason,
):
    trace = _make_trace(
        [[2.0, 1.0]] * 5,
        moving=True,
        termination_reason=termination_reason,
    )

    result = replay_stationarity(trace, _config())

    assert result.status == "terminated_before_qualification"
    assert result.checks == []


def test_detector_ignores_initial_observation_values():
    first = _make_trace([[2.0, 1.0]] * 8, moving=True)
    second = _make_trace([[2.0, 1.0]] * 8, moving=True)
    second.measured_query_answers[0] = np.array([4.0, 4.0])
    second.observations[0]["current_normalized_l1"] = 999.0
    second.observations[0]["current_squared_loss"] = 999.0
    second.observations[1]["actual_changed_query_count"] = 2
    second.observations[1]["normalized_query_l1_movement_mean"] = (
        2.5 / 4
    )

    first_result = replay_stationarity(first, _config())
    second_result = replay_stationarity(second, _config())

    assert first_result.status == second_result.status
    assert first_result.candidate_state_index == (
        second_result.candidate_state_index
    )
    assert first_result.checks == second_result.checks
    assert first_result.to_dict()["trace"]["trace_identity_sha256"] != (
        second_result.to_dict()["trace"]["trace_identity_sha256"]
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"window_size": 0}, "window_size"),
        ({"window_size": 1}, "window_size"),
        ({"query_mean_shift_tolerance": -1.0}, "query_mean"),
        ({"minimum_active_round_rate": 0.0}, "active_round_rate"),
        ({"minimum_active_round_rate": 1.1}, "active_round_rate"),
        (
            {"minimum_mean_changed_row_fraction": 0.0},
            "changed_row_fraction",
        ),
        (
            {"minimum_mean_changed_row_fraction": 1.1},
            "changed_row_fraction",
        ),
        ({"stall_patience_checks": 0}, "stall_patience"),
    ],
)
def test_detector_config_fails_closed(changes, match):
    with pytest.raises(ValueError, match=match):
        _config(**changes)


def test_detector_api_has_no_initializer_or_generation_parameters():
    assert list(inspect.signature(replay_stationarity).parameters) == [
        "trace",
        "config",
    ]
