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


def _make_trace(query_vectors, *, moving, termination_reason="max_rounds"):
    target = np.zeros(len(query_vectors[0]), dtype=float)
    frame_a = pd.DataFrame({"x": [0, 0, 1, 1]})
    frame_b = pd.DataFrame({"x": [1, 1, 0, 0]})
    initial_q = np.asarray(query_vectors[0], dtype=float)
    observations = [
        build_stationarity_observation(
            frame=frame_a,
            target=target,
            current_query_answers=initial_q,
            n_records=4,
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
    for index, values in enumerate(query_vectors, start=1):
        current_q = np.asarray(values, dtype=float)
        current_frame = (
            frame_b if moving and index % 2 else frame_a
        )
        changed = current_frame.ne(previous_frame).to_numpy(dtype=bool)
        delta_q = current_q - previous_q
        observations.append(
            build_stationarity_observation(
                frame=current_frame,
                target=target,
                current_query_answers=current_q,
                n_records=4,
                squared_loss=float(0.5 * np.dot(current_q, current_q)),
                state_index=index,
                round_index=index,
                phase="post_round",
                proposal_attempt_count=1,
                proposal_accepted=True,
                applied_attempt_index=1,
                attempted_participating_row_count=4,
                applied_participating_row_count=4,
                actual_changed_row_count=int(
                    np.any(changed, axis=1).sum()
                ),
                actual_changed_cell_count=int(changed.sum()),
                actual_changed_query_count=int(np.count_nonzero(delta_q)),
                normalized_query_l1_movement_mean=float(
                    np.mean(np.abs(delta_q)) / 4
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
        n_records=4,
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
        "l1_location_tolerance": 1e-12,
        "l1_spread_tolerance": 1e-12,
        "unique_row_rate_tolerance": 1e-12,
        "normalized_row_entropy_tolerance": 1e-12,
        "minimum_changed_state_rate": 0.5,
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


def test_dynamic_stability_requires_two_completed_checks():
    trace = _make_trace([[2.0, 1.0]] * 8, moving=True)

    result = replay_stationarity(trace, _config())

    assert result.contract_version == STATIONARITY_REPLAY_CONTRACT_VERSION
    assert result.status == "stationary_qualified"
    assert result.candidate_round_index == 8
    assert len(result.checks) == 2
    assert result.checks[0]["moving_stability_streak"] == 1
    assert result.checks[1]["moving_stability_streak"] == 2
    json.dumps(result.to_dict(), allow_nan=False)


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


def test_one_low_movement_check_is_only_insufficient_movement():
    trace = _make_trace(
        [[2.0, 1.0]] * 6,
        moving=False,
        termination_reason="in_progress",
    )

    result = replay_stationarity(trace, _config())

    assert result.status == "insufficient_movement"
    assert result.candidate_round_index is None


def test_persistent_drift_is_horizon_limited():
    query_vectors = [[0.5 * index, 0.0] for index in range(8)]
    trace = _make_trace(query_vectors, moving=True)

    result = replay_stationarity(trace, _config())

    assert result.status == "horizon_limited"
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

    assert replay_stationarity(first, _config()).to_dict() == (
        replay_stationarity(second, _config()).to_dict()
    )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"window_size": 0}, "window_size"),
        ({"query_mean_shift_tolerance": -1.0}, "query_mean"),
        ({"minimum_changed_state_rate": 1.1}, "changed_state_rate"),
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
