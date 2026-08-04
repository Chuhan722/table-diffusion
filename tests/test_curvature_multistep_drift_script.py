"""曲率核多步漂移与内禀扩散时钟诊断测试。"""

import copy

import pytest

import scripts.diagnose_curvature_multistep_drift as diagnostic


def _clock_row(seed, losses, query_clock, changed=None):
    rounds = len(losses) - 1
    if changed is None:
        changed = [1] * rounds
    return {
        "seed": seed,
        "rounds_run": rounds,
        "loss_history": list(losses),
        "changed_cells_history": list(changed),
        "cumulative_query_quadratic_variation_history": list(query_clock),
    }


def _query_gate_row():
    query_counts = [[0], [1]]
    return {
        "seed": 0,
        "rounds_run": 1,
        "query_clock_recorded": True,
        "query_count_history": query_counts,
        "query_state_sha256_history": [
            diagnostic.dynamics._query_vector_sha256(values)
            for values in query_counts
        ],
        "count_residual_l2_squared_history": [4.0, 1.0],
        "query_delta_l2_squared_history": [1.0],
        "linear_gain_history": [2.0],
        "quadratic_cost_history": [0.5],
        "gain_identity_error_history": [0.0],
        "gain_identity_max_abs_error": 0.0,
        "cumulative_query_quadratic_variation_history": [0.0, 1.0],
        "loss_history": [2.0, 0.5],
        "gain_history": [1.5],
        "changed_cells_history": [1],
    }


def test_clock_compression_keeps_last_identical_duplicate_state():
    row = _clock_row(0, [10.0, 10.0, 8.0], [0.0, 0.0, 2.0])

    clock, losses = diagnostic._clock_and_loss(
        row, "query_quadratic_variation"
    )

    assert clock.tolist() == [0.0, 2.0]
    assert losses.tolist() == [10.0, 8.0]


def test_clock_rejects_loss_change_without_intrinsic_motion():
    row = _clock_row(0, [10.0, 9.0], [0.0, 0.0])

    with pytest.raises(ValueError, match="loss"):
        diagnostic._clock_and_loss(row, "query_quadratic_variation")


def test_matched_clock_uses_preregistered_last_quarter_grid():
    baseline = _clock_row(
        3,
        [10.0, 8.0, 6.0, 4.0, 2.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    )
    candidate = _clock_row(
        3,
        [11.0, 9.0, 7.0, 5.0, 3.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    )

    result = diagnostic._matched_clock_pair(
        baseline,
        candidate,
        "query_quadratic_variation",
    )

    assert result["matched_start_clock"] == 3.0
    assert result["common_final_clock"] == 4.0
    assert result["grid_points"] == 251
    assert result["matched_mean_loss_difference"] == pytest.approx(1.0)
    assert result["matched_endpoint_loss_difference"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "interval,mean,calendar_mean,expected",
    [
        (
            [-2.0, -0.1],
            -1.0,
            5.0,
            "clock_efficiency_advantage_but_round_slowdown",
        ),
        (
            [0.1, 2.0],
            1.0,
            5.0,
            "drift_disadvantage_after_clock_matching",
        ),
        (
            [-1.0, 1.0],
            2.0,
            5.0,
            "time_rescaling_material_residual_inconclusive",
        ),
        (
            [-1.0, 1.0],
            3.0,
            5.0,
            "mixed_or_inconclusive_multistep_effect",
        ),
    ],
)
def test_multistep_classification_follows_preregistered_rule(
    interval, mean, calendar_mean, expected
):
    query_clock = {
        "matched_mean_loss_difference": {
            "mean": mean,
            "mean_t_interval_95": interval,
        }
    }
    calendar = {"difference": {"mean": calendar_mean}}

    assert diagnostic._classify_multistep_effect(
        query_clock, calendar
    ) == expected


def test_replay_audit_ignores_only_timing_and_additive_diagnostics():
    reference_row = {
        "seed": 0,
        "rounds_run": 2,
        "loss_history": [10.0, 8.0, 7.0],
        "gain_history": [2.0, 1.0],
        "elapsed_sec": 3.0,
    }
    replay_row = {
        **reference_row,
        "elapsed_sec": 4.0,
        "query_clock_recorded": True,
    }
    reference = {
        "runs": {
            "baseline": [reference_row],
            "candidate": [copy.deepcopy(reference_row)],
        }
    }
    runs = {
        "baseline": [replay_row],
        "candidate": [copy.deepcopy(replay_row)],
    }

    assert diagnostic._audit_replay(reference, runs)["passed"] is True

    runs["candidate"][0]["gain_history"][1] = 0.5
    failed = diagnostic._audit_replay(reference, runs)
    assert failed["passed"] is False
    assert failed["failures"] == [{
        "variant": "candidate",
        "seed": 0,
        "key": "gain_history",
    }]


def test_replay_audit_rejects_missing_none_field_and_unexpected_field():
    reference_row = {
        "seed": 0,
        "rounds_run": 0,
        "optional": None,
        "elapsed_sec": 1.0,
    }
    reference = {
        "runs": {
            "baseline": [reference_row],
            "candidate": [copy.deepcopy(reference_row)],
        }
    }
    runs = {
        "baseline": [{"seed": 0, "rounds_run": 0, "elapsed_sec": 2.0}],
        "candidate": [{**reference_row, "unplanned": 1}],
    }

    failed = diagnostic._audit_replay(reference, runs)

    assert failed["passed"] is False
    assert failed["failures"] == [
        {"variant": "baseline", "seed": 0, "key": "missing:optional"},
        {
            "variant": "candidate",
            "seed": 0,
            "key": "unexpected:unplanned",
        },
    ]


def test_query_clock_gate_recomputes_absolute_loss_and_gain():
    row = _query_gate_row()
    runs = {"baseline": [row]}

    passed = diagnostic._query_clock_gate(runs, 1, [2.0])

    assert passed["passed"] is True
    assert passed["checked_query_vectors"] == 2
    assert passed["checked_transitions"] == 1

    row["loss_history"] = [3.0, 1.5]
    failed = diagnostic._query_clock_gate(runs, 1, [2.0])
    assert failed["passed"] is False
    assert failed["failures"][0]["reason"] == (
        "recomputed_loss_history_mismatch"
    )


def test_query_clock_gate_reports_missing_history_instead_of_raising():
    row = _query_gate_row()
    del row["query_count_history"]

    failed = diagnostic._query_clock_gate(
        {"candidate": [row]}, 1, [2.0]
    )

    assert failed["passed"] is False
    assert failed["failures"] == [{
        "variant": "candidate",
        "seed": 0,
        "reason": "missing_field_or_round_count",
    }]
