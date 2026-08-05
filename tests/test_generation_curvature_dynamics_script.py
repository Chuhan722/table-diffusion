"""整代曲率 Gibbs 无接受动力学脚本的协议与回归测试。"""

import json

import numpy as np
import pytest

import scripts.compare_generation_curvature_unfiltered as experiment
import scripts.diagnose_curvature_multistep_drift as drift_diagnostic
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


def _inputs():
    schema = load_schema(str(experiment.SCHEMA_PATH))
    queries = load_queries(str(experiment.QUERY_PATH))
    target = np.asarray([query["result"] for query in queries], dtype=float)
    marginals = load_marginals(str(experiment.MARGINALS_PATH))
    return target, queries, schema, marginals


def _decision_rows(
    tail_values,
    *,
    entropy=0.68,
    unique=280,
    maximum=1500.0,
):
    rows = []
    for value in tail_values:
        rows.append({
            "late_250_mean_loss": float(value),
            "rounds_run": 250,
            "conditional_probability_count": 10,
            "conditional_entropy_mean": float(entropy),
            "conditional_probability_min": 0.1,
            "conditional_probability_max": 0.9,
            "conditional_logit_abs_max": 2.2,
            "conditional_logit_clipped_count": 0,
            "all_conditionals_bidirectional": True,
            "conditional_probability_count_history": [10] * 250,
            "conditional_entropy_mean_history": [float(entropy)] * 250,
            "conditional_probability_min_history": [0.1] * 250,
            "conditional_probability_max_history": [0.9] * 250,
            "conditional_logit_abs_max_history": [2.2] * 250,
            "conditional_logit_clipped_count_history": [0] * 250,
            "conditional_bidirectional_history": [True] * 250,
            "final_unique_states": int(unique),
            "maximum_loss": float(maximum),
        })
    return rows


@pytest.mark.parametrize(
    "candidate_values,expected",
    [
        ([90.0] * 20, "supports_unfiltered_curvature_dynamics"),
        ([97.0] * 20, "curvature_dynamics_inconclusive"),
        (
            [90.0] * 12 + [110.0] * 8,
            "curvature_dynamics_inconclusive",
        ),
        ([110.0] * 20, "curvature_dynamics_not_supported"),
        (
            [80.0] * 10 + [105.0] * 10,
            "curvature_dynamics_not_supported",
        ),
    ],
)
def test_decision_follows_preregistered_seed_rule(
    candidate_values, expected
):
    baseline = _decision_rows([100.0] * 20)
    candidate = _decision_rows(candidate_values)
    comparison = experiment._paired(
        candidate,
        baseline,
        "late_250_mean_loss",
        lower_is_better=True,
    )

    result = experiment._decision_from_runs(
        baseline,
        candidate,
        {"late_250_mean_loss": comparison},
    )

    assert result["decision"] == expected


def test_decision_reports_preregistered_risks():
    baseline = _decision_rows(
        [100.0] * 20,
        entropy=0.68,
        unique=280,
        maximum=1000.0,
    )
    candidate = _decision_rows(
        [90.0] * 20,
        entropy=0.55,
        unique=250,
        maximum=1300.0,
    )
    comparison = experiment._paired(
        candidate,
        baseline,
        "late_250_mean_loss",
        lower_is_better=True,
    )

    result = experiment._decision_from_runs(
        baseline,
        candidate,
        {"late_250_mean_loss": comparison},
    )

    assert result["conditional_entropy_concentration_risk"] is True
    assert result["support_contraction_risk"] is True
    assert result["stage_explosion_risk"] is True


def test_decision_entropy_risk_uses_tail_instead_of_full_trajectory():
    baseline = _decision_rows([100.0] * 20, entropy=0.68)
    candidate = _decision_rows([90.0] * 20, entropy=0.55)
    for row in baseline:
        row["conditional_entropy_mean"] = 0.40
    for row in candidate:
        row["conditional_entropy_mean"] = 0.69
    comparison = experiment._paired(
        candidate,
        baseline,
        "late_250_mean_loss",
        lower_is_better=True,
    )

    result = experiment._decision_from_runs(
        baseline,
        candidate,
        {"late_250_mean_loss": comparison},
    )

    assert result["conditional_entropy_relative_change"] == pytest.approx(
        0.55 / 0.68 - 1.0
    )
    assert result["conditional_entropy_concentration_risk"] is True


def test_conditional_summary_weights_microsteps():
    rows = _decision_rows([1.0, 1.0], entropy=0.4)
    rows[1]["conditional_probability_count"] = 30
    rows[1]["conditional_entropy_mean"] = 0.6
    rows[1]["conditional_probability_min"] = 0.05
    rows[1]["conditional_probability_max"] = 0.95
    rows[1]["conditional_logit_abs_max"] = 3.0

    result = experiment._aggregate_conditional(rows)

    assert result["n_microsteps"] == 40
    assert result["mean_entropy"] == pytest.approx(0.55)
    assert result["min_probability"] == 0.05
    assert result["max_probability"] == 0.95
    assert result["max_abs_logit"] == 3.0
    assert result["logit_clip_hits"] == 0
    assert result["all_bidirectional"] is True


def test_tail_conditional_summary_uses_only_preregistered_window():
    rows = _decision_rows([1.0], entropy=0.6)
    rows[0]["rounds_run"] = 251
    rows[0]["conditional_probability_count_history"] = [100] + [10] * 250
    rows[0]["conditional_entropy_mean_history"] = [0.1] + [0.6] * 250
    rows[0]["conditional_probability_min_history"] = [0.01] + [0.1] * 250
    rows[0]["conditional_probability_max_history"] = [0.99] + [0.9] * 250
    rows[0]["conditional_logit_abs_max_history"] = [8.0] + [2.2] * 250
    rows[0]["conditional_logit_clipped_count_history"] = [1] + [0] * 250
    rows[0]["conditional_bidirectional_history"] = [False] + [True] * 250

    result = experiment._aggregate_tail_conditional(rows)

    assert result["n_round_observations"] == 250
    assert result["n_microsteps"] == 2500
    assert result["mean_entropy"] == pytest.approx(0.6)
    assert result["min_probability"] == 0.1
    assert result["max_probability"] == 0.9
    assert result["max_abs_logit"] == 2.2
    assert result["logit_clip_hits"] == 0
    assert result["all_bidirectional"] is True


def test_tail_conditional_summary_handles_rounds_without_microsteps():
    rows = _decision_rows([1.0], entropy=0.6)
    rows[0]["rounds_run"] = 2
    rows[0]["conditional_probability_count_history"] = [0, 10]
    rows[0]["conditional_entropy_mean_history"] = [None, 0.6]
    rows[0]["conditional_probability_min_history"] = [None, 0.1]
    rows[0]["conditional_probability_max_history"] = [None, 0.9]
    rows[0]["conditional_logit_abs_max_history"] = [None, 2.2]
    rows[0]["conditional_logit_clipped_count_history"] = [0, 0]
    rows[0]["conditional_bidirectional_history"] = [True, True]

    result = experiment._aggregate_tail_conditional(rows)

    assert result["n_microsteps"] == 10
    assert result["mean_entropy"] == pytest.approx(0.6)
    assert result["max_abs_logit"] == 2.2


def test_tail_conditional_summary_rejects_incomplete_history():
    rows = _decision_rows([1.0])
    rows[0]["conditional_probability_count_history"].pop()

    with pytest.raises(ValueError, match="长度"):
        experiment._aggregate_tail_conditional(rows)


def test_paired_descriptive_metric_has_no_win_semantics():
    baseline = [{"work": 10.0}, {"work": 20.0}]
    candidate = [{"work": 12.0}, {"work": 18.0}]

    result = experiment._paired(
        candidate,
        baseline,
        "work",
        lower_is_better=None,
    )

    assert result["preference"] == "descriptive_only"
    assert result["lower_is_better"] is None
    assert result["wins"] is None
    assert result["ties"] is None
    assert result["losses"] is None


def test_round_addressed_gibbs_seeds_are_reproducible_and_distinct():
    first = experiment._gibbs_round_seed(3, 7)

    assert first == experiment._gibbs_round_seed(3, 7)
    assert first != experiment._gibbs_round_seed(3, 8)
    assert first != experiment._gibbs_round_seed(4, 7)


def test_gamma_zero_short_trajectory_matches_existing_factorized_path():
    target, queries, schema, marginals = _inputs()

    result = experiment._verify_gamma_zero_reference(
        target,
        queries,
        schema,
        marginals,
        seed=5,
        rounds=4,
        temperature=2.0,
        sweeps=2,
        device="numpy",
    )

    assert result["passed"] is True
    assert result["rounds"] == 4
    assert len(result["loss_history"]) == 5
    assert result["table_exact_rounds"] == 4
    assert result["loss_exact_rounds"] == 4
    assert result["primary_rng_exact_rounds"] == 4
    assert result["gibbs_rng_exact_rounds"] == 4
    assert result["common_diagnostics_exact_rounds"] == 4
    assert len(result["gibbs_round_seed_sha256"]) == 64
    assert len(result["gibbs_round_endpoint_sha256"]) == 64
    assert result["gamma_zero_conditional_probability_max_error"] == 0.0
    assert result["logit_clip_hits"] == 0


def test_two_variants_run_full_unfiltered_trajectory_with_aligned_primary_rng():
    target, queries, schema, marginals = _inputs()
    common = {
        "seed": 7,
        "rounds": 12,
        "temperature": 2.0,
        "sweeps": 2,
        "device": "numpy",
        "record_query_clock": True,
    }

    baseline = experiment._run_one(
        target,
        queries,
        schema,
        marginals,
        curvature_weight=0.0,
        **common,
    )
    candidate = experiment._run_one(
        target,
        queries,
        schema,
        marginals,
        curvature_weight=1.0,
        **common,
    )

    assert baseline["rounds_run"] == candidate["rounds_run"] == 12
    assert len(baseline["loss_history"]) == 13
    assert len(candidate["loss_history"]) == 13
    assert len(baseline["gain_history"]) == 12
    assert len(candidate["gain_history"]) == 12
    assert len(baseline["query_state_sha256_history"]) == 13
    assert len(candidate["query_state_sha256_history"]) == 13
    assert len(baseline["query_count_history"]) == 13
    assert len(candidate["query_count_history"]) == 13
    assert len(baseline["query_delta_l2_squared_history"]) == 12
    assert len(candidate["query_delta_l2_squared_history"]) == 12
    assert len(
        baseline["cumulative_query_quadratic_variation_history"]
    ) == 13
    assert len(
        candidate["cumulative_query_quadratic_variation_history"]
    ) == 13
    assert baseline["gain_identity_max_abs_error"] == 0.0
    assert candidate["gain_identity_max_abs_error"] == 0.0
    np.testing.assert_array_equal(
        np.asarray(baseline["count_residual_l2_squared_history"]),
        2.0 * np.asarray(baseline["loss_history"]),
    )
    np.testing.assert_array_equal(
        np.asarray(candidate["count_residual_l2_squared_history"]),
        2.0 * np.asarray(candidate["loss_history"]),
    )
    query_clock_gate = drift_diagnostic._query_clock_gate(
        {"baseline": [baseline], "candidate": [candidate]},
        12,
        target,
    )
    assert query_clock_gate["passed"] is True
    assert query_clock_gate["checked_query_vectors"] == 26
    assert query_clock_gate["checked_transitions"] == 24
    assert len(baseline["conditional_probability_count_history"]) == 12
    assert len(candidate["conditional_probability_count_history"]) == 12
    baseline_tail = experiment._aggregate_tail_conditional([baseline])
    candidate_tail = experiment._aggregate_tail_conditional([candidate])
    assert baseline_tail["n_round_observations"] == 12
    assert candidate_tail["n_round_observations"] == 12
    assert baseline["late_250_conditional_entropy_mean"] == pytest.approx(
        baseline_tail["mean_entropy"]
    )
    assert candidate["late_250_conditional_entropy_mean"] == pytest.approx(
        candidate_tail["mean_entropy"]
    )
    assert baseline["initial_table_sha256"] == candidate[
        "initial_table_sha256"
    ]
    assert baseline["initial_loss"] == candidate["initial_loss"]
    assert baseline["direction_reference_scale"] == candidate[
        "direction_reference_scale"
    ]
    assert baseline["direction_reference_scale_round"] == 0
    assert candidate["direction_reference_scale_round"] == 0
    assert baseline["primary_rng_state_sha256"] == candidate[
        "primary_rng_state_sha256"
    ]
    assert baseline["gibbs_round_seed_sha256"] == candidate[
        "gibbs_round_seed_sha256"
    ]
    assert baseline[
        "gamma_zero_reference_probability_max_error"
    ] == 0.0
    assert candidate[
        "gamma_zero_reference_probability_max_error"
    ] is None
    assert baseline["conditional_logit_clipped_count"] == 0
    assert candidate["conditional_logit_clipped_count"] == 0
    assert baseline["all_conditionals_bidirectional"] is True
    assert candidate["all_conditionals_bidirectional"] is True
    assert sum(baseline["gain_history"]) == pytest.approx(
        baseline["initial_loss"] - baseline["final_loss"]
    )
    assert sum(candidate["gain_history"]) == pytest.approx(
        candidate["initial_loss"] - candidate["final_loss"]
    )
    assert baseline["best_loss_diagnostic_only"] == min(
        baseline["loss_history"]
    )
    assert candidate["best_loss_diagnostic_only"] == min(
        candidate["loss_history"]
    )


def test_json_writer_rejects_nonfinite_values(tmp_path):
    output = tmp_path / "invalid.json"

    with pytest.raises(ValueError, match="非有限"):
        experiment._write_json(output, {"bad": np.inf})

    assert not output.exists()


def test_json_writer_replaces_complete_payload_atomically(tmp_path):
    output = tmp_path / "result.json"

    experiment._write_json(output, {"status": "running", "count": 1})
    experiment._write_json(output, {"status": "complete", "count": 2})

    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "complete",
        "count": 2,
    }
    assert not output.with_name("result.json.tmp").exists()


def test_main_rejects_rounds_shorter_than_preregistered_window(monkeypatch):
    monkeypatch.setattr(
        experiment.sys,
        "argv",
        [
            "compare_generation_curvature_unfiltered.py",
            "--rounds",
            str(experiment.PRIMARY_TAIL_WINDOW - 1),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        experiment.main()
