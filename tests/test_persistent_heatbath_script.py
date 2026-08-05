"""持久化热浴扩散正式脚本的配对、分类、落盘与独立审计测试。"""

import copy
import json

import numpy as np
import pytest

import scripts.probe_persistent_heatbath as probe
from table_diffevo.schema import AttributeBlock, Schema


def _tiny_problem():
    schema = Schema([
        AttributeBlock(
            name="a", type="categorical", description="", values=[0, 1]
        ),
        AttributeBlock(
            name="b", type="categorical", description="", values=[0, 1]
        ),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
    ]
    return schema, queries, np.asarray([1.0, 1.0, 1.0])


def _write_tiny_public_inputs(tmp_path):
    schema_path = tmp_path / "schema.yaml"
    query_path = tmp_path / "queries.json"
    marginals_path = tmp_path / "marginals.json"
    schema_path.write_text(json.dumps({
        "attributes": [
            {
                "name": "a",
                "type": "categorical",
                "values": [0, 1],
            },
            {
                "name": "b",
                "type": "categorical",
                "values": [0, 1],
            },
        ]
    }), encoding="utf-8")
    _, queries, target = _tiny_problem()
    measured_queries = [
        {**query, "result": int(result)}
        for query, result in zip(queries, target)
    ]
    query_path.write_text(
        json.dumps({"queries": measured_queries}), encoding="utf-8"
    )
    marginals = {
        "n_records": 2,
        "attributes": {
            "a": {
                "type": "categorical",
                "values": [0, 1],
                "counts": [1, 1],
            },
            "b": {
                "type": "categorical",
                "values": [0, 1],
                "counts": [1, 1],
            },
        },
    }
    marginals_path.write_text(json.dumps(marginals), encoding="utf-8")
    return {
        "schema": schema_path,
        "queries": query_path,
        "marginals": marginals_path,
    }, marginals


def _fake_metrics(value):
    return {
        "tail_mean_loss": float(value),
        "final_loss": float(value),
        "trajectory_mean_loss": float(value + 1.0),
        "diagnostic_best_loss": float(value - 1.0),
        "positive_steps": 4,
        "zero_steps": 3,
        "negative_steps": 3,
        "positive_gain_mean": 1.0,
        "negative_gain_abs_mean": 0.5,
        "changed_value_rate": 0.5,
        "conditional_expected_state_gain_mean": 0.1,
        "expected_gain_over_reference_mean": 0.2,
        "conditional_normalized_entropy_mean": 0.8,
        "uphill_probability_mass_mean": 0.1,
        "probability_min": 0.01,
        "probability_max": 0.99,
        "candidate_state_evaluations": 20,
        "query_indicator_evaluations": 40,
        "kernel_elapsed_sec": 0.2,
        "full_table_state_audits": 2,
    }


def _fake_run(seed, baseline, candidate, *, gate=True):
    return {
        "seed": seed,
        "n_steps": 10,
        "scale_elapsed_sec": 0.1,
        "elapsed_sec": 0.5,
        "baseline": {"metrics": _fake_metrics(baseline)},
        "candidate": {"metrics": _fake_metrics(candidate)},
        "gates": {
            "initial_scale_positive": gate,
            "initial_states_aligned": True,
            "random_inputs_aligned": True,
            "random_rng_endpoints_aligned": True,
            "full_audit_counts_complete": True,
            "all_full_state_audits_exact": True,
            "gain_identity_max_error": 0.0,
            "conditional_expectation_violation_max": 0.0,
            "all_probabilities_strictly_positive": True,
        },
    }


def test_exact_oracle_passes_every_preregistered_semantic_gate():
    oracle = probe.run_exact_oracle()

    assert oracle["passed"] is True
    assert oracle["states"] == 16
    assert oracle["betas"] == [0.0, 0.7, 1.3]
    assert oracle["conditional_probability_max_error"] <= 1e-12
    assert oracle["query_increment_max_error"] <= 1e-12
    assert oracle["loss_gain_identity_max_error"] <= 1e-12
    assert oracle["expected_loss_monotonic"] is True
    assert oracle["derivative_identity_max_error"] <= 1e-9
    for row in oracle["by_beta"]:
        assert row["all_allowed_single_coordinate_transitions_positive"]
        assert row["all_multi_coordinate_transitions_zero"]
        assert row["irreducible"]
        assert row["all_state_self_loops_positive"]
        assert row["detailed_balance_max_error"] <= 1e-12
        assert row["stationarity_max_error"] <= 1e-12


def test_run_seed_uses_common_randomness_and_produces_auditable_trajectories(
    monkeypatch,
):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        40,
        schema,
        queries,
        target,
        None,
        steps=12,
        tail=4,
        tau=1.0,
        verify_every=5,
    )

    assert run["n_steps"] == 12
    assert len(run["coordinate_history"]) == 12
    assert run["gates"]["initial_scale_positive"] is True
    assert run["gates"]["initial_states_aligned"] is True
    assert run["gates"]["random_inputs_aligned"] is True
    assert run["gates"]["random_rng_endpoints_aligned"] is True
    assert run["gates"]["all_full_state_audits_exact"] is True
    assert run["gates"]["gain_identity_max_error"] <= 1e-12
    assert run["baseline_rng_final_state_sha256"] == (
        run["candidate_rng_final_state_sha256"]
    )
    for variant in ("baseline", "candidate"):
        trajectory = run[variant]
        assert len(trajectory["loss_history"]) == 13
        assert len(trajectory["query_delta_sparse_history"]) == 12
        assert [row["step"] for row in trajectory["full_state_audits"]] == [
            0, 5, 10, 12
        ]
        assert trajectory["final_query_answers"] == (
            trajectory["full_state_audits"][-1]["query_answers"]
        )
        assert len(trajectory["final_table_records"]) == 2
        assert trajectory["final_table_sha256"] == (
            trajectory["full_state_audits"][-1]["table_sha256"]
        )

    aggregate = probe.aggregate_results(
        [run], {"passed": True}, [40], steps=12
    )
    payload = {
        "protocol": {
            "n_records": 2,
            "seeds": [40],
            "steps": 12,
            "tail_window": 4,
            "tau": 1.0,
        },
        "target": target.tolist(),
        "exact_oracle": {"passed": True},
        "runs": [run],
        "aggregate": aggregate,
        "public_input_sha256": {},
    }
    audit = probe.independent_audit(payload)

    assert audit["passed"] is True
    assert audit["checked_transitions"] == 24


def test_independent_audit_rebuilds_public_states_and_random_schedule(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    input_paths, marginals = _write_tiny_public_inputs(tmp_path)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        42,
        schema,
        queries,
        target,
        marginals,
        steps=8,
        tail=3,
        tau=1.0,
        verify_every=4,
    )
    aggregate = probe.aggregate_results(
        [run], {"passed": True}, [42], steps=8, classify=False
    )
    payload = {
        "formal_protocol": False,
        "protocol": {
            "n_records": 2,
            "seeds": [42],
            "steps": 8,
            "tail_window": 3,
            "tau": 1.0,
            "verify_every": 4,
        },
        "target": target.tolist(),
        "exact_oracle": probe.run_exact_oracle(),
        "runs": [run],
        "aggregate": aggregate,
        "public_input_sha256": {
            name: probe._sha256_file(path)
            for name, path in input_paths.items()
        },
    }

    audit = probe.independent_audit(payload, input_paths=input_paths)

    assert audit["passed"] is True
    assert audit["checked_public_initial_states"] == 1
    assert audit["checked_random_schedules"] == 1
    assert audit["checked_final_tables"] == 2


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (
            lambda payload: payload["runs"][0]["coordinate_history"].__setitem__(
                0,
                (payload["runs"][0]["coordinate_history"][0] + 1) % 4,
            ),
            "random_schedule_replay_mismatch",
        ),
        (
            lambda payload: payload["runs"][0]["candidate"][
                "final_table_records"
            ][0].__setitem__("a", 2),
            "final_table_audit_failure",
        ),
    ],
)
def test_independent_audit_detects_public_state_or_random_tampering(
    monkeypatch, tmp_path, mutate, reason
):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    input_paths, marginals = _write_tiny_public_inputs(tmp_path)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        43,
        schema,
        queries,
        target,
        marginals,
        steps=5,
        tail=2,
        tau=1.0,
        verify_every=5,
    )
    payload = {
        "protocol": {
            "n_records": 2,
            "seeds": [43],
            "steps": 5,
            "tail_window": 2,
            "tau": 1.0,
            "verify_every": 5,
        },
        "target": target.tolist(),
        "exact_oracle": probe.run_exact_oracle(),
        "runs": [run],
        "aggregate": probe.aggregate_results(
            [run], {"passed": True}, [43], steps=5
        ),
        "public_input_sha256": {
            name: probe._sha256_file(path)
            for name, path in input_paths.items()
        },
    }
    mutate(payload)

    audit = probe.independent_audit(payload, input_paths=input_paths)

    assert audit["passed"] is False
    assert any(
        failure.get("reason") == reason for failure in audit["failures"]
    )


def test_independent_audit_detects_sparse_delta_tampering(monkeypatch):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        41,
        schema,
        queries,
        target,
        None,
        steps=6,
        tail=2,
        tau=1.0,
        verify_every=3,
    )
    payload = {
        "protocol": {
            "n_records": 2,
            "seeds": [41],
            "steps": 6,
            "tail_window": 2,
            "tau": 1.0,
        },
        "target": target.tolist(),
        "exact_oracle": {"passed": True},
        "runs": [run],
        "aggregate": probe.aggregate_results(
            [run], {"passed": True}, [41], steps=6
        ),
        "public_input_sha256": {},
    }
    tampered = copy.deepcopy(payload)
    tampered["runs"][0]["candidate"]["loss_history"][1] += 1.0

    audit = probe.independent_audit(tampered)

    assert audit["passed"] is False
    assert any(
        failure.get("reason") == "recomputed_loss_mismatch"
        for failure in audit["failures"]
    )


@pytest.mark.parametrize(
    "candidate,gate,expected",
    [
        (90.0, True, "supports_persistent_heatbath_smoke"),
        (97.0, True, "persistent_heatbath_smoke_inconclusive"),
        (101.0, True, "persistent_heatbath_smoke_not_supported"),
        (90.0, False, "implementation_or_experiment_failure"),
    ],
)
def test_classification_follows_preregistered_seed_level_rules(
    candidate, gate, expected
):
    runs = [
        _fake_run(seed, 100.0, candidate, gate=gate)
        for seed in range(40, 60)
    ]

    aggregate = probe.aggregate_results(
        runs, {"passed": True}, list(range(40, 60)), steps=10
    )

    assert aggregate["classification"] == expected
    assert aggregate["primary"]["candidate_minus_baseline"]["wins"] == (
        20 if candidate < 100.0 else 0
    )


def test_custom_protocol_never_receives_formal_classification():
    runs = [_fake_run(seed, 100.0, 90.0) for seed in range(40, 60)]

    aggregate = probe.aggregate_results(
        runs,
        {"passed": True},
        list(range(40, 60)),
        steps=10,
        classify=False,
    )

    assert aggregate["classification"] == (
        "exploratory_protocol_no_formal_classification"
    )


def test_trajectory_metric_tail_excludes_initial_state_and_counts_directions():
    trajectory = {
        "loss_history": [10.0, 8.0, 9.0, 9.0, 7.0],
        "gain_history": [2.0, -1.0, 0.0, 2.0],
        "changed_history": [True, True, False, True],
        "conditional_expected_state_gain_history": [0.0] * 4,
        "expected_gain_over_reference_history": [0.0] * 4,
        "conditional_normalized_entropy_history": [1.0] * 4,
        "uphill_probability_mass_history": [0.2] * 4,
        "probability_min_history": [0.1] * 4,
        "probability_max_history": [0.9] * 4,
        "candidate_state_evaluations_history": [2] * 4,
        "query_indicator_evaluations_history": [6] * 4,
        "full_state_audits": [],
        "kernel_elapsed_sec": 0.4,
    }

    metrics = probe._trajectory_metrics(trajectory, tail=2)

    assert metrics["tail_mean_loss"] == 8.0
    assert metrics["trajectory_mean_loss"] == 8.25
    assert metrics["positive_steps"] == 2
    assert metrics["negative_steps"] == 1
    assert metrics["zero_steps"] == 1
    assert metrics["changed_value_rate"] == 0.75
    assert metrics["full_table_state_audits"] == 0


def test_atomic_writer_refuses_overwrite_and_cleans_invalid_temp(tmp_path):
    output = tmp_path / "result.json"
    probe._write_json_atomic(output, {"value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    with pytest.raises(FileExistsError):
        probe._write_json_atomic(output, {"value": 2})
    assert json.loads(output.read_text(encoding="utf-8")) == {"value": 1}

    invalid = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        probe._write_json_atomic(invalid, {"value": np.nan})
    assert not invalid.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_formal_protocol_match_requires_every_frozen_field():
    args = type("Args", (), {
        "seeds": list(probe.FORMAL_SEEDS),
        "steps": probe.FORMAL_STEPS,
        "tail": probe.FORMAL_TAIL,
        "tau": probe.FORMAL_TAU,
        "verify_every": probe.FORMAL_VERIFY_EVERY,
        "device": probe.FORMAL_DEVICE,
    })()
    assert probe._formal_protocol_matches(args) is True

    args.steps -= 1
    assert probe._formal_protocol_matches(args) is False


def test_formal_payload_match_requires_protocol_and_no_selection():
    payload = {
        "protocol": {
            "n_records": probe.N_RECORDS,
            "seeds": list(probe.FORMAL_SEEDS),
            "steps": probe.FORMAL_STEPS,
            "tail_window": probe.FORMAL_TAIL,
            "tau": probe.FORMAL_TAU,
            "verify_every": probe.FORMAL_VERIFY_EVERY,
            "device": probe.FORMAL_DEVICE,
            "acceptance_or_checkpoint_selection": False,
        }
    }
    assert probe._formal_payload_matches(payload) is True

    payload["protocol"]["acceptance_or_checkpoint_selection"] = True
    assert probe._formal_payload_matches(payload) is False
