"""L1 持久热浴阶段 I 脚本的配对、重放与协议测试。"""

import copy
import json

import numpy as np
import pytest

import scripts.probe_l1_persistent_heatbath as probe
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


def _write_public_inputs(tmp_path):
    schema_path = tmp_path / "schema.yaml"
    query_path = tmp_path / "queries.json"
    marginals_path = tmp_path / "marginals.json"
    schema_path.write_text(json.dumps({
        "attributes": [
            {"name": "a", "type": "categorical", "values": [0, 1]},
            {"name": "b", "type": "categorical", "values": [0, 1]},
        ]
    }), encoding="utf-8")
    _, queries, target = _tiny_problem()
    query_path.write_text(json.dumps({
        "queries": [
            {**query, "result": int(result)}
            for query, result in zip(queries, target)
        ]
    }), encoding="utf-8")
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
    paths = {
        "schema": schema_path,
        "queries": query_path,
        "marginals": marginals_path,
    }
    return paths, marginals


def _payload(run, paths, target, *, formal=False):
    oracle = probe.run_exact_oracle()
    aggregate = probe.aggregate_results(
        [run], oracle, [run["seed"]], run["n_steps"], formal=formal
    )
    return {
        "experiment": "l1_persistent_workload_heatbath",
        "formal_protocol": formal,
        "protocol": {
            "n_records": 2,
            "seeds": [run["seed"]],
            "steps": run["n_steps"],
            "tail_window": run["tail_window"],
            "tau": run["tau"],
            "verify_every": 4,
            "device": "cpu",
            "baseline_energy_mode": "squared",
            "candidate_energy_mode": "normalized_l1",
            "acceptance_or_checkpoint_selection": False,
        },
        "public_input_sha256": {
            name: probe.common._sha256_file(path)
            for name, path in paths.items()
        },
        "target": target.tolist(),
        "exact_oracle": oracle,
        "runs": [run],
        "aggregate": aggregate,
    }


def test_l1_exact_oracle_passes_all_semantic_gates():
    oracle = probe.run_exact_oracle()

    assert oracle["passed"] is True
    assert oracle["states"] == 16
    assert oracle["query_increment_max_error"] <= 1e-12
    assert oracle["energy_identity_max_error"] <= 1e-12
    assert oracle["conditional_probability_max_error"] <= 1e-12
    assert oracle["expected_energy_monotonic"] is True
    assert oracle["derivative_identity_max_error"] <= 1e-9
    for result in oracle["by_inverse_energy_scale"]:
        assert result["minimum_positive_transition"] > 0.0
        assert result["all_state_self_loops_positive"] is True
        assert result["irreducible"] is True
        assert result["detailed_balance_max_error"] <= 1e-12
        assert result["stationarity_max_error"] <= 1e-12


def test_run_seed_pairs_square_and_l1_with_shared_randomness(monkeypatch):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        60,
        schema,
        queries,
        target,
        None,
        steps=12,
        tail=4,
        tau=1.0,
        verify_every=5,
    )

    assert run["baseline"]["energy_mode"] == "squared"
    assert run["candidate"]["energy_mode"] == "normalized_l1"
    assert run["gates"]["random_inputs_aligned"] is True
    assert run["gates"]["random_rng_endpoints_aligned"] is True
    assert run["gates"]["all_full_state_audits_exact"] is True
    assert run["gates"]["energy_identity_max_error"] <= 1e-12
    assert run["gates"]["all_probabilities_strictly_positive"] is True
    assert len(run["coordinate_history"]) == 12
    for variant in ("baseline", "candidate"):
        trajectory = run[variant]
        assert len(trajectory["energy_history"]) == 13
        assert len(trajectory["normalized_l1_history"]) == 13
        assert [row["step"] for row in trajectory["full_state_audits"]] == [
            0, 5, 10, 12
        ]
        assert trajectory["final_query_answers"] == (
            trajectory["full_state_audits"][-1]["query_answers"]
        )


def test_independent_audit_replays_both_energy_modes(monkeypatch, tmp_path):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    paths, marginals = _write_public_inputs(tmp_path)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        61,
        schema,
        queries,
        target,
        marginals,
        steps=8,
        tail=3,
        tau=1.0,
        verify_every=4,
    )
    payload = _payload(run, paths, target)

    audit = probe.independent_audit(payload, input_paths=paths)

    assert audit["passed"] is True
    assert audit["checked_seed_trajectories"] == 2
    assert audit["checked_transitions"] == 16
    assert audit["checked_final_tables"] == 2


@pytest.mark.parametrize(
    "mutate,field",
    [
        (
            lambda payload: payload["runs"][0]["candidate"][
                "energy_history"
            ].__setitem__(1, 999.0),
            "energy_history",
        ),
        (
            lambda payload: payload["runs"][0]["coordinate_history"].__setitem__(
                0,
                (payload["runs"][0]["coordinate_history"][0] + 1) % 4,
            ),
            None,
        ),
    ],
)
def test_independent_audit_detects_tampering(
    monkeypatch, tmp_path, mutate, field
):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    paths, marginals = _write_public_inputs(tmp_path)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        62,
        schema,
        queries,
        target,
        marginals,
        steps=6,
        tail=2,
        tau=1.0,
        verify_every=4,
    )
    payload = _payload(run, paths, target)
    tampered = copy.deepcopy(payload)
    mutate(tampered)

    audit = probe.independent_audit(tampered, input_paths=paths)

    assert audit["passed"] is False
    if field is not None:
        assert any(
            failure.get("field") == field
            for failure in audit["failures"]
        )


def test_generation_aggregate_waits_for_offline_quality(monkeypatch):
    monkeypatch.setattr(probe, "N_RECORDS", 2)
    schema, queries, target = _tiny_problem()
    run = probe.run_seed(
        63,
        schema,
        queries,
        target,
        None,
        steps=5,
        tail=2,
        tau=1.0,
        verify_every=5,
    )

    formal = probe.aggregate_results(
        [run], probe.run_exact_oracle(), [63], 5, formal=True
    )
    exploratory = probe.aggregate_results(
        [run], probe.run_exact_oracle(), [63], 5, formal=False
    )

    assert formal["classification"] == (
        "generation_complete_pending_offline_quality"
    )
    assert formal["all_diagnostic_gates_passed"] is True
    assert "final_energy" not in formal["paired_metrics"]
    assert "optimized_energy_by_variant" in formal
    assert formal["paired_metrics"][
        "conditional_normalized_entropy_mean"
    ]["candidate_minus_baseline"]["lower_is_better"] is False
    assert exploratory["classification"] == (
        "exploratory_protocol_no_formal_classification"
    )


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

    args.seeds = args.seeds[:-1]
    assert probe._formal_protocol_matches(args) is False


def test_run_seed_rejects_invalid_protocol_before_initialization():
    schema, queries, target = _tiny_problem()
    with pytest.raises(ValueError, match="steps"):
        probe.run_seed(
            64,
            schema,
            queries,
            target,
            None,
            steps=0,
            tail=1,
            tau=1.0,
            verify_every=1,
        )
