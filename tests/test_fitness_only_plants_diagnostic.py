"""Tests for the plants medium-scale diagnostic protocol."""

import math
from pathlib import Path

import pytest

from scripts import run_fitness_only_plants_diagnostic as diagnostic


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("诊断 plan 不得读取输入或结果")

    monkeypatch.setattr(diagnostic, "_sha256_file", forbidden)
    monkeypatch.setattr(diagnostic, "_load_json_object", forbidden)
    plan = diagnostic.build_plan()
    assert plan["protocol_sha256"] == diagnostic.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 2
    assert plan["generation_started"] is False
    protocol = plan["protocol"]
    assert protocol["dataset"] == "plants"
    assert protocol["n_records"] == 17412
    assert protocol["generation_input"]["expected_query_count"] == 980
    assert protocol["shared_generation_config"]["configuration_source"] == (
        "v3_frozen_configuration_verbatim_no_retuning"
    )
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diagnostic.protocol_sha256() == diagnostic.FROZEN_PROTOCOL_SHA256
    assert diagnostic.assert_frozen_protocol_identity() == (
        diagnostic.FROZEN_PROTOCOL_SHA256
    )


def test_generation_input_audit_pins_measured_workload():
    audit = diagnostic._audit_generation_inputs(Path("."))
    assert audit["query_count"] == 980
    assert len(audit["queries"]) == len(audit["targets"]) == 980
    assert len({
        diagnostic.query_fingerprint(query) for query in audit["queries"]
    }) == 980
    assert all(
        len(query["conditions"]) >= 2 for query in audit["queries"]
    )
    assert audit["input_sha256"] == {
        key: diagnostic.INPUT_SHA256[key]
        for key in ("schema", "marginals", "measured")
    }
    assert all(
        math.isfinite(target) and target >= 0
        for target in audit["targets"]
    )


def test_wrong_confirmation_fails_before_output_creation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(diagnostic, "_repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="protocol SHA-256"):
        diagnostic.run("wrong")


def test_existing_output_is_not_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(diagnostic, "OUTPUT_DIR", Path("existing"))
    with pytest.raises(FileExistsError, match="不覆盖"):
        diagnostic.run(diagnostic.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_comparison_reports_residual_minus_equal():
    values = {
        "residual": {
            "measured": {"normalized_l1_mean": 1.0},
            "one_way_safety": {"normalized_l1_mean": 3.0},
            "heldout": {
                "3way": {"normalized_l1_mean": 4.0},
                "4way": {"normalized_l1_mean": 5.0},
                "combined": {"normalized_l1_mean": 4.5},
            },
        },
        "equal": {
            "measured": {"normalized_l1_mean": 1.5},
            "one_way_safety": {"normalized_l1_mean": 2.0},
            "heldout": {
                "3way": {"normalized_l1_mean": 3.0},
                "4way": {"normalized_l1_mean": 6.0},
                "combined": {"normalized_l1_mean": 4.5},
            },
        },
    }
    result = diagnostic._comparison(values)
    deltas = result["delta_residual_minus_equal"]
    assert deltas["measured"]["residual_minus_equal"] == -0.5
    assert deltas["one_way_safety"]["residual_minus_equal"] == 1.0
    assert deltas["heldout_3way"]["residual_minus_equal"] == 1.0
    assert deltas["heldout_4way"]["residual_minus_equal"] == -1.0
    assert deltas["heldout_combined"]["residual_minus_equal"] == 0.0
    assert result["interpretation"] == (
        "diagnostic_only_lower_is_better_no_promotion_gate"
    )


def test_expected_rho_schedule_matches_v3_three_phase_shape():
    values = diagnostic._expected_rho_schedule(diagnostic.N_ROUNDS)
    assert len(values) == 6000
    assert values[0] == pytest.approx(0.01)
    assert values[diagnostic.RHO_ANNEAL_START_ROUND] == pytest.approx(0.01)
    anneal_stop = (
        diagnostic.RHO_ANNEAL_START_ROUND + diagnostic.RHO_ANNEAL_ROUNDS
    )
    assert values[anneal_stop] == pytest.approx(0.001)
    assert values[-1] == pytest.approx(0.001)
    assert all(b <= a + 1e-15 for a, b in zip(values, values[1:]))

    audit = diagnostic._audit_rho_schedule(
        {"rho_schedule_history": list(values)}
    )
    assert audit["passed"] is True
    assert audit["max_abs_deviation"] == 0.0

    drifted = list(values)
    drifted[3000] += 1e-6
    with pytest.raises(RuntimeError, match="数值漂移"):
        diagnostic._audit_rho_schedule({"rho_schedule_history": drifted})
    with pytest.raises(RuntimeError, match="长度漂移"):
        diagnostic._audit_rho_schedule(
            {"rho_schedule_history": list(values[:-1])}
        )


def test_floor_morphology_observation_flags():
    descending = [1.0 - index * 1e-4 for index in range(6000)]
    result = diagnostic._floor_morphology(descending)
    assert result["descending"] is True
    assert result["last_best_update_round"] == 6000
    assert result["tail_window_mean_5000_6000"] < (
        result["prev_window_mean_4000_5000"]
    )
    assert result["interpretation"] == "observation_only_no_gate"

    plateau = [1.0 - index * 1e-4 for index in range(4000)]
    plateau.extend([plateau[-1]] * 2000)
    stalled = diagnostic._floor_morphology(plateau)
    assert stalled["descending"] is False
    assert stalled["last_best_update_round"] == 4000

    with pytest.raises(RuntimeError, match="长度漂移"):
        diagnostic._floor_morphology([1.0] * 5999)
