import json

import pytest

from scripts import audit_issue49_confirmation as auditor
from scripts import audit_issue49_stage_a as stage_a_auditor
from scripts import audit_issue49_stage_b as stage_b_auditor
from scripts import run_issue49_confirmation as runner
from scripts import run_issue49_stage_a as stage_a_runner
from scripts import run_issue49_stage_b as stage_b_runner


def _stage_b_selection(candidate):
    candidate_id = candidate["config_id"] if candidate else None
    independent = {
        stage_b_runner._config_id("independent", tau, 0): {
            "eligible": True
        }
        for tau in (4.0, 5.0, 6.0, 7.0, 8.0)
    }
    return {
        "unique_candidate": candidate_id,
        "unique_candidate_config": candidate,
        "i_star": (
            candidate_id
            if candidate and candidate["kernel"] == "independent"
            else "independent_tau_5"
        ),
        "g_star": (
            candidate_id
            if candidate and candidate["kernel"] == "factor" else None
        ),
        "independent": independent,
        "self_state_review": {
            "passed": bool(candidate and candidate["kernel"] == "factor")
        },
    }


def _stage_b_report(candidate):
    return {"selection": _stage_b_selection(candidate)}


def _paired_rows(candidate_id, baseline_id, differences, seeds):
    rows = []
    for seed, difference in zip(seeds, differences):
        rows.extend([
            {
                "config_id": candidate_id,
                "run": {
                    "seed": seed,
                    "late_window_current_loss_mean": 100.0 + difference,
                },
            },
            {
                "config_id": baseline_id,
                "run": {
                    "seed": seed,
                    "late_window_current_loss_mean": 100.0,
                },
            },
        ])
    return rows


def _decision_aggregate(config_id, mean):
    return {
        "config": {"config_id": config_id},
        "metrics": {"late_window_current_loss": {"mean": float(mean)}},
    }


def test_confirmation_protocol_freezes_new_seeds_and_full_tau_grid():
    formal = runner._protocol("formal")
    smoke = runner._protocol("smoke")

    assert formal["confirmation_seeds"] == list(range(110, 120))
    assert formal["rounds"] == 1000
    assert formal["snapshot_rounds"] == [0, 500, 1000]
    assert formal["independent_temperatures"] == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert formal["minimum_paired_wins"] == 6
    assert formal["paired_t_critical_95_df9"] == pytest.approx(
        2.2621571627409915
    )
    assert smoke["confirmation_seeds"] == [99]
    assert smoke["rounds"] == 12
    assert smoke["minimum_paired_wins"] == 1


def test_confirmation_grid_follows_frozen_candidate_instead_of_tau_four():
    protocol = runner._protocol("formal")
    independent_candidate = stage_b_runner._config("independent", 6.0, 0)
    configs, candidate = runner._confirmation_configs(
        protocol, _stage_b_report(independent_candidate)
    )

    assert candidate["config_id"] == "independent_tau_6"
    assert [row["config_id"] for row in configs] == [
        "independent_tau_4",
        "independent_tau_5",
        "independent_tau_6",
        "independent_tau_7",
        "independent_tau_8",
    ]

    factor_candidate = stage_b_runner._config("factor", 5.0, 16)
    configs, candidate = runner._confirmation_configs(
        protocol, _stage_b_report(factor_candidate)
    )
    assert candidate == factor_candidate
    assert len(configs) == 6
    assert configs[-1] == factor_candidate


def test_confirmation_refuses_to_run_without_a_frozen_candidate():
    with pytest.raises(RuntimeError, match="未冻结唯一候选"):
        runner._confirmation_configs(
            runner._protocol("formal"), _stage_b_report(None)
        )


def test_factor_comparison_deduplicates_same_tau_i_star():
    protocol = runner._protocol("formal")
    candidate = stage_b_runner._config("factor", 5.0, 16)
    selection = _stage_b_selection(candidate)
    selection["i_star"] = "independent_tau_5"

    plan = runner._comparison_plan(candidate, selection, protocol)

    assert plan == [{
        "baseline_config_id": "independent_tau_5",
        "roles": ["same_temperature_independent", "stage_b_i_star"],
    }]


def test_formal_paired_t_interval_is_computed_and_gated():
    protocol = runner._protocol("formal")
    candidate_id = "independent_tau_6"
    baseline_id = "independent_tau_8"
    differences = [-5.0, -4.0, -6.0, -5.5, -4.5] * 2
    rows = _paired_rows(
        candidate_id,
        baseline_id,
        differences,
        protocol["confirmation_seeds"],
    )

    result = runner._paired_comparison(
        rows,
        candidate_id,
        baseline_id,
        ["old_grid_incumbent_independent_tau_8"],
        protocol,
    )

    assert result["sample_size"] == 10
    assert result["wins"] == 10
    assert result["confidence_interval"][1] < 0.0
    assert result["passed"] is True

    candidate = stage_b_runner._config("independent", 6.0, 0)
    aggregates = {
        candidate_id: _decision_aggregate(candidate_id, 50.0),
        baseline_id: _decision_aggregate(baseline_id, 55.0),
    }
    eligibility = {
        config_id: {
            "eligible": True,
            "gates": {
                "all_confirmation_seeds_retained": True,
                "all_confirmation_rounds_complete": True,
            },
        }
        for config_id in aggregates
    }
    decision = runner._confirmation_decision(
        "formal",
        candidate,
        aggregates,
        eligibility,
        {baseline_id: result},
        {"applicable": False, "passed": None},
        {"all_trajectory_identity_gates_passed": True},
        True,
    )
    assert decision["confirmed"] is True
    assert decision["status"] == "confirmed"


def test_formal_t_interval_can_reject_mean_median_and_six_wins():
    protocol = runner._protocol("formal")
    candidate_id = "independent_tau_6"
    baseline_id = "independent_tau_8"
    rows = _paired_rows(
        candidate_id,
        baseline_id,
        [-10.0] * 6 + [9.0] * 4,
        protocol["confirmation_seeds"],
    )

    result = runner._paired_comparison(
        rows,
        candidate_id,
        baseline_id,
        ["old_grid_incumbent_independent_tau_8"],
        protocol,
    )

    assert result["mean_difference"] < 0.0
    assert result["median_difference"] < 0.0
    assert result["wins"] == 6
    assert result["confidence_interval"][1] > 0.0
    assert result["passed"] is False


def test_smoke_never_claims_a_t_interval_or_confirmation():
    protocol = runner._protocol("smoke")
    candidate = stage_b_runner._config("independent", 6.0, 0)
    baseline_id = "independent_tau_8"
    rows = _paired_rows(
        candidate["config_id"],
        baseline_id,
        [-5.0],
        protocol["confirmation_seeds"],
    )
    comparison = runner._paired_comparison(
        rows,
        candidate["config_id"],
        baseline_id,
        ["old_grid_incumbent_independent_tau_8"],
        protocol,
    )
    aggregates = {
        candidate["config_id"]: _decision_aggregate(
            candidate["config_id"], 50.0
        ),
        baseline_id: _decision_aggregate(baseline_id, 55.0),
    }
    eligibility = {
        config_id: {
            "eligible": True,
            "gates": {
                "all_confirmation_seeds_retained": True,
                "all_confirmation_rounds_complete": True,
            },
        }
        for config_id in aggregates
    }
    decision = runner._confirmation_decision(
        "smoke",
        candidate,
        aggregates,
        eligibility,
        {baseline_id: comparison},
        {"applicable": False, "passed": None},
        {"all_trajectory_identity_gates_passed": True},
        False,
    )

    assert comparison["confidence_interval"] is None
    assert comparison["passed"] is False
    assert decision["confirmed"] is None
    assert decision["status"] == "smoke_only_not_evidence"


def test_failed_confirmation_keeps_candidate_and_never_uses_runner_up():
    candidate = stage_b_runner._config("independent", 6.0, 0)
    aggregates = {
        "independent_tau_5": _decision_aggregate(
            "independent_tau_5", 40.0
        ),
        candidate["config_id"]: _decision_aggregate(
            candidate["config_id"], 50.0
        ),
    }
    eligibility = {
        config_id: {
            "eligible": True,
            "gates": {
                "all_confirmation_seeds_retained": True,
                "all_confirmation_rounds_complete": True,
            },
        }
        for config_id in aggregates
    }
    decision = runner._confirmation_decision(
        "formal",
        candidate,
        aggregates,
        eligibility,
        {},
        {"applicable": False, "passed": None},
        {"all_trajectory_identity_gates_passed": True},
        True,
    )

    assert decision["confirmed"] is False
    assert decision["status"] == "not_confirmed_no_reselection"
    assert decision["frozen_candidate"] == "independent_tau_6"
    assert decision["lowest_eligible_config_ids"] == ["independent_tau_5"]
    assert decision["no_runner_up_after_failure"] is True


def test_invalid_formal_identity_cannot_claim_confirmation():
    candidate = stage_b_runner._config("independent", 8.0, 0)
    aggregates = {
        candidate["config_id"]: _decision_aggregate(
            candidate["config_id"], 40.0
        )
    }
    eligibility = {
        candidate["config_id"]: {
            "eligible": True,
            "gates": {
                "all_confirmation_seeds_retained": True,
                "all_confirmation_rounds_complete": True,
            },
        }
    }

    decision = runner._confirmation_decision(
        "formal",
        candidate,
        aggregates,
        eligibility,
        {},
        {"applicable": False, "passed": None},
        {"all_trajectory_identity_gates_passed": True},
        False,
    )

    assert decision["confirmed"] is None
    assert decision["formal_confirmation_assessed"] is False
    assert decision["status"] == "invalid_formal_identity_not_assessed"


def test_factor_candidate_self_review_wrapper_is_independently_recomputed():
    protocol = runner._protocol("smoke")
    target, queries, schema, marginals, _ = stage_a_runner._load_inputs()
    candidate = stage_b_runner._config("factor", 4.0, 8)
    configs = [candidate]
    rows = runner._run_trajectories(
        target, queries, schema, marginals, protocol, configs
    )

    review = runner._candidate_self_review(
        candidate,
        rows,
        configs,
        target,
        queries,
        schema,
        protocol,
    )
    recomputed = auditor._candidate_self_review(
        review,
        candidate,
        rows,
        configs,
        protocol,
        target,
        queries,
        schema,
    )

    assert recomputed == review
    assert review["applicable"] is True
    assert review["candidate_config_id"] == candidate["config_id"]
    assert review["state_count"] == 3
    assert review["status"] in {
        "passed", "failed_confirmation_no_reselection"
    }


def test_confirmation_smoke_is_auditable_and_non_overwriting(tmp_path):
    stage_a_dir = tmp_path / "stage_a"
    stage_a_report, library, _ = stage_a_runner.run_stage_a(
        "smoke", stage_a_dir
    )
    stage_a_audit, _ = stage_a_auditor.audit_stage_a(
        stage_a_report, library, stage_a_dir / "stage_t_a_audit.json"
    )
    stage_b_dir = tmp_path / "stage_b"
    stage_b_report, _ = stage_b_runner.run_stage_b(
        "smoke", stage_a_report, stage_a_audit, stage_b_dir
    )
    stage_b_audit, _ = stage_b_auditor.audit_stage_b(
        stage_b_report, stage_b_dir / "stage_b_audit.json"
    )
    confirmation_dir = tmp_path / "confirmation"
    report_path, report = runner.run_confirmation(
        "smoke", stage_b_report, stage_b_audit, confirmation_dir
    )

    assert report["formal_result_valid"] is False
    assert report["decision"]["confirmed"] is None
    assert report["decision"]["status"] == "smoke_only_not_evidence"
    assert report["protocol"]["confirmation_seeds"] == [99]
    assert [
        row["temperature"] for row in report["configurations"][:5]
    ] == [4.0, 5.0, 6.0, 7.0, 8.0]
    assert report["frozen_candidate"]["config_id"] == report[
        "decision"
    ]["frozen_candidate"]

    audit_path, audit = auditor.audit_confirmation(
        report_path, confirmation_dir / "confirmation_audit.json"
    )
    assert audit_path.exists()
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert audit["decision"] == report["decision"]

    with pytest.raises(FileExistsError, match="尚未启动任何确认轨迹"):
        runner.run_confirmation(
            "smoke", stage_b_report, stage_b_audit, confirmation_dir
        )

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    candidate_id = tampered["frozen_candidate"]["config_id"]
    tampered["aggregates"][candidate_id]["metrics"][
        "late_window_current_loss"
    ]["mean"] += 1.0
    tampered_path = tmp_path / "tampered_confirmation_report.json"
    tampered_path.write_text(
        json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="report.aggregates"):
        auditor.audit_confirmation(
            tampered_path, tmp_path / "tampered_confirmation_audit.json"
        )
