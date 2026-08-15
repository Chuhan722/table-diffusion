import json

import pytest

from scripts import audit_issue49_stage_a as stage_a_auditor
from scripts import audit_issue49_stage_b as auditor
from scripts import run_issue49_stage_a as stage_a_runner
from scripts import run_issue49_stage_b as runner


def _summary(mean, *, median=None):
    value = float(mean)
    return {
        "mean": value,
        "std": 0.0,
        "median": value if median is None else float(median),
        "min": value,
        "max": value,
        "values": [value] * 10,
    }


def _aggregate(spec, late):
    return {
        "config": dict(spec),
        "trajectory_count": 10,
        "seeds": list(range(100, 110)),
        "all_rounds_complete": True,
        "metrics": {
            "late_window_current_loss": _summary(late),
            "current_loss_auc": _summary(late * 1000.0),
            "final_current_loss": _summary(late + 1.0),
        },
        "total_clip_hit_count": 0,
        "all_conditionals_finite_and_bidirectional": True,
    }


def _selection_fixture():
    protocol = runner._protocol("formal")
    configs = [
        runner._config("independent", tau, 0)
        for tau in protocol["independent_temperatures"]
    ]
    configs.extend([
        runner._config("factor", 4.0, 8),
        runner._config("factor", 5.0, 16),
    ])
    late = {
        "independent_tau_4": 60.0,
        "independent_tau_5": 50.0,
        "independent_tau_6": 70.0,
        "independent_tau_7": 80.0,
        "independent_tau_8": 90.0,
        "factor_tau_4_sweeps_8": 45.0,
        "factor_tau_5_sweeps_16": 40.0,
    }
    aggregates = {
        spec["config_id"]: _aggregate(spec, late[spec["config_id"]])
        for spec in configs
    }
    comparisons = {
        spec["config_id"]: {
            "mean_difference": -5.0,
            "median_difference": -4.0,
            "wins": 7,
        }
        for spec in configs if spec["kernel"] == "factor"
    }
    upstream = {
        "a0": {
            "classification": {
                "temperatures": {
                    runner.stage_a._tau_key(tau): {
                        "eligible_for_mixing": True
                    }
                    for tau in protocol["independent_temperatures"]
                }
            }
        }
    }
    return protocol, configs, aggregates, comparisons, upstream


def test_stage_b_protocol_freezes_formal_and_smoke_win_counts():
    formal = runner._protocol("formal")
    smoke = runner._protocol("smoke")

    assert formal["stage_b_seeds"] == list(range(100, 110))
    assert formal["snapshot_rounds"] == [0, 500, 1000]
    assert formal["minimum_paired_wins"] == 6
    assert smoke["stage_b_seeds"] == [99]
    assert smoke["snapshot_rounds"] == [0, 6, 12]
    assert smoke["minimum_paired_wins"] == 1
    assert formal["factor_builder"] == "legacy_rowwise"
    assert formal["self_review_probe_mu"] == 0.0
    assert formal["self_review_max_active_attributes"] == 12
    assert formal["self_review_required_groups"] == [
        "global", "initial", "mid", "late"
    ]
    assert formal["ranking_tie_breakers"][-2:] == [
        "fewer_sweeps", "lower_temperature"
    ]


def test_stage_b_refuses_all_a0_ineligible_before_trajectories():
    protocol = stage_a_runner._protocol("formal")
    temperatures = {
        stage_a_runner._tau_key(tau): {"eligible_for_mixing": False}
        for tau in protocol["evaluation_temperatures"]
    }
    upstream = {
        "protocol": protocol,
        "a0": {
            "classification": {
                "temperatures": temperatures,
                "eligible_temperatures": [],
            }
        },
    }

    with pytest.raises(RuntimeError, match="不得运行 Stage B"):
        runner._all_configs(runner._protocol("formal"), upstream)
    with pytest.raises(RuntimeError, match="不得存在 Stage B 报告"):
        auditor._expected_configs(upstream)


def test_g0_review_failure_does_not_promote_factor_runner_up():
    protocol, configs, aggregates, comparisons, upstream = (
        _selection_fixture()
    )
    preliminary = runner._preliminary_selection(
        configs, aggregates, comparisons, upstream, protocol
    )

    assert preliminary["i_star"] == "independent_tau_5"
    assert preliminary["g0"] == "factor_tau_5_sweeps_16"

    selection = runner._final_selection(
        preliminary,
        {
            "applicable": True,
            "status": "failed_no_factor_fallback",
            "passed": False,
        },
        configs,
        aggregates,
    )

    assert selection["g_star"] is None
    assert selection["unique_candidate"] == "independent_tau_5"
    assert selection["unique_candidate"] != "factor_tau_4_sweeps_8"
    assert selection["no_runner_up_after_g0_review_failure"] is True


def test_factor_can_freeze_only_after_g0_self_review_passes():
    protocol, configs, aggregates, comparisons, upstream = (
        _selection_fixture()
    )
    preliminary = runner._preliminary_selection(
        configs, aggregates, comparisons, upstream, protocol
    )
    selection = runner._final_selection(
        preliminary,
        {"applicable": True, "status": "passed", "passed": True},
        configs,
        aggregates,
    )

    assert selection["g_star"] == "factor_tau_5_sweeps_16"
    assert selection["unique_candidate"] == "factor_tau_5_sweeps_16"
    assert selection["status"] == "factor_candidate_frozen"


def test_g0_self_review_result_is_independently_recomputed():
    protocol = runner._protocol("smoke")
    target, queries, schema, marginals, _ = stage_a_runner._load_inputs()
    configs = [runner._config("factor", 4.0, 8)]
    rows = runner._run_trajectories(
        target, queries, schema, marginals, protocol, configs
    )
    identity = runner._trajectory_identity_gates(
        rows, configs, protocol
    )
    review = runner._self_review(
        configs[0]["config_id"],
        rows,
        configs,
        target,
        queries,
        schema,
        protocol,
    )

    recomputed = auditor._self_review(
        review,
        configs[0]["config_id"],
        rows,
        configs,
        protocol,
        target,
        queries,
        schema,
    )

    assert identity["all_trajectory_identity_gates_passed"] is True
    assert rows[0]["run"]["factor_conditional_logit_diagnostics"][
        "condition_count"
    ] == rows[0]["run"]["gibbs_microsteps"]
    assert recomputed == review
    assert review["applicable"] is True
    assert review["state_count"] == 3
    assert review["passed"] is False
    assert review["status"] == "failed_no_factor_fallback"


def test_factor_clip_hit_excludes_only_that_factor_from_g0():
    protocol, configs, aggregates, comparisons, upstream = (
        _selection_fixture()
    )
    aggregates["factor_tau_5_sweeps_16"]["total_clip_hit_count"] = 1

    preliminary = runner._preliminary_selection(
        configs, aggregates, comparisons, upstream, protocol
    )

    assert preliminary["g0"] == "factor_tau_4_sweeps_8"
    assert preliminary["factor"]["factor_tau_5_sweeps_16"][
        "eligible_for_g0"
    ] is False


def test_stage_b_smoke_is_reloadable_auditable_and_non_overwriting(tmp_path):
    stage_a_dir = tmp_path / "stage_a"
    stage_a_report_path, library_path, _ = stage_a_runner.run_stage_a(
        "smoke", stage_a_dir
    )
    stage_a_audit_path, _ = stage_a_auditor.audit_stage_a(
        stage_a_report_path,
        library_path,
        stage_a_dir / "stage_t_a_audit.json",
    )
    stage_b_dir = tmp_path / "stage_b"
    report_path, report = runner.run_stage_b(
        "smoke", stage_a_report_path, stage_a_audit_path, stage_b_dir
    )

    assert report["formal_result_valid"] is False
    assert report["interpretation"] == "pipeline_smoke_only_not_evidence"
    assert report["protocol"]["stage_b_seeds"] == [99]
    assert len(report["configurations"]) >= 5
    assert report["trajectory_identity_gates"][
        "all_trajectory_identity_gates_passed"
    ] is True
    assert report["selection"]["unique_candidate"] is not None

    audit_path, audit = auditor.audit_stage_b(
        report_path, stage_b_dir / "stage_b_audit.json"
    )
    assert audit_path.exists()
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert audit["selection"] == report["selection"]

    with pytest.raises(FileExistsError, match="尚未启动任何 Stage B 轨迹"):
        runner.run_stage_b(
            "smoke", stage_a_report_path, stage_a_audit_path, stage_b_dir
        )

    tampered = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = tampered["selection"]["unique_candidate"]
    tampered["aggregates"][candidate]["metrics"][
        "late_window_current_loss"
    ]["mean"] += 1.0
    tampered_path = tmp_path / "tampered_stage_b_report.json"
    tampered_path.write_text(
        json.dumps(tampered, ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="report.aggregates"):
        auditor.audit_stage_b(
            tampered_path, tmp_path / "tampered_stage_b_audit.json"
        )
