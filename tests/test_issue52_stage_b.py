import copy
import json

import pytest

from scripts import audit_issue52_stage_a_mixing as stage_a_auditor
from scripts import audit_issue52_stage_a_state_library as library_auditor
from scripts import audit_issue52_stage_b as auditor
from scripts import audit_issue52_stage_t as stage_t_auditor
from scripts import build_issue52_stage_a_state_library as library_builder
from scripts import issue52_protocol as protocol
from scripts import run_issue52_stage_a_mixing as stage_a_runner
from scripts import run_issue52_stage_b as runner
from scripts import run_issue52_stage_t as stage_t_runner
from table_diffevo.experiment_parallel import assert_scientifically_equal


def _write_payload(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def smoke_chain(tmp_path_factory):
    root = tmp_path_factory.mktemp("issue52-stage-b")
    stage_t_report_path, stage_t_report = stage_t_runner.run_stage_t(
        "smoke", root / "stage_t", max_workers=2
    )
    stage_t_audit_path, _ = stage_t_auditor.audit_stage_t(
        stage_t_report_path, root / "stage_t_audit.json"
    )
    library_path, _ = library_builder.build_state_library(
        stage_t_report_path,
        stage_t_audit_path,
        root / "state_library.json",
    )
    library_audit_path, _ = library_auditor.audit_state_library(
        stage_t_report_path,
        stage_t_audit_path,
        library_path,
        root / "state_library_audit.json",
    )
    stage_a_report_path, stage_a_report = stage_a_runner.run_stage_a_mixing(
        "smoke",
        library_path,
        library_audit_path,
        root / "stage_a",
        max_workers=2,
    )
    stage_a_audit_path, _ = stage_a_auditor.audit_stage_a_mixing(
        stage_a_report_path,
        library_path,
        library_audit_path,
        root / "stage_a_audit.json",
    )
    serial_path, serial = runner.run_stage_b(
        "smoke",
        stage_t_report_path,
        stage_t_audit_path,
        stage_a_report_path,
        stage_a_audit_path,
        root / "stage_b_serial",
        max_workers=1,
    )
    parallel_path, parallel = runner.run_stage_b(
        "smoke",
        stage_t_report_path,
        stage_t_audit_path,
        stage_a_report_path,
        stage_a_audit_path,
        root / "stage_b_parallel",
        max_workers=2,
    )
    audit_path, audit = auditor.audit_stage_b(
        parallel_path, root / "stage_b_audit.json"
    )
    return {
        "root": root,
        "stage_t_report_path": stage_t_report_path,
        "stage_t_report": stage_t_report,
        "stage_t_audit_path": stage_t_audit_path,
        "stage_a_report_path": stage_a_report_path,
        "stage_a_report": stage_a_report,
        "stage_a_audit_path": stage_a_audit_path,
        "serial_path": serial_path,
        "serial": serial,
        "parallel_path": parallel_path,
        "parallel": parallel,
        "audit_path": audit_path,
        "audit": audit,
    }


def test_formal_protocol_binds_upstreams_horizon_and_selection_rules():
    formal = protocol.stage_b_protocol("formal")
    assert formal["stage_b_seeds"] == list(range(200, 210))
    assert formal["rounds"] == 3000
    assert formal["trend_checkpoints"] == [
        500, 1000, 1500, 2000, 2500, 3000
    ]
    assert formal["late_window_size"] == 500
    assert formal["factor_builder"] == "compiled_batch"
    assert formal["expected_formal_i_star"] == "independent_tau_5"
    assert formal["expected_formal_stage_a_selection"] == {
        "minimal_sufficient_sweeps": {
            "tau_1": 8,
            "tau_2": 8,
            "tau_3": 16,
            "tau_4": None,
            "tau_5": None,
        },
        "qualified_temperatures": [1.0, 2.0, 3.0],
        "unqualified_temperatures": [4.0, 5.0],
    }
    assert formal["expected_stage_t_artifact_sha256"] == (
        protocol.FORMAL_STAGE_T_ARTIFACT_SHA256
    )
    assert formal["expected_stage_a_artifact_sha256"] == (
        protocol.FORMAL_STAGE_A_MIXING_ARTIFACT_SHA256
    )
    assert formal["no_reselection"] is True


def _comparison(candidate_mean, baseline_mean):
    primary = {
        "candidate": {"mean": candidate_mean},
        "baseline": {"mean": baseline_mean},
    }
    return {"metrics": {"late_window_current_loss": primary}}


def test_selection_uses_late_mean_then_sweeps_tau_and_two_point_gates():
    frozen = protocol.stage_b_protocol("smoke")
    configs = [
        runner._factor_config(1.0, 16),
        runner._factor_config(2.0, 8),
        runner._factor_config(3.0, 8),
    ]
    aggregates = {
        config["config_id"]: {
            "late_window_current_loss": {"mean": 10.0}
        }
        for config in configs
    }
    comparisons = {
        config["config_id"]: {
            "same_temperature_independent": _comparison(10.0, 11.0),
            "i_star": _comparison(10.0, 12.0),
        }
        for config in configs
    }

    selected = runner._selection(
        configs, aggregates, comparisons, "independent_tau_5", frozen
    )
    assert selected["g_star"] == "factor_tau_2_sweeps_8"
    assert selected["stage_c_allowed"] is True
    assert selected["status"] == "factor_candidate_selected"

    comparisons[selected["g_star"]]["i_star"] = _comparison(10.0, 9.0)
    stopped = runner._selection(
        configs, aggregates, comparisons, "independent_tau_5", frozen
    )
    assert stopped["g_star"] == selected["g_star"]
    assert stopped["stage_c_candidate"] is None
    assert stopped["status"] == "no_factor_candidate"
    assert stopped["no_reselection"] is True


def test_smoke_runs_only_stage_a_qualified_factors_and_audits(smoke_chain):
    serial = smoke_chain["serial"]
    report = smoke_chain["parallel"]
    audit = smoke_chain["audit"]
    selection = smoke_chain["stage_a_report"]["selection"]
    expected_configs = runner._factor_configs_from_selection(
        selection, protocol.stage_b_protocol("smoke")
    )

    assert report["factor_configurations"] == expected_configs
    assert report["execution"]["task_count"] == len(expected_configs)
    assert all(
        row["kernel"] == "factor"
        for row in report["stage_b"]["factor_trajectories"]
    )
    assert all(
        "independent" not in task
        for task in report["execution"]["task_order"]
    )
    assert report["independent_reference"]["i_star"].startswith(
        "independent_tau_"
    )
    assert report["stage_b"]["identity_gates"][
        "all_identity_gates_passed"
    ] is True
    assert report["formal_result_valid"] is False
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert audit["recomputed"]["selection"] == report["stage_b"][
        "selection"
    ]
    assert_scientifically_equal(
        serial["stage_b"]["factor_trajectories"],
        report["stage_b"]["factor_trajectories"],
    )

    with pytest.raises(FileExistsError, match="尚未启动任何 Stage B 轨迹"):
        runner.run_stage_b(
            "smoke",
            smoke_chain["stage_t_report_path"],
            smoke_chain["stage_t_audit_path"],
            smoke_chain["stage_a_report_path"],
            smoke_chain["stage_a_audit_path"],
            smoke_chain["parallel_path"].parent,
            max_workers=1,
        )
    with pytest.raises(FileExistsError, match="审计输出已存在"):
        auditor.audit_stage_b(
            smoke_chain["parallel_path"], smoke_chain["audit_path"]
        )


def _tampered_report(smoke_chain):
    return copy.deepcopy(smoke_chain["parallel"])


def test_audit_rejects_history_and_pairing_tamper(smoke_chain):
    history = _tampered_report(smoke_chain)
    history["stage_b"]["factor_trajectories"][0]["run"][
        "current_loss_after_round_history"
    ][0] += 1.0
    history_path = _write_payload(
        smoke_chain["root"] / "tampered_stage_b_history.json", history
    )
    with pytest.raises(RuntimeError, match="trajectory.*trend"):
        auditor.audit_stage_b(
            history_path,
            smoke_chain["root"] / "tampered_stage_b_history_audit.json",
        )

    pairing = _tampered_report(smoke_chain)
    first = pairing["factor_configurations"][0]["config_id"]
    pairing["stage_b"]["comparisons"][first][
        "same_temperature_independent"
    ]["metrics"]["late_window_current_loss"]["wins"] += 1
    pairing_path = _write_payload(
        smoke_chain["root"] / "tampered_stage_b_pairing.json", pairing
    )
    with pytest.raises(RuntimeError, match="stage_b.comparisons"):
        auditor.audit_stage_b(
            pairing_path,
            smoke_chain["root"] / "tampered_stage_b_pairing_audit.json",
        )


def test_audit_rejects_selection_and_upstream_binding_tamper(smoke_chain):
    selection = _tampered_report(smoke_chain)
    selection["stage_b"]["selection"]["stage_c_allowed"] = not selection[
        "stage_b"
    ]["selection"]["stage_c_allowed"]
    selection_path = _write_payload(
        smoke_chain["root"] / "tampered_stage_b_selection.json", selection
    )
    with pytest.raises(RuntimeError, match="stage_b.selection"):
        auditor.audit_stage_b(
            selection_path,
            smoke_chain["root"] / "tampered_stage_b_selection_audit.json",
        )

    upstream = _tampered_report(smoke_chain)
    upstream["upstream"]["stage_t_report_sha256"] = "0" * 64
    upstream_path = _write_payload(
        smoke_chain["root"] / "tampered_stage_b_upstream.json", upstream
    )
    with pytest.raises(RuntimeError, match="stage_t_report_sha256"):
        auditor.audit_stage_b(
            upstream_path,
            smoke_chain["root"] / "tampered_stage_b_upstream_audit.json",
        )


def test_formal_dirty_tree_refuses_before_loading_upstreams(monkeypatch, tmp_path):
    monkeypatch.setattr(
        runner.common,
        "_git_identity",
        lambda: {"commit": "test", "worktree_clean": False},
    )
    monkeypatch.setattr(
        runner,
        "_validate_stage_t_upstream",
        lambda *args: pytest.fail("dirty formal run must stop before inputs"),
    )
    with pytest.raises(RuntimeError, match="工作树干净"):
        runner.run_stage_b(
            "formal",
            "missing-stage-t-report",
            "missing-stage-t-audit",
            "missing-stage-a-report",
            "missing-stage-a-audit",
            tmp_path / "formal",
            max_workers=8,
        )
