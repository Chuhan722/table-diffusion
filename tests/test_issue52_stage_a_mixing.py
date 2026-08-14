import copy
import json

import pytest

from scripts import audit_issue52_stage_a_mixing as auditor
from scripts import audit_issue52_stage_a_state_library as library_auditor
from scripts import audit_issue52_stage_t as stage_t_auditor
from scripts import build_issue52_stage_a_state_library as library_builder
from scripts import issue52_protocol as protocol
from scripts import run_issue52_stage_a_mixing as runner
from scripts import run_issue52_stage_t as stage_t_runner


def _write_payload(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory):
    root = tmp_path_factory.mktemp("issue52-stage-a-mixing")
    stage_t_report_path, _ = stage_t_runner.run_stage_t(
        "smoke", root / "stage_t", max_workers=2
    )
    stage_t_audit_path, _ = stage_t_auditor.audit_stage_t(
        stage_t_report_path, root / "stage_t_audit.json"
    )
    library_path, library = library_builder.build_state_library(
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
    report_path, report = runner.run_stage_a_mixing(
        "smoke",
        library_path,
        library_audit_path,
        root / "mixing",
        max_workers=2,
    )
    audit_path, audit = auditor.audit_stage_a_mixing(
        report_path,
        library_path,
        library_audit_path,
        root / "mixing_audit.json",
    )
    return {
        "root": root,
        "library_path": library_path,
        "library": library,
        "library_audit_path": library_audit_path,
        "report_path": report_path,
        "report": report,
        "audit_path": audit_path,
        "audit": audit,
    }


def test_protocol_freezes_full_tau_grid_incremental_sweeps_and_48_states():
    formal = protocol.stage_a_mixing_protocol("formal")
    smoke = protocol.stage_a_mixing_protocol("smoke")

    assert formal["evaluation_temperatures"] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert formal["candidate_sweeps"] == [8, 16, 32]
    assert formal["sweeps_hard_cap"] == 32
    assert formal["expected_state_count"] == 48
    assert formal["states_per_group"] == 3
    assert len(formal["required_state_groups"]) == 16
    assert formal["tvd_threshold"] == 0.05
    assert formal["recovery_threshold"] == 0.80
    assert formal["expected_state_library_artifact_sha256"] == {
        "library": (
            "4945f52a644e059a710decf66659b82fa"
            "693c4443d799ca2e3948a1ca61d33fc"
        ),
        "audit": (
            "837a90851795dcc05d4f15f88fea17658"
            "2dcc92ebf9227cadbe40ae21e161545"
        ),
    }
    assert smoke["expected_state_count"] == 16
    assert smoke["states_per_group"] == 1
    assert smoke["expected_state_library_artifact_sha256"] is None


def _attempt(sweeps, passed):
    return {"sweeps": sweeps, "passed": passed}


def test_incremental_stop_rule_selects_first_pass_and_exhausts_failures():
    frozen = protocol.stage_a_mixing_protocol("smoke")

    at_eight = auditor._sequence_result(
        1.0, [_attempt(8, True)], frozen
    )
    at_sixteen = auditor._sequence_result(
        2.0, [_attempt(8, False), _attempt(16, True)], frozen
    )
    exhausted = auditor._sequence_result(
        3.0,
        [_attempt(8, False), _attempt(16, False), _attempt(32, False)],
        frozen,
    )

    assert at_eight["minimal_sufficient_sweeps"] == 8
    assert at_sixteen["minimal_sufficient_sweeps"] == 16
    assert exhausted["status"] == "unqualified_at_sweeps_cap"
    assert exhausted["minimal_sufficient_sweeps"] is None


@pytest.mark.parametrize(
    "attempts, message",
    [
        ([_attempt(8, True), _attempt(16, True)], "首次通过后"),
        ([_attempt(8, False)], "错误提前停止"),
        ([_attempt(16, True)], "缺失、乱序或跳级"),
    ],
)
def test_incremental_stop_rule_rejects_wrong_execution(attempts, message):
    frozen = protocol.stage_a_mixing_protocol("smoke")
    with pytest.raises(RuntimeError, match=message):
        auditor._sequence_result(1.0, attempts, frozen)


def test_real_smoke_is_auditable_complete_and_non_formal(smoke_result):
    report = smoke_result["report"]
    audit = smoke_result["audit"]

    assert report["status"] == "complete"
    assert report["formal_result_valid"] is False
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert report["protocol"]["expected_state_count"] == 16
    assert list(report["temperatures"]) == [
        "tau_1", "tau_2", "tau_3", "tau_4", "tau_5"
    ]
    assert all(report["execution_gates"].values())
    assert all(
        result["attempted_sweeps"]
        == report["protocol"]["candidate_sweeps"][:len(result["attempts"])]
        for result in report["temperatures"].values()
    )
    assert audit["execution_scientific_sha256"] == report[
        "execution_scientific_sha256"
    ]

    with pytest.raises(FileExistsError, match="尚未启动任何混合任务"):
        runner.run_stage_a_mixing(
            "smoke",
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["report_path"].parent,
            max_workers=1,
        )
    with pytest.raises(FileExistsError, match="审计输出已存在"):
        auditor.audit_stage_a_mixing(
            smoke_result["report_path"],
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["audit_path"],
        )


def _tampered_report(smoke_result):
    return copy.deepcopy(smoke_result["report"])


def test_audit_rejects_missing_and_reordered_states(smoke_result):
    missing = _tampered_report(smoke_result)
    first_attempt = missing["temperatures"]["tau_1"]["attempts"][0]
    first_attempt["state_results"].pop()
    missing_path = _write_payload(
        smoke_result["root"] / "missing_state_report.json", missing
    )
    with pytest.raises(RuntimeError, match="缺失、重复或乱序"):
        auditor.audit_stage_a_mixing(
            missing_path,
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["root"] / "missing_state_audit.json",
        )

    reordered = _tampered_report(smoke_result)
    rows = reordered["temperatures"]["tau_1"]["attempts"][0][
        "state_results"
    ]
    rows[0], rows[1] = rows[1], rows[0]
    reordered_path = _write_payload(
        smoke_result["root"] / "reordered_state_report.json", reordered
    )
    with pytest.raises(RuntimeError, match="缺失、重复或乱序"):
        auditor.audit_stage_a_mixing(
            reordered_path,
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["root"] / "reordered_state_audit.json",
        )


def test_audit_rejects_shared_tape_tamper_wrong_stop_and_sweeps_above_cap(
    smoke_result,
):
    tape = _tampered_report(smoke_result)
    tape["temperatures"]["tau_1"]["attempts"][0][
        "shared_condition_scientific_sha256"
    ] = "0" * 64
    tape_path = _write_payload(
        smoke_result["root"] / "tape_tamper_report.json", tape
    )
    with pytest.raises(RuntimeError, match="shared_condition"):
        auditor.audit_stage_a_mixing(
            tape_path,
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["root"] / "tape_tamper_audit.json",
        )

    wrong_stop = _tampered_report(smoke_result)
    result = wrong_stop["temperatures"]["tau_1"]
    result["attempts"] = result["attempts"][:1]
    result["attempted_sweeps"] = [8]
    result["status"] = "unqualified_at_sweeps_cap"
    result["minimal_sufficient_sweeps"] = None
    wrong_stop_path = _write_payload(
        smoke_result["root"] / "wrong_stop_report.json", wrong_stop
    )
    first_passed = result["attempts"][0]["passed"]
    expected_message = "report.temperatures" if first_passed else "错误提前停止"
    with pytest.raises(RuntimeError, match=expected_message):
        auditor.audit_stage_a_mixing(
            wrong_stop_path,
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["root"] / "wrong_stop_audit.json",
        )

    above_cap = _tampered_report(smoke_result)
    above_cap["temperatures"]["tau_1"]["attempts"][0]["sweeps"] = 64
    above_cap_path = _write_payload(
        smoke_result["root"] / "above_cap_report.json", above_cap
    )
    with pytest.raises(RuntimeError, match="非法 sweeps"):
        auditor.audit_stage_a_mixing(
            above_cap_path,
            smoke_result["library_path"],
            smoke_result["library_audit_path"],
            smoke_result["root"] / "above_cap_audit.json",
        )
