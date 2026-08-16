import copy
import json

import pytest

from scripts import audit_issue52_stage_t as auditor
from scripts import run_issue52_stage_t as runner


@pytest.fixture(scope="module")
def smoke_report(tmp_path_factory):
    root = tmp_path_factory.mktemp("issue52-stage-t-audit")
    report_path, report = runner.run_stage_t(
        "smoke", root / "run", max_workers=2
    )
    return root, report_path, report


def _write_payload(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def test_smoke_report_passes_independent_audit_and_refuses_overwrite(
    smoke_report
):
    root, report_path, report = smoke_report
    audit_path, audit = auditor.audit_stage_t(
        report_path, root / "stage_t_audit.json"
    )

    assert audit_path.exists()
    assert audit["passed"] is True
    assert audit["formal_result_valid"] is False
    assert audit["report_sha256"] == auditor.common._sha256_file(report_path)
    assert audit["recomputed"]["aggregates"] == report["stage_t"][
        "aggregates"
    ]
    assert audit["recomputed"]["trajectory_scientific_sha256"] == report[
        "execution"
    ]["trajectory_scientific_sha256"]

    with pytest.raises(FileExistsError, match="审计输出已存在"):
        auditor.audit_stage_t(report_path, audit_path)


def test_audit_rejects_tampered_current_loss_history(smoke_report):
    root, _, report = smoke_report
    tampered = copy.deepcopy(report)
    tampered["stage_t"]["trajectories"][0]["run"][
        "current_loss_after_round_history"
    ][0] += 1.0
    path = _write_payload(root / "tampered_history.json", tampered)

    with pytest.raises(RuntimeError, match="trajectory.*trend"):
        auditor.audit_stage_t(path, root / "tampered_history_audit.json")


def test_audit_rejects_tampered_snapshot_table(smoke_report):
    root, _, report = smoke_report
    tampered = copy.deepcopy(report)
    snapshot = tampered["stage_t"]["trajectories"][0]["run"][
        "state_snapshots"
    ][1]
    # Stronger tamper: replace the table and update its SHA too.  A hash-only
    # audit would accept this; re-evaluating the public queries must reject it.
    template = dict(snapshot["table_records"][0])
    for record in snapshot["table_records"]:
        for column, value in template.items():
            record[column] = value
    frame = auditor.pd.DataFrame(
        snapshot["table_records"], columns=snapshot["table_columns"]
    )
    snapshot["state_sha256"] = auditor.trajectory._frame_sha256(frame)
    path = _write_payload(root / "tampered_snapshot.json", tampered)

    with pytest.raises(RuntimeError, match="轨迹身份门禁失败"):
        auditor.audit_stage_t(path, root / "tampered_snapshot_audit.json")


def test_audit_rejects_tampered_aggregate(smoke_report):
    root, _, report = smoke_report
    tampered = copy.deepcopy(report)
    tampered["stage_t"]["aggregates"]["by_temperature"]["tau_1"][
        "late_window_current_loss"
    ]["mean"] += 1.0
    path = _write_payload(root / "tampered_aggregate.json", tampered)

    with pytest.raises(RuntimeError, match="stage_t.aggregates"):
        auditor.audit_stage_t(path, root / "tampered_aggregate_audit.json")
