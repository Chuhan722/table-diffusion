from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import issue53_gap_weight_dual_ar_screen_execution_protocol as source_protocol
from scripts import issue53_gap_weight_dual_ar_screen_recovery_protocol as recovery_protocol
from scripts import recover_issue53_gap_weight_dual_ar_screen as recovery


ROOT = Path(__file__).resolve().parents[1]


def _artifact_root() -> Path:
    shard_root = ROOT / source_protocol.SHARD_OUTPUT_ROOT
    recovered = shard_root / source_protocol.LOCAL_SHARD
    staging = shard_root / recovery_protocol.SOURCE_STAGING_BASENAME
    if recovered.is_dir():
        return recovered
    if staging.is_dir():
        return staging
    pytest.skip("本机没有冻结 A/R 暂存或恢复分片")


def _row(task):
    path = _artifact_root() / "cases" / task.task_id / recovery.CASE_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))["collection_row"]


def test_recovery_inventory_freezes_two_cases_and_eight_files():
    assert set(recovery_protocol.ARTIFACT_INVENTORY) == {
        task.task_id for task in source_protocol.task_plan().tasks
    }
    assert sum(
        len(files) for files in recovery_protocol.ARTIFACT_INVENTORY.values()
    ) == 8


def test_recovery_protocol_identity_and_confirmation_gate():
    assert (
        recovery_protocol.assert_frozen_recovery_identity(ROOT)
        == recovery_protocol.FROZEN_RECOVERY_SHA256
    )
    with pytest.raises(PermissionError, match="确认"):
        recovery_protocol.require_confirmation(None)


def test_actual_two_cases_validate_without_quality_interpretation():
    root = _artifact_root()
    recovery._validate_staging_manifest(root)
    rows = recovery._validate_case_matrix(root)
    assert [row["task_id"] for row in rows] == [
        task.task_id for task in source_protocol.task_plan().tasks
    ]
    assert all(
        row["state_evaluation_count"] == max(1, row["applied_rounds"])
        for row in rows
    )


def test_only_corrected_state_count_is_accepted_and_proxy_does_not_mutate():
    task = source_protocol.task_plan().tasks[0]
    row = copy.deepcopy(_row(task))
    original = copy.deepcopy(row)
    recovery._validate_case_row(task, row)
    proxy = recovery._legacy_validation_proxy(row)
    assert row == original
    assert proxy["state_evaluation_count"] == row["applied_rounds"] + 1
    row["state_evaluation_count"] = row["applied_rounds"] + 1
    with pytest.raises(RuntimeError, match="状态评价次数"):
        recovery._validate_case_row(task, row)


def test_dual_transition_guard_rejects_weight_identity_drift(tmp_path):
    task = source_protocol.task_plan().tasks[0]
    row = _row(task)
    source = (
        _artifact_root() / "cases" / task.task_id / recovery.TRANSITION_AUDIT
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["weighting_audit"]["expected_channel_aggregation"] = "sum"
    path = tmp_path / "transition.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="权重身份"):
        recovery._validate_transition(path, task, row)


def test_recovery_adapter_has_no_generation_or_quality_entrypoint():
    source = (ROOT / "scripts/recover_issue53_gap_weight_dual_ar_screen.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "run_evolution",
        "table_diffevo.evolution",
        "evaluate_table(",
        "classify_screen(",
    ):
        assert forbidden not in source

