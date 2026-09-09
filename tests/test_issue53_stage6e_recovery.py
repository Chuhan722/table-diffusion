from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts import audit_issue53_stage6e_recovered as recovery_auditor
from scripts import evaluate_issue53_stage6e_recovered as recovery_evaluator
from scripts import issue53_stage6e_autostop_protocol as generation_protocol
from scripts import issue53_stage6e_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6e as recovery


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _frozen_artifact_root() -> Path:
    root = _root()
    inventory = recovery_protocol.recovery_inventory(root)
    shard_id = generation_protocol.LOCAL_SHARD
    shard_root = root / recovery_protocol.SHARD_OUTPUT_ROOT
    staging = shard_root / inventory["shards"][shard_id][
        "source_staging_basename"
    ]
    recovered = shard_root / shard_id
    if staging.is_dir():
        return staging
    if recovered.is_dir():
        return recovered
    pytest.skip(
        "冻结 Stage 6E 暂存或恢复分片均不存在（gitignored 本地产物，"
        "仅在持有正式产物的机器上可复核；协议/入口只读契约测试不受影响）"
    )


def test_recovery_inventory_freezes_exact_single_shard_30_cases():
    inventory = recovery_protocol.recovery_inventory(_root())
    shard_id = generation_protocol.LOCAL_SHARD

    assert set(inventory["shards"]) == {shard_id}
    assert len(inventory["shards"][shard_id]["case_manifest_sha256"]) == 30
    assert set(inventory["shards"][shard_id]["case_manifest_sha256"]) == {
        task.task_id for task in generation_protocol.task_plan().tasks
    }


def test_recovery_protocol_and_plan_entrypoints_are_read_only():
    root = _root()

    assert recovery_protocol.assert_frozen_recovery_identity(root) == (
        recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
    )
    assert recovery.build_plan()["new_generation_allowed"] is False
    assert recovery.build_plan()["gpu_access_allowed"] is False
    assert recovery_evaluator.build_plan()["new_generation_allowed"] is False
    assert recovery_auditor.build_plan()["rerun_generation"] is False
    with pytest.raises(PermissionError, match="确认"):
        recovery_protocol.require_recovery_confirmation(None)


def test_actual_frozen_30_cases_validate_without_quality_interpretation():
    root = _root()
    inventory = recovery_protocol.recovery_inventory(root)
    artifact_root = _frozen_artifact_root()

    recovery._validate_staging_manifest(artifact_root, inventory)
    rows = recovery._validate_case_matrix(
        artifact_root,
        generation_protocol.task_plan().tasks,
        inventory,
    )

    assert len(rows) == 30
    assert recovery._termination_counts(rows) == {
        "fit_target_reached": 0,
        "early_stopped": 30,
        "resource_cap_reached": 0,
    }
    assert all(
        row["state_evaluation_count"] == max(1, row["applied_rounds"])
        for row in rows
    )


def test_old_applied_plus_one_assertion_is_rejected_by_erratum_validator():
    task = generation_protocol.task_plan().tasks[0]
    manifest = json.loads(
        (
            _frozen_artifact_root()
            / "cases"
            / task.task_id
            / recovery.CASE_MANIFEST
        ).read_text(encoding="utf-8")
    )
    row = copy.deepcopy(manifest["collection_row"])

    recovery._validate_case_row(task, row)
    row["state_evaluation_count"] = row["applied_rounds"] + 1
    with pytest.raises(RuntimeError, match="身份或计数"):
        recovery._validate_case_row(task, row)


def test_independent_validator_agrees_on_corrected_count_and_rejects_old_count():
    task = generation_protocol.task_plan().tasks[-1]
    manifest = json.loads(
        (
            _frozen_artifact_root()
            / "cases"
            / task.task_id
            / recovery.CASE_MANIFEST
        ).read_text(encoding="utf-8")
    )
    row = copy.deepcopy(manifest["collection_row"])

    recovery_auditor._audit_row_independently(task, row)
    row["state_evaluation_count"] = row["applied_rounds"] + 1
    with pytest.raises(RuntimeError, match="行审计失败"):
        recovery_auditor._audit_row_independently(task, row)
