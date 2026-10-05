from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_issue53_stage6d_v5_recovered as recovery_auditor
from scripts import evaluate_issue53_stage6d_v5_recovered as recovery_evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6d_v5 as recovery


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_row(task: joint.JointTrajectoryTask, rounds: int) -> dict:
    spec = generation_protocol.DATASETS[task.dataset]
    return {
        "task_id": task.task_id,
        "execution_shard_id": generation_protocol.task_shard_id(task),
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": task.seed,
        "requested_rounds": rounds,
        "applied_rounds": rounds,
        "termination_reason": "candidate_budget",
        "device": "cuda:0",
        "output_table_identity": "terminal_current",
        "all_applied_unconditionally": True,
        "proposal_attempt_count": rounds,
        "candidate_evaluation_count": rounds,
        "state_evaluation_count": rounds,
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
        "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        "gap_8k_identity": True,
        "gap_clip_hit_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
        "direction_logit_clipped_count": 0,
        "nonfinite_count": 0,
        "elapsed_sec": 1.0,
        "peak_allocated_bytes": 1,
        "peak_reserved_bytes": 1,
        "distance_evaluation_count": rounds,
        "direction_evaluation_count": rounds,
        "gap_microsteps": 0,
        "gap_expected_microsteps": 0,
        "factorized_gibbs_microsteps": 0,
        "factorized_gibbs_conditional_logit_evaluated_count": 0,
        "initial_table_sha256": "1" * 64,
        "primary_rng_post_initialization_sha256": "2" * 64,
        "primary_rng_endpoint_sha256": "3" * 64,
        "gap_reference_scale_established": False,
        "gap_reference_scale": None,
    }


def _small_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, joint.JointTrajectoryTask, dict, dict]:
    monkeypatch.setattr(generation_protocol, "ROUNDS", 2)
    monkeypatch.setattr(generation_protocol, "CANDIDATE_BUDGET", 2)
    monkeypatch.setattr(recovery_protocol, "OUTPUT_DIR", Path("out"))
    task = joint.JointTrajectoryTask(
        dataset="test_300x10",
        arm="independent_b_s0",
        seed=353,
        rounds=2,
    )
    output = tmp_path / "out"
    case_dir = output / "cases" / task.task_id
    case_dir.mkdir(parents=True)
    (case_dir / recovery.TERMINAL_TABLE).write_text("a\n0\n", encoding="utf-8")
    _write_json(case_dir / recovery.CHECKPOINT_ARTIFACT, {"opaque": True})
    transition = {
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "task_id": task.task_id,
        "round_count": 2,
        "all_applied_unconditionally": True,
        "clocks": [
            {
                "round": index,
                "state_index": index,
                "accepted_attempt": 1,
                "candidate_evaluation_count_cumulative": index,
                "gibbs_microsteps": 0,
            }
            for index in (1, 2)
        ],
        "gap_rounds": [],
    }
    _write_json(case_dir / recovery.TRANSITION_AUDIT, transition)
    row = _base_row(task, 2)
    prefix = Path("cases") / task.task_id
    for path_key, sha_key, name in (
        ("terminal_table_path", "terminal_table_sha256", recovery.TERMINAL_TABLE),
        (
            "checkpoint_artifact_path",
            "checkpoint_artifact_sha256",
            recovery.CHECKPOINT_ARTIFACT,
        ),
        (
            "transition_audit_path",
            "transition_audit_sha256",
            recovery.TRANSITION_AUDIT,
        ),
    ):
        row[path_key] = str(prefix / name)
        row[sha_key] = recovery_protocol.file_sha256(case_dir / name)
    manifest = {
        "collection_row": row,
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "execution_shard_id": generation_protocol.task_shard_id(task),
        "generator_params": generation_protocol.generator_params_manifest(
            task.dataset, task.arm, task.seed
        ),
        "method_comparison_emitted": False,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "raw_reference_data_accessed": False,
    }
    _write_json(case_dir / recovery.CASE_MANIFEST, manifest)
    shard_id = generation_protocol.task_shard_id(task)
    inventory = {
        "shards": {
            shard_id: {
                "case_manifest_sha256": {
                    task.task_id: recovery_protocol.file_sha256(
                        case_dir / recovery.CASE_MANIFEST
                    )
                }
            }
        }
    }
    return output, task, row, inventory


def test_recovery_inventory_freezes_exact_21_9_case_manifests():
    root = Path(__file__).resolve().parents[1]

    inventory = recovery_protocol.recovery_inventory(root)

    assert (
        len(
            inventory["shards"][generation_protocol.LOCAL_SHARD]["case_manifest_sha256"]
        )
        == 21
    )
    assert (
        len(
            inventory["shards"][generation_protocol.A6000_SHARD]["case_manifest_sha256"]
        )
        == 9
    )
    assert (
        sum(
            len(shard["case_manifest_sha256"]) for shard in inventory["shards"].values()
        )
        == 30
    )


def test_old_generation_source_fails_closed_and_recovery_plans_are_read_only(
    monkeypatch,
):
    root = Path(__file__).resolve().parents[1]

    with pytest.raises(RuntimeError, match="实现源码漂移：full_generator"):
        recovery_protocol.assert_frozen_recovery_identity(root)

    monkeypatch.setattr(
        recovery_protocol,
        "assert_frozen_recovery_identity",
        lambda _root: recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256,
    )
    assert recovery.build_plan()["new_generation_case_count"] == 0
    assert recovery.build_plan()["gpu_access_allowed"] is False
    assert recovery_evaluator.build_plan()["new_generation_allowed"] is False
    assert recovery_auditor.build_plan()["rerun_generation"] is False
    with pytest.raises(PermissionError, match="另行授权"):
        recovery_protocol.require_recovery_confirmation(None)


def test_actual_candidate_budget_state_count_is_2500_not_2501():
    task = joint.JointTrajectoryTask(
        dataset="nltcs",
        arm=generation_protocol.ARM_GAP,
        seed=353,
        rounds=2500,
    )
    row = _base_row(task, 2500)
    row.update(
        {
            "gap_reference_scale_established": True,
            "gap_reference_scale": 0.125,
            "gap_microsteps": 8,
            "gap_expected_microsteps": 8,
        }
    )

    recovery._validate_case_row(task, row)
    row["state_evaluation_count"] = 2501
    with pytest.raises(RuntimeError, match="计数"):
        recovery._validate_case_row(task, row)


def test_case_bytes_and_transition_are_verified_without_interpreting_checkpoint(
    tmp_path, monkeypatch
):
    output, task, row, inventory = _small_case(tmp_path, monkeypatch)

    assert recovery._validate_case(output, task, inventory) == row
    recovery_auditor._audit_case_independently(tmp_path, task, row, inventory)

    terminal = output / row["terminal_table_path"]
    terminal.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="附件 SHA-256"):
        recovery._validate_case(output, task, inventory)


def test_gap_transition_requires_eight_microsteps_per_active_switch(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(generation_protocol, "ROUNDS", 2)
    task = joint.JointTrajectoryTask(
        dataset="test_300x10",
        arm=generation_protocol.ARM_GAP,
        seed=353,
        rounds=2,
    )
    row = {"gap_microsteps": 16}
    transition = {
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "task_id": task.task_id,
        "round_count": 2,
        "all_applied_unconditionally": True,
        "clocks": [
            {
                "round": index,
                "state_index": index,
                "accepted_attempt": 1,
                "candidate_evaluation_count_cumulative": index,
                "gibbs_microsteps": 8,
            }
            for index in (1, 2)
        ],
        "gap_rounds": [
            {
                "round": index,
                "active_switches_k": 1,
                "n_sweeps": 8,
                "scan_applied": True,
                "clip_hit_count": 0,
                "nonfinite_condition_count": 0,
                "expected_microsteps": 8,
                "microsteps": 8,
            }
            for index in (1, 2)
        ],
    }
    path = tmp_path / "transition.json"
    _write_json(path, transition)

    recovery._validate_transition(path, task, row)
    transition["gap_rounds"][1]["microsteps"] = 7
    _write_json(path, transition)
    with pytest.raises(RuntimeError, match=r"8\*K"):
        recovery._validate_transition(path, task, row)


def test_recovered_reports_record_21_9_resumed_and_zero_new_cases():
    root = Path(__file__).resolve().parents[1]
    inventory = recovery_protocol.recovery_inventory(root)
    shard_reports = {}
    shard_hashes = {}
    all_rows = [
        {"task_id": task.task_id} for task in generation_protocol.task_plan().tasks
    ]
    by_id = {row["task_id"]: row for row in all_rows}
    for shard_id in generation_protocol.SHARD_ORDER:
        rows = [
            by_id[task.task_id]
            for task in generation_protocol.tasks_for_shard(shard_id)
        ]
        report = recovery._build_shard_report(
            shard_id=shard_id,
            recovery_commit="a" * 40,
            rows=rows,
            inventory=inventory,
            started_at="2026-08-28T00:00:00+08:00",
            finished_at="2026-08-28T00:00:01+08:00",
            elapsed_sec=1.0,
        )
        shard_reports[shard_id] = report
        shard_hashes[shard_id] = "f" * 64
        expected = 21 if shard_id == generation_protocol.LOCAL_SHARD else 9
        assert report["recovery_execution"]["resumed_case_count"] == expected
        assert report["recovery_execution"]["new_generation_case_count"] == 0
        assert report["source_gpu_monitoring_evidence"]["gpu_samples"] == []
        assert (
            report["source_gpu_monitoring_evidence"]["gpu_samples_reconstructed"]
            is False
        )
    collection = recovery._build_collection_report(
        recovery_commit="a" * 40,
        rows=all_rows,
        shard_reports=shard_reports,
        shard_report_sha256=shard_hashes,
        inventory=inventory,
    )

    assert collection["recovery_execution"]["resumed_case_counts"] == {
        generation_protocol.LOCAL_SHARD: 21,
        generation_protocol.A6000_SHARD: 9,
    }
    assert collection["recovery_execution"]["new_generation_case_count"] == 0
    assert collection["execution_monitoring_evidence_complete"] is False
    assert collection["l1_results_published_by_collection"] is False


def test_close_shard_atomically_moves_only_validated_frozen_case(tmp_path, monkeypatch):
    artifact_root, task, row, inventory = _small_case(tmp_path, monkeypatch)
    shard_id = generation_protocol.task_shard_id(task)
    shard_root = tmp_path / "shards"
    staging_name = f".{shard_id}.staging-frozen"
    staging = shard_root / staging_name
    shard_root.mkdir()
    artifact_root.rename(staging)
    staging_manifest = {
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256),
        "shard_id": shard_id,
        "task_ids": [task.task_id],
        "partial_quality_inspection_allowed": False,
    }
    _write_json(staging / recovery.STAGING_MANIFEST, staging_manifest)
    inventory["shards"][shard_id].update(
        {
            "source_hostname": generation_protocol.EXECUTION_SHARDS[shard_id][
                "hostname"
            ],
            "source_staging_basename": staging_name,
            "staging_manifest_sha256": recovery_protocol.file_sha256(
                staging / recovery.STAGING_MANIFEST
            ),
        }
    )
    source_hashes = {
        name: recovery_protocol.file_sha256(staging / "cases" / task.task_id / name)
        for name in (
            recovery.CASE_MANIFEST,
            recovery.TERMINAL_TABLE,
            recovery.CHECKPOINT_ARTIFACT,
            recovery.TRANSITION_AUDIT,
        )
    }
    monkeypatch.setattr(recovery, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(recovery_protocol, "SHARD_OUTPUT_ROOT", Path("shards"))
    monkeypatch.setattr(
        recovery_protocol, "assert_frozen_recovery_identity", lambda _root: "ok"
    )
    monkeypatch.setattr(recovery, "_assert_clean_recovery_tree", lambda _root: "a" * 40)
    monkeypatch.setattr(
        recovery_protocol, "recovery_inventory", lambda _root: inventory
    )
    monkeypatch.setattr(
        generation_protocol,
        "tasks_for_shard",
        lambda observed: (task,) if observed == shard_id else (),
    )
    monkeypatch.setattr(recovery, "_validate_pairing_for_tasks", lambda *_args: None)
    blocks = dict(generation_protocol.SHARD_BLOCKS)
    blocks[shard_id] = ((task.seed, task.dataset),)
    monkeypatch.setattr(generation_protocol, "SHARD_BLOCKS", blocks)

    report_path = recovery.close_shard(
        recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256,
        shard_id,
    )

    destination = shard_root / shard_id
    assert report_path == destination / recovery_protocol.SHARD_REPORT
    assert destination.is_dir()
    assert not staging.exists()
    report = recovery._load_json(report_path)
    assert report["recovery_execution"]["resumed_case_count"] == 1
    assert report["recovery_execution"]["new_generation_case_count"] == 0
    assert report["recovery_execution"]["generator_invoked"] is False
    assert {
        name: recovery_protocol.file_sha256(destination / "cases" / task.task_id / name)
        for name in source_hashes
    } == source_hashes
    assert row == report["raw_results"][0]


def test_evaluator_validates_recovered_collection_before_reference_access(monkeypatch):
    calls = []
    monkeypatch.setattr(
        recovery_protocol,
        "assert_frozen_recovery_identity",
        lambda _root: recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256,
    )
    monkeypatch.setattr(recovery, "_git_text", lambda *_args: "")

    def fake_load(_root, confirmed):
        calls.append(("collection", confirmed))
        raise RuntimeError("stop-before-reference")

    monkeypatch.setattr(recovery, "load_collection", fake_load)
    monkeypatch.setattr(
        recovery_evaluator.legacy_evaluator,
        "_evaluate_cases",
        lambda *_args: pytest.fail("collection 复核前不得读取参考数据"),
    )

    with pytest.raises(RuntimeError, match="stop-before-reference"):
        recovery_evaluator.evaluate("a" * 64)
    assert calls == [("collection", "a" * 64)]
