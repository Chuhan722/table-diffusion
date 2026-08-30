#!/usr/bin/env python3
"""只读复核并收口已经生成完成的 Issue #53 Stage 6E 轨迹。"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as generation_protocol
from scripts import issue53_stage6e_recovery_protocol as recovery_protocol


CASE_MANIFEST = "case_manifest.json"
STAGING_MANIFEST = "staging_manifest.json"
TERMINAL_TABLE = "terminal_current.csv"
CHECKPOINT_ARTIFACT = "checkpoint_query_answers.json"
TRANSITION_AUDIT = "transition_audit.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strict_json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"恢复报告已存在，不覆盖：{path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"恢复临时报告已存在：{temporary}")
    try:
        temporary.write_text(_strict_json_text(value), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _assert_clean_recovery_tree(root: Path) -> str:
    if _git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Stage 6E 恢复执行要求包含未跟踪文件在内的干净工作树")
    commit = _git_text(root, "rev-parse", "HEAD")
    if (
        _git_text(
            root,
            "merge-base",
            recovery_protocol.SOURCE_GENERATION_COMMIT,
            commit,
        )
        != recovery_protocol.SOURCE_GENERATION_COMMIT
    ):
        raise RuntimeError("Stage 6E 生成提交不是当前恢复提交的祖先")
    return commit


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _safe_artifact(root: Path, relative: Any, expected: str) -> Path:
    if relative != expected:
        raise RuntimeError(
            f"Stage 6E 恢复产物路径漂移：expected={expected}, observed={relative}"
        )
    part = Path(expected)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Stage 6E 恢复产物路径越界")
    base = root.resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("Stage 6E 恢复产物路径逃逸")
    return resolved


def _source_monitoring_limitation() -> dict[str, Any]:
    return {
        "code": recovery_protocol.MONITORING_LIMITATION_CODE,
        "gpu_samples_persisted": False,
        "gpu_samples": [],
        "gpu_samples_reconstructed": False,
        "reason": recovery_protocol.MONITORING_LIMITATION_REASON,
        "case_generation_result_bytes_affected": False,
        "quality_arithmetic_affected": False,
    }


def _validate_staging_manifest(
    staging: Path,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    shard_id = generation_protocol.LOCAL_SHARD
    path = staging / STAGING_MANIFEST
    expected_sha = inventory["shards"][shard_id]["staging_manifest_sha256"]
    if recovery_protocol.file_sha256(path) != expected_sha:
        raise RuntimeError("Stage 6E 恢复暂存清单 SHA-256 漂移")
    manifest = _load_json(path)
    required = {
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (
            recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256
        ),
        "shard_id": shard_id,
        "task_ids": [
            task.task_id for task in generation_protocol.tasks_for_shard(shard_id)
        ],
        "partial_quality_inspection_allowed": False,
    }
    if manifest != required:
        raise RuntimeError("Stage 6E 恢复暂存清单内容漂移")
    return manifest


def _validate_case_row(task: Any, row: dict[str, Any]) -> None:
    """只修正旧收集器的状态评价次数断言，其余护栏保持冻结。"""

    applied = row.get("applied_rounds")
    reason = row.get("termination_reason")
    stopping = row.get("inner_early_stopping")
    elapsed = row.get("elapsed_sec")
    average = row.get("average_sec_per_applied_round")
    if (
        row.get("task_id") != task.task_id
        or row.get("execution_shard_id") != generation_protocol.LOCAL_SHARD
        or row.get("dataset") != task.dataset
        or row.get("arm") != task.arm
        or row.get("seed") != task.seed
        or row.get("requested_rounds") != generation_protocol.ROUND_CAP
        or row.get("device") != "cuda:0"
        or row.get("output_table_identity") != "terminal_current"
        or row.get("all_applied_unconditionally") is not True
        or not _nonnegative_int(applied)
        or applied > generation_protocol.ROUND_CAP
        or reason not in generation_protocol.NORMAL_TERMINATION_REASONS
        or row.get("proposal_attempt_count") != applied
        or row.get("candidate_evaluation_count") != applied
        or row.get("state_evaluation_count") != max(1, applied)
        or not isinstance(stopping, dict)
        or stopping.get("enabled") is not True
        or stopping.get("patience_ticks") != generation_protocol.PATIENCE_TICKS
        or stopping.get("termination_reason") != reason
        or stopping.get("terminal_state_index") != applied
        or not _nonnegative_number(elapsed)
        or not _nonnegative_number(average)
        or not math.isclose(
            float(average),
            float(elapsed) / applied if applied else 0.0,
            rel_tol=1e-15,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(f"Stage 6E 恢复 case 行身份或计数漂移：{task.task_id}")
    if reason == "fit_target_reached" and stopping.get("terminal_loss") != 0.0:
        raise RuntimeError(f"Stage 6E 恢复 A 停止身份漂移：{task.task_id}")
    if reason == "early_stopped" and (
        not _nonnegative_int(stopping.get("consecutive_no_progress_ticks"))
        or stopping["consecutive_no_progress_ticks"]
        < generation_protocol.PATIENCE_TICKS
    ):
        raise RuntimeError(f"Stage 6E 恢复 B 停止身份漂移：{task.task_id}")
    if reason == "resource_cap_reached" and (
        applied != generation_protocol.ROUND_CAP
        or stopping.get("resource_cap_source_diagnostic_only")
        != "candidate_budget"
    ):
        raise RuntimeError(f"Stage 6E 恢复 C 停止身份漂移：{task.task_id}")

    spec = generation_protocol.DATASETS[task.dataset]
    required_identity = {
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
        "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        "gap_8k_identity": True,
        "gap_clip_hit_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
        "direction_logit_clipped_count": 0,
        "nonfinite_count": 0,
    }
    if any(row.get(key) != value for key, value in required_identity.items()):
        raise RuntimeError(f"Stage 6E 恢复 case 身份或数值护栏漂移：{task.task_id}")
    for name in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "distance_evaluation_count",
        "direction_evaluation_count",
        "direction_logit_evaluated_count",
        "gap_microsteps",
        "gap_expected_microsteps",
        "factorized_gibbs_microsteps",
        "factorized_gibbs_conditional_logit_evaluated_count",
    ):
        if not _nonnegative_number(row.get(name)):
            raise RuntimeError(f"Stage 6E 恢复成本字段漂移：{task.task_id}/{name}")
    if (
        row.get("distance_evaluation_count") != applied
        or row.get("direction_evaluation_count") != applied
        or row.get("factorized_gibbs_microsteps")
        != row.get("factorized_gibbs_conditional_logit_evaluated_count")
        or not _nonnegative_number(row.get("direction_reference_scale"))
        or row["direction_reference_scale"] <= 0.0
        or not all(
            _is_sha256(row.get(name))
            for name in (
                "initial_table_sha256",
                "primary_rng_post_initialization_sha256",
                "primary_rng_endpoint_sha256",
            )
        )
    ):
        raise RuntimeError(f"Stage 6E 恢复成本或随机流身份漂移：{task.task_id}")
    if task.arm == generation_protocol.ARM_GAP:
        if (
            row.get("gap_reference_scale_established") is not bool(applied)
            or (applied and not _nonnegative_number(row.get("gap_reference_scale")))
            or (applied and row["gap_reference_scale"] <= 0.0)
            or (not applied and row.get("gap_reference_scale") is not None)
            or row.get("gap_microsteps") != row.get("gap_expected_microsteps")
            or row.get("factorized_gibbs_microsteps") != 0
        ):
            raise RuntimeError(f"Stage 6E 恢复缺口核尺度或 8*K 漂移：{task.task_id}")
    elif (
        row.get("gap_reference_scale_established") is not False
        or row.get("gap_reference_scale") is not None
        or row.get("gap_microsteps") != 0
        or row.get("gap_expected_microsteps") != 0
    ):
        raise RuntimeError(f"Stage 6E 恢复基线含缺口核状态：{task.task_id}")
    if task.arm == "independent_b_s0" and row.get("factorized_gibbs_microsteps") != 0:
        raise RuntimeError(f"Stage 6E 恢复独立基线含 Gibbs 微步：{task.task_id}")


def _validate_transition(path: Path, task: Any, row: dict[str, Any]) -> None:
    transition = _load_json(path)
    applied = row["applied_rounds"]
    clocks = transition.get("clocks")
    gap_rounds = transition.get("gap_rounds")
    factor_rounds = transition.get("factor_rounds")
    if (
        transition.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or transition.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or transition.get("task_id") != task.task_id
        or transition.get("round_count") != applied
        or transition.get("all_applied_unconditionally") is not True
        or not isinstance(clocks, list)
        or len(clocks) != applied
        or not isinstance(gap_rounds, list)
        or not isinstance(factor_rounds, list)
    ):
        raise RuntimeError(f"Stage 6E 恢复转移总结构漂移：{task.task_id}")
    clock_microsteps = 0
    for index, clock in enumerate(clocks, start=1):
        integer_names = (
            "participating_rows",
            "changed_rows",
            "changed_cells",
            "changed_queries",
            "gibbs_microsteps",
        )
        if (
            not isinstance(clock, dict)
            or clock.get("round") != index
            or clock.get("state_index") != index
            or clock.get("accepted_attempt") != 1
            or clock.get("candidate_evaluation_count_cumulative") != index
            or not all(_nonnegative_int(clock.get(name)) for name in integer_names)
            or clock["changed_rows"] > clock["participating_rows"]
            or not _is_sha256(clock.get("post_current_table_sha256"))
            or not _is_sha256(clock.get("primary_rng_state_sha256"))
            or (
                not _is_sha256(clock.get("factorized_gibbs_rng_state_sha256"))
                if task.arm == "factor_b_s8"
                else clock.get("factorized_gibbs_rng_state_sha256") is not None
            )
        ):
            raise RuntimeError(f"Stage 6E 恢复转移时钟漂移：{task.task_id}/{index}")
        clock_microsteps += clock["gibbs_microsteps"]

    if task.arm == generation_protocol.ARM_GAP:
        if len(gap_rounds) != applied or factor_rounds:
            raise RuntimeError(f"Stage 6E 恢复缺口核轮次结构漂移：{task.task_id}")
        total = 0
        for index, item in enumerate(gap_rounds, start=1):
            active = item.get("active_switches_k") if isinstance(item, dict) else None
            expected = 8 * active if _nonnegative_int(active) else None
            if (
                not isinstance(item, dict)
                or item.get("round") != index
                or item.get("scan_applied") is not True
                or item.get("n_sweeps") != 8
                or item.get("microsteps") != expected
                or item.get("expected_microsteps") != expected
                or item.get("clip_hit_count") != 0
                or item.get("nonfinite_condition_count") != 0
            ):
                raise RuntimeError(f"Stage 6E 恢复缺口 8*K 漂移：{task.task_id}/{index}")
            total += expected
        if total != row["gap_microsteps"] or clock_microsteps != total:
            raise RuntimeError(f"Stage 6E 恢复缺口微步总量漂移：{task.task_id}")
    elif task.arm == "factor_b_s8":
        if gap_rounds or len(factor_rounds) != applied:
            raise RuntimeError(f"Stage 6E 恢复因子核轮次结构漂移：{task.task_id}")
        total = 0
        for index, item in enumerate(factor_rounds, start=1):
            if (
                not isinstance(item, dict)
                or item.get("round") != index
                or item.get("all_finite") is not True
                or not _nonnegative_int(item.get("gibbs_microsteps"))
            ):
                raise RuntimeError(f"Stage 6E 恢复因子核轮次漂移：{task.task_id}/{index}")
            total += item["gibbs_microsteps"]
        if total != row["factorized_gibbs_microsteps"] or clock_microsteps != total:
            raise RuntimeError(f"Stage 6E 恢复因子核微步总量漂移：{task.task_id}")
    elif gap_rounds or factor_rounds or clock_microsteps != 0:
        raise RuntimeError(f"Stage 6E 恢复独立核含 Gibbs 轨迹：{task.task_id}")


def _validate_case(
    artifact_root: Path,
    task: Any,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    shard_id = generation_protocol.LOCAL_SHARD
    case_relative = Path("cases") / task.task_id
    case_dir = artifact_root / case_relative
    expected_files = {
        CASE_MANIFEST,
        TERMINAL_TABLE,
        CHECKPOINT_ARTIFACT,
        TRANSITION_AUDIT,
    }
    if (
        not case_dir.is_dir()
        or {path.name for path in case_dir.iterdir()} != expected_files
    ):
        raise RuntimeError(f"Stage 6E 恢复 case 文件集合漂移：{task.task_id}")
    manifest_path = case_dir / CASE_MANIFEST
    expected_manifest_sha = inventory["shards"][shard_id][
        "case_manifest_sha256"
    ][task.task_id]
    if recovery_protocol.file_sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError(f"Stage 6E 恢复 case 清单 SHA-256 漂移：{task.task_id}")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or manifest.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or manifest.get("execution_commit")
        != recovery_protocol.SOURCE_GENERATION_COMMIT
        or manifest.get("execution_shard_id") != shard_id
        or manifest.get("generator_params")
        != generation_protocol.generator_params_manifest(
            task.dataset, task.arm, task.seed
        )
        or manifest.get("raw_reference_data_accessed") is not False
        or manifest.get("method_comparison_emitted") is not False
    ):
        raise RuntimeError(f"Stage 6E 恢复 case 来源身份漂移：{task.task_id}")
    row = manifest.get("collection_row")
    if not isinstance(row, dict):
        raise TypeError(f"Stage 6E 恢复 case 缺少 collection 行：{task.task_id}")
    _validate_case_row(task, row)
    attachments = (
        (
            "terminal_table_path",
            "terminal_table_sha256",
            str(case_relative / TERMINAL_TABLE),
        ),
        (
            "checkpoint_artifact_path",
            "checkpoint_artifact_sha256",
            str(case_relative / CHECKPOINT_ARTIFACT),
        ),
        (
            "transition_audit_path",
            "transition_audit_sha256",
            str(case_relative / TRANSITION_AUDIT),
        ),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, expected_relative in attachments:
        path = _safe_artifact(artifact_root, row.get(path_key), expected_relative)
        if (
            not _is_sha256(row.get(sha_key))
            or recovery_protocol.file_sha256(path) != row[sha_key]
        ):
            raise RuntimeError(
                f"Stage 6E 恢复 case 附件 SHA-256 漂移：{task.task_id}/{path_key}"
            )
        resolved[path_key] = path
    # 检查点在收口阶段只复核冻结字节；质量内容留给确认后的离线评价。
    _validate_transition(resolved["transition_audit_path"], task, row)
    return row


def _validate_pairing(
    rows: Sequence[dict[str, Any]], tasks: Sequence[Any]
) -> None:
    if [row.get("task_id") for row in rows] != [task.task_id for task in tasks]:
        raise RuntimeError("Stage 6E 恢复三核行顺序漂移")
    indexed = {(row["seed"], row["dataset"], row["arm"]): row for row in rows}
    if len(indexed) != len(tasks):
        raise RuntimeError("Stage 6E 恢复三核地址重复")
    for seed, dataset in sorted({(task.seed, task.dataset) for task in tasks}):
        paired = [indexed[(seed, dataset, arm)] for arm in joint.ARM_ORDER]
        for key in (
            "initial_table_sha256",
            "primary_rng_post_initialization_sha256",
            "direction_reference_scale",
            "query_identity_sha256",
            "target_vector_sha256",
            "trace_query_identity_sha256",
            "trace_target_vector_sha256",
        ):
            if len({row[key] for row in paired}) != 1:
                raise RuntimeError(
                    f"Stage 6E 恢复三核配对身份漂移：{seed}/{dataset}/{key}"
                )


def _validate_case_matrix(
    artifact_root: Path,
    tasks: Sequence[Any],
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    cases_root = artifact_root / "cases"
    expected_ids = {task.task_id for task in tasks}
    if (
        not cases_root.is_dir()
        or {path.name for path in cases_root.iterdir()} != expected_ids
    ):
        raise RuntimeError("Stage 6E 恢复 case 目录集合与冻结任务不一致")
    rows = [_validate_case(artifact_root, task, inventory) for task in tasks]
    _validate_pairing(rows, tasks)
    return rows


def _timing_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    elapsed = math.fsum(float(row["elapsed_sec"]) for row in rows)
    rounds = sum(int(row["applied_rounds"]) for row in rows)
    return {
        "sum_case_elapsed_sec": float(elapsed),
        "total_applied_rounds": int(rounds),
        "average_sec_per_applied_round": (
            float(elapsed / rounds) if rounds else 0.0
        ),
    }


def _termination_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        reason: sum(row["termination_reason"] == reason for row in rows)
        for reason in generation_protocol.NORMAL_TERMINATION_REASONS
    }


def _recovery_audit() -> dict[str, bool]:
    return {
        "all_30_cases_present": True,
        "all_10_dataset_seed_triplets_paired": True,
        "single_frozen_shard_complete": True,
        "all_case_manifests_match_frozen_inventory": True,
        "all_90_case_attachments_sha256_verified": True,
        "corrected_state_evaluation_identity_verified": True,
        "logical_state_count_remains_applied_rounds_plus_one": True,
        "all_existing_p6_autostop_enabled": True,
        "all_normal_abc_termination": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "zero_new_generation_cases": True,
        "source_case_files_rewritten": False,
        "gpu_monitoring_limitation_disclosed": True,
    }


def _build_shard_report(
    *,
    recovery_commit: str,
    rows: Sequence[dict[str, Any]],
    inventory: dict[str, Any],
    started_at: str,
    finished_at: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    shard_id = generation_protocol.LOCAL_SHARD
    return {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (
            recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
        ),
        "source_generation": {
            "contract_version": generation_protocol.PROTOCOL_VERSION,
            "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
            "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
            "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
            "generator_params_manifest_sha256": (
                recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
            ),
            "shard_assignment_sha256": (
                recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256
            ),
        },
        "recovery_commit": recovery_commit,
        "recovery_collector_sha256": recovery_protocol.file_sha256(Path(__file__)),
        "recovery_inventory_sha256": recovery_protocol.RECOVERY_INVENTORY_SHA256,
        "shard_id": shard_id,
        "source_hostname": inventory["shards"][shard_id]["source_hostname"],
        "source_staging_manifest": {
            "path": STAGING_MANIFEST,
            "sha256": inventory["shards"][shard_id]["staging_manifest_sha256"],
            "observed_original_basename": inventory["shards"][shard_id][
                "source_staging_basename"
            ],
        },
        "source_expected_execution_environment": {
            "hostname": generation_protocol.EXECUTION_SHARDS[shard_id]["hostname"],
            "gpu": dict(
                generation_protocol.EXECUTION_SHARDS[shard_id]["expected_gpu"]
            ),
            "software": dict(generation_protocol.EXPECTED_SOFTWARE),
            "status": "frozen_expectation_not_reconstructed_runtime_sample",
        },
        "source_gpu_monitoring_evidence": _source_monitoring_limitation(),
        "recovery_execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": float(elapsed_sec),
            "resumed_case_count": 30,
            "new_generation_case_count": 0,
            "generator_invoked": False,
            "gpu_accessed": False,
            "raw_reference_data_accessed": False,
            "quality_metrics_interpreted": False,
        },
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "task_ids": [row["task_id"] for row in rows],
        "case_manifest_sha256": dict(
            inventory["shards"][shard_id]["case_manifest_sha256"]
        ),
        "raw_results": list(rows),
        "timing_summary": _timing_summary(rows),
        "termination_reason_counts": _termination_counts(rows),
        "recovery_audit": _recovery_audit(),
        "formal_shard_complete": True,
        "formal_shard_recovered_after_generation": True,
        "execution_monitoring_evidence_complete": False,
        "partial_shard_comparison_emitted": False,
    }


def _validate_shard_report(
    shard_root: Path,
    recovery_commit: str,
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    shard_id = generation_protocol.LOCAL_SHARD
    tasks = generation_protocol.tasks_for_shard(shard_id)
    _validate_staging_manifest(shard_root, inventory)
    rows = _validate_case_matrix(shard_root, tasks, inventory)
    path = shard_root / recovery_protocol.SHARD_REPORT
    report = _load_json(path)
    required = {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (
            recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
        ),
        "recovery_commit": recovery_commit,
        "recovery_collector_sha256": recovery_protocol.IMPLEMENTATION_SOURCES[
            "recovery_collector"
        ]["sha256"],
        "recovery_inventory_sha256": recovery_protocol.RECOVERY_INVENTORY_SHA256,
        "shard_id": shard_id,
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "formal_shard_complete": True,
        "formal_shard_recovered_after_generation": True,
        "execution_monitoring_evidence_complete": False,
        "partial_shard_comparison_emitted": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("Stage 6E 恢复分片报告身份漂移")
    if (
        report.get("raw_results") != rows
        or report.get("task_ids") != [task.task_id for task in tasks]
        or report.get("case_manifest_sha256")
        != inventory["shards"][shard_id]["case_manifest_sha256"]
        or report.get("timing_summary") != _timing_summary(rows)
        or report.get("termination_reason_counts") != _termination_counts(rows)
        or report.get("recovery_audit") != _recovery_audit()
        or report.get("source_gpu_monitoring_evidence")
        != _source_monitoring_limitation()
    ):
        raise RuntimeError("Stage 6E 恢复分片报告内容漂移")
    execution = report.get("recovery_execution")
    if (
        not isinstance(execution, dict)
        or execution.get("resumed_case_count") != 30
        or execution.get("new_generation_case_count") != 0
        or execution.get("generator_invoked") is not False
        or execution.get("gpu_accessed") is not False
        or execution.get("raw_reference_data_accessed") is not False
        or execution.get("quality_metrics_interpreted") is not False
    ):
        raise RuntimeError("Stage 6E 恢复分片执行边界漂移")
    return report, rows, recovery_protocol.file_sha256(path)


def _find_frozen_staging(root: Path, inventory: dict[str, Any]) -> Path:
    shard_id = generation_protocol.LOCAL_SHARD
    expected = (
        root
        / recovery_protocol.SHARD_OUTPUT_ROOT
        / inventory["shards"][shard_id]["source_staging_basename"]
    )
    if not expected.is_dir():
        raise FileNotFoundError(f"冻结 Stage 6E 暂存目录不存在：{expected}")
    _validate_staging_manifest(expected, inventory)
    return expected


def close_shard(confirmed_recovery_sha256: str) -> Path:
    recovery_protocol.require_recovery_confirmation(confirmed_recovery_sha256)
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    recovery_commit = _assert_clean_recovery_tree(root)
    inventory = recovery_protocol.recovery_inventory(root)
    shard_id = generation_protocol.LOCAL_SHARD
    destination = root / recovery_protocol.SHARD_OUTPUT_ROOT / shard_id
    if destination.exists():
        report, _, _ = _validate_shard_report(
            destination, recovery_commit, inventory
        )
        if report.get("formal_shard_complete") is True:
            return destination / recovery_protocol.SHARD_REPORT
    staging = _find_frozen_staging(root, inventory)
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    rows = _validate_case_matrix(
        staging,
        generation_protocol.tasks_for_shard(shard_id),
        inventory,
    )
    finished_at = datetime.now().astimezone().isoformat()
    report_path = staging / recovery_protocol.SHARD_REPORT
    if report_path.exists():
        _validate_shard_report(staging, recovery_commit, inventory)
    else:
        _write_json_new(
            report_path,
            _build_shard_report(
                recovery_commit=recovery_commit,
                rows=rows,
                inventory=inventory,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_sec=time.perf_counter() - started,
            ),
        )
        _validate_shard_report(staging, recovery_commit, inventory)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return destination / recovery_protocol.SHARD_REPORT


def _generation_input_sha256() -> dict[str, dict[str, str]]:
    return {
        dataset: {
            key: spec["input_sha256"][key]
            for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in generation_protocol.DATASETS.items()
    }


def _build_collection_report(
    *,
    recovery_commit: str,
    rows: Sequence[dict[str, Any]],
    shard_report: dict[str, Any],
    shard_report_sha256: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    shard_id = generation_protocol.LOCAL_SHARD
    return {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (
            recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
        ),
        "recovery_protocol": recovery_protocol.json_roundtrip(
            recovery_protocol.recovery_protocol_manifest(_repo_root())
        ),
        "source_generation_protocol_sha256": (
            recovery_protocol.SOURCE_PROTOCOL_SHA256
        ),
        "source_generation_protocol": recovery_protocol.json_roundtrip(
            generation_protocol.frozen_protocol_manifest()
        ),
        "generation_execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "source_generation_runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (
            recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256
        ),
        "recovery_commit": recovery_commit,
        "recovery_collector_sha256": recovery_protocol.file_sha256(Path(__file__)),
        "recovery_inventory_sha256": recovery_protocol.RECOVERY_INVENTORY_SHA256,
        "generation_input_sha256": _generation_input_sha256(),
        "shard_report_sha256": {shard_id: shard_report_sha256},
        "shard_report_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / recovery_protocol.SHARD_REPORT),
                "sha256": shard_report_sha256,
            }
        },
        "source_staging_manifest_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / STAGING_MANIFEST),
                "sha256": inventory["shards"][shard_id][
                    "staging_manifest_sha256"
                ],
            }
        },
        "source_expected_execution_environment": {
            shard_id: shard_report["source_expected_execution_environment"]
        },
        "source_gpu_monitoring_evidence": {
            shard_id: shard_report["source_gpu_monitoring_evidence"]
        },
        "recovery_execution": {
            "shard_order": [shard_id],
            "resumed_case_counts": {shard_id: 30},
            "new_generation_case_count": 0,
            "generator_invoked": False,
            "gpu_accessed": False,
            "raw_reference_data_accessed": False,
            "quality_metrics_interpreted": False,
        },
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "raw_results": list(rows),
        "timing_summary": _timing_summary(rows),
        "termination_reason_counts": _termination_counts(rows),
        "collection_audit": _recovery_audit(),
        "formal_result_valid": True,
        "formal_collection_recovered": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            recovery_protocol.MONITORING_LIMITATION_CODE
        ),
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def merge_shard(confirmed_recovery_sha256: str) -> Path:
    recovery_protocol.require_recovery_confirmation(confirmed_recovery_sha256)
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    recovery_commit = _assert_clean_recovery_tree(root)
    inventory = recovery_protocol.recovery_inventory(root)
    destination = root / recovery_protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"Stage 6E 恢复 collection 已存在，不覆盖：{destination}")
    shard_id = generation_protocol.LOCAL_SHARD
    source_root = root / recovery_protocol.SHARD_OUTPUT_ROOT / shard_id
    shard_report, rows, shard_sha = _validate_shard_report(
        source_root, recovery_commit, inventory
    )
    report = _build_collection_report(
        recovery_commit=recovery_commit,
        rows=rows,
        shard_report=shard_report,
        shard_report_sha256=shard_sha,
        inventory=inventory,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.recovery-merge-", dir=destination.parent
        )
    )
    try:
        for task in generation_protocol.task_plan().tasks:
            source = source_root / "cases" / task.task_id
            target = staging / "cases" / task.task_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        copied_rows = _validate_case_matrix(
            staging, generation_protocol.task_plan().tasks, inventory
        )
        if copied_rows != rows:
            raise RuntimeError("Stage 6E 恢复合并后 case 行漂移")
        evidence = staging / "shards" / shard_id
        evidence.mkdir(parents=True, exist_ok=True)
        for name, expected_sha in (
            (recovery_protocol.SHARD_REPORT, shard_sha),
            (
                STAGING_MANIFEST,
                inventory["shards"][shard_id]["staging_manifest_sha256"],
            ),
        ):
            shutil.copy2(source_root / name, evidence / name)
            if recovery_protocol.file_sha256(evidence / name) != expected_sha:
                raise RuntimeError(f"Stage 6E 恢复分片证据复制漂移：{name}")
        _write_json_new(staging / recovery_protocol.COLLECTION_REPORT, report)
        os.replace(staging, destination)
    except Exception as exc:
        raise RuntimeError(
            f"Stage 6E 恢复合并失败；源分片未改动，暂存已保留：{staging}"
        ) from exc
    return destination / recovery_protocol.COLLECTION_REPORT


def _validate_commit_ancestry(
    root: Path, generation_commit: Any, recovery_commit: Any
) -> None:
    for value, name in (
        (generation_commit, "生成提交"),
        (recovery_commit, "恢复提交"),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Stage 6E {name}格式无效")
    current = _git_text(root, "rev-parse", "HEAD")
    if _git_text(root, "merge-base", generation_commit, recovery_commit) != generation_commit:
        raise RuntimeError("Stage 6E 生成提交不是恢复提交祖先")
    if _git_text(root, "merge-base", recovery_commit, current) != recovery_commit:
        raise RuntimeError("Stage 6E 恢复提交不是当前提交祖先")


def load_collection(
    root: Path,
    confirmed_collection_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    """供恢复版评价器复核完整 collection；不读取 reference 或质量字段。"""

    recovery_protocol.require_collection_confirmation(confirmed_collection_sha256)
    recovery_protocol.assert_frozen_recovery_identity(root)
    output = root / recovery_protocol.OUTPUT_DIR
    path = output / recovery_protocol.COLLECTION_REPORT
    if recovery_protocol.file_sha256(path) != confirmed_collection_sha256:
        raise ValueError("Stage 6E 恢复 collection SHA-256 与确认值不一致")
    report = _load_json(path)
    inventory = recovery_protocol.recovery_inventory(root)
    required = {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (
            recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
        ),
        "source_generation_protocol_sha256": (
            recovery_protocol.SOURCE_PROTOCOL_SHA256
        ),
        "generation_execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "source_generation_runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (
            recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256
        ),
        "recovery_inventory_sha256": recovery_protocol.RECOVERY_INVENTORY_SHA256,
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "formal_result_valid": True,
        "formal_collection_recovered": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            recovery_protocol.MONITORING_LIMITATION_CODE
        ),
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("Stage 6E 恢复 collection 顶层契约漂移")
    if report.get("recovery_protocol") != recovery_protocol.json_roundtrip(
        recovery_protocol.recovery_protocol_manifest(root)
    ) or report.get("source_generation_protocol") != recovery_protocol.json_roundtrip(
        generation_protocol.frozen_protocol_manifest()
    ):
        raise RuntimeError("Stage 6E 恢复 collection 内嵌协议漂移")
    if (
        report.get("recovery_collector_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_collector"]["sha256"]
        or report.get("generation_input_sha256") != _generation_input_sha256()
        or report.get("collection_audit") != _recovery_audit()
        or report.get("timing_summary")
        != _timing_summary(report.get("raw_results", []))
        or report.get("termination_reason_counts")
        != _termination_counts(report.get("raw_results", []))
    ):
        raise RuntimeError("Stage 6E 恢复 collection 身份或汇总漂移")
    _validate_commit_ancestry(
        root,
        report.get("generation_execution_commit"),
        report.get("recovery_commit"),
    )
    shard_id = generation_protocol.LOCAL_SHARD
    expected_execution = {
        "shard_order": [shard_id],
        "resumed_case_counts": {shard_id: 30},
        "new_generation_case_count": 0,
        "generator_invoked": False,
        "gpu_accessed": False,
        "raw_reference_data_accessed": False,
        "quality_metrics_interpreted": False,
    }
    if report.get("recovery_execution") != expected_execution:
        raise RuntimeError("Stage 6E 恢复 collection 执行边界漂移")
    expected_monitoring = {shard_id: _source_monitoring_limitation()}
    if report.get("source_gpu_monitoring_evidence") != expected_monitoring:
        raise RuntimeError("Stage 6E 恢复 collection 监控限制披露漂移")

    report_meta = report.get("shard_report_artifacts", {}).get(shard_id)
    staging_meta = report.get("source_staging_manifest_artifacts", {}).get(shard_id)
    expected_report_relative = str(
        Path("shards") / shard_id / recovery_protocol.SHARD_REPORT
    )
    expected_staging_relative = str(Path("shards") / shard_id / STAGING_MANIFEST)
    shard_sha = report.get("shard_report_sha256", {}).get(shard_id)
    if (
        report_meta
        != {"path": expected_report_relative, "sha256": shard_sha}
        or staging_meta
        != {
            "path": expected_staging_relative,
            "sha256": inventory["shards"][shard_id]["staging_manifest_sha256"],
        }
    ):
        raise RuntimeError("Stage 6E 恢复 collection 分片证据地址漂移")
    for relative, digest in (
        (expected_report_relative, shard_sha),
        (expected_staging_relative, staging_meta["sha256"]),
    ):
        artifact = _safe_artifact(output, relative, relative)
        if not _is_sha256(digest) or recovery_protocol.file_sha256(artifact) != digest:
            raise RuntimeError("Stage 6E 恢复 collection 分片证据哈希漂移")

    tasks = generation_protocol.task_plan().tasks
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 30:
        raise RuntimeError("Stage 6E 恢复 collection 不是完整 30 条")
    loaded_rows = _validate_case_matrix(output, tasks, inventory)
    if loaded_rows != rows:
        raise RuntimeError("Stage 6E 恢复 collection case 与报告行不一致")
    shard_payload = _load_json(output / expected_report_relative)
    if (
        recovery_protocol.file_sha256(output / expected_report_relative) != shard_sha
        or shard_payload.get("raw_results") != rows
        or shard_payload.get("recovery_commit") != report["recovery_commit"]
    ):
        raise RuntimeError("Stage 6E 恢复 collection 与分片报告不一致")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for task, row in zip(tasks, rows, strict=True):
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError("Stage 6E 恢复 collection case 重复")
        indexed[key] = row
    return report, indexed


def build_plan() -> dict[str, Any]:
    plan = recovery_protocol.build_plan(_repo_root())
    plan.update(
        {
            "recovery_collector_wired": True,
            "closeout_commands": ["close-shard", "merge-shard"],
            "source_case_rewrite_allowed": False,
            "evaluation_performed": False,
        }
    )
    return plan


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    close = subparsers.add_parser("close-shard")
    close.add_argument("--confirm-recovery-sha", required=True)
    merge = subparsers.add_parser("merge-shard")
    merge.add_argument("--confirm-recovery-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    if args.command == "close-shard":
        path = close_shard(args.confirm_recovery_sha)
        print(f"Stage 6E 恢复 shard 报告 -> {path}")
        print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")
        return
    path = merge_shard(args.confirm_recovery_sha)
    print(f"Stage 6E 恢复 collection 报告 -> {path}")
    print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
