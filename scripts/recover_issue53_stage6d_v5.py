#!/usr/bin/env python3
"""只读复核并收口已经完成的 Issue #53 Stage 6D（阶段 6D）第五版轨迹。"""

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
from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol

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
        raise RuntimeError("恢复执行要求包含未跟踪文件在内的干净工作树")
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
        raise RuntimeError("第五版生成提交不是当前恢复提交的祖先")
    return commit


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_artifact(root: Path, relative: Any, expected: str) -> Path:
    if relative != expected:
        raise RuntimeError(
            f"恢复产物路径漂移：expected={expected}, observed={relative}"
        )
    part = Path(expected)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("恢复产物路径越界")
    base = root.resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("恢复产物路径逃逸")
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
    shard_id: str,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    path = staging / STAGING_MANIFEST
    expected_sha = inventory["shards"][shard_id]["staging_manifest_sha256"]
    if recovery_protocol.file_sha256(path) != expected_sha:
        raise RuntimeError(f"恢复暂存清单 SHA-256 漂移：{shard_id}")
    manifest = _load_json(path)
    expected_task_ids = [
        task.task_id for task in generation_protocol.tasks_for_shard(shard_id)
    ]
    required = {
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256),
        "shard_id": shard_id,
        "task_ids": expected_task_ids,
        "partial_quality_inspection_allowed": False,
    }
    if manifest != required:
        raise RuntimeError(f"恢复暂存清单内容漂移：{shard_id}")
    return manifest


def _validate_case_row(task: Any, row: dict[str, Any]) -> None:
    """按实际第五版候选预算计数语义复核，不复用旧收集器的错误断言。"""

    rounds = generation_protocol.ROUNDS
    spec = generation_protocol.DATASETS[task.dataset]
    required = {
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
        # 初始当前状态 1 次，加第 2..2500 轮前的 2499 次；终点已计入候选评价。
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
    }
    if any(row.get(key) != expected for key, expected in required.items()):
        raise RuntimeError(f"恢复 case 身份、计数或数值护栏漂移：{task.task_id}")
    for name in (
        "elapsed_sec",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "distance_evaluation_count",
        "direction_evaluation_count",
        "gap_microsteps",
        "gap_expected_microsteps",
        "factorized_gibbs_microsteps",
        "factorized_gibbs_conditional_logit_evaluated_count",
    ):
        value = row.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise RuntimeError(f"恢复 case 非负成本字段漂移：{task.task_id}/{name}")
    if (
        not _is_sha256(row.get("initial_table_sha256"))
        or not _is_sha256(row.get("primary_rng_post_initialization_sha256"))
        or not _is_sha256(row.get("primary_rng_endpoint_sha256"))
    ):
        raise RuntimeError(f"恢复 case 随机流或初表身份漂移：{task.task_id}")
    if task.arm == generation_protocol.ARM_GAP:
        if (
            row.get("gap_reference_scale_established") is not True
            or not isinstance(row.get("gap_reference_scale"), (int, float))
            or isinstance(row.get("gap_reference_scale"), bool)
            or row["gap_reference_scale"] <= 0
            or row.get("gap_microsteps") != row.get("gap_expected_microsteps")
        ):
            raise RuntimeError(f"恢复缺口核尺度或 8*K 汇总漂移：{task.task_id}")
    elif row.get("gap_microsteps") != 0 or row.get("gap_expected_microsteps") != 0:
        raise RuntimeError(f"恢复基线包含缺口核微步：{task.task_id}")


def _validate_transition(
    path: Path,
    task: Any,
    row: dict[str, Any],
) -> None:
    transition = _load_json(path)
    rounds = generation_protocol.ROUNDS
    clocks = transition.get("clocks")
    if (
        transition.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or transition.get("protocol_sha256") != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or transition.get("task_id") != task.task_id
        or transition.get("round_count") != rounds
        or transition.get("all_applied_unconditionally") is not True
        or not isinstance(clocks, list)
        or len(clocks) != rounds
    ):
        raise RuntimeError(f"恢复无门控转移总结构漂移：{task.task_id}")
    clock_microsteps = 0
    for index, clock in enumerate(clocks, start=1):
        if (
            not isinstance(clock, dict)
            or clock.get("round") != index
            or clock.get("state_index") != index
            or clock.get("accepted_attempt") != 1
            or clock.get("candidate_evaluation_count_cumulative") != index
            or isinstance(clock.get("gibbs_microsteps"), bool)
            or not isinstance(clock.get("gibbs_microsteps"), int)
            or clock["gibbs_microsteps"] < 0
        ):
            raise RuntimeError(f"恢复无门控转移时钟漂移：{task.task_id}/{index}")
        clock_microsteps += clock["gibbs_microsteps"]
    gap_rounds = transition.get("gap_rounds")
    if not isinstance(gap_rounds, list):
        raise TypeError(f"恢复缺口轮次结构缺失：{task.task_id}")
    if task.arm == generation_protocol.ARM_GAP:
        if len(gap_rounds) != rounds:
            raise RuntimeError(f"恢复缺口轮次数量漂移：{task.task_id}")
        gap_total = 0
        for index, item in enumerate(gap_rounds, start=1):
            active = item.get("active_switches_k") if isinstance(item, dict) else None
            expected = (
                8 * active
                if isinstance(active, int) and not isinstance(active, bool)
                else None
            )
            if (
                not isinstance(item, dict)
                or item.get("round") != index
                or item.get("n_sweeps") != 8
                or item.get("scan_applied") is not True
                or item.get("clip_hit_count") != 0
                or item.get("nonfinite_condition_count") != 0
                or item.get("expected_microsteps") != expected
                or item.get("microsteps") != expected
            ):
                raise RuntimeError(f"恢复缺口 8*K 结构漂移：{task.task_id}/{index}")
            gap_total += expected
        if gap_total != row["gap_microsteps"] or clock_microsteps != gap_total:
            raise RuntimeError(f"恢复缺口 8*K 总量漂移：{task.task_id}")
    elif gap_rounds:
        raise RuntimeError(f"恢复基线出现缺口核轮次：{task.task_id}")
    elif (
        task.arm == "factor_b_s8"
        and clock_microsteps != row["factorized_gibbs_microsteps"]
    ):
        raise RuntimeError(f"恢复 Gibbs（吉布斯）微步总量漂移：{task.task_id}")
    elif task.arm == "independent_b_s0" and clock_microsteps != 0:
        raise RuntimeError(f"恢复独立基线出现 Gibbs（吉布斯）微步：{task.task_id}")


def _validate_case(
    artifact_root: Path,
    task: Any,
    inventory: dict[str, Any],
) -> dict[str, Any]:
    shard_id = generation_protocol.task_shard_id(task)
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
        raise RuntimeError(f"恢复 case 文件集合漂移：{task.task_id}")
    manifest_path = case_dir / CASE_MANIFEST
    expected_manifest_sha = inventory["shards"][shard_id]["case_manifest_sha256"][
        task.task_id
    ]
    if recovery_protocol.file_sha256(manifest_path) != expected_manifest_sha:
        raise RuntimeError(f"恢复 case 清单 SHA-256 漂移：{task.task_id}")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or manifest.get("protocol_sha256") != recovery_protocol.SOURCE_PROTOCOL_SHA256
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
        raise RuntimeError(f"恢复 case 第五版来源身份漂移：{task.task_id}")
    row = manifest.get("collection_row")
    if not isinstance(row, dict):
        raise TypeError(f"恢复 case 缺少 collection 行：{task.task_id}")
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
                f"恢复 case 附件 SHA-256 漂移：{task.task_id}/{path_key}"
            )
        resolved[path_key] = path
    # 检查点只验证冻结字节哈希；质量字段留到用户确认后的离线评价阶段。
    _validate_transition(resolved["transition_audit_path"], task, row)
    return row


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
        raise RuntimeError("恢复 case 目录集合与冻结任务不一致")
    rows = [_validate_case(artifact_root, task, inventory) for task in tasks]
    _validate_pairing_for_tasks(rows, tasks)
    return rows


def _validate_pairing_for_tasks(
    rows: Sequence[dict[str, Any]],
    tasks: Sequence[Any],
) -> None:
    if [row.get("task_id") for row in rows] != [task.task_id for task in tasks]:
        raise RuntimeError("恢复三方法配对行顺序漂移")
    indexed = {(row["seed"], row["dataset"], row["arm"]): row for row in rows}
    if len(indexed) != len(tasks):
        raise RuntimeError("恢复三方法配对地址重复")
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
                raise RuntimeError(f"恢复三方法配对身份漂移：{seed}/{dataset}/{key}")


def _build_shard_report(
    *,
    shard_id: str,
    recovery_commit: str,
    rows: Sequence[dict[str, Any]],
    inventory: dict[str, Any],
    started_at: str,
    finished_at: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    tasks = generation_protocol.tasks_for_shard(shard_id)
    task_ids = [task.task_id for task in tasks]
    if [row["task_id"] for row in rows] != task_ids:
        raise RuntimeError(f"恢复分片行顺序漂移：{shard_id}")
    count = len(tasks)
    return {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256),
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
            "gpu": dict(generation_protocol.EXECUTION_SHARDS[shard_id]["expected_gpu"]),
            "software": dict(generation_protocol.EXPECTED_SOFTWARE),
            "status": "frozen_expectation_not_reconstructed_runtime_sample",
        },
        "source_gpu_monitoring_evidence": _source_monitoring_limitation(),
        "recovery_execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec": float(elapsed_sec),
            "resumed_case_count": count,
            "new_generation_case_count": 0,
            "generator_invoked": False,
            "gpu_accessed": False,
            "raw_reference_data_accessed": False,
            "quality_metrics_interpreted": False,
        },
        "case_count": count,
        "paired_dataset_seed_count": len(generation_protocol.SHARD_BLOCKS[shard_id]),
        "task_ids": task_ids,
        "case_manifest_sha256": dict(
            inventory["shards"][shard_id]["case_manifest_sha256"]
        ),
        "raw_results": list(rows),
        "recovery_audit": {
            "all_assigned_cases_present": True,
            "all_case_manifests_match_frozen_inventory": True,
            "all_case_attachments_sha256_verified": True,
            "all_candidate_budget_counts_use_actual_semantics": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_gap_8k_identity": True,
            "all_zero_clip_and_finite": True,
            "exact_shard_resumed_count": True,
            "zero_new_generation_cases": True,
            "source_case_files_rewritten": False,
            "gpu_monitoring_limitation_disclosed": True,
        },
        "formal_shard_recovered": True,
        "raw_reference_data_accessed": False,
        "method_comparison_emitted": False,
        "l1_results_published": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def _validate_shard_report(
    shard_root: Path,
    shard_id: str,
    recovery_commit: str,
    rows: Sequence[dict[str, Any]],
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    path = shard_root / recovery_protocol.SHARD_REPORT
    report_sha = recovery_protocol.file_sha256(path)
    report = _load_json(path)
    tasks = generation_protocol.tasks_for_shard(shard_id)
    count = len(tasks)
    execution = report.get("recovery_execution")
    source = report.get("source_generation")
    expected_environment = {
        "hostname": generation_protocol.EXECUTION_SHARDS[shard_id]["hostname"],
        "gpu": dict(generation_protocol.EXECUTION_SHARDS[shard_id]["expected_gpu"]),
        "software": dict(generation_protocol.EXPECTED_SOFTWARE),
        "status": "frozen_expectation_not_reconstructed_runtime_sample",
    }
    expected_source = {
        "contract_version": generation_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256,
    }
    if (
        report.get("contract_version") != recovery_protocol.RECOVERY_PROTOCOL_VERSION
        or report.get("recovery_protocol_sha256")
        != recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256
        or source != expected_source
        or report.get("recovery_commit") != recovery_commit
        or report.get("recovery_collector_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_collector"]["sha256"]
        or report.get("recovery_inventory_sha256")
        != recovery_protocol.RECOVERY_INVENTORY_SHA256
        or report.get("shard_id") != shard_id
        or report.get("source_hostname")
        != inventory["shards"][shard_id]["source_hostname"]
        or report.get("source_expected_execution_environment") != expected_environment
        or report.get("task_ids") != [task.task_id for task in tasks]
        or report.get("case_count") != count
        or report.get("paired_dataset_seed_count")
        != len(generation_protocol.SHARD_BLOCKS[shard_id])
        or report.get("case_manifest_sha256")
        != inventory["shards"][shard_id]["case_manifest_sha256"]
        or report.get("raw_results") != list(rows)
        or report.get("source_gpu_monitoring_evidence")
        != _source_monitoring_limitation()
        or not isinstance(execution, dict)
        or not isinstance(execution.get("started_at"), str)
        or not isinstance(execution.get("finished_at"), str)
        or isinstance(execution.get("elapsed_sec"), bool)
        or not isinstance(execution.get("elapsed_sec"), (int, float))
        or not math.isfinite(float(execution["elapsed_sec"]))
        or execution["elapsed_sec"] < 0
        or execution.get("resumed_case_count") != count
        or execution.get("new_generation_case_count") != 0
        or execution.get("generator_invoked") is not False
        or execution.get("gpu_accessed") is not False
        or execution.get("raw_reference_data_accessed") is not False
        or execution.get("quality_metrics_interpreted") is not False
        or report.get("formal_shard_recovered") is not True
        or report.get("raw_reference_data_accessed") is not False
        or report.get("method_comparison_emitted") is not False
        or report.get("l1_results_published") is not False
        or report.get("parameter_retuning_performed") is not False
        or report.get("privacy_budget_consumed") is not False
    ):
        raise RuntimeError(f"恢复分片报告身份或边界漂移：{shard_id}")
    expected_staging = {
        "path": STAGING_MANIFEST,
        "sha256": inventory["shards"][shard_id]["staging_manifest_sha256"],
        "observed_original_basename": inventory["shards"][shard_id][
            "source_staging_basename"
        ],
    }
    if report.get("source_staging_manifest") != expected_staging:
        raise RuntimeError(f"恢复分片暂存来源证据漂移：{shard_id}")
    expected_audit = {
        "all_assigned_cases_present": True,
        "all_case_manifests_match_frozen_inventory": True,
        "all_case_attachments_sha256_verified": True,
        "all_candidate_budget_counts_use_actual_semantics": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "exact_shard_resumed_count": True,
        "zero_new_generation_cases": True,
        "source_case_files_rewritten": False,
        "gpu_monitoring_limitation_disclosed": True,
    }
    if report.get("recovery_audit") != expected_audit:
        raise RuntimeError(f"恢复分片审计字段漂移：{shard_id}")
    return report, report_sha


def validate_recovered_shard(
    root: Path,
    shard_id: str,
    recovery_commit: str,
    inventory: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    shard_root = root / recovery_protocol.SHARD_OUTPUT_ROOT / shard_id
    _validate_staging_manifest(shard_root, shard_id, inventory)
    rows = _validate_case_matrix(
        shard_root,
        generation_protocol.tasks_for_shard(shard_id),
        inventory,
    )
    report, report_sha = _validate_shard_report(
        shard_root, shard_id, recovery_commit, rows, inventory
    )
    return report, rows, report_sha


def _find_frozen_staging(
    root: Path,
    shard_id: str,
    inventory: dict[str, Any],
) -> Path:
    base = root / recovery_protocol.SHARD_OUTPUT_ROOT
    expected = inventory["shards"][shard_id]
    matches = []
    for candidate in sorted(base.glob(f".{shard_id}.staging-*")):
        manifest = candidate / STAGING_MANIFEST
        if (
            manifest.is_file()
            and recovery_protocol.file_sha256(manifest)
            == expected["staging_manifest_sha256"]
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(f"必须恰好找到一个冻结恢复暂存目录：{shard_id}")
    if matches[0].name != expected["source_staging_basename"]:
        raise RuntimeError(f"恢复暂存目录名与结果前清单不一致：{shard_id}")
    return matches[0]


def close_shard(confirmed_recovery_sha256: str, shard_id: str) -> Path:
    recovery_protocol.require_recovery_confirmation(confirmed_recovery_sha256)
    if shard_id not in generation_protocol.SHARD_ORDER:
        raise ValueError(f"未知恢复分片：{shard_id!r}")
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    recovery_commit = _assert_clean_recovery_tree(root)
    inventory = recovery_protocol.recovery_inventory(root)
    destination = root / recovery_protocol.SHARD_OUTPUT_ROOT / shard_id
    if destination.exists():
        validate_recovered_shard(root, shard_id, recovery_commit, inventory)
        return destination / recovery_protocol.SHARD_REPORT
    if (root / recovery_protocol.OUTPUT_DIR).exists():
        raise FileExistsError("完整正式输出已存在，不允许再收口分片")
    staging = _find_frozen_staging(root, shard_id, inventory)
    _validate_staging_manifest(staging, shard_id, inventory)
    tasks = generation_protocol.tasks_for_shard(shard_id)
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    rows = _validate_case_matrix(staging, tasks, inventory)
    finished_at = datetime.now().astimezone().isoformat()
    report_path = staging / recovery_protocol.SHARD_REPORT
    if report_path.exists():
        _validate_shard_report(staging, shard_id, recovery_commit, rows, inventory)
    else:
        report = _build_shard_report(
            shard_id=shard_id,
            recovery_commit=recovery_commit,
            rows=rows,
            inventory=inventory,
            started_at=started_at,
            finished_at=finished_at,
            elapsed_sec=time.perf_counter() - started,
        )
        _write_json_new(report_path, report)
        _validate_shard_report(staging, shard_id, recovery_commit, rows, inventory)
    os.replace(staging, destination)
    return destination / recovery_protocol.SHARD_REPORT


def _generation_input_sha256() -> dict[str, dict[str, str]]:
    return {
        dataset: {
            key: spec["input_sha256"][key] for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in generation_protocol.DATASETS.items()
    }


def _build_collection_report(
    *,
    recovery_commit: str,
    rows: Sequence[dict[str, Any]],
    shard_reports: dict[str, dict[str, Any]],
    shard_report_sha256: dict[str, str],
    inventory: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = [task.task_id for task in generation_protocol.task_plan().tasks]
    if [row["task_id"] for row in rows] != expected_ids:
        raise RuntimeError("恢复 collection 行顺序漂移")
    return {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256),
        "recovery_protocol": recovery_protocol.recovery_protocol_manifest(_repo_root()),
        "source_generation_protocol_sha256": (recovery_protocol.SOURCE_PROTOCOL_SHA256),
        "source_generation_protocol": generation_protocol.frozen_protocol_manifest(),
        "generation_execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "source_generation_runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256),
        "recovery_commit": recovery_commit,
        "recovery_collector_sha256": recovery_protocol.file_sha256(Path(__file__)),
        "recovery_inventory_sha256": recovery_protocol.RECOVERY_INVENTORY_SHA256,
        "generation_input_sha256": _generation_input_sha256(),
        "shard_report_sha256": dict(shard_report_sha256),
        "shard_report_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / recovery_protocol.SHARD_REPORT),
                "sha256": shard_report_sha256[shard_id],
            }
            for shard_id in generation_protocol.SHARD_ORDER
        },
        "source_staging_manifest_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / STAGING_MANIFEST),
                "sha256": inventory["shards"][shard_id]["staging_manifest_sha256"],
            }
            for shard_id in generation_protocol.SHARD_ORDER
        },
        "source_gpu_monitoring_evidence": {
            shard_id: shard_reports[shard_id]["source_gpu_monitoring_evidence"]
            for shard_id in generation_protocol.SHARD_ORDER
        },
        "recovery_execution": {
            "shard_order": list(generation_protocol.SHARD_ORDER),
            "resumed_case_counts": {
                shard_id: len(generation_protocol.tasks_for_shard(shard_id))
                for shard_id in generation_protocol.SHARD_ORDER
            },
            "new_generation_case_count": 0,
            "generator_invoked": False,
            "gpu_accessed": False,
            "raw_reference_data_accessed": False,
            "quality_metrics_interpreted": False,
        },
        "case_count": len(rows),
        "paired_dataset_seed_count": 10,
        "raw_results": list(rows),
        "collection_audit": {
            "all_30_cases_present": True,
            "all_10_dataset_seed_triplets_paired": True,
            "exact_21_9_shard_assignment": True,
            "both_recovered_shard_reports_verified_before_merge": True,
            "all_case_manifests_match_frozen_inventory": True,
            "all_case_attachments_sha256_verified": True,
            "all_candidate_budget_counts_use_actual_semantics": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_gap_8k_identity": True,
            "all_zero_clip_and_finite": True,
            "zero_new_generation_cases": True,
            "source_case_files_rewritten": False,
            "gpu_monitoring_limitation_disclosed": True,
        },
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


def merge_shards(confirmed_recovery_sha256: str) -> Path:
    recovery_protocol.require_recovery_confirmation(confirmed_recovery_sha256)
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    recovery_commit = _assert_clean_recovery_tree(root)
    inventory = recovery_protocol.recovery_inventory(root)
    destination = root / recovery_protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"恢复 collection 已存在，不覆盖：{destination}")
    shard_reports: dict[str, dict[str, Any]] = {}
    shard_rows: dict[str, dict[str, dict[str, Any]]] = {}
    shard_report_sha256: dict[str, str] = {}
    for shard_id in generation_protocol.SHARD_ORDER:
        report, rows, report_sha = validate_recovered_shard(
            root, shard_id, recovery_commit, inventory
        )
        shard_reports[shard_id] = report
        shard_rows[shard_id] = {row["task_id"]: row for row in rows}
        shard_report_sha256[shard_id] = report_sha
    rows = [
        shard_rows[generation_protocol.task_shard_id(task)][task.task_id]
        for task in generation_protocol.task_plan().tasks
    ]
    report = _build_collection_report(
        recovery_commit=recovery_commit,
        rows=rows,
        shard_reports=shard_reports,
        shard_report_sha256=shard_report_sha256,
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
            shard_id = generation_protocol.task_shard_id(task)
            source = (
                root
                / recovery_protocol.SHARD_OUTPUT_ROOT
                / shard_id
                / "cases"
                / task.task_id
            )
            target = staging / "cases" / task.task_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
            if (
                _validate_case(staging, task, inventory)
                != shard_rows[shard_id][task.task_id]
            ):
                raise RuntimeError(f"恢复合并后 case 行漂移：{task.task_id}")
        for shard_id in generation_protocol.SHARD_ORDER:
            source_root = root / recovery_protocol.SHARD_OUTPUT_ROOT / shard_id
            target_root = staging / "shards" / shard_id
            target_root.mkdir(parents=True, exist_ok=True)
            for name, expected_sha in (
                (recovery_protocol.SHARD_REPORT, shard_report_sha256[shard_id]),
                (
                    STAGING_MANIFEST,
                    inventory["shards"][shard_id]["staging_manifest_sha256"],
                ),
            ):
                shutil.copy2(source_root / name, target_root / name)
                if recovery_protocol.file_sha256(target_root / name) != expected_sha:
                    raise RuntimeError(f"恢复合并分片证据漂移：{shard_id}/{name}")
        _write_json_new(staging / recovery_protocol.COLLECTION_REPORT, report)
        os.replace(staging, destination)
    except Exception as exc:
        raise RuntimeError(
            f"恢复双分片合并失败；源分片未改动，合并暂存已保留：{staging}"
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
            raise RuntimeError(f"{name}格式无效")
    current = _git_text(root, "rev-parse", "HEAD")
    if (
        _git_text(root, "merge-base", generation_commit, recovery_commit)
        != generation_commit
    ):
        raise RuntimeError("生成提交不是恢复提交祖先")
    if _git_text(root, "merge-base", recovery_commit, current) != recovery_commit:
        raise RuntimeError("恢复提交不是当前提交祖先")


def load_collection(
    root: Path,
    confirmed_collection_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    """为恢复版评价器验证完整 collection；不读取参考数据或质量字段。"""

    recovery_protocol.require_collection_confirmation(confirmed_collection_sha256)
    output = root / recovery_protocol.OUTPUT_DIR
    report_path = output / recovery_protocol.COLLECTION_REPORT
    if recovery_protocol.file_sha256(report_path) != confirmed_collection_sha256:
        raise ValueError("恢复 collection 报告 SHA-256 与确认值不一致")
    report = _load_json(report_path)
    inventory = recovery_protocol.recovery_inventory(root)
    required = {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": (recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256),
        "source_generation_protocol_sha256": (recovery_protocol.SOURCE_PROTOCOL_SHA256),
        "generation_execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "source_generation_runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": (recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256),
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
    if any(report.get(key) != expected for key, expected in required.items()):
        raise RuntimeError("恢复 collection 顶层身份或信息边界漂移")
    if report.get("recovery_protocol") != recovery_protocol.recovery_protocol_manifest(
        root
    ):
        raise RuntimeError("恢复 collection 内嵌恢复协议漂移")
    if (
        report.get("source_generation_protocol")
        != generation_protocol.frozen_protocol_manifest()
    ):
        raise RuntimeError("恢复 collection 内嵌生成协议漂移")
    if (
        report.get("recovery_collector_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_collector"]["sha256"]
    ):
        raise RuntimeError("恢复 collection 收口器身份漂移")
    if report.get("generation_input_sha256") != _generation_input_sha256():
        raise RuntimeError("恢复 collection 生成输入身份漂移")
    expected_collection_audit = {
        "all_30_cases_present": True,
        "all_10_dataset_seed_triplets_paired": True,
        "exact_21_9_shard_assignment": True,
        "both_recovered_shard_reports_verified_before_merge": True,
        "all_case_manifests_match_frozen_inventory": True,
        "all_case_attachments_sha256_verified": True,
        "all_candidate_budget_counts_use_actual_semantics": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "zero_new_generation_cases": True,
        "source_case_files_rewritten": False,
        "gpu_monitoring_limitation_disclosed": True,
    }
    if report.get("collection_audit") != expected_collection_audit:
        raise RuntimeError("恢复 collection 审计字段漂移")
    expected_monitoring = {
        shard_id: _source_monitoring_limitation()
        for shard_id in generation_protocol.SHARD_ORDER
    }
    if report.get("source_gpu_monitoring_evidence") != expected_monitoring:
        raise RuntimeError("恢复 collection 显卡监控限制披露漂移")
    _validate_commit_ancestry(
        root,
        report.get("generation_execution_commit"),
        report.get("recovery_commit"),
    )
    execution = report.get("recovery_execution")
    if execution != {
        "shard_order": list(generation_protocol.SHARD_ORDER),
        "resumed_case_counts": {
            shard_id: len(generation_protocol.tasks_for_shard(shard_id))
            for shard_id in generation_protocol.SHARD_ORDER
        },
        "new_generation_case_count": 0,
        "generator_invoked": False,
        "gpu_accessed": False,
        "raw_reference_data_accessed": False,
        "quality_metrics_interpreted": False,
    }:
        raise RuntimeError("恢复 collection 21/9 计数或执行边界漂移")
    shard_payloads: dict[str, dict[str, Any]] = {}
    shard_hashes = report.get("shard_report_sha256")
    shard_artifacts = report.get("shard_report_artifacts")
    staging_artifacts = report.get("source_staging_manifest_artifacts")
    if not all(
        isinstance(value, dict)
        for value in (shard_hashes, shard_artifacts, staging_artifacts)
    ):
        raise TypeError("恢复 collection 分片证据缺失")
    for shard_id in generation_protocol.SHARD_ORDER:
        report_relative = str(
            Path("shards") / shard_id / recovery_protocol.SHARD_REPORT
        )
        staging_relative = str(Path("shards") / shard_id / STAGING_MANIFEST)
        report_artifact = shard_artifacts.get(shard_id)
        staging_artifact = staging_artifacts.get(shard_id)
        if (
            not isinstance(report_artifact, dict)
            or report_artifact.get("path") != report_relative
            or report_artifact.get("sha256") != shard_hashes.get(shard_id)
            or not isinstance(staging_artifact, dict)
            or staging_artifact.get("path") != staging_relative
            or staging_artifact.get("sha256")
            != inventory["shards"][shard_id]["staging_manifest_sha256"]
        ):
            raise RuntimeError(f"恢复 collection 分片证据地址漂移：{shard_id}")
        for relative, digest in (
            (report_relative, report_artifact["sha256"]),
            (staging_relative, staging_artifact["sha256"]),
        ):
            artifact = _safe_artifact(output, relative, relative)
            if recovery_protocol.file_sha256(artifact) != digest:
                raise RuntimeError(f"恢复 collection 分片证据哈希漂移：{shard_id}")
        shard_payloads[shard_id] = _load_json(output / report_relative)
    rows = report.get("raw_results")
    tasks = generation_protocol.task_plan().tasks
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("恢复 collection 不是完整 30 条")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    loaded_rows = _validate_case_matrix(output, tasks, inventory)
    if loaded_rows != rows:
        raise RuntimeError("恢复 collection case 与报告行不一致")
    for task, row in zip(tasks, rows):
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError("恢复 collection case 重复")
        indexed[key] = row
    for shard_id in generation_protocol.SHARD_ORDER:
        expected_rows = [
            indexed[(task.seed, task.dataset, task.arm)]
            for task in generation_protocol.tasks_for_shard(shard_id)
        ]
        payload = shard_payloads[shard_id]
        report_rows = [
            indexed[(task.seed, task.dataset, task.arm)]
            for task in generation_protocol.tasks_for_shard(shard_id)
        ]
        validated_payload, validated_sha = _validate_shard_report(
            output / "shards" / shard_id,
            shard_id,
            report["recovery_commit"],
            report_rows,
            inventory,
        )
        if (
            validated_payload != payload
            or validated_sha != shard_hashes[shard_id]
            or payload.get("raw_results") != expected_rows
            or payload.get("source_gpu_monitoring_evidence")
            != _source_monitoring_limitation()
            or payload.get("recovery_execution", {}).get("resumed_case_count")
            != len(expected_rows)
            or payload.get("recovery_execution", {}).get("new_generation_case_count")
            != 0
        ):
            raise RuntimeError(f"恢复 collection 与分片报告不一致：{shard_id}")
    return report, indexed


def build_plan() -> dict[str, Any]:
    plan = recovery_protocol.build_plan(_repo_root())
    plan.update(
        {
            "recovery_collector_wired": True,
            "closeout_commands": ["close-shard", "merge-shards"],
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
    close.add_argument(
        "--shard", choices=generation_protocol.SHARD_ORDER, required=True
    )
    close.add_argument("--confirm-recovery-sha", required=True)
    merge = subparsers.add_parser("merge-shards")
    merge.add_argument("--confirm-recovery-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    if args.command == "close-shard":
        path = close_shard(args.confirm_recovery_sha, args.shard)
        print(f"恢复 shard（分片）报告 -> {path}")
        print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")
        return
    path = merge_shards(args.confirm_recovery_sha)
    print(f"恢复 collection（采集）报告 -> {path}")
    print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
