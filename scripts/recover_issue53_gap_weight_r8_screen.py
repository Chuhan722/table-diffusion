#!/usr/bin/env python3
"""结果盲复核并收口已经生成完成的 Issue #53 R8 四条轨迹。"""

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

from scripts import issue53_gap_weight_r8_screen_execution_protocol as source_protocol
from scripts import issue53_gap_weight_r8_screen_recovery_protocol as recovery_protocol
from scripts import run_issue53_gap_weight_r8_screen as source_collector


CASE_MANIFEST = source_collector.CASE_MANIFEST
TERMINAL_TABLE = source_collector.TERMINAL_TABLE
CHECKPOINT_ARTIFACT = source_collector.CHECKPOINT_ARTIFACT
TRANSITION_AUDIT = source_collector.TRANSITION_AUDIT
STAGING_MANIFEST = "staging_manifest.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strict_json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"R8 恢复 JSON 根必须是对象：{path}")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"R8 恢复报告已存在，不覆盖：{path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"R8 恢复临时报告已存在：{temporary}")
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
        raise RuntimeError("R8 恢复执行要求包含未跟踪文件在内的干净工作树")
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
        raise RuntimeError("R8 生成提交不是当前恢复提交的祖先")
    return commit


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _nonnegative_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _safe_artifact(root: Path, relative: Any, expected: str) -> Path:
    if relative != expected:
        raise RuntimeError(
            f"R8 恢复产物路径漂移：expected={expected}, observed={relative}"
        )
    part = Path(expected)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("R8 恢复产物路径越界")
    base = root.resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("R8 恢复产物路径逃逸")
    return resolved


def _expected_staging_manifest() -> dict[str, Any]:
    shard_id = source_protocol.LOCAL_SHARD
    return {
        "contract_version": source_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "shard_id": shard_id,
        "shard_assignment_sha256": source_protocol.shard_assignment_sha256(),
        "generator_params_manifest_sha256": (
            source_protocol.generator_params_manifest_sha256()
        ),
        "task_ids": [
            task.task_id for task in source_protocol.tasks_for_shard(shard_id)
        ],
        "partial_quality_inspection_allowed": False,
    }


def _validate_staging_manifest(root: Path) -> dict[str, Any]:
    path = root / STAGING_MANIFEST
    if (
        recovery_protocol.file_sha256(path)
        != recovery_protocol.SOURCE_STAGING_MANIFEST_SHA256
    ):
        raise RuntimeError("R8 恢复暂存清单 SHA-256 漂移")
    manifest = _load_json(path)
    if manifest != _expected_staging_manifest():
        raise RuntimeError("R8 恢复暂存清单内容漂移")
    return manifest


def _legacy_validation_proxy(row: dict[str, Any]) -> dict[str, Any]:
    """仅供调用旧校验器；绝不写回冻结 row。"""

    applied = row.get("applied_rounds")
    if not _nonnegative_int(applied):
        raise RuntimeError("R8 恢复 applied_rounds 非法")
    proxy = dict(row)
    proxy["state_evaluation_count"] = applied + 1
    return proxy


def _validate_case_row(task: Any, row: dict[str, Any]) -> None:
    """只勘误状态评价次数；旧 Stage 6E 与 R8 护栏继续原样执行。"""

    applied = row.get("applied_rounds")
    if (
        not _nonnegative_int(applied)
        or applied > source_protocol.ROUND_CAP
        or row.get("state_evaluation_count") != max(1, applied)
    ):
        raise RuntimeError(f"R8 恢复状态评价次数漂移：{task.task_id}")
    with source_collector._execution_runtime():
        source_collector._validate_case_row(
            task,
            _legacy_validation_proxy(row),
        )

    spec = source_protocol.DATASETS[task.dataset]
    required = {
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
        "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        "gap_8k_identity": True,
        "gap_clip_hit_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
        "direction_logit_clipped_count": 0,
        "nonfinite_count": 0,
        "factorized_gibbs_microsteps": 0,
        "factorized_gibbs_conditional_logit_evaluated_count": 0,
    }
    if any(row.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"R8 恢复 case 身份或数值护栏漂移：{task.task_id}")
    for name in (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "distance_evaluation_count",
        "direction_evaluation_count",
        "direction_logit_evaluated_count",
        "gap_microsteps",
        "gap_expected_microsteps",
    ):
        if not _nonnegative_number(row.get(name)):
            raise RuntimeError(f"R8 恢复成本字段漂移：{task.task_id}/{name}")
    if (
        row.get("distance_evaluation_count") != applied
        or row.get("direction_evaluation_count") != applied
        or row.get("gap_reference_scale_established") is not bool(applied)
        or (applied and not _nonnegative_number(row.get("gap_reference_scale")))
        or (applied and float(row["gap_reference_scale"]) <= 0.0)
        or (not applied and row.get("gap_reference_scale") is not None)
        or row.get("gap_microsteps") != row.get("gap_expected_microsteps")
        or not _nonnegative_number(row.get("direction_reference_scale"))
        or float(row["direction_reference_scale"]) <= 0.0
        or not all(
            _is_sha256(row.get(name))
            for name in (
                "initial_table_sha256",
                "primary_rng_post_initialization_sha256",
                "primary_rng_endpoint_sha256",
            )
        )
    ):
        raise RuntimeError(f"R8 恢复成本、尺度或随机流身份漂移：{task.task_id}")


def _expected_weighting(task: Any) -> tuple[str, float | None, float | None]:
    if task.arm == "gap_legacy_s8":
        return source_protocol.LEGACY_WEIGHTING, None, None
    if task.arm == "gap_bounded_r8_s8":
        return (
            source_protocol.BOUNDED_WEIGHTING,
            float(source_protocol.MAX_WEIGHT_RATIO),
            float(source_protocol.smoothing_count(task.dataset)),
        )
    raise RuntimeError(f"R8 恢复遇到未知臂：{task.arm}")


def _validate_transition(path: Path, task: Any, row: dict[str, Any]) -> None:
    transition = _load_json(path)
    applied = row["applied_rounds"]
    clocks = transition.get("clocks")
    gap_rounds = transition.get("gap_rounds")
    factor_rounds = transition.get("factor_rounds")
    weighting = transition.get("weighting_audit")
    expected_weight, expected_ratio, expected_smoothing = _expected_weighting(task)
    if (
        transition.get("contract_version") != source_protocol.PROTOCOL_VERSION
        or transition.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or transition.get("task_id") != task.task_id
        or transition.get("round_count") != applied
        or transition.get("all_applied_unconditionally") is not True
        or not isinstance(clocks, list)
        or len(clocks) != applied
        or not isinstance(gap_rounds, list)
        or len(gap_rounds) != applied
        or factor_rounds != []
        or not isinstance(weighting, dict)
        or weighting.get("expected_weighting") != expected_weight
        or weighting.get("expected_max_weight_ratio") != expected_ratio
        or weighting.get("expected_smoothing_count") != expected_smoothing
        or weighting.get("scan_round_count")
        != row.get("gap_weighting_scan_round_count")
        or weighting.get("guard_passed") is not True
    ):
        raise RuntimeError(f"R8 恢复转移总结构或权重身份漂移：{task.task_id}")

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
            or clock.get("factorized_gibbs_rng_state_sha256") is not None
        ):
            raise RuntimeError(f"R8 恢复转移时钟漂移：{task.task_id}/{index}")
        clock_microsteps += clock["gibbs_microsteps"]

    total_microsteps = 0
    for index, item in enumerate(gap_rounds, start=1):
        active = item.get("active_switches_k") if isinstance(item, dict) else None
        expected_microsteps = 8 * active if _nonnegative_int(active) else None
        common_invalid = (
            not isinstance(item, dict)
            or item.get("round") != index
            or item.get("scan_applied") is not True
            or item.get("n_sweeps") != 8
            or item.get("microsteps") != expected_microsteps
            or item.get("expected_microsteps") != expected_microsteps
            or item.get("clip_hit_count") != 0
            or item.get("nonfinite_condition_count") != 0
            or item.get("expected_weighting") != expected_weight
            or item.get("expected_max_weight_ratio") != expected_ratio
            or item.get("expected_smoothing_count") != expected_smoothing
        )
        if task.arm == "gap_legacy_s8":
            weighting_invalid = (
                item.get("observed_kernel") != "gap_l1_global_random_scan"
                or item.get("observed_weighting")
                != source_protocol.LEGACY_WEIGHTING
                or item.get("observed_max_weight_ratio") is not None
                or item.get("observed_smoothing_count") is not None
                or item.get("observed_actual_weight_ratio") is not None
            )
        else:
            actual_ratio = item.get("observed_actual_weight_ratio")
            weighting_invalid = (
                item.get("observed_kernel")
                != "gap_l1_global_random_scan_bounded_relative"
                or item.get("observed_weighting")
                != source_protocol.BOUNDED_WEIGHTING
                or item.get("observed_max_weight_ratio") != expected_ratio
                or item.get("observed_smoothing_count") != expected_smoothing
                or not _nonnegative_number(actual_ratio)
                or not 1.0
                <= float(actual_ratio)
                <= source_protocol.MAX_WEIGHT_RATIO * (1.0 + 1e-12)
            )
        if common_invalid or weighting_invalid:
            raise RuntimeError(f"R8 恢复缺口 8*K 或权重漂移：{task.task_id}/{index}")
        total_microsteps += expected_microsteps
    if (
        total_microsteps != row["gap_microsteps"]
        or clock_microsteps != total_microsteps
    ):
        raise RuntimeError(f"R8 恢复缺口微步总量漂移：{task.task_id}")


def _validate_checkpoint(path: Path, task: Any, row: dict[str, Any]) -> None:
    """只复核检查点身份与位置，不计算或解释其中的答案和质量字段。"""

    payload = _load_json(path)
    spec = source_protocol.DATASETS[task.dataset]
    fixed = payload.get("fixed_checkpoints")
    terminal = payload.get("terminal")
    expected_states = [
        value
        for value in source_protocol.CHECKPOINT_ROUNDS
        if value <= row["applied_rounds"]
    ]
    if (
        payload.get("contract_version") != source_protocol.PROTOCOL_VERSION
        or payload.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or payload.get("task_id") != task.task_id
        or payload.get("dataset") != task.dataset
        or payload.get("arm") != task.arm
        or payload.get("seed") != task.seed
        or payload.get("n_records") != spec["n_records"]
        or payload.get("query_count") != spec["query_count"]
        or payload.get("query_identity_sha256")
        != spec["query_identity_sha256"]
        or payload.get("target_vector_sha256") != spec["target_vector_sha256"]
        or payload.get("trace_query_identity_sha256")
        != spec["trace_query_identity_sha256"]
        or payload.get("trace_target_vector_sha256")
        != spec["trace_target_vector_sha256"]
        or payload.get("fixed_checkpoint_rounds_requested")
        != list(source_protocol.CHECKPOINT_ROUNDS)
        or payload.get("historical_best_included") is not False
        or not isinstance(fixed, list)
        or [item.get("state_index") for item in fixed] != expected_states
        or not isinstance(terminal, dict)
    ):
        raise RuntimeError(f"R8 恢复检查点身份或位置漂移：{task.task_id}")
    expected_entry_keys = {
        "absolute_count_error_sum",
        "gap_e",
        "kind",
        "normalized_l1",
        "phase",
        "query_answers",
        "round",
        "squared_loss",
        "state_index",
    }
    for state, item in zip(expected_states, fixed, strict=True):
        if (
            not isinstance(item, dict)
            or set(item) != expected_entry_keys
            or item.get("kind") != "fixed_checkpoint"
            or item.get("phase") != ("initial" if state == 0 else "post_round")
            or item.get("round") != state
            or item.get("state_index") != state
            or not isinstance(item.get("query_answers"), list)
            or len(item["query_answers"]) != spec["query_count"]
        ):
            raise RuntimeError(f"R8 恢复固定检查点结构漂移：{task.task_id}/{state}")
    if (
        set(terminal) != expected_entry_keys
        or terminal.get("kind") != "terminal"
        or terminal.get("phase")
        != ("initial" if row["applied_rounds"] == 0 else "post_round")
        or terminal.get("round") != row["applied_rounds"]
        or terminal.get("state_index") != row["applied_rounds"]
        or not isinstance(terminal.get("query_answers"), list)
        or len(terminal["query_answers"]) != spec["query_count"]
    ):
        raise RuntimeError(f"R8 恢复终态检查点结构漂移：{task.task_id}")


def _validate_case(artifact_root: Path, task: Any) -> dict[str, Any]:
    case_relative = Path("cases") / task.task_id
    case_dir = artifact_root / case_relative
    expected_inventory = recovery_protocol.ARTIFACT_INVENTORY[task.task_id]
    expected_files = set(expected_inventory)
    if (
        not case_dir.is_dir()
        or {path.name for path in case_dir.iterdir()} != expected_files
    ):
        raise RuntimeError(f"R8 恢复 case 文件集合漂移：{task.task_id}")
    for name, identity in expected_inventory.items():
        path = case_dir / name
        if (
            path.stat().st_size != identity["size"]
            or recovery_protocol.file_sha256(path) != identity["sha256"]
        ):
            raise RuntimeError(f"R8 恢复冻结字节漂移：{task.task_id}/{name}")

    manifest = _load_json(case_dir / CASE_MANIFEST)
    if (
        manifest.get("contract_version") != source_protocol.PROTOCOL_VERSION
        or manifest.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or manifest.get("execution_commit")
        != recovery_protocol.SOURCE_GENERATION_COMMIT
        or manifest.get("execution_shard_id") != source_protocol.LOCAL_SHARD
        or manifest.get("generator_params")
        != source_protocol.generator_params_manifest(
            task.dataset, task.arm, task.seed
        )
        or manifest.get("raw_reference_data_accessed") is not False
        or manifest.get("method_comparison_emitted") is not False
    ):
        raise RuntimeError(f"R8 恢复 case 来源身份漂移：{task.task_id}")
    row = manifest.get("collection_row")
    if not isinstance(row, dict):
        raise TypeError(f"R8 恢复 case 缺少 collection 行：{task.task_id}")
    _validate_case_row(task, row)
    attachments = (
        ("terminal_table_path", "terminal_table_sha256", TERMINAL_TABLE),
        (
            "checkpoint_artifact_path",
            "checkpoint_artifact_sha256",
            CHECKPOINT_ARTIFACT,
        ),
        ("transition_audit_path", "transition_audit_sha256", TRANSITION_AUDIT),
    )
    resolved: dict[str, Path] = {}
    for path_key, sha_key, filename in attachments:
        expected_relative = str(case_relative / filename)
        path = _safe_artifact(artifact_root, row.get(path_key), expected_relative)
        if (
            not _is_sha256(row.get(sha_key))
            or recovery_protocol.file_sha256(path) != row[sha_key]
            or row[sha_key] != expected_inventory[filename]["sha256"]
        ):
            raise RuntimeError(
                f"R8 恢复 case 附件 SHA-256 漂移：{task.task_id}/{path_key}"
            )
        resolved[path_key] = path
    _validate_transition(resolved["transition_audit_path"], task, row)
    _validate_checkpoint(resolved["checkpoint_artifact_path"], task, row)
    return row


def _validate_pairing(rows: Sequence[dict[str, Any]], tasks: Sequence[Any]) -> None:
    if [row.get("task_id") for row in rows] != [task.task_id for task in tasks]:
        raise RuntimeError("R8 恢复四条行顺序漂移")
    proxies = [_legacy_validation_proxy(row) for row in rows]
    with source_collector._execution_runtime():
        source_collector._validate_pairing_for_tasks(proxies, tasks)


def _validate_case_matrix(
    artifact_root: Path,
    tasks: Sequence[Any] | None = None,
) -> list[dict[str, Any]]:
    frozen_tasks = list(tasks or source_protocol.task_plan().tasks)
    cases_root = artifact_root / "cases"
    expected_ids = {task.task_id for task in frozen_tasks}
    if (
        not cases_root.is_dir()
        or {path.name for path in cases_root.iterdir()} != expected_ids
    ):
        raise RuntimeError("R8 恢复 case 目录集合与冻结四条任务不一致")
    rows = [_validate_case(artifact_root, task) for task in frozen_tasks]
    _validate_pairing(rows, frozen_tasks)
    return rows


def _generation_input_sha256() -> dict[str, dict[str, str]]:
    return {
        dataset: {
            key: spec["input_sha256"][key]
            for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in source_protocol.DATASETS.items()
    }


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
        for reason in source_protocol.NORMAL_TERMINATION_REASONS
    }


def _source_environment() -> dict[str, Any]:
    return {
        "shard_id": source_protocol.LOCAL_SHARD,
        "hostname": source_protocol.EXECUTION_SHARDS[
            source_protocol.LOCAL_SHARD
        ]["hostname"],
        **source_protocol.EXPECTED_GPU,
        **source_protocol.EXPECTED_SOFTWARE,
        "cuda_available": True,
        "deterministic_algorithms": True,
        "evidence_origin": "successful_source_preflight_before_generation",
    }


def _elapsed_seconds(value: str) -> int:
    hours, minutes, seconds = (int(part) for part in value.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def _gpu_samples() -> list[dict[str, Any]]:
    return [
        {
            **sample,
            "elapsed_sec": float(_elapsed_seconds(sample["elapsed"])),
            "memory_total_mib": source_protocol.EXPECTED_GPU[
                "memory_total_mib"
            ],
            "evidence_source": "supervising_agent_nvidia_smi_observation",
            "source_runner_sample": False,
        }
        for sample in recovery_protocol.SUPERVISING_GPU_SAMPLES
    ]


def _shard_audit() -> dict[str, bool]:
    return {
        "all_assigned_cases_present": True,
        "all_assigned_dataset_seed_triplets_paired": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "all_gap_8k_identity": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_shard_physical_index": True,
    }


def _collection_audit() -> dict[str, bool]:
    return {
        "all_4_cases_present": True,
        "both_dataset_seed_pairs_complete": True,
        "single_rtx4090_shard_assignment": True,
        "both_pairs_single_shard": True,
        "complete_shard_verified_before_merge": True,
        "all_existing_p6_autostop_enabled": True,
        "all_normal_abc_termination": True,
        "all_terminal_current": True,
        "all_applied_unconditionally": True,
        "both_arms_gap_8k_identity": True,
        "all_weighting_diagnostics_valid": True,
        "all_zero_clip_and_finite": True,
        "all_artifact_sha256_verified": True,
        "all_generator_params_preflighted_before_gpu": True,
        "completed_cases_resumed_without_rerun": True,
        "all_gpu_samples_match_physical_index_1": True,
    }


def _recovery_evidence(recovery_commit: str) -> dict[str, Any]:
    return {
        "contract_version": recovery_protocol.RECOVERY_VERSION,
        "protocol_sha256": recovery_protocol.FROZEN_RECOVERY_SHA256,
        "protocol": recovery_protocol.frozen_recovery_manifest(),
        "recovery_commit": recovery_commit,
        "recovery_collector_sha256": recovery_protocol.file_sha256(Path(__file__)),
        "source_staging_manifest_sha256": (
            recovery_protocol.SOURCE_STAGING_MANIFEST_SHA256
        ),
        "corrected_state_evaluation_identity_verified": True,
        "source_case_files_rewritten": False,
        "new_generation_case_count": 0,
        "generator_invoked": False,
        "raw_reference_data_accessed": False,
        "quality_metrics_interpreted": False,
        "source_runner_gpu_samples_persisted": False,
        "supervising_gpu_samples_present": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
    }


def _build_shard_report(
    rows: Sequence[dict[str, Any]],
    recovery_commit: str,
    *,
    started_at: str,
    finished_at: str,
    elapsed_sec: float,
) -> dict[str, Any]:
    shard_id = source_protocol.LOCAL_SHARD
    samples = _gpu_samples()
    return {
        "contract_version": source_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "shard_id": shard_id,
        "shard_assignment_sha256": source_protocol.shard_assignment_sha256(),
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            source_protocol.generator_params_manifest_sha256()
        ),
        "environment": {
            "python": "source preflight bound by python_major_minor",
            "platform": "source host bound by hostname",
            "numpy": source_protocol.EXPECTED_SOFTWARE["numpy"],
            "gpu": _source_environment(),
        },
        "generation_input_sha256": _generation_input_sha256(),
        "execution": {
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed_sec_this_invocation": float(elapsed_sec),
            "resumed_case_count": len(rows),
            "new_case_count": 0,
            "max_workers": source_protocol.MAX_WORKERS,
            "multiprocessing_start_method": "none_recovery_only",
            "task_order": "seed_then_dataset_then_arm",
            "generator_invoked": False,
        },
        "gpu_samples": samples,
        "gpu_summary": {
            "sample_count": len(samples),
            "evidence_source": "supervising_agent_nvidia_smi_observation",
            "source_runner_sample_count": 0,
        },
        "case_count": len(rows),
        "paired_dataset_seed_count": 2,
        "task_ids": [row["task_id"] for row in rows],
        "case_manifest_sha256": {
            task_id: files[CASE_MANIFEST]["sha256"]
            for task_id, files in recovery_protocol.ARTIFACT_INVENTORY.items()
        },
        "raw_results": list(rows),
        "timing_summary": _timing_summary(rows),
        "termination_reason_counts": _termination_counts(rows),
        "shard_audit": _shard_audit(),
        "postcollection_recovery": _recovery_evidence(recovery_commit),
        "formal_shard_complete": True,
        "formal_shard_recovered_after_generation": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "source_runner_gpu_samples_persisted": False,
        "supervising_gpu_samples_present": True,
        "raw_reference_data_accessed": False,
        "partial_shard_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_shard": False,
        "checkpoint_vectors_persisted_for_later_evaluation": True,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def _validate_shard_report(
    shard_root: Path,
    recovery_commit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    _validate_staging_manifest(shard_root)
    tasks = source_protocol.tasks_for_shard(source_protocol.LOCAL_SHARD)
    rows = _validate_case_matrix(shard_root, tasks)
    path = shard_root / source_protocol.SHARD_REPORT
    report = _load_json(path)
    required = {
        "contract_version": source_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "shard_id": source_protocol.LOCAL_SHARD,
        "shard_assignment_sha256": source_protocol.shard_assignment_sha256(),
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            source_protocol.generator_params_manifest_sha256()
        ),
        "generation_input_sha256": _generation_input_sha256(),
        "case_count": 4,
        "paired_dataset_seed_count": 2,
        "shard_audit": _shard_audit(),
        "formal_shard_complete": True,
        "formal_shard_recovered_after_generation": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "source_runner_gpu_samples_persisted": False,
        "supervising_gpu_samples_present": True,
        "raw_reference_data_accessed": False,
        "partial_shard_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_shard": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("R8 恢复分片报告身份或边界漂移")
    if (
        report.get("environment", {}).get("gpu") != _source_environment()
        or report.get("gpu_samples") != _gpu_samples()
        or report.get("task_ids") != [task.task_id for task in tasks]
        or report.get("raw_results") != rows
        or report.get("timing_summary") != _timing_summary(rows)
        or report.get("termination_reason_counts") != _termination_counts(rows)
        or report.get("postcollection_recovery")
        != _recovery_evidence(recovery_commit)
    ):
        raise RuntimeError("R8 恢复分片报告内容漂移")
    execution = report.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("resumed_case_count") != 4
        or execution.get("new_case_count") != 0
        or execution.get("generator_invoked") is not False
    ):
        raise RuntimeError("R8 恢复分片执行边界漂移")
    return report, rows, recovery_protocol.file_sha256(path)


def _find_source_staging(root: Path) -> Path:
    expected = (
        root
        / source_protocol.SHARD_OUTPUT_ROOT
        / recovery_protocol.SOURCE_STAGING_BASENAME
    )
    if not expected.is_dir():
        raise FileNotFoundError(f"冻结 R8 暂存目录不存在：{expected}")
    _validate_staging_manifest(expected)
    return expected


def close_shard(confirmed_recovery_sha256: str) -> Path:
    recovery_protocol.require_confirmation(confirmed_recovery_sha256)
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    recovery_commit = _assert_clean_recovery_tree(root)
    destination = (
        root / source_protocol.SHARD_OUTPUT_ROOT / source_protocol.LOCAL_SHARD
    )
    if destination.exists():
        _validate_shard_report(destination, recovery_commit)
        return destination / source_protocol.SHARD_REPORT
    staging = _find_source_staging(root)
    started_at = datetime.now().astimezone().isoformat()
    started = time.perf_counter()
    rows = _validate_case_matrix(staging)
    finished_at = datetime.now().astimezone().isoformat()
    report_path = staging / source_protocol.SHARD_REPORT
    if report_path.exists():
        _validate_shard_report(staging, recovery_commit)
    else:
        _write_json_new(
            report_path,
            _build_shard_report(
                rows,
                recovery_commit,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_sec=time.perf_counter() - started,
            ),
        )
        _validate_shard_report(staging, recovery_commit)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    return destination / source_protocol.SHARD_REPORT


def _build_collection_report(
    rows: Sequence[dict[str, Any]],
    shard_report: dict[str, Any],
    shard_sha256: str,
    recovery_commit: str,
) -> dict[str, Any]:
    shard_id = source_protocol.LOCAL_SHARD
    report = {
        "contract_version": source_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "scientific_protocol_sha256": source_protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "protocol": source_protocol.frozen_protocol_manifest(),
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            source_protocol.generator_params_manifest_sha256()
        ),
        "shard_assignment_sha256": source_protocol.shard_assignment_sha256(),
        "shard_report_sha256": {shard_id: shard_sha256},
        "shard_report_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / source_protocol.SHARD_REPORT),
                "sha256": shard_sha256,
            }
        },
        "source_staging_manifest_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / STAGING_MANIFEST),
                "sha256": recovery_protocol.SOURCE_STAGING_MANIFEST_SHA256,
            }
        },
        "environment": {"shards": {shard_id: shard_report["environment"]}},
        "generation_input_sha256": _generation_input_sha256(),
        "execution": {
            "shard_order": [shard_id],
            "task_counts": {shard_id: 4},
            "paired_block_counts": {shard_id: 2},
            "max_workers_by_shard": {shard_id: source_protocol.MAX_WORKERS},
            "multiprocessing_start_method": "spawn_source_then_none_recovery",
            "task_order": "seed_then_dataset_then_arm",
            "shard_execution": {shard_id: shard_report["execution"]},
        },
        "case_count": 4,
        "paired_dataset_seed_count": 2,
        "raw_results": list(rows),
        "timing_summary": _timing_summary(rows),
        "termination_reason_counts": _termination_counts(rows),
        "collection_audit": _collection_audit(),
        "postcollection_recovery": _recovery_evidence(recovery_commit),
        "formal_result_valid": True,
        "screen_collection_valid": True,
        "formal_collection_recovered": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "source_runner_gpu_samples_persisted": False,
        "supervising_gpu_samples_present": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "quality_results_published_by_collection": False,
        "l1_results_published_by_collection": False,
        "checkpoint_vectors_persisted_for_later_evaluation": True,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    return report


def merge_shard(confirmed_recovery_sha256: str) -> Path:
    recovery_protocol.require_confirmation(confirmed_recovery_sha256)
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    recovery_commit = _assert_clean_recovery_tree(root)
    destination = root / source_protocol.OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"R8 恢复 collection 已存在，不覆盖：{destination}")
    source_root = (
        root / source_protocol.SHARD_OUTPUT_ROOT / source_protocol.LOCAL_SHARD
    )
    shard_report, rows, shard_sha = _validate_shard_report(
        source_root,
        recovery_commit,
    )
    report = _build_collection_report(
        rows,
        shard_report,
        shard_sha,
        recovery_commit,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.recovery-merge-",
            dir=destination.parent,
        )
    )
    try:
        for task in source_protocol.task_plan().tasks:
            source = source_root / "cases" / task.task_id
            target = staging / "cases" / task.task_id
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target)
        if _validate_case_matrix(staging) != rows:
            raise RuntimeError("R8 恢复合并后 case 行漂移")
        evidence = staging / "shards" / source_protocol.LOCAL_SHARD
        evidence.mkdir(parents=True, exist_ok=True)
        for name, expected_sha in (
            (source_protocol.SHARD_REPORT, shard_sha),
            (STAGING_MANIFEST, recovery_protocol.SOURCE_STAGING_MANIFEST_SHA256),
        ):
            shutil.copy2(source_root / name, evidence / name)
            if recovery_protocol.file_sha256(evidence / name) != expected_sha:
                raise RuntimeError(f"R8 恢复分片证据复制漂移：{name}")
        _write_json_new(staging / source_protocol.COLLECTION_REPORT, report)
        os.replace(staging, destination)
    except Exception as exc:
        raise RuntimeError(
            f"R8 恢复合并失败；源分片未改动，暂存已保留：{staging}"
        ) from exc
    return destination / source_protocol.COLLECTION_REPORT


def _validate_commit_ancestry(root: Path, recovery_commit: Any) -> None:
    if not isinstance(recovery_commit, str) or len(recovery_commit) != 40:
        raise RuntimeError("R8 恢复提交格式无效")
    current = _git_text(root, "rev-parse", "HEAD")
    if (
        _git_text(
            root,
            "merge-base",
            recovery_protocol.SOURCE_GENERATION_COMMIT,
            recovery_commit,
        )
        != recovery_protocol.SOURCE_GENERATION_COMMIT
        or _git_text(root, "merge-base", recovery_commit, current)
        != recovery_commit
    ):
        raise RuntimeError("R8 生成、恢复与当前提交祖先关系漂移")


def load_collection(
    root: Path,
    confirmed_collection_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    """结果盲复核完整恢复 collection，供后续另行授权的评价适配器使用。"""

    source_protocol.require_collection_confirmation(confirmed_collection_sha256)
    recovery_protocol.assert_frozen_recovery_identity(root)
    output = root / source_protocol.OUTPUT_DIR
    path = output / source_protocol.COLLECTION_REPORT
    if recovery_protocol.file_sha256(path) != confirmed_collection_sha256:
        raise ValueError("R8 恢复 collection SHA-256 与确认值不一致")
    report = _load_json(path)
    required = {
        "contract_version": source_protocol.PROTOCOL_VERSION,
        "protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "scientific_protocol_sha256": source_protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "protocol": source_protocol.frozen_protocol_manifest(),
        "execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            source_protocol.generator_params_manifest_sha256()
        ),
        "shard_assignment_sha256": source_protocol.shard_assignment_sha256(),
        "generation_input_sha256": _generation_input_sha256(),
        "case_count": 4,
        "paired_dataset_seed_count": 2,
        "collection_audit": _collection_audit(),
        "formal_result_valid": True,
        "screen_collection_valid": True,
        "formal_collection_recovered": True,
        "execution_monitoring_evidence_complete": False,
        "execution_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "source_runner_gpu_samples_persisted": False,
        "supervising_gpu_samples_present": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "quality_results_published_by_collection": False,
        "l1_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("R8 恢复 collection 顶层契约或边界漂移")
    recovery = report.get("postcollection_recovery")
    if not isinstance(recovery, dict):
        raise TypeError("R8 collection 缺少恢复证据")
    recovery_commit = recovery.get("recovery_commit")
    _validate_commit_ancestry(root, recovery_commit)
    if recovery != _recovery_evidence(recovery_commit):
        raise RuntimeError("R8 collection 恢复证据漂移")
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 4:
        raise RuntimeError("R8 恢复 collection 不是完整四条")
    loaded_rows = _validate_case_matrix(output)
    if loaded_rows != rows:
        raise RuntimeError("R8 恢复 collection case 与报告行不一致")

    shard_id = source_protocol.LOCAL_SHARD
    shard_relative = str(Path("shards") / shard_id / source_protocol.SHARD_REPORT)
    staging_relative = str(Path("shards") / shard_id / STAGING_MANIFEST)
    shard_meta = report.get("shard_report_artifacts", {}).get(shard_id)
    staging_meta = report.get("source_staging_manifest_artifacts", {}).get(shard_id)
    shard_sha = report.get("shard_report_sha256", {}).get(shard_id)
    if (
        shard_meta != {"path": shard_relative, "sha256": shard_sha}
        or staging_meta
        != {
            "path": staging_relative,
            "sha256": recovery_protocol.SOURCE_STAGING_MANIFEST_SHA256,
        }
        or not _is_sha256(shard_sha)
        or recovery_protocol.file_sha256(output / shard_relative) != shard_sha
        or recovery_protocol.file_sha256(output / staging_relative)
        != recovery_protocol.SOURCE_STAGING_MANIFEST_SHA256
    ):
        raise RuntimeError("R8 恢复 collection 分片证据绑定漂移")
    shard = _load_json(output / shard_relative)
    if (
        shard.get("raw_results") != rows
        or shard.get("postcollection_recovery") != recovery
        or shard.get("gpu_samples") != _gpu_samples()
    ):
        raise RuntimeError("R8 恢复 collection 与分片报告不一致")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    for task, row in zip(source_protocol.task_plan().tasks, rows, strict=True):
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError("R8 恢复 collection case 重复")
        indexed[key] = row
    return report, indexed


def build_plan() -> dict[str, Any]:
    root = _repo_root()
    protocol_sha = recovery_protocol.assert_frozen_recovery_identity(root)
    return {
        "mode": "result_blind_postcollection_recovery_only",
        "recovery_protocol_sha256": protocol_sha,
        "source_protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "source_staging_basename": recovery_protocol.SOURCE_STAGING_BASENAME,
        "task_ids": [task.task_id for task in source_protocol.task_plan().tasks],
        "case_count": 4,
        "source_case_rewrite_allowed": False,
        "case_rerun_allowed": False,
        "generator_invoked": False,
        "raw_reference_access_allowed": False,
        "quality_metric_interpretation_allowed": False,
        "evaluation_performed": False,
    }


def recover(confirmed_recovery_sha256: str) -> Path:
    close_shard(confirmed_recovery_sha256)
    return merge_shard(confirmed_recovery_sha256)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    close = subparsers.add_parser("close-shard")
    close.add_argument("--confirm-recovery-sha", required=True)
    merge = subparsers.add_parser("merge-shard")
    merge.add_argument("--confirm-recovery-sha", required=True)
    run = subparsers.add_parser("recover")
    run.add_argument("--confirm-recovery-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    if args.command == "close-shard":
        path = close_shard(args.confirm_recovery_sha)
        print(f"R8 恢复 shard 报告 -> {path}")
        print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")
        return
    if args.command == "merge-shard":
        path = merge_shard(args.confirm_recovery_sha)
    else:
        path = recover(args.confirm_recovery_sha)
    print(f"R8 恢复 collection 报告 -> {path}")
    print(f"报告 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
