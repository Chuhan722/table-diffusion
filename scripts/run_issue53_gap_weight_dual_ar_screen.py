#!/usr/bin/env python3
"""Collect the two Issue #53 A/R dual-channel development trajectories."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

from scripts import issue53_gap_weight_dual_ar_screen_execution_protocol as protocol
from scripts import run_issue53_gap_weight_sqrt_screen as sqrt_runner


stage6d_runner = sqrt_runner.stage6d_runner
stage6e_runner = sqrt_runner.stage6e_runner
CASE_MANIFEST = sqrt_runner.CASE_MANIFEST
TERMINAL_TABLE = sqrt_runner.TERMINAL_TABLE
CHECKPOINT_ARTIFACT = sqrt_runner.CHECKPOINT_ARTIFACT
TRANSITION_AUDIT = sqrt_runner.TRANSITION_AUDIT

_BASE_EXECUTE_ACTIVE = sqrt_runner._BASE_EXECUTE_ACTIVE
_BASE_VALIDATE_CASE_ROW = sqrt_runner._BASE_VALIDATE_CASE_ROW
_BASE_EXTRACT_TRANSITION_AUDIT = sqrt_runner._BASE_EXTRACT_TRANSITION_AUDIT


@contextlib.contextmanager
def _execution_runtime():
    """只在本入口调用域内绑定 A/R 差量，结束后恢复共享模块。"""

    stage6e_names = (
        "protocol",
        "__file__",
        "_execute_trajectory_task",
        "_validate_case_row",
        "_collection_report",
    )
    stage6d_names = (
        "_extract_transition_audit",
        "_validate_pairing_for_tasks",
    )
    original_stage6e = {
        name: getattr(stage6e_runner, name) for name in stage6e_names
    }
    original_stage6d = {
        name: getattr(stage6d_runner, name) for name in stage6d_names
    }
    stage6e_runner.protocol = protocol
    stage6e_runner.__file__ = __file__
    stage6e_runner._execute_trajectory_task = _execute_trajectory_task
    stage6e_runner._validate_case_row = _validate_case_row
    stage6e_runner._collection_report = _collection_report
    stage6d_runner._extract_transition_audit = _extract_transition_audit
    stage6d_runner._validate_pairing_for_tasks = _validate_pairing_for_tasks
    try:
        yield
    finally:
        for name, value in original_stage6d.items():
            setattr(stage6d_runner, name, value)
        for name, value in original_stage6e.items():
            setattr(stage6e_runner, name, value)


def _strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"


def _attempt_rows(
    task: Any, diagnostics: dict[str, Any], applied_rounds: int
) -> list[dict[str, Any]]:
    nested = diagnostics.get("gap_l1_attempt_diagnostics_history")
    if not isinstance(nested, list) or len(nested) != applied_rounds:
        raise RuntimeError(f"{task.task_id} 缺口核诊断轮数不完整")
    flattened = []
    for round_index, attempts in enumerate(nested, start=1):
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise RuntimeError(
                f"{task.task_id} 第{round_index}轮缺口核不是唯一尝试"
            )
        item = attempts[0]
        if not isinstance(item, dict):
            raise TypeError(f"{task.task_id} 第{round_index}轮诊断非法")
        flattened.append(item)
    return flattened


def _expected_weight_identity(dataset: str) -> dict[str, Any]:
    expected = protocol.EXPECTED_WEIGHT_AUDIT[dataset]
    return {
        "positive_target_query_count": expected[
            "positive_target_query_count"
        ],
        "relative_inverse_target_normalizer": expected[
            "relative_inverse_target_normalizer"
        ],
        "relative_positive_weight_ratio": expected[
            "relative_positive_weight_ratio"
        ],
    }


def _validate_weighting_diagnostics(
    task: Any,
    diagnostics: dict[str, Any],
    artifact: dict[str, Any],
    summary: dict[str, Any],
    *,
    applied_rounds: int,
) -> None:
    if task.arm != protocol.ARM_GAP:
        raise RuntimeError(f"未知 A/R 双通道筛查臂：{task.arm!r}")
    params = diagnostics.get("params")
    if (
        not isinstance(params, dict)
        or params.get("gap_l1_weighting") != protocol.DUAL_WEIGHTING
        or params.get("gap_l1_max_weight_ratio") is not None
    ):
        raise RuntimeError(f"{task.task_id} A/R 双通道生成参数诊断漂移")

    attempts = _attempt_rows(task, diagnostics, applied_rounds)
    compact = artifact.get("gap_rounds")
    if not isinstance(compact, list) or len(compact) != len(attempts):
        raise RuntimeError(f"{task.task_id} 缺口核紧凑审计轮数漂移")
    expected = _expected_weight_identity(task.dataset)
    observed_normalizers: list[float] = []
    observed_ratios: list[float] = []
    scan_rounds = 0
    for round_index, (item, row) in enumerate(
        zip(attempts, compact, strict=True), start=1
    ):
        row.update({
            "expected_weighting": protocol.DUAL_WEIGHTING,
            "expected_channel_aggregation": "max",
            "expected_zero_target_policy": "absolute_channel_only",
            "expected_floor_applied": False,
            "expected_max_weight_ratio": None,
            "expected_smoothing_count": None,
            **{f"expected_{key}": value for key, value in expected.items()},
        })
        if item.get("gap_l1_scan_applied") is not True:
            row.update({
                "observed_kernel": item.get("kernel"),
                "observed_weighting": None,
                "observed_channel_aggregation": None,
                "observed_zero_target_policy": None,
                "observed_floor_applied": None,
                **{f"observed_{key}": None for key in expected},
            })
            continue

        scan_rounds += 1
        normalizer = item.get("gap_l1_relative_inverse_target_normalizer")
        ratio = item.get("gap_l1_relative_positive_weight_ratio")
        if (
            item.get("kernel")
            != "gap_l1_global_random_scan_dual_abs_relative_max"
            or item.get("gap_l1_weighting") != protocol.DUAL_WEIGHTING
            or item.get("gap_l1_channel_aggregation") != "max"
            or item.get("gap_l1_absolute_channel_query_weighting")
            != "uniform"
            or item.get("gap_l1_relative_channel_query_weighting")
            != "normalized_inverse_positive_target"
            or item.get("gap_l1_zero_target_policy")
            != "absolute_channel_only"
            or item.get("gap_l1_floor_applied") is not False
            or item.get("gap_l1_max_weight_ratio") is not None
            or item.get("gap_l1_smoothing_count") is not None
            or item.get("gap_l1_actual_weight_ratio") != 1.0
            or item.get("gap_l1_positive_target_query_count")
            != expected["positive_target_query_count"]
            or normalizer != expected["relative_inverse_target_normalizer"]
            or ratio != expected["relative_positive_weight_ratio"]
        ):
            raise RuntimeError(
                f"{task.task_id} 第{round_index}轮 A/R 双通道权重诊断漂移"
            )
        observed_normalizers.append(float(normalizer))
        observed_ratios.append(float(ratio))
        row.update({
            "observed_kernel": item["kernel"],
            "observed_weighting": item["gap_l1_weighting"],
            "observed_channel_aggregation": item[
                "gap_l1_channel_aggregation"
            ],
            "observed_zero_target_policy": item[
                "gap_l1_zero_target_policy"
            ],
            "observed_floor_applied": item["gap_l1_floor_applied"],
            "observed_positive_target_query_count": item[
                "gap_l1_positive_target_query_count"
            ],
            "observed_relative_inverse_target_normalizer": float(normalizer),
            "observed_relative_positive_weight_ratio": float(ratio),
        })

    if applied_rounds > 0 and scan_rounds == 0:
        raise RuntimeError(f"{task.task_id} 从未实际执行 A/R 双通道缺口扫描")
    summary.update({
        "gap_weighting": protocol.DUAL_WEIGHTING,
        "gap_channel_aggregation": "max",
        "gap_zero_target_policy": "absolute_channel_only",
        "gap_floor_applied": False,
        "gap_max_weight_ratio": None,
        "gap_smoothing_count": None,
        "gap_expected_positive_target_query_count": expected[
            "positive_target_query_count"
        ],
        "gap_expected_relative_inverse_target_normalizer": expected[
            "relative_inverse_target_normalizer"
        ],
        "gap_expected_relative_positive_weight_ratio": expected[
            "relative_positive_weight_ratio"
        ],
        "gap_weighting_scan_round_count": scan_rounds,
        "gap_relative_inverse_target_normalizer_min": (
            min(observed_normalizers) if observed_normalizers else None
        ),
        "gap_relative_inverse_target_normalizer_max": (
            max(observed_normalizers) if observed_normalizers else None
        ),
        "gap_relative_positive_weight_ratio_min": (
            min(observed_ratios) if observed_ratios else None
        ),
        "gap_relative_positive_weight_ratio_max": (
            max(observed_ratios) if observed_ratios else None
        ),
        "gap_weighting_guard_passed": True,
    })
    artifact["weighting_audit"] = {
        "expected_weighting": protocol.DUAL_WEIGHTING,
        "expected_channel_aggregation": "max",
        "expected_zero_target_policy": "absolute_channel_only",
        "expected_floor_applied": False,
        "expected_max_weight_ratio": None,
        "expected_smoothing_count": None,
        **{f"expected_{key}": value for key, value in expected.items()},
        "scan_round_count": scan_rounds,
        "guard_passed": True,
    }


def _extract_transition_audit(
    task: Any,
    diagnostics: dict[str, Any],
    *,
    applied_rounds: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_protocol = stage6d_runner.protocol
    stage6d_runner.protocol = protocol
    try:
        artifact, summary = _BASE_EXTRACT_TRANSITION_AUDIT(
            task, diagnostics, applied_rounds=applied_rounds
        )
    finally:
        stage6d_runner.protocol = original_protocol
    _validate_weighting_diagnostics(
        task,
        diagnostics,
        artifact,
        summary,
        applied_rounds=applied_rounds,
    )
    return artifact, summary


def _validate_case_row(task: Any, row: dict[str, Any]) -> None:
    _BASE_VALIDATE_CASE_ROW(task, row)
    expected = _expected_weight_identity(task.dataset)
    if (
        row.get("gap_weighting") != protocol.DUAL_WEIGHTING
        or row.get("gap_channel_aggregation") != "max"
        or row.get("gap_zero_target_policy") != "absolute_channel_only"
        or row.get("gap_floor_applied") is not False
        or row.get("gap_max_weight_ratio") is not None
        or row.get("gap_smoothing_count") is not None
        or row.get("gap_expected_positive_target_query_count")
        != expected["positive_target_query_count"]
        or row.get("gap_expected_relative_inverse_target_normalizer")
        != expected["relative_inverse_target_normalizer"]
        or row.get("gap_expected_relative_positive_weight_ratio")
        != expected["relative_positive_weight_ratio"]
        or row.get("gap_weighting_guard_passed") is not True
        or not isinstance(row.get("gap_weighting_scan_round_count"), int)
        or row["gap_weighting_scan_round_count"] <= 0
        or row.get("gap_relative_inverse_target_normalizer_min")
        != expected["relative_inverse_target_normalizer"]
        or row.get("gap_relative_inverse_target_normalizer_max")
        != expected["relative_inverse_target_normalizer"]
        or row.get("gap_relative_positive_weight_ratio_min")
        != expected["relative_positive_weight_ratio"]
        or row.get("gap_relative_positive_weight_ratio_max")
        != expected["relative_positive_weight_ratio"]
    ):
        raise RuntimeError(
            f"A/R 双通道权重筛查 case 身份漂移：{task.task_id}"
        )


def _validate_pairing_for_tasks(
    rows: Sequence[dict[str, Any]], tasks: Sequence[Any]
) -> None:
    if len(rows) != len(tasks) or len(rows) != 2:
        raise RuntimeError("A/R 双通道权重筛查 case 数量不完整")
    if [row.get("task_id") for row in rows] != [
        task.task_id for task in tasks
    ]:
        raise RuntimeError("A/R 双通道权重筛查 case 顺序漂移")
    addresses = {
        (row.get("seed"), row.get("dataset"), row.get("arm"))
        for row in rows
    }
    expected = {
        (task.seed, task.dataset, task.arm) for task in tasks
    }
    if addresses != expected or len(addresses) != len(rows):
        raise RuntimeError("A/R 双通道权重筛查 case 地址缺失或重复")
    for task, row in zip(tasks, rows, strict=True):
        _validate_case_row(task, row)


def _execute_trajectory_task(*args: Any, **kwargs: Any) -> dict[str, Any]:
    # spawn 子进程会重新导入本模块，因此保留顶层入口。
    with _execution_runtime():
        return _BASE_EXECUTE_ACTIVE(*args, **kwargs)


def _collection_report(
    *,
    execution_commit: str,
    runner_sha256: str,
    generator_params_manifest_sha256: str,
    input_sha256: dict[str, dict[str, str]],
    shard_reports: dict[str, dict[str, Any]],
    shard_report_sha256: dict[str, str],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    tasks = protocol.task_plan().tasks
    _validate_pairing_for_tasks(rows, tasks)
    stage6d_runner._validate_numeric_rows(rows)
    shard_id = protocol.LOCAL_SHARD
    if set(shard_reports) != {shard_id} or set(shard_report_sha256) != {
        shard_id
    }:
        raise RuntimeError("A/R 双通道权重筛查单分片报告不完整")
    total_applied = sum(int(row["applied_rounds"]) for row in rows)
    total_case_time = math.fsum(float(row["elapsed_sec"]) for row in rows)
    termination_counts = {
        reason: sum(row["termination_reason"] == reason for row in rows)
        for reason in protocol.NORMAL_TERMINATION_REASONS
    }
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "scientific_protocol_sha256": protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "protocol": protocol.frozen_protocol_manifest(),
        "execution_commit": execution_commit,
        "runner_sha256": runner_sha256,
        "generator_params_manifest_sha256": generator_params_manifest_sha256,
        "shard_assignment_sha256": protocol.shard_assignment_sha256(),
        "shard_report_sha256": dict(shard_report_sha256),
        "shard_report_artifacts": {
            shard_id: {
                "path": str(Path("shards") / shard_id / protocol.SHARD_REPORT),
                "sha256": shard_report_sha256[shard_id],
            }
        },
        "environment": {
            "shards": {shard_id: shard_reports[shard_id]["environment"]}
        },
        "generation_input_sha256": input_sha256,
        "execution": {
            "shard_order": [shard_id],
            "task_counts": {shard_id: len(tasks)},
            "dataset_block_counts": {
                shard_id: len(protocol.SHARD_BLOCKS[shard_id])
            },
            "max_workers_by_shard": {shard_id: protocol.MAX_WORKERS},
            "multiprocessing_start_method": "spawn",
            "task_order": "seed_then_dataset",
            "shard_execution": {
                shard_id: shard_reports[shard_id]["execution"]
            },
        },
        "case_count": len(rows),
        "dataset_seed_count": len(protocol.SHARD_BLOCKS[shard_id]),
        "raw_results": list(rows),
        "timing_summary": {
            "sum_case_elapsed_sec": float(total_case_time),
            "total_applied_rounds": int(total_applied),
            "average_sec_per_applied_round": (
                float(total_case_time / total_applied)
                if total_applied else 0.0
            ),
        },
        "termination_reason_counts": termination_counts,
        "collection_audit": {
            "all_2_candidate_cases_present": len(rows) == 2,
            "both_dataset_seed_tasks_complete": True,
            "single_rtx4090_shard_assignment": True,
            "complete_shard_verified_before_merge": True,
            "all_existing_p6_autostop_enabled": True,
            "all_normal_abc_termination": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_dual_ar_gap_8k_identity": True,
            "all_weighting_diagnostics_valid": True,
            "all_zero_clip_and_finite": True,
            "all_artifact_sha256_verified": True,
            "all_generator_params_preflighted_before_gpu": True,
            "completed_cases_resumed_without_rerun": True,
            "all_gpu_samples_match_physical_index_1": True,
        },
        "screen_collection_valid": True,
        "raw_reference_data_accessed": False,
        "reused_baseline_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "quality_results_published_by_collection": False,
        "l1_results_published_by_collection": False,
        "checkpoint_vectors_persisted_for_later_evaluation": True,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def build_plan() -> dict[str, Any]:
    with _execution_runtime():
        plan = stage6e_runner.build_plan()
    plan.update({
        "scientific_protocol_sha256": protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "dataset_block_count": 2,
        "run_mode": "single_explicit_shard_then_verified_merge",
        "generation_started": False,
        "screen_generation_authorized": False,
        "next_collect_requires_later_user_confirmation": True,
    })
    return plan


def preflight() -> dict[str, Any]:
    """只读核对启动条件；不调用生成器、不创建正式输出。"""

    with _execution_runtime():
        result = stage6e_runner.preflight()
    result.update({
        "scientific_protocol_sha256": protocol.SCIENTIFIC_PROTOCOL_SHA256,
        "screen_generation_authorized": False,
        "requires_explicit_later_user_confirmation": True,
    })
    return result


def collect(confirmed_protocol_sha256: str) -> Path:
    """真实运行两条轨迹；只有用户后续明确授权才可调用。"""

    protocol.require_run_confirmation(confirmed_protocol_sha256)
    with _execution_runtime():
        return stage6e_runner.collect(confirmed_protocol_sha256)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("preflight")
    collect_parser = subparsers.add_parser("collect")
    collect_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(_strict_json_text(build_plan()), end="")
        return
    if args.command == "preflight":
        print(_strict_json_text(preflight()), end="")
        return
    report = collect(args.confirm_protocol_sha)
    print(f"dual A/R max screen collection -> {report}")
    print(f"collection SHA-256 -> {protocol.file_sha256(report)}")


if __name__ == "__main__":
    main()
