#!/usr/bin/env python3
"""采集 Issue #53 Stage 6E 自动停止三核正式比较。"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as protocol
from scripts import run_issue53_stage6d_formal as stage6d_runner
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


CASE_MANIFEST = stage6d_runner.CASE_MANIFEST
TERMINAL_TABLE = stage6d_runner.TERMINAL_TABLE
CHECKPOINT_ARTIFACT = stage6d_runner.CHECKPOINT_ARTIFACT
TRANSITION_AUDIT = stage6d_runner.TRANSITION_AUDIT

# 复用已经审计过的数据身份、GPU监控、原子暂存和转移证据基础设施；只在
# Stage 6E 调用的动态范围内临时绑定，避免导入本模块污染历史 Stage 6D。


@contextlib.contextmanager
def _stage6d_runtime():
    names = (
        "protocol",
        "__file__",
        "_write_case_artifacts",
        "_execute_trajectory_task",
        "_validate_case_row",
        "_collection_report",
    )
    original = {name: getattr(stage6d_runner, name) for name in names}
    stage6d_runner.protocol = protocol
    # 复用函数内部以模块 ``__file__`` 计算 runner SHA。
    stage6d_runner.__file__ = __file__
    stage6d_runner._write_case_artifacts = _write_case_artifacts
    stage6d_runner._execute_trajectory_task = _execute_trajectory_task
    stage6d_runner._validate_case_row = _validate_case_row
    stage6d_runner._collection_report = _collection_report
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(stage6d_runner, name, value)


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
    return stage6d_runner._load_json(path)


def _git_text(root: Path, *arguments: str) -> str:
    return stage6d_runner._git_text(root, *arguments)


def _metrics_from_answers(*args, **kwargs):
    return stage6d_runner._metrics_from_answers(*args, **kwargs)


def _runtime_executable_audit() -> dict[str, Any]:
    """确认 spawn 解释器和 Triton CUDA 工具具有执行权限。"""

    import triton

    python_path = Path(sys.executable).resolve()
    triton_bin = (
        Path(triton.__file__).resolve().parent / "backends" / "nvidia" / "bin"
    )
    tool_paths = {
        name: triton_bin / name
        for name in ("ptxas", "ptxas-blackwell", "cuobjdump", "nvdisasm")
    }
    paths = {"python": python_path, **tool_paths}
    for name, path in paths.items():
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(f"Stage 6E 运行时不可执行：{name}={path}")
    return {
        "python": str(python_path),
        "triton_cuda_tools": {
            name: str(path) for name, path in tool_paths.items()
        },
        "all_executable": True,
    }


def _stop_summary(
    task: joint.JointTrajectoryTask,
    diagnostics: dict[str, Any],
    *,
    applied_rounds: int,
) -> dict[str, Any]:
    stopping = diagnostics.get("inner_early_stopping")
    if not isinstance(stopping, dict) or stopping.get("enabled") is not True:
        raise RuntimeError(f"{task.task_id} 没有启用既有 inner early stopping")
    if stopping.get("patience_ticks") != protocol.PATIENCE_TICKS:
        raise RuntimeError(f"{task.task_id} P=6 自动停止配置漂移")
    decision = stopping.get("last_decision")
    if not isinstance(decision, dict):
        raise RuntimeError(f"{task.task_id} 缺少自动停止最终决定")
    reason = diagnostics.get("termination_reason")
    if (
        reason not in protocol.NORMAL_TERMINATION_REASONS
        or decision.get("termination_reason") != reason
        or decision.get("state_index") != applied_rounds
        or decision.get("terminal_output_state_index") != applied_rounds
    ):
        raise RuntimeError(f"{task.task_id} 自动停止终点身份漂移")

    clocks = diagnostics.get("transition_clock_history")
    if not isinstance(clocks, list) or len(clocks) != applied_rounds:
        raise RuntimeError(f"{task.task_id} 自动停止转移时钟不完整")
    participating_rows = 0
    for round_index, clock in enumerate(clocks, start=1):
        attempts = clock.get("attempts")
        if not isinstance(attempts, list) or len(attempts) != 1:
            raise RuntimeError(f"{task.task_id} 第{round_index}轮尝试数漂移")
        participating_rows += int(attempts[0]["participating_rows"])
    n_records = int(protocol.DATASETS[task.dataset]["n_records"])
    normalized_work = participating_rows / n_records
    if (
        decision.get("cumulative_participating_rows") != participating_rows
        or not math.isclose(
            float(decision.get("normalized_work")),
            normalized_work,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or decision.get("completed_work_ticks") != participating_rows // n_records
    ):
        raise RuntimeError(f"{task.task_id} 自然工作量复算漂移")

    current_loss = float(decision["current_loss"])
    if not math.isfinite(current_loss) or current_loss < 0.0:
        raise RuntimeError(f"{task.task_id} 自动停止当前损失非法")
    if reason == "fit_target_reached":
        if current_loss != 0.0 or decision.get("fit_target_reached") is not True:
            raise RuntimeError(f"{task.task_id} A 停止条件漂移")
    elif reason == "early_stopped":
        if (
            decision.get("work_tick_completed") is not True
            or int(decision.get("consecutive_no_progress_ticks", -1))
            < protocol.PATIENCE_TICKS
            or decision.get("inner_complete") is not True
        ):
            raise RuntimeError(f"{task.task_id} B 停止条件漂移")
    else:
        if (
            decision.get("external_resource_cap_reached") is not True
            or applied_rounds != protocol.ROUND_CAP
            or stopping.get("resource_cap_source_diagnostic_only")
            != "candidate_budget"
        ):
            raise RuntimeError(f"{task.task_id} C 资源上限条件漂移")

    return {
        "enabled": True,
        "patience_ticks": protocol.PATIENCE_TICKS,
        "termination_reason": reason,
        "terminal_state_index": applied_rounds,
        "completed_work_ticks": int(decision["completed_work_ticks"]),
        "cumulative_participating_rows": participating_rows,
        "normalized_work_at_stop": float(normalized_work),
        "consecutive_no_progress_ticks": int(
            decision["consecutive_no_progress_ticks"]
        ),
        "best_state_index_diagnostic_only": int(
            decision["best_state_index_diagnostic_only"]
        ),
        "best_loss_diagnostic_only": float(
            decision["best_loss_diagnostic_only"]
        ),
        "terminal_loss": current_loss,
        "resource_cap_source_diagnostic_only": stopping.get(
            "resource_cap_source_diagnostic_only"
        ),
    }


def _write_case_artifacts_active(
    task: joint.JointTrajectoryTask,
    staging_root: Path,
    final_table: Any,
    diagnostics: dict[str, Any],
    target: np.ndarray,
    *,
    elapsed_sec: float,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
    execution_commit: str,
    execution_shard_id: str,
    generator_params_manifest: dict[str, Any],
) -> dict[str, Any]:
    if protocol.task_shard_id(task) != execution_shard_id:
        raise RuntimeError(f"{task.task_id} 被提交到错误执行分片")
    final_dir = staging_root / "cases" / task.task_id
    if final_dir.exists():
        raise FileExistsError(f"Stage 6E 正式 case 已存在，不覆盖：{final_dir}")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        stage6d_runner.tempfile.mkdtemp(
            prefix=f".{task.task_id}.tmp-", dir=final_dir.parent
        )
    )
    try:
        accept_history = list(diagnostics["accept_history"])
        applied_rounds = len(accept_history)
        if accept_history != [True] * applied_rounds:
            raise RuntimeError(f"{task.task_id} 出现拒绝或未应用候选")
        if diagnostics["proposal_attempts_history"] != [1] * applied_rounds:
            raise RuntimeError(f"{task.task_id} 出现多次尝试")
        if diagnostics["accepted_attempt_history"] != [1] * applied_rounds:
            raise RuntimeError(f"{task.task_id} 应用候选不是唯一尝试")
        if int(diagnostics["rounds_run"]) != applied_rounds:
            raise RuntimeError(f"{task.task_id} 生成器轮次与状态转移数不一致")
        stop_summary = _stop_summary(
            task, diagnostics, applied_rounds=applied_rounds
        )

        terminal_sha = stage6d_runner._frame_sha256(final_table)
        expected_terminal_sha = (
            diagnostics["transition_clock_history"][-1][
                "post_current_table_sha256"
            ]
            if applied_rounds
            else diagnostics["initial_table_sha256"]
        )
        if terminal_sha != expected_terminal_sha:
            raise RuntimeError(f"{task.task_id} 最后当前表与状态时钟不一致")

        checkpoint_artifact = stage6d_runner._extract_checkpoint_artifact(
            task, diagnostics, target, applied_rounds=applied_rounds
        )
        transition_artifact, kernel_summary = (
            stage6d_runner._extract_transition_audit(
                task, diagnostics, applied_rounds=applied_rounds
            )
        )

        table_path = temporary / TERMINAL_TABLE
        final_table.reset_index(drop=True).to_csv(table_path, index=False)
        if protocol.file_sha256(table_path) != terminal_sha:
            raise RuntimeError(f"{task.task_id} 终表落盘哈希不一致")
        checkpoint_path = temporary / CHECKPOINT_ARTIFACT
        stage6d_runner._write_json(
            checkpoint_path, stage6d_runner._json_safe(checkpoint_artifact)
        )
        transition_path = temporary / TRANSITION_AUDIT
        stage6d_runner._write_json(
            transition_path, stage6d_runner._json_safe(transition_artifact)
        )

        relative_base = Path("cases") / task.task_id
        collection_row = {
            "task_id": task.task_id,
            "execution_shard_id": execution_shard_id,
            "dataset": task.dataset,
            "arm": task.arm,
            "seed": int(task.seed),
            "requested_rounds": int(task.rounds),
            "applied_rounds": applied_rounds,
            "termination_reason": stop_summary["termination_reason"],
            "inner_early_stopping": stop_summary,
            "device": "cuda:0",
            "output_table_identity": "terminal_current",
            "all_applied_unconditionally": True,
            "proposal_attempt_count": int(
                diagnostics["candidate_evaluation_count"]
            ),
            "query_identity_sha256": protocol.DATASETS[task.dataset][
                "query_identity_sha256"
            ],
            "target_vector_sha256": protocol.DATASETS[task.dataset][
                "target_vector_sha256"
            ],
            "trace_query_identity_sha256": diagnostics[
                "stationarity_trace"
            ].query_identity_sha256,
            "trace_target_vector_sha256": diagnostics[
                "stationarity_trace"
            ].target_identity_sha256,
            "initial_table_sha256": diagnostics["initial_table_sha256"],
            "primary_rng_post_initialization_sha256": diagnostics[
                "primary_rng_post_initialization_state_sha256"
            ],
            "primary_rng_endpoint_sha256": diagnostics[
                "primary_rng_state_sha256"
            ],
            "direction_reference_scale": (
                float(diagnostics["direction_reference_scale"])
                if diagnostics["direction_reference_scale"] is not None
                else None
            ),
            "terminal_table_path": str(relative_base / TERMINAL_TABLE),
            "terminal_table_sha256": terminal_sha,
            "checkpoint_artifact_path": str(
                relative_base / CHECKPOINT_ARTIFACT
            ),
            "checkpoint_artifact_sha256": protocol.file_sha256(
                checkpoint_path
            ),
            "transition_audit_path": str(relative_base / TRANSITION_AUDIT),
            "transition_audit_sha256": protocol.file_sha256(transition_path),
            "elapsed_sec": float(elapsed_sec),
            "average_sec_per_applied_round": (
                float(elapsed_sec / applied_rounds) if applied_rounds else 0.0
            ),
            "peak_allocated_bytes": int(peak_allocated_bytes),
            "peak_reserved_bytes": int(peak_reserved_bytes),
            "state_evaluation_count": int(diagnostics["state_evaluation_count"]),
            "candidate_evaluation_count": int(
                diagnostics["candidate_evaluation_count"]
            ),
            "distance_evaluation_count": int(
                diagnostics["distance_evaluation_count"]
            ),
            "direction_evaluation_count": int(
                diagnostics["direction_evaluation_count"]
            ),
            **kernel_summary,
        }
        manifest = {
            "contract_version": protocol.PROTOCOL_VERSION,
            "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
            "execution_commit": execution_commit,
            "execution_shard_id": execution_shard_id,
            "generator_params": generator_params_manifest,
            "collection_row": collection_row,
            "raw_reference_data_accessed": False,
            "method_comparison_emitted": False,
        }
        stage6d_runner._write_json(temporary / CASE_MANIFEST, manifest)
        os.replace(temporary, final_dir)
        return collection_row
    except Exception as exc:
        raise RuntimeError(
            f"{task.task_id} 生成后产物收口失败；临时目录已保留且不得自动重跑："
            f"{temporary}"
        ) from exc


def _write_case_artifacts(*args, **kwargs):
    with _stage6d_runtime():
        return _write_case_artifacts_active(*args, **kwargs)


def _execute_trajectory_task_active(
    task: joint.JointTrajectoryTask,
    *,
    repository_root: str,
    staging_root: str,
    execution_commit: str,
    execution_shard_id: str,
    generator_params_by_task: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    root = Path(repository_root)
    spec = protocol.DATASETS[task.dataset]
    manifest = generator_params_by_task.get(task.task_id)
    if manifest != protocol.generator_params_manifest(
        task.dataset, task.arm, task.seed
    ):
        raise RuntimeError(f"正式任务预检参数清单漂移：{task.task_id}")
    schema = load_schema(str(root / spec["schema"]))
    queries, target, _identity = stage6d_runner._query_target_identity_audit(
        root, task.dataset
    )
    marginals = load_marginals(str(root / spec["marginals"]))

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("工作进程没有恰好一张可用显卡")
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize(0)
    started = time.perf_counter()
    returned_table, diagnostics = stage6d_runner._call_generator_silently(
        task.task_id,
        target=target,
        queries=queries,
        schema=schema,
        marginals=marginals,
        **protocol.task_generator_params(task.dataset, task.arm, task.seed),
    )
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    if "final_table" not in diagnostics:
        raise RuntimeError(f"{task.task_id} 缺少最后当前表")
    final_table = diagnostics.pop("final_table")
    del returned_table
    return _write_case_artifacts(
        task,
        Path(staging_root),
        final_table,
        diagnostics,
        target,
        elapsed_sec=elapsed,
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)),
        execution_commit=execution_commit,
        execution_shard_id=execution_shard_id,
        generator_params_manifest=manifest,
    )


def _execute_trajectory_task(*args, **kwargs):
    # spawn 子进程通过本包装器进入临时 Stage 6E 绑定。
    with _stage6d_runtime():
        return _execute_trajectory_task_active(*args, **kwargs)


def _validate_case_row(
    task: joint.JointTrajectoryTask, row: dict[str, Any]
) -> None:
    applied = row.get("applied_rounds")
    reason = row.get("termination_reason")
    stopping = row.get("inner_early_stopping")
    if (
        row.get("task_id") != task.task_id
        or row.get("execution_shard_id") != protocol.LOCAL_SHARD
        or row.get("dataset") != task.dataset
        or row.get("arm") != task.arm
        or row.get("seed") != task.seed
        or row.get("requested_rounds") != protocol.ROUND_CAP
        or row.get("device") != "cuda:0"
        or row.get("output_table_identity") != "terminal_current"
        or row.get("all_applied_unconditionally") is not True
        or not isinstance(applied, int)
        or isinstance(applied, bool)
        or not 0 <= applied <= protocol.ROUND_CAP
        or reason not in protocol.NORMAL_TERMINATION_REASONS
        or row.get("proposal_attempt_count") != applied
        or row.get("candidate_evaluation_count") != applied
        or row.get("state_evaluation_count") != applied + 1
        or not isinstance(stopping, dict)
        or stopping.get("enabled") is not True
        or stopping.get("patience_ticks") != protocol.PATIENCE_TICKS
        or stopping.get("termination_reason") != reason
        or stopping.get("terminal_state_index") != applied
        or not math.isclose(
            float(row.get("average_sec_per_applied_round", -1.0)),
            float(row.get("elapsed_sec", -1.0)) / applied if applied else 0.0,
            rel_tol=1e-15,
            abs_tol=0.0,
        )
    ):
        raise RuntimeError(f"Stage 6E case 行身份漂移：{task.task_id}")
    if reason == "fit_target_reached" and stopping.get("terminal_loss") != 0.0:
        raise RuntimeError(f"Stage 6E A 停止身份漂移：{task.task_id}")
    if reason == "early_stopped" and (
        stopping.get("consecutive_no_progress_ticks", -1)
        < protocol.PATIENCE_TICKS
    ):
        raise RuntimeError(f"Stage 6E B 停止身份漂移：{task.task_id}")
    if reason == "resource_cap_reached" and (
        applied != protocol.ROUND_CAP
        or stopping.get("resource_cap_source_diagnostic_only")
        != "candidate_budget"
    ):
        raise RuntimeError(f"Stage 6E C 停止身份漂移：{task.task_id}")


def _collection_report_active(
    *,
    execution_commit: str,
    runner_sha256: str,
    generator_params_manifest_sha256: str,
    input_sha256: dict[str, dict[str, str]],
    shard_reports: dict[str, dict[str, Any]],
    shard_report_sha256: dict[str, str],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    tasks = protocol.task_plan().tasks
    if [row["task_id"] for row in rows] != [task.task_id for task in tasks]:
        raise RuntimeError("Stage 6E collection 行顺序漂移")
    stage6d_runner._validate_pairing(rows)
    stage6d_runner._validate_numeric_rows(rows)
    shard_id = protocol.LOCAL_SHARD
    total_applied = sum(int(row["applied_rounds"]) for row in rows)
    total_case_time = math.fsum(float(row["elapsed_sec"]) for row in rows)
    termination_counts = {
        reason: sum(row["termination_reason"] == reason for row in rows)
        for reason in protocol.NORMAL_TERMINATION_REASONS
    }
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
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
        "environment": {"shards": {shard_id: shard_reports[shard_id]["environment"]}},
        "generation_input_sha256": input_sha256,
        "execution": {
            "shard_order": [shard_id],
            "task_counts": {shard_id: len(tasks)},
            "paired_block_counts": {shard_id: 10},
            "max_workers_by_shard": {shard_id: protocol.MAX_WORKERS},
            "multiprocessing_start_method": "spawn",
            "task_order": "seed_then_arm_then_dataset",
            "shard_execution": {
                shard_id: shard_reports[shard_id]["execution"]
            },
        },
        "case_count": len(rows),
        "paired_dataset_seed_count": 10,
        "raw_results": list(rows),
        "timing_summary": {
            "sum_case_elapsed_sec": float(total_case_time),
            "total_applied_rounds": int(total_applied),
            "average_sec_per_applied_round": (
                float(total_case_time / total_applied) if total_applied else 0.0
            ),
        },
        "termination_reason_counts": termination_counts,
        "collection_audit": {
            "all_30_cases_present": len(rows) == 30,
            "all_10_dataset_seed_triplets_paired": True,
            "single_rtx4090_shard_assignment": True,
            "all_triplets_single_shard": True,
            "complete_shard_verified_before_merge": True,
            "all_existing_p6_autostop_enabled": True,
            "all_normal_abc_termination": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_gap_8k_identity": True,
            "all_zero_clip_and_finite": True,
            "all_artifact_sha256_verified": True,
            "all_generator_params_preflighted_before_gpu": True,
            "completed_cases_resumed_without_rerun": True,
            "all_gpu_samples_match_physical_index_1": True,
        },
        "formal_result_valid": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "checkpoint_vectors_persisted_for_later_evaluation": True,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }


def _collection_report(*args, **kwargs):
    with _stage6d_runtime():
        return _collection_report_active(*args, **kwargs)


def build_plan() -> dict[str, Any]:
    plan = protocol.build_plan(_repo_root())
    with _stage6d_runtime():
        manifests, manifest_sha = (
            stage6d_runner._preflight_generator_param_manifests(
                protocol.task_plan().tasks
            )
        )
    if manifests != protocol.generator_params_manifest_matrix():
        raise RuntimeError("Stage 6E 30条生成参数清单漂移")
    plan.update(
        {
            "collector_wired": True,
            "collector_sha256": protocol.file_sha256(Path(__file__)),
            "generator_params_manifest_sha256": manifest_sha,
            "all_generator_params_preflighted_without_gpu": True,
            "resume_only_from_same_protocol_commit_and_artifacts": True,
            "run_mode": "single_explicit_shard_then_verified_merge",
            "next_command_would_start_generation": True,
        }
    )
    return plan


def preflight() -> dict[str, Any]:
    """只读核对启动条件；不调用生成器、不创建正式输出。"""

    root = _repo_root()
    protocol_sha = protocol.assert_frozen_protocol_identity(root)
    execution_commit = stage6d_runner._assert_clean_worktree(root)
    with _stage6d_runtime():
        input_sha = stage6d_runner._generation_input_audit(root)
        manifests, manifest_sha = (
            stage6d_runner._preflight_generator_param_manifests(
                protocol.task_plan().tasks
            )
        )
    if manifests != protocol.generator_params_manifest_matrix():
        raise RuntimeError("Stage 6E 预检生成参数矩阵漂移")
    if (root / protocol.OUTPUT_DIR).exists():
        raise FileExistsError("Stage 6E 正式输出目录已经存在")
    with _stage6d_runtime():
        gpu = stage6d_runner._gpu_idle_audit(protocol.LOCAL_SHARD)
    return {
        "mode": "read_only_preflight_no_generator_call_no_output_created",
        "protocol_sha256": protocol_sha,
        "execution_commit": execution_commit,
        "worktree_clean_including_untracked": True,
        "generation_input_sha256": input_sha,
        "generator_params_manifest_sha256": manifest_sha,
        "trajectory_count": len(protocol.task_plan().tasks),
        "runtime_executables": _runtime_executable_audit(),
        "gpu": gpu,
        "generation_started": False,
        "ready_for_final_user_confirmation": True,
    }


def collect(confirmed_protocol_sha256: str) -> Path:
    """运行单分片并完成严格合并；这是会真正启动生成器的唯一入口。"""

    protocol.require_run_confirmation(confirmed_protocol_sha256)
    _runtime_executable_audit()
    with _stage6d_runtime():
        stage6d_runner.run_shard(
            confirmed_protocol_sha256, protocol.LOCAL_SHARD
        )
        return stage6d_runner.merge_shards(confirmed_protocol_sha256)


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
    print(f"Stage 6E collection -> {report}")
    print(f"collection SHA-256 -> {protocol.file_sha256(report)}")


if __name__ == "__main__":
    main()
