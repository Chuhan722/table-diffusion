#!/usr/bin/env python3
"""独立复核 Issue #53 Stage 6E 自动停止三核正式评价。"""

from __future__ import annotations

import argparse
import contextlib
import math
from pathlib import Path
from typing import Any

from scripts import audit_issue53_stage6d_formal as stage6d_auditor
from scripts import evaluate_issue53_stage6e_autostop as evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as protocol
from scripts import run_issue53_stage6e_autostop as collection


AUDIT_VERSION = "issue53-stage6e-autostop-formal-independent-audit-v1"

_legacy_recompute_cases = stage6d_auditor._recompute_cases
_legacy_summary_independent = stage6d_auditor._summary_independent


@contextlib.contextmanager
def _stage6d_audit_runtime():
    names = (
        "protocol",
        "collection",
        "AUDIT_VERSION",
        "build_plan",
        "_audit_collection_independently",
        "_recompute_cases",
        "_summary_independent",
        "_audit_evaluation",
    )
    original = {name: getattr(stage6d_auditor, name) for name in names}
    stage6d_auditor.protocol = protocol
    stage6d_auditor.collection = collection
    stage6d_auditor.AUDIT_VERSION = AUDIT_VERSION
    stage6d_auditor.build_plan = build_plan
    stage6d_auditor._audit_collection_independently = (
        _audit_collection_independently
    )
    stage6d_auditor._recompute_cases = _recompute_cases
    stage6d_auditor._summary_independent = _summary_independent
    stage6d_auditor._audit_evaluation = _audit_evaluation
    try:
        yield
    finally:
        for name, value in original.items():
            setattr(stage6d_auditor, name, value)


def _repo_root() -> Path:
    return collection._repo_root()


def build_plan() -> dict[str, Any]:
    protocol_sha = protocol.assert_frozen_protocol_identity(_repo_root())
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generation_access",
        "protocol_sha256": protocol_sha,
        "collection_report": str(protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT),
        "evaluation_report": str(protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT),
        "l1_results_csv": str(protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV),
        "audit_report": str(protocol.OUTPUT_DIR / protocol.AUDIT_REPORT),
        "recompute_terminal_tables": True,
        "recompute_checkpoint_vectors": True,
        "recompute_variable_round_timing": True,
        "recompute_l1_csv": True,
        "recompute_frozen_classification": True,
        "rerun_generation": False,
    }


def _safe_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("Stage 6E artifact 路径必须是字符串")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Stage 6E artifact 路径越界")
    base = (root / protocol.OUTPUT_DIR).resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("Stage 6E artifact 路径逃逸")
    return resolved


def _audit_collection_independently(
    root: Path,
    confirmed_sha: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    stage6d_auditor._require_sha(confirmed_sha, "Stage 6E collection SHA-256")
    path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("Stage 6E collection SHA-256 不一致")
    report = stage6d_auditor._load_json(path)
    required_top = {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "case_count": 30,
        "paired_dataset_seed_count": 10,
        "formal_result_valid": True,
        "raw_reference_data_accessed": False,
        "partial_matrix_comparison_emitted": False,
        "method_ranking_emitted": False,
        "l1_results_published_by_collection": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required_top.items()):
        raise RuntimeError("Stage 6E collection 顶层契约漂移")
    if report.get("protocol") != protocol.frozen_protocol_manifest():
        raise RuntimeError("Stage 6E collection 内嵌协议漂移")
    if (
        report.get("runner_sha256")
        != protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"]
        or report.get("generator_params_manifest_sha256")
        != protocol.generator_params_manifest_sha256()
        or report.get("shard_assignment_sha256")
        != protocol.shard_assignment_sha256()
    ):
        raise RuntimeError("Stage 6E collection 执行器或参数清单身份漂移")
    expected_inputs = {
        dataset: {
            key: spec["input_sha256"][key]
            for key in ("schema", "queries", "marginals")
        }
        for dataset, spec in protocol.DATASETS.items()
    }
    if report.get("generation_input_sha256") != expected_inputs:
        raise RuntimeError("Stage 6E collection 输入身份漂移")
    expected_audit = {
        "all_30_cases_present": True,
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
    }
    if report.get("collection_audit") != expected_audit:
        raise RuntimeError("Stage 6E collection 结构审计漂移")

    execution_commit = report.get("execution_commit")
    if (
        not isinstance(execution_commit, str)
        or len(execution_commit) != 40
        or any(character not in "0123456789abcdef" for character in execution_commit)
    ):
        raise RuntimeError("Stage 6E collection 执行提交无效")
    current_commit = collection.stage6d_runner._git_text(root, "rev-parse", "HEAD")
    if (
        collection.stage6d_runner._git_text(
            root, "merge-base", execution_commit, current_commit
        )
        != execution_commit
    ):
        raise RuntimeError("Stage 6E collection 执行提交不是当前提交祖先")

    shard_id = protocol.LOCAL_SHARD
    environments = report.get("environment", {}).get("shards")
    if not isinstance(environments, dict) or set(environments) != {shard_id}:
        raise RuntimeError("Stage 6E 单分片环境缺失")
    gpu = environments[shard_id].get("gpu")
    expected_shard = protocol.EXECUTION_SHARDS[shard_id]
    if (
        not isinstance(gpu, dict)
        or gpu.get("shard_id") != shard_id
        or gpu.get("hostname") != expected_shard["hostname"]
        or any(
            gpu.get(key) != expected
            for key, expected in expected_shard["expected_gpu"].items()
        )
        or any(
            gpu.get(key) != expected
            for key, expected in protocol.EXPECTED_SOFTWARE.items()
        )
    ):
        raise RuntimeError("Stage 6E collection 执行环境漂移")

    shard_meta = report.get("shard_report_artifacts", {}).get(shard_id)
    shard_sha = report.get("shard_report_sha256", {}).get(shard_id)
    shard_relative = str(Path("shards") / shard_id / protocol.SHARD_REPORT)
    if (
        not isinstance(shard_meta, dict)
        or shard_meta.get("path") != shard_relative
        or shard_meta.get("sha256") != shard_sha
    ):
        raise RuntimeError("Stage 6E shard report 元数据漂移")
    shard_path = _safe_artifact(root, shard_relative)
    if protocol.file_sha256(shard_path) != shard_sha:
        raise RuntimeError("Stage 6E shard report 哈希漂移")
    shard_report = stage6d_auditor._load_json(shard_path)
    if (
        shard_report.get("contract_version") != protocol.PROTOCOL_VERSION
        or shard_report.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or shard_report.get("shard_id") != shard_id
        or shard_report.get("execution_commit") != execution_commit
        or shard_report.get("formal_shard_complete") is not True
        or shard_report.get("partial_shard_comparison_emitted") is not False
    ):
        raise RuntimeError("Stage 6E shard report 内容漂移")
    samples = shard_report.get("gpu_samples")
    if (
        not isinstance(samples, list)
        or not samples
        or any(sample.get("physical_index") != 1 for sample in samples)
    ):
        raise RuntimeError("Stage 6E shard GPU 监控样本漂移")

    tasks = protocol.task_plan().tasks
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("Stage 6E collection 行数不完整")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    total_elapsed = 0.0
    total_applied = 0
    termination_counts = {
        reason: 0 for reason in protocol.NORMAL_TERMINATION_REASONS
    }
    for task, row in zip(tasks, rows, strict=True):
        applied = row.get("applied_rounds")
        reason = row.get("termination_reason")
        stopping = row.get("inner_early_stopping")
        elapsed = row.get("elapsed_sec")
        average = row.get("average_sec_per_applied_round")
        if (
            row.get("task_id") != task.task_id
            or row.get("execution_shard_id") != shard_id
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
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or float(elapsed) < 0.0
            or isinstance(average, bool)
            or not isinstance(average, (int, float))
            or not math.isclose(
                float(average),
                float(elapsed) / applied if applied else 0.0,
                rel_tol=1e-15,
                abs_tol=0.0,
            )
            or row.get("gap_8k_identity") is not True
            or row.get("gap_clip_hit_count") != 0
            or row.get("factorized_gibbs_conditional_logit_clipped_count") != 0
            or row.get("direction_logit_clipped_count") != 0
            or row.get("nonfinite_count") != 0
        ):
            raise RuntimeError(f"Stage 6E case 契约漂移：{task.task_id}")
        if reason == "fit_target_reached" and stopping.get("terminal_loss") != 0.0:
            raise RuntimeError(f"Stage 6E A 停止证据漂移：{task.task_id}")
        if reason == "early_stopped" and (
            stopping.get("consecutive_no_progress_ticks", -1)
            < protocol.PATIENCE_TICKS
        ):
            raise RuntimeError(f"Stage 6E B 停止证据漂移：{task.task_id}")
        if reason == "resource_cap_reached" and (
            applied != protocol.ROUND_CAP
            or stopping.get("resource_cap_source_diagnostic_only")
            != "candidate_budget"
        ):
            raise RuntimeError(f"Stage 6E C 停止证据漂移：{task.task_id}")
        spec = protocol.DATASETS[task.dataset]
        if (
            row.get("query_identity_sha256") != spec["query_identity_sha256"]
            or row.get("target_vector_sha256") != spec["target_vector_sha256"]
            or row.get("trace_query_identity_sha256")
            != spec["trace_query_identity_sha256"]
            or row.get("trace_target_vector_sha256")
            != spec["trace_target_vector_sha256"]
        ):
            raise RuntimeError(f"Stage 6E 查询身份漂移：{task.task_id}")
        for path_key, sha_key in (
            ("terminal_table_path", "terminal_table_sha256"),
            ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
            ("transition_audit_path", "transition_audit_sha256"),
        ):
            artifact = _safe_artifact(root, row[path_key])
            if protocol.file_sha256(artifact) != row[sha_key]:
                raise RuntimeError(f"Stage 6E artifact 漂移：{task.task_id}")
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError(f"Stage 6E case 地址重复：{key}")
        indexed[key] = row
        total_elapsed += float(elapsed)
        total_applied += applied
        termination_counts[reason] += 1

    timing = report.get("timing_summary")
    if (
        not isinstance(timing, dict)
        or not math.isclose(
            float(timing.get("sum_case_elapsed_sec", -1.0)),
            total_elapsed,
            rel_tol=1e-15,
            abs_tol=0.0,
        )
        or timing.get("total_applied_rounds") != total_applied
        or not math.isclose(
            float(timing.get("average_sec_per_applied_round", -1.0)),
            total_elapsed / total_applied if total_applied else 0.0,
            rel_tol=1e-15,
            abs_tol=0.0,
        )
        or report.get("termination_reason_counts") != termination_counts
    ):
        raise RuntimeError("Stage 6E 汇总时间或停止原因复算漂移")
    if shard_report.get("raw_results") != rows:
        raise RuntimeError("Stage 6E shard 与 collection 行不一致")
    return report, indexed


def _recompute_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with _stage6d_audit_runtime(), collection._stage6d_runtime():
        cases, identities = _legacy_recompute_cases(root, indexed)
    for case in cases:
        source = indexed[(case["seed"], case["dataset"], case["arm"])]
        case["inner_early_stopping"] = source["inner_early_stopping"]
        case["cost"]["applied_rounds"] = source["applied_rounds"]
        case["cost"]["average_sec_per_applied_round"] = source[
            "average_sec_per_applied_round"
        ]
    return cases, identities


def _summary_independent(cases):
    with _stage6d_audit_runtime():
        result = _legacy_summary_independent(cases)
    for dataset in joint.DATASET_ORDER:
        for arm in joint.ARM_ORDER:
            cell = result[dataset][arm]
            cell["applied_rounds"] = stage6d_auditor._cell_summary(
                cases, dataset, arm, "cost.applied_rounds"
            )
            cell["average_sec_per_applied_round"] = (
                stage6d_auditor._cell_summary(
                    cases,
                    dataset,
                    arm,
                    "cost.average_sec_per_applied_round",
                )
            )
            selected = stage6d_auditor._cell(cases, dataset, arm)
            total_elapsed = math.fsum(
                float(case["cost"]["elapsed_sec"]) for case in selected
            )
            total_rounds = sum(
                int(case["cost"]["applied_rounds"]) for case in selected
            )
            cell["timing_total_and_average_per_round"] = {
                "total_elapsed_sec": float(total_elapsed),
                "total_applied_rounds": int(total_rounds),
                "average_sec_per_applied_round": (
                    float(total_elapsed / total_rounds)
                    if total_rounds
                    else 0.0
                ),
            }
    return result


def _audit_evaluation(
    root: Path,
    confirmed_sha: str,
    collection_sha: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    stage6d_auditor._require_sha(confirmed_sha, "Stage 6E evaluation SHA-256")
    path = root / protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
    if protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("Stage 6E evaluation SHA-256 不一致")
    report = stage6d_auditor._load_json(path)
    if (
        report.get("contract_version") != evaluator.EVALUATION_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("collection_report_sha256") != collection_sha
        or report.get("collection_execution_commit")
        != collection_report["execution_commit"]
        or report.get("case_count") != 30
        or report.get("raw_reference_data_accessed") is not True
        or report.get("l1_fully_emitted") is not True
        or report.get("historical_best_used") is not False
        or report.get("checkpoint_selected_as_output") is not False
        or report.get("new_generation_performed_by_evaluator") is not False
        or report.get("parameter_retuning_performed") is not False
        or report.get("privacy_budget_consumed") is not False
    ):
        raise RuntimeError("Stage 6E evaluation 顶层契约漂移")
    evaluation_commit = report.get("evaluation_commit")
    if (
        not isinstance(evaluation_commit, str)
        or len(evaluation_commit) != 40
        or any(character not in "0123456789abcdef" for character in evaluation_commit)
    ):
        raise RuntimeError("Stage 6E evaluation 执行提交无效")
    current_commit = collection.stage6d_runner._git_text(root, "rev-parse", "HEAD")
    if (
        collection.stage6d_runner._git_text(
            root, "merge-base", evaluation_commit, current_commit
        )
        != evaluation_commit
        or collection.stage6d_runner._git_text(
            root,
            "merge-base",
            collection_report["execution_commit"],
            evaluation_commit,
        )
        != collection_report["execution_commit"]
    ):
        raise RuntimeError("Stage 6E collection/evaluation 提交祖先关系漂移")
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("Stage 6E evaluation 查询/参考身份漂移")
    if report.get("cases") != cases:
        raise RuntimeError("Stage 6E evaluation 30条指标与独立复算不一致")
    expected_summary = _summary_independent(cases)
    if report.get("summary") != expected_summary:
        raise RuntimeError("Stage 6E evaluation 汇总与独立复算不一致")
    expected_classification = stage6d_auditor._classification_independent(cases)
    if report.get("frozen_classification") != expected_classification:
        raise RuntimeError("Stage 6E evaluation 冻结分类与独立复算不一致")
    csv_audit = stage6d_auditor._audit_csv(
        root, report, stage6d_auditor._csv_rows_independent(cases)
    )
    return report, {
        "evaluation_report_sha256": confirmed_sha,
        "all_30_case_metrics_exactly_recomputed": True,
        "all_terminal_and_checkpoint_l1_exactly_recomputed": True,
        "all_variable_round_timing_exactly_recomputed": True,
        "all_pairwise_gates_exactly_recomputed": True,
        "classification_exactly_reproduced": True,
        "l1_csv": csv_audit,
        "pass": True,
    }


def audit(
    confirmed_collection_report_sha256: str,
    confirmed_evaluation_report_sha256: str,
) -> Path:
    with _stage6d_audit_runtime(), collection._stage6d_runtime():
        return stage6d_auditor.audit(
            confirmed_collection_report_sha256,
            confirmed_evaluation_report_sha256,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--confirm-collection-sha", required=True)
    audit_parser.add_argument("--confirm-evaluation-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(collection._strict_json_text(build_plan()), end="")
        return
    path = audit(args.confirm_collection_sha, args.confirm_evaluation_sha)
    print(f"Stage 6E independent audit -> {path}")
    print(f"audit SHA-256 -> {protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
