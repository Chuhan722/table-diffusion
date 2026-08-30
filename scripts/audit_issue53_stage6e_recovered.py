#!/usr/bin/env python3
"""独立复核经勘误恢复的 Issue #53 Stage 6E 正式评价。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from scripts import audit_issue53_stage6d_formal as stage6d_auditor
from scripts import audit_issue53_stage6e_autostop as source_auditor
from scripts import evaluate_issue53_stage6e_recovered as evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as generation_protocol
from scripts import issue53_stage6e_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6e as recovery_collection


AUDIT_VERSION = "issue53-stage6e-recovered-independent-audit-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"JSON 根必须是对象：{path}")
    return value


def _require_sha(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是完整小写 SHA-256")


def _safe_artifact(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise TypeError("Stage 6E 恢复版 artifact 路径必须是字符串")
    part = Path(relative)
    if part.is_absolute() or ".." in part.parts:
        raise ValueError("Stage 6E 恢复版 artifact 路径越界")
    base = (root / recovery_protocol.OUTPUT_DIR).resolve()
    resolved = (base / part).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError("Stage 6E 恢复版 artifact 路径逃逸")
    return resolved


def _assert_commit_ancestry(
    root: Path, generation_commit: Any, recovery_commit: Any, evaluation_commit: Any = None
) -> None:
    values = [(generation_commit, "生成提交"), (recovery_commit, "恢复提交")]
    if evaluation_commit is not None:
        values.append((evaluation_commit, "评价提交"))
    for value, name in values:
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"Stage 6E {name}无效")
    if (
        recovery_collection._git_text(
            root, "merge-base", generation_commit, recovery_commit
        )
        != generation_commit
    ):
        raise RuntimeError("Stage 6E 生成提交不是恢复提交祖先")
    current = recovery_collection._git_text(root, "rev-parse", "HEAD")
    tail = evaluation_commit if evaluation_commit is not None else recovery_commit
    if recovery_collection._git_text(root, "merge-base", tail, current) != tail:
        raise RuntimeError("Stage 6E 恢复/评价提交不是当前提交祖先")
    if evaluation_commit is not None and (
        recovery_collection._git_text(
            root, "merge-base", recovery_commit, evaluation_commit
        )
        != recovery_commit
    ):
        raise RuntimeError("Stage 6E 恢复提交不是评价提交祖先")


def build_plan() -> dict[str, Any]:
    recovery_sha = recovery_protocol.assert_frozen_recovery_identity(_repo_root())
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_artifact_reference_or_generation_access",
        "recovery_protocol_sha256": recovery_sha,
        "source_generation_protocol_sha256": (
            recovery_protocol.SOURCE_PROTOCOL_SHA256
        ),
        "collection_report": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.COLLECTION_REPORT
        ),
        "evaluation_report": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.EVALUATION_REPORT
        ),
        "l1_results_csv": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.L1_RESULTS_CSV
        ),
        "audit_report": str(
            recovery_protocol.OUTPUT_DIR / recovery_protocol.AUDIT_REPORT
        ),
        "independently_revalidate_recovered_collection": True,
        "recompute_terminal_tables": True,
        "recompute_checkpoint_vectors": True,
        "recompute_variable_round_timing": True,
        "recompute_l1_csv": True,
        "recompute_frozen_classification": True,
        "rerun_generation": False,
    }


def _audit_row_independently(task: Any, row: dict[str, Any]) -> None:
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
        or isinstance(applied, bool)
        or not isinstance(applied, int)
        or not 0 <= applied <= generation_protocol.ROUND_CAP
        or reason not in generation_protocol.NORMAL_TERMINATION_REASONS
        or row.get("proposal_attempt_count") != applied
        or row.get("candidate_evaluation_count") != applied
        or row.get("state_evaluation_count") != max(1, applied)
        or not isinstance(stopping, dict)
        or stopping.get("enabled") is not True
        or stopping.get("patience_ticks") != generation_protocol.PATIENCE_TICKS
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
    ):
        raise RuntimeError(f"Stage 6E 独立恢复行审计失败：{task.task_id}")
    if reason == "fit_target_reached" and stopping.get("terminal_loss") != 0.0:
        raise RuntimeError(f"Stage 6E 独立 A 停止审计失败：{task.task_id}")
    if reason == "early_stopped" and (
        stopping.get("consecutive_no_progress_ticks", -1)
        < generation_protocol.PATIENCE_TICKS
    ):
        raise RuntimeError(f"Stage 6E 独立 B 停止审计失败：{task.task_id}")
    if reason == "resource_cap_reached" and (
        applied != generation_protocol.ROUND_CAP
        or stopping.get("resource_cap_source_diagnostic_only") != "candidate_budget"
    ):
        raise RuntimeError(f"Stage 6E 独立 C 停止审计失败：{task.task_id}")
    spec = generation_protocol.DATASETS[task.dataset]
    expected = {
        "query_identity_sha256": spec["query_identity_sha256"],
        "target_vector_sha256": spec["target_vector_sha256"],
        "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
        "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        "gap_8k_identity": True,
        "gap_clip_hit_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
        "direction_logit_clipped_count": 0,
        "nonfinite_count": 0,
        "distance_evaluation_count": applied,
        "direction_evaluation_count": applied,
    }
    if any(row.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Stage 6E 独立身份/护栏审计失败：{task.task_id}")
    if row.get("factorized_gibbs_microsteps") != row.get(
        "factorized_gibbs_conditional_logit_evaluated_count"
    ):
        raise RuntimeError(f"Stage 6E 独立 Gibbs 计数审计失败：{task.task_id}")
    if task.arm == generation_protocol.ARM_GAP:
        if (
            row.get("gap_microsteps") != row.get("gap_expected_microsteps")
            or row.get("factorized_gibbs_microsteps") != 0
        ):
            raise RuntimeError(f"Stage 6E 独立缺口 8*K 审计失败：{task.task_id}")
    elif row.get("gap_microsteps") != 0 or row.get("gap_expected_microsteps") != 0:
        raise RuntimeError(f"Stage 6E 独立基线核审计失败：{task.task_id}")


def _audit_transition_independently(path: Path, task: Any, row: dict[str, Any]) -> None:
    transition = _load_json(path)
    applied = row["applied_rounds"]
    clocks = transition.get("clocks")
    gap_rows = transition.get("gap_rounds")
    factor_rows = transition.get("factor_rounds")
    if (
        transition.get("contract_version") != generation_protocol.PROTOCOL_VERSION
        or transition.get("protocol_sha256")
        != recovery_protocol.SOURCE_PROTOCOL_SHA256
        or transition.get("task_id") != task.task_id
        or transition.get("round_count") != applied
        or transition.get("all_applied_unconditionally") is not True
        or not isinstance(clocks, list)
        or len(clocks) != applied
        or not isinstance(gap_rows, list)
        or not isinstance(factor_rows, list)
    ):
        raise RuntimeError(f"Stage 6E 独立转移结构审计失败：{task.task_id}")
    clock_sum = 0
    for index, item in enumerate(clocks, start=1):
        microsteps = item.get("gibbs_microsteps") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or item.get("round") != index
            or item.get("state_index") != index
            or item.get("accepted_attempt") != 1
            or item.get("candidate_evaluation_count_cumulative") != index
            or isinstance(microsteps, bool)
            or not isinstance(microsteps, int)
            or microsteps < 0
        ):
            raise RuntimeError(f"Stage 6E 独立转移时钟审计失败：{task.task_id}")
        clock_sum += microsteps
    if task.arm == generation_protocol.ARM_GAP:
        if factor_rows or len(gap_rows) != applied:
            raise RuntimeError(f"Stage 6E 独立缺口轮次审计失败：{task.task_id}")
        total = 0
        for index, item in enumerate(gap_rows, start=1):
            active = item.get("active_switches_k") if isinstance(item, dict) else None
            expected = (
                8 * active
                if isinstance(active, int) and not isinstance(active, bool) and active >= 0
                else None
            )
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
                raise RuntimeError(f"Stage 6E 独立 8*K 审计失败：{task.task_id}")
            total += expected
        if total != row["gap_microsteps"] or clock_sum != total:
            raise RuntimeError(f"Stage 6E 独立缺口总量审计失败：{task.task_id}")
    elif task.arm == "factor_b_s8":
        if gap_rows or len(factor_rows) != applied:
            raise RuntimeError(f"Stage 6E 独立因子轮次审计失败：{task.task_id}")
        total = 0
        for index, item in enumerate(factor_rows, start=1):
            microsteps = item.get("gibbs_microsteps") if isinstance(item, dict) else None
            if (
                not isinstance(item, dict)
                or item.get("round") != index
                or item.get("all_finite") is not True
                or isinstance(microsteps, bool)
                or not isinstance(microsteps, int)
                or microsteps < 0
            ):
                raise RuntimeError(f"Stage 6E 独立因子轮次审计失败：{task.task_id}")
            total += microsteps
        if total != row["factorized_gibbs_microsteps"] or clock_sum != total:
            raise RuntimeError(f"Stage 6E 独立因子总量审计失败：{task.task_id}")
    elif gap_rows or factor_rows or clock_sum:
        raise RuntimeError(f"Stage 6E 独立核转移审计失败：{task.task_id}")


def _audit_collection_independently(
    root: Path, confirmed_sha: str
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    _require_sha(confirmed_sha, "Stage 6E 恢复 collection SHA-256")
    path = root / recovery_protocol.OUTPUT_DIR / recovery_protocol.COLLECTION_REPORT
    if recovery_protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("Stage 6E 恢复 collection SHA-256 不一致")
    report = _load_json(path)
    required = {
        "contract_version": recovery_protocol.RECOVERY_PROTOCOL_VERSION,
        "recovery_protocol_sha256": recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256,
        "source_generation_protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "generation_execution_commit": recovery_protocol.SOURCE_GENERATION_COMMIT,
        "source_generation_runner_sha256": recovery_protocol.SOURCE_RUNNER_SHA256,
        "generator_params_manifest_sha256": (
            recovery_protocol.SOURCE_GENERATOR_PARAMS_MANIFEST_SHA256
        ),
        "shard_assignment_sha256": recovery_protocol.SOURCE_SHARD_ASSIGNMENT_SHA256,
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
        raise RuntimeError("Stage 6E 独立恢复 collection 顶层审计失败")
    if (
        report.get("recovery_protocol")
        != recovery_protocol.json_roundtrip(
            recovery_protocol.recovery_protocol_manifest(root)
        )
        or report.get("source_generation_protocol")
        != recovery_protocol.json_roundtrip(
            generation_protocol.frozen_protocol_manifest()
        )
        or report.get("recovery_collector_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_collector"]["sha256"]
        or report.get("collection_audit") != recovery_collection._recovery_audit()
        or report.get("source_gpu_monitoring_evidence")
        != {
            generation_protocol.LOCAL_SHARD: (
                recovery_collection._source_monitoring_limitation()
            )
        }
    ):
        raise RuntimeError("Stage 6E 独立恢复协议/限制审计失败")
    _assert_commit_ancestry(
        root,
        report.get("generation_execution_commit"),
        report.get("recovery_commit"),
    )
    inventory = recovery_protocol.recovery_inventory(root)
    tasks = generation_protocol.task_plan().tasks
    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise RuntimeError("Stage 6E 独立恢复 collection 行数不完整")
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    total_elapsed = 0.0
    total_rounds = 0
    termination = {
        reason: 0 for reason in generation_protocol.NORMAL_TERMINATION_REASONS
    }
    for task, row in zip(tasks, rows, strict=True):
        _audit_row_independently(task, row)
        case_dir = _safe_artifact(root, str(Path("cases") / task.task_id))
        manifest_path = case_dir / recovery_collection.CASE_MANIFEST
        expected_manifest_sha = inventory["shards"][generation_protocol.LOCAL_SHARD][
            "case_manifest_sha256"
        ][task.task_id]
        if recovery_protocol.file_sha256(manifest_path) != expected_manifest_sha:
            raise RuntimeError(f"Stage 6E 独立 case 清单审计失败：{task.task_id}")
        manifest = _load_json(manifest_path)
        if (
            manifest.get("collection_row") != row
            or manifest.get("contract_version")
            != generation_protocol.PROTOCOL_VERSION
            or manifest.get("protocol_sha256")
            != recovery_protocol.SOURCE_PROTOCOL_SHA256
            or manifest.get("execution_commit")
            != recovery_protocol.SOURCE_GENERATION_COMMIT
            or manifest.get("generator_params")
            != generation_protocol.generator_params_manifest(
                task.dataset, task.arm, task.seed
            )
            or manifest.get("raw_reference_data_accessed") is not False
            or manifest.get("method_comparison_emitted") is not False
        ):
            raise RuntimeError(f"Stage 6E 独立 case 来源审计失败：{task.task_id}")
        for path_key, sha_key in (
            ("terminal_table_path", "terminal_table_sha256"),
            ("checkpoint_artifact_path", "checkpoint_artifact_sha256"),
            ("transition_audit_path", "transition_audit_sha256"),
        ):
            artifact = _safe_artifact(root, row[path_key])
            if recovery_protocol.file_sha256(artifact) != row[sha_key]:
                raise RuntimeError(f"Stage 6E 独立附件审计失败：{task.task_id}")
        _audit_transition_independently(
            _safe_artifact(root, row["transition_audit_path"]), task, row
        )
        key = (task.seed, task.dataset, task.arm)
        if key in indexed:
            raise RuntimeError("Stage 6E 独立 case 地址重复")
        indexed[key] = row
        total_elapsed += float(row["elapsed_sec"])
        total_rounds += int(row["applied_rounds"])
        termination[row["termination_reason"]] += 1
    expected_timing = {
        "sum_case_elapsed_sec": float(math.fsum(float(row["elapsed_sec"]) for row in rows)),
        "total_applied_rounds": total_rounds,
        "average_sec_per_applied_round": (
            float(math.fsum(float(row["elapsed_sec"]) for row in rows) / total_rounds)
            if total_rounds
            else 0.0
        ),
    }
    if (
        report.get("timing_summary") != expected_timing
        or report.get("termination_reason_counts") != termination
    ):
        raise RuntimeError("Stage 6E 独立时间/停止汇总审计失败")
    shard_id = generation_protocol.LOCAL_SHARD
    shard_meta = report.get("shard_report_artifacts", {}).get(shard_id)
    staging_meta = report.get("source_staging_manifest_artifacts", {}).get(shard_id)
    shard_path = str(Path("shards") / shard_id / recovery_protocol.SHARD_REPORT)
    staging_path = str(Path("shards") / shard_id / recovery_collection.STAGING_MANIFEST)
    if (
        not isinstance(shard_meta, dict)
        or shard_meta.get("path") != shard_path
        or recovery_protocol.file_sha256(_safe_artifact(root, shard_path))
        != shard_meta.get("sha256")
        or not isinstance(staging_meta, dict)
        or staging_meta.get("path") != staging_path
        or staging_meta.get("sha256")
        != inventory["shards"][shard_id]["staging_manifest_sha256"]
        or recovery_protocol.file_sha256(_safe_artifact(root, staging_path))
        != staging_meta.get("sha256")
    ):
        raise RuntimeError("Stage 6E 独立分片证据审计失败")
    shard_report = _load_json(_safe_artifact(root, shard_path))
    if shard_report.get("raw_results") != rows:
        raise RuntimeError("Stage 6E 独立分片/collection 行审计失败")
    for seed in generation_protocol.FORMAL_SEEDS:
        for dataset in joint.DATASET_ORDER:
            triplet = [indexed[(seed, dataset, arm)] for arm in joint.ARM_ORDER]
            for name in (
                "initial_table_sha256",
                "primary_rng_post_initialization_sha256",
                "direction_reference_scale",
                "query_identity_sha256",
                "target_vector_sha256",
            ):
                if len({row[name] for row in triplet}) != 1:
                    raise RuntimeError("Stage 6E 独立三核配对审计失败")
    return report, indexed


def _classification_independent(cases: list[dict[str, Any]]) -> dict[str, Any]:
    with source_auditor._stage6d_audit_runtime():
        return stage6d_auditor._classification_independent(cases)


def _audit_evaluation(
    root: Path,
    confirmed_sha: str,
    collection_sha: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    _require_sha(confirmed_sha, "Stage 6E 恢复 evaluation SHA-256")
    path = root / recovery_protocol.OUTPUT_DIR / recovery_protocol.EVALUATION_REPORT
    if recovery_protocol.file_sha256(path) != confirmed_sha:
        raise ValueError("Stage 6E 恢复 evaluation SHA-256 不一致")
    report = _load_json(path)
    required = {
        "contract_version": evaluator.EVALUATION_VERSION,
        "recovery_protocol_sha256": recovery_protocol.FROZEN_RECOVERY_PROTOCOL_SHA256,
        "source_generation_protocol_sha256": recovery_protocol.SOURCE_PROTOCOL_SHA256,
        "collection_report_sha256": collection_sha,
        "collection_generation_commit": collection_report[
            "generation_execution_commit"
        ],
        "collection_recovery_commit": collection_report["recovery_commit"],
        "collection_recovered_after_generation": True,
        "collection_execution_monitoring_evidence_complete": False,
        "collection_monitoring_limitation_code": (
            recovery_protocol.MONITORING_LIMITATION_CODE
        ),
        "case_count": 30,
        "raw_reference_data_accessed": True,
        "l1_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("Stage 6E 恢复 evaluation 顶层审计失败")
    if (
        report.get("recovery_evaluator_sha256")
        != recovery_protocol.IMPLEMENTATION_SOURCES["recovery_evaluator"]["sha256"]
    ):
        raise RuntimeError("Stage 6E 恢复评价器身份审计失败")
    _assert_commit_ancestry(
        root,
        collection_report["generation_execution_commit"],
        collection_report["recovery_commit"],
        report.get("evaluation_commit"),
    )
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("Stage 6E 恢复查询/reference 身份复算不一致")
    if report.get("cases") != cases:
        raise RuntimeError("Stage 6E 恢复 30 条指标与独立复算不一致")
    expected_summary = source_auditor._summary_independent(cases)
    if report.get("summary") != expected_summary:
        raise RuntimeError("Stage 6E 恢复汇总与独立复算不一致")
    expected_classification = _classification_independent(cases)
    if report.get("frozen_classification") != expected_classification:
        raise RuntimeError("Stage 6E 恢复冻结分类与独立复算不一致")
    with source_auditor._stage6d_audit_runtime():
        csv_audit = stage6d_auditor._audit_csv(
            root,
            report,
            stage6d_auditor._csv_rows_independent(cases),
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
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("Stage 6E 恢复版独立审计要求干净工作树")
    collection_report, indexed = _audit_collection_independently(
        root, confirmed_collection_report_sha256
    )
    cases, identities = source_auditor._recompute_cases(root, indexed)
    _, evaluation_audit = _audit_evaluation(
        root,
        confirmed_evaluation_report_sha256,
        confirmed_collection_report_sha256,
        collection_report,
        cases,
        identities,
    )
    report = {
        **build_plan(),
        "mode": "independent_audit_after_recovered_evaluation",
        "audit_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovery_independent_auditor_sha256": recovery_protocol.file_sha256(
            Path(__file__)
        ),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "evaluation_report_sha256": confirmed_evaluation_report_sha256,
        "collection_independent_audit": {
            "all_30_frozen_case_manifests_verified": True,
            "all_90_case_attachment_hashes_verified": True,
            "corrected_state_evaluation_identity_independently_verified": True,
            "all_transition_clocks_and_kernel_microsteps_verified": True,
            "all_three_kernel_pairing_identities_verified": True,
            "monitoring_evidence_limitation_preserved": True,
            "pass": True,
        },
        "evaluation_independent_audit": evaluation_audit,
        "raw_reference_data_accessed_by_auditor": True,
        "generation_rerun": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
        "pass": True,
    }
    path = root / recovery_protocol.OUTPUT_DIR / recovery_protocol.AUDIT_REPORT
    recovery_collection._write_json_new(path, report)
    return path


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
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    path = audit(args.confirm_collection_sha, args.confirm_evaluation_sha)
    print(f"Stage 6E 恢复版 independent audit -> {path}")
    print(f"审计 SHA-256 -> {recovery_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
