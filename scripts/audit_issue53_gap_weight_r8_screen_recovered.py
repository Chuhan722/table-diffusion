#!/usr/bin/env python3
"""独立复核经勘误恢复的 Issue #53 R8 四轨迹评价。"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

from scripts import audit_issue53_gap_weight_r8_screen as source_auditor
from scripts import issue53_gap_weight_r8_recovered_audit_protocol as audit_protocol
from scripts import issue53_gap_weight_r8_recovered_evaluation_protocol as evaluation_protocol
from scripts import issue53_gap_weight_r8_screen_execution_protocol as source_protocol
from scripts import recover_issue53_gap_weight_r8_screen as recovery_collection


AUDIT_VERSION = audit_protocol.AUDIT_ADAPTER_VERSION


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_plan() -> dict[str, Any]:
    audit_sha = audit_protocol.assert_frozen_audit_adapter_identity(_repo_root())
    return {
        "contract_version": AUDIT_VERSION,
        "mode": "plan_only_no_reference_recomputation_or_generation_access",
        "audit_adapter_protocol_sha256": audit_sha,
        "source_independent_auditor_version": (
            audit_protocol.SOURCE_AUDITOR_VERSION
        ),
        "source_independent_auditor_sha256": (
            audit_protocol.SOURCE_AUDITOR_SHA256
        ),
        "source_evaluation_adapter_protocol_sha256": (
            audit_protocol.SOURCE_EVALUATION_ADAPTER_SHA256
        ),
        "collection_report_sha256": audit_protocol.SOURCE_COLLECTION_SHA256,
        "evaluation_report_sha256": audit_protocol.SOURCE_EVALUATION_SHA256,
        "screen_metrics_csv_sha256": (
            audit_protocol.SOURCE_METRICS_CSV_SHA256
        ),
        "audit_report": str(audit_protocol.AUDIT_OUTPUT_PATH),
        "delegated_source_functions": list(
            audit_protocol.DELEGATED_SOURCE_FUNCTIONS
        ),
        "imports_primary_r8_evaluator_arithmetic": False,
        "independently_recompute_terminal_and_checkpoints": True,
        "independently_recompute_bins_groups_ratios_and_classification": True,
        "independently_recompute_and_compare_csv": True,
        "audit_started": False,
        "raw_reference_data_accessed": False,
        "rerun_generation": False,
        "evaluation_artifacts_modified": False,
    }


def _assert_audit_output_absent(root: Path) -> None:
    destination = root / audit_protocol.AUDIT_OUTPUT_PATH
    if destination.exists():
        raise FileExistsError(
            "R8 恢复版独立审计报告已存在，"
            f"不读取参考表且不覆盖：{destination}"
        )


def _validate_commit_ancestry(
    root: Path,
    generation_commit: Any,
    recovery_commit: Any,
    evaluation_commit: Any,
) -> None:
    values = (
        (generation_commit, "生成提交"),
        (recovery_commit, "恢复提交"),
        (evaluation_commit, "评价提交"),
    )
    for value, name in values:
        if (
            not isinstance(value, str)
            or len(value) != 40
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise RuntimeError(f"R8 恢复版独立审计{name}无效")
    current = recovery_collection._git_text(root, "rev-parse", "HEAD")
    for ancestor, descendant, label in (
        (generation_commit, recovery_commit, "生成/恢复"),
        (recovery_commit, evaluation_commit, "恢复/评价"),
        (evaluation_commit, current, "评价/审计"),
    ):
        if (
            recovery_collection._git_text(
                root, "merge-base", ancestor, descendant
            )
            != ancestor
        ):
            raise RuntimeError(f"R8 恢复版独立审计{label}祖先关系漂移")


def _audit_recovered_evaluation(
    root: Path,
    confirmed_evaluation_sha256: str,
    confirmed_collection_sha256: str,
    collection_report: dict[str, Any],
    cases: list[dict[str, Any]],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_auditor._require_sha(
        confirmed_evaluation_sha256,
        "R8 恢复版 evaluation SHA-256",
    )
    path = root / audit_protocol.SOURCE_EVALUATION_PATH
    if audit_protocol.file_sha256(path) != confirmed_evaluation_sha256:
        raise ValueError("R8 恢复版独立审计 evaluation SHA-256 不一致")
    report = source_auditor._load_json(path)
    recovery = collection_report["postcollection_recovery"]
    required = {
        "contract_version": evaluation_protocol.ADAPTER_VERSION,
        "evaluation_adapter_protocol_sha256": (
            evaluation_protocol.FROZEN_ADAPTER_SHA256
        ),
        "source_evaluation_contract_version": (
            evaluation_protocol.SOURCE_EVALUATOR_VERSION
        ),
        "source_evaluator_sha256": evaluation_protocol.SOURCE_EVALUATOR_SHA256,
        "source_recovery_protocol_sha256": (
            evaluation_protocol.SOURCE_RECOVERY_PROTOCOL_SHA256
        ),
        "collection_report_sha256": confirmed_collection_sha256,
        "collection_generation_commit": collection_report["execution_commit"],
        "collection_recovery_commit": recovery["recovery_commit"],
        "collection_recovered_after_generation": True,
        "collection_execution_monitoring_evidence_complete": False,
        "collection_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "case_count": 4,
        "evaluation_started": True,
        "generation_started": False,
        "source_metric_arithmetic_modified": False,
        "source_classification_modified": False,
        "raw_reference_data_accessed": True,
        "all_four_cases_evaluated": True,
        "terminal_and_reached_checkpoints_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
        "automatic_followup_authorized": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("R8 恢复版独立审计 evaluation 顶层契约漂移")
    if (
        report.get("recovered_evaluator_sha256")
        != evaluation_protocol.IMPLEMENTATION_SOURCES[
            "recovered_evaluation_adapter"
        ]["sha256"]
        or report.get("delegated_source_functions")
        != list(evaluation_protocol.DELEGATED_SOURCE_FUNCTIONS)
    ):
        raise RuntimeError("R8 恢复版独立审计 evaluator 源码或委托漂移")
    evaluation_commit = report.get("evaluation_commit")
    if evaluation_commit != audit_protocol.SOURCE_EVALUATION_COMMIT:
        raise RuntimeError("R8 恢复版独立审计评价提交漂移")
    _validate_commit_ancestry(
        root,
        collection_report["execution_commit"],
        recovery["recovery_commit"],
        evaluation_commit,
    )
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("R8 恢复版独立审计查询/参考身份漂移")
    source_auditor._assert_equal(report.get("cases"), cases, "cases")
    summary, execution_valid, classification = (
        source_auditor._summary_independent(cases)
    )
    source_auditor._assert_equal(report.get("summary"), summary, "summary")
    if (
        report.get("execution_valid") is not execution_valid
        or report.get("quality_interpretation_allowed") is not execution_valid
        or report.get("frozen_classification") != classification
        or report.get("classification_precedence")
        != source_protocol.scientific.frozen_protocol_manifest()[
            "screen_decision"
        ]["precedence"]
    ):
        raise RuntimeError("R8 恢复版独立审计冻结分类漂移")
    csv_audit = source_auditor._audit_csv(
        root,
        report,
        source_auditor._csv_rows_independent(cases),
    )
    return report, {
        "evaluation_report_sha256": confirmed_evaluation_sha256,
        "all_4_case_metrics_recomputed": True,
        "all_terminal_and_checkpoint_metrics_recomputed": True,
        "both_proxy_objectives_recomputed": True,
        "all_target_count_bins_recomputed": True,
        "all_query_orders_recomputed": True,
        "all_offline_groups_recomputed": True,
        "summary_independently_reproduced": True,
        "classification_independently_reproduced": True,
        "verified_frozen_classification": classification,
        "screen_metrics_csv": csv_audit,
        "pass": True,
    }


def _publish_audit(root: Path, payload: dict[str, Any]) -> Path:
    destination = root / audit_protocol.AUDIT_OUTPUT_PATH
    if destination.exists():
        raise FileExistsError("R8 恢复版独立审计报告已存在，不覆盖")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".r8-recovered-screen-audit.tmp-",
            dir=destination.parent,
        )
    )
    try:
        temporary_path = temporary / destination.name
        temporary_path.write_text(
            recovery_collection._strict_json_text(payload),
            encoding="utf-8",
        )
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary.rmdir()
        except OSError:
            pass
    return destination


def audit(
    confirmed_audit_adapter_sha256: str,
    confirmed_collection_sha256: str,
    confirmed_evaluation_sha256: str,
) -> Path:
    audit_protocol.require_audit_confirmation(
        confirmed_audit_adapter_sha256,
        confirmed_collection_sha256,
        confirmed_evaluation_sha256,
    )
    root = _repo_root()
    audit_protocol.assert_frozen_audit_adapter_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "R8 恢复版独立审计要求包含未跟踪文件在内的干净工作树"
        )
    _assert_audit_output_absent(root)
    collection_report, indexed = recovery_collection.load_collection(
        root,
        confirmed_collection_sha256,
    )

    # 从这里起才读取参考表；不导入主 evaluator。
    # 全部复算来自原独立 auditor。
    cases, identities = source_auditor._recompute_cases(root, indexed)
    evaluation_report, evidence = _audit_recovered_evaluation(
        root,
        confirmed_evaluation_sha256,
        confirmed_collection_sha256,
        collection_report,
        cases,
        identities,
    )
    payload = {
        **build_plan(),
        "mode": "independent_recomputation_after_recovered_evaluation",
        "audit_started": True,
        "audit_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovered_auditor_sha256": audit_protocol.file_sha256(Path(__file__)),
        "collection_report_sha256": confirmed_collection_sha256,
        "evaluation_report_sha256": confirmed_evaluation_sha256,
        "evaluation_commit": evaluation_report["evaluation_commit"],
        **evidence,
        "raw_reference_data_accessed": True,
        "collection_execution_monitoring_evidence_complete": False,
        "collection_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "evaluator_arithmetic_imported": False,
        "source_independent_auditor_reused": True,
        "generation_rerun": False,
        "evaluation_artifacts_modified": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
        "automatic_followup_authorized": False,
    }
    return _publish_audit(root, payload)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--confirm-audit-adapter-sha", required=True)
    audit_parser.add_argument("--confirm-collection-sha", required=True)
    audit_parser.add_argument("--confirm-evaluation-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    path = audit(
        args.confirm_audit_adapter_sha,
        args.confirm_collection_sha,
        args.confirm_evaluation_sha,
    )
    print(f"R8 recovered independent audit -> {path}")
    print(f"audit SHA-256 -> {audit_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
