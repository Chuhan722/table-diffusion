#!/usr/bin/env python3
"""离线评价经勘误恢复的 Issue #53 R8 四轨迹 collection。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts import evaluate_issue53_gap_weight_r8_screen as source_evaluator
from scripts import issue53_gap_weight_r8_recovered_evaluation_protocol as adapter_protocol
from scripts import issue53_gap_weight_r8_screen_execution_protocol as source_protocol
from scripts import recover_issue53_gap_weight_r8_screen as recovery_collection


EVALUATION_VERSION = adapter_protocol.ADAPTER_VERSION


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_plan() -> dict[str, Any]:
    adapter_sha = adapter_protocol.assert_frozen_adapter_identity(_repo_root())
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_reference_quality_or_generation_access",
        "evaluation_adapter_protocol_sha256": adapter_sha,
        "source_evaluation_contract_version": (
            adapter_protocol.SOURCE_EVALUATOR_VERSION
        ),
        "source_evaluator_sha256": adapter_protocol.SOURCE_EVALUATOR_SHA256,
        "source_recovery_protocol_sha256": (
            adapter_protocol.SOURCE_RECOVERY_PROTOCOL_SHA256
        ),
        "collection_report": str(adapter_protocol.SOURCE_COLLECTION_PATH),
        "collection_report_sha256": adapter_protocol.SOURCE_COLLECTION_SHA256,
        "evaluation_report": str(
            source_protocol.OUTPUT_DIR / source_protocol.EVALUATION_REPORT
        ),
        "screen_metrics_csv": str(
            source_protocol.OUTPUT_DIR / source_protocol.L1_RESULTS_CSV
        ),
        "delegated_source_functions": list(
            adapter_protocol.DELEGATED_SOURCE_FUNCTIONS
        ),
        "source_metric_arithmetic_modified": False,
        "source_classification_modified": False,
        "new_generation_allowed": False,
        "generation_started": False,
        "raw_reference_data_accessed": False,
        "evaluation_started": False,
        "requires_explicit_later_user_instruction": True,
        "requires_adapter_and_collection_sha_confirmation": True,
    }


def _assert_outputs_absent(root: Path) -> None:
    destination = root / source_protocol.OUTPUT_DIR
    existing = [
        path
        for path in (
            destination / source_protocol.EVALUATION_REPORT,
            destination / source_protocol.L1_RESULTS_CSV,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "R8 恢复版评价产物已存在，不读取参考表且不覆盖："
            + ", ".join(str(path) for path in existing)
        )


def evaluate(
    confirmed_adapter_sha256: str,
    confirmed_collection_sha256: str,
) -> tuple[Path, Path]:
    adapter_protocol.require_evaluation_confirmation(
        confirmed_adapter_sha256,
        confirmed_collection_sha256,
    )
    root = _repo_root()
    adapter_protocol.assert_frozen_adapter_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("R8 恢复版评价要求包含未跟踪文件在内的干净工作树")
    _assert_outputs_absent(root)
    collection_report, indexed = recovery_collection.load_collection(
        root,
        confirmed_collection_sha256,
    )

    # 从这里起才允许访问参考表。
    # 全部算术与判定直接委托给冻结原 evaluator。
    cases, identities = source_evaluator._evaluate_cases(root, indexed)
    summary = source_evaluator._summary_and_decision(cases)
    recovery = collection_report["postcollection_recovery"]
    report = {
        **build_plan(),
        "mode": "offline_evaluation_after_recovered_complete_r8_collection",
        "evaluation_started": True,
        "evaluation_commit": recovery_collection._git_text(
            root, "rev-parse", "HEAD"
        ),
        "recovered_evaluator_sha256": adapter_protocol.file_sha256(
            Path(__file__)
        ),
        "collection_report_sha256": confirmed_collection_sha256,
        "collection_generation_commit": collection_report["execution_commit"],
        "collection_recovery_commit": recovery["recovery_commit"],
        "collection_recovered_after_generation": True,
        "collection_execution_monitoring_evidence_complete": (
            collection_report["execution_monitoring_evidence_complete"]
        ),
        "collection_monitoring_limitation_code": collection_report[
            "execution_monitoring_limitation_code"
        ],
        "query_and_reference_identity_audit": identities,
        "case_count": len(cases),
        "cases": cases,
        "summary": summary["datasets"],
        "execution_valid": summary["execution_valid"],
        "quality_interpretation_allowed": summary[
            "quality_interpretation_allowed"
        ],
        "frozen_classification": summary["frozen_classification"],
        "classification_precedence": summary["classification_precedence"],
        "automatic_followup_authorized": False,
        "raw_reference_data_accessed": True,
        "all_four_cases_evaluated": len(cases) == 4,
        "terminal_and_reached_checkpoints_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    return source_evaluator._publish(
        root / source_protocol.OUTPUT_DIR,
        report,
        source_evaluator._csv_rows(cases),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-adapter-sha", required=True)
    evaluate_parser.add_argument("--confirm-collection-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    report, csv_path = evaluate(
        args.confirm_adapter_sha,
        args.confirm_collection_sha,
    )
    print(f"R8 recovered screen evaluation -> {report}")
    print(f"R8 recovered screen metrics -> {csv_path}")
    print(f"evaluation SHA-256 -> {adapter_protocol.file_sha256(report)}")


if __name__ == "__main__":
    main()
