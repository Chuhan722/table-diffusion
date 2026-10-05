#!/usr/bin/env python3
"""通过冻结 JSON 键兼容层评价恢复后的 Stage 6D（阶段 6D）第五版结果。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts import evaluate_issue53_stage6d_v5_recovered as recovery_evaluator_v1
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as generation_protocol
from scripts import issue53_stage6d_v5_evaluation_compat_protocol as compat_protocol
from scripts import issue53_stage6d_v5_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6d_v5 as recovery_collection
from scripts import validate_issue53_stage6d_v5_recovered_collection as compat_validator

EVALUATION_VERSION = "issue53-stage6d-v5-recovered-evaluation-json-key-compat-v2"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_plan() -> dict[str, Any]:
    compatibility_sha = compat_protocol.assert_frozen_compatibility_identity(
        _repo_root()
    )
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_reference_or_generation_access",
        "compatibility_protocol_sha256": compatibility_sha,
        "recovery_protocol_sha256": compat_protocol.SOURCE_RECOVERY_PROTOCOL_SHA256,
        "source_generation_protocol_sha256": (
            compat_protocol.SOURCE_GENERATION_PROTOCOL_SHA256
        ),
        "collection_report": str(
            compat_protocol.OUTPUT_DIR / compat_protocol.COLLECTION_REPORT
        ),
        "collection_report_sha256": (
            compat_protocol.CONFIRMED_COLLECTION_REPORT_SHA256
        ),
        "evaluation_report": str(
            compat_protocol.OUTPUT_DIR / compat_protocol.EVALUATION_REPORT
        ),
        "l1_results_csv": str(
            compat_protocol.OUTPUT_DIR / compat_protocol.L1_RESULTS_CSV
        ),
        "candidate_arm": generation_protocol.ARM_GAP,
        "baseline_arms": list(generation_protocol.BASELINE_ARMS),
        "stable_win_minimum": generation_protocol.STABLE_WIN_MINIMUM,
        "paired_seed_count": len(generation_protocol.FORMAL_SEEDS),
        "json_key_compatibility_validation_required": True,
        "new_generation_allowed": False,
        "generation_started": False,
        "evaluation_started": False,
    }


def evaluate(
    confirmed_collection_report_sha256: str,
    confirmed_compatibility_protocol_sha256: str,
) -> tuple[Path, Path]:
    """先验证兼容身份与完整采集，再使用第一版冻结指标算子评价。"""

    root = _repo_root()
    compat_protocol.assert_frozen_compatibility_identity(root)
    compat_protocol.require_evaluation_confirmation(
        confirmed_collection_report_sha256,
        confirmed_compatibility_protocol_sha256,
    )
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("第二版正式评价要求包含未跟踪文件在内的干净工作树")
    collection_report, indexed = compat_validator.load_collection(
        root,
        confirmed_collection_report_sha256,
        confirmed_compatibility_protocol_sha256,
    )
    cases, identities = recovery_evaluator_v1.legacy_evaluator._evaluate_cases(
        root, indexed
    )
    decisions = {
        dataset: recovery_evaluator_v1.legacy_evaluator._dataset_decision(
            cases, dataset
        )
        for dataset in joint.DATASET_ORDER
    }
    report = {
        **build_plan(),
        "mode": "offline_evaluation_after_json_key_compatible_collection_validation",
        "evaluation_started": True,
        "evaluation_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovery_evaluator_sha256": compat_protocol.file_sha256(Path(__file__)),
        "evaluation_compatibility_protocol_sha256": (
            confirmed_compatibility_protocol_sha256
        ),
        "compatibility_validator_sha256": compat_protocol.IMPLEMENTATION_SOURCES[
            "compatibility_validator"
        ]["sha256"],
        "json_key_compatibility": compat_validator.compatibility_evidence(),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_generation_commit": collection_report[
            "generation_execution_commit"
        ],
        "collection_recovery_commit": collection_report["recovery_commit"],
        "collection_recovered_after_generation": True,
        "collection_monitoring_limitation_code": collection_report[
            "execution_monitoring_limitation_code"
        ],
        "raw_reference_data_accessed": True,
        "query_and_reference_identity_audit": identities,
        "case_count": len(cases),
        "cases": cases,
        "summary": recovery_evaluator_v1.legacy_evaluator._summary(cases),
        "frozen_classification": {
            "by_dataset": decisions,
            "cross_dataset_response": (
                recovery_evaluator_v1.legacy_evaluator._cross_dataset(decisions)
            ),
        },
        "l1_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "cross_dataset_or_query_group_weighted_score_present": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    return recovery_evaluator_v1.legacy_evaluator._publish_evaluation(
        root / compat_protocol.OUTPUT_DIR,
        report,
        recovery_evaluator_v1.legacy_evaluator._l1_csv_rows(cases),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-collection-sha", required=True)
    evaluate_parser.add_argument("--confirm-compatibility-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    report, l1_csv = evaluate(
        args.confirm_collection_sha,
        args.confirm_compatibility_sha,
    )
    print(f"第二版 evaluation（评价） -> {report}")
    print(f"评价报告 SHA-256 -> {recovery_protocol.file_sha256(report)}")
    print(f"L1（平均绝对误差）结果 -> {l1_csv}")
    print(f"L1 CSV（逗号分隔表格）SHA-256 -> {recovery_protocol.file_sha256(l1_csv)}")


if __name__ == "__main__":
    main()
