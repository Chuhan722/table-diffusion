#!/usr/bin/env python3
"""离线评价经勘误恢复的 Issue #53 Stage 6E 完整结果。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts import evaluate_issue53_stage6d_formal as stage6d_evaluator
from scripts import evaluate_issue53_stage6e_autostop as source_evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as generation_protocol
from scripts import issue53_stage6e_recovery_protocol as recovery_protocol
from scripts import recover_issue53_stage6e as recovery_collection


EVALUATION_VERSION = "issue53-stage6e-recovered-evaluation-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_plan() -> dict[str, Any]:
    recovery_sha = recovery_protocol.assert_frozen_recovery_identity(_repo_root())
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_reference_or_generation_access",
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
        "candidate_arm": generation_protocol.ARM_GAP,
        "baseline_arms": list(generation_protocol.BASELINE_ARMS),
        "stable_win_minimum": generation_protocol.STABLE_WIN_MINIMUM,
        "paired_seed_count": len(generation_protocol.FORMAL_SEEDS),
        "lower_risk_ratio_max": generation_protocol.LOWER_RISK_RATIO_MAX,
        "higher_quality_ratio_min": generation_protocol.HIGHER_QUALITY_RATIO_MIN,
        "terminal_current_only": True,
        "variable_applied_rounds_expected": True,
        "new_generation_allowed": False,
        "generation_started": False,
        "evaluation_started": False,
    }


def _classification(cases: list[dict[str, Any]]) -> dict[str, Any]:
    with source_evaluator._stage6d_evaluation_runtime():
        decisions = {
            dataset: stage6d_evaluator._dataset_decision(cases, dataset)
            for dataset in joint.DATASET_ORDER
        }
        cross = stage6d_evaluator._cross_dataset(decisions)
    return {"by_dataset": decisions, "cross_dataset_response": cross}


def evaluate(confirmed_collection_report_sha256: str) -> tuple[Path, Path]:
    root = _repo_root()
    recovery_protocol.assert_frozen_recovery_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("Stage 6E 恢复版正式评价要求干净工作树")
    collection_report, indexed = recovery_collection.load_collection(
        root, confirmed_collection_report_sha256
    )
    cases, identities = source_evaluator._evaluate_cases(root, indexed)
    report = {
        **build_plan(),
        "mode": "offline_evaluation_after_recovered_complete_collection",
        "evaluation_started": True,
        "evaluation_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovery_evaluator_sha256": recovery_protocol.file_sha256(Path(__file__)),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_generation_commit": collection_report[
            "generation_execution_commit"
        ],
        "collection_recovery_commit": collection_report["recovery_commit"],
        "collection_recovered_after_generation": True,
        "collection_execution_monitoring_evidence_complete": False,
        "collection_monitoring_limitation_code": collection_report[
            "execution_monitoring_limitation_code"
        ],
        "raw_reference_data_accessed": True,
        "query_and_reference_identity_audit": identities,
        "case_count": len(cases),
        "cases": cases,
        "summary": source_evaluator._summary(cases),
        "frozen_classification": _classification(cases),
        "l1_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "cross_dataset_or_query_group_weighted_score_present": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
    }
    with source_evaluator._stage6d_evaluation_runtime():
        return stage6d_evaluator._publish_evaluation(
            root / recovery_protocol.OUTPUT_DIR,
            report,
            stage6d_evaluator._l1_csv_rows(cases),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-collection-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    report, l1_csv = evaluate(args.confirm_collection_sha)
    print(f"Stage 6E 恢复版 evaluation -> {report}")
    print(f"评价报告 SHA-256 -> {recovery_protocol.file_sha256(report)}")
    print(f"Stage 6E L1 CSV -> {l1_csv}")
    print(f"L1 CSV SHA-256 -> {recovery_protocol.file_sha256(l1_csv)}")


if __name__ == "__main__":
    main()
