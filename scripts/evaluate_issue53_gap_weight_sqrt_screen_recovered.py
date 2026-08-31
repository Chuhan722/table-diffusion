#!/usr/bin/env python3
"""离线评价经勘误恢复的 Issue #53 平方根权重两轨迹 collection。"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import evaluate_issue53_gap_weight_r8_screen as source_evaluator
from scripts import issue53_gap_weight_sqrt_recovered_evaluation_protocol as protocol
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific
from scripts import recover_issue53_gap_weight_sqrt_screen as recovery_collection


EVALUATION_VERSION = protocol.ADAPTER_VERSION
LEGACY_ARM = "gap_legacy_s8"
R8_ARM = "gap_bounded_r8_s8"
PRESENTATION_ARM_ORDER = (LEGACY_ARM, R8_ARM, protocol.CANDIDATE_ARM)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _source_evaluator_runtime():
    """仅在调用域内把冻结 R8 指标算术绑定到平方根两 case。"""

    original_protocol = source_evaluator.protocol
    source_evaluator.protocol = protocol
    try:
        yield
    finally:
        source_evaluator.protocol = original_protocol


def build_plan() -> dict[str, Any]:
    adapter_sha = protocol.assert_frozen_adapter_identity(_repo_root())
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_reference_quality_or_generation_access",
        "evaluation_adapter_protocol_sha256": adapter_sha,
        "source_scientific_protocol_sha256": (
            scientific.FROZEN_PROTOCOL_SHA256
        ),
        "source_evaluator_version": protocol.SOURCE_EVALUATOR_VERSION,
        "source_evaluator_sha256": protocol.SOURCE_EVALUATOR_SHA256,
        "source_recovery_protocol_sha256": (
            protocol.SOURCE_RECOVERY_PROTOCOL_SHA256
        ),
        "candidate_collection_report": str(protocol.SOURCE_COLLECTION_PATH),
        "candidate_collection_report_sha256": (
            protocol.SOURCE_COLLECTION_SHA256
        ),
        "baseline_artifacts": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in protocol.BASELINE_ARTIFACTS.items()
        },
        "evaluation_report": str(
            protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT
        ),
        "screen_metrics_csv": str(
            protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV
        ),
        "delegated_source_functions": list(
            protocol.DELEGATED_SOURCE_FUNCTIONS
        ),
        "candidate_metric_arithmetic_modified": False,
        "ratio_zero_denominator_rule_modified": False,
        "classification_source": "frozen_sqrt_scientific_protocol",
        "screen_gates": scientific.frozen_protocol_manifest()[
            "screen_gates"
        ],
        "baseline_cases_recomputed": False,
        "new_generation_allowed": False,
        "generation_started": False,
        "raw_reference_data_accessed": False,
        "candidate_quality_accessed": False,
        "baseline_quality_values_accessed": False,
        "evaluation_started": False,
        "requires_explicit_later_user_instruction": True,
        "requires_adapter_and_collection_sha_confirmation": True,
    }


def _assert_outputs_absent(root: Path) -> None:
    destination = root / protocol.OUTPUT_DIR
    existing = [
        path
        for path in (
            destination / protocol.EVALUATION_REPORT,
            destination / protocol.L1_RESULTS_CSV,
        )
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "平方根评价产物已存在，不读取参考表且不覆盖："
            + ", ".join(str(path) for path in existing)
        )


def preflight() -> dict[str, Any]:
    """只校验哈希、干净工作树与输出不存在，不读质量。"""

    root = _repo_root()
    plan = build_plan()
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "平方根评价预检要求包含未跟踪文件在内的干净工作树"
        )
    _assert_outputs_absent(root)
    return {
        **plan,
        "mode": "read_only_preflight_no_quality_or_reference_access",
        "evaluation_commit": recovery_collection._git_text(
            root, "rev-parse", "HEAD"
        ),
        "worktree_clean_including_untracked": True,
        "evaluation_outputs_absent": True,
        "ready_for_explicit_evaluation_confirmation": True,
    }


def _load_frozen_baseline(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """只在评价确认门之后读取已独立审计的基线质量产物。"""

    evaluation_path = root / protocol.BASELINE_EVALUATION_PATH
    audit_path = root / protocol.BASELINE_AUDIT_PATH
    if (
        protocol.file_sha256(evaluation_path)
        != protocol.BASELINE_EVALUATION_SHA256
        or protocol.file_sha256(audit_path) != protocol.BASELINE_AUDIT_SHA256
    ):
        raise RuntimeError("平方根评价复用的 R8 基线哈希漂移")
    evaluation = recovery_collection._load_json(evaluation_path)
    audit = recovery_collection._load_json(audit_path)
    expected_csv = protocol.BASELINE_ARTIFACTS["metrics_csv"]
    required_evaluation = {
        "contract_version": (
            "issue53-gap-weight-r8-recovered-evaluation-adapter-v1"
        ),
        "evaluation_adapter_protocol_sha256": (
            protocol.BASELINE_EVALUATION_ADAPTER_SHA256
        ),
        "source_evaluator_sha256": protocol.SOURCE_EVALUATOR_SHA256,
        "collection_report_sha256": protocol.BASELINE_ARTIFACTS[
            "collection"
        ]["sha256"],
        "case_count": 4,
        "execution_valid": True,
        "quality_interpretation_allowed": True,
        "all_four_cases_evaluated": True,
        "terminal_and_reached_checkpoints_fully_emitted": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
    }
    if any(
        evaluation.get(key) != value
        for key, value in required_evaluation.items()
    ):
        raise RuntimeError("平方根评价复用的 R8 evaluation 契约漂移")
    csv_meta = evaluation.get("screen_metrics_csv")
    if (
        not isinstance(csv_meta, dict)
        or csv_meta.get("path") != expected_csv["path"].name
        or csv_meta.get("sha256") != expected_csv["sha256"]
    ):
        raise RuntimeError("平方根评价复用的 R8 CSV 绑定漂移")
    required_audit = {
        "contract_version": (
            "issue53-gap-weight-r8-recovered-independent-audit-v1"
        ),
        "audit_adapter_protocol_sha256": (
            protocol.BASELINE_AUDIT_ADAPTER_SHA256
        ),
        "evaluation_report_sha256": protocol.BASELINE_EVALUATION_SHA256,
        "pass": True,
        "classification_independently_reproduced": True,
        "summary_independently_reproduced": True,
        "evaluator_arithmetic_imported": False,
        "generation_rerun": False,
        "evaluation_artifacts_modified": False,
    }
    if any(audit.get(key) != value for key, value in required_audit.items()):
        raise RuntimeError("平方根评价复用的 R8 独立审计契约漂移")
    cases = evaluation.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise RuntimeError("平方根评价复用的 R8 基线不是完整四条")
    expected_addresses = {
        (dataset, arm, scientific.DEVELOPMENT_SEED)
        for dataset in scientific.DATASET_ORDER
        for arm in (LEGACY_ARM, R8_ARM)
    }
    addresses = {
        (case.get("dataset"), case.get("arm"), case.get("seed"))
        for case in cases
        if isinstance(case, dict)
    }
    if addresses != expected_addresses or len(addresses) != len(cases):
        raise RuntimeError("平方根评价复用的 R8 基线地址漂移")
    identities = evaluation.get("query_and_reference_identity_audit")
    if not isinstance(identities, dict):
        raise TypeError("平方根评价复用的 R8 查询/参考身份缺失")
    return evaluation, cases, audit


def _evaluate_candidate_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with _source_evaluator_runtime():
        cases, identities = source_evaluator._evaluate_cases(root, indexed)
    expected = [task.task_id for task in protocol.task_plan().tasks]
    if (
        [case.get("task_id") for case in cases] != expected
        or len(cases) != 2
        or any(case.get("arm") != protocol.CANDIDATE_ARM for case in cases)
    ):
        raise RuntimeError("平方根评价候选 case 身份漂移")
    return cases, identities


def _case(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str
) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if case.get("dataset") == dataset and case.get("arm") == arm
    ]
    if (
        len(selected) != 1
        or selected[0].get("seed") != scientific.DEVELOPMENT_SEED
    ):
        raise RuntimeError(f"{dataset}/{arm} 缺少唯一冻结开发种子")
    return selected[0]


def _comparison(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    dataset: str,
    path: str,
    baseline_arm: str,
) -> dict[str, Any]:
    candidate = source_evaluator._nested(
        _case(candidate_cases, dataset, protocol.CANDIDATE_ARM), path
    )
    baseline_value = source_evaluator._nested(
        _case(baseline_cases, dataset, baseline_arm), path
    )
    source_record = source_evaluator._ratio_record(candidate, baseline_value)
    return {
        "metric": path,
        "candidate_arm": protocol.CANDIDATE_ARM,
        "baseline_arm": baseline_arm,
        "candidate_value": source_record["bounded_r8"],
        "baseline_value": source_record["legacy"],
        "candidate_over_baseline": source_record["bounded_over_legacy"],
        "ratio_state": source_record["ratio_state"],
        "lower_is_better": source_record["lower_is_better"],
    }


def _both_baselines(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    dataset: str,
    path: str,
) -> dict[str, Any]:
    return {
        "vs_legacy": _comparison(
            candidate_cases, baseline_cases, dataset, path, LEGACY_ARM
        ),
        "vs_r8": _comparison(
            candidate_cases, baseline_cases, dataset, path, R8_ARM
        ),
    }


def _decision_ratio(record: Mapping[str, Any]) -> float:
    return source_evaluator._decision_ratio({
        "ratio_state": record["ratio_state"],
        "bounded_over_legacy": record["candidate_over_baseline"],
    })


def _checkpoint_curves(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    dataset: str,
) -> dict[str, Any]:
    combined = [*baseline_cases, *candidate_cases]
    curves: dict[str, Any] = {}
    for arm in PRESENTATION_ARM_ORDER:
        case = _case(combined, dataset, arm)
        by_round = {
            int(row["round"]): row
            for row in case["trajectory"]["fixed_checkpoints"]
        }
        curves[arm] = {
            str(round_index): (
                {
                    "normalized_l1": by_round[round_index]["normalized_l1"],
                    "legacy_relative_gap_proxy": by_round[round_index][
                        "legacy_relative_gap_proxy"
                    ],
                    "bounded_r8_gap_proxy": by_round[round_index][
                        "bounded_r8_gap_proxy"
                    ],
                    "selection_role": "diagnostic_only",
                }
                if round_index in by_round
                else None
            )
            for round_index in scientific.CHECKPOINT_ROUNDS
        }
    return curves


def _gate_record(
    comparison: Mapping[str, Any],
    *,
    operator: str,
    threshold: float,
    execution_valid: bool,
) -> dict[str, Any]:
    ratio = _decision_ratio(comparison)
    if operator == "strictly_less_than":
        passed = ratio < threshold
    elif operator == "less_than_or_equal":
        passed = ratio <= threshold
    else:
        raise ValueError(f"未知平方根筛查门操作符：{operator}")
    return {
        "comparison": dict(comparison),
        "operator": operator,
        "threshold": float(threshold),
        "pass": bool(passed) if execution_valid else None,
        "interpreted": bool(execution_valid),
    }


def _summary_and_decision(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    baseline_evaluation: Mapping[str, Any],
    baseline_audit: Mapping[str, Any],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in scientific.DATASET_ORDER:
        measured = _both_baselines(
            candidate_cases,
            baseline_cases,
            dataset,
            "metrics.measured.overall.normalized_l1_mean",
        )
        proxies = {
            name: _both_baselines(
                candidate_cases,
                baseline_cases,
                dataset,
                f"metrics.measured.proxy_objectives.{name}",
            )
            for name in ("legacy_relative_gap", "bounded_r8_gap")
        }
        target_bins = {
            item["name"]: _both_baselines(
                candidate_cases,
                baseline_cases,
                dataset,
                "metrics.measured.target_count_bins."
                f"{item['name']}.absolute_count_error_mean",
            )
            for item in scientific.TARGET_COUNT_BINS[dataset]
        }
        orders = {
            str(order): _both_baselines(
                candidate_cases,
                baseline_cases,
                dataset,
                f"metrics.measured.by_order.{order}.absolute_count_error_mean",
            )
            for order in scientific.DATASETS[dataset]["order_counts"]
        }
        group_names = (
            protocol.TEST_GROUP_ORDER
            if dataset == "test_300x10"
            else tuple(protocol.NLTCS_GROUP_COUNTS)
        )
        offline_groups = {
            name: _both_baselines(
                candidate_cases,
                baseline_cases,
                dataset,
                f"metrics.offline_query_groups.{name}.normalized_l1_mean",
            )
            for name in group_names
        }
        combined = [*baseline_cases, *candidate_cases]
        datasets[dataset] = {
            "measured_normalized_l1": measured,
            "proxy_objectives": proxies,
            "target_count_bins": target_bins,
            "query_orders": orders,
            "offline_query_groups": offline_groups,
            "checkpoint_curves": _checkpoint_curves(
                candidate_cases, baseline_cases, dataset
            ),
            "terminal_cost_by_arm": {
                arm: _case(combined, dataset, arm)["cost"]
                for arm in PRESENTATION_ARM_ORDER
            },
        }

    candidate_execution_valid = all(
        source_evaluator._nested(case, "metrics.validity.valid_row_rate")
        == 1.0
        and case["kernel_audit"]["gap_weighting_guard_passed"] is True
        for case in candidate_cases
    )
    baseline_execution_valid = (
        baseline_evaluation.get("execution_valid") is True
        and baseline_audit.get("pass") is True
        and all(
            source_evaluator._nested(case, "metrics.validity.valid_row_rate")
            == 1.0
            and case["kernel_audit"]["gap_weighting_guard_passed"] is True
            for case in baseline_cases
        )
    )
    execution_valid = bool(
        candidate_execution_valid and baseline_execution_valid
    )
    nltcs = datasets["nltcs"]
    test = datasets["test_300x10"]
    ratios = {
        "nltcs_measured_l1": nltcs["measured_normalized_l1"][
            "vs_legacy"
        ],
        "nltcs_common_bin": nltcs["target_count_bins"][
            scientific.NLTCS_COMMON_BIN
        ]["vs_legacy"],
        "nltcs_rare_vs_legacy": nltcs["target_count_bins"][
            scientific.NLTCS_RARE_BIN
        ]["vs_legacy"],
        "nltcs_rare_vs_r8": nltcs["target_count_bins"][
            scientific.NLTCS_RARE_BIN
        ]["vs_r8"],
        "test_measured_l1": test["measured_normalized_l1"]["vs_legacy"],
        "one_way_by_dataset": {
            dataset: datasets[dataset]["offline_query_groups"][
                "one_way_safety"
            ]["vs_legacy"]
            for dataset in scientific.DATASET_ORDER
        },
    }
    classification = scientific.classify_screen(
        execution_valid=execution_valid,
        nltcs_measured_l1_ratio=_decision_ratio(
            ratios["nltcs_measured_l1"]
        ),
        nltcs_common_bin_ratio=_decision_ratio(
            ratios["nltcs_common_bin"]
        ),
        nltcs_rare_vs_legacy_ratio=_decision_ratio(
            ratios["nltcs_rare_vs_legacy"]
        ),
        nltcs_rare_vs_r8_ratio=_decision_ratio(
            ratios["nltcs_rare_vs_r8"]
        ),
        test_measured_l1_ratio=_decision_ratio(
            ratios["test_measured_l1"]
        ),
        one_way_ratio_by_dataset={
            dataset: _decision_ratio(record)
            for dataset, record in ratios["one_way_by_dataset"].items()
        },
    )
    gates = {
        "nltcs_measured_l1_vs_legacy": _gate_record(
            ratios["nltcs_measured_l1"],
            operator="strictly_less_than",
            threshold=1.0,
            execution_valid=execution_valid,
        ),
        "nltcs_common_bin_vs_legacy": _gate_record(
            ratios["nltcs_common_bin"],
            operator="strictly_less_than",
            threshold=1.0,
            execution_valid=execution_valid,
        ),
        "nltcs_rare_bin_vs_legacy": _gate_record(
            ratios["nltcs_rare_vs_legacy"],
            operator="less_than_or_equal",
            threshold=scientific.NLTCS_RARE_RATIO_MAX,
            execution_valid=execution_valid,
        ),
        "nltcs_rare_bin_vs_r8": _gate_record(
            ratios["nltcs_rare_vs_r8"],
            operator="strictly_less_than",
            threshold=1.0,
            execution_valid=execution_valid,
        ),
        "test_measured_l1_vs_legacy": _gate_record(
            ratios["test_measured_l1"],
            operator="less_than_or_equal",
            threshold=scientific.TEST_OVERALL_RATIO_MAX,
            execution_valid=execution_valid,
        ),
        "one_way_safety_vs_legacy": {
            dataset: _gate_record(
                record,
                operator="less_than_or_equal",
                threshold=scientific.ONE_WAY_RATIO_MAX,
                execution_valid=execution_valid,
            )
            for dataset, record in ratios["one_way_by_dataset"].items()
        },
    }
    return {
        "datasets": datasets,
        "screen_gate_evidence": gates,
        "candidate_execution_valid": bool(candidate_execution_valid),
        "baseline_execution_valid": bool(baseline_execution_valid),
        "execution_valid": execution_valid,
        "quality_interpretation_allowed": execution_valid,
        "frozen_classification": classification,
        "classification_precedence": scientific.frozen_protocol_manifest()[
            "screen_gates"
        ]["classification_order"],
        "all_six_gate_families_passed": (
            classification == "advance_to_fresh_seed_confirmation"
        ),
        "automatic_followup_authorized": False,
    }


def evaluate(
    confirmed_adapter_sha256: str,
    confirmed_collection_sha256: str,
) -> tuple[Path, Path]:
    # 确认门必须在任何 collection、基线或参考访问之前执行。
    protocol.require_evaluation_confirmation(
        confirmed_adapter_sha256,
        confirmed_collection_sha256,
    )
    root = _repo_root()
    protocol.assert_frozen_adapter_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "平方根评价要求包含未跟踪文件在内的干净工作树"
        )
    _assert_outputs_absent(root)
    collection_report, indexed = recovery_collection.load_collection(
        root,
        confirmed_collection_sha256,
    )

    # 从这里起才允许读取基线质量、候选终表/检查点和参考表。
    baseline_evaluation, baseline_cases, baseline_audit = (
        _load_frozen_baseline(root)
    )
    candidate_cases, identities = _evaluate_candidate_cases(root, indexed)
    if (
        identities
        != baseline_evaluation.get("query_and_reference_identity_audit")
    ):
        raise RuntimeError(
            "平方根候选与冻结基线的查询/参考身份不一致"
        )
    summary = _summary_and_decision(
        candidate_cases,
        baseline_cases,
        baseline_evaluation,
        baseline_audit,
    )
    recovery = collection_report["postcollection_recovery"]
    report = {
        **build_plan(),
        "mode": (
            "offline_evaluation_after_recovered_complete_sqrt_collection"
        ),
        "evaluation_started": True,
        "evaluation_commit": recovery_collection._git_text(
            root, "rev-parse", "HEAD"
        ),
        "evaluator_adapter_sha256": protocol.file_sha256(Path(__file__)),
        "candidate_collection_report_sha256": confirmed_collection_sha256,
        "candidate_collection_generation_commit": collection_report[
            "execution_commit"
        ],
        "candidate_collection_recovery_commit": recovery["recovery_commit"],
        "candidate_collection_recovered_after_generation": True,
        "candidate_execution_monitoring_evidence_complete": (
            collection_report["execution_monitoring_evidence_complete"]
        ),
        "candidate_monitoring_limitation_code": collection_report[
            "execution_monitoring_limitation_code"
        ],
        "baseline_evaluation_report_sha256": (
            protocol.BASELINE_EVALUATION_SHA256
        ),
        "baseline_independent_audit_sha256": protocol.BASELINE_AUDIT_SHA256,
        "baseline_frozen_classification": baseline_evaluation[
            "frozen_classification"
        ],
        "baseline_cases_recomputed": False,
        "baseline_metrics_reused_after_independent_audit": True,
        "query_and_reference_identity_audit": identities,
        "candidate_case_count": len(candidate_cases),
        "baseline_case_count": len(baseline_cases),
        "candidate_cases": candidate_cases,
        "summary": summary["datasets"],
        "screen_gate_evidence": summary["screen_gate_evidence"],
        "candidate_execution_valid": summary["candidate_execution_valid"],
        "baseline_execution_valid": summary["baseline_execution_valid"],
        "execution_valid": summary["execution_valid"],
        "quality_interpretation_allowed": summary[
            "quality_interpretation_allowed"
        ],
        "frozen_classification": summary["frozen_classification"],
        "classification_precedence": summary["classification_precedence"],
        "all_six_gate_families_passed": summary[
            "all_six_gate_families_passed"
        ],
        "automatic_followup_authorized": False,
        "raw_reference_data_accessed": True,
        "candidate_terminal_and_checkpoints_fully_evaluated": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
        "candidate_metrics_csv_only": True,
        "baseline_metrics_csv_reused_sha256": protocol.BASELINE_ARTIFACTS[
            "metrics_csv"
        ]["sha256"],
        "independent_audit_automatically_run": False,
    }
    with _source_evaluator_runtime():
        return source_evaluator._publish(
            root / protocol.OUTPUT_DIR,
            report,
            source_evaluator._csv_rows(candidate_cases),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    subparsers.add_parser("preflight")
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--confirm-adapter-sha", required=True)
    evaluate_parser.add_argument("--confirm-collection-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(recovery_collection._strict_json_text(build_plan()), end="")
        return
    if args.command == "preflight":
        print(recovery_collection._strict_json_text(preflight()), end="")
        return
    report, csv_path = evaluate(
        args.confirm_adapter_sha,
        args.confirm_collection_sha,
    )
    print(f"平方根恢复版筛查评价 -> {report}")
    print(f"平方根恢复版筛查指标 -> {csv_path}")
    print(f"evaluation SHA-256 -> {protocol.file_sha256(report)}")


if __name__ == "__main__":
    main()
