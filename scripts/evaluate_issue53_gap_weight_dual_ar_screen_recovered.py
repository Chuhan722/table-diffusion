#!/usr/bin/env python3
"""离线评价经勘误恢复的 Issue #53 A/R 双通道两轨迹 collection。"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import evaluate_issue53_gap_weight_r8_screen as source_evaluator
from scripts import evaluate_issue53_gap_weight_sqrt_screen_recovered as sqrt_evaluator
from scripts import issue53_gap_weight_dual_ar_recovered_evaluation_protocol as protocol
from scripts import issue53_gap_weight_dual_ar_screen_protocol as scientific
from scripts import recover_issue53_gap_weight_dual_ar_screen as recovery_collection


EVALUATION_VERSION = protocol.ADAPTER_VERSION
LEGACY_ARM = "gap_legacy_s8"
R8_ARM = "gap_bounded_r8_s8"
SQRT_ARM = "gap_sqrt_target_s8"
PRESENTATION_ARM_ORDER = (
    LEGACY_ARM,
    R8_ARM,
    SQRT_ARM,
    protocol.CANDIDATE_ARM,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _source_evaluator_runtime():
    """把冻结 R8 指标算术临时绑定到 A/R 两 case。"""

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
        "source_scientific_protocol_sha256": scientific.FROZEN_PROTOCOL_SHA256,
        "source_evaluator_version": protocol.SOURCE_EVALUATOR_VERSION,
        "source_evaluator_sha256": protocol.SOURCE_EVALUATOR_SHA256,
        "source_recovery_protocol_sha256": protocol.SOURCE_RECOVERY_PROTOCOL_SHA256,
        "candidate_collection_report": str(protocol.SOURCE_COLLECTION_PATH),
        "candidate_collection_report_sha256": protocol.SOURCE_COLLECTION_SHA256,
        "baseline_artifacts": {
            family: {
                name: {"path": str(item["path"]), "sha256": item["sha256"]}
                for name, item in artifacts.items()
            }
            for family, artifacts in protocol.BASELINE_ARTIFACTS.items()
        },
        "evaluation_report": str(protocol.OUTPUT_DIR / protocol.EVALUATION_REPORT),
        "screen_metrics_csv": str(protocol.OUTPUT_DIR / protocol.L1_RESULTS_CSV),
        "candidate_metric_arithmetic_modified": False,
        "checkpoint_gap_e_legacy_diagnostic_preserved": True,
        "ratio_zero_denominator_rule_modified": False,
        "classification_source": "frozen_dual_ar_scientific_protocol",
        "screen_gates": scientific.frozen_protocol_manifest()["screen_gates"],
        "baseline_cases_recomputed": False,
        "new_generation_allowed": False,
        "generation_started": False,
        "raw_reference_data_accessed": False,
        "candidate_quality_accessed": False,
        "baseline_quality_values_accessed": False,
        "evaluation_started": False,
        "explicit_user_evaluation_authorization_received": True,
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
            "A/R 评价产物已存在，不读取参考表且不覆盖："
            + ", ".join(str(path) for path in existing)
        )


def preflight() -> dict[str, Any]:
    root = _repo_root()
    plan = build_plan()
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("A/R 评价预检要求包含未跟踪文件在内的干净工作树")
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


def _load_sqrt_baseline(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    evaluation_spec = protocol.SQRT_BASELINE["evaluation"]
    audit_spec = protocol.SQRT_BASELINE["independent_audit"]
    evaluation = recovery_collection._load_json(root / evaluation_spec["path"])
    audit = recovery_collection._load_json(root / audit_spec["path"])
    required_evaluation = {
        "contract_version": "issue53-gap-weight-sqrt-target-recovered-evaluation-adapter-v1",
        "evaluation_adapter_protocol_sha256": protocol.SQRT_EVALUATION_ADAPTER_SHA256,
        "candidate_collection_report_sha256": protocol.SQRT_BASELINE["collection"]["sha256"],
        "candidate_case_count": 2,
        "execution_valid": True,
        "quality_interpretation_allowed": True,
        "candidate_terminal_and_checkpoints_fully_evaluated": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
    }
    if any(evaluation.get(key) != value for key, value in required_evaluation.items()):
        raise RuntimeError("A/R 评价复用的 sqrt evaluation 契约漂移")
    csv_meta = evaluation.get("screen_metrics_csv")
    if (
        not isinstance(csv_meta, dict)
        or csv_meta.get("path") != protocol.SQRT_BASELINE["metrics_csv"]["path"].name
        or csv_meta.get("sha256") != protocol.SQRT_BASELINE["metrics_csv"]["sha256"]
    ):
        raise RuntimeError("A/R 评价复用的 sqrt CSV 绑定漂移")
    required_audit = {
        "contract_version": "issue53-gap-weight-sqrt-target-recovered-independent-audit-v1",
        "audit_adapter_protocol_sha256": protocol.SQRT_AUDIT_ADAPTER_SHA256,
        "candidate_evaluation_report_sha256": evaluation_spec["sha256"],
        "pass": True,
        "classification_independently_reproduced": True,
        "dual_baseline_summary_independently_reproduced": True,
        "evaluator_arithmetic_imported": False,
        "generation_rerun": False,
        "evaluation_artifacts_modified": False,
    }
    if any(audit.get(key) != value for key, value in required_audit.items()):
        raise RuntimeError("A/R 评价复用的 sqrt 独立审计契约漂移")
    cases = evaluation.get("candidate_cases")
    if not isinstance(cases, list) or len(cases) != 2:
        raise RuntimeError("A/R 评价复用的 sqrt 基线不是完整两条")
    expected_addresses = {
        (dataset, SQRT_ARM, scientific.DEVELOPMENT_SEED)
        for dataset in scientific.DATASET_ORDER
    }
    addresses = {
        (case.get("dataset"), case.get("arm"), case.get("seed"))
        for case in cases
        if isinstance(case, dict)
    }
    if addresses != expected_addresses or len(addresses) != len(cases):
        raise RuntimeError("A/R 评价复用的 sqrt case 地址漂移")
    identities = evaluation.get("query_and_reference_identity_audit")
    if not isinstance(identities, dict):
        raise TypeError("A/R 评价复用的 sqrt 查询/参考身份缺失")
    return evaluation, cases, audit


def _load_frozen_baselines(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    r8_evaluation, r8_cases, r8_audit = sqrt_evaluator._load_frozen_baseline(root)
    sqrt_evaluation, sqrt_cases, sqrt_audit = _load_sqrt_baseline(root)
    r8_identities = r8_evaluation.get("query_and_reference_identity_audit")
    sqrt_identities = sqrt_evaluation.get("query_and_reference_identity_audit")
    if r8_identities != sqrt_identities:
        raise RuntimeError("A/R 评价复用的 R8 与 sqrt 查询/参考身份不一致")
    return (
        [*r8_cases, *sqrt_cases],
        r8_identities,
        {
            "r8_evaluation": r8_evaluation,
            "r8_audit": r8_audit,
            "sqrt_evaluation": sqrt_evaluation,
            "sqrt_audit": sqrt_audit,
        },
    )


def _candidate_proxy_rows(
    indexed: Mapping[tuple[int, str, str], dict[str, Any]],
) -> dict[tuple[int, str, str], dict[str, Any]]:
    proxies: dict[tuple[int, str, str], dict[str, Any]] = {}
    for key, row in indexed.items():
        proxy = dict(row)
        proxy["gap_actual_weight_ratio_min"] = row[
            "gap_relative_positive_weight_ratio_min"
        ]
        proxy["gap_actual_weight_ratio_max"] = row[
            "gap_relative_positive_weight_ratio_max"
        ]
        proxies[key] = proxy
    return proxies


def _evaluate_candidate_cases(
    root: Path,
    indexed: dict[tuple[int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with _source_evaluator_runtime():
        cases, identities = source_evaluator._evaluate_cases(
            root, _candidate_proxy_rows(indexed)
        )
    expected = [task.task_id for task in protocol.task_plan().tasks]
    if (
        [case.get("task_id") for case in cases] != expected
        or len(cases) != 2
        or any(case.get("arm") != protocol.CANDIDATE_ARM for case in cases)
    ):
        raise RuntimeError("A/R 评价候选 case 身份漂移")
    for case in cases:
        row = indexed[(case["seed"], case["dataset"], case["arm"])]
        case["kernel_audit"] = {
            key: row[key]
            for key in (
                "gap_weighting",
                "gap_channel_aggregation",
                "gap_zero_target_policy",
                "gap_floor_applied",
                "gap_max_weight_ratio",
                "gap_smoothing_count",
                "gap_expected_positive_target_query_count",
                "gap_expected_relative_inverse_target_normalizer",
                "gap_expected_relative_positive_weight_ratio",
                "gap_relative_inverse_target_normalizer_min",
                "gap_relative_inverse_target_normalizer_max",
                "gap_relative_positive_weight_ratio_min",
                "gap_relative_positive_weight_ratio_max",
                "gap_weighting_scan_round_count",
                "gap_weighting_guard_passed",
            )
        }
    return cases, identities


def _case(cases: Sequence[dict[str, Any]], dataset: str, arm: str) -> dict[str, Any]:
    selected = [
        case
        for case in cases
        if case.get("dataset") == dataset and case.get("arm") == arm
    ]
    if len(selected) != 1 or selected[0].get("seed") != scientific.DEVELOPMENT_SEED:
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
    record = source_evaluator._ratio_record(candidate, baseline_value)
    return {
        "metric": path,
        "candidate_arm": protocol.CANDIDATE_ARM,
        "baseline_arm": baseline_arm,
        "candidate_value": record["bounded_r8"],
        "baseline_value": record["legacy"],
        "candidate_over_baseline": record["bounded_over_legacy"],
        "ratio_state": record["ratio_state"],
        "lower_is_better": record["lower_is_better"],
    }


def _all_baselines(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    dataset: str,
    path: str,
) -> dict[str, Any]:
    return {
        "vs_legacy": _comparison(
            candidate_cases, baseline_cases, dataset, path, LEGACY_ARM
        ),
        "vs_r8": _comparison(candidate_cases, baseline_cases, dataset, path, R8_ARM),
        "vs_sqrt": _comparison(
            candidate_cases, baseline_cases, dataset, path, SQRT_ARM
        ),
    }


def _decision_ratio(record: Mapping[str, Any]) -> float:
    return source_evaluator._decision_ratio(
        {
            "ratio_state": record["ratio_state"],
            "bounded_over_legacy": record["candidate_over_baseline"],
        }
    )


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
        raise ValueError(f"未知 A/R 筛查门操作符：{operator}")
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
    baseline_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in scientific.DATASET_ORDER:
        measured = _all_baselines(
            candidate_cases,
            baseline_cases,
            dataset,
            "metrics.measured.overall.normalized_l1_mean",
        )
        proxies = {
            name: _all_baselines(
                candidate_cases,
                baseline_cases,
                dataset,
                f"metrics.measured.proxy_objectives.{name}",
            )
            for name in ("legacy_relative_gap", "bounded_r8_gap")
        }
        target_bins = {
            item["name"]: _all_baselines(
                candidate_cases,
                baseline_cases,
                dataset,
                "metrics.measured.target_count_bins."
                f"{item['name']}.absolute_count_error_mean",
            )
            for item in scientific.TARGET_COUNT_BINS[dataset]
        }
        orders = {
            str(order): _all_baselines(
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
            name: _all_baselines(
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
        source_evaluator._nested(case, "metrics.validity.valid_row_rate") == 1.0
        and case["kernel_audit"]["gap_weighting_guard_passed"] is True
        and case["kernel_audit"]["gap_weighting"] == protocol.DUAL_WEIGHTING
        for case in candidate_cases
    )
    r8_evaluation = baseline_provenance["r8_evaluation"]
    r8_audit = baseline_provenance["r8_audit"]
    sqrt_evaluation = baseline_provenance["sqrt_evaluation"]
    sqrt_audit = baseline_provenance["sqrt_audit"]
    baseline_execution_valid = (
        r8_evaluation.get("execution_valid") is True
        and r8_audit.get("pass") is True
        and sqrt_evaluation.get("execution_valid") is True
        and sqrt_audit.get("pass") is True
        and all(
            source_evaluator._nested(case, "metrics.validity.valid_row_rate")
            == 1.0
            and case["kernel_audit"]["gap_weighting_guard_passed"] is True
            for case in baseline_cases
        )
    )
    execution_valid = bool(candidate_execution_valid and baseline_execution_valid)
    nltcs = datasets["nltcs"]
    test = datasets["test_300x10"]
    ratios = {
        "nltcs_measured_l1": nltcs["measured_normalized_l1"]["vs_legacy"],
        "nltcs_common_bin": nltcs["target_count_bins"][
            scientific.NLTCS_COMMON_BIN
        ]["vs_legacy"],
        "nltcs_rare_vs_legacy": nltcs["target_count_bins"][
            scientific.NLTCS_RARE_BIN
        ]["vs_legacy"],
        "nltcs_rare_vs_sqrt": nltcs["target_count_bins"][
            scientific.NLTCS_RARE_BIN
        ]["vs_sqrt"],
        "test_measured_l1": test["measured_normalized_l1"]["vs_legacy"],
        "one_way_by_dataset": {
            dataset: datasets[dataset]["offline_query_groups"]["one_way_safety"][
                "vs_legacy"
            ]
            for dataset in scientific.DATASET_ORDER
        },
    }
    classification = scientific.classify_screen(
        execution_valid=execution_valid,
        nltcs_measured_l1_ratio=_decision_ratio(ratios["nltcs_measured_l1"]),
        nltcs_common_bin_ratio=_decision_ratio(ratios["nltcs_common_bin"]),
        nltcs_rare_vs_legacy_ratio=_decision_ratio(
            ratios["nltcs_rare_vs_legacy"]
        ),
        nltcs_rare_vs_sqrt_ratio=_decision_ratio(ratios["nltcs_rare_vs_sqrt"]),
        test_measured_l1_ratio=_decision_ratio(ratios["test_measured_l1"]),
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
        "nltcs_rare_bin_vs_sqrt": _gate_record(
            ratios["nltcs_rare_vs_sqrt"],
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
    protocol.require_evaluation_confirmation(
        confirmed_adapter_sha256, confirmed_collection_sha256
    )
    root = _repo_root()
    protocol.assert_frozen_adapter_identity(root)
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("A/R 评价要求包含未跟踪文件在内的干净工作树")
    _assert_outputs_absent(root)
    collection_report, indexed = recovery_collection.load_collection(
        root, confirmed_collection_sha256
    )

    baseline_cases, baseline_identities, provenance = _load_frozen_baselines(root)
    candidate_cases, identities = _evaluate_candidate_cases(root, indexed)
    if identities != baseline_identities:
        raise RuntimeError("A/R 候选与冻结基线的查询/参考身份不一致")
    summary = _summary_and_decision(candidate_cases, baseline_cases, provenance)
    recovery = collection_report["postcollection_recovery"]
    report = {
        **build_plan(),
        "mode": "offline_evaluation_after_recovered_complete_dual_ar_collection",
        "evaluation_started": True,
        "evaluation_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "evaluator_adapter_sha256": protocol.file_sha256(Path(__file__)),
        "candidate_collection_report_sha256": confirmed_collection_sha256,
        "candidate_collection_generation_commit": collection_report[
            "execution_commit"
        ],
        "candidate_collection_recovery_commit": recovery["recovery_commit"],
        "candidate_collection_recovered_after_generation": True,
        "candidate_source_case_files_rewritten": False,
        "baseline_artifact_sha256": {
            family: {name: item["sha256"] for name, item in artifacts.items()}
            for family, artifacts in protocol.BASELINE_ARTIFACTS.items()
        },
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
    run = subparsers.add_parser("evaluate")
    run.add_argument("--confirm-adapter-sha", required=True)
    run.add_argument("--confirm-collection-sha", required=True)
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
        args.confirm_adapter_sha, args.confirm_collection_sha
    )
    print(f"A/R 恢复版筛查评价 -> {report}")
    print(f"A/R 恢复版筛查指标 -> {csv_path}")
    print(f"evaluation SHA-256 -> {protocol.file_sha256(report)}")


if __name__ == "__main__":
    main()
