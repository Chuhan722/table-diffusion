#!/usr/bin/env python3
"""独立复核经勘误恢复的 Issue #53 平方根权重筛查评价。"""

from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scripts import audit_issue53_gap_weight_r8_screen as source_auditor
from scripts import issue53_gap_weight_sqrt_recovered_audit_protocol as audit_protocol
from scripts import issue53_gap_weight_sqrt_recovered_evaluation_protocol as evaluation_protocol
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific
from scripts import recover_issue53_gap_weight_sqrt_screen as recovery_collection


AUDIT_VERSION = audit_protocol.AUDIT_ADAPTER_VERSION
LEGACY_ARM = "gap_legacy_s8"
R8_ARM = "gap_bounded_r8_s8"
PRESENTATION_ARM_ORDER = (LEGACY_ARM, R8_ARM, evaluation_protocol.CANDIDATE_ARM)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def _source_auditor_runtime():
    """仅在调用域内把冻结独立 auditor 绑定到平方根两条候选。"""

    original_protocol = source_auditor.protocol
    source_auditor.protocol = evaluation_protocol
    try:
        yield
    finally:
        source_auditor.protocol = original_protocol


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
        "candidate_collection_report_sha256": (
            audit_protocol.SOURCE_COLLECTION_SHA256
        ),
        "candidate_evaluation_report_sha256": (
            audit_protocol.SOURCE_EVALUATION_SHA256
        ),
        "candidate_screen_metrics_csv_sha256": (
            audit_protocol.SOURCE_METRICS_CSV_SHA256
        ),
        "audited_baseline_artifacts": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in evaluation_protocol.BASELINE_ARTIFACTS.items()
        },
        "audit_report": str(audit_protocol.AUDIT_OUTPUT_PATH),
        "delegated_source_functions": list(
            audit_protocol.DELEGATED_SOURCE_FUNCTIONS
        ),
        "imports_primary_sqrt_evaluator_arithmetic": False,
        "imports_primary_r8_evaluator_arithmetic": False,
        "independently_recompute_candidate_terminal_and_checkpoints": True,
        "independently_recompute_candidate_bins_groups_and_csv": True,
        "independently_rebuild_dual_baseline_summary": True,
        "independently_rebuild_six_gates_and_classification": True,
        "baseline_cases_recomputed": False,
        "audit_started": False,
        "raw_reference_data_accessed": False,
        "candidate_results_accessed": False,
        "baseline_quality_values_accessed": False,
        "rerun_generation": False,
        "evaluation_artifacts_modified": False,
    }


def _assert_audit_output_absent(root: Path) -> None:
    destination = root / audit_protocol.AUDIT_OUTPUT_PATH
    if destination.exists():
        raise FileExistsError(
            "平方根独立审计报告已存在，不读取参考表且不覆盖："
            f"{destination}"
        )


def preflight() -> dict[str, Any]:
    """只检查冻结身份、干净工作树和输出不存在，不读取质量或参考表。"""

    root = _repo_root()
    plan = build_plan()
    if recovery_collection._git_text(
        root, "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError(
            "平方根独立审计预检要求包含未跟踪文件在内的干净工作树"
        )
    _assert_audit_output_absent(root)
    return {
        **plan,
        "mode": "read_only_preflight_no_quality_or_reference_access",
        "audit_commit": recovery_collection._git_text(
            root, "rev-parse", "HEAD"
        ),
        "worktree_clean_including_untracked": True,
        "audit_output_absent": True,
        "ready_for_explicit_audit_confirmation": True,
    }


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
            raise RuntimeError(f"平方根独立审计{name}无效")
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
            raise RuntimeError(f"平方根独立审计{label}祖先关系漂移")


def _load_audited_baseline(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """按哈希加载此前已独立复算通过的 legacy/R8 四条基线。"""

    evaluation_path = root / evaluation_protocol.BASELINE_EVALUATION_PATH
    audit_path = root / evaluation_protocol.BASELINE_AUDIT_PATH
    if (
        audit_protocol.file_sha256(evaluation_path)
        != evaluation_protocol.BASELINE_EVALUATION_SHA256
        or audit_protocol.file_sha256(audit_path)
        != evaluation_protocol.BASELINE_AUDIT_SHA256
    ):
        raise RuntimeError("平方根独立审计复用的基线哈希漂移")
    evaluation = source_auditor._load_json(evaluation_path)
    audit = source_auditor._load_json(audit_path)
    expected_csv = evaluation_protocol.BASELINE_ARTIFACTS["metrics_csv"]
    required_evaluation = {
        "contract_version": (
            "issue53-gap-weight-r8-recovered-evaluation-adapter-v1"
        ),
        "evaluation_adapter_protocol_sha256": (
            evaluation_protocol.BASELINE_EVALUATION_ADAPTER_SHA256
        ),
        "source_evaluator_sha256": evaluation_protocol.SOURCE_EVALUATOR_SHA256,
        "collection_report_sha256": evaluation_protocol.BASELINE_ARTIFACTS[
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
        raise RuntimeError("平方根独立审计复用的基线 evaluation 契约漂移")
    csv_meta = evaluation.get("screen_metrics_csv")
    if (
        not isinstance(csv_meta, dict)
        or csv_meta.get("path") != expected_csv["path"].name
        or csv_meta.get("sha256") != expected_csv["sha256"]
    ):
        raise RuntimeError("平方根独立审计复用的基线 CSV 绑定漂移")
    required_audit = {
        "contract_version": (
            "issue53-gap-weight-r8-recovered-independent-audit-v1"
        ),
        "audit_adapter_protocol_sha256": (
            evaluation_protocol.BASELINE_AUDIT_ADAPTER_SHA256
        ),
        "evaluation_report_sha256": (
            evaluation_protocol.BASELINE_EVALUATION_SHA256
        ),
        "pass": True,
        "classification_independently_reproduced": True,
        "summary_independently_reproduced": True,
        "evaluator_arithmetic_imported": False,
        "generation_rerun": False,
        "evaluation_artifacts_modified": False,
    }
    if any(audit.get(key) != value for key, value in required_audit.items()):
        raise RuntimeError("平方根独立审计复用的基线 audit 契约漂移")
    cases = evaluation.get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise RuntimeError("平方根独立审计复用的基线不是完整四条")
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
        raise RuntimeError("平方根独立审计复用的基线地址漂移")
    identities = evaluation.get("query_and_reference_identity_audit")
    if not isinstance(identities, dict):
        raise TypeError("平方根独立审计复用的基线查询/参考身份缺失")
    return evaluation, cases, audit


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
        raise RuntimeError(f"平方根独立审计 {dataset}/{arm} case 不唯一")
    return selected[0]


def _comparison_independent(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    dataset: str,
    path: str,
    baseline_arm: str,
) -> dict[str, Any]:
    candidate = source_auditor._nested(
        _case(candidate_cases, dataset, evaluation_protocol.CANDIDATE_ARM),
        path,
    )
    baseline = source_auditor._nested(
        _case(baseline_cases, dataset, baseline_arm), path
    )
    independent = source_auditor._independent_ratio(candidate, baseline)
    return {
        "metric": path,
        "candidate_arm": evaluation_protocol.CANDIDATE_ARM,
        "baseline_arm": baseline_arm,
        "candidate_value": independent["bounded_r8"],
        "baseline_value": independent["legacy"],
        "candidate_over_baseline": independent["bounded_over_legacy"],
        "ratio_state": independent["ratio_state"],
        "lower_is_better": independent["lower_is_better"],
    }


def _both_baselines_independent(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    dataset: str,
    path: str,
) -> dict[str, Any]:
    return {
        "vs_legacy": _comparison_independent(
            candidate_cases, baseline_cases, dataset, path, LEGACY_ARM
        ),
        "vs_r8": _comparison_independent(
            candidate_cases, baseline_cases, dataset, path, R8_ARM
        ),
    }


def _decision_ratio_independent(record: Mapping[str, Any]) -> float:
    return source_auditor._ratio_value(
        {
            "ratio_state": record["ratio_state"],
            "bounded_over_legacy": record["candidate_over_baseline"],
        }
    )


def _checkpoint_curves_independent(
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


def _gate_independent(
    comparison: Mapping[str, Any],
    *,
    operator: str,
    threshold: float,
    execution_valid: bool,
) -> dict[str, Any]:
    ratio = _decision_ratio_independent(comparison)
    if operator == "strictly_less_than":
        passed = ratio < threshold
    elif operator == "less_than_or_equal":
        passed = ratio <= threshold
    else:
        raise ValueError(f"未知平方根独立审计门操作符：{operator}")
    return {
        "comparison": dict(comparison),
        "operator": operator,
        "threshold": float(threshold),
        "pass": bool(passed) if execution_valid else None,
        "interpreted": bool(execution_valid),
    }


def _summary_independent(
    candidate_cases: Sequence[dict[str, Any]],
    baseline_cases: Sequence[dict[str, Any]],
    baseline_evaluation: Mapping[str, Any],
    baseline_audit: Mapping[str, Any],
) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for dataset in scientific.DATASET_ORDER:
        measured = _both_baselines_independent(
            candidate_cases,
            baseline_cases,
            dataset,
            "metrics.measured.overall.normalized_l1_mean",
        )
        proxies = {
            name: _both_baselines_independent(
                candidate_cases,
                baseline_cases,
                dataset,
                f"metrics.measured.proxy_objectives.{name}",
            )
            for name in ("legacy_relative_gap", "bounded_r8_gap")
        }
        target_bins = {
            item["name"]: _both_baselines_independent(
                candidate_cases,
                baseline_cases,
                dataset,
                "metrics.measured.target_count_bins."
                f"{item['name']}.absolute_count_error_mean",
            )
            for item in scientific.TARGET_COUNT_BINS[dataset]
        }
        orders = {
            str(order): _both_baselines_independent(
                candidate_cases,
                baseline_cases,
                dataset,
                f"metrics.measured.by_order.{order}.absolute_count_error_mean",
            )
            for order in scientific.DATASETS[dataset]["order_counts"]
        }
        group_names = (
            evaluation_protocol.TEST_GROUP_ORDER
            if dataset == "test_300x10"
            else tuple(evaluation_protocol.NLTCS_GROUP_COUNTS)
        )
        offline_groups = {
            name: _both_baselines_independent(
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
            "checkpoint_curves": _checkpoint_curves_independent(
                candidate_cases, baseline_cases, dataset
            ),
            "terminal_cost_by_arm": {
                arm: _case(combined, dataset, arm)["cost"]
                for arm in PRESENTATION_ARM_ORDER
            },
        }

    candidate_execution_valid = all(
        source_auditor._nested(case, "metrics.validity.valid_row_rate") == 1.0
        and case["kernel_audit"]["gap_weighting_guard_passed"] is True
        for case in candidate_cases
    )
    baseline_execution_valid = (
        baseline_evaluation.get("execution_valid") is True
        and baseline_audit.get("pass") is True
        and all(
            source_auditor._nested(
                case, "metrics.validity.valid_row_rate"
            )
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
        "nltcs_measured_l1": nltcs["measured_normalized_l1"]["vs_legacy"],
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
    decision_ratios = {
        name: _decision_ratio_independent(record)
        for name, record in ratios.items()
        if name != "one_way_by_dataset"
    }
    one_way_ratios = {
        dataset: _decision_ratio_independent(record)
        for dataset, record in ratios["one_way_by_dataset"].items()
    }

    # 独立展开结果前冻结的优先级；不调用 scientific.classify_screen。
    if not execution_valid:
        classification = "execution_invalid"
    elif (
        decision_ratios["nltcs_measured_l1"] >= 1.0
        or decision_ratios["nltcs_common_bin"] >= 1.0
    ):
        classification = "common_mechanism_not_retained"
    elif (
        decision_ratios["nltcs_rare_vs_legacy"]
        > scientific.NLTCS_RARE_RATIO_MAX
        or decision_ratios["nltcs_rare_vs_r8"] >= 1.0
    ):
        classification = "rare_query_protection_not_recovered"
    elif (
        decision_ratios["test_measured_l1"]
        > scientific.TEST_OVERALL_RATIO_MAX
        or any(
            value > scientific.ONE_WAY_RATIO_MAX
            for value in one_way_ratios.values()
        )
    ):
        classification = "measured_or_one_way_safety_risk"
    else:
        classification = "advance_to_fresh_seed_confirmation"

    gates = {
        "nltcs_measured_l1_vs_legacy": _gate_independent(
            ratios["nltcs_measured_l1"],
            operator="strictly_less_than",
            threshold=1.0,
            execution_valid=execution_valid,
        ),
        "nltcs_common_bin_vs_legacy": _gate_independent(
            ratios["nltcs_common_bin"],
            operator="strictly_less_than",
            threshold=1.0,
            execution_valid=execution_valid,
        ),
        "nltcs_rare_bin_vs_legacy": _gate_independent(
            ratios["nltcs_rare_vs_legacy"],
            operator="less_than_or_equal",
            threshold=scientific.NLTCS_RARE_RATIO_MAX,
            execution_valid=execution_valid,
        ),
        "nltcs_rare_bin_vs_r8": _gate_independent(
            ratios["nltcs_rare_vs_r8"],
            operator="strictly_less_than",
            threshold=1.0,
            execution_valid=execution_valid,
        ),
        "test_measured_l1_vs_legacy": _gate_independent(
            ratios["test_measured_l1"],
            operator="less_than_or_equal",
            threshold=scientific.TEST_OVERALL_RATIO_MAX,
            execution_valid=execution_valid,
        ),
        "one_way_safety_vs_legacy": {
            dataset: _gate_independent(
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


def _audit_recovered_evaluation(
    root: Path,
    confirmed_evaluation_sha256: str,
    confirmed_collection_sha256: str,
    collection_report: dict[str, Any],
    candidate_cases: list[dict[str, Any]],
    identities: dict[str, Any],
    baseline_evaluation: dict[str, Any],
    baseline_cases: list[dict[str, Any]],
    baseline_audit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_auditor._require_sha(
        confirmed_evaluation_sha256,
        "平方根 evaluation SHA-256",
    )
    path = root / audit_protocol.SOURCE_EVALUATION_PATH
    if audit_protocol.file_sha256(path) != confirmed_evaluation_sha256:
        raise ValueError("平方根独立审计 evaluation SHA-256 不一致")
    report = source_auditor._load_json(path)
    recovery = collection_report["postcollection_recovery"]
    required = {
        "contract_version": evaluation_protocol.ADAPTER_VERSION,
        "mode": (
            "offline_evaluation_after_recovered_complete_sqrt_collection"
        ),
        "evaluation_adapter_protocol_sha256": (
            evaluation_protocol.FROZEN_ADAPTER_SHA256
        ),
        "source_scientific_protocol_sha256": scientific.FROZEN_PROTOCOL_SHA256,
        "source_evaluator_version": evaluation_protocol.SOURCE_EVALUATOR_VERSION,
        "source_evaluator_sha256": evaluation_protocol.SOURCE_EVALUATOR_SHA256,
        "source_recovery_protocol_sha256": (
            evaluation_protocol.SOURCE_RECOVERY_PROTOCOL_SHA256
        ),
        "candidate_collection_report": str(
            evaluation_protocol.SOURCE_COLLECTION_PATH
        ),
        "candidate_collection_report_sha256": confirmed_collection_sha256,
        "baseline_artifacts": {
            name: {"path": str(item["path"]), "sha256": item["sha256"]}
            for name, item in evaluation_protocol.BASELINE_ARTIFACTS.items()
        },
        "evaluation_report": str(audit_protocol.SOURCE_EVALUATION_PATH),
        "delegated_source_functions": list(
            evaluation_protocol.DELEGATED_SOURCE_FUNCTIONS
        ),
        "candidate_metric_arithmetic_modified": False,
        "ratio_zero_denominator_rule_modified": False,
        "classification_source": "frozen_sqrt_scientific_protocol",
        "screen_gates": scientific.frozen_protocol_manifest()["screen_gates"],
        "baseline_cases_recomputed": False,
        "new_generation_allowed": False,
        "generation_started": False,
        "evaluation_started": True,
        "evaluator_adapter_sha256": evaluation_protocol.IMPLEMENTATION_SOURCES[
            "evaluation_adapter"
        ]["sha256"],
        "candidate_collection_generation_commit": collection_report[
            "execution_commit"
        ],
        "candidate_collection_recovery_commit": recovery["recovery_commit"],
        "candidate_collection_recovered_after_generation": True,
        "candidate_execution_monitoring_evidence_complete": False,
        "candidate_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "baseline_evaluation_report_sha256": (
            evaluation_protocol.BASELINE_EVALUATION_SHA256
        ),
        "baseline_independent_audit_sha256": (
            evaluation_protocol.BASELINE_AUDIT_SHA256
        ),
        "baseline_frozen_classification": baseline_evaluation[
            "frozen_classification"
        ],
        "baseline_metrics_reused_after_independent_audit": True,
        "candidate_case_count": 2,
        "baseline_case_count": 4,
        "automatic_followup_authorized": False,
        "raw_reference_data_accessed": True,
        "candidate_terminal_and_checkpoints_fully_evaluated": True,
        "historical_best_used": False,
        "checkpoint_selected_as_output": False,
        "new_generation_performed_by_evaluator": False,
        "parameter_retuning_performed": False,
        "privacy_budget_consumed": False,
        "candidate_metrics_csv_only": True,
        "baseline_metrics_csv_reused_sha256": (
            evaluation_protocol.BASELINE_ARTIFACTS["metrics_csv"]["sha256"]
        ),
        "independent_audit_automatically_run": False,
    }
    if any(report.get(key) != value for key, value in required.items()):
        raise RuntimeError("平方根独立审计 evaluation 顶层契约漂移")
    evaluation_commit = report.get("evaluation_commit")
    if evaluation_commit != audit_protocol.SOURCE_EVALUATION_COMMIT:
        raise RuntimeError("平方根独立审计评价提交漂移")
    _validate_commit_ancestry(
        root,
        collection_report["execution_commit"],
        recovery["recovery_commit"],
        evaluation_commit,
    )
    if report.get("query_and_reference_identity_audit") != identities:
        raise RuntimeError("平方根独立审计查询/参考身份漂移")
    source_auditor._assert_equal(
        report.get("candidate_cases"), candidate_cases, "candidate_cases"
    )
    summary = _summary_independent(
        candidate_cases,
        baseline_cases,
        baseline_evaluation,
        baseline_audit,
    )
    source_auditor._assert_equal(
        report.get("summary"), summary["datasets"], "summary"
    )
    source_auditor._assert_equal(
        report.get("screen_gate_evidence"),
        summary["screen_gate_evidence"],
        "screen_gate_evidence",
    )
    for key in (
        "candidate_execution_valid",
        "baseline_execution_valid",
        "execution_valid",
        "quality_interpretation_allowed",
        "frozen_classification",
        "classification_precedence",
        "all_six_gate_families_passed",
        "automatic_followup_authorized",
    ):
        source_auditor._assert_equal(report.get(key), summary[key], key)
    with _source_auditor_runtime():
        csv_rows = source_auditor._csv_rows_independent(candidate_cases)
        csv_audit = source_auditor._audit_csv(root, report, csv_rows)
    return report, {
        "candidate_evaluation_report_sha256": confirmed_evaluation_sha256,
        "all_2_candidate_case_metrics_recomputed": True,
        "all_candidate_terminal_and_checkpoint_metrics_recomputed": True,
        "both_candidate_proxy_objectives_recomputed": True,
        "all_candidate_target_count_bins_recomputed": True,
        "all_candidate_query_orders_recomputed": True,
        "all_candidate_offline_groups_recomputed": True,
        "audited_baseline_hashes_and_contract_verified": True,
        "dual_baseline_summary_independently_reproduced": True,
        "all_six_gate_records_independently_reproduced": True,
        "classification_independently_reproduced": True,
        "verified_frozen_classification": summary["frozen_classification"],
        "candidate_screen_metrics_csv": csv_audit,
        "pass": True,
    }


def _publish_audit(root: Path, payload: dict[str, Any]) -> Path:
    destination = root / audit_protocol.AUDIT_OUTPUT_PATH
    if destination.exists():
        raise FileExistsError("平方根独立审计报告已存在，不覆盖")
    temporary = Path(
        tempfile.mkdtemp(
            prefix=".sqrt-recovered-screen-audit.tmp-",
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
    # 确认门必须先于工作树、collection、基线、候选结果与参考表访问。
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
            "平方根独立审计要求包含未跟踪文件在内的干净工作树"
        )
    _assert_audit_output_absent(root)
    collection_report, indexed = recovery_collection.load_collection(
        root,
        confirmed_collection_sha256,
    )
    baseline_evaluation, baseline_cases, baseline_audit = (
        _load_audited_baseline(root)
    )

    # 从这里起才读取候选终表、检查点和参考表；不导入主 evaluator。
    with _source_auditor_runtime():
        candidate_cases, identities = source_auditor._recompute_cases(
            root, indexed
        )
    expected_task_ids = [
        task.task_id for task in evaluation_protocol.task_plan().tasks
    ]
    if (
        [case.get("task_id") for case in candidate_cases] != expected_task_ids
        or len(candidate_cases) != 2
        or any(
            case.get("arm") != evaluation_protocol.CANDIDATE_ARM
            for case in candidate_cases
        )
    ):
        raise RuntimeError("平方根独立审计候选 case 身份漂移")
    if (
        identities
        != baseline_evaluation.get("query_and_reference_identity_audit")
    ):
        raise RuntimeError("平方根独立审计候选与基线查询/参考身份不一致")
    evaluation_report, evidence = _audit_recovered_evaluation(
        root,
        confirmed_evaluation_sha256,
        confirmed_collection_sha256,
        collection_report,
        candidate_cases,
        identities,
        baseline_evaluation,
        baseline_cases,
        baseline_audit,
    )
    payload = {
        **build_plan(),
        "mode": "independent_recomputation_after_recovered_sqrt_evaluation",
        "audit_started": True,
        "audit_commit": recovery_collection._git_text(root, "rev-parse", "HEAD"),
        "recovered_auditor_sha256": audit_protocol.file_sha256(Path(__file__)),
        "candidate_collection_report_sha256": confirmed_collection_sha256,
        "candidate_evaluation_report_sha256": confirmed_evaluation_sha256,
        "evaluation_commit": evaluation_report["evaluation_commit"],
        **evidence,
        "raw_reference_data_accessed": True,
        "candidate_results_accessed": True,
        "baseline_quality_values_accessed": True,
        "candidate_execution_monitoring_evidence_complete": False,
        "candidate_monitoring_limitation_code": (
            "runner_samples_lost_before_shard_report_write"
        ),
        "evaluator_arithmetic_imported": False,
        "source_independent_auditor_reused": True,
        "baseline_cases_recomputed": False,
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
    subparsers.add_parser("preflight")
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
    if args.command == "preflight":
        print(recovery_collection._strict_json_text(preflight()), end="")
        return
    path = audit(
        args.confirm_audit_adapter_sha,
        args.confirm_collection_sha,
        args.confirm_evaluation_sha,
    )
    print(f"平方根恢复版独立审计 -> {path}")
    print(f"audit SHA-256 -> {audit_protocol.file_sha256(path)}")


if __name__ == "__main__":
    main()
