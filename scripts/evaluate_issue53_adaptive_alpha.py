#!/usr/bin/env python3
"""离线评价 Issue #53 两档自适应 alpha 的冻结采集。"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import evaluate_issue53_fixed_alpha_calibration as fixed_evaluation
    from scripts import run_issue53_adaptive_alpha as collection
else:
    import evaluate_issue53_fixed_alpha_calibration as fixed_evaluation
    import run_issue53_adaptive_alpha as collection


EVALUATION_VERSION = "issue53-adaptive-alpha-evaluation-v1"
EVALUATION_REPORT = "evaluation_report.json"
RISK_RATIO_MAX = 1.05
DIVERSITY_RATIO_MIN = 0.95
WORK_RATIO_MAX = 1.05
STABLE_WIN_MINIMUM = 4
NORMAL_REASONS = {"fit_target_reached", "early_stopped"}

REFERENCE_PATHS = fixed_evaluation.REFERENCE_PATHS
REFERENCE_SHA256 = fixed_evaluation.REFERENCE_SHA256
TEST_GROUP_ORDER = fixed_evaluation.TEST_GROUP_ORDER
TEST_GROUP_COUNTS = fixed_evaluation.TEST_GROUP_COUNTS
TEST_GROUP_IDENTITIES = fixed_evaluation.TEST_GROUP_IDENTITIES
NLTCS_GROUP_COUNTS = fixed_evaluation.NLTCS_GROUP_COUNTS
NLTCS_GROUP_IDENTITIES = fixed_evaluation.NLTCS_GROUP_IDENTITIES


def build_plan() -> dict[str, Any]:
    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_or_reference_read",
        "collection_protocol_sha256": collection.FROZEN_PROTOCOL_SHA256,
        "collection_report": str(
            collection.OUTPUT_DIR / collection.COLLECTION_REPORT
        ),
        "evaluation_report": str(collection.OUTPUT_DIR / EVALUATION_REPORT),
        "reference_sha256": REFERENCE_SHA256,
        "test_groups": {
            "counts": TEST_GROUP_COUNTS,
            "identity_sha256": TEST_GROUP_IDENTITIES,
        },
        "nltcs_groups": {
            "counts": NLTCS_GROUP_COUNTS,
            "identity_sha256": NLTCS_GROUP_IDENTITIES,
        },
        "primary_candidate_arm": collection.ARM_ADAPTIVE,
        "primary_baseline_arm": collection.ARM_FIXED_16,
        "mechanism_control_arm": collection.ARM_FIXED_12,
        "stable_measured_gain": {
            "candidate_mean_lt_fixed16_mean": True,
            "paired_seed_wins_minimum": STABLE_WIN_MINIMUM,
            "paired_seed_count": len(collection.SEEDS),
        },
        "offline_risk_ratio_max": RISK_RATIO_MAX,
        "diversity_ratio_min": DIVERSITY_RATIO_MIN,
        "normalized_work_ratio_max": WORK_RATIO_MAX,
        "fixed12_uses_same_full_gate_vs_fixed16": True,
        "adaptive_vs_fixed12_direct_gate_present": False,
        "cross_dataset_or_cross_group_score_allowed": False,
        "new_generation_allowed": False,
        "generation_started": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return collection._load_json(path)


def _load_runtime():
    return fixed_evaluation._load_runtime()


def _table_path(
    root: Path, *, seed: int, dataset: str, arm: str
) -> Path:
    return (
        root
        / collection.OUTPUT_DIR
        / f"seed_{seed}"
        / dataset
        / collection._arm_label(arm)
        / "terminal_current.csv"
    )


def _audit_collection(
    root: Path,
    confirmed_report_sha256: str,
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    report_path = root / collection.OUTPUT_DIR / collection.COLLECTION_REPORT
    observed_sha = collection._sha256_file(report_path)
    if observed_sha != confirmed_report_sha256:
        raise ValueError(
            "collection report SHA 与显式确认值不一致："
            f"confirmed={confirmed_report_sha256}, observed={observed_sha}"
        )
    report = _load_json(report_path)
    if (
        report.get("contract_version") != collection.PROTOCOL_VERSION
        or report.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256
        or report.get("protocol") != collection.frozen_protocol_manifest()
        or report.get("case_count") != 30
    ):
        raise RuntimeError("collection 协议或 30-case 身份漂移")
    if (
        report.get("raw_reference_data_accessed")
        or report.get("privacy_budget_consumed")
        or report.get("parameter_retuning_performed")
    ):
        raise RuntimeError("collection 信息流或研究边界漂移")

    execution_commit = report.get("execution_git_commit")
    if not isinstance(execution_commit, str):
        raise TypeError("collection execution commit 缺失")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if collection._git_text(
        root, "merge-base", execution_commit, current_commit
    ) != execution_commit:
        raise RuntimeError("collection execution commit 不是评价提交的祖先")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 30:
        raise RuntimeError("collection raw results 不完整")
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["dataset"], row["arm"])
        if key in indexed:
            raise RuntimeError(f"collection case 重复：{key}")
        if (
            key[0] not in collection.SEEDS
            or key[1] not in collection.DATASETS
            or key[2] not in collection.ARMS
            or row.get("protocol_sha256") != collection.FROZEN_PROTOCOL_SHA256
            or row.get("git_commit") != execution_commit
        ):
            raise RuntimeError(f"collection case 身份漂移：{key}")
        spec = collection.DATASETS[key[1]]
        if (
            row.get("query_identity_sha256")
            != spec["query_identity_sha256"]
            or row.get("target_vector_sha256")
            != spec["target_vector_sha256"]
        ):
            raise RuntimeError(f"collection query/target 身份漂移：{key}")
        expected_arm = collection._arm_generator_identity(key[2])
        if (
            row.get("alpha_schedule_mode")
            != expected_arm["alpha_schedule_mode"]
            or row.get("fixed_alpha") != expected_arm["fixed_alpha"]
        ):
            raise RuntimeError(f"collection alpha schedule 身份漂移：{key}")
        path = _table_path(
            root, seed=key[0], dataset=key[1], arm=key[2]
        )
        if collection._sha256_file(path) != row["terminal_table_sha256"]:
            raise RuntimeError(f"collection terminal table SHA 漂移：{key}")
        indexed[key] = row
    expected = {
        (seed, dataset, arm)
        for seed in collection.SEEDS
        for dataset, arm in collection.CASE_ORDER
    }
    if set(indexed) != expected:
        raise RuntimeError("collection 30-case 身份不完整")
    return report, indexed


def _nested(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {path} 不是数值")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"metric {path} 不是有限数")
    return value


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _summary(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str, path: str
) -> dict[str, Any]:
    selected = sorted(
        (
            case
            for case in cases
            if case["dataset"] == dataset and case["arm"] == arm
        ),
        key=lambda case: case["seed"],
    )
    if len(selected) != len(collection.SEEDS):
        raise RuntimeError(f"{dataset}/{arm}/{path} 缺少五种子")
    values = [_nested(case, path) for case in selected]
    return {
        "mean": _mean(values),
        "median": float(statistics.median(values)),
        "values_by_seed": {
            str(case["seed"]): value for case, value in zip(selected, values)
        },
    }


def _paired(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    candidate_arm: str,
    baseline_arm: str,
    path: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    indexed = {
        (case["seed"], case["arm"]): _nested(case, path)
        for case in cases
        if case["dataset"] == dataset
    }
    differences = []
    candidate_values = []
    baseline_values = []
    for seed in collection.SEEDS:
        candidate = indexed[(seed, candidate_arm)]
        baseline = indexed[(seed, baseline_arm)]
        candidate_values.append(candidate)
        baseline_values.append(baseline)
        differences.append(candidate - baseline)
    better = [value < 0 if lower_is_better else value > 0 for value in differences]
    worse = [value > 0 if lower_is_better else value < 0 for value in differences]
    mean_difference = _mean(differences)
    difference_std = statistics.stdev(differences)
    half_width = 2.7764451051977987 * difference_std / math.sqrt(len(differences))
    candidate_mean = _mean(candidate_values)
    baseline_mean = _mean(baseline_values)
    return {
        "metric": path,
        "candidate_arm": candidate_arm,
        "baseline_arm": baseline_arm,
        "lower_is_better": lower_is_better,
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "mean_difference": mean_difference,
        "candidate_over_baseline": (
            candidate_mean / baseline_mean if baseline_mean != 0 else None
        ),
        "paired_wins": int(sum(better)),
        "paired_ties": int(sum(value == 0 for value in differences)),
        "paired_losses": int(sum(worse)),
        "differences_by_seed": {
            str(seed): value
            for seed, value in zip(collection.SEEDS, differences)
        },
        "difference_95pct_t_interval": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
    }


def _noninferior_ratio(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    candidate_arm: str,
    path: str,
    maximum: float,
) -> tuple[bool, dict[str, Any]]:
    comparison = _paired(
        cases,
        dataset,
        candidate_arm,
        collection.ARM_FIXED_16,
        path,
        lower_is_better=True,
    )
    baseline = comparison["baseline_mean"]
    candidate = comparison["candidate_mean"]
    passed = candidate == 0 if baseline == 0 else candidate / baseline <= maximum
    comparison["maximum_ratio"] = maximum
    comparison["pass"] = bool(passed)
    return bool(passed), comparison


def _safety_paths(dataset: str) -> dict[str, str]:
    if dataset == "test_300x10":
        return {
            name: f"metrics.offline_query_groups.{name}.normalized_l1_mean"
            for name in TEST_GROUP_ORDER
        }
    return {
        "one_way_safety": (
            "metrics.offline_query_groups.one_way_safety.normalized_l1_mean"
        ),
        "unmeasured_3way": (
            "metrics.offline_query_groups.unmeasured_3way.normalized_l1_mean"
        ),
        "all_4way": (
            "metrics.offline_query_groups.all_4way.normalized_l1_mean"
        ),
        "binned_joint_tvd": "metrics.binned_joint.tvd",
    }


def _classify_arm(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    arm: str,
    all_normal: bool,
) -> dict[str, Any]:
    measured = _paired(
        cases,
        dataset,
        arm,
        collection.ARM_FIXED_16,
        "metrics.measured.overall.normalized_l1_mean",
        lower_is_better=True,
    )
    stable_measured_gain = (
        measured["candidate_mean"] < measured["baseline_mean"]
        and measured["paired_wins"] >= STABLE_WIN_MINIMUM
    )
    safety = {}
    safety_pass = True
    for name, path in _safety_paths(dataset).items():
        passed, comparison = _noninferior_ratio(
            cases, dataset, arm, path, RISK_RATIO_MAX
        )
        safety[name] = comparison
        safety_pass = safety_pass and passed

    diversity = {}
    diversity_pass = True
    for name, path in {
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
    }.items():
        comparison = _paired(
            cases,
            dataset,
            arm,
            collection.ARM_FIXED_16,
            path,
            lower_is_better=False,
        )
        passed = (
            comparison["candidate_mean"]
            >= DIVERSITY_RATIO_MIN * comparison["baseline_mean"]
        )
        comparison["minimum_ratio"] = DIVERSITY_RATIO_MIN
        comparison["pass"] = bool(passed)
        diversity[name] = comparison
        diversity_pass = diversity_pass and passed

    validity_values = [
        _nested(case, "metrics.validity.valid_row_rate")
        for case in cases
        if case["dataset"] == dataset and case["arm"] == arm
    ]
    validity_pass = len(validity_values) == len(collection.SEEDS) and all(
        value == 1.0 for value in validity_values
    )
    compute_pass, compute = _noninferior_ratio(
        cases,
        dataset,
        arm,
        "normalized_work_at_stop",
        WORK_RATIO_MAX,
    )

    supported_name = (
        "supported_adaptive_escape"
        if arm == collection.ARM_ADAPTIVE
        else "supported_fixed_alpha_12"
    )
    if not all_normal:
        classification = "inconclusive_resource_cap"
    elif not stable_measured_gain:
        classification = "no_stable_measured_gain"
    elif not (safety_pass and diversity_pass and validity_pass):
        classification = "measured_gain_with_quality_or_diversity_risk"
    elif not compute_pass:
        classification = "quality_supported_with_compute_tradeoff"
    else:
        classification = supported_name
    return {
        "classification": classification,
        "full_support": classification == supported_name,
        "stable_measured_gain": stable_measured_gain,
        "measured": measured,
        "offline_safety_pass": safety_pass,
        "offline_safety": safety,
        "diversity_pass": diversity_pass,
        "diversity": diversity,
        "validity_pass": validity_pass,
        "validity_values": validity_values,
        "compute_pass": compute_pass,
        "compute": compute,
    }


def _adaptive_activation(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    selected = sorted(
        (
            case
            for case in cases
            if case["dataset"] == dataset
            and case["arm"] == collection.ARM_ADAPTIVE
        ),
        key=lambda case: case["seed"],
    )
    if len(selected) != len(collection.SEEDS):
        raise RuntimeError(f"{dataset} adaptive activation 缺少五种子")
    counts = [
        int(case["adaptive_alpha_summary"]["escape_count"])
        for case in selected
    ]
    triggered = [value > 0 for value in counts]
    return {
        "classification": (
            "adaptive_not_exercised"
            if not any(triggered) else "adaptive_exercised"
        ),
        "mechanism_claim_allowed": bool(any(triggered)),
        "triggered_case_count": int(sum(triggered)),
        "total_escape_count": int(sum(counts)),
        "escape_counts_by_seed": {
            str(case["seed"]): count
            for case, count in zip(selected, counts)
        },
        "new_best_during_escape_count": int(
            sum(
                case["adaptive_alpha_summary"][
                    "new_best_during_escape_count"
                ]
                for case in selected
            )
        ),
    }


def _direct_adaptive_vs_fixed12(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    comparisons = {
        "measured_l1": (
            "metrics.measured.overall.normalized_l1_mean",
            True,
        ),
        "normalized_work": ("normalized_work_at_stop", True),
        "unique_row_rate": ("metrics.diversity.unique_row_rate", False),
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio",
            False,
        ),
    }
    result = {
        name: _paired(
            cases,
            dataset,
            collection.ARM_ADAPTIVE,
            collection.ARM_FIXED_12,
            path,
            lower_is_better=lower_is_better,
        )
        for name, (path, lower_is_better) in comparisons.items()
    }
    result["offline_safety"] = {
        name: _paired(
            cases,
            dataset,
            collection.ARM_ADAPTIVE,
            collection.ARM_FIXED_12,
            path,
            lower_is_better=True,
        )
        for name, path in _safety_paths(dataset).items()
    }
    result["hard_gate_applied"] = False
    return result


def _mechanism_interpretation(
    adaptive_supported: bool, fixed12_supported: bool
) -> str:
    if adaptive_supported and not fixed12_supported:
        return "supports_timed_escape_beyond_always_low_alpha"
    if adaptive_supported and fixed12_supported:
        return "adaptive_and_always_low_both_supported"
    if not adaptive_supported and fixed12_supported:
        return "supports_always_low_not_adaptive"
    return "no_supported_alpha12_strategy"


def _build_summaries(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "measured_l1": "metrics.measured.overall.normalized_l1_mean",
        "measured_count_error": (
            "metrics.measured.overall.absolute_count_error_mean"
        ),
        "normalized_work": "normalized_work_at_stop",
        "rounds": "rounds_run",
        "row_max_prob_tail_mean": (
            "donor_concentration.row_max_prob_mean.tail_mean"
        ),
        "effective_donor_fraction_tail_mean": (
            "donor_concentration.effective_donor_fraction.tail_mean"
        ),
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
        "escape_count": "adaptive_alpha_summary.escape_count",
        "alpha12_normalized_work": (
            "adaptive_alpha_summary.alpha12_normalized_work"
        ),
    }
    result = {}
    for dataset in collection.DATASET_ORDER:
        result[dataset] = {}
        for arm in collection.ARMS:
            arm_paths = dict(paths)
            if arm != collection.ARM_ADAPTIVE:
                arm_paths.pop("alpha12_normalized_work")
            result[dataset][arm] = {
                name: _summary(cases, dataset, arm, path)
                for name, path in arm_paths.items()
            }
    return result


def _frozen_classification(
    cases: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    all_normal = all(
        case["termination_reason"] in NORMAL_REASONS for case in cases
    )
    adaptive = {
        dataset: _classify_arm(
            cases,
            dataset,
            collection.ARM_ADAPTIVE,
            all_normal,
        )
        for dataset in collection.DATASET_ORDER
    }
    fixed12 = {
        dataset: _classify_arm(
            cases,
            dataset,
            collection.ARM_FIXED_12,
            all_normal,
        )
        for dataset in collection.DATASET_ORDER
    }
    interpretation = {
        dataset: _mechanism_interpretation(
            adaptive[dataset]["full_support"],
            fixed12[dataset]["full_support"],
        )
        for dataset in collection.DATASET_ORDER
    }
    activation = {
        dataset: _adaptive_activation(cases, dataset)
        for dataset in collection.DATASET_ORDER
    }
    supported_count = sum(
        adaptive[dataset]["full_support"]
        for dataset in collection.DATASET_ORDER
    )
    if not all_normal:
        cross_dataset = "inconclusive_resource_cap"
    elif supported_count == len(collection.DATASET_ORDER):
        cross_dataset = "shared_adaptive_support"
    elif supported_count == 1:
        cross_dataset = "dataset_dependent_adaptive_response"
    else:
        cross_dataset = "no_shared_adaptive_support"
    return {
        "all_30_cases_normal": all_normal,
        "normal_case_count": sum(
            case["termination_reason"] in NORMAL_REASONS for case in cases
        ),
        "resource_cap_case_count": sum(
            case["termination_reason"] == "resource_cap_reached"
            for case in cases
        ),
        "adaptive_vs_fixed16": adaptive,
        "fixed12_vs_fixed16": fixed12,
        "adaptive_activation": activation,
        "mechanism_claim_status": {
            dataset: (
                "allowed"
                if activation[dataset]["mechanism_claim_allowed"]
                else "prohibited_adaptive_not_exercised"
            )
            for dataset in collection.DATASET_ORDER
        },
        "mechanism_interpretation": interpretation,
        "adaptive_vs_fixed12_descriptive": {
            dataset: _direct_adaptive_vs_fixed12(cases, dataset)
            for dataset in collection.DATASET_ORDER
        },
        "cross_dataset_response": cross_dataset,
        "fixed_alpha_selected": None,
        "public_default_changed": False,
    }


def evaluate(confirmed_collection_report_sha256: str) -> Path:
    if len(confirmed_collection_report_sha256) != 64:
        raise ValueError("必须显式确认完整 collection report SHA-256")
    root = collection._repo_root()
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式评价要求包含 untracked 在内的干净工作树")
    destination = root / collection.OUTPUT_DIR
    report_path = destination / EVALUATION_REPORT
    if report_path.exists():
        raise FileExistsError(f"评价报告已存在，不覆盖：{report_path}")

    collection_report, indexed = _audit_collection(
        root, confirmed_collection_report_sha256
    )
    test_groups, test_identity_audit = fixed_evaluation._freeze_test_groups(root)
    nltcs_groups, nltcs_identity_audit = fixed_evaluation._freeze_nltcs_groups(root)
    runtime = _load_runtime()
    references, reference_sha = fixed_evaluation._load_references(root, runtime)

    dataset_inputs = {}
    for dataset in collection.DATASET_ORDER:
        spec = collection.DATASETS[dataset]
        schema = runtime.load_schema(str(root / spec["schema"]))
        queries = runtime.load_queries(str(root / spec["queries"]))
        payload = _load_json(root / spec["queries"])
        targets = [query["result"] for query in payload["queries"]]
        dataset_inputs[dataset] = {
            "schema": schema,
            "queries": queries,
            "targets": targets,
        }

    test_group_targets = {
        name: runtime.evaluate_table(references["test_300x10"], queries)
        for name, queries in test_groups.items()
    }
    nltcs_marginals = _load_json(
        root / collection.DATASETS["nltcs"]["marginals"]
    )
    nltcs_domains = runtime.offline._discretization_domains(nltcs_marginals)
    nltcs_measured_triples = runtime.offline._measured_cell_keys(
        dataset_inputs["nltcs"]["queries"], nltcs_marginals, order=3
    )

    cases = []
    for seed in collection.SEEDS:
        for dataset, arm in collection.CASE_ORDER:
            source = indexed[(seed, dataset, arm)]
            path = _table_path(root, seed=seed, dataset=dataset, arm=arm)
            table = runtime.pd.read_csv(path)
            inputs = dataset_inputs[dataset]
            if dataset == "test_300x10":
                metrics = fixed_evaluation._evaluate_test_case(
                    runtime,
                    table,
                    inputs["queries"],
                    inputs["targets"],
                    test_groups,
                    test_group_targets,
                    inputs["schema"],
                    references[dataset],
                )
            else:
                metrics = fixed_evaluation._evaluate_nltcs_case(
                    runtime,
                    table,
                    inputs["queries"],
                    inputs["targets"],
                    nltcs_groups["one_way_safety"],
                    inputs["schema"],
                    nltcs_marginals,
                    nltcs_domains,
                    nltcs_measured_triples,
                    references[dataset],
                )
            measured_l1 = metrics["measured"]["overall"]["normalized_l1_mean"]
            if not fixed_evaluation._measured_l1_matches_collection(
                measured_l1, source["terminal_current_normalized_l1"]
            ):
                raise RuntimeError(
                    f"{dataset}/{arm}/seed{seed} measured L1 复算漂移"
                )
            cases.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "seed": seed,
                    "termination_reason": source["termination_reason"],
                    "rounds_run": source["rounds_run"],
                    "normalized_work_at_stop": source[
                        "normalized_work_at_stop"
                    ],
                    "terminal_table_sha256": source["terminal_table_sha256"],
                    "donor_concentration": source["donor_concentration"],
                    "adaptive_alpha_summary": source[
                        "adaptive_alpha_summary"
                    ],
                    "metrics": metrics,
                }
            )
            print(
                f"[evaluate {dataset}/{arm}/seed={seed}] "
                f"L1={measured_l1:.10f}",
                flush=True,
            )

    report = {
        **build_plan(),
        "mode": "evaluate_frozen_collection_after_identity_audit",
        "evaluation_git_commit": collection._git_text(root, "rev-parse", "HEAD"),
        "collection_report_sha256": confirmed_collection_report_sha256,
        "collection_execution_git_commit": collection_report[
            "execution_git_commit"
        ],
        "query_identity_frozen_before_reference_load": True,
        "query_identity_audit": {
            "test_300x10": test_identity_audit,
            "nltcs": nltcs_identity_audit,
        },
        "reference_sha256": reference_sha,
        "case_count": len(cases),
        "cases": cases,
        "summary": _build_summaries(cases),
        "frozen_classification": _frozen_classification(cases),
        "cross_dataset_or_cross_group_score_present": False,
        "new_generation_performed_by_evaluator": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "claim_scope": "two_level_adaptive_alpha_evaluation_only",
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=".evaluation-report.",
        suffix=".tmp",
        dir=destination,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
    os.replace(temporary, report_path)
    return report_path


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
        print(json.dumps(build_plan(), ensure_ascii=False, sort_keys=True, indent=2))
        return
    path = evaluate(args.confirm_collection_sha)
    print(f"adaptive alpha evaluation -> {path}")
    print(f"evaluation SHA-256 -> {collection._sha256_file(path)}")


if __name__ == "__main__":
    main()
