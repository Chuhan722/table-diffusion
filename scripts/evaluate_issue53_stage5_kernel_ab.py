#!/usr/bin/env python3
"""Offline evaluation for the frozen Issue #53 Stage 5 kernel A/B."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

if __package__:
    from scripts import evaluate_issue53_fixed_alpha_calibration as fixed_evaluation
    from scripts import issue53_stage5_protocol as protocol
    from scripts import run_issue53_stage5_kernel_ab as collection
else:
    import evaluate_issue53_fixed_alpha_calibration as fixed_evaluation
    import issue53_stage5_protocol as protocol
    import run_issue53_stage5_kernel_ab as collection


EVALUATION_VERSION = "issue53-stage5-kernel-ab-evaluation-v1"
EVALUATION_REPORT = protocol.EVALUATION_REPORT
REFERENCE_PATHS = protocol.REFERENCE_PATHS
REFERENCE_SHA256 = protocol.REFERENCE_SHA256
TEST_GROUP_ORDER = protocol.TEST_GROUP_ORDER
TEST_GROUP_COUNTS = protocol.TEST_GROUP_COUNTS
TEST_GROUP_IDENTITIES = protocol.TEST_GROUP_IDENTITIES
NLTCS_GROUP_COUNTS = protocol.NLTCS_GROUP_COUNTS
NLTCS_GROUP_IDENTITIES = protocol.NLTCS_GROUP_IDENTITIES

LOWER_RISK_RATIO_MAX = protocol.LOWER_RISK_RATIO_MAX
HIGHER_QUALITY_RATIO_MIN = protocol.HIGHER_QUALITY_RATIO_MIN
OUTER_COMPUTE_RATIO_MAX = protocol.OUTER_COMPUTE_RATIO_MAX
STABLE_WIN_MINIMUM = protocol.STABLE_WIN_MINIMUM
T_CRITICAL_DF9_95 = 2.2621571628540993


def build_plan() -> dict[str, Any]:
    """Describe evaluation without reading a collection or reference table."""

    return {
        "contract_version": EVALUATION_VERSION,
        "mode": "plan_only_no_collection_reference_or_generator_access",
        "collection_protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "collection_report": str(
            protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
        ),
        "evaluation_report": str(protocol.OUTPUT_DIR / EVALUATION_REPORT),
        "reference_sha256": REFERENCE_SHA256,
        "test_groups": {
            "order": list(TEST_GROUP_ORDER),
            "counts": TEST_GROUP_COUNTS,
            "identity_sha256": TEST_GROUP_IDENTITIES,
        },
        "nltcs_groups": {
            "counts": NLTCS_GROUP_COUNTS,
            "identity_sha256": NLTCS_GROUP_IDENTITIES,
        },
        "candidate_arm": protocol.ARM_FACTOR,
        "baseline_arm": protocol.ARM_INDEPENDENT,
        "stable_measured_gain": {
            "aggregate_absolute_count_error_sum_strictly_lower": True,
            "paired_strict_wins_minimum": STABLE_WIN_MINIMUM,
            "paired_seed_count": len(protocol.FORMAL_SEEDS),
        },
        "lower_is_better_ratio_max": LOWER_RISK_RATIO_MAX,
        "higher_is_better_ratio_min": HIGHER_QUALITY_RATIO_MIN,
        "outer_compute_ratio_max": OUTER_COMPUTE_RATIO_MAX,
        "cross_dataset_or_cross_group_weighted_score_allowed": False,
        "new_generation_allowed": False,
        "generation_started": False,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return collection._load_json(path)


def _load_runtime():
    return fixed_evaluation._load_runtime()


def _table_path(root: Path, source: dict[str, Any]) -> Path:
    return (
        root
        / protocol.OUTPUT_DIR
        / f"seed_{source['seed']}"
        / source["terminal_table_path"]
    )


def _audit_source_generator_params(row: dict[str, Any]) -> None:
    dataset = row["dataset"]
    arm = row["arm"]
    params = row.get("generator_params")
    if not isinstance(params, dict):
        raise RuntimeError("collection generator params 缺失")
    expected = {
        "n_records": protocol.DATASETS[dataset]["n_records"],
        "n_rounds": protocol.ROUND_CAP,
        "seed": row["seed"],
        "beta": 1.0,
        "h": 0.8,
        "rho": 0.01,
        "eta": 0.5,
        "mu": 0.01,
        "tol": "positive_infinity",
        "device": protocol.DATASETS[dataset]["device"],
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "marginal",
        "distance_mode": "geometric",
        "p": None,
        "lambda": 0.5,
        "alpha_min": None,
        "alpha_max": None,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": protocol.FIXED_ALPHA,
        "delta": 0.05,
        "winsorize_quantiles": [0.01, 0.99],
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": protocol.TAU,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_reference_scale": None,
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_sweeps": (
            0 if arm == protocol.ARM_INDEPENDENT else protocol.FACTOR_SWEEPS
        ),
        "factorized_gibbs_max_order": protocol.DATASETS[dataset][
            "max_factor_order"
        ],
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": arm == protocol.ARM_FACTOR,
        "candidate_budget": protocol.CANDIDATE_BUDGET,
        "residual_self_cooling": None,
        "self_cooling_monotone": None,
        "self_cooling_stop_ratio": None,
        "rho_anneal_end": None,
        "rho_anneal_rounds": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "record_transition_clocks": True,
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": protocol.PATIENCE_TICKS,
        "horizon_invariant": False,
    }
    drift = {
        key: {"expected": value, "observed": params.get(key)}
        for key, value in expected.items()
        if params.get(key) != value
    }
    expected_index = protocol.case_order_for_seed(row["seed"]).index(
        (dataset, arm)
    )
    if drift or row.get("execution_order_index") != expected_index:
        raise RuntimeError(
            f"collection generator/order identity 漂移："
            f"seed={row['seed']} {dataset}/{arm}, drift={drift}"
        )


def _audit_collection(
    root: Path, confirmed_report_sha256: str
) -> tuple[dict[str, Any], dict[tuple[int, str, str], dict[str, Any]]]:
    report_path = root / protocol.OUTPUT_DIR / protocol.COLLECTION_REPORT
    observed_sha = collection._sha256_file(report_path)
    if observed_sha != confirmed_report_sha256:
        raise ValueError(
            "collection report SHA 与显式确认值不一致："
            f"confirmed={confirmed_report_sha256}, observed={observed_sha}"
        )
    report = _load_json(report_path)
    if (
        report.get("contract_version") != protocol.PROTOCOL_VERSION
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("formal_result_valid") is not True
        or report.get("case_count") != 40
        or report.get("paired_dataset_seed_count") != 20
        or report.get("raw_reference_data_accessed") is not False
        or report.get("partial_matrix_comparison_emitted") is not False
        or report.get("offline_classification_emitted") is not False
        or report.get("parameter_retuning_performed") is not False
    ):
        raise RuntimeError("collection 协议、完整性或信息边界漂移")
    expected_audit = {
        "all_40_cases_present": True,
        "all_20_pairs_present": True,
        "all_artifact_sha256_verified": True,
        "all_initial_state_pairing_verified": True,
        "all_case_order_verified": True,
        "single_clean_execution_commit": True,
        "compiled_equivalence_smoke_gate_passed": True,
    }
    if report.get("collection_audit") != expected_audit:
        raise RuntimeError("collection audit gate 未完整通过")

    execution_commit = report.get("execution_git_commit")
    if not isinstance(execution_commit, str) or len(execution_commit) != 40:
        raise RuntimeError("collection execution commit 缺失")
    current_commit = collection._git_text(root, "rev-parse", "HEAD")
    if collection._git_text(
        root, "merge-base", execution_commit, current_commit
    ) != execution_commit:
        raise RuntimeError("collection execution commit 不是 evaluator commit 的祖先")

    rows = report.get("raw_results")
    if not isinstance(rows, list) or len(rows) != 40:
        raise RuntimeError("collection raw_results 不是完整 40 cases")
    indexed = {}
    for row in rows:
        key = (int(row["seed"]), row["dataset"], row["arm"])
        if key in indexed:
            raise RuntimeError(f"collection case 重复：{key}")
        if (
            key[0] not in protocol.FORMAL_SEEDS
            or key[1] not in protocol.DATASETS
            or key[2] not in protocol.ARMS
            or row.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
            or row.get("git_commit") != execution_commit
            or row.get("mode") != "formal"
            or row.get("formal_result_valid") is not True
            or row.get("artifact_role") != "kernel_arm"
            or row.get("raw_reference_data_accessed") is not False
        ):
            raise RuntimeError(f"collection case 身份漂移：{key}")
        spec = protocol.DATASETS[key[1]]
        if (
            row.get("query_identity_sha256")
            != spec["query_identity_sha256"]
            or row.get("target_vector_sha256")
            != spec["target_vector_sha256"]
        ):
            raise RuntimeError(f"collection query/target 身份漂移：{key}")
        shard_root = root / protocol.OUTPUT_DIR / f"seed_{key[0]}"
        collection._audit_result_artifacts(shard_root, row)
        _audit_source_generator_params(row)
        indexed[key] = row
    expected = {
        (seed, dataset, arm)
        for seed in protocol.FORMAL_SEEDS
        for dataset in protocol.DATASET_ORDER
        for arm in protocol.ARMS
    }
    if set(indexed) != expected:
        raise RuntimeError("collection 40-case 身份不完整")
    for seed in protocol.FORMAL_SEEDS:
        collection._pairing_from_unordered_rows(
            [row for key, row in indexed.items() if key[0] == seed], seed
        )
    return report, indexed


def _nested(record: dict[str, Any], path: str) -> float:
    value: Any = record
    for part in path.split("."):
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"metric {path} 不是数值")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"metric {path} 不是有限数")
    return normalized


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values))


def _selected_cases(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str
) -> list[dict[str, Any]]:
    selected = sorted(
        [
            case
            for case in cases
            if case["dataset"] == dataset and case["arm"] == arm
        ],
        key=lambda case: case["seed"],
    )
    if len(selected) != len(protocol.FORMAL_SEEDS):
        raise RuntimeError(f"{dataset}/{arm} 缺少十种子")
    return selected


def _arm_summary(
    cases: Sequence[dict[str, Any]], dataset: str, arm: str, path: str
) -> dict[str, Any]:
    selected = _selected_cases(cases, dataset, arm)
    values = [_nested(case, path) for case in selected]
    return {
        "mean": _mean(values),
        "median": float(statistics.median(values)),
        "minimum": min(values),
        "maximum": max(values),
        "values_by_seed": {
            str(case["seed"]): value for case, value in zip(selected, values)
        },
    }


def _paired(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    path: str,
    *,
    lower_is_better: bool,
) -> dict[str, Any]:
    factor = _selected_cases(cases, dataset, protocol.ARM_FACTOR)
    independent = _selected_cases(cases, dataset, protocol.ARM_INDEPENDENT)
    factor_by_seed = {case["seed"]: _nested(case, path) for case in factor}
    independent_by_seed = {
        case["seed"]: _nested(case, path) for case in independent
    }
    differences = [
        factor_by_seed[seed] - independent_by_seed[seed]
        for seed in protocol.FORMAL_SEEDS
    ]
    wins = [value < 0 if lower_is_better else value > 0 for value in differences]
    losses = [value > 0 if lower_is_better else value < 0 for value in differences]
    mean_difference = _mean(differences)
    std = statistics.stdev(differences)
    half_width = T_CRITICAL_DF9_95 * std / math.sqrt(len(differences))
    factor_mean = _mean(list(factor_by_seed.values()))
    independent_mean = _mean(list(independent_by_seed.values()))
    return {
        "metric": path,
        "candidate_arm": protocol.ARM_FACTOR,
        "baseline_arm": protocol.ARM_INDEPENDENT,
        "lower_is_better": lower_is_better,
        "candidate_mean": factor_mean,
        "baseline_mean": independent_mean,
        "candidate_over_baseline": (
            factor_mean / independent_mean if independent_mean != 0.0 else None
        ),
        "mean_paired_difference": mean_difference,
        "paired_wins": int(sum(wins)),
        "paired_ties": int(sum(value == 0.0 for value in differences)),
        "paired_losses": int(sum(losses)),
        "factor_values_by_seed": {
            str(seed): factor_by_seed[seed] for seed in protocol.FORMAL_SEEDS
        },
        "independent_values_by_seed": {
            str(seed): independent_by_seed[seed]
            for seed in protocol.FORMAL_SEEDS
        },
        "factor_minus_independent_by_seed": {
            str(seed): value
            for seed, value in zip(protocol.FORMAL_SEEDS, differences)
        },
        "paired_difference_95pct_t_interval_diagnostic_only": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
    }


def _measured_stability(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    path = "metrics.measured.overall.absolute_count_error_sum"
    comparison = _paired(cases, dataset, path, lower_is_better=True)
    factor_sum = int(sum(comparison["factor_values_by_seed"].values()))
    independent_sum = int(
        sum(comparison["independent_values_by_seed"].values())
    )
    factor_stable = (
        factor_sum < independent_sum
        and comparison["paired_wins"] >= STABLE_WIN_MINIMUM
    )
    independent_stable = (
        independent_sum < factor_sum
        and comparison["paired_losses"] >= STABLE_WIN_MINIMUM
    )
    return {
        **comparison,
        "factor_10_seed_aggregate_count_error_sum": factor_sum,
        "independent_10_seed_aggregate_count_error_sum": independent_sum,
        "stable_factor_measured_gain": bool(factor_stable),
        "stable_independent_measured_advantage": bool(independent_stable),
        "mixed_no_stable_kernel_winner": bool(
            not factor_stable and not independent_stable
        ),
        "paired_strict_wins_required": STABLE_WIN_MINIMUM,
    }


def _lower_gate(
    cases: Sequence[dict[str, Any]], dataset: str, path: str
) -> dict[str, Any]:
    result = _paired(cases, dataset, path, lower_is_better=True)
    candidate = result["candidate_mean"]
    baseline = result["baseline_mean"]
    passed = candidate == 0.0 if baseline == 0.0 else (
        candidate / baseline <= LOWER_RISK_RATIO_MAX
    )
    result.update(
        {
            "maximum_ratio": LOWER_RISK_RATIO_MAX,
            "zero_baseline_requires_zero_candidate": baseline == 0.0,
            "pass": bool(passed),
        }
    )
    return result


def _higher_gate(
    cases: Sequence[dict[str, Any]], dataset: str, path: str
) -> dict[str, Any]:
    result = _paired(cases, dataset, path, lower_is_better=False)
    candidate = result["candidate_mean"]
    baseline = result["baseline_mean"]
    passed = candidate >= HIGHER_QUALITY_RATIO_MIN * baseline
    result.update(
        {
            "minimum_ratio": HIGHER_QUALITY_RATIO_MIN,
            "ratio_evidence_available": baseline != 0.0,
            "zero_baseline_report_only": baseline == 0.0,
            "pass": bool(passed),
        }
    )
    return result


def _quality_gates(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    offline_paths = (
        {
            name: f"metrics.offline_query_groups.{name}.normalized_l1_mean"
            for name in TEST_GROUP_ORDER
        }
        if dataset == "test_300x10"
        else {
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
    )
    offline = {
        name: _lower_gate(cases, dataset, path)
        for name, path in offline_paths.items()
    }
    support_paths = {
        "synthetic_mass_in_reference_support": (
            "metrics.reference_support.synthetic_mass_in_reference_support"
        ),
        "reference_mass_covered": (
            "metrics.reference_support.reference_mass_covered"
        ),
    }
    diversity_paths = {
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
        "attribute_effective_support_ratio_mean": (
            "metrics.diversity.attribute_effective_support_ratio_mean"
        ),
        "attribute_effective_support_ratio_min": (
            "metrics.diversity.attribute_effective_support_ratio_min"
        ),
    }
    support = {
        name: _higher_gate(cases, dataset, path)
        for name, path in support_paths.items()
    }
    diversity = {
        name: _higher_gate(cases, dataset, path)
        for name, path in diversity_paths.items()
    }
    factor_validity = {
        str(case["seed"]): _nested(case, "metrics.validity.valid_row_rate")
        for case in _selected_cases(cases, dataset, protocol.ARM_FACTOR)
    }
    validity_pass = all(value == 1.0 for value in factor_validity.values())
    return {
        "offline_safety": offline,
        "offline_safety_pass": all(item["pass"] for item in offline.values()),
        "reference_support": support,
        "reference_support_pass": all(item["pass"] for item in support.values()),
        "diversity": diversity,
        "diversity_pass": all(item["pass"] for item in diversity.values()),
        "factor_valid_row_rate_by_seed": factor_validity,
        "validity_pass": validity_pass,
        "all_quality_gates_pass": bool(
            all(item["pass"] for item in offline.values())
            and all(item["pass"] for item in support.values())
            and all(item["pass"] for item in diversity.values())
            and validity_pass
        ),
    }


def _outer_compute_gate(
    cases: Sequence[dict[str, Any]], dataset: str
) -> dict[str, Any]:
    metrics = {
        "normalized_work": _lower_gate(
            cases, dataset, "terminal_behavior.normalized_work_at_stop"
        ),
        "candidate_evaluation_count": _lower_gate(
            cases, dataset, "cost.candidate_evaluation_count"
        ),
    }
    for value in metrics.values():
        value["maximum_ratio"] = OUTER_COMPUTE_RATIO_MAX
        baseline = value["baseline_mean"]
        candidate = value["candidate_mean"]
        value["pass"] = bool(
            candidate == 0.0
            if baseline == 0.0
            else candidate / baseline <= OUTER_COMPUTE_RATIO_MAX
        )
    return {
        "metrics": metrics,
        "pass": all(value["pass"] for value in metrics.values()),
        "gibbs_internal_cost_is_additional_and_not_gated_against_zero": True,
    }


def _execution_qualification(
    cases: Sequence[dict[str, Any]], *, collection_audit_ok: bool = True
) -> dict[str, Any]:
    expected = {
        (seed, dataset, arm)
        for seed in protocol.FORMAL_SEEDS
        for dataset in protocol.DATASET_ORDER
        for arm in protocol.ARMS
    }
    identities = {
        (case.get("seed"), case.get("dataset"), case.get("arm"))
        for case in cases
    }
    structural = collection_audit_ok and len(cases) == 40 and identities == expected
    known_reasons = set(protocol.NORMAL_REASONS) | {protocol.RESOURCE_CAP_REASON}
    structural = structural and all(
        case.get("termination_reason") in known_reasons for case in cases
    )
    resource_cases = [
        case
        for case in cases
        if case.get("termination_reason") == protocol.RESOURCE_CAP_REASON
    ]
    direction_clip = sum(
        int(case.get("direction_logit_clipped_count", 0)) for case in cases
    )
    factor_clip = sum(
        int(case.get("cost", {}).get(
            "factorized_gibbs_conditional_logit_clipped_count", 0
        ))
        for case in cases
        if case.get("arm") == protocol.ARM_FACTOR
    )
    if not structural:
        status = "execution_invalid"
    elif resource_cases:
        status = "inconclusive_resource_cap"
    elif direction_clip or factor_clip:
        status = "outside_stage4_qualified_unclipped_regime"
    else:
        status = "qualified"
    return {
        "status": status,
        "collection_structure_valid": structural,
        "case_count": len(cases),
        "pair_count": len(cases) // 2 if structural else None,
        "normal_case_count": sum(
            case.get("termination_reason") in protocol.NORMAL_REASONS
            for case in cases
        ),
        "resource_cap_case_count": len(resource_cases),
        "direction_logit_clipped_count": direction_clip,
        "factor_conditional_logit_clipped_count": factor_clip,
    }


def _classify_dataset(
    cases: Sequence[dict[str, Any]],
    dataset: str,
    execution: dict[str, Any],
) -> dict[str, Any]:
    measured = _measured_stability(cases, dataset)
    quality = _quality_gates(cases, dataset)
    compute = _outer_compute_gate(cases, dataset)
    if execution["status"] != "qualified":
        classification = execution["status"]
    elif not measured["stable_factor_measured_gain"]:
        classification = "no_stable_factor_gain"
    elif not quality["all_quality_gates_pass"]:
        classification = "factor_measured_gain_with_quality_or_diversity_risk"
    elif not compute["pass"]:
        classification = "factor_quality_supported_with_outer_compute_tradeoff"
    else:
        classification = "factor_quality_supported_outer_efficient"
    return {
        "classification": classification,
        "measured_stability": measured,
        "no_stable_factor_gain_detail": (
            "stable_independent_measured_advantage"
            if measured["stable_independent_measured_advantage"]
            else "mixed_no_stable_kernel_winner"
            if measured["mixed_no_stable_kernel_winner"]
            else None
        ),
        "quality_gates": quality,
        "outer_compute_gate": compute,
    }


def _cross_dataset_classification(
    by_dataset: dict[str, dict[str, Any]], execution: dict[str, Any]
) -> str:
    if execution["status"] != "qualified":
        return "inconclusive_or_invalid_stage5"
    values = [row["classification"] for row in by_dataset.values()]
    efficient = "factor_quality_supported_outer_efficient"
    tradeoff = "factor_quality_supported_with_outer_compute_tradeoff"
    quality_supported = {efficient, tradeoff}
    supported_count = sum(value in quality_supported for value in values)
    if all(value == efficient for value in values):
        return "shared_factor_support_at_tau2"
    if supported_count == len(protocol.DATASET_ORDER):
        return "shared_factor_quality_support_with_compute_tradeoff"
    if supported_count == 1:
        return "dataset_dependent_kernel_response"
    return "no_shared_factor_support"


def _frozen_classification(
    cases: Sequence[dict[str, Any]], *, collection_audit_ok: bool = True
) -> dict[str, Any]:
    execution = _execution_qualification(
        cases, collection_audit_ok=collection_audit_ok
    )
    by_dataset = {
        dataset: _classify_dataset(cases, dataset, execution)
        for dataset in protocol.DATASET_ORDER
    }
    return {
        "execution_qualification": execution,
        "by_dataset": by_dataset,
        "cross_dataset_response": _cross_dataset_classification(
            by_dataset, execution
        ),
        "tau_selected": None,
        "kernel_default_changed": False,
        "additional_seed_requested": False,
    }


def _add_exact_measured_count_sums(
    runtime: Any,
    metrics: dict[str, Any],
    table: Any,
    queries: Sequence[dict[str, Any]],
    targets: Sequence[int],
) -> int:
    answers = runtime.np.asarray(runtime.evaluate_table(table, list(queries)))
    target_array = runtime.np.asarray(targets)
    total = collection._exact_count_error_sum(target_array, answers, runtime)
    metrics["measured"]["overall"]["absolute_count_error_sum"] = total
    orders = sorted({len(query["conditions"]) for query in queries})
    for order in orders:
        indices = [
            index
            for index, query in enumerate(queries)
            if len(query["conditions"]) == order
        ]
        subtotal = collection._exact_count_error_sum(
            target_array[indices], answers[indices], runtime
        )
        metrics["measured"]["by_order"][str(order)][
            "absolute_count_error_sum"
        ] = subtotal
    return total


def _summary(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    paths = {
        "measured_count_error_sum": (
            "metrics.measured.overall.absolute_count_error_sum"
        ),
        "measured_normalized_l1": (
            "metrics.measured.overall.normalized_l1_mean"
        ),
        "measured_squared_loss": (
            "metrics.measured.overall.squared_loss_diagnostic_only"
        ),
        "normalized_work": "terminal_behavior.normalized_work_at_stop",
        "candidate_evaluations": "cost.candidate_evaluation_count",
        "rounds": "cost.raw_rounds",
        "case_wall_clock_sec": "cost.case_wall_clock_elapsed_sec",
        "generator_elapsed_sec": "cost.generator_elapsed_sec",
        "gibbs_microsteps": "cost.factorized_gibbs_microsteps",
        "conditional_logit_evaluations": (
            "cost.factorized_gibbs_conditional_logit_evaluated_count"
        ),
        "unique_row_rate": "metrics.diversity.unique_row_rate",
        "effective_unique_row_ratio": (
            "metrics.diversity.effective_unique_row_ratio"
        ),
        "synthetic_mass_in_reference_support": (
            "metrics.reference_support.synthetic_mass_in_reference_support"
        ),
        "reference_mass_covered": (
            "metrics.reference_support.reference_mass_covered"
        ),
    }
    return {
        dataset: {
            arm: {
                name: _arm_summary(cases, dataset, arm, path)
                for name, path in paths.items()
            }
            for arm in protocol.ARMS
        }
        for dataset in protocol.DATASET_ORDER
    }


def _wall_clock_diagnostics(
    cases: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    path = "cost.case_wall_clock_elapsed_sec"
    result = {}
    for dataset in protocol.DATASET_ORDER:
        paired = _paired(cases, dataset, path, lower_is_better=True)
        strata = {}
        for arm in protocol.ARMS:
            selected = _selected_cases(cases, dataset, arm)
            first = [
                _nested(case, path)
                for case in selected
                if case["pair_execution_position"] == 1
            ]
            second = [
                _nested(case, path)
                for case in selected
                if case["pair_execution_position"] == 2
            ]
            strata[arm] = {
                "executed_first_count": len(first),
                "executed_first_mean_sec": _mean(first),
                "executed_second_count": len(second),
                "executed_second_mean_sec": _mean(second),
            }
        result[dataset] = {
            "paired": paired,
            "execution_order_strata": strata,
            "hard_gate_applied": False,
        }
    return result


def evaluate(confirmed_collection_report_sha256: str) -> Path:
    if (
        len(confirmed_collection_report_sha256) != 64
        or any(
            char not in "0123456789abcdef"
            for char in confirmed_collection_report_sha256
        )
    ):
        raise ValueError("必须显式确认完整小写 collection report SHA-256")
    root = collection._repo_root()
    if collection._git_text(root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("正式 evaluator 要求包含 untracked 在内的干净工作树")
    destination = root / protocol.OUTPUT_DIR
    report_path = destination / EVALUATION_REPORT
    if report_path.exists():
        raise FileExistsError(f"evaluation report 已存在，不覆盖：{report_path}")

    collection_report, indexed = _audit_collection(
        root, confirmed_collection_report_sha256
    )
    # Query identities are frozen before either reference file is opened.
    test_groups, test_identity_audit = fixed_evaluation._freeze_test_groups(root)
    nltcs_groups, nltcs_identity_audit = fixed_evaluation._freeze_nltcs_groups(root)
    runtime = _load_runtime()
    references, reference_sha = fixed_evaluation._load_references(root, runtime)

    dataset_inputs = {}
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
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
    nltcs_marginals = _load_json(root / protocol.DATASETS["nltcs"]["marginals"])
    nltcs_domains = runtime.offline._discretization_domains(nltcs_marginals)
    nltcs_measured_triples = runtime.offline._measured_cell_keys(
        dataset_inputs["nltcs"]["queries"], nltcs_marginals, order=3
    )

    cases = []
    for seed in protocol.FORMAL_SEEDS:
        for dataset in protocol.DATASET_ORDER:
            for arm in protocol.ARMS:
                source = indexed[(seed, dataset, arm)]
                table = runtime.pd.read_csv(_table_path(root, source))
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
                count_sum = _add_exact_measured_count_sums(
                    runtime,
                    metrics,
                    table,
                    inputs["queries"],
                    inputs["targets"],
                )
                measured_l1 = metrics["measured"]["overall"][
                    "normalized_l1_mean"
                ]
                if (
                    count_sum
                    != source["measured"]["absolute_count_error_sum"]
                    or not fixed_evaluation._measured_l1_matches_collection(
                        measured_l1, source["measured"]["normalized_l1_mean"]
                    )
                    or metrics["measured"]["overall"][
                        "squared_loss_diagnostic_only"
                    ]
                    != source["measured"]["squared_loss"]
                ):
                    raise RuntimeError(
                        f"{dataset}/{arm}/seed{seed} measured 复算漂移"
                    )
                cases.append(
                    {
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "pair_execution_position": (
                            int(source["execution_order_index"]) % 2 + 1
                        ),
                        "termination_reason": source["termination_reason"],
                        "terminal_behavior": source["terminal_behavior"],
                        "cost": source["cost"],
                        "direction_logit_clipped_count": source[
                            "direction_logit_clipped_count"
                        ],
                        "terminal_table_sha256": source[
                            "terminal_table_sha256"
                        ],
                        "metrics": metrics,
                    }
                )
                print(
                    f"[offline evaluate {dataset}/{arm}/seed={seed}] complete",
                    flush=True,
                )

    report = {
        **build_plan(),
        "mode": "evaluate_complete_frozen_collection_after_identity_audit",
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
        "summary": _summary(cases),
        "wall_clock_diagnostics": _wall_clock_diagnostics(cases),
        "frozen_classification": _frozen_classification(cases),
        "cross_dataset_or_cross_group_weighted_score_present": False,
        "new_generation_performed_by_evaluator": False,
        "raw_reference_data_accessed": True,
        "privacy_budget_consumed": False,
        "parameter_retuning_performed": False,
        "claim_scope": "stage5_same_tau_kernel_ab_only",
    }
    collection._atomic_write_json(report_path, report)
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
    print(f"Stage 5 evaluation -> {path}")
    print(f"evaluation SHA-256 -> {collection._sha256_file(path)}")


if __name__ == "__main__":
    main()
