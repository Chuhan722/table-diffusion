#!/usr/bin/env python3
"""Evaluate frozen Stage 6A proposal geometry after structural audit.

This evaluator is strictly post-hoc.  It consumes one complete proposal
collection and the passing structural audit that is cryptographically bound
to that exact collection.  It performs no generation, probability update,
accept/reject decision, retry, rollback, or proposal selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

if __package__:
    from scripts import audit_issue53_stage6a_proposal_structure as auditor
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import collect_issue53_stage6a_proposals as collector
    from scripts import issue53_stage6a_protocol as protocol
else:
    import audit_issue53_stage6a_proposal_structure as auditor
    import build_issue53_stage6a_state_library as state_builder
    import collect_issue53_stage6a_proposals as collector
    import issue53_stage6a_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_FORMAT = "issue53_stage6a_frozen_evaluation_v1"
EVALUATION_FILENAME = "frozen_evaluation.json"

LEG_NAMES = ("copy_only", "mutation_given_copy", "full")
EXACT_METRICS = ("b2", "c2", "g2", "cself2", "ccross2")
WORK_METRICS = (
    "participating_rows",
    "copied_rows",
    "copied_cells",
    "mutation_rows",
    "mutation_changed_cells",
    "mutation_overwrote_copied_cells",
    "copy_query_changed_rows",
    "mutation_query_changed_rows",
    "full_query_changed_rows",
)
SIGNS = ("negative", "zero", "positive")
SIGN_TRANSITIONS = tuple(
    f"{left}_to_{right}" for left in SIGNS for right in SIGNS
)
CATEGORY_TRANSITIONS = tuple(
    f"{copy_category}_to_{full_category}"
    for copy_category in protocol.FULL_GAIN_CATEGORIES
    for full_category in protocol.FULL_GAIN_CATEGORIES
)
PHASES = ("initial", "primary")

EVALUATION_BOUNDARY = {
    "post_hoc_only": True,
    "proposal_generation_run": False,
    "proposal_probability_modified": False,
    "proposal_feedback_to_generation": False,
    "proposal_acceptance_or_rejection": False,
    "proposal_retry_or_rollback": False,
    "proposal_selection_or_ranking": False,
    "structural_audit_required": True,
    "structural_audit_must_pass_and_bind_exact_collection": True,
    "initial_excluded_from_primary_dominance": True,
    "statistical_independence_unit": "source_trajectory_seed",
    "proposal_rows_are_not_independent_seeds": True,
    "equal_seed_weighting": True,
    "adaptive_thresholds_or_subgroups": False,
    "reference_or_held_out_read": False,
    "smoke_is_mechanism_evidence": False,
}

DESCRIPTIVE_STATISTICS_SPEC = {
    "exact_value_authority": "integer_numerator_over_positive_denominator",
    "reported_float_is_diagnostic_only": True,
    "mean": "exact arithmetic mean",
    "quartiles": (
        "Hyndman-Fan type 7 linear interpolation at p=0.25,0.50,0.75; "
        "interpolation is performed as an exact rational"
    ),
    "empty_distribution": "count_zero_and_all_statistics_null",
    "dataset_aggregate": (
        "proposal-level pooled distribution is descriptive only; the frozen "
        "dataset aggregate is the distribution of per-seed means/shares, "
        "with every seed weighted equally"
    ),
    "undefined_share": (
        "a seed with zero axis denominator is retained as undefined and is "
        "never silently converted to zero"
    ),
}


def _json_value(value: Any) -> Any:
    return state_builder._json_safe(value)


def _mode_seeds(mode: str) -> tuple[int, ...]:
    if mode == "formal":
        return protocol.FORMAL_SEEDS
    if mode == "smoke":
        return (protocol.SMOKE_SEED,)
    raise ValueError("mode 必须是 formal 或 smoke")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "value_diagnostic_only": float(value),
    }


def _type7_quantile(
    ordered: Sequence[Fraction], probability: Fraction
) -> Fraction:
    if not ordered:
        raise ValueError("empty distribution 没有 quantile")
    if probability < 0 or probability > 1:
        raise ValueError("quantile probability 必须在 [0, 1]")
    position = Fraction(len(ordered) - 1, 1) * probability
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _exact_summary(values: Sequence[Fraction]) -> dict[str, Any]:
    ordered = sorted(Fraction(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "q25": None,
            "median": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    mean = sum(ordered, Fraction(0, 1)) / len(ordered)
    return {
        "count": len(ordered),
        "mean": _fraction_payload(mean),
        "q25": _fraction_payload(
            _type7_quantile(ordered, Fraction(1, 4))
        ),
        "median": _fraction_payload(
            _type7_quantile(ordered, Fraction(1, 2))
        ),
        "q75": _fraction_payload(
            _type7_quantile(ordered, Fraction(3, 4))
        ),
        "minimum": _fraction_payload(ordered[0]),
        "maximum": _fraction_payload(ordered[-1]),
    }


def _float_type7_quantile(
    ordered: Sequence[float], probability: Fraction
) -> float:
    if not ordered:
        raise ValueError("empty float distribution 没有 quantile")
    position = Fraction(len(ordered) - 1, 1) * probability
    lower = position.numerator // position.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = float(position - lower)
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _float_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise RuntimeError("墙钟统计包含非有限数")
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "q25": None,
            "median": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "q25": _float_type7_quantile(ordered, Fraction(1, 4)),
        "median": _float_type7_quantile(ordered, Fraction(1, 2)),
        "q75": _float_type7_quantile(ordered, Fraction(3, 4)),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _exact_value(
    pair: Mapping[str, Any], leg: str, metric: str
) -> Fraction:
    exact = pair["exact"][leg]
    numerator = exact[f"{metric}_numerator"]
    denominator = exact["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise RuntimeError("proposal exact rational 结构失败")
    return Fraction(numerator, denominator)


def _interaction_value(pair: Mapping[str, Any]) -> Fraction:
    sequential = pair["exact"]["sequential"]
    numerator = sequential["copy_mutation_interaction2_numerator"]
    denominator = sequential["denominator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise RuntimeError("proposal interaction rational 结构失败")
    return Fraction(numerator, denominator)


def _work_value(pair: Mapping[str, Any], metric: str) -> Fraction:
    value = pair["work"][metric]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("proposal work metric 必须是非负整数")
    return Fraction(value, 1)


def _elapsed_value(pair: Mapping[str, Any]) -> float:
    value = pair["elapsed_sec_diagnostic_only"]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0.0
    ):
        raise RuntimeError("proposal wallclock 必须是非负有限数")
    return float(value)


def _copy_full_category_transition(pair: Mapping[str, Any]) -> str:
    return (
        f"{pair['exact']['copy_only']['category']}_to_"
        f"{pair['exact']['full']['category']}"
    )


def _fixed_counts(
    observed: Sequence[str | None], labels: Sequence[str]
) -> dict[str, int]:
    allowed = set(labels)
    unexpected = {value for value in observed if value not in allowed | {None}}
    if unexpected:
        raise RuntimeError(f"出现未冻结类别：{sorted(unexpected)}")
    counts = Counter(value for value in observed if value is not None)
    return {label: int(counts[label]) for label in labels}


def _summarize_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    legs: dict[str, Any] = {}
    for leg in LEG_NAMES:
        leg_rows = [pair["exact"][leg] for pair in pairs]
        legs[leg] = {
            "metrics": {
                metric: _exact_summary([
                    _exact_value(pair, leg, metric) for pair in pairs
                ])
                for metric in EXACT_METRICS
            },
            "category_counts": _fixed_counts(
                [row["category"] for row in leg_rows],
                protocol.FULL_GAIN_CATEGORIES,
            ),
            "curvature_role_counts": _fixed_counts(
                [row["curvature_role"] for row in leg_rows],
                protocol.CURVATURE_ROLE_CATEGORIES,
            ),
            "cross_label_counts": _fixed_counts(
                [row["cross_label"] for row in leg_rows],
                protocol.CROSS_LABELS,
            ),
        }
    return {
        "pair_count": len(pairs),
        "legs": legs,
        "sequential": {
            "copy_mutation_interaction2": _exact_summary([
                _interaction_value(pair) for pair in pairs
            ]),
            "copy_full_gain_sign_transition_counts": _fixed_counts(
                [
                    pair["exact"]["copy_full_gain_sign_transition"]
                    for pair in pairs
                ],
                SIGN_TRANSITIONS,
            ),
            "copy_full_category_transition_counts": _fixed_counts(
                [
                    _copy_full_category_transition(pair)
                    for pair in pairs
                ],
                CATEGORY_TRANSITIONS,
            ),
            "mutation_failure_source_counts": _fixed_counts(
                [
                    pair["exact"]["mutation_failure_source"]
                    for pair in pairs
                ],
                protocol.MUTATION_FAILURE_LABELS,
            ),
        },
        "work": {
            metric: _exact_summary([
                _work_value(pair, metric) for pair in pairs
            ])
            for metric in WORK_METRICS
        },
        "wallclock_sec_diagnostic_only": _float_summary([
            _elapsed_value(pair) for pair in pairs
        ]),
    }


def _mean_fraction(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise RuntimeError("seed aggregate 不得含空 proposal stratum")
    return sum(values, Fraction(0, 1)) / len(values)


def _equal_seed_mean_summary(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], Fraction],
) -> dict[str, Any]:
    per_seed = []
    seed_means = []
    for seed, pairs in pairs_by_seed.items():
        mean = _mean_fraction([extractor(pair) for pair in pairs])
        seed_means.append(mean)
        per_seed.append({
            "seed": seed,
            "proposal_count": len(pairs),
            "mean": _fraction_payload(mean),
        })
    return {
        "per_seed": per_seed,
        "equal_seed_distribution": _exact_summary(seed_means),
    }


def _equal_seed_float_mean_summary(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    per_seed = []
    seed_means = []
    for seed, pairs in pairs_by_seed.items():
        values = [float(extractor(pair)) for pair in pairs]
        if not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError("seed wallclock aggregate 无效")
        mean = sum(values) / len(values)
        seed_means.append(mean)
        per_seed.append({
            "seed": seed,
            "proposal_count": len(pairs),
            "mean": mean,
        })
    return {
        "per_seed": per_seed,
        "equal_seed_distribution": _float_summary(seed_means),
    }


def _equal_seed_share_summary(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], str | None],
    axis_labels: Sequence[str],
    selected_label: str,
) -> dict[str, Any]:
    if selected_label not in axis_labels:
        raise ValueError("selected_label 不在 share axis")
    per_seed = []
    defined_shares: list[Fraction] = []
    for seed, pairs in pairs_by_seed.items():
        values = [extractor(pair) for pair in pairs]
        counts = _fixed_counts(values, axis_labels)
        denominator = sum(counts.values())
        share = (
            Fraction(counts[selected_label], denominator)
            if denominator
            else None
        )
        if share is not None:
            defined_shares.append(share)
        per_seed.append({
            "seed": seed,
            "count": counts[selected_label],
            "denominator": denominator,
            "share": _fraction_payload(share) if share is not None else None,
        })
    return {
        "per_seed": per_seed,
        "defined_seed_count": len(defined_shares),
        "undefined_seed_count": len(pairs_by_seed) - len(defined_shares),
        "equal_seed_distribution": _exact_summary(defined_shares),
    }


def _equal_seed_aggregate(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    legs: dict[str, Any] = {}
    for leg in LEG_NAMES:
        legs[leg] = {
            "metric_seed_means": {
                metric: _equal_seed_mean_summary(
                    pairs_by_seed,
                    lambda pair, leg=leg, metric=metric: _exact_value(
                        pair, leg, metric
                    ),
                )
                for metric in EXACT_METRICS
            },
            "category_seed_shares": {
                label: _equal_seed_share_summary(
                    pairs_by_seed,
                    lambda pair, leg=leg: pair["exact"][leg]["category"],
                    protocol.FULL_GAIN_CATEGORIES,
                    label,
                )
                for label in protocol.FULL_GAIN_CATEGORIES
            },
            "curvature_role_seed_shares": {
                label: _equal_seed_share_summary(
                    pairs_by_seed,
                    lambda pair, leg=leg: pair["exact"][leg][
                        "curvature_role"
                    ],
                    protocol.CURVATURE_ROLE_CATEGORIES,
                    label,
                )
                for label in protocol.CURVATURE_ROLE_CATEGORIES
            },
            "cross_label_seed_shares": {
                label: _equal_seed_share_summary(
                    pairs_by_seed,
                    lambda pair, leg=leg: pair["exact"][leg]["cross_label"],
                    protocol.CROSS_LABELS,
                    label,
                )
                for label in protocol.CROSS_LABELS
            },
        }
    return {
        "seed_count": len(pairs_by_seed),
        "seed_order": list(pairs_by_seed),
        "seed_pair_counts": {
            str(seed): len(pairs) for seed, pairs in pairs_by_seed.items()
        },
        "legs": legs,
        "sequential": {
            "copy_mutation_interaction2_seed_means": (
                _equal_seed_mean_summary(
                    pairs_by_seed, _interaction_value
                )
            ),
            "copy_full_gain_sign_transition_seed_shares": {
                label: _equal_seed_share_summary(
                    pairs_by_seed,
                    lambda pair: pair["exact"][
                        "copy_full_gain_sign_transition"
                    ],
                    SIGN_TRANSITIONS,
                    label,
                )
                for label in SIGN_TRANSITIONS
            },
            "copy_full_category_transition_seed_shares": {
                label: _equal_seed_share_summary(
                    pairs_by_seed,
                    _copy_full_category_transition,
                    CATEGORY_TRANSITIONS,
                    label,
                )
                for label in CATEGORY_TRANSITIONS
            },
            "mutation_failure_source_seed_shares": {
                label: _equal_seed_share_summary(
                    pairs_by_seed,
                    lambda pair: pair["exact"]["mutation_failure_source"],
                    protocol.MUTATION_FAILURE_LABELS,
                    label,
                )
                for label in protocol.MUTATION_FAILURE_LABELS
            },
        },
        "work_seed_means": {
            metric: _equal_seed_mean_summary(
                pairs_by_seed,
                lambda pair, metric=metric: _work_value(pair, metric),
            )
            for metric in WORK_METRICS
        },
        "wallclock_sec_diagnostic_only_seed_means": (
            _equal_seed_float_mean_summary(
                pairs_by_seed,
                _elapsed_value,
            )
        ),
    }


def _state_pair_index(
    collection: Mapping[str, Any],
) -> dict[tuple[str, int, str], list[Mapping[str, Any]]]:
    index = {}
    for state in collection["states"]:
        key = (state["dataset"], state["seed"], state["state_group"])
        if key in index:
            raise RuntimeError("proposal collection 出现重复 state stratum")
        index[key] = list(state["pairs"])
    return index


def _phase_groups(phase: str) -> tuple[str, ...]:
    if phase == "initial":
        return ("initial",)
    if phase == "primary":
        return tuple(protocol.PRIMARY_STATE_GROUPS)
    raise ValueError("phase 必须是 initial 或 primary")


def _build_strata(
    collection: Mapping[str, Any], seeds: Sequence[int]
) -> dict[str, Any]:
    index = _state_pair_index(collection)
    dataset_seed_state = []
    dataset_seed_phase = []
    dataset_phase_equal_seed = []

    for dataset in protocol.DATASET_ORDER:
        for seed in seeds:
            for state_group in protocol.STATE_GROUPS:
                pairs = index[(dataset, seed, state_group)]
                dataset_seed_state.append({
                    "dataset": dataset,
                    "seed": seed,
                    "state_group": state_group,
                    "phase": (
                        "initial" if state_group == "initial" else "primary"
                    ),
                    "summary": _summarize_pairs(pairs),
                })

            for phase in PHASES:
                groups = _phase_groups(phase)
                pairs = [
                    pair
                    for state_group in groups
                    for pair in index[(dataset, seed, state_group)]
                ]
                dataset_seed_phase.append({
                    "dataset": dataset,
                    "seed": seed,
                    "phase": phase,
                    "state_groups": list(groups),
                    "summary": _summarize_pairs(pairs),
                })

        for phase in PHASES:
            groups = _phase_groups(phase)
            pairs_by_seed = {
                seed: [
                    pair
                    for state_group in groups
                    for pair in index[(dataset, seed, state_group)]
                ]
                for seed in seeds
            }
            pooled = [
                pair for pairs in pairs_by_seed.values() for pair in pairs
            ]
            dataset_phase_equal_seed.append({
                "dataset": dataset,
                "phase": phase,
                "state_groups": list(groups),
                "pooled_proposals_descriptive_only": _summarize_pairs(pooled),
                "equal_seed_aggregate": _equal_seed_aggregate(pairs_by_seed),
            })

    return {
        "dataset_seed_state": dataset_seed_state,
        "dataset_seed_phase": dataset_seed_phase,
        "dataset_phase_equal_seed": dataset_phase_equal_seed,
    }


def _axis_counts_by_seed(
    collection: Mapping[str, Any],
    dataset: str,
    labels: Sequence[str],
    extractor: Callable[[Mapping[str, Any]], str | None],
) -> dict[int, dict[str, int]]:
    index = _state_pair_index(collection)
    result = {}
    for seed in protocol.FORMAL_SEEDS:
        pairs = [
            pair
            for state_group in protocol.PRIMARY_STATE_GROUPS
            for pair in index[(dataset, seed, state_group)]
        ]
        result[seed] = _fixed_counts(
            [extractor(pair) for pair in pairs], labels
        )
    return result


def _dominance_with_counts(
    counts_by_seed: Mapping[int, Mapping[str, int]],
    labels: Sequence[str],
) -> dict[str, Any]:
    result = protocol.classify_dataset_dominance(counts_by_seed, labels)
    return {
        "axis_labels": list(labels),
        "counts_by_seed": {
            str(seed): dict(counts_by_seed[seed])
            for seed in protocol.FORMAL_SEEDS
        },
        **result,
    }


def _shared_secondary_result(
    dataset_results: Mapping[str, Mapping[str, Any]],
    *,
    applicable: bool,
    not_applicable_reason: str,
) -> dict[str, Any]:
    if not applicable:
        return {
            "status": "not_applicable",
            "reason": not_applicable_reason,
            "shared_label": None,
        }
    rows = [dataset_results[dataset] for dataset in protocol.DATASET_ORDER]
    if any(row["status"] == "insufficient_support" for row in rows):
        return {
            "status": "insufficient_support",
            "reason": None,
            "shared_label": None,
        }
    labels = [
        row["label"] if row["status"] == "supported" else None
        for row in rows
    ]
    if labels[0] is not None and labels[0] == labels[1]:
        return {
            "status": "supported",
            "reason": None,
            "shared_label": labels[0],
        }
    return {
        "status": "no_shared_label",
        "reason": None,
        "shared_label": None,
    }


def _build_formal_dominance(
    collection: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    dataset_results: dict[str, Any] = {}
    geometry_results: dict[str, Any] = {}
    mutation_results: dict[str, Any] = {}
    cross_results: dict[str, Any] = {}

    for dataset in protocol.DATASET_ORDER:
        geometry_counts = _axis_counts_by_seed(
            collection,
            dataset,
            protocol.GEOMETRY_LABELS,
            lambda pair: (
                pair["exact"]["full"]["category"]
                if pair["exact"]["full"]["category"]
                in protocol.GEOMETRY_LABELS
                else None
            ),
        )
        mutation_counts = _axis_counts_by_seed(
            collection,
            dataset,
            protocol.MUTATION_FAILURE_LABELS,
            lambda pair: pair["exact"]["mutation_failure_source"],
        )
        cross_counts = _axis_counts_by_seed(
            collection,
            dataset,
            protocol.CROSS_LABELS,
            lambda pair: pair["exact"]["full"]["cross_label"],
        )
        for seed in protocol.FORMAL_SEEDS:
            if sum(geometry_counts[seed].values()) != sum(
                mutation_counts[seed].values()
            ):
                raise RuntimeError("full negative 与 mutation source 分母不一致")
            if cross_counts[seed]["self_sufficient_overrun"] + (
                cross_counts[seed]["cross_required_overrun"]
            ) != geometry_counts[seed]["curvature_overrun"]:
                raise RuntimeError("curvature 与 self/cross 分母不一致")

        geometry = _dominance_with_counts(
            geometry_counts, protocol.GEOMETRY_LABELS
        )
        mutation = _dominance_with_counts(
            mutation_counts, protocol.MUTATION_FAILURE_LABELS
        )
        if geometry["status"] == "supported" and geometry["label"] == (
            "curvature_overrun"
        ):
            cross = _dominance_with_counts(
                cross_counts, protocol.CROSS_LABELS
            )
        else:
            cross = {
                "axis_labels": list(protocol.CROSS_LABELS),
                "counts_by_seed": {
                    str(seed): dict(cross_counts[seed])
                    for seed in protocol.FORMAL_SEEDS
                },
                "status": "not_applicable_without_dataset_curvature_support",
                "label": None,
                "minimum_per_seed": protocol.MIN_FAILURES_PER_SEED,
            }
        geometry_results[dataset] = geometry
        mutation_results[dataset] = mutation
        cross_results[dataset] = cross
        dataset_results[dataset] = {
            "geometry": geometry,
            "curvature_self_cross": cross,
            "mutation_failure_source": mutation,
        }

    overall_geometry = protocol.shared_geometry_result(geometry_results)
    shared_cross = _shared_secondary_result(
        cross_results,
        applicable=overall_geometry == "shared_curvature_overrun",
        not_applicable_reason=(
            "overall geometry is not shared_curvature_overrun"
        ),
    )
    shared_mutation = _shared_secondary_result(
        mutation_results,
        applicable=True,
        not_applicable_reason="",
    )
    return ({
        "primary_state_groups": list(protocol.PRIMARY_STATE_GROUPS),
        "initial_included": False,
        "minimum_failures_per_seed": protocol.MIN_FAILURES_PER_SEED,
        "required_same_seed_majorities": (
            protocol.REQUIRED_SEED_MAJORITY
        ),
        "dataset_rule": (
            "all five seed denominators >= 10; at least 4/5 strict seed "
            "majorities agree; that label's equal-seed mean share > 0.5"
        ),
        "dataset_results": dataset_results,
        "shared_geometry": overall_geometry,
        "shared_curvature_self_cross": shared_cross,
        "shared_mutation_failure_source": shared_mutation,
    }, overall_geometry)


def scientific_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "evaluation_format": report["evaluation_format"],
        "mode": report["mode"],
        "formal_result_valid": report["formal_result_valid"],
        "mechanism_evidence_emitted": report[
            "mechanism_evidence_emitted"
        ],
        "protocol_sha256": report["protocol_sha256"],
        "evaluator_git_commit": report["evaluator_git_commit"],
        "input_audit": report["input_audit"],
        "runtime_targets": report["runtime_targets"],
        "artifact_identity": report["artifact_identity"],
        "evaluation_boundary": report["evaluation_boundary"],
        "descriptive_statistics_spec": report[
            "descriptive_statistics_spec"
        ],
        "strata": report["strata"],
        "formal_dominance": report["formal_dominance"],
        "formal_overall_result": report["formal_overall_result"],
        "manifest": report["manifest"],
    }


def _validate_evaluation_report(
    report: Mapping[str, Any],
    *,
    mode: str,
    selected_seeds: Sequence[int],
) -> None:
    required_keys = {
        "evaluation_format",
        "status",
        "mode",
        "formal_result_valid",
        "mechanism_evidence_emitted",
        "protocol",
        "protocol_sha256",
        "evaluator_git_commit",
        "git",
        "environment",
        "input_audit",
        "runtime_targets",
        "artifact_paths_diagnostic_only",
        "artifact_identity",
        "evaluation_boundary",
        "descriptive_statistics_spec",
        "strata",
        "formal_dominance",
        "formal_overall_result",
        "manifest",
        "elapsed_sec_diagnostic_only",
        "evaluation_scientific_sha256",
    }
    seeds = tuple(selected_seeds)
    manifest = report.get("manifest", {})
    strata = report.get("strata", {})
    expected_state_count = (
        len(protocol.DATASET_ORDER)
        * len(seeds)
        * len(protocol.STATE_GROUPS)
    )
    expected_pair_count = sum(
        len(seeds)
        * len(protocol.STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    expected_primary_count = sum(
        len(seeds)
        * len(protocol.PRIMARY_STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    if (
        set(report) != required_keys
        or report.get("evaluation_format") != EVALUATION_FORMAT
        or report.get("status") != "complete"
        or report.get("mode") != mode
        or report.get("formal_result_valid") is not (mode == "formal")
        or report.get("mechanism_evidence_emitted") is not (mode == "formal")
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("evaluation_boundary") != EVALUATION_BOUNDARY
        or report.get("descriptive_statistics_spec")
        != DESCRIPTIVE_STATISTICS_SPEC
        or not isinstance(strata, dict)
        or set(strata)
        != {
            "dataset_seed_state",
            "dataset_seed_phase",
            "dataset_phase_equal_seed",
        }
        or len(strata["dataset_seed_state"]) != expected_state_count
        or len(strata["dataset_seed_phase"])
        != len(protocol.DATASET_ORDER) * len(seeds) * len(PHASES)
        or len(strata["dataset_phase_equal_seed"])
        != len(protocol.DATASET_ORDER) * len(PHASES)
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(seeds)
        or manifest.get("state_group_order") != list(protocol.STATE_GROUPS)
        or manifest.get("primary_state_group_order")
        != list(protocol.PRIMARY_STATE_GROUPS)
        or manifest.get("state_summary_count") != expected_state_count
        or manifest.get("pair_count") != expected_pair_count
        or manifest.get("initial_pair_count")
        != expected_pair_count - expected_primary_count
        or manifest.get("primary_pair_count") != expected_primary_count
    ):
        raise RuntimeError("Stage 6A evaluation report 结构/覆盖失败")

    artifact = report.get("artifact_identity")
    if (
        not isinstance(artifact, dict)
        or any(
            not _is_sha256(artifact.get(key))
            for key in (
                "proposal_collection_file_sha256",
                "proposal_collection_scientific_sha256",
                "structural_audit_file_sha256",
                "structural_audit_scientific_sha256",
            )
        )
    ):
        raise RuntimeError("Stage 6A evaluation artifact identity 失败")

    state_rows = strata["dataset_seed_state"]
    expected_state_keys = [
        (dataset, seed, state_group)
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for state_group in protocol.STATE_GROUPS
    ]
    observed_state_keys = [
        (row.get("dataset"), row.get("seed"), row.get("state_group"))
        for row in state_rows
    ]
    if observed_state_keys != expected_state_keys:
        raise RuntimeError("evaluation dataset/seed/state 固定顺序失败")
    if sum(row["summary"]["pair_count"] for row in state_rows) != (
        expected_pair_count
    ):
        raise RuntimeError("evaluation pair coverage 失败")

    if mode == "formal":
        if (
            report.get("formal_overall_result")
            not in protocol.ALLOWED_OVERALL_RESULTS
            or not isinstance(report.get("formal_dominance"), dict)
            or report["formal_dominance"].get("shared_geometry")
            != report.get("formal_overall_result")
        ):
            raise RuntimeError("formal evaluation 结论结构失败")
    elif (
        report.get("formal_dominance") is not None
        or report.get("formal_overall_result") is not None
    ):
        raise RuntimeError("smoke 不得输出正式机制结论")

    observed_scientific_sha = protocol.canonical_sha256(
        scientific_payload(report)
    )
    if report.get("evaluation_scientific_sha256") != (
        observed_scientific_sha
    ):
        raise RuntimeError("Stage 6A evaluation scientific SHA-256 失败")


def _validate_bound_inputs(
    proposal: Mapping[str, Any],
    structural_audit: Mapping[str, Any],
    *,
    mode: str,
    proposal_file_sha256: str,
) -> tuple[int, ...]:
    seeds = _mode_seeds(mode)
    collector._validate_collection_structure(
        proposal,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    auditor._validate_structural_audit_report(
        structural_audit,
        mode=mode,
        selected_seeds=seeds,
    )
    artifact = structural_audit["artifact_identity"]
    proposal_git_commit = proposal.get("git", {}).get("commit")
    if (
        structural_audit.get("checks") != auditor.PASS_CHECKS
        or artifact["proposal_collection_file_sha256"]
        != proposal_file_sha256
        or artifact["proposal_collection_scientific_sha256"]
        != proposal["proposal_scientific_sha256"]
        or artifact["proposal_collection_git_commit"]
        != proposal_git_commit
        or structural_audit.get("input_audit") != proposal.get("input_audit")
        or structural_audit.get("runtime_targets")
        != proposal.get("runtime_targets")
    ):
        raise RuntimeError(
            "structural audit 未通过或未精确绑定本次 proposal collection"
        )
    audit_states = structural_audit["state_audits"]
    proposal_states = proposal["states"]
    for audit_state, proposal_state in zip(audit_states, proposal_states):
        if any(
            audit_state.get(key) != proposal_state.get(key)
            for key in (
                "state_id",
                "source_state_scientific_sha256",
                "source_trajectory_scientific_sha256",
                "current_table_sha256",
            )
        ):
            raise RuntimeError("structural audit state identity 未绑定 proposal")
    return seeds


def build_plan(mode: str) -> dict[str, Any]:
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    seeds = _mode_seeds(mode)
    pair_count = sum(
        len(seeds)
        * len(protocol.STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    return {
        "mode": "plan_only_no_input_read_no_evaluation_no_generation",
        "requested_mode": mode,
        "formal_result_valid": False,
        "mechanism_evidence_emitted": False,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "evaluation_format": EVALUATION_FORMAT,
        "required_inputs": [
            "complete_proposal_collection",
            "passing_structural_audit_bound_to_exact_collection",
        ],
        "dataset_order": list(protocol.DATASET_ORDER),
        "seed_order": list(seeds),
        "state_group_order": list(protocol.STATE_GROUPS),
        "primary_state_group_order": list(protocol.PRIMARY_STATE_GROUPS),
        "pair_count": pair_count,
        "evaluation_boundary": dict(EVALUATION_BOUNDARY),
        "generation_started": False,
        "input_read_started": False,
        "formal_confirmation_consumed": False,
    }


def evaluate(
    mode: str,
    proposal_collection_path: str | Path,
    structural_audit_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_proposal_collection_sha256: str,
    confirmed_structural_audit_sha256: str,
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Create the frozen descriptive report and, only in formal mode, labels."""

    protocol.require_formal_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"evaluation 输出已存在，不覆盖：{output}")

    proposal_path = Path(proposal_collection_path).resolve()
    audit_path = Path(structural_audit_path).resolve()
    proposal_sha = protocol.file_sha256(proposal_path)
    audit_sha = protocol.file_sha256(audit_path)
    if confirmed_proposal_collection_sha256 != proposal_sha:
        raise PermissionError("proposal collection 显式确认 SHA-256 不匹配")
    if confirmed_structural_audit_sha256 != audit_sha:
        raise PermissionError("structural audit 显式确认 SHA-256 不匹配")

    proposal = state_builder._strict_load_json(proposal_path)
    structural_audit = state_builder._strict_load_json(audit_path)
    seeds = _validate_bound_inputs(
        proposal,
        structural_audit,
        mode=mode,
        proposal_file_sha256=proposal_sha,
    )
    if (
        protocol.file_sha256(proposal_path) != proposal_sha
        or protocol.file_sha256(audit_path) != audit_sha
    ):
        raise RuntimeError("evaluation 输入在读取校验期间改变")

    git = state_builder._git_identity(REPOSITORY_ROOT)
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=False,
    )
    if mode == "formal":
        commits = (
            proposal.get("git", {}).get("commit"),
            structural_audit.get("audit_git_commit"),
            structural_audit["artifact_identity"].get(
                "state_library_git_commit"
            ),
            structural_audit["artifact_identity"].get(
                "proposal_collection_git_commit"
            ),
        )
        if (
            any(commit != git["commit"] for commit in commits)
            or proposal.get("git", {}).get(
                "worktree_clean_including_untracked"
            )
            is not True
            or structural_audit.get("git", {}).get(
                "worktree_clean_including_untracked"
            )
            is not True
        ):
            raise RuntimeError(
                "formal evaluator 要求 state/proposal/audit/current "
                "同一 clean commit"
            )

    started = time.perf_counter()
    strata = _build_strata(proposal, seeds)
    if mode == "formal":
        formal_dominance, overall_result = _build_formal_dominance(proposal)
    else:
        formal_dominance = None
        overall_result = None

    artifact_identity = {
        "proposal_collection_file_sha256": proposal_sha,
        "proposal_collection_scientific_sha256": proposal[
            "proposal_scientific_sha256"
        ],
        "proposal_collection_git_commit": proposal.get("git", {}).get(
            "commit"
        ),
        "structural_audit_file_sha256": audit_sha,
        "structural_audit_scientific_sha256": structural_audit[
            "structural_audit_scientific_sha256"
        ],
        "structural_audit_git_commit": structural_audit[
            "audit_git_commit"
        ],
        "structural_audit_bound_artifact_identity": structural_audit[
            "artifact_identity"
        ],
    }
    expected_pair_count = sum(
        len(seeds)
        * len(protocol.STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    primary_pair_count = sum(
        len(seeds)
        * len(protocol.PRIMARY_STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    report = _json_value({
        "evaluation_format": EVALUATION_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "mechanism_evidence_emitted": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "evaluator_git_commit": git["commit"],
        "git": git,
        "environment": environment,
        "input_audit": proposal["input_audit"],
        "runtime_targets": proposal["runtime_targets"],
        "artifact_paths_diagnostic_only": {
            "proposal_collection": str(proposal_path),
            "structural_audit": str(audit_path),
        },
        "artifact_identity": artifact_identity,
        "evaluation_boundary": dict(EVALUATION_BOUNDARY),
        "descriptive_statistics_spec": dict(DESCRIPTIVE_STATISTICS_SPEC),
        "strata": strata,
        "formal_dominance": formal_dominance,
        "formal_overall_result": overall_result,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "primary_state_group_order": list(
                protocol.PRIMARY_STATE_GROUPS
            ),
            "phase_order": list(PHASES),
            "leg_order": list(LEG_NAMES),
            "exact_metric_order": list(EXACT_METRICS),
            "work_metric_order": list(WORK_METRICS),
            "copy_full_category_transition_order": list(
                CATEGORY_TRANSITIONS
            ),
            "state_summary_count": (
                len(protocol.DATASET_ORDER)
                * len(seeds)
                * len(protocol.STATE_GROUPS)
            ),
            "dataset_seed_phase_summary_count": (
                len(protocol.DATASET_ORDER) * len(seeds) * len(PHASES)
            ),
            "dataset_phase_equal_seed_summary_count": (
                len(protocol.DATASET_ORDER) * len(PHASES)
            ),
            "pair_count": expected_pair_count,
            "initial_pair_count": expected_pair_count - primary_pair_count,
            "primary_pair_count": primary_pair_count,
            "source_pair_ids_in_fixed_order_sha256": (
                protocol.canonical_sha256(
                    proposal["manifest"]["pair_ids_in_fixed_order"]
                )
            ),
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    })
    report["evaluation_scientific_sha256"] = protocol.canonical_sha256(
        scientific_payload(report)
    )
    _validate_evaluation_report(
        report, mode=mode, selected_seeds=seeds
    )

    if (
        protocol.file_sha256(proposal_path) != proposal_sha
        or protocol.file_sha256(audit_path) != audit_sha
    ):
        raise RuntimeError("evaluation 输入在汇总期间改变")
    if mode == "formal" and state_builder._git_identity(
        REPOSITORY_ROOT
    ) != git:
        raise RuntimeError("formal evaluation 期间 git identity 改变")
    published = state_builder._exclusive_write_json(output, report)
    print(f"Stage 6A frozen evaluation：{published}", flush=True)
    return published, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    evaluate_parser.add_argument("--proposal-collection", required=True)
    evaluate_parser.add_argument("--structural-audit", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument(
        "--confirmed-proposal-collection-sha256", required=True
    )
    evaluate_parser.add_argument(
        "--confirmed-structural-audit-sha256", required=True
    )
    evaluate_parser.add_argument("--confirmed-protocol-sha256")
    evaluate_parser.add_argument("--confirmed-execution-commit")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(
            build_plan(args.mode),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    evaluate(
        args.mode,
        args.proposal_collection,
        args.structural_audit,
        args.output,
        confirmed_proposal_collection_sha256=(
            args.confirmed_proposal_collection_sha256
        ),
        confirmed_structural_audit_sha256=(
            args.confirmed_structural_audit_sha256
        ),
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
