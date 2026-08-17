#!/usr/bin/env python
"""Issue #53 P=6 未见轨迹质量—计算验收的结果前冻结协议。

本模块只构造并校验公开人工 family、case 矩阵、资源护栏和验收元数据。
它不导入生成器、不生成轨迹，也不读取项目数据；命令行只能打印计划。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

PROTOCOL_VERSION = "issue53-p6-unseen-quality-compute-v1"
FROZEN_PROTOCOL_SHA256 = (
    "759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317"
)

PRIMARY_PATIENCE_TICKS = 6
PRIMARY_SEEDS = (20260819, 20260820, 20260821)
FALLBACK_SEEDS = (20260822, 20260823, 20260824)
EXCLUDED_DEVELOPMENT_SEEDS = (20260816, 20260817, 20260818)
RHOS = (1.0, 0.25)
SHADOW_WORK_OFFSETS = (6, 12)
EXPECTED_NORMALIZED_WORK_CAP = 60.0

_RESOURCE_CAPS = (
    (1.0, 60),
    (0.25, 240),
)

_FALLBACK_BRANCHES = (
    ("quality_only_failure", 12),
    ("compute_only_failure", 4),
)

_FAMILY_DEFINITIONS = (
    (
        "binary_chain_4",
        (
            ("a", (0, 1)),
            ("b", (0, 1)),
            ("c", (0, 1)),
            ("d", (0, 1)),
        ),
        (
            ((0, 0, 0, 0), 6),
            ((0, 0, 0, 1), 2),
            ((0, 0, 1, 0), 2),
            ((0, 0, 1, 1), 2),
            ((1, 1, 0, 0), 2),
            ((1, 1, 0, 1), 2),
            ((1, 1, 1, 0), 2),
            ((1, 1, 1, 1), 6),
            ((0, 1, 0, 1), 2),
            ((1, 0, 1, 0), 2),
            ((0, 1, 1, 0), 2),
            ((1, 0, 0, 1), 2),
        ),
        (
            (("a", 1),),
            (("b", 1),),
            (("c", 1),),
            (("d", 1),),
            (("a", 1), ("b", 1)),
            (("b", 1), ("c", 1)),
            (("c", 1), ("d", 1)),
            (("a", 1), ("d", 1)),
            (("a", 1), ("b", 1), ("c", 1)),
            (("b", 1), ("c", 1), ("d", 1)),
            (("a", 1), ("b", 1), ("c", 1), ("d", 1)),
        ),
        (16, 16, 16, 16, 12, 10, 8, 10, 8, 6, 6),
    ),
    (
        "mixed_2x3x2",
        (
            ("x", (0, 1)),
            ("y", (0, 1, 2)),
            ("z", (0, 1)),
        ),
        (
            ((0, 0, 0), 6),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 1),
            ((0, 2, 0), 1),
            ((0, 2, 1), 5),
            ((1, 0, 0), 2),
            ((1, 0, 1), 4),
            ((1, 1, 0), 1),
            ((1, 1, 1), 5),
            ((1, 2, 0), 4),
            ((1, 2, 1), 2),
        ),
        (
            (("x", 1),),
            (("y", 0),),
            (("y", 1),),
            (("y", 2),),
            (("z", 1),),
            (("x", 1), ("y", 0)),
            (("x", 1), ("y", 1)),
            (("x", 1), ("y", 2)),
            (("y", 0), ("z", 1)),
            (("y", 1), ("z", 1)),
            (("y", 2), ("z", 1)),
            (("x", 1), ("z", 1)),
            (("x", 1), ("y", 0), ("z", 1)),
            (("x", 1), ("y", 1), ("z", 1)),
            (("x", 1), ("y", 2), ("z", 1)),
        ),
        (18, 14, 10, 12, 19, 6, 6, 6, 6, 6, 7, 11, 4, 5, 2),
    ),
)


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def _validate_exact_keys(
    value: Any,
    expected: set[str],
    name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是对象")
    observed = set(value)
    if observed != expected:
        raise ValueError(
            f"{name} 字段不一致；missing="
            f"{sorted(expected - observed)}，unknown="
            f"{sorted(observed - expected)}"
        )


def _strict_value_key(value: Any) -> bytes:
    try:
        return _strict_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("类别值必须可严格 JSON 序列化") from exc


def recompute_family_arithmetic(family: Mapping[str, Any]) -> dict[str, Any]:
    """从公开 reference multiset 重新计算 N 和 ordered target。"""

    if not isinstance(family, Mapping):
        raise TypeError("family 必须是对象")
    schema = family.get("schema")
    if not isinstance(schema, Mapping):
        raise TypeError("family.schema 必须是对象")
    _validate_exact_keys(schema, {"attributes"}, "family.schema")
    attributes = schema["attributes"]
    if not isinstance(attributes, list) or not attributes:
        raise ValueError("family.schema.attributes 必须是非空列表")

    attribute_order: list[str] = []
    domains: dict[str, set[bytes]] = {}
    for index, attribute in enumerate(attributes):
        name = f"family.schema.attributes[{index}]"
        _validate_exact_keys(
            attribute,
            {"name", "type", "values"},
            name,
        )
        attribute_name = attribute["name"]
        if not isinstance(attribute_name, str) or not attribute_name:
            raise ValueError(f"{name}.name 必须是非空字符串")
        if attribute_name in domains:
            raise ValueError("schema attribute 名称不得重复")
        if attribute["type"] != "categorical":
            raise ValueError("本协议 family 只允许 categorical attribute")
        values = attribute["values"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}.values 必须是非空列表")
        value_keys = [_strict_value_key(value) for value in values]
        if len(set(value_keys)) != len(value_keys):
            raise ValueError(f"{name}.values 不得重复")
        attribute_order.append(attribute_name)
        domains[attribute_name] = set(value_keys)

    declared_order = family.get("attribute_order")
    if declared_order != attribute_order:
        raise ValueError("family.attribute_order 与 schema 顺序不一致")

    reference = family.get("reference_multiset")
    if not isinstance(reference, list) or not reference:
        raise ValueError("family.reference_multiset 必须是非空列表")
    materialized_rows: list[tuple[dict[str, Any], int]] = []
    seen_states: set[bytes] = set()
    n_records = 0
    for index, row in enumerate(reference):
        name = f"family.reference_multiset[{index}]"
        _validate_exact_keys(row, {"state", "count"}, name)
        state = row["state"]
        count = row["count"]
        if not isinstance(state, list) or len(state) != len(attribute_order):
            raise ValueError(f"{name}.state 维度与 schema 不一致")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"{name}.count 必须是正整数")
        state_key = _strict_json_bytes(state)
        if state_key in seen_states:
            raise ValueError("reference state 不得重复")
        seen_states.add(state_key)
        for attribute_name, value in zip(attribute_order, state, strict=True):
            if _strict_value_key(value) not in domains[attribute_name]:
                raise ValueError(f"{name}.state 的 {attribute_name} 值不在公开 domain")
        row_values = dict(zip(attribute_order, state, strict=True))
        materialized_rows.append((row_values, count))
        n_records += count

    queries = family.get("ordered_queries")
    if not isinstance(queries, list) or not queries:
        raise ValueError("family.ordered_queries 必须是非空列表")
    targets: list[int] = []
    query_signatures: set[bytes] = set()
    for query_index, query in enumerate(queries):
        query_name = f"family.ordered_queries[{query_index}]"
        _validate_exact_keys(query, {"conditions"}, query_name)
        conditions = query["conditions"]
        if not isinstance(conditions, list) or not conditions:
            raise ValueError(f"{query_name}.conditions 必须是非空列表")
        conditioned_attributes: set[str] = set()
        for condition_index, condition in enumerate(conditions):
            condition_name = f"{query_name}.conditions[{condition_index}]"
            _validate_exact_keys(
                condition,
                {"attribute", "operator", "value"},
                condition_name,
            )
            attribute_name = condition["attribute"]
            if attribute_name not in domains:
                raise ValueError(f"{condition_name}.attribute 未在 schema 中声明")
            if attribute_name in conditioned_attributes:
                raise ValueError("单个 query 不得重复约束同一 attribute")
            conditioned_attributes.add(attribute_name)
            if condition["operator"] != "==":
                raise ValueError("本协议 family query 只允许 ==")
            condition_value_key = _strict_value_key(condition["value"])
            if condition_value_key not in domains[attribute_name]:
                raise ValueError(f"{condition_name}.value 不在公开 domain")
        query_signature = _strict_json_bytes(query)
        if query_signature in query_signatures:
            raise ValueError("ordered queries 不得重复")
        query_signatures.add(query_signature)
        targets.append(
            sum(
                count
                for row_values, count in materialized_rows
                if all(
                    row_values[condition["attribute"]] == condition["value"]
                    for condition in conditions
                )
            )
        )

    declared_n_records = family.get("n_records")
    if declared_n_records != n_records:
        raise ValueError("family.n_records 与 reference count 总和不一致")
    declared_query_count = family.get("query_count")
    if declared_query_count != len(queries):
        raise ValueError("family.query_count 与 ordered queries 数量不一致")
    if family.get("ordered_targets") != targets:
        raise ValueError("family.ordered_targets 与 reference 复算结果不一致")

    claimed_identity = family.get("family_identity_sha256")
    if claimed_identity is not None:
        identity_payload = dict(family)
        del identity_payload["family_identity_sha256"]
        if claimed_identity != _sha256_json(identity_payload):
            raise ValueError("family_identity_sha256 与 family 内容不一致")

    return {
        "n_records": n_records,
        "query_count": len(queries),
        "ordered_targets": targets,
    }


def _query(conditions: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {
        "conditions": [
            {"attribute": attribute, "operator": "==", "value": value}
            for attribute, value in conditions
        ]
    }


def _build_family_manifest(definition: tuple[Any, ...]) -> dict[str, Any]:
    name, domains, reference, queries, expected_targets = definition
    family = {
        "family": name,
        "schema": {
            "attributes": [
                {
                    "name": attribute,
                    "type": "categorical",
                    "values": list(values),
                }
                for attribute, values in domains
            ]
        },
        "attribute_order": [attribute for attribute, _ in domains],
        "reference_multiset": [
            {"state": list(state), "count": count} for state, count in reference
        ],
        "n_records": sum(count for _, count in reference),
        "ordered_queries": [_query(conditions) for conditions in queries],
        "query_count": len(queries),
        "ordered_targets": list(expected_targets),
        "reference_use": "target_materialization_before_generation_only",
    }
    arithmetic = recompute_family_arithmetic(family)
    if arithmetic["ordered_targets"] != list(expected_targets):
        raise RuntimeError(f"{name} 的冻结 target 与 reference 复算不一致")
    family["family_identity_sha256"] = _sha256_json(family)
    recompute_family_arithmetic(family)
    return family


def family_manifests() -> list[dict[str, Any]]:
    """Return fresh, fully validated U1/U2 manifests in frozen order."""

    return [_build_family_manifest(definition) for definition in _FAMILY_DEFINITIONS]


def resource_cap_for_rho(rho: Any) -> dict[str, Any]:
    """Translate one frozen rho to the pure C guard; reject all other values."""

    if (
        isinstance(rho, bool)
        or not isinstance(rho, (int, float))
        or not math.isfinite(rho)
    ):
        raise ValueError("rho 必须是冻结集合中的有限数值")
    normalized_rho = float(rho)
    for frozen_rho, raw_cap in _RESOURCE_CAPS:
        if normalized_rho == frozen_rho:
            return {
                "rho": frozen_rho,
                "expected_normalized_work_cap": EXPECTED_NORMALIZED_WORK_CAP,
                "n_rounds": raw_cap,
                "candidate_budget": raw_cap,
            }
    raise ValueError(f"rho 不在冻结集合 {RHOS} 中")


def _rho_token(rho: float) -> str:
    if rho == 1.0:
        return "1p0"
    if rho == 0.25:
        return "0p25"
    raise ValueError("未知 rho")


def _case_matrix(
    *,
    cohort: str,
    seeds: Sequence[int],
    patience_ticks: int,
) -> list[dict[str, Any]]:
    families = family_manifests()
    cases = []
    for family in families:
        for seed in seeds:
            for rho in RHOS:
                cap = resource_cap_for_rho(rho)
                cases.append(
                    {
                        "case_id": (
                            f"{cohort}__{family['family']}__seed_{seed}__"
                            f"rho_{_rho_token(rho)}__p_{patience_ticks}"
                        ),
                        "cohort": cohort,
                        "family": family["family"],
                        "n_records": family["n_records"],
                        "seed": seed,
                        "rho": rho,
                        "patience_ticks": patience_ticks,
                        "n_rounds": cap["n_rounds"],
                        "candidate_budget": cap["candidate_budget"],
                    }
                )
    return cases


def primary_case_matrix() -> list[dict[str, Any]]:
    """Return the only frozen P=6 primary matrix (12 cases)."""

    return _case_matrix(
        cohort="primary",
        seeds=PRIMARY_SEEDS,
        patience_ticks=PRIMARY_PATIENCE_TICKS,
    )


def fallback_case_matrix(branch: str) -> list[dict[str, Any]]:
    """Return one preregistered fallback matrix; arbitrary P is impossible."""

    patience_by_branch = dict(_FALLBACK_BRANCHES)
    if branch not in patience_by_branch:
        raise ValueError("branch 必须是 quality_only_failure 或 compute_only_failure")
    return _case_matrix(
        cohort=f"fallback_{branch}",
        seeds=FALLBACK_SEEDS,
        patience_ticks=patience_by_branch[branch],
    )


def frozen_protocol_manifest() -> dict[str, Any]:
    """Build the complete result-blind protocol manifest."""

    families = family_manifests()
    primary_cases = primary_case_matrix()
    fallback_branches = {
        branch: {
            "patience_ticks": patience,
            "seeds": list(FALLBACK_SEEDS),
            "expected_case_count": len(fallback_case_matrix(branch)),
            "cases": fallback_case_matrix(branch),
        }
        for branch, patience in _FALLBACK_BRANCHES
    }
    manifest = {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "purpose": "p6_unseen_terminal_current_quality_compute_acceptance",
        "freeze_status": "frozen_before_unseen_seed_generation",
        "families": families,
        "scope": {
            "primary_seeds": list(PRIMARY_SEEDS),
            "fallback_seeds": list(FALLBACK_SEEDS),
            "excluded_seen_development_seeds": list(EXCLUDED_DEVELOPMENT_SEEDS),
            "rhos": list(RHOS),
            "primary_expected_case_count": 12,
            "reads_project_dataset": False,
            "reads_saved_real_trajectory": False,
            "uses_gpu": False,
            "consumes_privacy_budget": False,
        },
        "generator": {
            "init_method": "random",
            "eta": 0.45,
            "mu": 0.02,
            "distance_mode": "geometric",
            "alpha_schedule_mode": "fixed",
            "fixed_alpha": 6.0,
            "residual_directed_diffusion": True,
            "diffusion_direction_strength": 0.8,
            "diffusion_direction_normalization": "fixed",
            "diffusion_direction_reference_scale": 1.25,
            "diffusion_direction_logit_clip": 9.0,
            "factorized_gibbs_sweeps": 0,
            "residual_self_cooling": None,
            "tol": "positive_infinity",
            "max_retries": 0,
            "device": "numpy",
            "horizon_invariant": True,
            "stop_on_exact_residual": True,
            "return_final_table": True,
            "record_transition_clocks": True,
        },
        "online_stopping": {
            "primary_patience_ticks": PRIMARY_PATIENCE_TICKS,
            "natural_work": "cumulative_applied_participating_rows/n_records",
            "work_tick": "floor(natural_work)",
            "priority": ["A", "B", "C"],
            "A": {
                "condition": "current_squared_loss_equals_zero",
                "reason": "fit_target_reached",
                "inner_complete": True,
            },
            "B": {
                "condition": "P_completed_ticks_without_strict_best_refresh",
                "reason": "early_stopped",
                "inner_complete": True,
            },
            "C": {
                "condition": "n_rounds_or_candidate_budget_reached",
                "reason": "resource_cap_reached",
                "inner_complete": False,
            },
            "output_identity": "terminal_current_at_trigger",
            "historical_best_role": "progress_clock_and_diagnostic_only",
            "l1_used_online": False,
            "reference_table_used_online": False,
        },
        "resource_guard": {
            "role": "hang_prevention_only_not_quality_endpoint",
            "expected_normalized_work_cap": EXPECTED_NORMALIZED_WORK_CAP,
            "mapping": [resource_cap_for_rho(rho) for rho in RHOS],
            "actual_terminal_work_may_differ": True,
        },
        "primary": {
            "patience_ticks": PRIMARY_PATIENCE_TICKS,
            "seeds": list(PRIMARY_SEEDS),
            "expected_case_count": len(primary_cases),
            "cases": primary_cases,
        },
        "shadow_continuation": {
            "b_only": True,
            "work_offsets": list(SHADOW_WORK_OFFSETS),
            "state_rule": "first_real_current_state_at_or_beyond_target_work",
            "interpolation_allowed": False,
            "historical_best_allowed": False,
            "right_censor_if_unobserved_before_c": True,
            "c_terminal_imputation_allowed": False,
            "delta_l1_definition": "l1_at_b_minus_l1_at_continuation",
            "positive_delta_means_early_stop_is_worse": True,
        },
        "acceptance": {
            "normal_completion_reasons": [
                "fit_target_reached",
                "early_stopped",
            ],
            "normal_completion_minimum_count": 10,
            "resource_cap_maximum_count": 2,
            "b_case_minimum_count": 6,
            "checkpoint_coverage_minimum_fraction": 0.80,
            "median_delta_l1_maximum_each_checkpoint": 0.01,
            "large_degradation_definition": "delta_l1_strictly_greater_than_0.02",
            "large_degradation_maximum_fraction_each_checkpoint": 0.25,
            "per_family_median_delta_l1_maximum_each_checkpoint": 0.02,
            "saving_12_definition": "12/(stop_work+12)",
            "saving_12_population": ("b_cases_with_observed_plus_12_checkpoint"),
            "median_saving_12_minimum": 0.30,
            "all_conditions_required": True,
        },
        "fallback": {
            "branches_are_mutually_exclusive": True,
            "maximum_fallback_attempts": 1,
            "third_patience_candidate_allowed": False,
            "branches": fallback_branches,
            "quality_only_failure_trigger": (
                "quality_fails_and_compute_and_evidence_pass"
            ),
            "compute_only_failure_trigger": (
                "compute_fails_and_quality_and_evidence_pass"
            ),
            "quality_and_compute_failure_action": "reject_b_and_redesign",
            "opposite_family_direction_action": "reject_b_and_redesign",
            "insufficient_evidence_trigger": (
                "too_many_c_or_fewer_than_6_b_or_checkpoint_coverage_fails"
            ),
            "insufficient_evidence_action": (
                "review_c_and_observation_range_without_changing_p"
            ),
        },
        "execution_safety": {
            "this_entry_computes_targets_from_public_multisets": True,
            "this_entry_expands_multisets_to_n_row_tables": False,
            "this_entry_runs_generation": False,
            "this_entry_accesses_unseen_seed_results": False,
            "formal_run_requires_separate_user_authorization": True,
        },
    }
    _strict_json_bytes(manifest)
    return manifest


def protocol_sha256() -> str:
    return _sha256_json(frozen_protocol_manifest())


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("Issue53 P6 未见轨迹协议内容与结果前冻结 SHA-256 不一致")
    return observed


def build_plan() -> dict[str, Any]:
    """Return the inspectable plan; no execution mode exists in this module."""

    protocol = frozen_protocol_manifest()
    plan = {
        "contract_version": PROTOCOL_VERSION,
        "protocol_sha256": assert_frozen_protocol_identity(),
        "mode": "plan_only_no_generation_or_unseen_result_access",
        "protocol": protocol,
        "primary_case_count": len(primary_case_matrix()),
        "primary_cases": primary_case_matrix(),
        "fallback_case_count_if_triggered": 12,
        "unseen_seed_results_accessed": False,
        "generation_started": False,
        "execution_authorized_by_this_command": False,
    }
    _strict_json_bytes(plan)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan",), default="plan")
    parser.parse_args(argv)
    print(
        json.dumps(
            build_plan(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
