"""Issue #53 P=6 未见轨迹冻结 manifest 的纯契约测试。"""

from __future__ import annotations

import copy
import inspect
import json

import pytest

from scripts import issue53_p6_unseen_protocol as protocol

EXPECTED_PROTOCOL_SHA256 = (
    "759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317"
)


def _reference_counts(family):
    return {tuple(row["state"]): row["count"] for row in family["reference_multiset"]}


def _query_order(family):
    return [
        tuple(
            (condition["attribute"], condition["value"])
            for condition in query["conditions"]
        )
        for query in family["ordered_queries"]
    ]


def test_u1_reference_counts_query_order_and_targets_recompute_exactly():
    family = protocol.family_manifests()[0]

    assert family["family"] == "binary_chain_4"
    assert family["attribute_order"] == ["a", "b", "c", "d"]
    assert _reference_counts(family) == {
        (0, 0, 0, 0): 6,
        (0, 0, 0, 1): 2,
        (0, 0, 1, 0): 2,
        (0, 0, 1, 1): 2,
        (1, 1, 0, 0): 2,
        (1, 1, 0, 1): 2,
        (1, 1, 1, 0): 2,
        (1, 1, 1, 1): 6,
        (0, 1, 0, 1): 2,
        (1, 0, 1, 0): 2,
        (0, 1, 1, 0): 2,
        (1, 0, 0, 1): 2,
    }
    assert _query_order(family) == [
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
    ]
    assert protocol.recompute_family_arithmetic(family) == {
        "n_records": 32,
        "query_count": 11,
        "ordered_targets": [16, 16, 16, 16, 12, 10, 8, 10, 8, 6, 6],
    }
    assert family["family_identity_sha256"] == (
        "c47200c0b68c6c3bcf4818b7b9322f85666584eaa1459d94a19d216642f447ee"
    )


def test_u2_reference_counts_query_order_and_targets_recompute_exactly():
    family = protocol.family_manifests()[1]

    assert family["family"] == "mixed_2x3x2"
    assert family["attribute_order"] == ["x", "y", "z"]
    assert _reference_counts(family) == {
        (0, 0, 0): 6,
        (0, 0, 1): 2,
        (0, 1, 0): 3,
        (0, 1, 1): 1,
        (0, 2, 0): 1,
        (0, 2, 1): 5,
        (1, 0, 0): 2,
        (1, 0, 1): 4,
        (1, 1, 0): 1,
        (1, 1, 1): 5,
        (1, 2, 0): 4,
        (1, 2, 1): 2,
    }
    assert _query_order(family) == [
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
    ]
    assert protocol.recompute_family_arithmetic(family) == {
        "n_records": 36,
        "query_count": 15,
        "ordered_targets": [
            18,
            14,
            10,
            12,
            19,
            6,
            6,
            6,
            6,
            6,
            7,
            11,
            4,
            5,
            2,
        ],
    }
    assert family["family_identity_sha256"] == (
        "db3af48d083e1e4905a16362b63ba4bbbe7c55045efd3ae6e6a580f82a58bbab"
    )


@pytest.mark.parametrize("mutation", ["count", "target", "query_order"])
def test_family_arithmetic_and_identity_fail_closed(mutation):
    family = copy.deepcopy(protocol.family_manifests()[0])
    if mutation == "count":
        family["reference_multiset"][0]["count"] += 1
    elif mutation == "target":
        family["ordered_targets"][4] += 1
    else:
        family["ordered_queries"][4], family["ordered_queries"][5] = (
            family["ordered_queries"][5],
            family["ordered_queries"][4],
        )
        family["ordered_targets"][4], family["ordered_targets"][5] = (
            family["ordered_targets"][5],
            family["ordered_targets"][4],
        )

    with pytest.raises(ValueError):
        protocol.recompute_family_arithmetic(family)


def test_primary_case_matrix_is_exactly_the_frozen_twelve_cases():
    cases = protocol.primary_case_matrix()
    observed = [
        (
            case["family"],
            case["n_records"],
            case["seed"],
            case["rho"],
            case["patience_ticks"],
            case["n_rounds"],
            case["candidate_budget"],
        )
        for case in cases
    ]
    expected = [
        (family, n_records, seed, rho, 6, raw_cap, raw_cap)
        for family, n_records in (
            ("binary_chain_4", 32),
            ("mixed_2x3x2", 36),
        )
        for seed in (20260819, 20260820, 20260821)
        for rho, raw_cap in ((1.0, 60), (0.25, 240))
    ]

    assert observed == expected
    assert len(cases) == 12
    assert len({case["case_id"] for case in cases}) == 12
    assert cases[0]["case_id"] == (
        "primary__binary_chain_4__seed_20260819__rho_1p0__p_6"
    )
    assert cases[-1]["case_id"] == (
        "primary__mixed_2x3x2__seed_20260821__rho_0p25__p_6"
    )


def test_c_resource_mapping_is_exact_and_unknown_rho_fails_closed():
    assert protocol.resource_cap_for_rho(1.0) == {
        "rho": 1.0,
        "expected_normalized_work_cap": 60.0,
        "n_rounds": 60,
        "candidate_budget": 60,
    }
    assert protocol.resource_cap_for_rho(0.25) == {
        "rho": 0.25,
        "expected_normalized_work_cap": 60.0,
        "n_rounds": 240,
        "candidate_budget": 240,
    }
    assert protocol.resource_cap_for_rho(1) == protocol.resource_cap_for_rho(1.0)

    for invalid in (True, None, "1.0", 0.0, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            protocol.resource_cap_for_rho(invalid)


def test_only_two_preregistered_fallback_branches_exist():
    quality = protocol.fallback_case_matrix("quality_only_failure")
    compute = protocol.fallback_case_matrix("compute_only_failure")

    assert len(quality) == len(compute) == 12
    assert {case["seed"] for case in quality} == {
        20260822,
        20260823,
        20260824,
    }
    assert {case["seed"] for case in compute} == {
        20260822,
        20260823,
        20260824,
    }
    assert {case["patience_ticks"] for case in quality} == {12}
    assert {case["patience_ticks"] for case in compute} == {4}
    assert set(protocol.PRIMARY_SEEDS).isdisjoint(protocol.FALLBACK_SEEDS)
    assert set(protocol.FALLBACK_SEEDS).isdisjoint(protocol.EXCLUDED_DEVELOPMENT_SEEDS)
    with pytest.raises(ValueError):
        protocol.fallback_case_matrix("try_another_p")


def test_manifest_freezes_gate_free_terminal_current_contract_and_gates():
    manifest = protocol.frozen_protocol_manifest()

    assert manifest["generator"] == {
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
    }
    assert manifest["online_stopping"]["priority"] == ["A", "B", "C"]
    assert manifest["online_stopping"]["output_identity"] == (
        "terminal_current_at_trigger"
    )
    assert manifest["online_stopping"]["historical_best_role"] == (
        "progress_clock_and_diagnostic_only"
    )
    assert manifest["online_stopping"]["l1_used_online"] is False
    assert manifest["online_stopping"]["reference_table_used_online"] is False
    assert manifest["shadow_continuation"]["work_offsets"] == [6, 12]
    assert manifest["acceptance"] == {
        "normal_completion_reasons": ["fit_target_reached", "early_stopped"],
        "normal_completion_minimum_count": 10,
        "resource_cap_maximum_count": 2,
        "b_case_minimum_count": 6,
        "checkpoint_coverage_minimum_fraction": 0.80,
        "median_delta_l1_maximum_each_checkpoint": 0.01,
        "large_degradation_definition": ("delta_l1_strictly_greater_than_0.02"),
        "large_degradation_maximum_fraction_each_checkpoint": 0.25,
        "per_family_median_delta_l1_maximum_each_checkpoint": 0.02,
        "saving_12_definition": "12/(stop_work+12)",
        "saving_12_population": "b_cases_with_observed_plus_12_checkpoint",
        "median_saving_12_minimum": 0.30,
        "all_conditions_required": True,
    }


def test_fallback_routing_is_frozen_without_an_open_p_search():
    fallback = protocol.frozen_protocol_manifest()["fallback"]

    assert fallback["branches_are_mutually_exclusive"] is True
    assert fallback["maximum_fallback_attempts"] == 1
    assert fallback["third_patience_candidate_allowed"] is False
    assert fallback["quality_only_failure_trigger"] == (
        "quality_fails_and_compute_and_evidence_pass"
    )
    assert fallback["compute_only_failure_trigger"] == (
        "compute_fails_and_quality_and_evidence_pass"
    )
    assert fallback["quality_and_compute_failure_action"] == ("reject_b_and_redesign")
    assert fallback["opposite_family_direction_action"] == ("reject_b_and_redesign")
    assert fallback["insufficient_evidence_trigger"] == (
        "too_many_c_or_fewer_than_6_b_or_checkpoint_coverage_fails"
    )
    assert fallback["insufficient_evidence_action"] == (
        "review_c_and_observation_range_without_changing_p"
    )
    assert set(fallback["branches"]) == {
        "quality_only_failure",
        "compute_only_failure",
    }


def test_protocol_hash_is_exact_stable_and_manifest_calls_are_fresh():
    first = protocol.frozen_protocol_manifest()
    second = protocol.frozen_protocol_manifest()

    assert first == second
    assert protocol.FROZEN_PROTOCOL_SHA256 == EXPECTED_PROTOCOL_SHA256
    assert protocol.protocol_sha256() == EXPECTED_PROTOCOL_SHA256
    assert protocol.assert_frozen_protocol_identity() == EXPECTED_PROTOCOL_SHA256
    first["primary"]["cases"][0]["seed"] = -1
    first["families"][0]["ordered_targets"][0] = -1
    assert protocol.frozen_protocol_manifest() == second
    assert protocol.protocol_sha256() == EXPECTED_PROTOCOL_SHA256
    json.dumps(second, allow_nan=False)


def test_plan_entry_cannot_run_generation_or_access_unseen_results():
    plan = protocol.build_plan()
    module_source = inspect.getsource(protocol)

    assert plan["mode"] == "plan_only_no_generation_or_unseen_result_access"
    assert plan["primary_case_count"] == 12
    assert plan["fallback_case_count_if_triggered"] == 12
    assert plan["unseen_seed_results_accessed"] is False
    assert plan["generation_started"] is False
    assert plan["execution_authorized_by_this_command"] is False
    assert list(inspect.signature(protocol.build_plan).parameters) == []
    assert "from table_diffevo" not in module_source
    assert "import table_diffevo" not in module_source
    assert not hasattr(protocol, "run_evolution")
