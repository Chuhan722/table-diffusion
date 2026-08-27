"""Frozen, result-blind contracts for Issue #53 Stage 5."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_issue53_stage5_kernel_ab as auditor
from scripts import evaluate_issue53_stage5_kernel_ab as evaluator
from scripts import issue53_stage5_protocol as protocol
from scripts import run_issue53_stage5_kernel_ab as runner


ROOT = Path(__file__).resolve().parents[1]


def test_protocol_document_manifest_and_plan_are_frozen() -> None:
    plan = protocol.build_plan()

    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    assert protocol.file_sha256(ROOT / protocol.PROTOCOL_DOC) == (
        protocol.PROTOCOL_DOC_SHA256
    )
    assert plan["protocol_sha256"] == protocol.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    assert plan["scientific_overrides_allowed"] is False
    frozen = plan["protocol"]
    assert frozen["formal_seeds"] == list(range(338, 348))
    assert frozen["formal_case_count"] == 40
    assert frozen["paired_dataset_seed_count"] == 20
    assert frozen["smoke"] == {
        "seed": 9905,
        "round_cap": 3,
        "formal_result_valid": False,
    }
    assert frozen["upstream_stage4"]["result"] == "qualified_random_scan_s8"
    assert frozen["formal_generation_started"] is False


def test_plan_does_not_import_generator_or_instantiate_rng() -> None:
    program = """
import sys
import numpy as np
from scripts import issue53_stage5_protocol as p

def forbidden(*args, **kwargs):
    raise AssertionError("RNG instantiated")

np.random.default_rng = forbidden
p.build_plan()
assert "table_diffevo.evolution" not in sys.modules
assert "table_diffevo.generator" not in sys.modules
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:."
    subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_only_kernel_fields_differ_and_max_order_is_dataset_bound() -> None:
    allowed = {
        "factorized_gibbs_sweeps",
        "factorized_gibbs_use_compiled_workload",
    }
    for dataset, max_order in (("test_300x10", 4), ("nltcs", 3)):
        independent = protocol.generator_params(
            dataset, 338, protocol.ARM_INDEPENDENT
        )
        factor = protocol.generator_params(dataset, 338, protocol.ARM_FACTOR)
        differing = {
            key
            for key in independent
            if independent[key] != factor[key]
        }
        assert differing == allowed
        assert independent["factorized_gibbs_sweeps"] == 0
        assert factor["factorized_gibbs_sweeps"] == 8
        assert independent["factorized_gibbs_use_compiled_workload"] is False
        assert factor["factorized_gibbs_use_compiled_workload"] is True
        assert independent["factorized_gibbs_max_order"] == max_order
        assert factor["factorized_gibbs_max_order"] == max_order
        for params in (independent, factor):
            assert params["diffusion_direction_strength"] == 2.0
            assert params["diffusion_direction_normalization"] == "initial_rms"
            assert params["fixed_alpha"] == 16.0
            assert params["inner_early_stopping_patience_ticks"] == 6
            assert params["record_stationarity_trace"] is False
            assert params["record_natural_work_snapshots"] is False


def test_seed_order_is_counterbalanced_and_not_overridable() -> None:
    assert protocol.arm_order_for_seed(338) == (
        protocol.ARM_INDEPENDENT,
        protocol.ARM_FACTOR,
    )
    assert protocol.arm_order_for_seed(339) == (
        protocol.ARM_FACTOR,
        protocol.ARM_INDEPENDENT,
    )
    assert protocol.case_order_for_seed(338) == (
        ("test_300x10", protocol.ARM_INDEPENDENT),
        ("test_300x10", protocol.ARM_FACTOR),
        ("nltcs", protocol.ARM_INDEPENDENT),
        ("nltcs", protocol.ARM_FACTOR),
    )
    smoke = protocol.generator_params(
        "nltcs", 9905, protocol.ARM_FACTOR, mode="smoke"
    )
    assert smoke["n_rounds"] == smoke["candidate_budget"] == 3
    with pytest.raises(ValueError, match="seed"):
        protocol.generator_params(
            "nltcs", 337, protocol.ARM_FACTOR, mode="formal"
        )
    with pytest.raises(ValueError, match="seed"):
        protocol.generator_params(
            "nltcs", 338, protocol.ARM_FACTOR, mode="smoke"
        )
    with pytest.raises(ValueError, match="arm"):
        protocol.generator_params("nltcs", 338, "factor_s16")
    with pytest.raises(ValueError, match="dataset"):
        protocol.generator_params("adult", 338, protocol.ARM_FACTOR)


def _pair_rows(seed: int) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "arm": arm,
            "seed": seed,
            "execution_order_index": index,
            "initial_table_sha256": f"initial-{dataset}",
            "primary_rng_post_initialization_state_sha256": f"rng-{dataset}",
            "primary_rng_state_sha256_by_round": [
                f"round-1-{dataset}",
                f"round-2-{dataset}",
            ],
            "direction_reference_scale": 1.25,
            "hostname": "host-a",
            "cuda_device_name": "gpu-a",
        }
        for index, (dataset, arm) in enumerate(
            protocol.case_order_for_seed(seed)
        )
    ]


def test_pairing_requires_exact_order_s0_rng_scale_host_and_device() -> None:
    rows = _pair_rows(338)
    pairing = runner._assert_seed_pairing(rows, 338)
    assert set(pairing) == set(protocol.DATASET_ORDER)

    broken = deepcopy(rows)
    broken[1]["direction_reference_scale"] = 1.5
    with pytest.raises(RuntimeError, match="未配对"):
        runner._assert_seed_pairing(broken, 338)

    rng_drift = deepcopy(rows)
    rng_drift[1]["primary_rng_state_sha256_by_round"][1] = "different"
    with pytest.raises(RuntimeError, match="common prefix"):
        runner._assert_seed_pairing(rng_drift, 338)

    reversed_pair = deepcopy(rows)
    reversed_pair[0], reversed_pair[1] = reversed_pair[1], reversed_pair[0]
    with pytest.raises(RuntimeError, match="顺序"):
        runner._assert_seed_pairing(reversed_pair, 338)


def test_exact_count_error_sum_rejects_noninteger_answers() -> None:
    runtime = SimpleNamespace(np=np)
    assert runner._exact_count_error_sum(
        np.array([3, 8]), np.array([1, 9]), runtime
    ) == 3
    with pytest.raises(RuntimeError, match="不是整数"):
        runner._exact_count_error_sum(
            np.array([3, 8]), np.array([1.5, 9]), runtime
        )
    assert runner._json_safe({"tol": float("inf")}) == {
        "tol": "positive_infinity"
    }
    with pytest.raises(ValueError, match="NaN"):
        runner._json_safe({"metric": float("nan")})
    with pytest.raises(ValueError, match="infinity"):
        runner._json_safe({"metric": float("inf")})


def test_kernel_counter_audit_fails_closed() -> None:
    independent = {
        "direction_logit_evaluated_count_history": [2, 3],
        "direction_logit_clipped_count_history": [0, 0],
        "factorized_gibbs_microsteps": 0,
        "factorized_gibbs_conditional_logit_evaluated_count": 0,
        "factorized_gibbs_active_blocks": 0,
        "factorized_gibbs_initial_rng_state_sha256": None,
        "factorized_gibbs_rng_state_sha256": None,
    }
    runner._audit_kernel_counters(independent, protocol.ARM_INDEPENDENT, 2)

    factor = deepcopy(independent)
    factor.update(
        {
            "factorized_gibbs_microsteps": 40,
            "factorized_gibbs_conditional_logit_evaluated_count": 40,
            "factorized_gibbs_active_blocks": 5,
            "factorized_gibbs_initial_rng_state_sha256": "a" * 64,
            "factorized_gibbs_rng_state_sha256": "b" * 64,
        }
    )
    runner._audit_kernel_counters(factor, protocol.ARM_FACTOR, 2)
    factor["factorized_gibbs_microsteps"] = 39
    with pytest.raises(RuntimeError, match="microstep"):
        runner._audit_kernel_counters(factor, protocol.ARM_FACTOR, 2)


def _metric(value: float) -> dict[str, float]:
    return {
        "normalized_l1_mean": value,
        "normalized_l1_median": value,
        "normalized_l1_p90": value,
        "normalized_l1_max": value,
        "squared_loss_diagnostic_only": value,
    }


def _synthetic_case(dataset: str, arm: str, seed: int) -> dict:
    is_factor = arm == protocol.ARM_FACTOR
    measured_sum = 80 if is_factor else 100
    offline = {"one_way_safety": _metric(1.0)}
    if dataset == "test_300x10":
        for name in protocol.TEST_GROUP_ORDER[1:]:
            offline[name] = _metric(1.0)
    else:
        offline["unmeasured_3way"] = _metric(1.0)
        offline["all_4way"] = _metric(1.0)
    metrics = {
        "measured": {
            "overall": {
                **_metric(measured_sum / 1000),
                "absolute_count_error_sum": measured_sum,
            }
        },
        "offline_query_groups": offline,
        "reference_support": {
            "synthetic_mass_in_reference_support": 0.8,
            "reference_mass_covered": 0.7,
        },
        "diversity": {
            "unique_row_rate": 0.8,
            "effective_unique_row_ratio": 0.7,
            "attribute_effective_support_ratio_mean": 0.6,
            "attribute_effective_support_ratio_min": 0.5,
        },
        "validity": {"valid_row_rate": 1.0},
    }
    if dataset == "nltcs":
        metrics["binned_joint"] = {"tvd": 1.0}
    return {
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "pair_execution_position": (
            protocol.arm_order_for_seed(seed).index(arm) + 1
        ),
        "termination_reason": "early_stopped",
        "terminal_behavior": {"normalized_work_at_stop": 1.0},
        "cost": {
            "candidate_evaluation_count": 10,
            "case_wall_clock_elapsed_sec": 2.0 if is_factor else 1.0,
            "factorized_gibbs_conditional_logit_clipped_count": 0,
        },
        "direction_logit_clipped_count": 0,
        "metrics": metrics,
    }


def _synthetic_cases() -> list[dict]:
    return [
        _synthetic_case(dataset, arm, seed)
        for seed in protocol.FORMAL_SEEDS
        for dataset in protocol.DATASET_ORDER
        for arm in protocol.ARMS
    ]


def test_frozen_classification_supports_factor_only_after_all_gates() -> None:
    cases = _synthetic_cases()
    result = evaluator._frozen_classification(cases)

    assert result["execution_qualification"]["status"] == "qualified"
    assert result["cross_dataset_response"] == "shared_factor_support_at_tau2"
    for dataset in protocol.DATASET_ORDER:
        cell = result["by_dataset"][dataset]
        assert cell["classification"] == (
            "factor_quality_supported_outer_efficient"
        )
        measured = cell["measured_stability"]
        assert measured["factor_10_seed_aggregate_count_error_sum"] == 800
        assert measured["independent_10_seed_aggregate_count_error_sum"] == 1000
        assert measured["paired_wins"] == 10
        assert measured["stable_factor_measured_gain"] is True

    independent = auditor._independent_classification(cases)
    assert independent["cross_dataset_response"] == (
        result["cross_dataset_response"]
    )
    assert all(
        independent["datasets"][dataset]["classification"]
        == result["by_dataset"][dataset]["classification"]
        for dataset in protocol.DATASET_ORDER
    )
    auditor._audit_evaluator_classification(
        {"frozen_classification": result}, independent
    )


def test_frozen_classification_distinguishes_no_gain_risk_and_compute() -> None:
    no_gain = _synthetic_cases()
    for case in no_gain:
        if case["arm"] == protocol.ARM_FACTOR:
            case["metrics"]["measured"]["overall"][
                "absolute_count_error_sum"
            ] = 100
    result = evaluator._frozen_classification(no_gain)
    assert all(
        cell["classification"] == "no_stable_factor_gain"
        for cell in result["by_dataset"].values()
    )
    assert all(
        cell["no_stable_factor_gain_detail"]
        == "mixed_no_stable_kernel_winner"
        for cell in result["by_dataset"].values()
    )

    independent_wins = deepcopy(no_gain)
    for case in independent_wins:
        if case["arm"] == protocol.ARM_FACTOR:
            case["metrics"]["measured"]["overall"][
                "absolute_count_error_sum"
            ] = 120
    result = evaluator._frozen_classification(independent_wins)
    assert all(
        cell["no_stable_factor_gain_detail"]
        == "stable_independent_measured_advantage"
        for cell in result["by_dataset"].values()
    )

    risk = _synthetic_cases()
    for case in risk:
        if case["arm"] == protocol.ARM_FACTOR:
            case["metrics"]["offline_query_groups"]["one_way_safety"][
                "normalized_l1_mean"
            ] = 1.2
    result = evaluator._frozen_classification(risk)
    assert all(
        cell["classification"]
        == "factor_measured_gain_with_quality_or_diversity_risk"
        for cell in result["by_dataset"].values()
    )

    compute = _synthetic_cases()
    for case in compute:
        if case["arm"] == protocol.ARM_FACTOR:
            case["terminal_behavior"]["normalized_work_at_stop"] = 1.2
    result = evaluator._frozen_classification(compute)
    assert all(
        cell["classification"]
        == "factor_quality_supported_with_outer_compute_tradeoff"
        for cell in result["by_dataset"].values()
    )
    assert result["cross_dataset_response"] == (
        "shared_factor_quality_support_with_compute_tradeoff"
    )


def test_frozen_classification_is_dataset_specific_and_execution_first() -> None:
    cases = _synthetic_cases()
    for case in cases:
        if (
            case["dataset"] == "test_300x10"
            and case["arm"] == protocol.ARM_FACTOR
        ):
            case["metrics"]["diversity"]["unique_row_rate"] = 0.5
    result = evaluator._frozen_classification(cases)
    assert result["cross_dataset_response"] == "dataset_dependent_kernel_response"

    capped = _synthetic_cases()
    capped[0]["termination_reason"] = "resource_cap_reached"
    result = evaluator._frozen_classification(capped)
    assert result["execution_qualification"]["status"] == (
        "inconclusive_resource_cap"
    )
    assert all(
        cell["classification"] == "inconclusive_resource_cap"
        for cell in result["by_dataset"].values()
    )

    clipped = _synthetic_cases()
    clipped[-1]["cost"][
        "factorized_gibbs_conditional_logit_clipped_count"
    ] = 1
    result = evaluator._frozen_classification(clipped)
    assert result["execution_qualification"]["status"] == (
        "outside_stage4_qualified_unclipped_regime"
    )


def test_evaluation_and_audit_plans_are_result_blind() -> None:
    evaluation = evaluator.build_plan()
    audit = auditor.build_plan()

    assert evaluation["collection_protocol_sha256"] == (
        protocol.FROZEN_PROTOCOL_SHA256
    )
    assert evaluation["stable_measured_gain"][
        "paired_strict_wins_minimum"
    ] == 8
    assert evaluation["new_generation_allowed"] is False
    assert audit["recompute_all_40_case_metrics"] is True
    assert audit["trust_evaluator_derived_values"] is False
    assert audit["new_generation_allowed"] is False


def test_compiled_equivalence_detects_non_timing_drift(tmp_path: Path) -> None:
    base_diagnostics = {
        "current_state_metrics_history": [{"current_squared_loss": 2.0}],
        "transition_clock_history": [],
        "loss_history": [2.0],
        "accept_history": [],
        "rho_schedule_history": [],
        "alpha_history": [],
        "direction_reference_scale_history": [],
        "direction_logit_evaluated_count_history": [],
        "direction_logit_clipped_count_history": [],
        "factorized_gibbs_active_rows": 0,
        "factorized_gibbs_active_blocks": 0,
        "factorized_gibbs_factor_count": 0,
        "factorized_gibbs_factor_table_entries": 0,
        "factorized_gibbs_microsteps": 0,
        "factorized_gibbs_conditional_logit_evaluated_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
    }
    compiled_path = tmp_path / "compiled.json"
    rowwise_path = tmp_path / "rowwise.json"
    runner._write_json(compiled_path, base_diagnostics)
    runner._write_json(rowwise_path, base_diagnostics)
    common = {
        "dataset": "test_300x10",
        "terminal_table_sha256": "a" * 64,
        "termination_reason": "early_stopped",
        "terminal_behavior": {"terminal_state_index": 0},
        "measured": {"absolute_count_error_sum": 1},
        "initial_table_sha256": "b" * 64,
        "primary_rng_post_initialization_state_sha256": "c" * 64,
        "primary_rng_state_sha256": "d" * 64,
        "factorized_gibbs_initial_rng_state_sha256": "e" * 64,
        "factorized_gibbs_rng_state_sha256": "f" * 64,
        "direction_reference_scale": 1.0,
        "result_artifact_sha256": "1" * 64,
    }
    compiled = {**common, "diagnostics_path": compiled_path.name}
    rowwise = {**common, "diagnostics_path": rowwise_path.name}
    result = runner._compiled_equivalence(tmp_path, compiled, rowwise)
    assert result["compiled_equals_rowwise"] is True

    drifted = deepcopy(base_diagnostics)
    drifted["loss_history"] = [3.0]
    runner._write_json(rowwise_path, drifted)
    with pytest.raises(RuntimeError, match="不等价"):
        runner._compiled_equivalence(tmp_path, compiled, rowwise)
