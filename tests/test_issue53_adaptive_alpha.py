"""Frozen collection and evaluation contracts for adaptive alpha."""

from copy import deepcopy

import pytest

from scripts import evaluate_issue53_adaptive_alpha as evaluator
from scripts import run_issue53_adaptive_alpha as runner


def test_collection_plan_freezes_three_arms_and_fresh_seeds() -> None:
    plan = runner.build_plan()

    assert plan["protocol_sha256"] == runner.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    assert plan["scientific_overrides_allowed"] is False
    protocol = plan["protocol"]
    assert protocol["arm_order"] == [
        runner.ARM_FIXED_16,
        runner.ARM_FIXED_12,
        runner.ARM_ADAPTIVE,
    ]
    assert protocol["seeds"] == [328, 329, 330, 331, 332]
    assert protocol["trajectory_count"] == 30
    assert protocol["primary_comparison"].startswith(runner.ARM_ADAPTIVE)
    assert protocol["mechanism_comparison"].startswith(runner.ARM_FIXED_12)
    assert protocol["formal_generation_started"] is False


def test_protocol_document_and_manifest_identities_are_frozen() -> None:
    root = runner._repo_root()

    assert runner.protocol_sha256() == runner.FROZEN_PROTOCOL_SHA256
    assert runner._sha256_file(root / runner.PROTOCOL_DOC) == (
        runner.PROTOCOL_DOC_SHA256
    )
    text = (root / runner.PROTOCOL_DOC).read_text(encoding="utf-8")
    assert "连续 2 个无进展自然工作刻度" in text
    assert "恰好 2 个自然工作刻度" in text
    assert "seeds = [328, 329, 330, 331, 332]" in text
    assert "measured_50query_30_15_5.json" in text
    assert "measured_50query.json" not in text


def test_only_frozen_alpha_schedule_fields_change_between_arms() -> None:
    fixed16 = runner.generator_params(328, runner.ARM_FIXED_16)
    fixed12 = runner.generator_params(328, runner.ARM_FIXED_12)
    adaptive = runner.generator_params(328, runner.ARM_ADAPTIVE)

    assert fixed16["fixed_alpha"] == 16.0
    assert fixed12["fixed_alpha"] == 12.0
    assert adaptive["fixed_alpha"] is None
    assert adaptive["alpha_schedule_mode"] == runner.ADAPTIVE_SCHEDULE_MODE
    assert fixed16["alpha_schedule_mode"] == fixed12["alpha_schedule_mode"] == "fixed"
    for params in (fixed16, fixed12, adaptive):
        assert params["residual_geometry"] == "relative"
        assert params["residual_geometry_floor"] == 8.0
        assert params["selection_scale_invariant"] is True
        assert params["inner_early_stopping_patience_ticks"] == 6
        assert params["factorized_gibbs_sweeps"] == 0

    ignored = {"alpha_schedule_mode", "fixed_alpha"}
    assert {
        key: value for key, value in fixed16.items() if key not in ignored
    } == {
        key: value for key, value in fixed12.items() if key not in ignored
    } == {
        key: value for key, value in adaptive.items() if key not in ignored
    }
    with pytest.raises(ValueError, match="seed"):
        runner.generator_params(327, runner.ARM_FIXED_16)
    with pytest.raises(ValueError, match="arm"):
        runner.generator_params(328, "adaptive_alpha_16_24")


def test_public_inputs_keep_updated_no_one_way_workloads() -> None:
    audit = runner._audit_inputs(runner._repo_root())

    assert audit["test_300x10"]["order_counts"] == {2: 30, 3: 15, 4: 5}
    assert audit["nltcs"]["order_counts"] == {2: 479, 3: 522}
    assert audit["test_300x10"]["query_count"] == 50
    assert audit["nltcs"]["query_count"] == 1001


def test_adaptive_schedule_audit_preserves_full_event_history() -> None:
    observations = [
        {
            "state_index": 1,
            "alpha_used": 16.0,
            "events": (),
            "completed_work_ticks": 1,
            "applied_participating_rows": 4,
            "cumulative_participating_rows": 4,
            "normalized_work": 1.0,
            "phase_before": "normal",
            "escape_index_observed": None,
            "progress_epoch_before": 0,
            "best_updated": False,
        },
        {
            "state_index": 2,
            "alpha_used": 16.0,
            "events": ("escape_started",),
            "completed_work_ticks": 2,
            "applied_participating_rows": 4,
            "cumulative_participating_rows": 8,
            "normalized_work": 2.0,
            "phase_before": "normal",
            "escape_index_observed": 1,
            "progress_epoch_before": 0,
            "best_updated": False,
        },
        {
            "state_index": 3,
            "alpha_used": 12.0,
            "events": ("new_best", "escape_tick_completed"),
            "completed_work_ticks": 3,
            "applied_participating_rows": 4,
            "cumulative_participating_rows": 12,
            "normalized_work": 3.0,
            "phase_before": "escape",
            "escape_index_observed": 1,
            "progress_epoch_before": 0,
            "best_updated": True,
        },
        {
            "state_index": 4,
            "alpha_used": 12.0,
            "events": ("escape_tick_completed", "escape_completed"),
            "completed_work_ticks": 4,
            "applied_participating_rows": 4,
            "cumulative_participating_rows": 16,
            "normalized_work": 4.0,
            "phase_before": "escape",
            "escape_index_observed": 1,
            "progress_epoch_before": 1,
            "best_updated": False,
        },
    ]
    diagnostics = {
        "rounds_run": 4,
        "alpha_history": [16.0, 16.0, 12.0, 12.0],
        "params": {
            "alpha_schedule_mode": runner.ADAPTIVE_SCHEDULE_MODE,
            "fixed_alpha": None,
            "adaptive_alpha_config": runner.ADAPTIVE_CONFIG,
            "selection_scale_invariant": True,
            "residual_geometry": "relative",
            "residual_geometry_floor": 8.0,
            "factorized_gibbs_sweeps": 0,
            "inner_early_stopping_patience_ticks": 6,
        },
        "adaptive_alpha": {
            "enabled": True,
            "config": runner.ADAPTIVE_CONFIG,
            "escape_count": 1,
            "observation_history": observations,
        },
        "row_max_prob_mean_history": [0.8, 0.7, 0.5, 0.4],
        "row_max_prob_max_history": [0.9, 0.8, 0.6, 0.5],
        "effective_donors_mean_history": [1.2, 1.5, 2.1, 2.4],
        "donor_top_share_history": [0.75, 0.5, 0.5, 0.25],
    }

    summary = runner._audit_schedule(
        diagnostics, runner.ARM_ADAPTIVE, n_records=4
    )

    assert summary["escape_count"] == 1
    assert summary["trigger_completed_work_ticks"] == [2]
    assert summary["escape_completed_count"] == 1
    assert summary["new_best_during_escape_count"] == 1
    assert summary["alpha12_round_count"] == 2
    assert summary["alpha12_normalized_work"] == 2.0
    phases = summary["phase_diagnostics"]
    assert phases["phase_concentration"]["normal_alpha16_all"][
        "round_count"
    ] == 2
    assert phases["phase_concentration"]["escape_alpha12_all"][
        "round_count"
    ] == 2
    segment = phases["escape_segments"][0]
    assert segment["progress_epoch_at_trigger"] == 0
    assert segment["trigger"]["state_index"] == 2
    assert segment["completion"]["state_index"] == 4
    assert segment["first_new_best_location"] == "during_escape"
    assert segment["donor_concentration"]["escape_alpha12"][
        "round_count"
    ] == 2

    restored = deepcopy(diagnostics)
    restored["rounds_run"] = 5
    restored["alpha_history"].append(16.0)
    restored["adaptive_alpha"]["observation_history"][2]["events"] = (
        "escape_tick_completed",
    )
    restored["adaptive_alpha"]["observation_history"][2][
        "best_updated"
    ] = False
    restored["adaptive_alpha"]["observation_history"].append(
        {
            "state_index": 5,
            "alpha_used": 16.0,
            "events": ("new_best",),
            "completed_work_ticks": 5,
            "applied_participating_rows": 4,
            "cumulative_participating_rows": 20,
            "normalized_work": 5.0,
            "phase_before": "normal",
            "escape_index_observed": None,
            "progress_epoch_before": 0,
            "best_updated": True,
        }
    )
    restored["row_max_prob_mean_history"].append(0.75)
    restored["row_max_prob_max_history"].append(0.85)
    restored["effective_donors_mean_history"].append(1.4)
    restored["donor_top_share_history"].append(0.5)

    restored_summary = runner._audit_schedule(
        restored, runner.ARM_ADAPTIVE, n_records=4
    )
    restored_segment = restored_summary["phase_diagnostics"][
        "escape_segments"
    ][0]
    assert restored_segment["first_new_best_location"] == "after_restore"
    assert restored_segment[
        "first_new_best_after_restore_state_index"
    ] == 5
    assert restored_segment["donor_concentration"][
        "restored_alpha16_until_next_escape"
    ]["round_count"] == 1


def test_seed_pairing_requires_all_three_arms_and_same_initial_state() -> None:
    rows = [
        {
            "dataset": dataset,
            "arm": arm,
            "initial_table_sha256": f"table-{dataset}",
            "primary_rng_post_initialization_state_sha256": f"rng-{dataset}",
        }
        for dataset, arm in runner.CASE_ORDER
    ]

    pairing = runner._assert_seed_pairing(rows, 328)
    assert set(pairing) == set(runner.DATASET_ORDER)

    broken = deepcopy(rows)
    broken[2]["initial_table_sha256"] = "different"
    with pytest.raises(RuntimeError, match="初始状态未配对"):
        runner._assert_seed_pairing(broken, 328)


def test_evaluation_plan_freezes_primary_and_mechanism_comparisons() -> None:
    plan = evaluator.build_plan()

    assert plan["collection_protocol_sha256"] == runner.FROZEN_PROTOCOL_SHA256
    assert plan["primary_candidate_arm"] == runner.ARM_ADAPTIVE
    assert plan["primary_baseline_arm"] == runner.ARM_FIXED_16
    assert plan["mechanism_control_arm"] == runner.ARM_FIXED_12
    assert plan["stable_measured_gain"]["paired_seed_wins_minimum"] == 4
    assert plan["fixed12_uses_same_full_gate_vs_fixed16"] is True
    assert plan["adaptive_vs_fixed12_direct_gate_present"] is False
    assert plan["new_generation_allowed"] is False


def _metric_block(value: float) -> dict:
    return {"normalized_l1_mean": value}


def _synthetic_case(dataset: str, arm: str, seed: int) -> dict:
    measured = {
        runner.ARM_FIXED_16: 1.0,
        runner.ARM_FIXED_12: 0.9,
        runner.ARM_ADAPTIVE: 0.8,
    }[arm]
    offline_groups = {"one_way_safety": _metric_block(1.0)}
    metrics = {
        "measured": {
            "overall": {
                "normalized_l1_mean": measured,
                "absolute_count_error_mean": measured,
            }
        },
        "offline_query_groups": offline_groups,
        "validity": {"valid_row_rate": 1.0},
        "diversity": {
            "unique_row_rate": 0.8,
            "effective_unique_row_ratio": 0.7,
        },
    }
    if dataset == "test_300x10":
        for name in evaluator.TEST_GROUP_ORDER[1:]:
            offline_groups[name] = _metric_block(1.0)
    else:
        offline_groups["unmeasured_3way"] = _metric_block(1.0)
        offline_groups["all_4way"] = _metric_block(1.0)
        metrics["binned_joint"] = {"tvd": 1.0}
    return {
        "dataset": dataset,
        "arm": arm,
        "seed": seed,
        "termination_reason": "early_stopped",
        "rounds_run": 10.0,
        "normalized_work_at_stop": 1.0,
        "donor_concentration": {
            "effective_donor_fraction": {"tail_mean": 0.1},
            "row_max_prob_mean": {"tail_mean": 0.2},
        },
        "adaptive_alpha_summary": {
            "escape_count": 1 if arm == runner.ARM_ADAPTIVE else 0,
            "triggered": arm == runner.ARM_ADAPTIVE,
            "new_best_during_escape_count": (
                1 if arm == runner.ARM_ADAPTIVE else 0
            ),
            "alpha12_normalized_work": (
                2.0 if arm == runner.ARM_ADAPTIVE else None
            ),
        },
        "metrics": metrics,
    }


def _synthetic_cases() -> list[dict]:
    return [
        _synthetic_case(dataset, arm, seed)
        for dataset in runner.DATASET_ORDER
        for arm in runner.ARMS
        for seed in runner.SEEDS
    ]


def test_frozen_classification_distinguishes_timing_from_always_low() -> None:
    cases = _synthetic_cases()

    both = evaluator._frozen_classification(cases)
    assert both["cross_dataset_response"] == "shared_adaptive_support"
    assert all(
        value == "adaptive_and_always_low_both_supported"
        for value in both["mechanism_interpretation"].values()
    )
    assert all(
        value["classification"] == "adaptive_exercised"
        for value in both["adaptive_activation"].values()
    )
    assert all(
        value["mechanism_claim_allowed"]
        for value in both["adaptive_activation"].values()
    )

    timed_only = deepcopy(cases)
    for case in timed_only:
        if case["arm"] == runner.ARM_FIXED_12:
            case["metrics"]["measured"]["overall"][
                "normalized_l1_mean"
            ] = 1.1
    result = evaluator._frozen_classification(timed_only)
    assert all(
        value == "supports_timed_escape_beyond_always_low_alpha"
        for value in result["mechanism_interpretation"].values()
    )


def test_frozen_classification_keeps_risk_compute_and_resource_boundaries() -> None:
    cases = _synthetic_cases()

    risk = deepcopy(cases)
    for case in risk:
        if (
            case["dataset"] == "test_300x10"
            and case["arm"] == runner.ARM_ADAPTIVE
        ):
            case["metrics"]["offline_query_groups"]["one_way_safety"][
                "normalized_l1_mean"
            ] = 1.1
    result = evaluator._frozen_classification(risk)
    assert result["adaptive_vs_fixed16"]["test_300x10"][
        "classification"
    ] == "measured_gain_with_quality_or_diversity_risk"
    assert result["cross_dataset_response"] == (
        "dataset_dependent_adaptive_response"
    )

    compute = deepcopy(cases)
    for case in compute:
        if case["arm"] == runner.ARM_ADAPTIVE:
            case["normalized_work_at_stop"] = 1.1
    result = evaluator._frozen_classification(compute)
    assert all(
        value["classification"] == "quality_supported_with_compute_tradeoff"
        for value in result["adaptive_vs_fixed16"].values()
    )

    capped = deepcopy(cases)
    capped[0]["termination_reason"] = "resource_cap_reached"
    result = evaluator._frozen_classification(capped)
    assert result["cross_dataset_response"] == "inconclusive_resource_cap"
    assert result["all_30_cases_normal"] is False


def test_zero_activation_is_reported_without_changing_quality_gate() -> None:
    cases = _synthetic_cases()
    for case in cases:
        if case["arm"] == runner.ARM_ADAPTIVE:
            case["adaptive_alpha_summary"]["escape_count"] = 0
            case["adaptive_alpha_summary"]["triggered"] = False

    result = evaluator._frozen_classification(cases)

    assert all(
        value["classification"] == "adaptive_not_exercised"
        for value in result["adaptive_activation"].values()
    )
    assert all(
        not value["mechanism_claim_allowed"]
        for value in result["adaptive_activation"].values()
    )
    assert set(result["mechanism_claim_status"].values()) == {
        "prohibited_adaptive_not_exercised"
    }
    assert all(
        value["classification"] == "supported_adaptive_escape"
        for value in result["adaptive_vs_fixed16"].values()
    )


def test_offline_group_identities_are_reused_without_reference_answers() -> None:
    root = runner._repo_root()
    test_groups, test_audit = evaluator.fixed_evaluation._freeze_test_groups(root)
    nltcs_groups, nltcs_audit = evaluator.fixed_evaluation._freeze_nltcs_groups(root)

    assert {name: len(value) for name, value in test_groups.items()} == (
        evaluator.TEST_GROUP_COUNTS
    )
    assert test_audit["identity_sha256"] == evaluator.TEST_GROUP_IDENTITIES
    assert {name: len(value) for name, value in nltcs_groups.items()} == (
        evaluator.NLTCS_GROUP_COUNTS
    )
    assert nltcs_audit["identity_sha256"] == evaluator.NLTCS_GROUP_IDENTITIES
