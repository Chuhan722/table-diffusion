from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import audit_issue53_stage6d_formal as auditor
from scripts import evaluate_issue53_stage6d_formal as evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6d_formal_protocol as protocol
from scripts import run_issue53_stage6d_formal as runner
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import AttributeBlock, Schema


def test_frozen_identity_and_all_plan_entrypoints_are_read_only():
    root = Path(__file__).resolve().parents[1]

    assert protocol.assert_frozen_protocol_identity(root) == (
        protocol.FROZEN_PROTOCOL_SHA256
    )
    collection_plan = runner.build_plan()
    evaluation_plan = evaluator.build_plan()
    audit_plan = auditor.build_plan()
    assert collection_plan["trajectory_count"] == 30
    assert collection_plan["formal_collection_authorized"] is False
    assert evaluation_plan["new_generation_allowed"] is False
    assert audit_plan["rerun_generation"] is False
    with pytest.raises(PermissionError, match="另行授权"):
        protocol.require_run_confirmation(None)


def test_each_dataset_passes_both_frozen_identity_conventions():
    root = Path(__file__).resolve().parents[1]

    for dataset, spec in protocol.DATASETS.items():
        queries, targets, observed = runner._query_target_identity_audit(root, dataset)
        assert len(queries) == spec["query_count"]
        assert len(targets) == spec["query_count"]
        assert observed == {
            "query_identity_sha256": spec["query_identity_sha256"],
            "target_vector_sha256": spec["target_vector_sha256"],
            "trace_query_identity_sha256": spec["trace_query_identity_sha256"],
            "trace_target_vector_sha256": spec["trace_target_vector_sha256"],
        }
        assert (
            observed["query_identity_sha256"] != observed["trace_query_identity_sha256"]
        )
        assert (
            observed["target_vector_sha256"] != observed["trace_target_vector_sha256"]
        )


def test_frozen_matrix_and_no_gate_params_are_complete():
    plan = protocol.task_plan()

    assert protocol.PROTOCOL_VERSION.endswith("-v2")
    assert protocol.OUTPUT_DIR.name.endswith("_v2")
    assert plan.seeds == (353, 354, 355, 356, 357)
    assert len(plan.tasks) == 30
    assert [task.task_id for task in plan.tasks[:6]] == [
        "seed_353__factor_b_s8__test_300x10",
        "seed_353__factor_b_s8__nltcs",
        "seed_353__independent_b_s0__test_300x10",
        "seed_353__independent_b_s0__nltcs",
        "seed_353__gap_l1_global_s8__test_300x10",
        "seed_353__gap_l1_global_s8__nltcs",
    ]
    for task in plan.tasks:
        params = protocol.task_generator_params(task.dataset, task.arm, task.seed)
        assert params["n_rounds"] == 2500
        assert params["candidate_budget"] == 2500
        assert params["tol"] == float("inf")
        assert params["max_retries"] == 0
        assert params["inner_early_stopping_patience_ticks"] is None
        assert params["return_final_table"] is True
        assert params["record_stationarity_trace"] is True
        assert params["record_transition_clocks"] is True
        assert params["gap_l1_sweeps"] == (8 if task.arm == protocol.ARM_GAP else 0)


def test_protocol_requires_both_datasets_and_both_baselines():
    manifest = protocol.frozen_protocol_manifest()

    assert manifest["task_matrix"]["trajectory_count"] == 30
    assert manifest["task_matrix"]["datasets"] == ["test_300x10", "nltcs"]
    assert manifest["primary_quality"]["must_beat_each_baseline_separately"]
    assert manifest["l1_output"] == {
        "evaluation_json": protocol.EVALUATION_REPORT,
        "long_form_csv": protocol.L1_RESULTS_CSV,
        "per_case_terminal": True,
        "per_seed_pairwise_differences": True,
        "five_seed_summaries": True,
        "fixed_checkpoint_curve": [0, 500, 1000, 1500, 2000, 2500],
    }
    assert manifest["target_kind"] == "exact_source_query_counts_not_noisy"
    assert manifest["dataset_identity_contract"]["all_four_identities_required"]
    assert not manifest["dataset_identity_contract"][
        "cross_convention_hash_equality_expected"
    ]


def test_metrics_from_answers_reports_integer_sum_l1_and_gap_e():
    metrics = runner._metrics_from_answers(
        np.asarray([10.0, 2.0]),
        np.asarray([7.0, 4.0]),
        100,
    )

    assert metrics["absolute_count_error_sum"] == 5
    assert metrics["normalized_l1"] == 0.025
    assert metrics["squared_loss"] == 6.5
    assert metrics["gap_e"] == pytest.approx((3 / 10 + 2 / 8) / 2)


@pytest.mark.parametrize(
    ("target", "answers", "n_records"),
    [
        ([1.5], [1.0], 10),
        ([1.0], [1.5], 10),
        ([11.0], [1.0], 10),
        ([1.0], [-1.0], 10),
        ([], [], 10),
    ],
)
def test_l1_arithmetic_rejects_non_count_vectors(target, answers, n_records):
    with pytest.raises(ValueError):
        runner._metrics_from_answers(np.asarray(target), np.asarray(answers), n_records)
    with pytest.raises(RuntimeError):
        auditor._arithmetic(np.asarray(target), np.asarray(answers), n_records)


def test_checkpoint_artifact_contains_fixed_and_actual_terminal_l1(monkeypatch):
    task = joint.JointTrajectoryTask(
        dataset="test_300x10",
        arm="independent_b_s0",
        seed=353,
        rounds=2500,
    )
    target = np.asarray([2.0, 1.0])
    answers = np.asarray([[0.0, 0.0], [1.0, 1.0]])
    query_identity = protocol.DATASETS["test_300x10"]["trace_query_identity_sha256"]
    target_identity = runner.target_answer_identity_sha256(target)
    monkeypatch.setitem(
        protocol.DATASETS["test_300x10"],
        "trace_target_vector_sha256",
        target_identity,
    )
    diagnostics = {
        "stationarity_trace": SimpleNamespace(
            measured_query_answers=answers,
            observations=[object(), object()],
            query_identity_sha256=query_identity,
            target_identity_sha256=target_identity,
        ),
        "current_state_metrics_history": [
            {
                "state_index": 0,
                "round": 0,
                "phase": "initial",
                "current_normalized_l1": 0.005,
                "current_squared_loss": 2.5,
            },
            {
                "state_index": 1,
                "round": 1,
                "phase": "post_round",
                "current_normalized_l1": 1 / 600,
                "current_squared_loss": 0.5,
            },
        ],
    }

    artifact = runner._extract_checkpoint_artifact(
        task, diagnostics, target, applied_rounds=1
    )

    assert [row["round"] for row in artifact["fixed_checkpoints"]] == [0]
    assert artifact["fixed_checkpoints"][0]["normalized_l1"] == 0.005
    assert artifact["terminal"]["round"] == 1
    assert artifact["terminal"]["normalized_l1"] == 1 / 600
    assert (
        artifact["query_identity_sha256"]
        == protocol.DATASETS["test_300x10"]["query_identity_sha256"]
    )
    assert artifact["trace_query_identity_sha256"] == query_identity
    assert artifact["trace_target_vector_sha256"] == target_identity
    assert artifact["historical_best_included"] is False


def _gap_transition_diagnostics():
    return {
        "transition_clock_history": [
            {
                "state_index": 1,
                "round": 1,
                "attempts": [
                    {
                        "participating_rows": 3,
                        "changed_rows": 2,
                        "changed_cells": 4,
                        "changed_queries": 1,
                        "gibbs_microsteps": 24,
                    }
                ],
                "accepted_attempt": 1,
                "candidate_evaluation_count_cumulative": 1,
                "post_current_table_sha256": "1" * 64,
                "primary_rng_state_sha256": "2" * 64,
                "factorized_gibbs_rng_state_sha256": None,
            }
        ],
        "gap_l1_attempt_diagnostics_history": [
            [
                {
                    "no_gate": True,
                    "n_sweeps": 8,
                    "gap_l1_scan_applied": True,
                    "active_switches_k": 3,
                    "gibbs_microsteps": 24,
                    "clip_hit_count": 0,
                    "nonfinite_condition_count": 0,
                }
            ]
        ],
        "factorized_gibbs_attempt_diagnostics_history": [],
        "gap_l1_microsteps": 24,
        "gap_l1_reference_scale": 0.125,
        "gap_l1_clip_hit_count": 0,
        "factorized_gibbs_microsteps": 0,
        "factorized_gibbs_conditional_logit_evaluated_count": 0,
        "factorized_gibbs_conditional_logit_clipped_count": 0,
        "direction_logit_evaluated_count_history": [3],
        "direction_logit_clipped_count_history": [0],
    }


def test_transition_audit_proves_gap_8k_and_unconditional_application():
    task = joint.JointTrajectoryTask(
        dataset="nltcs",
        arm=protocol.ARM_GAP,
        seed=353,
        rounds=2500,
    )

    artifact, summary = runner._extract_transition_audit(
        task, _gap_transition_diagnostics(), applied_rounds=1
    )

    assert artifact["all_applied_unconditionally"] is True
    assert artifact["clocks"][0]["accepted_attempt"] == 1
    assert artifact["gap_rounds"][0]["microsteps"] == 24
    assert summary["gap_expected_microsteps"] == 24
    assert summary["gap_8k_identity"] is True
    assert summary["gap_reference_scale"] == 0.125


def test_transition_audit_rejects_gap_microstep_drift():
    task = joint.JointTrajectoryTask(
        dataset="nltcs",
        arm=protocol.ARM_GAP,
        seed=353,
        rounds=2500,
    )
    diagnostics = _gap_transition_diagnostics()
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0]["gibbs_microsteps"] = 16

    with pytest.raises(RuntimeError, match="护栏失败"):
        runner._extract_transition_audit(task, diagnostics, applied_rounds=1)


def test_case_row_requires_one_candidate_for_every_applied_round():
    task = joint.JointTrajectoryTask(
        dataset="nltcs",
        arm=protocol.ARM_GAP,
        seed=353,
        rounds=2500,
    )
    row = {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": task.seed,
        "requested_rounds": task.rounds,
        "applied_rounds": 2500,
        "termination_reason": "candidate_budget",
        "device": "cuda:0",
        "output_table_identity": "terminal_current",
        "all_applied_unconditionally": True,
        "proposal_attempt_count": 2500,
        "candidate_evaluation_count": 2500,
        "state_evaluation_count": 2501,
    }

    runner._validate_case_row(task, row)
    row["candidate_evaluation_count"] = 2501
    with pytest.raises(RuntimeError, match="身份漂移"):
        runner._validate_case_row(task, row)


@pytest.mark.parametrize("arm", joint.ARM_ORDER)
def test_real_artificial_generator_diagnostics_feed_formal_audits(arm, monkeypatch):
    schema = Schema(
        [
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in ("a", "b", "c")
        ]
    )
    queries = [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 1}]},
        {"conditions": [{"attribute": "b", "operator": "==", "value": 1}]},
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 0},
                {"attribute": "b", "operator": "==", "value": 0},
                {"attribute": "c", "operator": "==", "value": 1},
            ]
        },
    ]
    # 这四个整数目标不能由同一张二元三属性表同时精确满足。
    target = np.asarray([20.0, 20.0, 20.0, 20.0])
    params = {
        "n_records": 20,
        "n_rounds": 2,
        "candidate_budget": 2,
        "seed": 353,
        "rho": 0.8,
        "eta": 0.5,
        "mu": 0.1,
        "tol": float("inf"),
        "max_retries": 0,
        "device": "numpy",
        "distance_mode": "geometric",
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_use_compiled_workload": arm == "factor_b_s8",
        "return_final_table": True,
        "record_transition_clocks": True,
        "record_stationarity_trace": True,
        "log_every": 100,
        **joint.arm_kernel_params(arm),
    }
    _, diagnostics = run_evolution(target, queries, schema, **params)
    diagnostics.pop("final_table")
    task = joint.JointTrajectoryTask(
        dataset="test_300x10", arm=arm, seed=353, rounds=2500
    )
    monkeypatch.setitem(protocol.DATASETS["test_300x10"], "n_records", 20)
    monkeypatch.setitem(
        protocol.DATASETS["test_300x10"],
        "trace_query_identity_sha256",
        diagnostics["stationarity_trace"].query_identity_sha256,
    )
    monkeypatch.setitem(
        protocol.DATASETS["test_300x10"],
        "trace_target_vector_sha256",
        diagnostics["stationarity_trace"].target_identity_sha256,
    )

    transition, summary = runner._extract_transition_audit(
        task, diagnostics, applied_rounds=2
    )
    checkpoints = runner._extract_checkpoint_artifact(
        task, diagnostics, target, applied_rounds=2
    )

    assert transition["round_count"] == 2
    assert transition["all_applied_unconditionally"] is True
    assert summary["gap_8k_identity"] is True
    assert checkpoints["terminal"]["round"] == 2


def _quality_metrics(dataset: str, count_error: int) -> dict:
    n_records = protocol.DATASETS[dataset]["n_records"]
    query_count = protocol.DATASETS[dataset]["query_count"]
    offline_names = (
        protocol.TEST_GROUP_ORDER
        if dataset == "test_300x10"
        else ("one_way_safety", "unmeasured_3way", "all_4way")
    )
    return {
        "measured": {
            "overall": {
                "absolute_count_error_sum": count_error,
                "normalized_l1_mean": count_error / (query_count * n_records),
                "gap_e": count_error / 1000,
                "squared_loss_diagnostic_only": float(count_error),
            }
        },
        "offline_query_groups": {
            name: {"normalized_l1_mean": 0.1} for name in offline_names
        },
        "binned_joint": {"tvd": 0.1},
        "validity": {"valid_row_rate": 1.0},
        "diversity": {
            "unique_row_rate": 0.8,
            "effective_unique_row_ratio": 0.7,
            "attribute_effective_support_ratio_mean": 0.9,
            "attribute_effective_support_ratio_min": 0.8,
        },
        "reference_support": {
            "synthetic_mass_in_reference_support": 0.9,
            "reference_mass_covered": 0.8,
        },
    }


def _formal_cases() -> list[dict]:
    cases = []
    for dataset in joint.DATASET_ORDER:
        for arm in joint.ARM_ORDER:
            for seed in protocol.FORMAL_SEEDS:
                if arm == protocol.ARM_GAP:
                    count_error = 80 if seed != 357 else 120
                elif arm == "factor_b_s8":
                    count_error = 100
                else:
                    count_error = 110
                checkpoint = {
                    "kind": "fixed_checkpoint",
                    "state_index": 2500,
                    "round": 2500,
                    "phase": "post_round",
                    "absolute_count_error_sum": count_error,
                    "normalized_l1": count_error
                    / (
                        protocol.DATASETS[dataset]["query_count"]
                        * protocol.DATASETS[dataset]["n_records"]
                    ),
                    "squared_loss": float(count_error),
                    "gap_e": count_error / 1000,
                }
                cases.append(
                    {
                        "task_id": f"seed_{seed}__{arm}__{dataset}",
                        "dataset": dataset,
                        "arm": arm,
                        "seed": seed,
                        "termination_reason": "candidate_budget",
                        "applied_rounds": 2500,
                        "terminal_table_sha256": "1" * 64,
                        "metrics": _quality_metrics(dataset, count_error),
                        "trajectory_l1": {
                            "fixed_checkpoints": [checkpoint],
                            "terminal": checkpoint,
                        },
                        "cost": {
                            "elapsed_sec": 1.0,
                            "gap_microsteps": (80 if arm == protocol.ARM_GAP else 0),
                            "factorized_gibbs_microsteps": (
                                80 if arm == "factor_b_s8" else 0
                            ),
                        },
                    }
                )
    return sorted(
        cases,
        key=lambda case: (
            protocol.FORMAL_SEEDS.index(case["seed"]),
            joint.ARM_ORDER.index(case["arm"]),
            joint.DATASET_ORDER.index(case["dataset"]),
        ),
    )


def test_frozen_classification_requires_four_of_five_against_both_baselines():
    cases = _formal_cases()

    decisions = {
        dataset: evaluator._dataset_decision(cases, dataset)
        for dataset in joint.DATASET_ORDER
    }

    assert all(
        decision["classification"] == "gap_quality_supported"
        for decision in decisions.values()
    )
    for decision in decisions.values():
        assert all(
            comparison["count_error"]["paired_wins"] == 4
            for comparison in decision["measured_by_baseline"].values()
        )
    assert evaluator._cross_dataset(decisions) == "shared_gap_kernel_support"
    assert auditor._classification_independent(cases) == {
        "by_dataset": decisions,
        "cross_dataset_response": "shared_gap_kernel_support",
    }


def test_quality_risk_downgrades_supported_measured_gain():
    cases = _formal_cases()
    for case in cases:
        if case["dataset"] == "test_300x10" and case["arm"] == protocol.ARM_GAP:
            case["metrics"]["offline_query_groups"]["one_way_safety"][
                "normalized_l1_mean"
            ] = 0.2

    decision = evaluator._dataset_decision(cases, "test_300x10")

    assert decision["measured_pass_against_both_baselines"] is True
    assert decision["safety_pass_against_both_baselines"] is False
    assert decision["classification"] == "gap_measured_gain_with_quality_risk"


def test_l1_csv_rows_include_fixed_terminal_without_historical_best():
    cases = _formal_cases()

    rows = evaluator._l1_csv_rows(cases)

    assert len(rows) == 30
    assert all(row["record_kind"] == "fixed_checkpoint" for row in rows)
    assert all(row["round"] == 2500 for row in rows)
    assert all(row["is_terminal"] is True for row in rows)
    assert rows == auditor._csv_rows_independent(cases)


def test_protocol_rejects_nonformal_seed():
    with pytest.raises(ValueError, match="不在正式集合"):
        protocol.task_generator_params("nltcs", protocol.ARM_GAP, 9907)
