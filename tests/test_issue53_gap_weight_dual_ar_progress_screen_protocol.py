"""A/R 相对初始进度开发筛查的结果前协议测试。"""

import json
from pathlib import Path

import pytest

from scripts import audit_issue53_gap_weight_dual_ar_progress_offline as audit
from scripts import issue53_gap_weight_dual_ar_progress_screen_protocol as protocol
from scripts import issue53_gap_weight_dual_ar_screen_protocol as prior
from table_diffevo import gap_l1_diffusion as gap


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_formula_freezes_relative_initial_progress_without_new_tuning():
    assert protocol.frozen_protocol_manifest()["formula"] == {
        "weighting": "dual_abs_relative_progress_max",
        "absolute": "sum_all(abs(target-count))/(N*J)",
        "relative": (
            "sum_positive(abs(target-count)/target)"
            "/(N*sum_positive(1/target))"
        ),
        "aggregation": "max(A/A_init,R/R_init)",
        "reference_source": "initial_current_before_round_1",
        "reference_recomputed_by_round": False,
        "zero_target_policy": "absolute_channel_only",
        "empty_positive_target_policy": "A/A_init_only",
        "zero_positive_reference_policy": "fail_closed",
        "floor_applied": False,
        "max_weight_ratio": None,
        "smoothing_count": None,
        "tunable_parameter_count": 0,
    }
    assert protocol.DUAL_WEIGHTING == (
        gap.GAP_L1_WEIGHTING_DUAL_ABS_RELATIVE_PROGRESS_MAX
    )


def test_task_matrix_only_adds_two_candidate_trajectories():
    plan = protocol.task_plan()
    assert plan.seed == 9908
    assert [task.dataset for task in plan.tasks] == list(
        protocol.DATASET_ORDER
    )
    assert {task.arm for task in plan.tasks} == {protocol.CANDIDATE_ARM}
    assert all(task.rounds == 6000 for task in plan.tasks)
    assert len(plan.tasks) == 2


def test_candidate_only_changes_aggregation_mode_from_prior_dual_arm():
    for dataset in protocol.DATASET_ORDER:
        candidate = protocol.task_generator_params(
            dataset, protocol.CANDIDATE_ARM
        )
        baseline = prior.task_generator_params(
            dataset, prior.CANDIDATE_ARM
        )
        differing = {
            key for key in candidate if candidate[key] != baseline[key]
        }
        assert differing == {"gap_l1_weighting"}
        assert candidate["gap_l1_sweeps"] == 8
        assert candidate["gap_l1_max_weight_ratio"] is None


def test_offline_audit_reconstructs_frozen_initial_references_and_activation():
    rebuilt = audit.build_audit(REPOSITORY_ROOT)
    assert rebuilt["audit"] == {
        "read_only": True,
        "runtime_float64_reduction_order_reproduced": True,
        "historical_candidate_checkpoint_answers_only": True,
        "new_candidate_generated": False,
        "raw_reference_table_accessed": False,
        "quality_evaluation_performed": False,
        "parameter_search_performed": False,
    }
    for dataset, expected in (
        protocol.EXPECTED_INITIAL_CHANNEL_REFERENCES.items()
    ):
        observed = rebuilt["datasets"][dataset]
        assert observed["absolute_initial"] == expected["absolute_initial"]
        assert observed["relative_initial"] == expected["relative_initial"]
        assert observed["terminal"]["progress_dominant_channel"] == "relative"
        assert observed["summary"]["terminal_dominance_changed"] is True


def test_frozen_references_match_runtime_builder_bit_for_bit():
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
        query_rows = json.loads(
            (REPOSITORY_ROOT / spec["queries"]).read_text(encoding="utf-8")
        )["queries"]
        targets = [row["result"] for row in query_rows]
        checkpoint = (
            REPOSITORY_ROOT
            / prior.OUTPUT_DIR
            / "cases"
            / (
                f"seed_{prior.DEVELOPMENT_SEED}__"
                f"{prior.CANDIDATE_ARM}__{dataset}"
            )
            / "checkpoint_query_answers.json"
        )
        initial_counts = json.loads(
            checkpoint.read_text(encoding="utf-8")
        )["fixed_checkpoints"][0]["query_answers"]
        reference = gap.build_gap_l1_channel_reference(
            initial_counts,
            targets,
            n_records=spec["n_records"],
        )
        assert reference.absolute_initial == (
            protocol.EXPECTED_INITIAL_CHANNEL_REFERENCES[dataset][
                "absolute_initial"
            ]
        )
        assert reference.relative_initial == (
            protocol.EXPECTED_INITIAL_CHANNEL_REFERENCES[dataset][
                "relative_initial"
            ]
        )


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"execution_valid": False}, "execution_invalid"),
        (
            {"relative_channel_activated_by_dataset": {
                "test_300x10": False,
                "nltcs": True,
            }},
            "relative_channel_not_activated",
        ),
        (
            {"measured_l1_ratio_by_dataset": {
                "test_300x10": 1.0,
                "nltcs": 1.0,
            }},
            "common_query_improvement_not_recovered",
        ),
        (
            {"nltcs_common_bin_ratio": 1.0},
            "common_query_improvement_not_recovered",
        ),
        ({"nltcs_rare_bin_ratio": 1.050001}, "rare_query_safety_risk"),
        (
            {"one_way_ratio_by_dataset": {
                "test_300x10": 1.0,
                "nltcs": 1.050001,
            }},
            "measured_or_one_way_safety_risk",
        ),
        ({}, "advance_to_fresh_seed_confirmation"),
    ],
)
def test_classification_uses_frozen_priority(overrides, expected):
    values = {
        "execution_valid": True,
        "relative_channel_activated_by_dataset": {
            "test_300x10": True,
            "nltcs": True,
        },
        "measured_l1_ratio_by_dataset": {
            "test_300x10": 1.05,
            "nltcs": 0.99,
        },
        "nltcs_common_bin_ratio": 0.99,
        "nltcs_rare_bin_ratio": 1.05,
        "one_way_ratio_by_dataset": {
            "test_300x10": 1.05,
            "nltcs": 1.05,
        },
    }
    values.update(overrides)
    assert protocol.classify_screen(**values) == expected


def test_run_requires_the_scientific_protocol_hash():
    with pytest.raises(PermissionError, match="显式确认"):
        protocol.require_run_confirmation("wrong")
    protocol.require_run_confirmation(protocol.FROZEN_PROTOCOL_SHA256)


def test_protocol_identity_is_fully_frozen():
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
