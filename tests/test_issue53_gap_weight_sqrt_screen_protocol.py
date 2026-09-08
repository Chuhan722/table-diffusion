"""平方根查询权重开发筛查的结果前协议与目标审计测试。"""

import json
from pathlib import Path

import pytest

from scripts import issue53_gap_weight_r8_screen_protocol as r8
from scripts import issue53_gap_weight_sqrt_screen_protocol as protocol
from table_diffevo import gap_l1_diffusion as gap


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_formula_has_no_dataset_specific_parameter():
    manifest = protocol.frozen_protocol_manifest()
    assert manifest["formula"] == {
        "weighting": "sqrt_target_relative",
        "denominator": "sqrt(max(target_count,1))",
        "target_count_quantum": 1,
        "max_weight_ratio": None,
        "smoothing_count": None,
        "tunable_parameter_count": 0,
    }
    assert protocol.SQRT_WEIGHTING == (
        gap.GAP_L1_WEIGHTING_SQRT_TARGET_RELATIVE
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


def test_candidate_only_changes_c_weighting_from_frozen_legacy_arm():
    for dataset in protocol.DATASET_ORDER:
        candidate = protocol.task_generator_params(
            dataset, protocol.CANDIDATE_ARM
        )
        legacy = r8.task_generator_params(dataset, "gap_legacy_s8")
        differing = {
            key
            for key in candidate
            if candidate[key] != legacy[key]
        }
        assert differing == {"gap_l1_weighting"}
        assert candidate["gap_l1_weighting"] == "sqrt_target_relative"
        assert candidate["gap_l1_max_weight_ratio"] is None
        assert candidate["residual_geometry"] == "relative"
        assert candidate["residual_geometry_floor"] == 8.0
        assert candidate["gap_l1_sweeps"] == 8


def test_target_weight_audit_rebuilds_frozen_artifact_exactly():
    expected = json.loads(
        (REPOSITORY_ROOT / protocol.TARGET_WEIGHT_AUDIT).read_text(
            encoding="utf-8"
        )
    )
    rebuilt = protocol.target_weight_audit(REPOSITORY_ROOT)
    assert rebuilt == expected
    assert rebuilt["audit"] == {
        "query_targets_only": True,
        "kernel_implementation_imported": False,
        "raw_reference_data_accessed": False,
        "legacy_or_r8_terminal_tables_accessed": False,
        "candidate_generation_started": False,
        "candidate_results_accessed": False,
        "all_query_identities_verified": True,
        "all_target_identities_verified": True,
        "all_frozen_bin_identities_verified": True,
    }


def test_target_weight_audit_matches_kernel_denominators():
    audit = protocol.target_weight_audit(REPOSITORY_ROOT)
    for dataset in protocol.DATASET_ORDER:
        rows = audit["datasets"][dataset]["queries"]
        targets = [row["target"] for row in rows]
        denominators, spec = gap._build_gap_l1_denominators(
            targets,
            floor=8.0,
            weighting=protocol.SQRT_WEIGHTING,
            max_weight_ratio=None,
            n_records=protocol.DATASETS[dataset]["n_records"],
        )
        assert denominators.tolist() == [
            row["denominator"] for row in rows
        ]
        assert spec.actual_weight_ratio == (
            audit["datasets"][dataset]["summary"]["actual_weight_ratio"]
        )


def test_reused_baseline_artifact_hashes_are_still_exact():
    for artifact in protocol.BASELINE_ARTIFACTS.values():
        path = REPOSITORY_ROOT / artifact["path"]
        if not path.exists():
            pytest.skip(
                "冻结基线产物不在本机（gitignored），产物持有机复核"
            )
        assert protocol.file_sha256(path) == artifact["sha256"]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"execution_valid": False}, "execution_invalid"),
        (
            {"nltcs_measured_l1_ratio": 1.0},
            "common_mechanism_not_retained",
        ),
        (
            {"nltcs_common_bin_ratio": 1.0},
            "common_mechanism_not_retained",
        ),
        (
            {"nltcs_rare_vs_legacy_ratio": 1.2500001},
            "rare_query_protection_not_recovered",
        ),
        (
            {"nltcs_rare_vs_r8_ratio": 1.0},
            "rare_query_protection_not_recovered",
        ),
        (
            {"test_measured_l1_ratio": 1.0500001},
            "measured_or_one_way_safety_risk",
        ),
        (
            {"one_way_ratio_by_dataset": {
                "test_300x10": 1.0,
                "nltcs": 1.0500001,
            }},
            "measured_or_one_way_safety_risk",
        ),
        ({}, "advance_to_fresh_seed_confirmation"),
    ],
)
def test_classification_uses_frozen_priority(kwargs, expected):
    values = {
        "execution_valid": True,
        "nltcs_measured_l1_ratio": 0.9,
        "nltcs_common_bin_ratio": 0.9,
        "nltcs_rare_vs_legacy_ratio": 1.25,
        "nltcs_rare_vs_r8_ratio": 0.999,
        "test_measured_l1_ratio": 1.05,
        "one_way_ratio_by_dataset": {
            "test_300x10": 1.05,
            "nltcs": 1.05,
        },
    }
    values.update(kwargs)
    assert protocol.classify_screen(**values) == expected


def test_protocol_identity_is_fully_frozen():
    # 收束线：协议冻结的是历史实验身份；活实现源码此后演进属预期，
    # 守卫应失败关闭。死记录（协议自身清单 SHA）仍须自恰。
    assert protocol.canonical_sha256(protocol.frozen_protocol_manifest()) == (
        protocol.FROZEN_PROTOCOL_SHA256
    )
    with pytest.raises(RuntimeError, match="漂移"):
        protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
