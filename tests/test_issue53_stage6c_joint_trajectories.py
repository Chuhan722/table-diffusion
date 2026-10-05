from __future__ import annotations

import math

import numpy as np
import pytest

from scripts import issue53_stage6c_joint_trajectories as joint


def test_joint_plan_lists_both_datasets_for_every_seed_and_arm():
    plan = joint.build_joint_trajectory_plan(
        seeds=[901, 902],
        rounds_by_dataset={"test_300x10": 100, "nltcs": 600},
    )

    assert plan.tasks == tuple(
        joint.JointTrajectoryTask(dataset, arm, seed, rounds)
        for seed in (901, 902)
        for arm in joint.ARM_ORDER
        for dataset, rounds in (("test_300x10", 100), ("nltcs", 600))
    )
    assert len(plan.tasks) == 12
    assert len({task.task_id for task in plan.tasks}) == 12
    assert plan.protocol_frozen is False
    assert plan.generation_started is False


def test_plan_manifest_is_explicitly_result_blind_and_not_executable():
    plan = joint.build_joint_trajectory_plan(
        seeds=[903],
        rounds_by_dataset={"nltcs": 700, "test_300x10": 120},
    )

    manifest = joint.plan_manifest(plan)

    assert manifest["mode"] == "plan_only_no_data_read_no_generation"
    assert manifest["datasets"] == ["test_300x10", "nltcs"]
    assert manifest["arms"] == [
        "factor_b_s8",
        "independent_b_s0",
        "gap_l1_global_s8",
    ]
    assert manifest["trajectory_count"] == 6
    assert manifest["protocol_frozen"] is False
    assert manifest["generation_started"] is False
    assert manifest["partial_dataset_plan_allowed"] is False
    assert manifest["result_dependent_task_removal_allowed"] is False


@pytest.mark.parametrize(
    ("arm", "factor_sweeps", "compiled", "gap_sweeps"),
    [
        ("factor_b_s8", 8, True, 0),
        ("independent_b_s0", 0, False, 0),
        ("gap_l1_global_s8", 0, False, 8),
    ],
)
def test_arm_deltas_change_only_the_kernel(
    arm, factor_sweeps, compiled, gap_sweeps
):
    params = joint.arm_kernel_params(arm)

    assert params == {
        "tol": float("inf"),
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "return_final_table": True,
        "factorized_gibbs_sweeps": factor_sweeps,
        "factorized_gibbs_use_compiled_workload": compiled,
        "gap_l1_sweeps": gap_sweeps,
    }
    assert math.isinf(params["tol"])


@pytest.mark.parametrize(
    "seeds",
    [[], [1, 1], [-1], [True], [1.5], np.asarray([1, 2])],
)
def test_joint_plan_rejects_invalid_seed_lists(seeds):
    error = TypeError if isinstance(seeds, np.ndarray) else ValueError
    with pytest.raises(error):
        joint.build_joint_trajectory_plan(
            seeds=seeds,
            rounds_by_dataset={"test_300x10": 100, "nltcs": 600},
        )


@pytest.mark.parametrize(
    "rounds",
    [
        {"test_300x10": 100},
        {"test_300x10": 100, "nltcs": 600, "plants": 10},
        {"test_300x10": 0, "nltcs": 600},
        {"test_300x10": 100, "nltcs": True},
    ],
)
def test_joint_plan_requires_exactly_both_datasets_with_positive_rounds(rounds):
    with pytest.raises(ValueError):
        joint.build_joint_trajectory_plan(seeds=[904], rounds_by_dataset=rounds)


def test_arm_parameters_are_fresh_and_unknown_arms_fail():
    first = joint.arm_kernel_params("gap_l1_global_s8")
    first["gap_l1_sweeps"] = 0

    assert joint.arm_kernel_params("gap_l1_global_s8")["gap_l1_sweeps"] == 8
    with pytest.raises(ValueError, match="未知长轨迹方法"):
        joint.arm_kernel_params("not_an_arm")
