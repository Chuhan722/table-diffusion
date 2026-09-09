from __future__ import annotations

from pathlib import Path

import pytest

from scripts import issue53_stage6c_joint_smoke_protocol as protocol
from scripts import issue53_stage6c_joint_trajectories as joint


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_protocol_manifest_is_frozen_and_old_source_fails_closed():
    assert protocol.file_sha256(REPOSITORY_ROOT / protocol.PROTOCOL_DOC) == (
        protocol.PROTOCOL_DOC_SHA256
    )
    drifted_sources = [
        name
        for name, binding in protocol.IMPLEMENTATION_SOURCES.items()
        if protocol.file_sha256(REPOSITORY_ROOT / binding["path"])
        != binding["sha256"]
    ]
    # 2026-09-06：update.py 因 lottery_first_donor_selection（先抽签后
    # 选供体提速，等价性见 tests/test_lottery_first_donor_selection.py）
    # 合法演进，加入预期漂移清单。
    assert drifted_sources == [
        "full_generator", "shared_update_plan", "gap_kernel",
    ]
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    with pytest.raises(RuntimeError, match="实现源码漂移：full_generator"):
        protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)


def test_plan_is_read_only_and_lists_the_complete_joint_matrix(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "assert_frozen_protocol_identity",
        lambda _root: protocol.FROZEN_PROTOCOL_SHA256,
    )
    plan = protocol.build_plan(REPOSITORY_ROOT)

    assert plan == {
        "mode": "plan_only_no_dataset_read_no_generation",
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "task_ids": [
            "seed_9907__factor_b_s8__test_300x10",
            "seed_9907__factor_b_s8__nltcs",
            "seed_9907__independent_b_s0__test_300x10",
            "seed_9907__independent_b_s0__nltcs",
            "seed_9907__gap_l1_global_s8__test_300x10",
            "seed_9907__gap_l1_global_s8__nltcs",
        ],
        "trajectory_count": 6,
        "max_workers": 2,
        "output_dir": "outputs/issue53_stage6c_joint_smoke_seed9907_v1",
        "quality_values_emitted": False,
        "runner_wired": False,
        "generation_started": False,
        "confirmation_consumed": False,
    }


def test_frozen_task_matrix_cannot_drop_a_dataset_or_arm():
    manifest = protocol.frozen_protocol_manifest()
    matrix = manifest["task_matrix"]

    assert matrix["datasets"] == list(joint.DATASET_ORDER)
    assert matrix["arms"] == list(joint.ARM_ORDER)
    assert matrix["seeds"] == [9907]
    assert matrix["rounds_by_dataset"] == {
        "test_300x10": 20,
        "nltcs": 20,
    }
    assert matrix["trajectory_count"] == 6
    assert matrix["all_tasks_declared_before_any_result"] is True
    assert matrix["partial_dataset_or_arm_plan_allowed"] is False
    assert matrix["result_dependent_task_removal_allowed"] is False
    assert matrix["smoke_seed_excluded_from_formal_effect_seeds"] is True


def test_common_generator_configuration_is_fixed_budget_and_no_gate():
    params = protocol.common_generator_params()

    assert params["n_rounds"] == 20
    assert params["candidate_budget"] == 20
    assert params["seed"] == 9907
    assert params["device"] == "cuda"
    assert params["tol"] == float("inf")
    assert params["max_retries"] == 0
    assert params["rho"] == 0.01
    assert params["eta"] == 0.5
    assert params["mu"] == 0.01
    assert params["alpha_schedule_mode"] == "fixed"
    assert params["fixed_alpha"] == 16.0
    assert params["residual_geometry"] == "relative"
    assert params["residual_geometry_floor"] == 8.0
    assert params["diffusion_direction_strength"] == 2.0
    assert "factorized_gibbs_use_compiled_workload" not in params
    assert params["inner_early_stopping_patience_ticks"] is None
    assert params["stop_on_exact_residual"] is True
    assert params["return_final_table"] is True


@pytest.mark.parametrize(
    ("dataset", "n_records", "max_order", "query_path"),
    [
        (
            "test_300x10",
            300,
            4,
            "configs/test_300x10/measured_50query_30_15_5.json",
        ),
        ("nltcs", 16_181, 3, "configs/nltcs/measured_1000query.json"),
    ],
)
def test_both_datasets_use_the_frozen_workload_and_cuda(
    dataset, n_records, max_order, query_path
):
    spec = protocol.DATASETS[dataset]
    params = protocol.task_generator_params(dataset, "independent_b_s0")

    assert str(spec["queries"]) == query_path
    assert params["n_records"] == n_records
    assert params["factorized_gibbs_max_order"] == max_order
    assert params["device"] == "cuda"


@pytest.mark.parametrize(
    ("arm", "factor_sweeps", "compiled", "gap_sweeps"),
    [
        ("factor_b_s8", 8, True, 0),
        ("independent_b_s0", 0, False, 0),
        ("gap_l1_global_s8", 0, False, 8),
    ],
)
def test_only_the_kernel_delta_changes_between_arms(
    arm, factor_sweeps, compiled, gap_sweeps
):
    params = protocol.task_generator_params("nltcs", arm)

    assert params["factorized_gibbs_sweeps"] == factor_sweeps
    assert params["factorized_gibbs_use_compiled_workload"] is compiled
    assert params["gap_l1_sweeps"] == gap_sweeps
    assert params["tol"] == float("inf")
    assert params["max_retries"] == 0
    assert params["return_final_table"] is True


def test_gpu_and_concurrency_are_fixed_without_fallback():
    execution = protocol.frozen_protocol_manifest()["execution"]

    assert execution["max_workers"] == 2
    assert execution["multiprocessing_start_method"] == "spawn"
    assert execution["task_reordering"] is False
    assert execution["adaptive_concurrency"] is False
    assert execution["automatic_cpu_fallback"] is False
    assert execution["expected_gpu"] == {
        "physical_index": 1,
        "cuda_visible_devices": "1",
        "uuid": "GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02",
        "name": "NVIDIA GeForce RTX 4090",
        "process_device": "cuda:0",
        "visible_device_count": 1,
    }


def test_report_contract_suppresses_quality_and_table_contents():
    boundary = protocol.frozen_protocol_manifest()["report_boundary"]

    assert boundary["quality_values_emitted"] is False
    assert boundary["method_ranking_or_classification_emitted"] is False
    assert boundary["raw_reference_or_heldout_evaluation"] is False
    assert boundary["generated_table_contents_persisted"] is False
    assert "elapsed_sec" in boundary["allowed_task_fields"]
    assert "terminal_table_sha256" in boundary["allowed_task_fields"]
    assert "loss" in boundary["forbidden_quality_fields"]
    assert "normalized_l1_error" in boundary["forbidden_quality_fields"]
    assert "final_table" in boundary["forbidden_quality_fields"]
    assert not (
        set(boundary["allowed_task_fields"])
        & set(boundary["forbidden_quality_fields"])
    )


def test_protocol_freeze_does_not_authorize_runner_or_smoke_execution():
    boundary = protocol.frozen_protocol_manifest()["authorization_boundary"]

    assert boundary == {
        "protocol_frozen": True,
        "runner_wired_at_protocol_freeze": False,
        "smoke_run_authorized_at_protocol_freeze": False,
        "formal_effect_protocol_frozen": False,
        "formal_run_authorized": False,
        "public_default_kernel_changed": False,
    }
    with pytest.raises(PermissionError, match="另行授权"):
        protocol.require_run_confirmation(None)
    with pytest.raises(PermissionError, match="另行授权"):
        protocol.require_run_confirmation("wrong")
    protocol.require_run_confirmation(protocol.FROZEN_PROTOCOL_SHA256)


def test_unknown_dataset_or_arm_is_rejected():
    with pytest.raises(ValueError, match="未知测速数据集"):
        protocol.task_generator_params("plants", "independent_b_s0")
    with pytest.raises(ValueError, match="未知长轨迹方法"):
        protocol.task_generator_params("nltcs", "unknown")
