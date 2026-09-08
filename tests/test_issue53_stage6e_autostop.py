from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts import audit_issue53_stage6e_autostop as auditor
from scripts import evaluate_issue53_stage6e_autostop as evaluator
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import issue53_stage6e_autostop_protocol as protocol
from scripts import run_issue53_stage6e_autostop as runner


def _task(arm: str = "gap_l1_global_s8", dataset: str = "test_300x10"):
    return next(
        task
        for task in protocol.task_plan().tasks
        if task.arm == arm and task.dataset == dataset
    )


def _clock(state_index: int, participating_rows: int) -> dict:
    return {
        "state_index": state_index,
        "round": state_index,
        "attempts": [{"participating_rows": participating_rows}],
    }


def _stop_diagnostics(
    task,
    reason: str,
    participating_rows: list[int],
    *,
    consecutive: int,
    terminal_loss: float,
    resource_source: str | None = None,
) -> dict:
    cumulative = sum(participating_rows)
    applied = len(participating_rows)
    completed = cumulative // protocol.DATASETS[task.dataset]["n_records"]
    return {
        "termination_reason": reason,
        "transition_clock_history": [
            _clock(index, rows)
            for index, rows in enumerate(participating_rows, start=1)
        ],
        "inner_early_stopping": {
            "enabled": True,
            "patience_ticks": protocol.PATIENCE_TICKS,
            "resource_cap_source_diagnostic_only": resource_source,
            "last_decision": {
                "termination_reason": reason,
                "state_index": applied,
                "terminal_output_state_index": applied,
                "cumulative_participating_rows": cumulative,
                "normalized_work": (
                    cumulative / protocol.DATASETS[task.dataset]["n_records"]
                ),
                "completed_work_ticks": completed,
                "current_loss": terminal_loss,
                "fit_target_reached": reason == "fit_target_reached",
                "work_tick_completed": reason == "early_stopped",
                "consecutive_no_progress_ticks": consecutive,
                "inner_complete": reason in {"fit_target_reached", "early_stopped"},
                "external_resource_cap_reached": reason
                == "resource_cap_reached",
                "best_state_index_diagnostic_only": 0,
                "best_loss_diagnostic_only": 1.0,
            },
        },
    }


def _case_row(task, reason: str = "early_stopped", applied: int = 12) -> dict:
    elapsed = 6.0
    return {
        "task_id": task.task_id,
        "execution_shard_id": protocol.LOCAL_SHARD,
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": task.seed,
        "requested_rounds": protocol.ROUND_CAP,
        "applied_rounds": applied,
        "termination_reason": reason,
        "inner_early_stopping": {
            "enabled": True,
            "patience_ticks": protocol.PATIENCE_TICKS,
            "termination_reason": reason,
            "terminal_state_index": applied,
            "terminal_loss": 0.0 if reason == "fit_target_reached" else 1.0,
            "consecutive_no_progress_ticks": (
                protocol.PATIENCE_TICKS if reason == "early_stopped" else 0
            ),
            "resource_cap_source_diagnostic_only": (
                "candidate_budget" if reason == "resource_cap_reached" else None
            ),
        },
        "device": "cuda:0",
        "output_table_identity": "terminal_current",
        "all_applied_unconditionally": True,
        "proposal_attempt_count": applied,
        "candidate_evaluation_count": applied,
        "state_evaluation_count": applied + 1,
        "elapsed_sec": elapsed,
        "average_sec_per_applied_round": elapsed / applied if applied else 0.0,
    }


def test_protocol_restores_existing_p6_flow_and_only_kernel_switches():
    assert len(protocol.task_plan().tasks) == 30
    assert protocol.FORMAL_SEEDS == (353, 354, 355, 356, 357)
    assert protocol.ROUND_CAP == protocol.CANDIDATE_BUDGET == 6000
    assert protocol.PATIENCE_TICKS == 6
    assert protocol.SHARD_ORDER == (protocol.LOCAL_SHARD,)
    assert len(protocol.tasks_for_shard(protocol.LOCAL_SHARD)) == 30

    manifests = {
        arm: protocol.task_generator_params("nltcs", arm, 353)
        for arm in joint.ARM_ORDER
    }
    kernel_keys = {
        "factorized_gibbs_sweeps",
        "factorized_gibbs_use_compiled_workload",
        "gap_l1_sweeps",
    }
    common = []
    for params in manifests.values():
        assert params["inner_early_stopping_patience_ticks"] == 6
        assert params["n_rounds"] == params["candidate_budget"] == 6000
        assert params["return_final_table"] is True
        assert math.isinf(params["tol"]) and params["tol"] > 0
        common.append({key: value for key, value in params.items() if key not in kernel_keys})
    assert common[0] == common[1] == common[2]
    assert manifests["factor_b_s8"]["factorized_gibbs_sweeps"] == 8
    assert manifests["independent_b_s0"]["factorized_gibbs_sweeps"] == 0
    assert manifests["gap_l1_global_s8"]["gap_l1_sweeps"] == 8


def test_frozen_protocol_identity_fails_closed_after_kernel_source_change():
    root = runner._repo_root()
    drifted_sources = [
        name
        for name, binding in protocol.IMPLEMENTATION_SOURCES.items()
        if protocol.file_sha256(root / binding["path"]) != binding["sha256"]
    ]

    assert drifted_sources == ["full_generator", "gap_kernel"]
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    assert protocol.file_sha256(root / protocol.PROTOCOL_DOC) == (
        protocol.PROTOCOL_DOC_SHA256
    )
    with pytest.raises(RuntimeError, match="实现源码漂移：full_generator"):
        protocol.assert_frozen_protocol_identity(root)


def test_final_confirmation_guard_fails_before_collection(monkeypatch):
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("不得进入正式采集")

    monkeypatch.setattr(runner.stage6d_runner, "run_shard", forbidden)
    with pytest.raises(PermissionError):
        runner.collect("0" * 64)
    assert called is False


def test_a_b_c_stop_summaries_are_accepted_and_audited():
    task = _task()
    n_records = protocol.DATASETS[task.dataset]["n_records"]
    fit = _stop_diagnostics(
        task,
        "fit_target_reached",
        [n_records],
        consecutive=0,
        terminal_loss=0.0,
    )
    assert runner._stop_summary(task, fit, applied_rounds=1)[
        "termination_reason"
    ] == "fit_target_reached"

    early = _stop_diagnostics(
        task,
        "early_stopped",
        [n_records] * protocol.PATIENCE_TICKS,
        consecutive=protocol.PATIENCE_TICKS,
        terminal_loss=1.0,
    )
    summary = runner._stop_summary(
        task, early, applied_rounds=protocol.PATIENCE_TICKS
    )
    assert summary["completed_work_ticks"] == protocol.PATIENCE_TICKS
    assert summary["normalized_work_at_stop"] == float(protocol.PATIENCE_TICKS)

    resource = _stop_diagnostics(
        task,
        "resource_cap_reached",
        [1] * protocol.ROUND_CAP,
        consecutive=0,
        terminal_loss=1.0,
        resource_source="candidate_budget",
    )
    assert runner._stop_summary(
        task, resource, applied_rounds=protocol.ROUND_CAP
    )["resource_cap_source_diagnostic_only"] == "candidate_budget"


def test_stop_summary_rejects_disabled_or_inconsistent_patience():
    task = _task()
    diagnostics = _stop_diagnostics(
        task,
        "early_stopped",
        [protocol.DATASETS[task.dataset]["n_records"]] * 6,
        consecutive=6,
        terminal_loss=1.0,
    )
    diagnostics["inner_early_stopping"]["patience_ticks"] = None
    with pytest.raises(RuntimeError, match="P=6"):
        runner._stop_summary(task, diagnostics, applied_rounds=6)


@pytest.mark.parametrize(
    ("reason", "applied"),
    [
        ("fit_target_reached", 1),
        ("early_stopped", 12),
        ("resource_cap_reached", protocol.ROUND_CAP),
    ],
)
def test_case_row_accepts_variable_normal_termination(reason, applied):
    task = _task()
    runner._validate_case_row(task, _case_row(task, reason, applied))


def test_case_row_rejects_old_fixed_budget_reason():
    task = _task()
    row = _case_row(task)
    row["termination_reason"] = "candidate_budget"
    row["inner_early_stopping"]["termination_reason"] = "candidate_budget"
    with pytest.raises(RuntimeError, match="case 行身份漂移"):
        runner._validate_case_row(task, row)


def test_collection_report_recomputes_total_and_average_round_time():
    rows = []
    for task in protocol.task_plan().tasks:
        pairing = f"{task.seed}:{task.dataset}"
        rows.append(
            {
                "task_id": task.task_id,
                "seed": task.seed,
                "dataset": task.dataset,
                "arm": task.arm,
                "applied_rounds": 10,
                "termination_reason": "early_stopped",
                "elapsed_sec": 5.0,
                "initial_table_sha256": pairing,
                "primary_rng_post_initialization_sha256": pairing,
                "direction_reference_scale": 1.0,
                "query_identity_sha256": pairing,
                "target_vector_sha256": pairing,
                "trace_query_identity_sha256": pairing,
                "trace_target_vector_sha256": pairing,
                "gap_8k_identity": True,
                "gap_clip_hit_count": 0,
                "factorized_gibbs_conditional_logit_clipped_count": 0,
                "direction_logit_clipped_count": 0,
                "nonfinite_count": 0,
            }
        )
    shard_id = protocol.LOCAL_SHARD
    report = runner._collection_report(
        execution_commit="c" * 40,
        runner_sha256="r" * 64,
        generator_params_manifest_sha256="g" * 64,
        input_sha256={},
        shard_reports={
            shard_id: {"environment": {}, "execution": {"elapsed_sec": 1.0}}
        },
        shard_report_sha256={shard_id: "s" * 64},
        rows=rows,
    )
    assert report["timing_summary"] == {
        "sum_case_elapsed_sec": 150.0,
        "total_applied_rounds": 300,
        "average_sec_per_applied_round": 0.5,
    }
    assert report["termination_reason_counts"] == {
        "fit_target_reached": 0,
        "early_stopped": 30,
        "resource_cap_reached": 0,
    }


def test_legacy_stage6d_bindings_are_restored_after_stage6e_contexts():
    old_runner_protocol = runner.stage6d_runner.protocol
    old_runner_file = runner.stage6d_runner.__file__
    with runner._stage6d_runtime():
        assert runner.stage6d_runner.protocol is protocol
        assert Path(runner.stage6d_runner.__file__).name == (
            "run_issue53_stage6e_autostop.py"
        )
    assert runner.stage6d_runner.protocol is old_runner_protocol
    assert runner.stage6d_runner.__file__ == old_runner_file

    old_evaluator_protocol = evaluator.stage6d_evaluator.protocol
    with evaluator._stage6d_evaluation_runtime():
        assert evaluator.stage6d_evaluator.protocol is protocol
    assert evaluator.stage6d_evaluator.protocol is old_evaluator_protocol

    old_auditor_protocol = auditor.stage6d_auditor.protocol
    with auditor._stage6d_audit_runtime():
        assert auditor.stage6d_auditor.protocol is protocol
    assert auditor.stage6d_auditor.protocol is old_auditor_protocol


def test_plan_entrypoints_are_result_blind_and_generation_free(monkeypatch):
    monkeypatch.setattr(
        protocol,
        "assert_frozen_protocol_identity",
        lambda _root: "a" * 64,
    )
    collection_plan = runner.build_plan()
    evaluation_plan = evaluator.build_plan()
    audit_plan = auditor.build_plan()
    assert collection_plan["trajectory_count"] == 30
    assert collection_plan["generation_started"] is False
    assert collection_plan["next_command_would_start_generation"] is True
    assert evaluation_plan["new_generation_allowed"] is False
    assert evaluation_plan["variable_applied_rounds_expected"] is True
    assert audit_plan["rerun_generation"] is False
    assert audit_plan["recompute_variable_round_timing"] is True


def test_preflight_does_not_call_generator_or_create_output(monkeypatch, tmp_path):
    try:
        runner._runtime_executable_audit()
    except RuntimeError as error:
        # 冻结审计要求 Triton CUDA 四件套（含 ptxas-blackwell）；
        # 旧版 triton（如 py3.9 上限 3.4）不带该文件，属环境差异非回归。
        pytest.skip(f"本机 Triton 运行时不满足 6E 冻结审计要求：{error}")
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(runner, "_repo_root", lambda: root)
    monkeypatch.setattr(
        protocol, "assert_frozen_protocol_identity", lambda _root: "b" * 64
    )
    monkeypatch.setattr(
        runner.stage6d_runner, "_assert_clean_worktree", lambda _root: "c" * 40
    )
    monkeypatch.setattr(
        runner.stage6d_runner, "_generation_input_audit", lambda _root: {"ok": {}}
    )
    monkeypatch.setattr(
        runner.stage6d_runner,
        "_preflight_generator_param_manifests",
        lambda _tasks: (
            protocol.generator_params_manifest_matrix(),
            protocol.generator_params_manifest_sha256(),
        ),
    )
    monkeypatch.setattr(
        runner.stage6d_runner,
        "_gpu_idle_audit",
        lambda _shard: {"physical_index": 1, "cuda_available": True},
    )
    monkeypatch.setattr(
        runner.stage6d_runner,
        "_call_generator_silently",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("preflight 不得调用生成器")
        ),
    )
    result = runner.preflight()
    assert result["generation_started"] is False
    assert result["ready_for_final_user_confirmation"] is True
    assert not (root / protocol.OUTPUT_DIR).exists()


def test_protocol_source_paths_are_real_files():
    root = runner._repo_root()
    assert (root / protocol.PROTOCOL_DOC).is_file()
    assert set(protocol.IMPLEMENTATION_SOURCES) >= {
        "full_generator",
        "gap_kernel",
        "gap_triton_kernel",
        "inner_early_stopping",
        "collector",
        "evaluator",
        "independent_auditor",
    }
    for binding in protocol.IMPLEMENTATION_SOURCES.values():
        assert (root / Path(binding["path"])).is_file()
