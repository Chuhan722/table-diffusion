from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import issue53_stage6c_joint_smoke_protocol as protocol
from scripts import issue53_stage6c_joint_trajectories as joint
from scripts import run_issue53_stage6c_joint_smoke as runner
from table_diffevo.schema import AttributeBlock, Schema


def _artificial_worker(task):
    return {"task_id": task.task_id, "worker_pid": os.getpid()}


def _artificial_failing_worker(task):
    if task.task_id == "seed_9907__independent_b_s0__nltcs":
        raise RuntimeError("artificial worker failure")
    return {"task_id": task.task_id}


def _diagnostics(final_table, *, arm="independent_b_s0"):
    final_sha = runner._frame_sha256(final_table)
    gap = arm == "gap_l1_global_s8"
    factor = arm == "factor_b_s8"
    return {
        "final_table": final_table.copy(),
        "rounds_run": 2,
        "accept_history": [True, True],
        "transition_clock_history": [
            {"post_current_table_sha256": "1" * 64},
            {"post_current_table_sha256": final_sha},
        ],
        "initial_table_sha256": "0" * 64,
        "termination_reason": "candidate_budget",
        "primary_rng_state_sha256": "2" * 64,
        "gap_l1_attempt_diagnostics_history": (
            [
                [
                    {
                        "no_gate": True,
                        "gap_l1_scan_applied": True,
                        "gibbs_microsteps": 24,
                        "n_sweeps": 8,
                        "active_switches_k": 3,
                        "nonfinite_condition_count": 0,
                    }
                ],
                [
                    {
                        "no_gate": True,
                        "gap_l1_scan_applied": True,
                        "gibbs_microsteps": 0,
                        "n_sweeps": 8,
                        "active_switches_k": 0,
                        "nonfinite_condition_count": 0,
                    }
                ],
            ]
            if gap
            else []
        ),
        "factorized_gibbs_attempt_diagnostics_history": (
            [
                [{"factor_conditional_logit_diagnostics": {"all_finite": True}}],
                [{"factor_conditional_logit_diagnostics": {"all_finite": True}}],
            ]
            if factor
            else []
        ),
        "gap_l1_microsteps": 24 if gap else 0,
        "gap_l1_reference_scale": 0.125 if gap else None,
        "gap_l1_clip_hit_count": 0,
    }


def _task_row(task):
    digest = hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()
    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "arm": task.arm,
        "seed": task.seed,
        "requested_rounds": task.rounds,
        "rounds_run": task.rounds,
        "termination_reason": "candidate_budget",
        "completed": True,
        "device": "cuda:0",
        "elapsed_sec": 1.0,
        "sec_per_round": 0.05,
        "peak_allocated_bytes": 1024,
        "peak_reserved_bytes": 2048,
        "output_table_identity": "terminal_current",
        "terminal_table_sha256": digest,
        "main_rng_endpoint_sha256": "3" * 64,
        "all_applied_unconditionally": True,
        "applied_round_count": task.rounds,
        "gap_reference_scale_established": (
            task.arm == "gap_l1_global_s8"
        ),
        "gap_microsteps": 80 if task.arm == "gap_l1_global_s8" else 0,
        "gap_expected_microsteps": (
            80 if task.arm == "gap_l1_global_s8" else 0
        ),
        "gap_8k_identity": True,
        "gap_clip_hit_count": 0,
        "nonfinite_count": 0,
    }


def _valid_report():
    tasks = runner._frozen_tasks()
    rows = [_task_row(task) for task in tasks]
    samples = [
        {
            "elapsed_sec": 0.0,
            "utilization_percent": 0,
            "memory_used_mib": 10,
            "memory_total_mib": 24_000,
        },
        {
            "elapsed_sec": 0.25,
            "utilization_percent": 80,
            "memory_used_mib": 12_000,
            "memory_total_mib": 24_000,
        },
    ]
    return {
        "contract_version": protocol.PROTOCOL_VERSION,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "execution_commit": "4" * 40,
        "runner_sha256": "5" * 64,
        "environment": {"python": "artificial", "gpu": "artificial"},
        "input_sha256": {"artificial": True},
        "execution": {
            "started_at": "artificial",
            "finished_at": "artificial",
            "elapsed_sec": 2.0,
            "max_workers": 2,
            "multiprocessing_start_method": "spawn",
            "task_order": "seed_then_arm_then_dataset",
            "atomic_output": True,
        },
        "gpu_samples": samples,
        "gpu_summary": runner._summarize_gpu_samples(samples),
        "tasks": rows,
        "summary": {
            "trajectory_count": 6,
            "completed_count": 6,
            "all_tasks_completed": True,
            "all_terminal_current": True,
            "all_applied_unconditionally": True,
            "all_gap_8k_identity": True,
            "total_nonfinite_count": 0,
            "quality_compared_or_emitted": False,
            "method_ranking_emitted": False,
            "formal_effect_claim_emitted": False,
        },
    }


def test_plan_does_not_load_data_check_gpu_or_generate(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan must remain read only")

    monkeypatch.setattr(runner, "_audit_inputs", forbidden)
    monkeypatch.setattr(runner, "_gpu_preflight", forbidden)
    monkeypatch.setattr(runner, "run_evolution", forbidden)

    plan = runner.build_plan()

    assert plan["trajectory_count"] == 6
    assert plan["max_workers"] == 2
    assert plan["quality_values_emitted"] is False
    assert plan["runner_wired_at_protocol_freeze"] is False
    assert plan["runner_wired"] is True
    assert len(plan["runner_sha256"]) == 64
    assert plan["smoke_run_authorized"] is False
    assert plan["generation_started"] is False


@pytest.mark.parametrize("arm", joint.ARM_ORDER)
def test_each_frozen_arm_reaches_the_generator_on_artificial_data(arm):
    schema = Schema([
        AttributeBlock(
            name="x",
            type="categorical",
            description="x",
            values=[0, 1],
        )
    ])
    queries = [{
        "conditions": [{"attribute": "x", "operator": "==", "value": 1}]
    }]
    params = protocol.task_generator_params("test_300x10", arm)
    params.update({
        "n_records": 8,
        "n_rounds": 1,
        "candidate_budget": 1,
        "device": "numpy",
        "init_method": "random",
        "marginals": None,
        "log_every": 2,
    })

    returned_table, diagnostics = runner._call_generator_silently(
        f"artificial_{arm}",
        target=np.asarray([3.5]),
        queries=queries,
        schema=schema,
        **params,
    )

    assert diagnostics["candidate_evaluation_count"] == 1
    assert diagnostics["accept_history"] == [True]
    report = runner._build_task_report(
        joint.JointTrajectoryTask("test_300x10", arm, 9907, 1),
        returned_table,
        diagnostics,
        elapsed_sec=0.1,
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
    )
    assert report["completed"] is True
    assert report["output_table_identity"] == "terminal_current"


def test_artificial_two_process_task_matrix_is_complete_and_ordered():
    rows = runner.run_frozen_task_matrix(_artificial_worker)

    assert [row["task_id"] for row in rows] == [
        task.task_id for task in runner._frozen_tasks()
    ]
    assert len(rows) == 6


def test_artificial_worker_failure_invalidates_the_complete_matrix():
    with pytest.raises(RuntimeError, match="并行任务.*执行失败"):
        runner.run_frozen_task_matrix(_artificial_failing_worker)


def test_task_report_uses_terminal_current_not_generator_best_return():
    task = joint.JointTrajectoryTask(
        "test_300x10", "independent_b_s0", 9907, 20
    )
    historical_best_return = pd.DataFrame({"value": [99, 99]})
    terminal_current = pd.DataFrame({"value": [1, 2]})

    report = runner._build_task_report(
        task,
        historical_best_return,
        _diagnostics(terminal_current),
        elapsed_sec=2.0,
        peak_allocated_bytes=100,
        peak_reserved_bytes=200,
    )

    assert report["output_table_identity"] == "terminal_current"
    assert report["terminal_table_sha256"] == runner._frame_sha256(
        terminal_current
    )
    assert report["terminal_table_sha256"] != runner._frame_sha256(
        historical_best_return
    )
    assert set(report) == set(protocol.ALLOWED_TASK_REPORT_FIELDS)


def test_gap_task_report_rechecks_8k_microsteps():
    task = joint.JointTrajectoryTask(
        "nltcs", "gap_l1_global_s8", 9907, 20
    )
    terminal = pd.DataFrame({"value": [0, 1]})

    report = runner._build_task_report(
        task,
        terminal,
        _diagnostics(terminal, arm=task.arm),
        elapsed_sec=1.0,
        peak_allocated_bytes=10,
        peak_reserved_bytes=20,
    )

    assert report["gap_reference_scale_established"] is True
    assert report["gap_microsteps"] == 24
    assert report["gap_expected_microsteps"] == 24
    assert report["gap_8k_identity"] is True
    assert report["nonfinite_count"] == 0


def test_gap_microstep_mismatch_fails_closed():
    task = joint.JointTrajectoryTask(
        "nltcs", "gap_l1_global_s8", 9907, 20
    )
    terminal = pd.DataFrame({"value": [0, 1]})
    diagnostics = _diagnostics(terminal, arm=task.arm)
    diagnostics["gap_l1_microsteps"] = 23

    with pytest.raises(RuntimeError, match="缺口微步累计不一致"):
        runner._build_task_report(
            task,
            terminal,
            diagnostics,
            elapsed_sec=1.0,
            peak_allocated_bytes=10,
            peak_reserved_bytes=20,
        )


def test_gap_applied_rounds_without_reference_scale_fail_closed():
    task = joint.JointTrajectoryTask(
        "nltcs", "gap_l1_global_s8", 9907, 20
    )
    terminal = pd.DataFrame({"value": [0, 1]})
    diagnostics = _diagnostics(terminal, arm=task.arm)
    diagnostics["gap_l1_reference_scale"] = None

    with pytest.raises(RuntimeError, match="仍未建立缺口定尺"):
        runner._build_task_report(
            task,
            terminal,
            diagnostics,
            elapsed_sec=1.0,
            peak_allocated_bytes=10,
            peak_reserved_bytes=20,
        )


def test_factor_task_requires_complete_finite_diagnostics():
    task = joint.JointTrajectoryTask(
        "nltcs", "factor_b_s8", 9907, 20
    )
    terminal = pd.DataFrame({"value": [0, 1]})
    diagnostics = _diagnostics(terminal, arm=task.arm)
    diagnostics["factorized_gibbs_attempt_diagnostics_history"][1] = [{}]

    with pytest.raises(RuntimeError, match="缺少有限值护栏"):
        runner._build_task_report(
            task,
            terminal,
            diagnostics,
            elapsed_sec=1.0,
            peak_allocated_bytes=10,
            peak_reserved_bytes=20,
        )


def test_exact_zero_stop_reports_zero_applied_rounds():
    task = joint.JointTrajectoryTask(
        "test_300x10", "independent_b_s0", 9907, 20
    )
    terminal = pd.DataFrame({"value": [0, 1]})
    diagnostics = _diagnostics(terminal)
    diagnostics.update(
        {
            "rounds_run": 1,
            "accept_history": [],
            "transition_clock_history": [],
            "initial_table_sha256": runner._frame_sha256(terminal),
            "termination_reason": "exact_residual",
        }
    )

    report = runner._build_task_report(
        task,
        pd.DataFrame({"value": [99, 99]}),
        diagnostics,
        elapsed_sec=0.25,
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
    )

    assert report["rounds_run"] == 0
    assert report["applied_round_count"] == 0
    assert report["termination_reason"] == "exact_residual"
    assert report["sec_per_round"] == 0.0


def test_generator_console_and_sensitive_exception_are_suppressed(
    monkeypatch, capsys
):
    def failing_generator(**_kwargs):
        print("loss=123.0 normalized_l1_error=0.5")
        raise ValueError("secret loss=123.0")

    monkeypatch.setattr(runner, "run_evolution", failing_generator)

    with pytest.raises(RuntimeError) as caught:
        runner._call_generator_silently("artificial_task")

    assert "loss" not in str(caught.value)
    assert "123" not in str(caught.value)
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""


def test_task_report_whitelist_rejects_quality_fields():
    row = _task_row(runner._frozen_tasks()[0])
    row["loss"] = 1.0

    with pytest.raises(RuntimeError, match="白名单"):
        runner._validate_task_report(row)


def test_atomic_report_contains_no_quality_and_refuses_overwrite(tmp_path):
    report = _valid_report()
    destination = tmp_path / "joint-smoke"

    path = runner._write_report_atomically(destination, report)

    assert path == destination / runner.REPORT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert runner._forbidden_keys(payload) == set()
    assert list(destination.iterdir()) == [path]
    with pytest.raises(FileExistsError, match="不覆盖"):
        runner._write_report_atomically(destination, report)


def test_invalid_report_creates_no_partial_output(tmp_path):
    report = _valid_report()
    report["loss"] = 1.0
    destination = tmp_path / "joint-smoke"

    with pytest.raises(RuntimeError, match="顶层报告字段"):
        runner._write_report_atomically(destination, report)

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_gpu_identity_and_resource_summary_are_strict():
    snapshot = {
        **protocol.EXPECTED_GPU,
        "cuda_available": True,
        "deterministic_algorithms": True,
    }
    assert runner._validate_gpu_snapshot(snapshot) == snapshot

    wrong = {**snapshot, "uuid": "wrong"}
    with pytest.raises(RuntimeError, match="显卡身份不一致"):
        runner._validate_gpu_snapshot(wrong)

    summary = runner._summarize_gpu_samples(
        [
            {
                "utilization_percent": 0,
                "memory_used_mib": 100,
                "memory_total_mib": 24_000,
            },
            {
                "utilization_percent": 80,
                "memory_used_mib": 12_000,
                "memory_total_mib": 24_000,
            },
        ]
    )
    assert summary["sample_count"] == 2
    assert summary["utilization_mean_percent"] == 40.0
    assert summary["utilization_max_percent"] == 80
    assert summary["utilization_nonzero_fraction"] == 0.5
    assert summary["memory_used_max_mib"] == 12_000


def test_cli_exposes_no_scientific_or_resource_overrides():
    parser = runner._build_parser()
    parsed = parser.parse_args(
        ["run", "--confirm-protocol-sha", protocol.FROZEN_PROTOCOL_SHA256]
    )
    assert set(vars(parsed)) == {"command", "confirm_protocol_sha"}

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--confirm-protocol-sha",
                protocol.FROZEN_PROTOCOL_SHA256,
                "--rounds",
                "1",
            ]
        )
