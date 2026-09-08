"""平方根查询权重筛查执行接线的结果前测试。"""

import sys
import ast
from pathlib import Path

import numpy as np
import pytest

from scripts import issue53_gap_weight_sqrt_screen_execution_protocol as protocol
from scripts import issue53_gap_weight_sqrt_screen_protocol as scientific
from scripts import run_issue53_gap_weight_sqrt_screen as runner
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import AttributeBlock, Schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_execution_protocol_is_frozen_and_inherits_science_exactly():
    # 收束线：活树身份守卫预期失败关闭；科学继承与任务矩阵仍须自恰。
    with pytest.raises(RuntimeError, match="漂移"):
        protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    protocol.assert_scientific_inheritance()
    assert protocol.SCIENTIFIC_PROTOCOL_SHA256 == (
        scientific.FROZEN_PROTOCOL_SHA256
    )
    assert protocol.task_plan().tasks == scientific.task_plan().tasks
    assert protocol.generator_params_manifest_matrix() == (
        scientific.generator_params_manifest_matrix()
    )


def test_execution_matrix_contains_only_two_candidate_tasks():
    tasks = protocol.task_plan().tasks
    assert [task.task_id for task in tasks] == [
        "seed_9908__gap_sqrt_target_s8__test_300x10",
        "seed_9908__gap_sqrt_target_s8__nltcs",
    ]
    assert protocol.tasks_for_shard(protocol.LOCAL_SHARD) == tasks
    assert all(
        protocol.task_shard_id(task) == protocol.LOCAL_SHARD
        for task in tasks
    )
    with pytest.raises(ValueError, match="未知平方根"):
        protocol.tasks_for_shard("other")


def test_execution_wrapper_rejects_other_seed_and_requires_own_hash():
    assert protocol.task_generator_params(
        "nltcs", scientific.CANDIDATE_ARM, 9908
    ) == scientific.task_generator_params(
        "nltcs", scientific.CANDIDATE_ARM
    )
    with pytest.raises(ValueError, match="不是 9908"):
        protocol.task_generator_params(
            "nltcs", scientific.CANDIDATE_ARM, 9909
        )
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        protocol.require_run_confirmation(None)
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        protocol.require_run_confirmation(
            scientific.FROZEN_PROTOCOL_SHA256
        )
    protocol.require_run_confirmation(protocol.FROZEN_PROTOCOL_SHA256)


def _weight_diagnostic(dataset="test_300x10"):
    task = next(
        item for item in protocol.task_plan().tasks if item.dataset == dataset
    )
    ratio = protocol.EXPECTED_WEIGHT_AUDIT[dataset]["actual_weight_ratio"]
    diagnostics = {
        "params": {
            "gap_l1_weighting": protocol.SQRT_WEIGHTING,
            "gap_l1_max_weight_ratio": None,
        },
        "gap_l1_attempt_diagnostics_history": [[{
            "gap_l1_scan_applied": True,
            "kernel": "gap_l1_global_random_scan_sqrt_target_relative",
            "gap_l1_weighting": protocol.SQRT_WEIGHTING,
            "gap_l1_max_weight_ratio": None,
            "gap_l1_smoothing_count": None,
            "gap_l1_target_count_quantum": 1.0,
            "gap_l1_actual_weight_ratio": ratio,
        }]],
    }
    return task, diagnostics, {"gap_rounds": [{}]}, {}


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_accepts_exact_sqrt_weight_identity():
    task, diagnostics, artifact, summary = _weight_diagnostic()
    runner._validate_weighting_diagnostics(
        task,
        diagnostics,
        artifact,
        summary,
        applied_rounds=1,
    )
    expected = protocol.EXPECTED_WEIGHT_AUDIT[task.dataset][
        "actual_weight_ratio"
    ]
    assert summary == {
        "gap_weighting": protocol.SQRT_WEIGHTING,
        "gap_max_weight_ratio": None,
        "gap_smoothing_count": None,
        "gap_target_count_quantum": 1.0,
        "gap_expected_actual_weight_ratio": expected,
        "gap_weighting_scan_round_count": 1,
        "gap_actual_weight_ratio_min": expected,
        "gap_actual_weight_ratio_max": expected,
        "gap_weighting_guard_passed": True,
    }
    assert artifact["weighting_audit"]["guard_passed"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("gap_l1_actual_weight_ratio", 9.0),
        ("gap_l1_target_count_quantum", 8.0),
        ("gap_l1_smoothing_count", 1.0),
        ("gap_l1_max_weight_ratio", 8.0),
    ],
)
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_rejects_any_sqrt_weight_identity_drift(field, value):
    task, diagnostics, artifact, summary = _weight_diagnostic()
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0][field] = value
    with pytest.raises(RuntimeError, match="平方根权重诊断漂移"):
        runner._validate_weighting_diagnostics(
            task,
            diagnostics,
            artifact,
            summary,
            applied_rounds=1,
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_rejects_configured_but_never_executed_scan():
    task, diagnostics, artifact, summary = _weight_diagnostic()
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0] = {
        "gap_l1_scan_applied": False,
        "kernel": "independent_b_unscaled_gap_fallback",
    }
    with pytest.raises(RuntimeError, match="从未实际执行"):
        runner._validate_weighting_diagnostics(
            task,
            diagnostics,
            artifact,
            summary,
            applied_rounds=1,
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_real_one_round_transition_passes_sqrt_guard():
    schema = Schema([
        AttributeBlock(
            name="a",
            type="categorical",
            description="a",
            values=[0, 1],
        )
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1}
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 0}
        ]},
    ]
    _, diagnostics = run_evolution(
        np.asarray([0.0, 88.0]),
        queries,
        schema,
        n_records=300,
        n_rounds=1,
        seed=17,
        rho=0.8,
        eta=0.5,
        mu=0.1,
        device="numpy",
        distance_mode="geometric",
        residual_directed_diffusion=True,
        diffusion_direction_strength=2.0,
        diffusion_direction_normalization="initial_rms",
        log_every=100,
        tol=float("inf"),
        gap_l1_sweeps=8,
        gap_l1_weighting=protocol.SQRT_WEIGHTING,
        record_transition_clocks=True,
    )
    task = next(
        item
        for item in protocol.task_plan().tasks
        if item.dataset == "test_300x10"
    )
    artifact, summary = runner._extract_transition_audit(
        task, diagnostics, applied_rounds=1
    )
    assert summary["gap_weighting_guard_passed"] is True
    assert summary["gap_actual_weight_ratio_max"] == (
        protocol.EXPECTED_WEIGHT_AUDIT["test_300x10"][
            "actual_weight_ratio"
        ]
    )
    assert artifact["weighting_audit"]["guard_passed"] is True


def test_runtime_binding_is_scoped_and_restored():
    original_protocol = runner.stage6e_runner.protocol
    original_file = runner.stage6e_runner.__file__
    original_pairing = runner.stage6d_runner._validate_pairing_for_tasks
    original_transition = runner.stage6d_runner._extract_transition_audit
    with runner._execution_runtime():
        assert runner.stage6e_runner.protocol is protocol
        assert Path(runner.stage6e_runner.__file__).resolve() == Path(
            runner.__file__
        ).resolve()
        assert runner.stage6d_runner._validate_pairing_for_tasks is (
            runner._validate_pairing_for_tasks
        )
        assert runner.stage6d_runner._extract_transition_audit is (
            runner._extract_transition_audit
        )
    assert runner.stage6e_runner.protocol is original_protocol
    assert runner.stage6e_runner.__file__ == original_file
    assert runner.stage6d_runner._validate_pairing_for_tasks is original_pairing
    assert runner.stage6d_runner._extract_transition_audit is original_transition


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_pairing_requires_exact_two_task_order(monkeypatch):
    tasks = protocol.task_plan().tasks
    rows = [
        {
            "task_id": task.task_id,
            "seed": task.seed,
            "dataset": task.dataset,
            "arm": task.arm,
        }
        for task in tasks
    ]
    monkeypatch.setattr(runner, "_validate_case_row", lambda *_args: None)
    runner._validate_pairing_for_tasks(rows, tasks)
    with pytest.raises(RuntimeError, match="顺序漂移"):
        runner._validate_pairing_for_tasks(list(reversed(rows)), tasks)


def test_runner_has_no_evaluator_or_baseline_import():
    source = (REPOSITORY_ROOT / "scripts/run_issue53_gap_weight_sqrt_screen.py").read_text(
        encoding="utf-8"
    )
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        name and ("evaluate" in name or "audit_issue53" in name)
        for name in imported
    )
    assert "evaluation_report.json" not in source
    assert "screen_metrics.csv" not in source


def test_collect_wrong_hash_fails_before_shared_runner(monkeypatch):
    monkeypatch.setattr(
        runner.stage6e_runner,
        "collect",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("错误哈希不得进入共享采集器")
        ),
    )
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        runner.collect("0" * 64)
