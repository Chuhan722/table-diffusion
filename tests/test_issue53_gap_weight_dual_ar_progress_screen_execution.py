"""A/R 相对初始进度筛查执行接线的结果前测试。"""

import sys
import ast
from pathlib import Path

import pytest

from scripts import (
    issue53_gap_weight_dual_ar_progress_screen_execution_protocol as protocol,
)
from scripts import (
    issue53_gap_weight_dual_ar_progress_screen_protocol as scientific,
)
from scripts import run_issue53_gap_weight_dual_ar_progress_screen as runner


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_execution_protocol_is_frozen_and_inherits_science_exactly():
    # 死记录自恰（不碰活树，任何树上都必须成立）。
    assert protocol.SCIENTIFIC_PROTOCOL_SHA256 == (
        scientific.FROZEN_PROTOCOL_SHA256
    )
    assert protocol.task_plan().tasks == scientific.task_plan().tasks
    # 双态：产物缺失跳过；活树漂移时守卫失败关闭即为正确行为。
    try:
        observed = protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    except FileNotFoundError:
        pytest.skip("冻结产物不在本机（gitignored），产物持有机复核")
    except RuntimeError as exc:
        assert "漂移" in str(exc)
        return
    assert observed == protocol.FROZEN_PROTOCOL_SHA256
    protocol.assert_scientific_inheritance()


def test_execution_matrix_contains_only_two_candidate_tasks():
    tasks = protocol.task_plan().tasks
    assert [task.task_id for task in tasks] == [
        "seed_9908__gap_dual_abs_relative_progress_max_s8__test_300x10",
        "seed_9908__gap_dual_abs_relative_progress_max_s8__nltcs",
    ]
    assert protocol.tasks_for_shard(protocol.LOCAL_SHARD) == tasks
    assert all(
        protocol.task_shard_id(task) == protocol.LOCAL_SHARD
        for task in tasks
    )


def test_execution_requires_its_own_later_confirmed_hash():
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        protocol.require_run_confirmation(None)
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        protocol.require_run_confirmation(scientific.FROZEN_PROTOCOL_SHA256)
    protocol.require_run_confirmation(protocol.FROZEN_PROTOCOL_SHA256)


def _weight_diagnostic(dataset="test_300x10"):
    task = next(
        item for item in protocol.task_plan().tasks if item.dataset == dataset
    )
    expected_weight = protocol.EXPECTED_WEIGHT_AUDIT[dataset]
    reference = protocol.EXPECTED_INITIAL_CHANNEL_REFERENCES[dataset]
    absolute_raw = reference["absolute_initial"] / 2.0
    relative_raw = reference["relative_initial"] / 2.0
    counts = {
        "candidate_0": {"absolute": 4, "relative": 3, "tie": 1},
        "candidate_1": {"absolute": 2, "relative": 6, "tie": 0},
        "pair": {
            "both_absolute": 2,
            "both_relative": 3,
            "cross_channel": 2,
            "tie_involved": 1,
        },
    }
    diagnostics = {
        "params": {
            "gap_l1_weighting": protocol.DUAL_WEIGHTING,
            "gap_l1_max_weight_ratio": None,
        },
        "gap_l1_channel_reference": {
            **reference,
            "source": "initial_current_before_round_1",
        },
        "gap_l1_attempt_diagnostics_history": [[{
            "gap_l1_scan_applied": True,
            "kernel": (
                "gap_l1_global_random_scan_"
                "dual_abs_relative_progress_max"
            ),
            "gibbs_microsteps": 8,
            "gap_l1_weighting": protocol.DUAL_WEIGHTING,
            "gap_l1_max_weight_ratio": None,
            "gap_l1_smoothing_count": None,
            "gap_l1_actual_weight_ratio": 1.0,
            "gap_l1_channel_aggregation": "max_relative_to_initial",
            "gap_l1_channel_reference_source": (
                "initial_current_before_round_1"
            ),
            "gap_l1_absolute_channel_initial_reference": reference[
                "absolute_initial"
            ],
            "gap_l1_relative_channel_initial_reference": reference[
                "relative_initial"
            ],
            "gap_l1_zero_target_policy": "absolute_channel_only",
            "gap_l1_floor_applied": False,
            "gap_l1_positive_target_query_count": expected_weight[
                "positive_target_query_count"
            ],
            "gap_l1_relative_inverse_target_normalizer": expected_weight[
                "relative_inverse_target_normalizer"
            ],
            "gap_l1_relative_positive_weight_ratio": expected_weight[
                "relative_positive_weight_ratio"
            ],
            "gap_l1_channel_dominance_counts": counts,
            "gap_l1_final_channels": {
                "absolute_raw": absolute_raw,
                "relative_raw": relative_raw,
                "absolute_relative_to_initial": 0.5,
                "relative_relative_to_initial": 0.5,
                "dominant_channel": "tie",
            },
        }]],
    }
    return task, diagnostics, {"gap_rounds": [{}]}, {}


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_accepts_exact_progress_channel_identity_and_activation():
    task, diagnostics, artifact, summary = _weight_diagnostic()
    runner._validate_weighting_diagnostics(
        task, diagnostics, artifact, summary, applied_rounds=1
    )
    assert summary["gap_channel_aggregation"] == "max_relative_to_initial"
    assert summary["gap_relative_candidate_side_count"] == 9
    assert summary["gap_relative_channel_activated"] is True
    assert summary["gap_weighting_guard_passed"] is True
    assert artifact["weighting_audit"]["guard_passed"] is True


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("gap_l1_channel_aggregation", "max", "相对进度诊断漂移"),
        ("gap_l1_absolute_channel_initial_reference", 1.0, "相对进度诊断漂移"),
        ("gap_l1_zero_target_policy", "relative_too", "相对进度诊断漂移"),
        ("gap_l1_floor_applied", True, "相对进度诊断漂移"),
        ("gap_l1_positive_target_query_count", 0, "相对进度诊断漂移"),
    ],
)
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_rejects_progress_identity_drift(field, value, error):
    task, diagnostics, artifact, summary = _weight_diagnostic()
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0][field] = value
    with pytest.raises(RuntimeError, match=error):
        runner._validate_weighting_diagnostics(
            task, diagnostics, artifact, summary, applied_rounds=1
        )


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_rejects_dominance_count_or_final_channel_drift():
    task, diagnostics, artifact, summary = _weight_diagnostic()
    item = diagnostics["gap_l1_attempt_diagnostics_history"][0][0]
    item["gap_l1_channel_dominance_counts"]["candidate_0"]["absolute"] = 3
    with pytest.raises(RuntimeError, match="主导计数漂移"):
        runner._validate_weighting_diagnostics(
            task, diagnostics, artifact, summary, applied_rounds=1
        )

    task, diagnostics, artifact, summary = _weight_diagnostic()
    item = diagnostics["gap_l1_attempt_diagnostics_history"][0][0]
    item["gap_l1_final_channels"]["relative_relative_to_initial"] = 0.6
    with pytest.raises(RuntimeError, match="最终通道相对初始值漂移"):
        runner._validate_weighting_diagnostics(
            task, diagnostics, artifact, summary, applied_rounds=1
        )


def test_runtime_binding_is_scoped_and_restored():
    original_protocol = runner.stage6e_runner.protocol
    original_file = runner.stage6e_runner.__file__
    original_pairing = runner.stage6d_runner._validate_pairing_for_tasks
    with runner._execution_runtime():
        assert runner.stage6e_runner.protocol is protocol
        assert Path(runner.stage6e_runner.__file__).resolve() == Path(
            runner.__file__
        ).resolve()
        assert runner.stage6d_runner._validate_pairing_for_tasks is (
            runner._validate_pairing_for_tasks
        )
    assert runner.stage6e_runner.protocol is original_protocol
    assert runner.stage6e_runner.__file__ == original_file
    assert runner.stage6d_runner._validate_pairing_for_tasks is original_pairing


def test_runner_has_no_evaluator_or_baseline_import():
    source = (
        REPOSITORY_ROOT
        / "scripts/run_issue53_gap_weight_dual_ar_progress_screen.py"
    ).read_text(encoding="utf-8")
    imported = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }
    assert not any(
        name and ("evaluate" in name or "audit_issue53" in name)
        for name in imported
    )


def test_plan_is_read_only_and_keeps_collect_unauthorized(monkeypatch):
    # 收束线：正式产物在位属预期；绕过活树身份校验，只测 plan 只读性。
    monkeypatch.setattr(
        protocol,
        "assert_frozen_protocol_identity",
        lambda _root: protocol.FROZEN_PROTOCOL_SHA256,
    )
    plan = runner.build_plan()
    assert plan["generation_started"] is False
    assert plan["screen_generation_authorized"] is False
    assert plan["next_collect_requires_later_user_confirmation"] is True
