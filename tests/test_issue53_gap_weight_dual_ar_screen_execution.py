"""A/R 双通道筛查执行接线的结果前测试。"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import issue53_gap_weight_dual_ar_screen_execution_protocol as protocol
from scripts import issue53_gap_weight_dual_ar_screen_protocol as scientific
from scripts import run_issue53_gap_weight_dual_ar_screen as runner
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import AttributeBlock, Schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_execution_protocol_is_frozen_and_inherits_science_exactly():
    assert protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT) == (
        protocol.FROZEN_PROTOCOL_SHA256
    )
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
        "seed_9908__gap_dual_abs_relative_max_s8__test_300x10",
        "seed_9908__gap_dual_abs_relative_max_s8__nltcs",
    ]
    assert protocol.tasks_for_shard(protocol.LOCAL_SHARD) == tasks
    assert all(
        protocol.task_shard_id(task) == protocol.LOCAL_SHARD
        for task in tasks
    )
    with pytest.raises(ValueError, match="未知 A/R"):
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
    expected = protocol.EXPECTED_WEIGHT_AUDIT[dataset]
    diagnostics = {
        "params": {
            "gap_l1_weighting": protocol.DUAL_WEIGHTING,
            "gap_l1_max_weight_ratio": None,
        },
        "gap_l1_attempt_diagnostics_history": [[{
            "gap_l1_scan_applied": True,
            "kernel": "gap_l1_global_random_scan_dual_abs_relative_max",
            "gap_l1_weighting": protocol.DUAL_WEIGHTING,
            "gap_l1_max_weight_ratio": None,
            "gap_l1_smoothing_count": None,
            "gap_l1_actual_weight_ratio": 1.0,
            "gap_l1_channel_aggregation": "max",
            "gap_l1_absolute_channel_query_weighting": "uniform",
            "gap_l1_relative_channel_query_weighting": (
                "normalized_inverse_positive_target"
            ),
            "gap_l1_zero_target_policy": "absolute_channel_only",
            "gap_l1_floor_applied": False,
            "gap_l1_positive_target_query_count": expected[
                "positive_target_query_count"
            ],
            "gap_l1_relative_inverse_target_normalizer": expected[
                "relative_inverse_target_normalizer"
            ],
            "gap_l1_relative_positive_weight_ratio": expected[
                "relative_positive_weight_ratio"
            ],
        }]],
    }
    return task, diagnostics, {"gap_rounds": [{}]}, {}


def test_collector_accepts_exact_dual_channel_weight_identity():
    task, diagnostics, artifact, summary = _weight_diagnostic()
    runner._validate_weighting_diagnostics(
        task, diagnostics, artifact, summary, applied_rounds=1
    )
    expected = protocol.EXPECTED_WEIGHT_AUDIT[task.dataset]
    assert summary["gap_weighting"] == protocol.DUAL_WEIGHTING
    assert summary["gap_channel_aggregation"] == "max"
    assert summary["gap_zero_target_policy"] == "absolute_channel_only"
    assert summary["gap_floor_applied"] is False
    assert summary["gap_expected_positive_target_query_count"] == (
        expected["positive_target_query_count"]
    )
    assert summary[
        "gap_relative_inverse_target_normalizer_min"
    ] == expected["relative_inverse_target_normalizer"]
    assert summary[
        "gap_relative_positive_weight_ratio_max"
    ] == expected["relative_positive_weight_ratio"]
    assert summary["gap_weighting_guard_passed"] is True
    assert artifact["weighting_audit"]["guard_passed"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("gap_l1_channel_aggregation", "sum"),
        ("gap_l1_zero_target_policy", "relative_too"),
        ("gap_l1_floor_applied", True),
        ("gap_l1_positive_target_query_count", 46),
        ("gap_l1_relative_inverse_target_normalizer", 5.0),
        ("gap_l1_relative_positive_weight_ratio", 8.0),
        ("gap_l1_smoothing_count", 1.0),
        ("gap_l1_max_weight_ratio", 8.0),
    ],
)
def test_collector_rejects_any_dual_channel_identity_drift(field, value):
    task, diagnostics, artifact, summary = _weight_diagnostic()
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0][field] = value
    with pytest.raises(RuntimeError, match="A/R 双通道权重诊断漂移"):
        runner._validate_weighting_diagnostics(
            task, diagnostics, artifact, summary, applied_rounds=1
        )


def test_collector_rejects_configured_but_never_executed_scan():
    task, diagnostics, artifact, summary = _weight_diagnostic()
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0] = {
        "gap_l1_scan_applied": False,
        "kernel": "independent_b_unscaled_gap_fallback",
    }
    with pytest.raises(RuntimeError, match="从未实际执行"):
        runner._validate_weighting_diagnostics(
            task, diagnostics, artifact, summary, applied_rounds=1
        )


def test_real_one_round_transition_passes_dual_channel_guard():
    schema = Schema([
        AttributeBlock(
            name="a",
            type="categorical",
            description="a",
            values=[0, 1],
        )
    ])
    query_file = REPOSITORY_ROOT / protocol.DATASETS["test_300x10"][
        "queries"
    ]
    target_rows = json.loads(query_file.read_text(encoding="utf-8"))[
        "queries"
    ]
    targets = np.asarray([row["result"] for row in target_rows], dtype=float)
    queries = [{
        "conditions": [{
            "attribute": "a", "operator": "==", "value": 1
        }]
    } for _ in targets]
    _, diagnostics = run_evolution(
        targets,
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
        gap_l1_weighting=protocol.DUAL_WEIGHTING,
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
    expected = protocol.EXPECTED_WEIGHT_AUDIT["test_300x10"]
    assert summary["gap_weighting_guard_passed"] is True
    assert summary[
        "gap_relative_inverse_target_normalizer_max"
    ] == expected["relative_inverse_target_normalizer"]
    assert summary[
        "gap_relative_positive_weight_ratio_max"
    ] == expected["relative_positive_weight_ratio"]
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
    source = (
        REPOSITORY_ROOT / "scripts/run_issue53_gap_weight_dual_ar_screen.py"
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


def test_plan_and_preflight_are_read_only_and_keep_collect_unauthorized():
    assert not protocol.OUTPUT_DIR.exists()
    assert not protocol.SHARD_OUTPUT_ROOT.exists()
    plan = runner.build_plan()
    preflight = runner.preflight()
    assert plan["generation_started"] is False
    assert plan["screen_generation_authorized"] is False
    assert preflight["generation_started"] is False
    assert preflight["screen_generation_authorized"] is False
    assert preflight["requires_explicit_later_user_confirmation"] is True
    assert not protocol.OUTPUT_DIR.exists()
    assert not protocol.SHARD_OUTPUT_ROOT.exists()
