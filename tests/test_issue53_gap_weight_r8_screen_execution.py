from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import audit_issue53_gap_weight_r8_screen as auditor
from scripts import evaluate_issue53_gap_weight_r8_screen as evaluator
from scripts import issue53_gap_weight_r8_screen_execution_protocol as protocol
from scripts import issue53_gap_weight_r8_screen_protocol as scientific
from scripts import run_issue53_gap_weight_r8_screen as runner
from table_diffevo.evolution import run_evolution
from table_diffevo.schema import AttributeBlock, Schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_execution_delta_document_sources_and_manifest_are_frozen():
    # 收束线：只校验死记录自恰；活实现源码此后演进属预期，守卫应失败关闭。
    assert protocol.file_sha256(REPOSITORY_ROOT / protocol.PROTOCOL_DOC) == (
        protocol.PROTOCOL_DOC_SHA256
    )
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    with pytest.raises(RuntimeError, match="实现源码漂移"):
        protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)


def test_execution_delta_inherits_scientific_protocol_without_changes():
    protocol.assert_scientific_inheritance()

    assert protocol.SCIENTIFIC_PROTOCOL_SHA256 == (
        "50e49ab9786db1dd74476363e0489eeedf76bbdc44a1aa6940a1c7f80e34147e"
    )
    assert protocol.task_plan().tasks == scientific.task_plan().tasks
    assert protocol.generator_params_manifest_matrix() == (
        scientific.generator_params_manifest_matrix()
    )
    assert protocol.generator_params_manifest_sha256() == (
        "7e99f7a63fcaae34d359d2cce697be10ad5c4675363dbaf9276a2be6bd5e5630"
    )
    manifest = protocol.frozen_protocol_manifest()
    assert manifest["scientific_protocol"]["scientific_fields_modified_by_delta"] == []
    assert manifest["authorization_boundary"] == {
        "runner_wired": True,
        "generation_authorized_at_execution_freeze": False,
        "matching_hash_alone_authorizes_generation": False,
        "explicit_later_user_confirmation_required": True,
        "current_wiring_work_may_run_generator": False,
        "pr69_change_allowed": False,
    }


def test_execution_task_and_single_shard_matrix_are_exact():
    tasks = protocol.task_plan().tasks

    assert [task.task_id for task in tasks] == [
        "seed_9908__gap_legacy_s8__test_300x10",
        "seed_9908__gap_bounded_r8_s8__test_300x10",
        "seed_9908__gap_legacy_s8__nltcs",
        "seed_9908__gap_bounded_r8_s8__nltcs",
    ]
    assert protocol.tasks_for_shard(protocol.LOCAL_SHARD) == tasks
    assert protocol.SHARD_BLOCKS == {
        protocol.LOCAL_SHARD: (
            (9908, "test_300x10"),
            (9908, "nltcs"),
        )
    }
    assert all(protocol.task_shard_id(task) == protocol.LOCAL_SHARD for task in tasks)
    with pytest.raises(ValueError, match="未知 R8"):
        protocol.tasks_for_shard("other")


def test_execution_generator_wrapper_rejects_any_other_seed():
    expected = scientific.task_generator_params(
        "nltcs", "gap_bounded_r8_s8"
    )
    assert protocol.task_generator_params(
        "nltcs", "gap_bounded_r8_s8", 9908
    ) == expected
    with pytest.raises(ValueError, match="不是 9908"):
        protocol.task_generator_params(
            "nltcs", "gap_bounded_r8_s8", 9909
        )
    with pytest.raises(ValueError, match="不是 9908"):
        protocol.generator_params_manifest(
            "nltcs", "gap_bounded_r8_s8", 9909
        )


def test_run_confirmation_is_a_separate_exact_execution_hash_gate():
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        protocol.require_run_confirmation(None)
    with pytest.raises(PermissionError, match="后续用户明确授权"):
        protocol.require_run_confirmation(scientific.FROZEN_PROTOCOL_SHA256)
    protocol.require_run_confirmation(protocol.FROZEN_PROTOCOL_SHA256)


def _weight_diagnostic(
    *, arm: str, dataset: str = "test_300x10"
) -> tuple[object, dict, dict, dict]:
    task = next(
        item
        for item in protocol.task_plan().tasks
        if item.arm == arm and item.dataset == dataset
    )
    if arm == "gap_bounded_r8_s8":
        params = {
            "gap_l1_weighting": protocol.BOUNDED_WEIGHTING,
            "gap_l1_max_weight_ratio": 8.0,
        }
        attempt = {
            "gap_l1_scan_applied": True,
            "kernel": "gap_l1_global_random_scan_bounded_relative",
            "gap_l1_weighting": protocol.BOUNDED_WEIGHTING,
            "gap_l1_max_weight_ratio": 8.0,
            "gap_l1_smoothing_count": protocol.smoothing_count(dataset),
            "gap_l1_actual_weight_ratio": 8.0,
        }
    else:
        params = {}
        attempt = {
            "gap_l1_scan_applied": True,
            "kernel": "gap_l1_global_random_scan",
        }
    diagnostics = {
        "params": params,
        "gap_l1_attempt_diagnostics_history": [[attempt]],
    }
    artifact = {"gap_rounds": [{"scan_applied": True}]}
    summary: dict = {}
    return task, diagnostics, artifact, summary


@pytest.mark.parametrize(
    "arm,expected,ratio,smoothing",
    [
        ("gap_legacy_s8", "legacy_relative", None, None),
        ("gap_bounded_r8_s8", "bounded_relative", 8.0, 300 / 7),
    ],
)
@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_weighting_guard_accepts_both_frozen_arms(
    arm, expected, ratio, smoothing
):
    task, diagnostics, artifact, summary = _weight_diagnostic(arm=arm)

    runner._validate_weighting_diagnostics(
        task,
        diagnostics,
        artifact,
        summary,
        applied_rounds=1,
    )

    assert summary["gap_weighting"] == expected
    assert summary["gap_max_weight_ratio"] == ratio
    assert summary["gap_smoothing_count"] == smoothing
    assert summary["gap_weighting_scan_round_count"] == 1
    assert summary["gap_weighting_guard_passed"] is True
    assert artifact["weighting_audit"]["guard_passed"] is True
    observed = artifact["gap_rounds"][0]
    assert observed["observed_weighting"] == expected


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
def test_collector_rejects_rounded_r8_smoothing():
    task, diagnostics, artifact, summary = _weight_diagnostic(
        arm="gap_bounded_r8_s8"
    )
    diagnostics["gap_l1_attempt_diagnostics_history"][0][0][
        "gap_l1_smoothing_count"
    ] = round(protocol.smoothing_count(task.dataset))

    with pytest.raises(RuntimeError, match="R8 权重诊断漂移"):
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
def test_collector_rejects_configured_but_never_executed_r8_scan():
    task, diagnostics, artifact, summary = _weight_diagnostic(
        arm="gap_bounded_r8_s8"
    )
    attempt = diagnostics["gap_l1_attempt_diagnostics_history"][0][0]
    attempt.clear()
    attempt.update(
        {
            "gap_l1_scan_applied": False,
            "kernel": "independent_b_unscaled_gap_fallback",
        }
    )
    artifact["gap_rounds"][0]["scan_applied"] = False

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
@pytest.mark.parametrize("arm", protocol.ARM_ORDER)
def test_real_synthetic_transition_diagnostics_pass_collector_guard(arm):
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
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1}
            ]
        },
        {
            "conditions": [
                {"attribute": "b", "operator": "==", "value": 1}
            ]
        },
        {
            "conditions": [
                {"attribute": "a", "operator": "==", "value": 1},
                {"attribute": "b", "operator": "==", "value": 1},
            ]
        },
    ]
    generator_params = {
        "n_records": 300,
        "n_rounds": 1,
        "seed": 17,
        "rho": 0.8,
        "eta": 0.5,
        "mu": 0.1,
        "device": "numpy",
        "distance_mode": "geometric",
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 2.0,
        "diffusion_direction_normalization": "initial_rms",
        "log_every": 100,
        "tol": float("inf"),
        "gap_l1_sweeps": 8,
        "record_transition_clocks": True,
        **protocol.arm_kernel_params(arm),
    }
    _table, diagnostics = run_evolution(
        np.asarray([80.5, 150.5, 40.5]),
        queries,
        schema,
        **generator_params,
    )
    task = next(
        item
        for item in protocol.task_plan().tasks
        if item.dataset == "test_300x10" and item.arm == arm
    )

    artifact, summary = runner._extract_transition_audit(
        task, diagnostics, applied_rounds=1
    )

    assert summary["gap_weighting"] == protocol.arm_kernel_params(arm)[
        "gap_l1_weighting"
    ]
    assert summary["gap_weighting_scan_round_count"] == 1
    assert summary["gap_weighting_guard_passed"] is True
    assert artifact["weighting_audit"]["guard_passed"] is True
    if arm == "gap_bounded_r8_s8":
        assert summary["gap_smoothing_count"] == 300 / 7
        assert summary["gap_actual_weight_ratio_max"] <= 8.0
    else:
        assert summary["gap_smoothing_count"] is None
        assert summary["gap_actual_weight_ratio_max"] is None


def _paired_rows(dataset: str) -> tuple[tuple[object, ...], list[dict]]:
    tasks = tuple(
        task for task in protocol.task_plan().tasks if task.dataset == dataset
    )
    rows = [
        {
            "task_id": task.task_id,
            "seed": task.seed,
            "dataset": task.dataset,
            "arm": task.arm,
            "initial_table_sha256": "a" * 64,
            "primary_rng_post_initialization_sha256": "b" * 64,
            "query_identity_sha256": "c" * 64,
            "target_vector_sha256": "d" * 64,
            "trace_query_identity_sha256": "e" * 64,
            "trace_target_vector_sha256": "f" * 64,
            "direction_reference_scale": float(index + 1),
        }
        for index, task in enumerate(tasks)
    ]
    return tasks, rows


def test_two_arm_pairing_allows_derived_reference_scale_to_differ(monkeypatch):
    tasks, rows = _paired_rows("nltcs")
    monkeypatch.setattr(runner, "_validate_case_row", lambda _task, _row: None)

    runner._validate_pairing_for_tasks(rows, tasks)

    rows[1]["initial_table_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="initial_table_sha256"):
        runner._validate_pairing_for_tasks(rows, tasks)


def test_execution_runtime_binds_spawn_worker_and_restores_legacy_modules():
    original_stage6e_protocol = runner.stage6e_runner.protocol
    original_stage6e_file = runner.stage6e_runner.__file__
    original_pairing = runner.stage6d_runner._validate_pairing_for_tasks
    original_transition = runner.stage6d_runner._extract_transition_audit

    with runner._execution_runtime():
        assert runner.stage6e_runner.protocol is protocol
        assert Path(runner.stage6e_runner.__file__).resolve() == Path(
            runner.__file__
        ).resolve()
        assert runner.stage6e_runner._execute_trajectory_task is (
            runner._execute_trajectory_task
        )
        assert runner.stage6d_runner._validate_pairing_for_tasks is (
            runner._validate_pairing_for_tasks
        )
        assert runner.stage6d_runner._extract_transition_audit is (
            runner._extract_transition_audit
        )
        with runner.stage6e_runner._stage6d_runtime():
            assert runner.stage6d_runner.protocol is protocol
            assert runner.stage6d_runner._execute_trajectory_task is (
                runner._execute_trajectory_task
            )
            assert runner.stage6d_runner._validate_case_row is (
                runner._validate_case_row
            )
            assert runner.stage6d_runner._collection_report is (
                runner._collection_report
            )

    assert runner.stage6e_runner.protocol is original_stage6e_protocol
    assert runner.stage6e_runner.__file__ == original_stage6e_file
    assert runner.stage6d_runner._validate_pairing_for_tasks is original_pairing
    assert runner.stage6d_runner._extract_transition_audit is original_transition


def test_collection_report_uses_four_cases_and_two_pair_blocks(monkeypatch):
    rows = [
        {
            "task_id": task.task_id,
            "applied_rounds": index + 1,
            "elapsed_sec": float(index + 2),
            "termination_reason": "early_stopped",
        }
        for index, task in enumerate(protocol.task_plan().tasks)
    ]
    monkeypatch.setattr(
        runner, "_validate_pairing_for_tasks", lambda _rows, _tasks: None
    )
    monkeypatch.setattr(
        runner.stage6d_runner, "_validate_numeric_rows", lambda _rows: None
    )
    shard = protocol.LOCAL_SHARD
    report = runner._collection_report(
        execution_commit="a" * 40,
        runner_sha256=protocol.IMPLEMENTATION_SOURCES["collector"]["sha256"],
        generator_params_manifest_sha256=(
            protocol.generator_params_manifest_sha256()
        ),
        input_sha256={"inputs": {}},
        shard_reports={
            shard: {
                "environment": {"gpu": {}},
                "execution": {"elapsed_sec_this_invocation": 1.0},
            }
        },
        shard_report_sha256={shard: "b" * 64},
        rows=rows,
    )

    assert report["case_count"] == 4
    assert report["paired_dataset_seed_count"] == 2
    assert report["execution"]["task_order"] == "seed_then_dataset_then_arm"
    assert report["timing_summary"] == {
        "sum_case_elapsed_sec": 14.0,
        "total_applied_rounds": 10,
        "average_sec_per_applied_round": 1.4,
    }
    assert report["termination_reason_counts"] == {
        "fit_target_reached": 0,
        "early_stopped": 4,
        "resource_cap_reached": 0,
    }
    assert report["collection_audit"]["all_4_cases_present"] is True
    assert "all_30_cases_present" not in report["collection_audit"]
    assert report["quality_results_published_by_collection"] is False


def _query_payload(dataset: str) -> list[dict]:
    path = REPOSITORY_ROOT / protocol.DATASETS[dataset]["queries"]
    return json.loads(path.read_text(encoding="utf-8"))["queries"]


@pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="冻结筛查脚本使用 zip(strict=True)（需 py3.10+）；收束线按字节冻结不回改",
)
@pytest.mark.parametrize("dataset", protocol.DATASET_ORDER)
def test_primary_and_independent_vector_arithmetic_match(dataset):
    raw = _query_payload(dataset)
    targets = [row["result"] for row in raw]
    answers = list(targets)
    n_records = protocol.DATASETS[dataset]["n_records"]
    answers[0] = targets[0] - 1 if targets[0] == n_records else targets[0] + 1
    answers[-1] = targets[-1] - 1 if targets[-1] == n_records else targets[-1] + 1

    primary = evaluator._vector_metrics(
        dataset, raw, targets, answers, n_records
    )
    independent = auditor._independent_vector_metrics(
        dataset, raw, targets, answers, n_records
    )

    assert primary == independent
    assert primary["bounded_r8_smoothing_count"] == n_records / 7
    assert sum(
        item["query_count"] for item in primary["target_count_bins"].values()
    ) == len(raw)
    assert primary["absolute_count_error_sum"] == 2


def test_ratio_zero_baseline_rule_is_json_safe():
    assert evaluator._ratio_record(0.0, 0.0) == {
        "bounded_r8": 0.0,
        "legacy": 0.0,
        "bounded_over_legacy": 1.0,
        "ratio_state": "both_zero_ratio_one",
        "lower_is_better": True,
    }
    infinite = evaluator._ratio_record(1.0, 0.0)
    assert infinite["bounded_over_legacy"] is None
    assert infinite["ratio_state"] == "positive_infinity"
    assert evaluator._decision_ratio(infinite) == float("inf")
    json.dumps(infinite, allow_nan=False)


def _state_metrics(dataset: str, value: float) -> dict:
    return {
        "kind": "fixed_checkpoint",
        "state_index": 0,
        "round": 0,
        "phase": "initial",
        "squared_loss": value,
        "query_count": protocol.DATASETS[dataset]["query_count"],
        "absolute_count_error_sum": 1,
        "normalized_l1": value,
        "legacy_relative_gap_proxy": value,
        "bounded_r8_gap_proxy": value,
        "bounded_r8_smoothing_count": protocol.smoothing_count(dataset),
        "target_count_bins": {
            item["name"]: {
                "query_count": item["query_count"],
                "absolute_count_error_sum": 1,
                "absolute_count_error_mean": value,
                "normalized_l1": value,
                "membership_sha256": item["membership_sha256"],
            }
            for item in protocol.TARGET_COUNT_BINS[dataset]
        },
        "query_orders": {
            str(order): {
                "query_count": count,
                "absolute_count_error_sum": 1,
                "absolute_count_error_mean": value,
                "normalized_l1": value,
            }
            for order, count in protocol.DATASETS[dataset]["order_counts"].items()
        },
    }


def _screen_cases(
    *, nltcs_common_bounded: float = 9.0
) -> list[dict]:
    cases = []
    for dataset in protocol.DATASET_ORDER:
        group_names = (
            protocol.TEST_GROUP_ORDER
            if dataset == "test_300x10"
            else tuple(protocol.NLTCS_GROUP_COUNTS)
        )
        for arm in protocol.ARM_ORDER:
            bounded = arm == "gap_bounded_r8_s8"
            overall = (
                (10.5 if bounded else 10.0)
                if dataset == "test_300x10"
                else (9.0 if bounded else 10.0)
            )
            bins = {}
            for item in protocol.TARGET_COUNT_BINS[dataset]:
                value = 10.0
                if bounded:
                    if dataset == "nltcs" and item["name"] == protocol.NLTCS_COMMON_BIN:
                        value = nltcs_common_bounded
                    elif dataset == "nltcs" and item["name"] == protocol.NLTCS_RARE_BIN:
                        value = 12.5
                    else:
                        value = 10.0
                bins[item["name"]] = {
                    "absolute_count_error_mean": value
                }
            state = _state_metrics(dataset, overall)
            cases.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "seed": 9908,
                    "metrics": {
                        "measured": {
                            "overall": {"normalized_l1_mean": overall},
                            "proxy_objectives": {
                                "legacy_relative_gap": overall,
                                "bounded_r8_gap": overall,
                            },
                            "target_count_bins": bins,
                            "by_order": {
                                str(order): {
                                    "absolute_count_error_mean": overall
                                }
                                for order in protocol.DATASETS[dataset]["order_counts"]
                            },
                        },
                        "offline_query_groups": {
                            name: {
                                "normalized_l1_mean": (
                                    10.5 if bounded else 10.0
                                )
                            }
                            for name in group_names
                        },
                        "validity": {"valid_row_rate": 1.0},
                    },
                    "trajectory": {
                        "fixed_checkpoints": [state],
                        "terminal": state,
                    },
                    "kernel_audit": {
                        "gap_weighting_guard_passed": True
                    },
                    "cost": {"elapsed_sec": float(2 if bounded else 1)},
                }
            )
    return cases


def test_primary_and_independent_summary_reproduce_frozen_classification():
    cases = _screen_cases()

    primary = evaluator._summary_and_decision(cases)
    independent_summary, independent_valid, independent_classification = (
        auditor._summary_independent(cases)
    )

    assert primary["datasets"] == independent_summary
    assert primary["execution_valid"] is independent_valid is True
    assert primary["frozen_classification"] == independent_classification == (
        "advance_to_five_seed_confirmation"
    )

    failed = evaluator._summary_and_decision(
        _screen_cases(nltcs_common_bounded=10.0)
    )
    assert failed["frozen_classification"] == (
        "bounded_weighting_mechanism_not_supported"
    )


def test_primary_and_independent_csv_rows_match():
    cases = _screen_cases()

    assert evaluator._csv_rows(cases) == auditor._csv_rows_independent(cases)
    assert all(
        set(row) == set(evaluator.CSV_FIELDS)
        for row in evaluator._csv_rows(cases)
    )


def test_independent_auditor_does_not_import_r8_evaluator():
    source_path = REPOSITORY_ROOT / "scripts/audit_issue53_gap_weight_r8_screen.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    assert "scripts.evaluate_issue53_gap_weight_r8_screen" not in imported


@pytest.mark.parametrize(
    "function",
    [evaluator._artifact_path, auditor._safe_artifact],
)
def test_artifact_paths_fail_closed_on_escape(function):
    with pytest.raises(ValueError, match="路径越界"):
        function(REPOSITORY_ROOT, "../outside")
    with pytest.raises(ValueError, match="路径越界"):
        function(REPOSITORY_ROOT, "/absolute")


def test_runner_plan_is_read_only_and_never_calls_generator(monkeypatch):
    # 收束线：绕过活树身份校验，只测 plan 只读性。
    monkeypatch.setattr(
        protocol,
        "assert_frozen_protocol_identity",
        lambda _root: protocol.FROZEN_PROTOCOL_SHA256,
    )
    monkeypatch.setattr(
        runner.stage6d_runner,
        "_call_generator_silently",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("plan 不得调用生成器")
        ),
    )

    plan = runner.build_plan()

    assert plan["trajectory_count"] == 4
    assert plan["paired_block_count"] == 2
    assert plan["generation_started"] is False
    assert plan["screen_generation_authorized"] is False
    assert plan["next_collect_requires_later_user_confirmation"] is True
    # 正式运行已完成，OUTPUT_DIR 在位属预期。


def test_read_only_preflight_does_not_call_generator_or_create_output(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(runner, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        runner.stage6e_runner, "_repo_root", lambda: tmp_path
    )
    monkeypatch.setattr(
        protocol, "assert_frozen_protocol_identity", lambda _root: "a" * 64
    )
    monkeypatch.setattr(
        runner.stage6d_runner, "_assert_clean_worktree", lambda _root: "b" * 40
    )
    monkeypatch.setattr(
        runner.stage6d_runner,
        "_generation_input_audit",
        lambda _root: {"inputs": {}},
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
        lambda _shard: {"physical_index": 1},
    )
    monkeypatch.setattr(
        runner.stage6e_runner,
        "_runtime_executable_audit",
        lambda: {"all_executable": True},
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
    assert result["screen_generation_authorized"] is False
    assert result["requires_explicit_later_user_confirmation"] is True
    assert not (tmp_path / protocol.OUTPUT_DIR).exists()


def test_plan_clis_emit_read_only_plans():
    # 收束线：CLI 不经 monkeypatch。run 的 plan 触发源码身份守卫；
    # evaluate/audit 的 plan 在正式产物在位时按"不覆盖"合同拒绝。
    # 三者都应失败关闭且不产生任何输出改动。
    environment = {
        "PYTHONPATH": f"{REPOSITORY_ROOT}:{REPOSITORY_ROOT / 'src'}"
    }
    modules = (
        "scripts.run_issue53_gap_weight_r8_screen",
        "scripts.evaluate_issue53_gap_weight_r8_screen",
        "scripts.audit_issue53_gap_weight_r8_screen",
    )
    expected_markers = ("漂移", "已存在", "已经存在")
    for module in modules:
        result = subprocess.run(
            [sys.executable, "-m", module, "plan"],
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, module
        assert any(marker in result.stderr for marker in expected_markers), (
            module,
            result.stderr[-500:],
        )
