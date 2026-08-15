"""Issue #53 Stage 2B 无阈值量程报告契约测试。"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import analyze_issue53_stage2b_range_finding as analyzer
from table_diffevo.stationarity import save_stationarity_trace
from tests.test_stationarity import _make_trace


def test_plan_freezes_range_grid_without_threshold_or_validation_access():
    plan = analyzer.build_plan(
        Path("input"), Path("output")
    )

    assert plan["candidate_window_sizes"] == [100, 200, 400, 800, 1000]
    assert plan["expected_trajectory_count"] == 12
    assert plan["threshold_parameters_present"] is False
    assert plan["classification_output_present"] is False
    assert plan["validation_seeds_may_be_read"] is False
    assert set(inspect.signature(analyzer.generate_report).parameters) == {
        "input_dir",
        "output_dir",
    }


def test_range_frames_are_threshold_free_and_keep_block_alignment(
    tmp_path, monkeypatch
):
    trace = _make_trace([[2.0, 1.0]] * 10, moving=True)
    run_dir = tmp_path / "test_300x10" / "seed_999" / "independent"
    save_stationarity_trace(trace, run_dir / "trace")
    item = analyzer.AuditedTraceInput(
        dataset="test_300x10",
        kernel="independent",
        seed=999,
        run_dir=run_dir,
        manifest={},
        manifest_sha256="0" * 64,
    )
    monkeypatch.setattr(analyzer, "CANDIDATE_WINDOW_SIZES", (2, 3))
    monkeypatch.setattr(
        analyzer.collector,
        "DEVELOPMENT_ROUND_BUDGET",
        10,
    )
    monkeypatch.setattr(
        analyzer,
        "ROUND_BAND_LABELS",
        ("q1", "q2", "q3", "q4"),
    )

    range_frame, state_frame = analyzer.build_range_frames([item])

    assert list(range_frame["window_size"]) == [2, 2, 2, 3]
    assert list(range_frame["round_index"]) == [6, 8, 10, 9]
    assert len(state_frame) == 11
    assert analyzer.FORBIDDEN_CLASSIFICATION_FIELDS.isdisjoint(
        range_frame.columns
    )
    assert range_frame.loc[0, "window_1_start_round"] == 1
    assert range_frame.loc[0, "window_3_end_round"] == 6

    summary = analyzer.build_descriptive_summary(
        range_frame,
        state_frame,
        {"sealed_validation_seeds_read": False},
    )
    assert summary["role"] == {
        "purpose": "descriptive_range_finding_only",
        "threshold_parameters": "absent",
        "stationarity_or_stall_classification": "absent",
        "candidate_stop_round": "absent",
        "generator_rerun": False,
        "validation_seed_access": False,
    }
    assert summary["source_audit"][
        "sealed_validation_seeds_read"
    ] is False


def test_flatten_check_fails_closed_on_classification_field():
    item = SimpleNamespace(
        dataset="test_300x10", kernel="independent", seed=999
    )
    check = {"stable": True}

    with pytest.raises(RuntimeError, match="分类字段"):
        analyzer._flatten_check(item, check)


def test_formal_report_rejects_dirty_tree_before_reading_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        analyzer,
        "analysis_environment_manifest",
        lambda: {"git_worktree_clean_including_untracked": False},
    )
    touched = False

    def forbidden_audit(_path):
        nonlocal touched
        touched = True
        raise AssertionError("dirty-tree guard must run first")

    monkeypatch.setattr(analyzer, "audit_formal_inputs", forbidden_audit)

    with pytest.raises(RuntimeError, match="干净工作树"):
        analyzer.generate_report(tmp_path / "input", tmp_path / "output")
    assert touched is False
