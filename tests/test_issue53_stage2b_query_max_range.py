"""Issue #53 Stage 2B 最坏查询漂移无阈值量程契约测试。"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_issue53_stage2b_query_max_range as analyzer
from table_diffevo.stationarity import collect_stationarity_range_evidence
from tests.test_stationarity import _make_trace


def test_plan_has_no_threshold_classification_or_validation_access():
    plan = analyzer.build_plan(Path("input"), Path("output"))

    assert plan["window_size"] == 400
    assert plan["calibration_round_range"] == [6001, 8000]
    assert plan["terminal_rounds"] == [7200, 7600, 8000]
    assert plan["threshold_parameters_present"] is False
    assert plan["classification_output_present"] is False
    assert plan["detector_config_changed"] is False
    assert plan["generator_rerun"] is False
    assert plan["validation_seeds_may_be_read"] is False
    assert set(inspect.signature(analyzer.generate_report).parameters) == {
        "input_dir",
        "output_dir",
    }


def test_query_max_exposes_sparse_drift_hidden_by_linear_p95(monkeypatch):
    monkeypatch.setattr(analyzer, "WINDOW_SIZE", 2)
    query_vectors = []
    for value in (1.0, 1.0, 3.0, 3.0, 5.0, 5.0):
        vector = np.zeros(21)
        vector[7] = value
        query_vectors.append(vector)
    trace = _make_trace(query_vectors, moving=True, n_records=10)
    raw = collect_stationarity_range_evidence(trace, [2])[0]

    evidence = analyzer.compute_query_max_evidence(
        trace, raw["window_round_ranges"]
    )

    assert raw["query_mean_shift"] == pytest.approx(0.4 / 21)
    assert raw["query_p95_shift"] == 0.0
    assert evidence["query_mean_shift"] == raw["query_mean_shift"]
    assert evidence["query_p95_shift"] == raw["query_p95_shift"]
    assert evidence["query_max_shift"] == pytest.approx(0.4)
    assert evidence["max_query_index"] == 7
    assert evidence["max_shift_window_pair"] == [1, 3]


def test_query_max_uses_all_three_window_pairs(monkeypatch):
    monkeypatch.setattr(analyzer, "WINDOW_SIZE", 2)
    query_vectors = []
    for value in (1.0, 1.0, 5.0, 5.0, 3.0, 3.0):
        vector = np.zeros(3)
        vector[1] = value
        query_vectors.append(vector)
    trace = _make_trace(query_vectors, moving=True, n_records=10)

    evidence = analyzer.compute_query_max_evidence(
        trace, [[1, 2], [3, 4], [5, 6]]
    )

    assert [(row["left_window"], row["right_window"]) for row in (
        evidence["pairwise"]
    )] == [(1, 2), (1, 3), (2, 3)]
    assert evidence["query_max_shift"] == pytest.approx(0.4)
    assert evidence["max_shift_window_pair"] == [1, 2]


def test_formal_report_rejects_dirty_tree_before_reading_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        analyzer.range_analyzer,
        "analysis_environment_manifest",
        lambda: {"git_worktree_clean_including_untracked": False},
    )
    touched = False

    def forbidden_audit(_path):
        nonlocal touched
        touched = True
        raise AssertionError("dirty-tree guard must run first")

    monkeypatch.setattr(
        analyzer.range_analyzer, "audit_formal_inputs", forbidden_audit
    )

    with pytest.raises(RuntimeError, match="干净工作树"):
        analyzer.generate_report(tmp_path / "input", tmp_path / "output")
    assert touched is False
