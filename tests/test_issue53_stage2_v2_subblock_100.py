"""Issue #53 V2 单一 100 轮小块假设审查契约测试。"""

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_issue53_stage2_v2_subblock_100 as analyzer
from table_diffevo.stationarity_v2 import (
    compute_v2_candidate_evidence,
    compute_v2_scalar_evidence,
)
from tests.test_stationarity_v2 import _make_candidate_collection


def test_plan_contains_one_hypothesis_without_selection_or_decision() -> None:
    plan = analyzer.build_plan(Path("input"), Path("output"))

    assert plan["mode"] == "plan_only_no_trace_read"
    assert plan["tested_subblock_round_counts"] == [100]
    assert plan["alternative_length_selection"] is False
    assert plan["dataset_specific_window_rule"] is False
    assert plan["threshold_parameters_present"] is False
    assert plan["classification_output_present"] is False
    assert plan["validation_seeds_may_be_read"] is False
    assert set(inspect.signature(analyzer.generate_report).parameters) == {
        "input_dir",
        "output_dir",
    }


def test_candidate_schedule_is_fixed_and_block_aligned() -> None:
    starts = analyzer.candidate_first_subblock_numbers(80)

    assert starts == tuple(range(1, 70, 4))
    assert len(starts) == 18
    assert [100 * (start + 11) for start in starts] == list(
        range(1200, 8001, 400)
    )
    assert analyzer.candidate_first_subblock_numbers(11) == ()
    with pytest.raises(ValueError, match="integer"):
        analyzer.candidate_first_subblock_numbers(True)


def test_residual_lag_one_correlation_has_explicit_undefined_rule() -> None:
    constant = compute_v2_scalar_evidence([0.2] * 12)
    alternating = compute_v2_scalar_evidence([0.0, 1.0] * 6)

    assert analyzer.residual_lag_one_correlation(constant) is None
    assert analyzer.residual_lag_one_correlation(alternating) == pytest.approx(
        -1.0
    )


def test_candidate_row_keeps_zero_scale_and_dependence_explicit() -> None:
    values = [
        [0.2, 0.4 if index % 2 == 0 else 0.6]
        for index in range(12)
    ]
    collection = _make_candidate_collection(values)
    evidence = compute_v2_candidate_evidence(
        collection, first_subblock_number=1
    )
    item = SimpleNamespace(
        dataset="test_300x10", kernel="independent", seed=999
    )

    row = analyzer.flatten_candidate_evidence(item, evidence)

    assert row["subblock_round_count"] == 100
    assert row["start_round_index"] == 1
    assert row["end_round_index"] == 1200
    assert row["query_count"] == 2
    assert row["query_zero_scale_count"] >= 1
    assert row["query_residual_lag1_undefined_count"] >= 1
    assert row["query_residual_lag1_absolute_maximum"] == pytest.approx(1.0)
    assert analyzer.FORBIDDEN_DECISION_FIELDS.isdisjoint(row)
    json_bytes = analyzer._strict_json_bytes(row)
    assert b"Infinity" not in json_bytes
    assert b"NaN" not in json_bytes


def test_cell_summary_is_descriptive_and_has_no_hidden_verdict() -> None:
    template = {
        metric: 0.5 for metric in analyzer.SUMMARY_METRICS
    }
    rows = []
    for dataset in ("nltcs", "test_300x10"):
        for kernel in ("factorized_gibbs", "independent"):
            for index in range(54):
                rows.append({
                    "dataset": dataset,
                    "kernel": kernel,
                    "seed": 200 + index % 3,
                    "query_count": 1001 if dataset == "nltcs" else 50,
                    **template,
                })

    summary = analyzer.build_descriptive_summary(
        rows, {"sealed_validation_seeds_read": False}
    )

    assert len(summary["cell_summaries"]) == 4
    assert summary["role"]["tested_subblock_round_counts"] == [100]
    assert summary["role"]["threshold_parameters"] == "absent"
    assert summary["role"]["convergence_or_stall_classification"] == (
        "absent"
    )
    assert summary["query_count_boundary"][
        "query_count_maximum_correction"
    ] == "not_yet_defined"
    assert np.isfinite(
        summary["cell_summaries"][0]["metrics"][
            analyzer.SUMMARY_METRICS[0]
        ]["median"]
    )


def test_formal_report_rejects_dirty_tree_before_reading_inputs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        analyzer.range_analyzer,
        "analysis_environment_manifest",
        lambda: {"git_worktree_clean_including_untracked": False},
    )
    touched = False

    def forbidden_audit(_path):
        nonlocal touched
        touched = True
        raise AssertionError("dirty-tree guard must run before input audit")

    monkeypatch.setattr(
        analyzer.range_analyzer, "audit_formal_inputs", forbidden_audit
    )

    with pytest.raises(RuntimeError, match="干净工作树"):
        analyzer.generate_report(tmp_path / "input", tmp_path / "output")
    assert touched is False
