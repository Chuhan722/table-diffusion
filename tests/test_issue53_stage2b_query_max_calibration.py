"""Issue #53 Stage 2B 最坏查询护栏开发候选契约测试。"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from scripts import calibrate_issue53_stage2b_query_max_detector as calibration


def _calibration_frame():
    rows = []
    for dataset, kernel, scale in (
        ("d1", "k1", 1.0),
        ("d1", "k2", 2.0),
        ("d2", "k1", 3.0),
        ("d2", "k2", 4.0),
    ):
        for index in range(9):
            rows.append({
                "dataset": dataset,
                "kernel": kernel,
                "query_max_shift": scale + index,
            })
    return pd.DataFrame(rows)


def _raw_checks(stable_flags):
    rows = []
    for index, stable in enumerate(stable_flags, start=1):
        row = {
            "round_index": (index + 2) * 400,
            "query_max_shift": 0.0 if stable else 100.0,
        }
        for metric in calibration.base_calibration.STABILITY_CONFIG_FIELDS:
            row[metric] = 0.0
        for metric in calibration.base_calibration.MOVEMENT_CONFIG_FIELDS:
            row[metric] = 1.0
        rows.append(row)
    return rows


def test_plan_is_development_only_and_preserves_old_detector():
    plan = calibration.build_plan(Path("input"), Path("output"))

    assert plan["window_size"] == 400
    assert plan["calibration_round_range"] == [6001, 8000]
    assert plan["calibration_terminal_rounds"] == [7200, 7600, 8000]
    assert plan["stability_quantile"] == 0.95
    assert plan["common_rule"] == "maximum_of_four_cell_p95"
    assert plan["threshold_override_parameters_present"] is False
    assert plan["old_frozen_detector_modified"] is False
    assert plan["generator_rerun"] is False
    assert plan["validation_seeds_may_be_read"] is False
    assert set(inspect.signature(calibration.generate_report).parameters) == {
        "input_dir",
        "output_dir",
    }


def test_candidate_uses_cell_p95_max_and_exact_old_base_config():
    config, derivation = calibration.derive_candidate_config(
        _calibration_frame()
    )

    assert config.query_max_shift_tolerance == pytest.approx(11.6)
    assert config.base_config.to_dict() == (
        calibration.old_protocol.FROZEN_DETECTOR_CONFIG.to_dict()
    )
    assert derivation["common_rule"] == "maximum_of_four_cell_p95"
    assert derivation["manual_margin_or_rounding"] is False


def test_candidate_rejects_incomplete_cell_evidence():
    with pytest.raises(ValueError, match="四格"):
        calibration.derive_candidate_config(_calibration_frame().iloc[:-1])


def test_one_bad_block_can_cause_three_failures_without_redrift():
    config, _ = calibration.derive_candidate_config(_calibration_frame())

    _, audit = calibration.annotate_full_checks(
        _raw_checks([True, False, False, False, True]),
        config,
        candidate_round_index=1200,
    )

    assert audit["maximum_post_candidate_unstable_streak"] == 3
    assert audit["persistent_redrift_detected"] is False


def test_four_consecutive_failures_trigger_persistent_redrift():
    config, _ = calibration.derive_candidate_config(_calibration_frame())

    _, audit = calibration.annotate_full_checks(
        _raw_checks([True, False, False, False, False]),
        config,
        candidate_round_index=1200,
    )

    assert audit["maximum_post_candidate_unstable_streak"] == 4
    assert audit["persistent_redrift_detected"] is True


def test_formal_report_rejects_dirty_tree_before_reading_inputs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        calibration.range_analyzer,
        "analysis_environment_manifest",
        lambda: {"git_worktree_clean_including_untracked": False},
    )
    touched = False

    def forbidden_build(*_args):
        nonlocal touched
        touched = True
        raise AssertionError("dirty-tree guard must run first")

    monkeypatch.setattr(calibration, "build_report", forbidden_build)

    with pytest.raises(RuntimeError, match="干净工作树"):
        calibration.generate_report(
            tmp_path / "input", tmp_path / "output"
        )
    assert touched is False
