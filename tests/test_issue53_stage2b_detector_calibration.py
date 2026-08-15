"""Issue #53 Stage 2B 通用 detector 开发校准报告契约测试。"""

import inspect
from pathlib import Path

import pandas as pd
import pytest

from scripts import calibrate_issue53_stage2b_detector as calibration


def _calibration_frame():
    rows = []
    cells = (
        ("d1", "k1", 1.0),
        ("d1", "k2", 2.0),
        ("d2", "k1", 3.0),
        ("d2", "k2", 4.0),
    )
    for dataset, kernel, scale in cells:
        for index in range(9):
            row = {"dataset": dataset, "kernel": kernel}
            for metric in calibration.STABILITY_CONFIG_FIELDS:
                row[metric] = scale + index
            for metric in calibration.MOVEMENT_CONFIG_FIELDS:
                row[metric] = (scale + index) / 100.0
            rows.append(row)
    return pd.DataFrame(rows)


def _raw_checks(stable_flags):
    rows = []
    for index, stable in enumerate(stable_flags, start=1):
        row = {
            "window_size": calibration.WINDOW_SIZE,
            "completed_block_count": index + 2,
            "state_index": (index + 2) * calibration.WINDOW_SIZE,
            "round_index": (index + 2) * calibration.WINDOW_SIZE,
            "window_round_ranges": [[1, 2], [3, 4], [5, 6]],
        }
        for metric in calibration.STABILITY_CONFIG_FIELDS:
            row[metric] = 0.0 if stable else 100.0
        for metric in calibration.MOVEMENT_CONFIG_FIELDS:
            row[metric] = 1.0
        rows.append(row)
    return rows


def test_plan_freezes_confirmed_rules_without_threshold_override():
    plan = calibration.build_plan(Path("input"), Path("output"))

    assert plan["window_size"] == 400
    assert plan["calibration_round_range"] == [6001, 8000]
    assert plan["calibration_terminal_rounds"] == [7200, 7600, 8000]
    assert plan["stability_quantile"] == 0.95
    assert plan["movement_quantile"] == 0.05
    assert plan["stall_patience_checks"] == 4
    assert plan["persistent_redrift_checks"] == 4
    assert plan["threshold_override_parameters_present"] is False
    assert plan["validation_seeds_may_be_read"] is False
    assert set(inspect.signature(calibration.generate_report).parameters) == {
        "input_dir",
        "output_dir",
    }


def test_candidate_config_uses_cell_p95_max_and_cell_p05_min():
    frame = _calibration_frame()

    config, derivation = calibration.derive_candidate_config(frame)

    # Linear P95 of 4..12 is 11.6 and is the largest cell envelope.
    assert config.query_mean_shift_tolerance == pytest.approx(11.6)
    # Linear P05 of 0.01..0.09 is 0.014 and is the smallest cell floor.
    assert config.minimum_active_round_rate == pytest.approx(0.014)
    assert config.window_size == 400
    assert config.stall_patience_checks == 4
    assert derivation["stability"]["query_mean_shift"][
        "common_rule"
    ] == "maximum_of_four_cell_p95"
    assert derivation["movement"][
        "minimum_observed_active_round_rate"
    ]["common_rule"] == "minimum_of_four_cell_p05"


def test_candidate_config_rejects_incomplete_cell_evidence():
    with pytest.raises(ValueError, match="四格"):
        calibration.derive_candidate_config(
            _calibration_frame().iloc[:-1]
        )


def test_one_bad_block_can_cause_three_failures_without_redrift():
    config, _ = calibration.derive_candidate_config(_calibration_frame())
    checks = _raw_checks([True, False, False, False, True])

    _, audit = calibration.annotate_full_checks(
        checks, config, candidate_round_index=1200
    )

    assert audit["maximum_post_candidate_unstable_streak"] == 3
    assert audit["persistent_redrift_detected"] is False


def test_four_consecutive_failures_trigger_persistent_redrift():
    config, _ = calibration.derive_candidate_config(_calibration_frame())
    checks = _raw_checks([True, False, False, False, False])

    _, audit = calibration.annotate_full_checks(
        checks, config, candidate_round_index=1200
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
