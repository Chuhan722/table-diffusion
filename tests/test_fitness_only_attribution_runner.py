"""Tests for the result-blind fitness-only attribution runner."""

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_fitness_only_attribution as runner


def test_plan_is_frozen_and_does_not_read_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(runner, "_sha256_file", forbidden)
    monkeypatch.setattr(runner, "_load_json_object", forbidden)
    plan = runner.build_plan()

    assert plan["protocol_sha256"] == runner.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 4
    assert plan["generation_started"] is False
    assert plan["protocol"]["phase_boundary"][
        "heldout_and_reference_loaded_only_after_all_generation"
    ] is True
    assert plan["protocol"]["pairing"]["only_treatment_difference"] == (
        "row_fitness"
    )


def test_protocol_identity_is_fail_closed():
    assert runner.protocol_sha256() == runner.FROZEN_PROTOCOL_SHA256
    assert runner.assert_frozen_protocol_identity() == (
        runner.FROZEN_PROTOCOL_SHA256
    )


def test_wrong_confirmation_fails_before_any_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_repo_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        runner,
        "_source_snapshot",
        lambda *_args, **_kwargs: called.append("source"),
    )

    with pytest.raises(ValueError, match="protocol SHA-256"):
        runner.run("not-the-frozen-hash")
    assert called == []


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "already-there"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(runner, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(runner, "OUTPUT_DIR", Path("already-there"))

    with pytest.raises(FileExistsError, match="不覆盖"):
        runner.run(runner.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_grouped_metrics_keep_zero_target_as_own_bucket():
    queries = [
        {"conditions": [{"attribute": "a", "operator": "==", "value": 0}]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 0},
            {"attribute": "b", "operator": "==", "value": 1},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": 1},
            {"attribute": "b", "operator": "==", "value": 1},
            {"attribute": "c", "operator": "==", "value": 1},
        ]},
    ]
    target = np.asarray([0.0, 8.0, 100.0])
    current = np.asarray([3.0, 4.0, 90.0])
    result = runner._grouped_error_metrics(queries, target, current, 100)

    assert set(result["by_order"]) == {"1way", "2way", "3way"}
    assert result["by_target_bucket"]["target_0"]["absolute_error_mean"] == 3.0
    assert result["by_target_bucket"]["target_1_8"]["absolute_error_mean"] == 4.0
    assert result["by_target_bucket"]["target_51_500"]["absolute_error_mean"] == 10.0


def test_jsonable_diagnostics_is_strict():
    value = runner._jsonable({
        "positive": float("inf"),
        "array": np.asarray([1, 2], dtype=np.int64),
    })
    assert value == {
        "positive": "positive_infinity",
        "array": [1, 2],
    }
    json.dumps(value, allow_nan=False)


def test_target_bucket_boundaries():
    assert runner._target_bucket(0) == "target_0"
    assert runner._target_bucket(8) == "target_1_8"
    assert runner._target_bucket(9) == "target_9_50"
    assert runner._target_bucket(50) == "target_9_50"
    assert runner._target_bucket(51) == "target_51_500"
    assert runner._target_bucket(500) == "target_51_500"
    assert runner._target_bucket(501) == "target_501_2000"
    assert runner._target_bucket(2001) == "target_over_2000"
