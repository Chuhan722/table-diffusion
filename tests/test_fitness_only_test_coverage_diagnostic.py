"""Tests for the isolated workload-coverage diagnostic protocol."""

import json
from pathlib import Path

import pytest

from scripts import run_fitness_only_test_coverage_diagnostic as diagnostic


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("诊断 plan 不得读取输入或结果")

    monkeypatch.setattr(diagnostic, "_sha256_file", forbidden)
    monkeypatch.setattr(diagnostic, "_load_json_object", forbidden)
    plan = diagnostic.build_plan()
    assert plan["protocol_sha256"] == diagnostic.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 2
    assert plan["protocol"]["generation_input"]["combined_query_count"] == 75
    assert plan["protocol"]["interpretation"]["diagnostic_only"] is True


def test_protocol_identity_is_frozen():
    assert diagnostic.protocol_sha256() == diagnostic.FROZEN_PROTOCOL_SHA256
    assert diagnostic.assert_frozen_protocol_identity() == (
        diagnostic.FROZEN_PROTOCOL_SHA256
    )


def test_generation_input_audit_builds_exact_b_plus_one_way_partition():
    audit = diagnostic._audit_generation_inputs(Path("."))
    assert audit["workload_b_count"] == 50
    assert audit["one_way_count"] == 25
    assert audit["combined_count"] == 75
    assert len(audit["queries"]) == len(audit["targets"]) == 75
    assert len({
        diagnostic.query_fingerprint(query) for query in audit["queries"]
    }) == 75


def test_wrong_confirmation_fails_before_output_creation(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostic, "_repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="protocol SHA-256"):
        diagnostic.run("wrong")


def test_existing_output_is_not_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diagnostic, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(diagnostic, "OUTPUT_DIR", Path("existing"))
    with pytest.raises(FileExistsError, match="不覆盖"):
        diagnostic.run(diagnostic.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_comparison_reports_residual_minus_equal():
    values = {
        "residual": {
            "measured": {"normalized_l1_mean": 1.0},
            "workload_b_measured": {"normalized_l1_mean": 2.0},
            "one_way_safety": {"normalized_l1_mean": 3.0},
            "heldout": {
                "3way": {"normalized_l1_mean": 4.0},
                "4way": {"normalized_l1_mean": 5.0},
            },
        },
        "equal": {
            "measured": {"normalized_l1_mean": 1.5},
            "workload_b_measured": {"normalized_l1_mean": 1.5},
            "one_way_safety": {"normalized_l1_mean": 2.0},
            "heldout": {
                "3way": {"normalized_l1_mean": 3.0},
                "4way": {"normalized_l1_mean": 6.0},
            },
        },
    }
    result = diagnostic._comparison(values)
    deltas = result["delta_residual_minus_equal"]
    assert deltas["combined_measured"]["residual_minus_equal"] == -0.5
    assert deltas["one_way_safety"]["residual_minus_equal"] == 1.0
    assert deltas["heldout_4way"]["residual_minus_equal"] == -1.0
