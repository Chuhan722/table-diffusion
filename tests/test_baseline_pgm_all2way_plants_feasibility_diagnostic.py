"""Tests for the plants all-2way PGM infeasibility receipt protocol."""

import math
from pathlib import Path

import pytest

from scripts import (
    run_baseline_pgm_all2way_plants_feasibility_diagnostic as diag,
)
from scripts import run_baseline_pgm_plants_diagnostic as pgm980


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("收据 plan 不得读取输入或结果")

    monkeypatch.setattr(diag, "_sha256_file", forbidden)
    monkeypatch.setattr(diag, "_load_json_object", forbidden)
    plan = diag.build_plan()
    assert plan["protocol_sha256"] == diag.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    protocol = plan["protocol"]
    assert protocol["dataset"] == "plants"
    assert protocol["n_records"] == 17412
    assert protocol["n_attributes"] == 69
    precheck = protocol["precheck"]
    assert precheck["expected_verdict"] == "infeasible"
    assert precheck["estimation_attempted"] is False
    assert precheck["synthetic_table_produced"] is False
    assert precheck["cap_mb"] == pgm980.FEASIBILITY_CAP_MB
    generation_input = protocol["generation_input"]
    assert generation_input["expected_cell_query_count"] == 9384
    assert generation_input["expected_pair_clique_count"] == 2346
    assert generation_input["expected_cells_per_clique"] == 4
    assert generation_input["input_sha256"]["measured"] == (
        diag.ALL2WAY_SHA256
    )
    assert "contrast_receipts" in protocol
    assert protocol["failure_policy"][
        "precheck_unexpectedly_feasible"
    ] == "fail_closed_abort_receipt_claim_would_be_wrong"
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diag.protocol_sha256() == diag.FROZEN_PROTOCOL_SHA256
    assert diag.assert_frozen_protocol_identity() == (
        diag.FROZEN_PROTOCOL_SHA256
    )


def test_audit_pins_all2way_workload_and_pair_structure():
    audit = diag._audit_inputs(Path("."))
    assert audit["query_count"] == 9384
    assert len(audit["pair_cliques"]) == 2346
    assert all(len(clique) == 2 for clique in audit["pair_cliques"])
    assert audit["input_sha256"]["measured"] == diag.ALL2WAY_SHA256
    assert len(audit["query_identity_sha256"]) == 64


def test_precheck_reports_structural_infeasibility():
    audit = diag._audit_inputs(Path("."))
    precheck = diag._precheck(Path("."), audit)
    assert precheck["feasible"] is False
    assert precheck["max_clique_attribute_count"] == 69
    assert precheck["max_clique_cells"] == 2**69
    assert precheck["model_size_mb"] > precheck["cap_mb"]
    assert precheck["over_cap_factor"] > 1e11
    assert precheck["pair_clique_count"] == 2346
    assert precheck["one_way_clique_count"] == 69
    assert math.isfinite(precheck["model_size_exabytes_observation_only"])


def test_contrast_receipt_pins_old_feasible_report():
    contrast = diag._verify_contrast_receipt(Path("."))
    assert contrast["sha256"] == diag.OLD_PGM_REPORT_SHA256
    assert contrast["old_exam_feasibility"]["feasible"] is True
    assert contrast["old_exam_feasibility"]["model_size_mb"] < 4096.0


def test_unexpectedly_feasible_precheck_aborts_before_writing(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(diag, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        diag, "_audit_inputs", lambda _root: {"query_count": 0}
    )
    monkeypatch.setattr(
        diag, "_precheck", lambda _root, _audit: {"feasible": True}
    )
    with pytest.raises(RuntimeError, match="意外可行"):
        diag.run(diag.FROZEN_PROTOCOL_SHA256)
    assert not (tmp_path / diag.OUTPUT_DIR).exists()


def test_wrong_confirmation_fails_before_output_creation(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(diag, "_repo_root", lambda: tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        diag.run("wrong")


def test_existing_output_is_not_overwritten(monkeypatch, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(diag, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(diag, "OUTPUT_DIR", Path("existing"))
    with pytest.raises(FileExistsError, match="不覆盖"):
        diag.run(diag.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"
