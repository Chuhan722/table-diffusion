"""Tests for the halfspace-pool plants diagnostic protocol."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import (
    run_fitness_only_all2way_pool_plants_diagnostic as all2way,
)
from scripts import (
    run_fitness_only_halfspace_pool_plants_diagnostic as diag,
)
from scripts import (
    run_fitness_only_oneway_pool_plants_diagnostic as oneway,
)
from scripts import (
    run_fitness_only_polish_budget_plants_diagnostic as polish,
)
from table_diffevo.quality import (
    query_fingerprint,
    validate_query_partition,
)


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(diag, "_sha256_file", forbidden)
    monkeypatch.setattr(diag, "_load_json_object", forbidden)
    plan = diag.build_plan()
    assert plan["protocol_sha256"] == diag.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    assert plan["trajectory_count"] == 1
    protocol = plan["protocol"]
    assert protocol["dataset"] == "plants"
    assert protocol["n_records"] == 17412
    assert protocol["n_attributes"] == 69
    assert protocol["arms"] == ["residual"]
    budget = protocol["budget"]
    assert budget["n_rounds"] == 27000
    assert budget["rho_anneal_start_round"] == 4050
    assert budget["rho_anneal_rounds"] == 2700
    assert protocol["early_stopping"]["patience_ticks"] == 6
    generation_input = protocol["generation_input"]
    assert generation_input["expected_halfspace_total"] == 127
    assert generation_input["expected_halfspace_measured"] == 64
    assert generation_input["expected_halfspace_heldout"] == 63
    assert generation_input["expected_tier_counts"] == {
        "rowsum": {"measured": 32, "heldout": 31},
        "general": {"measured": 32, "heldout": 32},
    }
    assert generation_input["expected_pool_count"] == 9586
    assert generation_input["halfspace_exam_sha256"] == (
        diag.HALFSPACE_SHA256
    )
    assert "boundary_note" in protocol
    grouping = protocol["evaluation_grouping"]
    assert grouping["halfspace_measured_group_added"] is True
    assert grouping["halfspace_heldout_group_added"] is True
    assert grouping["halfspace_tier_buckets"] == ["rowsum", "general"]
    reference = protocol["comparison_reference"]
    assert reference["all2way_report_sha256"] == (
        diag.ALL2WAY_REPORT_SHA256
    )
    assert reference["bare_guess_table_sha256"] == (
        diag.BARE_GUESS_TABLE_SHA256
    )
    assert "pgm_absent_by_design" in reference
    observation = protocol["observation_points"]
    assert set(observation) == {
        "H1_optimizability",
        "H2_generalization",
        "H3_no_regression",
        "H4_difficulty_tiers",
    }
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diag.protocol_sha256() == diag.FROZEN_PROTOCOL_SHA256
    assert diag.assert_frozen_protocol_identity() == (
        diag.FROZEN_PROTOCOL_SHA256
    )


def test_budget_constants_imported_from_polish_protocol():
    assert diag.N_ROUNDS is polish.N_ROUNDS
    assert diag.RHO_ANNEAL_START_ROUND is polish.RHO_ANNEAL_START_ROUND
    assert diag.RHO_ANNEAL_ROUNDS is polish.RHO_ANNEAL_ROUNDS
    assert diag.EARLY_STOPPING_PATIENCE_TICKS is (
        polish.EARLY_STOPPING_PATIENCE_TICKS
    )
    assert not hasattr(diag, "_config")
    assert diag.EXPECTED_POOL_COUNT == all2way.EXPECTED_POOL_COUNT + 64
    identity = polish.assert_polish_budget_identity()
    assert identity["minimality_and_rounding"] == "pass"


def test_halfspace_exam_pinned_and_wellformed():
    exam = diag._load_halfspace_exam(Path("."))
    assert exam["halfspace_sha256"] == diag.HALFSPACE_SHA256
    assert len(exam["measured_queries"]) == 64
    assert len(exam["heldout_queries"]) == 63
    fingerprints = {
        query_fingerprint(query)
        for query in exam["measured_queries"] + exam["heldout_queries"]
    }
    assert len(fingerprints) == 127
    for query in exam["measured_queries"] + exam["heldout_queries"]:
        spec = query["halfspace"]
        assert len(spec["attributes"]) == len(spec["weights"])
        if query["tier"] == "rowsum":
            assert len(spec["attributes"]) == 69
            assert all(weight == 1 for weight in spec["weights"])
        else:
            assert len(spec["attributes"]) == 16
            assert all(weight in (-1, 1) for weight in spec["weights"])


def test_halfspace_counts_match_source_data_spot_checks():
    payload = json.loads(
        Path(diag.HALFSPACE_PATH).read_text(encoding="utf-8")
    )
    df = pd.read_csv("data/plants/plants.csv")
    queries = payload["queries"]
    for index in (0, 31, 63, 64, 95, 126):
        query = queries[index]
        spec = query["halfspace"]
        projected = df[spec["attributes"]].to_numpy() @ np.asarray(
            spec["weights"], dtype=float
        )
        recomputed = int(np.sum(projected >= float(spec["theta"])))
        assert recomputed == int(query["result"])


def test_audit_generation_inputs_pool_assembly():
    audit = diag._audit_generation_inputs(Path("."))
    assert audit["query_count"] == 980
    assert audit["all2way_count"] == 9384
    assert audit["one_way_count"] == 138
    assert audit["halfspace_measured_count"] == 64
    assert audit["halfspace_heldout_count"] == 63
    assert audit["pool_query_count"] == 9586
    assert audit["pool_queries"][:9522] == (
        list(audit["all2way_queries"]) + list(audit["one_way_queries"])
    )
    assert audit["pool_queries"][9522:] == (
        audit["halfspace_measured_queries"]
    )
    assert audit["pool_targets"][9522:] == (
        audit["halfspace_measured_targets"]
    )
    for key in (
        "halfspace_measured_identity_sha256",
        "halfspace_heldout_identity_sha256",
        "pool_identity_sha256",
        "pool_target_vector_sha256",
    ):
        assert len(audit[key]) == 64


def test_pool_disjoint_from_all_heldout_groups():
    audit = diag._audit_generation_inputs(Path("."))
    heldout_queries, _, _ = oneway._load_heldout_and_reference(Path("."))
    validate_query_partition(
        list(audit["pool_queries"]),
        list(heldout_queries) + list(audit["halfspace_heldout_queries"]),
    )


def test_halfspace_group_metrics_buckets_by_tier():
    queries = [
        {"tier": "rowsum"},
        {"tier": "rowsum"},
        {"tier": "general"},
    ]
    targets = [100.0, 200.0, 300.0]
    answers = [110.0, 190.0, 340.0]
    metrics = diag._halfspace_group_metrics(queries, targets, answers)
    assert metrics["overall"]["query_count"] == 3
    assert metrics["overall"]["absolute_error_mean"] == pytest.approx(
        20.0
    )
    assert set(metrics["by_tier"]) == {"rowsum", "general"}
    assert metrics["by_tier"]["rowsum"]["query_count"] == 2
    assert metrics["by_tier"]["rowsum"][
        "absolute_error_mean"
    ] == pytest.approx(10.0)
    assert metrics["by_tier"]["general"][
        "absolute_error_max"
    ] == pytest.approx(40.0)
    assert metrics["by_tier"]["general"][
        "normalized_l1_mean"
    ] == pytest.approx(40.0 / 17412)


def _constant_stats(value: float) -> dict:
    return {
        f"normalized_l1_{name}": value
        for name in ("mean", "median", "p90", "max")
    }


def _constant_quality(value: float) -> dict:
    stats = _constant_stats(value)
    return {
        "measured": dict(stats),
        "heldout": {
            "3way": dict(stats),
            "4way": dict(stats),
            "combined": dict(stats),
        },
        "one_way_safety_by_target_bucket": {
            "by_order": {"1way": dict(stats)},
        },
        "all2way_pool": dict(stats),
    }


def _constant_halfspace_group(
    rowsum: float, general: float
) -> dict:
    overall = _constant_stats((rowsum + general) / 2.0)
    return {
        "overall": overall,
        "by_tier": {
            "rowsum": _constant_stats(rowsum),
            "general": _constant_stats(general),
        },
    }


def test_comparison_reports_h_blocks():
    quality = _constant_quality(2.0)
    quality["halfspace_measured"] = _constant_halfspace_group(1.0, 3.0)
    quality["halfspace_heldout"] = _constant_halfspace_group(2.0, 6.0)
    quality["halfspace_bare_guess_baseline"] = {
        "measured": _constant_halfspace_group(10.0, 30.0),
        "heldout": _constant_halfspace_group(8.0, 24.0),
    }
    references = {
        "all2way_residual": oneway._metric_snapshot(
            _constant_quality(3.0)
        ),
        "all2way_pool_anchor": {
            "mean": 3.0,
            "median": 3.0,
            "max": 3.0,
        },
    }
    result = diag._comparison(quality, references)
    assert set(result["new_residual_snapshot"]) == {
        "measured",
        "heldout_3way",
        "heldout_4way",
        "heldout_combined",
        "one_way_safety",
    }
    h1 = result["H1_optimizability_measured"]
    assert h1["new"]["overall_normalized_l1_mean"] == pytest.approx(2.0)
    assert h1["bare_guess_floor"][
        "overall_normalized_l1_mean"
    ] == pytest.approx(20.0)
    assert h1[
        "improvement_ratio_bare_over_new_observation_only"
    ] == pytest.approx(10.0)
    h2 = result["H2_generalization_heldout"]
    assert h2[
        "improvement_ratio_bare_over_new_observation_only"
    ] == pytest.approx(4.0)
    for entry in result["H3_no_regression_vs_all2way_report"].values():
        assert entry["delta_mean"] == pytest.approx(-1.0)
    drift = result["H3_all2way_pool_drift"]
    assert drift["new"]["mean"] == pytest.approx(2.0)
    assert drift["reference"]["mean"] == pytest.approx(3.0)
    tiers = result["H4_difficulty_tiers"]
    assert tiers["measured_by_tier"] == {
        "rowsum": pytest.approx(1.0),
        "general": pytest.approx(3.0),
    }
    assert tiers["bare_guess_heldout_by_tier"] == {
        "rowsum": pytest.approx(8.0),
        "general": pytest.approx(24.0),
    }


def test_reference_extraction_pins_and_shapes():
    try:
        references = diag._load_reference_reports(Path("."))
    except FileNotFoundError:
        pytest.skip("冻结产物不在本机（gitignored），产物持有机复核")
    assert references["reference_sha256"] == {
        "all2way_report": diag.ALL2WAY_REPORT_SHA256,
    }
    snapshot = references["all2way_residual"]
    for stats in snapshot.values():
        assert all(
            math.isfinite(value) and value >= 0
            for value in stats.values()
        )
    assert snapshot["heldout_3way"]["mean"] == pytest.approx(
        0.005712, abs=1e-5
    )
    assert snapshot["one_way_safety"]["mean"] == pytest.approx(
        0.000257, abs=1e-5
    )
    assert references["all2way_pool_anchor"]["mean"] == pytest.approx(
        0.001292, abs=1e-5
    )


def test_bare_guess_table_pin_matches_all2way_report():
    report_path = Path(diag.ALL2WAY_REPORT_PATH)
    table_path = Path(diag.BARE_GUESS_TABLE_PATH)
    if not report_path.exists() or not table_path.exists():
        pytest.skip("冻结产物不在本机（gitignored），产物持有机复核")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["generation"]["arms"]["residual"][
        "terminal_table_sha256"
    ] == diag.BARE_GUESS_TABLE_SHA256
    assert diag._sha256_file(table_path) == (
        diag.BARE_GUESS_TABLE_SHA256
    )


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
