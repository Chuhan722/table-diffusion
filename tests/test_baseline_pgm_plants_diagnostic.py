"""Tests for the Private-PGM plants baseline protocol."""

import math
from pathlib import Path

import numpy as np
import pytest

from scripts import run_baseline_pgm_plants_diagnostic as diagnostic


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("基线 plan 不得读取输入或结果")

    monkeypatch.setattr(diagnostic, "_sha256_file", forbidden)
    monkeypatch.setattr(diagnostic, "_load_json_object", forbidden)
    plan = diagnostic.build_plan()
    assert plan["protocol_sha256"] == diagnostic.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    protocol = plan["protocol"]
    assert protocol["dataset"] == "plants"
    assert protocol["n_records"] == 17412
    baseline = protocol["baseline"]
    assert baseline["iters"] == 1000
    assert baseline["tuning_allowed"] is False
    assert baseline["feasibility_precheck"]["cap_mb"] == 4096.0
    assert protocol["comparison_reference"]["rerun_our_arms"] is False
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diagnostic.protocol_sha256() == diagnostic.FROZEN_PROTOCOL_SHA256
    assert diagnostic.assert_frozen_protocol_identity() == (
        diagnostic.FROZEN_PROTOCOL_SHA256
    )


def test_generation_input_audit_pins_measured_workload():
    audit = diagnostic._audit_generation_inputs(Path("."))
    assert audit["query_count"] == 980
    assert all(
        len(query["conditions"]) >= 2 for query in audit["queries"]
    )
    assert audit["input_sha256"] == {
        key: diagnostic.INPUT_SHA256[key]
        for key in ("schema", "marginals", "measured")
    }
    assert all(
        math.isfinite(target) and target >= 0
        for target in audit["targets"]
    )


def test_cell_measurements_group_cliques_and_match_targets():
    root = Path(".")
    audit = diagnostic._audit_generation_inputs(root)
    _, names, cards = diagnostic._schema_domain(root)
    measurements, measurement_audit = diagnostic._build_cell_measurements(
        root, audit, names, cards
    )
    assert len(measurements) == 375
    assert sum(item["cell_count"] for item in measurement_audit) == 980

    order_index = {name: position for position, name in enumerate(names)}
    lookup = {
        tuple(measurement.clique): measurement
        for measurement in measurements
    }
    for query, target in zip(audit["queries"], audit["targets"]):
        conditions = sorted(
            query["conditions"],
            key=lambda c: order_index[c["attribute"]],
        )
        clique = tuple(c["attribute"] for c in conditions)
        measurement = lookup[clique]
        dims = tuple(cards[a] for a in clique)
        flat = int(np.ravel_multi_index(
            tuple(int(c["value"]) for c in conditions), dims
        ))
        position = measurement.query.indices.index(flat)
        assert float(measurement.noisy_measurement[position]) == target


def test_one_way_measurements_cover_all_attributes():
    root = Path(".")
    _, names, cards = diagnostic._schema_domain(root)
    measurements, measurement_audit, safety = (
        diagnostic._build_one_way_measurements(root, names, cards)
    )
    assert len(measurements) == 69
    assert sum(item["cell_count"] for item in measurement_audit) == 138
    assert len(safety["queries"]) == 138
    for measurement in measurements:
        total = float(np.sum(np.asarray(measurement.noisy_measurement)))
        assert total == pytest.approx(17412.0)


def test_feasibility_precheck_reports_junction_tree_size():
    root = Path(".")
    audit = diagnostic._audit_generation_inputs(root)
    domain, names, cards = diagnostic._schema_domain(root)
    cell_measurements, _ = diagnostic._build_cell_measurements(
        root, audit, names, cards
    )
    one_way_measurements, _, _ = diagnostic._build_one_way_measurements(
        root, names, cards
    )
    feasibility = diagnostic._feasibility_precheck(
        domain, cell_measurements + one_way_measurements
    )
    assert feasibility["feasible"] is True
    assert feasibility["model_size_mb"] <= feasibility["cap_mb"]
    assert feasibility["max_clique_cells"] >= 1
    assert feasibility["maximal_clique_count"] >= 1


def test_wrong_confirmation_fails_before_output_creation(
    monkeypatch, tmp_path
):
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


def _constant_arm_quality(value: float) -> dict:
    stats = {
        f"normalized_l1_{name}": value
        for name in ("mean", "median", "p90", "max")
    }
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
    }


def test_comparison_reports_pgm_minus_each_arm():
    pgm_quality = _constant_arm_quality(2.0)
    reference = {
        "arms": {
            "residual": diagnostic._arm_snapshot(_constant_arm_quality(0.5)),
            "equal": diagnostic._arm_snapshot(_constant_arm_quality(3.0)),
        },
    }
    result = diagnostic._comparison(pgm_quality, reference)
    deltas = result["normalized_l1_mean_deltas"]
    assert set(deltas) == {
        "measured",
        "heldout_3way",
        "heldout_4way",
        "heldout_combined",
        "one_way_safety",
    }
    for entry in deltas.values():
        assert entry["pgm_minus_residual"] == pytest.approx(1.5)
        assert entry["pgm_minus_equal"] == pytest.approx(-1.0)


def test_reference_report_extraction_pins_and_shapes():
    reference = diagnostic._load_reference_report(Path("."))
    assert reference["sha256"] == diagnostic.INPUT_SHA256["reference_report"]
    assert set(reference["arms"]) == {"residual", "equal"}
    for arm_snapshot in reference["arms"].values():
        assert set(arm_snapshot) == {
            "measured",
            "heldout_3way",
            "heldout_4way",
            "heldout_combined",
            "one_way_safety",
        }
        for stats in arm_snapshot.values():
            assert all(
                math.isfinite(value) and value >= 0
                for value in stats.values()
            )
