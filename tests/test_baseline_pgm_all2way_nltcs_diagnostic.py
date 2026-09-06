"""Tests for the Private-PGM all-2way nltcs refit baseline protocol."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_baseline_pgm_all2way_nltcs_diagnostic as diag
from scripts import run_baseline_pgm_nltcs_diagnostic as pgm980
from table_diffevo.quality import query_fingerprint


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("基线 plan 不得读取输入或结果")

    monkeypatch.setattr(diag, "_sha256_file", forbidden)
    monkeypatch.setattr(diag, "_load_json_object", forbidden)
    plan = diag.build_plan()
    assert plan["protocol_sha256"] == diag.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    protocol = plan["protocol"]
    assert protocol["dataset"] == "nltcs"
    assert protocol["n_records"] == 16181
    baseline = protocol["baseline"]
    assert baseline["estimator"] == "estimation.MirrorDescent"
    assert baseline["iters"] == 1000
    assert baseline["tuning_allowed"] is False
    generation_input = protocol["generation_input"]
    assert generation_input["expected_cell_query_count"] == 480
    assert generation_input["expected_clique_count"] == 120
    assert generation_input["expected_cells_per_clique"] == 4
    assert generation_input["input_sha256"]["measured"] == (
        diag.ALL2WAY_SHA256
    )
    assert "feasibility_precheck" in protocol
    reference = protocol["comparison_reference"]
    assert reference["comparable_groups"] == list(diag.COMPARABLE_GROUPS)
    assert "measured_sets_note" in reference
    assert reference["rerun_referenced_arms"] is False
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diag.protocol_sha256() == diag.FROZEN_PROTOCOL_SHA256
    assert diag.assert_frozen_protocol_identity() == (
        diag.FROZEN_PROTOCOL_SHA256
    )


def test_baseline_constants_imported_from_pgm980_protocol():
    assert diag.MD_ITERS is pgm980.MD_ITERS
    assert diag.MEASUREMENT_STDDEV is pgm980.MEASUREMENT_STDDEV
    assert diag.SAMPLING_METHOD is pgm980.SAMPLING_METHOD
    assert diag.MBI_SOURCE_COMMIT is pgm980.MBI_SOURCE_COMMIT
    assert diag.SCHEMA_PATH is pgm980.SCHEMA_PATH
    assert diag.MARGINALS_PATH is pgm980.MARGINALS_PATH
    assert diag.HELDOUT_PATH is pgm980.HELDOUT_PATH
    assert diag.REFERENCE_PATH is pgm980.REFERENCE_PATH


def test_all2way_workload_file_pinned_and_wellformed():
    path = Path(diag.ALL2WAY_PATH)
    assert diag._sha256_file(path) == diag.ALL2WAY_SHA256
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 16181
    assert payload["query_count"] == 480
    queries = payload["queries"]
    assert len(queries) == 480
    assert all(q["type"] == "double" for q in queries)
    fingerprints = [query_fingerprint(q) for q in queries]
    assert len(set(fingerprints)) == len(fingerprints)
    group_sums: dict[str, int] = {}
    group_cells: dict[str, int] = {}
    for query in queries:
        group_sums[query["group"]] = (
            group_sums.get(query["group"], 0) + int(query["result"])
        )
        group_cells[query["group"]] = group_cells.get(query["group"], 0) + 1
    assert len(group_sums) == 120
    assert all(count == 4 for count in group_cells.values())
    assert all(total == 16181 for total in group_sums.values())


def test_all2way_counts_match_source_data_spot_pairs():
    payload = json.loads(
        Path(diag.ALL2WAY_PATH).read_text(encoding="utf-8")
    )
    by_fingerprint = {
        query_fingerprint(q): int(q["result"])
        for q in payload["queries"]
    }
    df = pd.read_csv("data/nltcs/nltcs.csv")
    columns = list(df.columns)
    values = df.to_numpy()
    spot_pairs = [(0, 1), (0, 15), (7, 8), (14, 15)]
    for i, j in spot_pairs:
        counts = np.bincount(values[:, i] * 2 + values[:, j], minlength=4)
        for va, vb in itertools.product((0, 1), repeat=2):
            probe = {
                "conditions": [
                    {
                        "attribute": columns[i],
                        "operator": "==",
                        "value": va,
                    },
                    {
                        "attribute": columns[j],
                        "operator": "==",
                        "value": vb,
                    },
                ],
            }
            assert by_fingerprint[query_fingerprint(probe)] == int(
                counts[va * 2 + vb]
            )


def test_generation_input_audit_pins_all2way_workload():
    audit = diag._audit_generation_inputs(Path("."))
    assert audit["query_count"] == 480
    assert len(audit["queries"]) == len(audit["targets"]) == 480
    assert audit["input_sha256"]["measured"] == diag.ALL2WAY_SHA256
    assert all(
        math.isfinite(target) and target >= 0
        for target in audit["targets"]
    )


def test_cell_measurements_group_full_pair_marginals():
    root = Path(".")
    audit = diag._audit_generation_inputs(root)
    _, names, cards = pgm980._schema_domain(root)
    measurements, measurement_audit = diag._build_cell_measurements(
        audit, names, cards
    )
    assert len(measurements) == 120
    assert all(item["cell_count"] == 4 for item in measurement_audit)

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
        assert measurement.stddev == diag.MEASUREMENT_STDDEV


def test_feasibility_precheck_passes_with_tiny_model():
    root = Path(".")
    audit = diag._audit_generation_inputs(root)
    domain, names, cards = pgm980._schema_domain(root)
    cell_measurements, _ = diag._build_cell_measurements(
        audit, names, cards
    )
    one_way_measurements, _, _ = pgm980._build_one_way_measurements(
        root, names, cards
    )
    precheck = diag._feasibility_precheck(
        domain, cell_measurements + one_way_measurements
    )
    assert precheck["feasible"] is True
    assert precheck["max_clique_attribute_count"] == 16
    assert precheck["max_clique_cells"] == 65536
    assert precheck["model_size_mb"] < 1.0
    assert precheck["cap_mb"] == 4096.0


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


def test_comparison_restricted_to_comparable_groups():
    pgm_quality = _constant_arm_quality(2.0)
    references = {
        "engine_polish_budget_residual": {
            "snapshot": pgm980._arm_snapshot(_constant_arm_quality(0.5)),
        },
        "old_pgm_980": {
            "snapshot": pgm980._arm_snapshot(_constant_arm_quality(3.0)),
        },
    }
    result = diag._comparison(pgm_quality, references)
    deltas = result["normalized_l1_mean_deltas_comparable_groups_only"]
    assert set(deltas) == set(diag.COMPARABLE_GROUPS)
    assert "measured" not in deltas
    for entry in deltas.values():
        assert entry["pgm_all2way"]["mean"] == 2.0
        assert entry[
            "pgm_all2way_minus_engine_polish_budget_residual"
        ] == pytest.approx(1.5)
        assert entry["pgm_all2way_minus_old_pgm_980"] == pytest.approx(
            -1.0
        )
    anchors = result["measured_anchors_not_compared"]
    assert anchors["pgm_all2way_on_480_family"]["mean"] == 2.0
    assert anchors["engine_polish_on_1001_exam"]["mean"] == 0.5
    assert anchors["old_pgm_on_1001_exam"]["mean"] == 3.0


def test_reference_extraction_pins_and_shapes():
    references = diag._load_references(Path("."))
    polish = references["engine_polish_budget_residual"]
    old_pgm = references["old_pgm_980"]
    assert polish["sha256"] == diag.POLISH_REPORT_SHA256
    assert old_pgm["sha256"] == diag.OLD_PGM_REPORT_SHA256
    for reference in (polish, old_pgm):
        snapshot = reference["snapshot"]
        assert set(snapshot) == {
            "measured",
            "heldout_3way",
            "heldout_4way",
            "heldout_combined",
            "one_way_safety",
        }
        for stats in snapshot.values():
            assert all(
                math.isfinite(value) and value >= 0
                for value in stats.values()
            )
    assert polish["snapshot"]["heldout_3way"]["mean"] == pytest.approx(
        0.000433, abs=1e-5
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
