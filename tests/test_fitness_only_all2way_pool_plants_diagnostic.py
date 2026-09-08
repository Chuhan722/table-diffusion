"""Tests for the all-2way full-workload pool plants diagnostic protocol."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import (
    run_fitness_only_all2way_pool_plants_diagnostic as diag,
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
    stopping = protocol["early_stopping"]
    assert stopping["patience_ticks"] == 6
    assert stopping["stop_on_exact_residual"] is True
    assert set(stopping["termination_labels"]) == {
        "fit_target_reached",
        "early_stopped",
        "resource_cap_reached",
    }
    generation_input = protocol["generation_input"]
    assert generation_input["expected_pair_count"] == 2346
    assert generation_input["expected_all2way_count"] == 9384
    assert generation_input["expected_one_way_count"] == 138
    assert generation_input["expected_pool_count"] == 9522
    assert generation_input["all2way_sha256"] == diag.ALL2WAY_SHA256
    assert protocol["shared_generation_config"][
        "changed_relative_to_polish_budget"
    ] == ["query_pool"]
    assert "pgm_information_asymmetry_caveat" in protocol
    grouping = protocol["evaluation_grouping"]
    assert grouping["measured_group_is_original_980_exam_continuity"]
    assert grouping["measured_980_doubles_now_in_pool"] == 460
    assert grouping["measured_980_triples_now_out_of_pool"] == 520
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
    # 配置整体复用 polish 的 _config：本协议不得自带配置定义。
    assert not hasattr(diag, "_config")
    identity = polish.assert_polish_budget_identity()
    assert identity["minimality_and_rounding"] == "pass"


def test_all2way_workload_file_pinned_and_wellformed():
    path = Path(diag.ALL2WAY_PATH)
    assert diag._sha256_file(path) == diag.ALL2WAY_SHA256
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["record_count"] == 17412
    assert payload["query_count"] == 9384
    queries = payload["queries"]
    assert len(queries) == 9384
    assert all(len(q["conditions"]) == 2 for q in queries)
    assert all(q["type"] == "double" for q in queries)
    fingerprints = [query_fingerprint(q) for q in queries]
    assert len(set(fingerprints)) == len(fingerprints)
    group_sums: dict[str, int] = {}
    group_cells: dict[str, int] = {}
    for query in queries:
        group_sums[query["group"]] = (
            group_sums.get(query["group"], 0) + int(query["result"])
        )
        group_cells[query["group"]] = (
            group_cells.get(query["group"], 0) + 1
        )
    assert len(group_sums) == 2346
    assert all(count == 4 for count in group_cells.values())
    assert all(total == 17412 for total in group_sums.values())


def test_all2way_counts_match_source_data_spot_pairs():
    payload = json.loads(
        Path(diag.ALL2WAY_PATH).read_text(encoding="utf-8")
    )
    by_fingerprint = {
        query_fingerprint(q): int(q["result"])
        for q in payload["queries"]
    }
    df = pd.read_csv("data/plants/plants.csv")
    columns = list(df.columns)
    values = df.to_numpy()
    spot_pairs = [
        (0, 1),
        (0, len(columns) - 1),
        (len(columns) - 2, len(columns) - 1),
        (10, 26),
    ]
    for i, j in spot_pairs:
        counts = np.bincount(
            values[:, i] * 2 + values[:, j], minlength=4
        )
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


def test_audit_generation_inputs_pool_assembly():
    audit = diag._audit_generation_inputs(Path("."))
    assert audit["query_count"] == 980
    assert audit["all2way_count"] == 9384
    assert audit["one_way_count"] == 138
    assert audit["pool_query_count"] == 9522
    assert audit["measured_doubles_subsumed_by_all2way"] == 460
    assert audit["pool_queries"][:9384] == audit["all2way_queries"]
    assert audit["pool_queries"][9384:] == audit["one_way_queries"]
    assert audit["pool_targets"][:9384] == audit["all2way_targets"]
    assert audit["pool_targets"][9384:] == audit["one_way_targets"]
    assert audit["all2way_sha256"] == diag.ALL2WAY_SHA256
    for key in (
        "all2way_identity_sha256",
        "all2way_target_vector_sha256",
        "pool_identity_sha256",
        "pool_target_vector_sha256",
    ):
        assert len(audit[key]) == 64


def test_all2way_pool_disjoint_from_heldout():
    audit = diag._audit_generation_inputs(Path("."))
    heldout_queries, _, _ = oneway._load_heldout_and_reference(Path("."))
    validate_query_partition(
        list(audit["pool_queries"]), list(heldout_queries)
    )


def _constant_quality(value: float) -> dict:
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
        "measured_by_order_and_target_bucket": {
            "by_order": {
                "2way": {"normalized_l1_mean": value},
                "3way": {"normalized_l1_mean": value},
            },
        },
        "all2way_pool": dict(stats),
    }


def test_comparison_reports_deltas_and_probes():
    references = {
        "polish_budget_residual": oneway._metric_snapshot(
            _constant_quality(3.0)
        ),
        "polish_budget_measured_by_order": {
            "2way": {"normalized_l1_mean": 3.5},
            "3way": {"normalized_l1_mean": 4.0},
        },
        "oneway_pool_residual": oneway._metric_snapshot(
            _constant_quality(6.0)
        ),
        "frozen_residual": oneway._metric_snapshot(
            _constant_quality(5.0)
        ),
        "pgm": oneway._metric_snapshot(_constant_quality(1.0)),
    }
    result = diag._comparison(_constant_quality(2.0), references)
    assert set(result["new_residual_snapshot"]) == {
        "measured",
        "heldout_3way",
        "heldout_4way",
        "heldout_combined",
        "one_way_safety",
    }
    for entry in result[
        "new_residual_minus_polish_budget_residual"
    ].values():
        assert entry["delta_mean"] == pytest.approx(-1.0)
    for entry in result[
        "new_residual_minus_oneway_pool_residual"
    ].values():
        assert entry["delta_mean"] == pytest.approx(-4.0)
    for entry in result[
        "new_residual_minus_pgm_context_only"
    ].values():
        assert entry["delta_mean"] == pytest.approx(1.0)
    probe = result["out_of_pool_triples_probe"]
    assert probe["new_measured_3way_bucket_mean"] == pytest.approx(2.0)
    assert probe["polish_in_pool_3way_bucket_mean"] == pytest.approx(4.0)
    anchor = result["all2way_pool_anchor"]
    assert anchor["normalized_l1_mean"] == pytest.approx(2.0)
    assert anchor["normalized_l1_max"] == pytest.approx(2.0)


def test_reference_extraction_pins_and_shapes():
    try:
        references = diag._load_reference_reports(Path("."))
    except FileNotFoundError as exc:
        pytest.skip(f"冻结产物不在本机（gitignored），产物持有机复核：{exc}")
    assert references["reference_sha256"] == diag.REFERENCE_SHA256
    for key in (
        "polish_budget_residual",
        "oneway_pool_residual",
        "oneway_pool_equal_context_only",
        "frozen_residual",
        "pgm",
    ):
        snapshot = references[key]
        for stats in snapshot.values():
            assert all(
                math.isfinite(value) and value >= 0
                for value in stats.values()
            )
    assert references["polish_budget_residual"]["heldout_3way"][
        "mean"
    ] == pytest.approx(0.032006, abs=1e-5)
    assert references["polish_budget_residual"]["one_way_safety"][
        "mean"
    ] == pytest.approx(0.002007, abs=1e-5)
    by_order = references["polish_budget_measured_by_order"]
    assert by_order["3way"]["normalized_l1_mean"] == pytest.approx(
        0.001646, abs=1e-5
    )
    assert references["pgm"]["heldout_3way"]["mean"] == pytest.approx(
        0.015229, abs=1e-5
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
