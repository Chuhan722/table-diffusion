"""Tests for the one-way-in-pool nltcs diagnostic protocol."""

import math
from pathlib import Path

import pytest

# plants 诊断 runner 的 _config() 是 v3 冻结配置的逐字权威源（两数据集共用）。
from scripts import run_fitness_only_plants_diagnostic as v3_frozen
from scripts import run_fitness_only_oneway_pool_nltcs_diagnostic as diag


def test_plan_is_frozen_and_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(diag, "_sha256_file", forbidden)
    monkeypatch.setattr(diag, "_load_json_object", forbidden)
    plan = diag.build_plan()
    assert plan["protocol_sha256"] == diag.FROZEN_PROTOCOL_SHA256
    assert plan["generation_started"] is False
    protocol = plan["protocol"]
    assert protocol["dataset"] == "nltcs"
    assert protocol["n_records"] == 16181
    gen = protocol["generation_input"]
    assert gen["expected_measured_count"] == 1001
    assert gen["expected_one_way_count"] == 32
    assert gen["expected_pool_count"] == 1033
    grouping = protocol["evaluation_grouping"]
    assert grouping["one_way_now_in_evolution_pool"] is True
    assert grouping[
        "measured_metric_uses_only_original_measured_queries"
    ] is True
    reference = protocol["comparison_reference"]
    assert reference["rerun_frozen_arms"] is False
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diag.protocol_sha256() == diag.FROZEN_PROTOCOL_SHA256
    assert diag.assert_frozen_protocol_identity() == (
        diag.FROZEN_PROTOCOL_SHA256
    )


def test_config_matches_frozen_v3_configuration_verbatim():
    assert diag._config() == v3_frozen._config()


def test_one_way_pool_queries_match_marginals():
    queries, targets = diag._one_way_pool_queries(Path("."))
    assert len(queries) == 32
    assert len(targets) == 32
    assert all(len(q["conditions"]) == 1 for q in queries)
    assert all(q["type"] == "single" for q in queries)
    assert all(
        math.isfinite(t) and t >= 0 for t in targets
    )
    assert [q["id"] for q in queries] == [
        f"OW{i:04d}" for i in range(1, 33)
    ]
    per_attribute: dict[str, float] = {}
    for query, target in zip(queries, targets):
        assert float(query["result"]) == target
        attribute = query["conditions"][0]["attribute"]
        per_attribute[attribute] = per_attribute.get(attribute, 0.0) + target
    assert len(per_attribute) == 16
    for total in per_attribute.values():
        assert total == pytest.approx(16181.0)


def test_generation_input_audit_builds_disjoint_pool():
    audit = diag._audit_generation_inputs(Path("."))
    assert audit["query_count"] == 1001
    assert audit["one_way_count"] == 32
    assert audit["pool_query_count"] == 1033
    assert len(audit["pool_queries"]) == 1033
    assert len(audit["pool_targets"]) == 1033
    assert audit["pool_queries"][:1001] == audit["queries"]
    assert audit["pool_targets"][:1001] == audit["targets"]
    assert audit["pool_identity_sha256"] != audit["query_identity_sha256"]
    assert all(
        len(query["conditions"]) >= 2 for query in audit["queries"]
    )
    assert all(
        len(query["conditions"]) == 1
        for query in audit["pool_queries"][1001:]
    )


def test_metric_snapshot_extracts_all_groups():
    stats = {
        f"normalized_l1_{name}": 0.25
        for name in ("mean", "median", "p90", "max")
    }
    quality = {
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
    snapshot = diag._metric_snapshot(quality)
    assert set(snapshot) == {
        "measured",
        "heldout_3way",
        "heldout_4way",
        "heldout_combined",
        "one_way_safety",
    }
    for entry in snapshot.values():
        assert entry == {
            "mean": 0.25, "median": 0.25, "p90": 0.25, "max": 0.25,
        }


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
    }


def test_comparison_reports_all_reference_deltas():
    quality = {
        "residual": _constant_quality(2.0),
        "equal": _constant_quality(5.0),
    }
    references = {
        "frozen_arms": {
            "residual": diag._metric_snapshot(_constant_quality(3.0)),
            "equal": diag._metric_snapshot(_constant_quality(5.0)),
        },
        "pgm": diag._metric_snapshot(_constant_quality(1.5)),
        "frozen_equal_terminal_table_sha256": "abc",
    }
    generation = {
        "arms": {"equal": {"terminal_table_sha256": "abc"}},
    }
    result = diag._comparison(quality, references, generation)
    assert result["equal_table_identity_match"] is True
    for entry in result["internal_residual_minus_equal"].values():
        assert entry["delta_mean"] == pytest.approx(-3.0)
    for entry in result["new_residual_minus_frozen_residual"].values():
        assert entry["delta_mean"] == pytest.approx(-1.0)
    for entry in result["new_equal_minus_frozen_equal"].values():
        assert entry["delta_mean"] == pytest.approx(0.0)
    for entry in result["new_residual_minus_pgm"].values():
        assert entry["delta_mean"] == pytest.approx(0.5)

    generation_mismatch = {
        "arms": {"equal": {"terminal_table_sha256": "other"}},
    }
    mismatch = diag._comparison(quality, references, generation_mismatch)
    assert mismatch["equal_table_identity_match"] is False


def test_frozen_reference_extraction_pins_and_shapes():
    try:
        references = diag._load_frozen_references(Path("."))
    except FileNotFoundError as exc:
        pytest.skip(f"冻结产物不在本机（gitignored），产物持有机复核：{exc}")
    assert references["frozen_report_sha256"] == (
        diag.INPUT_SHA256["frozen_report"]
    )
    assert references["pgm_report_sha256"] == (
        diag.INPUT_SHA256["pgm_report"]
    )
    assert set(references["frozen_arms"]) == {"residual", "equal"}
    assert len(references["frozen_equal_terminal_table_sha256"]) == 64
    for snapshot in (
        *references["frozen_arms"].values(),
        references["pgm"],
    ):
        for stats in snapshot.values():
            assert all(
                math.isfinite(value) and value >= 0
                for value in stats.values()
            )


def test_expected_rho_schedule_shape():
    schedule = diag._expected_rho_schedule(diag.N_ROUNDS)
    assert len(schedule) == 6000
    assert all(value == pytest.approx(0.01) for value in schedule[:900])
    assert all(value == pytest.approx(0.001) for value in schedule[1500:])
    assert schedule[900] > schedule[1200] > schedule[1499]


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
