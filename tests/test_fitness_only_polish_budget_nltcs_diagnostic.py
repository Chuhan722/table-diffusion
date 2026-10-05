"""Tests for the unified polish-density recipe nltcs archive protocol."""

import dataclasses
import math
from pathlib import Path

import pytest

from scripts import (
    run_fitness_only_oneway_pool_nltcs_diagnostic as oneway,
)
from scripts import (
    run_fitness_only_polish_budget_nltcs_diagnostic as diag,
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
    assert protocol["dataset"] == "nltcs"
    assert protocol["n_records"] == 16181
    assert protocol["n_attributes"] == 16
    assert protocol["arms"] == ["residual"]
    budget = protocol["budget"]
    assert budget["n_rounds"] == 7000
    assert budget["rho_anneal_start_round"] == 1050
    assert budget["rho_anneal_rounds"] == 700
    principle = protocol["polish_density_principle"]
    assert principle["target_kappa"] == 1.0
    assert principle["polish_target_per_row"] == 16.0
    assert principle["solved_minimal_rounds"] == 6057
    assert principle["cap_rounding_unit"] == 1000
    stopping = protocol["early_stopping"]
    assert stopping["patience_ticks"] == 6
    assert stopping["stop_on_exact_residual"] is True
    assert protocol["equal_arm_cut"]["rerun"] is False
    assert protocol["interpretation"]["diagnostic_only"] is True
    assert protocol["interpretation"]["formal_claim_allowed"] is False


def test_protocol_identity_is_frozen():
    assert diag.protocol_sha256() == diag.FROZEN_PROTOCOL_SHA256
    assert diag.assert_frozen_protocol_identity() == (
        diag.FROZEN_PROTOCOL_SHA256
    )


def test_polish_budget_identity_minimality_and_rounding():
    identity = diag.assert_polish_budget_identity()
    assert identity["minimality_and_rounding"] == "pass"
    assert identity["solved_minimal_rounds"] == 6057
    assert identity["budget_cap_rounds"] == 7000
    assert identity["polish_at_solved"] >= 16.0
    assert identity["polish_below_solved"] < 16.0
    assert identity["polish_at_cap"] >= 16.0
    assert identity["kappa_at_cap"] == pytest.approx(1.1557, abs=1e-3)


def test_config_differs_from_oneway_pool_only_in_predeclared_fields():
    expected = dataclasses.replace(
        oneway._config(),
        n_rounds=7000,
        rho_anneal_start_round=1050,
        rho_anneal_rounds=700,
        inner_early_stopping_patience_ticks=6,
    )
    assert diag._config() == expected


def test_relative_schedule_matches_frozen_constants():
    assert diag._relative_schedule(diag.N_ROUNDS) == (1050, 700)
    assert diag._relative_schedule(6000) == (900, 600)


def test_expected_rho_schedule_shape():
    schedule = diag._expected_rho_schedule(
        diag.N_ROUNDS,
        diag.RHO_ANNEAL_START_ROUND,
        diag.RHO_ANNEAL_ROUNDS,
    )
    assert len(schedule) == 7000
    assert all(value == pytest.approx(0.01) for value in schedule[:1050])
    assert all(value == pytest.approx(0.001) for value in schedule[1750:])
    assert schedule[1050] > schedule[1400] > schedule[1749]
    assert sum(schedule) == pytest.approx(18.4906, abs=1e-3)


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
    references = {
        "oneway_pool_residual": oneway._metric_snapshot(
            _constant_quality(3.0)
        ),
        "frozen_residual": oneway._metric_snapshot(_constant_quality(5.0)),
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
        "new_residual_minus_oneway_pool_residual"
    ].values():
        assert entry["delta_mean"] == pytest.approx(-1.0)
    for entry in result["new_residual_minus_frozen_residual"].values():
        assert entry["delta_mean"] == pytest.approx(-3.0)
    for entry in result["new_residual_minus_pgm"].values():
        assert entry["delta_mean"] == pytest.approx(1.0)


def test_reference_extraction_pins_and_shapes():
    try:
        references = diag._load_reference_reports(Path("."))
    except FileNotFoundError as exc:
        pytest.skip(f"冻结产物不在本机（gitignored），产物持有机复核：{exc}")
    assert references["reference_sha256"] == diag.REFERENCE_SHA256
    for key in (
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
    assert references["oneway_pool_residual"]["one_way_safety"][
        "mean"
    ] == pytest.approx(0.000108, abs=1e-5)
    assert references["pgm"]["measured"]["mean"] == pytest.approx(
        0.000550, abs=1e-5
    )


def test_early_stopping_summary_requires_enabled():
    with pytest.raises(RuntimeError, match="缺失"):
        diag._early_stopping_summary({
            "inner_early_stopping": {"enabled": False},
            "termination_reason": "max_rounds",
            "stopped_early": False,
        })


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
