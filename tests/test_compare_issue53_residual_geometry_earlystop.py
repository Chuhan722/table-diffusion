from __future__ import annotations

import pytest

from scripts import compare_issue53_residual_geometry_earlystop as comparison
from scripts import run_issue53_p6_dataset_smoke as base


def test_generator_matrix_changes_only_seed_and_residual_geometry():
    baseline = base.generator_params()

    for seed in comparison.SEEDS:
        for arm in comparison.ARMS:
            params = comparison.generator_params(seed, arm)
            assert params["seed"] == seed
            assert params["residual_geometry"] == arm
            assert params["residual_geometry_floor"] == 8.0
            differing = {
                key for key in params if params[key] != baseline[key]
            }
            expected = {"seed", "residual_geometry"}
            if seed == baseline["seed"]:
                expected.remove("seed")
            if arm == baseline["residual_geometry"]:
                expected.remove("residual_geometry")
            assert differing == expected


def test_generator_rejects_nonfrozen_seed_and_arm():
    with pytest.raises(ValueError, match="seed 不在冻结矩阵"):
        comparison.generator_params(999, "absolute")
    with pytest.raises(ValueError, match="arm 不在冻结矩阵"):
        comparison.generator_params(310, "unknown")


def test_protocol_scope_and_execution_are_fixed():
    protocol = comparison.frozen_protocol_manifest()

    assert protocol["datasets"][0]["name"] == "test_300x10"
    assert protocol["datasets"][1]["name"] == "nltcs"
    assert protocol["arms"] == ["absolute", "sqrt_relative", "relative"]
    assert protocol["seeds"] == [310, 311, 312]
    assert protocol["trajectory_count"] == 18
    assert protocol["common_generator"]["inner_early_stopping_patience_ticks"] == 6
    assert protocol["common_generator"]["n_rounds"] == 6000
    assert protocol["common_generator"]["candidate_budget"] == 6000
    assert protocol["common_generator"]["rho"] == 0.01
    assert protocol["common_generator"]["fixed_alpha"] == 16.0
    assert protocol["common_generator"]["factorized_gibbs_sweeps"] == 0
    assert protocol["execution_concurrency"]["visible_gpu_count"] == 1
    assert protocol["execution_concurrency"]["worker_count"] == 1
    assert protocol["execution_concurrency"]["seed_shards_serial"] is True
    assert protocol["parameter_retuning_allowed"] is False
    assert protocol["canonical_selection_allowed"] is False
    assert protocol["online_l1_used"] is False


def test_plan_is_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan must not read inputs or generate")

    monkeypatch.setattr(comparison, "_load_dataset", forbidden)
    monkeypatch.setattr(comparison, "run_evolution", forbidden)
    monkeypatch.setattr(comparison.base, "_audit_inputs", forbidden)

    plan = comparison.build_plan()

    assert plan["protocol_sha256"] == comparison.FROZEN_PROTOCOL_SHA256
    assert [item["seed"] for item in plan["shards"]] == [310, 311, 312]
    assert [item["case_count"] for item in plan["shards"]] == [6, 6, 6]
    assert len(plan["case_order_within_shard"]) == 6
    assert plan["scientific_overrides_allowed"] is False
    assert plan["generation_started"] is False


def test_cli_has_only_execution_shard_override():
    parser = comparison._build_parser()
    parsed = parser.parse_args(
        [
            "run-shard",
            "--confirm-protocol-sha",
            comparison.FROZEN_PROTOCOL_SHA256,
            "--shard-index",
            "1",
        ]
    )
    assert vars(parsed) == {
        "command": "run-shard",
        "confirm_protocol_sha": comparison.FROZEN_PROTOCOL_SHA256,
        "shard_index": 1,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-shard",
                "--confirm-protocol-sha",
                comparison.FROZEN_PROTOCOL_SHA256,
                "--shard-index",
                "0",
                "--rho",
                "0.1",
            ]
        )


def test_wrong_sha_fails_before_environment_or_generation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong SHA must fail first")

    monkeypatch.setattr(comparison.base, "_environment", forbidden)
    monkeypatch.setattr(comparison, "run_evolution", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        comparison.run_shard("0" * 64, 0)


def _fake_row(
    arm: str,
    seed: int,
    l1: float,
    *,
    reason: str = "early_stopped",
) -> dict:
    return {
        "arm": arm,
        "seed": seed,
        "termination_reason": reason,
        "terminal_current_normalized_l1": l1,
        "terminal_current_squared_loss": l1 * 1000,
        "rounds_run": 1000 + seed,
        "normalized_work_at_stop": 10.0 + seed / 1000,
        "elapsed_sec": 1.0,
    }


def test_summary_reports_mean_winner_pair_wins_and_resource_caps():
    rows = []
    for seed in comparison.SEEDS:
        rows.extend(
            [
                _fake_row("absolute", seed, 0.3),
                _fake_row("sqrt_relative", seed, 0.1),
                _fake_row(
                    "relative",
                    seed,
                    0.2,
                    reason=(
                        "resource_cap_reached"
                        if seed == comparison.SEEDS[-1]
                        else "early_stopped"
                    ),
                ),
            ]
        )

    summary = comparison._summarize_dataset(rows)

    assert summary["lowest_mean_terminal_l1_arms"] == ["sqrt_relative"]
    assert summary["paired_seed_l1_win_counts"] == {"sqrt_relative": 3}
    assert summary["resource_cap_case_count"] == 1
    assert summary["arms"]["relative"]["termination_counts"] == {
        "early_stopped": 2,
        "resource_cap_reached": 1,
    }
