from __future__ import annotations

import pytest

from scripts import run_issue53_p6_dataset_smoke as base
from scripts import run_issue53_test_residual_confirmation as confirmation


def test_generator_matrix_changes_only_seed_and_geometry():
    baseline = base.generator_params()

    for seed in confirmation.SEEDS:
        for arm in confirmation.ARMS:
            params = confirmation.generator_params(seed, arm)
            assert params["seed"] == seed
            assert params["residual_geometry"] == arm
            differing = {
                key for key in params if params[key] != baseline[key]
            }
            expected = {"seed", "residual_geometry"}
            if seed == baseline["seed"]:
                expected.remove("seed")
            if arm == baseline["residual_geometry"]:
                expected.remove("residual_geometry")
            assert differing == expected


def test_generator_rejects_nonfrozen_cases():
    with pytest.raises(ValueError, match="seed 不在冻结矩阵"):
        confirmation.generator_params(999, "absolute")
    with pytest.raises(ValueError, match="arm 不在冻结矩阵"):
        confirmation.generator_params(313, "new_arm")


def test_frozen_protocol_is_exact_five_seed_numpy_collection():
    protocol = confirmation.frozen_protocol_manifest()

    assert confirmation.protocol_sha256() == confirmation.FROZEN_PROTOCOL_SHA256
    assert protocol["dataset"]["name"] == "test_300x10"
    assert protocol["dataset"]["device"] == "numpy"
    assert protocol["arms"] == ["absolute", "sqrt_relative", "relative"]
    assert protocol["seeds"] == [313, 314, 315, 316, 317]
    assert protocol["trajectory_count"] == 15
    assert protocol["common_generator"][
        "inner_early_stopping_patience_ticks"
    ] == 6
    assert protocol["common_generator"]["n_rounds"] == 6000
    assert protocol["common_generator"]["candidate_budget"] == 6000
    assert protocol["common_generator"]["rho"] == 0.01
    assert protocol["common_generator"]["fixed_alpha"] == 16.0
    assert protocol["execution_concurrency"]["worker_count"] == 5
    assert protocol["execution_concurrency"]["cuda_visible_devices"] == "empty"
    assert protocol["parameter_retuning_allowed"] is False


def test_plan_is_result_blind(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan must not read inputs or generate")

    monkeypatch.setattr(confirmation, "_load_dataset", forbidden)
    monkeypatch.setattr(confirmation, "run_evolution", forbidden)
    monkeypatch.setattr(confirmation, "_audit_inputs", forbidden)

    plan = confirmation.build_plan()

    assert [row["seed"] for row in plan["shards"]] == [313, 314, 315, 316, 317]
    assert [row["case_count"] for row in plan["shards"]] == [3] * 5
    assert plan["case_order_within_shard"] == list(confirmation.ARMS)
    assert plan["scientific_overrides_allowed"] is False
    assert plan["generation_started"] is False


def test_cli_exposes_only_execution_shard_override():
    parser = confirmation._build_parser()
    parsed = parser.parse_args(
        [
            "run-shard",
            "--confirm-protocol-sha",
            confirmation.FROZEN_PROTOCOL_SHA256,
            "--shard-index",
            "2",
        ]
    )
    assert vars(parsed) == {
        "command": "run-shard",
        "confirm_protocol_sha": confirmation.FROZEN_PROTOCOL_SHA256,
        "shard_index": 2,
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-shard",
                "--confirm-protocol-sha",
                confirmation.FROZEN_PROTOCOL_SHA256,
                "--shard-index",
                "0",
                "--rho",
                "0.1",
            ]
        )


def test_wrong_sha_fails_before_environment_or_generation(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("wrong SHA must fail first")

    monkeypatch.setattr(confirmation, "_environment", forbidden)
    monkeypatch.setattr(confirmation, "run_evolution", forbidden)
    with pytest.raises(ValueError, match="显式确认"):
        confirmation.run_shard("0" * 64, 0)


def _row(seed: int, arm: str, reason: str = "early_stopped") -> dict:
    return {
        "seed": seed,
        "arm": arm,
        "termination_reason": reason,
        "terminal_current_normalized_l1": seed / 100000,
        "terminal_current_squared_loss": float(seed),
        "rounds_run": seed,
        "normalized_work_at_stop": seed / 10,
        "elapsed_sec": 1.0,
    }


def test_collection_summary_keeps_arms_separate_and_counts_resource_caps():
    rows = [
        _row(
            seed,
            arm,
            reason=(
                "resource_cap_reached"
                if seed == 317 and arm == "relative"
                else "early_stopped"
            ),
        )
        for seed in confirmation.SEEDS
        for arm in confirmation.ARMS
    ]

    summary = confirmation._summarize(rows)

    assert set(summary["arms"]) == set(confirmation.ARMS)
    assert all(row["case_count"] == 5 for row in summary["arms"].values())
    assert summary["resource_cap_case_count"] == 1
    assert summary["normal_completion_case_count"] == 14
