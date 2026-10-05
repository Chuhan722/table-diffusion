"""Tests for the fitness-only three-phase schedule v2 delta runner."""

import dataclasses
import json
from pathlib import Path

import pytest

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2


def test_plan_is_frozen_and_does_not_read_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(v1, "_sha256_file", forbidden)
    monkeypatch.setattr(v1, "_load_json_object", forbidden)
    plan = v2.build_plan()

    assert plan["protocol_sha256"] == v2.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 4
    assert plan["generation_started"] is False
    assert plan["output_dir"] == str(v2.OUTPUT_DIR)
    assert plan["protocol"]["inherits"]["base_protocol_sha256"] == (
        v1.FROZEN_PROTOCOL_SHA256
    )
    assert plan["protocol"]["rho_schedule"][
        "constants_frozen_before_any_v2_result"
    ] is True


def test_protocol_identity_is_fail_closed():
    assert v2.protocol_sha256() == v2.FROZEN_PROTOCOL_SHA256
    assert v2.assert_frozen_protocol_identity() == v2.FROZEN_PROTOCOL_SHA256


def test_wrong_confirmation_fails_before_any_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        v2,
        "_source_snapshot",
        lambda *_args, **_kwargs: called.append("source"),
    )

    with pytest.raises(ValueError, match="protocol SHA-256"):
        v2.run("not-the-frozen-hash")
    assert called == []


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "already-there"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(v2, "OUTPUT_DIR", Path("already-there"))

    with pytest.raises(FileExistsError, match="不覆盖"):
        v2.run(v2.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_generation_config_inherits_v1_exactly_except_schedule():
    v1_config = v1._json_protocol_manifest()["generation_config"]
    v2_config = v2._json_protocol_manifest()["generation_config"]

    schedule_keys = {
        "rho_anneal_end",
        "rho_anneal_rounds",
        "rho_anneal_start_round",
    }
    assert set(v2_config) == set(v1_config) | schedule_keys
    for key in set(v1_config) - schedule_keys:
        assert v2_config[key] == v1_config[key], key
    assert v1_config["rho_anneal_end"] is None
    assert v2_config["rho_anneal_end"] == 0.001
    assert v2_config["rho_anneal_rounds"] == 600
    assert v2_config["rho_anneal_start_round"] == 900


def test_fitness_config_delta_is_only_the_schedule():
    for spec in v1.DATASETS.values():
        base = dataclasses.asdict(v1._fitness_config(spec))
        delta = dataclasses.asdict(v2._fitness_config(spec))
        changed = {
            key for key in base
            if base[key] != delta[key]
        }
        assert changed == {
            "rho_anneal_start_round",
            "rho_anneal_rounds",
            "rho_anneal_end",
        }
        assert base["rho_anneal_end"] is None
        assert delta["rho_anneal_start_round"] == 900
        assert delta["rho_anneal_rounds"] == 600
        assert delta["rho_anneal_end"] == 0.001


def test_expected_schedule_three_phases_and_boundaries():
    schedule = v2._expected_schedule(v1.N_ROUNDS)
    assert len(schedule) == 3000
    assert schedule[0] == 0.01
    assert schedule[899] == 0.01
    assert schedule[900] == 0.01
    assert schedule[901] == pytest.approx(0.01 * 0.1 ** (1 / 600))
    assert schedule[1499] == pytest.approx(0.01 * 0.1 ** (599 / 600))
    assert schedule[1500] == pytest.approx(0.001)
    assert schedule[2999] == schedule[1500]
    assert min(schedule) == schedule[1500]


def test_audit_schedule_rejects_tampered_history(tmp_path):
    staging = tmp_path
    dataset = next(iter(v1.DATASETS))
    arm_dir = staging / "generation" / dataset
    arm_dir.mkdir(parents=True)
    tampered = v2._expected_schedule(v1.N_ROUNDS)
    tampered[1200] *= 1.5
    (arm_dir / f"{v1.ARMS[0]}_diagnostics.json").write_text(
        json.dumps({
            "rho_schedule_history": tampered,
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v2._audit_schedule(staging)


def _fake_support_inputs(drift, l1):
    schedule_audit = {"arms": {}}
    quality = {}
    for name in v1.DATASETS:
        schedule_audit["arms"][name] = {
            "residual": {
                "best_loss_diagnostic_only": 100.0,
                "output_squared_loss": 100.0 * drift[name],
            },
        }
        quality[name] = {
            "residual": {"measured": {"normalized_l1_mean": l1[name]}},
        }
    return schedule_audit, quality


def test_support_evaluation_applies_frozen_thresholds():
    passing = v2._support_evaluation(*_fake_support_inputs(
        drift={"test_300x10": 2.0, "nltcs": 1.05},
        l1={"test_300x10": 0.0034, "nltcs": 0.00027},
    ))
    assert passing["verdict"] == "schedule_dev_supported"
    assert passing["formal_claim_allowed"] is False

    drift_fail = v2._support_evaluation(*_fake_support_inputs(
        drift={"test_300x10": 2.2, "nltcs": 1.05},
        l1={"test_300x10": 0.0034, "nltcs": 0.00027},
    ))
    assert drift_fail["verdict"] == "schedule_dev_unsupported"
    assert drift_fail["per_dataset_residual_arm"]["test_300x10"][
        "drift_criterion_pass"
    ] is False

    l1_fail = v2._support_evaluation(*_fake_support_inputs(
        drift={"test_300x10": 2.0, "nltcs": 1.05},
        l1={"test_300x10": 0.0034, "nltcs": 0.000288},
    ))
    assert l1_fail["verdict"] == "schedule_dev_unsupported"
    assert l1_fail["per_dataset_residual_arm"]["nltcs"][
        "measured_criterion_pass"
    ] is False
