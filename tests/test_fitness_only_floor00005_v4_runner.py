"""Tests for the fitness-only lower-floor (0.0005) v4 delta runner."""

import dataclasses
import json
from pathlib import Path

import pytest

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2
from scripts import run_fitness_only_schedule_v3_t6000 as v3
from scripts import run_fitness_only_floor00005_v4 as v4


def test_plan_is_frozen_and_does_not_read_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(v1, "_sha256_file", forbidden)
    monkeypatch.setattr(v1, "_load_json_object", forbidden)
    plan = v4.build_plan()

    assert plan["protocol_sha256"] == v4.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 4
    assert plan["generation_started"] is False
    assert plan["output_dir"] == str(v4.OUTPUT_DIR)
    assert plan["protocol"]["inherits"]["base_protocol_sha256"] == (
        v3.FROZEN_PROTOCOL_SHA256
    )
    hypothesis = plan["protocol"]["lower_floor_hypothesis"]
    assert hypothesis["changes"]["rho_anneal_end"] == {
        "from": 0.001, "to": 0.0005,
    }
    assert hypothesis["changes"]["n_rounds"] == {"from": 6000, "to": 9000}
    assert "truncation_hypothesis" not in plan["protocol"]
    schedule = plan["protocol"]["rho_schedule"]
    assert schedule["floor"] == 0.0005
    assert schedule["floor_ratio"] == 0.05
    assert schedule["hold_rounds_H"] == 900
    assert schedule["descent_rounds_D"] == 600


def test_protocol_identity_is_fail_closed():
    assert v4.protocol_sha256() == v4.FROZEN_PROTOCOL_SHA256
    assert v4.assert_frozen_protocol_identity() == v4.FROZEN_PROTOCOL_SHA256


def test_wrong_confirmation_fails_before_any_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        v4,
        "_verify_v3_reference_artifacts",
        lambda *_args, **_kwargs: called.append("v3-audit"),
    )
    monkeypatch.setattr(
        v4,
        "_source_snapshot",
        lambda *_args, **_kwargs: called.append("source"),
    )

    with pytest.raises(ValueError, match="protocol SHA-256"):
        v4.run("not-the-frozen-hash")
    assert called == []


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "already-there"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(v4, "OUTPUT_DIR", Path("already-there"))

    with pytest.raises(FileExistsError, match="不覆盖"):
        v4.run(v4.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_fitness_config_delta_is_floor_and_budget_only():
    for spec in v1.DATASETS.values():
        base = dataclasses.asdict(v3._fitness_config(spec))
        delta = dataclasses.asdict(v4._fitness_config(spec))
        changed = {key for key in base if base[key] != delta[key]}
        assert changed == {"rho_anneal_end", "n_rounds"}
        assert base["rho_anneal_end"] == 0.001
        assert delta["rho_anneal_end"] == 0.0005
        assert base["n_rounds"] == 6000
        assert delta["n_rounds"] == 9000
        assert delta["rho_anneal_start_round"] == 900
        assert delta["rho_anneal_rounds"] == 600


def test_manifest_inherits_v3_except_floor_and_rounds():
    v3_manifest = v3._json_protocol_manifest()
    v4_manifest = v4._json_protocol_manifest()

    assert v3_manifest["n_rounds"] == 6000
    assert v4_manifest["n_rounds"] == 9000
    v3_config = dict(v3_manifest["generation_config"])
    v4_config = dict(v4_manifest["generation_config"])
    assert v3_config.pop("rho_anneal_end") == 0.001
    assert v4_config.pop("rho_anneal_end") == 0.0005
    assert v4_config == v3_config
    assert v4_manifest["datasets"] == v3_manifest["datasets"]
    assert v4_manifest["seed"] == v3_manifest["seed"]


def test_expected_schedule_prefix_and_floor_boundaries():
    schedule_v4 = v4._expected_schedule(v4.N_ROUNDS)
    schedule_v3 = v2._expected_schedule(v3.N_ROUNDS)

    assert len(schedule_v4) == 9000
    # 前 901 轮（t∈[0,900]）与 v3 逐位一致；t=901 起分叉。
    assert schedule_v4[:901] == schedule_v3[:901]
    assert schedule_v4[900] == 0.01
    assert schedule_v4[901] != schedule_v3[901]
    assert schedule_v4[901] == pytest.approx(0.01 * 0.05 ** (1 / 600))
    assert schedule_v4[1499] == pytest.approx(0.01 * 0.05 ** (599 / 600))
    floor = schedule_v4[1500]
    assert floor == pytest.approx(0.0005)
    assert all(value == floor for value in schedule_v4[1500:])
    assert schedule_v4[0] == 0.01


def test_audit_schedule_rejects_tampered_or_v3_history(tmp_path):
    dataset = next(iter(v1.DATASETS))
    arm_dir = tmp_path / "generation" / dataset
    arm_dir.mkdir(parents=True)

    tampered = v4._expected_schedule(v4.N_ROUNDS)
    tampered[7000] *= 1.5
    (arm_dir / f"{v1.ARMS[0]}_diagnostics.json").write_text(
        json.dumps({
            "rho_schedule_history": tampered,
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v4._audit_schedule(tmp_path)

    # v3 的 0.001 地板时间表（即使补长到 9000 轮）也必须被拒。
    stale = v2._expected_schedule(v3.N_ROUNDS)
    stale = stale + [stale[-1]] * 3000
    (arm_dir / f"{v1.ARMS[0]}_diagnostics.json").write_text(
        json.dumps({
            "rho_schedule_history": stale,
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v4._audit_schedule(tmp_path)


def test_v3_reference_artifacts_are_hash_verified(tmp_path):
    base = tmp_path / v4.V3_OUTPUT_DIR
    with pytest.raises(RuntimeError, match="缺失"):
        v4._verify_v3_reference_artifacts(tmp_path)

    for rel_path in v4.V3_REFERENCE_ARTIFACT_SHA256:
        path = base / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{\"forged\": true}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="身份漂移"):
        v4._verify_v3_reference_artifacts(tmp_path)


def _diag(loss, rho, init="init-hash", rng="rng-hash"):
    return {
        "loss_history": list(loss),
        "rho_schedule_history": list(rho),
        "initial_table_sha256": init,
        "primary_rng_post_initialization_state_sha256": rng,
    }


def _paired_diagnostics():
    shared_loss = [float(9000 - t) for t in range(901)]
    v3_loss = shared_loss + [1.0] * (6000 - 901)
    v4_loss = shared_loss + [2.0] * (9000 - 901)
    v3_rho = v2._expected_schedule(v3.N_ROUNDS)
    v4_rho = v4._expected_schedule(v4.N_ROUNDS)
    old = {
        name: {arm: _diag(v3_loss, v3_rho) for arm in v1.ARMS}
        for name in v1.DATASETS
    }
    new = {
        name: {arm: _diag(v4_loss, v4_rho) for arm in v1.ARMS}
        for name in v1.DATASETS
    }
    return new, old


def test_prefix_audit_passes_on_identical_901_prefix():
    new, old = _paired_diagnostics()
    audit = v4._prefix_audit(new, old)
    assert audit["prefix_rounds"] == 901
    assert audit["passed"] is True
    assert audit["failure_label"] is None
    for name in v1.DATASETS:
        for arm in v1.ARMS:
            assert audit["arms"][name][arm]["passed"] is True


def test_prefix_audit_rejects_any_forged_prefix():
    # 前缀最后一轮（t=900）被篡改必须被抓住。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][900] += 0.5
    audit = v4._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["failure_label"] == "schedule_blindness_violated"
    detail = audit["arms"]["nltcs"]["residual"]
    assert detail["loss_history_prefix_identical"] is False
    assert detail["loss_history_first_mismatch_round"] == 900
    assert audit["arms"]["nltcs"]["equal"]["passed"] is True

    # 前缀之外（t=901）的差异不构成违规。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][901] += 123.0
    audit = v4._prefix_audit(new, old)
    assert audit["passed"] is True

    new, old = _paired_diagnostics()
    new["test_300x10"]["equal"]["rho_schedule_history"][0] *= 2.0
    audit = v4._prefix_audit(new, old)
    assert audit["passed"] is False
    detail = audit["arms"]["test_300x10"]["equal"]
    assert detail["rho_schedule_history_prefix_identical"] is False
    assert detail["rho_schedule_first_mismatch_round"] == 0

    new, old = _paired_diagnostics()
    new["nltcs"]["equal"]["initial_table_sha256"] = "other"
    audit = v4._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["nltcs"]["equal"][
        "initial_table_sha256_identical"
    ] is False

    new, old = _paired_diagnostics()
    new["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_sha256"
    ] = "other"
    audit = v4._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_identical"
    ] is False


def test_floor_morphology_descending_rule_uses_v4_windows():
    descending = v4._floor_morphology(
        [float(9000 - t) for t in range(9000)]
    )
    assert descending["descending"] is True
    assert descending["last_running_best_update_round"] == 8999
    assert descending["tail_window"] == [8000, 9000]
    assert descending["prev_window"] == [7000, 8000]

    flat_tail = [float(9000 - t) for t in range(8000)] + [2000.0] * 1000
    plateau = v4._floor_morphology(flat_tail)
    assert plateau["descending"] is False
    assert plateau["tail_below_prev"] is False

    early_best = [1.0] + [200.0] * 7999 + [100.0] * 1000
    stale = v4._floor_morphology(early_best)
    assert stale["tail_below_prev"] is True
    assert stale["last_running_best_update_round"] == 0
    assert stale["descending"] is False


def _fake_verdict_inputs(nltcs_drift, nltcs_l1, test_l1, loss_history):
    schedule_audit = {"arms": {}}
    quality = {}
    values = {
        "nltcs": {"drift": nltcs_drift, "l1": nltcs_l1},
        "test_300x10": {"drift": 2.0, "l1": test_l1},
    }
    for name in v1.DATASETS:
        schedule_audit["arms"][name] = {
            "residual": {
                "best_loss_diagnostic_only": 100.0,
                "output_squared_loss": 100.0 * values[name]["drift"],
            },
        }
        quality[name] = {
            "residual": {
                "measured": {"normalized_l1_mean": values[name]["l1"]},
            },
        }
    return schedule_audit, quality, loss_history


DESCENDING_LOSS = [float(9000 - t) for t in range(9000)]
PLATEAU_LOSS = [float(9000 - t) for t in range(8000)] + [2000.0] * 1000


def test_verdict_quality_gate_failure_dominates():
    evaluation = v4._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.000198,
        test_l1=0.002,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == (
        "quality_regression_under_lower_floor"
    )
    assert evaluation["quality_gate"]["pass"] is False

    evaluation = v4._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.00015,
        test_l1=0.002641,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == (
        "quality_regression_under_lower_floor"
    )


def test_verdict_supported_when_drift_and_quality_pass():
    evaluation = v4._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "lower_floor_supported"
    assert evaluation["primary_drift"]["pass"] is True
    assert evaluation["formal_claim_allowed"] is False


def test_verdict_budget_insufficient_when_descending():
    evaluation = v4._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.15,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == "lower_floor_budget_insufficient"
    assert evaluation["nltcs_floor_morphology"]["descending"] is True


def test_verdict_rejected_when_not_descending():
    evaluation = v4._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.15,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "lower_floor_rejected"
    assert evaluation["nltcs_floor_morphology"]["descending"] is False


def test_verdict_boundary_values_use_frozen_thresholds():
    evaluation = v4._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.100,
        nltcs_l1=0.000197,
        test_l1=0.002640,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "lower_floor_supported"
