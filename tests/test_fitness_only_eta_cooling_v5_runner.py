"""Tests for the fitness-only floor-phase eta-cooling v5 delta runner."""

import dataclasses
import json
from pathlib import Path

import pytest

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2
from scripts import run_fitness_only_schedule_v3_t6000 as v3
from scripts import run_fitness_only_floor00005_v4 as v4
from scripts import run_fitness_only_eta_cooling_v5 as v5


def test_plan_is_frozen_and_does_not_read_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(v1, "_sha256_file", forbidden)
    monkeypatch.setattr(v1, "_load_json_object", forbidden)
    plan = v5.build_plan()

    assert plan["protocol_sha256"] == v5.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 4
    assert plan["generation_started"] is False
    assert plan["output_dir"] == str(v5.OUTPUT_DIR)
    assert plan["protocol"]["inherits"]["base_protocol_sha256"] == (
        v3.FROZEN_PROTOCOL_SHA256
    )
    assert plan["protocol"]["inherits"]["v4_protocol_sha256"] == (
        v4.FROZEN_PROTOCOL_SHA256
    )
    hypothesis = plan["protocol"]["jump_magnitude_hypothesis"]
    assert hypothesis["changes"]["n_rounds"] == {"from": 6000, "to": 9000}
    assert hypothesis["rho_schedule_v3_verbatim"] is True
    assert "truncation_hypothesis" not in plan["protocol"]
    assert "lower_floor_hypothesis" not in plan["protocol"]
    # rho 时间表必须是 v3 原样（floor 0.001，不是 v4 的 0.0005）。
    rho_schedule = plan["protocol"]["rho_schedule"]
    assert rho_schedule["floor"] == 0.001
    assert rho_schedule["hold_rounds_H"] == 900
    assert rho_schedule["descent_rounds_D"] == 600
    eta_schedule = plan["protocol"]["eta_schedule"]
    assert eta_schedule["eta0"] == 0.5
    assert eta_schedule["hold_rounds_H_eta"] == 1500
    assert eta_schedule["descent_rounds_D_eta"] == 600
    assert eta_schedule["floor"] == 0.25
    assert eta_schedule["staggered_after_rho_floor_landing"] is True


def test_protocol_identity_is_fail_closed():
    assert v5.protocol_sha256() == v5.FROZEN_PROTOCOL_SHA256
    assert v5.assert_frozen_protocol_identity() == v5.FROZEN_PROTOCOL_SHA256


def test_wrong_confirmation_fails_before_any_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        v4,
        "_verify_v3_reference_artifacts",
        lambda *_args, **_kwargs: called.append("v3-audit"),
    )
    monkeypatch.setattr(
        v5,
        "_source_snapshot",
        lambda *_args, **_kwargs: called.append("source"),
    )

    with pytest.raises(ValueError, match="protocol SHA-256"):
        v5.run("not-the-frozen-hash")
    assert called == []


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "already-there"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(v5, "OUTPUT_DIR", Path("already-there"))

    with pytest.raises(FileExistsError, match="不覆盖"):
        v5.run(v5.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_fitness_config_delta_is_eta_schedule_and_budget_only():
    for spec in v1.DATASETS.values():
        base = dataclasses.asdict(v3._fitness_config(spec))
        delta = dataclasses.asdict(v5._fitness_config(spec))
        changed = {key for key in base if base[key] != delta[key]}
        assert changed == {
            "eta_anneal_start_round",
            "eta_anneal_rounds",
            "eta_anneal_end",
            "n_rounds",
        }
        # rho 时间表 v3 原样：floor 0.001（不是 v4 的 0.0005）。
        assert delta["rho_anneal_end"] == 0.001
        assert delta["rho_anneal_start_round"] == 900
        assert delta["rho_anneal_rounds"] == 600
        assert base["eta_anneal_end"] is None
        assert delta["eta_anneal_start_round"] == 1500
        assert delta["eta_anneal_rounds"] == 600
        assert delta["eta_anneal_end"] == 0.25
        assert delta["eta"] == 0.5
        assert base["n_rounds"] == 6000
        assert delta["n_rounds"] == 9000


def test_manifest_inherits_v3_except_eta_schedule_and_rounds():
    v3_manifest = v3._json_protocol_manifest()
    v5_manifest = v5._json_protocol_manifest()

    assert v3_manifest["n_rounds"] == 6000
    assert v5_manifest["n_rounds"] == 9000
    v3_config = dict(v3_manifest["generation_config"])
    v5_config = dict(v5_manifest["generation_config"])
    assert v5_config.pop("eta_anneal_start_round") == 1500
    assert v5_config.pop("eta_anneal_rounds") == 600
    assert v5_config.pop("eta_anneal_end") == 0.25
    assert v5_config == v3_config
    assert v5_config["rho_anneal_end"] == 0.001
    assert v5_manifest["rho_schedule"] == v3_manifest["rho_schedule"]
    assert v5_manifest["datasets"] == v3_manifest["datasets"]
    assert v5_manifest["seed"] == v3_manifest["seed"]


def test_expected_rho_schedule_is_v3_verbatim_extended_to_9000():
    schedule_v5 = v5._expected_rho_schedule(v5.N_ROUNDS)
    schedule_v3 = v2._expected_schedule(v3.N_ROUNDS)

    assert len(schedule_v5) == 9000
    # 与 v3 的 6000 轮逐位一致；[6000,9000) 恒为 0.001 地板延伸。
    assert schedule_v5[:6000] == schedule_v3
    floor = schedule_v5[1500]
    assert floor == pytest.approx(0.001)
    assert all(value == floor for value in schedule_v5[1500:])
    assert schedule_v5[0] == 0.01
    assert schedule_v5[900] == 0.01


def test_expected_eta_schedule_boundaries():
    schedule = v5._expected_eta_schedule(v5.N_ROUNDS)

    assert len(schedule) == 9000
    # t∈[0,1500] 恒为 0.5（t=1500 progress 恰为 0，浮点精确）。
    assert schedule[:1501] == [0.5] * 1501
    assert schedule[1500] == 0.5
    assert schedule[1501] != 0.5
    assert schedule[1501] == pytest.approx(0.5 * 0.5 ** (1 / 600))
    assert schedule[2099] == pytest.approx(0.5 * 0.5 ** (599 / 600))
    floor = schedule[2100]
    assert floor == pytest.approx(0.25)
    assert all(value == floor for value in schedule[2100:])


def _write_diag(path, rho_history, eta_history):
    path.write_text(
        json.dumps({
            "rho_schedule_history": rho_history,
            "eta_schedule_history": eta_history,
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )


def test_audit_schedule_rejects_tampered_rho_or_eta_history(tmp_path):
    dataset = next(iter(v1.DATASETS))
    arm_dir = tmp_path / "generation" / dataset
    arm_dir.mkdir(parents=True)
    diag_path = arm_dir / f"{v1.ARMS[0]}_diagnostics.json"

    good_rho = v5._expected_rho_schedule(v5.N_ROUNDS)
    good_eta = v5._expected_eta_schedule(v5.N_ROUNDS)

    # rho 被篡改（例如错用了 v4 的 0.0005 地板）必须被拒。
    v4_rho = v4._expected_schedule(v4.N_ROUNDS)
    _write_diag(diag_path, v4_rho, list(good_eta))
    with pytest.raises(RuntimeError, match="偏离 v3 原样时间表"):
        v5._audit_schedule(tmp_path)

    # eta 被篡改必须被拒。
    tampered_eta = list(good_eta)
    tampered_eta[5000] *= 1.5
    _write_diag(diag_path, list(good_rho), tampered_eta)
    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v5._audit_schedule(tmp_path)

    # eta 恒定 0.5（未启用退火的旧行为）也必须被拒。
    _write_diag(diag_path, list(good_rho), [0.5] * v5.N_ROUNDS)
    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v5._audit_schedule(tmp_path)


def test_v3_reference_constants_are_shared_with_v4():
    assert v5.V3_REFERENCE_ARTIFACT_SHA256 is v4.V3_REFERENCE_ARTIFACT_SHA256
    assert v5.V3_REFERENCE is v4.V3_REFERENCE
    assert v5.V3_OUTPUT_DIR == v3.OUTPUT_DIR


def _diag(loss, rho, init="init-hash", rng="rng-hash"):
    return {
        "loss_history": list(loss),
        "rho_schedule_history": list(rho),
        "initial_table_sha256": init,
        "primary_rng_post_initialization_state_sha256": rng,
    }


def _paired_diagnostics():
    shared_loss = [float(9000 - t) for t in range(1501)]
    v3_loss = shared_loss + [1.0] * (6000 - 1501)
    v5_loss = shared_loss + [2.0] * (9000 - 1501)
    v3_rho = v2._expected_schedule(v3.N_ROUNDS)
    v5_rho = v5._expected_rho_schedule(v5.N_ROUNDS)
    old = {
        name: {arm: _diag(v3_loss, v3_rho) for arm in v1.ARMS}
        for name in v1.DATASETS
    }
    new = {
        name: {arm: _diag(v5_loss, v5_rho) for arm in v1.ARMS}
        for name in v1.DATASETS
    }
    return new, old


def test_prefix_audit_passes_on_identical_1501_prefix():
    new, old = _paired_diagnostics()
    audit = v5._prefix_audit(new, old)
    assert audit["prefix_rounds"] == 1501
    assert audit["passed"] is True
    assert audit["failure_label"] is None
    for name in v1.DATASETS:
        for arm in v1.ARMS:
            assert audit["arms"][name][arm]["passed"] is True


def test_prefix_audit_rejects_any_forged_prefix():
    # 前缀最后一轮（t=1500）被篡改必须被抓住。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][1500] += 0.5
    audit = v5._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["failure_label"] == "schedule_blindness_violated"
    detail = audit["arms"]["nltcs"]["residual"]
    assert detail["loss_history_prefix_identical"] is False
    assert detail["loss_history_first_mismatch_round"] == 1500
    assert audit["arms"]["nltcs"]["equal"]["passed"] is True

    # 前缀之外（t=1501）的差异不构成违规。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][1501] += 123.0
    audit = v5._prefix_audit(new, old)
    assert audit["passed"] is True

    new, old = _paired_diagnostics()
    new["test_300x10"]["equal"]["rho_schedule_history"][0] *= 2.0
    audit = v5._prefix_audit(new, old)
    assert audit["passed"] is False
    detail = audit["arms"]["test_300x10"]["equal"]
    assert detail["rho_schedule_history_prefix_identical"] is False
    assert detail["rho_schedule_first_mismatch_round"] == 0

    new, old = _paired_diagnostics()
    new["nltcs"]["equal"]["initial_table_sha256"] = "other"
    audit = v5._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["nltcs"]["equal"][
        "initial_table_sha256_identical"
    ] is False

    new, old = _paired_diagnostics()
    new["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_sha256"
    ] = "other"
    audit = v5._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_identical"
    ] is False


def test_floor_morphology_descending_rule_uses_v5_windows():
    descending = v5._floor_morphology(
        [float(9000 - t) for t in range(9000)]
    )
    assert descending["descending"] is True
    assert descending["last_running_best_update_round"] == 8999
    assert descending["tail_window"] == [8000, 9000]
    assert descending["prev_window"] == [7000, 8000]
    assert descending["eta_floor_start_round"] == 2100
    segments = descending["observation_floor_segment_means"]
    assert segments[0]["rounds"] == [2100, 2600]

    flat_tail = [float(9000 - t) for t in range(8000)] + [2000.0] * 1000
    plateau = v5._floor_morphology(flat_tail)
    assert plateau["descending"] is False
    assert plateau["tail_below_prev"] is False

    early_best = [1.0] + [200.0] * 7999 + [100.0] * 1000
    stale = v5._floor_morphology(early_best)
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
    evaluation = v5._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.000198,
        test_l1=0.002,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == (
        "quality_regression_under_eta_cooling"
    )
    assert evaluation["quality_gate"]["pass"] is False

    evaluation = v5._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.00015,
        test_l1=0.002641,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == (
        "quality_regression_under_eta_cooling"
    )


def test_verdict_supported_when_drift_and_quality_pass():
    evaluation = v5._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "eta_cooling_supported"
    assert evaluation["primary_drift"]["pass"] is True
    assert evaluation["formal_claim_allowed"] is False


def test_verdict_budget_insufficient_when_descending():
    evaluation = v5._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.15,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == "eta_cooling_budget_insufficient"
    assert evaluation["nltcs_floor_morphology"]["descending"] is True


def test_verdict_rejected_when_not_descending():
    evaluation = v5._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.15,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "eta_cooling_rejected"
    assert evaluation["nltcs_floor_morphology"]["descending"] is False


def test_verdict_boundary_values_use_frozen_thresholds():
    evaluation = v5._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.100,
        nltcs_l1=0.000197,
        test_l1=0.002640,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "eta_cooling_supported"


def test_baseline_v4_record_matches_frozen_v4_report_values():
    manifest = v5._json_protocol_manifest()
    record = manifest["baseline_v4_record"]
    assert record["verdict"] == "quality_regression_under_lower_floor"
    assert record["nltcs_drift_ratio"] == 1.1753874591212854
    assert record["nltcs_measured_normalized_l1_mean"] == (
        0.0001647817604804194
    )
    assert record["test_measured_normalized_l1_mean"] == (
        0.0027333333333333333
    )
    assert record["test_drift_ratio"] == 3.1363636363636362
    assert record["protocol_sha256"] == v4.FROZEN_PROTOCOL_SHA256
