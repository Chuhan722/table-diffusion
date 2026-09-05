"""Tests for the fitness-only T=6000 truncation-hypothesis v3 delta runner."""

import dataclasses
import json
from pathlib import Path

import pytest

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2
from scripts import run_fitness_only_schedule_v3_t6000 as v3


def test_plan_is_frozen_and_does_not_read_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(v1, "_sha256_file", forbidden)
    monkeypatch.setattr(v1, "_load_json_object", forbidden)
    plan = v3.build_plan()

    assert plan["protocol_sha256"] == v3.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 4
    assert plan["generation_started"] is False
    assert plan["output_dir"] == str(v3.OUTPUT_DIR)
    assert plan["protocol"]["inherits"]["base_protocol_sha256"] == (
        v2.FROZEN_PROTOCOL_SHA256
    )
    assert plan["protocol"]["inherits"]["v1_protocol_sha256"] == (
        v1.FROZEN_PROTOCOL_SHA256
    )
    hypothesis = plan["protocol"]["truncation_hypothesis"]
    assert hypothesis["only_change"] == {
        "n_rounds": {"from": 3000, "to": 6000},
    }
    assert hypothesis["schedule_constants_unchanged"] is True
    assert hypothesis["v2_unsupported_record_stands"] is True


def test_protocol_identity_is_fail_closed():
    assert v3.protocol_sha256() == v3.FROZEN_PROTOCOL_SHA256
    assert v3.assert_frozen_protocol_identity() == v3.FROZEN_PROTOCOL_SHA256


def test_wrong_confirmation_fails_before_any_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        v3,
        "_verify_v2_reference_artifacts",
        lambda *_args, **_kwargs: called.append("v2-audit"),
    )
    monkeypatch.setattr(
        v3,
        "_source_snapshot",
        lambda *_args, **_kwargs: called.append("source"),
    )

    with pytest.raises(ValueError, match="protocol SHA-256"):
        v3.run("not-the-frozen-hash")
    assert called == []


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "already-there"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(v3, "OUTPUT_DIR", Path("already-there"))

    with pytest.raises(FileExistsError, match="不覆盖"):
        v3.run(v3.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_manifest_inherits_v2_exactly_except_rounds():
    v2_manifest = v2._json_protocol_manifest()
    v3_manifest = v3._json_protocol_manifest()

    assert v2_manifest["n_rounds"] == 3000
    assert v3_manifest["n_rounds"] == 6000
    # generation_config 与时间表定义逐键继承，无任何改动。
    assert v3_manifest["generation_config"] == v2_manifest["generation_config"]
    assert v3_manifest["rho_schedule"] == v2_manifest["rho_schedule"]
    assert v3_manifest["datasets"] == v2_manifest["datasets"]
    assert v3_manifest["seed"] == v2_manifest["seed"]


def test_fitness_config_delta_is_only_the_round_budget():
    for spec in v1.DATASETS.values():
        base = dataclasses.asdict(v2._fitness_config(spec))
        delta = dataclasses.asdict(v3._fitness_config(spec))
        changed = {key for key in base if base[key] != delta[key]}
        assert changed == {"n_rounds"}
        assert base["n_rounds"] == 3000
        assert delta["n_rounds"] == 6000
        assert delta["rho_anneal_start_round"] == 900
        assert delta["rho_anneal_rounds"] == 600
        assert delta["rho_anneal_end"] == 0.001


def test_expected_schedule_prefix_matches_v2_and_floor_extends():
    schedule_v3 = v2._expected_schedule(v3.N_ROUNDS)
    schedule_v2 = v2._expected_schedule(v1.N_ROUNDS)

    assert len(schedule_v3) == 6000
    assert schedule_v3[:3000] == schedule_v2
    floor = schedule_v3[1500]
    assert floor == pytest.approx(0.001)
    assert all(value == floor for value in schedule_v3[1500:])
    assert schedule_v3[0] == 0.01
    assert schedule_v3[899] == 0.01


def test_audit_schedule_rejects_tampered_or_short_history(tmp_path):
    dataset = next(iter(v1.DATASETS))
    arm_dir = tmp_path / "generation" / dataset
    arm_dir.mkdir(parents=True)

    tampered = v2._expected_schedule(v3.N_ROUNDS)
    tampered[4200] *= 1.5
    (arm_dir / f"{v1.ARMS[0]}_diagnostics.json").write_text(
        json.dumps({
            "rho_schedule_history": tampered,
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v3._audit_schedule(tmp_path)

    # v2 长度（3000 轮）的历史在 v3 审计下同样必须被拒。
    (arm_dir / f"{v1.ARMS[0]}_diagnostics.json").write_text(
        json.dumps({
            "rho_schedule_history": v2._expected_schedule(v1.N_ROUNDS),
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="偏离预冻结时间表"):
        v3._audit_schedule(tmp_path)


def test_v2_reference_artifacts_are_hash_verified(tmp_path):
    base = tmp_path / v3.V2_OUTPUT_DIR
    with pytest.raises(RuntimeError, match="缺失"):
        v3._verify_v2_reference_artifacts(tmp_path)

    for rel_path in v3.V2_REFERENCE_ARTIFACT_SHA256:
        path = base / rel_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{\"forged\": true}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="身份漂移"):
        v3._verify_v2_reference_artifacts(tmp_path)


def _diag(loss, rho, init="init-hash", rng="rng-hash"):
    return {
        "loss_history": list(loss),
        "rho_schedule_history": list(rho),
        "initial_table_sha256": init,
        "primary_rng_post_initialization_state_sha256": rng,
    }


def _paired_diagnostics():
    v2_loss = [float(3000 - t) for t in range(3000)]
    v2_rho = v2._expected_schedule(v1.N_ROUNDS)
    v3_loss = v2_loss + [1.0] * 3000
    v3_rho = v2._expected_schedule(v3.N_ROUNDS)
    old = {
        name: {arm: _diag(v2_loss, v2_rho) for arm in v1.ARMS}
        for name in v1.DATASETS
    }
    new = {
        name: {arm: _diag(v3_loss, v3_rho) for arm in v1.ARMS}
        for name in v1.DATASETS
    }
    return new, old


def test_prefix_audit_passes_on_identical_prefix():
    new, old = _paired_diagnostics()
    audit = v3._prefix_audit(new, old)
    assert audit["passed"] is True
    assert audit["failure_label"] is None
    for name in v1.DATASETS:
        for arm in v1.ARMS:
            assert audit["arms"][name][arm]["passed"] is True


def test_prefix_audit_rejects_any_forged_prefix():
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][137] += 0.5
    audit = v3._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["failure_label"] == "horizon_invariance_violated"
    detail = audit["arms"]["nltcs"]["residual"]
    assert detail["loss_history_prefix_identical"] is False
    assert detail["loss_history_first_mismatch_round"] == 137
    # 其余臂不受牵连（配对细节仍可读）。
    assert audit["arms"]["nltcs"]["equal"]["passed"] is True

    new, old = _paired_diagnostics()
    new["test_300x10"]["equal"]["rho_schedule_history"][2999] *= 2.0
    audit = v3._prefix_audit(new, old)
    assert audit["passed"] is False
    detail = audit["arms"]["test_300x10"]["equal"]
    assert detail["rho_schedule_history_prefix_identical"] is False
    assert detail["rho_schedule_first_mismatch_round"] == 2999

    new, old = _paired_diagnostics()
    new["nltcs"]["equal"]["initial_table_sha256"] = "other"
    audit = v3._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["nltcs"]["equal"][
        "initial_table_sha256_identical"
    ] is False

    new, old = _paired_diagnostics()
    new["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_sha256"
    ] = "other"
    audit = v3._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_identical"
    ] is False


def test_floor_morphology_descending_rule():
    # 单调下降：尾窗均值低于前窗且 best 刷新持续到最后 → descending。
    descending = v3._floor_morphology(
        [float(6000 - t) for t in range(6000)]
    )
    assert descending["descending"] is True
    assert descending["last_running_best_update_round"] == 5999
    assert descending["tail_below_prev"] is True

    # 尾段横盘（均值不再下降）→ 非 descending。
    flat_tail = [float(6000 - t) for t in range(5000)] + [2000.0] * 1000
    plateau = v3._floor_morphology(flat_tail)
    assert plateau["descending"] is False
    assert plateau["tail_below_prev"] is False

    # 尾窗均值更低但 best 早已不刷新（全局最低在 t=0）→ 非 descending。
    early_best = [1.0] + [200.0] * 4999 + [100.0] * 1000
    stale = v3._floor_morphology(early_best)
    assert stale["tail_below_prev"] is True
    assert stale["last_running_best_update_round"] == 0
    assert stale["descending"] is False


def _fake_verdict_inputs(nltcs_drift, nltcs_l1, test_l1, loss_history):
    schedule_audit = {"arms": {}}
    quality = {}
    values = {
        "nltcs": {"drift": nltcs_drift, "l1": nltcs_l1},
        "test_300x10": {"drift": 1.5, "l1": test_l1},
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


DESCENDING_LOSS = [float(6000 - t) for t in range(6000)]
PLATEAU_LOSS = [float(6000 - t) for t in range(5000)] + [2000.0] * 1000


def test_verdict_quality_gate_failure_dominates():
    evaluation = v3._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.000201,
        test_l1=0.002,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == (
        "quality_regression_under_extended_budget"
    )
    assert evaluation["quality_gate"]["pass"] is False
    assert evaluation["quality_gate"]["per_dataset"]["nltcs"]["pass"] is False

    evaluation = v3._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.00015,
        test_l1=0.002787,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == (
        "quality_regression_under_extended_budget"
    )


def test_verdict_supported_when_drift_and_quality_pass():
    evaluation = v3._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.05,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "truncation_hypothesis_supported"
    assert evaluation["primary_drift"]["pass"] is True
    assert evaluation["formal_claim_allowed"] is False


def test_verdict_budget_still_insufficient_when_descending():
    evaluation = v3._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.2,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=DESCENDING_LOSS,
    ))
    assert evaluation["primary_verdict"] == "budget_still_insufficient"
    assert evaluation["primary_drift"]["pass"] is False
    assert evaluation["nltcs_floor_morphology"]["descending"] is True


def test_verdict_rejected_when_not_descending():
    evaluation = v3._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.2,
        nltcs_l1=0.00015,
        test_l1=0.002,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "truncation_hypothesis_rejected"
    assert evaluation["nltcs_floor_morphology"]["descending"] is False


def test_verdict_boundary_values_use_frozen_thresholds():
    # drift 恰在 1.100 → 通过；L1 恰在阈值 → 通过（≤ 为闭区间）。
    evaluation = v3._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_drift=1.100,
        nltcs_l1=0.000200,
        test_l1=0.002786,
        loss_history=PLATEAU_LOSS,
    ))
    assert evaluation["primary_verdict"] == "truncation_hypothesis_supported"
