"""Tests for the fitness-only multiplicative-weights v6 delta runner."""

import dataclasses
import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2
from scripts import run_fitness_only_schedule_v3_t6000 as v3
from scripts import run_fitness_only_floor00005_v4 as v4
from scripts import run_fitness_only_eta_cooling_v5 as v5
from scripts import run_fitness_only_mw_v6 as v6


def test_plan_is_frozen_and_does_not_read_inputs(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("plan 不得读取输入或结果")

    monkeypatch.setattr(v1, "_sha256_file", forbidden)
    monkeypatch.setattr(v1, "_load_json_object", forbidden)
    plan = v6.build_plan()

    assert plan["protocol_sha256"] == v6.FROZEN_PROTOCOL_SHA256
    assert plan["trajectory_count"] == 2
    assert plan["arms_run"] == ["residual"]
    assert plan["generation_started"] is False
    assert plan["output_dir"] == str(v6.OUTPUT_DIR)
    assert plan["protocol"]["inherits"]["base_protocol_sha256"] == (
        v3.FROZEN_PROTOCOL_SHA256
    )
    assert plan["protocol"]["inherits"]["v5_protocol_sha256"] == (
        v5.FROZEN_PROTOCOL_SHA256
    )
    hypothesis = plan["protocol"]["mw_aggregation_hypothesis"]
    assert hypothesis["rho_eta_schedules_v3_verbatim"] is True
    assert hypothesis["n_rounds_unchanged_same_budget_comparison"] is True
    assert "truncation_hypothesis" not in plan["protocol"]
    # rho 时间表 v3 原样；eta 无退火（generation_config 无 eta anneal 值）。
    rho_schedule = plan["protocol"]["rho_schedule"]
    assert rho_schedule["floor"] == 0.001
    assert rho_schedule["hold_rounds_H"] == 900
    assert rho_schedule["descent_rounds_D"] == 600
    config = plan["protocol"]["generation_config"]
    assert config["mw_query_weight_eta"] == 0.002
    assert config["mw_signal_cap"] == 8.0
    assert config["mw_weight_cap"] == 8.0
    assert config["mw_start_round"] == 1500
    assert plan["protocol"]["n_rounds"] == 6000
    single_arm = plan["protocol"]["single_arm_design"]
    assert single_arm["arms_run"] == ["residual"]


def test_protocol_identity_is_fail_closed():
    assert v6.protocol_sha256() == v6.FROZEN_PROTOCOL_SHA256
    assert v6.assert_frozen_protocol_identity() == v6.FROZEN_PROTOCOL_SHA256


def test_wrong_confirmation_fails_before_any_generation(monkeypatch, tmp_path):
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    called = []
    monkeypatch.setattr(
        v4,
        "_verify_v3_reference_artifacts",
        lambda *_args, **_kwargs: called.append("v3-audit"),
    )
    monkeypatch.setattr(
        v6,
        "_source_snapshot",
        lambda *_args, **_kwargs: called.append("source"),
    )

    with pytest.raises(ValueError, match="protocol SHA-256"):
        v6.run("not-the-frozen-hash")
    assert called == []


def test_existing_output_is_never_overwritten(monkeypatch, tmp_path):
    destination = tmp_path / "already-there"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(v1, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(v6, "OUTPUT_DIR", Path("already-there"))

    with pytest.raises(FileExistsError, match="不覆盖"):
        v6.run(v6.FROZEN_PROTOCOL_SHA256)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_fitness_config_delta_is_mw_constants_only():
    for spec in v1.DATASETS.values():
        base = dataclasses.asdict(v3._fitness_config(spec))
        delta = dataclasses.asdict(v6._fitness_config(spec))
        changed = {key for key in base if base[key] != delta[key]}
        assert changed == {
            "mw_query_weight_eta",
            "mw_signal_cap",
            "mw_weight_cap",
            "mw_start_round",
        }
        # v3 全部常数逐字保留：rho 三段式、eta 恒 0.5、同预算 6000 轮。
        assert delta["rho_anneal_end"] == 0.001
        assert delta["rho_anneal_start_round"] == 900
        assert delta["rho_anneal_rounds"] == 600
        assert delta["eta"] == 0.5
        assert delta["eta_anneal_end"] is None
        assert delta["n_rounds"] == 6000
        assert base["mw_query_weight_eta"] is None
        assert delta["mw_query_weight_eta"] == 0.002
        assert delta["mw_signal_cap"] == 8.0
        assert delta["mw_weight_cap"] == 8.0
        assert delta["mw_start_round"] == 1500


def test_manifest_inherits_v3_except_mw_constants():
    v3_manifest = v3._json_protocol_manifest()
    v6_manifest = v6._json_protocol_manifest()

    assert v6_manifest["n_rounds"] == v3_manifest["n_rounds"] == 6000
    v3_config = dict(v3_manifest["generation_config"])
    v6_config = dict(v6_manifest["generation_config"])
    assert v6_config.pop("mw_query_weight_eta") == 0.002
    assert v6_config.pop("mw_signal_cap") == 8.0
    assert v6_config.pop("mw_weight_cap") == 8.0
    assert v6_config.pop("mw_start_round") == 1500
    assert v6_config == v3_config
    assert v6_manifest["rho_schedule"] == v3_manifest["rho_schedule"]
    assert v6_manifest["datasets"] == v3_manifest["datasets"]
    assert v6_manifest["seed"] == v3_manifest["seed"]


def test_v3_reference_constants_are_shared_with_v4():
    assert v6.V3_REFERENCE_ARTIFACT_SHA256 is v4.V3_REFERENCE_ARTIFACT_SHA256
    assert v6.V3_REFERENCE is v4.V3_REFERENCE
    assert v6.V3_OUTPUT_DIR == v3.OUTPUT_DIR


def _stats_row(round_index, max_=1.0, min_=1.0, mean=1.0, upper=0, lower=0):
    return {
        "round": round_index,
        "max": max_,
        "min": min_,
        "mean": mean,
        "at_upper_cap": upper,
        "at_lower_cap": lower,
    }


def _good_stats():
    rows = [_stats_row(t) for t in range(v6.MW_START_ROUND)]
    rows += [
        _stats_row(t, max_=2.0, min_=0.5, mean=0.98)
        for t in range(v6.MW_START_ROUND, v6.N_ROUNDS)
    ]
    return rows


def _write_diag(path, rho_history, eta_history, stats, snapshots=None):
    if snapshots is None:
        snapshots = [{"round": 0, "weights": [1.0]}]
    path.write_text(
        json.dumps({
            "rho_schedule_history": rho_history,
            "eta_schedule_history": eta_history,
            "mw_weight_stats_history": stats,
            "mw_weight_snapshot_history": snapshots,
            "output_squared_loss": 1.0,
            "best_loss_diagnostic_only": 1.0,
        }),
        encoding="utf-8",
    )


def test_audit_schedule_rejects_tampered_schedules_or_weights(tmp_path):
    dataset = next(iter(v1.DATASETS))
    arm_dir = tmp_path / "generation" / dataset
    arm_dir.mkdir(parents=True)
    diag_path = arm_dir / "residual_diagnostics.json"

    good_rho = v5._expected_rho_schedule(v6.N_ROUNDS)
    good_eta = [0.5] * v6.N_ROUNDS

    # rho 被篡改（错用 v4 的 0.0005 地板）必须被拒。
    v4_rho = v4._expected_schedule(v6.N_ROUNDS)
    _write_diag(diag_path, v4_rho, list(good_eta), _good_stats())
    with pytest.raises(RuntimeError, match="偏离 v3 原样时间表"):
        v6._audit_schedule(tmp_path)

    # eta 被篡改（错带 v5 的降温）必须被拒。
    tampered_eta = list(good_eta)
    tampered_eta[3000] = 0.25
    _write_diag(diag_path, list(good_rho), tampered_eta, _good_stats())
    with pytest.raises(RuntimeError, match="恒 0.5"):
        v6._audit_schedule(tmp_path)

    # 保温段权重非平凡必须被拒（MW 提前开启=前缀合同被破坏）。
    early_stats = _good_stats()
    early_stats[100] = _stats_row(100, max_=1.5, mean=1.0)
    _write_diag(diag_path, list(good_rho), list(good_eta), early_stats)
    with pytest.raises(RuntimeError, match="保温段合同被破坏"):
        v6._audit_schedule(tmp_path)

    # 越出 [1/cap, cap] 围栏必须被拒。
    outlaw_stats = _good_stats()
    outlaw_stats[3000] = _stats_row(3000, max_=9.0, min_=0.5, mean=1.2)
    _write_diag(diag_path, list(good_rho), list(good_eta), outlaw_stats)
    with pytest.raises(RuntimeError, match="围栏"):
        v6._audit_schedule(tmp_path)

    # 长度错误必须被拒。
    _write_diag(
        diag_path, list(good_rho), list(good_eta), _good_stats()[:-1],
    )
    with pytest.raises(RuntimeError, match="长度异常"):
        v6._audit_schedule(tmp_path)


def _diag(loss, rho, state, stats, init="init-hash", rng="rng-hash"):
    return {
        "loss_history": list(loss),
        "rho_schedule_history": list(rho),
        "current_state_metrics_history": list(state),
        "mw_weight_stats_history": stats,
        "initial_table_sha256": init,
        "primary_rng_post_initialization_state_sha256": rng,
    }


def _paired_diagnostics():
    shared_loss = [float(9000 - t) for t in range(1501)]
    v3_loss = shared_loss + [1.0] * (6000 - 1501)
    v6_loss = shared_loss + [2.0] * (6000 - 1501)
    shared_state = [{"round": -1, "phase": "initial"}] + [
        {"round": t, "loss": loss} for t, loss in enumerate(shared_loss)
    ]
    v3_state = shared_state + [
        {"round": t, "loss": 1.0} for t in range(1501, 6000)
    ]
    v6_state = shared_state + [
        {"round": t, "loss": 2.0} for t in range(1501, 6000)
    ]
    rho = v2._expected_schedule(v3.N_ROUNDS)
    old = {
        name: {
            "residual": _diag(v3_loss, rho, v3_state, None),
        }
        for name in v1.DATASETS
    }
    new = {
        name: {
            "residual": _diag(v6_loss, rho, v6_state, _good_stats()),
        }
        for name in v1.DATASETS
    }
    return new, old


def test_prefix_audit_passes_on_identical_1501_prefix():
    new, old = _paired_diagnostics()
    audit = v6._prefix_audit(new, old)
    assert audit["prefix_rounds"] == 1501
    assert audit["passed"] is True
    assert audit["failure_label"] is None
    for name in v1.DATASETS:
        assert audit["arms"][name]["residual"]["passed"] is True
        assert audit["arms"][name]["residual"][
            "mw_weights_all_ones_before_start_round"
        ] is True


def test_prefix_audit_rejects_any_forged_prefix():
    # 前缀最后一轮（t=1500）被篡改必须被抓住。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][1500] += 0.5
    audit = v6._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["failure_label"] == "schedule_blindness_violated"
    detail = audit["arms"]["nltcs"]["residual"]
    assert detail["loss_history_prefix_identical"] is False
    assert detail["loss_history_first_mismatch_round"] == 1500

    # 前缀之外（t=1501）的差异不构成违规。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["loss_history"][1501] += 123.0
    audit = v6._prefix_audit(new, old)
    assert audit["passed"] is True

    # current_state_metrics_history 前缀被篡改必须被抓住。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["current_state_metrics_history"][10] = {
        "round": 9, "loss": -1.0,
    }
    audit = v6._prefix_audit(new, old)
    assert audit["passed"] is False
    detail = audit["arms"]["nltcs"]["residual"]
    assert detail[
        "current_state_metrics_history_prefix_identical"
    ] is False
    assert detail["current_state_metrics_first_mismatch_entry"] == 10

    new, old = _paired_diagnostics()
    new["test_300x10"]["residual"]["initial_table_sha256"] = "other"
    audit = v6._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["test_300x10"]["residual"][
        "initial_table_sha256_identical"
    ] is False

    new, old = _paired_diagnostics()
    new["test_300x10"]["residual"][
        "primary_rng_post_initialization_state_sha256"
    ] = "other"
    audit = v6._prefix_audit(new, old)
    assert audit["passed"] is False

    # 保温段权重非平凡也是前缀违规。
    new, old = _paired_diagnostics()
    new["nltcs"]["residual"]["mw_weight_stats_history"][100] = _stats_row(
        100, max_=1.5,
    )
    audit = v6._prefix_audit(new, old)
    assert audit["passed"] is False
    assert audit["arms"]["nltcs"]["residual"][
        "mw_weights_all_ones_before_start_round"
    ] is False


def test_stage_v3_equal_tables_verifies_frame_sha(monkeypatch, tmp_path):
    frame = pd.DataFrame({"a": ["0", "1"], "b": ["1", "0"]})
    for name in v1.DATASETS:
        source_dir = tmp_path / v6.V3_OUTPUT_DIR / "generation" / name
        source_dir.mkdir(parents=True)
        frame.to_csv(source_dir / "equal_terminal_current.csv", index=False)
        staging_dir = tmp_path / "staging" / "generation" / name
        staging_dir.mkdir(parents=True)
    good_sha = v1._frame_sha256(pd.read_csv(
        tmp_path / v6.V3_OUTPUT_DIR / "generation"
        / next(iter(v1.DATASETS)) / "equal_terminal_current.csv"
    ))
    v3_report = {
        "generation": {
            name: {"arms": {"equal": {"terminal_table_sha256": good_sha}}}
            for name in v1.DATASETS
        },
    }

    staged = v6._stage_v3_equal_tables(
        tmp_path, tmp_path / "staging", v3_report,
    )
    for name in v1.DATASETS:
        assert staged[name]["terminal_table_sha256"] == good_sha
        assert staged[name]["reused_not_rerun"] is True
        copied = (
            tmp_path / "staging" / "generation" / name
            / "equal_terminal_current.csv"
        )
        assert copied.is_file()

    # 表内容被篡改（frame SHA 与 v3 report 不符）必须被拒。
    bad_report = {
        "generation": {
            name: {
                "arms": {"equal": {"terminal_table_sha256": "not-this"}},
            }
            for name in v1.DATASETS
        },
    }
    with pytest.raises(RuntimeError, match="frame SHA 漂移"):
        v6._stage_v3_equal_tables(tmp_path, tmp_path / "staging", bad_report)


V3_TEST_3WAY = 0.0033398437500000017
V3_TEST_4WAY = 0.00099609375000000045


def _fake_verdict_inputs(
    *,
    nltcs_l1=0.00015,
    test_l1=0.002,
    nltcs_3way=-0.061,
    nltcs_4way=-0.037,
    test_3way=-0.001,
    test_4way=-0.001,
):
    schedule_audit = {"arms": {}}
    quality = {}
    values = {
        "nltcs": {"l1": nltcs_l1},
        "test_300x10": {"l1": test_l1},
    }
    for name in v1.DATASETS:
        schedule_audit["arms"][name] = {
            "residual": {
                "best_loss_diagnostic_only": 100.0,
                "output_squared_loss": 120.0,
            },
        }
        quality[name] = {
            "residual": {
                "measured": {"normalized_l1_mean": values[name]["l1"]},
            },
        }
    paired = {
        "nltcs": {
            "delta_residual_minus_equal": {
                "heldout_3way_normalized_l1_mean": nltcs_3way,
                "heldout_4way_normalized_l1_mean": nltcs_4way,
            },
        },
        "test_300x10": {
            "delta_residual_minus_equal": {
                "heldout_3way_normalized_l1_mean": test_3way,
                "heldout_4way_normalized_l1_mean": test_4way,
            },
        },
    }
    return schedule_audit, quality, paired


def test_verdict_guard_violation_dominates():
    # nltcs held-out 优势失守（3way 超过 −0.050 上限）。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_3way=-0.049, test_3way=-0.001, test_4way=-0.001,
    ))
    assert evaluation["primary_verdict"] == "quality_regression_under_mw"
    assert evaluation["guards_pass"] is False

    # nltcs measured L1 超阈。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_l1=0.000198,
    ))
    assert evaluation["primary_verdict"] == "quality_regression_under_mw"

    # test measured L1 超阈。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_l1=0.002641,
    ))
    assert evaluation["primary_verdict"] == "quality_regression_under_mw"


def test_verdict_supported_when_both_test_deltas_negative():
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_3way=-0.0001, test_4way=-0.0001,
    ))
    assert evaluation["primary_verdict"] == "mw_supported"
    assert evaluation["primary_test_heldout_paired_delta"][
        "both_strictly_negative"
    ] is True
    assert evaluation["formal_claim_allowed"] is False


def test_verdict_partial_when_one_improved_50pct_and_none_worse():
    # 3way 改善 ≥50%（≤ 0.5×v3），4way 未恶化但未达 50%。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_3way=0.5 * V3_TEST_3WAY,
        test_4way=0.9 * V3_TEST_4WAY,
    ))
    assert evaluation["primary_verdict"] == "mw_partial_improvement"

    # 一项转负一项持平于 v3（未恶化）：负值本身就满足 ≥50% 改善。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_3way=-0.0001,
        test_4way=V3_TEST_4WAY,
    ))
    assert evaluation["primary_verdict"] == "mw_partial_improvement"


def test_verdict_rejected_when_no_meaningful_improvement():
    # 两项都略改善但都不足 50%。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_3way=0.9 * V3_TEST_3WAY,
        test_4way=0.9 * V3_TEST_4WAY,
    ))
    assert evaluation["primary_verdict"] == "mw_rejected"

    # 一项大改善但另一项恶化（比 v3 更差）→ 不满足 partial。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_3way=0.1 * V3_TEST_3WAY,
        test_4way=1.1 * V3_TEST_4WAY,
    ))
    assert evaluation["primary_verdict"] == "mw_rejected"


def test_verdict_boundary_values_use_frozen_thresholds():
    # 护栏边界：恰好等于阈值 = 通过。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        nltcs_l1=0.000197,
        test_l1=0.002640,
        nltcs_3way=-0.050,
        nltcs_4way=-0.030,
        test_3way=-0.0001,
        test_4way=-0.0001,
    ))
    assert evaluation["primary_verdict"] == "mw_supported"

    # 主判据边界：delta == 0 不是"严格转负"。
    evaluation = v6._verdict_evaluation(*_fake_verdict_inputs(
        test_3way=0.0,
        test_4way=-0.0001,
    ))
    assert evaluation["primary_verdict"] == "mw_partial_improvement"


def test_baseline_v5_record_matches_frozen_v5_values():
    manifest = v6._json_protocol_manifest()
    record = manifest["baseline_v5_record"]
    assert record["verdict"] == "quality_regression_under_eta_cooling"
    assert record["protocol_sha256"] == v5.FROZEN_PROTOCOL_SHA256
    record_v4 = manifest["baseline_v4_record"]
    assert record_v4["verdict"] == "quality_regression_under_lower_floor"
    assert record_v4["protocol_sha256"] == v4.FROZEN_PROTOCOL_SHA256
