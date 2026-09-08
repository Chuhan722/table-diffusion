"""纯形态 vs 方向场主线正式 A/B 协议脚本测试（Issue #75，nltcs 与 plants 两协议）。

结构合同（validate_formal_artifact）的完整篡改矩阵已在
tests/test_residual_geometry_formal_script.py 对同构实现锚定；本文件聚焦
两协议的新逻辑：非劣/优效判定、方向场观察项、协议身份与统一入口端到端。
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
MODULE_NAMES = ("probe_pure_form_nltcs_formal", "probe_pure_form_plants_formal")


def _load(name):
    scripts_dir = str(SCRIPTS)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", SCRIPTS / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module", params=MODULE_NAMES)
def proto(request):
    return _load(request.param)


def test_protocol_sha_matches_frozen_constant(proto):
    assert proto.protocol_sha256() == proto.FROZEN_PROTOCOL_SHA256


def test_protocol_identity_content(proto):
    manifest = proto.canonical_protocol_manifest()
    assert manifest["issue"] == 75
    assert manifest["seeds"] == [400, 401, 402, 403, 404]
    assert set(manifest["arms"]) == {"directed_ds2", "pure_ds0"}
    assert manifest["arms"]["directed_ds2"]["residual_directed_diffusion"] is True
    assert manifest["arms"]["directed_ds2"]["diffusion_direction_strength"] == 2.0
    assert manifest["arms"]["pure_ds0"]["residual_directed_diffusion"] is False
    assert manifest["arms"]["pure_ds0"]["diffusion_direction_strength"] == 0.0
    # 唯一变量：共享参数中不再含方向场三参数
    for key in ("residual_directed_diffusion", "diffusion_direction_strength",
                "diffusion_direction_normalization"):
        assert key not in manifest["shared_params"]
    thresholds = manifest["judgement_thresholds"]
    assert thresholds["noninferiority_margin"] == 0.05
    assert thresholds["superiority_margin"] == 0.05
    if manifest["primary_dataset"] == "nltcs":
        assert manifest["rounds"] == 2000 and manifest["frozen_si_alpha"] == 16.0
        assert manifest["frozen_rho"] == 0.01
    else:
        assert manifest["rounds"] == 16000 and manifest["frozen_si_alpha"] == 24.0
        assert manifest["frozen_rho"] == 0.005
        assert manifest["offline_sampling"]["n_3way"] == 2000


def test_expected_hashes_match_repo_files(proto):
    import hashlib

    for ds_name, expected in proto.EXPECTED_INPUT_SHA256.items():
        spec = proto.DATASETS[ds_name]
        for kind, digest in expected.items():
            assert hashlib.sha256(Path(spec[kind]).read_bytes()).hexdigest() == digest
    for ds_name, expected in proto.EXPECTED_REFERENCE_SHA256.items():
        for ref_name, digest in expected.items():
            path = proto.DATASETS[ds_name]["references"][ref_name]
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


def _quality(**overrides):
    base = {
        "unmeasured_3way_l1": 100.0, "unmeasured_4way_l1": 80.0,
        "raw_joint_tvd": 0.9, "binned_joint_tvd": 0.5,
        "raw_unique_states": 10, "raw_support_overlap": 5,
    }
    base.update(overrides)
    return base


def _fake_runs(proto, directed, pure, pure_quality=None, directed_quality=None):
    runs = []
    for seed in proto.FORMAL_SEEDS:
        runs.append({
            "seed": seed, "arm": proto.DIRECTED_ARM,
            "final_table_measured_l1": directed[seed],
            "tail_copy_prob_positive_direction": 0.5006,
            "tail_copy_prob_negative_direction": 0.4996,
            "offline": {"train": _quality(**(directed_quality or {}))},
        })
        runs.append({
            "seed": seed, "arm": proto.PURE_ARM,
            "final_table_measured_l1": pure[seed],
            "tail_copy_prob_positive_direction": None,
            "tail_copy_prob_negative_direction": None,
            "offline": {"train": _quality(**(pure_quality or {}))},
        })
    return runs


def test_judge_noninferior_when_pure_within_margin(proto):
    seeds = proto.FORMAL_SEEDS
    directed = {s: 0.000250 for s in seeds}
    pure = {s: 0.000255 for s in seeds}  # +2%，全部种子在 5% 内
    result = proto._judge(_fake_runs(proto, directed, pure))
    assert result["classification"] == "pure_form_noninferior"
    assert result["pure_seeds_within_margin"] == 5
    assert result["superior"] is False
    obs = result["observations"]
    assert obs["directed_tail_copy_prob_positive_direction_mean"] == pytest.approx(0.5006)
    assert obs["pure_vs_pgm_ratio"] == pytest.approx(0.000255 / proto.PGM_REFERENCE_MEASURED_L1_MEAN)


def test_judge_superior_when_pure_clearly_better(proto):
    seeds = proto.FORMAL_SEEDS
    directed = {s: 0.000300 for s in seeds}
    pure = {s: 0.000270 for s in seeds}  # −10%，5/5 胜
    result = proto._judge(_fake_runs(proto, directed, pure))
    assert result["classification"] == "pure_form_superior"
    assert result["pure_paired_wins"] == 5


def test_judge_directed_required_when_pure_worse(proto):
    seeds = proto.FORMAL_SEEDS
    directed = {s: 0.000250 for s in seeds}
    pure = {s: 0.000300 for s in seeds}  # +20%
    result = proto._judge(_fake_runs(proto, directed, pure))
    assert result["classification"] == "directed_required"
    assert result["noninferior"] is False and result["superior"] is False


def test_judge_noninferiority_needs_min_seeds(proto):
    seeds = proto.FORMAL_SEEDS
    directed = {s: 0.000250 for s in seeds}
    # 均值在 5% 内但只有 2 个种子在 5% 内 → 不满足 ≥3 种子条件
    pure = {seeds[0]: 0.000250, seeds[1]: 0.000250, seeds[2]: 0.000270,
            seeds[3]: 0.000270, seeds[4]: 0.000270}
    result = proto._judge(_fake_runs(proto, directed, pure))
    assert result["pure_seeds_within_margin"] == 2
    assert result["classification"] == "directed_required"


def test_judge_boundary_margin(proto):
    """边界两侧：略在 5% 内 → 非劣；略在 5% 外 → directed_required。
    （精确等于阈值时受 np.mean 末位舍入影响，不作为合同断言。）"""
    seeds = proto.FORMAL_SEEDS
    directed = {s: 0.000200 for s in seeds}
    inside = {s: 0.000200 * 1.049 for s in seeds}
    outside = {s: 0.000200 * 1.051 for s in seeds}
    assert proto._judge(_fake_runs(proto, directed, inside))["classification"] == (
        "pure_form_noninferior"
    )
    assert proto._judge(_fake_runs(proto, directed, outside))["classification"] == (
        "directed_required"
    )


def test_judge_quality_risk_suffix_only_on_pass(proto):
    seeds = proto.FORMAL_SEEDS
    directed = {s: 0.000250 for s in seeds}
    pure_ok = {s: 0.000252 for s in seeds}
    worse_quality = {"unmeasured_3way_l1": 100.0 * 1.10}
    result = proto._judge(_fake_runs(proto, directed, pure_ok, pure_quality=worse_quality))
    assert result["classification"] == "pure_form_noninferior_with_quality_risk"
    assert result["quality_risks"]["unmeasured_3way_l1"]["flagged"] is True
    pure_bad = {s: 0.000400 for s in seeds}
    result2 = proto._judge(_fake_runs(proto, directed, pure_bad, pure_quality=worse_quality))
    assert result2["classification"] == "directed_required"
    assert result2["any_quality_risk"] is True


def test_tail_mean_helper(proto):
    assert proto._tail_mean(None) is None
    assert proto._tail_mean([]) is None
    assert proto._tail_mean([None, None]) is None
    assert proto._tail_mean([0.2, None, 0.4], n=100) == pytest.approx(0.3)
    assert proto._tail_mean(list(range(200)), n=100) == pytest.approx(149.5)


def test_no_legacy_whitelist(proto):
    assert proto.KNOWN_LEGACY_ARTIFACT_SHA256 == set()


def test_run_record_fields_include_direction_observations(proto):
    assert "tail_copy_prob_positive_direction" in proto.FORMAL_RUN_RECORD_FIELDS
    assert "tail_copy_prob_positive_direction" in proto.FORMAL_RUN_NULLABLE_FIELDS
    assert "tail_copy_prob_negative_direction" in proto.FORMAL_RUN_NULLABLE_FIELDS


@pytest.fixture(scope="module")
def nltcs_proto():
    return _load("probe_pure_form_nltcs_formal")


@pytest.fixture(scope="module")
def nltcs_initial_state(nltcs_proto):
    return nltcs_proto.recompute_initial_state("nltcs")


def _formal_base(proto, real_initial_state, ds_name):
    ist = {
        "measured_l1_mean": float(np.mean(list(real_initial_state["measured_l1_by_seed"].values()))),
        "measured_l1_by_seed": dict(real_initial_state["measured_l1_by_seed"]),
        "loss_by_seed": dict(real_initial_state["loss_by_seed"]),
        "note": "n_rounds=0 的 marginal 初始化状态（种子相关）",
    }
    runs = []
    for seed in proto.FORMAL_SEEDS:
        for arm in (proto.DIRECTED_ARM, proto.PURE_ARM):
            directed = arm == proto.DIRECTED_ARM
            runs.append({
                "dataset": ds_name, "arm": arm, "seed": seed,
                "rounds_run": proto.FORMAL_ROUNDS,
                "candidate_evaluations": proto.FORMAL_ROUNDS,
                "pre_final_proposal_loss": 123.0, "final_loss": 120.0,
                "final_table_measured_l1": 0.00025 if directed else 0.000252,
                "best_loss": 119.0,
                "rare_query_mean_abs_residual": None,
                "common_query_mean_abs_residual": 1.5,
                "exact_match_queries": 3,
                "row_max_prob_mean_final": 0.5,
                "effective_donors_mean_final": 10.0,
                "tail_mean_pre_proposal_loss": 121.0,
                "final_table_sha256": "0" * 64,
                "elapsed_sec": 1.0,
                "tail_copy_prob_positive_direction": 0.5006 if directed else None,
                "tail_copy_prob_negative_direction": 0.4996 if directed else None,
                "offline": {"train": _quality()},
            })
    payload = {
        "artifact_schema_version": proto.ARTIFACT_SCHEMA_VERSION,
        "protocol": proto.canonical_protocol_manifest(),
        "run_config": {
            "seeds": list(proto.FORMAL_SEEDS), "rounds": proto.FORMAL_ROUNDS,
            "datasets": sorted(proto.DATASETS),
        },
        "provenance": {
            "git_commit": "a" * 40, "git_dirty": False,
            "protocol_sha256": proto.FROZEN_PROTOCOL_SHA256,
            "protocol_match": True, "formal": True,
            "started_at": "2026-01-01T00:00:00+08:00",
            "finished_at": "2026-01-01T01:00:00+08:00",
            "environment": {"python": "test"},
            "input_sha256": {n: dict(e) for n, e in proto.EXPECTED_INPUT_SHA256.items()},
            "input_hash_mismatches": [],
            "command": f"scripts/{ds_name}",
        },
        "datasets": {ds_name: {
            "initial_state": ist,
            "reference_sha256": dict(proto.EXPECTED_REFERENCE_SHA256[ds_name]),
            "runs": runs,
        }},
    }
    payload["datasets"][ds_name]["judgment"] = proto._judge(runs)
    return payload


def test_validate_accepts_valid_artifact_nltcs(nltcs_proto, nltcs_initial_state):
    payload = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    nltcs_proto.validate_formal_artifact(payload, source_kind="generator")
    assert payload["datasets"]["nltcs"]["judgment"]["classification"] == (
        "pure_form_noninferior"
    )


def test_validate_rejects_core_tampering_nltcs(nltcs_proto, nltcs_initial_state):
    cases = []
    p = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    p["protocol"]["arms"]["pure_ds0"]["residual_directed_diffusion"] = True
    cases.append(p)
    p = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    p["datasets"]["nltcs"]["judgment"]["classification"] = "pure_form_superior"
    cases.append(p)
    p = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    p["datasets"]["nltcs"]["runs"][0]["tail_copy_prob_positive_direction"] = "0.5"
    cases.append(p)
    p = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    del p["datasets"]["nltcs"]["runs"][1]["tail_copy_prob_negative_direction"]
    cases.append(p)
    p = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    p["datasets"]["nltcs"]["runs"] = [
        r for r in p["datasets"]["nltcs"]["runs"] if r["arm"] != nltcs_proto.PURE_ARM
    ]
    cases.append(p)
    for payload in cases:
        with pytest.raises(nltcs_proto.FormalArtifactError):
            nltcs_proto.validate_formal_artifact(payload, source_kind="generator")


def test_audit_end_to_end_nltcs(nltcs_proto, nltcs_initial_state, tmp_path):
    import subprocess as _sp

    payload = _formal_base(nltcs_proto, nltcs_initial_state, "nltcs")
    json_path = tmp_path / "pure.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False))
    result = _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", "probe_pure_form_nltcs_formal", "--json", str(json_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audited = json.loads((tmp_path / "pure.json.audited.json").read_text())
    assert audited["audit"]["source_kind"] == "v2"
