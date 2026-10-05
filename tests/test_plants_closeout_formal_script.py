"""plants 收口正式协议脚本的身份、判定与统一验证入口测试（Issue #46）。

结构合同(validate_formal_artifact)的完整篡改矩阵已在
tests/test_residual_geometry_formal_script.py 中对同构实现锚定；本文件
聚焦收口协议的新逻辑：PGM 靶判定、冻结抽样离线口径、协议身份与
统一入口在新协议模块上的端到端行为。
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "probe_plants_closeout_formal.py"
)


@pytest.fixture(scope="module")
def closeout_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "probe_plants_closeout_formal_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_protocol_sha_matches_frozen_constant(closeout_module):
    """协议身份可复算且与冻结常量一致（漂移即显式重新预注册）"""
    assert closeout_module.protocol_sha256() == (
        closeout_module.FROZEN_PROTOCOL_SHA256
    )


def test_protocol_identity_content(closeout_module):
    """协议清单覆盖收口判定所需的全部冻结项"""
    manifest = closeout_module.canonical_protocol_manifest()
    assert manifest["issue"] == 46
    assert manifest["primary_dataset"] == "plants"
    assert manifest["seeds"] == [300, 301, 302, 303, 304]
    assert manifest["rounds"] == 16000
    thresholds = manifest["judgement_thresholds"]
    assert thresholds["pgm_target_measured_l1_mean"] == (
        closeout_module.PGM_TARGET_MEASURED_L1_MEAN
    )
    assert thresholds["tie_band"] == 0.03
    assert thresholds["pgm_target_quality"]["unmeasured_3way_l1"] > 0
    assert manifest["offline_sampling"] == {
        "seed": closeout_module.OFFLINE_SAMPLE_SEED,
        "n_3way": 2000,
        "n_4way": 2000,
    }


def test_expected_input_hashes_match_repo_files(closeout_module):
    """冻结输入哈希与仓库实际文件一致（协议绑定这组公开输入）"""
    import hashlib

    for ds_name, expected in closeout_module.EXPECTED_INPUT_SHA256.items():
        spec = closeout_module.DATASETS[ds_name]
        for kind, digest in expected.items():
            actual = hashlib.sha256(
                Path(spec[kind]).read_bytes()
            ).hexdigest()
            assert actual == digest, f"{ds_name}/{kind}"
    for ds_name, expected in (
        closeout_module.EXPECTED_REFERENCE_SHA256.items()
    ):
        for ref_name, digest in expected.items():
            path = closeout_module.DATASETS[ds_name]["references"][ref_name]
            import hashlib as _h

            assert _h.sha256(
                Path(path).read_bytes()
            ).hexdigest() == digest, f"{ds_name}/{ref_name}"


def _fake_runs(closeout_module, l1_by_seed, quality=None):
    quality = quality or {
        "unmeasured_3way_l1": 260.0, "unmeasured_4way_l1": 200.0,
        "raw_joint_tvd": 0.99, "binned_joint_tvd": 0.95,
        "raw_unique_states": 10, "raw_support_overlap": 5,
    }
    return [
        {
            "seed": seed, "arm": closeout_module.CLOSEOUT_ARM,
            "final_table_measured_l1": l1,
            "offline": {"train": dict(quality)},
        }
        for seed, l1 in l1_by_seed.items()
    ]


def test_judge_closeout_pass_within_tie_band(closeout_module):
    """均值 ≤ 靶 ×1.03 → closeout_pass（打平带）"""
    target = closeout_module.PGM_TARGET_MEASURED_L1_MEAN
    runs = _fake_runs(closeout_module, {
        seed: target * 1.01 for seed in closeout_module.FORMAL_SEEDS
    })
    result = closeout_module._judge(runs)
    assert result["primary_pass"] is True
    assert result["classification"] == "closeout_pass"
    assert result["pgm_target_mean"] == target


def test_judge_not_closed_beyond_tie_band(closeout_module):
    """均值 > 靶 ×1.03 → not_closed（dev 预期路径）"""
    target = closeout_module.PGM_TARGET_MEASURED_L1_MEAN
    runs = _fake_runs(closeout_module, {
        seed: target * 1.5 for seed in closeout_module.FORMAL_SEEDS
    })
    result = closeout_module._judge(runs)
    assert result["primary_pass"] is False
    assert result["classification"] == "not_closed"
    assert result["relative_gap_vs_target"] == pytest.approx(0.5)


def test_judge_boundary_exact_threshold(closeout_module):
    """恰好等于阈值 → 打平（≤ 语义）"""
    target = closeout_module.PGM_TARGET_MEASURED_L1_MEAN
    threshold = target * (1.0 + closeout_module.TIE_BAND)
    runs = _fake_runs(closeout_module, {
        seed: threshold for seed in closeout_module.FORMAL_SEEDS
    })
    assert closeout_module._judge(runs)["primary_pass"] is True


def test_judge_quality_risk_downgrades_pass(closeout_module):
    """通过但质量指标相对 PGM 靶劣化 >5% → 降级 with_quality_risk；
    not_closed 时仅记录不改分类"""
    target = closeout_module.PGM_TARGET_MEASURED_L1_MEAN
    bad_quality = {
        "unmeasured_3way_l1": (
            closeout_module.PGM_TARGET_QUALITY["unmeasured_3way_l1"] * 1.10
        ),
        "unmeasured_4way_l1": 200.0,
        "raw_joint_tvd": 0.99, "binned_joint_tvd": 0.95,
        "raw_unique_states": 10, "raw_support_overlap": 5,
    }
    runs = _fake_runs(
        closeout_module,
        {seed: target for seed in closeout_module.FORMAL_SEEDS},
        quality=bad_quality,
    )
    result = closeout_module._judge(runs)
    assert result["classification"] == "closeout_pass_with_quality_risk"
    runs_fail = _fake_runs(
        closeout_module,
        {seed: target * 2 for seed in closeout_module.FORMAL_SEEDS},
        quality=bad_quality,
    )
    result_fail = closeout_module._judge(runs_fail)
    assert result_fail["classification"] == "not_closed"
    assert result_fail["any_quality_risk"] is True


def test_sampled_combinations_deterministic(closeout_module):
    """冻结抽样：两次调用同结果、数量正确、组合互异且属于全空间"""
    columns = [f"attr_{i}" for i in range(1, 70)]
    combos_a = closeout_module.sampled_offline_combinations(
        columns, 3, 2000
    )
    combos_b = closeout_module.sampled_offline_combinations(
        columns, 3, 2000
    )
    assert combos_a == combos_b
    assert len(combos_a) == 2000
    assert len(set(combos_a)) == 2000
    assert all(len(c) == 3 and all(a in columns for a in c)
               for c in combos_a)
    combos4 = closeout_module.sampled_offline_combinations(
        columns, 4, 2000
    )
    assert len(combos4) == 2000
    # 小空间退化为全枚举
    small = closeout_module.sampled_offline_combinations(
        columns[:5], 3, 2000
    )
    assert len(small) == 10


def test_no_legacy_whitelist(closeout_module):
    """新协议不存在 legacy 白名单（任何缺版本字段的产物都被拒）"""
    assert closeout_module.KNOWN_LEGACY_ARTIFACT_SHA256 == set()


@pytest.fixture(scope="module")
def real_initial_state(closeout_module):
    return closeout_module.recompute_initial_state("plants")


def _formal_base(closeout_module, real_initial_state):
    import numpy as np

    ist = {
        "measured_l1_mean": float(np.mean(
            list(real_initial_state["measured_l1_by_seed"].values())
        )),
        "measured_l1_by_seed": dict(
            real_initial_state["measured_l1_by_seed"]
        ),
        "loss_by_seed": dict(real_initial_state["loss_by_seed"]),
        "note": "n_rounds=0 的 marginal 初始化状态（种子相关）",
    }
    runs = [
        {
            "dataset": "plants", "arm": closeout_module.CLOSEOUT_ARM,
            "seed": seed,
            "rounds_run": closeout_module.FORMAL_ROUNDS,
            "candidate_evaluations": closeout_module.FORMAL_ROUNDS,
            "pre_final_proposal_loss": 123.0, "final_loss": 120.0,
            "final_table_measured_l1": 0.0005, "best_loss": 119.0,
            "rare_query_mean_abs_residual": None,
            "common_query_mean_abs_residual": 1.5,
            "exact_match_queries": 3,
            "row_max_prob_mean_final": 0.5,
            "effective_donors_mean_final": 10.0,
            "tail_mean_pre_proposal_loss": 121.0,
            "final_table_sha256": "0" * 64,
            "elapsed_sec": 1.0,
            "offline": {"train": {
                "unmeasured_3way_l1": 260.0, "unmeasured_4way_l1": 200.0,
                "raw_joint_tvd": 0.99, "binned_joint_tvd": 0.95,
                "raw_unique_states": 10, "raw_support_overlap": 5,
            }},
        }
        for seed in closeout_module.FORMAL_SEEDS
    ]
    payload = {
        "artifact_schema_version": closeout_module.ARTIFACT_SCHEMA_VERSION,
        "protocol": closeout_module.canonical_protocol_manifest(),
        "run_config": {
            "seeds": list(closeout_module.FORMAL_SEEDS),
            "rounds": closeout_module.FORMAL_ROUNDS,
            "datasets": sorted(closeout_module.DATASETS),
        },
        "provenance": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "protocol_sha256": closeout_module.FROZEN_PROTOCOL_SHA256,
            "protocol_match": True,
            "formal": True,
            "started_at": "2026-01-01T00:00:00+08:00",
            "finished_at": "2026-01-01T01:00:00+08:00",
            "environment": {"python": "test"},
            "input_sha256": {
                name: dict(expected)
                for name, expected in
                closeout_module.EXPECTED_INPUT_SHA256.items()
            },
            "input_hash_mismatches": [],
            "command": "scripts/probe_plants_closeout_formal.py",
        },
        "datasets": {
            "plants": {
                "initial_state": ist,
                "reference_sha256": dict(
                    closeout_module.EXPECTED_REFERENCE_SHA256["plants"]
                ),
                "runs": runs,
                "judgment": closeout_module._judge(runs),
            }
        },
    }
    return payload


def test_validate_accepts_valid_artifact(
    closeout_module, real_initial_state
):
    """统一验证入口在收口协议上接受合法产物（含初态重算对拍）"""
    closeout_module.validate_formal_artifact(
        _formal_base(closeout_module, real_initial_state),
        source_kind="generator",
    )


def test_validate_rejects_core_tampering(
    closeout_module, real_initial_state
):
    """核心拒绝抽查：版本/协议篡改/判定重算/缺初态/多余字段/非布尔"""
    cases = []
    p = _formal_base(closeout_module, real_initial_state)
    p["artifact_schema_version"] = "wrong-v0"
    cases.append(p)
    p = _formal_base(closeout_module, real_initial_state)
    p["protocol"]["judgement_thresholds"]["pgm_target_measured_l1_mean"] = 1.0
    cases.append(p)
    p = _formal_base(closeout_module, real_initial_state)
    p["datasets"]["plants"]["judgment"]["classification"] = "closeout_pass"
    cases.append(p)
    p = _formal_base(closeout_module, real_initial_state)
    del p["datasets"]["plants"]["initial_state"]
    cases.append(p)
    p = _formal_base(closeout_module, real_initial_state)
    p["extra"] = 1
    cases.append(p)
    p = _formal_base(closeout_module, real_initial_state)
    p["provenance"]["formal"] = "true"
    cases.append(p)
    p = _formal_base(closeout_module, real_initial_state)
    p["datasets"]["plants"]["initial_state"]["measured_l1_by_seed"][
        str(closeout_module.FORMAL_SEEDS[0])
    ] += 1e-3
    cases.append(p)
    for payload in cases:
        with pytest.raises(closeout_module.FormalArtifactError):
            closeout_module.validate_formal_artifact(
                payload, source_kind="generator"
            )


def test_audit_end_to_end_on_closeout_protocol(
    closeout_module, real_initial_state, tmp_path
):
    """审计器在新协议模块上端到端：合法产物通过并带 audit 段；
    篡改产物拒绝且不留输出"""
    import subprocess as _sp

    payload = _formal_base(closeout_module, real_initial_state)
    json_path = tmp_path / "closeout.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False))
    result = _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", "probe_plants_closeout_formal",
         "--json", str(json_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audited = json.loads(
        (tmp_path / "closeout.json.audited.json").read_text()
    )
    assert audited["audit"]["source_kind"] == "v2"
    # 篡改判定 → 拒绝且不留输出
    payload2 = _formal_base(closeout_module, real_initial_state)
    payload2["datasets"]["plants"]["judgment"] = None
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(payload2, ensure_ascii=False))
    result2 = _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", "probe_plants_closeout_formal",
         "--json", str(bad_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
    )
    assert result2.returncode == 1
    assert not (tmp_path / "bad.json.audited.json").exists()
