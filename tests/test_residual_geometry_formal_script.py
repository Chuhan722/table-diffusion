"""残差几何正式协议脚本的正式身份、安全门禁与判定逻辑测试（Issue #57）。"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / (
    "probe_residual_geometry_formal.py"
)


@pytest.fixture(scope="module")
def formal_module():
    scripts_dir = str(SCRIPT.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "probe_residual_geometry_formal_under_test", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_main(module, monkeypatch, tmp_path, argv, dirty, out_name="o.json"):
    out = tmp_path / out_name
    monkeypatch.setattr(
        sys, "argv",
        ["probe_residual_geometry_formal.py", "--output", str(out)] + argv,
    )
    monkeypatch.setattr(
        module, "_git",
        lambda *args: ("M x.py" if dirty else "") if "status" in args
        else "a" * 40,
    )
    monkeypatch.setattr(module, "DATASETS", {})
    monkeypatch.setattr(module, "EXPECTED_INPUT_SHA256", {})
    monkeypatch.setattr(module, "EXPECTED_REFERENCE_SHA256", {})
    monkeypatch.setattr(module, "_run_dataset", lambda *a, **k: ([], {}, {}))
    # DATASETS 被 mock 后协议身份变化，同步冻结常量以隔离测试其它门禁
    monkeypatch.setattr(
        module, "FROZEN_PROTOCOL_SHA256", module.protocol_sha256()
    )
    module.main()
    return json.loads(out.read_text())


def test_allow_dirty_forces_informal_even_with_formal_params(
    formal_module, monkeypatch, tmp_path
):
    payload = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=["--allow-dirty"], dirty=False,
    )
    assert payload["provenance"]["formal"] is False


def test_clean_tree_formal_params_is_formal(
    formal_module, monkeypatch, tmp_path
):
    assert formal_module.OUTPUT_PATH.name == (
        "formal_residual_geometry_5seed_2000round.json"
    )
    payload = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=[], dirty=False,
    )
    # DATASETS 被 mock 为空集，默认 --datasets 即"预注册全集"；干净树 +
    # 预注册 seeds/rounds/datasets → formal=True（正条件锚定）。
    assert payload["provenance"]["formal"] is True


def test_formal_requires_prereg_seeds_rounds(
    formal_module, monkeypatch, tmp_path
):
    payload = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=["--rounds", "30"], dirty=False,
    )
    assert payload["provenance"]["formal"] is False
    payload2 = _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=["--seeds", "1", "2"], dirty=False,
        out_name="o2.json",
    )
    assert payload2["provenance"]["formal"] is False


def test_dirty_tree_without_flag_refuses(
    formal_module, monkeypatch, tmp_path
):
    with pytest.raises(SystemExit, match="干净树"):
        _run_main(
            formal_module, monkeypatch, tmp_path,
            argv=[], dirty=True,
        )


def test_existing_output_refuses_overwrite(
    formal_module, monkeypatch, tmp_path
):
    out = tmp_path / "exists.json"
    out.write_text("{}")
    monkeypatch.setattr(
        sys, "argv",
        ["probe_residual_geometry_formal.py", "--output", str(out)],
    )
    with pytest.raises(SystemExit, match="拒绝覆盖"):
        formal_module.main()


# ---- 判定逻辑单元测试 ----

def _fake_runs(base_l1, cand_l1, floors=None, offline_rel=0.0):
    """构造 absolute 与 relative_f8（及可选 floor 臂）的假 runs。

    offline_rel：candidate 相对 baseline 的质量指标劣化比例。
    """
    runs = []
    seeds = list(range(len(base_l1)))
    quality_base = 0.01

    def _offline(rel):
        value = quality_base * (1.0 + rel)
        return {
            "train": {
                "unmeasured_3way_l1": value,
                "unmeasured_4way_l1": value,
                "raw_joint_tvd": value,
                "binned_joint_tvd": value,
                "raw_unique_states": 10,
                "raw_support_overlap": 10,
            }
        }

    for seed, value in zip(seeds, base_l1):
        runs.append({
            "dataset": "nltcs", "arm": "absolute", "seed": seed,
            "final_table_measured_l1": value, "offline": _offline(0.0),
        })
    for seed, value in zip(seeds, cand_l1):
        runs.append({
            "dataset": "nltcs", "arm": "relative_f8", "seed": seed,
            "final_table_measured_l1": value,
            "offline": _offline(offline_rel),
        })
    floors = floors or {}
    for arm, values in floors.items():
        for seed, value in zip(seeds, values):
            runs.append({
                "dataset": "nltcs", "arm": arm, "seed": seed,
                "final_table_measured_l1": value,
                "offline": _offline(0.0),
            })
    return runs


def _full_floor_arms(cand_l1, worse=1.5):
    """floor 次要臂默认全部劣于主 candidate。"""
    return {
        arm: [v * worse for v in cand_l1]
        for arm in ("relative_f1", "relative_f4", "relative_f16")
    }


def test_judge_supports_when_all_seeds_improve(formal_module):
    base = [0.0010, 0.0011, 0.0010, 0.0009, 0.0010]
    cand = [0.0003, 0.0004, 0.0003, 0.0003, 0.0004]
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs)
    assert judgement["paired_wins"] == 5
    assert judgement["classification"] == "supports_relative_geometry"
    assert judgement["floor_suboptimal_flag"] is False


def test_judge_mixed_when_partial_wins(formal_module):
    base = [0.0010, 0.0011, 0.0010, 0.0009, 0.0010]
    cand = [0.0008, 0.0009, 0.0008, 0.0012, 0.0011]  # 3/5 胜
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs)
    assert judgement["paired_wins"] == 3
    assert judgement["classification"] == "mixed"


def test_judge_not_supported_when_no_gain(formal_module):
    base = [0.0010] * 5
    cand = [0.0012] * 5
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs)
    assert judgement["classification"] == "not_supported"


def test_judge_supports_requires_min_improvement(formal_module):
    """5/5 胜但均值改善 <30% → mixed 而非 supports"""
    base = [0.0010] * 5
    cand = [0.00085] * 5  # 改善 15%
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs)
    assert judgement["paired_wins"] == 5
    assert judgement["classification"] == "mixed"


def test_judge_quality_risk_downgrades(formal_module):
    base = [0.0010] * 5
    cand = [0.0003] * 5
    runs = _fake_runs(
        base, cand, floors=_full_floor_arms(cand), offline_rel=0.10,
    )
    judgement = formal_module._judge(runs)
    assert judgement["any_quality_risk"] is True
    assert judgement["classification"] == (
        "supports_relative_geometry_with_quality_risk"
    )


def test_judge_floor_suboptimal_flag(formal_module):
    base = [0.0010] * 5
    cand = [0.0003] * 5
    floors = _full_floor_arms(cand)
    floors["relative_f4"] = [0.0002] * 5  # f4 优于主臂 f8
    runs = _fake_runs(base, cand, floors=floors)
    judgement = formal_module._judge(runs)
    assert judgement["floor_best_arm"] == "relative_f4"
    assert judgement["floor_suboptimal_flag"] is True
    # 不改变主分类
    assert judgement["classification"] == "supports_relative_geometry"


def test_load_reference_headerless_data_file(formal_module, tmp_path):
    """无表头 .data 参考文件必须完整读取（首行是数据不是表头）"""
    path = tmp_path / "ref.data"
    path.write_text("0,1\n1,0\n1,1\n")
    frame = formal_module._load_reference(path, ["a", "b"])
    assert len(frame) == 3
    assert list(frame.columns) == ["a", "b"]
    assert frame.iloc[0].tolist() == [0, 1]


def test_load_reference_csv_with_header(formal_module, tmp_path):
    """.csv 参考文件按表头读取并对齐列名"""
    path = tmp_path / "ref.csv"
    path.write_text("x,y\n0,1\n1,0\n")
    frame = formal_module._load_reference(path, ["a", "b"])
    assert len(frame) == 2
    assert list(frame.columns) == ["a", "b"]


def test_input_hash_mismatch_fails_closed(formal_module, monkeypatch, tmp_path):
    """公开输入与冻结 EXPECTED_INPUT_SHA256 不符时拒绝正式运行；
    --allow-dirty 探索模式可继续但 formal=False 且偏差入档"""
    out = tmp_path / "o.json"
    monkeypatch.setattr(
        sys, "argv",
        ["probe_residual_geometry_formal.py", "--output", str(out)],
    )
    monkeypatch.setattr(
        formal_module, "_git",
        lambda *args: "" if "status" in args else "testcommit",
    )
    fake_spec = {
        "schema": tmp_path / "s.yaml",
        "queries": tmp_path / "q.json",
        "marginals": tmp_path / "m.json",
    }
    for p in fake_spec.values():
        p.write_text("x")
    import json as _json
    fake_spec["queries"].write_text(_json.dumps({"record_count": 1}))
    monkeypatch.setattr(formal_module, "DATASETS", {"nltcs": fake_spec})
    monkeypatch.setattr(
        formal_module, "_run_dataset", lambda *a, **k: ([], {}, {})
    )
    monkeypatch.setattr(
        formal_module, "FROZEN_PROTOCOL_SHA256",
        formal_module.protocol_sha256(),
    )
    with pytest.raises(SystemExit, match="EXPECTED_INPUT_SHA256 不符"):
        formal_module.main()
    # 探索模式：允许继续但 formal=False，偏差记录进 provenance
    monkeypatch.setattr(
        sys, "argv",
        ["probe_residual_geometry_formal.py", "--output", str(out),
         "--allow-dirty"],
    )
    formal_module.main()
    payload = _json.loads(out.read_text())
    assert payload["provenance"]["formal"] is False
    assert len(payload["provenance"]["input_hash_mismatches"]) == 3


def test_expected_input_hashes_match_repo_files(formal_module):
    """冻结的 EXPECTED_INPUT_SHA256 与仓库当前公开输入逐一相符
    （防冻结常量与实际文件漂移）"""
    for name, spec in formal_module.DATASETS.items():
        expected = formal_module.EXPECTED_INPUT_SHA256[name]
        for kind in ("schema", "queries", "marginals"):
            actual = formal_module._sha256_file(spec[kind])
            assert actual == expected[kind], f"{name}/{kind} 哈希漂移"


# ---- Issue #60 fail-closed 三类硬化 ----

def test_protocol_sha_matches_frozen_constant(formal_module):
    """协议 SHA 可独立复算且与冻结常量一致（协议漂移即失败）"""
    assert formal_module.protocol_sha256() == (
        formal_module.FROZEN_PROTOCOL_SHA256
    )


def test_protocol_drift_refuses_formal_run(formal_module, monkeypatch, tmp_path):
    """协议常量被修改（未重新预注册）时拒绝正式运行；探索模式降级"""
    out = tmp_path / "o.json"
    monkeypatch.setattr(
        sys, "argv",
        ["probe_residual_geometry_formal.py", "--output", str(out)],
    )
    monkeypatch.setattr(
        formal_module, "_git",
        lambda *args: "" if "status" in args else "testcommit",
    )
    monkeypatch.setattr(formal_module, "PRIMARY_MIN_IMPROVEMENT", 0.01)
    with pytest.raises(SystemExit, match="FROZEN_PROTOCOL_SHA256 不符"):
        formal_module.main()
    monkeypatch.setattr(formal_module, "DATASETS", {})
    monkeypatch.setattr(
        sys, "argv",
        ["probe_residual_geometry_formal.py", "--output", str(out),
         "--allow-dirty"],
    )
    formal_module.main()
    import json as _json
    payload = _json.loads(out.read_text())
    assert payload["provenance"]["formal"] is False
    assert payload["provenance"]["protocol_match"] is False


def test_reference_hash_verified_before_parse(formal_module, tmp_path):
    """真实生产函数 _load_verified_reference：正确哈希通过并返回解析
    结果；错误内容在解析前即被拒绝（先验后用，生产与测试共用入口）"""
    import hashlib as _hashlib

    good = tmp_path / "ref.data"
    good.write_text("0,1\n1,0\n")
    digest = _hashlib.sha256(good.read_bytes()).hexdigest()
    # 注册临时数据集的冻结哈希
    formal_module.EXPECTED_REFERENCE_SHA256["_tmp_ds"] = {"train": digest}
    try:
        frame, got = formal_module._load_verified_reference(
            "_tmp_ds", "train", good, ["a", "b"]
        )
        assert got == digest and len(frame) == 2
        # 内容被篡改 → 解析前拒绝
        good.write_text("1,1\n0,0\n")
        with pytest.raises(RuntimeError, match="SHA-256 与冻结值不符"):
            formal_module._load_verified_reference(
                "_tmp_ds", "train", good, ["a", "b"]
            )
    finally:
        formal_module.EXPECTED_REFERENCE_SHA256.pop("_tmp_ds")


def test_reference_unregistered_dataset_skips_hash(formal_module, tmp_path):
    """未登记冻结哈希的数据集跳过对拍（正式身份由协议 SHA 罩住）"""
    path = tmp_path / "new.data"
    path.write_text("0\n1\n")
    frame, digest = formal_module._load_verified_reference(
        "_unregistered", "train", path, ["a"]
    )
    assert len(frame) == 2 and len(digest) == 64


def test_offline_missing_fails_formal_passes_exploratory(formal_module, capsys):
    """正式运行缺失离线指标即失败；探索运行警告继续"""
    runs = [{
        "seed": 0, "arm": "absolute",
        "offline": {"train": {
            "unmeasured_3way_l1": 0.1, "unmeasured_4way_l1": None,
            "binned_joint_tvd": 0.2,
        }},
    }]
    with pytest.raises(RuntimeError, match="缺失或非有限"):
        formal_module._assert_offline_complete("nltcs", runs, formal=True)
    formal_module._assert_offline_complete("nltcs", runs, formal=False)
    assert "警告" in capsys.readouterr().out


def test_offline_empty_fails_formal(formal_module):
    """正式运行 run 完全没有 offline 也失败"""
    runs = [{"seed": 0, "arm": "absolute", "offline": {}}]
    with pytest.raises(RuntimeError, match="缺失或非有限"):
        formal_module._assert_offline_complete("x", runs, formal=True)


def test_offline_nonfinite_fails_formal(formal_module):
    """NaN/inf/bool 离线指标在正式运行下同样 fail-closed（PR #62 意见 2）"""
    for bad in (float("nan"), float("inf"), True):
        runs = [{
            "seed": 0, "arm": "absolute",
            "offline": {"train": {
                "unmeasured_3way_l1": 0.1, "unmeasured_4way_l1": bad,
                "binned_joint_tvd": 0.2,
            }},
        }]
        with pytest.raises(RuntimeError, match="缺失或非有限"):
            formal_module._assert_offline_complete(
                "nltcs", runs, formal=True
            )


# ---- PR #62 第六轮最终合同：唯一验证入口 + 冻结结构 + 审计安全 ----
#
# 所有结构反例直接调用协议模块的 validate_formal_artifact（测试不得
# 重新实现另一套验证逻辑）；端到端行为（迁移/幂等/路径与输出安全）
# 用子进程审计器验证。


@pytest.fixture(scope="module")
def real_initial_states(formal_module):
    """按冻结协议真实重算的初始状态（module 级缓存供合法产物复用）。"""
    import numpy as _np

    states = {}
    for name in formal_module.DATASETS:
        rebuilt = formal_module.recompute_initial_state(name)
        states[name] = {
            "measured_l1_mean": float(_np.mean(
                list(rebuilt["measured_l1_by_seed"].values())
            )),
            "measured_l1_by_seed": rebuilt["measured_l1_by_seed"],
            "loss_by_seed": rebuilt["loss_by_seed"],
            "note": "n_rounds=0 的 marginal 初始化状态（种子相关）",
        }
    return states


def _full_run(formal_module, ds_name, seed, arm, l1=0.001):
    refs = list(formal_module.DATASETS[ds_name]["references"])
    return {
        "dataset": ds_name, "arm": arm, "seed": seed,
        "rounds_run": formal_module.FORMAL_ROUNDS,
        "candidate_evaluations": formal_module.FORMAL_ROUNDS,
        "pre_final_proposal_loss": 123.0, "final_loss": 120.0,
        "final_table_measured_l1": l1, "best_loss": 119.0,
        "rare_query_mean_abs_residual": None,
        "common_query_mean_abs_residual": 1.5,
        "exact_match_queries": 3,
        "row_max_prob_mean_final": 0.5,
        "effective_donors_mean_final": 10.0,
        "tail_mean_pre_proposal_loss": 121.0,
        "final_table_sha256": "0" * 64,
        "elapsed_sec": 1.0,
        "offline": {ref: {
            "unmeasured_3way_l1": 0.1, "unmeasured_4way_l1": 0.1,
            "raw_joint_tvd": 0.2, "binned_joint_tvd": 0.1,
            "raw_unique_states": 10, "raw_support_overlap": 5,
        } for ref in refs},
    }


def _formal_base(formal_module, real_initial_states):
    """构造通过统一严格验证的合法 v2 formal 产物。"""
    payload = {
        "artifact_schema_version": formal_module.ARTIFACT_SCHEMA_VERSION,
        "protocol": formal_module.canonical_protocol_manifest(),
        "run_config": {
            "seeds": list(formal_module.FORMAL_SEEDS),
            "rounds": formal_module.FORMAL_ROUNDS,
            "datasets": sorted(formal_module.DATASETS),
        },
        "provenance": {
            "git_commit": "a" * 40,
            "git_dirty": False,
            "protocol_sha256": formal_module.FROZEN_PROTOCOL_SHA256,
            "protocol_match": True,
            "formal": True,
            "started_at": "2026-01-01T00:00:00+08:00",
            "finished_at": "2026-01-01T01:00:00+08:00",
            "environment": {"python": "test"},
            "input_sha256": {
                name: dict(expected)
                for name, expected in
                formal_module.EXPECTED_INPUT_SHA256.items()
            },
            "input_hash_mismatches": [],
            "command": "scripts/probe_residual_geometry_formal.py",
        },
        "datasets": {
            name: {
                "initial_state": json.loads(json.dumps(
                    real_initial_states[name]
                )),
                "reference_sha256": dict(
                    formal_module.EXPECTED_REFERENCE_SHA256[name]
                ),
                "runs": [
                    _full_run(formal_module, name, seed, arm)
                    for seed in formal_module.FORMAL_SEEDS
                    for arm in formal_module.ARMS
                ],
            }
            for name in formal_module.DATASETS
        },
    }
    for ds in payload["datasets"].values():
        ds["judgment"] = formal_module._judge(ds["runs"])
    return payload


def _expect_reject(formal_module, payload, match=None):
    with pytest.raises(formal_module.FormalArtifactError) as excinfo:
        formal_module.validate_formal_artifact(
            payload, source_kind="generator"
        )
    if match is not None:
        assert match in str(excinfo.value), str(excinfo.value)


def _run_audit(tmp_path, payload, protocol_name, name="a.json"):
    import subprocess as _sp

    json_path = tmp_path / name
    json_path.write_text(json.dumps(payload, ensure_ascii=False))
    return _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", protocol_name, "--json", str(json_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": "src:scripts"},
    )


def test_validate_accepts_valid_v2(formal_module, real_initial_states):
    """验收：合法新格式正式产物通过统一验证入口（含初始状态重算对拍）"""
    formal_module.validate_formal_artifact(
        _formal_base(formal_module, real_initial_states),
        source_kind="generator",
    )


def test_audit_accepts_valid_v2_artifact(
    formal_module, real_initial_states, tmp_path
):
    """验收：合法 v2 产物子进程端到端审计通过，输出带冻结 audit 段"""
    result = _run_audit(
        tmp_path, _formal_base(formal_module, real_initial_states),
        "probe_residual_geometry_formal",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    audited = json.loads((tmp_path / "a.json.audited.json").read_text())
    assert audited["audit"]["source_kind"] == "v2"
    assert len(audited["audit"]["source_sha256"]) == 64


def test_validate_rejects_run_config_cleared(
    formal_module, real_initial_states
):
    """六轮反例 1：run_config 的种子/轮数/数据集全部清空 → 拒绝"""
    payload = _formal_base(formal_module, real_initial_states)
    payload["run_config"] = {"seeds": [], "rounds": None, "datasets": []}
    _expect_reject(formal_module, payload, match="run_config")


def test_validate_rejects_string_and_nonbool_flags(
    formal_module, real_initial_states
):
    """六轮反例 2：formal/protocol_match/git_dirty 必须是真正的布尔值"""
    for field, bad in [
        ("formal", "true"), ("formal", 1), ("formal", None),
        ("formal", False),
        ("protocol_match", "true"), ("protocol_match", 1),
        ("git_dirty", "false"), ("git_dirty", 0), ("git_dirty", True),
    ]:
        payload = _formal_base(formal_module, real_initial_states)
        payload["provenance"][field] = bad
        _expect_reject(formal_module, payload, match=field)


def test_validate_rejects_extra_fields_all_levels(
    formal_module, real_initial_states
):
    """六轮反例 3 + 验收矩阵：各层多余/冲突字段全部拒绝"""
    def mutate(path):
        payload = _formal_base(formal_module, real_initial_states)
        node = payload
        for key in path:
            node = node[key]
        node["unexpected_extra_field"] = 1
        return payload

    for path in [
        (), ("run_config",), ("provenance",), ("protocol",),
        ("datasets", "nltcs"),
        ("datasets", "nltcs", "initial_state"),
        ("datasets", "nltcs", "runs", 0),
        ("datasets", "nltcs", "runs", 0, "offline", "train"),
        ("provenance", "input_sha256", "nltcs"),
        ("datasets", "nltcs", "reference_sha256"),
    ]:
        _expect_reject(formal_module, mutate(path))


def test_validate_rejects_hash_map_tampering(
    formal_module, real_initial_states
):
    """六轮反例 4：输入/参考哈希整份相等——增键、删键、改值全拒"""
    cases = []
    p = _formal_base(formal_module, real_initial_states)
    p["provenance"]["input_sha256"]["nltcs"]["forged"] = "0" * 64
    cases.append(p)
    p = _formal_base(formal_module, real_initial_states)
    del p["provenance"]["input_sha256"]["nltcs"]["schema"]
    cases.append(p)
    p = _formal_base(formal_module, real_initial_states)
    p["provenance"]["input_sha256"]["nltcs"]["schema"] = "f" * 64
    cases.append(p)
    p = _formal_base(formal_module, real_initial_states)
    p["datasets"]["nltcs"]["reference_sha256"]["forged"] = "0" * 64
    cases.append(p)
    p = _formal_base(formal_module, real_initial_states)
    p["datasets"]["nltcs"]["reference_sha256"] = {}
    cases.append(p)
    for payload in cases:
        _expect_reject(formal_module, payload)


def test_validate_rejects_old_spelling_in_v2(
    formal_module, real_initial_states
):
    """六轮反例 5：新版产物使用旧拼写 judgement（含双拼写）→ 拒绝"""
    payload = _formal_base(formal_module, real_initial_states)
    payload["datasets"]["nltcs"]["judgement"] = (
        payload["datasets"]["nltcs"].pop("judgment")
    )
    _expect_reject(formal_module, payload, match="judgement")
    payload2 = _formal_base(formal_module, real_initial_states)
    payload2["datasets"]["nltcs"]["judgement"] = dict(
        payload2["datasets"]["nltcs"]["judgment"]
    )
    _expect_reject(formal_module, payload2)


def test_validate_rejects_missing_initial_state_no_backfill(
    formal_module, real_initial_states, tmp_path
):
    """六轮反例 6：新版缺 initial_state 直接拒绝；审计器不自动补写"""
    payload = _formal_base(formal_module, real_initial_states)
    del payload["datasets"]["nltcs"]["initial_state"]
    _expect_reject(formal_module, payload, match="initial_state")
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert not (tmp_path / "a.json.audited.json").exists()


def test_validate_rejects_initial_state_recompute_mismatch(
    formal_module, real_initial_states
):
    """初始状态存在但与冻结种子重算值不一致 → 拒绝（不能伪造）"""
    payload = _formal_base(formal_module, real_initial_states)
    ist = payload["datasets"]["test_300x10"]["initial_state"]
    seed_key = str(formal_module.FORMAL_SEEDS[0])
    ist["measured_l1_by_seed"][seed_key] += 1e-3
    ist["measured_l1_mean"] = float(
        sum(ist["measured_l1_by_seed"].values())
        / len(ist["measured_l1_by_seed"])
    )
    _expect_reject(formal_module, payload, match="重算对拍不一致")


def test_validate_protected_field_matrix(
    formal_module, real_initial_states
):
    """验收矩阵：整份结构每个必需字段 × 删除/空值/错误类型/内容篡改。

    程序化遍历顶层、protocol 清单、run_config、provenance、数据集、
    initial_state、运行记录与 offline 指标的全部受保护字段。

    边界（显式声明）：进入科学判定的一切数值（final_table_measured_l1、
    unmeasured_3/4way、binned_joint_tvd、initial_state 种子映射）都被
    判定/初态重算锚定，内容篡改必拒；下面豁免集合是"无冻结锚点的运行
    诊断记录"——审计器不重跑 2000 轮实验，数学上无法区分被篡改的
    墙钟/loss 快照与真实值。这些字段的删除/空值/类型错误仍然全部拒绝，
    仅内容篡改不可判定。
    """
    # (容器名, 字段) → 豁免的变换集合
    free_value = {"tamper"}
    nullable = {"tamper", "null"}
    exemptions = {
        ("provenance", "started_at"): free_value,
        ("provenance", "finished_at"): free_value,
        ("provenance", "command"): free_value,
        ("provenance", "environment"): free_value,
        ("initial_state", "note"): free_value,
        ("run", "pre_final_proposal_loss"): free_value,
        ("run", "final_loss"): free_value,
        ("run", "best_loss"): free_value,
        ("run", "tail_mean_pre_proposal_loss"): free_value,
        ("run", "elapsed_sec"): free_value,
        ("run", "exact_match_queries"): free_value,
        ("run", "rare_query_mean_abs_residual"): free_value,
        ("run", "common_query_mean_abs_residual"): nullable,
        ("run", "row_max_prob_mean_final"): nullable,
        ("run", "effective_donors_mean_final"): nullable,
        ("offline", "raw_joint_tvd"): free_value,
        ("offline", "raw_unique_states"): free_value,
        ("offline", "raw_support_overlap"): free_value,
    }

    def containers(payload):
        yield "top", payload, sorted(formal_module.FORMAL_TOP_LEVEL_FIELDS)
        yield "protocol", payload["protocol"], sorted(payload["protocol"])
        yield "run_config", payload["run_config"], sorted(
            formal_module.FORMAL_RUN_CONFIG_FIELDS
        )
        yield "provenance", payload["provenance"], sorted(
            formal_module.FORMAL_PROVENANCE_FIELDS
        )
        yield "dataset", payload["datasets"]["nltcs"], sorted(
            formal_module.FORMAL_DATASET_FIELDS
        )
        yield (
            "initial_state",
            payload["datasets"]["nltcs"]["initial_state"],
            sorted(formal_module.FORMAL_INITIAL_STATE_FIELDS),
        )
        yield "run", payload["datasets"]["nltcs"]["runs"][0], sorted(
            formal_module.FORMAL_RUN_RECORD_FIELDS
        )
        yield (
            "offline",
            payload["datasets"]["nltcs"]["runs"][0]["offline"]["train"],
            sorted(formal_module.FORMAL_OFFLINE_METRIC_FIELDS),
        )

    n_checked = 0
    template = _formal_base(formal_module, real_initial_states)
    layout = [
        (index, cname, fields)
        for index, (cname, _, fields) in enumerate(containers(template))
    ]
    for container_index, cname, fields in layout:
        for field in fields:
            for mutation in ("delete", "null", "wrong_type", "tamper"):
                if mutation in exemptions.get((cname, field), set()):
                    continue
                payload = _formal_base(formal_module, real_initial_states)
                container = list(containers(payload))[container_index][1]
                original = container[field]
                if mutation == "delete":
                    del container[field]
                elif mutation == "null":
                    if original is None:
                        continue
                    container[field] = None
                elif mutation == "wrong_type":
                    container[field] = (
                        "wrong" if not isinstance(original, str)
                        else 12345
                    )
                else:
                    if isinstance(original, bool):
                        container[field] = not original
                    elif isinstance(original, (int, float)) and (
                        original is not None
                    ):
                        container[field] = (original or 1) * 31 + 7
                    elif isinstance(original, str):
                        container[field] = original + "_tampered"
                    elif isinstance(original, list):
                        container[field] = original + ["tampered"]
                    elif isinstance(original, dict):
                        container[field] = {**original, "tampered": 1}
                    elif original is None:
                        container[field] = 0.5
                with pytest.raises(formal_module.FormalArtifactError):
                    formal_module.validate_formal_artifact(
                        payload, source_kind="generator"
                    )
                n_checked += 1
    assert n_checked > 150


def test_validate_rejects_list_anomalies(
    formal_module, real_initial_states
):
    """验收矩阵：列表缺项、重复项和顺序错误全部拒绝"""
    seeds = list(formal_module.FORMAL_SEEDS)
    for bad_seeds in [
        seeds[:-1],                       # 缺项
        seeds + [seeds[0]],               # 重复
        list(reversed(seeds)),            # 顺序错误
        [True] + seeds[1:],               # 布尔伪装整数
    ]:
        payload = _formal_base(formal_module, real_initial_states)
        payload["run_config"]["seeds"] = bad_seeds
        _expect_reject(formal_module, payload, match="seeds")
    payload = _formal_base(formal_module, real_initial_states)
    payload["run_config"]["datasets"] = sorted(
        formal_module.DATASETS, reverse=True
    )
    _expect_reject(formal_module, payload, match="datasets")
    # runs：缺组合 / 重复组合
    payload = _formal_base(formal_module, real_initial_states)
    payload["datasets"]["nltcs"]["runs"] = (
        payload["datasets"]["nltcs"]["runs"][:-1]
    )
    _expect_reject(formal_module, payload)
    payload = _formal_base(formal_module, real_initial_states)
    payload["datasets"]["nltcs"]["runs"].append(
        json.loads(json.dumps(payload["datasets"]["nltcs"]["runs"][0]))
    )
    _expect_reject(formal_module, payload, match="重复")


def test_validate_rejects_judgment_variants(
    formal_module, real_initial_states
):
    """验收矩阵：判定缺失/空值/空映射/错误类型/重算不一致全部拒绝"""
    for mutate in [
        lambda ds: ds.pop("judgment"),
        lambda ds: ds.__setitem__("judgment", None),
        lambda ds: ds.__setitem__("judgment", {}),
        lambda ds: ds.__setitem__("judgment", "supports"),
        lambda ds: ds["judgment"].__setitem__(
            "classification", "forged_supports"
        ),
    ]:
        payload = _formal_base(formal_module, real_initial_states)
        mutate(payload["datasets"]["nltcs"])
        _expect_reject(formal_module, payload)


def test_validate_rejects_forged_protocol_fields(
    formal_module, real_initial_states
):
    """五轮反例回归：issue/primary_dataset/arms/shared_params 篡改全拒"""
    for field, value in [
        ("issue", 999), ("primary_dataset", "adult"),
        ("seeds", [1, 2, 3]), ("rounds", 10),
    ]:
        payload = _formal_base(formal_module, real_initial_states)
        payload["protocol"][field] = value
        _expect_reject(formal_module, payload, match="protocol")
    payload = _formal_base(formal_module, real_initial_states)
    payload["protocol"]["arms"]["forged_arm"] = {}
    _expect_reject(formal_module, payload, match="protocol")
    payload = _formal_base(formal_module, real_initial_states)
    payload["protocol"]["shared_params"]["rho"] = 0.99
    _expect_reject(formal_module, payload, match="protocol")


def test_validate_rejects_identity_tampering(
    formal_module, real_initial_states
):
    """回归：schema version/protocol_sha256/git_commit 身份字段篡改全拒"""
    payload = _formal_base(formal_module, real_initial_states)
    payload["artifact_schema_version"] = "residual-geometry-formal-v999"
    _expect_reject(formal_module, payload, match="结构版本")
    payload = _formal_base(formal_module, real_initial_states)
    payload["provenance"]["protocol_sha256"] = "0" * 64
    _expect_reject(formal_module, payload, match="protocol_sha256")
    payload = _formal_base(formal_module, real_initial_states)
    payload["provenance"]["git_commit"] = "not-a-commit"
    _expect_reject(formal_module, payload, match="git_commit")
    payload = _formal_base(formal_module, real_initial_states)
    payload["provenance"]["input_hash_mismatches"] = ["x"]
    _expect_reject(formal_module, payload, match="input_hash_mismatches")


def test_validate_rejects_run_record_task_mismatch(
    formal_module, real_initial_states
):
    """回归：运行记录的数据集名/种子/臂/轮数必须与所属任务一致"""
    for field, value, match in [
        ("dataset", "test_300x10", "所属"),
        ("seed", 999, "种子"),
        ("arm", "forged_arm", "臂"),
        ("rounds_run", 1999, "轮数"),
        ("candidate_evaluations", 0, "candidate_evaluations"),
        ("final_table_measured_l1", True, "final_table_measured_l1"),
        ("final_table_measured_l1", float("nan"),
         "final_table_measured_l1"),
        ("final_table_measured_l1", "0.001", "final_table_measured_l1"),
        ("final_table_sha256", "zz", "final_table_sha256"),
    ]:
        payload = _formal_base(formal_module, real_initial_states)
        payload["datasets"]["nltcs"]["runs"][0][field] = value
        _expect_reject(formal_module, payload, match=match)


def test_validate_rejects_wrong_offline_reference_names(
    formal_module, real_initial_states
):
    """回归（四轮意见 3）：离线参考名集合必须与冻结参考精确一致"""
    payload = _formal_base(formal_module, real_initial_states)
    run = payload["datasets"]["nltcs"]["runs"][0]
    run["offline"] = {"forged_ref": run["offline"]["train"]}
    _expect_reject(formal_module, payload, match="参考名")
    payload2 = _formal_base(formal_module, real_initial_states)
    payload2["datasets"]["nltcs"]["runs"][0]["offline"] = {}
    _expect_reject(formal_module, payload2, match="参考名")


def test_validate_source_kind_contract(
    formal_module, real_initial_states
):
    """audit 段合同：generator 禁止 audit 段；audit 要求冻结结构段"""
    payload = _formal_base(formal_module, real_initial_states)
    payload["audit"] = {"source_sha256": "0" * 64, "source_kind": "v2"}
    _expect_reject(formal_module, payload, match="多余")
    good_audit = _formal_base(formal_module, real_initial_states)
    good_audit["audit"] = {
        "source_sha256": "0" * 64, "source_kind": "v2",
    }
    formal_module.validate_formal_artifact(good_audit, source_kind="audit")
    for bad in [
        {"source_sha256": "0" * 64},                        # 缺字段
        {"source_sha256": "0" * 64, "source_kind": "v2",
         "extra": 1},                                       # 多字段
        {"source_sha256": "xyz", "source_kind": "v2"},      # 非 hex
        {"source_sha256": "0" * 64, "source_kind": "v3"},   # 非法枚举
    ]:
        payload = _formal_base(formal_module, real_initial_states)
        payload["audit"] = bad
        with pytest.raises(formal_module.FormalArtifactError):
            formal_module.validate_formal_artifact(
                payload, source_kind="audit"
            )
    missing = _formal_base(formal_module, real_initial_states)
    with pytest.raises(formal_module.FormalArtifactError):
        formal_module.validate_formal_artifact(missing, source_kind="audit")
    with pytest.raises(ValueError):
        formal_module.validate_formal_artifact(
            _formal_base(formal_module, real_initial_states),
            source_kind="unknown",
        )


def test_generator_invokes_unified_validator(
    formal_module, monkeypatch, tmp_path
):
    """生成器必须在正式结果写入前调用唯一验证入口（合同一）"""
    calls = []

    def spy(payload, *, source_kind):
        calls.append(source_kind)

    monkeypatch.setattr(formal_module, "validate_formal_artifact", spy)
    payload = _run_main(
        formal_module, monkeypatch, tmp_path, argv=[], dirty=False,
    )
    assert payload["provenance"]["formal"] is True
    assert calls == ["generator"]
    # 探索性运行（formal=False）不经过正式合同校验
    calls.clear()
    _run_main(
        formal_module, monkeypatch, tmp_path,
        argv=["--allow-dirty"], dirty=False, out_name="o2.json",
    )
    assert calls == []


def test_audit_rejects_non_formal_artifact(formal_module, tmp_path):
    """审计器语义收窄：非 formal（探索性）产物一律拒绝"""
    payload = {
        "artifact_schema_version": formal_module.ARTIFACT_SCHEMA_VERSION,
        "protocol": formal_module.canonical_protocol_manifest(),
        "run_config": {"seeds": [1], "rounds": 5, "datasets": ["nltcs"]},
        "provenance": {"formal": False},
        "datasets": {},
    }
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert not (tmp_path / "a.json.audited.json").exists()


def test_audit_rejects_unknown_legacy_artifact(formal_module, tmp_path):
    """缺 artifact_schema_version 且文件 SHA 不在白名单 → FATAL"""
    payload = {"provenance": {"formal": True}, "protocol": {"seeds": []},
               "datasets": {}}
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "白名单" in result.stdout


def test_audit_legacy_idempotent_and_source_untouched(
    formal_module, tmp_path
):
    """验收：已知 legacy 连续审计两次均通过；原文件逐字节不变；两次
    输出逐字节相同（audit 段无时间戳）"""
    import hashlib as _hashlib
    import shutil as _shutil
    import subprocess as _sp

    archived = (
        Path(__file__).resolve().parents[1]
        / "docs/实验结果/formal_residual_geometry_5seed_2000round.json"
    )
    json_path = tmp_path / "legacy.json"
    _shutil.copy(archived, json_path)
    before = _hashlib.sha256(json_path.read_bytes()).hexdigest()
    assert before in formal_module.KNOWN_LEGACY_ARTIFACT_SHA256

    def run(out_name):
        return _sp.run(
            [sys.executable, "scripts/audit_formal_json.py",
             "--protocol", "probe_residual_geometry_formal",
             "--json", str(json_path),
             "--output", str(tmp_path / out_name)],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
        )

    first = run("audit1.json")
    assert first.returncode == 0, first.stdout + first.stderr
    second = run("audit2.json")
    assert second.returncode == 0, second.stdout + second.stderr
    after = _hashlib.sha256(json_path.read_bytes()).hexdigest()
    assert after == before
    out1 = (tmp_path / "audit1.json").read_bytes()
    out2 = (tmp_path / "audit2.json").read_bytes()
    assert out1 == out2
    audited = json.loads(out1)
    assert audited["audit"]["source_kind"] == "legacy_migrated"
    assert audited["audit"]["source_sha256"] == before
    # 迁移后的输出本身满足新版全部结构合同
    core = dict(audited)
    del core["audit"]
    formal_module.validate_formal_artifact(core, source_kind="generator")


def test_audit_rejects_same_path_alias_symlink_hardlink(
    formal_module, real_initial_states, tmp_path
):
    """验收：输入输出同路径/路径别名/符号链接/硬链接全部拒绝，且任一
    场景原文件逐字节不变"""
    import hashlib as _hashlib
    import os as _os
    import subprocess as _sp

    payload = _formal_base(formal_module, real_initial_states)
    json_path = tmp_path / "src.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False))
    before = _hashlib.sha256(json_path.read_bytes()).hexdigest()

    alias = str(tmp_path / "." / "src.json")
    link_path = tmp_path / "src_link.json"
    _os.symlink(json_path, link_path)
    hard_path = tmp_path / "src_hard.json"
    _os.link(json_path, hard_path)

    for output in [str(json_path), alias, str(link_path), str(hard_path)]:
        result = _sp.run(
            [sys.executable, "scripts/audit_formal_json.py",
             "--protocol", "probe_residual_geometry_formal",
             "--json", str(json_path), "--output", output],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
        )
        assert result.returncode == 1, output
        assert "只读不变量" in result.stdout or "相同" in result.stdout
        assert _hashlib.sha256(
            json_path.read_bytes()
        ).hexdigest() == before, output


def test_audit_refuses_foreign_or_unparseable_existing_output(
    formal_module, real_initial_states, tmp_path
):
    """验收（输出安全 4）：已有输出不属于同一源文件或无法解析 → 拒绝
    覆盖且原有输出保持不变"""
    import subprocess as _sp

    def audit(src_name, out_name):
        return _sp.run(
            [sys.executable, "scripts/audit_formal_json.py",
             "--protocol", "probe_residual_geometry_formal",
             "--json", str(tmp_path / src_name),
             "--output", str(tmp_path / out_name)],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
        )

    payload = _formal_base(formal_module, real_initial_states)
    (tmp_path / "one.json").write_text(
        json.dumps(payload, ensure_ascii=False)
    )
    assert audit("one.json", "out.json").returncode == 0
    original_output = (tmp_path / "out.json").read_bytes()
    # 同源重复审计：允许覆盖（幂等）
    assert audit("one.json", "out.json").returncode == 0
    assert (tmp_path / "out.json").read_bytes() == original_output
    # 另一源文件（字节不同 → source_sha256 不同）指向既有输出 → 拒绝
    payload2 = _formal_base(formal_module, real_initial_states)
    payload2["provenance"]["command"] = "another run"
    (tmp_path / "two.json").write_text(
        json.dumps(payload2, ensure_ascii=False)
    )
    blocked = audit("two.json", "out.json")
    assert blocked.returncode == 1
    assert "拒绝覆盖" in blocked.stdout
    assert (tmp_path / "out.json").read_bytes() == original_output
    # 无法解析的既有输出 → 拒绝覆盖
    (tmp_path / "corrupt.json").write_text("not json")
    blocked2 = audit("one.json", "corrupt.json")
    assert blocked2.returncode == 1
    assert (tmp_path / "corrupt.json").read_text() == "not json"


def test_audit_failure_leaves_no_output(
    formal_module, real_initial_states, tmp_path
):
    """验收：任一失败场景不得留下可被误认为成功的审计输出"""
    payload = _formal_base(formal_module, real_initial_states)
    payload["datasets"]["nltcs"]["judgment"] = None
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    leftovers = [
        p for p in tmp_path.iterdir()
        if p.name != "a.json" and not p.name.startswith(".")
    ]
    assert leftovers == []


def test_audit_requires_unified_entry(formal_module, tmp_path):
    """审计器只支持实现统一验证入口合同的协议模块"""
    import subprocess as _sp

    (tmp_path / "stub_protocol.py").write_text("X = 1\n")
    json_path = tmp_path / "a.json"
    json_path.write_text("{}")
    result = _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", "stub_protocol", "--json", str(json_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": f"src:scripts:{tmp_path}"},
    )
    assert result.returncode == 1
    assert "统一验证入口" in result.stdout
