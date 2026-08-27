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
        else "testcommit",
    )
    monkeypatch.setattr(module, "DATASETS", {})
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


def _run_audit(tmp_path, payload, protocol_name):
    import json as _json
    import subprocess as _sp

    json_path = tmp_path / "a.json"
    json_path.write_text(_json.dumps(payload, ensure_ascii=False))
    return _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", protocol_name, "--json", str(json_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ,
             "PYTHONPATH": "src:scripts"},
    )


def test_audit_rejects_protocol_sha_mismatch(formal_module, tmp_path):
    """审计器对新产物复核协议 SHA：不一致 FATAL 退出（PR #62 意见 3）"""
    payload = {
        "provenance": {
            "protocol_sha256": "0" * 64, "protocol_match": True,
            "formal": True,
        },
        "protocol": {"seeds": []},
        "datasets": {},
    }
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "协议身份不一致" in result.stdout


def test_audit_accepts_matching_protocol_sha_and_legacy(
    formal_module, tmp_path
):
    """审计器：协议 SHA 一致的非正式产物通过（完整性检查只对 formal
    执行）；白名单命中的真实归档 legacy 产物端到端通过"""
    good = {
        "provenance": {
            "protocol_sha256": formal_module.protocol_sha256(),
            "protocol_match": True, "formal": False,
        },
        "protocol": {"seeds": []},
        "datasets": {},
    }
    result = _run_audit(tmp_path, good, "probe_residual_geometry_formal")
    assert result.returncode == 0
    assert "协议 SHA 复核一致" in result.stdout
    # legacy 接受路径：只有白名单命中的真实归档产物可通过
    import shutil as _shutil

    archived = (
        Path(__file__).resolve().parents[1]
        / "docs/实验结果/formal_residual_geometry_5seed_2000round.json"
    )
    json_path = tmp_path / "legacy.json"
    _shutil.copy(archived, json_path)
    import subprocess as _sp

    result2 = _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", "probe_residual_geometry_formal",
         "--json", str(json_path)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
    )
    assert result2.returncode == 0
    assert "已知 legacy 产物" in result2.stdout


def test_audit_rejects_formal_without_protocol_match(
    formal_module, tmp_path
):
    """formal=true 但 protocol_match=false 的产物被审计器拒绝"""
    payload = {
        "provenance": {
            "protocol_sha256": formal_module.protocol_sha256(),
            "protocol_match": False, "formal": True,
        },
        "protocol": {"seeds": []},
        "datasets": {},
    }
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "protocol_match" in result.stdout


def test_audit_rejects_unknown_legacy_artifact(formal_module, tmp_path):
    """缺 protocol_sha256 且文件 SHA 不在白名单 → FATAL（PR #62 二轮）"""
    payload = {"provenance": {"formal": True}, "protocol": {"seeds": []},
               "datasets": {}}
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "白名单" in result.stdout


def test_audit_rejects_empty_datasets_formal(formal_module, tmp_path):
    """身份字段全部正确但 datasets 为空的 formal 产物 → FATAL（空壳拒绝）"""
    payload = _formal_base(formal_module)
    payload["datasets"] = {}
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    # 空 datasets 先在 reference_sha256 对拍处失败（同为 fail-closed）
    assert "reference_sha256" in result.stdout or "数据集不完整" in result.stdout


def test_audit_rejects_missing_combo_and_nonfinite(formal_module, tmp_path):
    """缺 seed×arm 组合或指标非有限的 formal 产物 → FATAL"""
    # 缺一个组合
    payload = _formal_base(formal_module)
    payload["datasets"]["nltcs"]["runs"] = [
        run for run in payload["datasets"]["nltcs"]["runs"]
        if (run["seed"], run["arm"]) != (
            formal_module.FORMAL_SEEDS[0], "absolute"
        )
    ]
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "组合不完整" in result.stdout
    # NaN 主指标
    payload2 = _formal_base(formal_module)
    payload2["datasets"]["nltcs"]["runs"][0][
        "final_table_measured_l1"
    ] = float("nan")
    result2 = _run_audit(tmp_path, payload2, "probe_residual_geometry_formal")
    assert result2.returncode == 1
    assert "缺失或非有限" in result2.stdout


# ---- PR #62 三轮审查：审计器身份对拍/恰好一次/类型检查 ----

def _formal_base(formal_module):
    """构造通过全部统一验证的 v2 结构 formal 产物骨架。"""
    def _runs(ds_name):
        refs = list(formal_module.DATASETS[ds_name]["references"])
        return [
            {
                "seed": seed, "arm": arm,
                "final_table_measured_l1": 0.001,
                "offline": {ref: {
                    "unmeasured_3way_l1": 0.1,
                    "unmeasured_4way_l1": 0.1,
                    "binned_joint_tvd": 0.1,
                } for ref in refs},
            }
            for seed in formal_module.FORMAL_SEEDS
            for arm in formal_module.ARMS
        ]

    payload = {
        "artifact_schema_version": formal_module.ARTIFACT_SCHEMA_VERSION,
        "protocol": formal_module.canonical_protocol_manifest(),
        "run_config": {
            "seeds": list(formal_module.FORMAL_SEEDS),
            "rounds": formal_module.FORMAL_ROUNDS,
            "datasets": list(formal_module.DATASETS),
        },
        "provenance": {
            "protocol_sha256": formal_module.protocol_sha256(),
            "protocol_match": True, "formal": True,
            "input_sha256": {
                name: dict(expected)
                for name, expected in
                formal_module.EXPECTED_INPUT_SHA256.items()
            },
        },
        "datasets": {
            name: {
                "runs": _runs(name),
                "reference_sha256": dict(
                    formal_module.EXPECTED_REFERENCE_SHA256[name]
                ),
            }
            for name in formal_module.DATASETS
        },
    }
    # 判定字段填真实重算值（无条件重算要求精确一致）
    for name, ds in payload["datasets"].items():
        ds["judgment"] = formal_module._judge(ds["runs"])
    return payload


def test_audit_accepts_valid_v2_artifact(formal_module, tmp_path):
    """验收矩阵：一份合法的新格式正式产物能够通过"""
    result = _run_audit(
        tmp_path, _formal_base(formal_module),
        "probe_residual_geometry_formal",
    )
    assert result.returncode == 0, result.stdout
    assert "canonical 清单整份对拍一致" in result.stdout
    assert "判定无条件重算一致" in result.stdout


def test_audit_rejects_missing_input_hashes(formal_module, tmp_path):
    """删除全部 input_sha256 的 formal 产物 → FATAL（三轮意见 1）"""
    payload = _formal_base(formal_module)
    payload["provenance"].pop("input_sha256")
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "input_sha256" in result.stdout


def test_audit_rejects_missing_reference_hashes(formal_module, tmp_path):
    """删除数据集 reference_sha256 的 formal 产物 → FATAL（三轮意见 1）"""
    payload = _formal_base(formal_module)
    payload["datasets"]["nltcs"].pop("reference_sha256")
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "reference_sha256" in result.stdout


def test_audit_rejects_forged_seeds(formal_module, tmp_path):
    """伪造 protocol.seeds 的 formal 产物 → FATAL（三轮意见 1，
    初态重建同时改为固定使用 FORMAL_SEEDS）"""
    payload = _formal_base(formal_module)
    payload["protocol"]["seeds"] = []
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "protocol" in result.stdout  # 整份对拍或 seeds 检查拒绝


def test_audit_rejects_duplicate_runs(formal_module, tmp_path):
    """追加重复 seed×arm run → FATAL（三轮意见 2：Counter 恰好一次）"""
    payload = _formal_base(formal_module)
    runs = payload["datasets"]["nltcs"]["runs"]
    runs.append(dict(runs[0]))
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "重复" in result.stdout


def test_audit_rejects_boolean_metrics(formal_module, tmp_path):
    """offline 指标为 true 的 formal 产物 → FATAL（三轮意见 3：
    np.isfinite(True) 为真，需显式拒绝 bool）"""
    payload = _formal_base(formal_module)
    for run in payload["datasets"]["nltcs"]["runs"]:
        run["offline"]["train"]["unmeasured_3way_l1"] = True
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "非有限数值" in result.stdout


def test_audit_rejects_string_metrics(formal_module, tmp_path):
    """主指标为字符串 → FATAL（类型检查与生成端一致）"""
    payload = _formal_base(formal_module)
    payload["datasets"]["nltcs"]["runs"][0][
        "final_table_measured_l1"
    ] = "0.001"
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "非有限数值" in result.stdout


# ---- PR #62 四轮审查：完整协议对拍/judgment 强制/参考名/幂等 ----

def test_audit_rejects_forged_arms_and_shared_params(
    formal_module, tmp_path
):
    """伪造 protocol.arms 或 shared_params（保留正确协议哈希）→ FATAL
    （四轮意见 1）"""
    payload = _formal_base(formal_module)
    payload["protocol"]["arms"] = {"absolute": {"residual_geometry": "chi2"}}
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "整份对拍失败" in result.stdout
    payload2 = _formal_base(formal_module)
    payload2["protocol"]["shared_params"]["rho"] = 0.99
    result2 = _run_audit(tmp_path, payload2, "probe_residual_geometry_formal")
    assert result2.returncode == 1
    assert "整份对拍失败" in result2.stdout


def test_audit_rejects_missing_judgment_formal(formal_module, tmp_path):
    """formal 产物删除全部判定字段 → FATAL（四轮意见 2）"""
    payload = _formal_base(formal_module)
    for ds in payload["datasets"].values():
        ds.pop("judgment", None)
        ds.pop("judgement", None)
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "缺失判定字段" in result.stdout


def test_audit_rejects_dual_judgment_spellings(formal_module, tmp_path):
    """judgment 与 judgement 同时存在 → FATAL（拼写歧义）"""
    payload = _formal_base(formal_module)
    ds = payload["datasets"]["nltcs"]
    ds["judgement"] = ds["judgment"]
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "同时存在" in result.stdout


def test_audit_rejects_wrong_offline_reference_names(
    formal_module, tmp_path
):
    """offline 参考名改为任意错误名称 → FATAL（四轮意见 3）"""
    payload = _formal_base(formal_module)
    for run in payload["datasets"]["test_300x10"]["runs"]:
        run["offline"] = {"bogus": run["offline"].popitem()[1]}
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "离线参考名" in result.stdout


def test_audit_is_idempotent_on_legacy_artifact(formal_module, tmp_path):
    """连续两次审计同一 legacy 归档产物均成功且原文件字节不变
    （四轮意见 4：审计输出写 .audited.json，原产物永不改写）"""
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

    def _audit_once():
        return _sp.run(
            [sys.executable, "scripts/audit_formal_json.py",
             "--protocol", "probe_residual_geometry_formal",
             "--json", str(json_path)],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
        )

    first = _audit_once()
    assert first.returncode == 0, first.stdout
    assert "已知 legacy 产物" in first.stdout
    after_first = _hashlib.sha256(json_path.read_bytes()).hexdigest()
    assert after_first == before, "审计不得改写原产物文件"
    assert (tmp_path / "legacy.json.audited.json").exists()
    second = _audit_once()
    assert second.returncode == 0, second.stdout
    assert "已知 legacy 产物" in second.stdout


# ---- PR #62 五轮审查：统一不变量验收矩阵 ----

def test_audit_rejects_null_judgment(formal_module, tmp_path):
    """判定字段值为 null 的 formal 产物 → FATAL（五轮反例 1）"""
    payload = _formal_base(formal_module)
    payload["datasets"]["nltcs"]["judgment"] = None
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "非空映射" in result.stdout


def test_audit_rejects_judgment_recompute_mismatch(formal_module, tmp_path):
    """判定内容与无条件重算不一致 → FATAL"""
    payload = _formal_base(formal_module)
    payload["datasets"]["nltcs"]["judgment"]["classification"] = "forged"
    result = _run_audit(tmp_path, payload, "probe_residual_geometry_formal")
    assert result.returncode == 1
    assert "判定重算与记录不一致" in result.stdout


def test_audit_protected_manifest_field_matrix(formal_module, tmp_path):
    """规范协议清单每个受保护字段的删除/空值/篡改全部拒绝（五轮反例 2
    与验收矩阵）"""
    manifest = formal_module.canonical_protocol_manifest()
    for field in sorted(manifest):
        # 删除
        payload = _formal_base(formal_module)
        del payload["protocol"][field]
        result = _run_audit(
            tmp_path, payload, "probe_residual_geometry_formal",
        )
        assert result.returncode == 1, f"删除 {field} 未被拒绝"
        # 空值
        payload = _formal_base(formal_module)
        payload["protocol"][field] = None
        result = _run_audit(
            tmp_path, payload, "probe_residual_geometry_formal",
        )
        assert result.returncode == 1, f"{field}=null 未被拒绝"
        # 内容篡改
        payload = _formal_base(formal_module)
        payload["protocol"][field] = "tampered"
        result = _run_audit(
            tmp_path, payload, "probe_residual_geometry_formal",
        )
        assert result.returncode == 1, f"篡改 {field} 未被拒绝"


def test_audit_rejects_same_path_alias_symlink_hardlink(
    formal_module, tmp_path
):
    """输入输出同路径/相对别名/符号链接/硬链接全部拒绝且不改输入
    （五轮反例 3 + 不变量 4）"""
    import hashlib as _hashlib
    import subprocess as _sp

    src = tmp_path / "artifact.json"
    import json as _json
    src.write_text(_json.dumps(_formal_base(formal_module)))
    before = _hashlib.sha256(src.read_bytes()).hexdigest()

    def _audit_with_output(out):
        return _sp.run(
            [sys.executable, "scripts/audit_formal_json.py",
             "--protocol", "probe_residual_geometry_formal",
             "--json", str(src), "--output", str(out)],
            capture_output=True, text=True,
            env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
        )

    # 同路径
    r1 = _audit_with_output(src)
    assert r1.returncode != 0 and "拒绝审计" in (r1.stdout + r1.stderr)
    # 相对路径别名
    alias = tmp_path / "sub" / ".." / "artifact.json"
    r2 = _audit_with_output(alias)
    assert r2.returncode != 0
    # 符号链接
    link = tmp_path / "link.json"
    link.symlink_to(src)
    r3 = _audit_with_output(link)
    assert r3.returncode != 0
    # 硬链接
    hard = tmp_path / "hard.json"
    __import__("os").link(src, hard)
    r4 = _audit_with_output(hard)
    assert r4.returncode != 0
    after = _hashlib.sha256(src.read_bytes()).hexdigest()
    assert after == before, "任一拒绝场景不得修改输入文件"


def test_audit_failure_leaves_no_output(formal_module, tmp_path):
    """任一失败场景不得留下可被误认为成功的审计输出"""
    payload = _formal_base(formal_module)
    payload["datasets"]["nltcs"]["judgment"] = None
    import json as _json
    src = tmp_path / "bad.json"
    src.write_text(_json.dumps(payload))
    import subprocess as _sp
    result = _sp.run(
        [sys.executable, "scripts/audit_formal_json.py",
         "--protocol", "probe_residual_geometry_formal",
         "--json", str(src)],
        capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": "src:scripts"},
    )
    assert result.returncode == 1
    assert not (tmp_path / "bad.json.audited.json").exists()
    assert not list(tmp_path.glob("*.tmp"))
