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
    judgement = formal_module._judge(runs, "train")
    assert judgement["paired_wins"] == 5
    assert judgement["classification"] == "supports_relative_geometry"
    assert judgement["floor_suboptimal_flag"] is False


def test_judge_mixed_when_partial_wins(formal_module):
    base = [0.0010, 0.0011, 0.0010, 0.0009, 0.0010]
    cand = [0.0008, 0.0009, 0.0008, 0.0012, 0.0011]  # 3/5 胜
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs, "train")
    assert judgement["paired_wins"] == 3
    assert judgement["classification"] == "mixed"


def test_judge_not_supported_when_no_gain(formal_module):
    base = [0.0010] * 5
    cand = [0.0012] * 5
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs, "train")
    assert judgement["classification"] == "not_supported"


def test_judge_supports_requires_min_improvement(formal_module):
    """5/5 胜但均值改善 <30% → mixed 而非 supports"""
    base = [0.0010] * 5
    cand = [0.00085] * 5  # 改善 15%
    runs = _fake_runs(base, cand, floors=_full_floor_arms(cand))
    judgement = formal_module._judge(runs, "train")
    assert judgement["paired_wins"] == 5
    assert judgement["classification"] == "mixed"


def test_judge_quality_risk_downgrades(formal_module):
    base = [0.0010] * 5
    cand = [0.0003] * 5
    runs = _fake_runs(
        base, cand, floors=_full_floor_arms(cand), offline_rel=0.10,
    )
    judgement = formal_module._judge(runs, "train")
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
    judgement = formal_module._judge(runs, "train")
    assert judgement["floor_best_arm"] == "relative_f4"
    assert judgement["floor_suboptimal_flag"] is True
    # 不改变主分类
    assert judgement["classification"] == "supports_relative_geometry"
