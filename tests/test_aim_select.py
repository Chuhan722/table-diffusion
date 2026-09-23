"""AIM 选择外壳组件测试，投影器与考卷写出与指数机制与候选权重。"""
import importlib.util
import json
from pathlib import Path

import numpy as np

from resevo.dataset import TableSchema, load_queries


def _load_shell():
    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_aim_select", root / "scripts" / "run_aim_select.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _schema():
    return TableSchema(("a", "b", "c"), (("0", "1"), ("0", "1", "2"), ("x", "y")))


def test_projector_counts_match_manual():
    """投影计数与手工统计一致，展平序为值序 C-order。"""
    sh = _load_shell()
    rows = [("0", "1", "x"), ("0", "1", "y"), ("1", "2", "x"), ("0", "0", "x")]
    p = sh.TableProjector(_schema())
    p.encode(rows)
    v = p.project(("a", "b"))
    assert v.shape == (6,)
    assert v[0 * 3 + 1] == 2.0
    assert v[1 * 3 + 2] == 1.0
    assert v[0 * 3 + 0] == 1.0
    assert v.sum() == 4.0
    v1 = p.project(("c",))
    assert list(v1) == [3.0, 1.0]


def test_write_exam_roundtrip():
    """考卷写出后 load_queries 读回，题数与答案与条件逐位对上。"""
    sh = _load_shell()
    p = sh.TableProjector(_schema())
    meas = [
        {"clique": ("a",), "y": np.array([3.0, -1.5]), "sigma": 2.0},
        {"clique": ("a", "c"), "y": np.arange(4.0), "sigma": 1.0},
    ]
    out = Path("/tmp/test_aimsel_exam.json")
    sh.write_exam(out, meas, p, 4, "测试卷")
    specs = load_queries(str(out))
    assert len(specs) == 2 + 4
    assert specs[1].result == -1.5
    assert specs[1].conditions[0] == {"attribute": "a", "operator": "==", "value": "1"}
    last = specs[-1]
    assert last.result == 3.0
    assert last.conditions == (
        {"attribute": "a", "operator": "==", "value": "1"},
        {"attribute": "c", "operator": "==", "value": "y"},
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["record_count"] == 4
    assert payload["result_unit"] == "records"


def test_compile_workload_weights():
    """downward closure 权重照官方语义，交集大小加权求和。"""
    sh = _load_shell()
    wl = [(("a", "b"), 1.0), (("a", "c"), 1.0), (("b", "c"), 1.0)]
    cand = sh.compile_workload(wl)
    assert cand[("a",)] == 2.0
    assert cand[("a", "b")] == 4.0
    assert len(cand) == 3 + 3


def test_exponential_mechanism_prefers_max():
    """epsilon 大时几乎必选最大效用，固定种子可复现。"""
    sh = _load_shell()
    q = {"lo": 0.0, "mid": 5.0, "hi": 10.0}
    picks = {
        sh.exponential_mechanism(q, 100.0, 1.0, np.random.default_rng(s))
        for s in range(10)
    }
    assert picks == {"hi"}
    a = sh.exponential_mechanism(q, 0.5, 1.0, np.random.default_rng(7))
    b = sh.exponential_mechanism(q, 0.5, 1.0, np.random.default_rng(7))
    assert a == b


def test_noise_floor_mixed_sigma():
    """混合 sigma 地板逐题累加，0.5 乘格子数乘方差。"""
    sh = _load_shell()
    meas = [
        {"clique": ("a",), "y": np.zeros(2), "sigma": 3.0},
        {"clique": ("b",), "y": np.zeros(4), "sigma": 0.5},
    ]
    assert sh.noise_floor(meas) == 0.5 * (2 * 9.0 + 4 * 0.25)
