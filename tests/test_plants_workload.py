"""plants workload v2 的确定性检查（PR #61 审查意见 4）。

锚定：查询总数、类型构成、唯一性、成组完整性、答案复算、
三属性取值模式覆盖、计数阈值。数据与 workload 文件均入库，
本测试完全确定性。
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD = ROOT / "configs/plants/measured_1000query.json"
DATA = ROOT / "data/plants/plants.csv"


@pytest.fixture(scope="module")
def workload():
    return json.loads(WORKLOAD.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frame():
    return pd.read_csv(DATA)


def test_counts_and_composition(workload):
    queries = workload["queries"]
    assert workload["query_count"] == len(queries) == 980
    types = Counter(q["type"] for q in queries)
    assert types == {"double": 460, "triple": 520}
    assert workload["workload_version"] == 2
    assert workload["record_count"] == 16181 + 1231  # 17412


def test_no_single_queries(workload):
    """1-way 不入主 workload（由 init_marginals 初始化层提供）"""
    assert all(len(q["conditions"]) >= 2 for q in workload["queries"])


def test_uniqueness(workload):
    keys = [
        tuple(sorted(
            (c["attribute"], c["value"]) for c in q["conditions"]
        ))
        for q in workload["queries"]
    ]
    assert len(keys) == len(set(keys)), "存在重复查询"


def test_double_groups_are_complete_marginals(workload):
    """每个入选属性对恰好 4 个 cell（完整 2×2 成组）且模式互异"""
    groups = {}
    for q in workload["queries"]:
        if q["type"] != "double":
            continue
        attrs = tuple(sorted(c["attribute"] for c in q["conditions"]))
        pattern = tuple(
            c["value"] for c in sorted(
                q["conditions"], key=lambda c: c["attribute"]
            )
        )
        groups.setdefault(attrs, set()).add(pattern)
    assert len(groups) == 115
    for attrs, patterns in groups.items():
        assert patterns == {(0, 0), (0, 1), (1, 0), (1, 1)}, attrs


def test_triple_groups_two_distinct_patterns(workload):
    """每个三元组恰好 2 条且取值模式不同"""
    groups = {}
    for q in workload["queries"]:
        if q["type"] != "triple":
            continue
        attrs = tuple(sorted(c["attribute"] for c in q["conditions"]))
        pattern = tuple(
            c["value"] for c in sorted(
                q["conditions"], key=lambda c: c["attribute"]
            )
        )
        groups.setdefault(attrs, []).append(pattern)
    assert len(groups) == 260
    for attrs, patterns in groups.items():
        assert len(patterns) == 2 and patterns[0] != patterns[1], attrs


def test_triple_pattern_coverage(workload):
    """三属性取值模式覆盖：无 000 塌缩（v1 缺陷），≥6 种模式，
    单一模式占比 ≤60%"""
    patterns = Counter()
    for q in workload["queries"]:
        if q["type"] != "triple":
            continue
        pattern = "".join(str(c["value"]) for c in sorted(
            q["conditions"], key=lambda c: c["attribute"]
        ))
        patterns[pattern] += 1
    assert "000" not in patterns
    assert len(patterns) >= 6
    assert max(patterns.values()) / sum(patterns.values()) <= 0.60


def test_answers_recompute_exactly(workload, frame):
    """全部 980 条答案与源数据逐条精确复算一致"""
    mat = frame.to_numpy()
    cols = {a: i for i, a in enumerate(frame.columns)}
    for q in workload["queries"]:
        mask = np.ones(len(frame), dtype=bool)
        for c in q["conditions"]:
            mask &= mat[:, cols[c["attribute"]]] == c["value"]
        assert int(mask.sum()) == q["result"], q["id"]


def test_triple_min_count_threshold(workload):
    """triple cell 计数不低于阈值 30（double 成组不筛，完整 marginal
    语义优先）"""
    for q in workload["queries"]:
        if q["type"] == "triple":
            assert q["result"] >= 30, q["id"]
