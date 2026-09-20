"""初始化模块测试，一阶边缘化对拍真实表与按比例抽样分布。"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from resevo.dataset import QuerySpec, load_queries, load_table
from resevo.initialization import (
    _atom_signature,
    _atom_values,
    clean_first_order,
    derive_first_order,
    sample_initial_rows,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_300x10"


@pytest.fixture(scope="module")
def loaded():
    schema, rows = load_table(str(DATA_DIR / "test_300x10.csv"))
    specs = load_queries(str(DATA_DIR / "measured_698query.json"))
    return schema, rows, specs


def test_derive_first_order_matches_real_table(loaded):
    """二阶边缘化出的一阶原子计数逐个等于真实表直接统计。"""
    schema, rows, specs = loaded
    marginals = derive_first_order(specs, schema, len(rows))
    assert len(marginals) == schema.num_fields
    for j, atoms in enumerate(marginals):
        total = 0
        for cond, count in atoms:
            covered = set(_atom_values(cond, schema.domains[j]))
            true_count = sum(1 for r in rows if r[j] in covered)
            assert count == true_count, f"字段 {schema.fields[j]} 原子 {cond} 计数不符"
            total += count
        assert total == len(rows)


def test_sample_initial_rows_proportions(loaded):
    """抽样起点表的原子比例贴近边缘化计数，字段值都在域内。"""
    schema, rows, specs = loaded
    marginals = derive_first_order(specs, schema, len(rows))
    rng = np.random.default_rng(7)
    n = 6000
    sampled = sample_initial_rows(marginals, schema, n, rng)
    assert len(sampled) == n
    for j, atoms in enumerate(marginals):
        domain = set(schema.domains[j])
        assert all(r[j] in domain for r in sampled)
        for cond, count in atoms:
            covered = set(_atom_values(cond, schema.domains[j]))
            got = sum(1 for r in sampled if r[j] in covered) / n
            want = count / len(rows)
            assert abs(got - want) < 0.03, f"字段 {schema.fields[j]} 原子比例偏差过大"


def test_derive_rejects_incomplete_coverage():
    """覆盖不全的考卷拒绝边缘化，防止静默错比例。"""
    from resevo.dataset import TableSchema

    schema = TableSchema(("x", "y"), (("0", "1"), ("a", "b")))
    specs = [
        QuerySpec("q1", (
            {"attribute": "x", "operator": "==", "value": "0"},
            {"attribute": "y", "operator": "==", "value": "a"},
        ), 3.0),
    ]
    with pytest.raises(ValueError):
        derive_first_order(specs, schema, 10)


def test_atom_signature_normalizes():
    """同语义不同写法的原子签名一致。"""
    a = {"attribute": "age", "operator": "between", "lower": 18, "upper": 24}
    b = {"attribute": "age", "operator": "between", "lower": 18.0, "upper": 24.0}
    assert _atom_signature(a) == _atom_signature(b)


def test_derive_first_order_noisy_tolerance():
    """噪声考卷浮点答案总和偏离行数，容差放宽可过，默认精确拒绝。"""
    from resevo.dataset import TableSchema

    schema = TableSchema(("x", "y"), (("0", "1"), ("a", "b")))
    cells = [("0", "a", 2.7), ("0", "b", 3.4), ("1", "a", 1.9), ("1", "b", 2.3)]
    specs = [
        QuerySpec(f"q{i}", (
            {"attribute": "x", "operator": "==", "value": vx},
            {"attribute": "y", "operator": "==", "value": vy},
        ), r)
        for i, (vx, vy, r) in enumerate(cells)
    ]
    with pytest.raises(ValueError):
        derive_first_order(specs, schema, 10)
    marginals = derive_first_order(specs, schema, 10, tol_rows=0.05)
    got_x = {(_atom_signature(c)): v for c, v in marginals[0]}
    key0 = _atom_signature({"attribute": "x", "operator": "==", "value": "0"})
    assert abs(got_x[key0] - (2.7 + 3.4)) < 1e-12


def test_cdp_rho_roundtrip():
    """cdp_rho 换算回代 delta 自洽，锚住与官方 snsynth 同源的实现。"""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_noisy_exam import _cdp_delta, cdp_rho

    rho = cdp_rho(1.0, 1e-5)
    assert 0.02 < rho < 0.05
    assert abs(_cdp_delta(rho, 1.0) - 1e-5) < 1e-9
    assert cdp_rho(2.0, 1e-5) > rho


def test_clean_first_order_noisy_counts():
    """负计数截零，总和恰归行数，等额摊符合单纯形投影手算。"""
    cond = {"attribute": "x", "operator": "==", "value": "0"}
    marginals = [[(cond, -2.0), (cond, 4.0), (cond, 6.0)]]
    cleaned = clean_first_order(marginals, 10)
    vals = np.array([v for _, v in cleaned[0]])
    assert (vals >= 0.0).all()
    assert abs(vals.sum() - 10.0) < 1e-9
    assert np.allclose(vals, [0.0, 4.0, 6.0])


def test_clean_first_order_keeps_clean_input():
    """已在单纯形内的干净计数投影后逐位不变。"""
    cond = {"attribute": "x", "operator": "==", "value": "0"}
    marginals = [[(cond, 3.0), (cond, 7.0)]]
    cleaned = clean_first_order(marginals, 10)
    assert np.allclose([v for _, v in cleaned[0]], [3.0, 7.0])
