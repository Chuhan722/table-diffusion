"""多表逆方差加权平均一阶边缘化测试，零噪声一致性，加权正确性，缺格防御。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from resevo.dataset import QuerySpec, TableSchema, load_queries, load_table
from resevo.initialization import (
    _atom_signature,
    derive_first_order,
    derive_first_order_avg,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_300x10"


def _eq(attr, val):
    return {"attribute": attr, "operator": "==", "value": val}


def test_avg_matches_greedy_on_clean_exam():
    """零噪声考卷每张表都精确，平均与贪心逐原子近似一致。"""
    schema, rows = load_table(str(DATA_DIR / "test_300x10.csv"))
    specs = load_queries(str(DATA_DIR / "measured_698query.json"))
    greedy = derive_first_order(specs, schema, len(rows))
    avg = derive_first_order_avg(specs, schema, len(rows))
    assert len(avg) == len(greedy)
    for atoms_g, atoms_a in zip(greedy, avg):
        sig_g = {_atom_signature(c): v for c, v in atoms_g}
        sig_a = {_atom_signature(c): v for c, v in atoms_a}
        assert sig_g.keys() == sig_a.keys()
        for k in sig_g:
            assert sig_a[k] == pytest.approx(sig_g[k], abs=1e-9)


def test_avg_inverse_variance_weighting():
    """两张伙伴域大小不同的带偏表，平均值等于手算逆方差加权。"""
    schema = TableSchema(
        ("x", "y", "z"), (("0", "1"), ("a", "b"), ("p", "q", "r", "s")),
    )
    specs = [
        QuerySpec("xy00", (_eq("x", "0"), _eq("y", "a")), 5.0),
        QuerySpec("xy01", (_eq("x", "0"), _eq("y", "b")), 3.0),
        QuerySpec("xy10", (_eq("x", "1"), _eq("y", "a")), 2.0),
        QuerySpec("xy11", (_eq("x", "1"), _eq("y", "b")), 2.0),
        QuerySpec("xz0p", (_eq("x", "0"), _eq("z", "p")), 1.0),
        QuerySpec("xz0q", (_eq("x", "0"), _eq("z", "q")), 1.0),
        QuerySpec("xz0r", (_eq("x", "0"), _eq("z", "r")), 1.0),
        QuerySpec("xz0s", (_eq("x", "0"), _eq("z", "s")), 1.0),
        QuerySpec("xz1p", (_eq("x", "1"), _eq("z", "p")), 2.0),
        QuerySpec("xz1q", (_eq("x", "1"), _eq("z", "q")), 2.0),
        QuerySpec("xz1r", (_eq("x", "1"), _eq("z", "r")), 1.0),
        QuerySpec("xz1s", (_eq("x", "1"), _eq("z", "s")), 1.0),
    ]
    avg = derive_first_order_avg(specs, schema, 10)
    x_counts = {_atom_signature(c): v for c, v in avg[0]}
    # x0 两份估计 8 与 4，权重 1/2 与 1/4，加权 (4+1)/0.75
    assert x_counts[("==", "0")] == pytest.approx((0.5 * 8 + 0.25 * 4) / 0.75)
    assert x_counts[("==", "1")] == pytest.approx((0.5 * 4 + 0.25 * 6) / 0.75)


def test_avg_skips_incomplete_table():
    """缺格表被跳过不进平均，结果等于唯一完整表。"""
    schema = TableSchema(
        ("x", "y", "z"), (("0", "1"), ("a", "b"), ("p", "q")),
    )
    specs = [
        QuerySpec("xy00", (_eq("x", "0"), _eq("y", "a")), 5.0),
        QuerySpec("xy01", (_eq("x", "0"), _eq("y", "b")), 3.0),
        QuerySpec("xy10", (_eq("x", "1"), _eq("y", "a")), 2.0),
        QuerySpec("xy11", (_eq("x", "1"), _eq("y", "b")), 2.0),
        QuerySpec("xz0p", (_eq("x", "0"), _eq("z", "p")), 4.0),
        QuerySpec("xz0q", (_eq("x", "0"), _eq("z", "q")), 4.0),
    ]
    avg = derive_first_order_avg(specs, schema, 10)
    x_counts = {_atom_signature(c): v for c, v in avg[0]}
    assert x_counts[("==", "0")] == pytest.approx(8.0)
    assert x_counts[("==", "1")] == pytest.approx(4.0)


def test_avg_rejects_uncovered_field():
    """无任何二阶配对覆盖的字段拒绝平均边缘化。"""
    schema = TableSchema(("x", "y", "w"), (("0", "1"), ("a", "b"), ("0", "1")))
    specs = [
        QuerySpec("xy00", (_eq("x", "0"), _eq("y", "a")), 5.0),
        QuerySpec("xy01", (_eq("x", "0"), _eq("y", "b")), 3.0),
        QuerySpec("xy10", (_eq("x", "1"), _eq("y", "a")), 2.0),
        QuerySpec("xy11", (_eq("x", "1"), _eq("y", "b")), 2.0),
    ]
    with pytest.raises(ValueError):
        derive_first_order_avg(specs, schema, 10)
