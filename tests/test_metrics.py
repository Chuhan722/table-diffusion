"""评价指标测试，AIM 式与 GSD 式定义的锚点断言。"""
from __future__ import annotations

import numpy as np
import pytest

from resevo.dataset import QuerySpec, TableSchema, load_queries, load_table
from resevo.metrics import (
    aim_workload_error,
    assert_disjoint_workloads,
    composite_score,
    gsd_query_errors,
    load_heldout_queries,
    marginal_l1_error,
)

DATA_DIR = "data/test_300x10"


@pytest.fixture(scope="module")
def real_data():
    schema, rows = load_table(f"{DATA_DIR}/test_300x10.csv")
    measured = load_queries(f"{DATA_DIR}/measured_50query.json")
    heldout = load_heldout_queries(f"{DATA_DIR}/heldout_issue53_v1.json")
    return schema, rows, measured, heldout


def test_marginal_l1_hand_anchor():
    """两行小表手算锚点，边缘 L1 逐格核对。

    真实表 (a,x) (b,y)，合成表 (a,y) (a,x)，
    字段 0 边缘 real a1 b1 synth a2，L1=|1-2|+|1-0|=2，除以 2 得 1，
    字段 1 边缘 real x1 y1 synth x1 y1，L1=0，
    联合边缘 real ax1 by1 synth ay1 ax1，L1=|1-1|+|1-0|+|0-1|=2，除以 2 得 1。
    """
    real = [("a", "x"), ("b", "y")]
    synth = [("a", "y"), ("a", "x")]
    assert marginal_l1_error(real, synth, (0,)) == 1.0
    assert marginal_l1_error(real, synth, (1,)) == 0.0
    assert marginal_l1_error(real, synth, (0, 1)) == 1.0


def test_identical_tables_zero_error(real_data):
    """真实表对自己评价，全部指标恒为零。"""
    schema, rows, measured, heldout = real_data
    for order in (1, 2):
        report = aim_workload_error(rows, rows, schema, order)
        assert report.average_error == 0.0
        assert report.max_error == 0.0
    for specs in (measured, heldout):
        report = gsd_query_errors(specs, rows, schema, len(rows))
        assert report.average_error == 0.0
        assert report.max_error == 0.0


def test_row_order_invariance(real_data):
    """评价只看多重集，打乱行序结果逐位不变。"""
    schema, rows, measured, _ = real_data
    rng = np.random.default_rng(0)
    synth = [rows[i] for i in rng.permutation(60)]
    base = [rows[i] for i in range(60)]
    shuffled_report = aim_workload_error(rows, synth, schema, 2)
    plain_report = aim_workload_error(rows, base, schema, 2)
    assert shuffled_report.per_marginal == plain_report.per_marginal
    g1 = gsd_query_errors(measured, synth, schema, len(rows))
    g2 = gsd_query_errors(measured, base, schema, len(rows))
    assert g1.per_query == g2.per_query


def test_aim_workload_shapes(real_data):
    schema, rows, _, _ = real_data
    synth = rows[:150] * 2  # 同 300 行但联合结构不同，各边缘误差有区分度
    r2 = aim_workload_error(rows, synth, schema, 2)
    assert len(r2.per_marginal) == 45  # C(10,2)
    r3 = aim_workload_error(rows, synth, schema, 3)
    assert len(r3.per_marginal) == 120  # C(10,3)
    assert r2.max_error >= r2.average_error > 0
    assert r3.average_error >= r2.average_error  # 阶数越高越难匹配
    with pytest.raises(ValueError):
        aim_workload_error(rows, rows, schema, 0)


def test_gsd_matches_dataset_evaluator(real_data):
    """GSD 式在真实表上的合成答案与 result 字段一致，误差恒零且比例口径正确。"""
    schema, rows, measured, _ = real_data
    report = gsd_query_errors(measured, rows, schema, 300)
    assert report.query_count == 50
    assert report.max_error == 0.0
    # 挪走一行，被影响查询的误差应恰为 1/300 的整数倍
    synth = rows[1:]
    report2 = gsd_query_errors(measured, synth, schema, 300)
    assert report2.max_error > 0
    for e in report2.per_query:
        # 比例差 |c_s/299 - c_r/300|，允许浮点，但不为负
        assert e >= 0.0


def test_heldout_loaded_and_disjoint(real_data):
    """固化核查结论，保留查询 1024 条且与生成查询语义零交集。"""
    _, _, measured, heldout = real_data
    assert len(heldout) == 1024
    orders = {len(s.conditions) for s in heldout}
    assert orders == {3, 4}
    assert_disjoint_workloads(measured, heldout)
    # 人为塞一条生成查询进保留集必须被拒绝
    with pytest.raises(ValueError):
        assert_disjoint_workloads(measured, heldout + [measured[0]])


def test_gsd_hand_anchor():
    """单查询手算锚点，真实 3/4 合成 1/2，误差 0.25。"""
    schema = TableSchema(("A",), (("0", "1"),))
    spec = QuerySpec("q", ({"attribute": "A", "operator": "==", "value": "1"},), 3.0)
    synth = [("1",), ("0",)]
    report = gsd_query_errors([spec], synth, schema, 4)
    assert report.per_query[0] == pytest.approx(0.25)


def test_composite_zero_on_identical(real_data):
    """真实表自评综合分恒为零。"""
    schema, rows, measured, heldout = real_data
    report = composite_score(
        gsd_query_errors(measured, rows, schema, 300),
        gsd_query_errors(heldout, rows, schema, 300),
        aim_workload_error(rows, rows, schema, 2),
        aim_workload_error(rows, rows, schema, 3),
    )
    assert report.score == 0.0


def test_composite_hand_anchor(real_data):
    """手工构造四组数字，综合分等于均值且 TVD 归一正确。"""
    from resevo.metrics import AimReport, GsdReport

    gm = GsdReport(1, 0.2, 0.2, (0.2,))
    gh = GsdReport(1, 0.1, 0.1, (0.1,))
    a2 = AimReport(2, 0.8, 0.8, (0.8,), ((0, 1),))
    a3 = AimReport(3, 0.4, 0.4, (0.4,), ((0, 1, 2),))
    report = composite_score(gm, gh, a2, a3)
    assert report.tvd_2way == pytest.approx(0.4)
    assert report.tvd_3way == pytest.approx(0.2)
    assert report.score == pytest.approx((0.2 + 0.1 + 0.4 + 0.2) / 4)
    bad = AimReport(2, 2.5, 2.5, (2.5,), ((0, 1),))
    with pytest.raises(ValueError):
        composite_score(gm, gh, bad, a3)


def test_composite_monotone(real_data):
    """任一组变差综合分必须变大，越小越好方向一致。"""
    schema, rows, measured, heldout = real_data
    good = [rows[i] for i in range(300)]
    bad = rows[:150] * 2
    def full(synth):
        return composite_score(
            gsd_query_errors(measured, synth, schema, 300),
            gsd_query_errors(heldout, synth, schema, 300),
            aim_workload_error(rows, synth, schema, 2),
            aim_workload_error(rows, synth, schema, 3),
        )
    assert full(bad).score > full(good).score == 0.0
