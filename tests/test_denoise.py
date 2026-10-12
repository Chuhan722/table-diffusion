"""测量后处理（平分清洗）单元测试。"""
import numpy as np

from resevo.dataset import QuerySpec, TableSchema
from resevo.denoise import pool_oneway, pool_pairs, pool_vector


def test_pool_vector_rules():
    sigma = np.full(5, 10.0)
    # 全部测得出：恒等（含负值原样，交给下游）
    y = np.array([100.0, 50.0, 30.0, 25.0, -21.0])
    assert np.array_equal(pool_vector(y, sigma, 300.0, 2.0), y)
    # 两格未测出：保留格原样，剩余均分，总量守恒
    y = np.array([100.0, 50.0, 30.0, 5.0, -15.0])
    out = pool_vector(y, sigma, 300.0, 2.0)
    assert np.allclose(out[:3], [100.0, 50.0, 30.0])
    assert np.allclose(out[3:], [60.0, 60.0])
    assert np.isclose(out.sum(), 300.0)
    # 权重分配：按权重比例
    out = pool_vector(y, sigma, 300.0, 2.0, weights=np.array([1, 1, 1, 3.0, 1.0]))
    assert np.allclose(out[3:], [90.0, 30.0])
    # 保留量超总数：退回欧氏投影，非负且和为总数
    y = np.array([200.0, 150.0, 30.0, 5.0, -15.0])
    out = pool_vector(y, sigma, 300.0, 2.0)
    assert out.min() >= 0.0 and np.isclose(out.sum(), 300.0)


def _specs():
    schema = TableSchema(("x", "y"), (("0", "1", "2"), ("a", "b")))
    specs = []
    one = {("x", "0"): 70.0, ("x", "1"): 3.0, ("x", "2"): -8.0, ("y", "a"): 60.0, ("y", "b"): 40.0}
    for (f, v), r in one.items():
        specs.append(QuerySpec(f"{f}{v}", ({"attribute": f, "operator": "==", "value": v},), r, 10.0))
    two = {("0", "a"): 45.0, ("0", "b"): 25.0, ("1", "a"): 4.0, ("1", "b"): -3.0, ("2", "a"): 12.0, ("2", "b"): 2.0}
    for (a, b), r in two.items():
        specs.append(QuerySpec(
            f"p{a}{b}",
            ({"attribute": "x", "operator": "==", "value": a}, {"attribute": "y", "operator": "==", "value": b}),
            r, 5.0,
        ))
    return schema, specs


def test_pool_oneway_and_pairs():
    schema, specs = _specs()
    new, ones, st = pool_oneway(specs, schema, 100, 2.0)
    assert st == {"fields": 2, "cells": 5, "pooled": 2}
    # x：70 保留，3 与 -8 未测出各分 15；y：全测得出恒等
    assert np.allclose(ones[0], [70.0, 15.0, 15.0]) and np.allclose(ones[1], [60.0, 40.0])
    got = {q.query_id: q.result for q in new}
    assert got["x1"] == 15.0 and got["x2"] == 15.0 and got["ya"] == 60.0
    # 二阶：|y|<10 的格子 (1,a)(1,b)(2,b) 未测出，剩余 100-45-25-12=18 按一阶乘积分
    new2, st2 = pool_pairs(new, schema, 100, 2.0, ones)
    assert st2 == {"pairs": 1, "cells": 6, "pooled": 3}
    got = {q.query_id: q.result for q in new2}
    w = {"p1a": 15 * 60, "p1b": 15 * 40, "p2b": 15 * 40}
    tot = sum(w.values())
    for k, v in w.items():
        assert np.isclose(got[k], 18.0 * v / tot)
    assert np.isclose(sum(got[k] for k in ("p0a", "p0b", "p1a", "p1b", "p2a", "p2b")), 100.0)
    # 未动的题对象原样保留
    assert new2[0] is new[0]


def test_pool_small_sigma_is_identity():
    schema, specs = _specs()
    tiny = [QuerySpec(q.query_id, q.conditions, q.result, 0.01) for q in specs]
    new, ones, st = pool_oneway(tiny, schema, 100, 2.0)
    assert st["pooled"] == 0 and all(a.result == b.result for a, b in zip(new, tiny))
    new2, st2 = pool_pairs(new, schema, 100, 2.0, ones)
    assert st2["pooled"] == 0 and all(a.result == b.result for a, b in zip(new2, tiny))


def test_pool_oneway_scope_uncovered():
    schema, specs = _specs()
    # x、y 都出现在二阶题里 → uncovered 范围下一个字段都不清洗，题单原样
    new, ones, st = pool_oneway(specs, schema, 100, 2.0, scope="uncovered")
    assert st == {"fields": 0, "cells": 0, "pooled": 0} and ones == {} and all(a is b for a, b in zip(new, specs))
    # 去掉二阶题后 x 未被覆盖 → 清洗
    only_one = [q for q in specs if len(q.conditions) == 1]
    new, ones, st = pool_oneway(only_one, schema, 100, 2.0, scope="uncovered")
    assert st["fields"] == 2 and np.allclose(ones[0], [70.0, 15.0, 15.0])
