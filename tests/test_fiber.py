"""纤维洗牌平均（src/resevo/fiber.py）单元测试：守恒自检、与脚本版逐格一致、伪测量题式。"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from resevo.dataset import QuerySpec, TableSchema, load_queries, load_table
from resevo.fiber import (
    PSEUDO_PREFIX,
    encode_rows,
    fiber_average,
    fiber_specs,
    marg,
    measured_sets,
    neighbors,
    pseudo_sigma,
)

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    # 三字段 x,y,z：测过 (x,y) 二阶与 z 一阶，(x,z)、(y,z) 没测
    schema = TableSchema(("x", "y", "z"), (("0", "1"), ("a", "b", "c"), ("p", "q")))
    specs = []
    for vx in schema.domains[0]:
        for vy in schema.domains[1]:
            specs.append(QuerySpec(f"xy{vx}{vy}", (
                {"attribute": "x", "operator": "==", "value": vx},
                {"attribute": "y", "operator": "==", "value": vy}), 1.0, 4.0))
    for vz in schema.domains[2]:
        specs.append(QuerySpec(f"z{vz}", ({"attribute": "z", "operator": "==", "value": vz},), 1.0, 9.0))
    return schema, specs


def test_measured_sets_and_neighbors():
    schema, specs = _toy()
    sets = measured_sets(specs + [QuerySpec("P2_00001", specs[0].conditions, 1.0, 1.0)], schema)
    assert sets == [(0, 1), (2,)]
    assert neighbors(3, sets) == [[1], [0], []]


def test_fiber_average_preserves_measured_and_scrambles_unmeasured():
    schema, _ = _toy()
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, n)
    y = np.where(x == 1, rng.integers(1, 3, n), rng.integers(0, 2, n))  # y 依赖 x
    z = x.copy()  # z 与 x 完全相关，但 (x,z) 没测 → 洗后应接近独立
    X = np.stack([x, y, z], 1).astype(np.int64)
    doms = [2, 3, 2]
    sets = [(0, 1), (2,)]
    before_xy, before_z = marg(X, (0, 1), doms), marg(X, (2,), doms)
    X0 = X.copy()
    res = fiber_average(X, doms, sets, passes=6, rng=np.random.default_rng(1))
    assert res.unmeasured_pairs == [(0, 2), (1, 2)] and res.measured_pairs == [(0, 1)] and res.averaged == 5
    assert np.array_equal(marg(X, (0, 1), doms), before_xy) and np.array_equal(marg(X, (2,), doms), before_z)
    # 没测的 (x,z)：起点完全相关（对角占满），平均后接近独立乘积
    avg = res.averages[(0, 2)].reshape(2, 2)
    indep = np.outer(marg(X0, (0,), doms), marg(X0, (2,), doms)) / n
    assert np.abs(avg - indep).sum() / n < 0.05 and np.abs(marg(X0, (0, 2), doms).reshape(2, 2) - indep).sum() / n > 0.4
    assert np.isclose(avg.sum(), n)


def test_fiber_specs_and_sigma():
    schema, specs = _toy()
    assert pseudo_sigma(specs) == 4.0 and pseudo_sigma(specs, 1.5) == 6.0
    assert pseudo_sigma([q for q in specs if len(q.conditions) == 1]) is None
    X = np.zeros((10, 3), dtype=np.int64)
    res = fiber_average(X, [2, 3, 2], [(0, 1), (2,)], passes=2, rng=np.random.default_rng(0))
    out = fiber_specs(res, schema, 4.0)
    assert len(out) == 2 * 2 + 3 * 2 and out[0].query_id == f"{PSEUDO_PREFIX}00001" and out[-1].query_id == f"{PSEUDO_PREFIX}00010"
    assert out[0].conditions[0]["attribute"] == "x" and out[0].conditions[1] == {"attribute": "z", "operator": "==", "value": "p"}
    assert all(q.sigma == 4.0 for q in out) and np.isclose(sum(q.result for q in out[:4]), 10.0)


def test_matches_script_version_on_real_table():
    """与 scripts/fiber_shuffle.py 的逐步实现同种子同表逐格一致（编码无关）。"""
    table = ROOT / "results/stage2/adult_eps0p1_hybridp1u_seed1.csv"
    exam = sorted((ROOT / "data/adult_aimrep0p1_s1").glob("measured_*query.json"))
    if not table.exists() or not exam:
        import pytest

        pytest.skip("缺 adult 收敛表或考卷")
    spec = importlib.util.spec_from_file_location("fiber_shuffle", ROOT / "scripts/fiber_shuffle.py")
    fs = importlib.util.module_from_spec(spec)
    sys.modules["fiber_shuffle"] = fs
    spec.loader.exec_module(fs)

    schema, rows = load_table(str(table))
    specs = load_queries(str(exam[0]))
    sets = measured_sets(specs, schema)
    doms = [len(d) for d in schema.domains]
    X = encode_rows(rows, schema)
    res = fiber_average(X, doms, sets, passes=3, rng=np.random.default_rng(1))

    # 脚本版：整数取值矩阵 + 字段名集合
    V = np.array([[int(v) for v in r] for r in rows], dtype=np.int64)
    sets_named = [tuple(schema.fields[j] for j in s) for s in sets]
    nb = fs.neighbors_from_sets(list(schema.fields), sets_named)
    doms_v = [int(V[:, j].max()) + 1 for j in range(V.shape[1])]
    rng = np.random.default_rng(1)
    acc = {p: np.zeros(doms_v[p[0]] * doms_v[p[1]]) for p in res.unmeasured_pairs}
    for t in range(1, 4):
        for a in rng.permutation(V.shape[1]):
            fs.shuffle_column(V, int(a), fs.group_ids(V, nb[int(a)]), rng)
        if t >= 2:
            for p in acc:
                acc[p] += fs.marg(V, p, doms_v)
    for (a, b), m in res.averages.items():
        ours = m.reshape(doms[a], doms[b])
        theirs = (acc[(a, b)] / 2).reshape(doms_v[a], doms_v[b])
        # 编码下标 → 整数取值
        ia = [int(v) for v in schema.domains[a]]
        ib = [int(v) for v in schema.domains[b]]
        assert np.array_equal(ours, theirs[np.ix_(ia, ib)])
