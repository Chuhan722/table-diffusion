"""test300 新考卷生成器，全二阶格子加高阶加半空间，一阶不出题。

设计拍板，
一，age 分五箱，其余九个分类字段用原值，
二，全部二阶格子逐格出等值计数题共 548 条，一阶真值可由其精确边缘化，
三，三阶四阶各 50 条随机字段组合等值题，
四，半空间 50 条，字段数值化取随机权重，权重乘一百万量化成整数，
    行的整数总分不小于阈值才通过，阈值取真实分数分布的随机分位点，
五，保留考卷等值高阶 1024 条加半空间 200 条，与训练卷语义零交集，
真值全部由 evaluate_query 在真实表上逐行求和得出。
用法，./.venv/bin/python scripts/gen_test300_exam.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.dataset import QuerySpec, evaluate_query, load_table  # noqa: E402
from resevo.metrics import (  # noqa: E402
    assert_disjoint_workloads,
    canonical_condition_set,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_300x10"

AGE_BINS = ((18, 24), (25, 34), (35, 49), (50, 64), (65, 200))

MEASURED_SEED = 20260917
HELDOUT_SEED = 20260918
NUM_3WAY = 50
NUM_4WAY = 50
NUM_HS = 50
NUM_HELDOUT_EQ = 1024
NUM_HELDOUT_HS = 200
QUANT = 1_000_000


def field_atoms(schema, j):
    """字段的原子条件列表，age 是五个分箱，其余是逐值等值。"""
    name = schema.fields[j]
    if name == "age":
        return [
            {"attribute": "age", "operator": "between", "lower": lo, "upper": hi}
            for lo, hi in AGE_BINS
        ]
    return [
        {"attribute": name, "operator": "==", "value": v}
        for v in schema.domains[j]
    ]


def answer(conditions, real_rows, schema):
    spec = QuerySpec("tmp", tuple(conditions), 0.0)
    return int(sum(evaluate_query(spec, r, schema) for r in real_rows))


def eq_query(qid, conditions, real_rows, schema):
    parts = []
    for c in conditions:
        if c["operator"] == "between":
            parts.append(f"{c['lower']} <= {c['attribute']} <= {c['upper']}")
        else:
            parts.append(f"{c['attribute']} == {c['value']}")
    return {
        "id": qid,
        "type": f"eq{len(conditions)}way",
        "expression": " AND ".join(parts),
        "conditions": list(conditions),
        "result": answer(conditions, real_rows, schema),
    }


def gen_all_2way(schema, real_rows):
    out = []
    k = 0
    for a, b in combinations(range(schema.num_fields), 2):
        for ca, cb in product(field_atoms(schema, a), field_atoms(schema, b)):
            k += 1
            out.append(eq_query(f"W2_{k:04d}", (ca, cb), real_rows, schema))
    return out


def gen_random_eq(schema, real_rows, order, count, rng, prefix, taken):
    """随机高阶等值题，规范形式去重且避开 taken 集合。"""
    out = []
    guard = 0
    while len(out) < count:
        guard += 1
        if guard > 100000:
            raise RuntimeError("高阶组合枯竭，检查条数设置")
        fields = sorted(rng.choice(schema.num_fields, size=order, replace=False))
        conds = []
        for j in fields:
            atoms = field_atoms(schema, int(j))
            conds.append(atoms[int(rng.integers(len(atoms)))])
        key = canonical_condition_set(QuerySpec("t", tuple(conds), 0.0))
        if key in taken:
            continue
        taken.add(key)
        out.append(eq_query(f"{prefix}_{len(out) + 1:04d}", conds, real_rows, schema))
    return out


def gen_halfspaces(schema, real_rows, count, rng, prefix):
    """随机整数权重半空间题，阈值取真实分数分布的随机分位点。

    age 按标准化数值乘单权重给分，分类字段每取值独立权重，
    全部分数乘一百万四舍五入成整数，查询定义即整数分表，无浮点歧义。
    """
    ages = np.array([float(r[schema.field_index("age")]) for r in real_rows])
    age_mean, age_std = float(ages.mean()), float(ages.std())
    out = []
    for t in range(count):
        scores: dict[str, dict[str, int]] = {}
        for j, name in enumerate(schema.fields):
            if name == "age":
                w = float(rng.normal())
                tab = {
                    v: int(round(w * (float(v) - age_mean) / age_std * QUANT))
                    for v in schema.domains[j]
                }
            else:
                tab = {
                    v: int(round(float(rng.normal()) * QUANT))
                    for v in schema.domains[j]
                }
            scores[name] = tab
        row_scores = np.array(
            [
                sum(scores[name][r[j]] for j, name in enumerate(schema.fields))
                for r in real_rows
            ],
            dtype=np.int64,
        )
        q = float(rng.uniform(0.15, 0.85))
        threshold = int(np.quantile(row_scores, q, method="higher"))
        cond = {"operator": "halfspace", "scores": scores, "threshold": threshold}
        result = int((row_scores >= threshold).sum())
        out.append(
            {
                "id": f"{prefix}_{t + 1:04d}",
                "type": "halfspace",
                "expression": f"随机整数权重半空间，阈值分位点 {q:.3f}",
                "conditions": [cond],
                "result": result,
            }
        )
    return out


def dump(path, queries, real_rows, description):
    payload = {
        "dataset": "test_300x10.csv",
        "record_count": len(real_rows),
        "query_count": len(queries),
        "result_unit": "records",
        "description": description,
        "queries": queries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def main():
    schema, real_rows = load_table(str(DATA_DIR / "test_300x10.csv"))

    m_rng = np.random.default_rng(MEASURED_SEED)
    taken: set = set()
    two = gen_all_2way(schema, real_rows)
    for q in two:
        taken.add(canonical_condition_set(QuerySpec("t", tuple(q["conditions"]), 0.0)))
    three = gen_random_eq(schema, real_rows, 3, NUM_3WAY, m_rng, "W3", taken)
    four = gen_random_eq(schema, real_rows, 4, NUM_4WAY, m_rng, "W4", taken)
    hs = gen_halfspaces(schema, real_rows, NUM_HS, m_rng, "WH")
    measured = two + three + four + hs

    h_rng = np.random.default_rng(HELDOUT_SEED)
    h3 = gen_random_eq(schema, real_rows, 3, NUM_HELDOUT_EQ // 2, h_rng, "H3", taken)
    h4 = gen_random_eq(schema, real_rows, 4, NUM_HELDOUT_EQ // 2, h_rng, "H4", taken)
    hhs = gen_halfspaces(schema, real_rows, NUM_HELDOUT_HS, h_rng, "HH")
    heldout = h3 + h4 + hhs

    m_specs = [QuerySpec(q["id"], tuple(q["conditions"]), float(q["result"])) for q in measured]
    h_specs = [QuerySpec(q["id"], tuple(q["conditions"]), float(q["result"])) for q in heldout]
    assert_disjoint_workloads(m_specs, h_specs)

    m_path = DATA_DIR / f"measured_{len(measured)}query.json"
    h_path = DATA_DIR / f"heldout_{len(heldout)}query.json"
    dump(
        m_path, measured, real_rows,
        "训练考卷，全二阶格子加三阶四阶随机等值加整数权重半空间，一阶不出题",
    )
    dump(
        h_path, heldout, real_rows,
        "保留考卷，评价专用，等值高阶加半空间与训练卷语义零交集",
    )
    print(f"训练卷 {len(measured)} 条，二阶 {len(two)} 三阶 {len(three)} 四阶 {len(four)} 半空间 {len(hs)}")
    print(f"保留卷 {len(heldout)} 条，三阶 {len(h3)} 四阶 {len(h4)} 半空间 {len(hhs)}")
    print(f"写入 {m_path.name} 与 {h_path.name}")


if __name__ == "__main__":
    main()
