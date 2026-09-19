"""plants 考卷生成器，训练卷全二阶格子，保留卷高阶加半空间。

设计拍板，
一，69 个 01 字段全部二阶格子逐格出等值计数题，2346 对乘 4 格共 9384 条，
    与 AIM 和 GSD 文献的全二阶边缘 workload 同口径，一阶不出题可精确边缘化，
二，保留考卷三阶四阶随机等值各 512 条加整数权重半空间 200 条共 1224 条，
    与训练卷语义零交集，只做评价不参与训练，
三，真值全部用 numpy 矢量化在真实表上数出来，再随机抽查与逐行求值对拍。
用法，./.venv/bin/python scripts/gen_plants_exam.py
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

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

HELDOUT_SEED = 20260919
NUM_HELDOUT_EQ = 1024
NUM_HELDOUT_HS = 200
QUANT = 1_000_000


def eq_query(qid, conditions, result):
    parts = [f"{c['attribute']} == {c['value']}" for c in conditions]
    return {
        "id": qid,
        "type": f"eq{len(conditions)}way",
        "expression": " AND ".join(parts),
        "conditions": list(conditions),
        "result": int(result),
    }


def gen_all_2way(schema, X):
    """全二阶格子，按实际值域出格，常值字段的对只出存在的格。"""
    out = []
    k = 0
    for a, b in combinations(range(schema.num_fields), 2):
        counts = np.bincount(X[:, a] * 2 + X[:, b], minlength=4)
        for va, vb in product(schema.domains[a], schema.domains[b]):
            k += 1
            conds = (
                {"attribute": schema.fields[a], "operator": "==", "value": va},
                {"attribute": schema.fields[b], "operator": "==", "value": vb},
            )
            out.append(eq_query(f"W2_{k:05d}", conds, counts[int(va) * 2 + int(vb)]))
    return out


def gen_random_eq(schema, X, order, count, rng, prefix, taken):
    """随机高阶等值题，规范形式去重，掩码矢量化数答案。"""
    out = []
    guard = 0
    while len(out) < count:
        guard += 1
        if guard > 100000:
            raise RuntimeError("高阶组合枯竭，检查条数设置")
        fields = sorted(int(j) for j in rng.choice(schema.num_fields, order, replace=False))
        values = [
            schema.domains[j][int(rng.integers(len(schema.domains[j])))] for j in fields
        ]
        conds = tuple(
            {"attribute": schema.fields[j], "operator": "==", "value": v}
            for j, v in zip(fields, values)
        )
        key = canonical_condition_set(QuerySpec("t", conds, 0.0))
        if key in taken:
            continue
        taken.add(key)
        mask = np.ones(len(X), dtype=bool)
        for j, v in zip(fields, values):
            mask &= X[:, j] == int(v)
        out.append(eq_query(f"{prefix}_{len(out) + 1:04d}", conds, mask.sum()))
    return out


def gen_halfspaces(schema, X, count, rng, prefix):
    """随机整数权重半空间题，阈值取真实分数分布的随机分位点。

    01 字段每取值独立正态权重乘一百万量化成整数，
    行总分即整数分表求和，不小于阈值才通过，无浮点歧义。
    """
    out = []
    for t in range(count):
        w = np.round(rng.normal(size=(schema.num_fields, 2)) * QUANT).astype(np.int64)
        scores = {
            name: {v: int(w[j, int(v)]) for v in schema.domains[j]}
            for j, name in enumerate(schema.fields)
        }
        # 行分即每字段按取值查表求和，01 字段等价于基础分加取 1 字段的差分
        row_scores = X.astype(np.int64) @ (w[:, 1] - w[:, 0]) + int(w[:, 0].sum())
        q = float(rng.uniform(0.15, 0.85))
        threshold = int(np.quantile(row_scores, q, method="higher"))
        cond = {"operator": "halfspace", "scores": scores, "threshold": threshold}
        out.append(
            {
                "id": f"{prefix}_{t + 1:04d}",
                "type": "halfspace",
                "expression": f"随机整数权重半空间，阈值分位点 {q:.3f}",
                "conditions": [cond],
                "result": int((row_scores >= threshold).sum()),
            }
        )
    return out


def spot_check(queries, schema, real_rows, rng, count=60):
    """随机抽查矢量化答案与逐行求值一致。"""
    picks = rng.choice(len(queries), size=min(count, len(queries)), replace=False)
    for i in picks:
        q = queries[int(i)]
        spec = QuerySpec(q["id"], tuple(q["conditions"]), 0.0)
        slow = sum(evaluate_query(spec, r, schema) for r in real_rows)
        if slow != q["result"]:
            raise AssertionError(f"{q['id']} 矢量化 {q['result']} 逐行 {slow} 不一致")
    return len(picks)


def dump(path, queries, num_rows, description, dataset):
    payload = {
        "dataset": dataset,
        "record_count": num_rows,
        "query_count": len(queries),
        "result_unit": "records",
        "description": description,
        "queries": queries,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="二阶格子考卷生成器")
    parser.add_argument("--data", type=str, default="plants", help="数据目录名，表名须同名")
    args = parser.parse_args()
    data_dir = DATA_ROOT / args.data
    schema, real_rows = load_table(str(data_dir / f"{args.data}.csv"))
    if any(any(v not in ("0", "1") for v in d) for d in schema.domains):
        raise AssertionError(f"{args.data} 字段取值必须落在 01 内")
    constant = [schema.fields[j] for j, d in enumerate(schema.domains) if len(d) == 1]
    if constant:
        print(f"常值字段 {len(constant)} 个 {constant}，其格子按实际值域出")
    X = np.array([[int(v) for v in r] for r in real_rows], dtype=np.int8)

    taken: set = set()
    two = gen_all_2way(schema, X)
    for q in two:
        taken.add(canonical_condition_set(QuerySpec("t", tuple(q["conditions"]), 0.0)))

    h_rng = np.random.default_rng(HELDOUT_SEED)
    h3 = gen_random_eq(schema, X, 3, NUM_HELDOUT_EQ // 2, h_rng, "H3", taken)
    h4 = gen_random_eq(schema, X, 4, NUM_HELDOUT_EQ // 2, h_rng, "H4", taken)
    hhs = gen_halfspaces(schema, X, NUM_HELDOUT_HS, h_rng, "HH")
    heldout = h3 + h4 + hhs

    m_specs = [QuerySpec(q["id"], tuple(q["conditions"]), float(q["result"])) for q in two]
    h_specs = [QuerySpec(q["id"], tuple(q["conditions"]), float(q["result"])) for q in heldout]
    assert_disjoint_workloads(m_specs, h_specs)

    check_rng = np.random.default_rng(HELDOUT_SEED + 1)
    checked = spot_check(two + heldout, schema, real_rows, check_rng)

    sums: dict[frozenset, int] = {}
    for q in two:
        key = frozenset(c["attribute"] for c in q["conditions"])
        sums[key] = sums.get(key, 0) + q["result"]
    if any(s != len(real_rows) for s in sums.values()):
        raise AssertionError("某字段对全格答案之和不等于行数")

    m_path = data_dir / f"measured_{len(two)}query.json"
    h_path = data_dir / f"heldout_{len(heldout)}query.json"
    dump(m_path, two, len(real_rows), "训练考卷，全二阶格子逐格出题，一阶不出题，与文献全二阶边缘同口径", f"{args.data}.csv")
    dump(h_path, heldout, len(real_rows), "保留考卷，评价专用，三阶四阶等值加整数权重半空间，与训练卷语义零交集", f"{args.data}.csv")
    print(f"训练卷 {len(two)} 条全二阶，字段对 {len(sums)} 个格子和均为 {len(real_rows)}")
    print(f"保留卷 {len(heldout)} 条，三阶 {len(h3)} 四阶 {len(h4)} 半空间 {len(hhs)}")
    print(f"抽查 {checked} 条矢量化与逐行求值一致")
    print(f"写入 {m_path.name} 与 {h_path.name}")


if __name__ == "__main__":
    main()
