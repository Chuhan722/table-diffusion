"""
为 plants 数据集生成查询工作负载（1001 个，与 nltcs workload 同规模）。

与 gen_nltcs_queries.py 的构成对比（跨数据集同预算口径，总量一致）：
- 单属性：138 个（69×2，客观上限拉满，覆盖各属性基本分布）
- 双属性：480 个——候选空间 C(69,2)×4=9384 远超 nltcs 的 480，
  **不再拉满**，按偏离独立假设的信息量 |观测−独立预期| 取 top 480
  （每个属性对最多 2 种取值组合，避免高度相关）；
- 三属性：补足到 1001（=383 个）——全组合 C(69,3)×8≈40 万次计数过重，
  候选池限制为"双属性信息量 top 30 属性"的三元组，同样按三阶偏离
  信息量筛选，每个属性三元组最多 2 种取值组合。

覆盖语义变化（有意保留）：nltcs 的 1001 查询近乎拉满其低阶组合空间，
plants 的 1001 查询只覆盖其组合空间的一小部分——低覆盖场景更接近
真实 DP 应用，且 workload 对本方法与第一层基线双侧对称，不影响
同信息条件公平性。

设计原则（沿用 nltcs 生成器）：
- 避免重复或逻辑高度相关的查询；
- 极低计数阈值：双属性 ≥50、三属性 ≥30（与 nltcs 相同，两数据集
  行数同量级 17412 vs 16181）；
- 固定 SEED（本脚本当前流程无随机抽样，保留字段仅为协议完整性）。
"""
import itertools
import json

import numpy as np
import pandas as pd

DATA_PATH = "data/plants/plants.csv"
OUT_PATH = "configs/plants/measured_1000query.json"
SEED = 0

TOTAL = 1001
N_SINGLE = 138            # 69×2 拉满
N_DOUBLE = 480            # 信息量 top（候选 9384）
MIN_COUNT = 50            # 双属性最小计数
MIN_COUNT_TRIPLE = 30     # 三属性最小计数
MAX_PER_PAIR = 2          # 每个属性对最多取值组合数
MAX_PER_TRIPLE = 2        # 每个属性三元组最多取值组合数
TRIPLE_ATTR_POOL = 30     # 三属性候选池：双属性信息量 top 属性数


def eval_count(df, conds):
    mask = np.ones(len(df), dtype=bool)
    for attr, val in conds:
        mask &= (df[attr].values == val)
    return int(mask.sum())


def main():
    df = pd.read_csv(DATA_PATH)
    N = len(df)
    attrs = list(df.columns)
    values = [0, 1]
    assert len(attrs) == 69, len(attrs)

    queries = []

    # ---------- 1. 单属性：69×2 全纳入 ----------
    single_count = 0
    for a in attrs:
        for v in values:
            c = eval_count(df, [(a, v)])
            queries.append({
                "id": f"S{single_count+1:03d}",
                "type": "single",
                "expression": f"{a} == {v}",
                "conditions": [
                    {"attribute": a, "operator": "==", "value": v},
                ],
                "result": c,
            })
            single_count += 1
    print(f"单属性查询: {single_count} 个（69 属性 × 2 取值）")

    # ---------- 2. 双属性：信息量 top 480 ----------
    p1 = {a: (df[a].values == 1).mean() for a in attrs}
    double_candidates = []
    for a, b in itertools.combinations(attrs, 2):
        for va in values:
            for vb in values:
                c = eval_count(df, [(a, va), (b, vb)])
                if c < MIN_COUNT:
                    continue
                pa = p1[a] if va == 1 else 1 - p1[a]
                pb = p1[b] if vb == 1 else 1 - p1[b]
                expected = pa * pb * N
                double_candidates.append({
                    "attrs": (a, va, b, vb), "count": c,
                    "info": abs(c - expected),
                })
    double_candidates.sort(key=lambda x: -x["info"])

    selected_double = []
    pair_use = {}
    attr_info = {a: 0.0 for a in attrs}  # 供三属性候选池排序
    for cand in double_candidates:
        if len(selected_double) >= N_DOUBLE:
            break
        a, va, b, vb = cand["attrs"]
        key = (a, b)
        if pair_use.get(key, 0) >= MAX_PER_PAIR:
            continue
        selected_double.append(cand)
        pair_use[key] = pair_use.get(key, 0) + 1
        attr_info[a] += cand["info"]
        attr_info[b] += cand["info"]

    for cand in selected_double:
        a, va, b, vb = cand["attrs"]
        qid = len(queries) + 1
        queries.append({
            "id": f"D{qid:04d}",
            "type": "double",
            "expression": f"{a} == {va} AND {b} == {vb}",
            "conditions": [
                {"attribute": a, "operator": "==", "value": va},
                {"attribute": b, "operator": "==", "value": vb},
            ],
            "result": cand["count"],
        })
    n_double_actual = len(queries) - single_count
    print(f"双属性查询: {n_double_actual} 个（候选 {len(double_candidates)}）")

    # ---------- 3. 三属性：top 池组合补足 ----------
    n_triple_target = TOTAL - single_count - n_double_actual
    pool = sorted(attrs, key=lambda a: -attr_info[a])[:TRIPLE_ATTR_POOL]
    triple_candidates = []
    for a, b, c_attr in itertools.combinations(pool, 3):
        for va in values:
            for vb in values:
                for vc in values:
                    cnt = eval_count(
                        df, [(a, va), (b, vb), (c_attr, vc)]
                    )
                    if cnt < MIN_COUNT_TRIPLE:
                        continue
                    pa = p1[a] if va == 1 else 1 - p1[a]
                    pb = p1[b] if vb == 1 else 1 - p1[b]
                    pc = p1[c_attr] if vc == 1 else 1 - p1[c_attr]
                    expected = pa * pb * pc * N
                    triple_candidates.append({
                        "attrs": (a, va, b, vb, c_attr, vc),
                        "count": cnt, "info": abs(cnt - expected),
                    })
    triple_candidates.sort(key=lambda x: -x["info"])

    selected_triple = []
    triple_use = {}
    for cand in triple_candidates:
        if len(selected_triple) >= n_triple_target:
            break
        a, va, b, vb, c_attr, vc = cand["attrs"]
        tri_key = (a, b, c_attr)
        if triple_use.get(tri_key, 0) >= MAX_PER_TRIPLE:
            continue
        selected_triple.append(cand)
        triple_use[tri_key] = triple_use.get(tri_key, 0) + 1

    for cand in selected_triple[:n_triple_target]:
        a, va, b, vb, c_attr, vc = cand["attrs"]
        qid = len(queries) + 1
        queries.append({
            "id": f"T{qid:04d}",
            "type": "triple",
            "expression": (
                f"{a} == {va} AND {b} == {vb} AND {c_attr} == {vc}"
            ),
            "conditions": [
                {"attribute": a, "operator": "==", "value": va},
                {"attribute": b, "operator": "==", "value": vb},
                {"attribute": c_attr, "operator": "==", "value": vc},
            ],
            "result": cand["count"],
        })
    print(f"三属性查询: {len(selected_triple)} 个（池 {TRIPLE_ATTR_POOL} 属性）")

    out = {
        "dataset": "plants.csv",
        "record_count": N,
        "query_count": len(queries),
        "result_unit": "records",
        "description": (
            f"{len(queries)} 个查询（single 138 拉满，double 信息量 "
            f"top {n_double_actual}，triple 从信息量 top "
            f"{TRIPLE_ATTR_POOL} 属性池精选）"
        ),
        "queries": queries,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    counts = [q["result"] for q in queries]
    print(f"\n总查询数: {len(queries)}")
    print(f"已保存: {OUT_PATH}")
    print(f"计数分布: min={min(counts)}, max={max(counts)}, "
          f"median={int(np.median(counts))}")


if __name__ == "__main__":
    main()
