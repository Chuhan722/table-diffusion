"""
为 nltcs 数据集生成查询工作负载（1000 个）

分布（受 16 个二值属性的组合空间限制）：
- 单属性：32 个（16×2，客观上限，全覆盖各属性基本分布）
- 双属性：约 478 个（C(16,2)×4 计数达标者拉满，覆盖两两相关关系）
- 三属性：补足到 1000（业务重要、计数不太稀疏、无法被低阶充分表达的组合）

设计原则：
- 避免重复或逻辑高度相关的查询
- 减少极低计数条件（设最小计数阈值）
- 让不同属性/类别的出现次数尽量均衡
"""
import json
import itertools
import numpy as np
import pandas as pd

DATA_PATH = "data/nltcs/nltcs.csv"
OUT_PATH = "configs/nltcs/measured_1000query.json"
SEED = 0

# 总计 1000 个查询：单属性、双属性拉满（客观上限），其余全给三属性
# 单属性上限 16×2=32；双属性上限 C(16,2)×4=480（计数>=50 的约 478）
# 三属性动态补足到 TOTAL
TOTAL = 1000
N_SINGLE = 32
N_DOUBLE = 480

# 极低计数阈值：查询计数低于此值视为过稀疏，尽量避免
MIN_COUNT = 50          # 双属性最小计数
MIN_COUNT_TRIPLE = 30   # 三属性最小计数（放宽一些）


def eval_count(df, conds):
    """给定条件列表（AND），返回满足的记录数。"""
    mask = np.ones(len(df), dtype=bool)
    for attr, val in conds:
        mask &= (df[attr].values == val)
    return int(mask.sum())


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(DATA_PATH)
    N = len(df)
    attrs = list(df.columns)
    values = [0, 1]

    queries = []
    qid = 0

    # 出现次数计数器，用于均衡
    attr_use = {a: 0 for a in attrs}

    # ---------- 1. 单属性：100 个 ----------
    # 16 属性 × 2 取值 = 32 个基础单属性查询，全部纳入
    # 剩余 68 个：无更多单属性组合，故单属性上限是 32
    # 因此单属性实际生成 32 个，把差额补到双属性
    single_specs = [(a, v) for a in attrs for v in values]
    single_count = 0
    for (a, v) in single_specs:
        c = eval_count(df, [(a, v)])
        queries.append({
            "id": f"S{single_count+1:03d}",
            "type": "single",
            "expression": f"{a} == {v}",
            "conditions": [{"attribute": a, "operator": "==", "value": v}],
            "result": c,
        })
        attr_use[a] += 1
        single_count += 1
    print(f"单属性查询: {single_count} 个（16 属性 × 2 取值）")

    # ---------- 2. 双属性 ----------
    # 所有 (attr_i, attr_j) 对 × 4 种取值组合，按信息量筛选
    # 信息量：偏离独立假设的程度（|观测计数 - 独立预期|），优先选相关性强的
    double_candidates = []
    p1 = {a: (df[a].values == 1).mean() for a in attrs}
    for a, b in itertools.combinations(attrs, 2):
        for va in values:
            for vb in values:
                c = eval_count(df, [(a, va), (b, vb)])
                if c < MIN_COUNT:
                    continue  # 减少极低计数
                # 独立预期
                pa = p1[a] if va == 1 else 1 - p1[a]
                pb = p1[b] if vb == 1 else 1 - p1[b]
                expected = pa * pb * N
                info = abs(c - expected)  # 偏离独立的程度
                double_candidates.append({
                    "attrs": (a, va, b, vb), "count": c, "info": info,
                })

    # 双属性拉满：计数达标的组合全部纳入（客观上限约 478/480）
    # 按信息量降序排列，仅为让"更相关"的排在前面，不做数量裁剪
    double_candidates.sort(key=lambda x: -x["info"])
    selected_double = double_candidates[:N_DOUBLE]
    for cand in selected_double:
        a, va, b, vb = cand["attrs"]
        attr_use[a] += 1
        attr_use[b] += 1

    for cand in selected_double[:N_DOUBLE]:
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
    print(f"双属性查询: {n_double_actual} 个")

    # ---------- 3. 三属性 ----------
    # 动态补足到总数 1000：三属性目标 = 1000 − 单属性 − 双属性实际数
    n_triple_target = TOTAL - single_count - n_double_actual
    # 只选：计数不太稀疏 + 无法被低阶充分表达（三阶交互信息大）
    # 三阶偏离：|观测 - 基于二阶的预期|，用独立近似衡量高阶结构
    triple_candidates = []
    # 为控制计算量，从相关性强的属性子集里组合
    for a, b, c_attr in itertools.combinations(attrs, 3):
        for va in values:
            for vb in values:
                for vc in values:
                    cnt = eval_count(df, [(a, va), (b, vb), (c_attr, vc)])
                    if cnt < MIN_COUNT_TRIPLE:
                        continue
                    pa = p1[a] if va == 1 else 1 - p1[a]
                    pb = p1[b] if vb == 1 else 1 - p1[b]
                    pc = p1[c_attr] if vc == 1 else 1 - p1[c_attr]
                    expected = pa * pb * pc * N
                    info = abs(cnt - expected)
                    triple_candidates.append({
                        "attrs": (a, va, b, vb, c_attr, vc),
                        "count": cnt, "info": info,
                    })

    triple_candidates.sort(key=lambda x: -x["info"])
    selected_triple = []
    triple_use = {}
    for cand in triple_candidates:
        if len(selected_triple) >= n_triple_target:
            break
        a, va, b, vb, c_attr, vc = cand["attrs"]
        tri_key = (a, b, c_attr)
        tu = triple_use.get(tri_key, 0)
        # 每个属性三元组最多取 2 种取值组合，避免高度相关
        if tu >= 2:
            continue
        selected_triple.append(cand)
        triple_use[tri_key] = tu + 1

    if len(selected_triple) < n_triple_target:
        for cand in triple_candidates:
            if len(selected_triple) >= n_triple_target:
                break
            if cand in selected_triple:
                continue
            selected_triple.append(cand)

    for cand in selected_triple[:n_triple_target]:
        a, va, b, vb, c_attr, vc = cand["attrs"]
        qid = len(queries) + 1
        queries.append({
            "id": f"T{qid:04d}",
            "type": "triple",
            "expression": f"{a} == {va} AND {b} == {vb} AND {c_attr} == {vc}",
            "conditions": [
                {"attribute": a, "operator": "==", "value": va},
                {"attribute": b, "operator": "==", "value": vb},
                {"attribute": c_attr, "operator": "==", "value": vc},
            ],
            "result": cand["count"],
        })
    print(f"三属性查询: {min(len(selected_triple), n_triple_target)} 个")

    # ---------- 输出 ----------
    out = {
        "dataset": "nltcs.csv",
        "record_count": N,
        "query_count": len(queries),
        "result_unit": "records",
        "description": f"{len(queries)} 个真实数据计数查询及其在 nltcs.csv 上的查询结果",
        "queries": queries,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n总查询数: {len(queries)}")
    print(f"已保存: {OUT_PATH}")

    # 诊断：计数分布、属性均衡
    counts = [q["result"] for q in queries]
    print(f"\n计数分布: min={min(counts)}, max={max(counts)}, "
          f"中位数={int(np.median(counts))}, 均值={int(np.mean(counts))}")


if __name__ == "__main__":
    main()
