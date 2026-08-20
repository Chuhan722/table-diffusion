"""
为 plants 数据集生成查询工作负载 v2（980 个，按 PR #61 审查意见重设计）。

## v2 相对 v1 的修改（审查意见逐项）

1. **移除 1-way 查询**（意见 3）：1-way 已由初始化层 init_marginals.json
   单独提供精确边缘，不再计入主 measured workload；plants 构成对齐
   nltcs 的"0 单 / double / triple"口径。
2. **2-way 完整 marginal 成组**（意见 2）：不再按原始 cell 分数散选。
   先按完整 2×2 列联表的归一化关联度 φ²=χ²/N 选 top 115 个属性对，
   每对纳入全部 4 个 cell（联合单元完整成组，计 460 条）。
3. **3-way 模式覆盖**（意见 1）：v1 用原始计数偏差 |obs−exp| 选 cell，
   偏向高频全 0 模式（383 条全部 000）。v2 改为：
   - 属性三元组按三阶专属关联度选择：G²_total − G²_AB − G²_AC − G²_BC
     （完整 2×2×2 列联表对独立模型的 G² 中扣除三个二阶对独立的 G²，
     近似三阶交互强度），取 top 260 组；
   - 组内 cell 按标准化残差 |obs−exp_indep|/√max(exp,1) 排序取 top-2，
     且强制两个 cell 取值模式不同（计 520 条）；
   - 输出模式直方图供覆盖检查（tests 锚定非全 0 覆盖下限）。
4. 极低计数阈值保留（cell 计数 ≥30 才可入选 triple cell；2-way 成组
   不筛 cell——完整 marginal 语义优先，低计数 cell 是该 marginal 的
   真实组成部分）。

## 构成对比

nltcs（已冻结不动）：0 single / 479 double(cell 级近似成组) / 522 triple
plants v2：0 single / 460 double(115 对完整成组) / 520 triple(260 组×2)
总数 980 vs 1001：不追求逐条对齐，以"同数量级、同预算口径、双侧对称"
为可比性基础；构成差异在文档披露。

## 数据来源

公开密度估计基准系列（Lowd & Davis / Van Haaren & Davis 文献族）中的
plants 数据集，train/valid/test 划分沿用原始发布。来源与哈希见
data/plants/README.md。
"""
import itertools
import json

import numpy as np
import pandas as pd

DATA_PATH = "data/plants/plants.csv"
OUT_PATH = "configs/plants/measured_1000query.json"
SEED = 0

N_DOUBLE_PAIRS = 115      # ×4 cell = 460
N_TRIPLE_GROUPS = 260     # ×2 cell = 520
MIN_COUNT_TRIPLE_CELL = 30


def _g2(observed, expected):
    """似然比统计量 G² = 2·Σ obs·ln(obs/exp)（obs=0 项为 0）。"""
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    mask = observed > 0
    return float(
        2.0 * np.sum(observed[mask] * np.log(
            observed[mask] / np.maximum(expected[mask], 1e-12)
        ))
    )


def _table2(df_vals, a_col, b_col):
    """2×2 联合计数表（按 (va, vb) 展平序 00,01,10,11）。"""
    counts = np.zeros(4)
    joint = 2 * df_vals[:, a_col] + df_vals[:, b_col]
    for cell in range(4):
        counts[cell] = np.sum(joint == cell)
    return counts


def _table3(df_vals, cols):
    counts = np.zeros(8)
    joint = (4 * df_vals[:, cols[0]] + 2 * df_vals[:, cols[1]]
             + df_vals[:, cols[2]])
    for cell in range(8):
        counts[cell] = np.sum(joint == cell)
    return counts


def main():
    df = pd.read_csv(DATA_PATH)
    N = len(df)
    attrs = list(df.columns)
    assert len(attrs) == 69
    vals = df.to_numpy()
    p1 = vals.mean(axis=0)  # P(attr=1)

    def marg1(j):
        return np.array([1 - p1[j], p1[j]])

    # ---------- 2-way：φ² 关联度 top 115 对，完整成组 ----------
    pair_scores = []
    for i, j in itertools.combinations(range(69), 2):
        obs = _table2(vals, i, j)
        expected = np.outer(marg1(i), marg1(j)).ravel() * N
        chi2 = float(np.sum(
            (obs - expected) ** 2 / np.maximum(expected, 1e-12)
        ))
        pair_scores.append((chi2 / N, i, j, obs))
    pair_scores.sort(key=lambda x: -x[0])

    queries = []
    for phi2, i, j, obs in pair_scores[:N_DOUBLE_PAIRS]:
        for cell in range(4):
            va, vb = int(cell) >> 1, int(cell) & 1
            qid = len(queries) + 1
            queries.append({
                "id": f"D{qid:04d}",
                "type": "double",
                "expression": f"{attrs[i]} == {va} AND {attrs[j]} == {vb}",
                "conditions": [
                    {"attribute": attrs[i], "operator": "==", "value": va},
                    {"attribute": attrs[j], "operator": "==", "value": vb},
                ],
                "result": int(obs[cell]),
                "group": f"pair_{attrs[i]}_{attrs[j]}",
            })
    n_double = len(queries)
    print(f"双属性：{N_DOUBLE_PAIRS} 对完整成组 = {n_double} 条 "
          f"(φ² 范围 {pair_scores[N_DOUBLE_PAIRS-1][0]:.4f}"
          f"~{pair_scores[0][0]:.4f})", flush=True)

    # ---------- 3-way：三阶专属关联度 top 260 组 × 模式互异 2 cell ----
    # 候选池：φ² top 40 属性（控制 C(40,3)=9880 次列联表计算）
    attr_score = np.zeros(69)
    for phi2, i, j, _ in pair_scores:
        attr_score[i] += phi2
        attr_score[j] += phi2
    pool = np.argsort(-attr_score)[:40]

    triple_scores = []
    for a, b, c in itertools.combinations(sorted(pool), 3):
        obs3 = _table3(vals, (a, b, c))
        exp3_ind = (
            np.einsum("i,j,k->ijk", marg1(a), marg1(b), marg1(c)).ravel()
            * N
        )
        g2_total = _g2(obs3, exp3_ind)
        # 三个二阶对独立的 G²
        g2_pairs = 0.0
        for x, y in ((a, b), (a, c), (b, c)):
            obs2 = _table2(vals, x, y)
            exp2 = np.outer(marg1(x), marg1(y)).ravel() * N
            g2_pairs += _g2(obs2, exp2)
        interaction = g2_total - g2_pairs
        triple_scores.append((interaction / N, a, b, c, obs3, exp3_ind))
    triple_scores.sort(key=lambda x: -x[0])

    n_groups = 0
    pattern_hist = {}
    for score, a, b, c, obs3, exp3 in triple_scores:
        if n_groups >= N_TRIPLE_GROUPS:
            break
        # 组内 cell：标准化残差排序，模式互异，计数阈值
        std_resid = np.abs(obs3 - exp3) / np.sqrt(np.maximum(exp3, 1.0))
        order = np.argsort(-std_resid)
        chosen = []
        for cell in order:
            if obs3[cell] < MIN_COUNT_TRIPLE_CELL:
                continue
            pattern = (int(cell) >> 2, (int(cell) >> 1) & 1, int(cell) & 1)
            if chosen and pattern == chosen[0][1]:
                continue
            chosen.append((cell, pattern))
            if len(chosen) == 2:
                break
        if len(chosen) < 2:
            continue  # 该组可选 cell 不足，跳过（不足时向后补组）
        n_groups += 1
        for cell, pattern in chosen:
            qid = len(queries) + 1
            pattern_str = "".join(map(str, pattern))
            pattern_hist[pattern_str] = pattern_hist.get(pattern_str, 0) + 1
            queries.append({
                "id": f"T{qid:04d}",
                "type": "triple",
                "expression": (
                    f"{attrs[a]} == {pattern[0]} AND "
                    f"{attrs[b]} == {pattern[1]} AND "
                    f"{attrs[c]} == {pattern[2]}"
                ),
                "conditions": [
                    {"attribute": attrs[a], "operator": "==",
                     "value": pattern[0]},
                    {"attribute": attrs[b], "operator": "==",
                     "value": pattern[1]},
                    {"attribute": attrs[c], "operator": "==",
                     "value": pattern[2]},
                ],
                "result": int(obs3[cell]),
                "group": f"triple_{attrs[a]}_{attrs[b]}_{attrs[c]}",
            })
    n_triple = len(queries) - n_double
    print(f"三属性：{n_groups} 组 × 2 = {n_triple} 条", flush=True)
    print(f"三属性取值模式直方图：{dict(sorted(pattern_hist.items()))}",
          flush=True)

    out = {
        "dataset": "plants.csv",
        "record_count": N,
        "query_count": len(queries),
        "result_unit": "records",
        "workload_version": 2,
        "description": (
            f"{len(queries)} 个查询（v2：无 single——1-way 由初始化边缘"
            f"单独提供；double {N_DOUBLE_PAIRS} 对完整 2×2 成组共 "
            f"{n_double} 条，φ² 关联度选对；triple {n_groups} 组各 2 条"
            f"模式互异 cell，三阶专属 G² 交互度选组，标准化残差选 cell）"
        ),
        "queries": queries,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    counts = [q["result"] for q in queries]
    print(f"\n总查询数：{len(queries)}（double {n_double} / triple {n_triple}）",
          flush=True)
    print(f"计数分布：min={min(counts)} max={max(counts)} "
          f"median={int(np.median(counts))}", flush=True)
    print(f"已保存：{OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
