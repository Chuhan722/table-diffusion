"""二阶清洗离线扫描（第二阶段首战 Step 0，工作笔记第五十七步第 16 条）。

用 resevo.denoise 的规则直接处理考卷目标，不跑引擎，对真表评"已测对"上的二阶误差
（L1 记录数之和），回答：二阶题也做清洗是帮忙还是添乱？对照列：
  原始噪声   = 考卷目标原样；
  逐对投影   = 每对格子欧氏投影到总行数单纯形（近似拟合终点）；
  仅一阶 z=2 = 只清洗一阶，二阶目标原样（hybridp1 臂的目标）；
  一阶+二阶 z= 一阶后二阶也清洗（hybridp12 臂的目标）；
  hybrid / PGM 终表 = 第一阶段 1B 的实际终表在同一批对上的误差。
用法: .venv/bin/python logs_eps/stage1/pairpool_scan.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "logs_eps/stage1"))
from grade_stage1 import arm_path  # noqa: E402
from resevo.dataset import load_queries, load_table  # noqa: E402
from resevo.denoise import pool_oneway, pool_pairs  # noqa: E402
from resevo.initialization import _project_simplex  # noqa: E402


def pair_cells(specs, schema):
    """{(字段a,字段b): (题下标数组, 值a数组, 值b数组, σ)}，只收恰两条等值条件的题。"""
    out: dict = {}
    for i, q in enumerate(specs):
        if len(q.conditions) != 2 or any(c["operator"] != "==" for c in q.conditions):
            continue
        (ca, cb) = sorted(q.conditions, key=lambda c: schema.field_index(c["attribute"]))
        out.setdefault((ca["attribute"], cb["attribute"]), []).append((i, int(ca["value"]), int(cb["value"]), q.sigma))
    return out


def table_pair_counts(df: pd.DataFrame, a: str, b: str, va, vb) -> np.ndarray:
    ct = pd.crosstab(df[a], df[b])
    return np.array([float(ct.at[x, y]) if (x in ct.index and y in ct.columns) else 0.0 for x, y in zip(va, vb)])


def main() -> None:
    real = pd.read_csv(ROOT / "data/adult/adult.csv")
    n = len(real)
    lines = ["## 二阶清洗离线扫描（已测对上的二阶 L1 误差之和，记录数；×10⁻³）", "",
             "| ε | 种子 | 对数 | 格子 | 原始噪声 | 逐对投影 | 仅一阶 z=2 | 一阶+二阶 z=2 | 一阶+二阶 z=3 | hybrid 终表 | PGM 终表 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag, eps in [("0p1", "0.1"), ("0p3", "0.3"), ("1", "1"), ("3", "3"), ("10", "10")]:
        for s in (1, 2, 3):
            ddir = ROOT / f"data/adult_aimrep{tag}_s{s}"
            schema, _ = load_table(str(ddir / f"adult_aimrep{tag}_s{s}.csv"))
            specs = load_queries(glob.glob(str(ddir / "measured_*query.json"))[0])
            p1, ones, _ = pool_oneway(specs, schema, n, 2.0)
            p12, _ = pool_pairs(p1, schema, n, 2.0, ones)
            q1, ones3, _ = pool_oneway(specs, schema, n, 3.0)
            p12z3, _ = pool_pairs(q1, schema, n, 3.0, ones3)
            cells = pair_cells(specs, schema)
            tables = {}
            for arm in ("hybrid", "pgmrep10k"):
                p = arm_path(arm, tag, s)
                tables[arm] = pd.read_csv(p) if p.exists() else None
            err = {k: 0.0 for k in ("raw", "proj", "p1", "p12", "p12z3", "hybrid", "pgmrep10k")}
            ncell = 0
            for (a, b), items in cells.items():
                idx = np.array([t[0] for t in items])
                va = [t[1] for t in items]
                vb = [t[2] for t in items]
                truth = table_pair_counts(real, a, b, va, vb)
                ncell += len(idx)
                y = np.array([specs[i].result for i in idx])
                err["raw"] += np.abs(y - truth).sum()
                err["proj"] += np.abs(_project_simplex(y, float(n)) - truth).sum()
                err["p1"] += np.abs(np.array([p1[i].result for i in idx]) - truth).sum()
                err["p12"] += np.abs(np.array([p12[i].result for i in idx]) - truth).sum()
                err["p12z3"] += np.abs(np.array([p12z3[i].result for i in idx]) - truth).sum()
                for arm, df in tables.items():
                    if df is not None:
                        err[arm] += np.abs(table_pair_counts(df, a, b, va, vb) - truth).sum()
            f = lambda k: f"{err[k] / 1e3:,.1f}" if tables.get(k, 1) is not None else "—"
            lines.append(f"| {eps} | {s} | {len(cells)} | {ncell} | " + " | ".join(f(k) for k in err) + " |")
    text = "\n".join(lines) + "\n"
    (ROOT / "logs_eps/scores/stage2_pairpool_scan.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
