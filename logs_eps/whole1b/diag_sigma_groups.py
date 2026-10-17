"""第六十七步补充诊断：按 σ 档拆开看，第一段地板停表 vs 旧收敛表在 1B 考卷上各档的拟合程度。

每档给出  对带噪测量的加权损失 ÷ 该档地板  和  对真表的加权距离 ÷ 该档地板（档地板 = ½·档题数·σ_min²）。
地板停表 = results/whole1b/*_ourswhole_seed*_stage1.csv；收敛表 = results/stage1/*_hybrid_seed*.csv（10-15 旧第一段）。
只读已有表，不跑任何实验。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TAGS = ["0p1", "0p3", "1", "3", "10"]
SEEDS = [1, 2, 3]
REAL = pd.read_csv(ROOT / "data/adult/adult.csv")


def counts(df: pd.DataFrame, qs: list[dict]) -> np.ndarray:
    cache: dict = {}
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        attrs = tuple(c["attribute"] for c in q["conditions"])
        vals = tuple(int(c["value"]) for c in q["conditions"])
        if attrs not in cache:
            g = df.groupby(list(attrs)).size()
            cache[attrs] = {(k if isinstance(k, tuple) else (k,)): v for k, v in g.items()}
        out[i] = cache[attrs].get(vals, 0)
    return out


def load_exam(tag: str, seed: int) -> list[dict]:
    path = glob.glob(str(ROOT / f"data/adult_aimrep{tag}_s{seed}/measured_*query.json"))[0]
    return json.load(open(path))["queries"]


def main() -> None:
    lines = ["# 1B 第一段：按 σ 档拆开的拟合程度（地板停 vs 收敛）", ""]
    summary = []
    for tag in TAGS:
        for seed in SEEDS:
            qs = load_exam(tag, seed)
            y = np.array([q["result"] for q in qs])
            sig = np.array([q["sigma"] for q in qs])
            w = (sig.min() / sig) ** 2
            tr = counts(REAL, qs)
            marg = [tuple(c["attribute"] for c in q["conditions"]) for q in qs]
            tabs = {
                "地板停": pd.read_csv(ROOT / f"results/whole1b/adult_eps{tag}_ourswhole_seed{seed}_stage1.csv"),
                "收敛": pd.read_csv(ROOT / f"results/stage1/adult_eps{tag}_hybrid_seed{seed}.csv"),
            }
            qq = {k: counts(v, qs) for k, v in tabs.items()}
            F = 0.5 * np.sum(w * sig**2)
            keys = np.round(sig, 1)
            lines += [f"## ε={tag} 种子 {seed}：题 {len(qs)}，σ {sig.min():.1f}–{sig.max():.1f}", ""]
            lines += ["| σ | 权重 | 题数 | 边际数 | 地板份额 | 地板停 对带噪÷档地板 | 地板停 对真值÷档地板 | 收敛 对带噪÷档地板 | 收敛 对真值÷档地板 |",
                      "|---|---|---|---|---|---|---|---|---|"]
            rows = []
            for u in sorted(set(keys)):
                m = keys == u
                Fg = 0.5 * np.sum((w * sig**2)[m])
                a = {k: 0.5 * np.sum((w * (y - qq[k]) ** 2)[m]) / Fg for k in tabs}
                b = {k: 0.5 * np.sum((w * (tr - qq[k]) ** 2)[m]) / Fg for k in tabs}
                nm = len({marg[i] for i in np.flatnonzero(m)})
                rows.append((u, w[m][0], int(m.sum()), nm, Fg / F, a["地板停"], b["地板停"], a["收敛"], b["收敛"]))
                lines.append(f"| {u:.1f} | {w[m][0]:.2f} | {m.sum()} | {nm} | {Fg / F:.1%} | {a['地板停']:.2f} | {b['地板停']:.2f} | {a['收敛']:.2f} | {b['收敛']:.2f} |")
            lines.append("")
            worst = rows[-1]  # σ 最大、权重最小的一档
            rest_m = keys != worst[0]
            Fr = 0.5 * np.sum((w * sig**2)[rest_m])
            rest_floor = 0.5 * np.sum((w * (y - qq["地板停"]) ** 2)[rest_m]) / Fr
            rest_conv = 0.5 * np.sum((w * (y - qq["收敛"]) ** 2)[rest_m]) / Fr
            summary.append((tag, seed, worst[0], worst[1], worst[3], worst[4], worst[5], worst[6], worst[7], worst[8], rest_floor, rest_conv))
    lines += ["## 汇总：σ 最大（权重最小）一档 vs 其余各档", "",
              "| ε | 种子 | 最大σ | 权重 | 边际数 | 地板份额 | 该档地板停 对带噪 | 该档地板停 对真值 | 该档收敛 对带噪 | 该档收敛 对真值 | 其余档地板停 对带噪 | 其余档收敛 对带噪 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r[0]} | {r[1]} | {r[2]:.1f} | {r[3]:.2f} | {r[4]} | {r[5]:.1%} | {r[6]:.2f} | {r[7]:.2f} | {r[8]:.2f} | {r[9]:.2f} | {r[10]:.2f} | {r[11]:.2f} |")
    out = ROOT / "logs_eps/scores/whole1b_sigma_groups.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-17:]))
    print("->", out)


if __name__ == "__main__":
    main()
