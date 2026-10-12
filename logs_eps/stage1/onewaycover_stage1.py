"""一阶覆盖分析：按字段是否被 AIM 测过的对覆盖分组，看一阶误差来自哪里。

工作笔记第五十七步第 14 条。对每个 ε、种子：
  仅一阶测量 = 字段不在任何 ≥2 阶测量团里（只有 σ 很大的一阶题）；
  被对覆盖   = 字段至少出现在一个测过的对/三元组里。
四列都是对真表的一阶误差（×10⁻⁶ L1/N，组内字段均值，3 种子均）：
  夹零按比例 = 噪声一阶负数夹零后按比例归一（第一阶段 hybrid 起点，--init-oneway 不带 --init-clean）；
  欧氏投影   = 同一噪声一阶投影到总和 N 的非负单纯形（--init-clean 的 clean_first_order）；
  hybrid 终表 / PGM 终表 = 两臂最终表的一阶。

用法: .venv/bin/python logs_eps/stage1/onewaycover_stage1.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps/stage1"))
sys.path.insert(0, str(ROOT / "src"))
from grade_stage1 import arm_path  # noqa: E402
from resevo.initialization import _project_simplex  # noqa: E402

TAGS = ("0p1", "0p3", "1", "3", "10")
SEEDS = (1, 2, 3)


def main() -> None:
    real = pd.read_csv(ROOT / "data/adult/adult.csv")
    cols = list(real.columns)
    n = len(real)
    sizes = {c: int(real[c].max()) + 1 for c in cols}
    ro = {c: np.bincount(real[c], minlength=sizes[c]).astype(float) for c in cols}

    def clipprop(v):
        v = np.maximum(v, 0.0)
        return v / v.sum() * n

    lines = [
        "## 1B 一阶覆盖分析：一阶误差按字段是否被测过的对覆盖分组（×10⁻⁶ L1/N，3 种子均）",
        "",
        "仅一阶测量 = 字段只有一阶题（不在任何测过的对里）；被对覆盖 = 字段至少在一个测过的对里。",
        "夹零按比例 = 第一阶段 hybrid 的起点清洗；欧氏投影 = --init-clean 的清洗；后两列为两臂终表。",
        "",
        "| ε | 字段组 | 字段数 | 夹零按比例(起点) | 欧氏投影 | hybrid 终表 | PGM 终表 | Δ终表 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        acc: dict[str, list] = {}
        for s in SEEDS:
            dump = json.loads((ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{s}_measurements.json").read_text())
            covered: set[str] = set()
            for m in dump["measurements"]:
                if len(m["clique"]) >= 2:
                    covered.update(m["clique"])
            exam = json.loads(Path(glob.glob(str(ROOT / f"data/adult_aimrep{tag}_s{s}/measured_*query.json"))[0]).read_text())
            tgt: dict[str, dict[int, float]] = {}
            for q in exam["queries"]:
                if len(q["conditions"]) == 1:
                    c = q["conditions"][0]
                    tgt.setdefault(c["attribute"], {})[int(c["value"])] = q["result"]
            syn = {a: pd.read_csv(arm_path(a, tag, s)) for a in ("hybrid", "pgmrep10k")}
            groups = {"仅一阶测量": [c for c in cols if c not in covered], "被对覆盖": [c for c in cols if c in covered]}
            for g, fs in groups.items():
                if not fs:
                    continue
                r = []
                for c in fs:
                    t = np.array([tgt[c].get(v, 0.0) for v in range(sizes[c])])
                    e = lambda h, c=c: np.abs(h - ro[c]).sum() / n * 1e6
                    r.append((
                        e(clipprop(t)), e(_project_simplex(t, float(n))),
                        e(np.bincount(syn["hybrid"][c], minlength=sizes[c])),
                        e(np.bincount(syn["pgmrep10k"][c], minlength=sizes[c])), len(fs),
                    ))
                acc.setdefault(g, []).append(np.mean(r, axis=0))
        for g in ("仅一阶测量", "被对覆盖"):
            if g not in acc:
                lines.append(f"| {tag.replace('p', '.')} | {g} | 0 | — | — | — | — | — |")
                continue
            m = np.mean(acc[g], axis=0)
            d = (m[2] - m[3]) / m[3] * 100
            lines.append(
                f"| {tag.replace('p', '.')} | {g} | {m[4]:.1f} | {m[0]:,.0f} | {m[1]:,.0f} | {m[2]:,.0f} | {m[3]:,.0f} | {d:+.1f}% |"
            )
    text = "\n".join(lines) + "\n"
    (ROOT / "logs_eps/scores/stage1_onewaycover.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
