"""一阶去噪估计量扫描（第二阶段首战选药，工作笔记第五十七步第 15 条）。

只用考卷里的噪声一阶题（结果 + σ，纯后处理零预算），对真表评一阶误差（×10⁻⁶ L1/N），
与 PGM 终表一阶对照。候选：
  欧氏投影   = 现状（hybrid 终表恰落在此点）；
  平分池 z   = |y| < zσ 的格子视为"测不出"，剩余质量在它们之间平分，其余格子取 max(y,0)；
  EG早停 T   = 指数梯度从均匀起步、L2 损失、固定 T 步早停（模仿 PGM 的隐式正则）。
按种子逐个报，便于看稳定性。用法: .venv/bin/python logs_eps/stage1/onewaypool_scan.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "logs_eps/stage1"))
from grade_stage1 import arm_path  # noqa: E402
from resevo.initialization import _project_simplex  # noqa: E402


def eg_earlystop(y: np.ndarray, sigma: float, n: int, steps: int, lr: float = 0.5) -> np.ndarray:
    p = np.full(len(y), 1.0 / len(y))
    for _ in range(steps):
        p = p * np.exp(-lr * (n * p - y) / n)
        p /= p.sum()
    return n * p


def pool(y: np.ndarray, sigma: float, n: int, z: float) -> np.ndarray:
    known = np.abs(y) >= z * sigma
    out = np.zeros_like(y)
    out[known] = np.maximum(y[known], 0.0)
    rest = n - out.sum()
    if rest < 0 or not (~known).any():
        return _project_simplex(y, float(n))
    out[~known] = rest / (~known).sum()
    return out


def main() -> None:
    real = pd.read_csv(ROOT / "data/adult/adult.csv")
    cols = list(real.columns)
    n = len(real)
    sizes = {c: int(real[c].max()) + 1 for c in cols}
    ro = {c: np.bincount(real[c], minlength=sizes[c]).astype(float) for c in cols}
    ests = {"欧氏投影": lambda y, s: _project_simplex(y, float(n))}
    for z in (1, 2, 3):
        ests[f"平分池 z={z}"] = lambda y, s, z=z: pool(y, s, n, z)
    for st in (50, 100, 300):
        ests[f"EG早停 {st}步"] = lambda y, s, st=st: eg_earlystop(y, s, n, st)
    lines = ["## 一阶去噪估计量扫描（对真表一阶误差，×10⁻⁶ L1/N；字段均值）", ""]
    for tag, which, label in [("0p1", "unc", "ε=0.1 仅一阶字段"), ("0p3", "unc", "ε=0.3 仅一阶字段"),
                              ("0p1", "all", "ε=0.1 全部字段"), ("1", "all", "ε=1 全部字段"), ("10", "all", "ε=10 全部字段")]:
        lines += [f"### {label}", "", "| 种子 | 字段数 | PGM 终表 | " + " | ".join(ests) + " |",
                  "|---|---|---|" + "---|" * len(ests)]
        for s in (1, 2, 3):
            dump = json.loads((ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{s}_measurements.json").read_text())
            cov: set[str] = set()
            for m in dump["measurements"]:
                if len(m["clique"]) >= 2:
                    cov.update(m["clique"])
            exam = json.loads(Path(glob.glob(str(ROOT / f"data/adult_aimrep{tag}_s{s}/measured_*query.json"))[0]).read_text())
            tgt: dict[str, dict[int, float]] = {}
            sg: dict[str, float] = {}
            for q in exam["queries"]:
                if len(q["conditions"]) == 1:
                    c = q["conditions"][0]
                    tgt.setdefault(c["attribute"], {})[int(c["value"])] = q["result"]
                    sg[c["attribute"]] = q["sigma"]
            pgm = pd.read_csv(arm_path("pgmrep10k", tag, s))
            fs = [c for c in cols if c not in cov] if which == "unc" else cols
            if not fs:
                lines.append(f"| {s} | 0 | — |" + " — |" * len(ests))
                continue
            e = lambda h, c: np.abs(h - ro[c]).sum() / n * 1e6
            pg = np.mean([e(np.bincount(pgm[c], minlength=sizes[c]), c) for c in fs])
            vals = []
            for k, f in ests.items():
                vals.append(np.mean([e(f(np.array([tgt[c].get(v, 0.0) for v in range(sizes[c])]), sg[c]), c) for c in fs]))
            lines.append(f"| {s} | {len(fs)} | {pg:,.0f} | " + " | ".join(f"{v:,.0f}" for v in vals) + " |")
        lines.append("")
    text = "\n".join(lines)
    (ROOT / "logs_eps/scores/stage1_onewaypool_scan.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
