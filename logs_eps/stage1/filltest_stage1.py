"""填充测试：没测过的对上，表的实际成绩 vs "拿表自己的一阶做独立乘积"的成绩。

目的（工作笔记第五十七步第 14 条）：区分 1B 低 ε 输给 PGM 的两种解释——
(a) 一阶传导：没测的对两家都≈按一阶独立铺，输只因一阶差；
(b) 结构注入：我们逐行改表顺带往没测的对里灌了噪声结构。
判据：若 实际 ≈ 自身乘积 → 纯传导；若 实际 > 自身乘积 → 注入有害结构；
若 实际 < 自身乘积 → 表里带的结构有益（与真结构同向）。
同时报 "自身结构量" = |表 − 自身乘积|（表在该组对上离独立多远）。

用法: .venv/bin/python logs_eps/stage1/filltest_stage1.py [--arms hybrid,pgmrep10k]
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "logs_eps/stage1"))
from grade_stage1 import arm_path  # noqa: E402

TAGS = ("0p1", "0p3", "1", "3", "10")
SEEDS = (1, 2, 3)


def hist2(df: pd.DataFrame, a: str, b: str, sa: int, sb: int) -> np.ndarray:
    key = df[a].to_numpy(dtype=np.int64) * sb + df[b].to_numpy(dtype=np.int64)
    return np.bincount(key, minlength=sa * sb).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="hybrid,pgmrep10k")
    ap.add_argument("--out", default=str(ROOT / "logs_eps/scores/stage1_filltest.md"))
    args = ap.parse_args()
    arms = args.arms.split(",")

    real = pd.read_csv(ROOT / "data/adult/adult.csv")
    cols = list(real.columns)
    sizes = {c: int(real[c].max()) + 1 for c in cols}
    n = len(real)
    pairs = list(combinations(cols, 2))
    real_pairs = {p: hist2(real, *p, sizes[p[0]], sizes[p[1]]) for p in pairs}
    real_ones = {c: np.bincount(real[c].to_numpy(), minlength=sizes[c]).astype(np.float64) for c in cols}

    def l1(pred: dict, ps) -> float:
        return float(np.mean([np.abs(pred[p] - real_pairs[p]).sum() / n for p in ps])) * 1e6

    def diff(a: dict, b: dict, ps) -> float:
        return float(np.mean([np.abs(a[p] - b[p]).sum() / n for p in ps])) * 1e6

    lines = [
        "## 1B 填充测试：没测过的对上，实际 vs 自身一阶独立乘积（×10⁻⁶ L1/N，3 种子均）",
        "",
        "实际 = 表自己的二阶误差；自身乘积 = 用该表的一阶做独立乘积当二阶的误差；",
        "自身结构量 = |表 − 自身乘积|（表离独立多远）；真结构量 = |真表 − 真一阶乘积|。",
        "实际 ≈ 自身乘积 ⇒ 没测的对纯靠一阶传导；实际 > 自身乘积 ⇒ 表里灌了有害结构。",
        "",
        "| ε | 组 | 臂 | 实际 | 自身乘积 | 实际−乘积 | 自身结构量 | 真结构量 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag in TAGS:
        acc: dict[tuple[str, str], list[list[float]]] = {}
        for seed in SEEDS:
            dump = json.loads((ROOT / f"results/aimreplay/adult_eps{tag}_aimfull_seed{seed}_measurements.json").read_text())
            measured = set()
            for m in dump["measurements"]:
                if len(m["clique"]) == 2:
                    a, b = m["clique"]
                    measured.add((a, b) if cols.index(a) < cols.index(b) else (b, a))
            groups = {"没测": [p for p in pairs if p not in measured], "测过": [p for p in pairs if p in measured]}
            true_prod = {p: np.outer(real_ones[p[0]], real_ones[p[1]]).ravel() / n for p in pairs}
            for arm in arms:
                syn = pd.read_csv(arm_path(arm, tag, seed))
                scale = n / len(syn)
                ones = {c: np.bincount(syn[c].to_numpy(), minlength=sizes[c]).astype(np.float64) * scale for c in cols}
                actual = {p: hist2(syn, *p, sizes[p[0]], sizes[p[1]]) * scale for p in pairs}
                prod = {p: np.outer(ones[p[0]], ones[p[1]]).ravel() / n for p in pairs}
                for g, ps in groups.items():
                    acc.setdefault((g, arm), []).append([
                        l1(actual, ps), l1(prod, ps), diff(actual, prod, ps), diff(real_pairs, true_prod, ps),
                    ])
        for g in ("没测", "测过"):
            for arm in arms:
                v = np.mean(acc[(g, arm)], axis=0)
                lines.append(
                    f"| {tag.replace('p', '.')} | {g} | {arm} | {v[0]:,.0f} | {v[1]:,.0f} | {v[0]-v[1]:+,.0f} | {v[2]:,.0f} | {v[3]:,.0f} |"
                )
    text = "\n".join(lines) + "\n"
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
