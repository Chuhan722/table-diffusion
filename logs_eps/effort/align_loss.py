"""第六十步 第 1 步：统一损失口径。

共同损失 ℒ(X) = Σ_{14123 格} (噪声目标 − X 的计数)²，两家都吃同一份 measured_2way.json。
我们：old_loss = ½ eᵀe（--exam-weights none）→ ℒ = 2·old_loss。
GSD：fitness = Σ(目标频率 − 频率)² → ℒ = N²·fitness；日志只打了 L1 均值（频率尺度），用它核对本脚本的计数口径。
用法：.venv/bin/python logs_eps/effort/align_loss.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAGS = ["0p1", "0p3", "1", "3", "10"]
SEEDS = [1, 2, 3]


def load_exam(path: Path):
    d = json.load(open(path, encoding="utf-8"))
    pairs, targets = [], []
    for q in d["queries"]:
        (a, va), (b, vb) = [(c["attribute"], int(c["value"])) for c in q["conditions"]]
        pairs.append((a, va, b, vb))
        targets.append(float(q["result"]))
    return d["record_count"], pairs, np.asarray(targets)


def load_codes(path: Path):
    with open(path, encoding="utf-8") as fh:
        r = csv.reader(fh)
        cols = next(r)
        rows = np.asarray([[int(x) for x in line] for line in r], dtype=np.int64)
    return cols, rows


def cell_counts(cols, rows, pairs):
    idx = {c: i for i, c in enumerate(cols)}
    out = np.empty(len(pairs))
    cache = {}
    for k, (a, va, b, vb) in enumerate(pairs):
        key = (a, b)
        if key not in cache:
            ca, cb = rows[:, idx[a]], rows[:, idx[b]]
            ka, kb = int(ca.max()) + 1, int(cb.max()) + 1
            cache[key] = (np.bincount(ca * kb + cb, minlength=ka * kb), kb)
        bc, kb = cache[key]
        flat = va * kb + vb
        out[k] = bc[flat] if flat < bc.size else 0
    return out


def curve_final_loss(path: Path) -> float:
    last = None
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["round"] == "final":
                return float(row["old_loss"])
            last = row
    return float(last["old_loss"])


def gsd_log_l1(path: Path):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    m = re.search(r"对噪声目标残差 平均 ([0-9.eE+-]+) 最大 ([0-9.eE+-]+)", txt)
    s = re.search(r"搜索 ([0-9.]+) 秒", txt)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None), float(s.group(1)) if s else None


def main() -> None:
    print("| ε | 种子 | ℒ 我们 | 2·曲线末 old_loss | 相对差 | ℒ GSD | GSD L1 均(本算) | GSD L1 均(日志) | GSD/我们 |")
    print("|---|---|---|---|---|---|---|---|---|")
    worst = 0.0
    for tag in TAGS:
        for s in SEEDS:
            n, pairs, tgt = load_exam(ROOT / f"data/adult_alloc{tag}base_s{s}/measured_2way.json")
            ours = ROOT / f"results/stage1/adult_eps{tag}_ours2way_seed{s}.csv"
            gsd = ROOT / f"results/baseline2/adult_eps{tag}_gsd_seed{s}.csv"
            co, ro = load_codes(ours)
            cg, rg = load_codes(gsd)
            assert ro.shape[0] == n and rg.shape[0] == n
            lo = float(np.sum((tgt - cell_counts(co, ro, pairs)) ** 2))
            cg_counts = cell_counts(cg, rg, pairs)
            lg = float(np.sum((tgt - cg_counts) ** 2))
            curve = 2 * curve_final_loss(ROOT / f"results/stage1/adult_eps{tag}_ours2way_seed{s}_curve.csv")
            rel = abs(lo - curve) / curve
            worst = max(worst, rel)
            (l1_log, _), _ = gsd_log_l1(ROOT / f"logs_eps/baseline2/adult_eps{tag}_gsd_seed{s}.log")
            l1_mine = float(np.mean(np.abs(tgt - cg_counts)) / n)
            print(f"| {tag} | {s} | {lo:,.0f} | {curve:,.0f} | {rel:.1e} | {lg:,.0f} | {l1_mine:.6f} | {l1_log} | {lg / lo:.3f} |")
    print(f"\n我们：ℒ 与 2·曲线末 old_loss 最大相对差 {worst:.1e}")


if __name__ == "__main__":
    main()
