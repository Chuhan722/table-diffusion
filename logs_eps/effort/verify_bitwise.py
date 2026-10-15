"""第六十步记账改动的逐位核对：记账只写不算，终表必须与旧场逐位相同。

我们：results/effort 终表 vs results/stage1（1A 旧表）；曲线前 10 列 vs results/speed2 甲曲线（同稀疏通路，应全同）、
      vs results/stage1 旧曲线（稠密通路，7 列应同、3 列末位允许异）。
GSD：GPU 搜索本身非逐位确定（笔记第 386 行，原版脚本同 seed 两跑亦不同），故只能行为学核对：
      新表 ℒ vs baseline2 旧表 ℒ（同一份考卷算），记账末行 N²·best_fitness 必须等于新表 ℒ。
用法：.venv/bin/python logs_eps/effort/verify_bitwise.py [tags...]
"""
from __future__ import annotations

import csv
import filecmp
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "logs_eps/effort"))
from align_loss import cell_counts, load_codes, load_exam  # noqa: E402
TAGS = sys.argv[1:] or ["0p1", "0p3", "1", "3", "10"]
SEED = 1
OLD_COLS = ["round", "old_loss", "beta", "direction_gain", "interaction", "step",
            "expected_loss", "status", "max_gain_sum", "temp"]
EXACT = {"round", "status", "old_loss", "beta", "direction_gain", "max_gain_sum", "temp"}


def read(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def curve_compare(new: Path, ref: Path) -> str:
    a, b = read(new), read(ref)
    if len(a) != len(b):
        return f"行数异 {len(a)} vs {len(b)}"
    worst: dict[str, float] = {}
    for ra, rb in zip(a, b):
        for c in OLD_COLS:
            if ra[c] == rb[c]:
                continue
            if c in EXACT:
                return f"精确列 {c} 异于 round {ra['round']}: {ra[c]} vs {rb[c]}"
            try:
                d = abs(float(ra[c]) - float(rb[c])) / max(abs(float(rb[c])), 1.0)  # |参考|<1 时退化为绝对差
            except ValueError:
                return f"列 {c} 非数值异"
            worst[c] = max(worst.get(c, 0.0), d)
    if not worst:
        return "前 10 列全同"
    return "精确列同；" + " ".join(f"{c} 相对异≤{v:.1e}" for c, v in sorted(worst.items()))


def main() -> None:
    for tag in TAGS:
        ours = REPO / f"results/effort/adult_eps{tag}_ours2way_seed{SEED}.csv"
        if ours.exists():
            same = filecmp.cmp(ours, REPO / f"results/stage1/adult_eps{tag}_ours2way_seed{SEED}.csv", shallow=False)
            print(f"ε={tag} 我们终表 vs 1A: {'逐位同' if same else '不同'}")
            cur = ours.with_name(ours.stem + "_curve.csv")
            jia = REPO / f"results/speed2/adult_eps{tag}_ours2way_jia_seed{SEED}_curve.csv"
            if jia.exists():
                print(f"       曲线 vs 甲曲线: {curve_compare(cur, jia)}")
            print(f"       曲线 vs 1A 曲线: {curve_compare(cur, REPO / f'results/stage1/adult_eps{tag}_ours2way_seed{SEED}_curve.csv')}")
            rows = read(cur)
            cands = [int(r["candidates"]) for r in rows if r["candidates"] not in ("", "-1")]
            print(f"       轮数 {len(rows) - 1}，有候选数记录的轮 {len(cands)}，累计候选 {sum(cands):,}，"
                  f"末轮 elapsed_s {rows[-2]['elapsed_s']}")
        else:
            print(f"ε={tag} 我们终表尚无")
        gsd = REPO / f"results/effort/adult_eps{tag}_gsd_seed{SEED}.csv"
        if gsd.exists():
            old = REPO / f"results/baseline2/adult_eps{tag}_gsd_seed{SEED}.csv"
            same = filecmp.cmp(gsd, old, shallow=False)
            n, pairs, targets = load_exam(REPO / f"data/adult_alloc{tag}base_s{SEED}/measured_2way.json")
            loss = {}
            for name, path in (("新", gsd), ("旧", old)):
                cols, rows = load_codes(path)
                loss[name] = float(np.sum((targets - cell_counts(cols, rows, pairs)) ** 2))
            gens = read(gsd.with_name(gsd.stem + "_gens.csv"))
            last = gens[-1]
            l_log = n * n * float(last["best_fitness"])
            print(f"ε={tag} GSD 终表 vs baseline2: {'逐位同' if same else '不同（GSD GPU 搜索本身非逐位确定）'}；"
                  f"新表 ℒ {loss['新']:.6g} vs 旧表 ℒ {loss['旧']:.6g}（新/旧 {loss['新'] / loss['旧']:.4f}）；"
                  f"记账末行 N²·best_fitness {l_log:.6g}（与新表 ℒ 相对差 {abs(l_log - loss['新']) / loss['新']:.1e}）")
            print(f"       停止代数 {int(last['gen']):,}，累计候选 {int(last['cum_candidates']):,}，循环 {float(last['elapsed_s']):.0f} s")
        else:
            print(f"ε={tag} GSD 终表尚无")


if __name__ == "__main__":
    main()
