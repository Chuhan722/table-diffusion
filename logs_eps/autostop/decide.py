#!/usr/bin/env python
"""第七十二步 探针并行自动停：判停法。

用法：decide.py 模拟卷.json 地板停表G.csv 终表.csv 输出.json
对模拟卷 y−e 的每题算绝对误差均值 L1_B = mean|y_B − q(表)|（第七十一步定的判据，含一阶题，不清洗）：
G 的 L1_B 低于终表 → 有谷 → 地板停（floor）；否则收敛停（conv）。只用带噪考卷，不看真表。
"""
import json
import sys
import time

import numpy as np


def load(path: str):
    """读整数表（不依赖 pandas，远端 venv 没装）。"""
    with open(path, encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    arr = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.int64, ndmin=2)
    return {name: arr[:, j] for j, name in enumerate(header)}


def counts(cols: dict, qs: list) -> np.ndarray:
    cache: dict = {}
    out = np.empty(len(qs))
    for i, q in enumerate(qs):
        attrs = tuple(c["attribute"] for c in q["conditions"])
        vals = tuple(int(c["value"]) for c in q["conditions"])
        if attrs not in cache:
            keys, n = np.unique(np.stack([cols[a] for a in attrs], axis=1), axis=0, return_counts=True)
            cache[attrs] = {tuple(int(v) for v in k): int(c) for k, c in zip(keys, n)}
        out[i] = cache[attrs].get(vals, 0)
    return out


def main() -> None:
    t0 = time.perf_counter()
    exam_b, g_csv, fin_csv, out_json = sys.argv[1:5]
    qb = json.load(open(exam_b, encoding="utf-8"))["queries"]
    yb = np.array([q["result"] for q in qb], dtype=np.float64)
    l1 = {}
    for key, path in (("G", g_csv), ("final", fin_csv)):
        l1[key] = float(np.mean(np.abs(yb - counts(load(path), qb))))
    decision = "floor" if l1["G"] < l1["final"] else "conv"
    res = dict(decision=decision, L1_B_G=l1["G"], L1_B_final=l1["final"],
               G_vs_final=(l1["G"] - l1["final"]) / l1["final"], questions=len(qb),
               seconds=time.perf_counter() - t0)
    json.dump(res, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"判停法 {decision}：L1_B G={l1['G']:.4f} 终={l1['final']:.4f}（G 对终 {res['G_vs_final']:+.2%}，{len(qb)} 题，{res['seconds']:.1f} 秒）")
    print(decision)


if __name__ == "__main__":
    main()
