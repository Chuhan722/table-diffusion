"""按 AIM 是否测过分组阅二阶卷：测过的对 / 没测的对 / 一阶，外加独立填充参照。

用法:
  python scripts/score_pair_groups.py --real data/adult/adult.csv \
      --dump results/aimreplay/adult_eps0p1_aimfull_seed1_measurements.json \
      标签=表.csv [标签=表.csv ...]

每对误差 = L1/N（合成表与真表该对二阶计数差绝对值之和除以行数），
组内取均值，单位 1e-6。独立参照 = 真一阶乘积（Π p_a p_b）当合成边际，
是"不花预算只知道一阶"的下界参照。--dump 给出 AIM 测过的团，没给则只报总体。
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def pair_hist(df: pd.DataFrame, a: str, b: str, sa: int, sb: int) -> np.ndarray:
    key = df[a].to_numpy(dtype=np.int64) * sb + df[b].to_numpy(dtype=np.int64)
    return np.bincount(key, minlength=sa * sb).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True)
    ap.add_argument("--dump", default=None, help="AIM 测量录像 JSON，提供测过的团")
    ap.add_argument("tables", nargs="+", help="标签=路径")
    args = ap.parse_args()

    real = pd.read_csv(args.real)
    cols = list(real.columns)
    sizes = {c: int(real[c].max()) + 1 for c in cols}
    n = len(real)
    pairs = list(combinations(cols, 2))

    measured_pairs: set[tuple[str, str]] = set()
    if args.dump:
        dump = json.loads(Path(args.dump).read_text(encoding="utf-8"))
        for m in dump["measurements"]:
            if len(m["clique"]) == 2:
                a, b = m["clique"]
                measured_pairs.add((a, b) if cols.index(a) < cols.index(b) else (b, a))

    real_pairs = {(a, b): pair_hist(real, a, b, sizes[a], sizes[b]) for a, b in pairs}
    real_ones = {c: np.bincount(real[c].to_numpy(), minlength=sizes[c]).astype(np.float64) for c in cols}

    def pair_errors(get_hist) -> dict[tuple[str, str], float]:
        return {p: float(np.abs(get_hist(p) - real_pairs[p]).sum()) / n for p in pairs}

    rows = []
    indep = pair_errors(lambda p: np.outer(real_ones[p[0]], real_ones[p[1]]).ravel() / n)
    rows.append(("独立参照(真一阶)", indep, None))
    for item in args.tables:
        label, path = item.split("=", 1)
        syn = pd.read_csv(path)
        if list(syn.columns) != cols:
            raise SystemExit(f"{path} 列名与真表不一致")
        errs = pair_errors(lambda p, syn=syn: pair_hist(syn, p[0], p[1], sizes[p[0]], sizes[p[1]]))
        one = np.mean([
            np.abs(np.bincount(syn[c].to_numpy(), minlength=sizes[c]) - real_ones[c]).sum() / n for c in cols
        ])
        rows.append((label, errs, float(one)))

    meas = [p for p in pairs if p in measured_pairs]
    unmeas = [p for p in pairs if p not in measured_pairs]
    cells = lambda ps: np.mean([sizes[a] * sizes[b] for a, b in ps]) if ps else float("nan")
    print(f"行数 {n}，字段 {len(cols)}，对数 {len(pairs)}；AIM 测过 {len(meas)} 对（均 {cells(meas):.0f} 格），"
          f"没测 {len(unmeas)} 对（均 {cells(unmeas):.0f} 格）")
    hdr = f"{'表':<28}{'二阶总体':>10}{'测过的对':>10}{'没测的对':>10}{'一阶':>8}"
    print(hdr)
    for label, errs, one in rows:
        tot = np.mean([errs[p] for p in pairs]) * 1e6
        m = np.mean([errs[p] for p in meas]) * 1e6 if meas else float("nan")
        u = np.mean([errs[p] for p in unmeas]) * 1e6 if unmeas else float("nan")
        one_s = f"{one * 1e6:8.0f}" if one is not None else f"{'-':>8}"
        print(f"{label:<28}{tot:10.0f}{m:10.0f}{u:10.0f}{one_s}")


if __name__ == "__main__":
    main()
