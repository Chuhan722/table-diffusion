"""纤维平均：从我们自己的收敛表洗牌出"没测的对"的目标（工作笔记第五十八步）。

一条链连洗 --passes 趟（scripts/fiber_shuffle.py 的组内精确重排，测过统计逐格不变），
从第 2 趟起每趟把全部字段对的二阶边际累加平均。平均值就是纤维平均（≈最大熵）的估计，
是小数目标不是表；落表交给引擎（add_infer_targets.py 把它写进考卷）。
只用收敛表 + 考卷里"哪些属性组合测过"，零对手产物、零额外预算。

输出 npz 键与 pgm_exact_marginals.py 对齐（columns、doms、rows、measured_pairs、m2_i_j），
测过的对写表自身的计数（洗牌不改它们），没测的对写平均值；另记 passes、seed、source。

用法: .venv/bin/python scripts/fiber_targets.py --table results/stage1/adult_eps10_hybrid_seed1.csv \
        --exam data/adult_aimrep10_s1/measured_6767query.json --out results/stage2/adult_eps10_fiber_seed1_targets.npz [--passes 100] [--seed 1]
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fiber_shuffle import group_ids, load_exam_sets, marg, neighbors_from_sets, shuffle_column  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--real", default=str(ROOT / "data/adult/adult.csv"), help="只借真表定值域上界，不读其计数")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    columns = list(df.columns)
    X = np.stack([df[c].to_numpy(dtype=np.int64) for c in columns], axis=1)
    N, F = X.shape
    sets = load_exam_sets(Path(args.exam))
    nb = neighbors_from_sets(columns, sets)
    real = pd.read_csv(args.real)
    assert list(real.columns) == columns
    doms = [int(max(real[c].max(), X[:, j].max())) + 1 for j, c in enumerate(columns)]
    pairs = list(combinations(range(F), 2))
    measured = sorted({tuple(sorted((columns.index(s[0]), columns.index(s[1])))) for s in sets if len(s) == 2})
    mset = set(measured)
    unmeas = [p for p in pairs if p not in mset]

    before = {p: marg(X, p, doms) for p in measured}
    acc = {p: np.zeros(doms[p[0]] * doms[p[1]], dtype=np.float64) for p in unmeas}
    rng = np.random.default_rng(args.seed)
    n = 0
    for t in range(1, args.passes + 1):
        for a in rng.permutation(F):
            shuffle_column(X, int(a), group_ids(X, nb[int(a)]), rng)
        if t >= 2:
            for p in unmeas:
                acc[p] += marg(X, p, doms)
            n += 1
    for p in measured:
        if not np.array_equal(marg(X, p, doms), before[p]):
            raise SystemExit(f"自检失败：测过的对 {p} 边际变了")

    out = {"columns": np.array(columns), "doms": np.array(doms), "rows": np.array(N),
           "measured_pairs": np.array(measured, dtype=np.int64).reshape(-1, 2),
           "passes": np.array(args.passes), "averaged": np.array(n), "seed": np.array(args.seed),
           "source": np.array("fiber_avg"), "table": np.array(str(Path(args.table).resolve().relative_to(ROOT)))}
    for p in measured:
        out[f"m2_{p[0]}_{p[1]}"] = before[p]
    for p in unmeas:
        out[f"m2_{p[0]}_{p[1]}"] = acc[p] / n
    np.savez(args.out, **out)
    print(f"写出 {args.out}：测过 {len(measured)} 对原计数，没测 {len(unmeas)} 对取 {n} 趟平均（共洗 {args.passes} 趟）")


if __name__ == "__main__":
    main()
