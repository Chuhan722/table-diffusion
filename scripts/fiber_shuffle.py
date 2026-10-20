"""纤维洗牌：收敛表在「测过的统计看不出区别」的行之间重排属性值，只洗没测的结构。

原则（工作笔记第五十八步）：满足测过统计的表构成一个纤维，典型成员≈最大熵；引擎沿下降路径停在
纤维里一张带路径指纹的少数派表上。洗牌在纤维内做公平随机走动，停下的那张是典型成员。
动作：对属性 a，把行按 a 的全部测过邻居（考卷里与 a 同题的属性）取值分组，组内随机重排 a 列。
组内邻居全同 ⇒ a 的一维、a 与每个邻居的测过边际（含更高阶的邻居联合）精确不变，损失精确不变；
变的只有 a 与非邻居（没测的对）。15 个属性随机顺序各洗一次为一趟，固定趟数。

只用考卷里「哪些属性组合测过」这一信息，不用任何对手产物。诊断（可选 --npz）只用于记录每趟
离真值/离 PGM 分布的距离，不参与洗牌。

用法: .venv/bin/python scripts/fiber_shuffle.py --table results/stage1/adult_eps10_hybrid_seed1.csv \
        --exam data/adult_aimrep10_s1/measured_6767query.json --out results/stage2/adult_eps10_hybridshuf_seed1.csv \
        [--passes 30] [--seed 1] [--npz results/stage2/adult_eps10_pgmexact_seed1_exact.npz] [--log 路径]
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_exam_sets(path: Path) -> tuple[list[str], list[tuple[str, ...]]]:
    """返回考卷字段顺序无关的测过字段集合列表（去重）。"""
    q = json.load(open(path, encoding="utf-8"))["queries"]
    sets = sorted({tuple(sorted(c["attribute"] for c in x["conditions"])) for x in q})
    for x in q:
        for c in x["conditions"]:
            if c["operator"] != "==":
                raise SystemExit(f"考卷含非等值题 {c['operator']}，纤维洗牌只支持边际格子题")
    return sets


def neighbors_from_sets(columns: list[str], sets: list[tuple[str, ...]]) -> list[list[int]]:
    pos = {c: j for j, c in enumerate(columns)}
    nb: list[set[int]] = [set() for _ in columns]
    for s in sets:
        idx = [pos[c] for c in s]
        for a in idx:
            nb[a].update(b for b in idx if b != a)
    return [sorted(v) for v in nb]


def group_ids(X: np.ndarray, cols: list[int]) -> np.ndarray:
    if not cols:
        return np.zeros(len(X), dtype=np.int64)
    _, inv = np.unique(X[:, cols], axis=0, return_inverse=True)
    return inv.reshape(-1).astype(np.int64)


def shuffle_column(X: np.ndarray, a: int, gid: np.ndarray, rng: np.random.Generator) -> None:
    """组内随机重排第 a 列：按（组号，随机数）排序取值，按（组号，原序）排序落位。"""
    n = len(X)
    src = np.lexsort((rng.random(n), gid))
    dst = np.lexsort((np.arange(n), gid))
    X[dst, a] = X[src, a].copy()


def marg(X: np.ndarray, cols: tuple[int, ...], doms: list[int]) -> np.ndarray:
    key = X[:, cols[0]].astype(np.int64)
    size = doms[cols[0]]
    for c in cols[1:]:
        key = key * doms[c] + X[:, c]
        size *= doms[c]
    return np.bincount(key, minlength=size).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--npz", default="", help="PGM 精确边际，仅诊断用")
    ap.add_argument("--real", default=str(ROOT / "data/adult/adult.csv"), help="真表，仅诊断用")
    ap.add_argument("--log", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    columns = list(df.columns)
    X = np.stack([df[c].to_numpy(dtype=np.int64) for c in columns], axis=1)
    N, F = X.shape
    sets = load_exam_sets(Path(args.exam))
    nb = neighbors_from_sets(columns, sets)
    real = pd.read_csv(args.real)
    assert list(real.columns) == columns
    R = np.stack([real[c].to_numpy(dtype=np.int64) for c in columns], axis=1)
    doms = [int(max(R[:, j].max(), X[:, j].max())) + 1 for j in range(F)]
    pairs = list(combinations(range(F), 2))
    measured = {tuple(sorted((columns.index(s[0]), columns.index(s[1])))) for s in sets if len(s) == 2}
    unmeas = [p for p in pairs if p not in measured]
    truth2 = {p: marg(R, p, doms) for p in pairs}
    pgm2 = None
    if args.npz:
        z = np.load(args.npz)
        assert list(z["columns"]) == columns
        pgm2 = {p: z["m2_" + "_".join(map(str, p))] for p in pairs}

    out_lines: list[str] = []

    def say(s: str) -> None:
        print(s)
        out_lines.append(s)

    # 测过的统计快照：每个测过字段集合的完整边际表，洗后必须逐格相同
    before = {s: marg(X, tuple(columns.index(c) for c in s), doms) for s in sets}
    X0 = X.copy()

    def err(mm: dict, ps: list) -> float:
        return float(np.mean([np.abs(marg(X, p, doms) - mm[p]).sum() / N for p in ps]) * 1e6)

    def dist0(ps: list) -> float:
        return float(np.mean([np.abs(marg(X, p, doms) - marg(X0, p, doms)).sum() / N for p in ps]) * 1e6)

    say(f"表 {args.table} 行 {N} 字段 {F}；考卷测过集合 {len(sets)}（二阶 {len(measured)}，没测 {len(unmeas)}）；趟数 {args.passes} 种子 {args.seed}")
    for a in range(F):
        gid = group_ids(X, nb[a])
        cnt = np.bincount(gid)
        movable = float(cnt[cnt >= 2].sum() / N)
        say(f"  {columns[a]} 邻居 {len(nb[a])} 组数 {len(cnt)} 可换行占比 {movable:.3f}")
    hdr = "趟 | 没测对 vs 真值 | 测过对 vs 真值 | 没测对离起点"
    if pgm2 is not None:
        hdr += " | 没测对 vs PGM分布 | 没测对离PGM分布"
    say(hdr)

    def report(t: int) -> None:
        row = f"{t:3d} | {err(truth2, unmeas):10,.0f} | {err(truth2, sorted(measured)):10,.0f} | {dist0(unmeas):10,.0f}"
        if pgm2 is not None:
            e_pgm = float(np.mean([np.abs(pgm2[p] - truth2[p]).sum() / N for p in unmeas]) * 1e6)
            row += f" | {e_pgm:10,.0f} | {err(pgm2, unmeas):10,.0f}"
        say(row)

    rng = np.random.default_rng(args.seed)
    report(0)
    for t in range(1, args.passes + 1):
        for a in rng.permutation(F):
            shuffle_column(X, int(a), group_ids(X, nb[int(a)]), rng)
        if t <= 5 or t % 5 == 0 or t == args.passes:
            report(t)

    for s in sets:
        after = marg(X, tuple(columns.index(c) for c in s), doms)
        if not np.array_equal(after, before[s]):
            raise SystemExit(f"自检失败：测过集合 {s} 边际变了")
    say(f"自检通过：{len(sets)} 个测过集合的边际逐格不变")
    pd.DataFrame(X, columns=columns).to_csv(args.out, index=False)
    say(f"已写 {args.out}")
    if args.log:
        Path(args.log).write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
