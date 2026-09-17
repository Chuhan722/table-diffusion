"""plants 评价脚本，AIM 式与 GSD 式指标对比任意合成表与真实表。

评价与生成严格分离，只读表文件与查询文件不触碰引擎，
规模原因全部矢量化，等值题掩码计数，半空间题整数分表矩阵乘，
AIM 式二阶取全部字段对，三阶全组合五万个跑不动，取固定种子随机抽样，
综合分口径与 test300 一致，measured 加 heldout 加二阶三阶 TVD 四轴平均。
用法，./.venv/bin/python scripts/eval_plants.py 表1.csv 表2.csv ...
不给表时只报真实表自评与随机表基线。
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from resevo.dataset import load_queries, load_table  # noqa: E402
from resevo.metrics import assert_disjoint_workloads, load_heldout_queries  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "plants"
AIM3_SEED = 20260920
AIM3_SAMPLES = 300


def to_matrix(schema, rows):
    idx = [{v: k for k, v in enumerate(d)} for d in schema.domains]
    return np.array([[idx[j][r[j]] for j in range(len(r))] for r in rows], dtype=np.int8)


def spec_answers(specs, schema, X):
    """矢量化算一组查询在表上的计数答案。"""
    fpos = {name: j for j, name in enumerate(schema.fields)}
    out = np.zeros(len(specs), dtype=np.int64)
    for i, s in enumerate(specs):
        if s.conditions[0]["operator"] == "halfspace":
            c = s.conditions[0]
            w = np.zeros((schema.num_fields, 2), dtype=np.int64)
            for name, tab in c["scores"].items():
                j = fpos[name]
                for v, sc in tab.items():
                    w[j, int(v)] = int(sc)
            rs = X.astype(np.int64) @ (w[:, 1] - w[:, 0]) + int(w[:, 0].sum())
            out[i] = int((rs >= int(c["threshold"])).sum())
        else:
            mask = np.ones(len(X), dtype=bool)
            for c in s.conditions:
                j = fpos[c["attribute"]]
                v = list(schema.domains[j]).index(c["value"])
                mask &= X[:, j] == v
            out[i] = int(mask.sum())
    return out


def gsd_axis(specs, schema, X, real_total):
    ans = spec_answers(specs, schema, X)
    real = np.array([s.result for s in specs])
    err = np.abs(ans / len(X) - real / real_total)
    return float(err.mean()), float(err.max())


def marginal_l1(Xr, Xs, attrs):
    """一组字段上的边缘分布 L1 距离，比例口径。"""
    base = 1
    idx_r = np.zeros(len(Xr), dtype=np.int64)
    idx_s = np.zeros(len(Xs), dtype=np.int64)
    for j in attrs:
        idx_r = idx_r * 2 + Xr[:, j]
        idx_s = idx_s * 2 + Xs[:, j]
        base *= 2
    pr = np.bincount(idx_r, minlength=base) / len(Xr)
    ps = np.bincount(idx_s, minlength=base) / len(Xs)
    return float(np.abs(pr - ps).sum())


def aim_axes(schema, Xr, Xs):
    two = [marginal_l1(Xr, Xs, ab) for ab in combinations(range(schema.num_fields), 2)]
    rng = np.random.default_rng(AIM3_SEED)
    triples = set()
    while len(triples) < AIM3_SAMPLES:
        triples.add(tuple(sorted(int(j) for j in rng.choice(schema.num_fields, 3, replace=False))))
    three = [marginal_l1(Xr, Xs, t) for t in sorted(triples)]
    return (
        float(np.mean(two)), float(np.max(two)),
        float(np.mean(three)), float(np.max(three)),
    )


def evaluate_one(name, schema, Xs, Xr, measured, heldout, real_total):
    gm_avg, gm_max = gsd_axis(measured, schema, Xs, real_total)
    gh_avg, gh_max = gsd_axis(heldout, schema, Xs, real_total)
    h_eq = [s for s in heldout if s.conditions[0]["operator"] != "halfspace"]
    h_hs = [s for s in heldout if s.conditions[0]["operator"] == "halfspace"]
    ge_avg, _ = gsd_axis(h_eq, schema, Xs, real_total)
    gs_avg, _ = gsd_axis(h_hs, schema, Xs, real_total)
    a2_avg, a2_max, a3_avg, a3_max = aim_axes(schema, Xr, Xs)
    comp = float(np.mean([gm_avg, gh_avg, a2_avg / 2, a3_avg / 2]))
    print(f"\n== {name}，行数 {len(Xs)} ==")
    print(f"GSD 式 measured {len(measured)}   平均 {gm_avg:.6f}  最大 {gm_max:.6f}")
    print(f"GSD 式 heldout {len(heldout)}  平均 {gh_avg:.6f}  最大 {gh_max:.6f}")
    print(f"  其中等值高阶 {len(h_eq)} 平均 {ge_avg:.6f}，半空间 {len(h_hs)} 平均 {gs_avg:.6f}")
    print(f"AIM 式 全部二阶边缘 2346   平均 {a2_avg:.6f}  最大 {a2_max:.6f}")
    print(f"AIM 式 抽样三阶边缘 {AIM3_SAMPLES}  平均 {a3_avg:.6f}  最大 {a3_max:.6f}")
    print(
        f"综合误差分 {comp:.6f}，四组 measured {gm_avg:.6f}，heldout {gh_avg:.6f}，"
        f"二阶 TVD {a2_avg / 2:.6f}，三阶 TVD {a3_avg / 2:.6f}"
    )
    return comp


def main() -> None:
    parser = argparse.ArgumentParser(description="plants AIM 与 GSD 式评价对比")
    parser.add_argument("tables", nargs="*", help="待评合成表 CSV 路径列表")
    parser.add_argument("--baseline-seeds", type=int, default=3, help="随机基线种子数")
    args = parser.parse_args()

    schema, real_rows = load_table(str(DATA_DIR / "plants.csv"))
    measured = load_queries(str(DATA_DIR / "measured_9248query.json"))
    heldout = load_heldout_queries(str(DATA_DIR / "heldout_1224query.json"))
    assert_disjoint_workloads(measured, heldout)
    print(f"评价集校验通过，measured {len(measured)} 条与 heldout {len(heldout)} 条语义零交集")
    Xr = to_matrix(schema, real_rows)
    real_total = len(real_rows)

    evaluate_one("真实表自评，全零基准", schema, Xr, Xr, measured, heldout, real_total)

    comps = []
    for seed in range(args.baseline_seeds):
        rng = np.random.default_rng(seed)
        Xs = np.stack(
            [rng.integers(len(schema.domains[j]), size=real_total).astype(np.int8)
             for j in range(schema.num_fields)], axis=1,
        )
        comps.append(
            evaluate_one(f"随机表基线 种子 {seed}", schema, Xs, Xr, measured, heldout, real_total)
        )
    if comps:
        print(f"\n随机基线综合误差分均值 {np.mean(comps):.6f} ± {np.std(comps):.6f}")

    for path in args.tables:
        s2, rows = load_table(path)
        if tuple(s2.fields) != tuple(schema.fields):
            raise AssertionError(f"{path} 字段名与真实表不一致")
        idx = [{v: k for k, v in enumerate(schema.domains[j])} for j in range(schema.num_fields)]
        Xs = np.array(
            [[idx[j][r[j]] for j in range(len(r))] for r in rows], dtype=np.int8
        )
        evaluate_one(path, schema, Xs, Xr, measured, heldout, real_total)


if __name__ == "__main__":
    main()
