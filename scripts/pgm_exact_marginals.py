"""PGM 机制核查：同一份拟合好的 PGM 模型，分别「不出表直接算边际」与「出表后数边际」。

目的是查 1B 高 ε 我们赢 PGM 的来源：是 PGM 拟合出的分布本身差，还是分布→有限表那一步
（mbi synthetic_data 的分组条件随机取整）把精度丢了。拟合部分与 scripts/run_pgm_replay.py
逐字同配置（同录像、iters、known_total），之后：
  1. np.random.seed(seed) 出 N 行表写 CSV（应与第一阶段 pgmrep10k 表逐格相同，本地核对）；
  2. 模型精确边际：全部一阶、全部二阶、三阶（与 score_any 同选法：≤2000 全取，否则 rng(11) 抽 2000）；
  3. 等值留出题精确答案（模型投影到题目字段后取格）；
  4. 半空间留出题不能精确算，用同模型出 mult 倍行数的大表近似（答案 / mult），大表边际一并存下做校验。
全部存进一个 npz，阅卷在本地 logs_eps/stage2/grade_pgm_exact.py 做（本地有真表、我们的表、留出题）。

用法（A6000 aim2 环境，纯 CPU）：
  PYTHONPATH=/root/private-pgm/src JAX_PLATFORMS=cpu /root/miniconda3/envs/aim2/bin/python scripts/pgm_exact_marginals.py \
      --dump results/aimreplay/adult_eps10_aimfull_seed1_measurements.json --seed 1 --iters 10000 \
      --heldout data/adult/heldout_1224query.json --out-prefix results/stage2/adult_eps10_pgmexact_seed1
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations

import numpy as np
import pandas as pd


def select_triples(F: int, cap: int = 2000) -> list[tuple[int, ...]]:
    """与 scripts/score_any.py 逐字同一套三元组选法。"""
    all_trips = list(combinations(range(F), 3))
    rng = np.random.default_rng(11)
    if len(all_trips) > cap:
        idx = rng.choice(len(all_trips), size=cap, replace=False)
        return [all_trips[i] for i in sorted(idx)]
    return all_trips


def table_marg(X: np.ndarray, cols: tuple[int, ...], doms: list[int]) -> np.ndarray:
    key = X[:, cols[0]].astype(np.int64)
    size = doms[cols[0]]
    for c in cols[1:]:
        key = key * doms[c] + X[:, c]
        size *= doms[c]
    return np.bincount(key, minlength=size).astype(np.float64)


def table_answers(queries: list[dict], X: np.ndarray, fpos: dict[str, int], doms: list[int]) -> np.ndarray:
    out = np.zeros(len(queries), dtype=np.float64)
    for i, q in enumerate(queries):
        c0 = q["conditions"][0]
        if c0["operator"] == "halfspace":
            w = np.zeros((len(doms), max(doms)), dtype=np.int64)
            for name, tab in c0["scores"].items():
                j = fpos[name]
                for v, sc in tab.items():
                    w[j, int(v)] = int(sc)
            rs = np.zeros(len(X), dtype=np.int64)
            for j in range(len(doms)):
                rs += w[j, X[:, j]]
            out[i] = float((rs >= int(c0["threshold"])).sum())
        else:
            mask = np.ones(len(X), dtype=bool)
            for c in q["conditions"]:
                mask &= X[:, fpos[c["attribute"]]] == int(c["value"])
            out[i] = float(mask.sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="PGM 模型精确边际 vs 出表边际")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--known-total", choices=["rows", "auto"], default="rows")
    ap.add_argument("--heldout", required=True, help="留出题 json（等值精确算，半空间用大表近似）")
    ap.add_argument("--big-mult", type=int, default=20, help="近似半空间用的大表行数倍数")
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    from mbi import Domain, estimation, junction_tree
    from mbi.marginal_loss import LinearMeasurement

    with open(args.dump, "r", encoding="utf-8") as fh:
        dump = json.load(fh)
    columns = list(dump["columns"])
    doms = [int(s) for s in dump["domain_sizes"]]
    domain = Domain(columns, doms)
    rows = int(dump["num_rows"])
    F = len(columns)
    fpos = {c: j for j, c in enumerate(columns)}

    measurements = []
    for rec in dump["measurements"]:
        cl = tuple(rec["clique"])
        y = np.asarray(rec["y"], dtype=np.float64)
        measurements.append(LinearMeasurement(y, cl, stddev=float(rec["sigma"])))
    cliques = [m.clique for m in measurements]
    model_mb = junction_tree.hypothetical_model_size(domain, cliques)
    print(f"{dump['data']} eps={dump['epsilon']} 录像种子={dump['seed']} 测量 {len(measurements)} 条，"
          f"联结树参数表 {model_mb:.1f} MB，iters={args.iters} rows={rows} known_total={args.known_total}", flush=True)

    t0 = time.time()
    known_total = float(rows) if args.known_total == "rows" else None
    model = estimation.MirrorDescent().estimate(domain, measurements, known_total=known_total, iters=args.iters)
    print(f"拟合 {time.time() - t0:.0f} 秒", flush=True)

    # 1. 与第一阶段逐字同法出 N 行表
    np.random.seed(args.seed)
    synth = model.synthetic_data(rows=rows)
    X = np.stack([np.asarray(synth.data[c], dtype=np.int64) for c in columns], axis=1)
    pd.DataFrame(X, columns=columns).to_csv(args.out_prefix + ".csv", index=False)

    out: dict[str, np.ndarray] = {}
    out["columns"] = np.array(columns)
    out["doms"] = np.array(doms)
    out["rows"] = np.array(rows)
    out["measured_pairs"] = np.array(sorted({tuple(sorted(fpos[a] for a in cl)) for cl in cliques if len(cl) == 2}),
                                     dtype=np.int64).reshape(-1, 2)

    # 2. 模型精确边际。每个新字段组合都让 JAX 编译一个新程序，几千个程序会把进程内存映射数顶到
    #    vm.max_map_count（65530）而崩（LLVM "Cannot allocate memory"），故每 50 次投影清一次编译缓存。
    import jax
    ncall = [0]

    def project_np(attrs: tuple[str, ...], flatten: bool) -> np.ndarray:
        ncall[0] += 1
        if ncall[0] % 50 == 0:
            jax.clear_caches()
        f = model.project(attrs).transpose(attrs)
        return np.asarray(f.datavector(flatten=flatten), dtype=np.float64)

    def exact(cols: tuple[int, ...]) -> np.ndarray:
        return project_np(tuple(columns[c] for c in cols), True)

    t1 = time.time()
    for j in range(F):
        out[f"m1_{j}"] = exact((j,))
    pairs = list(combinations(range(F), 2))
    for p in pairs:
        out["m2_" + "_".join(map(str, p))] = exact(p)
    trips = select_triples(F)
    for t in trips:
        out["m3_" + "_".join(map(str, t))] = exact(t)
    out["triples"] = np.array(trips, dtype=np.int64)
    print(f"精确边际 一阶 {F} 二阶 {len(pairs)} 三阶 {len(trips)}，{time.time() - t1:.0f} 秒", flush=True)

    # 3. 等值题精确答案
    with open(args.heldout, "r", encoding="utf-8") as fh:
        hq = json.load(fh)["queries"]
    eq = [q for q in hq if q["conditions"][0]["operator"] != "halfspace"]
    hs = [q for q in hq if q["conditions"][0]["operator"] == "halfspace"]
    t2 = time.time()
    cache: dict[tuple[int, ...], np.ndarray] = {}
    eq_ans = np.zeros(len(eq), dtype=np.float64)
    reused = 0
    for i, q in enumerate(eq):
        cs = sorted((fpos[c["attribute"]], int(c["value"])) for c in q["conditions"])
        cols = tuple(c for c, _ in cs)
        if cols not in cache:
            key = f"m{len(cols)}_" + "_".join(map(str, cols))
            if key in out:  # 三阶（及以下）字段组前面已精确算过，直接复用，省一次 JAX 编译
                cache[cols] = out[key].reshape([doms[c] for c in cols])
                reused += 1
            else:
                cache[cols] = project_np(tuple(columns[c] for c in cols), False)
        eq_ans[i] = cache[cols][tuple(v for _, v in cs)]
    out["eq_exact"] = eq_ans
    print(f"等值题 {len(eq)} 道精确答案（{len(cache)} 个不同字段组，其中 {reused} 组复用已算边际），"
          f"{time.time() - t2:.0f} 秒", flush=True)

    # 4. 大表近似（半空间专用，边际一并存下校验），以及 N 行表自身的答案
    t3 = time.time()
    np.random.seed(args.seed + 1000)
    big = model.synthetic_data(rows=rows * args.big_mult)
    XB = np.stack([np.asarray(big.data[c], dtype=np.int64) for c in columns], axis=1)
    out["big_mult"] = np.array(args.big_mult)
    for j in range(F):
        out[f"big_m1_{j}"] = table_marg(XB, (j,), doms) / args.big_mult
    for p in pairs:
        out["big_m2_" + "_".join(map(str, p))] = table_marg(XB, p, doms) / args.big_mult
    for t in trips:
        out["big_m3_" + "_".join(map(str, t))] = table_marg(XB, t, doms) / args.big_mult
    out["hs_big"] = table_answers(hs, XB, fpos, doms) / args.big_mult
    out["eq_big"] = table_answers(eq, XB, fpos, doms) / args.big_mult
    print(f"大表 {len(XB)} 行近似：半空间 {len(hs)} 道，{time.time() - t3:.0f} 秒", flush=True)

    np.savez_compressed(args.out_prefix + "_exact.npz", **out)
    print(f"写出 {args.out_prefix}.csv 与 {args.out_prefix}_exact.npz，总耗时 {time.time() - t0:.0f} 秒", flush=True)


if __name__ == "__main__":
    main()
