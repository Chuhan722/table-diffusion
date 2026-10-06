"""官方 AIM 端到端基线驱动，直调 mechanisms/aim.py 的 AIM 类。

口径与第二十一步 nltcs aimfull 档案完全一致，官方默认 unbounded、
max_model_size 80、workload 全二阶字段对与考卷同卷、行数等于原表、
prng 注入种子可复现。AIM 吃真数据自己做 DP 选择加测量，不吃噪声卷。

jax 每轮换团结构重编译耗尽 JIT 内存池的老坑，包装 MirrorDescent.estimate
每次调用后 clear_caches，官方代码零改动。

用法，需能 import mbi 2.0 与 mechanisms（PYTHONPATH 或 venv）:
  python scripts/run_aim_full.py --data plants --seed 1 \
      --out results/plants_eps1_aimfull_seed1.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="官方 AIM 端到端基线")
    ap.add_argument("--data", required=True, help="数据目录名，如 plants")
    ap.add_argument("--data-root", default=None, help="data 根目录，默认仓库 data/")
    ap.add_argument("--mechanisms", default="/home/chuhan/projects/private-pgm/mechanisms",
                    help="官方 mechanisms 目录")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-model-size", type=float, default=80.0)
    ap.add_argument("--clear-caches-every", type=int, default=0,
                    help="每 N 次 estimate 后 jax.clear_caches，0 为不清；防 LLVM JIT 内存池崩")
    ap.add_argument("--clear-caches-every-projects", type=int, default=0,
                    help="每 N 次模型投影后 jax.clear_caches，0 为不清；大候选池科目单轮即可爆 JIT 池，须轮内清")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-measurements", default="",
                    help="把 AIM 全部测量（轮次、团、sigma、加噪向量）录成 JSON，官方代码零改动，"
                         "靠包装 MirrorDescent.estimate 截取末次调用的测量列表")
    args = ap.parse_args()

    sys.path.insert(0, args.mechanisms)

    import jax
    jax.config.update("jax_enable_compilation_cache", False)

    from mbi import Dataset, Domain, estimation

    captured: dict = {"measurements": None, "calls": 0}
    if args.dump_measurements:
        orig_estimate_rec = estimation.MirrorDescent.estimate

        def recording_estimate(self, domain, loss_fn, *a, **k):
            captured["calls"] += 1
            if isinstance(loss_fn, list):
                # AIM 每轮都把累计全部测量整列传入，末次调用即完整录像，顺序即轮次
                captured["measurements"] = list(loss_fn)
            return orig_estimate_rec(self, domain, loss_fn, *a, **k)

        estimation.MirrorDescent.estimate = recording_estimate

    if args.clear_caches_every > 0:
        orig_estimate = estimation.MirrorDescent.estimate
        counter = {"n": 0}

        def patched_estimate(self, *a, **k):
            out = orig_estimate(self, *a, **k)
            counter["n"] += 1
            if counter["n"] % args.clear_caches_every == 0:
                jax.clear_caches()
            return out

        estimation.MirrorDescent.estimate = patched_estimate

    if args.clear_caches_every_projects > 0:
        from mbi import markov_random_field
        orig_project = markov_random_field.MarkovRandomField.project
        pcounter = {"n": 0}

        def patched_project(self, *a, **k):
            out = orig_project(self, *a, **k)
            pcounter["n"] += 1
            if pcounter["n"] % args.clear_caches_every_projects == 0:
                jax.clear_caches()
            return out

        markov_random_field.MarkovRandomField.project = patched_project

    from aim import AIM

    root = args.data_root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    csv = os.path.join(root, args.data, f"{args.data}.csv")
    df = pd.read_csv(csv)
    sizes = []
    for c in df.columns:
        lo, hi, uniq = int(df[c].min()), int(df[c].max()), df[c].nunique()
        if lo != 0 or uniq != hi + 1:
            raise SystemExit(f"{c} 域不连续 min={lo} max={hi} unique={uniq}，需人工核对")
        sizes.append(hi + 1)
    domain = Domain(list(df.columns), sizes)
    data = Dataset(df.to_dict("list"), domain)
    workload = [(cl, 1.0) for cl in itertools.combinations(domain, 2)]
    print(f"{args.data} 字段{len(sizes)} 行{len(df)} 全二阶{len(workload)}对 "
          f"eps={args.epsilon} delta={args.delta} seed={args.seed} "
          f"max_model_size={args.max_model_size}", flush=True)

    t0 = time.time()
    mech = AIM(args.epsilon, args.delta,
               prng=np.random.RandomState(args.seed),
               max_model_size=args.max_model_size)
    _, synth = mech.run(data, workload, num_synth_rows=len(df))
    out_df = pd.DataFrame({c: np.asarray(synth.data[c]) for c in df.columns})
    out_df.to_csv(args.out, index=False)
    print(f"完成 {time.time() - t0:.0f}s 行{len(out_df)} -> {args.out}", flush=True)

    if args.dump_measurements:
        import json

        from mbi.marginal_loss import LinearMeasurement

        ms = captured["measurements"]
        if not ms:
            raise SystemExit("录像为空：estimate 从未收到测量列表，请核对 mbi 版本")
        records = []
        for i, m in enumerate(ms):
            if not isinstance(m, LinearMeasurement):
                raise SystemExit(f"第 {i} 条测量不是 LinearMeasurement: {type(m)}")
            cl = tuple(m.clique)
            shape = [int(domain[a]) for a in cl]
            y = np.asarray(m.noisy_measurement, dtype=np.float64).ravel()
            if y.size != int(np.prod(shape)):
                raise SystemExit(f"第 {i} 条测量长度 {y.size} 与团域 {shape} 不符")
            records.append({
                "round": i, "clique": list(cl), "shape": shape,
                "sigma": float(m.stddev), "y": y.tolist(),
            })
        payload = {
            "data": args.data, "epsilon": args.epsilon, "delta": args.delta,
            "seed": args.seed, "rho": float(mech.rho),
            "num_rows": int(len(df)), "columns": list(df.columns),
            "domain_sizes": sizes, "max_model_size": args.max_model_size,
            "estimate_calls": captured["calls"],
            "layout": "y 为 C 序展平的团边际计数向量，shape 按 clique 顺序；值编码 0..k-1 与 CSV 同",
            "measurements": records,
        }
        with open(args.dump_measurements, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        n1 = sum(1 for r in records if len(r["clique"]) == 1)
        print(f"录像 {len(records)} 条（一阶 {n1}，二阶 {len(records) - n1}）"
              f" estimate 调用 {captured['calls']} 次 -> {args.dump_measurements}", flush=True)


if __name__ == "__main__":
    main()
