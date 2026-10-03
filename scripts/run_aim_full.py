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
    args = ap.parse_args()

    sys.path.insert(0, args.mechanisms)

    import jax
    jax.config.update("jax_enable_compilation_cache", False)

    from mbi import Dataset, Domain, estimation

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


if __name__ == "__main__":
    main()
