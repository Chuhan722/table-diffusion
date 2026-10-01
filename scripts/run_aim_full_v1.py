"""官方 AIM 经典实现（numpy mbi 1.x）驱动，plants 专用。

背景，mbi 2.0 jax 版在 69 字段全二阶 workload 下每轮需对约 2415 个候选
做模型投影，每个触发一次 XLA 编译，实测单轮约 35 分钟、预算烧速指向数
百轮，单场天级不可跑。经典 numpy 版（2024-06 commit 4152cc5）算法主体
与 2.0 版逐行同款（选择、预算、退火），仅推断底座为 numpy，无编译开销。

老版怪癖如实保留，AIM 的 super().__init__(epsilon, delta, prng) 把 prng
传进了 bounded 参数位，噪声实际走全局 np.random，故种子用 np.random.seed
控制，rho 换算 cdp_rho 与 2.0 版同款。

用法（需 mbi 1.x 环境，如 A6000 aim 环境）:
  python scripts/run_aim_full_v1.py --data plants --seed 1 \
      --mechanisms /root/pgm_v1 --out results/plants_eps1_aimfull_seed1.csv
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
    ap = argparse.ArgumentParser(description="官方 AIM 经典实现端到端基线")
    ap.add_argument("--data", required=True, help="数据目录名，如 plants")
    ap.add_argument("--data-root", default=None, help="data 根目录，默认仓库 data/")
    ap.add_argument("--mechanisms", required=True, help="pgm_v1 根目录，内含 mechanisms 包")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-model-size", type=float, default=80.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.mechanisms)
    sys.path.insert(0, os.path.join(args.mechanisms, "mechanisms"))
    np.random.seed(args.seed)

    from mbi import Dataset, Domain
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
    data = Dataset(df, domain)
    workload = [(cl, 1.0) for cl in itertools.combinations(data.domain, 2)]
    print(f"{args.data} 字段{len(sizes)} 行{len(df)} 全二阶{len(workload)}对 "
          f"eps={args.epsilon} delta={args.delta} seed={args.seed} "
          f"max_model_size={args.max_model_size} 实现=classic-numpy", flush=True)

    t0 = time.time()
    mech = AIM(args.epsilon, args.delta, max_model_size=args.max_model_size)
    _, synth = mech.run(data, workload, num_synth_rows=len(df))
    out_df = synth.df[list(df.columns)]
    out_df.to_csv(args.out, index=False)
    print(f"完成 {time.time() - t0:.0f}s 行{len(out_df)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
