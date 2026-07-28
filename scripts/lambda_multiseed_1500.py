"""
λ0.5 vs λ0.6 多种子精修（nltcs, 1500 轮, seed 0/1/2, α1.5→6）

目的：定论 λ=0.5 与 0.6 的差异是否显著（还是落在种子噪声内）。
1500 轮单种子曾显示 λ0.5 优 3.6%，但单种子不足以区分，本次补多种子 + 配对 t 检验。

种子复用：seed=0 的两点已跑过，本脚本只补 seed=1/2（每 λ 2 次）。
- λ0.5 seed=0: 7.3124e7  (outputs/geometric_alpha1p5_6_2026-07-28_0559/geometric_a1p5_6)
- λ0.6 seed=0: 7.5786e7  (outputs/geometric_lambda0p6_1500_2026-07-28_1003/geometric_a1p5_6_lambda0p6)

用法（单 λ、指定补跑种子，配合 CUDA_VISIBLE_DEVICES 并行）：
    CUDA_VISIBLE_DEVICES=1 python scripts/lambda_multiseed_1500.py --lambda 0.5 --seeds 1 2
    CUDA_VISIBLE_DEVICES=2 python scripts/lambda_multiseed_1500.py --lambda 0.6 --seeds 1 2

两个进程各自把新种子结果写到独立目录，避免冲突。t 检验汇总由 aggregate 步骤单独做。
"""
import os
import json
import time
import argparse
import numpy as np
import pandas as pd

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.io import save_run

SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

N_RECORDS = 16181
N_ROUNDS = 1500

PARAMS = {
    "beta": 1.0, "h": 0.8, "rho": 0.01, "eta": 0.5, "mu": 0.01,
    "device": "cuda", "eval_method": "vectorized", "batch_size": 256,
    "init_method": "marginal", "log_every": 100,
    "distance_mode": "geometric",
    "alpha_min": 1.5, "alpha_max": 6.0, "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", type=float, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    lam_tag = str(args.lam).replace(".", "p")
    out_root = f"outputs/lambda_multiseed_1500/lambda{lam_tag}"
    os.makedirs(out_root, exist_ok=True)

    for seed in args.seeds:
        print(f"\n{'='*60}\nλ={args.lam}  seed={seed}  @ {N_ROUNDS} 轮\n{'='*60}")
        start = time.perf_counter()
        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS, n_rounds=N_ROUNDS, seed=seed,
            marginals=marginals, lambda_param=args.lam, **PARAMS,
        )
        elapsed = time.perf_counter() - start
        run_dir = os.path.join(out_root, f"seed{seed}")
        save_run(best_S, diag, run_dir=run_dir)

        rec = {
            "lambda": args.lam, "seed": seed,
            "best_loss": float(diag["best_loss"]),
            "normalized_l1_error": float(diag["normalized_l1_error"]),
            "accept_rate": float(sum(diag["accept_history"]) / len(diag["accept_history"])),
            "elapsed_sec": float(elapsed),
        }
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        print(f"best_loss: {rec['best_loss']:.4e}  L1: {rec['normalized_l1_error']:.4f}  "
              f"接受率: {rec['accept_rate']*100:.1f}%  耗时: {elapsed/60:.1f}min")


if __name__ == "__main__":
    main()
