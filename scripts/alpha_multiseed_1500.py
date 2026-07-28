"""
α1.5→6 vs α2→10 多种子精修（nltcs, 1500 轮, seed 0/1/2, λ0.5）

背景：小数据 + nltcs 单种子均显示 α2→10 精度不比 α1.5→6 差，nltcs 上还好一个
数量级（7.24e6 vs 7.31e7），且不坍缩（L1 0.0054 vs 0.0185，接受率 96.8% vs 90.3%）。
loss 曲线核实无异常骤降，10 倍优势来自后期高锐度加速收敛。
本脚本补多种子 + 配对 t 检验，确认是否该把推荐配置从 α1.5→6 改为 α2→10。

种子复用：seed=0 已跑过，本脚本只补 seed 1/2。
- α1.5→6 seed=0: 7.3124e7  (outputs/geometric_alpha1p5_6_2026-07-28_0559/geometric_a1p5_6)
- α2→10  seed=0: 7.2428e6  (outputs/geometric_alpha2_10_2026-07-28_0530/geometric_a2_10)

用法（单配置、指定补跑种子）：
    CUDA_VISIBLE_DEVICES=1 python scripts/alpha_multiseed_1500.py --config a1p5_6 --seeds 1 2
    CUDA_VISIBLE_DEVICES=1 python scripts/alpha_multiseed_1500.py --config a2_10 --seeds 1 2
"""
import os
import json
import time
import argparse
import numpy as np

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

CONFIGS = {
    "a1p5_6": {"alpha_min": 1.5, "alpha_max": 6.0},
    "a2_10": {"alpha_min": 2.0, "alpha_max": 10.0},
}

PARAMS = {
    "beta": 1.0, "h": 0.8, "rho": 0.01, "eta": 0.5, "mu": 0.01,
    "device": "cuda", "eval_method": "vectorized", "batch_size": 256,
    "init_method": "marginal", "log_every": 100,
    "distance_mode": "geometric", "lambda_param": 0.5, "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    args = ap.parse_args()
    cfg = CONFIGS[args.config]

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    out_root = f"outputs/alpha_multiseed_1500/{args.config}"
    os.makedirs(out_root, exist_ok=True)

    for seed in args.seeds:
        print(f"\n{'='*60}\n{args.config} (α{cfg['alpha_min']}→{cfg['alpha_max']})  "
              f"seed={seed}  @ {N_ROUNDS} 轮\n{'='*60}")
        start = time.perf_counter()
        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS, n_rounds=N_ROUNDS, seed=seed,
            marginals=marginals,
            alpha_min=cfg["alpha_min"], alpha_max=cfg["alpha_max"], **PARAMS,
        )
        elapsed = time.perf_counter() - start
        run_dir = os.path.join(out_root, f"seed{seed}")
        save_run(best_S, diag, run_dir=run_dir)

        rec = {
            "config": args.config, "seed": seed,
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
