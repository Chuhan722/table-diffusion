"""
调参实验：geometric 单跑（nltcs, 1500轮, 单种子）
α_min=2.0, α_max=10.0，其余参数与首次对比实验一致。

与现有结果对比：
- geometric (α 0.5→4.0): outputs/geometric_vs_multiplicative_2026-07-28_0431/geometric
- multiplicative (p=1.0):  outputs/geometric_vs_multiplicative_2026-07-28_0431/multiplicative
"""
import os
import json
import time
import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.io import create_parent_dir, save_run

SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

N_RECORDS = 16181
N_ROUNDS = 1500
SEED = 0
LOG_EVERY = 50

# 前次实验的参考结果目录
PREV_DIR = "outputs/geometric_vs_multiplicative_2026-07-28_0431"

COMMON = {
    "beta": 1.0,
    "h": 0.8,
    "rho": 0.01,
    "eta": 0.5,
    "mu": 0.01,
    "device": "cuda",
    "eval_method": "vectorized",
    "batch_size": 256,
    "init_method": "marginal",
    "log_every": LOG_EVERY,
}

GEOM_CONFIG = {
    "distance_mode": "geometric",
    "lambda_param": 0.5,
    "alpha_min": 2.0,
    "alpha_max": 10.0,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def _load_prev(name):
    path = os.path.join(PREV_DIR, name, "diagnostics.json")
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    lh = d["loss_history"]
    acc = d["accept_history"]
    return {
        "best_loss": float(d["best_loss"]),
        "normalized_l1_error": float(d["normalized_l1_error"]),
        "accept_rate": float(sum(acc) / len(acc)),
        "initial_loss": float(lh[0]),
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    parent_dir = create_parent_dir(prefix="geometric_alpha2_10")
    print(f"输出目录: {parent_dir}/\n")

    print(f"{'='*60}")
    print(f"运行：geometric  α: {GEOM_CONFIG['alpha_min']}→{GEOM_CONFIG['alpha_max']}")
    print(f"{'='*60}")

    start = time.perf_counter()
    best_S, diag = run_evolution(
        target, queries, schema,
        n_records=N_RECORDS,
        n_rounds=N_ROUNDS,
        seed=SEED,
        marginals=marginals,
        **COMMON,
        **GEOM_CONFIG,
    )
    elapsed = time.perf_counter() - start

    mode_dir = os.path.join(parent_dir, "geometric_a2_10")
    save_run(best_S, diag, run_dir=mode_dir)

    lh = diag["loss_history"]
    best_loss = diag["best_loss"]
    nl1 = diag["normalized_l1_error"]
    accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])

    cur = {
        "best_loss": float(best_loss),
        "normalized_l1_error": float(nl1),
        "accept_rate": float(accept_rate),
        "initial_loss": float(lh[0]),
        "elapsed_sec": float(elapsed),
    }

    print(f"\n===== geometric (α 2→10) 结果 =====")
    print(f"初始 loss: {lh[0]:.2e}")
    print(f"最优 loss: {best_loss:.2e}")
    print(f"下降比例: {(1 - best_loss / lh[0]) * 100:.1f}%")
    print(f"归一化 L1: {nl1:.4f}")
    print(f"接受率: {accept_rate * 100:.1f}%")
    print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")

    # 与旧结果对比
    geom_old = _load_prev("geometric")
    mult = _load_prev("multiplicative")

    print(f"\n{'='*72}")
    print("三方对比")
    print(f"{'='*72}")
    print(f"{'指标':<16} {'geom α2→10':<15} {'geom α0.5→4':<15} {'multiplicative':<15}")
    print(f"{'-'*72}")

    def row(label, key, fmt):
        v_new = cur[key]
        v_old = geom_old[key] if geom_old else float('nan')
        v_mul = mult[key] if mult else float('nan')
        print(f"{label:<16} {fmt.format(v_new):<15} {fmt.format(v_old):<15} {fmt.format(v_mul):<15}")

    row("最优 loss", "best_loss", "{:.2e}")
    row("归一化 L1", "normalized_l1_error", "{:.4f}")
    row("接受率", "accept_rate", "{:.3f}")

    comparison = {
        "experiment": "geometric_alpha2_10_vs_prev",
        "dataset": "nltcs",
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "seed": SEED,
        "common_params": COMMON,
        "geom_config": GEOM_CONFIG,
        "results": {
            "geometric_a2_10": cur,
            "geometric_a05_4_prev": geom_old,
            "multiplicative_prev": mult,
        },
    }
    comparison_path = os.path.join(parent_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n对比结果已保存: {comparison_path}")


if __name__ == "__main__":
    main()
