"""
对比实验：geometric vs multiplicative（nltcs, 1500轮, 单种子）

固定参数：
- geometric: λ=0.5, α=0.5→4.0, δ=0.05, winsorize=(0.01,0.99)
- multiplicative: p=1.0
- 共享: β=1.0, ρ=0.01, η=0.5, μ=0.01, init=marginal, device=cuda
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

# 实验配置
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"

N_RECORDS = 16181
N_ROUNDS = 1500
SEED = 0
LOG_EVERY = 50

# 共享参数
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

# 两组配置
CONFIGS = {
    "geometric": {
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": 0.5,
        "alpha_max": 4.0,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
    },
    "multiplicative": {
        "distance_mode": "multiplicative",
        "p": 1.0,
    },
}

def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    # 创建父目录
    parent_dir = create_parent_dir(prefix="geometric_vs_multiplicative")
    print(f"输出目录: {parent_dir}/\n")

    results = {}

    for mode_name, mode_config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"运行：{mode_name}")
        print(f"{'='*60}")

        start = time.perf_counter()
        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=N_RECORDS,
            n_rounds=N_ROUNDS,
            seed=SEED,
            marginals=marginals,
            **COMMON,
            **mode_config,
        )
        elapsed = time.perf_counter() - start

        # 保存结果
        mode_dir = os.path.join(parent_dir, mode_name)
        save_run(best_S, diag, run_dir=mode_dir)

        lh = diag["loss_history"]
        best_loss = diag["best_loss"]
        nl1 = diag["normalized_l1_error"]
        accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])

        print(f"\n===== {mode_name} 结果 =====")
        print(f"初始 loss: {lh[0]:.2e}")
        print(f"最优 loss: {best_loss:.2e}")
        print(f"下降比例: {(1 - best_loss / lh[0]) * 100:.1f}%")
        print(f"归一化 L1: {nl1:.4f}")
        print(f"接受率: {accept_rate * 100:.1f}%")
        print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
        print(f"已存: {mode_dir}/")

        results[mode_name] = {
            "best_loss": float(best_loss),
            "normalized_l1_error": float(nl1),
            "accept_rate": float(accept_rate),
            "initial_loss": float(lh[0]),
            "elapsed_sec": float(elapsed),
            "loss_history": [float(x) for x in lh],
        }

    # 对比汇总
    print(f"\n{'='*60}")
    print("对比汇总")
    print(f"{'='*60}")

    geom = results["geometric"]
    mult = results["multiplicative"]

    loss_diff = (geom["best_loss"] - mult["best_loss"]) / mult["best_loss"] * 100
    nl1_diff = (geom["normalized_l1_error"] - mult["normalized_l1_error"]) / mult["normalized_l1_error"] * 100

    print(f"\n{'指标':<20} {'geometric':<15} {'multiplicative':<15} {'相对差异'}")
    print(f"{'-'*65}")
    print(f"{'最优 loss':<20} {geom['best_loss']:<15.2e} {mult['best_loss']:<15.2e} {loss_diff:+.1f}%")
    print(f"{'归一化 L1':<20} {geom['normalized_l1_error']:<15.4f} {mult['normalized_l1_error']:<15.4f} {nl1_diff:+.1f}%")
    print(f"{'接受率':<20} {geom['accept_rate']*100:<15.1f} {mult['accept_rate']*100:<15.1f}")
    print(f"{'耗时(秒)':<20} {geom['elapsed_sec']:<15.1f} {mult['elapsed_sec']:<15.1f}")

    # 保存对比
    comparison = {
        "experiment": "geometric_vs_multiplicative",
        "dataset": "nltcs",
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "seed": SEED,
        "common_params": COMMON,
        "configs": CONFIGS,
        "results": results,
        "summary": {
            "best_loss_diff_pct": float(loss_diff),
            "normalized_l1_diff_pct": float(nl1_diff),
        }
    }

    comparison_path = os.path.join(parent_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n对比结果已保存: {comparison_path}")

    # 判断
    if loss_diff < -5:
        print(f"\n✅ geometric 明显更好（loss 低 {abs(loss_diff):.1f}%）")
    elif loss_diff > 5:
        print(f"\n❌ geometric 更差（loss 高 {loss_diff:.1f}%）")
    else:
        print(f"\n➖ 两者接近（loss 差异 {loss_diff:+.1f}%）")


if __name__ == "__main__":
    main()
