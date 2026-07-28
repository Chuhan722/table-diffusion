"""
调参实验：geometric 单跑（nltcs, 1500轮, 单种子）
α_min=1.5, α_max=6.0，寻找精度-多样性平衡点。

与现有结果对比：
- geometric (α 2→10):   outputs/geometric_alpha2_10_2026-07-28_0530/geometric_a2_10
- geometric (α 0.5→4):  outputs/geometric_vs_multiplicative_2026-07-28_0431/geometric
- multiplicative (p=1): outputs/geometric_vs_multiplicative_2026-07-28_0431/multiplicative
"""
import os
import json
import time
import numpy as np
import pandas as pd

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

PREV_RESULTS = {
    "geom_a2_10": "outputs/geometric_alpha2_10_2026-07-28_0530/geometric_a2_10",
    "geom_a05_4": "outputs/geometric_vs_multiplicative_2026-07-28_0431/geometric",
    "multiplicative": "outputs/geometric_vs_multiplicative_2026-07-28_0431/multiplicative",
}

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
    "alpha_min": 1.5,
    "alpha_max": 6.0,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def _analyze_diversity(csv_path):
    """分析合成数据的多样性"""
    df = pd.read_csv(csv_path)
    n_records = len(df)
    n_unique = len(df.drop_duplicates())
    dup_rate = (n_records - n_unique) / n_records * 100

    value_counts = df.value_counts()
    max_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
    max_pct = max_count / n_records * 100

    top5_sum = value_counts.iloc[:5].sum() if len(value_counts) >= 5 else max_count
    top5_pct = top5_sum / n_records * 100

    return {
        "n_unique": int(n_unique),
        "duplicate_rate_pct": float(dup_rate),
        "max_freq_count": int(max_count),
        "max_freq_pct": float(max_pct),
        "top5_freq_pct": float(top5_pct),
    }


def _load_prev(result_dir):
    """加载前次结果"""
    diag_path = os.path.join(result_dir, "diagnostics.json")
    csv_path = os.path.join(result_dir, "best_synthetic.csv")

    if not os.path.exists(diag_path) or not os.path.exists(csv_path):
        return None

    d = json.load(open(diag_path))
    lh = d["loss_history"]
    acc = d["accept_history"]
    diversity = _analyze_diversity(csv_path)

    return {
        "best_loss": float(d["best_loss"]),
        "normalized_l1_error": float(d["normalized_l1_error"]),
        "accept_rate": float(sum(acc) / len(acc)),
        "initial_loss": float(lh[0]),
        "diversity": diversity,
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)

    parent_dir = create_parent_dir(prefix="geometric_alpha1p5_6")
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

    mode_dir = os.path.join(parent_dir, "geometric_a1p5_6")
    save_run(best_S, diag, run_dir=mode_dir)

    lh = diag["loss_history"]
    best_loss = diag["best_loss"]
    nl1 = diag["normalized_l1_error"]
    accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])

    csv_path = os.path.join(mode_dir, "best_synthetic.csv")
    diversity = _analyze_diversity(csv_path)

    cur = {
        "best_loss": float(best_loss),
        "normalized_l1_error": float(nl1),
        "accept_rate": float(accept_rate),
        "initial_loss": float(lh[0]),
        "elapsed_sec": float(elapsed),
        "diversity": diversity,
    }

    print(f"\n===== geometric (α 1.5→6) 结果 =====")
    print(f"初始 loss: {lh[0]:.2e}")
    print(f"最优 loss: {best_loss:.2e}")
    print(f"下降比例: {(1 - best_loss / lh[0]) * 100:.1f}%")
    print(f"归一化 L1: {nl1:.4f}")
    print(f"接受率: {accept_rate * 100:.1f}%")
    print(f"耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"\n多样性:")
    print(f"  唯一记录: {diversity['n_unique']} ({diversity['n_unique']/N_RECORDS*100:.1f}%)")
    print(f"  重复率: {diversity['duplicate_rate_pct']:.2f}%")
    print(f"  最高频记录: {diversity['max_freq_pct']:.2f}%")
    print(f"  前5记录占比: {diversity['top5_freq_pct']:.2f}%")

    # 加载旧结果
    results = {"geometric_a1p5_6": cur}
    for name, result_dir in PREV_RESULTS.items():
        prev = _load_prev(result_dir)
        if prev:
            results[name] = prev

    print(f"\n{'='*85}")
    print("四方对比")
    print(f"{'='*85}")
    print(f"{'指标':<18} {'α1.5→6':<15} {'α2→10':<15} {'α0.5→4':<15} {'multiplicative':<15}")
    print(f"{'-'*85}")

    def row(label, key, fmt):
        vals = []
        for name in ["geometric_a1p5_6", "geom_a2_10", "geom_a05_4", "multiplicative"]:
            if name in results and key in results[name]:
                vals.append(fmt.format(results[name][key]))
            else:
                vals.append("—")
        print(f"{label:<18} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15} {vals[3]:<15}")

    row("最优 loss", "best_loss", "{:.2e}")
    row("归一化 L1", "normalized_l1_error", "{:.4f}")
    row("接受率", "accept_rate", "{:.3f}")

    print(f"\n{'多样性':<18}")

    def div_row(label, key, fmt):
        vals = []
        for name in ["geometric_a1p5_6", "geom_a2_10", "geom_a05_4", "multiplicative"]:
            if name in results and "diversity" in results[name]:
                vals.append(fmt.format(results[name]["diversity"][key]))
            else:
                vals.append("—")
        print(f"  {label:<16} {vals[0]:<15} {vals[1]:<15} {vals[2]:<15} {vals[3]:<15}")

    div_row("唯一记录", "n_unique", "{:d}")
    div_row("重复率%", "duplicate_rate_pct", "{:.2f}")
    div_row("最高频%", "max_freq_pct", "{:.2f}")
    div_row("前5占比%", "top5_freq_pct", "{:.2f}")

    comparison = {
        "experiment": "geometric_alpha1p5_6_vs_all",
        "dataset": "nltcs",
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "seed": SEED,
        "common_params": COMMON,
        "geom_config": GEOM_CONFIG,
        "results": results,
    }

    comparison_path = os.path.join(parent_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n对比结果已保存: {comparison_path}")


if __name__ == "__main__":
    main()
