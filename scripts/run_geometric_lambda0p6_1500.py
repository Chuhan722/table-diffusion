"""
补齐公平对比：geometric α1.5→6 λ=0.6 @ 1500 轮（nltcs, seed=0）

背景：文档 3.2 节的 linear/multiplicative baseline 都是 1500 轮，而 λ 扫描得到的
λ=0.6 是 1000 轮。轮数不对齐无法公平对比。本脚本用 1500 轮 seed=0 跑 λ=0.6，
补齐与 baseline 同轮数的数字，用于修正文档结论。

同轮数（1500）baseline（seed=0 单种子口径）：
- multiplicative p=1:     best_loss 7.166e7  (outputs/geometric_vs_multiplicative_2026-07-28_0431/multiplicative)
- geometric α1.5→6 λ0.5:  best_loss 7.31e7   (PROJECT_STATUS 记录)
参考（1500 轮，3 种子均值口径）：
- linear:  best_loss 1.034e8
- squared: best_loss 1.30e8
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
REAL_DATA_PATH = "data/nltcs/nltcs.csv"

N_RECORDS = 16181
N_ROUNDS = 1500
SEED = 0

PARAMS = {
    "beta": 1.0,          # geometric 分支不使用，占位
    "h": 0.8,             # geometric 分支不使用，占位
    "rho": 0.01,
    "eta": 0.5,
    "mu": 0.01,
    "device": "cuda",
    "eval_method": "vectorized",
    "batch_size": 256,
    "init_method": "marginal",
    "log_every": 50,
    "distance_mode": "geometric",
    "lambda_param": 0.6,
    "alpha_min": 1.5,
    "alpha_max": 6.0,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def analyze_diversity(synth_df, real_df, top_k=10):
    n_synth, n_real = len(synth_df), len(real_df)
    synth_unique = len(synth_df.drop_duplicates())
    synth_vc = synth_df.value_counts()
    real_vc = real_df.value_counts()
    synth_max_pct = synth_vc.iloc[0] / n_synth * 100 if len(synth_vc) > 0 else 0.0
    real_topk = real_vc.head(top_k)
    covered = sum(1 for idx in real_topk.index if idx in synth_vc.index)
    return {
        "synth_unique": int(synth_unique),
        "synth_dup_rate": float((n_synth - synth_unique) / n_synth * 100),
        "synth_max_pct": float(synth_max_pct),
        "real_topk_covered": int(covered),
        "top_k": int(top_k),
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)
    real_data = pd.read_csv(REAL_DATA_PATH)

    parent_dir = create_parent_dir(prefix="geometric_lambda0p6_1500")
    print(f"输出目录: {parent_dir}/")
    print(f"{'='*60}")
    print(f"geometric α1.5→6 λ=0.6 @ {N_ROUNDS} 轮 seed={SEED}")
    print(f"{'='*60}")

    start = time.perf_counter()
    best_S, diag = run_evolution(
        target, queries, schema,
        n_records=N_RECORDS,
        n_rounds=N_ROUNDS,
        seed=SEED,
        marginals=marginals,
        **PARAMS,
    )
    elapsed = time.perf_counter() - start

    run_dir = os.path.join(parent_dir, "geometric_a1p5_6_lambda0p6")
    save_run(best_S, diag, run_dir=run_dir)

    lh = diag["loss_history"]
    best_loss = diag["best_loss"]
    nl1 = diag["normalized_l1_error"]
    accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])
    diversity = analyze_diversity(best_S, real_data)

    print(f"\n初始 loss: {lh[0]:.3e} → 最优 loss: {best_loss:.3e} "
          f"(降 {(1 - best_loss/lh[0])*100:.1f}%)")
    print(f"归一化 L1: {nl1:.4f}  接受率: {accept_rate*100:.1f}%  "
          f"耗时: {elapsed/60:.1f}min")
    print(f"多样性: 唯一 {diversity['synth_unique']} "
          f"({diversity['synth_unique']/N_RECORDS*100:.1f}%), "
          f"重复率 {diversity['synth_dup_rate']:.2f}%, "
          f"最高频 {diversity['synth_max_pct']:.2f}%, "
          f"Top-10 覆盖 {diversity['real_topk_covered']}/10")

    # 同轮数对比表（1500 轮）
    print(f"\n{'='*72}")
    print("1500 轮同轮数对比（nltcs, seed=0）")
    print(f"{'='*72}")
    print(f"{'方法':<28} {'轮数':<6} {'最优 loss':<12} {'归一化 L1':<10} {'口径'}")
    print('-' * 72)
    print(f"{'linear':<28} {'1500':<6} {'1.034e+08':<12} {'0.0230':<10} 3种子均值")
    print(f"{'squared':<28} {'1500':<6} {'1.30e+08':<12} {'0.0262':<10} 3种子均值")
    print(f"{'multiplicative p=1':<28} {'1500':<6} {'7.17e+07':<12} {'0.0188':<10} 单种子")
    print(f"{'geometric α1.5→6 λ0.5':<28} {'1500':<6} {'7.31e+07':<12} {'0.0185':<10} 单种子")
    print(f"{'geometric α1.5→6 λ0.6':<28} {'1500':<6} {best_loss:<12.3e} {nl1:<10.4f} 单种子 ← 本次")

    result = {
        "experiment": "geometric_lambda0p6_1500rounds",
        "dataset": "nltcs",
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "seed": SEED,
        "params": {k: v for k, v in PARAMS.items()},
        "best_loss": float(best_loss),
        "initial_loss": float(lh[0]),
        "reduction_pct": float((1 - best_loss / lh[0]) * 100),
        "normalized_l1_error": float(nl1),
        "accept_rate": float(accept_rate),
        "elapsed_sec": float(elapsed),
        "diversity": diversity,
        "baselines_1500rounds": {
            "linear": {"best_loss": 1.034e8, "nl1": 0.0230, "note": "3种子均值"},
            "squared": {"best_loss": 1.30e8, "nl1": 0.0262, "note": "3种子均值"},
            "multiplicative_p1": {"best_loss": 7.17e7, "nl1": 0.0188, "note": "单种子seed0"},
            "geometric_lambda0p5": {"best_loss": 7.31e7, "nl1": 0.0185, "note": "单种子seed0"},
        },
    }
    out_path = os.path.join(parent_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
