"""
λ 扫描（geometric 模式，α 锁定 1.5→6，OAT 只变 λ）

目的：α 已定 1.5→6，扫剩下唯一没碰过的核心自由度 λ（适应度-距离权衡），
回答两件事：
1. λ=0.5 是否通用最优，还是需要偏相似度（λ<0.5）/偏适应度（λ>0.5）
2. 两个分布迥异的数据集（nltcs 83% 重复 vs test_300x10 0% 重复）是否偏好不同 λ

方法：第 1 步单种子粗扫地形（A 方案）。看趋势后再决定是否多种子精修。

数据集：
- test_300x10：300×50，100 轮，seed=42，全扫 5 点（几乎不花时间）
- nltcs：16181×1000，1000 轮，seed=0，5 点（用户定 1000 轮）

固定：alpha 1.5→6，rho=0.01, eta=0.5, mu=0.01, delta=0.05, winsorize=(0.01,0.99)
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

LAMBDAS = [0.3, 0.4, 0.5, 0.6, 0.7]

DATASETS = {
    "test_300x10": {
        "schema": "configs/test_300x10/schema.yaml",
        "query": "configs/test_300x10/measured_50query.json",
        "marginals": "configs/test_300x10/init_marginals.json",
        "real_data": "data/test_300x10/test_300x10.csv",
        "n_records": 300,
        "n_rounds": 100,
        "seed": 42,
        "top_k": 5,
    },
    "nltcs": {
        "schema": "configs/nltcs/schema.yaml",
        "query": "configs/nltcs/measured_1000query.json",
        "marginals": "configs/nltcs/init_marginals.json",
        "real_data": "data/nltcs/nltcs.csv",
        "n_records": 16181,
        "n_rounds": 1000,
        "seed": 0,
        "top_k": 10,
    },
}

COMMON = {
    "beta": 1.0,          # geometric 分支不使用，仅占位
    "h": 0.8,             # geometric 分支不使用，仅占位
    "rho": 0.01,
    "eta": 0.5,
    "mu": 0.01,
    "device": "cuda",
    "eval_method": "vectorized",
    "batch_size": 256,
    "init_method": "marginal",
    "log_every": 50,
    "distance_mode": "geometric",
    "alpha_min": 1.5,
    "alpha_max": 6.0,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
}


def analyze_diversity(synth_df, real_df, top_k):
    """多样性指标：唯一数、重复率、最高频、Top-k 覆盖。"""
    n_synth = len(synth_df)
    n_real = len(real_df)

    synth_unique = len(synth_df.drop_duplicates())
    real_unique = len(real_df.drop_duplicates())

    synth_vc = synth_df.value_counts()
    real_vc = real_df.value_counts()

    synth_max_pct = synth_vc.iloc[0] / n_synth * 100 if len(synth_vc) > 0 else 0.0
    real_max_pct = real_vc.iloc[0] / n_real * 100 if len(real_vc) > 0 else 0.0

    real_topk = real_vc.head(top_k)
    covered = sum(1 for idx in real_topk.index if idx in synth_vc.index)

    return {
        "synth_unique": int(synth_unique),
        "synth_dup_rate": float((n_synth - synth_unique) / n_synth * 100),
        "real_unique": int(real_unique),
        "real_dup_rate": float((n_real - real_unique) / n_real * 100),
        "synth_max_pct": float(synth_max_pct),
        "real_max_pct": float(real_max_pct),
        "real_topk_covered": int(covered),
        "top_k": int(top_k),
    }


def run_dataset(ds_name, cfg, parent_dir):
    schema = load_schema(cfg["schema"])
    queries = load_queries(cfg["query"])
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(cfg["marginals"])
    real_data = pd.read_csv(cfg["real_data"])

    print(f"\n{'#'*70}")
    print(f"# 数据集: {ds_name}  ({cfg['n_records']} 条 × {len(queries)} 查询, "
          f"{cfg['n_rounds']} 轮, seed={cfg['seed']})")
    print(f"# 真实唯一 {len(real_data.drop_duplicates())}, "
          f"重复率 {(1 - len(real_data.drop_duplicates())/len(real_data))*100:.2f}%")
    print(f"{'#'*70}")

    results = []
    for lam in LAMBDAS:
        print(f"\n{'='*60}")
        print(f"{ds_name}  λ={lam}  (α 1.5→6)")
        print(f"{'='*60}")

        start = time.perf_counter()
        best_S, diag = run_evolution(
            target, queries, schema,
            n_records=cfg["n_records"],
            n_rounds=cfg["n_rounds"],
            seed=cfg["seed"],
            marginals=marginals,
            lambda_param=lam,
            **COMMON,
        )
        elapsed = time.perf_counter() - start

        run_dir = os.path.join(parent_dir, f"{ds_name}_lambda{lam}")
        save_run(best_S, diag, run_dir=run_dir)

        lh = diag["loss_history"]
        best_loss = diag["best_loss"]
        nl1 = diag["normalized_l1_error"]
        accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])
        diversity = analyze_diversity(best_S, real_data, cfg["top_k"])

        print(f"初始 loss: {lh[0]:.2e} → 最优 loss: {best_loss:.2e} "
              f"(降 {(1 - best_loss/lh[0])*100:.1f}%)")
        print(f"归一化 L1: {nl1:.4f}  接受率: {accept_rate*100:.1f}%  "
              f"耗时: {elapsed/60:.1f}min")
        print(f"多样性: 唯一 {diversity['synth_unique']} "
              f"({diversity['synth_unique']/cfg['n_records']*100:.1f}%), "
              f"重复率 {diversity['synth_dup_rate']:.2f}%, "
              f"最高频 {diversity['synth_max_pct']:.2f}%, "
              f"Top-{cfg['top_k']} 覆盖 {diversity['real_topk_covered']}/{cfg['top_k']}")

        results.append({
            "lambda": lam,
            "best_loss": float(best_loss),
            "initial_loss": float(lh[0]),
            "reduction_pct": float((1 - best_loss / lh[0]) * 100),
            "normalized_l1_error": float(nl1),
            "accept_rate": float(accept_rate),
            "elapsed_sec": float(elapsed),
            "diversity": diversity,
        })

    # 汇总表
    print(f"\n{'='*80}")
    print(f"{ds_name} — λ 扫描汇总")
    print(f"{'='*80}")
    header = f"{'指标':<16}" + "".join(f"λ={l:<9}" for l in LAMBDAS)
    print(header)
    print('-' * 80)

    def row(label, getter, fmt):
        vals = "".join(f"{fmt.format(getter(r)):<11}" for r in results)
        print(f"{label:<16}{vals}")

    row("最优 loss", lambda r: r["best_loss"], "{:.2e}")
    row("归一化 L1", lambda r: r["normalized_l1_error"], "{:.4f}")
    row("下降%", lambda r: r["reduction_pct"], "{:.1f}")
    row("接受率", lambda r: r["accept_rate"], "{:.3f}")
    row("合成唯一", lambda r: r["diversity"]["synth_unique"], "{:d}")
    row("重复率%", lambda r: r["diversity"]["synth_dup_rate"], "{:.2f}")
    row("最高频%", lambda r: r["diversity"]["synth_max_pct"], "{:.2f}")
    row(f"Top-{cfg['top_k']}覆盖", lambda r: r["diversity"]["real_topk_covered"], "{:d}")

    real_ref = results[0]["diversity"]
    print(f"\n参考真实: 唯一 {real_ref['real_unique']}, "
          f"重复率 {real_ref['real_dup_rate']:.2f}%, "
          f"最高频 {real_ref['real_max_pct']:.2f}%")

    return {
        "dataset": ds_name,
        "n_records": cfg["n_records"],
        "n_rounds": cfg["n_rounds"],
        "seed": cfg["seed"],
        "top_k": cfg["top_k"],
        "real_unique": real_ref["real_unique"],
        "real_dup_rate": real_ref["real_dup_rate"],
        "real_max_pct": real_ref["real_max_pct"],
        "results": results,
    }


def main():
    parent_dir = create_parent_dir(prefix="lambda_sweep")
    print(f"输出目录: {parent_dir}/")
    print(f"λ 网格: {LAMBDAS}  |  α 锁定 1.5→6")

    all_results = {}
    for ds_name, cfg in DATASETS.items():
        all_results[ds_name] = run_dataset(ds_name, cfg, parent_dir)

    comparison = {
        "experiment": "lambda_sweep_alpha1p5_6",
        "lambdas": LAMBDAS,
        "common_params": {k: v for k, v in COMMON.items()
                          if k != "winsorize_quantiles"},
        "winsorize_quantiles": list(COMMON["winsorize_quantiles"]),
        "datasets": all_results,
    }
    comparison_path = os.path.join(parent_dir, "comparison.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n对比结果已保存: {comparison_path}")


if __name__ == "__main__":
    main()
