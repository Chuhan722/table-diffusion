"""
诊断：geometric 高锐度下的"自身抽样率"

问题：候选池含自身（K=N，全对全），自身距离=0、相似度=1。α 越大 softmax 越尖，
理论上可能把概率质量堆到自己身上 → 抽到自己 = 复制自己 = 表不变 = 整代必接受，
使接受率虚高。本脚本直接测量 donor_idx==i 的比例（历史 diagnostics 未记 donor_idx，
故需复现单轮抽样实测）。

做法：加载各配置的 best 表，在若干 α_t（模拟早/中/末轮锐度）下复现单轮抽样，
统计每条记录抽到自己的比例，并给出概率视角（自身概率 p_ii 的平均/分位）。
只测量，不改主代码。
"""
import numpy as np
import pandas as pd

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.vectorized_eval import evaluate_vectorized
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.sampling import compute_sampling_probs, sample_donors

# (标签, best表路径, schema, query, distance_mode, 该配置的 α_min→α_max)
CASES = [
    ("nltcs α2→10", "outputs/alpha_multiseed_1500/a2_10/seed1/best_synthetic.csv",
     "configs/nltcs/schema.yaml", "configs/nltcs/measured_1000query.json", "geometric", (2.0, 10.0)),
    ("nltcs α1.5→6", "outputs/alpha_multiseed_1500/a1p5_6/seed1/best_synthetic.csv",
     "configs/nltcs/schema.yaml", "configs/nltcs/measured_1000query.json", "geometric", (1.5, 6.0)),
    # 小表：300 条、0% 重复，无等价副本兜底 —— 验证高锐度是否会抽爆自己的最强场景
    ("test_300x10 α2→10", "outputs/lambda_sweep_2026-07-28_0812/test_300x10_lambda0.5/best_synthetic.csv",
     "configs/test_300x10/schema.yaml", "configs/test_300x10/measured_50query.json", "geometric", (2.0, 10.0)),
]

# 复现单轮抽样时用哪些 α_t（覆盖早/中/末轮锐度）；末轮 α_t = α_max
ALPHA_T_PROBES = ["min", "mid", "max"]
SEED = 12345  # 抽样种子（多抽几次取均值更稳）
N_DRAWS = 5


def alpha_t_value(which, amin, amax):
    return {"min": amin, "mid": (amin + amax) / 2, "max": amax}[which]


def main():
    for label, path, schema_path, query_path, mode, (amin, amax) in CASES:
        schema = load_schema(schema_path)
        queries = load_queries(query_path)
        target = np.array([q["result"] for q in queries], dtype=float)
        df = pd.read_csv(path)
        N = len(df)
        # 现算 fitness（与主循环同口径：无噪声、权重全 1）
        _, _, fitness = evaluate_vectorized(
            df, queries, schema, target=target, n_records=N,
            device="numpy", want_fitness=True, verbose=False,
        )
        distances = pairwise_block_distance(df, df, schema, device="numpy")

        print(f"\n{'='*66}\n{label}  (N={N}, mode={mode}, α {amin}→{amax})\n{'='*66}")
        # 自身距离应为 0 —— 先确认
        self_d = np.diag(distances)
        print(f"  对角线(自身)距离: mean={self_d.mean():.4f} max={self_d.max():.4f}  (应≈0)")

        for which in ALPHA_T_PROBES:
            at = alpha_t_value(which, amin, amax)
            probs = compute_sampling_probs(
                fitness, distances, beta=1.0, h=0.8, device="numpy",
                distance_mode=mode, p=1.0, lambda_param=0.5, alpha=at,
                delta=0.05, winsorize_quantiles=(0.01, 0.99),
            )
            # 概率视角：自身概率 p_ii
            p_ii = probs[np.arange(N), np.arange(N)]
            # 抽样视角：实际抽到自己的比例（多抽几次取均值）
            self_rates = []
            for k in range(N_DRAWS):
                rng = np.random.default_rng(SEED + k)
                donor_idx = sample_donors(probs, rng, device="numpy")
                self_rates.append(float(np.mean(donor_idx == np.arange(N))))
            sr = np.array(self_rates)
            print(f"  α_t={at:5.2f} ({which:3s}): "
                  f"自身概率 p_ii 均值={p_ii.mean()*100:5.2f}% 中位={np.median(p_ii)*100:5.2f}% "
                  f"P90={np.percentile(p_ii,90)*100:5.2f}% max={p_ii.max()*100:5.2f}%  |  "
                  f"实测自身抽样率={sr.mean()*100:5.2f}% (±{sr.std()*100:.2f})")


if __name__ == "__main__":
    main()
