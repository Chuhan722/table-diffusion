"""
一次性探针：验证 F 的量级是否随查询数按 √m 缩放。

理论预测（随机游走论证）：每个查询对 F 的贡献均值为 0、有正有负半抵消，
故 F 的 std ∝ √(查询数 m)。本脚本在同一张初始表上，从全部查询里随机抽
不同大小的子集，量 F 的 std / 跨度，看是否按 √m 走。

只测量、不训练、不落盘。
    conda run -p ./.conda python scripts/probe_f_scaling.py
"""
import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.marginals import load_marginals
from table_diffevo.generator import init_synthetic_table
from table_diffevo.vectorized_eval import evaluate_vectorized

SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"
N_RECORDS = 16181
DEVICE = 'cuda'
BATCH_SIZE = 256
SEED = 0
SUBSET_SIZES = [50, 100, 250, 500, 1001]   # 抽多少个查询
N_REPEAT = 5                                 # 每个大小抽几次不同子集取平均


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target_all = np.array([q["result"] for q in queries], dtype=float)
    marginals = load_marginals(MARGINALS_PATH)
    rng = np.random.default_rng(SEED)

    # 固定一张初始表，所有子集都在它上面测（控制变量）
    S = init_synthetic_table(N_RECORDS, schema, rng, marginals=marginals)
    m_total = len(queries)

    print(f"预测：F 的 std ∝ √m。下面看实测 std / √m 是否近似常数。\n")
    print(f"{'m':>6} | {'F跨度均值':>10} | {'F std均值':>10} | "
          f"{'std/√m':>8} | {'跨度/√m':>8}")
    print("-" * 56)

    base_ratio = None
    for m in SUBSET_SIZES:
        spans, stds = [], []
        for r in range(N_REPEAT):
            if m >= m_total:
                idx = np.arange(m_total)   # 全集，无需重复抽
            else:
                idx = rng.choice(m_total, size=m, replace=False)
            sub_q = [queries[i] for i in idx]
            sub_target = target_all[idx]
            _, _, fitness = evaluate_vectorized(
                S, sub_q, schema, target=sub_target, n_records=N_RECORDS,
                batch_size=BATCH_SIZE, device=DEVICE, want_fitness=True,
                verbose=False,
            )
            spans.append(float(fitness.max() - fitness.min()))
            stds.append(float(fitness.std()))
            if m >= m_total:
                break   # 全集只有一种，不重复
        span_mean = np.mean(spans)
        std_mean = np.mean(stds)
        sqrt_m = np.sqrt(m)
        std_ratio = std_mean / sqrt_m
        span_ratio = span_mean / sqrt_m
        if base_ratio is None:
            base_ratio = std_ratio
        print(f"{m:>6} | {span_mean:>10.3f} | {std_mean:>10.3f} | "
              f"{std_ratio:>8.4f} | {span_ratio:>8.4f}")

    print(f"\n若 'std/√m' 一列近似常数，则 F ∝ √m 成立（每个查询贡献半抵消累加）。")


if __name__ == "__main__":
    main()
