"""
测量抽样 logit 两项的量级：适应度项 β·F vs 距离项 γ·d

目的：判断 (s, λ) 重参数化里 λ 的刻度是否均匀。
- 若 F 和 d 量级相当 → λ 刻度正常
- 若 F 量级碾压 d → λ 死区，必须先把权重归一化到和为 1

测两个状态：
1. 初始态（marginal init）
2. 收敛态（加载已跑好的 linear best_synthetic.csv）

每个状态报告：
- fitness 分布（原始权重全 1）
- distance 分布
- β·F 项 vs γ·d 项的量级（当前 β=1, h=0.8 → γ=1.25）
- 权重归一化到和为 1 后 F 的预期量级（= F / m，m=查询数）
"""
import sys
import numpy as np
import pandas as pd

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.marginals import load_marginals, init_from_marginals
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.vectorized_eval import evaluate_vectorized

# ===== 配置 =====
SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"
CONVERGED_CSV = "outputs/distance_mode_experiment_2026-07-25_1000/linear/0-0/best_synthetic.csv"
N_RECORDS = 16181
BETA = 1.0
H = 0.8
DEVICE = 'cuda'
DIST_SUBSAMPLE = 2000  # 距离矩阵子采样，避免 16181² 爆内存
SEED = 0
# ================


def describe(name, arr):
    arr = np.asarray(arr, dtype=float).ravel()
    print(f"  {name}:")
    print(f"    mean={arr.mean():.4f}  std={arr.std():.4f}  mean|·|={np.abs(arr).mean():.4f}")
    print(f"    min={arr.min():.4f}  p50={np.median(arr):.4f}  p90={np.percentile(arr,90):.4f}  max={arr.max():.4f}")


def measure_state(label, S, queries, schema, target, m):
    print(f"\n{'='*64}\n状态: {label}  (N={len(S)})\n{'='*64}")

    # fitness（原始权重全 1）
    _, _, fitness = evaluate_vectorized(
        S, queries, schema, target=target, n_records=N_RECORDS,
        device=DEVICE, want_fitness=True, verbose=False,
    )
    describe("fitness F (权重全1)", fitness)

    # 距离（子采样）
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(S), size=min(DIST_SUBSAMPLE, len(S)), replace=False)
    S_sub = S.iloc[idx].reset_index(drop=True)
    dist = pairwise_block_distance(S_sub, S_sub, schema, device=DEVICE, return_tensor=False)
    # 去掉对角线（自己到自己=0）
    off_diag = dist[~np.eye(len(S_sub), dtype=bool)]
    describe("distance d", off_diag)

    # 两项量级对比
    beta_F = BETA * np.abs(fitness).mean()
    gamma = 1.0 / H
    gamma_d = gamma * off_diag.mean()
    print(f"\n  --- logit 两项量级（取典型值 mean|·|）---")
    print(f"    适应度项 β·|F|  = {BETA} × {np.abs(fitness).mean():.4f} = {beta_F:.4f}")
    print(f"    距离项   γ·d    = {gamma:.4f} × {off_diag.mean():.4f} = {gamma_d:.4f}")
    print(f"    量级比 (β·F)/(γ·d) = {beta_F/gamma_d:.1f}x")

    # 权重归一化到和为 1 后的预期
    F_norm = np.abs(fitness).mean() / m
    print(f"\n  --- 若权重归一化到和为1（w_j=1/m, m={m}）---")
    print(f"    F 预期量级 ≈ |F|/m = {np.abs(fitness).mean():.4f} / {m} = {F_norm:.6f}")
    print(f"    归一化后量级比 (F_norm)/(d) = {F_norm/off_diag.mean():.4f}x  (目标：接近 1)")

    return np.abs(fitness).mean(), off_diag.mean()


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(MARGINALS_PATH)
    m = len(queries)
    print(f"查询数 m = {m}")

    # 状态1：初始态
    rng = np.random.default_rng(SEED)
    S_init = init_from_marginals(N_RECORDS, schema, marginals, rng=rng)
    measure_state("初始态 (marginal init)", S_init, queries, schema, target, m)

    # 状态2：收敛态
    try:
        S_conv = pd.read_csv(CONVERGED_CSV)
        measure_state("收敛态 (linear 1500轮 best)", S_conv, queries, schema, target, m)
    except FileNotFoundError:
        print(f"\n[跳过收敛态] 找不到 {CONVERGED_CSV}")

    print(f"\n{'='*64}\n结论提示：看『量级比』和『归一化后量级比』两个数\n{'='*64}")


if __name__ == "__main__":
    main()
