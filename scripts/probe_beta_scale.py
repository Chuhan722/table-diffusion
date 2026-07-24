"""
一次性探针：实测适应度 F 的数值量级，用来定 β 扫描网格的数量级。

打分公式 ℓ = β·F(z) − d²/(2h²)，softmax 只看两项的相对差异。
- 距离项 d²/(2h²)：d∈[0,1] 归一化，h=0.8 时范围 [0, 0.78]。
- 适应度项 β·F：F 的动态范围未知，本脚本实测。

判据：要让"距离项"在选择中还能起作用，β·F 的跨度需与 0.78 可比。
若 F 的候选间跨度是"几十"，则 β 需 ~0.01 量级才平衡；若 F 跨度 ~1，则 β~1 合理。

只测量、不训练、不落盘。跑几十轮取几个时间点看 F 分布。
    conda run -p ./.conda python scripts/probe_beta_scale.py
"""
import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.marginals import load_marginals
from table_diffevo.generator import init_synthetic_table
from table_diffevo.vectorized_eval import evaluate_vectorized
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.objective import compute_loss
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.update import evolve_step

SCHEMA_PATH = "configs/nltcs/schema.yaml"
QUERY_PATH = "configs/nltcs/measured_1000query.json"
MARGINALS_PATH = "configs/nltcs/init_marginals.json"
N_RECORDS = 16181
DEVICE = 'cuda'
BATCH_SIZE = 256
SEED = 0
# baseline 更新参数（推进表演化，与 sweep BASELINE 一致）
RHO, ETA, MU = 0.01, 0.5, 0.01
H = 0.8                 # 用于把 β·F 的跨度和距离项 d²/(2h²) 做对比
CHECKPOINTS = [0, 1, 5, 20, 50, 100, 200, 350, 500]   # 覆盖后期，看 F 量级是否随收敛塌缩


def _describe(name, arr):
    arr = np.asarray(arr, dtype=float)
    span = arr.max() - arr.min()
    print(f"  {name}: min={arr.min():.4g} max={arr.max():.4g} "
          f"mean={arr.mean():.4g} std={arr.std():.4g} "
          f"P10={np.percentile(arr,10):.4g} P90={np.percentile(arr,90):.4g} "
          f"| 跨度(max-min)={span:.4g}")
    return span


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.array([q["result"] for q in queries], dtype=float)
    marginals = load_marginals(MARGINALS_PATH)
    rng = np.random.default_rng(SEED)

    S = init_synthetic_table(N_RECORDS, schema, rng, marginals=marginals)

    dist_penalty_span = 1.0 / (2 * H**2)   # d∈[0,1] → d²/(2h²) 最大值
    print(f"距离项 d²/(2h²) 在 h={H} 时范围 [0, {dist_penalty_span:.4g}]"
          f"（候选间最大跨度 {dist_penalty_span:.4g}）\n")

    max_round = max(CHECKPOINTS)
    for t in range(max_round + 1):
        q, residual, fitness = evaluate_vectorized(
            S, queries, schema, target=target, n_records=N_RECORDS,
            batch_size=BATCH_SIZE, device=DEVICE, want_fitness=True,
            verbose=False,
        )
        if t in CHECKPOINTS:
            loss = compute_loss(target, q)
            n_active = int(np.count_nonzero(residual))
            print(f"[轮 {t}] loss={loss:.3e} | 未达标查询 {n_active}/{len(queries)}")
            f_span = _describe("F(适应度)", fitness)
            # 建议的平衡 β：让 β·F 跨度 ≈ 距离项跨度
            if f_span > 0:
                beta_bal = dist_penalty_span / f_span
                print(f"    → 使 β·F 跨度≈距离项 的 β ≈ {beta_bal:.4g} "
                      f"（β·F 与距离项此时相当）")
            print()

        if t == max_round:
            break

        # 推进一步（用 baseline 参数），让 F 反映训练中期的真实状态
        use_torch = DEVICE in ('cuda', 'cpu')
        distances = pairwise_block_distance(
            S, S, schema, device=DEVICE, return_tensor=use_torch)
        probs = compute_sampling_probs(fitness, distances, beta=1.0, h=H,
                                       device=DEVICE)
        donor_idx = sample_donors(probs, rng, device=DEVICE)
        donors = S.iloc[donor_idx].reset_index(drop=True)
        S = evolve_step(S, donors, schema, rho=RHO, eta=ETA, mu=MU, rng=rng)


if __name__ == "__main__":
    main()
