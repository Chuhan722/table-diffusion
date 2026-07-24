"""
对照实验：random vs marginal 初始化

在 test_300x10 上对比两种初始化的起点 loss 和演化效果。
"""
import numpy as np
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries, evaluate_table
from table_diffevo.objective import compute_loss
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals

schema = load_schema("configs/test_300x10/schema.yaml")
queries = load_queries("configs/test_300x10/measured_50query.json")
target = np.array([q["result"] for q in queries])
marginals = load_marginals("configs/test_300x10/init_marginals.json")

print("=" * 60)
print("对照实验：test_300x10（N=300, 50 查询）")
print("=" * 60)

# 实验 1：random 初始化
print("\n[1] 纯随机初始化（baseline）")
best_S_rand, diag_rand = run_evolution(
    target, queries, schema, n_records=300, n_rounds=10, seed=42,
    beta=1.0, h=0.8, rho=0.1, eta=0.5, mu=0.01,
    device='numpy', eval_method='vectorized', batch_size=256,
    init_method='random',
)
print(f"  初始 loss: {diag_rand['loss_history'][0]:.1f}")
print(f"  最优 loss: {diag_rand['best_loss']:.1f}")
print(f"  降低比例: {(1 - diag_rand['best_loss']/diag_rand['loss_history'][0])*100:.1f}%")

# 实验 2：marginal 初始化
print("\n[2] 按 1-way 边缘初始化")
best_S_marg, diag_marg = run_evolution(
    target, queries, schema, n_records=300, n_rounds=10, seed=42,
    beta=1.0, h=0.8, rho=0.1, eta=0.5, mu=0.01,
    device='numpy', eval_method='vectorized', batch_size=256,
    init_method='marginal', marginals=marginals,
)
print(f"  初始 loss: {diag_marg['loss_history'][0]:.1f}")
print(f"  最优 loss: {diag_marg['best_loss']:.1f}")
print(f"  降低比例: {(1 - diag_marg['best_loss']/diag_marg['loss_history'][0])*100:.1f}%")

# 对比
print("\n" + "=" * 60)
print("对比结果")
print("=" * 60)
ratio = diag_rand['loss_history'][0] / diag_marg['loss_history'][0]
print(f"初始 loss 降低: {ratio:.2f}× ({diag_rand['loss_history'][0]:.0f} → {diag_marg['loss_history'][0]:.0f})")
print(f"最优 loss 对比: random={diag_rand['best_loss']:.0f}, marginal={diag_marg['best_loss']:.0f}")
print(f"marginal 初始化使起点 loss 降低 {(1-1/ratio)*100:.1f}%")
