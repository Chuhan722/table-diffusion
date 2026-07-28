"""
小数据测试 geometric 模式（test_300x10, 100 轮）

确认：
- 能跑通
- loss 下降
- 接受率正常
- 无 NaN/inf
"""
import numpy as np
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals

# 小数据配置
SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"

N_RECORDS = 300
N_ROUNDS = 100
SEED = 0

# geometric 模式参数
DISTANCE_MODE = 'geometric'
LAMBDA = 0.5
ALPHA_MIN = 0.5
ALPHA_MAX = 4.0
DELTA = 0.05
WINSORIZE = (0.01, 0.99)

# 共享参数
BETA = 1.0
H = 0.8
RHO = 0.01
ETA = 0.5
MU = 0.01

schema = load_schema(SCHEMA_PATH)
queries = load_queries(QUERY_PATH)
target = np.array([q["result"] for q in queries])
marginals = load_marginals(MARGINALS_PATH)

print("===== 小数据测试：geometric 模式 =====")
print(f"数据：{N_RECORDS} 条 × {len(queries)} 查询")
print(f"参数：λ={LAMBDA}, α={ALPHA_MIN}→{ALPHA_MAX}, δ={DELTA}")
print()

best_S, diag = run_evolution(
    target, queries, schema,
    n_records=N_RECORDS,
    n_rounds=N_ROUNDS,
    seed=SEED,
    beta=BETA, h=H, rho=RHO, eta=ETA, mu=MU,
    device='numpy',
    eval_method='vectorized',
    init_method='marginal',
    marginals=marginals,
    log_every=0,
    distance_mode=DISTANCE_MODE,
    lambda_param=LAMBDA,
    alpha_min=ALPHA_MIN,
    alpha_max=ALPHA_MAX,
    delta=DELTA,
    winsorize_quantiles=WINSORIZE,
)

lh = diag["loss_history"]
print(f"\n===== 结果 =====")
print(f"初始 loss: {lh[0]:.2e}")
print(f"最优 loss: {diag['best_loss']:.2e}")
print(f"下降比例: {(1 - diag['best_loss'] / lh[0]) * 100:.1f}%")
print(f"接受率: {sum(diag['accept_history']) / len(diag['accept_history']) * 100:.1f}%")
print(f"归一化 L1: {diag['normalized_l1_error']:.4f}")
print(f"α 范围: {diag['alpha_history'][0]:.2f} → {diag['alpha_history'][-1]:.2f}")

# 检查无 NaN/inf
assert not np.any(np.isnan(lh)), "loss_history 含 NaN"
assert not np.any(np.isinf(lh)), "loss_history 含 inf"
assert diag['best_loss'] < lh[0], "loss 未下降"
print("\n✅ 测试通过：无 NaN/inf，loss 下降")
