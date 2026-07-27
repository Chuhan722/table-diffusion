"""
第 1 步：小数据查错 - multiplicative 模式

目标：确认 multiplicative 模式能跑、不崩、行为正常
数据：test_300x10
轮数：100
模式：multiplicative
参数：p=1.0, β=1.0
设备：cuda (gpu1)
"""
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from table_diffevo.schema import load_schema
from table_diffevo.evolution import run_evolution
from table_diffevo.io import create_parent_dir

# 参数配置
DATASET = 'test_300x10'
N_ROUNDS = 100
SEED = 0
DISTANCE_MODE = 'multiplicative'
P = 1.0
BETA = 1.0
H = 0.8  # multiplicative 不用,但传进去
RHO = 0.01
ETA = 0.5
MU = 0.01
DEVICE = 'cuda'
INIT_METHOD = 'marginal'

# 路径
data_path = project_root / 'data' / DATASET / f'{DATASET}.csv'
queries_path = project_root / 'configs' / DATASET / 'measured_50query.json'
schema_path = project_root / 'configs' / DATASET / 'schema.yaml'
marginals_path = project_root / 'configs' / DATASET / 'init_marginals.json'
output_dir = project_root / 'outputs' / f'multiplicative_step1_{DATASET}'

print("=" * 60)
print("第 1 步：小数据查错 - multiplicative 模式")
print("=" * 60)
print(f"数据集: {DATASET}")
print(f"轮数: {N_ROUNDS}")
print(f"模式: {DISTANCE_MODE}")
print(f"参数: p={P}, β={BETA}, ρ={RHO}, η={ETA}, μ={MU}")
print(f"设备: {DEVICE}")
print(f"初始化: {INIT_METHOD}")
print(f"输出: {output_dir}")
print("=" * 60)

# 加载数据
print("\n[1/5] 加载数据...")
df_real = pd.read_csv(data_path)
n_records = len(df_real)
print(f"  - 记录数: {n_records}")

with open(queries_path) as f:
    queries_data = json.load(f)
    queries = queries_data['queries']  # 查询在 'queries' 键下
print(f"  - 查询数: {len(queries)}")

schema = load_schema(schema_path)
print(f"  - 属性数: {len(schema.attributes)}")

# 加载边缘分布(marginal init)
with open(marginals_path) as f:
    marginals = json.load(f)

# 目标计数(真实数据上的查询结果)
from table_diffevo.queries import evaluate_table
target = evaluate_table(df_real, queries)
print(f"  - 目标计数范围: [{target.min()}, {target.max()}]")

# 运行演化
print("\n[2/5] 运行演化...")
print(f"  - 开始时间: {pd.Timestamp.now()}")

best_S, diagnostics = run_evolution(
    target=target,
    queries=queries,
    schema=schema,
    n_records=n_records,
    n_rounds=N_ROUNDS,
    seed=SEED,
    beta=BETA,
    h=H,
    rho=RHO,
    eta=ETA,
    mu=MU,
    device=DEVICE,
    init_method=INIT_METHOD,
    marginals=marginals,
    log_every=10,  # 每 10 轮打印一次
    distance_mode=DISTANCE_MODE,
    p=P,
)

print(f"  - 结束时间: {pd.Timestamp.now()}")
if 'wall_time_seconds' in diagnostics:
    print(f"  - 耗时: {diagnostics['wall_time_seconds']:.2f} 秒")

# 检查结果
print("\n[3/5] 检查结果...")
print(f"  - 初始 loss: {diagnostics['loss_history'][0]:.6e}")
print(f"  - 最优 loss: {diagnostics['best_loss']:.6e}")
print(f"  - 最终 loss: {diagnostics['loss_history'][-1]:.6e}")
print(f"  - loss 下降: {(1 - diagnostics['best_loss']/diagnostics['loss_history'][0])*100:.1f}%")
print(f"  - 运行轮数: {diagnostics['rounds_run']}")
print(f"  - 接受率: {np.mean(diagnostics['accept_history'])*100:.1f}%")
print(f"  - 提前停止: {diagnostics['stopped_early']}")

# 检查是否有异常
loss_history = diagnostics['loss_history']
if np.any(np.isnan(loss_history)):
    print("  ⚠️ 警告: loss_history 中有 NaN!")
elif np.any(np.isinf(loss_history)):
    print("  ⚠️ 警告: loss_history 中有 inf!")
elif diagnostics['best_loss'] >= diagnostics['loss_history'][0]:
    print("  ⚠️ 警告: loss 没有下降!")
elif np.mean(diagnostics['accept_history']) == 0:
    print("  ⚠️ 警告: 接受率为 0,全部被拒!")
elif np.mean(diagnostics['accept_history']) == 1:
    print("  ⚠️ 警告: 接受率为 100%,全部接受!")
else:
    print("  ✅ 正常: 无 NaN/inf, loss 下降, 接受率合理")

# 检查 diagnostics 中的参数
print("\n[4/5] 检查参数记录...")
params = diagnostics['params']
assert params['distance_mode'] == DISTANCE_MODE, f"distance_mode 不对: {params['distance_mode']}"
assert params['p'] == P, f"p 不对: {params['p']}"
assert params['beta'] == BETA, f"beta 不对: {params['beta']}"
print(f"  ✅ distance_mode: {params['distance_mode']}")
print(f"  ✅ p: {params['p']}")
print(f"  ✅ β: {params['beta']}")

# 保存结果
print("\n[5/5] 保存结果...")
create_parent_dir(output_dir)
output_dir.mkdir(exist_ok=True)

best_S.to_csv(output_dir / 'best_synthetic.csv', index=False)
print(f"  ✅ 保存最优表: best_synthetic.csv")

with open(output_dir / 'diagnostics.json', 'w') as f:
    # 转换 numpy 类型为 Python 类型
    diagnostics_serializable = {}
    for k, v in diagnostics.items():
        if isinstance(v, np.ndarray):
            diagnostics_serializable[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            diagnostics_serializable[k] = v.item()
        elif isinstance(v, dict):
            diagnostics_serializable[k] = {
                kk: vv.item() if isinstance(vv, (np.integer, np.floating)) else vv
                for kk, vv in v.items()
            }
        else:
            diagnostics_serializable[k] = v
    json.dump(diagnostics_serializable, f, indent=2)
print(f"  ✅ 保存诊断: diagnostics.json")

print("\n" + "=" * 60)
print("第 1 步完成 ✅")
print(f"输出目录: {output_dir}")
print("=" * 60)
