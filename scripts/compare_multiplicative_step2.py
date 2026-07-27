"""
第 2 步：大数据单种子对比 - multiplicative vs linear

目标：用一个种子快速判断 multiplicative 和 linear 谁好
数据：nltcs
轮数：1500
模式：multiplicative(p=1.0, β=1.0) vs linear(h=0.8, β=1.0)
种子：单种子 0
设备：cuda (gpu1)
"""
import sys
import json
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from table_diffevo.schema import load_schema
from table_diffevo.evolution import run_evolution
from table_diffevo.queries import evaluate_table

# 参数配置
DATASET = 'nltcs'
N_ROUNDS = 1500
SEED = 0
BETA = 1.0
H = 0.8
P = 1.0
RHO = 0.01
ETA = 0.5
MU = 0.01
DEVICE = 'cuda'
INIT_METHOD = 'marginal'

# 两组实验配置
EXPERIMENTS = [
    {'name': 'multiplicative', 'distance_mode': 'multiplicative', 'p': P, 'h': H},
    {'name': 'linear', 'distance_mode': 'linear', 'p': P, 'h': H},
]

# 路径
data_path = project_root / 'data' / DATASET / f'{DATASET}.csv'
queries_path = project_root / 'configs' / DATASET / 'measured_1000query.json'
schema_path = project_root / 'configs' / DATASET / 'schema.yaml'
marginals_path = project_root / 'configs' / DATASET / 'init_marginals.json'

stamp = time.strftime('%Y-%m-%d_%H%M')
output_base = project_root / 'outputs' / f'multiplicative_step2_{DATASET}_{stamp}'
output_base.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("第 2 步：大数据单种子对比 - multiplicative vs linear")
print("=" * 60)
print(f"数据集: {DATASET}")
print(f"轮数: {N_ROUNDS}")
print(f"种子: {SEED}")
print(f"参数: β={BETA}, ρ={RHO}, η={ETA}, μ={MU}")
print(f"  - multiplicative: p={P}")
print(f"  - linear: h={H}")
print(f"设备: {DEVICE}")
print(f"输出: {output_base}")
print("=" * 60)

# 加载数据
print("\n加载数据...")
df_real = pd.read_csv(data_path)
n_records = len(df_real)

with open(queries_path) as f:
    queries_data = json.load(f)
    queries = queries_data['queries']

schema = load_schema(schema_path)

with open(marginals_path) as f:
    marginals = json.load(f)

target = evaluate_table(df_real, queries)
print(f"  - 记录数: {n_records}, 查询数: {len(queries)}, 属性数: {len(schema.attributes)}")
print(f"  - 目标计数范围: [{target.min()}, {target.max()}]")


def serialize_diagnostics(diagnostics):
    """把 diagnostics 转成可 JSON 序列化的格式"""
    out = {}
    for k, v in diagnostics.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, dict):
            out[k] = {
                kk: (vv.item() if isinstance(vv, (np.integer, np.floating)) else vv)
                for kk, vv in v.items()
            }
        else:
            out[k] = v
    return out


# 跑两组实验
results = {}
for exp in EXPERIMENTS:
    name = exp['name']
    print(f"\n{'='*60}")
    print(f"运行: {name} (distance_mode={exp['distance_mode']}, p={exp['p']}, h={exp['h']})")
    print(f"{'='*60}")

    t0 = time.time()
    best_S, diagnostics = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=n_records,
        n_rounds=N_ROUNDS,
        seed=SEED,
        beta=BETA,
        h=exp['h'],
        rho=RHO,
        eta=ETA,
        mu=MU,
        device=DEVICE,
        init_method=INIT_METHOD,
        marginals=marginals,
        log_every=100,
        distance_mode=exp['distance_mode'],
        p=exp['p'],
    )
    elapsed = time.time() - t0

    # 保存
    exp_dir = output_base / name
    exp_dir.mkdir(exist_ok=True)
    best_S.to_csv(exp_dir / 'best_synthetic.csv', index=False)
    with open(exp_dir / 'diagnostics.json', 'w') as f:
        json.dump(serialize_diagnostics(diagnostics), f, indent=2)

    results[name] = {
        'best_loss': float(diagnostics['best_loss']),
        'normalized_l1': float(diagnostics.get('normalized_l1_error', -1)),
        'init_loss': float(diagnostics['loss_history'][0]),
        'final_loss': float(diagnostics['loss_history'][-1]),
        'accept_rate': float(np.mean(diagnostics['accept_history'])),
        'elapsed_sec': elapsed,
    }

    print(f"\n  - 初始 loss: {results[name]['init_loss']:.4e}")
    print(f"  - 最优 loss: {results[name]['best_loss']:.4e}")
    print(f"  - 归一化 L1: {results[name]['normalized_l1']:.4f}")
    print(f"  - 接受率: {results[name]['accept_rate']*100:.1f}%")
    print(f"  - 耗时: {elapsed:.1f} 秒")


# 对比汇总
print(f"\n{'='*60}")
print("对比结果")
print(f"{'='*60}")
print(f"{'模式':<16} {'最优Loss':<14} {'归一化L1':<12} {'接受率':<10} {'耗时(s)':<10}")
print("-" * 60)
for name in ['multiplicative', 'linear']:
    r = results[name]
    print(f"{name:<16} {r['best_loss']:<14.4e} {r['normalized_l1']:<12.4f} "
          f"{r['accept_rate']*100:<10.1f} {r['elapsed_sec']:<10.1f}")

# 相对差异
mult_loss = results['multiplicative']['best_loss']
lin_loss = results['linear']['best_loss']
diff_pct = (mult_loss - lin_loss) / lin_loss * 100
print("-" * 60)
print(f"multiplicative vs linear: loss 相对差 {diff_pct:+.1f}%")
if diff_pct < -5:
    print("  → multiplicative 更好 (loss 更低)")
elif diff_pct > 5:
    print("  → linear 更好 (multiplicative loss 更高)")
else:
    print("  → 两者接近 (差异 < 5%)")

# 保存对比
with open(output_base / 'comparison.json', 'w') as f:
    json.dump({
        'config': {
            'dataset': DATASET, 'n_rounds': N_ROUNDS, 'seed': SEED,
            'beta': BETA, 'rho': RHO, 'eta': ETA, 'mu': MU,
            'multiplicative_p': P, 'linear_h': H,
        },
        'results': results,
        'diff_pct': diff_pct,
    }, f, indent=2)

print(f"\n{'='*60}")
print(f"第 2 步完成 ✅  对比已保存: {output_base / 'comparison.json'}")
print(f"{'='*60}")
