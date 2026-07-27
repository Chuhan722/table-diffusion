"""
第 3 步：大数据多种子严格对比 - multiplicative vs linear

目标：3 种子严格验证,统计显著性检验
数据：nltcs
轮数：1500
模式：multiplicative(p=1.0, β=1.0) vs linear(h=0.8, β=1.0)
种子：0, 1, 2
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
from scipy import stats

from table_diffevo.schema import load_schema
from table_diffevo.evolution import run_evolution
from table_diffevo.queries import evaluate_table

# 参数配置
DATASET = 'nltcs'
N_ROUNDS = 1500
SEEDS = [0, 1, 2]
BETA = 1.0
H = 0.8
P = 1.0
RHO = 0.01
ETA = 0.5
MU = 0.01
DEVICE = 'cuda'
INIT_METHOD = 'marginal'

# 两组实验配置
MODES = ['multiplicative', 'linear']
MODE_CONFIGS = {
    'multiplicative': {'distance_mode': 'multiplicative', 'p': P, 'h': H},
    'linear': {'distance_mode': 'linear', 'p': P, 'h': H},
}

# 路径
data_path = project_root / 'data' / DATASET / f'{DATASET}.csv'
queries_path = project_root / 'configs' / DATASET / 'measured_1000query.json'
schema_path = project_root / 'configs' / DATASET / 'schema.yaml'
marginals_path = project_root / 'configs' / DATASET / 'init_marginals.json'

stamp = time.strftime('%Y-%m-%d_%H%M')
output_base = project_root / 'outputs' / f'multiplicative_step3_{DATASET}_{stamp}'
output_base.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("第 3 步：大数据多种子严格对比 - multiplicative vs linear")
print("=" * 70)
print(f"数据集: {DATASET}")
print(f"轮数: {N_ROUNDS}")
print(f"种子: {SEEDS}")
print(f"参数: β={BETA}, ρ={RHO}, η={ETA}, μ={MU}")
print(f"  - multiplicative: p={P}")
print(f"  - linear: h={H}")
print(f"设备: {DEVICE}")
print(f"输出: {output_base}")
print("=" * 70)

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


# 运行所有实验
all_results = {mode: {'best_loss': [], 'normalized_l1': []} for mode in MODES}

for mode in MODES:
    print(f"\n{'='*70}")
    print(f"模式: {mode}")
    print(f"{'='*70}")

    mode_dir = output_base / mode
    mode_dir.mkdir(exist_ok=True)

    config = MODE_CONFIGS[mode]

    for seed in SEEDS:
        print(f"\n--- 种子 {seed} ---")

        t0 = time.time()
        best_S, diagnostics = run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=n_records,
            n_rounds=N_ROUNDS,
            seed=seed,
            beta=BETA,
            h=config['h'],
            rho=RHO,
            eta=ETA,
            mu=MU,
            device=DEVICE,
            init_method=INIT_METHOD,
            marginals=marginals,
            log_every=500,  # 只打印关键轮次
            distance_mode=config['distance_mode'],
            p=config['p'],
        )
        elapsed = time.time() - t0

        # 保存
        seed_dir = mode_dir / f'seed_{seed}'
        seed_dir.mkdir(exist_ok=True)
        best_S.to_csv(seed_dir / 'best_synthetic.csv', index=False)
        with open(seed_dir / 'diagnostics.json', 'w') as f:
            json.dump(serialize_diagnostics(diagnostics), f, indent=2)

        best_loss = float(diagnostics['best_loss'])
        norm_l1 = float(diagnostics.get('normalized_l1_error', -1))
        accept_rate = float(np.mean(diagnostics['accept_history']))

        all_results[mode]['best_loss'].append(best_loss)
        all_results[mode]['normalized_l1'].append(norm_l1)

        print(f"  最优 loss: {best_loss:.4e}, 归一化 L1: {norm_l1:.4f}, "
              f"接受率: {accept_rate*100:.1f}%, 耗时: {elapsed:.1f}s")


# 汇总统计
print(f"\n{'='*70}")
print("汇总统计 (3 种子)")
print(f"{'='*70}")
print(f"{'模式':<16} {'最优Loss (均值±std)':<30} {'归一化L1 (均值±std)':<25}")
print("-" * 70)

summary = {}
for mode in MODES:
    losses = np.array(all_results[mode]['best_loss'])
    l1s = np.array(all_results[mode]['normalized_l1'])

    summary[mode] = {
        'best_loss_mean': float(losses.mean()),
        'best_loss_std': float(losses.std()),
        'best_loss_values': losses.tolist(),
        'normalized_l1_mean': float(l1s.mean()),
        'normalized_l1_std': float(l1s.std()),
        'normalized_l1_values': l1s.tolist(),
    }

    print(f"{mode:<16} {losses.mean():.4e} ± {losses.std():.4e}    "
          f"{l1s.mean():.4f} ± {l1s.std():.4f}")

# 统计显著性检验
mult_losses = np.array(all_results['multiplicative']['best_loss'])
lin_losses = np.array(all_results['linear']['best_loss'])
t_stat, p_value = stats.ttest_ind(mult_losses, lin_losses)

diff_pct = (mult_losses.mean() - lin_losses.mean()) / lin_losses.mean() * 100

print("-" * 70)
print(f"multiplicative vs linear:")
print(f"  - 平均 loss 相对差: {diff_pct:+.1f}%")
print(f"  - t 检验: t={t_stat:.4f}, p={p_value:.6f}")

if p_value < 0.001:
    sig_label = "高度显著 (p<0.001)"
elif p_value < 0.01:
    sig_label = "非常显著 (p<0.01)"
elif p_value < 0.05:
    sig_label = "显著 (p<0.05)"
else:
    sig_label = "不显著 (p≥0.05)"

print(f"  - 显著性: {sig_label}")

# 判断
if p_value < 0.01 and diff_pct < -10:
    conclusion = "multiplicative 显著更好,建议改默认值"
elif p_value < 0.05 and abs(diff_pct) < 5:
    conclusion = "两者接近,保持 linear 默认"
elif p_value >= 0.05:
    conclusion = "差异不显著,保持 linear 默认"
else:
    conclusion = f"multiplicative 差异 {diff_pct:+.1f}%,需结合实际决策"

print(f"  - 建议: {conclusion}")

# 保存汇总
with open(output_base / 'summary.json', 'w') as f:
    json.dump({
        'config': {
            'dataset': DATASET,
            'n_rounds': N_ROUNDS,
            'seeds': SEEDS,
            'beta': BETA, 'rho': RHO, 'eta': ETA, 'mu': MU,
            'multiplicative_p': P,
            'linear_h': H,
        },
        'summary': summary,
        'comparison': {
            'diff_pct': diff_pct,
            't_stat': float(t_stat),
            'p_value': float(p_value),
            'significance': sig_label,
            'conclusion': conclusion,
        },
    }, f, indent=2)

print(f"\n{'='*70}")
print(f"第 3 步完成 ✅")
print(f"汇总已保存: {output_base / 'summary.json'}")
print(f"{'='*70}")
