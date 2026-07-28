"""
分析假设：α 的最优值是否与数据集的内在重复率相关？

假设：
- 高重复数据集（nltcs 83%）→ 高 α 更优
- 低重复数据集（test_300x10 0%）→ 低 α 更优

实验设计：
1. 计算两个数据集的"重复率相关指标"
2. 看 α 对 L1 的影响曲线是否与重复率相关

如果假设成立，说明 α 需要根据数据集特征调整，不存在"通用最优 α"。
如果假设不成立，α1.5→6 可能真的是鲁棒最优。
"""

import json
import matplotlib.pyplot as plt
import numpy as np

# 读取实验结果
results = {
    'nltcs': {
        'real_dup_rate': 83.49,
        'alpha_configs': {
            '0.5→4': {'L1': 0.0359, 'unique': 5877, 'dup_rate': 63.68},
            '1.5→6': {'L1': 0.0185, 'unique': 5042, 'dup_rate': 68.84},
            '2→10': {'L1': 0.0054, 'unique': 3251, 'dup_rate': 79.91},
        }
    },
    'test_300x10': {
        'real_dup_rate': 0.00,
        'alpha_configs': {
            '0.5→4': {'L1': 0.0146, 'unique': 300, 'dup_rate': 0.00},
            '1.5→6': {'L1': 0.0136, 'unique': 297, 'dup_rate': 1.00},
            '2→10': {'L1': 0.0147, 'unique': 294, 'dup_rate': 2.00},
        }
    }
}

# 用数值代表 α 范围（取几何平均作为代表）
alpha_map = {
    '0.5→4': np.sqrt(0.5 * 4),  # ≈1.41
    '1.5→6': np.sqrt(1.5 * 6),  # ≈3.0
    '2→10': np.sqrt(2 * 10),    # ≈4.47
}

print("=" * 70)
print("假设检验：α 最优值是否与数据集重复率相关？")
print("=" * 70)

for dataset, data in results.items():
    print(f"\n【{dataset}】真实重复率 = {data['real_dup_rate']:.2f}%")
    print(f"{'α 范围':<12} {'代表值':<8} {'L1':<10} {'合成重复率%':<12} {'判断'}")
    print("-" * 70)

    l1_values = []
    alpha_values = []

    for alpha_range, metrics in data['alpha_configs'].items():
        alpha_val = alpha_map[alpha_range]
        l1 = metrics['L1']
        dup = metrics['dup_rate']

        alpha_values.append(alpha_val)
        l1_values.append(l1)

        # 找最优
        is_best = l1 == min([m['L1'] for m in data['alpha_configs'].values()])
        mark = " ← 最优" if is_best else ""

        print(f"{alpha_range:<12} {alpha_val:<8.2f} {l1:<10.4f} {dup:<12.2f} {mark}")

    # 计算 L1 相对于 α 的趋势
    if len(alpha_values) == 3:
        # 线性拟合
        coeffs = np.polyfit(alpha_values, l1_values, 1)
        slope = coeffs[0]

        print(f"\nL1 vs α 斜率: {slope:.6f}")
        if slope < -0.001:
            print("  → L1 随 α 增大而**单调下降**（高 α 有利）")
        elif slope > 0.001:
            print("  → L1 随 α 增大而**单调上升**（低 α 有利）")
        else:
            print("  → L1 与 α 无明显单调关系")

# 关键对比
print("\n" + "=" * 70)
print("核心发现")
print("=" * 70)

nltcs_best_alpha = '2→10'
nltcs_best_l1 = results['nltcs']['alpha_configs'][nltcs_best_alpha]['L1']

test_best_alpha = '1.5→6'
test_best_l1 = results['test_300x10']['alpha_configs'][test_best_alpha]['L1']

print(f"\nnltcs (高重复 83%)  最优 α = {nltcs_best_alpha}, L1 = {nltcs_best_l1:.4f}")
print(f"test_300x10 (低重复 0%)  最优 α = {test_best_alpha}, L1 = {test_best_l1:.4f}")

if nltcs_best_alpha != test_best_alpha:
    print("\n✓ 假设**部分成立**：不同数据集的最优 α 不同")
    print("  → 高重复数据偏好高 α")
    print("  → 低重复数据偏好中等 α")
    print("\n结论：不存在单一的通用最优 α，需根据数据特征调整")
else:
    print("\n✗ 假设**不成立**：两个数据集的最优 α 相同")
    print(f"  → α={nltcs_best_alpha} 是鲁棒最优")

# 进一步检验：α1.5→6 的鲁棒性
print("\n" + "=" * 70)
print("α1.5→6 的鲁棒性检验")
print("=" * 70)

for dataset, data in results.items():
    l1_15_6 = data['alpha_configs']['1.5→6']['L1']
    l1_best = min([m['L1'] for m in data['alpha_configs'].values()])
    gap_pct = (l1_15_6 - l1_best) / l1_best * 100

    print(f"\n{dataset}:")
    print(f"  α1.5→6 的 L1 = {l1_15_6:.4f}")
    print(f"  该数据集最优 L1 = {l1_best:.4f}")
    print(f"  相对差距 = +{gap_pct:.1f}%")

    if gap_pct < 10:
        print(f"  → 接近最优（<10% 差距）✓")
    elif gap_pct < 50:
        print(f"  → 可接受（10-50% 差距）")
    else:
        print(f"  → 明显次优（>50% 差距）")

print("\n" + "=" * 70)
print("最终建议")
print("=" * 70)

avg_gap = np.mean([
    (results['nltcs']['alpha_configs']['1.5→6']['L1'] - min([m['L1'] for m in results['nltcs']['alpha_configs'].values()])) / min([m['L1'] for m in results['nltcs']['alpha_configs'].values()]) * 100,
    (results['test_300x10']['alpha_configs']['1.5→6']['L1'] - min([m['L1'] for m in results['test_300x10']['alpha_configs'].values()])) / min([m['L1'] for m in results['test_300x10']['alpha_configs'].values()]) * 100
])

print(f"\nα1.5→6 在两个数据集上平均偏离最优 {avg_gap:.1f}%")

if avg_gap < 50:
    print("\n推荐：α1.5→6 作为通用配置")
    print("  理由：虽不是每个数据集的最优，但鲁棒性好、泛化能力强")
else:
    print("\n推荐：根据数据集重复率动态调整 α")
    print("  高重复（>50%）→ α2→10")
    print("  低重复（<20%）→ α1.5→6 或 α0.5→4")
