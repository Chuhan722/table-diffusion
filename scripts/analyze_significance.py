"""
统计显著性检验

用法：
    python scripts/analyze_significance.py outputs/distance_mode_experiment_2026-07-25_0812/
"""
import json
import os
import sys
from scipy.stats import ttest_ind

def load_group_losses(exp_dir, mode):
    """从某组的 summary.json 提取所有种子的 best_loss"""
    summary_path = os.path.join(exp_dir, mode, "summary.json")
    with open(summary_path, 'r') as f:
        data = json.load(f)
    return [s['best_loss'] for s in data['per_seed']]

def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/analyze_significance.py <实验目录>")
        print("例如: python scripts/analyze_significance.py outputs/distance_mode_experiment_2026-07-25_0812/")
        sys.exit(1)

    exp_dir = sys.argv[1]

    # 读取三组数据
    squared_losses = load_group_losses(exp_dir, 'squared')
    linear_losses = load_group_losses(exp_dir, 'linear')
    none_losses = load_group_losses(exp_dir, 'none')

    print(f"===== 统计显著性检验（{exp_dir}）=====\n")

    # LINEAR vs SQUARED
    t_stat, p_value = ttest_ind(linear_losses, squared_losses)
    print(f"LINEAR vs SQUARED:")
    print(f"  LINEAR 均值: {sum(linear_losses)/len(linear_losses):.2e}")
    print(f"  SQUARED 均值: {sum(squared_losses)/len(squared_losses):.2e}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  显著？ {'是' if p_value < 0.05 else '否'} (p<0.05)\n")

    # LINEAR vs NONE
    t_stat, p_value = ttest_ind(linear_losses, none_losses)
    print(f"LINEAR vs NONE:")
    print(f"  LINEAR 均值: {sum(linear_losses)/len(linear_losses):.2e}")
    print(f"  NONE 均值: {sum(none_losses)/len(none_losses):.2e}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  显著？ {'是' if p_value < 0.05 else '否'} (p<0.05)\n")

    # SQUARED vs NONE
    t_stat, p_value = ttest_ind(squared_losses, none_losses)
    print(f"SQUARED vs NONE:")
    print(f"  SQUARED 均值: {sum(squared_losses)/len(squared_losses):.2e}")
    print(f"  NONE 均值: {sum(none_losses)/len(none_losses):.2e}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.6f}")
    print(f"  显著？ {'是' if p_value < 0.05 else '否'} (p<0.05)\n")

if __name__ == "__main__":
    main()
