"""
分析 donor 诊断数据：对比 squared vs linear 选中的 donor 特征

用法：
    python scripts/analyze_donor_diagnostics.py \
        outputs/.../squared/0-0/diagnostics.json \
        outputs/.../linear/0-0/diagnostics.json
"""
import json
import sys
import matplotlib.pyplot as plt
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("用法: python scripts/analyze_donor_diagnostics.py <squared诊断> <linear诊断>")
        sys.exit(1)

    squared_path = sys.argv[1]
    linear_path = sys.argv[2]

    with open(squared_path, 'r') as f:
        squared = json.load(f)
    with open(linear_path, 'r') as f:
        linear = json.load(f)

    # 提取数据
    sq_fitness = squared['donor_fitness_history']
    sq_distance = squared['donor_distance_history']
    lin_fitness = linear['donor_fitness_history']
    lin_distance = linear['donor_distance_history']

    rounds = len(sq_fitness)
    x = np.arange(1, rounds + 1)

    # 画图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 上图：donor 平均适应度
    ax1.plot(x, sq_fitness, label='SQUARED', color='red', alpha=0.7)
    ax1.plot(x, lin_fitness, label='LINEAR', color='blue', alpha=0.7)
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('选中 donor 的平均适应度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Donor 适应度对比（越高越好）')

    # 下图：到 donor 的平均距离
    ax2.plot(x, sq_distance, label='SQUARED', color='red', alpha=0.7)
    ax2.plot(x, lin_distance, label='LINEAR', color='blue', alpha=0.7)
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('到选中 donor 的平均距离')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Donor 距离对比')

    plt.tight_layout()
    plt.savefig('donor_diagnostics.png', dpi=150)
    print("图已保存: donor_diagnostics.png")

    # 打印统计
    print(f"\n===== 统计对比 =====")
    print(f"SQUARED - 平均 donor 适应度: {np.mean(sq_fitness):.6f}")
    print(f"LINEAR  - 平均 donor 适应度: {np.mean(lin_fitness):.6f}")
    diff_pct = (np.mean(lin_fitness) - np.mean(sq_fitness)) / abs(np.mean(sq_fitness)) * 100
    print(f"LINEAR 优势: {diff_pct:+.2f}%\n")

    print(f"SQUARED - 平均 donor 距离: {np.mean(sq_distance):.4f}")
    print(f"LINEAR  - 平均 donor 距离: {np.mean(lin_distance):.4f}")
    print(f"LINEAR 距离{'更远' if np.mean(lin_distance) > np.mean(sq_distance) else '更近'}: {abs(np.mean(lin_distance) - np.mean(sq_distance)):.4f}")

if __name__ == "__main__":
    main()
