#!/usr/bin/env python3
"""演示实验基础设施的集成使用（仅接线演示，非真实实验）。

此脚本展示如何把 metrics、experiment_logger、experiment_config、acceptance
四个模块接线到一起，跑通"配置→模拟循环→记录→汇总"的完整数据流。

注意：这里的候选生成是 `current + 高斯噪声` 的玩具模拟（simulate_evolution），
不是 evolution.py 的真实演化逻辑，也不加载真实数据。它只用于验证基础设施接口
能协同工作，产出的数字没有实验意义。输出统一写到被 gitignore 的
`experiments/results/.demo/`，不进入版本库。
"""

import sys
from pathlib import Path
import numpy as np

# 添加 src 到 path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from table_diffevo.metrics import compute_all_metrics
from table_diffevo.experiment_logger import ExperimentLogger
from table_diffevo.experiment_config import (
    ExperimentConfig, DataConfig, AcceptanceRuleConfig, AlphaScheduleConfig
)
from table_diffevo.acceptance import check_acceptance


def simulate_evolution(config: ExperimentConfig, seed: int, logger: ExperimentLogger):
    """模拟一次演化过程（简化版）。

    Args:
        config: 实验配置
        seed: 随机种子
        logger: 日志记录器
    """
    np.random.seed(seed)

    # 模拟目标向量（实际使用中从配置文件加载）
    n_queries = 100
    target = np.random.randint(50, 500, size=n_queries).astype(float)

    # 初始化：当前合成表答案（从较差的状态开始）
    current = target + np.random.randn(n_queries) * 100

    # 追踪最佳状态
    best_L1 = float('inf')
    best_Q = float('inf')
    best_current = current.copy()

    # 演化参数
    alpha = config.alpha_schedule.alpha_value
    alpha_min = config.alpha_schedule.alpha_min
    alpha_max = config.alpha_schedule.alpha_max
    u = (alpha - alpha_min) / (alpha_max - alpha_min)  # 归一化 α

    n_rounds = config.n_rounds
    n_records = config.data.n_records

    # 接受规则参数
    eps_L1 = config.acceptance_rule.eps_L1
    eps_Q = config.acceptance_rule.eps_Q
    rule = config.acceptance_rule.rule

    candidate_evaluations = 0

    print(f"  种子 {seed}:")
    print(f"    接受规则: {rule}, α={alpha:.1f}")

    for round_idx in range(n_rounds):
        # 模拟候选生成（实际使用中调用 evolution.py 的逻辑）
        candidate = current + np.random.randn(n_queries) * 50
        candidate_evaluations += 1

        # 在覆盖前计算差值并判断接受（使用统一接口）
        accepted, delta_L1, delta_Q = check_acceptance(
            rule=rule,
            target=target,
            current=current,
            candidate=candidate,
            n_records=n_records,
            eps_L1=eps_L1,
            eps_Q=eps_Q
        )

        # 计算当前度量（用于日志和最佳状态更新）
        current_L1, current_Q, _ = compute_all_metrics(target, current, n_records)

        # 更新状态
        if accepted:
            current = candidate
            # 重新计算 current_L1 和 current_Q（已被覆盖）
            current_L1, current_Q, _ = compute_all_metrics(target, current, n_records)

        # 更新最佳
        if current_L1 < best_L1:
            best_L1 = current_L1
            best_Q = current_Q
            best_current = current.copy()

        # 记录日志（差值无论接受与否都记录）
        logger.log_round(
            seed=seed,
            arm=rule,
            round=round_idx + 1,
            block=(round_idx + 1) // 10,  # 假设 10 轮为一块
            alpha=alpha,
            u=u,
            L1_current=current_L1,
            best_L1=best_L1,
            Q_current=current_Q,
            accepted=accepted,
            delta_L1=delta_L1,
            delta_Q=delta_Q,
            candidate_evaluations=candidate_evaluations
        )

    print(f"    最终 best_L1: {best_L1:.6f}")
    print(f"    总候选评估数: {candidate_evaluations}")

    return best_L1, candidate_evaluations


def main():
    print("=" * 70)
    print("实验基础设施集成演示")
    print("=" * 70)
    print()

    # 步骤 1：加载配置
    print("步骤 1: 加载配置")
    config_path = Path(__file__).parent.parent / "experiments/configs/example_phase_a.yaml"

    if not config_path.exists():
        print(f"  ⚠ 配置文件不存在: {config_path}")
        print("  使用默认参数创建演示配置...")

        # 创建演示配置（实际使用中从 YAML 加载）
        config = ExperimentConfig(
            experiment_name="demo_infrastructure",
            data=DataConfig(
                dataset_name="demo",
                target_path="",
                measured_target_path="",
                init_marginals_path="",
                n_records=1000
            ),
            acceptance_rule=AcceptanceRuleConfig(
                rule="A1",
                eps_L1=1e-5,
                eps_Q=0.0
            ),
            alpha_schedule=AlphaScheduleConfig(
                mode="fixed",
                alpha_value=5.0,
                alpha_min=2.0,
                alpha_max=10.0
            ),
            seeds=[42, 43],
            n_rounds=50,  # 演示用少量轮数
            output_dir="experiments/results/.demo",  # 加 . 前缀避免进入 git
            beta=1.0,
            eta=0.5,
            h=0.8,
            mu=0.01,
            lambda_=0.5,
            delta=0.05,
            winsorize_limits=[0.01, 0.99]
        )
    else:
        config = ExperimentConfig.from_yaml(config_path)
        # 演示用少量轮数
        config.n_rounds = 50
        config.seeds = config.seeds[:2]  # 只用前两个种子
        config.output_dir = "experiments/results/.demo"  # 加 . 前缀避免进入 git

    print(f"  ✓ 配置已加载: {config.experiment_name}")
    print(f"    接受规则: {config.acceptance_rule.rule}")
    print(f"    α 模式: {config.alpha_schedule.mode} (α={config.alpha_schedule.alpha_value})")
    print(f"    种子: {config.seeds}")
    print(f"    轮数: {config.n_rounds}")
    print()

    # 步骤 2：创建日志记录器
    print("步骤 2: 创建日志记录器")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = ExperimentLogger(output_dir)
    print(f"  ✓ 日志目录: {output_dir}")
    print()

    # 步骤 3：运行演化（简化模拟）
    print("步骤 3: 运行演化（简化模拟）")
    all_best_L1 = []
    all_evaluations = []

    for seed in config.seeds:
        best_L1, total_evals = simulate_evolution(config, seed, logger)
        all_best_L1.append(best_L1)
        all_evaluations.append(total_evals)

    print()

    # 步骤 4：保存统计信息和日志
    print("步骤 4: 保存统计信息和日志")
    logger.add_stat("config_name", config.experiment_name)
    logger.add_stat("acceptance_rule", config.acceptance_rule.rule)
    logger.add_stat("alpha_mode", config.alpha_schedule.mode)
    logger.add_stat("alpha_value", config.alpha_schedule.alpha_value)
    logger.add_stat("seeds", config.seeds)
    logger.add_stat("n_rounds", config.n_rounds)
    logger.add_stat("mean_best_L1", float(np.mean(all_best_L1)))
    logger.add_stat("std_best_L1", float(np.std(all_best_L1)))
    logger.add_stat("mean_evaluations", float(np.mean(all_evaluations)))

    logger.save()

    print(f"  ✓ 日志已保存到 {output_dir}")
    print(f"    - rounds.csv: 每轮详细日志")
    print(f"    - summary.json: 统计信息")
    print()

    # 步骤 5：显示汇总结果
    print("步骤 5: 汇总结果")
    print(f"  平均 best_L1: {np.mean(all_best_L1):.6f} ± {np.std(all_best_L1):.6f}")
    print(f"  平均候选评估数: {np.mean(all_evaluations):.0f}")
    print()

    print("=" * 70)
    print("✅ 演示完成")
    print("=" * 70)
    print()
    print("提示:")
    print(f"  - 查看详细日志: cat {output_dir}/rounds.csv")
    print(f"  - 查看统计信息: cat {output_dir}/summary.json")
    print()


if __name__ == "__main__":
    main()
