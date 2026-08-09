#!/usr/bin/env python3
"""
Mini Pilot：在 test_300x10 上验证 A0 vs A1 配对对照

目标：
1. 验证配对对照设计正确（相同种子、相同初始化）
2. 检查四象限统计在多种子下的稳定性
3. 验证能正确生成对比图表
4. 快速反馈（5 分钟）

数据集：test_300x10（300 条记录，10 个属性，50 个查询）
种子：[42, 43, 44]（dev seeds）
轮数：100
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
import json

# 数据集配置
DATASET = "test_300x10"
CONFIG_DIR = Path("configs") / DATASET
SEEDS = [42, 43, 44]
N_ROUNDS = 500  # 用户 2026-08-09 决定：小表跑 500 轮

# 固定参数（与 nltcs 正式实验一致）
PARAMS = {
    "beta": 1.0,
    "h": 0.8,
    "eta": 0.5,
    "mu": 0.01,
    "lambda_param": 0.5,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
    "distance_mode": "geometric",
    "init_method": "marginal",
    # 残差定向扩散：必须开，理由同 smoke_test_acceptance.py。关着时本表上 A1
    # 在 seed 42/43 上 0/100 接受、seed 44 仅 8/100，L1 死在初始值；开启后
    # 3 种子 × {100,500} 轮共 6 格全部解除冻结。强度 2.0 与归一化方式对齐
    # experiments/probe_convergence_a2.py 的 FROZEN_PARAMS。两条臂同设。
    "residual_directed_diffusion": True,
    "diffusion_direction_strength": 2.0,
    # 2.0 是相对首轮方向分数 RMS 的倍数，不是绝对量纲（见 evolution.py:749-753）。
    "diffusion_direction_normalization": "initial_rms",
    # alpha 范围：用户 2026-08-09 决定改为 2-10（原为 1-10）
    "alpha_min": 2.0,
    "alpha_max": 10.0,
    # 排除自抽样：避免空转（抽到自己 → ΔL1=0, ΔQ=0 → eps=0 必拒绝）
    "exclude_self": True,
}

# 输出目录
OUTPUT_DIR = Path("experiments/results/mini_pilot_acceptance")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """加载数据集"""
    schema = load_schema(CONFIG_DIR / "schema.yaml")
    queries = load_queries(CONFIG_DIR / "measured_50query.json")

    with open(CONFIG_DIR / "measured_50query.json") as f:
        queries_json = json.load(f)

    target = np.array([q["result"] for q in queries_json["queries"]], dtype=float)
    n_records = queries_json["record_count"]
    marginals = load_marginals(CONFIG_DIR / "init_marginals.json")
    return schema, queries, target, n_records, marginals


def analyze_quadrants(diag):
    """分析四象限统计"""
    delta_L1_history = diag["delta_L1_history"]
    delta_Q_history = diag["delta_Q_history"]
    accept_history = diag["accept_history"]

    # 展平所有尝试
    all_delta_L1 = []
    all_delta_Q = []
    all_accepted = []

    for round_idx, (round_L1, round_Q, accepted) in enumerate(
        zip(delta_L1_history, delta_Q_history, accept_history)
    ):
        for attempt_idx, (dL1, dQ) in enumerate(zip(round_L1, round_Q)):
            all_delta_L1.append(dL1)
            all_delta_Q.append(dQ)
            # 只有当轮接受且是最后一次尝试才算接受
            is_accepted = accepted and (attempt_idx == len(round_L1) - 1)
            all_accepted.append(is_accepted)

    total = len(all_delta_L1)
    if total == 0:
        return {
            "total": 0,
            "q_down_l1_down": 0,
            "q_up_l1_down": 0,
            "q_down_l1_up": 0,
            "q_up_l1_up": 0,
            "accepted_q_up_l1_down": 0,
        }

    # 四象限计数（所有尝试）
    q_down_l1_down = sum(
        1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 < 0 and dQ < 0
    )
    q_up_l1_down = sum(
        1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 < 0 and dQ >= 0
    )
    q_down_l1_up = sum(
        1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 >= 0 and dQ < 0
    )
    q_up_l1_up = sum(
        1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 >= 0 and dQ >= 0
    )

    # A1 专属区被接受的次数
    accepted_q_up_l1_down = sum(
        1
        for dL1, dQ, acc in zip(all_delta_L1, all_delta_Q, all_accepted)
        if dL1 < 0 and dQ >= 0 and acc
    )

    return {
        "total": total,
        "q_down_l1_down": q_down_l1_down,
        "q_up_l1_down": q_up_l1_down,
        "q_down_l1_up": q_down_l1_up,
        "q_up_l1_up": q_up_l1_up,
        "accepted_q_up_l1_down": accepted_q_up_l1_down,
    }


def run_mini_pilot():
    """运行 mini pilot"""
    print("=" * 60)
    print("Mini Pilot: A0 vs A1 接受规则配对对照")
    print("=" * 60)
    print(f"数据集: {DATASET}")
    print(f"种子: {SEEDS}")
    print(f"轮数: {N_ROUNDS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 加载数据
    print("加载数据集...")
    schema, queries, target, n_records, marginals = load_dataset()
    print(f"  记录数: {n_records}")
    print(f"  查询数: {len(queries)}")
    print(f"  属性数: {len(schema.attributes)}")
    print()

    results = []

    for seed in SEEDS:
        print(f"{'=' * 60}")
        print(f"种子 {seed}")
        print(f"{'=' * 60}")

        seed_results = {"seed": seed}

        for rule in ["A0", "A1"]:
            print(f"\n运行 {rule}...")
            print("-" * 60)

            # 运行演化
            best_S, diag = run_evolution(
                target=target,
                queries=queries,
                schema=schema,
                n_records=n_records,
                n_rounds=N_ROUNDS,
                seed=seed,
                acceptance_rule=rule,
                # eps_L1=0：平局带退化为「ΔL1 恰好为 0」。依据是 ΔL1 的量化性——
                # normalized_l1 的分子是整数计数之差，ΔL1 只能以 1/(m*N) 为步长
                # 跳变，0 与最小非零变化之间隔着一整个步长，无需容差吸收毛刺。
                # 定标史见 PROJECT_STATUS.md。本数据集（300x50）实测 9/500 次
                # 尝试为精确平局，故 Q 平局裁决那一路确实会触发。
                eps_L1=0.0,
                eps_Q=0.0,
                marginals=marginals,
                log_every=0,  # 不打印逐轮
                **PARAMS,
            )

            # 统计
            final_L1 = diag["normalized_l1_error"]
            final_Q = diag["best_loss"]
            accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])
            quadrants = analyze_quadrants(diag)

            seed_results[f"{rule}_final_L1"] = final_L1
            seed_results[f"{rule}_final_Q"] = final_Q
            seed_results[f"{rule}_accept_rate"] = accept_rate
            seed_results[f"{rule}_quadrants"] = quadrants

            print(f"  最终 L1: {final_L1:.6f}")
            print(f"  最终 Q: {final_Q:.2f}")
            print(f"  接受率: {accept_rate:.1%}")

        # 配对差
        delta_L1 = seed_results["A1_final_L1"] - seed_results["A0_final_L1"]
        delta_Q = seed_results["A1_final_Q"] - seed_results["A0_final_Q"]
        seed_results["delta_L1"] = delta_L1
        seed_results["delta_Q"] = delta_Q

        print(f"\n配对差:")
        print(f"  ΔL1 (A1 - A0): {delta_L1:+.6f}")
        print(f"  ΔQ (A1 - A0): {delta_Q:+.2f}")

        if delta_L1 < 0:
            winner = "A1"
        elif delta_L1 > 0:
            winner = "A0"
        else:
            winner = "Tie"
        seed_results["winner"] = winner
        print(f"  胜者: {winner}")

        results.append(seed_results)

    # 汇总结果
    print("\n" + "=" * 60)
    print("汇总结果")
    print("=" * 60)

    df = pd.DataFrame(results)
    print("\n配对差分布:")
    print(df[["seed", "delta_L1", "delta_Q", "winner"]])

    # 统计胜负
    wins_A1 = sum(1 for r in results if r["winner"] == "A1")
    wins_A0 = sum(1 for r in results if r["winner"] == "A0")
    ties = sum(1 for r in results if r["winner"] == "Tie")

    print(f"\n胜负统计:")
    print(f"  A1 获胜: {wins_A1}/{len(SEEDS)}")
    print(f"  A0 获胜: {wins_A0}/{len(SEEDS)}")
    print(f"  平局: {ties}/{len(SEEDS)}")

    # 保存结果
    csv_path = OUTPUT_DIR / "mini_pilot_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ 结果已保存到: {csv_path}")

    # 四象限汇总
    print(f"\n四象限统计（跨种子平均）:")
    for rule in ["A0", "A1"]:
        total_attempts = sum(r[f"{rule}_quadrants"]["total"] for r in results)
        q_down_l1_down = sum(r[f"{rule}_quadrants"]["q_down_l1_down"] for r in results)
        q_up_l1_down = sum(r[f"{rule}_quadrants"]["q_up_l1_down"] for r in results)
        q_down_l1_up = sum(r[f"{rule}_quadrants"]["q_down_l1_up"] for r in results)
        q_up_l1_up = sum(r[f"{rule}_quadrants"]["q_up_l1_up"] for r in results)

        print(f"\n{rule}:")
        print(f"  总尝试: {total_attempts}")
        print(f"  [Q↓,L1↓]: {q_down_l1_down} ({q_down_l1_down/total_attempts*100:.1f}%)")
        print(f"  [Q↑,L1↓]: {q_up_l1_down} ({q_up_l1_down/total_attempts*100:.1f}%)")
        print(f"  [Q↓,L1↑]: {q_down_l1_up} ({q_down_l1_up/total_attempts*100:.1f}%)")
        print(f"  [Q↑,L1↑]: {q_up_l1_up} ({q_up_l1_up/total_attempts*100:.1f}%)")

        if rule == "A1":
            accepted_q_up_l1_down = sum(
                r[f"{rule}_quadrants"]["accepted_q_up_l1_down"] for r in results
            )
            print(
                f"  A1专属区被接受: {accepted_q_up_l1_down} ({accepted_q_up_l1_down/total_attempts*100:.1f}%)"
            )

    print("\n" + "=" * 60)
    print("✓ Mini pilot 完成！")
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        results = run_mini_pilot()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Mini pilot 失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
