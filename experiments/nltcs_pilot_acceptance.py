#!/usr/bin/env python3
"""
NLTCS Pilot：在 nltcs 数据集上测试 A0 vs A1 接受规则

目标：
1. 验证 A1 在大规模数据集上是否有优势
2. 检查四象限分布是否与 test_300x10 不同
3. 配对对照设计：相同种子、相同初始化、只改接受规则

数据集：nltcs（134 条记录，16 个属性，1000 个查询）
种子：[42, 43, 44]
轮数：500
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
import json

# 数据集配置
DATASET = "nltcs"
CONFIG_DIR = Path("configs") / DATASET
SEEDS = [42, 43, 44]
N_ROUNDS = 1000  # 用户 2026-08-09 决定：nltcs 跑 1000 轮

# 固定参数
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
    # 残差定向扩散：必须开，理由同 smoke_test_acceptance.py（小表上关着时 A1
    # 完全冻结）。本数据集上的冻结未单独复现，但这里开它还有一条独立依据：
    # memory 记录的「单块+残差在 nltcs 上 L1 单调降、强度 2.0 达 -19.3%」正是
    # 开着它测出来的，关着等于拿一套未经验证的生成器跑正式对照。强度 2.0 与
    # 归一化方式对齐 experiments/probe_convergence_a2.py 的 FROZEN_PARAMS。
    # 两条臂（A0/A1）同设。
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


def load_dataset():
    """加载数据集"""
    schema = load_schema(CONFIG_DIR / "schema.yaml")
    queries = load_queries(CONFIG_DIR / "measured_1000query.json")

    # 从 JSON 文件直接读取 metadata
    with open(CONFIG_DIR / "measured_1000query.json") as f:
        query_data = json.load(f)

    n_records = query_data["record_count"]
    target = np.array([q["result"] for q in query_data["queries"]], dtype=float)

    # 加载初始 marginals
    marginals = load_marginals(str(CONFIG_DIR / "init_marginals.json"))

    return schema, queries, target, n_records, marginals


def analyze_quadrants(delta_L1_history, delta_Q_history):
    """分析四象限统计"""
    all_delta_L1 = [dL1 for round_attempts in delta_L1_history for dL1 in round_attempts]
    all_delta_Q = [dQ for round_attempts in delta_Q_history for dQ in round_attempts]

    quadrants = {
        "[Q↓,L1↓]": 0,  # 双赢区
        "[Q↑,L1↓]": 0,  # A1专属区
        "[Q↓,L1↑]": 0,  # A0专属区
        "[Q↑,L1↑]": 0,  # 双输区
    }

    for dL1, dQ in zip(all_delta_L1, all_delta_Q):
        if dQ <= 0 and dL1 <= 0:
            quadrants["[Q↓,L1↓]"] += 1
        elif dQ > 0 and dL1 <= 0:
            quadrants["[Q↑,L1↓]"] += 1
        elif dQ <= 0 and dL1 > 0:
            quadrants["[Q↓,L1↑]"] += 1
        else:
            quadrants["[Q↑,L1↑]"] += 1

    total = len(all_delta_L1)
    quadrant_pcts = {k: v / total * 100 if total > 0 else 0 for k, v in quadrants.items()}

    return quadrants, quadrant_pcts


def run_single_trial(rule: str, seed: int, schema, queries, target, n_records, marginals):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"规则: {rule}, 种子: {seed}")
    print(f"{'='*60}")

    best_S, diagnostics = run_evolution(
        schema=schema,
        queries=queries,
        target=target,
        n_records=n_records,
        n_rounds=N_ROUNDS,
        seed=seed,
        acceptance_rule=rule,
        # eps_L1=0：平局带退化为「ΔL1 恰好为 0」。依据是 ΔL1 的量化性——
        # normalized_l1 的分子是整数计数之差，ΔL1 只能以 1/(m*N) 为步长跳变，
        # 0 与最小非零变化之间隔着一整个步长，无需容差吸收毛刺。
        # 定标史见 PROJECT_STATUS.md。注意本数据集（16181x1001）实测 300 次尝试
        # 0 次精确平局，故 A1 在 nltcs 上事实上是纯 L1 判据，Q 平局裁决不触发；
        # 且 18% 的尝试 ΔL1 与 ΔQ 方向相反，这些全由 L1 单方裁决。这是 A1 该
        # 承担的风险，须在结果分析里披露，不得当成 A1 的设计已被完整检验。
        eps_L1=0.0,
        eps_Q=0.0,
        marginals=marginals,
        device='cuda',
        **PARAMS
    )

    # 提取关键指标
    final_L1 = diagnostics["normalized_l1_error"]
    final_Q = diagnostics["loss_history"][-1]
    accept_history = diagnostics["accept_history"]
    accept_rate = sum(accept_history) / len(accept_history) if accept_history else 0.0

    # 四象限统计
    delta_L1_hist = diagnostics["delta_L1_history"]
    delta_Q_hist = diagnostics["delta_Q_history"]
    quadrants, quadrant_pcts = analyze_quadrants(delta_L1_hist, delta_Q_hist)

    print(f"\n结果摘要:")
    print(f"  最终 L1: {final_L1:.6f}")
    print(f"  最终 Q: {final_Q:.2f}")
    print(f"  接受率: {accept_rate:.1%}")
    print(f"\n四象限分布:")
    for name, pct in quadrant_pcts.items():
        print(f"  {name}: {pct:.1f}%")

    return {
        "rule": rule,
        "seed": seed,
        "final_L1": final_L1,
        "final_Q": final_Q,
        "accept_rate": accept_rate,
        "quadrants": quadrants,
        "quadrant_pcts": quadrant_pcts,
    }


def main():
    print("="*60)
    print("NLTCS Pilot: A0 vs A1 接受规则对照实验")
    print("="*60)

    # 加载数据集
    print("\n加载数据集...")
    schema, queries, target, n_records, marginals = load_dataset()
    print(f"  数据集: {DATASET}")
    print(f"  记录数: {n_records}")
    print(f"  查询数: {len(queries)}")
    print(f"  种子: {SEEDS}")
    print(f"  轮数: {N_ROUNDS}")

    # 运行所有实验
    results = []
    for seed in SEEDS:
        for rule in ["A0", "A1"]:
            result = run_single_trial(rule, seed, schema, queries, target, n_records, marginals)
            results.append(result)

    # 保存结果
    output_dir = Path("experiments/results/nltcs_pilot_acceptance")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    csv_path = output_dir / "nltcs_pilot_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n结果已保存到: {csv_path}")

    # 配对对照分析
    print("\n" + "="*60)
    print("配对对照分析")
    print("="*60)

    for seed in SEEDS:
        A0_result = df[(df["rule"] == "A0") & (df["seed"] == seed)].iloc[0]
        A1_result = df[(df["rule"] == "A1") & (df["seed"] == seed)].iloc[0]

        delta_L1 = A1_result["final_L1"] - A0_result["final_L1"]
        winner = "A0" if delta_L1 > 0 else "A1"

        print(f"\n种子 {seed}:")
        print(f"  A0: L1={A0_result['final_L1']:.6f}, 接受率={A0_result['accept_rate']:.1%}")
        print(f"  A1: L1={A1_result['final_L1']:.6f}, 接受率={A1_result['accept_rate']:.1%}")
        print(f"  ΔL1 (A1-A0): {delta_L1:+.6f}")
        print(f"  胜者: {winner}")


if __name__ == "__main__":
    main()
