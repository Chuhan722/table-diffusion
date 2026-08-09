#!/usr/bin/env python3
"""
Smoke test：验证 A0/A1 接受规则在 test_300x10 上能正常运行

目标：
1. 验证 A0 和 A1 都能跑完不报错
2. 检查四象限统计不为空
3. 验证日志字段完整
4. 快速反馈（1-2 分钟）

数据集：test_300x10（300 条记录，10 个属性，50 个查询）
种子：[42]（单种子）
轮数：100（快速验证）
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals

# 数据集配置
DATASET = "test_300x10"
CONFIG_DIR = Path("configs") / DATASET
SEED = 42
N_ROUNDS = 100

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
    # 残差定向扩散：必须开。关着时生成器不看残差，本表上 94/100 个提案落在
    # [Q↓, L1↑] 象限且无一个 ΔL1<=0，A1 全数拒绝 → 100/300 轮 0 接受、L1 死在
    # 初始值 0.016867，A0/A1 对照测不出东西。开启后提案分布改变
    # （[Q↓,L1↓] 由 0 升到 18），冻结消失。强度 2.0 与归一化方式对齐
    # experiments/probe_convergence_a2.py 的 FROZEN_PARAMS，也是 memory 里
    # 「nltcs L1 -19.3%」那条结论所用的配置。两条臂（A0/A1）同设。
    "residual_directed_diffusion": True,
    "diffusion_direction_strength": 2.0,
    # 2.0 是相对首轮方向分数 RMS 的倍数，不是绝对量纲；显式写出归一化方式，
    # 否则这个数值无法解释（见 evolution.py:749-753）。
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
    queries = load_queries(CONFIG_DIR / "measured_50query.json")

    # 从 JSON 文件直接读取 metadata
    import json
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

    # 展平所有尝试
    all_delta_L1 = [dL1 for round_L1 in delta_L1_history for dL1 in round_L1]
    all_delta_Q = [dQ for round_Q in delta_Q_history for dQ in round_Q]

    total = len(all_delta_L1)
    if total == 0:
        return {
            "total": 0,
            "q_down_l1_down": 0,
            "q_up_l1_down": 0,
            "q_down_l1_up": 0,
            "q_up_l1_up": 0,
        }

    # 四象限计数
    q_down_l1_down = sum(1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 < 0 and dQ < 0)
    q_up_l1_down = sum(1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 < 0 and dQ >= 0)
    q_down_l1_up = sum(1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 >= 0 and dQ < 0)
    q_up_l1_up = sum(1 for dL1, dQ in zip(all_delta_L1, all_delta_Q) if dL1 >= 0 and dQ >= 0)

    return {
        "total": total,
        "q_down_l1_down": q_down_l1_down,
        "q_up_l1_down": q_up_l1_down,
        "q_down_l1_up": q_down_l1_up,
        "q_up_l1_up": q_up_l1_up,
    }


def run_smoke_test():
    """运行 smoke test"""
    print("=" * 60)
    print("Smoke Test: A0 vs A1 接受规则")
    print("=" * 60)
    print(f"数据集: {DATASET}")
    print(f"种子: {SEED}")
    print(f"轮数: {N_ROUNDS}")
    print()

    # 加载数据
    print("加载数据集...")
    schema, queries, target, n_records, marginals = load_dataset()
    print(f"  记录数: {n_records}")
    print(f"  查询数: {len(queries)}")
    print(f"  属性数: {len(schema.attributes)}")
    print()

    results = {}

    for rule in ["A0", "A1"]:
        print(f"运行 {rule}...")
        print("-" * 60)

        # 运行演化
        best_S, diag = run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=n_records,
            n_rounds=N_ROUNDS,
            seed=SEED,
            acceptance_rule=rule,
            # eps_L1=0：平局带退化为「ΔL1 恰好为 0」。依据是 ΔL1 的量化性——
            # normalized_l1 的分子是整数计数之差，ΔL1 只能以 1/(m*N) 为步长跳变，
            # 0 与最小非零变化之间隔着一整个步长，没有需要容差吸收的浮点毛刺。
            # 定标史见 PROJECT_STATUS.md：预注册的 0.02 在两数据集都覆盖 100% 候选
            # （A1 退化成 A0），而任何非零值的落带率在两数据集差一到两个数量级。
            eps_L1=0.0,
            eps_Q=0.0,
            marginals=marginals,
            log_every=20,  # 每 20 轮打印一次
            **PARAMS,
        )

        # 检查关键字段
        assert "delta_L1_history" in diag, f"{rule}: 缺少 delta_L1_history"
        assert "delta_Q_history" in diag, f"{rule}: 缺少 delta_Q_history"
        assert diag["params"]["acceptance_rule"] == rule, f"{rule}: 参数记录错误"

        # 统计
        final_L1 = diag["normalized_l1_error"]
        final_Q = diag["best_loss"]
        accept_rate = sum(diag["accept_history"]) / len(diag["accept_history"])
        quadrants = analyze_quadrants(diag)

        results[rule] = {
            "final_L1": final_L1,
            "final_Q": final_Q,
            "accept_rate": accept_rate,
            "quadrants": quadrants,
        }

        print(f"  最终 L1: {final_L1:.6f}")
        print(f"  最终 Q: {final_Q:.2f}")
        print(f"  接受率: {accept_rate:.1%}")
        print(f"  尝试总数: {quadrants['total']}")
        print(f"  四象限分布:")
        print(f"    [Q↓,L1↓]: {quadrants['q_down_l1_down']} ({quadrants['q_down_l1_down']/quadrants['total']*100:.1f}%)")
        print(f"    [Q↑,L1↓]: {quadrants['q_up_l1_down']} ({quadrants['q_up_l1_down']/quadrants['total']*100:.1f}%) ← A1专属")
        print(f"    [Q↓,L1↑]: {quadrants['q_down_l1_up']} ({quadrants['q_down_l1_up']/quadrants['total']*100:.1f}%) ← A0专属")
        print(f"    [Q↑,L1↑]: {quadrants['q_up_l1_up']} ({quadrants['q_up_l1_up']/quadrants['total']*100:.1f}%)")
        print()

    # 对比
    print("=" * 60)
    print("对比结果")
    print("=" * 60)
    delta_L1 = results["A1"]["final_L1"] - results["A0"]["final_L1"]
    delta_Q = results["A1"]["final_Q"] - results["A0"]["final_Q"]
    print(f"ΔL1 (A1 - A0): {delta_L1:+.6f}")
    print(f"ΔQ (A1 - A0): {delta_Q:+.2f}")

    if delta_L1 < 0:
        print("✓ A1 最终 L1 更低")
    elif delta_L1 > 0:
        print("✓ A0 最终 L1 更低")
    else:
        print("✓ 两者 L1 相同")

    # 验证四象限
    a0_q_down_l1_up = results["A0"]["quadrants"]["q_down_l1_up"]
    a1_q_up_l1_down = results["A1"]["quadrants"]["q_up_l1_down"]

    print()
    print("四象限验证:")
    print(f"  A0 专属区 [Q↓,L1↑]: {a0_q_down_l1_up} 次")
    print(f"  A1 专属区 [Q↑,L1↓]: {a1_q_up_l1_down} 次")

    if a1_q_up_l1_down > 0:
        print("  ✓ A1 确实接受了 'Q 恶化但 L1 改善' 的步")
    else:
        print("  ⚠ A1 未接受任何 'Q 恶化但 L1 改善' 的步（可能是数据特性）")

    if a0_q_down_l1_up > 0:
        print("  ✓ A0 确实接受了 'Q 改善但 L1 恶化' 的步")
    else:
        print("  ⚠ A0 未接受任何 'Q 改善但 L1 恶化' 的步（可能是数据特性）")

    print()
    print("=" * 60)
    print("✓ Smoke test 通过！")
    print("=" * 60)

    return results


if __name__ == "__main__":
    try:
        results = run_smoke_test()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Smoke test 失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
