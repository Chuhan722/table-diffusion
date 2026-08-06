#!/usr/bin/env python3
"""
test_300x10 正式实验：A0 vs A1 接受规则对照
配对对照设计，固定 ρ=0.01，与 nltcs_formal_acceptance 协议一致
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import json
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.experiment_logger import ExperimentLogger

# 数据集配置
DATASET = "test_300x10"
CONFIG_DIR = Path("configs") / DATASET
SEEDS = [42, 43, 44, 45, 46]
N_ROUNDS = 500

# 固定参数（与 nltcs 实验一致）
RHO = 0.01
W = 20  # probe 窗口大小

# 接受规则配置（eps_L1 与 nltcs 实验一致）
RULES = {
    "A0": {"rule_type": "A0", "eps_L1": None},
    "A1": {"rule_type": "A1", "eps_L1": 0.02}
}


def load_dataset():
    """加载数据集"""
    schema = load_schema(CONFIG_DIR / "schema.yaml")
    queries = load_queries(CONFIG_DIR / "measured_50query.json")

    with open(CONFIG_DIR / "measured_50query.json") as f:
        query_data = json.load(f)

    n_records = query_data["record_count"]
    target = np.array([q["result"] for q in query_data["queries"]], dtype=float)
    marginals = load_marginals(str(CONFIG_DIR / "init_marginals.json"))

    return schema, queries, target, n_records, marginals


def run_single_experiment(
    rule_type: str,
    eps_L1: float,
    seed: int,
    rule_name: str,
    output_dir: Path,
    n_rounds: int = 500,
    rho: float = 0.01,
    W: int = 20
):
    """运行单次实验"""
    logger = ExperimentLogger(output_dir=output_dir / f"seed_{seed}_{rule_name}")

    print(f"\n{'='*60}")
    print(f"运行: {rule_name.upper()} | 种子 {seed}")
    print(f"{'='*60}")

    schema, queries, target, n_records, marginals = load_dataset()

    best_S, diagnostics = run_evolution(
        schema=schema,
        queries=queries,
        target=target,
        n_records=n_records,
        n_rounds=n_rounds,
        seed=seed,

        # 接受规则参数
        acceptance_rule=rule_type,
        eps_L1=eps_L1 if rule_type == 'A1' else 1e-5,

        # 演化参数（与 nltcs 实验一致）
        distance_mode='geometric',
        alpha_min=2.0,
        alpha_max=10.0,
        lambda_param=0.5,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        exclude_self=True,
        marginals=marginals,
        device='cuda',
        rho=rho,
        beta=1.0,
        h=0.8,
        eta=0.5,
        mu=0.01
    )

    summary = {
        'seed': seed,
        'rule': rule_name,
        'rule_type': rule_type,
        'n_rounds': n_rounds,
        'rho': rho,
        'W': W,
        'eps_L1': eps_L1 if rule_type == 'A1' else None,
        'final_loss': float(diagnostics['loss_history'][-1]),
        'normalized_l1': float(diagnostics['normalized_l1_error']),
        'squared_loss': float(diagnostics['loss_history'][-1]),
        'accept_count': int(sum(diagnostics.get('accept_history', []))),
        'accept_rate': float(sum(diagnostics.get('accept_history', [])) / n_rounds)
    }

    summary_path = logger.output_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n结果:")
    print(f"  最终损失: {summary['final_loss']:.2f}")
    print(f"  归一化L1: {summary['normalized_l1']:.6f}")
    print(f"  接受率: {summary['accept_rate']:.1%}")

    return summary


def main():
    print("="*60)
    print("test_300x10 正式实验: A0 vs A1 接受规则")
    print(f"种子: {SEEDS}")
    print(f"轮数: {N_ROUNDS}")
    print(f"固定参数: ρ={RHO}, W={W}, eps_L1(A1)=0.02")
    print("="*60)

    output_base = Path("results") / "small_formal_acceptance"
    output_base.mkdir(parents=True, exist_ok=True)

    all_results = []

    for seed in SEEDS:
        for rule_name, rule_config in RULES.items():
            try:
                result = run_single_experiment(
                    rule_type=rule_config["rule_type"],
                    eps_L1=rule_config["eps_L1"],
                    seed=seed,
                    rule_name=rule_name,
                    output_dir=output_base,
                    n_rounds=N_ROUNDS,
                    rho=RHO,
                    W=W
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n❌ 实验失败 ({rule_name}, seed={seed}): {e}")
                import traceback
                traceback.print_exc()

    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_base / "all_results.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ 所有结果已保存到: {summary_path}")

        print("\n" + "="*60)
        print("分组统计（按规则）:")
        print("="*60)
        for rule in ["A0", "A1"]:
            rule_data = summary_df[summary_df['rule'] == rule]
            if len(rule_data) > 0:
                print(f"\n{rule}:")
                print(f"  归一化L1: {rule_data['normalized_l1'].mean():.6f} ± {rule_data['normalized_l1'].std():.6f}")
                print(f"  接受率:   {rule_data['accept_rate'].mean():.1%} ± {rule_data['accept_rate'].std():.1%}")
                print(f"  最终损失: {rule_data['final_loss'].mean():.2f} ± {rule_data['final_loss'].std():.2f}")


if __name__ == "__main__":
    main()
