"""
距离模式对比实验

对比三种距离项设计（squared / linear / none）的收敛效果。

实验设计：
- 3 种距离模式 × 3 个种子 = 9 次运行
- 固定其他参数（初始化方式、beta/h/rho/eta/mu）
- 对比指标：best_loss、normalized_l1_error、收敛速度

结果保存在：
outputs/distance_mode_experiment_YYYY-MM-DD_HHMM/
    ├── squared/         # 组 A（baseline）
    ├── linear/          # 组 B
    ├── none/            # 组 C
    └── comparison.json  # 汇总对比

运行方式：
    conda run -p ./.conda python scripts/compare_distance_modes.py
"""
import os
import json
from datetime import datetime
from typing import Dict, Any

import numpy as np

from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution
from table_diffevo.io import save_run, save_summary
from table_diffevo.marginals import load_marginals


# ========== 实验配置 ==========
# 数据集配置（两阶段：先小数据试水，再大数据正式跑）
DATASETS = [
    {
        "name": "test_300x10",
        "schema_path": "configs/test_300x10/schema.yaml",
        "query_path": "configs/test_300x10/measured_50query.json",
        "marginals_path": "configs/test_300x10/init_marginals.json",
        "n_records": 300,
        "n_rounds": 100,
    },
    {
        "name": "nltcs",
        "schema_path": "configs/nltcs/schema.yaml",
        "query_path": "configs/nltcs/measured_1000query.json",
        "marginals_path": "configs/nltcs/init_marginals.json",
        "n_records": 16181,
        "n_rounds": 1500,
    },
]

# 选择运行哪个数据集（0=test_300x10, 1=nltcs）
DATASET_INDEX = 1  # nltcs 大数据

# 实验组配置
DISTANCE_MODES = ['squared', 'linear']  # 不跑 none（已确认最差，无意义）
SEEDS = [0, 1, 2]  # 3 个种子

# 固定参数（保证对照公平）
DEVICE = 'cuda'
EVAL_METHOD = 'vectorized'
BATCH_SIZE = 256
INIT_METHOD = 'marginal'
BETA = 1.0
H = 0.8
RHO = 0.01
ETA = 0.5
MU = 0.01
LOG_EVERY = 50
# ==============================


def _aggregate(values):
    """一组标量的均值/标准差/最小/最大。"""
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def run_one_group(
    distance_mode: str,
    dataset_config: Dict[str, Any],
    parent_dir: str,
) -> Dict[str, Any]:
    """
    跑一组实验（一种距离模式 × 多个种子）。

    Returns
    -------
    dict
        该组的汇总结果（per_seed + aggregate）
    """
    print(f"\n{'='*60}")
    print(f"组别: {distance_mode.upper()}")
    print(f"{'='*60}")

    # 加载数据
    schema = load_schema(dataset_config["schema_path"])
    queries = load_queries(dataset_config["query_path"])
    target = np.array([q["result"] for q in queries])
    marginals = load_marginals(dataset_config["marginals_path"])

    # 创建该组的子文件夹
    group_dir = os.path.join(parent_dir, distance_mode)
    os.makedirs(group_dir, exist_ok=True)

    per_seed = []
    for i, seed in enumerate(SEEDS):
        print(f"\n--- 种子 {seed} ({i + 1}/{len(SEEDS)}) ---")

        best_S, diagnostics = run_evolution(
            target, queries, schema,
            n_records=dataset_config["n_records"],
            n_rounds=dataset_config["n_rounds"],
            seed=seed,
            beta=BETA, h=H, rho=RHO, eta=ETA, mu=MU,
            device=DEVICE,
            eval_method=EVAL_METHOD,
            batch_size=BATCH_SIZE,
            init_method=INIT_METHOD,
            marginals=marginals,
            log_every=LOG_EVERY,
            distance_mode=distance_mode,
        )

        # 保存到 group_dir/{顺序}-{种子}/
        sub_name = f"{i}-{seed}"
        run_dir = save_run(best_S, diagnostics,
                          run_dir=os.path.join(group_dir, sub_name))

        lh = diagnostics["loss_history"]
        print(f"  初始 loss : {lh[0]:.1f}  →  最优 loss : {diagnostics['best_loss']:.1f}")
        print(f"  平均归一化L1: {diagnostics['normalized_l1_error']:.4f}"
              f" | 中位: {diagnostics['normalized_l1_median']:.4f}"
              f" | P90: {diagnostics['normalized_l1_p90']:.4f}")
        print(f"  跑了轮数  : {diagnostics['rounds_run']}"
              f"（提前停止={diagnostics['stopped_early']}）"
              f" | 耗时: {diagnostics['elapsed_sec']:.1f}s")

        per_seed.append({
            "seed": seed,
            "run_dir": sub_name,
            "best_loss": diagnostics["best_loss"],
            "normalized_l1_error": diagnostics["normalized_l1_error"],
            "elapsed_sec": diagnostics["elapsed_sec"],
        })

    # 该组的汇总统计
    group_summary = {
        "distance_mode": distance_mode,
        "per_seed": per_seed,
        "aggregate": {
            "best_loss": _aggregate([s["best_loss"] for s in per_seed]),
            "normalized_l1_error": _aggregate(
                [s["normalized_l1_error"] for s in per_seed]
            ),
            "elapsed_sec": _aggregate([s["elapsed_sec"] for s in per_seed]),
        },
    }

    # 保存该组的 summary.json
    save_summary(group_dir, {
        "distance_mode": distance_mode,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": group_summary["aggregate"],
    })

    # 打印该组汇总
    bl = group_summary["aggregate"]["best_loss"]
    nl = group_summary["aggregate"]["normalized_l1_error"]
    print(f"\n--- 组别 {distance_mode.upper()} 汇总（{len(SEEDS)} 个种子）---")
    print(f"  最优 loss    : 均值 {bl['mean']:.3e} ± {bl['std']:.2e}"
          f"  (min {bl['min']:.3e}, max {bl['max']:.3e})")
    print(f"  平均归一化L1 : 均值 {nl['mean']:.4f} ± {nl['std']:.4f}"
          f"  (min {nl['min']:.4f}, max {nl['max']:.4f})")

    return group_summary


def main():
    dataset = DATASETS[DATASET_INDEX]
    print(f"数据集: {dataset['name']}")
    print(f"记录数: {dataset['n_records']}")
    print(f"轮数: {dataset['n_rounds']}")
    print(f"种子: {SEEDS}")
    print(f"距离模式: {DISTANCE_MODES}")

    # 创建实验父文件夹
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    parent_dir = os.path.join("outputs", f"distance_mode_experiment_{stamp}")
    os.makedirs(parent_dir, exist_ok=True)
    print(f"\n结果将保存到: {parent_dir}/")

    # 跑三组实验
    all_groups = {}
    for distance_mode in DISTANCE_MODES:
        group_summary = run_one_group(distance_mode, dataset, parent_dir)
        all_groups[distance_mode] = group_summary

    # 汇总对比
    comparison = {
        "experiment": "distance_mode_comparison",
        "dataset": dataset["name"],
        "timestamp": stamp,
        "common_params": {
            "n_records": dataset["n_records"],
            "n_rounds": dataset["n_rounds"],
            "seeds": list(SEEDS),
            "init_method": INIT_METHOD,
            "beta": BETA,
            "h": H,
            "rho": RHO,
            "eta": ETA,
            "mu": MU,
            "device": DEVICE,
            "eval_method": EVAL_METHOD,
            "batch_size": BATCH_SIZE,
        },
        "groups": {
            mode: {
                "distance_mode": mode,
                "best_loss": summary["aggregate"]["best_loss"],
                "normalized_l1_error": summary["aggregate"]["normalized_l1_error"],
            }
            for mode, summary in all_groups.items()
        },
    }

    # 保存总对比文件
    comparison_path = os.path.join(parent_dir, "comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"实验完成！总对比")
    print(f"{'='*60}")
    for mode in DISTANCE_MODES:
        bl = all_groups[mode]["aggregate"]["best_loss"]
        nl = all_groups[mode]["aggregate"]["normalized_l1_error"]
        print(f"\n{mode.upper()}:")
        print(f"  最优 loss    : {bl['mean']:.3e} ± {bl['std']:.2e}")
        print(f"  平均归一化L1 : {nl['mean']:.4f} ± {nl['std']:.4f}")

    print(f"\n完整结果已保存到: {parent_dir}/")
    print(f"查看对比: cat {comparison_path}")


if __name__ == "__main__":
    main()
