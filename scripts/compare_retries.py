"""
对比整代提案的缩步重试策略。

默认在 test_300x10 上用推荐的 geometric 配置跑 20 个配对种子：

- baseline: 被拒后不重试
- retry1: 最多重试 1 次，rho 减半
- retry2: 最多重试 2 次，rho 每次减半
- baseline120: 不重试但多跑 20% 轮数，作为近似等墙钟对照

记录 loss/L1、最终拒绝率、重试挽回率、提案评估数和耗时，
并对各重试配置与 baseline 做配对 t 检验。

用法：
    CUDA_VISIBLE_DEVICES=0 python scripts/compare_retries.py
"""
import json
import os

import numpy as np
from scipy import stats

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
OUTPUT_PATH = "outputs/retry_experiment_small/summary.json"

N_RECORDS = 300
N_ROUNDS = 100
SEEDS = list(range(20))

COMMON = {
    "beta": 1.0,
    "h": 0.8,
    "rho": 0.01,
    "eta": 0.5,
    "mu": 0.01,
    "device": "cuda",
    "eval_method": "vectorized",
    "batch_size": 256,
    "init_method": "marginal",
    "distance_mode": "geometric",
    "lambda_param": 0.5,
    "alpha_min": 2.0,
    "alpha_max": 10.0,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
    "exclude_self": True,
}

CONFIGS = [
    {
        "name": "baseline",
        "n_rounds": 100,
        "max_retries": 0,
        "retry_rho_decay": 0.5,
    },
    {
        "name": "retry1",
        "n_rounds": 100,
        "max_retries": 1,
        "retry_rho_decay": 0.5,
    },
    {
        "name": "retry2",
        "n_rounds": 100,
        "max_retries": 2,
        "retry_rho_decay": 0.5,
    },
    {
        # 根据首轮实测，retry2 比 baseline 慢约 20%；加这组用于
        # 回答“同样墙钟时间内，baseline 多跑 20 轮能否追上”。
        "name": "baseline120",
        "n_rounds": 120,
        "max_retries": 0,
        "retry_rho_decay": 0.5,
    },
]


def _aggregate(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _run_one(target, queries, schema, marginals, seed, config):
    n_rounds = config["n_rounds"]
    best_s, diag = run_evolution(
        target,
        queries,
        schema,
        n_records=N_RECORDS,
        n_rounds=n_rounds,
        seed=seed,
        marginals=marginals,
        max_retries=config["max_retries"],
        retry_rho_decay=config["retry_rho_decay"],
        log_every=n_rounds + 1,
        **COMMON,
    )

    attempts = np.asarray(diag["proposal_attempts_history"], dtype=int)
    accepted_attempts = np.asarray(diag["accepted_attempt_history"], dtype=int)
    value_counts = best_s.value_counts()
    return {
        "seed": seed,
        "n_rounds": n_rounds,
        "best_loss": float(diag["best_loss"]),
        "normalized_l1_error": float(diag["normalized_l1_error"]),
        "accept_rate": float(np.mean(accepted_attempts > 0)),
        "retry_rescue_rate": float(np.mean(accepted_attempts > 1)),
        "final_reject_rate": float(np.mean(accepted_attempts == 0)),
        "proposals_per_round": float(attempts.mean()),
        "elapsed_sec": float(diag["elapsed_sec"]),
        "n_unique": int(len(value_counts)),
        "dup_rate": float(1.0 - len(value_counts) / len(best_s)),
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)

    runs = {config["name"]: [] for config in CONFIGS}
    for seed in SEEDS:
        for config in CONFIGS:
            result = _run_one(target, queries, schema, marginals, seed, config)
            runs[config["name"]].append(result)
            print(
                f"seed={seed:02d} {config['name']:<8} "
                f"loss={result['best_loss']:.1f} "
                f"reject={result['final_reject_rate'] * 100:.1f}% "
                f"proposals/round={result['proposals_per_round']:.2f}"
            )

    metrics = [
        "best_loss",
        "normalized_l1_error",
        "accept_rate",
        "retry_rescue_rate",
        "final_reject_rate",
        "proposals_per_round",
        "elapsed_sec",
        "n_unique",
        "dup_rate",
    ]
    aggregate = {
        name: {
            metric: _aggregate([run[metric] for run in config_runs])
            for metric in metrics
        }
        for name, config_runs in runs.items()
    }

    def paired_comparison(candidate_name, reference_name):
        candidate_loss = np.asarray(
            [run["best_loss"] for run in runs[candidate_name]]
        )
        reference_loss = np.asarray(
            [run["best_loss"] for run in runs[reference_name]]
        )
        t_stat, p_value = stats.ttest_rel(candidate_loss, reference_loss)
        return {
            "mean_loss_change_pct": float(
                (candidate_loss.mean() / reference_loss.mean() - 1.0) * 100.0
            ),
            "paired_t": float(t_stat),
            "paired_p": float(p_value),
            "wins": int(np.sum(candidate_loss < reference_loss)),
            "ties": int(np.sum(candidate_loss == reference_loss)),
            "losses": int(np.sum(candidate_loss > reference_loss)),
        }

    comparisons = {
        config["name"]: paired_comparison(config["name"], "baseline")
        for config in CONFIGS[1:]
    }
    time_matched = paired_comparison("retry2", "baseline120")

    summary = {
        "experiment": "proposal_retry_small",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "seeds": SEEDS,
        "common_params": COMMON,
        "configs": CONFIGS,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons_vs_baseline": comparisons,
        "retry2_vs_time_matched_baseline120": time_matched,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n" + "=" * 78)
    print(f"{N_ROUNDS} 轮×{len(SEEDS)} 种子结果")
    print("=" * 78)
    for config in CONFIGS:
        name = config["name"]
        values = aggregate[name]
        print(
            f"{name:<8} loss={values['best_loss']['mean']:.1f}"
            f"±{values['best_loss']['std']:.1f}  "
            f"L1={values['normalized_l1_error']['mean']:.4f}  "
            f"最终拒绝={values['final_reject_rate']['mean'] * 100:.1f}%  "
            f"重试挽回={values['retry_rescue_rate']['mean'] * 100:.1f}%  "
            f"提案/轮={values['proposals_per_round']['mean']:.2f}  "
            f"耗时={values['elapsed_sec']['mean']:.3f}s"
        )
    for name, comparison in comparisons.items():
        print(
            f"{name} vs baseline: loss {comparison['mean_loss_change_pct']:+.2f}%  "
            f"p={comparison['paired_p']:.4g}  "
            f"胜/平/负={comparison['wins']}/{comparison['ties']}/{comparison['losses']}"
        )
    print(
        "retry2 vs baseline120 (等墙钟近似): "
        f"loss {time_matched['mean_loss_change_pct']:+.2f}%  "
        f"p={time_matched['paired_p']:.4g}  "
        f"胜/平/负={time_matched['wins']}/{time_matched['ties']}/"
        f"{time_matched['losses']}"
    )
    print(f"详细结果: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
