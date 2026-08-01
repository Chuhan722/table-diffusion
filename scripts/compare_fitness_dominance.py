"""适应度支配门控的小表配对筛选与等参与量控制。

第一阶段在 test_300x10 上比较：

- baseline：原始均匀记录参与；
- dominance_gate：rho 骰子不变，但仅允许 donor fitness 严格高于 recipient 的行参与；
- matched_rho_control：关闭门控，把 rho 调成候选实测平均支配率乘以原 rho，近似
  匹配候选的期望参与量，用于排除“只是更新行数更少”的解释。

这是便宜的机制筛选，不替代 nltcs 的配对多种子正式实验和离线泛化评价。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/compare_fitness_dominance.py
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
OUTPUT_PATH = "outputs/fitness_dominance_small/summary.json"

N_RECORDS = 300
N_ROUNDS = 100
SEEDS = list(range(20))
BASE_RHO = 0.01

COMMON = {
    "beta": 1.0,
    "h": 0.8,
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
    "max_retries": 0,
}


def _aggregate(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _run_one(target, queries, schema, marginals, seed, name, rho, gate):
    best_s, diagnostics = run_evolution(
        target,
        queries,
        schema,
        n_records=N_RECORDS,
        n_rounds=N_ROUNDS,
        seed=seed,
        rho=rho,
        marginals=marginals,
        fitness_dominance_gate=gate,
        log_every=N_ROUNDS + 1,
        **COMMON,
    )
    dominance_rates = np.asarray(
        diagnostics["fitness_dominance_rate_history"], dtype=float
    )
    accept_history = np.asarray(diagnostics["accept_history"], dtype=bool)
    value_counts = best_s.value_counts()
    result = {
        "name": name,
        "seed": seed,
        "rho": rho,
        "fitness_dominance_gate": gate,
        "best_loss": float(diagnostics["best_loss"]),
        "normalized_l1_error": float(diagnostics["normalized_l1_error"]),
        "accept_rate": float(accept_history.mean()),
        "mean_dominance_rate": float(dominance_rates.mean()),
        "elapsed_sec": float(diagnostics["elapsed_sec"]),
        "n_unique": int(len(value_counts)),
        "dup_rate": float(1.0 - len(value_counts) / len(best_s)),
    }
    print(
        f"seed={seed:02d} {name:<19} loss={result['best_loss']:.1f} "
        f"L1={result['normalized_l1_error']:.5f} "
        f"accept={result['accept_rate'] * 100:.1f}% "
        f"dominance={result['mean_dominance_rate'] * 100:.1f}%"
    )
    return result


def _paired(candidate, baseline, metric):
    candidate_values = np.asarray([row[metric] for row in candidate], dtype=float)
    baseline_values = np.asarray([row[metric] for row in baseline], dtype=float)
    t_stat, p_value = stats.ttest_rel(candidate_values, baseline_values)
    return {
        "candidate_mean": float(candidate_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "change_pct": float(
            (candidate_values.mean() / baseline_values.mean() - 1.0) * 100.0
        ),
        "paired_t": float(t_stat),
        "paired_p": float(p_value),
        "wins": int(np.sum(candidate_values < baseline_values)),
        "ties": int(np.sum(candidate_values == baseline_values)),
        "losses": int(np.sum(candidate_values > baseline_values)),
    }


def main():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)

    runs = {"baseline": [], "dominance_gate": [], "matched_rho_control": []}
    for seed in SEEDS:
        runs["baseline"].append(
            _run_one(
                target, queries, schema, marginals, seed,
                name="baseline", rho=BASE_RHO, gate=False,
            )
        )
        runs["dominance_gate"].append(
            _run_one(
                target, queries, schema, marginals, seed,
                name="dominance_gate", rho=BASE_RHO, gate=True,
            )
        )

    mean_gate_rate = float(np.mean([
        row["mean_dominance_rate"] for row in runs["dominance_gate"]
    ]))
    matched_rho = BASE_RHO * mean_gate_rate
    print(
        f"\n候选平均支配率={mean_gate_rate:.6f}，"
        f"等期望参与量控制 rho={matched_rho:.8f}\n"
    )
    for seed in SEEDS:
        runs["matched_rho_control"].append(
            _run_one(
                target, queries, schema, marginals, seed,
                name="matched_rho_control", rho=matched_rho, gate=False,
            )
        )

    metrics = [
        "best_loss",
        "normalized_l1_error",
        "accept_rate",
        "mean_dominance_rate",
        "elapsed_sec",
        "n_unique",
        "dup_rate",
    ]
    aggregate = {
        name: {
            metric: _aggregate([row[metric] for row in config_runs])
            for metric in metrics
        }
        for name, config_runs in runs.items()
    }
    comparisons = {
        "dominance_gate_vs_baseline": {
            metric: _paired(
                runs["dominance_gate"], runs["baseline"], metric
            )
            for metric in ("best_loss", "normalized_l1_error")
        },
        "dominance_gate_vs_matched_rho_control": {
            metric: _paired(
                runs["dominance_gate"], runs["matched_rho_control"], metric
            )
            for metric in ("best_loss", "normalized_l1_error")
        },
    }
    summary = {
        "experiment": "fitness_dominance_small_screen",
        "scope": "fixed_workload_exact_target_no_noise",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "n_rounds": N_ROUNDS,
        "seeds": SEEDS,
        "common_params": COMMON,
        "base_rho": BASE_RHO,
        "mean_gate_rate": mean_gate_rate,
        "matched_rho": matched_rho,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("\n===== 小表筛选汇总 =====")
    for name in runs:
        loss = aggregate[name]["best_loss"]
        l1 = aggregate[name]["normalized_l1_error"]
        print(
            f"{name:<19} loss={loss['mean']:.1f}±{loss['std']:.1f} "
            f"L1={l1['mean']:.5f}±{l1['std']:.5f}"
        )
    for name, comparison in comparisons.items():
        loss = comparison["best_loss"]
        print(
            f"{name}: loss {loss['change_pct']:+.2f}% "
            f"p={loss['paired_p']:.4g} "
            f"胜/平/负={loss['wins']}/{loss['ties']}/{loss['losses']}"
        )
    print(f"详细结果: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
