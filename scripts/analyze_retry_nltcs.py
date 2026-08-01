"""
分析 nltcs 缩步重试对照实验，包括未参与训练的边缘查询。

除 diagnostics 中的训练 workload loss/L1 外，额外评价：

- 未测量的 3-way 单元格查询（排除 measured_1000query.json 中的三阶查询）
- 全部 4-way 单元格查询（训练 workload 不含四阶）
- 完整 16 位联合分布 TVD 与基本多样性

用法：
    python scripts/analyze_retry_nltcs.py \
        --baseline outputs/retry_baseline_nltcs_YYYY-MM-DD_HHMM \
        --candidate outputs/retry2_nltcs_YYYY-MM-DD_HHMM
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _load_measured_triples(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    measured = set()
    for query in payload["queries"]:
        conditions = query["conditions"]
        if len(conditions) != 3:
            continue
        measured.add(
            tuple(sorted((condition["attribute"], int(condition["value"]))
                         for condition in conditions))
        )
    return measured


def _marginal_error(real_x, synth_x, columns, order, excluded_queries=None):
    """计算指定阶数的全部二值单元格查询误差。"""
    excluded_queries = excluded_queries or set()
    n_records = len(real_x)
    weights = 1 << np.arange(order, dtype=np.int64)
    errors = []

    for indices in itertools.combinations(range(real_x.shape[1]), order):
        idx = np.asarray(indices)
        real_codes = real_x[:, idx].astype(np.int64) @ weights
        synth_codes = synth_x[:, idx].astype(np.int64) @ weights
        real_counts = np.bincount(real_codes, minlength=2 ** order)
        synth_counts = np.bincount(synth_codes, minlength=2 ** order)

        for state in range(2 ** order):
            query_key = tuple(sorted(
                (columns[column_idx], (state >> bit_idx) & 1)
                for bit_idx, column_idx in enumerate(indices)
            ))
            if query_key in excluded_queries:
                continue
            errors.append(abs(int(real_counts[state]) - int(synth_counts[state]))
                          / n_records)

    values = np.asarray(errors, dtype=float)
    return {
        "n_queries": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _joint_and_diversity(real_x, synth_x):
    n_columns = real_x.shape[1]
    weights = 1 << np.arange(n_columns, dtype=np.int64)
    real_codes = real_x.astype(np.int64) @ weights
    synth_codes = synth_x.astype(np.int64) @ weights
    n_states = 2 ** n_columns
    real_counts = np.bincount(real_codes, minlength=n_states)
    synth_counts = np.bincount(synth_codes, minlength=n_states)
    real_probs = real_counts / len(real_x)
    synth_probs = synth_counts / len(synth_x)
    real_top10 = set(np.argsort(real_counts)[-10:])
    synth_top10 = set(np.argsort(synth_counts)[-10:])
    n_unique = int(np.count_nonzero(synth_counts))
    return {
        "joint_tvd": float(0.5 * np.abs(real_probs - synth_probs).sum()),
        "n_unique": n_unique,
        "dup_rate": float(1.0 - n_unique / len(synth_x)),
        "top1_rate": float(synth_counts.max() / len(synth_x)),
        "real_top10_coverage": int(len(real_top10 & synth_top10)),
    }


def _read_runs(root):
    runs = {}
    for diagnostics_path in sorted(Path(root).glob("*-*/diagnostics.json")):
        with diagnostics_path.open(encoding="utf-8") as handle:
            diagnostics = json.load(handle)
        seed = int(diagnostics["params"]["seed"])
        runs[seed] = {
            "diagnostics": diagnostics,
            "synthetic_path": diagnostics_path.parent / "best_synthetic.csv",
        }
    if not runs:
        raise ValueError(f"目录中没有找到完整运行: {root}")
    return runs


def _evaluate_run(run, real_x, columns, measured_triples):
    synthetic = pd.read_csv(run["synthetic_path"])
    synth_x = synthetic[columns].to_numpy()
    diagnostics = run["diagnostics"]
    attempts = np.asarray(diagnostics["proposal_attempts_history"], dtype=int)
    accepted_attempts = np.asarray(
        diagnostics["accepted_attempt_history"], dtype=int
    )
    metrics = {
        "best_loss": float(diagnostics["best_loss"]),
        "training_l1": float(diagnostics["normalized_l1_error"]),
        "elapsed_sec": float(diagnostics["elapsed_sec"]),
        "accept_rate": float(np.mean(accepted_attempts > 0)),
        "retry_accept_rate": float(np.mean(accepted_attempts > 1)),
        "proposals_per_round": float(attempts.mean()),
        "heldout_3way": _marginal_error(
            real_x, synth_x, columns, order=3,
            excluded_queries=measured_triples,
        ),
        "heldout_4way": _marginal_error(
            real_x, synth_x, columns, order=4,
        ),
    }
    metrics.update(_joint_and_diversity(real_x, synth_x))
    return metrics


def _aggregate(run_metrics, metric_path):
    def extract(record):
        value = record
        for key in metric_path:
            value = value[key]
        return value

    values = np.asarray([extract(record) for record in run_metrics], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "values": values.tolist(),
    }


def _paired(candidate, baseline, metric_path):
    def extract(record):
        value = record
        for key in metric_path:
            value = value[key]
        return value

    candidate_values = np.asarray([extract(record) for record in candidate])
    baseline_values = np.asarray([extract(record) for record in baseline])
    if len(candidate_values) > 1:
        t_stat, p_value = stats.ttest_rel(candidate_values, baseline_values)
    else:
        t_stat, p_value = np.nan, np.nan
    return {
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--real", default="data/nltcs/nltcs.csv")
    parser.add_argument(
        "--queries", default="configs/nltcs/measured_1000query.json"
    )
    parser.add_argument("--output")
    args = parser.parse_args()

    real = pd.read_csv(args.real)
    columns = list(real.columns)
    real_x = real.to_numpy()
    measured_triples = _load_measured_triples(args.queries)
    baseline_runs = _read_runs(args.baseline)
    candidate_runs = _read_runs(args.candidate)
    seeds = sorted(set(baseline_runs) & set(candidate_runs))
    if not seeds:
        raise ValueError("两组结果没有共同种子")

    baseline_metrics = [
        _evaluate_run(baseline_runs[seed], real_x, columns, measured_triples)
        for seed in seeds
    ]
    candidate_metrics = [
        _evaluate_run(candidate_runs[seed], real_x, columns, measured_triples)
        for seed in seeds
    ]
    metric_paths = {
        "best_loss": ("best_loss",),
        "training_l1": ("training_l1",),
        "heldout_3way_l1": ("heldout_3way", "mean"),
        "heldout_4way_l1": ("heldout_4way", "mean"),
        "joint_tvd": ("joint_tvd",),
        "elapsed_sec": ("elapsed_sec",),
    }
    summary = {
        "baseline_dir": args.baseline,
        "candidate_dir": args.candidate,
        "seeds": seeds,
        "measured_triple_queries_excluded": len(measured_triples),
        "baseline_runs": baseline_metrics,
        "candidate_runs": candidate_metrics,
        "aggregate": {
            "baseline": {
                name: _aggregate(baseline_metrics, path)
                for name, path in metric_paths.items()
            },
            "candidate": {
                name: _aggregate(candidate_metrics, path)
                for name, path in metric_paths.items()
            },
        },
        "candidate_vs_baseline": {
            name: _paired(candidate_metrics, baseline_metrics, path)
            for name, path in metric_paths.items()
        },
    }

    output = Path(args.output) if args.output else Path(args.candidate) / "comparison.json"
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"配对种子: {seeds}")
    for name in metric_paths:
        base = summary["aggregate"]["baseline"][name]
        cand = summary["aggregate"]["candidate"][name]
        comp = summary["candidate_vs_baseline"][name]
        print(
            f"{name:<18} {base['mean']:.6g} → {cand['mean']:.6g}  "
            f"({comp['change_pct']:+.2f}%, p={comp['paired_p']:.4g}, "
            f"胜/平/负={comp['wins']}/{comp['ties']}/{comp['losses']})"
        )
    print(f"详细结果: {output}")


if __name__ == "__main__":
    main()
