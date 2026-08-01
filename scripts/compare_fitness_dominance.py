"""适应度支配软门控的小表配对筛选与等复制量控制。

比较以下配置：

- baseline：原始均匀记录参与；
- soft gate：支配 pair 使用完整 rho，非支配 pair 的 donor 复制参与率为
  ``rho * exploration_rate``，随机变异仍使用完整 rho；
- matched-copy control：关闭门控、保持 rho 和变异不变，仅按候选实测平均复制缩放
  降低 eta，近似匹配期望 donor 块复制量。

探索率 1 是 baseline 端点，脚本逐种子核对合成 CSV 哈希。该筛选不替代 nltcs
正式多种子实验和离线泛化评价。

用法：
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python \
        scripts/compare_fitness_dominance.py \
        --rounds 1000 --candidate-rate 0.02 \
        --output outputs/fitness_soft_dominance_small/summary.json
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from scipy import stats

from table_diffevo.evolution import run_evolution
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"

N_RECORDS = 300
BASE_RHO = 0.01
BASE_ETA = 0.5


def _rate_name(rate):
    return f"soft_rate_{rate:g}".replace(".", "p")


def _aggregate(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _mean_or_zero(values):
    """空历史只会出现在初始即达标时，此时按零次事件记录为 0。"""
    return float(np.mean(values)) if len(values) else 0.0


def _paired(candidate, baseline, metric):
    candidate_values = np.asarray(
        [row[metric] for row in candidate], dtype=float
    )
    baseline_values = np.asarray(
        [row[metric] for row in baseline], dtype=float
    )
    candidate_mean = float(candidate_values.mean())
    baseline_mean = float(baseline_values.mean())
    if len(candidate_values) < 2 or np.array_equal(
        candidate_values, baseline_values
    ):
        t_stat, p_value = None, None
    else:
        t_stat, p_value = stats.ttest_rel(
            candidate_values, baseline_values
        )
        t_stat = float(t_stat) if np.isfinite(t_stat) else None
        p_value = float(p_value) if np.isfinite(p_value) else None
    return {
        "candidate_mean": candidate_mean,
        "baseline_mean": baseline_mean,
        "change_pct": (
            float((candidate_mean / baseline_mean - 1.0) * 100.0)
            if baseline_mean != 0.0 else None
        ),
        "paired_t": t_stat,
        "paired_p": p_value,
        "wins": int(np.sum(candidate_values < baseline_values)),
        "ties": int(np.sum(candidate_values == baseline_values)),
        "losses": int(np.sum(candidate_values > baseline_values)),
    }


def _run_one(
    target,
    queries,
    schema,
    marginals,
    *,
    seed,
    name,
    rounds,
    device,
    gate,
    exploration_rate,
    eta,
):
    with contextlib.redirect_stdout(io.StringIO()):
        best_s, diagnostics = run_evolution(
            target,
            queries,
            schema,
            n_records=N_RECORDS,
            n_rounds=rounds,
            seed=seed,
            beta=1.0,
            h=0.8,
            rho=BASE_RHO,
            eta=eta,
            mu=0.01,
            device=device,
            eval_method="vectorized",
            batch_size=256,
            init_method="marginal",
            marginals=marginals,
            log_every=rounds + 1,
            distance_mode="geometric",
            lambda_param=0.5,
            alpha_min=2.0,
            alpha_max=10.0,
            delta=0.05,
            winsorize_quantiles=(0.01, 0.99),
            exclude_self=True,
            max_retries=0,
            fitness_dominance_gate=gate,
            fitness_dominance_exploration_rate=exploration_rate,
        )
    loss_history = diagnostics["loss_history"]
    accept_history = diagnostics["accept_history"]
    loss_late_start = 3 * len(loss_history) // 4
    accept_late_start = 3 * len(accept_history) // 4
    result = {
        "name": name,
        "seed": seed,
        "rho": BASE_RHO,
        "eta": eta,
        "fitness_dominance_gate": gate,
        "fitness_dominance_exploration_rate": exploration_rate,
        "best_loss": float(diagnostics["best_loss"]),
        "normalized_l1_error": float(
            diagnostics["normalized_l1_error"]
        ),
        "accept_rate": _mean_or_zero(accept_history),
        "late_accept_rate": _mean_or_zero(
            accept_history[accept_late_start:]
        ),
        "late_loss_improvement": float(
            loss_history[loss_late_start] - diagnostics["best_loss"]
        ),
        "mean_dominance_rate": _mean_or_zero(
            diagnostics["fitness_dominance_rate_history"]
        ),
        "mean_copy_participation_scale": _mean_or_zero(
            diagnostics["fitness_copy_participation_scale_history"]
        ),
        "elapsed_sec": float(diagnostics["elapsed_sec"]),
        "n_unique": int(len(best_s.value_counts())),
        "csv_sha256": hashlib.sha256(
            best_s.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    print(
        f"seed={seed:02d} {name:<22} loss={result['best_loss']:.1f} "
        f"L1={result['normalized_l1_error']:.5f} "
        f"accept={result['accept_rate']:.1%} "
        f"copy_scale={result['mean_copy_participation_scale']:.1%}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument(
        "--rates", nargs="+", type=float, default=[0.0, 0.02, 0.1, 1.0]
    )
    parser.add_argument("--candidate-rate", type=float, default=0.02)
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default="outputs/fitness_soft_dominance_small/summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds 必须为正数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if len(set(args.rates)) != len(args.rates):
        parser.error("--rates 不得重复")
    if any(not 0.0 <= rate <= 1.0 for rate in args.rates):
        parser.error("--rates 必须全部位于 [0, 1]")
    if not 0.0 <= args.candidate_rate <= 1.0:
        parser.error("--candidate-rate 必须位于 [0, 1]")
    if args.candidate_rate not in args.rates:
        parser.error("--candidate-rate 必须包含在 --rates 中")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)

    runs = {"baseline": []}
    for rate in args.rates:
        runs[_rate_name(rate)] = []

    for seed in args.seeds:
        runs["baseline"].append(
            _run_one(
                target, queries, schema, marginals,
                seed=seed, name="baseline", rounds=args.rounds,
                device=args.device, gate=False, exploration_rate=0.0,
                eta=BASE_ETA,
            )
        )
        for rate in args.rates:
            name = _rate_name(rate)
            runs[name].append(
                _run_one(
                    target, queries, schema, marginals,
                    seed=seed, name=name, rounds=args.rounds,
                    device=args.device, gate=True,
                    exploration_rate=rate, eta=BASE_ETA,
                )
            )

    candidate_name = _rate_name(args.candidate_rate)
    mean_copy_scale = float(np.mean([
        row["mean_copy_participation_scale"]
        for row in runs[candidate_name]
    ]))
    matched_eta = BASE_ETA * mean_copy_scale
    runs["matched_copy_control"] = []
    print(
        f"\n候选平均复制参与率缩放={mean_copy_scale:.6f}，"
        f"等复制量控制 eta={matched_eta:.8f}\n",
        flush=True,
    )
    for seed in args.seeds:
        runs["matched_copy_control"].append(
            _run_one(
                target, queries, schema, marginals,
                seed=seed, name="matched_copy_control",
                rounds=args.rounds, device=args.device, gate=False,
                exploration_rate=0.0, eta=matched_eta,
            )
        )

    metrics = [
        "best_loss",
        "normalized_l1_error",
        "accept_rate",
        "late_accept_rate",
        "late_loss_improvement",
        "mean_dominance_rate",
        "mean_copy_participation_scale",
        "elapsed_sec",
        "n_unique",
    ]
    aggregate = {
        name: {
            metric: _aggregate([row[metric] for row in config_runs])
            for metric in metrics
        }
        for name, config_runs in runs.items()
    }
    comparisons = {
        f"{name}_vs_baseline": {
            metric: _paired(config_runs, runs["baseline"], metric)
            for metric in ("best_loss", "normalized_l1_error")
        }
        for name, config_runs in runs.items()
        if name != "baseline"
    }
    comparisons["candidate_vs_matched_copy_control"] = {
        metric: _paired(
            runs[candidate_name], runs["matched_copy_control"], metric
        )
        for metric in ("best_loss", "normalized_l1_error")
    }
    rate_one_name = _rate_name(1.0)
    rate_one_csv_exact = (
        rate_one_name in runs
        and all(
            left["csv_sha256"] == right["csv_sha256"]
            for left, right in zip(runs[rate_one_name], runs["baseline"])
        )
    )

    summary = {
        "experiment": "fitness_soft_dominance_small",
        "scope": "fixed_workload_exact_target_no_noise",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "n_rounds": args.rounds,
        "seeds": args.seeds,
        "rates": args.rates,
        "candidate_rate": args.candidate_rate,
        "base_rho": BASE_RHO,
        "base_eta": BASE_ETA,
        "candidate_mean_copy_participation_scale": mean_copy_scale,
        "matched_eta": matched_eta,
        "rate_one_csv_exact": rate_one_csv_exact,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(
            summary, handle, ensure_ascii=False, indent=2, allow_nan=False
        )

    print("\n===== 小表筛选汇总 =====")
    for name in runs:
        loss = aggregate[name]["best_loss"]
        l1 = aggregate[name]["normalized_l1_error"]
        print(
            f"{name:<22} loss={loss['mean']:.1f}±{loss['std']:.1f} "
            f"L1={l1['mean']:.5f}±{l1['std']:.5f}"
        )
    for name, comparison in comparisons.items():
        loss = comparison["best_loss"]
        change_text = (
            f"{loss['change_pct']:+.2f}%"
            if loss["change_pct"] is not None else "NA"
        )
        p_text = (
            f"{loss['paired_p']:.4g}"
            if loss["paired_p"] is not None else "NA"
        )
        print(
            f"{name}: loss {change_text} "
            f"p={p_text} "
            f"胜/平/负={loss['wins']}/{loss['ties']}/{loss['losses']}"
        )
    print(f"rate=1 CSV 逐种子等价：{rate_one_csv_exact}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
