"""小表配对比较残差驱动扩散核，主看整代接受前的原始提案。

本脚本固定仓库原始无噪声配置，只改变块复制概率对实际局部方向量的连续倾斜
强度。baseline 关闭新算子；strength=0 启用算子但必须精确退化到 baseline。
真实训练表和测试表均不在生成或本脚本评价中读取。
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


def _strength_name(strength):
    return f"strength_{strength:g}".replace(".", "p")


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
    return float(np.mean(values)) if len(values) else 0.0


def _first_attempts(nested):
    return [float(attempts[0]) for attempts in nested if attempts]


def _paired(candidate, baseline, metric, higher_is_better=False):
    candidate_values = np.asarray(
        [row[metric] for row in candidate], dtype=float
    )
    baseline_values = np.asarray(
        [row[metric] for row in baseline], dtype=float
    )
    differences = candidate_values - baseline_values
    if len(differences) < 2 or np.all(differences == 0.0):
        t_stat, p_value = None, None
    else:
        t_stat, p_value = stats.ttest_rel(candidate_values, baseline_values)
        t_stat = float(t_stat) if np.isfinite(t_stat) else None
        p_value = float(p_value) if np.isfinite(p_value) else None
    better = differences > 0.0 if higher_is_better else differences < 0.0
    worse = differences < 0.0 if higher_is_better else differences > 0.0
    return {
        "candidate_mean": float(candidate_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "mean_difference": float(differences.mean()),
        "paired_t": t_stat,
        "paired_p": p_value,
        "wins": int(np.sum(better)),
        "ties": int(np.sum(differences == 0.0)),
        "losses": int(np.sum(worse)),
    }


def _run_one(
    target,
    queries,
    schema,
    marginals,
    *,
    seed,
    rounds,
    device,
    enabled,
    strength,
    normalization,
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
            rho=0.01,
            eta=0.5,
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
            residual_directed_diffusion=enabled,
            diffusion_direction_strength=strength,
            diffusion_direction_normalization=normalization,
        )

    gains = _first_attempts(diagnostics["raw_proposal_gain_history"])
    linear = _first_attempts(
        diagnostics["raw_proposal_linear_gain_history"]
    )
    quadratic = _first_attempts(
        diagnostics["raw_proposal_quadratic_penalty_history"]
    )
    direction_means = [
        value for value in diagnostics["copy_direction_mean_history"]
        if value is not None
    ]
    direction_positive = [
        value
        for value in diagnostics["copy_direction_positive_rate_history"]
        if value is not None
    ]
    direction_negative = [
        value
        for value in diagnostics["copy_direction_negative_rate_history"]
        if value is not None
    ]
    negative_copy_probability = [
        value
        for value in diagnostics[
            "negative_direction_copy_probability_history"
        ]
        if value is not None
    ]
    positive_copy_probability = [
        value
        for value in diagnostics[
            "positive_direction_copy_probability_history"
        ]
        if value is not None
    ]
    copy_probability_entropy = [
        value
        for value in diagnostics["copy_probability_entropy_history"]
        if value is not None
    ]
    result = {
        "seed": int(seed),
        "enabled": bool(enabled),
        "strength": float(strength),
        "best_loss": float(diagnostics["best_loss"]),
        "normalized_l1_error": float(
            diagnostics["normalized_l1_error"]
        ),
        "accept_rate": _mean_or_zero(diagnostics["accept_history"]),
        "raw_proposal_gain_mean": _mean_or_zero(gains),
        "raw_proposal_gain_median": (
            float(np.median(gains)) if gains else 0.0
        ),
        "raw_proposal_positive_rate": _mean_or_zero(
            np.asarray(gains) > 0.0
        ),
        "raw_proposal_linear_gain_mean": _mean_or_zero(linear),
        "raw_proposal_quadratic_penalty_mean": _mean_or_zero(quadratic),
        "copy_direction_mean": _mean_or_zero(direction_means),
        "copy_direction_positive_rate": _mean_or_zero(direction_positive),
        "copy_direction_negative_rate": _mean_or_zero(direction_negative),
        "negative_direction_copy_probability": _mean_or_zero(
            negative_copy_probability
        ),
        "positive_direction_copy_probability": _mean_or_zero(
            positive_copy_probability
        ),
        "copy_probability_entropy": _mean_or_zero(
            copy_probability_entropy
        ),
        "elapsed_sec": float(diagnostics["elapsed_sec"]),
        "direction_evaluation_elapsed_sec": float(
            diagnostics["direction_evaluation_elapsed_sec"]
        ),
        "direction_reference_scale": (
            float(diagnostics["direction_reference_scale"])
            if diagnostics["direction_reference_scale"] is not None else 0.0
        ),
        "n_unique": int(len(best_s.value_counts())),
        "csv_sha256": hashlib.sha256(
            best_s.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    label = "baseline" if not enabled else _strength_name(strength)
    print(
        f"seed={seed:02d} {label:<14} loss={result['best_loss']:.1f} "
        f"raw_gain={result['raw_proposal_gain_mean']:+.1f} "
        f"raw_pos={result['raw_proposal_positive_rate']:.1%} "
        f"accept={result['accept_rate']:.1%}",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=300)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=[0.0, 1.0, 5.0, 20.0]
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "initial_rms"],
        default="initial_rms",
    )
    parser.add_argument(
        "--output",
        default="outputs/residual_directed_diffusion_small/summary.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds 必须为正数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if len(set(args.strengths)) != len(args.strengths):
        parser.error("--strengths 不得重复")
    if any(not np.isfinite(value) or value < 0.0 for value in args.strengths):
        parser.error("--strengths 必须全部为非负有限数值")
    if 0.0 not in args.strengths:
        parser.error("--strengths 必须包含 0，用于端点等价检查")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)

    runs = {"baseline": []}
    for strength in args.strengths:
        runs[_strength_name(strength)] = []

    for seed in args.seeds:
        runs["baseline"].append(
            _run_one(
                target,
                queries,
                schema,
                marginals,
                seed=seed,
                rounds=args.rounds,
                device=args.device,
                enabled=False,
                strength=0.0,
                normalization=args.normalization,
            )
        )
        for strength in args.strengths:
            runs[_strength_name(strength)].append(
                _run_one(
                    target,
                    queries,
                    schema,
                    marginals,
                    seed=seed,
                    rounds=args.rounds,
                    device=args.device,
                    enabled=True,
                    strength=strength,
                    normalization=args.normalization,
                )
            )

    metrics = (
        "best_loss",
        "normalized_l1_error",
        "accept_rate",
        "raw_proposal_gain_mean",
        "raw_proposal_gain_median",
        "raw_proposal_positive_rate",
        "raw_proposal_linear_gain_mean",
        "raw_proposal_quadratic_penalty_mean",
        "copy_direction_mean",
        "copy_direction_positive_rate",
        "copy_direction_negative_rate",
        "negative_direction_copy_probability",
        "positive_direction_copy_probability",
        "copy_probability_entropy",
        "elapsed_sec",
        "direction_evaluation_elapsed_sec",
        "direction_reference_scale",
        "n_unique",
    )
    aggregate = {
        name: {
            metric: _aggregate([row[metric] for row in config_runs])
            for metric in metrics
        }
        for name, config_runs in runs.items()
    }
    comparisons = {}
    for name, config_runs in runs.items():
        if name == "baseline":
            continue
        comparisons[f"{name}_vs_baseline"] = {
            "best_loss": _paired(
                config_runs, runs["baseline"], "best_loss"
            ),
            "raw_proposal_gain_mean": _paired(
                config_runs,
                runs["baseline"],
                "raw_proposal_gain_mean",
                higher_is_better=True,
            ),
            "raw_proposal_positive_rate": _paired(
                config_runs,
                runs["baseline"],
                "raw_proposal_positive_rate",
                higher_is_better=True,
            ),
        }

    endpoint_exact = all(
        candidate["csv_sha256"] == baseline["csv_sha256"]
        for candidate, baseline in zip(runs["strength_0"], runs["baseline"])
    )
    summary = {
        "experiment": "residual_directed_diffusion_small",
        "scope": "fixed_workload_exact_target_no_noise",
        "primary_evidence": "raw_proposal_before_generation_acceptance",
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "n_rounds": args.rounds,
        "seeds": args.seeds,
        "strengths": args.strengths,
        "device": args.device,
        "normalization": args.normalization,
        "strength_zero_csv_exact": endpoint_exact,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n===== 原始提案主指标 =====")
    for name in runs:
        loss = aggregate[name]["best_loss"]
        raw = aggregate[name]["raw_proposal_gain_mean"]
        positive = aggregate[name]["raw_proposal_positive_rate"]
        print(
            f"{name:<14} loss={loss['mean']:.1f}±{loss['std']:.1f} "
            f"raw_gain={raw['mean']:+.1f}±{raw['std']:.1f} "
            f"raw_pos={positive['mean']:.1%}"
        )
    print(f"strength=0 CSV 逐种子等价：{endpoint_exact}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
