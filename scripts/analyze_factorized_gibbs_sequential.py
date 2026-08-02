"""校验并汇总因子 Gibbs 无接受动力学的首批与顺序扩样输出。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from scipy import stats


def _load(path):
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return data, digest


def _candidate_name(data):
    return f"gibbs_{data['candidate_sweeps']}_sweeps"


def _derive_metrics(row, source_name, seed, variant):
    losses = np.asarray(row["loss_history"], dtype=float)
    if losses.shape != (1000,) or not np.all(np.isfinite(losses)):
        raise ValueError(
            f"{source_name} 的 seed={seed}、{variant} loss_history "
            "必须包含 1000 个有限值"
        )
    metrics = {
        "final_loss": float(row["final_loss"]),
        "best_loss_diagnostic_only": float(
            row["best_loss_diagnostic_only"]
        ),
        "mean_trajectory_loss": float(losses.mean()),
        "late_100_mean_loss": float(losses[-100:].mean()),
        "late_250_mean_loss": float(losses[-250:].mean()),
        "mean_raw_gain": float(row["mean_raw_gain"]),
        "positive_gain_rate": float(row["positive_gain_rate"]),
        "mean_positive_gain": float(row["mean_positive_gain"]),
        "mean_negative_gain": float(row["mean_negative_gain"]),
        "mean_changed_cells": float(row["mean_changed_cells"]),
        "final_unique_states": float(row["final_unique_states"]),
        "elapsed_sec": float(row["elapsed_sec"]),
        "direction_elapsed_sec": float(row["direction_elapsed_sec"]),
        "factor_build_elapsed_sec": float(
            row["factor_build_elapsed_sec"]
        ),
        "gibbs_sample_elapsed_sec": float(
            row["gibbs_sample_elapsed_sec"]
        ),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(
            f"{source_name} 的 seed={seed}、{variant} 包含非有限指标"
        )
    return metrics


def _validate_and_extract(data, source_name):
    required = {
        "experiment": "factorized_gibbs_unfiltered_dynamics",
        "primary_endpoint": "final_current_loss_not_best_loss",
        "dataset": "test_300x10",
        "n_rounds": 1000,
        "temperature": 2.0,
        "candidate_sweeps": 8,
        "rho": 0.01,
        "eta": 0.5,
        "mu": 0.01,
        "device": "cuda",
    }
    for key, expected in required.items():
        if data.get(key) != expected:
            raise ValueError(
                f"{source_name} 的 {key}={data.get(key)!r}，期望 {expected!r}"
            )
    if not data.get("primary_rng_aligned_all_seeds"):
        raise ValueError(f"{source_name} 的主 RNG 未全部对齐")

    seeds = data.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != len(set(seeds)):
        raise ValueError(f"{source_name} 的 seeds 必须是无重复列表")
    candidate_name = _candidate_name(data)
    baseline_rows = data.get("runs", {}).get("independent", [])
    candidate_rows = data.get("runs", {}).get(candidate_name, [])
    if len(seeds) != len(baseline_rows) or len(seeds) != len(candidate_rows):
        raise ValueError(f"{source_name} 的种子与运行行数不一致")
    extracted = []
    for seed, baseline, candidate in zip(
        seeds, baseline_rows, candidate_rows
    ):
        if baseline["seed"] != seed or candidate["seed"] != seed:
            raise ValueError(f"{source_name} 的运行顺序与 seeds 不一致")
        if baseline["rounds_run"] != 1000 or candidate["rounds_run"] != 1000:
            raise ValueError(f"{source_name} 存在未跑满 1000 轮的运行")
        if (
            baseline["primary_rng_state_sha256"]
            != candidate["primary_rng_state_sha256"]
        ):
            raise ValueError(f"{source_name} 的 seed={seed} 主 RNG 未对齐")
        if baseline["initial_loss"] != candidate["initial_loss"]:
            raise ValueError(f"{source_name} 的 seed={seed} 初始 loss 不一致")
        if (
            baseline["direction_reference_scale"]
            != candidate["direction_reference_scale"]
        ):
            raise ValueError(
                f"{source_name} 的 seed={seed} 首轮方向尺度不一致"
            )
        extracted.append({
            "seed": int(seed),
            "baseline": _derive_metrics(
                baseline, source_name, seed, "independent"
            ),
            "candidate": _derive_metrics(
                candidate, source_name, seed, candidate_name
            ),
        })
    return extracted


def _paired_summary(rows, metric, lower_is_better):
    baseline = np.asarray([
        row["baseline"][metric] for row in rows
    ], dtype=float)
    candidate = np.asarray([
        row["candidate"][metric] for row in rows
    ], dtype=float)
    difference = candidate - baseline
    n = len(difference)
    difference_std = float(difference.std(ddof=1)) if n > 1 else 0.0
    if n > 1 and not np.all(difference == 0.0):
        statistic, p_value = stats.ttest_rel(candidate, baseline)
        critical = float(stats.t.ppf(0.975, n - 1))
        half_width = critical * difference_std / np.sqrt(n)
        paired_t = float(statistic)
        paired_p = float(p_value)
    else:
        half_width = 0.0
        paired_t = None
        paired_p = None
    mean_difference = float(difference.mean())
    better = difference < 0.0 if lower_is_better else difference > 0.0
    worse = difference > 0.0 if lower_is_better else difference < 0.0
    baseline_mean = float(baseline.mean())
    candidate_mean = float(candidate.mean())
    return {
        "n": n,
        "baseline_mean": baseline_mean,
        "baseline_std": (
            float(baseline.std(ddof=1)) if n > 1 else 0.0
        ),
        "candidate_mean": candidate_mean,
        "candidate_std": (
            float(candidate.std(ddof=1)) if n > 1 else 0.0
        ),
        "mean_difference": mean_difference,
        "difference_std": difference_std,
        "difference_95pct_t_interval": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
        "relative_aggregate_change_pct": (
            (candidate_mean / baseline_mean - 1.0) * 100.0
            if baseline_mean != 0.0 else None
        ),
        "paired_t": paired_t,
        "paired_p_descriptive": paired_p,
        "wins": int(np.sum(better)),
        "ties": int(np.sum(difference == 0.0)),
        "losses": int(np.sum(worse)),
        "differences": difference.tolist(),
    }


def _summarize(rows):
    directions = {
        "final_loss": True,
        "best_loss_diagnostic_only": True,
        "mean_trajectory_loss": True,
        "late_100_mean_loss": True,
        "late_250_mean_loss": True,
        "mean_raw_gain": False,
        "positive_gain_rate": False,
        "mean_positive_gain": False,
        "mean_negative_gain": False,
        "mean_changed_cells": False,
        "final_unique_states": False,
        "elapsed_sec": True,
        "direction_elapsed_sec": True,
        "factor_build_elapsed_sec": True,
        "gibbs_sample_elapsed_sec": True,
    }
    return {
        metric: _paired_summary(rows, metric, lower_is_better)
        for metric, lower_is_better in directions.items()
    }


def _git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{args.output}")

    initial_data, initial_sha256 = _load(args.initial)
    extension_data, extension_sha256 = _load(args.extension)
    initial_rows = _validate_and_extract(initial_data, "initial")
    extension_rows = _validate_and_extract(extension_data, "extension")
    initial_seeds = {row["seed"] for row in initial_rows}
    extension_seeds = {row["seed"] for row in extension_rows}
    overlap = sorted(initial_seeds & extension_seeds)
    if overlap:
        raise ValueError(f"首批与追加种子不得重叠：{overlap}")
    if initial_seeds != set(range(10)):
        raise ValueError("首批种子必须恰好是 0..9")
    if extension_seeds != set(range(10, 30)):
        raise ValueError("追加种子必须恰好是 10..29")
    combined_rows = sorted(
        initial_rows + extension_rows,
        key=lambda row: row["seed"],
    )

    summary = {
        "analysis": "factorized_gibbs_sequential_dynamics_summary",
        "inference_boundary": (
            "extension was chosen after observing the initial cohort; "
            "combined p-values and intervals are descriptive"
        ),
        "git_commit": _git_commit(),
        "sources": {
            "initial": {
                "path": str(args.initial),
                "sha256": initial_sha256,
                "git_commit": initial_data["git_commit"],
                "seeds": initial_data["seeds"],
            },
            "extension": {
                "path": str(args.extension),
                "sha256": extension_sha256,
                "git_commit": extension_data["git_commit"],
                "seeds": extension_data["seeds"],
            },
        },
        "cohorts": {
            "initial_0_9": _summarize(initial_rows),
            "extension_10_29": _summarize(extension_rows),
            "combined_sequential_0_29": _summarize(combined_rows),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    for cohort_name, cohort in summary["cohorts"].items():
        final = cohort["final_loss"]
        late = cohort["late_250_mean_loss"]
        trajectory = cohort["mean_trajectory_loss"]
        print(
            f"{cohort_name}: final {final['baseline_mean']:.2f} -> "
            f"{final['candidate_mean']:.2f} "
            f"({final['wins']}/{final['ties']}/{final['losses']}); "
            f"late250 {late['baseline_mean']:.2f} -> "
            f"{late['candidate_mean']:.2f} "
            f"({late['wins']}/{late['ties']}/{late['losses']}); "
            f"trajectory {trajectory['baseline_mean']:.2f} -> "
            f"{trajectory['candidate_mean']:.2f}"
        )
    print(f"详细结果：{args.output}")


if __name__ == "__main__":
    main()
