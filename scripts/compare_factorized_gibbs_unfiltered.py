"""关闭整代接受，比较独立核与低阶因子 Gibbs 核的长期动力学。

两侧使用相同温度、初始化、donor 机制、轮数和主随机流。candidate 只增加固定数量
的随机扫描 Gibbs sweep；额外 Gibbs 随机量来自独立流，不会错位后续 donor、独立
mask 初值或 mutation。每个原始 proposal 无条件成为下一状态，主终点是最终当前表。
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
from scipy import stats

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    direction_rms_scale,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.factorized_diffusion import evolve_step_factorized_gibbs
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
N_RECORDS = 300
RHO = 0.01
ETA = 0.5
MU = 0.01


def _git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _environment(device):
    result = {
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device in ("cuda", "cpu"):
        import torch

        result["torch"] = torch.__version__
        result["torch_cuda_runtime"] = torch.version.cuda
        if device == "cuda":
            result["cuda_device_name"] = torch.cuda.get_device_name(0)
            result["cuda_device_capability"] = list(
                torch.cuda.get_device_capability(0)
            )
    return result


def _gibbs_seed(seed):
    sequence = np.random.SeedSequence([int(seed), 0x4749424253])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _rng_state_sha256(rng):
    serialized = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _run_one(
    target,
    queries,
    schema,
    marginals,
    *,
    seed,
    rounds,
    temperature,
    sweeps,
    device,
):
    rng = np.random.default_rng(seed)
    gibbs_rng = (
        np.random.default_rng(_gibbs_seed(seed)) if sweeps > 0 else None
    )
    state = init_synthetic_table(
        N_RECORDS, schema, rng, marginals=marginals
    )
    q, residual, fitness = evaluate_vectorized(
        state,
        queries,
        schema,
        target=target,
        n_records=N_RECORDS,
        batch_size=256,
        device=device,
        want_fitness=True,
        verbose=False,
    )
    initial_loss = float(compute_loss(target, q))
    best_loss = initial_loss
    direction_reference_scale = None
    loss_history = []
    gain_history = []
    changed_cells_history = []
    unique_history = []
    factor_build_elapsed = 0.0
    gibbs_sample_elapsed = 0.0
    active_gibbs_rows = 0
    active_blocks = 0
    factor_count = 0
    factor_table_entries = 0
    gibbs_microsteps = 0
    direction_elapsed = 0.0
    start = time.perf_counter()

    for round_index in range(rounds):
        current_loss = float(compute_loss(target, q))
        loss_history.append(current_loss)
        unique_history.append(int(len(state.value_counts())))
        if np.all(residual == 0.0):
            break

        use_torch = device in ("cuda", "cpu")
        distances = pairwise_block_distance(
            state,
            state,
            schema,
            device=device,
            return_tensor=use_torch,
        )
        progress = (
            round_index / (rounds - 1) if rounds > 1 else 1.0
        )
        alpha = 2.0 + 8.0 * progress
        probabilities = compute_sampling_probs(
            fitness,
            distances,
            beta=1.0,
            h=0.8,
            device=device,
            distance_mode="geometric",
            lambda_param=0.5,
            alpha=alpha,
            delta=0.05,
            winsorize_quantiles=(0.01, 0.99),
            exclude_self=True,
        )
        donor_indices = sample_donors(probabilities, rng, device=device)
        donors = state.iloc[donor_indices].reset_index(drop=True)

        direction_start = time.perf_counter()
        directions = compute_copy_direction_scores(
            state,
            donors,
            schema,
            queries,
            residual,
            batch_size=256,
            device=device,
        )
        direction_elapsed += time.perf_counter() - direction_start
        differs = np.column_stack([
            state[attr].reset_index(drop=True).to_numpy()
            != donors[attr].to_numpy()
            for attr in schema.attribute_names()
        ])
        if direction_reference_scale is None:
            candidate_scale = direction_rms_scale(directions[differs])
            if candidate_scale > 0.0:
                direction_reference_scale = candidate_scale
        effective_strength = (
            temperature / direction_reference_scale
            if direction_reference_scale is not None else 0.0
        )

        proposal, update_diagnostics = evolve_step_factorized_gibbs(
            state,
            donors,
            schema,
            queries,
            residual,
            rho=RHO,
            eta=ETA,
            mu=MU,
            copy_direction_scores=directions,
            copy_direction_strength=effective_strength,
            n_sweeps=sweeps,
            rng=rng,
            gibbs_rng=gibbs_rng,
            max_factor_order=3,
        )
        proposal_q, proposal_residual, proposal_fitness = evaluate_vectorized(
            proposal,
            queries,
            schema,
            target=target,
            n_records=N_RECORDS,
            batch_size=256,
            device=device,
            want_fitness=True,
            verbose=False,
        )
        proposal_loss = float(compute_loss(target, proposal_q))
        gain_history.append(current_loss - proposal_loss)
        changed_cells_history.append(int(
            (
                proposal.reset_index(drop=True)
                != state.reset_index(drop=True)
            ).to_numpy().sum()
        ))
        best_loss = min(best_loss, proposal_loss)
        factor_build_elapsed += update_diagnostics[
            "factor_build_elapsed_sec"
        ]
        gibbs_sample_elapsed += update_diagnostics[
            "gibbs_sample_elapsed_sec"
        ]
        active_gibbs_rows += update_diagnostics["active_gibbs_rows"]
        active_blocks += update_diagnostics["active_blocks"]
        factor_count += update_diagnostics["factor_count"]
        factor_table_entries += update_diagnostics["factor_table_entries"]
        gibbs_microsteps += update_diagnostics["gibbs_microsteps"]

        # 核心条件：不检查 proposal_loss，不重试，不回滚，无条件进入下一状态。
        state = proposal
        q = proposal_q
        residual = proposal_residual
        fitness = proposal_fitness

    final_loss = float(compute_loss(target, q))
    elapsed = time.perf_counter() - start
    gains = np.asarray(gain_history, dtype=float)
    losses = np.asarray(loss_history, dtype=float)
    positive_gains = gains[gains > 0.0]
    negative_gains = gains[gains < 0.0]
    label = "independent" if sweeps == 0 else f"gibbs_{sweeps}_sweeps"
    result = {
        "seed": int(seed),
        "name": label,
        "temperature": float(temperature),
        "sweeps": int(sweeps),
        "rounds_run": len(gain_history),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "best_loss_diagnostic_only": best_loss,
        "final_change_pct": (final_loss / initial_loss - 1.0) * 100.0,
        "mean_raw_gain": float(gains.mean()) if len(gains) else 0.0,
        "mean_trajectory_loss": (
            float(losses.mean()) if len(losses) else final_loss
        ),
        "late_100_mean_loss": (
            float(losses[-100:].mean()) if len(losses) else final_loss
        ),
        "late_250_mean_loss": (
            float(losses[-250:].mean()) if len(losses) else final_loss
        ),
        "positive_gain_rate": (
            float(np.mean(gains > 0.0)) if len(gains) else 0.0
        ),
        "negative_gain_rate": (
            float(np.mean(gains < 0.0)) if len(gains) else 0.0
        ),
        "mean_positive_gain": (
            float(positive_gains.mean()) if len(positive_gains) else 0.0
        ),
        "mean_negative_gain": (
            float(negative_gains.mean()) if len(negative_gains) else 0.0
        ),
        "mean_changed_cells": (
            float(np.mean(changed_cells_history))
            if changed_cells_history else 0.0
        ),
        "maximum_loss": float(max(loss_history + [final_loss])),
        "final_unique_states": int(len(state.value_counts())),
        "mean_unique_states": float(np.mean(unique_history)),
        "direction_reference_scale": direction_reference_scale,
        "direction_elapsed_sec": direction_elapsed,
        "factor_build_elapsed_sec": factor_build_elapsed,
        "gibbs_sample_elapsed_sec": gibbs_sample_elapsed,
        "active_gibbs_rows": active_gibbs_rows,
        "active_blocks": active_blocks,
        "factor_count": factor_count,
        "factor_table_entries": factor_table_entries,
        "gibbs_microsteps": gibbs_microsteps,
        "elapsed_sec": elapsed,
        "primary_rng_state_sha256": _rng_state_sha256(rng),
        "final_csv_sha256": hashlib.sha256(
            state.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
        "loss_history": loss_history,
        "gain_history": gain_history,
        "changed_cells_history": changed_cells_history,
    }
    print(
        f"seed={seed:02d} {label:<16} "
        f"loss={initial_loss:.1f}->{final_loss:.1f} "
        f"({result['final_change_pct']:+.1f}%) "
        f"raw_pos={result['positive_gain_rate']:.1%} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return result


def _aggregate(rows, key):
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values.tolist(),
    }


def _paired(candidate, baseline, key, lower_is_better):
    candidate_values = np.asarray([row[key] for row in candidate], dtype=float)
    baseline_values = np.asarray([row[key] for row in baseline], dtype=float)
    difference = candidate_values - baseline_values
    if len(difference) < 2 or np.all(difference == 0.0):
        paired_t = None
        paired_p = None
    else:
        statistic, p_value = stats.ttest_rel(
            candidate_values, baseline_values
        )
        paired_t = float(statistic) if np.isfinite(statistic) else None
        paired_p = float(p_value) if np.isfinite(p_value) else None
    better = difference < 0.0 if lower_is_better else difference > 0.0
    worse = difference > 0.0 if lower_is_better else difference < 0.0
    return {
        "candidate_mean": float(candidate_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "mean_difference": float(difference.mean()),
        "paired_t": paired_t,
        "paired_p": paired_p,
        "wins": int(np.sum(better)),
        "ties": int(np.sum(difference == 0.0)),
        "losses": int(np.sum(worse)),
        "values": difference.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default="outputs/factorized_gibbs/unfiltered_tau2_sweep8.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds 必须为正整数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")

    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)
    runs = {"independent": [], f"gibbs_{args.sweeps}_sweeps": []}
    experiment_start = time.perf_counter()
    for seed in args.seeds:
        runs["independent"].append(_run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=0,
            device=args.device,
        ))
        runs[f"gibbs_{args.sweeps}_sweeps"].append(_run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=args.sweeps,
            device=args.device,
        ))

    metrics = (
        "initial_loss",
        "final_loss",
        "best_loss_diagnostic_only",
        "final_change_pct",
        "mean_raw_gain",
        "mean_trajectory_loss",
        "late_100_mean_loss",
        "late_250_mean_loss",
        "positive_gain_rate",
        "negative_gain_rate",
        "mean_positive_gain",
        "mean_negative_gain",
        "mean_changed_cells",
        "maximum_loss",
        "final_unique_states",
        "mean_unique_states",
        "direction_elapsed_sec",
        "factor_build_elapsed_sec",
        "gibbs_sample_elapsed_sec",
        "active_gibbs_rows",
        "active_blocks",
        "factor_count",
        "factor_table_entries",
        "gibbs_microsteps",
        "elapsed_sec",
    )
    aggregate = {
        name: {key: _aggregate(rows, key) for key in metrics}
        for name, rows in runs.items()
    }
    candidate_name = f"gibbs_{args.sweeps}_sweeps"
    comparisons = {
        "final_loss": _paired(
            runs[candidate_name], runs["independent"], "final_loss", True
        ),
        "mean_raw_gain": _paired(
            runs[candidate_name],
            runs["independent"],
            "mean_raw_gain",
            False,
        ),
        "mean_trajectory_loss": _paired(
            runs[candidate_name],
            runs["independent"],
            "mean_trajectory_loss",
            True,
        ),
        "late_100_mean_loss": _paired(
            runs[candidate_name],
            runs["independent"],
            "late_100_mean_loss",
            True,
        ),
        "late_250_mean_loss": _paired(
            runs[candidate_name],
            runs["independent"],
            "late_250_mean_loss",
            True,
        ),
        "positive_gain_rate": _paired(
            runs[candidate_name],
            runs["independent"],
            "positive_gain_rate",
            False,
        ),
        "final_unique_states": _paired(
            runs[candidate_name],
            runs["independent"],
            "final_unique_states",
            False,
        ),
    }
    primary_rng_aligned = all(
        baseline["primary_rng_state_sha256"]
        == candidate["primary_rng_state_sha256"]
        for baseline, candidate in zip(
            runs["independent"], runs[candidate_name]
        )
    )
    initial_loss_aligned = all(
        baseline["initial_loss"] == candidate["initial_loss"]
        for baseline, candidate in zip(
            runs["independent"], runs[candidate_name]
        )
    )
    direction_scale_aligned = all(
        baseline["direction_reference_scale"]
        == candidate["direction_reference_scale"]
        for baseline, candidate in zip(
            runs["independent"], runs[candidate_name]
        )
    )
    summary = {
        "experiment": "factorized_gibbs_unfiltered_dynamics",
        "scope": (
            "same_temperature_and_primary_rng_every_raw_proposal_becomes_"
            "next_state_no_loss_acceptance"
        ),
        "primary_endpoint": "final_current_loss_not_best_loss",
        "dataset": "test_300x10",
        "n_rounds": args.rounds,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "candidate_sweeps": args.sweeps,
        "rho": RHO,
        "eta": ETA,
        "mu": MU,
        "device": args.device,
        "git_commit": _git_commit(),
        "command_argv": sys.argv,
        "environment": _environment(args.device),
        "primary_rng_aligned_all_seeds": primary_rng_aligned,
        "initial_loss_aligned_all_seeds": initial_loss_aligned,
        "direction_reference_scale_aligned_all_seeds": (
            direction_scale_aligned
        ),
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n===== 无整代接受的最终当前表 =====")
    for name in runs:
        final = aggregate[name]["final_loss"]
        positive = aggregate[name]["positive_gain_rate"]
        elapsed = aggregate[name]["elapsed_sec"]
        print(
            f"{name:<18} final_loss={final['mean']:.2f}±{final['std']:.2f} "
            f"raw_pos={positive['mean']:.1%} "
            f"elapsed={elapsed['mean']:.1f}s"
        )
    print(f"主随机流逐种子对齐：{primary_rng_aligned}")
    print(f"初始 loss 逐种子对齐：{initial_loss_aligned}")
    print(f"首轮方向尺度逐种子对齐：{direction_scale_aligned}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
