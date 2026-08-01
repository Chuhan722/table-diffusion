"""关闭整代贪心接受，检验扩散过程能否形成下降轨迹。

本脚本每轮都把原始 proposal 设为下一状态，不执行 loss 门控、缩步重试或
checkpoint 回滚。主报告使用最终当前表而不是 best 表，避免把历史最优选择误当成
扩散动力。donor 抽样仍按原算法由适应度和距离驱动，因此本实验隔离的是整代接受
筛选，不把结果解释为脱离 donor 选择后的纯随机扩散。实验只使用固定 workload
的精确 target，不读取真实 train/test。
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy import stats

from table_diffevo.directional_diffusion import compute_copy_direction_scores
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.generator import init_synthetic_table
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_loss
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.update import evolve_step
from table_diffevo.vectorized_eval import evaluate_vectorized


SCHEMA_PATH = "configs/test_300x10/schema.yaml"
QUERY_PATH = "configs/test_300x10/measured_50query.json"
MARGINALS_PATH = "configs/test_300x10/init_marginals.json"
N_RECORDS = 300


def _name(strength):
    return "baseline" if strength is None else (
        f"strength_{strength:g}".replace(".", "p")
    )


def _run_one(target, queries, schema, marginals, seed, rounds, strength, device):
    rng = np.random.default_rng(seed)
    state = init_synthetic_table(
        N_RECORDS, schema, rng, marginals=marginals
    )
    loss_history = []
    gain_history = []
    changed_cells_history = []
    start = time.perf_counter()

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
    initial_loss = compute_loss(target, q)
    best_loss = float(initial_loss)

    for round_index in range(rounds):
        loss = compute_loss(target, q)
        loss_history.append(float(loss))
        if np.all(residual == 0.0):
            break

        use_torch = device in ("cuda", "cpu")
        distances = pairwise_block_distance(
            state, state, schema, device=device, return_tensor=use_torch
        )
        progress = (
            round_index / (rounds - 1) if rounds > 1 else 1.0
        )
        alpha = 2.0 + 8.0 * progress
        probs = compute_sampling_probs(
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
        donor_idx = sample_donors(probs, rng, device=device)
        donors = state.iloc[donor_idx].reset_index(drop=True)

        if strength is None:
            direction_kwargs = {}
        else:
            directions = compute_copy_direction_scores(
                state,
                donors,
                schema,
                queries,
                residual,
                batch_size=256,
                device=device,
            )
            direction_kwargs = {
                "copy_direction_scores": directions,
                "copy_direction_strength": strength,
            }

        proposal = evolve_step(
            state,
            donors,
            schema,
            rho=0.01,
            eta=0.5,
            mu=0.01,
            rng=rng,
            **direction_kwargs,
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
        proposal_loss = compute_loss(target, proposal_q)
        gain_history.append(float(loss - proposal_loss))
        changed_cells_history.append(int(
            (proposal.reset_index(drop=True) != state.reset_index(drop=True))
            .to_numpy()
            .sum()
        ))

        # 核心实验条件：无论 proposal_loss 是否上升，都无条件进入下一状态。
        state = proposal
        q = proposal_q
        residual = proposal_residual
        fitness = proposal_fitness
        best_loss = min(best_loss, float(proposal_loss))

    final_loss = float(compute_loss(target, q))
    elapsed = time.perf_counter() - start
    gains = np.asarray(gain_history, dtype=float)
    positive_gains = gains[gains > 0.0]
    negative_gains = gains[gains < 0.0]
    label = _name(strength)
    result = {
        "seed": int(seed),
        "name": label,
        "strength": None if strength is None else float(strength),
        "rounds_run": len(gain_history),
        "initial_loss": float(initial_loss),
        "final_loss": final_loss,
        "best_loss_diagnostic_only": best_loss,
        "final_change_pct": float((final_loss / initial_loss - 1.0) * 100.0),
        "mean_raw_gain": float(gains.mean()) if len(gains) else 0.0,
        "positive_gain_rate": float(np.mean(gains > 0.0)) if len(gains) else 0.0,
        "negative_gain_rate": float(np.mean(gains < 0.0)) if len(gains) else 0.0,
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
        "elapsed_sec": float(elapsed),
        "n_unique": int(len(state.value_counts())),
        "final_csv_sha256": hashlib.sha256(
            state.to_csv(index=False).encode("utf-8")
        ).hexdigest(),
    }
    print(
        f"seed={seed:02d} {label:<14} "
        f"loss={initial_loss:.1f}->{final_loss:.1f} "
        f"({result['final_change_pct']:+.1f}%) "
        f"raw_pos={result['positive_gain_rate']:.1%}",
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
    diff = candidate_values - baseline_values
    if len(diff) < 2 or np.all(diff == 0.0):
        paired_t, paired_p = None, None
    else:
        paired_t, paired_p = stats.ttest_rel(candidate_values, baseline_values)
        paired_t = float(paired_t) if np.isfinite(paired_t) else None
        paired_p = float(paired_p) if np.isfinite(paired_p) else None
    better = diff < 0.0 if lower_is_better else diff > 0.0
    worse = diff > 0.0 if lower_is_better else diff < 0.0
    return {
        "candidate_mean": float(candidate_values.mean()),
        "baseline_mean": float(baseline_values.mean()),
        "mean_difference": float(diff.mean()),
        "paired_t": paired_t,
        "paired_p": paired_p,
        "wins": int(np.sum(better)),
        "ties": int(np.sum(diff == 0.0)),
        "losses": int(np.sum(worse)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument(
        "--strengths", nargs="+", type=float, default=[0.0, 20.0, 100.0]
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output",
        default="outputs/residual_directed_diffusion_small/unfiltered.json",
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
        runs[_name(strength)] = []
    for seed in args.seeds:
        runs["baseline"].append(_run_one(
            target, queries, schema, marginals, seed, args.rounds, None,
            args.device,
        ))
        for strength in args.strengths:
            runs[_name(strength)].append(_run_one(
                target, queries, schema, marginals, seed, args.rounds,
                strength, args.device,
            ))

    metrics = (
        "initial_loss",
        "final_loss",
        "best_loss_diagnostic_only",
        "final_change_pct",
        "mean_raw_gain",
        "positive_gain_rate",
        "negative_gain_rate",
        "mean_positive_gain",
        "mean_negative_gain",
        "mean_changed_cells",
        "maximum_loss",
        "elapsed_sec",
        "n_unique",
    )
    aggregate = {
        name: {key: _aggregate(rows, key) for key in metrics}
        for name, rows in runs.items()
    }
    comparisons = {
        f"{name}_vs_baseline": {
            "final_loss": _paired(
                rows, runs["baseline"], "final_loss", True
            ),
            "mean_raw_gain": _paired(
                rows, runs["baseline"], "mean_raw_gain", False
            ),
            "positive_gain_rate": _paired(
                rows, runs["baseline"], "positive_gain_rate", False
            ),
        }
        for name, rows in runs.items()
        if name != "baseline"
    }
    endpoint_exact = all(
        endpoint["final_csv_sha256"] == baseline["final_csv_sha256"]
        for endpoint, baseline in zip(runs["strength_0"], runs["baseline"])
    )
    summary = {
        "experiment": "unfiltered_residual_directed_diffusion",
        "scope": "every_raw_proposal_becomes_next_state_no_loss_acceptance",
        "primary_endpoint": "final_current_loss_not_best_loss",
        "dataset": "test_300x10",
        "n_rounds": args.rounds,
        "seeds": args.seeds,
        "strengths": args.strengths,
        "device": args.device,
        "strength_zero_csv_exact": endpoint_exact,
        "runs": runs,
        "aggregate": aggregate,
        "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)

    print("\n===== 无整代接受的最终当前表 =====")
    for name in runs:
        final = aggregate[name]["final_loss"]
        change = aggregate[name]["final_change_pct"]
        positive = aggregate[name]["positive_gain_rate"]
        print(
            f"{name:<14} final_loss={final['mean']:.1f}±{final['std']:.1f} "
            f"change={change['mean']:+.1f}% raw_pos={positive['mean']:.1%}"
        )
    print(f"strength=0 CSV 逐种子等价：{endpoint_exact}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
