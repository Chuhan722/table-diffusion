"""区分整代曲率核的时间减速与状态依赖多步漂移。

本脚本重放 Issue #24 的 gamma=0/1 无接受轨迹，只增加不参与决策的查询空间
诊断。主比较按累计查询二次变差对齐，不改变生成算法或原正式结论。
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import stats

if __package__:
    from scripts import compare_generation_curvature_unfiltered as dynamics
else:
    import compare_generation_curvature_unfiltered as dynamics

from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


FORMAL_REFERENCE_OUTPUT = Path(
    "outputs/generation_curvature_dynamics/"
    "formal_20seed_1000r_tau2_sweep8_ecf072c.json"
)
FORMAL_REFERENCE_SHA256 = (
    "3a6d185b9522550ce0807069b03a087780aefc0696da84a4c630339e10076234"
)
FORMAL_OUTPUT = Path(
    "outputs/curvature_multistep_drift/"
    "formal_20seed_1000r_tau2_sweep8_query_clock.json"
)
MATCHED_GRID_POINTS = 251
MATCHED_TAIL_FRACTION = 0.25
N_STATE_BINS = 10
MIN_TRANSITIONS_PER_SEED_BIN = 10
REPLAY_TIMING_KEYS = {
    "direction_elapsed_sec",
    "factor_build_elapsed_sec",
    "gibbs_sample_elapsed_sec",
    "elapsed_sec",
}
QUERY_CLOCK_KEYS = {
    "query_clock_recorded",
    "query_count_history",
    "query_state_sha256_history",
    "count_residual_l2_squared_history",
    "query_delta_l2_squared_history",
    "linear_gain_history",
    "quadratic_cost_history",
    "gain_identity_error_history",
    "gain_identity_max_abs_error",
    "cumulative_query_quadratic_variation_history",
}


def _mean_t_interval(values, confidence=0.95):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("区间输入必须是非空有限一维数组")
    mean = float(array.mean())
    if len(array) == 1:
        return [mean, mean]
    standard_error = float(array.std(ddof=1) / np.sqrt(len(array)))
    if standard_error == 0.0:
        return [mean, mean]
    critical = float(stats.t.ppf(
        0.5 + confidence / 2.0,
        df=len(array) - 1,
    ))
    return [
        float(mean - critical * standard_error),
        float(mean + critical * standard_error),
    ]


def _paired_difference_summary(values):
    array = np.asarray(values, dtype=float)
    summary = dynamics._summarize(array)
    summary.update({
        "mean_t_interval_95": _mean_t_interval(array),
        "wins": int(np.sum(array < 0.0)),
        "ties": int(np.sum(array == 0.0)),
        "losses": int(np.sum(array > 0.0)),
        "preference": "candidate_lower_is_better",
    })
    return summary


def _clock_and_loss(row, clock_kind):
    losses = np.asarray(row["loss_history"], dtype=float)
    rounds = int(row["rounds_run"])
    if losses.shape != (rounds + 1,) or not np.all(np.isfinite(losses)):
        raise ValueError("loss_history 与轮数不一致或包含非有限值")
    if clock_kind == "query_quadratic_variation":
        clock = np.asarray(
            row["cumulative_query_quadratic_variation_history"],
            dtype=float,
        )
    elif clock_kind == "changed_cells":
        changed = np.asarray(row["changed_cells_history"], dtype=float)
        if changed.shape != (rounds,):
            raise ValueError("changed_cells_history 与轮数不一致")
        clock = np.concatenate([[0.0], np.cumsum(changed)])
    else:
        raise ValueError(f"未知内禀时钟：{clock_kind!r}")
    if (
        clock.shape != (rounds + 1,)
        or not np.all(np.isfinite(clock))
        or np.any(clock < 0.0)
        or np.any(np.diff(clock) < 0.0)
    ):
        raise ValueError(f"{clock_kind} 时钟必须有限、非负且单调")
    duplicate = np.diff(clock) == 0.0
    if np.any(losses[1:][duplicate] != losses[:-1][duplicate]):
        raise ValueError("内禀时钟未前进时 loss 却发生变化")
    # 重复时钟对应相同查询状态；保留最后一个点，保证插值横坐标严格递增。
    keep = np.r_[clock[1:] != clock[:-1], True]
    return clock[keep], losses[keep]


def _matched_clock_pair(
    baseline,
    candidate,
    clock_kind,
    *,
    grid_points=MATCHED_GRID_POINTS,
    tail_fraction=MATCHED_TAIL_FRACTION,
):
    if grid_points < 2:
        raise ValueError("匹配时钟网格至少需要两个点")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction 必须位于 (0, 1]")
    baseline_clock, baseline_loss = _clock_and_loss(
        baseline, clock_kind
    )
    candidate_clock, candidate_loss = _clock_and_loss(
        candidate, clock_kind
    )
    common_final = float(min(
        baseline_clock[-1], candidate_clock[-1]
    ))
    if common_final <= 0.0:
        raise ValueError("共同内禀时钟终点必须为正")
    start = float((1.0 - tail_fraction) * common_final)
    grid = np.linspace(start, common_final, grid_points)
    baseline_interpolated = np.interp(
        grid, baseline_clock, baseline_loss
    )
    candidate_interpolated = np.interp(
        grid, candidate_clock, candidate_loss
    )
    difference = candidate_interpolated - baseline_interpolated
    return {
        "seed": int(baseline["seed"]),
        "clock": clock_kind,
        "baseline_final_clock": float(baseline_clock[-1]),
        "candidate_final_clock": float(candidate_clock[-1]),
        "candidate_to_baseline_final_clock_ratio": float(
            candidate_clock[-1] / baseline_clock[-1]
        ),
        "common_final_clock": common_final,
        "matched_start_clock": start,
        "grid_points": int(grid_points),
        "tail_fraction": float(tail_fraction),
        "baseline_matched_mean_loss": float(
            baseline_interpolated.mean()
        ),
        "candidate_matched_mean_loss": float(
            candidate_interpolated.mean()
        ),
        "matched_mean_loss_difference": float(difference.mean()),
        "matched_endpoint_loss_difference": float(difference[-1]),
    }


def _rows_by_seed(rows):
    result = {int(row["seed"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("轨迹 seed 不得重复")
    return result


def _matched_clock_summary(baseline, candidate, clock_kind):
    baseline_by_seed = _rows_by_seed(baseline)
    candidate_by_seed = _rows_by_seed(candidate)
    if set(baseline_by_seed) != set(candidate_by_seed):
        raise ValueError("baseline/candidate 的 seed 集合不一致")
    pairs = [
        _matched_clock_pair(
            baseline_by_seed[seed],
            candidate_by_seed[seed],
            clock_kind,
        )
        for seed in sorted(baseline_by_seed)
    ]
    differences = [
        pair["matched_mean_loss_difference"] for pair in pairs
    ]
    ratios = [
        pair["candidate_to_baseline_final_clock_ratio"]
        for pair in pairs
    ]
    endpoints = [
        pair["matched_endpoint_loss_difference"] for pair in pairs
    ]
    return {
        "clock": clock_kind,
        "grid_points": MATCHED_GRID_POINTS,
        "tail_fraction": MATCHED_TAIL_FRACTION,
        "pairs": pairs,
        "matched_mean_loss_difference": _paired_difference_summary(
            differences
        ),
        "matched_endpoint_loss_difference": _paired_difference_summary(
            endpoints
        ),
        "candidate_to_baseline_final_clock_ratio": dynamics._summarize(
            ratios
        ),
    }


def _calendar_tail_summary(baseline, candidate):
    baseline_by_seed = _rows_by_seed(baseline)
    candidate_by_seed = _rows_by_seed(candidate)
    if set(baseline_by_seed) != set(candidate_by_seed):
        raise ValueError("baseline/candidate 的 seed 集合不一致")
    pairs = []
    differences = []
    for seed in sorted(baseline_by_seed):
        baseline_value = float(
            baseline_by_seed[seed]["late_250_mean_loss"]
        )
        candidate_value = float(
            candidate_by_seed[seed]["late_250_mean_loss"]
        )
        difference = candidate_value - baseline_value
        pairs.append({
            "seed": seed,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "difference": difference,
        })
        differences.append(difference)
    return {
        "window": 250,
        "pairs": pairs,
        "difference": _paired_difference_summary(differences),
    }


def _classify_multistep_effect(query_clock, calendar):
    matched = query_clock["matched_mean_loss_difference"]
    lower, upper = matched["mean_t_interval_95"]
    if upper < 0.0:
        return "clock_efficiency_advantage_but_round_slowdown"
    if lower > 0.0:
        return "drift_disadvantage_after_clock_matching"
    calendar_mean = calendar["difference"]["mean"]
    if (
        calendar_mean != 0.0
        and abs(matched["mean"]) <= 0.5 * abs(calendar_mean)
    ):
        return "time_rescaling_material_residual_inconclusive"
    return "mixed_or_inconclusive_multistep_effect"


def _state_dependent_drift(baseline, candidate, n_queries):
    if not baseline or not candidate:
        raise ValueError("状态依赖漂移需要非空的 baseline/candidate 轨迹")
    if not isinstance(n_queries, (int, np.integer)) or n_queries <= 0:
        raise ValueError("n_queries 必须为正整数")
    baseline_radius = np.concatenate([
        np.sqrt(
            np.asarray(
                row["count_residual_l2_squared_history"][:-1],
                dtype=float,
            ) / n_queries
        )
        for row in baseline
    ])
    if not np.all(np.isfinite(baseline_radius)):
        raise ValueError("baseline 残差半径包含非有限值")
    internal = np.quantile(
        baseline_radius,
        np.linspace(0.0, 1.0, N_STATE_BINS + 1)[1:-1],
    )
    boundaries = np.concatenate(([-np.inf], internal, [np.inf]))
    result = {
        "n_bins": N_STATE_BINS,
        "minimum_transitions_per_seed_bin": MIN_TRANSITIONS_PER_SEED_BIN,
        "baseline_internal_rms_residual_boundaries": internal.tolist(),
        "bins": [],
    }
    for bin_index, (lower, upper) in enumerate(zip(
        boundaries[:-1], boundaries[1:]
    )):
        bin_result = {
            "bin": int(bin_index),
            "lower_inclusive": (
                float(lower) if np.isfinite(lower) else None
            ),
            "upper_exclusive": (
                float(upper) if np.isfinite(upper) else None
            ),
        }
        for label, rows in (("baseline", baseline), ("candidate", candidate)):
            seed_rows = []
            for row in rows:
                radius = np.sqrt(
                    np.asarray(
                        row["count_residual_l2_squared_history"][:-1],
                        dtype=float,
                    ) / n_queries
                )
                mask = (radius >= lower) & (radius < upper)
                count = int(mask.sum())
                if count < MIN_TRANSITIONS_PER_SEED_BIN:
                    continue
                gains = np.asarray(row["gain_history"], dtype=float)[mask]
                query_variation = np.asarray(
                    row["query_delta_l2_squared_history"], dtype=float
                )[mask]
                changed = np.asarray(
                    row["changed_cells_history"], dtype=float
                )[mask]
                total_variation = float(query_variation.sum())
                seed_rows.append({
                    "seed": int(row["seed"]),
                    "n_transitions": count,
                    "mean_gain": float(gains.mean()),
                    "mean_query_quadratic_variation": float(
                        query_variation.mean()
                    ),
                    "mean_changed_cells": float(changed.mean()),
                    "gain_per_query_quadratic_variation": (
                        float(gains.sum() / total_variation)
                        if total_variation > 0.0 else 0.0
                    ),
                    "positive_gain_rate": float(np.mean(gains > 0.0)),
                    "zero_gain_rate": float(np.mean(gains == 0.0)),
                    "negative_gain_rate": float(np.mean(gains < 0.0)),
                })
            metrics = {}
            for metric in (
                "mean_gain",
                "mean_query_quadratic_variation",
                "mean_changed_cells",
                "gain_per_query_quadratic_variation",
                "positive_gain_rate",
                "zero_gain_rate",
                "negative_gain_rate",
            ):
                metrics[metric] = (
                    dynamics._summarize([
                        row[metric] for row in seed_rows
                    ])
                    if seed_rows else None
                )
            bin_result[label] = {
                "n_seeds": len(seed_rows),
                "n_transitions": int(sum(
                    row["n_transitions"] for row in seed_rows
                )),
                "seed_rows": seed_rows,
                "metrics": metrics,
            }
        result["bins"].append(bin_result)
    return result


def _load_reference(path):
    if not path.is_file():
        raise FileNotFoundError(f"参考输出不存在：{path}")
    digest = dynamics._sha256_file(path)
    if digest != FORMAL_REFERENCE_SHA256:
        raise ValueError(
            "参考输出 SHA-256 与预注册值不一致："
            f"{digest}"
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        payload.get("status") != "complete"
        or payload.get("seeds") != dynamics.FORMAL_SEEDS
        or payload.get("rounds") != dynamics.FORMAL_ROUNDS
        or set(payload.get("runs", {})) != {"baseline", "candidate"}
    ):
        raise ValueError("参考输出不满足 Issue #24 正式协议")
    return payload, digest


def _audit_replay(reference, runs):
    failures = []
    checked_fields = 0
    for variant in ("baseline", "candidate"):
        reference_rows = _rows_by_seed(reference["runs"][variant])
        replay_rows = _rows_by_seed(runs[variant])
        if set(reference_rows) != set(replay_rows):
            failures.append({
                "variant": variant,
                "seed": None,
                "key": "seed_set",
            })
            continue
        for seed in sorted(reference_rows):
            expected = reference_rows[seed]
            actual = replay_rows[seed]
            missing = sorted(set(expected) - set(actual))
            if missing:
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "key": f"missing:{missing[0]}",
                })
                continue
            unexpected = sorted(
                set(actual) - set(expected) - QUERY_CLOCK_KEYS
            )
            if unexpected:
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "key": f"unexpected:{unexpected[0]}",
                })
                continue
            for key in sorted(set(expected) - REPLAY_TIMING_KEYS):
                checked_fields += 1
                if actual[key] != expected[key]:
                    failures.append({
                        "variant": variant,
                        "seed": seed,
                        "key": key,
                    })
                    break
    return {
        "passed": not failures,
        "n_runs": sum(len(rows) for rows in runs.values()),
        "checked_fields": checked_fields,
        "checked_transitions": int(sum(
            row["rounds_run"] for rows in runs.values() for row in rows
        )),
        "excluded_timing_keys": sorted(REPLAY_TIMING_KEYS),
        "additive_query_clock_keys": sorted(QUERY_CLOCK_KEYS),
        "failures": failures,
    }


def _query_clock_gate(runs, rounds, target):
    failures = []
    maximum_identity_error = 0.0
    checked_query_vectors = 0
    checked_transitions = 0
    if not isinstance(rounds, (int, np.integer)) or rounds < 0:
        raise ValueError("rounds 必须为非负整数")
    target_values = np.asarray(target, dtype=float)
    if (
        target_values.ndim != 1
        or len(target_values) == 0
        or not np.all(np.isfinite(target_values))
    ):
        raise ValueError("target 必须是有限一维数组")
    for variant, rows in runs.items():
        for row in rows:
            seed = row.get("seed")
            expected_state_length = rounds + 1
            expected_step_length = rounds
            state_keys = (
                "query_count_history",
                "query_state_sha256_history",
                "count_residual_l2_squared_history",
                "cumulative_query_quadratic_variation_history",
                "loss_history",
            )
            step_keys = (
                "query_delta_l2_squared_history",
                "linear_gain_history",
                "quadratic_cost_history",
                "gain_identity_error_history",
                "gain_history",
                "changed_cells_history",
            )
            required_keys = {
                *state_keys,
                *step_keys,
                "gain_identity_max_abs_error",
            }
            if (
                not row.get("query_clock_recorded")
                or row.get("rounds_run") != rounds
                or not required_keys.issubset(row)
            ):
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": "missing_field_or_round_count",
                })
                continue
            try:
                state_lengths = [len(row[key]) for key in state_keys]
                step_lengths = [len(row[key]) for key in step_keys]
                hashes = row["query_state_sha256_history"]
                hashes_valid = all(
                    isinstance(value, str)
                    and len(value) == 64
                    and all(character in "0123456789abcdef" for character in value)
                    for value in hashes
                )
            except TypeError:
                state_lengths = []
                step_lengths = []
                hashes_valid = False
            if (
                state_lengths != [expected_state_length] * len(state_keys)
                or step_lengths != [expected_step_length] * len(step_keys)
                or not hashes_valid
            ):
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": "history_length_or_hash",
                })
                continue
            try:
                query_counts = np.asarray(row["query_count_history"])
            except ValueError:
                query_counts = np.asarray([], dtype=float)
            if (
                query_counts.shape != (
                    expected_state_length, len(target_values)
                )
                or query_counts.dtype.kind not in "iu"
                or np.any(query_counts < 0)
            ):
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": "query_count_shape_or_dtype",
                })
                continue
            expected_hashes = [
                dynamics._query_vector_sha256(query_vector)
                for query_vector in query_counts
            ]
            if expected_hashes != row["query_state_sha256_history"]:
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": "query_state_hash_mismatch",
                })
                continue
            query_counts_float = query_counts.astype(float, copy=False)
            query_delta = np.diff(query_counts_float, axis=0)
            count_residual = target_values - query_counts_float
            query_variation = np.einsum(
                "ij,ij->i", query_delta, query_delta
            )
            residual_squared = np.einsum(
                "ij,ij->i", count_residual, count_residual
            )
            linear_gain = np.einsum(
                "ij,ij->i", count_residual[:-1], query_delta
            )
            quadratic_cost = 0.5 * query_variation
            recomputed_losses = 0.5 * residual_squared
            actual_gain = recomputed_losses[:-1] - recomputed_losses[1:]
            identity_error = actual_gain - (
                linear_gain - quadratic_cost
            )
            cumulative_variation = np.concatenate([
                [0.0], np.cumsum(query_variation)
            ])
            recomputed = {
                "loss_history": recomputed_losses,
                "gain_history": actual_gain,
                "count_residual_l2_squared_history": residual_squared,
                "query_delta_l2_squared_history": query_variation,
                "linear_gain_history": linear_gain,
                "quadratic_cost_history": quadratic_cost,
                "gain_identity_error_history": identity_error,
                "cumulative_query_quadratic_variation_history": (
                    cumulative_variation
                ),
            }
            try:
                mismatch = next((
                    key for key, expected in recomputed.items()
                    if not np.array_equal(
                        np.asarray(row[key], dtype=float), expected
                    )
                ), None)
                changed_cells = np.asarray(
                    row["changed_cells_history"]
                )
                changed_cells_valid = (
                    changed_cells.shape == (expected_step_length,)
                    and changed_cells.dtype.kind in "iu"
                    and np.all(changed_cells >= 0)
                )
            except (TypeError, ValueError):
                mismatch = "diagnostic_array"
                changed_cells_valid = False
            if mismatch is not None:
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": f"recomputed_{mismatch}_mismatch",
                })
                continue
            if not changed_cells_valid:
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": "changed_cells_invalid",
                })
                continue
            recomputed_max_error = float(
                np.max(np.abs(identity_error), initial=0.0)
            )
            if (
                not isinstance(
                    row["gain_identity_max_abs_error"],
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(row["gain_identity_max_abs_error"])
                or row["gain_identity_max_abs_error"]
                != recomputed_max_error
            ):
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": "gain_identity_max_abs_error_mismatch",
                })
                continue
            checked_query_vectors += expected_state_length
            checked_transitions += expected_step_length
            maximum_identity_error = max(
                maximum_identity_error,
                recomputed_max_error,
            )
            try:
                _clock_and_loss(row, "query_quadratic_variation")
                _clock_and_loss(row, "changed_cells")
            except ValueError as error:
                failures.append({
                    "variant": variant,
                    "seed": seed,
                    "reason": str(error),
                })
    return {
        "passed": not failures and maximum_identity_error <= 1e-12,
        "maximum_gain_identity_abs_error": float(
            maximum_identity_error
        ),
        "checked_query_vectors": int(checked_query_vectors),
        "checked_transitions": int(checked_transitions),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=dynamics.FORMAL_SEEDS
    )
    parser.add_argument(
        "--rounds", type=int, default=dynamics.FORMAL_ROUNDS
    )
    parser.add_argument(
        "--temperature", type=float, default=dynamics.FORMAL_TEMPERATURE
    )
    parser.add_argument(
        "--sweeps", type=int, default=dynamics.FORMAL_SWEEPS
    )
    parser.add_argument(
        "--reference-rounds",
        type=int,
        default=dynamics.FORMAL_REFERENCE_ROUNDS,
    )
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--reference-output", default=str(FORMAL_REFERENCE_OUTPUT)
    )
    parser.add_argument("--output", default=str(FORMAL_OUTPUT))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error("--seeds 必须非空、非负且不重复")
    if args.rounds < dynamics.PRIMARY_TAIL_WINDOW:
        parser.error(
            f"--rounds 不得小于主窗口 {dynamics.PRIMARY_TAIL_WINDOW}"
        )
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")
    if args.reference_rounds <= 0:
        parser.error("--reference-rounds 必须为正整数")

    reference_path = Path(args.reference_output)
    reference, reference_sha256 = _load_reference(reference_path)
    formal_protocol_matches = (
        args.seeds == dynamics.FORMAL_SEEDS
        and args.rounds == dynamics.FORMAL_ROUNDS
        and args.temperature == dynamics.FORMAL_TEMPERATURE
        and args.sweeps == dynamics.FORMAL_SWEEPS
        and args.reference_rounds == dynamics.FORMAL_REFERENCE_ROUNDS
        and args.device == "cuda"
        and reference_sha256 == FORMAL_REFERENCE_SHA256
    )
    output = Path(args.output)
    if output.exists() and (formal_protocol_matches or not args.overwrite):
        raise FileExistsError(f"输出已存在，不覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    environment = dynamics._environment_snapshot(args.device)
    if formal_protocol_matches and not environment["git_worktree_clean"]:
        raise RuntimeError("正式协议要求 tracked 工作树干净")
    if args.device == "cuda" and not environment["cuda_available"]:
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备")

    schema = load_schema(str(dynamics.SCHEMA_PATH))
    queries = load_queries(str(dynamics.QUERY_PATH))
    target = np.asarray(
        [query["result"] for query in queries], dtype=float
    )
    marginals = load_marginals(str(dynamics.MARGINALS_PATH))
    if (
        len(queries) != 50
        or target.shape != (50,)
        or not np.all(np.isfinite(target))
        or len(schema.attribute_names()) != 10
        or marginals.get("n_records") != dynamics.N_RECORDS
        or set(marginals.get("attributes", {}))
        != set(schema.attribute_names())
    ):
        raise ValueError("test_300x10 的公开输入与协议不一致")

    experiment_start = time.perf_counter()
    preflight = dynamics._verify_gamma_zero_reference(
        target,
        queries,
        schema,
        marginals,
        seed=0,
        rounds=args.reference_rounds,
        temperature=args.temperature,
        sweeps=args.sweeps,
        device=args.device,
    )
    runs = {"baseline": [], "candidate": []}
    partial = {
        "experiment": "curvature_multistep_drift_query_clock",
        "issue": 27,
        "status": "running",
        "formal_protocol_matches": formal_protocol_matches,
        "environment": environment,
        "reference_output": str(reference_path),
        "reference_output_sha256": reference_sha256,
        "preflight": preflight,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "runs": runs,
    }
    dynamics._write_json(output, partial)

    for seed in args.seeds:
        runs["baseline"].append(dynamics._run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=args.sweeps,
            curvature_weight=dynamics.BASELINE_CURVATURE,
            device=args.device,
            record_query_clock=True,
        ))
        partial["completed_baseline_seeds"] = len(runs["baseline"])
        dynamics._write_json(output, partial)
        runs["candidate"].append(dynamics._run_one(
            target,
            queries,
            schema,
            marginals,
            seed=seed,
            rounds=args.rounds,
            temperature=args.temperature,
            sweeps=args.sweeps,
            curvature_weight=dynamics.CANDIDATE_CURVATURE,
            device=args.device,
            record_query_clock=True,
        ))
        partial["completed_candidate_seeds"] = len(runs["candidate"])
        dynamics._write_json(output, partial)

    replay_audit = _audit_replay(reference, runs)
    query_clock_gate = _query_clock_gate(runs, args.rounds, target)
    query_clock = _matched_clock_summary(
        runs["baseline"],
        runs["candidate"],
        "query_quadratic_variation",
    )
    changed_cells_clock = _matched_clock_summary(
        runs["baseline"], runs["candidate"], "changed_cells"
    )
    calendar = _calendar_tail_summary(
        runs["baseline"], runs["candidate"]
    )
    state_dependent_drift = _state_dependent_drift(
        runs["baseline"], runs["candidate"], len(queries)
    )
    gates = {
        "reference_sha256_matches": (
            reference_sha256 == FORMAL_REFERENCE_SHA256
        ),
        "gamma_zero_reference_preflight": preflight["passed"],
        "reference_replay_exact": replay_audit["passed"],
        "query_clock_complete_and_valid": query_clock_gate["passed"],
        "maximum_gain_identity_abs_error": query_clock_gate[
            "maximum_gain_identity_abs_error"
        ],
        "all_trajectories_complete": all(
            row["rounds_run"] == args.rounds
            for rows in runs.values() for row in rows
        ),
        "logit_clip_hits": sum(
            row["conditional_logit_clipped_count"]
            for rows in runs.values() for row in rows
        ),
        "all_conditionals_bidirectional": all(
            row["all_conditionals_bidirectional"]
            for rows in runs.values() for row in rows
        ),
    }
    diagnostic_gate_passed = (
        gates["reference_sha256_matches"]
        and gates["gamma_zero_reference_preflight"]
        and gates["reference_replay_exact"]
        and gates["query_clock_complete_and_valid"]
        and gates["maximum_gain_identity_abs_error"] <= 1e-12
        and gates["all_trajectories_complete"]
        and gates["logit_clip_hits"] == 0
        and gates["all_conditionals_bidirectional"]
    )
    classification = (
        _classify_multistep_effect(query_clock, calendar)
        if formal_protocol_matches and diagnostic_gate_passed
        else (
            "diagnostic_gate_failed"
            if formal_protocol_matches else "non_formal_run_no_decision"
        )
    )
    summary = {
        "experiment": "curvature_multistep_drift_query_clock",
        "issue": 27,
        "status": "complete",
        "formal_protocol_matches": formal_protocol_matches,
        "diagnostic_gate_passed": diagnostic_gate_passed,
        "classification": classification,
        "scope": "diagnostic_only_no_algorithm_or_control_flow_change",
        "dataset": "test_300x10",
        "n_records": dynamics.N_RECORDS,
        "n_queries": len(queries),
        "seeds": args.seeds,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "sweeps": args.sweeps,
        "reference_rounds": args.reference_rounds,
        "baseline_curvature_weight": dynamics.BASELINE_CURVATURE,
        "candidate_curvature_weight": dynamics.CANDIDATE_CURVATURE,
        "rho": dynamics.RHO,
        "eta": dynamics.ETA,
        "mu": dynamics.MU,
        "gibbs_logit_clip": dynamics.GIBBS_LOGIT_CLIP,
        "device": args.device,
        "real_data_access": "none",
        "environment": environment,
        "public_input_sha256": {
            str(path): dynamics._sha256_file(path)
            for path in (
                dynamics.SCHEMA_PATH,
                dynamics.QUERY_PATH,
                dynamics.MARGINALS_PATH,
            )
        },
        "reference_output": str(reference_path),
        "reference_output_sha256": reference_sha256,
        "original_issue24_decision": reference["decision"],
        "preflight": preflight,
        "gates": gates,
        "reference_replay_audit": replay_audit,
        "query_clock_gate": query_clock_gate,
        "calendar_round_clock": calendar,
        "query_quadratic_variation_clock": query_clock,
        "changed_cells_clock_posthoc_control": changed_cells_clock,
        "state_dependent_drift_descriptive": state_dependent_drift,
        "runs": runs,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    dynamics._write_json(output, summary)

    query_difference = query_clock["matched_mean_loss_difference"]
    calendar_difference = calendar["difference"]
    print("\n===== 曲率核多步漂移与内禀扩散时钟 =====")
    print(
        "按轮数末 250 轮 loss 差："
        f"{calendar_difference['mean']:+.6g}, "
        f"{calendar_difference['wins']}/"
        f"{calendar_difference['ties']}/"
        f"{calendar_difference['losses']}"
    )
    print(
        "匹配查询二次变差后的 loss 差："
        f"{query_difference['mean']:+.6g}, "
        f"95%区间={query_difference['mean_t_interval_95']}, "
        f"{query_difference['wins']}/"
        f"{query_difference['ties']}/"
        f"{query_difference['losses']}"
    )
    print(f"诊断门禁：{diagnostic_gate_passed}")
    print(f"预注册分类：{classification}")
    print(f"详细结果：{output}")


if __name__ == "__main__":
    main()
