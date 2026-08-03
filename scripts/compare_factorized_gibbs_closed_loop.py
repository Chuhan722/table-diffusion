"""比较因子 Gibbs 与独立单块核在标准接受闭环中的生成效果。

生成阶段只读取公开 schema、记录数、预定义查询、已发布 target、1-way marginal
和算法参数。全部 baseline/candidate 生成结束后才读取真实参考表并离线评价；真实表
不参与初始化、接受、早停、参数选择或 checkpoint 选择。
"""

import argparse
import contextlib
from datetime import datetime
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

from table_diffevo.evolution import run_evolution
from table_diffevo.factorized_diffusion import DEFAULT_LOGIT_CLIP
from table_diffevo.io import save_run
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


SCHEMA_PATH = Path("configs/test_300x10/schema.yaml")
QUERY_PATH = Path("configs/test_300x10/measured_50query.json")
MARGINALS_PATH = Path("configs/test_300x10/init_marginals.json")
REAL_DATA_PATH = Path("data/test_300x10/test_300x10.csv")
N_RECORDS = 300
FORMAL_SEEDS = list(range(20))
FORMAL_ROUNDS = 500
FORMAL_TEMPERATURE = 2.0
FORMAL_SWEEPS = 8


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame_sha256(frame):
    return _sha256_bytes(frame.to_csv(index=False).encode("utf-8"))


def _git_text(*args):
    completed = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout.strip()


def _environment_snapshot(device):
    commit_code, commit = _git_text("rev-parse", "HEAD")
    status_code, status = _git_text("status", "--porcelain")
    snapshot = {
        "started_at": datetime.now().astimezone().isoformat(),
        "command": [sys.executable, *sys.argv],
        "git_commit": commit if commit_code == 0 else None,
        "git_worktree_clean": status_code == 0 and status == "",
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch
    except ImportError:
        snapshot.update({
            "torch": None,
            "torch_cuda_runtime": None,
            "cuda_available": False,
            "gpu": None,
        })
    else:
        cuda_available = bool(torch.cuda.is_available())
        snapshot.update({
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "gpu": (
                torch.cuda.get_device_name(0)
                if device == "cuda" and cuda_available else None
            ),
        })
    return snapshot


def _mean_or_zero(values):
    return float(np.mean(values)) if len(values) else 0.0


def _mean_without_none(values):
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else 0.0


def _first_attempts(nested):
    return [float(attempts[0]) for attempts in nested if attempts]


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
    max_factor_order,
    device,
):
    call_start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        best, diagnostics = run_evolution(
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
            tol=1e-9,
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
            residual_directed_diffusion=True,
            diffusion_direction_strength=temperature,
            diffusion_direction_normalization="initial_rms",
            factorized_gibbs_sweeps=sweeps,
            factorized_gibbs_max_order=max_factor_order,
            factorized_gibbs_logit_clip=DEFAULT_LOGIT_CLIP,
        )
    generation_call_wall_sec = time.perf_counter() - call_start

    losses = np.asarray(diagnostics["loss_history"], dtype=float)
    raw_gains = np.asarray(
        _first_attempts(diagnostics["raw_proposal_gain_history"]),
        dtype=float,
    )
    late_start = 3 * len(raw_gains) // 4
    late_raw_gains = raw_gains[late_start:]
    positive_gains = raw_gains[raw_gains > 0.0]
    negative_gains = raw_gains[raw_gains < 0.0]
    accept_history = diagnostics["accept_history"]
    variant = "independent_0_sweeps" if sweeps == 0 else (
        f"factorized_{sweeps}_sweeps"
    )
    result = {
        "seed": int(seed),
        "variant": variant,
        "temperature": float(temperature),
        "sweeps": int(sweeps),
        "gibbs_logit_clip": float(DEFAULT_LOGIT_CLIP),
        "rounds_run": int(diagnostics["rounds_run"]),
        "stopped_early": bool(diagnostics["stopped_early"]),
        "initial_loss": float(losses[0]),
        "best_loss": float(diagnostics["best_loss"]),
        "normalized_l1_error": float(
            diagnostics["normalized_l1_error"]
        ),
        "accept_rate": _mean_or_zero(accept_history),
        "late_accept_rate": _mean_or_zero(
            accept_history[3 * len(accept_history) // 4:]
        ),
        "late_loss_improvement": (
            float(losses[3 * len(losses) // 4] - diagnostics["best_loss"])
            if len(losses) else 0.0
        ),
        "raw_proposal_gain_mean": _mean_or_zero(raw_gains),
        "raw_proposal_positive_rate": _mean_or_zero(raw_gains > 0.0),
        "raw_positive_gain_mean": _mean_or_zero(positive_gains),
        "raw_negative_gain_mean": _mean_or_zero(negative_gains),
        "late_raw_proposal_gain_mean": _mean_or_zero(late_raw_gains),
        "late_raw_proposal_positive_rate": _mean_or_zero(
            late_raw_gains > 0.0
        ),
        "copy_direction_mean": _mean_without_none(
            diagnostics["copy_direction_mean_history"]
        ),
        "copy_direction_positive_rate": _mean_without_none(
            diagnostics["copy_direction_positive_rate_history"]
        ),
        "copy_direction_negative_rate": _mean_without_none(
            diagnostics["copy_direction_negative_rate_history"]
        ),
        "negative_direction_copy_probability": _mean_without_none(
            diagnostics["negative_direction_copy_probability_history"]
        ),
        "positive_direction_copy_probability": _mean_without_none(
            diagnostics["positive_direction_copy_probability_history"]
        ),
        "copy_probability_entropy": _mean_without_none(
            diagnostics["copy_probability_entropy_history"]
        ),
        "direction_reference_scale": float(
            diagnostics["direction_reference_scale"] or 0.0
        ),
        "state_evaluation_count": int(
            diagnostics["state_evaluation_count"]
        ),
        "distance_evaluation_count": int(
            diagnostics["distance_evaluation_count"]
        ),
        "direction_evaluation_count": int(
            diagnostics["direction_evaluation_count"]
        ),
        "factorized_gibbs_active_rows": int(
            diagnostics["factorized_gibbs_active_rows"]
        ),
        "factorized_gibbs_active_blocks": int(
            diagnostics["factorized_gibbs_active_blocks"]
        ),
        "factorized_gibbs_factor_count": int(
            diagnostics["factorized_gibbs_factor_count"]
        ),
        "factorized_gibbs_factor_table_entries": int(
            diagnostics["factorized_gibbs_factor_table_entries"]
        ),
        "factorized_gibbs_microsteps": int(
            diagnostics["factorized_gibbs_microsteps"]
        ),
        "run_evolution_loop_sec": float(diagnostics["elapsed_sec"]),
        "generation_call_wall_sec": generation_call_wall_sec,
        "direction_evaluation_elapsed_sec": float(
            diagnostics["direction_evaluation_elapsed_sec"]
        ),
        "factorized_gibbs_factor_build_elapsed_sec": float(
            diagnostics["factorized_gibbs_factor_build_elapsed_sec"]
        ),
        "factorized_gibbs_sample_elapsed_sec": float(
            diagnostics["factorized_gibbs_sample_elapsed_sec"]
        ),
        "initial_table_sha256": diagnostics["initial_table_sha256"],
        "synthetic_table_sha256": _frame_sha256(best),
        "primary_rng_post_initialization_state_sha256": diagnostics[
            "primary_rng_post_initialization_state_sha256"
        ],
        "primary_rng_state_sha256": diagnostics[
            "primary_rng_state_sha256"
        ],
        "factorized_gibbs_initial_rng_state_sha256": diagnostics[
            "factorized_gibbs_initial_rng_state_sha256"
        ],
        "factorized_gibbs_rng_state_sha256": diagnostics[
            "factorized_gibbs_rng_state_sha256"
        ],
    }
    print(
        f"seed={seed:02d} {variant:<22} "
        f"loss={result['best_loss']:.2f} "
        f"raw_gain={result['raw_proposal_gain_mean']:+.2f} "
        f"accept={result['accept_rate']:.1%} "
        f"wall={generation_call_wall_sec:.2f}s",
        flush=True,
    )
    return best, diagnostics, result


def _discretization_domains(marginals):
    domains = {}
    for attribute, specification in marginals["attributes"].items():
        if specification["type"] == "numeric":
            domains[attribute] = [
                f"[{int(lower)},{int(upper)}]"
                for lower, upper in specification["bins"]
            ]
        else:
            domains[attribute] = [
                str(value) for value in specification["values"]
            ]
    return domains


def _discretize(frame, marginals):
    result = pd.DataFrame(index=frame.index)
    for attribute, specification in marginals["attributes"].items():
        if specification["type"] == "numeric":
            values = pd.to_numeric(frame[attribute]).to_numpy()
            encoded = np.full(len(frame), None, dtype=object)
            for lower, upper in specification["bins"]:
                selected = (values >= lower) & (values <= upper)
                encoded[selected] = f"[{int(lower)},{int(upper)}]"
            if any(value is None for value in encoded):
                raise ValueError(f"{attribute} 存在公开分箱范围外的值")
            result[attribute] = encoded
        else:
            result[attribute] = frame[attribute].map(str)
            allowed = {str(value) for value in specification["values"]}
            observed = set(result[attribute].unique())
            if not observed <= allowed:
                raise ValueError(
                    f"{attribute} 存在 schema 外取值：{sorted(observed - allowed)}"
                )
    return result


def _condition_cell_value(condition, marginal_specification):
    operator = condition["operator"]
    if marginal_specification["type"] == "categorical":
        if operator != "==":
            return None
        return str(condition["value"])

    bins = marginal_specification["bins"]
    if operator == "between":
        lower = condition["lower"]
        upper = condition["upper"]
        matches = [
            (bin_lower, bin_upper)
            for bin_lower, bin_upper in bins
            if bin_lower == lower and bin_upper == upper
        ]
    elif operator == ">=":
        lower = condition["value"]
        maximum = max(bin_upper for _, bin_upper in bins)
        matches = [
            (bin_lower, bin_upper)
            for bin_lower, bin_upper in bins
            if bin_lower == lower and bin_upper == maximum
        ]
    else:
        matches = []
    if len(matches) != 1:
        return None
    return f"[{int(matches[0][0])},{int(matches[0][1])}]"


def _measured_cell_keys(queries, marginals, order):
    measured = set()
    specifications = marginals["attributes"]
    for query in queries:
        conditions = query["conditions"]
        if len(conditions) != order:
            continue
        key = []
        for condition in conditions:
            attribute = condition["attribute"]
            value = _condition_cell_value(
                condition, specifications[attribute]
            )
            if value is None:
                key = []
                break
            key.append((attribute, value))
        if len(key) == order:
            measured.add(tuple(sorted(key)))
    return measured


def _marginal_cell_error(
    reference,
    synthetic,
    domains,
    order,
    excluded=None,
):
    excluded = excluded or set()
    errors = []
    for attributes in itertools.combinations(reference.columns, order):
        index = pd.MultiIndex.from_product(
            [domains[attribute] for attribute in attributes],
            names=attributes,
        )
        reference_counts = (
            reference.groupby(list(attributes), dropna=False)
            .size()
            .reindex(index, fill_value=0)
        )
        synthetic_counts = (
            synthetic.groupby(list(attributes), dropna=False)
            .size()
            .reindex(index, fill_value=0)
        )
        for state, reference_count, synthetic_count in zip(
            index,
            reference_counts.to_numpy(),
            synthetic_counts.to_numpy(),
        ):
            key = tuple(sorted(zip(attributes, map(str, state))))
            if key in excluded:
                continue
            errors.append(
                abs(float(reference_count) - float(synthetic_count))
                / len(reference)
            )
    values = np.asarray(errors, dtype=float)
    return {
        "n_queries": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def _joint_metrics(reference, synthetic):
    reference_counts = reference.value_counts(sort=False)
    synthetic_counts = synthetic.value_counts(sort=False)
    support = reference_counts.index.union(synthetic_counts.index)
    reference_probability = (
        reference_counts.reindex(support, fill_value=0).to_numpy(dtype=float)
        / len(reference)
    )
    synthetic_probability = (
        synthetic_counts.reindex(support, fill_value=0).to_numpy(dtype=float)
        / len(synthetic)
    )
    reference_only = synthetic_probability == 0.0
    synthetic_only = reference_probability == 0.0
    return {
        "tvd": float(
            0.5 * np.abs(reference_probability - synthetic_probability).sum()
        ),
        "n_unique": int(len(synthetic_counts)),
        "dup_rate": float(1.0 - len(synthetic_counts) / len(synthetic)),
        "support_overlap": int(
            len(reference_counts.index.intersection(synthetic_counts.index))
        ),
        "missing_reference_mass": float(
            reference_probability[reference_only].sum()
        ),
        "novel_synthetic_mass": float(
            synthetic_probability[synthetic_only].sum()
        ),
    }


def _offline_metrics(
    reference,
    synthetic,
    marginals,
    domains,
    measured_triples,
):
    reference_discrete = _discretize(reference, marginals)
    synthetic_discrete = _discretize(synthetic, marginals)
    return {
        "unmeasured_3way": _marginal_cell_error(
            reference_discrete,
            synthetic_discrete,
            domains,
            3,
            excluded=measured_triples,
        ),
        "unmeasured_4way": _marginal_cell_error(
            reference_discrete,
            synthetic_discrete,
            domains,
            4,
        ),
        "raw_joint": _joint_metrics(reference, synthetic),
        "binned_joint": _joint_metrics(
            reference_discrete, synthetic_discrete
        ),
    }


def _aggregate(rows, metric):
    values = np.asarray([row[metric] for row in rows], dtype=float)
    if len(values) == 0:
        raise ValueError(f"{metric} 没有可汇总记录")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{metric} 包含非有限值")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        "values": values.tolist(),
    }


def _paired(candidate, baseline, metric, lower_is_better):
    candidate_values = np.asarray(
        [row[metric] for row in candidate], dtype=float
    )
    baseline_values = np.asarray(
        [row[metric] for row in baseline], dtype=float
    )
    difference = candidate_values - baseline_values
    n = len(difference)
    if n == 0 or len(baseline_values) != n:
        raise ValueError(f"{metric} 的配对记录数量无效")
    if not (
        np.all(np.isfinite(candidate_values))
        and np.all(np.isfinite(baseline_values))
    ):
        raise ValueError(f"{metric} 的配对值必须全部有限")
    mean_difference = float(difference.mean())
    difference_std = float(difference.std(ddof=1)) if n > 1 else 0.0
    zero_variance_nonzero_difference = (
        n > 1 and difference_std == 0.0 and mean_difference != 0.0
    )
    if n > 1 and difference_std > 0.0:
        statistic, p_value = stats.ttest_rel(
            candidate_values, baseline_values
        )
        critical = float(stats.t.ppf(0.975, n - 1))
        half_width = critical * difference_std / np.sqrt(n)
        statistic = (
            float(statistic) if np.isfinite(statistic) else None
        )
        p_value = float(p_value) if np.isfinite(p_value) else None
    else:
        half_width = 0.0
        statistic = None
        p_value = 0.0 if zero_variance_nonzero_difference else None
    better = difference < 0.0 if lower_is_better else difference > 0.0
    worse = difference > 0.0 if lower_is_better else difference < 0.0
    baseline_mean = float(baseline_values.mean())
    candidate_mean = float(candidate_values.mean())
    return {
        "n": n,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
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
        "paired_t": statistic,
        "paired_p": p_value,
        "zero_variance_nonzero_difference": (
            zero_variance_nonzero_difference
        ),
        "wins": int(np.sum(better)),
        "ties": int(np.sum(difference == 0.0)),
        "losses": int(np.sum(worse)),
        "differences": difference.tolist(),
    }


def _flatten_offline(record, offline):
    record.update({
        "unmeasured_3way_l1": offline["unmeasured_3way"]["mean"],
        "unmeasured_3way_median_l1": offline[
            "unmeasured_3way"
        ]["median"],
        "unmeasured_3way_p90_l1": offline["unmeasured_3way"]["p90"],
        "unmeasured_3way_max_l1": offline["unmeasured_3way"]["max"],
        "unmeasured_3way_query_count": offline[
            "unmeasured_3way"
        ]["n_queries"],
        "unmeasured_4way_l1": offline["unmeasured_4way"]["mean"],
        "unmeasured_4way_median_l1": offline[
            "unmeasured_4way"
        ]["median"],
        "unmeasured_4way_p90_l1": offline["unmeasured_4way"]["p90"],
        "unmeasured_4way_max_l1": offline["unmeasured_4way"]["max"],
        "unmeasured_4way_query_count": offline[
            "unmeasured_4way"
        ]["n_queries"],
        "raw_joint_tvd": offline["raw_joint"]["tvd"],
        "raw_n_unique": offline["raw_joint"]["n_unique"],
        "raw_dup_rate": offline["raw_joint"]["dup_rate"],
        "raw_support_overlap": offline["raw_joint"]["support_overlap"],
        "raw_missing_reference_mass": offline[
            "raw_joint"
        ]["missing_reference_mass"],
        "raw_novel_synthetic_mass": offline[
            "raw_joint"
        ]["novel_synthetic_mass"],
        "binned_joint_tvd": offline["binned_joint"]["tvd"],
        "binned_n_unique": offline["binned_joint"]["n_unique"],
        "binned_dup_rate": offline["binned_joint"]["dup_rate"],
        "binned_support_overlap": offline[
            "binned_joint"
        ]["support_overlap"],
        "binned_missing_reference_mass": offline[
            "binned_joint"
        ]["missing_reference_mass"],
        "binned_novel_synthetic_mass": offline[
            "binned_joint"
        ]["novel_synthetic_mass"],
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_SEEDS)
    parser.add_argument("--rounds", type=int, default=FORMAL_ROUNDS)
    parser.add_argument(
        "--temperature", type=float, default=FORMAL_TEMPERATURE
    )
    parser.add_argument("--sweeps", type=int, default=FORMAL_SWEEPS)
    parser.add_argument("--max-factor-order", type=int, default=3)
    parser.add_argument(
        "--device", choices=["cuda", "cpu", "numpy"], default="cuda"
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/factorized_gibbs_closed_loop/formal",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds 必须为正数")
    if (
        not args.seeds
        or len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error("--seeds 必须是非空、非负且无重复的列表")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    if isinstance(args.sweeps, bool) or args.sweeps <= 0:
        parser.error("--sweeps 必须是正整数")
    if not 1 <= args.max_factor_order <= 8:
        parser.error("--max-factor-order 必须在 1..8 内")

    formal_protocol_matches = (
        args.seeds == FORMAL_SEEDS
        and args.rounds == FORMAL_ROUNDS
        and args.temperature == FORMAL_TEMPERATURE
        and args.sweeps == FORMAL_SWEEPS
        and args.max_factor_order == 3
        and args.device == "cuda"
    )

    output_dir = Path(args.output_dir)
    output_has_contents = (
        output_dir.exists() and any(output_dir.iterdir())
    )
    if output_has_contents and (
        formal_protocol_matches or not args.overwrite
    ):
        raise FileExistsError(f"输出目录已存在且非空，不覆盖：{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = _environment_snapshot(args.device)
    if formal_protocol_matches and not environment["git_worktree_clean"]:
        raise RuntimeError("正式协议要求 tracked 工作树干净")
    if args.device == "cuda" and not environment["cuda_available"]:
        raise RuntimeError("请求 CUDA，但当前环境没有可用 CUDA 设备")
    schema = load_schema(str(SCHEMA_PATH))
    queries = load_queries(str(QUERY_PATH))
    target = np.asarray([query["result"] for query in queries], dtype=float)
    marginals = load_marginals(str(MARGINALS_PATH))
    if (
        len(queries) != 50
        or target.shape != (50,)
        or not np.all(np.isfinite(target))
        or len(schema.attribute_names()) != 10
        or marginals.get("n_records") != N_RECORDS
        or set(marginals.get("attributes", {}))
        != set(schema.attribute_names())
    ):
        raise ValueError("test_300x10 的公开输入与正式协议不一致")
    public_input_hashes = {
        str(path): _sha256_file(path)
        for path in (SCHEMA_PATH, QUERY_PATH, MARGINALS_PATH)
    }

    runs = {
        "independent_0_sweeps": [],
        f"factorized_{args.sweeps}_sweeps": [],
    }
    tables = {name: [] for name in runs}
    all_generation_start = time.perf_counter()
    for seed_position, seed in enumerate(args.seeds):
        order = [0, args.sweeps]
        if seed % 2 == 1:
            order.reverse()
        for sweeps in order:
            best, diagnostics, record = _run_one(
                target,
                queries,
                schema,
                marginals,
                seed=seed,
                rounds=args.rounds,
                temperature=args.temperature,
                sweeps=sweeps,
                max_factor_order=args.max_factor_order,
                device=args.device,
            )
            variant = record["variant"]
            run_dir = output_dir / "runs" / f"seed_{seed:02d}" / variant
            save_run(best, diagnostics, run_dir=str(run_dir))
            record["run_dir"] = str(run_dir.relative_to(output_dir))
            record["run_order_within_seed"] = order.index(sweeps)
            record["seed_position"] = seed_position
            runs[variant].append(record)
            tables[variant].append(best)
    all_generation_elapsed_sec = time.perf_counter() - all_generation_start

    baseline_name = "independent_0_sweeps"
    candidate_name = f"factorized_{args.sweeps}_sweeps"
    for name in runs:
        paired = sorted(
            zip(runs[name], tables[name]), key=lambda item: item[0]["seed"]
        )
        runs[name] = [item[0] for item in paired]
        tables[name] = [item[1] for item in paired]

    baseline_runs = runs[baseline_name]
    candidate_runs = runs[candidate_name]
    expected_seeds = sorted(args.seeds)
    if (
        [record["seed"] for record in baseline_runs] != expected_seeds
        or [record["seed"] for record in candidate_runs] != expected_seeds
    ):
        raise RuntimeError("baseline/candidate 的配对 seed 不完整或未对齐")
    initial_tables_aligned = all(
        baseline["initial_table_sha256"]
        == candidate["initial_table_sha256"]
        for baseline, candidate in zip(baseline_runs, candidate_runs)
    )
    post_initialization_rng_aligned = all(
        baseline["primary_rng_post_initialization_state_sha256"]
        == candidate["primary_rng_post_initialization_state_sha256"]
        for baseline, candidate in zip(baseline_runs, candidate_runs)
    )
    direction_scale_aligned = all(
        baseline["direction_reference_scale"]
        == candidate["direction_reference_scale"]
        for baseline, candidate in zip(baseline_runs, candidate_runs)
    )
    equal_round_pairs = [
        (baseline, candidate)
        for baseline, candidate in zip(baseline_runs, candidate_runs)
        if baseline["rounds_run"] == candidate["rounds_run"]
    ]
    same_rounds_all_seeds = len(equal_round_pairs) == len(args.seeds)
    primary_rng_aligned_for_equal_round_pairs = all(
        baseline["primary_rng_state_sha256"]
        == candidate["primary_rng_state_sha256"]
        for baseline, candidate in equal_round_pairs
    )
    primary_rng_aligned_all_seeds = (
        same_rounds_all_seeds
        and primary_rng_aligned_for_equal_round_pairs
    )

    # 隐私边界：只有全部生成调用完成后，才从这里开始读取真实参考表。
    offline_started_at = datetime.now().astimezone().isoformat()
    offline_start = time.perf_counter()
    reference = pd.read_csv(REAL_DATA_PATH)
    columns = schema.attribute_names()
    reference = reference[columns]
    if len(reference) != N_RECORDS:
        raise ValueError("离线真实参考表行数与公开记录数不一致")
    domains = _discretization_domains(marginals)
    measured_triples = _measured_cell_keys(queries, marginals, order=3)
    for name in runs:
        for record, synthetic in zip(runs[name], tables[name]):
            offline = _offline_metrics(
                reference,
                synthetic[columns],
                marginals,
                domains,
                measured_triples,
            )
            _flatten_offline(record, offline)
            run_dir = output_dir / record["run_dir"]
            with (run_dir / "evaluation.json").open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump(
                    {"run": record, "offline": offline},
                    handle,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
    offline_elapsed_sec = time.perf_counter() - offline_start

    metric_directions = {
        "best_loss": True,
        "normalized_l1_error": True,
        "accept_rate": False,
        "late_accept_rate": False,
        "late_loss_improvement": False,
        "raw_proposal_gain_mean": False,
        "raw_proposal_positive_rate": False,
        "raw_positive_gain_mean": False,
        "raw_negative_gain_mean": False,
        "late_raw_proposal_gain_mean": False,
        "late_raw_proposal_positive_rate": False,
        "raw_joint_tvd": True,
        "binned_joint_tvd": True,
        "unmeasured_3way_l1": True,
        "unmeasured_3way_median_l1": True,
        "unmeasured_3way_p90_l1": True,
        "unmeasured_3way_max_l1": True,
        "unmeasured_4way_l1": True,
        "unmeasured_4way_median_l1": True,
        "unmeasured_4way_p90_l1": True,
        "unmeasured_4way_max_l1": True,
        "raw_n_unique": False,
        "raw_dup_rate": True,
        "raw_support_overlap": False,
        "raw_missing_reference_mass": True,
        "raw_novel_synthetic_mass": True,
        "binned_n_unique": False,
        "binned_dup_rate": True,
        "binned_support_overlap": False,
        "binned_missing_reference_mass": True,
        "binned_novel_synthetic_mass": True,
        "state_evaluation_count": True,
        "distance_evaluation_count": True,
        "direction_evaluation_count": True,
        "factorized_gibbs_active_rows": True,
        "factorized_gibbs_active_blocks": True,
        "factorized_gibbs_factor_count": True,
        "factorized_gibbs_factor_table_entries": True,
        "factorized_gibbs_microsteps": True,
        "run_evolution_loop_sec": True,
        "generation_call_wall_sec": True,
        "direction_evaluation_elapsed_sec": True,
        "factorized_gibbs_factor_build_elapsed_sec": True,
        "factorized_gibbs_sample_elapsed_sec": True,
    }
    aggregate = {
        name: {
            metric: _aggregate(rows, metric)
            for metric in metric_directions
        }
        for name, rows in runs.items()
    }
    comparisons = {
        metric: _paired(
            candidate_runs,
            baseline_runs,
            metric,
            lower_is_better,
        )
        for metric, lower_is_better in metric_directions.items()
    }

    primary = comparisons["best_loss"]
    interval = primary["difference_95pct_t_interval"]
    if not formal_protocol_matches:
        primary_decision = "non_formal_run_no_decision"
    elif (
        primary["mean_difference"] < 0.0
        and interval[1] < 0.0
        and primary["wins"] >= 14
    ):
        primary_decision = "supports_candidate"
    elif primary["mean_difference"] < 0.0:
        primary_decision = "inconclusive"
    else:
        primary_decision = "does_not_support_candidate"

    relative_risk_metrics = (
        "binned_joint_tvd",
        "unmeasured_3way_l1",
        "unmeasured_4way_l1",
    )
    quality_regression_flags = {
        metric: (
            comparisons[metric]["relative_aggregate_change_pct"] is not None
            and comparisons[metric]["relative_aggregate_change_pct"] > 5.0
        )
        for metric in relative_risk_metrics
    }
    unique_change = comparisons["raw_n_unique"][
        "relative_aggregate_change_pct"
    ]
    quality_regression_flags["raw_n_unique_drop_over_5pct"] = (
        unique_change is not None and unique_change < -5.0
    )

    summary = {
        "experiment": "factorized_gibbs_standard_acceptance_closed_loop",
        "issue": 16,
        "scope": "fixed_workload_exact_target_no_noise",
        "baseline": baseline_name,
        "candidate": candidate_name,
        "only_algorithm_variable": (
            f"factorized_gibbs_sweeps_0_vs_{args.sweeps}"
        ),
        "primary_endpoint": (
            f"best_workload_loss_after_{args.rounds}_rounds"
        ),
        "primary_decision_rule": (
            "mean difference < 0, paired 95% interval upper < 0, "
            "and at least 14/20 wins"
        ),
        "primary_decision": primary_decision,
        "secondary_p_values_are_descriptive": True,
        "formal_protocol_matches": formal_protocol_matches,
        "dataset": "test_300x10",
        "n_records": N_RECORDS,
        "n_rounds": args.rounds,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "candidate_sweeps": args.sweeps,
        "max_factor_order": args.max_factor_order,
        "gibbs_logit_clip": float(DEFAULT_LOGIT_CLIP),
        "device": args.device,
        "generation_order": (
            "even seeds baseline first; odd seeds candidate first"
        ),
        "real_data_boundary": (
            "loaded only after every run_evolution call; offline evaluation "
            "never affects generation, acceptance, stopping, or selection"
        ),
        "real_data_loaded_after_all_generation": True,
        "offline_started_at": offline_started_at,
        "environment": environment,
        "public_input_sha256": public_input_hashes,
        "offline_reference_sha256": _sha256_file(REAL_DATA_PATH),
        "measured_3way_cells_excluded": len(measured_triples),
        "initial_tables_aligned_all_seeds": initial_tables_aligned,
        "post_initialization_primary_rng_aligned_all_seeds": (
            post_initialization_rng_aligned
        ),
        "direction_reference_scale_aligned_all_seeds": (
            direction_scale_aligned
        ),
        "same_rounds_all_seeds": same_rounds_all_seeds,
        "equal_round_pair_count": len(equal_round_pairs),
        "primary_rng_aligned_for_all_equal_round_pairs": (
            primary_rng_aligned_for_equal_round_pairs
        ),
        "primary_rng_aligned_all_seeds": primary_rng_aligned_all_seeds,
        "all_generation_elapsed_sec": all_generation_elapsed_sec,
        "offline_evaluation_elapsed_sec": offline_elapsed_sec,
        "quality_regression_flags": quality_regression_flags,
        "runs": runs,
        "aggregate": aggregate,
        "candidate_vs_baseline": comparisons,
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("\n===== 标准接受闭环汇总 =====")
    for metric in (
        "best_loss",
        "normalized_l1_error",
        "unmeasured_3way_l1",
        "unmeasured_4way_l1",
        "binned_joint_tvd",
        "raw_n_unique",
        "generation_call_wall_sec",
    ):
        comparison = comparisons[metric]
        print(
            f"{metric:<32} "
            f"{comparison['baseline_mean']:.6g} -> "
            f"{comparison['candidate_mean']:.6g} "
            f"({comparison['wins']}/{comparison['ties']}/"
            f"{comparison['losses']})"
        )
    print(f"主终点判断：{primary_decision}")
    print(f"初始表逐种子对齐：{initial_tables_aligned}")
    print(f"两侧全部跑相同轮数：{same_rounds_all_seeds}")
    print(f"主 RNG 端点逐种子对齐：{primary_rng_aligned_all_seeds}")
    print(f"详细结果：{summary_path}")


if __name__ == "__main__":
    main()
