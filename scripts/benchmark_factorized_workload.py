"""在逐步等价硬门禁下比较旧逐行构造与预编译批量构造性能。

每个 seed/构造器组合在独立子进程运行；偶数 seed 先跑旧版，奇数 seed 先跑
新版。两侧都是相同的 8-sweep 因子 Gibbs 算法，唯一变量是因子构造路径。
"""

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from compare_factorized_gibbs_unfiltered import (
    GIBBS_LOGIT_CLIP,
    MARGINALS_PATH,
    QUERY_PATH,
    SCHEMA_PATH,
    _environment,
    _git_commit,
    _run_one,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries
from table_diffevo.schema import load_schema


BUILDERS = ("legacy_rowwise", "compiled_batch")
EQUIVALENCE_EXCLUDED_KEYS = {
    "factor_builder",
    "direction_elapsed_sec",
    "factor_build_elapsed_sec",
    "compiled_validation_elapsed_sec",
    "workload_compile_elapsed_sec",
    "factor_pipeline_elapsed_sec",
    "gibbs_sample_elapsed_sec",
    "elapsed_sec",
    "raw_elapsed_sec",
    "trajectory_audit_elapsed_sec",
    "condition_evaluation_batches",
    "compiled_unique_conditions",
    "process_run_elapsed_sec",
    "peak_rss_mib",
    "cuda_peak_allocated_mib",
    "cuda_peak_reserved_mib",
}


def _metric_summary(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.all(np.isfinite(array)):
        raise ValueError("性能指标必须是非空有限一维数组")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
        "values": array.tolist(),
    }


def _paired_performance(legacy, compiled, key):
    legacy_values = np.asarray([row[key] for row in legacy], dtype=float)
    compiled_values = np.asarray([row[key] for row in compiled], dtype=float)
    if np.any(legacy_values <= 0.0):
        raise ValueError(f"{key} 的 baseline 必须为正")
    relative_change_pct = (
        compiled_values / legacy_values - 1.0
    ) * 100.0
    reduction_pct = -relative_change_pct
    return {
        "legacy": _metric_summary(legacy_values),
        "compiled": _metric_summary(compiled_values),
        "relative_change_pct": _metric_summary(relative_change_pct),
        "reduction_pct": _metric_summary(reduction_pct),
        "wins": int(np.sum(compiled_values < legacy_values)),
        "ties": int(np.sum(compiled_values == legacy_values)),
        "losses": int(np.sum(compiled_values > legacy_values)),
    }


def _equivalence_report(legacy, compiled):
    failures = []
    if len(legacy) != len(compiled):
        failures.append({
            "seed": None,
            "key": "run_count",
            "legacy": len(legacy),
            "compiled": len(compiled),
        })
    for baseline, candidate in zip(legacy, compiled):
        if baseline["seed"] != candidate["seed"]:
            failures.append({
                "seed": None,
                "key": "seed_order",
                "legacy": baseline["seed"],
                "compiled": candidate["seed"],
            })
            continue
        keys = sorted(
            (set(baseline) | set(candidate)) - EQUIVALENCE_EXCLUDED_KEYS
        )
        for key in keys:
            if baseline.get(key) != candidate.get(key):
                failures.append({
                    "seed": baseline["seed"],
                    "key": key,
                    "legacy": baseline.get(key),
                    "compiled": candidate.get(key),
                })
                break
    return {
        "passed": not failures,
        "n_pairs": len(legacy),
        "failed_pairs": len(failures),
        "failures": failures,
        "checked_state_hashes": int(sum(
            len(row["state_sha256_history"]) for row in legacy
        )),
    }


def _memory_risk(legacy, compiled):
    rss_relative = []
    cuda_allocated_relative = []
    cuda_allocated_absolute = []
    cuda_reserved_relative = []
    cuda_reserved_absolute = []
    for baseline, candidate in zip(legacy, compiled):
        rss_relative.append(
            (candidate["peak_rss_mib"] / baseline["peak_rss_mib"] - 1.0)
            * 100.0
        )
        for key, relative, absolute in (
            (
                "cuda_peak_allocated_mib",
                cuda_allocated_relative,
                cuda_allocated_absolute,
            ),
            (
                "cuda_peak_reserved_mib",
                cuda_reserved_relative,
                cuda_reserved_absolute,
            ),
        ):
            baseline_cuda = baseline[key]
            candidate_cuda = candidate[key]
            absolute.append(candidate_cuda - baseline_cuda)
            if baseline_cuda > 0.0:
                relative.append(
                    (candidate_cuda / baseline_cuda - 1.0) * 100.0
                )
            elif candidate_cuda == 0.0:
                relative.append(0.0)
            else:
                relative.append(float("inf"))
    rss_relative = np.asarray(rss_relative, dtype=float)
    cuda_allocated_relative = np.asarray(
        cuda_allocated_relative, dtype=float
    )
    cuda_allocated_absolute = np.asarray(
        cuda_allocated_absolute, dtype=float
    )
    cuda_reserved_relative = np.asarray(
        cuda_reserved_relative, dtype=float
    )
    cuda_reserved_absolute = np.asarray(
        cuda_reserved_absolute, dtype=float
    )
    cuda_relative_finite = (
        np.all(np.isfinite(cuda_allocated_relative))
        and np.all(np.isfinite(cuda_reserved_relative))
    )
    return {
        "rss_relative_change_pct": _metric_summary(rss_relative),
        "cuda_allocated_relative_change_pct": (
            _metric_summary(cuda_allocated_relative)
            if np.all(np.isfinite(cuda_allocated_relative)) else {
                "values": [
                    value if np.isfinite(value) else None
                    for value in cuda_allocated_relative.tolist()
                ],
                "has_infinite": True,
            }
        ),
        "cuda_allocated_change_mib": _metric_summary(
            cuda_allocated_absolute
        ),
        "cuda_reserved_relative_change_pct": (
            _metric_summary(cuda_reserved_relative)
            if np.all(np.isfinite(cuda_reserved_relative)) else {
                "values": [
                    value if np.isfinite(value) else None
                    for value in cuda_reserved_relative.tolist()
                ],
                "has_infinite": True,
            }
        ),
        "cuda_reserved_change_mib": _metric_summary(
            cuda_reserved_absolute
        ),
        "rss_risk_over_10pct": bool(np.any(rss_relative > 10.0)),
        "cuda_risk_over_10pct_or_64mib": bool(
            not cuda_relative_finite
            or np.any(cuda_allocated_relative > 10.0)
            or np.any(cuda_allocated_absolute > 64.0)
            or np.any(cuda_reserved_relative > 10.0)
            or np.any(cuda_reserved_absolute > 64.0)
        ),
    }


def _load_inputs():
    schema = load_schema(SCHEMA_PATH)
    queries = load_queries(QUERY_PATH)
    target = np.asarray([query["result"] for query in queries])
    marginals = load_marginals(MARGINALS_PATH)
    return target, queries, schema, marginals


def _run_worker(args):
    output = Path(args.worker_output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"worker 输出已存在，不覆盖：{output}")
    target, queries, schema, marginals = _load_inputs()
    cuda_peak_allocated = 0.0
    cuda_peak_reserved = 0.0
    torch = None
    if args.device == "cuda":
        import torch as torch_module

        torch = torch_module
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    process_start = time.perf_counter()
    run = _run_one(
        target,
        queries,
        schema,
        marginals,
        seed=args.worker_seed,
        rounds=args.rounds,
        temperature=args.temperature,
        sweeps=args.sweeps,
        device=args.device,
        factor_builder=args.worker_builder,
        record_state_hashes=True,
    )
    if torch is not None:
        torch.cuda.synchronize()
        cuda_peak_allocated = (
            torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        )
        cuda_peak_reserved = (
            torch.cuda.max_memory_reserved() / (1024.0 ** 2)
        )
    run["process_run_elapsed_sec"] = time.perf_counter() - process_start
    run["peak_rss_mib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    )
    run["cuda_peak_allocated_mib"] = float(cuda_peak_allocated)
    run["cuda_peak_reserved_mib"] = float(cuda_peak_reserved)
    payload = {
        "git_commit": _git_commit(),
        "environment": _environment(args.device),
        "run": run,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _worker_command(args, seed, builder, output):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-builder",
        builder,
        "--worker-seed",
        str(seed),
        "--worker-output",
        str(output),
        "--rounds",
        str(args.rounds),
        "--temperature",
        str(args.temperature),
        "--sweeps",
        str(args.sweeps),
        "--device",
        args.device,
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def _run_driver(args):
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"输出文件已存在，不覆盖：{output}")
    worker_dir = output.parent / f"{output.stem}_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    runs = {builder: [] for builder in BUILDERS}
    execution_order = []
    commands = []
    worker_commits = []
    worker_environment = None
    repository_root = Path(__file__).resolve().parents[1]
    experiment_start = time.perf_counter()
    for seed in args.seeds:
        order = BUILDERS if seed % 2 == 0 else tuple(reversed(BUILDERS))
        for builder in order:
            worker_output = worker_dir / f"seed{seed}_{builder}.json"
            command = _worker_command(args, seed, builder, worker_output)
            print(
                f"运行 seed={seed} builder={builder}", flush=True
            )
            subprocess.run(command, check=True, cwd=repository_root)
            with worker_output.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            runs[builder].append(payload["run"])
            worker_commits.append(payload["git_commit"])
            if worker_environment is None:
                worker_environment = payload["environment"]
            execution_order.append({"seed": seed, "builder": builder})
            commands.append(command)

    for builder in BUILDERS:
        runs[builder].sort(key=lambda row: row["seed"])
    legacy = runs["legacy_rowwise"]
    compiled = runs["compiled_batch"]
    equivalence = _equivalence_report(legacy, compiled)
    performance = {
        key: _paired_performance(legacy, compiled, key)
        for key in (
            "factor_pipeline_elapsed_sec",
            "factor_build_elapsed_sec",
            "elapsed_sec",
            "raw_elapsed_sec",
            "peak_rss_mib",
            "cuda_peak_allocated_mib",
            "cuda_peak_reserved_mib",
        )
        if key not in (
            "cuda_peak_allocated_mib",
            "cuda_peak_reserved_mib",
        ) or all(row[key] > 0.0 for row in legacy)
    }
    memory = _memory_risk(legacy, compiled)
    factor_reduction = performance[
        "factor_pipeline_elapsed_sec"
    ]["reduction_pct"]["median"]
    elapsed_change = performance[
        "elapsed_sec"
    ]["relative_change_pct"]
    elapsed_reduction = -elapsed_change["median"]
    no_seed_slowdown = max(elapsed_change["values"]) <= 2.0
    driver_commit = _git_commit()
    worker_commit_aligned = all(
        commit == driver_commit for commit in worker_commits
    )
    gates = {
        "equivalence_passed": equivalence["passed"],
        "worker_commit_aligned": worker_commit_aligned,
        "factor_pipeline_median_reduction_at_least_50pct": (
            factor_reduction >= 50.0
        ),
        "generation_median_reduction_at_least_15pct": (
            elapsed_reduction >= 15.0
        ),
        "no_seed_slowdown_over_2pct": no_seed_slowdown,
        "rss_within_10pct": not memory["rss_risk_over_10pct"],
        "cuda_within_10pct_and_64mib": not memory[
            "cuda_risk_over_10pct_or_64mib"
        ],
    }
    if not equivalence["passed"]:
        decision = "equivalence_failed"
    elif all(gates.values()):
        decision = "performance_success"
    else:
        decision = "performance_not_supported"
    summary = {
        "experiment": "factorized_workload_builder_performance",
        "scope": "same_8_sweep_algorithm_only_factor_builder_changes",
        "dataset": "test_300x10",
        "rounds": args.rounds,
        "seeds": args.seeds,
        "temperature": args.temperature,
        "sweeps": args.sweeps,
        "gibbs_logit_clip": GIBBS_LOGIT_CLIP,
        "device": args.device,
        "git_commit": driver_commit,
        "worker_git_commits": worker_commits,
        "environment": worker_environment,
        "command_argv": sys.argv,
        "worker_commands": commands,
        "execution_order": execution_order,
        "equivalence": equivalence,
        "performance": performance,
        "memory": memory,
        "gates": gates,
        "decision": decision,
        "runs": runs,
        "elapsed_sec": time.perf_counter() - experiment_start,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"逐步等价门禁：{equivalence['passed']}")
    print(f"因子管线中位降幅：{factor_reduction:.2f}%")
    print(f"生成墙钟中位降幅：{elapsed_reduction:.2f}%")
    print(f"正式判断：{decision}")
    print(f"输出：{output}")
    print(f"SHA-256：{output_sha256}")
    if not equivalence["passed"]:
        raise RuntimeError("逐步等价门禁失败，停止性能结论")


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
        default=(
            "outputs/factorized_workload/"
            "formal_10seed_1000r_tau2_sweep8.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--worker-builder", choices=BUILDERS, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds 必须为正整数")
    if args.sweeps <= 0:
        parser.error("--sweeps 必须为正整数")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds 不得重复")
    if not np.isfinite(args.temperature) or args.temperature < 0.0:
        parser.error("--temperature 必须是非负有限数值")
    worker_mode = args.worker_builder is not None
    if worker_mode:
        if args.worker_seed is None or args.worker_output is None:
            parser.error("worker 模式必须提供 seed 和输出路径")
        _run_worker(args)
    else:
        _run_driver(args)


if __name__ == "__main__":
    main()
