"""非整数目标的 GPU 成本/轨迹验证；仅使用公开查询与合成初表，不是 DP 实验。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--scenario", choices=("active", "frozen"), required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.seed < 0:
        parser.error("轮数须为正，种子不能为负")
    code = args.code_root.resolve()
    commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(code), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("验证源码工作树不干净")
    if args.out.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{args.out}")
    sys.path.insert(0, str(code / "src"))
    import cupy as cp
    import numpy as np
    import resevo
    import resevo.gpukernel as gpu
    from resevo.batchkernel import evolve_batch, make_batch_provider
    from resevo.dataset import QuerySpec, StateRegistry, TableSchema, load_queries
    from resevo.editspace import tuples_from_ids

    if Path(resevo.__file__).resolve().parent != code / "src" / "resevo":
        raise RuntimeError("实际导入来源与指定工作树不一致")
    raw = args.queries.read_bytes()
    payload = json.loads(raw)
    n = payload["record_count"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("公开 record_count 必须为正整数")
    specs = load_queries(str(args.queries), expected_record_count=n)
    if not specs or any(
        len(spec.conditions) != 2 or any(
            c["operator"] != "==" or str(c["value"]) not in ("0", "1")
            for c in spec.conditions
        )
        for spec in specs
    ):
        raise ValueError("成本协议仅接受公开二值二阶等值查询")
    names = sorted({c["attribute"] for spec in specs for c in spec.conditions})
    positions = {name: i for i, name in enumerate(names)}
    domains = tuple(tuple(sorted({
        str(c["value"]) for spec in specs for c in spec.conditions if c["attribute"] == name
    })) for name in names)
    schema = TableSchema(tuple(names), domains)
    initial_rng = np.random.default_rng(args.seed)
    initial = np.column_stack([
        np.asarray(domain, dtype=np.int8)[initial_rng.integers(0, len(domain), n)]
        for domain in domains
    ])
    rows = [tuple(str(int(value)) for value in row) for row in initial]
    jitter = np.random.default_rng(900000 + args.seed)
    if args.scenario == "active":
        target = np.array([spec.result for spec in specs]) + jitter.normal(0.0, 4.0, len(specs))
    else:
        counts = []
        for spec in specs:
            c0, c1 = spec.conditions
            counts.append(np.count_nonzero(
                (initial[:, positions[c0["attribute"]]] == int(c0["value"]))
                & (initial[:, positions[c1["attribute"]]] == int(c1["value"]))
            ))
        target = np.asarray(counts) + jitter.uniform(-0.125, 0.125, len(specs))
    specs = [QuerySpec(spec.query_id, spec.conditions, float(value)) for spec, value in zip(specs, target)]
    weights = np.ones(len(specs))
    counters = dict(kernel_builds=0, exact_recheck_calls=0, exact_recheck_paths=0)
    original_build = gpu.GpuBatchContext.build

    def build(*values, **kwargs):
        counters["kernel_builds"] += 1
        return original_build(*values, **kwargs)

    gpu.GpuBatchContext.build = build
    exact = getattr(gpu, "exact_sparse_path_gains", None)
    if exact is not None:
        def recheck(paths, *values, **kwargs):
            counters["exact_recheck_calls"] += 1
            counters["exact_recheck_paths"] += len(paths)
            return exact(paths, *values, **kwargs)
        gpu.exact_sparse_path_gains = recheck

    def run(rounds):
        registry = StateRegistry(schema, specs)
        registry.set_lazy_features(True)
        ids = registry.register_table(rows)
        menu_rng = np.random.default_rng(100000 + args.seed)
        sample_rng = np.random.default_rng(300000 + args.seed)
        provider = make_batch_provider(
            registry, target, weights, menu_rng, pairing=False, defer_workload=True,
        )
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        result = evolve_batch(
            ids, rounds, sample_rng, provider, registry, backend="gpu",
            max_frozen_retries=args.rounds, gpu_entry_budget=40000000,
        )
        cp.cuda.Device().synchronize()
        return result, registry, time.perf_counter() - start, menu_rng.bit_generator.state, sample_rng.bit_generator.state

    run(2)
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    counters.update(kernel_builds=0, exact_recheck_calls=0, exact_recheck_paths=0)
    result, registry, seconds, menu_rng, sample_rng = run(args.rounds)
    records = [asdict(record) for record in result.records]
    for record in records:
        del record["elapsed"]
    if len(records) != args.rounds or result.stop_reason != "round_limit":
        raise AssertionError("没有跑满固定验证预算")
    if args.scenario == "frozen" and any(record["status"] != "no_positive_direction" for record in records):
        raise AssertionError("冻结控制场景出现了正方向")
    args.out.mkdir(parents=True)
    for name, table in (("initial.csv", rows), ("current.csv", tuples_from_ids(registry, result.state_ids))):
        with (args.out / name).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(schema.fields)
            writer.writerows(table)
    data = dict(
        code_commit=commit, seed=args.seed, scenario=args.scenario, rounds=args.rounds,
        query_sha256=hashlib.sha256(raw).hexdigest(), target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
        rows=n, fields=len(names), queries=len(specs), noninteger_targets=int(np.count_nonzero(target != np.rint(target))),
        seconds=seconds, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        gpu_pool_reserved_mib=cp.get_default_memory_pool().total_bytes() / 2**20,
        kernel_builds=counters["kernel_builds"], evaluated_paths=sum(record["candidates"] for record in records),
        frozen_rounds=sum(record["status"] == "no_positive_direction" for record in records),
        exact_recheck_available=exact is not None,
        exact_recheck_calls=counters["exact_recheck_calls"] if exact is not None else None,
        exact_recheck_paths=counters["exact_recheck_paths"] if exact is not None else None,
        stop_reason=result.stop_reason, records=records, menu_rng=menu_rng, sample_rng=sample_rng,
        python=sys.version, numpy=np.__version__, cupy=cp.__version__,
        cuda_driver=cp.cuda.runtime.driverGetVersion(),
        device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        environment={key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        parameters=dict(alpha=0.5, stay=0.9, damping=1.0, weights="ones", pairing=False, sparse="auto",
                        warmup_rounds=2, initialization="uniform_query_domains"),
    )
    (args.out / "result.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{commit[:7]} {args.scenario} seed={args.seed} rounds={len(records)} seconds={seconds:.3f}", flush=True)


if __name__ == "__main__":
    main()
