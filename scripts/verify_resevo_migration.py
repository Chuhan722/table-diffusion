"""迁移对拍：独立进程加载指定工作树，仅使用公开输入，不读取原始参考表。"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--code-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--seed", type=int, required=True)
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--small", choices=("rows", "groups", "batch", "gpu"))
mode.add_argument("--dataset", choices=("nltcs", "plants"))
parser.add_argument("--variant", choices=("full", "working"))
parser.add_argument("--public-inputs", type=Path)
parser.add_argument("--baseline-output", type=Path)
parser.add_argument("--reference-api", action="store_true", help="用源版本基础 API 生成参考结果，不依赖新 CLI")
args = parser.parse_args()
code = args.code_root.resolve()
sys.path.insert(0, str(code / "src"))

import numpy as np
import resevo
from resevo.batchkernel import evolve_batch, make_batch_provider
from resevo.dataset import QuerySpec, StateRegistry, TableSchema, load_queries, target_from_specs
from resevo.editspace import make_edit_provider, tuples_from_ids
from resevo.engine import evolve
from resevo.grouping import evolve_grouped, make_grouped_provider
from resevo.initialization import derive_first_order, sample_initial_rows

if Path(resevo.__file__).resolve().parent != code / "src" / "resevo":
    raise RuntimeError("实际导入的 resevo 不是指定工作树")
commit = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
if subprocess.check_output(["git", "-C", str(code), "status", "--porcelain"], text=True).strip():
    raise RuntimeError(f"工作树不干净，拒绝记录不可复现的实现版本：{code}")
if args.out.exists():
    raise FileExistsError(f"拒绝覆盖已有结果：{args.out}")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def records(result):
    out = []
    for record in result.records:
        row = asdict(record)
        del row["elapsed"]
        if math.isnan(row["moved_rows"]):
            row["moved_rows"] = None  # 行/组旧日志未记录该诊断，不将其解释为零。
        out.append(row)
    return out


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_table(path, schema, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(schema.fields)
        writer.writerows(rows)


def small():
    schema = TableSchema(("a", "b"), (("0", "1"), ("0", "1")))
    ca = {"attribute": "a", "operator": "==", "value": "1"}
    cb = {"attribute": "b", "operator": "==", "value": "1"}
    specs = [QuerySpec("a", (ca,), 3.0), QuerySpec("b", (cb,), 1.0), QuerySpec("ab", (ca, cb), 1.0)]
    target, weights = target_from_specs(specs), np.ones(3)
    registry = StateRegistry(schema, specs)
    registry.set_lazy_features(args.small == "gpu")
    ids = registry.register_table([("0", "0"), ("0", "1"), ("1", "0"), ("1", "1")])
    menu_rng = np.random.default_rng(100000 + args.seed)
    sample_rng = np.random.default_rng(300000 + args.seed)
    kwargs = dict(damping=0.5, max_expected_rows=0.3, max_frozen_retries=6)
    if args.small == "rows":
        provider = make_edit_provider(registry, target, weights, menu_rng)
        result = evolve(None, ids, 50, sample_rng, support_provider=provider, **kwargs)
    elif args.small == "groups":
        provider = make_grouped_provider(registry, target, weights, menu_rng, pairing=False)
        result = evolve_grouped(ids, 50, sample_rng, provider, **kwargs)
    else:
        provider = make_batch_provider(registry, target, weights, menu_rng,
                                       defer_workload=args.small == "gpu", pairing=False)
        result = evolve_batch(ids, 50, sample_rng, provider, registry,
                              backend="gpu" if args.small == "gpu" else "cpu", **kwargs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump(args.out, dict(
        code_commit=commit, path=args.small, seed=args.seed, parameters=kwargs,
        rows=tuples_from_ids(registry, result.state_ids), state_ids=result.state_ids.tolist(),
        records=records(result), stop_reason=result.stop_reason,
        menu_rng=menu_rng.bit_generator.state, sample_rng=sample_rng.bit_generator.state,
        sampling_precision_reported=hasattr(result, "sampling_precision"),
        sampling_precision=[asdict(row) for row in getattr(result, "sampling_precision", None) or []],
    ))
    print("small", args.small, args.seed, result.stop_reason, len(result.records))


def large():
    if any(value is None for value in (args.dataset, args.variant, args.public_inputs)):
        raise ValueError("大表对拍需要 dataset、variant 和 public-inputs")
    if not args.reference_api and args.baseline_output is None:
        raise ValueError("候选模式需要 baseline-output")
    paths = list((args.public_inputs / args.dataset).glob("measured_*query.json"))
    if len(paths) != 1:
        raise ValueError("每个公开输入目录必须恰有一个 measured_*query.json")
    query_path = paths[0].resolve()
    payload = json.loads(query_path.read_text())
    specs = load_queries(str(query_path))
    if not specs or not all(len(spec.conditions) == 2 for spec in specs) or not all(
        c["operator"] == "==" and str(c["value"]) in ("0", "1")
        for spec in specs for c in spec.conditions
    ):
        raise ValueError("本对拍协议只用于发布的二值全二阶查询")
    fields = sorted({c["attribute"] for s in specs for c in s.conditions},
                    key=lambda name: int(name.rsplit("_", 1)[1]))
    domains = [sorted({str(c["value"]) for s in specs for c in s.conditions
                       if c["attribute"] == field}) for field in fields]
    n = int(payload["record_count"])
    args.out.mkdir(parents=True)
    schema_path = args.out / "public_schema.json"
    dump(schema_path, {"fields": fields, "domains": domains})
    output = args.out / "terminal.csv"
    backend = "gpu" if args.dataset == "plants" else "cpu"
    work_rows = max(1, int(n * 0.1 + 0.5)) if args.variant == "working" else 0
    observed = {}

    def observe(ids, rounds, rng, provider, registry, **kwargs):
        write_table(args.out / "initial.csv", registry.schema, tuples_from_ids(registry, ids))
        cp = None
        if backend == "gpu":
            import cupy as cp
            cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        result = evolve_batch(ids, rounds, rng, provider, registry, **kwargs)
        if cp is not None:
            cp.cuda.Device().synchronize()
        observed["seconds"] = time.perf_counter() - t0
        observed["result"] = result
        observed["gpu_pool_reserved_mib"] = (
            cp.get_default_memory_pool().total_bytes() / 2**20 if cp is not None else None)
        return result

    if args.reference_api:
        schema = TableSchema(tuple(fields), tuple(tuple(domain) for domain in domains))
        target = target_from_specs(specs)
        marginals = derive_first_order(specs, schema, n)
        init_rng = np.random.default_rng(args.seed)
        rows = sample_initial_rows(marginals, schema, n, init_rng)
        registry = StateRegistry(schema, specs)
        registry.set_lazy_features(backend == "gpu")
        ids = registry.register_table(rows)
        menu_rng = np.random.default_rng(100000 + args.seed)
        select_rng = np.random.default_rng(200000 + args.seed)
        provider = make_batch_provider(
            registry, target, np.ones(len(specs)), menu_rng,
            pairing=False, defer_workload=backend == "gpu",
            work_rows=work_rows, work_below_step=0.2 if work_rows else 0.0,
            select_rng=select_rng,
        )
        sample_rng = np.random.default_rng(300000 + args.seed)
        result = observe(ids, 400, sample_rng, provider, registry,
                         max_frozen_retries=6, backend=backend, gpu_entry_budget=40000000)
        write_table(output, schema, tuples_from_ids(registry, result.state_ids))
        dump(args.out / "state.json", {
            "state_ids": result.state_ids.tolist(),
            "init_rng": init_rng.bit_generator.state,
            "menu_rng": menu_rng.bit_generator.state,
            "select_rng": select_rng.bit_generator.state,
            "sample_rng": sample_rng.bit_generator.state,
        })
    else:
        from resevo import cli

        cli.evolve_batch = observe
        cli.main([
            "--schema", str(schema_path), "--queries", str(query_path), "--rows", str(n),
            "--output", str(output), "--rounds", "400", "--seed", str(args.seed),
            "--backend", backend, "--retries", "6", "--gpu-entry-budget", "40000000",
            "--work-rows", str(work_rows), "--work-below", "0.2" if work_rows else "0",
        ])
    result = observed["result"]
    x = np.loadtxt(output, delimiter=",", skiprows=1, dtype=np.int8, ndmin=2)
    positions = {field: i for i, field in enumerate(fields)}
    counts = []
    for spec in specs:
        mask = np.ones(n, dtype=bool)
        for condition in spec.conditions:
            mask &= x[:, positions[condition["attribute"]]] == int(condition["value"])
        counts.append(int(mask.sum()))
    error = target_from_specs(specs) - np.asarray(counts, dtype=np.float64)
    final_loss = float(error @ error / 2)
    next_losses = [r.old_loss for r in result.records[1:]] + [final_loss]
    with (args.out / "trajectory.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["round", "old_loss", "expected_loss", "actual_next_loss",
                         "D", "C", "h", "M", "candidate_paths", "seconds"])
        for record, next_loss in zip(result.records, next_losses):
            writer.writerow([record.round_index, record.old_loss, record.expected_loss, next_loss,
                             record.direction_gain, record.interaction, record.step,
                             record.max_gain_sum, record.candidates, record.elapsed])
    summary = dict(
        code_commit=commit, dataset=args.dataset, variant=args.variant, seed=args.seed,
        backend=backend, recorded_rounds=len(result.records), stop_reason=result.stop_reason,
        initial_sha256=sha(args.out / "initial.csv"), terminal_sha256=sha(output),
        final_loss=final_loss, measured_l1=float(np.abs(error).mean() / n),
        measured_rmse=float(np.sqrt(np.mean(error**2)) / n), measured_max=float(np.abs(error).max() / n),
        evaluated_paths=int(sum(r.candidates for r in result.records)),
        seconds=observed["seconds"], gpu_pool_reserved_mib=observed["gpu_pool_reserved_mib"],
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        source_query_sha256=sha(query_path),
        sampling_precision_reported=hasattr(result, "sampling_precision"),
        sampling_precision=[asdict(row) for row in getattr(result, "sampling_precision", None) or []],
    )
    dump(args.out / "records.json", records(result))
    dump(args.out / "summary.json", summary)
    if args.reference_api:
        print("REFERENCE", args.dataset, args.variant, args.seed, json.dumps(summary))
        return
    baseline = args.baseline_output / f"{args.dataset}_{args.variant}_s{args.seed}"
    expected = json.loads((baseline / "summary.json").read_text())
    expected_query_hash = expected.get("source_query_sha256", expected.get("public_input_sha256"))
    if summary["source_query_sha256"] != expected_query_hash:
        raise AssertionError("基线与候选的公开查询输入不同")
    for filename, key in (("initial.csv", "initial_sha256"), ("terminal.csv", "terminal_sha256")):
        if sha(baseline / filename) != expected[key]:
            raise AssertionError(f"基线表与其元数据不符：{filename}")
    for key in ("recorded_rounds", "stop_reason", "initial_sha256", "terminal_sha256",
                "final_loss", "measured_l1", "measured_rmse", "measured_max", "evaluated_paths"):
        if summary[key] != expected[key]:
            raise AssertionError((key, summary[key], expected[key]))
    with (baseline / "trajectory.csv").open() as stream:
        old = list(csv.DictReader(stream))
    with (args.out / "trajectory.csv").open() as stream:
        new = list(csv.DictReader(stream))
    if len(old) != len(new) or len(new) != 400:
        raise AssertionError("轨迹轮数不一致")
    for before, after in zip(old, new):
        del before["seconds"], after["seconds"]
        if before != after:
            raise AssertionError((before, after))
    dump(args.out / "equivalence.json", {
        "status": "PASS", "baseline_commit": expected.get("code_commit", expected.get("commit")),
        "candidate_commit": commit, "initial_and_terminal_bytes_equal": True,
        "recorded_nontiming_trajectory_equal": True, "candidate_counts_and_quality_equal": True, "rounds": 400,
    })
    print("PASS", args.dataset, args.variant, args.seed)


if args.small:
    small()
else:
    large()
