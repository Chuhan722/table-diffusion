"""只接收公开 schema、记录数和已提供查询目标的基础生成入口。"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .batchkernel import evolve_batch, make_batch_provider
from .dataset import StateRegistry, TableSchema, load_queries, target_from_specs
from .editspace import tuples_from_ids
from .failure import save_numerical_failure
from .fiber import encode_rows, measured_sets
from .hard_fiber import hard_fiber_sample
from .initialization import derive_first_order, sample_initial_rows
from .sampling import SAMPLING_CONTRACT, sampling_summary
from .stepsize import validate_step_parameters


def load_public_schema(path: Path) -> TableSchema:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("schema 必须是包含 fields 和 domains 的 JSON 对象")
    fields, domains = payload.get("fields"), payload.get("domains")
    if (
        not isinstance(fields, list) or not fields
        or any(not isinstance(field, str) or not field for field in fields)
        or len(set(fields)) != len(fields)
    ):
        raise ValueError("schema.fields 必须是非空、不重复的字段名列表")
    if not isinstance(domains, list) or len(domains) != len(fields):
        raise ValueError("schema.domains 必须与 fields 一一对应")
    for domain in domains:
        if (
            not isinstance(domain, list) or not domain
            or any(not isinstance(value, str) for value in domain)
            or len(set(domain)) != len(domain)
        ):
            raise ValueError("每个 domain 必须是非空、不重复的字符串列表")
    return TableSchema(tuple(fields), tuple(tuple(domain) for domain in domains))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True, help="公开 fields/domains JSON")
    parser.add_argument("--queries", type=Path, required=True, help="已提供目标，result_unit=records")
    parser.add_argument("--rows", type=int, required=True, help="公开记录数，须与查询文件 record_count 相同")
    parser.add_argument("--output", type=Path, required=True, help="新建合成 CSV，不覆盖已有文件")
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--init", choices=("marginal", "uniform"), default="marginal")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--stay", type=float, default=0.9)
    parser.add_argument("--damping", type=float, default=1.0)
    parser.add_argument("--max-expected-rows", type=float, default=None)
    parser.add_argument("--retries", type=int, default=0, help="无正方向时的候选刷新次数")
    parser.add_argument("--work-rows", type=int, default=0)
    parser.add_argument("--work-below", type=float, default=0.0)
    parser.add_argument("--gpu-entry-budget", type=int, default=0)
    parser.add_argument("--hard-fiber-steps", type=int, default=0,
                        help="可选CPU硬后处理更新数（含自环）；保持完整等值边缘和全部一阶，"
                        "要求至少一条含字段的等值题，默认0关闭")
    parser.add_argument("--hard-fiber-seed", type=int, default=None,
                        help="硬后处理独立随机种子，默认400000+seed；须配正的--hard-fiber-steps")
    args = parser.parse_args(argv)

    try:
        validate_step_parameters(args.damping, args.max_expected_rows)
    except ValueError as exc:
        parser.error(str(exc))
    if args.rows < 1 or args.rounds < 1:
        parser.error("--rows 和 --rounds 必须为正")
    if args.seed < 0 or args.retries < 0 or args.work_rows < 0 or args.gpu_entry_budget < 0:
        parser.error("种子、重试次数、工作批名额和 GPU 条目预算不能为负")
    if args.hard_fiber_steps < 0:
        parser.error("--hard-fiber-steps 不能为负")
    if args.hard_fiber_seed is not None and (args.hard_fiber_seed < 0 or args.hard_fiber_steps == 0):
        parser.error("--hard-fiber-seed 必须非负且配正的 --hard-fiber-steps")
    if not (0 < args.alpha < 1) or not (0 < args.stay < 1):
        parser.error("--alpha 和 --stay 必须落在 (0,1)")
    if not np.isfinite(args.work_below) or args.work_below < 0:
        parser.error("--work-below 必须为非负有限数")
    if args.output.exists():
        parser.error(f"拒绝覆盖已有输出：{args.output}")
    try:
        schema = load_public_schema(args.schema)
        specs = load_queries(str(args.queries), expected_record_count=args.rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    target = target_from_specs(specs)
    if not specs or not np.isfinite(target).all():
        parser.error("查询目标必须非空且全部有限")
    hard_sets = None
    if args.hard_fiber_steps > 0:
        try:
            hard_sets = measured_sets(specs, schema, ignore_prefix="")
        except ValueError as exc:
            parser.error(f"硬纤维约束校验失败：{exc}")
        if not any(hard_sets):
            parser.error("--hard-fiber-steps 需要至少一条含字段的等值查询；常量条目可与其混合")

    init_rng = np.random.default_rng(args.seed)
    if args.init == "marginal":
        try:
            marginals = derive_first_order(specs, schema, args.rows)
            rows = sample_initial_rows(marginals, schema, args.rows, init_rng)
        except ValueError as exc:
            parser.error(
                f"marginal 初始化失败：{exc}；覆盖不足或带噪边缘不一致时，可显式选择 --init uniform"
            )
    else:
        rows = [
            tuple(domain[int(init_rng.integers(len(domain)))] for domain in schema.domains)
            for _ in range(args.rows)
        ]
    registry = StateRegistry(schema, specs)
    registry.set_lazy_features(args.backend == "gpu")
    ids = registry.register_table(rows)
    provider = make_batch_provider(
        registry, target, np.ones(len(specs)), np.random.default_rng(100000 + args.seed),
        pairing=False, defer_workload=args.backend == "gpu",
        work_rows=args.work_rows, work_below_step=args.work_below,
        select_rng=np.random.default_rng(200000 + args.seed),
    )
    with save_numerical_failure(registry, args.output):
        result = evolve_batch(
            ids, args.rounds, np.random.default_rng(300000 + args.seed), provider, registry,
            alpha=args.alpha, stay_probability=args.stay, damping=args.damping,
            max_expected_rows=args.max_expected_rows, max_frozen_retries=args.retries,
            backend=args.backend, gpu_entry_budget=args.gpu_entry_budget,
        )
    final_rows = tuples_from_ids(registry, result.state_ids)
    hard_info = None
    if hard_sets is not None:
        hard_seed = 400000 + args.seed if args.hard_fiber_seed is None else args.hard_fiber_seed
        hard = hard_fiber_sample(
            encode_rows(final_rows, schema), [len(domain) for domain in schema.domains], hard_sets,
            args.hard_fiber_steps, np.random.default_rng(hard_seed),
        )
        final_rows = [
            tuple(schema.domains[j][int(value)] for j, value in enumerate(row))
            for row in hard.codes
        ]
        hard_info = {"seed": hard_seed, **asdict(hard.diagnostics)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(schema.fields)
        writer.writerows(final_rows)
    info = {
        "output": str(args.output), "rows": len(final_rows),
        "rounds": len(result.records), "stop_reason": result.stop_reason,
        "backend": args.backend, "seed": args.seed,
        "sampling_contract": SAMPLING_CONTRACT,
        "sampling_precision": sampling_summary(result.sampling_precision),
    }
    if hard_info is not None:
        info["hard_fiber"] = hard_info
    print(json.dumps(info, ensure_ascii=False))
