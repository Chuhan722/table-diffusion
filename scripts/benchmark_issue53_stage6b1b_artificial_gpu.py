#!/usr/bin/env python3
"""用 NLTCS 同尺寸人工表测量第 6B-1B 显卡实现。

只读取公开 schema（属性结构）与查询定义；不读取冻结状态、来源轨迹、
候选地址、参考表或任何第 6B-1B 结果，也不使用冻结随机地址。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from table_diffevo.gap_l1_diffusion import (
    evolve_step_gap_l1_global,
    isolated_gap_l1_scores,
    stable_nonzero_rms,
)
from table_diffevo.schema import load_schema
from table_diffevo.queries import load_queries
from table_diffevo.vectorized_eval import evaluate_vectorized

if __package__:
    from scripts import issue53_stage6b1b_independent_cuda as independent
    from scripts.check_issue53_stage6b1b_nltcs_gpu_environment import (
        validate_gpu_environment,
    )
else:
    import issue53_stage6b1b_independent_cuda as independent
    from check_issue53_stage6b1b_nltcs_gpu_environment import (
        validate_gpu_environment,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFICIAL_TABLE_SEED = 2026082601
ARTIFICIAL_SCAN_SEED = 2026082602


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16181)
    parser.add_argument("--participation-rate", type=float, default=0.01)
    parser.add_argument("--sweeps", type=int, default=8)
    return parser


def _measure(torch, operation):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = operation()
    torch.cuda.synchronize()
    return result, {
        "elapsed_sec": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.rows <= 0 or not 0.0 < args.participation_rate < 1.0:
        raise ValueError("人工表行数和参与率无效")
    if args.sweeps < 0:
        raise ValueError("人工扫描次数必须是非负整数")
    environment = validate_gpu_environment()
    import torch

    schema = load_schema(str(REPOSITORY_ROOT / "configs/nltcs/schema.yaml"))
    queries = load_queries(
        str(REPOSITORY_ROOT / "configs/nltcs/measured_1000query.json")
    )
    attributes = schema.attribute_names()
    rng = np.random.default_rng(ARTIFICIAL_TABLE_SEED)
    current = pd.DataFrame(
        rng.integers(0, 2, size=(args.rows, len(attributes)), dtype=np.int8),
        columns=attributes,
    )
    donor_indices = rng.integers(0, args.rows, size=args.rows)
    donors = current.iloc[donor_indices].reset_index(drop=True)
    participate = rng.random(args.rows) < args.participation_rate
    active = participate[:, None] & (
        current.to_numpy() != donors.to_numpy()
    )
    initial_mask = active & (
        rng.random(size=active.shape) < 0.5
    )
    current_counts, _, _ = evaluate_vectorized(
        current,
        queries,
        schema,
        device="cuda",
        want_fitness=False,
        verbose=False,
    )
    current_counts = np.asarray(current_counts, dtype=np.int64)
    offsets = (np.arange(len(queries), dtype=np.int64) % 5) - 2
    target = np.maximum(current_counts + offsets, 0).astype(np.float64)
    exact_numerators = target.astype(np.int64)

    production_isolated, production_calibration_timing = _measure(
        torch,
        lambda: isolated_gap_l1_scores(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            exact_target_numerators=exact_numerators,
            exact_target_denominator=1,
            device="cuda",
        ),
    )
    reference_scale, scale_distribution = stable_nonzero_rms(
        production_isolated["scores"]
    )
    if reference_scale <= 0.0:
        raise RuntimeError("人工负载没有形成正参考尺度")
    independent_isolated, independent_calibration_timing = _measure(
        torch,
        lambda: independent.isolated_gap_l1_scores_cuda(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            floor=8.0,
            exact_target_numerators=exact_numerators,
            exact_target_denominator=1,
        ),
    )
    if (
        not np.array_equal(
            production_isolated["coordinates"], independent_isolated[0]
        )
        or not np.array_equal(
            production_isolated["scores"], independent_isolated[1]
        )
    ):
        raise RuntimeError("人工同尺寸定尺的生产/独立显卡结果不一致")

    production_scan, production_scan_timing = _measure(
        torch,
        lambda: evolve_step_gap_l1_global(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            participate=participate,
            initial_mask=initial_mask,
            reference_scale=reference_scale,
            rng=np.random.default_rng(ARTIFICIAL_SCAN_SEED),
            n_sweeps=args.sweeps,
            device="cuda",
        ),
    )
    independent_scan, independent_scan_timing = _measure(
        torch,
        lambda: independent.replay_gap_l1_cuda(
            current,
            donors,
            schema,
            queries,
            target,
            current_counts,
            participate,
            initial_mask,
            reference_scale=reference_scale,
            seed=ARTIFICIAL_SCAN_SEED,
            n_sweeps=args.sweeps,
            eta=0.5,
            strength=2.0,
            floor=8.0,
            logit_clip=30.0,
        ),
    )
    production_diagnostics = production_scan[2]
    independent_diagnostics = independent_scan[2]
    if (
        not production_scan[0].equals(independent_scan[0])
        or not np.array_equal(production_scan[1], independent_scan[1])
        or production_diagnostics["microstep_trace_sha256"]
        != independent_diagnostics["trace_sha256"]
        or production_diagnostics["final_query_counts"]
        != independent_diagnostics["final_query_counts"].tolist()
    ):
        raise RuntimeError("人工同尺寸扫描的生产/独立显卡结果不一致")

    result = {
        "status": "artificial_nltcs_size_gpu_qualification_passed",
        "environment": environment,
        "artificial_workload": {
            "rows": args.rows,
            "attributes": len(attributes),
            "queries": len(queries),
            "participation_rate_requested": args.participation_rate,
            "participating_rows": int(np.sum(participate)),
            "active_switches_k": int(np.sum(active)),
            "sweeps": args.sweeps,
            "microsteps": int(args.sweeps * np.sum(active)),
            "reference_scale": reference_scale,
            "nonzero_isolated_scores": scale_distribution["nonzero_count"],
            "table_seed_non_frozen": ARTIFICIAL_TABLE_SEED,
            "scan_seed_non_frozen": ARTIFICIAL_SCAN_SEED,
        },
        "timings": {
            "production_calibration": production_calibration_timing,
            "independent_calibration": independent_calibration_timing,
            "production_scan": production_scan_timing,
            "independent_scan": independent_scan_timing,
        },
        "exact_checks": {
            "isolated_coordinates": True,
            "isolated_scores": True,
            "final_table": True,
            "final_mask": True,
            "final_query_counts": True,
            "microstep_trace_sha256": True,
        },
        "formal_state_or_address_read": False,
        "frozen_address_consumed": False,
        "mechanism_effect_evaluated": False,
        "candidate_artifact_written": False,
    }
    print(json.dumps(
        result,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ))


if __name__ == "__main__":
    main()
