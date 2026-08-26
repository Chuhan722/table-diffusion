#!/usr/bin/env python3
"""测量第 6B-1B 阶段多地址 CUDA 批量原型。

本脚本只复用公开 NLTCS schema（属性结构）和查询定义来构造人工表，不读取
冻结状态、来源轨迹、正式地址或任何方法效果。它逐地址调用现有生产核形成
参考，再测试批量 1/4/8/16/20 的吞吐、显存和一致性。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from typing import Any

import numpy as np

from table_diffevo import gap_l1_diffusion as production

if __package__:
    from scripts.benchmark_issue53_stage6b1b_artificial_gpu import (
        ARTIFICIAL_SCAN_SEED,
        build_artificial_workload,
    )
    from scripts.check_issue53_stage6b1b_nltcs_gpu_environment import (
        validate_gpu_environment,
    )
    from scripts.issue53_stage6b1b_batched_cuda_prototype import (
        run_same_workload_batched_prototype,
    )
else:
    from benchmark_issue53_stage6b1b_artificial_gpu import (
        ARTIFICIAL_SCAN_SEED,
        build_artificial_workload,
    )
    from check_issue53_stage6b1b_nltcs_gpu_environment import (
        validate_gpu_environment,
    )
    from issue53_stage6b1b_batched_cuda_prototype import (
        run_same_workload_batched_prototype,
    )


PROTOTYPE_INITIAL_MASK_SEED = 2026082610
FLOAT_TOLERANCE = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16181)
    parser.add_argument("--participation-rate", type=float, default=0.01)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=(1, 4, 8, 16, 20)
    )
    parser.add_argument("--reference-addresses", type=int, default=4)
    return parser


def _run_reference(
    workload: dict,
    initial_mask: np.ndarray,
    seed: int,
    *,
    reference_scale: float,
    sweeps: int,
) -> tuple[tuple[Any, ...], list[dict[str, Any]], float]:
    captured: list[dict[str, Any]] = []
    original = production._update_trace

    def capture(digest, **values):
        captured.append(dict(values))
        return original(digest, **values)

    production._update_trace = capture
    started = time.perf_counter()
    try:
        result = production.evolve_step_gap_l1_global(
            workload["current"],
            workload["donors"],
            workload["schema"],
            workload["queries"],
            workload["target"],
            workload["current_counts"],
            participate=workload["participate"],
            initial_mask=initial_mask,
            reference_scale=reference_scale,
            rng=np.random.default_rng(seed),
            n_sweeps=sweeps,
            device="cuda",
        )
    finally:
        production._update_trace = original
    return result, captured, time.perf_counter() - started


def _reference_arrays(records: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
    coordinates = np.asarray(
        [[row["row_index"], row["attribute_index"]] for row in records],
        dtype=np.int64,
    )
    floats = np.asarray([
        [
            row["e0"],
            row["e1"],
            row["score"],
            row["normalized_score"],
            row["raw_logit"],
            row["probability"],
            row["random_roll"],
        ]
        for row in records
    ], dtype=np.float64)
    booleans = np.asarray([
        [row["before"], row["after"], row["clipped"]]
        for row in records
    ], dtype=bool)
    return coordinates, floats, booleans


def _compare_to_references(
    prototype: dict,
    references: list[tuple[tuple[Any, ...], list[dict[str, Any]], float]],
) -> dict[str, Any]:
    verified = min(prototype["batch_size"], len(references))
    maximum_float_difference = 0.0
    coordinates_exact = True
    booleans_exact = True
    final_masks_exact = True
    final_counts_exact = True
    final_tables_exact = True
    legacy_trace_hashes_exact = True
    for index in range(verified):
        reference, records, _ = references[index]
        coordinates, floats, booleans = _reference_arrays(records)
        maximum_float_difference = max(
            maximum_float_difference,
            float(np.max(
                np.abs(prototype["float_values"][index] - floats),
                initial=0.0,
            )),
        )
        coordinates_exact &= np.array_equal(
            prototype["coordinates"][index], coordinates
        )
        booleans_exact &= np.array_equal(
            prototype["bool_values"][index], booleans
        )
        final_masks_exact &= np.array_equal(
            prototype["masks"][index], reference[1]
        )
        final_counts_exact &= np.array_equal(
            prototype["final_query_counts"][index],
            np.asarray(reference[2]["final_query_counts"], dtype=np.int64),
        )
        final_tables_exact &= prototype["tables"][index].equals(reference[0])
        legacy_trace_hashes_exact &= (
            prototype["trace_sha256"][index]
            == reference[2]["microstep_trace_sha256"]
        )
    finite = bool(np.all(np.isfinite(prototype["float_values"])))
    random_decisions_exact = bool(np.array_equal(
        prototype["bool_values"][:, :, 1],
        prototype["float_values"][:, :, 6]
        < prototype["float_values"][:, :, 5],
    ))
    tolerated_float_equivalence = (
        maximum_float_difference <= FLOAT_TOLERANCE
    )
    discrete_equivalence = all((
        coordinates_exact,
        booleans_exact,
        final_masks_exact,
        final_counts_exact,
        final_tables_exact,
        random_decisions_exact,
    ))
    return {
        "verified_address_count": verified,
        "all_finite": finite,
        "coordinates_and_random_rolls_exact": coordinates_exact,
        "all_microstep_booleans_exact": booleans_exact,
        "final_masks_exact": final_masks_exact,
        "final_query_counts_exact": final_counts_exact,
        "final_tables_exact": final_tables_exact,
        "random_decisions_match_probabilities": random_decisions_exact,
        "maximum_absolute_float_difference": maximum_float_difference,
        "float_tolerance": FLOAT_TOLERANCE,
        "within_frozen_float_tolerance": tolerated_float_equivalence,
        "legacy_single_address_trace_sha256_exact": (
            legacy_trace_hashes_exact
        ),
        "prototype_feasibility_passed": bool(
            finite and discrete_equivalence and tolerated_float_equivalence
        ),
    }


def main() -> None:
    args = _parser().parse_args()
    batches = tuple(sorted(set(args.batches)))
    if (
        not batches
        or batches[0] <= 0
        or batches[-1] > 20
        or args.sweeps < 0
        or args.reference_addresses <= 0
    ):
        raise ValueError("批量、扫描次数或参考地址数无效")
    environment = validate_gpu_environment()
    import torch

    workload = build_artificial_workload(
        rows=args.rows,
        participation_rate=args.participation_rate,
    )
    isolated = production.isolated_gap_l1_scores(
        workload["current"],
        workload["donors"],
        workload["schema"],
        workload["queries"],
        workload["target"],
        workload["current_counts"],
        exact_target_numerators=workload["exact_target_numerators"],
        exact_target_denominator=1,
        device="cuda",
    )
    reference_scale, scale_diagnostics = production.stable_nonzero_rms(
        isolated["scores"]
    )
    if reference_scale <= 0.0:
        raise RuntimeError("人工工作量没有形成正参考尺度")

    maximum_batch = batches[-1]
    mask_rng = np.random.default_rng(PROTOTYPE_INITIAL_MASK_SEED)
    initial_masks = [workload["initial_mask"]]
    initial_masks.extend(
        workload["active"] & (
            mask_rng.random(workload["active"].shape) < 0.5
        )
        for _ in range(maximum_batch - 1)
    )
    seeds = [ARTIFICIAL_SCAN_SEED + index for index in range(maximum_batch)]
    reference_count = min(args.reference_addresses, maximum_batch)
    references = []
    for index in range(reference_count):
        print(
            f"[batched prototype reference {index + 1}/{reference_count}]",
            file=sys.stderr,
            flush=True,
        )
        references.append(_run_reference(
            workload,
            initial_masks[index],
            seeds[index],
            reference_scale=reference_scale,
            sweeps=args.sweeps,
        ))
    reference_mean_elapsed = float(np.mean([
        row[2] for row in references
    ]))

    batch_results = []
    for batch_size in batches:
        print(
            f"[batched prototype batch={batch_size}]",
            file=sys.stderr,
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        prototype = run_same_workload_batched_prototype(
            workload["current"],
            workload["donors"],
            workload["schema"],
            workload["queries"],
            workload["target"],
            workload["current_counts"],
            participate=workload["participate"],
            initial_masks=initial_masks[:batch_size],
            reference_scale=reference_scale,
            seeds=seeds[:batch_size],
            n_sweeps=args.sweeps,
        )
        total_elapsed = prototype["timings"]["total_elapsed_sec"]
        comparison = _compare_to_references(prototype, references)
        batch_results.append({
            "batch_size": batch_size,
            "timings": dict(prototype["timings"]),
            "per_address_elapsed_sec": total_elapsed / batch_size,
            "throughput_speedup_vs_existing_single_address": (
                reference_mean_elapsed / (total_elapsed / batch_size)
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "comparison": comparison,
        })
        del prototype

    result = {
        "status": "artificial_batched_gpu_prototype_complete",
        "environment": environment,
        "artificial_workload": {
            "rows": args.rows,
            "attributes": len(workload["attributes"]),
            "queries": len(workload["queries"]),
            "participating_rows": int(np.sum(workload["participate"])),
            "active_switches_k": int(np.sum(workload["active"])),
            "sweeps": args.sweeps,
            "microsteps_per_address": int(
                args.sweeps * np.sum(workload["active"])
            ),
            "reference_scale": reference_scale,
            "nonzero_isolated_scores": scale_diagnostics["nonzero_count"],
        },
        "existing_single_address_reference": {
            "address_count": reference_count,
            "mean_elapsed_sec": reference_mean_elapsed,
            "elapsed_sec": [row[2] for row in references],
        },
        "batch_results": batch_results,
        "prototype_scope": {
            "shared_current_donors_participation_and_active_coordinates": True,
            "different_initial_masks_and_random_tapes": True,
            "production_pipeline_enabled": False,
            "independent_batched_auditor_implemented": False,
            "formal_state_or_address_read": False,
            "frozen_address_consumed": False,
            "mechanism_effect_evaluated": False,
            "candidate_artifact_written": False,
        },
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
