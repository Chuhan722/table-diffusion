#!/usr/bin/env python3
"""测量第 6B-1B 阶段不同地址的 CUDA（显卡计算）批量原型。

本脚本只复用公开 NLTCS（数据集）属性结构和查询定义来构造人工表，不读取
冻结状态、来源轨迹、正式地址或任何方法效果。每个人工地址使用不同供体、
参与行、活跃坐标、K、初始开关与随机带；现有单地址实现逐地址形成参考。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
        run_variable_workloads_batched_prototype,
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
        run_variable_workloads_batched_prototype,
    )


ARTIFICIAL_ADDRESS_SEED = 2026082611
FLOAT_TOLERANCE = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16181)
    parser.add_argument("--participation-rate", type=float, default=0.01)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=(1, 4, 8, 16, 20)
    )
    return parser


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _build_variable_addresses(
    workload: dict[str, Any],
    *,
    address_count: int,
    participation_rate: float,
) -> list[dict[str, Any]]:
    """从同一人工当前表构造结构不同的地址。"""

    if address_count <= 0:
        raise ValueError("人工地址数必须为正")
    current = workload["current"]
    current_values = current.to_numpy()
    rows, attributes = current_values.shape
    rng = np.random.default_rng(ARTIFICIAL_ADDRESS_SEED)
    addresses = []
    for address_index in range(address_count):
        donor_indices = rng.integers(0, rows, size=rows, dtype=np.int64)
        donors = current.iloc[donor_indices].reset_index(drop=True)
        participate = rng.random(rows) < participation_rate
        active = participate[:, None] & (
            current_values != donors.to_numpy()
        )
        initial_mask = active & (
            rng.random((rows, attributes)) < 0.5
        )
        if not np.any(active):
            raise RuntimeError(
                f"人工地址 {address_index} 没有形成活跃坐标"
            )
        addresses.append({
            "donors": donors,
            "donor_indices": donor_indices,
            "participate": participate,
            "active": active,
            "initial_mask": initial_mask,
        })

    donor_hashes = {
        _array_sha256(address["donor_indices"]) for address in addresses
    }
    participation_hashes = {
        _array_sha256(address["participate"]) for address in addresses
    }
    active_switches = [
        int(np.sum(address["active"])) for address in addresses
    ]
    if len(donor_hashes) != address_count:
        raise RuntimeError("人工地址未形成互不相同的供体")
    if len(participation_hashes) != address_count:
        raise RuntimeError("人工地址未形成互不相同的参与行")
    if address_count > 1 and len(set(active_switches)) == 1:
        raise RuntimeError("人工地址未形成不同 K")
    return addresses


def _run_reference(
    workload: dict[str, Any],
    address: dict[str, Any],
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
            address["donors"],
            workload["schema"],
            workload["queries"],
            workload["target"],
            workload["current_counts"],
            participate=address["participate"],
            initial_mask=address["initial_mask"],
            reference_scale=reference_scale,
            rng=np.random.default_rng(seed),
            n_sweeps=sweeps,
            device="cuda",
        )
    finally:
        production._update_trace = original
    return result, captured, time.perf_counter() - started


def _reference_arrays(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, ...]:
    coordinates = np.asarray(
        [[row["row_index"], row["attribute_index"]] for row in records],
        dtype=np.int64,
    ).reshape((-1, 2))
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
    ], dtype=np.float64).reshape((-1, 7))
    booleans = np.asarray([
        [row["before"], row["after"], row["clipped"]]
        for row in records
    ], dtype=bool).reshape((-1, 3))
    return coordinates, floats, booleans


def _compare_to_references(
    prototype: dict[str, Any],
    references: list[tuple[tuple[Any, ...], list[dict[str, Any]], float]],
) -> dict[str, Any]:
    verified = prototype["batch_size"]
    if len(references) < verified:
        raise ValueError("逐地址参考数量不足")
    maximum_float_difference = 0.0
    coordinates_exact = True
    random_rolls_exact = True
    microstep_lengths_exact = True
    booleans_exact = True
    final_masks_exact = True
    final_counts_exact = True
    final_tables_exact = True
    legacy_trace_hashes_exact = True
    sampling_decision_flip_count = 0
    all_finite = True
    random_decisions_exact = True
    for index in range(verified):
        reference, records, _ = references[index]
        coordinates, floats, booleans = _reference_arrays(records)
        prototype_floats = prototype["float_values"][index]
        prototype_booleans = prototype["bool_values"][index]
        microstep_lengths_exact &= (
            prototype["microsteps_by_address"][index] == len(records)
            and prototype_floats.shape == floats.shape
            and prototype_booleans.shape == booleans.shape
        )
        if prototype_floats.shape == floats.shape:
            maximum_float_difference = max(
                maximum_float_difference,
                float(np.max(
                    np.abs(prototype_floats - floats), initial=0.0
                )),
            )
        else:
            maximum_float_difference = float("inf")
        coordinates_exact &= np.array_equal(
            prototype["coordinates"][index], coordinates
        )
        random_rolls_exact &= np.array_equal(
            prototype["random_rolls"][index], floats[:, 6]
        )
        current_booleans_exact = np.array_equal(
            prototype_booleans, booleans
        )
        booleans_exact &= current_booleans_exact
        if prototype_booleans.shape == booleans.shape:
            sampling_decision_flip_count += int(np.sum(
                prototype_booleans[:, 1] != booleans[:, 1]
            ))
        else:
            sampling_decision_flip_count += max(
                len(prototype_booleans), len(booleans)
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
        all_finite &= bool(np.all(np.isfinite(prototype_floats)))
        random_decisions_exact &= bool(np.array_equal(
            prototype_booleans[:, 1],
            prototype_floats[:, 6] < prototype_floats[:, 5],
        ))

    valid_mask = prototype["valid_step_mask"]
    padding_mask_exact = True
    for index, microsteps in enumerate(prototype["microsteps_by_address"]):
        padding_mask_exact &= bool(
            np.all(valid_mask[index, :microsteps])
            and not np.any(valid_mask[index, microsteps:])
        )
    tolerated_float_equivalence = (
        maximum_float_difference <= FLOAT_TOLERANCE
    )
    discrete_equivalence = all((
        coordinates_exact,
        random_rolls_exact,
        microstep_lengths_exact,
        booleans_exact,
        final_masks_exact,
        final_counts_exact,
        final_tables_exact,
        random_decisions_exact,
        padding_mask_exact,
        sampling_decision_flip_count == 0,
    ))
    return {
        "verified_address_count": verified,
        "all_finite": all_finite,
        "microstep_lengths_exact": microstep_lengths_exact,
        "coordinates_exact": coordinates_exact,
        "random_rolls_bitwise_exact": random_rolls_exact,
        "all_microstep_booleans_exact": booleans_exact,
        "sampling_decision_flip_count": sampling_decision_flip_count,
        "final_masks_exact": final_masks_exact,
        "final_query_counts_exact": final_counts_exact,
        "final_tables_exact": final_tables_exact,
        "random_decisions_match_probabilities": random_decisions_exact,
        "padding_validity_mask_exact": padding_mask_exact,
        "maximum_absolute_float_difference": maximum_float_difference,
        "float_tolerance": FLOAT_TOLERANCE,
        "within_frozen_float_tolerance": tolerated_float_equivalence,
        "legacy_single_address_trace_sha256_exact": (
            legacy_trace_hashes_exact
        ),
        "prototype_feasibility_passed": bool(
            all_finite and discrete_equivalence and tolerated_float_equivalence
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
        or not 0.0 < args.participation_rate < 1.0
    ):
        raise ValueError("批量、参与率或扫描次数无效")
    environment = validate_gpu_environment()
    import torch

    workload = build_artificial_workload(
        rows=args.rows,
        participation_rate=args.participation_rate,
    )
    maximum_batch = batches[-1]
    addresses = _build_variable_addresses(
        workload,
        address_count=maximum_batch,
        participation_rate=args.participation_rate,
    )
    isolated = production.isolated_gap_l1_scores(
        workload["current"],
        addresses[0]["donors"],
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

    seeds = [ARTIFICIAL_SCAN_SEED + index for index in range(maximum_batch)]
    references = []
    for index in range(maximum_batch):
        print(
            f"[variable batch reference {index + 1}/{maximum_batch}]",
            file=sys.stderr,
            flush=True,
        )
        references.append(_run_reference(
            workload,
            addresses[index],
            seeds[index],
            reference_scale=reference_scale,
            sweeps=args.sweeps,
        ))

    batch_results = []
    for batch_size in batches:
        print(
            f"[variable batch prototype batch={batch_size}]",
            file=sys.stderr,
            flush=True,
        )
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        selected_addresses = addresses[:batch_size]
        prototype = run_variable_workloads_batched_prototype(
            workload["current"],
            [address["donors"] for address in selected_addresses],
            workload["schema"],
            workload["queries"],
            workload["target"],
            workload["current_counts"],
            participates=[
                address["participate"] for address in selected_addresses
            ],
            initial_masks=[
                address["initial_mask"] for address in selected_addresses
            ],
            reference_scale=reference_scale,
            seeds=seeds[:batch_size],
            n_sweeps=args.sweeps,
        )
        total_elapsed = prototype["timings"]["total_elapsed_sec"]
        comparison = _compare_to_references(prototype, references)
        reference_elapsed = float(np.sum([
            row[2] for row in references[:batch_size]
        ]))
        batch_results.append({
            "batch_size": batch_size,
            "active_switches_k_by_address": list(
                prototype["active_switches_k_by_address"]
            ),
            "microsteps_by_address": list(
                prototype["microsteps_by_address"]
            ),
            "maximum_padded_microsteps": prototype[
                "maximum_padded_microsteps"
            ],
            "padding_microsteps": prototype["padding_microsteps"],
            "timings": dict(prototype["timings"]),
            "per_address_elapsed_sec": total_elapsed / batch_size,
            "throughput_speedup_vs_existing_single_address": (
                reference_elapsed / total_elapsed
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "comparison": comparison,
        })
        del prototype

    active_switches = [
        int(np.sum(address["active"])) for address in addresses
    ]
    participating_rows = [
        int(np.sum(address["participate"])) for address in addresses
    ]
    result = {
        "status": "artificial_variable_batched_gpu_prototype_complete",
        "environment": environment,
        "artificial_workload": {
            "rows": args.rows,
            "attributes": len(workload["attributes"]),
            "queries": len(workload["queries"]),
            "address_count": maximum_batch,
            "participation_rate_requested": args.participation_rate,
            "participating_rows_by_address": participating_rows,
            "active_switches_k_by_address": active_switches,
            "minimum_active_switches_k": min(active_switches),
            "maximum_active_switches_k": max(active_switches),
            "sweeps": args.sweeps,
            "reference_scale": reference_scale,
            "nonzero_isolated_scores": scale_diagnostics["nonzero_count"],
            "address_seed_non_frozen": ARTIFICIAL_ADDRESS_SEED,
        },
        "existing_single_address_references": {
            "address_count": maximum_batch,
            "mean_elapsed_sec": float(np.mean([
                row[2] for row in references
            ])),
            "total_elapsed_sec": float(np.sum([
                row[2] for row in references
            ])),
            "elapsed_sec": [row[2] for row in references],
        },
        "batch_results": batch_results,
        "prototype_scope": {
            "shared_current_target_and_query_state": True,
            "different_donors": True,
            "different_participation_masks": True,
            "different_active_coordinates_and_k": True,
            "different_initial_masks_and_random_tapes": True,
            "full_query_state_batched": True,
            "variable_length_padding_and_masks": True,
            "strict_internal_address_order_preserved": True,
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
