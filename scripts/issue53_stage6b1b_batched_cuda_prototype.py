"""第 6B-1B 阶段多地址 CUDA（显卡计算）批量人工对拍封装。

本模块只为人工基准暴露生产批量核的逐微步内部轨迹。批量算术只在
``table_diffevo.gap_l1_diffusion（正式缺口核模块）`` 中实现一次；本文件
不再维护第二套扫描算法，也不被正式采集、审计或公共生成器导入。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from table_diffevo import gap_l1_diffusion as production


PROTOTYPE_BACKEND = "artificial_wrapper_over_production_batched_cuda_v2"


def run_variable_workloads_batched_prototype(
    current: pd.DataFrame,
    donor_tables: Sequence[pd.DataFrame],
    schema: Any,
    queries: list[dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participates: Sequence[Any],
    initial_masks: Sequence[Any],
    reference_scale: float,
    seeds: Sequence[int],
    n_sweeps: int = 8,
    eta: float = 0.5,
    strength: float = 2.0,
    floor: float = 8.0,
    logit_clip: float = 30.0,
) -> dict[str, Any]:
    """用人工种子调用唯一的生产批量算术并返回对拍轨迹。"""

    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    results, debug = production._evolve_step_gap_l1_global_cuda_batched(
        current,
        donor_tables,
        schema,
        queries,
        target,
        current_counts,
        participates=participates,
        initial_masks=initial_masks,
        reference_scale=reference_scale,
        rngs=rngs,
        n_sweeps=n_sweeps,
        eta=eta,
        strength=strength,
        floor=floor,
        logit_clip=logit_clip,
        compiled_workload=None,
        verify_full_recount=False,
    )
    return {
        "backend": PROTOTYPE_BACKEND,
        "production_batch_backend": debug["backend"],
        "batch_execution_format": debug["batch_execution_format"],
        "batch_size": debug["batch_size"],
        "active_switches_k_by_address": debug[
            "active_switches_k_by_address"
        ],
        "microsteps_by_address": debug["microsteps_by_address"],
        "maximum_padded_microsteps": debug["maximum_padded_microsteps"],
        "padding_microsteps": debug["padding_microsteps"],
        "valid_step_mask": debug["valid_step_mask"],
        "reference_scale": debug["reference_scale"],
        "tables": tuple(result[0] for result in results),
        "masks": np.stack([result[1] for result in results], axis=0),
        "final_query_counts": np.asarray([
            result[2]["final_query_counts"] for result in results
        ], dtype=np.int64),
        "trace_sha256": tuple(
            result[2]["microstep_trace_sha256"] for result in results
        ),
        "coordinates": debug["coordinates"],
        "random_rolls": debug["random_rolls"],
        "float_values": debug["float_values"],
        "bool_values": debug["bool_values"],
        "timings": dict(debug["timings"]),
        "different_address_structures_enabled": True,
        "strict_internal_order_preserved": True,
        "production_batch_kernel_called": True,
        "formal_state_or_address_read": False,
        "production_pipeline_enabled": False,
    }


def run_same_workload_batched_prototype(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    queries: list[dict[str, Any]],
    target: Any,
    current_counts: Any,
    *,
    participate: Any,
    initial_masks: Sequence[Any],
    reference_scale: float,
    seeds: Sequence[int],
    n_sweeps: int = 8,
    eta: float = 0.5,
    strength: float = 2.0,
    floor: float = 8.0,
    logit_clip: float = 30.0,
) -> dict[str, Any]:
    """兼容首版同结构人工基准的薄封装。"""

    batch_size = len(initial_masks)
    result = run_variable_workloads_batched_prototype(
        current,
        [donors] * batch_size,
        schema,
        queries,
        target,
        current_counts,
        participates=[participate] * batch_size,
        initial_masks=initial_masks,
        reference_scale=reference_scale,
        seeds=seeds,
        n_sweeps=n_sweeps,
        eta=eta,
        strength=strength,
        floor=floor,
        logit_clip=logit_clip,
    )
    active_switches = result["active_switches_k_by_address"]
    microsteps = result["microsteps_by_address"]
    if len(set(active_switches)) != 1 or len(set(microsteps)) != 1:
        raise RuntimeError("同结构兼容封装得到不同的地址长度")
    result["active_switches_k"] = active_switches[0]
    result["microsteps_per_address"] = microsteps[0]
    result["coordinates"] = np.stack(result["coordinates"], axis=0)
    result["random_rolls"] = np.stack(result["random_rolls"], axis=0)
    result["float_values"] = np.stack(result["float_values"], axis=0)
    result["bool_values"] = np.stack(result["bool_values"], axis=0)
    return result
