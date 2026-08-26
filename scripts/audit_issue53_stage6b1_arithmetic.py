#!/usr/bin/env python3
"""独立复算第 6B-1 阶段逐微步、表、指标和冻结结论。

本文件刻意不导入第 6B-1 阶段采集器、共享算术模块、结构审计器或评价器。
查询评价、稀疏表重建、绝对误差、吉布斯增量状态、门槛与分类均在此独立
实现。仅复用第 6A 阶段已经独立审计过的通用表/查询底层原语和公开生成部件。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import compute_copy_direction_scores
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.vectorized_eval import evaluate_vectorized

if __package__:
    from scripts import audit_issue53_stage6a_arithmetic as prior_audit
    from scripts.issue53_stage6b1_protocol_loader import load_protocol
else:
    import audit_issue53_stage6a_arithmetic as prior_audit
    from issue53_stage6b1_protocol_loader import load_protocol

protocol = load_protocol()


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT_FORMAT = "issue53_stage6b1_independent_arithmetic_audit_v1"
COLLECTION_FORMAT = "issue53_stage6b1_gap_l1_screen_collection_v1"
CALIBRATION_FORMAT = "issue53_stage6b1_gap_l1_calibration_v1"
STRUCTURAL_FORMAT = "issue53_stage6b1_structural_replay_audit_v1"
EVALUATION_FORMAT = "issue53_stage6b1_frozen_evaluation_v1"

SAMPLING_PARAMS = {
    "beta": 1.0,
    "h": 0.8,
    "distance_mode": "geometric",
    "lambda_param": 0.5,
    "alpha": 16.0,
    "delta": 0.05,
    "winsorize_quantiles": (0.01, 0.99),
    "exclude_self": True,
    "scale_invariant": True,
    "scale_invariant_min_spread": 1e-3,
}

AUDIT_BOUNDARY = {
    "stage6b1_collector_imported": False,
    "stage6b1_shared_arithmetic_imported": False,
    "stage6b1_structural_auditor_imported": False,
    "stage6b1_frozen_evaluator_imported": False,
    "protocol_classification_helpers_called": False,
    "queries_reevaluated_from_definitions": True,
    "calibration_scores_independently_recomputed": True,
    "gap_microsteps_independently_replayed": True,
    "sparse_tables_independently_reconstructed": True,
    "exact_gap_metrics_independently_recomputed": True,
    "old_squared_geometry_independently_recomputed": True,
    "seed_equal_weight_thresholds_independently_recomputed": True,
    "new_generation_performed": False,
    "proposal_acceptance_or_rejection": False,
    "reference_or_heldout_table_read": False,
}
if getattr(protocol, "GAP_L1_DEVICE", "numpy") == "cuda":
    AUDIT_BOUNDARY.update({
        "production_gap_cuda_math_imported": False,
        "independent_cuda_condition_math_used": True,
        "independent_cuda_final_recount_used": True,
    })


def _json_safe(value: Any, path: tuple[str, ...] = ()) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"strict JSON 禁止非有限数：{'.'.join(path)}")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, path + (str(key),))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path + (str(index),))
            for index, item in enumerate(value)
        ]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), path)
    if hasattr(value, "item"):
        return _json_safe(value.item(), path)
    raise TypeError(f"不可严格 JSON 序列化：{type(value)!r}")


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values)
    payload = (
        array.dtype.str.encode("utf-8")
        + repr(array.shape).encode("utf-8")
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.reset_index(drop=True).to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _strict_load(path: str | Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"JSON 包含非标准数值：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        result = json.load(handle, parse_constant=reject)
    if not isinstance(result, dict):
        raise TypeError("artifact JSON 根必须是对象")
    return result


def _exclusive_write(path: str | Path, value: Any) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"独立算术审计输出已存在，不覆盖：{output}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                _json_safe(value),
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError as error:
            raise FileExistsError(
                f"独立算术审计输出已存在，不覆盖：{output}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _git_identity() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "commit": commit,
        "worktree_clean_including_untracked": not status,
        "status": status,
    }


def _validate_execution(
    mode: str,
    git: Mapping[str, Any],
    confirmed_execution_commit: str | None,
) -> dict[str, Any]:
    if not git.get("worktree_clean_including_untracked"):
        raise RuntimeError("独立算术审计要求包含未跟踪文件在内的干净工作树")
    if confirmed_execution_commit != git.get("commit"):
        raise PermissionError("必须精确确认当前干净实现提交")
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if mode == "formal":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is None or not visible.strip() or "," in visible:
            raise RuntimeError("正式独立重放必须显式且只暴露一张 GPU")
    elif mode != "smoke":
        raise ValueError("mode 必须是 formal 或 smoke")
    if hasattr(protocol, "validate_runtime_environment"):
        environment.update(protocol.validate_runtime_environment(mode))
    return environment


def _rng_sha256(rng: np.random.Generator) -> str:
    return hashlib.sha256(json.dumps(
        _json_safe(rng.bit_generator.state),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load_queries(dataset: str) -> list[dict[str, Any]]:
    with (REPOSITORY_ROOT / protocol.DATASETS[dataset]["queries"]).open(
        encoding="utf-8"
    ) as handle:
        value = json.load(handle)
    queries = value.get("queries")
    if not isinstance(queries, list):
        raise RuntimeError("measured query 文件结构无效")
    return queries


def _snapshot_frame(snapshot: Mapping[str, Any]) -> pd.DataFrame:
    columns = snapshot.get("table_columns")
    records = snapshot.get("table_records")
    if not isinstance(columns, list) or not isinstance(records, list):
        raise RuntimeError("冻结当前表快照结构无效")
    return pd.DataFrame.from_records(records, columns=columns)


@dataclass(frozen=True)
class _SourceContext:
    dataset: str
    seed: int
    group: str
    mode: str
    current: pd.DataFrame
    schema: Any
    queries: list[dict[str, Any]]
    target: np.ndarray
    source_target: np.ndarray
    runtime_n: int
    source_n: int
    q: np.ndarray
    residual: np.ndarray
    fitness: np.ndarray
    probabilities: Any
    device: str
    trajectory: Mapping[str, Any]
    source_proposal_state: Mapping[str, Any]


def _build_source_context(
    dataset: str,
    seed: int,
    group: str,
    mode: str,
    state_library: Mapping[str, Any],
    source_state_index: Mapping[str, Mapping[str, Any]],
    trajectory_index: Mapping[tuple[str, int], Mapping[str, Any]],
    source_proposal_index: Mapping[str, Mapping[str, Any]],
) -> _SourceContext:
    identifier = protocol.state_id(dataset, seed, group, mode=mode)
    state = source_state_index[identifier]
    trajectory = trajectory_index[(dataset, seed)]
    proposal_state = source_proposal_index[identifier]
    current = _snapshot_frame(state["snapshot"])
    queries = _load_queries(dataset)
    schema = load_schema(str(REPOSITORY_ROOT / protocol.DATASETS[dataset]["schema"]))
    target_record = state_library["runtime_targets"][dataset]
    target = np.asarray(target_record["target_values"], dtype=np.float64)
    source_target = np.asarray(
        [query["result"] for query in queries], dtype=np.int64
    )
    runtime_n = int(target_record["runtime_n_records"])
    source_n = int(protocol.DATASETS[dataset]["n_records"])
    if not np.array_equal(
        target,
        source_target.astype(np.float64) * (runtime_n / source_n),
    ):
        raise RuntimeError("独立审计运行目标缩放失败")
    device = str(proposal_state["runtime_device"])
    q, residual, fitness = evaluate_vectorized(
        current,
        queries,
        schema,
        target=target,
        n_records=len(current),
        batch_size=256,
        device=device,
        want_fitness=True,
        verbose=False,
        residual_geometry="relative",
        residual_geometry_floor=protocol.GAP_L1_FLOOR,
    )
    q = np.asarray(q, dtype=np.int64)
    residual = np.asarray(residual, dtype=np.float64)
    fitness = np.asarray(fitness)
    independent_q = prior_audit._query_counts(current, queries)
    if (
        not np.array_equal(q, independent_q)
        or _frame_sha256(current) != state["snapshot"]["current_table_sha256"]
        or _array_sha256(q) != proposal_state["current_query_answers_sha256"]
        or _array_sha256(residual) != proposal_state["residual_signal_sha256"]
        or _array_sha256(fitness) != proposal_state["fitness_sha256"]
    ):
        raise RuntimeError("独立审计当前表/查询/残差信息对拍失败")
    distances = pairwise_block_distance(
        current,
        current,
        schema,
        device=device,
        return_tensor=device in ("cuda", "cpu"),
    )
    probabilities = compute_sampling_probs(
        fitness,
        distances,
        beta=SAMPLING_PARAMS["beta"],
        h=SAMPLING_PARAMS["h"],
        device=device,
        distance_mode=SAMPLING_PARAMS["distance_mode"],
        lambda_param=SAMPLING_PARAMS["lambda_param"],
        alpha=SAMPLING_PARAMS["alpha"],
        delta=SAMPLING_PARAMS["delta"],
        winsorize_quantiles=SAMPLING_PARAMS["winsorize_quantiles"],
        exclude_self=SAMPLING_PARAMS["exclude_self"],
        scale_invariant=SAMPLING_PARAMS["scale_invariant"],
        scale_invariant_min_spread=SAMPLING_PARAMS[
            "scale_invariant_min_spread"
        ],
    )
    return _SourceContext(
        dataset=dataset,
        seed=seed,
        group=group,
        mode=mode,
        current=current.reset_index(drop=True),
        schema=schema,
        queries=queries,
        target=target,
        source_target=source_target,
        runtime_n=runtime_n,
        source_n=source_n,
        q=q,
        residual=residual,
        fitness=fitness,
        probabilities=probabilities,
        device=device,
        trajectory=trajectory,
        source_proposal_state=proposal_state,
    )


def _source_pair(context: _SourceContext, proposal_index: int) -> Mapping[str, Any]:
    pair = context.source_proposal_state["pairs"][proposal_index]
    expected = protocol.pair_id(
        context.dataset,
        context.seed,
        context.group,
        proposal_index,
        mode=context.mode,
    )
    if pair.get("pair_id") != expected:
        raise RuntimeError("来源候选地址身份失败")
    return pair


def _replay_donor(
    context: _SourceContext,
    proposal_index: int,
) -> tuple[Mapping[str, Any], np.ndarray, pd.DataFrame]:
    pair = _source_pair(context, proposal_index)
    seed = protocol.stage6a.proposal_address_seed(
        context.dataset,
        context.seed,
        context.group,
        proposal_index,
        "donor",
        mode=context.mode,
    )
    rng = np.random.default_rng(seed)
    initial = _rng_sha256(rng)
    indices = np.asarray(sample_donors(
        context.probabilities, rng, device=context.device
    ), dtype=np.int64)
    if (
        pair["rng"]["donor_address_uint64"] != seed
        or pair["rng"]["donor_initial_state_sha256"] != initial
        or pair["rng"]["donor_endpoint_state_sha256"] != _rng_sha256(rng)
        or pair["donor_indices_sha256"] != _array_sha256(indices)
    ):
        raise RuntimeError("独立供体重放失败")
    return pair, indices, context.current.iloc[indices].reset_index(drop=True)


def _tilted_probabilities(
    scores: np.ndarray,
    strength: float,
) -> np.ndarray:
    logits = np.clip(strength * np.asarray(scores, dtype=np.float64), -30.0, 30.0)
    probabilities = np.empty_like(logits)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    probabilities[np.asarray(scores) == 0.0] = 0.5
    return probabilities


def _replay_common_update(
    context: _SourceContext,
    proposal_index: int,
    donors: pd.DataFrame,
    source_pair: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
    direction = np.asarray(compute_copy_direction_scores(
        context.current,
        donors,
        context.schema,
        context.queries,
        context.residual,
        batch_size=256,
        device=context.device,
    ), dtype=np.float64)
    if _array_sha256(direction) != source_pair["direction_scores_sha256"]:
        raise RuntimeError("独立 B 方向重放失败")
    reference = float(context.trajectory["direction_reference_scale"])
    strength = protocol.B_STRENGTH / reference
    update_seed = protocol.stage6a.proposal_address_seed(
        context.dataset,
        context.seed,
        context.group,
        proposal_index,
        "update",
        mode=context.mode,
    )
    rng = np.random.default_rng(update_seed)
    initial_rng_hash = _rng_sha256(rng)
    participate = rng.random(len(context.current)) < protocol.RHO
    attributes = context.schema.attribute_names()
    current_values = context.current.loc[:, attributes].to_numpy()
    donor_values = donors.loc[:, attributes].to_numpy()
    mask = np.zeros((len(context.current), len(attributes)), dtype=bool)
    probabilities = np.empty_like(direction)
    for attribute_index in range(len(attributes)):
        probabilities[:, attribute_index] = _tilted_probabilities(
            direction[:, attribute_index], strength
        )
        rolls = rng.random(len(context.current))
        mask[:, attribute_index] = (
            participate
            & (current_values[:, attribute_index] != donor_values[:, attribute_index])
            & (rolls < probabilities[:, attribute_index])
        )
    mutation_rolls = rng.random(len(context.current))
    pre_mutation_hash = _rng_sha256(rng)
    mutation_events = []
    for row_raw in np.flatnonzero(participate & (mutation_rolls < protocol.MU)):
        row = int(row_raw)
        attribute_index = int(rng.integers(0, len(attributes)))
        attribute = attributes[attribute_index]
        block = context.schema.get_block(attribute)
        if block.is_numeric():
            low, high = block.range
            sampled: Any = int(rng.integers(int(low), int(high) + 1))
        else:
            sampled = block.values[int(rng.integers(0, len(block.values)))]
        mutation_events.append({
            "row_index": row,
            "attribute_index": attribute_index,
            "attribute": attribute,
            "sampled_value": sampled,
        })
    stored_specs = [
        {
            "row_index": int(event["row_index"]),
            "attribute_index": int(event["attribute_index"]),
            "attribute": event["attribute"],
            "sampled_value": event["sampled_value"],
        }
        for event in source_pair["mutation_events"]
    ]
    stored_mask = np.zeros_like(mask)
    for attribute_index, attribute in enumerate(attributes):
        stored_mask[
            np.asarray(
                source_pair["copy_row_indices_by_attribute"][attribute],
                dtype=np.intp,
            ),
            attribute_index,
        ] = True
    if (
        source_pair["rng"]["update_address_uint64"] != update_seed
        or source_pair["rng"]["update_initial_state_sha256"] != initial_rng_hash
        or source_pair["rng"]["shared_pre_mutation_state_sha256"]
        != pre_mutation_hash
        or source_pair["rng"]["full_endpoint_state_sha256"] != _rng_sha256(rng)
        or source_pair["participating_row_indices"]
        != np.flatnonzero(participate).tolist()
        or not np.array_equal(mask, stored_mask)
        or stored_specs != mutation_events
        or source_pair["copy_probabilities_sha256"]
        != _array_sha256(probabilities)
    ):
        raise RuntimeError("独立参与、初始开关或突变随机重放失败")
    return participate, mask, mutation_events, direction


@dataclass
class _IndependentGapPlan:
    attributes: tuple[str, ...]
    queries: Sequence[Mapping[str, Any]]
    active_rows: np.ndarray
    coordinates: np.ndarray
    row_lookup: np.ndarray
    query_indices_by_attribute: tuple[np.ndarray, ...]
    current_attr_failures: tuple[tuple[np.ndarray, ...], ...]
    donor_attr_failures: tuple[tuple[np.ndarray, ...], ...]
    failures: np.ndarray
    indicators: np.ndarray
    plan_counts: np.ndarray
    target: np.ndarray
    denominators: np.ndarray
    error_terms: np.ndarray
    error_sum: float
    mask: np.ndarray


def _prepare_independent_gap(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    queries: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    q: np.ndarray,
    participate: np.ndarray,
    initial_mask: np.ndarray,
) -> _IndependentGapPlan:
    attributes = tuple(current.columns)
    positions = {name: index for index, name in enumerate(attributes)}
    current_values = current.loc[:, list(attributes)].to_numpy()
    donor_values = donors.loc[:, list(attributes)].to_numpy()
    active = participate[:, None] & (current_values != donor_values)
    if np.any(initial_mask & ~active):
        raise RuntimeError("独立新核初始开关越出活跃域")
    coordinates = np.argwhere(active).astype(np.intp)
    active_rows = np.flatnonzero(np.any(active, axis=1)).astype(np.intp)
    row_lookup = np.full(len(current), -1, dtype=np.intp)
    row_lookup[active_rows] = np.arange(len(active_rows))
    pair = pd.concat(
        [current.iloc[active_rows], donors.iloc[active_rows]],
        ignore_index=True,
    )
    condition_truth = []
    query_condition_indices = []
    per_attribute: list[dict[int, list[int]]] = [
        {} for _ in attributes
    ]
    for query_index, query in enumerate(queries):
        indices = []
        for condition in query["conditions"]:
            condition_index = len(condition_truth)
            truth = prior_audit._condition_mask(pair, condition)
            condition_truth.append(truth)
            indices.append(condition_index)
            attribute_index = positions[condition["attribute"]]
            per_attribute[attribute_index].setdefault(query_index, []).append(
                condition_index
            )
        query_condition_indices.append(indices)
    if condition_truth:
        truth_matrix = np.column_stack(condition_truth).astype(bool)
    else:
        truth_matrix = np.empty((len(pair), 0), dtype=bool)
    n_active_rows = len(active_rows)
    current_truth = truth_matrix[:n_active_rows]
    donor_truth = truth_matrix[n_active_rows:]
    query_indices_by_attribute = []
    current_attr_failures = []
    donor_attr_failures = []
    for mapping in per_attribute:
        query_indices = np.asarray(sorted(mapping), dtype=np.intp)
        query_indices_by_attribute.append(query_indices)
        current_values_by_query = []
        donor_values_by_query = []
        for query_index in query_indices:
            indices = np.asarray(mapping[int(query_index)], dtype=np.intp)
            current_values_by_query.append(
                np.sum(~current_truth[:, indices], axis=1, dtype=np.int16)
            )
            donor_values_by_query.append(
                np.sum(~donor_truth[:, indices], axis=1, dtype=np.int16)
            )
        current_attr_failures.append(tuple(current_values_by_query))
        donor_attr_failures.append(tuple(donor_values_by_query))
    failures = np.zeros((n_active_rows, len(queries)), dtype=np.int16)
    for query_index, indices in enumerate(query_condition_indices):
        failures[:, query_index] = np.sum(
            ~current_truth[:, np.asarray(indices, dtype=np.intp)],
            axis=1,
            dtype=np.int16,
        )
    current_indicators = failures == 0
    mask = np.asarray(initial_mask, dtype=bool).copy()
    for local_row, row in enumerate(active_rows):
        for attribute_raw in np.flatnonzero(mask[row]):
            attribute = int(attribute_raw)
            query_indices = query_indices_by_attribute[attribute]
            current_fail = np.asarray([
                values[local_row]
                for values in current_attr_failures[attribute]
            ], dtype=np.int16)
            donor_fail = np.asarray([
                values[local_row]
                for values in donor_attr_failures[attribute]
            ], dtype=np.int16)
            failures[local_row, query_indices] += donor_fail - current_fail
    indicators = failures == 0
    plan_counts = np.asarray(q, dtype=np.float64).copy()
    if n_active_rows:
        plan_counts += np.sum(
            indicators.astype(np.int64) - current_indicators.astype(np.int64),
            axis=0,
            dtype=np.int64,
        )
    target_values = np.asarray(target, dtype=np.float64)
    denominators = np.maximum(target_values, protocol.GAP_L1_FLOOR)
    error_terms = np.abs(target_values - plan_counts) / denominators
    return _IndependentGapPlan(
        attributes=attributes,
        queries=queries,
        active_rows=active_rows,
        coordinates=coordinates,
        row_lookup=row_lookup,
        query_indices_by_attribute=tuple(query_indices_by_attribute),
        current_attr_failures=tuple(current_attr_failures),
        donor_attr_failures=tuple(donor_attr_failures),
        failures=failures,
        indicators=indicators,
        plan_counts=plan_counts,
        target=target_values,
        denominators=denominators,
        error_terms=error_terms,
        error_sum=float(np.sum(error_terms, dtype=np.float64)),
        mask=mask,
    )


def _attr_failures(values: tuple[np.ndarray, ...], row: int) -> np.ndarray:
    return np.asarray([item[row] for item in values], dtype=np.int16)


def _independent_condition(
    plan: _IndependentGapPlan,
    row: int,
    attribute: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local = int(plan.row_lookup[row])
    query_indices = plan.query_indices_by_attribute[attribute]
    if len(query_indices) == 0:
        value = plan.error_sum / len(plan.queries)
        empty_int = np.zeros(0, dtype=np.int16)
        empty_bool = np.zeros(0, dtype=bool)
        return value, value, empty_int, empty_int, empty_bool, empty_bool
    current_fail = _attr_failures(plan.current_attr_failures[attribute], local)
    donor_fail = _attr_failures(plan.donor_attr_failures[attribute], local)
    old_fail = donor_fail if plan.mask[row, attribute] else current_fail
    base = plan.failures[local, query_indices] - old_fail
    failures0 = base + current_fail
    failures1 = base + donor_fail
    indicators0 = failures0 == 0
    indicators1 = failures1 == 0
    old_indicators = plan.indicators[local, query_indices]
    counts0 = (
        plan.plan_counts[query_indices]
        + indicators0.astype(np.int8) - old_indicators.astype(np.int8)
    )
    counts1 = (
        plan.plan_counts[query_indices]
        + indicators1.astype(np.int8) - old_indicators.astype(np.int8)
    )
    terms0 = np.abs(plan.target[query_indices] - counts0) / plan.denominators[
        query_indices
    ]
    terms1 = np.abs(plan.target[query_indices] - counts1) / plan.denominators[
        query_indices
    ]
    old_terms = plan.error_terms[query_indices]
    sum0 = plan.error_sum - float(np.sum(old_terms)) + float(np.sum(terms0))
    sum1 = plan.error_sum - float(np.sum(old_terms)) + float(np.sum(terms1))
    return (
        sum0 / len(plan.queries),
        sum1 / len(plan.queries),
        failures0,
        failures1,
        indicators0,
        indicators1,
    )


def _independent_set(
    plan: _IndependentGapPlan,
    row: int,
    attribute: int,
    selected: bool,
    failures: np.ndarray,
    indicators: np.ndarray,
) -> None:
    if selected == bool(plan.mask[row, attribute]):
        return
    local = int(plan.row_lookup[row])
    query_indices = plan.query_indices_by_attribute[attribute]
    if len(query_indices):
        old_indicators = plan.indicators[local, query_indices]
        plan.plan_counts[query_indices] += (
            indicators.astype(np.int8) - old_indicators.astype(np.int8)
        )
        plan.failures[local, query_indices] = failures
        plan.indicators[local, query_indices] = indicators
        new_terms = np.abs(
            plan.target[query_indices] - plan.plan_counts[query_indices]
        ) / plan.denominators[query_indices]
        plan.error_sum += float(
            np.sum(new_terms, dtype=np.float64)
            - np.sum(plan.error_terms[query_indices], dtype=np.float64)
        )
        plan.error_terms[query_indices] = new_terms
    plan.mask[row, attribute] = selected


def _independent_probability(
    score: float,
    scale: float,
) -> tuple[float, float, bool]:
    raw = float(protocol.GAP_L1_STRENGTH * score / scale)
    effective = float(np.clip(raw, -protocol.LOGIT_CLIP, protocol.LOGIT_CLIP))
    if effective >= 0.0:
        probability = float(1.0 / (1.0 + np.exp(-effective)))
    else:
        exponential = float(np.exp(effective))
        probability = exponential / (1.0 + exponential)
    if not np.isfinite(raw) or not 0.0 < probability < 1.0:
        raise RuntimeError("独立新核出现非有限或单向条件")
    return probability, raw, raw != effective


def _materialize_gap(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    mask: np.ndarray,
) -> pd.DataFrame:
    result = current.copy(deep=True)
    for attribute_index, attribute in enumerate(current.columns):
        selected = mask[:, attribute_index]
        if np.any(selected):
            values = result[attribute].to_numpy().copy()
            values[selected] = donors[attribute].to_numpy()[selected]
            result[attribute] = values
    return result


def _independent_cuda_module():
    if __package__:
        from scripts import issue53_stage6b1b_independent_cuda as module
    else:
        import issue53_stage6b1b_independent_cuda as module
    return module


def _independent_gap_replay(
    context: _SourceContext,
    donors: pd.DataFrame,
    participate: np.ndarray,
    initial_mask: np.ndarray,
    scale: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    if getattr(protocol, "GAP_L1_DEVICE", "numpy") == "cuda":
        return _independent_cuda_module().replay_gap_l1_cuda(
            context.current,
            donors,
            context.schema,
            context.queries,
            context.target,
            context.q,
            participate,
            initial_mask,
            reference_scale=scale,
            seed=seed,
            n_sweeps=protocol.GIBBS_SWEEPS,
            eta=protocol.ETA,
            strength=protocol.GAP_L1_STRENGTH,
            floor=protocol.GAP_L1_FLOOR,
            logit_clip=protocol.LOGIT_CLIP,
        )
    plan = _prepare_independent_gap(
        context.current,
        donors,
        context.queries,
        context.target,
        context.q,
        participate,
        initial_mask,
    )
    rng = np.random.default_rng(seed)
    initial_rng_sha = _rng_sha256(rng)
    trace = hashlib.sha256()
    clip_hits = 0
    k = len(plan.coordinates)
    steps = protocol.GIBBS_SWEEPS * k
    for step in range(steps):
        coordinate = int(rng.integers(0, k))
        row = int(plan.coordinates[coordinate, 0])
        attribute = int(plan.coordinates[coordinate, 1])
        before = bool(plan.mask[row, attribute])
        e0, e1, f0, f1, i0, i1 = _independent_condition(
            plan, row, attribute
        )
        score = float(e0 - e1)
        normalized = float(score / scale)
        probability, raw_logit, clipped = _independent_probability(score, scale)
        roll = float(rng.random())
        after = bool(roll < probability)
        _independent_set(
            plan,
            row,
            attribute,
            after,
            f1 if after else f0,
            i1 if after else i0,
        )
        trace.update(struct.pack(
            "<qqqddddddd???",
            step,
            row,
            attribute,
            e0,
            e1,
            score,
            normalized,
            raw_logit,
            probability,
            roll,
            before,
            after,
            clipped,
        ))
        clip_hits += int(clipped)
    table = _materialize_gap(context.current, donors, plan.mask)
    recounted = prior_audit._query_counts(table, context.queries)
    if not np.array_equal(recounted, plan.plan_counts):
        raise RuntimeError("独立增量查询计数与完整表不一致")
    return table, plan.mask, {
        "initial_rng_sha256": initial_rng_sha,
        "endpoint_rng_sha256": _rng_sha256(rng),
        "trace_sha256": trace.hexdigest(),
        "active_switches_k": k,
        "microsteps": steps,
        "clip_hits": clip_hits,
        "final_query_counts": recounted,
    }


def _isolated_score_summary(
    values: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    nonzero = values[values != 0.0]
    if len(nonzero):
        absolute = np.abs(nonzero)
        maximum = float(np.max(absolute))
        rms = float(maximum * np.sqrt(np.mean((nonzero / maximum) ** 2)))
        return rms, {
            "total_count": len(values),
            "nonzero_count": len(nonzero),
            "zero_count": len(values) - len(nonzero),
            "absolute_min": float(np.min(absolute)),
            "absolute_q25": float(np.quantile(absolute, 0.25)),
            "absolute_median": float(np.quantile(absolute, 0.5)),
            "absolute_q75": float(np.quantile(absolute, 0.75)),
            "rms": rms,
            "absolute_max": maximum,
            "absolute_max_over_rms": float(maximum / rms),
        }
    return 0.0, {
        "total_count": len(values),
        "nonzero_count": 0,
        "zero_count": len(values),
        "absolute_min": None,
        "absolute_q25": None,
        "absolute_median": None,
        "absolute_q75": None,
        "rms": 0.0,
        "absolute_max": None,
        "absolute_max_over_rms": None,
    }


def _independent_isolated_scores(
    context: _SourceContext,
    donors: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    if getattr(protocol, "GAP_L1_DEVICE", "numpy") == "cuda":
        coordinates, values, _ = (
            _independent_cuda_module().isolated_gap_l1_scores_cuda(
                context.current,
                donors,
                context.schema,
                context.queries,
                context.target,
                context.q,
                floor=protocol.GAP_L1_FLOOR,
                exact_target_numerators=(
                    context.source_target * context.runtime_n
                ),
                exact_target_denominator=context.source_n,
            )
        )
        rms, distribution = _isolated_score_summary(values)
        return coordinates, values, rms, distribution
    plan = _prepare_independent_gap(
        context.current,
        donors,
        context.queries,
        context.target,
        context.q,
        np.ones(len(context.current), dtype=bool),
        np.zeros((len(context.current), len(context.current.columns)), dtype=bool),
    )
    target_numerators = [
        int(value) * context.runtime_n for value in context.source_target
    ]
    raw_denominators = [
        max(value, int(protocol.GAP_L1_FLOOR) * context.source_n)
        for value in target_numerators
    ]
    divisors = [
        math.gcd(math.gcd(abs(value), context.source_n), denominator)
        for value, denominator in zip(target_numerators, raw_denominators)
    ]
    reduced_denominators = [
        denominator // divisor
        for denominator, divisor in zip(raw_denominators, divisors)
    ]
    common_multiple = math.lcm(*reduced_denominators)
    weights = [
        common_multiple // denominator for denominator in reduced_denominators
    ]
    scores = []
    for row_raw, attribute_raw in plan.coordinates:
        row = int(row_raw)
        attribute = int(attribute_raw)
        e0, e1, _, _, indicators0, indicators1 = _independent_condition(
            plan, row, attribute
        )
        score = float(e0 - e1)
        query_indices = plan.query_indices_by_attribute[attribute]
        local_row = int(plan.row_lookup[row])
        old_indicators = plan.indicators[local_row, query_indices]
        counts0 = (
            plan.plan_counts[query_indices]
            + indicators0.astype(np.int8)
            - old_indicators.astype(np.int8)
        ).astype(np.int64)
        counts1 = (
            plan.plan_counts[query_indices]
            + indicators1.astype(np.int8)
            - old_indicators.astype(np.int8)
        ).astype(np.int64)
        exact_units = 0
        for local_index, query_index_raw in enumerate(query_indices):
            query_index = int(query_index_raw)
            residual0 = abs(
                target_numerators[query_index]
                - int(counts0[local_index]) * context.source_n
            )
            residual1 = abs(
                target_numerators[query_index]
                - int(counts1[local_index]) * context.source_n
            )
            exact_units += (
                residual0 // divisors[query_index]
                - residual1 // divisors[query_index]
            ) * weights[query_index]
        if exact_units == 0:
            score = 0.0
        scores.append(score)
    values = np.asarray(scores, dtype=np.float64)
    rms, distribution = _isolated_score_summary(values)
    return plan.coordinates, values, rms, distribution


def _sparse_query_counts(
    before: Sequence[int],
    sparse: Sequence[Sequence[int]],
) -> np.ndarray:
    result = np.asarray(before, dtype=np.int64).copy()
    seen = set()
    for item in sparse:
        if not isinstance(item, list) or len(item) != 2:
            raise RuntimeError("稀疏查询改变量结构失败")
        index, delta = item
        if (
            isinstance(index, bool)
            or isinstance(delta, bool)
            or not isinstance(index, int)
            or not isinstance(delta, int)
            or not 0 <= index < len(result)
            or index in seen
            or delta == 0
        ):
            raise RuntimeError("稀疏查询改变量索引/数值失败")
        seen.add(index)
        result[index] += delta
    return result


def _gap_error_float(context: _SourceContext, counts: np.ndarray) -> float:
    return float(np.mean(
        np.abs(context.target - np.asarray(counts, dtype=float))
        / np.maximum(context.target, protocol.GAP_L1_FLOOR)
    ))


def _independent_query_geometry(
    context: _SourceContext,
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    delta = np.asarray(after, dtype=np.int64) - np.asarray(before, dtype=np.int64)
    target = context.source_target * context.runtime_n
    residual = target - np.asarray(before, dtype=np.int64) * context.source_n
    scaled_delta = delta * context.source_n
    denominators = np.maximum(
        target, int(protocol.GAP_L1_FLOOR) * context.source_n
    )
    labels = {
        "no_move": 0,
        "toward_not_crossed": 0,
        "crossed_target": 0,
        "away_from_target": 0,
        "left_exact_target": 0,
    }
    crossed = 0.0
    left = 0.0
    for r_raw, d_raw, denominator_raw in zip(
        residual, scaled_delta, denominators
    ):
        r, d, denominator = int(r_raw), int(d_raw), int(denominator_raw)
        if d == 0:
            labels["no_move"] += 1
        elif r == 0:
            labels["left_exact_target"] += 1
            left += abs(d) / denominator
        elif r * d < 0:
            labels["away_from_target"] += 1
        elif abs(d) <= abs(r):
            labels["toward_not_crossed"] += 1
        else:
            labels["crossed_target"] += 1
            crossed += (abs(d) - abs(r)) / denominator
    exact_before = int(np.sum(residual == 0))
    exact_after = int(np.sum(target - np.asarray(after) * context.source_n == 0))
    return {
        "labels": labels,
        "weighted_crossed_amount_float": float(crossed),
        "weighted_left_exact_amount_float": float(left),
        "query_l1_displacement": int(np.sum(np.abs(delta), dtype=np.int64)),
        "query_l2_squared_displacement": int(np.dot(delta, delta)),
        "exact_query_count_before": exact_before,
        "exact_query_count_after": exact_after,
        "exact_query_count_change": exact_after - exact_before,
    }


def _independent_old_squared(
    context: _SourceContext,
    copy_table: pd.DataFrame,
    full_table: pd.DataFrame,
) -> dict[str, Any]:
    columns = context.schema.attribute_names()
    copy_rows, copy_deltas = prior_audit._row_query_deltas(
        context.current, copy_table, context.queries, columns
    )
    mutation_rows, mutation_deltas = prior_audit._row_query_deltas(
        copy_table, full_table, context.queries, columns
    )
    full_rows, full_deltas = prior_audit._row_query_deltas(
        context.current, full_table, context.queries, columns
    )
    residual = (
        context.source_target * context.runtime_n
        - context.q * context.source_n
    ).astype(np.int64)
    copy = prior_audit._exact_decomposition(
        copy_deltas, residual, context.source_n
    )
    copy_delta = np.asarray(copy["delta_q"], dtype=np.int64)
    mutation = prior_audit._exact_decomposition(
        mutation_deltas,
        residual - context.source_n * copy_delta,
        context.source_n,
    )
    full = prior_audit._exact_decomposition(
        full_deltas, residual, context.source_n
    )
    sequential = prior_audit._sequential_payload(copy, mutation, full)
    return {
        "copy_only": copy,
        "mutation_given_copy": mutation,
        "full": full,
        "sequential": sequential,
        "changed_row_counts": {
            "copy_only": len(copy_rows),
            "mutation_given_copy": len(mutation_rows),
            "full": len(full_rows),
        },
    }


def _audit_arm_record(
    context: _SourceContext,
    donor_indices: np.ndarray,
    donors: pd.DataFrame,
    arm: Mapping[str, Any],
    *,
    query_device: str = "numpy",
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    columns = context.schema.attribute_names()
    copy_table, full_table, copied_cells = prior_audit._apply_sparse_logs(
        context.current,
        arm["copy_edits"],
        arm["mutation_events"],
        columns,
    )
    if (
        _frame_sha256(copy_table) != arm["table_sha256"]["copy_only"]
        or _frame_sha256(full_table) != arm["table_sha256"]["full"]
    ):
        raise RuntimeError("独立稀疏表重建哈希失败")
    for edit in arm["copy_edits"]:
        row = int(edit["row_index"])
        if int(edit["donor_index"]) != int(donor_indices[row]):
            raise RuntimeError("独立复制日志供体编号失败")
        for cell in edit["cells"]:
            if donors.at[row, cell["attribute"]] != cell["after"]:
                raise RuntimeError("独立复制日志不是供体值")
    if query_device == "cuda":
        cuda = _independent_cuda_module()
        copy_q = cuda.query_counts_cuda(
            copy_table, context.schema, context.queries
        )
        full_q = cuda.query_counts_cuda(
            full_table, context.schema, context.queries
        )
    elif query_device == "numpy":
        copy_q = prior_audit._query_counts(copy_table, context.queries)
        full_q = prior_audit._query_counts(full_table, context.queries)
    else:
        raise ValueError("独立查询计数设备只支持 numpy 或 cuda")
    if (
        not np.array_equal(
            copy_q,
            _sparse_query_counts(
                context.q, arm["query_delta"]["copy_only"]
            ),
        )
        or not np.array_equal(
            full_q,
            _sparse_query_counts(context.q, arm["query_delta"]["full"]),
        )
    ):
        raise RuntimeError("独立查询计数与稀疏改变量失败")
    current_error = _gap_error_float(context, context.q)
    copy_error = _gap_error_float(context, copy_q)
    full_error = _gap_error_float(context, full_q)
    expected_gap = {
        "current": current_error,
        "copy_only": {
            "after": copy_error,
            "gain": current_error - copy_error,
            "positive_gain": max(current_error - copy_error, 0.0),
            "negative_harm": max(copy_error - current_error, 0.0),
        },
        "full": {
            "after": full_error,
            "gain": current_error - full_error,
            "positive_gain": max(current_error - full_error, 0.0),
            "negative_harm": max(full_error - current_error, 0.0),
        },
    }
    if arm["gap_l1_float_reading_only"] != expected_gap:
        raise RuntimeError("阅读用浮点绝对误差重算失败")
    expected_geometry = {
        "copy_only": _independent_query_geometry(context, context.q, copy_q),
        "full": _independent_query_geometry(context, context.q, full_q),
    }
    if arm["query_geometry"] != expected_geometry:
        raise RuntimeError("查询方向/越界独立复算失败")
    expected_squared = _independent_old_squared(
        context, copy_table, full_table
    )
    if arm["old_squared_geometry"] != expected_squared:
        raise RuntimeError("旧 B/C/G 与复制突变恒等式独立复算失败")
    if arm["work"]["changed_cells_copy_only"] != copied_cells:
        raise RuntimeError("复制单元格工作量复算失败")
    return copy_table, full_table, copy_q, full_q


@dataclass(frozen=True)
class _ExactSystem:
    targets: tuple[int, ...]
    source_denominator: int
    divisors: tuple[int, ...]
    weights: tuple[int, ...]
    common_multiple: int
    error_denominator: int

    def units(self, counts: Sequence[int]) -> int:
        return sum(
            (
                abs(target - int(q) * self.source_denominator) // divisor
            ) * weight
            for q, target, divisor, weight in zip(
                counts, self.targets, self.divisors, self.weights
            )
        )


def _exact_system(context: _SourceContext) -> _ExactSystem:
    targets = [int(value) * context.runtime_n for value in context.source_target]
    raw_denominators = [
        max(target, int(protocol.GAP_L1_FLOOR) * context.source_n)
        for target in targets
    ]
    divisors = [
        math.gcd(math.gcd(abs(target), context.source_n), denominator)
        for target, denominator in zip(targets, raw_denominators)
    ]
    term_denominators = [
        denominator // divisor
        for denominator, divisor in zip(raw_denominators, divisors)
    ]
    common = math.lcm(*term_denominators)
    weights = [common // value for value in term_denominators]
    return _ExactSystem(
        targets=tuple(targets),
        source_denominator=context.source_n,
        divisors=tuple(divisors),
        weights=tuple(weights),
        common_multiple=common,
        error_denominator=len(targets) * common,
    )


def _exact_record(value: Fraction) -> dict[str, Any]:
    sign = "-" if value.numerator < 0 else ""
    return {
        "numerator_hex": sign + format(abs(value.numerator), "x"),
        "denominator_hex": format(value.denominator, "x"),
        "float_reading_only": float(value),
    }


def _mean(values: Sequence[Fraction]) -> Fraction:
    if not values:
        raise RuntimeError("独立精确均值分母不足")
    return sum(values, Fraction(0, 1)) / len(values)


def _comparison(
    new: Mapping[int, Fraction],
    baseline: Mapping[int, Fraction],
    relation: str,
) -> dict[str, Any]:
    seeds = tuple(new)
    new_mean = _mean([new[seed] for seed in seeds])
    baseline_mean = _mean([baseline[seed] for seed in seeds])
    if relation == "strict_lower":
        per_seed = {seed: new[seed] < baseline[seed] for seed in seeds}
        aggregate = new_mean < baseline_mean
    elif relation == "noninferior_1p05":
        per_seed = {
            seed: new[seed] <= Fraction(105, 100) * baseline[seed]
            for seed in seeds
        }
        aggregate = new_mean <= Fraction(105, 100) * baseline_mean
    else:
        raise RuntimeError("未知冻结比较")
    passing = sum(per_seed.values())
    return {
        "relation": relation,
        "new_equal_seed_mean": _exact_record(new_mean),
        "baseline_equal_seed_mean": _exact_record(baseline_mean),
        "per_seed": {
            str(seed): {
                "new": _exact_record(new[seed]),
                "baseline": _exact_record(baseline[seed]),
                "passes": per_seed[seed],
            }
            for seed in seeds
        },
        "passing_seed_count": passing,
        "passes": bool(
            aggregate and passing >= protocol.SEED_STABILITY_MINIMUM
        ),
    }


def _retention(
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    seeds: Sequence[int],
) -> dict[str, Any]:
    per_seed = {}
    missing = []
    for seed in seeds:
        selected = [
            row for row in rows
            if row["seed"] == seed and row["gain_units"][baseline] > 0
        ]
        denominator = sum(row["gain_units"][baseline] for row in selected)
        if denominator == 0:
            missing.append(seed)
        else:
            per_seed[seed] = Fraction(sum(
                max(row["gain_units"][protocol.ARM_GAP_L1], 0)
                for row in selected
            ), denominator)
    support = not missing
    if support:
        equal_mean = _mean(list(per_seed.values()))
        passing = sum(value >= Fraction(90, 100) for value in per_seed.values())
        passed = bool(
            equal_mean >= Fraction(95, 100)
            and passing >= protocol.SEED_STABILITY_MINIMUM
        )
    else:
        equal_mean = None
        passing = 0
        passed = False
    return {
        "support_sufficient": support,
        "missing_positive_gain_seed_denominators": missing,
        "equal_seed_mean": _exact_record(equal_mean) if equal_mean is not None else None,
        "per_seed": {
            str(seed): _exact_record(value) for seed, value in per_seed.items()
        },
        "passing_seed_count_at_0p90": passing,
        "passes": passed,
    }


def _dataset_label(summary: Mapping[str, bool]) -> str:
    if not summary["stable_gap_error_gain"]:
        return "no_stable_gap_error_gain"
    if not summary["transition_support"]:
        return "insufficient_transition_support"
    if not summary["harm_suppression"]:
        return "gap_gain_without_harm_suppression"
    if not summary["good_step_retention"]:
        return "gap_gain_with_good_step_suppression"
    if not summary["initial_safety"] or not summary["full_safety"]:
        return "gap_mechanism_with_transition_risk"
    return "gap_kernel_development_supported"


def _stage1_label(labels: Mapping[str, str]) -> str:
    if tuple(protocol.DATASET_ORDER) == ("nltcs",):
        if set(labels) != {"nltcs"}:
            raise RuntimeError("独立第二阶段标签覆盖失败")
        label = labels["nltcs"]
        if label in protocol.EXECUTION_FAILURE_LABELS:
            return "inconclusive_or_invalid_screen"
        if label == "gap_kernel_development_supported":
            return "shared_development_support"
        return "dataset_dependent_development_support"
    if set(labels) != {"test_300x10"}:
        raise RuntimeError("独立第一阶段标签覆盖失败")
    label = labels["test_300x10"]
    if label in protocol.EXECUTION_FAILURE_LABELS:
        return "stage1_inconclusive_or_invalid"
    if label == "gap_kernel_development_supported":
        return "advance_to_nltcs_gpu_protocol"
    return "stop_before_nltcs_no_test300_support"


def _independent_dataset_evaluation(
    dataset: str,
    mode: str,
    rows: Sequence[Mapping[str, Any]],
    system: _ExactSystem,
    runtime_n: int,
    source_n: int,
) -> dict[str, Any]:
    seeds = protocol.mode_seeds(mode)

    def seed_means(
        groups: Sequence[str],
        arm: str,
        field: str,
        transform=lambda value: value,
    ) -> dict[int, Fraction]:
        result = {}
        for seed in seeds:
            values = [
                transform(row[field][arm])
                for row in rows
                if row["eligible"]
                and row["seed"] == seed
                and row["state_group"] in groups
            ]
            result[seed] = (
                Fraction(sum(values), len(values) * system.error_denominator)
                if values else Fraction(0, 1)
            )
        return result

    baselines = (protocol.ARM_INDEPENDENT, protocol.ARM_FACTOR)
    new_copy = seed_means(
        protocol.PRIMARY_STATE_GROUPS, protocol.ARM_GAP_L1, "copy_units"
    )
    new_harm = seed_means(
        protocol.PRIMARY_STATE_GROUPS,
        protocol.ARM_GAP_L1,
        "gain_units",
        transform=lambda value: max(-value, 0),
    )
    new_initial = seed_means(
        ("initial",), protocol.ARM_GAP_L1, "copy_units"
    )
    new_full = seed_means(
        protocol.PRIMARY_STATE_GROUPS, protocol.ARM_GAP_L1, "full_units"
    )
    primary_rows = [
        row for row in rows
        if row["eligible"] and row["state_group"] in protocol.PRIMARY_STATE_GROUPS
    ]
    primary_gap = {}
    harm = {}
    retention = {}
    initial = {}
    full = {}
    for baseline in baselines:
        primary_gap[baseline] = _comparison(
            new_copy,
            seed_means(protocol.PRIMARY_STATE_GROUPS, baseline, "copy_units"),
            "strict_lower",
        )
        baseline_harm = seed_means(
            protocol.PRIMARY_STATE_GROUPS,
            baseline,
            "gain_units",
            transform=lambda value: max(-value, 0),
        )
        support = any(value > 0 for value in baseline_harm.values())
        harm[baseline] = {
            "support_sufficient": support,
            **_comparison(new_harm, baseline_harm, "strict_lower"),
        }
        if not support:
            harm[baseline]["passes"] = False
            harm[baseline]["support_label"] = "insufficient_harm_support"
        retention[baseline] = _retention(primary_rows, baseline, seeds)
        initial[baseline] = _comparison(
            new_initial,
            seed_means(("initial",), baseline, "copy_units"),
            "noninferior_1p05",
        )
        full[baseline] = _comparison(
            new_full,
            seed_means(protocol.PRIMARY_STATE_GROUPS, baseline, "full_units"),
            "noninferior_1p05",
        )
    boolean = {
        "stable_gap_error_gain": all(primary_gap[x]["passes"] for x in baselines),
        "transition_support": all(
            harm[x]["support_sufficient"]
            and retention[x]["support_sufficient"]
            for x in baselines
        ),
        "harm_suppression": all(harm[x]["passes"] for x in baselines),
        "good_step_retention": all(retention[x]["passes"] for x in baselines),
        "initial_safety": all(initial[x]["passes"] for x in baselines),
        "full_safety": all(full[x]["passes"] for x in baselines),
    }
    label = _dataset_label(boolean)
    return {
        "dataset": dataset,
        "exact_error_system": {
            "source_denominator": source_n,
            "runtime_n_records": runtime_n,
            "query_count": len(system.targets),
            "common_multiple_hex": format(system.common_multiple, "x"),
            "error_denominator_hex": format(system.error_denominator, "x"),
        },
        "eligible_primary_pair_count": len(primary_rows),
        "primary_gap_error": primary_gap,
        "negative_harm": harm,
        "good_step_retention": retention,
        "initial_safety": initial,
        "full_safety": full,
        "frozen_boolean_summary": boolean,
        "dataset_classification": label,
        "address_exact_metric_sha256": _canonical_sha256([
            {
                "pair_id": row["pair_id"],
                "current_units_hex": format(row["current_units"], "x"),
                "copy_units_hex": {
                    arm: format(row["copy_units"][arm], "x")
                    for arm in protocol.ARMS
                },
                "full_units_hex": {
                    arm: format(row["full_units"][arm], "x")
                    for arm in protocol.ARMS
                },
            }
            for row in rows
        ]),
    }


def _calibration_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {
            "elapsed_sec_diagnostic_only",
            "calibration_scientific_sha256",
            "artifact_paths_diagnostic_only",
        }
    }


def _strip_collection_diagnostic(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_collection_diagnostic(item)
            for key, item in value.items()
            if not key.endswith("_diagnostic_only")
            and not key.endswith("_elapsed_sec")
            and key != "environment"
        }
    if isinstance(value, list):
        return [_strip_collection_diagnostic(item) for item in value]
    return value


def _collection_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return _strip_collection_diagnostic({
        key: item
        for key, item in value.items()
        if key not in {
            "collection_scientific_sha256",
            "artifact_paths_diagnostic_only",
            "environment",
            "elapsed_sec_diagnostic_only",
        }
    })


def _validate_envelopes(
    mode: str,
    git_commit: str,
    calibration: Mapping[str, Any],
    collection: Mapping[str, Any],
    structural: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> None:
    expected_pairs = protocol.expected_pair_ids(mode)
    if (
        calibration.get("calibration_format") != CALIBRATION_FORMAT
        or collection.get("collection_format") != COLLECTION_FORMAT
        or structural.get("structural_audit_format") != STRUCTURAL_FORMAT
        or evaluation.get("evaluation_format") != EVALUATION_FORMAT
        or any(
            item.get("status") != "complete"
            or item.get("mode") != mode
            or item.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
            or item.get("git", {}).get("commit") != git_commit
            for item in (calibration, collection, structural, evaluation)
        )
        or structural.get("audit_passed") is not True
        or tuple(pair.get("pair_id") for pair in collection.get("pairs", []))
        != expected_pairs
        or collection.get("calibration_artifact", {}).get("file_sha256")
        != hashes["calibration"]
        or structural.get("artifact_identity", {}).get(
            "collection_file_sha256"
        ) != hashes["collection"]
        or evaluation.get("artifact_identity", {}).get(
            "structural_audit_file_sha256"
        ) != hashes["structural_audit"]
    ):
        raise RuntimeError("独立审计输入 envelope/产物链失败")
    if calibration.get("calibration_scientific_sha256") != _canonical_sha256(
        _calibration_payload(calibration)
    ):
        raise RuntimeError("参考尺度 scientific SHA-256 失败")
    if collection.get("collection_scientific_sha256") != _canonical_sha256(
        _collection_payload(collection)
    ):
        raise RuntimeError("采集集合 scientific SHA-256 失败")
    if mode == "smoke" and (
        evaluation.get("formal_result_valid") is not False
        or evaluation.get("mechanism_evidence_emitted") is not False
        or evaluation.get("final_screen_classification") is not None
        or evaluation.get("dataset_evaluations") is not None
    ):
        raise RuntimeError("smoke 评价越界发布机制证据")


def _audit_calibrations(
    mode: str,
    calibration: Mapping[str, Any],
    state_library: Mapping[str, Any],
    source_state_index: Mapping[str, Mapping[str, Any]],
    trajectory_index: Mapping[tuple[str, int], Mapping[str, Any]],
    source_proposal_index: Mapping[str, Mapping[str, Any]],
) -> dict[tuple[str, int], float | None]:
    scale_index = {}
    for row in calibration["calibrations"]:
        dataset = row["dataset"]
        seed = int(row["source_seed"])
        context = _build_source_context(
            dataset,
            seed,
            "initial",
            mode,
            state_library,
            source_state_index,
            trajectory_index,
            source_proposal_index,
        )
        exact = bool(np.all(
            context.source_target * context.runtime_n
            - context.q * context.source_n == 0
        ))
        if exact:
            if row["status"] != "already_exact":
                raise RuntimeError("独立定尺发现 already_exact 身份不一致")
            scale_index[(dataset, seed)] = None
            continue
        selected = None
        for searched in row["searched_addresses"]:
            proposal_index = int(searched["proposal_index"])
            source_pair, donor_indices, donors = _replay_donor(
                context, proposal_index
            )
            coordinates, scores, rms, distribution = (
                _independent_isolated_scores(context, donors)
            )
            if (
                searched["donor_indices_sha256"] != _array_sha256(donor_indices)
                or searched["isolated_coordinates_sha256"]
                != _array_sha256(coordinates)
                or searched["isolated_scores_sha256"] != _array_sha256(scores)
                or searched["distribution"] != distribution
                or searched["reference_scale"] != rms
            ):
                raise RuntimeError("独立孤立分数/RMS 定尺复算失败")
            if rms > 0.0 and selected is None:
                selected = proposal_index
        if row["status"] == "calibrated":
            if (
                selected != row["calibration_proposal_index"]
                or float(row["reference_scale"])
                != row["distribution"]["rms"]
            ):
                raise RuntimeError("独立定尺首个非零地址失败")
            scale_index[(dataset, seed)] = float(row["reference_scale"])
        elif row["status"] == "calibration_unsupported":
            if selected is not None:
                raise RuntimeError("unsupported 定尺仍存在非零地址")
            scale_index[(dataset, seed)] = None
        else:
            raise RuntimeError("未知定尺状态")
    return scale_index


def _mutation_specs(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "row_index": int(event["row_index"]),
            "attribute_index": int(event["attribute_index"]),
            "attribute": event["attribute"],
            "sampled_value": event["sampled_value"],
        }
        for event in events
    ]


def _audit_generated_pair(
    context: _SourceContext,
    pair: Mapping[str, Any],
    scale: float,
) -> dict[str, Any]:
    proposal_index = int(pair["proposal_index"])
    source_pair, donor_indices, donors = _replay_donor(context, proposal_index)
    participate, initial_mask, mutations, direction = _replay_common_update(
        context, proposal_index, donors, source_pair
    )
    shared = pair["shared_replay"]
    if (
        shared["donor_indices_sha256"] != _array_sha256(donor_indices)
        or shared["direction_scores_sha256"] != _array_sha256(direction)
        or shared["participate_sha256"] != _array_sha256(participate)
        or shared["initial_copy_mask_sha256"] != _array_sha256(initial_mask)
        or shared["mutation_specs_sha256"] != _canonical_sha256(mutations)
        or float(shared["gap_l1_reference_scale"]) != scale
    ):
        raise RuntimeError("独立共同随机身份复算失败")
    arm_tables = {}
    arm_counts = {}
    mutation_specs = []
    for arm_name in protocol.ARMS:
        record = pair["arms"][arm_name]
        copy_table, full_table, copy_q, full_q = _audit_arm_record(
            context,
            donor_indices,
            donors,
            record,
            query_device=(
                getattr(protocol, "GAP_L1_DEVICE", "numpy")
                if arm_name == protocol.ARM_GAP_L1
                else "numpy"
            ),
        )
        arm_tables[arm_name] = (copy_table, full_table)
        arm_counts[arm_name] = (copy_q, full_q)
        mutation_specs.append(_mutation_specs(record["mutation_events"]))
    if not mutation_specs[0] == mutation_specs[1] == mutation_specs[2] == mutations:
        raise RuntimeError("独立三组公共突变设值复算失败")
    gap_seed = protocol.gibbs_address_seed(
        context.dataset,
        context.seed,
        context.group,
        proposal_index,
        protocol.ARM_GAP_L1,
        mode=context.mode,
    )
    gap_table, gap_mask, gap_diag = _independent_gap_replay(
        context, donors, participate, initial_mask, scale, gap_seed
    )
    recorded_gap = pair["arms"][protocol.ARM_GAP_L1]
    try:
        pd.testing.assert_frame_equal(
            gap_table,
            arm_tables[protocol.ARM_GAP_L1][0],
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError("独立逐微步最终复制表失败") from error
    diagnostics = recorded_gap["kernel_diagnostics"]
    if (
        recorded_gap["gibbs_rng"]["address_uint64"] != gap_seed
        or recorded_gap["gibbs_rng"]["initial_state_sha256"]
        != gap_diag["initial_rng_sha256"]
        or recorded_gap["gibbs_rng"]["endpoint_state_sha256"]
        != gap_diag["endpoint_rng_sha256"]
        or diagnostics["microstep_trace_sha256"] != gap_diag["trace_sha256"]
        or diagnostics["active_switches_k"] != gap_diag["active_switches_k"]
        or diagnostics["gibbs_microsteps"] != gap_diag["microsteps"]
        or diagnostics["clip_hit_count"] != gap_diag["clip_hits"]
        or diagnostics["final_query_counts"]
        != gap_diag["final_query_counts"].tolist()
    ):
        raise RuntimeError("独立逐微步坐标/E0/E1/概率/随机结果失败")
    expected_backend = getattr(protocol, "NEW_KERNEL_BACKEND", None)
    if expected_backend is not None and (
        shared.get("new_kernel_device") != expected_backend
        or diagnostics.get("backend") != expected_backend
        or gap_diag.get("backend") != "independent_torch_cuda_float64"
    ):
        raise RuntimeError("独立审计没有绑定生产/独立 CUDA 后端")
    system = _exact_system(context)
    current_units = system.units(context.q)
    copy_units = {
        arm: system.units(arm_counts[arm][0]) for arm in protocol.ARMS
    }
    full_units = {
        arm: system.units(arm_counts[arm][1]) for arm in protocol.ARMS
    }
    return {
        "pair_id": pair["pair_id"],
        "seed": context.seed,
        "state_group": context.group,
        "eligible": True,
        "current_units": current_units,
        "copy_units": copy_units,
        "full_units": full_units,
        "gain_units": {
            arm: current_units - copy_units[arm] for arm in protocol.ARMS
        },
    }


def _audit_exact_pair(
    context: _SourceContext,
    pair: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        pair["address_status"] != "already_exact_deterministic_no_op"
        or pair["shared_replay"].get("stopped_before_donor") is not True
    ):
        raise RuntimeError("独立精确 no-op 地址身份失败")
    current_hash = _frame_sha256(context.current)
    expected_squared = _independent_old_squared(
        context, context.current, context.current
    )
    expected_geometry = _independent_query_geometry(
        context, context.q, context.q
    )
    current_error = _gap_error_float(context, context.q)
    counts = {}
    for arm in protocol.ARMS:
        record = pair["arms"][arm]
        if (
            record["copy_edits"]
            or record["mutation_events"]
            or record["table_sha256"]
            != {"copy_only": current_hash, "full": current_hash}
            or record["query_delta"] != {"copy_only": [], "full": []}
            or record["old_squared_geometry"] != expected_squared
            or record["query_geometry"]
            != {"copy_only": expected_geometry, "full": expected_geometry}
            or record["gap_l1_float_reading_only"]["current"]
            != current_error
        ):
            raise RuntimeError("独立精确 no-op 表/编辑失败")
        counts[arm] = (context.q, context.q)
    system = _exact_system(context)
    units = system.units(context.q)
    return {
        "pair_id": pair["pair_id"],
        "seed": context.seed,
        "state_group": context.group,
        "eligible": False,
        "current_units": units,
        "copy_units": {arm: units for arm in protocol.ARMS},
        "full_units": {arm: units for arm in protocol.ARMS},
        "gain_units": {arm: 0 for arm in protocol.ARMS},
    }


def _scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {
            "arithmetic_audit_scientific_sha256",
            "artifact_paths_diagnostic_only",
            "environment",
            "elapsed_sec_diagnostic_only",
        }
    }


def validate_arithmetic_audit(value: Mapping[str, Any], *, mode: str) -> None:
    if (
        value.get("arithmetic_audit_format") != AUDIT_FORMAT
        or value.get("status") != "complete"
        or value.get("mode") != mode
        or value.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or value.get("audit_boundary") != AUDIT_BOUNDARY
        or value.get("audit_passed") is not True
        or value.get("arithmetic_audit_scientific_sha256")
        != _canonical_sha256(_scientific_payload(value))
        or (mode == "smoke" and (
            value.get("formal_result_valid") is not False
            or value.get("mechanism_evidence_validated") is not False
            or value.get("independent_dataset_evaluations") is not None
            or value.get("independent_final_screen_classification") is not None
        ))
    ):
        raise RuntimeError("第 6B-1 阶段独立算术审计结构/身份失败")


def audit_arithmetic(
    mode: str,
    calibration_path: str | Path,
    collection_path: str | Path,
    structural_audit_path: str | Path,
    evaluation_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None,
    confirmed_execution_commit: str | None,
) -> tuple[Path, dict[str, Any]]:
    protocol.require_run_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"独立算术审计输出已存在，不覆盖：{output}")
    git = _git_identity()
    environment = _validate_execution(
        mode, git, confirmed_execution_commit
    )
    paths = {
        "calibration": Path(calibration_path).resolve(),
        "collection": Path(collection_path).resolve(),
        "structural_audit": Path(structural_audit_path).resolve(),
        "evaluation": Path(evaluation_path).resolve(),
    }
    hashes = {name: _file_sha256(path) for name, path in paths.items()}
    calibration = _strict_load(paths["calibration"])
    collection = _strict_load(paths["collection"])
    structural = _strict_load(paths["structural_audit"])
    evaluation = _strict_load(paths["evaluation"])
    _validate_envelopes(
        mode,
        git["commit"],
        calibration,
        collection,
        structural,
        evaluation,
        hashes,
    )
    source_hashes = {}
    for name, binding in protocol.SOURCE_ARTIFACTS[mode].items():
        observed = _file_sha256(REPOSITORY_ROOT / binding["path"])
        if observed != binding["sha256"]:
            raise RuntimeError(f"独立审计第 6A 来源产物漂移：{name}")
        source_hashes[name] = observed
    state_library = _strict_load(
        REPOSITORY_ROOT
        / protocol.SOURCE_ARTIFACTS[mode]["state_library"]["path"]
    )
    source_proposals = _strict_load(
        REPOSITORY_ROOT
        / protocol.SOURCE_ARTIFACTS[mode]["proposal_collection"]["path"]
    )
    if (
        state_library.get("status") != "complete"
        or source_proposals.get("status") != "complete"
        or state_library.get("mode") != mode
        or source_proposals.get("mode") != mode
    ):
        raise RuntimeError("独立审计第 6A 来源 envelope 失败")
    source_state_index = {
        row["state_id"]: row for row in state_library["states"]
    }
    trajectory_index = {
        (row["dataset"], int(row["seed"])): row
        for row in state_library["trajectories"]
    }
    source_proposal_index = {
        row["state_id"]: row for row in source_proposals["states"]
    }
    started = time.perf_counter()
    scale_index = _audit_calibrations(
        mode,
        calibration,
        state_library,
        source_state_index,
        trajectory_index,
        source_proposal_index,
    )
    pair_index = {row["pair_id"]: row for row in collection["pairs"]}
    state_manifest_index = {
        row["state_id"]: row for row in collection["state_manifest"]
    }
    dataset_rows: dict[str, list[dict[str, Any]]] = {
        dataset: [] for dataset in protocol.DATASET_ORDER
    }
    dataset_systems: dict[str, _ExactSystem] = {}
    dataset_dimensions: dict[str, tuple[int, int]] = {}
    state_audits = []
    audited_pairs = 0
    audited_microsteps = 0
    for dataset in protocol.DATASET_ORDER:
        for seed in protocol.mode_seeds(mode):
            for group in protocol.STATE_GROUPS:
                context = _build_source_context(
                    dataset,
                    seed,
                    group,
                    mode,
                    state_library,
                    source_state_index,
                    trajectory_index,
                    source_proposal_index,
                )
                dataset_systems.setdefault(dataset, _exact_system(context))
                dataset_dimensions.setdefault(
                    dataset, (context.runtime_n, context.source_n)
                )
                state_manifest = state_manifest_index[
                    protocol.state_id(dataset, seed, group, mode=mode)
                ]
                if (
                    state_manifest["current_query_counts"] != context.q.tolist()
                    or state_manifest["current_table_sha256"]
                    != _frame_sha256(context.current)
                ):
                    raise RuntimeError("独立采集状态清单复算失败")
                exact = bool(np.all(
                    context.source_target * context.runtime_n
                    - context.q * context.source_n == 0
                ))
                state_pair_count = 0
                state_microsteps = 0
                for proposal_index in range(
                    protocol.proposals_per_state(dataset, mode=mode)
                ):
                    pair_id = protocol.pair_id(
                        dataset, seed, group, proposal_index, mode=mode
                    )
                    pair = pair_index[pair_id]
                    if exact:
                        metric = _audit_exact_pair(context, pair)
                    else:
                        scale = scale_index[(dataset, seed)]
                        if scale is None or scale <= 0.0:
                            raise RuntimeError("非精确地址缺少独立正参考尺度")
                        metric = _audit_generated_pair(context, pair, scale)
                        microsteps = int(
                            pair["arms"][protocol.ARM_GAP_L1]
                            ["kernel_diagnostics"]["gibbs_microsteps"]
                        )
                        audited_microsteps += microsteps
                        state_microsteps += microsteps
                    dataset_rows[dataset].append(metric)
                    audited_pairs += 1
                    state_pair_count += 1
                state_audits.append({
                    "state_id": protocol.state_id(
                        dataset, seed, group, mode=mode
                    ),
                    "pair_count": state_pair_count,
                    "gap_microsteps_independently_replayed": state_microsteps,
                    "all_checks_passed": True,
                })
                print(
                    f"[Stage6B1 independent arithmetic {mode} "
                    f"{protocol.state_id(dataset, seed, group, mode=mode)}] passed",
                    flush=True,
                )
    if audited_pairs != len(protocol.expected_pair_ids(mode)):
        raise RuntimeError("独立算术审计地址覆盖失败")
    computed = {}
    for dataset in protocol.DATASET_ORDER:
        runtime_n, source_n = dataset_dimensions[dataset]
        computed[dataset] = _independent_dataset_evaluation(
            dataset,
            mode,
            dataset_rows[dataset],
            dataset_systems[dataset],
            runtime_n,
            source_n,
        )
    if mode == "formal":
        if computed != evaluation["dataset_evaluations"]:
            raise RuntimeError("独立随机种子等权汇总/全部冻结门槛不一致")
        labels = {
            dataset: computed[dataset]["dataset_classification"]
            for dataset in protocol.DATASET_ORDER
        }
        final = _stage1_label(labels)
        if final != evaluation["final_screen_classification"]:
            raise RuntimeError("独立第一阶段冻结决定不一致")
        published_computed: Any = computed
        published_final: Any = final
        mechanism_validated = True
    else:
        internal_hash = _canonical_sha256(computed)
        if internal_hash != evaluation["smoke_pipeline_validation"][
            "internal_result_sha256_not_mechanism_evidence"
        ]:
            raise RuntimeError("smoke 独立评价器接线哈希不一致")
        published_computed = None
        published_final = None
        mechanism_validated = False
    if any(_file_sha256(paths[name]) != hashes[name] for name in paths):
        raise RuntimeError("独立算术审计输入在运行期间改变")
    checks = {
        "frozen_protocol_identity_verified": True,
        "six_input_artifacts_cryptographically_bound": True,
        "source_donor_participation_initial_mask_mutation_replayed": True,
        "all_calibration_scores_and_fixed_rms_recomputed": True,
        "all_gap_microsteps_coordinates_scores_probabilities_replayed": True,
        "all_sparse_tables_and_query_counts_recomputed": True,
        "all_exact_gap_gain_harm_metrics_recomputed": True,
        "all_query_geometry_and_old_squared_identities_recomputed": True,
        "all_seed_equal_weight_thresholds_recomputed": True,
        "frozen_dataset_and_stage1_decision_match": True,
        "no_gate_and_complete_matrix_verified": True,
        "overall_pass": True,
    }
    value = {
        "arithmetic_audit_format": AUDIT_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "artifact_role": (
            "formal_final_independent_arithmetic_verification"
            if mode == "formal" else "pipeline_smoke_only"
        ),
        "mechanism_evidence_validated": mechanism_validated,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": environment,
        "audit_boundary": AUDIT_BOUNDARY,
        "artifact_identity": {
            **{f"{name}_file_sha256": digest for name, digest in hashes.items()},
            **{
                f"source_stage6a_{name}_sha256": digest
                for name, digest in source_hashes.items()
            },
        },
        "checks": checks,
        "state_audits": state_audits,
        "manifest": {
            "pair_count": audited_pairs,
            "gap_microsteps_independently_replayed": audited_microsteps,
            "state_count": len(state_audits),
        },
        "independent_dataset_evaluations": published_computed,
        "independent_final_screen_classification": published_final,
        "smoke_internal_recalculation_sha256": (
            _canonical_sha256(computed) if mode == "smoke" else None
        ),
        "audit_passed": True,
        "default_kernel_changed": False,
        "artifact_paths_diagnostic_only": {
            name: str(path) for name, path in paths.items()
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    value["arithmetic_audit_scientific_sha256"] = _canonical_sha256(
        _scientific_payload(value)
    )
    validate_arithmetic_audit(value, mode=mode)
    published = _exclusive_write(output, value)
    return published, value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run = commands.add_parser("run")
    run.add_argument("--mode", choices=("smoke", "formal"), required=True)
    run.add_argument("--calibration", required=True)
    run.add_argument("--collection", required=True)
    run.add_argument("--structural-audit", required=True)
    run.add_argument("--evaluation", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--confirmed-protocol-sha256", required=True)
    run.add_argument("--confirmed-execution-commit", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "plan":
        plan = protocol.build_plan(args.mode)
        plan.update({
            "audit_kind": "independent_arithmetic_and_full_gap_microstep_replay",
            "stage6b1_collector_imported": False,
            "stage6b1_evaluator_imported": False,
            "source_read_started": False,
            "audit_started": False,
        })
        print(json.dumps(
            plan, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ))
        return
    audit_arithmetic(
        args.mode,
        args.calibration,
        args.collection,
        args.structural_audit,
        args.evaluation,
        args.output,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
