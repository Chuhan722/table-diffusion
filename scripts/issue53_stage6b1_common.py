"""第 6B-1 阶段采集侧共享的只读重放与原始指标工具。

独立算术审计不得导入本模块；它必须自行重写相应算术。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import compute_copy_direction_scores
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.queries import evaluate_table, load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.step_diagnostics import compute_row_query_deltas
from table_diffevo.vectorized_eval import evaluate_vectorized

if __package__:
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import collect_issue53_stage6a_proposals as stage6a_collector
    from scripts import issue53_stage6b1_protocol as protocol
else:
    import build_issue53_stage6a_state_library as state_builder
    import collect_issue53_stage6a_proposals as stage6a_collector
    import issue53_stage6b1_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SourceBundle:
    mode: str
    state_library: Mapping[str, Any]
    proposal_collection: Mapping[str, Any]
    states: Mapping[str, Mapping[str, Any]]
    trajectories: Mapping[tuple[str, int], Mapping[str, Any]]
    proposal_states: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class DatasetRuntime:
    dataset: str
    schema: Any
    queries: list[dict[str, Any]]
    runtime_target: np.ndarray
    source_target: np.ndarray
    runtime_n_records: int
    source_n_records: int


@dataclass(frozen=True)
class StateContext:
    state: Mapping[str, Any]
    trajectory: Mapping[str, Any]
    source_proposal_state: Mapping[str, Any]
    runtime: DatasetRuntime
    current: pd.DataFrame
    query_counts: np.ndarray
    residual_signal: np.ndarray
    fitness: np.ndarray
    sampling_probabilities: Any
    runtime_device: str


@dataclass(frozen=True)
class PairReplay:
    source_pair: Mapping[str, Any]
    donor_indices: np.ndarray
    donors: pd.DataFrame
    direction_scores: np.ndarray
    common_update: Mapping[str, Any]


def array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values)
    payload = (
        array.dtype.str.encode("utf-8")
        + repr(array.shape).encode("utf-8")
        + array.tobytes()
    )
    return hashlib.sha256(payload).hexdigest()


def rng_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(
        state_builder._json_safe(rng.bit_generator.state),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frame_sha256(frame: pd.DataFrame) -> str:
    return state_builder.fixed_inputs._frame_sha256(
        frame.reset_index(drop=True)
    )


def load_source_bundle(
    mode: str,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> SourceBundle:
    """读取并验证绑定的第 6A 阶段完整来源产物。"""

    root = Path(repository_root)
    protocol.assert_frozen_protocol_identity(root)
    protocol.assert_source_artifact_identities(root, mode)
    bindings = protocol.SOURCE_ARTIFACTS[mode]
    library = state_builder._strict_load_json(
        root / bindings["state_library"]["path"]
    )
    proposals = state_builder._strict_load_json(
        root / bindings["proposal_collection"]["path"]
    )
    seeds = protocol.mode_seeds(mode)
    state_builder._validate_library_structure(
        library,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    stage6a_collector._validate_collection_structure(
        proposals,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    if (
        library.get("status") != "complete"
        or proposals.get("status") != "complete"
        or library.get("runtime_targets") != proposals.get("runtime_targets")
        or library.get("input_audit") != proposals.get("input_audit")
    ):
        raise RuntimeError("第 6A 阶段状态库与候选集合身份不一致")
    states = {row["state_id"]: row for row in library["states"]}
    trajectories = {
        (row["dataset"], int(row["seed"])): row
        for row in library["trajectories"]
    }
    proposal_states = {
        row["state_id"]: row for row in proposals["states"]
    }
    expected = {
        protocol.state_id(dataset, seed, group, mode=mode)
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for group in protocol.STATE_GROUPS
    }
    if set(states) != expected or set(proposal_states) != expected:
        raise RuntimeError("第 6A 阶段来源状态覆盖不完整")
    return SourceBundle(
        mode=mode,
        state_library=library,
        proposal_collection=proposals,
        states=states,
        trajectories=trajectories,
        proposal_states=proposal_states,
    )


def load_dataset_runtime(
    bundle: SourceBundle,
    dataset: str,
    repository_root: str | Path = REPOSITORY_ROOT,
) -> DatasetRuntime:
    if dataset not in protocol.DATASETS:
        raise ValueError("未知 dataset")
    root = Path(repository_root)
    config = protocol.DATASETS[dataset]
    queries = load_queries(str(root / config["queries"]))
    schema = load_schema(str(root / config["schema"]))
    source_target = np.asarray(
        [query["result"] for query in queries], dtype=np.int64
    )
    target_record = bundle.state_library["runtime_targets"][dataset]
    runtime_target = np.asarray(
        target_record["target_values"], dtype=np.float64
    )
    runtime_n_records = int(target_record["runtime_n_records"])
    source_n_records = int(config["n_records"])
    # 严格沿用第 6A 阶段的浮点运算顺序：先形成同一个缩放比，再逐项相乘。
    expected = source_target.astype(np.float64) * (
        runtime_n_records / source_n_records
    )
    if not np.array_equal(runtime_target, expected):
        raise RuntimeError("运行目标没有精确复现来源计数缩放")
    return DatasetRuntime(
        dataset=dataset,
        schema=schema,
        queries=queries,
        runtime_target=runtime_target,
        source_target=source_target,
        runtime_n_records=runtime_n_records,
        source_n_records=source_n_records,
    )


def build_state_context(
    bundle: SourceBundle,
    runtime: DatasetRuntime,
    seed: int,
    group: str,
) -> StateContext:
    identifier = protocol.state_id(
        runtime.dataset, seed, group, mode=bundle.mode
    )
    state = bundle.states[identifier]
    trajectory = bundle.trajectories[(runtime.dataset, seed)]
    source_proposal_state = bundle.proposal_states[identifier]
    current = state_builder._snapshot_frame(state["snapshot"])
    device = str(source_proposal_state["runtime_device"])
    if device != trajectory["runtime_device"]:
        raise RuntimeError("逐状态运行后端身份不一致")
    q, residual, fitness = evaluate_vectorized(
        current,
        runtime.queries,
        runtime.schema,
        target=runtime.runtime_target,
        n_records=len(current),
        batch_size=256,
        device=device,
        want_fitness=True,
        verbose=False,
        residual_geometry="relative",
        residual_geometry_floor=protocol.GAP_L1_FLOOR,
    )
    q = np.asarray(q)
    residual = np.asarray(residual, dtype=np.float64)
    fitness = np.asarray(fitness)
    if (
        q.dtype.kind not in "iu"
        or frame_sha256(current) != state["snapshot"]["current_table_sha256"]
        or array_sha256(q) != source_proposal_state["current_query_answers_sha256"]
        or array_sha256(residual) != source_proposal_state["residual_signal_sha256"]
        or array_sha256(fitness) != source_proposal_state["fitness_sha256"]
        or not np.array_equal(
            q.astype(np.float64),
            np.asarray(state["snapshot"]["current_query_answers"], dtype=float),
        )
    ):
        raise RuntimeError(f"{identifier} 当前状态重算身份失败")

    use_tensor = device in ("cuda", "cpu")
    distances = pairwise_block_distance(
        current,
        current,
        runtime.schema,
        device=device,
        return_tensor=use_tensor,
    )
    params = stage6a_collector.SAMPLING_PARAMS
    probabilities = compute_sampling_probs(
        fitness,
        distances,
        beta=params["beta"],
        h=params["h"],
        device=device,
        distance_mode=params["distance_mode"],
        lambda_param=params["lambda_param"],
        alpha=params["alpha"],
        delta=params["delta"],
        winsorize_quantiles=tuple(params["winsorize_quantiles"]),
        exclude_self=params["exclude_self"],
        scale_invariant=params["scale_invariant"],
        scale_invariant_min_spread=params["scale_invariant_min_spread"],
    )
    return StateContext(
        state=state,
        trajectory=trajectory,
        source_proposal_state=source_proposal_state,
        runtime=runtime,
        current=current.reset_index(drop=True),
        query_counts=q.astype(np.int64, copy=False),
        residual_signal=residual,
        fitness=fitness,
        sampling_probabilities=probabilities,
        runtime_device=device,
    )


def source_pair(
    context: StateContext,
    proposal_index: int,
    *,
    mode: str,
) -> Mapping[str, Any]:
    pairs = context.source_proposal_state["pairs"]
    if not 0 <= proposal_index < len(pairs):
        raise ValueError("proposal_index 超出来源地址范围")
    pair = pairs[proposal_index]
    expected_id = protocol.pair_id(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        mode=mode,
    )
    if pair["pair_id"] != expected_id or pair["proposal_index"] != proposal_index:
        raise RuntimeError("第 6A 阶段候选地址顺序漂移")
    return pair


def replay_donors(
    context: StateContext,
    proposal_index: int,
    *,
    mode: str,
) -> tuple[Mapping[str, Any], np.ndarray, pd.DataFrame, dict[str, str]]:
    pair = source_pair(context, proposal_index, mode=mode)
    donor_seed = protocol.stage6a.proposal_address_seed(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        "donor",
        mode=mode,
    )
    if pair["rng"]["donor_address_uint64"] != donor_seed:
        raise RuntimeError("来源供体随机地址漂移")
    rng = np.random.default_rng(donor_seed)
    initial_hash = rng_state_sha256(rng)
    donor_indices = np.asarray(sample_donors(
        context.sampling_probabilities,
        rng,
        device=context.runtime_device,
    ), dtype=np.int64)
    endpoint_hash = rng_state_sha256(rng)
    if (
        array_sha256(donor_indices) != pair["donor_indices_sha256"]
        or initial_hash != pair["rng"]["donor_initial_state_sha256"]
        or endpoint_hash != pair["rng"]["donor_endpoint_state_sha256"]
    ):
        raise RuntimeError("供体重放未对拍第 6A 阶段哈希")
    donors = context.current.iloc[donor_indices].reset_index(drop=True)
    return pair, donor_indices, donors, {
        "initial_state_sha256": initial_hash,
        "endpoint_state_sha256": endpoint_hash,
    }


def replay_pair(
    context: StateContext,
    proposal_index: int,
    *,
    mode: str,
) -> PairReplay:
    pair, donor_indices, donors, _ = replay_donors(
        context, proposal_index, mode=mode
    )
    direction_scores = compute_copy_direction_scores(
        context.current,
        donors,
        context.runtime.schema,
        context.runtime.queries,
        context.residual_signal,
        batch_size=256,
        device=context.runtime_device,
    )
    direction_scores = np.asarray(direction_scores, dtype=np.float64)
    if array_sha256(direction_scores) != pair["direction_scores_sha256"]:
        raise RuntimeError("现有 B 方向分数未对拍第 6A 阶段哈希")
    reference = float(context.trajectory["direction_reference_scale"])
    if reference != float(pair["direction_reference_scale"]):
        raise RuntimeError("现有 B 固定参考尺度漂移")
    strength = protocol.B_STRENGTH / reference
    update_seed = protocol.stage6a.proposal_address_seed(
        context.runtime.dataset,
        int(context.state["seed"]),
        context.state["state_group"],
        proposal_index,
        "update",
        mode=mode,
    )
    if pair["rng"]["update_address_uint64"] != update_seed:
        raise RuntimeError("来源更新随机地址漂移")
    common = stage6a_collector.replay_paired_update(
        context.current,
        donors,
        context.runtime.schema,
        direction_scores,
        strength,
        update_seed,
        rho=protocol.RHO,
        eta=protocol.ETA,
        full_mu=protocol.MU,
    )
    attributes = context.runtime.schema.attribute_names()
    stored_masks = np.zeros_like(common["copy_masks"], dtype=bool)
    for attribute_index, attribute in enumerate(attributes):
        stored_masks[
            np.asarray(pair["copy_row_indices_by_attribute"][attribute], dtype=int),
            attribute_index,
        ] = True
    if (
        not np.array_equal(stored_masks, common["copy_masks"])
        or np.flatnonzero(common["participate"]).tolist()
        != pair["participating_row_indices"]
        or common["mutation_events"] != pair["mutation_events"]
        or array_sha256(common["copy_probabilities"])
        != pair["copy_probabilities_sha256"]
        or frame_sha256(common["copy_table"])
        != pair["table_sha256"]["copy_only"]
        or frame_sha256(common["full_table"])
        != pair["table_sha256"]["full"]
    ):
        raise RuntimeError("共同参与、初始开关或突变重放身份失败")
    return PairReplay(
        source_pair=pair,
        donor_indices=donor_indices,
        donors=donors,
        direction_scores=direction_scores,
        common_update=common,
    )


def final_mask_from_copy_table(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    copy_table: pd.DataFrame,
    attributes: Sequence[str],
) -> np.ndarray:
    current_values = current.loc[:, list(attributes)].to_numpy()
    donor_values = donors.loc[:, list(attributes)].to_numpy()
    copy_values = copy_table.loc[:, list(attributes)].to_numpy()
    differs = current_values != donor_values
    if np.any(differs & (copy_values != current_values) & (copy_values != donor_values)):
        raise RuntimeError("复制表包含既非 current 也非 donor 的值")
    if np.any(~differs & (copy_values != current_values)):
        raise RuntimeError("非活跃单元格在复制阶段改变")
    return differs & (copy_values == donor_values) & (copy_values != current_values)


def apply_common_mutations(
    copy_table: pd.DataFrame,
    mutation_specs: Sequence[Mapping[str, Any]],
    final_mask: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    full = copy_table.reset_index(drop=True).copy(deep=True)
    events = []
    for spec in mutation_specs:
        row = int(spec["row_index"])
        attribute_index = int(spec["attribute_index"])
        attribute = str(spec["attribute"])
        before = full.at[row, attribute]
        sampled = spec["sampled_value"]
        full.at[row, attribute] = sampled
        events.append({
            "row_index": row,
            "attribute_index": attribute_index,
            "attribute": attribute,
            "before_copy": state_builder._json_safe(before),
            "sampled_value": sampled,
            "changed": bool(before != sampled),
            "overwrote_copied_cell": bool(final_mask[row, attribute_index]),
        })
    return full, events


def sparse_copy_edits(
    current: pd.DataFrame,
    copy_table: pd.DataFrame,
    donors: pd.DataFrame,
    donor_indices: np.ndarray,
    attributes: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    edits = []
    cell_count = 0
    for row_index in range(len(current)):
        cells = []
        for attribute in attributes:
            before = current.at[row_index, attribute]
            after = copy_table.at[row_index, attribute]
            if before == after:
                continue
            if after != donors.at[row_index, attribute]:
                raise RuntimeError("复制编辑不是对应供体值")
            cells.append({
                "attribute": attribute,
                "before": state_builder._json_safe(before),
                "after": state_builder._json_safe(after),
            })
        if cells:
            edits.append({
                "row_index": row_index,
                "donor_index": int(donor_indices[row_index]),
                "cells": cells,
            })
            cell_count += len(cells)
    return edits, cell_count


def sparse_query_delta(before: np.ndarray, after: np.ndarray) -> list[list[int]]:
    delta = np.asarray(after, dtype=np.int64) - np.asarray(before, dtype=np.int64)
    return [
        [int(index), int(delta[index])]
        for index in np.flatnonzero(delta)
    ]


def reconstruct_query_counts(
    before: Sequence[int], sparse_delta: Sequence[Sequence[int]]
) -> np.ndarray:
    result = np.asarray(before, dtype=np.int64).copy()
    seen = set()
    for item in sparse_delta:
        if len(item) != 2:
            raise ValueError("query delta 项必须是 [index, value]")
        index, value = map(int, item)
        if index in seen or not 0 <= index < len(result) or value == 0:
            raise ValueError("query delta 索引重复、越界或包含零")
        seen.add(index)
        result[index] += value
    return result


def gap_error_float(runtime: DatasetRuntime, counts: np.ndarray) -> float:
    denominators = np.maximum(runtime.runtime_target, protocol.GAP_L1_FLOOR)
    return float(np.mean(
        np.abs(runtime.runtime_target - np.asarray(counts, dtype=float))
        / denominators
    ))


def query_geometry(
    runtime: DatasetRuntime,
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, Any]:
    before = np.asarray(before, dtype=np.int64)
    after = np.asarray(after, dtype=np.int64)
    delta = after - before
    scale = runtime.source_n_records
    target_numerators = runtime.source_target * runtime.runtime_n_records
    residual = target_numerators - before * scale
    delta_scaled = delta * scale
    denominators = np.maximum(
        target_numerators, int(protocol.GAP_L1_FLOOR) * scale
    )
    labels = {
        "no_move": 0,
        "toward_not_crossed": 0,
        "crossed_target": 0,
        "away_from_target": 0,
        "left_exact_target": 0,
    }
    crossed_weighted = 0.0
    left_exact_weighted = 0.0
    for r, d, denominator in zip(residual, delta_scaled, denominators):
        r = int(r)
        d = int(d)
        if d == 0:
            labels["no_move"] += 1
        elif r == 0:
            labels["left_exact_target"] += 1
            left_exact_weighted += abs(d) / int(denominator)
        elif r * d < 0:
            labels["away_from_target"] += 1
        elif abs(d) <= abs(r):
            labels["toward_not_crossed"] += 1
        else:
            labels["crossed_target"] += 1
            crossed_weighted += (abs(d) - abs(r)) / int(denominator)
    exact_before = int(np.sum(residual == 0))
    exact_after = int(np.sum(target_numerators - after * scale == 0))
    return {
        "labels": labels,
        "weighted_crossed_amount_float": float(crossed_weighted),
        "weighted_left_exact_amount_float": float(left_exact_weighted),
        "query_l1_displacement": int(np.sum(np.abs(delta), dtype=np.int64)),
        "query_l2_squared_displacement": int(np.dot(delta, delta)),
        "exact_query_count_before": exact_before,
        "exact_query_count_after": exact_after,
        "exact_query_count_change": exact_after - exact_before,
    }


def _changed_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    attributes: Sequence[str],
) -> np.ndarray:
    differs = (
        before.loc[:, list(attributes)].to_numpy()
        != after.loc[:, list(attributes)].to_numpy()
    )
    return np.flatnonzero(np.any(differs, axis=1)).astype(np.intp)


def exact_quadratic_diagnostics(
    context: StateContext,
    copy_table: pd.DataFrame,
    full_table: pd.DataFrame,
) -> dict[str, Any]:
    attributes = context.runtime.schema.attribute_names()
    source_n = context.runtime.source_n_records
    residual_numerators = (
        context.runtime.source_target * context.runtime.runtime_n_records
        - context.query_counts * source_n
    ).astype(np.int64)

    def row_deltas(before: pd.DataFrame, after: pd.DataFrame):
        rows = _changed_rows(before, after, attributes)
        if len(rows) == 0:
            return rows, np.zeros(
                (0, len(context.runtime.queries)), dtype=np.int8
            )
        deltas = compute_row_query_deltas(
            before.iloc[rows].reset_index(drop=True),
            after.iloc[rows].reset_index(drop=True),
            context.runtime.queries,
        )
        return rows, np.asarray(deltas, dtype=np.int8)

    copy_rows, copy_row_deltas = row_deltas(context.current, copy_table)
    mutation_rows, mutation_row_deltas = row_deltas(copy_table, full_table)
    full_rows, full_row_deltas = row_deltas(context.current, full_table)
    copy = stage6a_collector.exact_scaled_doubled_decomposition(
        copy_row_deltas, residual_numerators, source_n
    )
    copy_delta = np.asarray(copy["delta_q"], dtype=np.int64)
    mutation_residual = residual_numerators - source_n * copy_delta
    mutation = stage6a_collector.exact_scaled_doubled_decomposition(
        mutation_row_deltas, mutation_residual, source_n
    )
    full = stage6a_collector.exact_scaled_doubled_decomposition(
        full_row_deltas, residual_numerators, source_n
    )
    mutation_delta = np.asarray(mutation["delta_q"], dtype=np.int64)
    if not np.array_equal(
        np.asarray(full["delta_q"], dtype=np.int64),
        copy_delta + mutation_delta,
    ):
        raise RuntimeError("完整查询改变量不等于复制加突变")
    interaction = source_n * 2 * int(np.dot(copy_delta, mutation_delta))
    sequential = stage6a_collector._validate_scaled_sequential_identity(
        copy=copy,
        mutation_given_copy=mutation,
        full=full,
        interaction2_numerator=interaction,
    )
    return state_builder._json_safe({
        "copy_only": copy,
        "mutation_given_copy": mutation,
        "full": full,
        "sequential": sequential,
        "changed_row_counts": {
            "copy_only": len(copy_rows),
            "mutation_given_copy": len(mutation_rows),
            "full": len(full_rows),
        },
    })


def arm_raw_metrics(
    context: StateContext,
    replay: PairReplay,
    copy_table: pd.DataFrame,
    final_mask: np.ndarray,
    mutation_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    attributes = context.runtime.schema.attribute_names()
    full_table, arm_mutations = apply_common_mutations(
        copy_table, mutation_events, final_mask
    )
    copy_edits, copied_cells = sparse_copy_edits(
        context.current,
        copy_table,
        replay.donors,
        replay.donor_indices,
        attributes,
    )
    copy_counts = evaluate_table(copy_table, context.runtime.queries).astype(
        np.int64
    )
    full_counts = evaluate_table(full_table, context.runtime.queries).astype(
        np.int64
    )
    current_error = gap_error_float(context.runtime, context.query_counts)
    copy_error = gap_error_float(context.runtime, copy_counts)
    full_error = gap_error_float(context.runtime, full_counts)
    return {
        "retained_unconditionally": True,
        "copy_edits": copy_edits,
        "mutation_events": arm_mutations,
        "table_sha256": {
            "copy_only": frame_sha256(copy_table),
            "full": frame_sha256(full_table),
        },
        "query_delta": {
            "copy_only": sparse_query_delta(context.query_counts, copy_counts),
            "full": sparse_query_delta(context.query_counts, full_counts),
        },
        "gap_l1_float_reading_only": {
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
        },
        "query_geometry": {
            "copy_only": query_geometry(
                context.runtime, context.query_counts, copy_counts
            ),
            "full": query_geometry(
                context.runtime, context.query_counts, full_counts
            ),
        },
        "old_squared_geometry": exact_quadratic_diagnostics(
            context, copy_table, full_table
        ),
        "work": {
            "participating_rows": int(np.sum(replay.common_update["participate"])),
            "active_switches": int(np.sum(
                replay.common_update["participate"][:, None]
                & (
                    context.current.loc[:, attributes].to_numpy()
                    != replay.donors.loc[:, attributes].to_numpy()
                )
            )),
            "final_on_switches": int(np.sum(final_mask)),
            "changed_rows_copy_only": int(len(_changed_rows(
                context.current, copy_table, attributes
            ))),
            "changed_cells_copy_only": int(copied_cells),
            "changed_rows_full": int(len(_changed_rows(
                context.current, full_table, attributes
            ))),
            "mutation_rows": len(arm_mutations),
            "mutation_changed_cells": sum(
                bool(event["changed"]) for event in arm_mutations
            ),
        },
    }
