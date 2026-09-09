#!/usr/bin/env python3
"""Collect frozen, gate-free Stage 6A proposal-pair diagnostics.

The collector reads a completed Stage 6A state library, creates every
addressed independent proposal pair, and records exact measured-workload
geometry.  It never accepts, rejects, retries, rolls back, ranks, or feeds a
proposal result into another proposal or into the source trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from table_diffevo.directional_diffusion import (
    compute_copy_direction_scores,
    tilted_copy_probabilities,
)
from table_diffevo.distance import pairwise_block_distance
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.step_diagnostics import compute_row_query_deltas
from table_diffevo.update import evolve_step
from table_diffevo.vectorized_eval import evaluate_vectorized
from table_diffevo.queries import load_queries

if __package__:
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import issue53_stage6a_protocol as protocol
else:
    import build_issue53_stage6a_state_library as state_builder
    import issue53_stage6a_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROPOSAL_COLLECTION_FORMAT = "issue53_stage6a_proposal_collection_v1"
PROPOSAL_SHARD_FORMAT = "issue53_stage6a_proposal_seed_shard_v1"
PROPOSAL_COLLECTION_FILENAME = "proposal_collection.json"
PROPOSAL_SHARD_DIRECTORY = "proposal_shards"

PROBE_BOUNDARY = {
    "kernel": "independent_directional_copy",
    "post_proposal_acceptance": False,
    "proposal_rejection": False,
    "proposal_retry": False,
    "proposal_rollback": False,
    "best_shadow_or_winner_selection": False,
    "probe_feedback_to_source_trajectory": False,
    "probe_feedback_to_later_address": False,
    "all_addressed_pairs_retained": True,
    "copy_full_pair_is_measurement_not_generation_arms": True,
}

SAMPLING_PARAMS = {
    "beta": 1.0,
    "h": 0.8,
    "distance_mode": "geometric",
    "lambda_param": 0.5,
    "alpha": protocol.FIXED_ALPHA,
    "delta": 0.05,
    "winsorize_quantiles": [0.01, 0.99],
    "exclude_self": True,
    "scale_invariant": True,
    "scale_invariant_min_spread": 1e-3,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.ascontiguousarray(values)
    payload = (
        array.dtype.str.encode("utf-8")
        + repr(array.shape).encode("utf-8")
        + array.tobytes()
    )
    return _sha256_bytes(payload)


def _rng_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(
        state_builder._json_safe(rng.bit_generator.state),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _json_value(value: Any) -> Any:
    return state_builder._json_safe(value)


def _mode_seeds(mode: str) -> tuple[int, ...]:
    if mode == "formal":
        return protocol.FORMAL_SEEDS
    if mode == "smoke":
        return (protocol.SMOKE_SEED,)
    raise ValueError("mode 必须是 formal 或 smoke")


def _expected_state_ids(seeds: Sequence[int]) -> list[str]:
    return [
        protocol.state_id(dataset, seed, state_group, mode=(
            "formal" if seed in protocol.FORMAL_SEEDS else "smoke"
        ))
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for state_group in protocol.STATE_GROUPS
    ]


def _pair_id(state_id: str, proposal_index: int) -> str:
    return f"{state_id}__proposal_{proposal_index:04d}"


def _expected_pair_ids(seeds: Sequence[int], *, mode: str) -> list[str]:
    return [
        _pair_id(
            protocol.state_id(dataset, seed, group, mode=mode),
            proposal_index,
        )
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for group in protocol.STATE_GROUPS
        for proposal_index in range(
            protocol.proposals_per_state(dataset, mode=mode)
        )
    ]


def _sign(value: int) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def _scaled_full_category(b2_numerator: int, c2_numerator: int) -> str:
    if c2_numerator < 0:
        raise ValueError("C2 numerator 必须非负")
    if c2_numerator == 0 and b2_numerator != 0:
        raise ValueError("C2=0 时 B2 也必须为 0")
    g2_numerator = b2_numerator - c2_numerator
    if g2_numerator > 0:
        return "improving"
    if g2_numerator == 0:
        return "unchanged" if c2_numerator == 0 else "exact_balance"
    return (
        "direction_failure"
        if b2_numerator <= 0
        else "curvature_overrun"
    )


def _scaled_curvature_role(
    b2_numerator: int,
    cself2_numerator: int,
    ccross2_numerator: int,
) -> str:
    if _scaled_full_category(
        b2_numerator,
        cself2_numerator + ccross2_numerator,
    ) != "curvature_overrun":
        raise ValueError("cross role 只对 curvature_overrun 定义")
    without_cross = b2_numerator - cself2_numerator
    if without_cross < 0:
        return "self_sufficient_overrun"
    if without_cross == 0:
        return "cross_breaks_tie"
    return "cross_decisive_overrun"


def exact_scaled_doubled_decomposition(
    row_query_deltas: np.ndarray,
    residual_numerators: np.ndarray,
    denominator: int,
) -> dict[str, Any]:
    """Return exact B2/C2 geometry in one common rational unit.

    Formal runs reduce exactly to the protocol's integer B2/C2/G2.  Smoke
    targets are scaled from integer source counts and can be rational, so the
    same identities are represented as signed int64 numerators over the
    frozen source-record denominator.
    """

    raw_deltas = np.asarray(row_query_deltas)
    raw_residual = np.asarray(residual_numerators)
    if (
        raw_deltas.ndim != 2
        or raw_deltas.dtype.kind not in "iu"
        or raw_deltas.dtype.kind == "b"
    ):
        raise ValueError("row_query_deltas 必须是整数二维数组")
    if (
        raw_residual.shape != (raw_deltas.shape[1],)
        or raw_residual.dtype.kind not in "iu"
        or raw_residual.dtype.kind == "b"
    ):
        raise ValueError("residual_numerators 必须是匹配查询数的整数向量")
    if (
        isinstance(denominator, (bool, np.bool_))
        or not isinstance(denominator, (int, np.integer))
        or denominator <= 0
    ):
        raise ValueError("denominator 必须是正整数")

    deltas = raw_deltas.astype(np.int64, copy=False)
    residual = raw_residual.astype(np.int64, copy=False)
    denominator = int(denominator)
    row_count, query_count = deltas.shape
    max_delta = int(np.max(np.abs(deltas))) if deltas.size else 0
    max_residual = int(np.max(np.abs(residual))) if residual.size else 0
    max_delta_q = row_count * max_delta
    bounds = {
        "b2_numerator": (
            2 * query_count * max_residual * max_delta_q
        ),
        "c2_numerator": (
            denominator * query_count * max_delta_q * max_delta_q
        ),
        "cself2_numerator": (
            denominator * row_count * query_count * max_delta * max_delta
        ),
    }
    if any(value > np.iinfo(np.int64).max for value in bounds.values()):
        raise OverflowError("scaled exact B2/C2 存在 int64 溢出风险")

    delta_q = deltas.sum(axis=0, dtype=np.int64)
    b2_numerator = 2 * int(np.dot(residual, delta_q))
    c2_numerator = denominator * int(np.dot(delta_q, delta_q))
    cself2_numerator = denominator * int(
        np.einsum("ij,ij->", deltas, deltas)
    )
    ccross2_numerator = c2_numerator - cself2_numerator
    g2_numerator = b2_numerator - c2_numerator
    category = _scaled_full_category(b2_numerator, c2_numerator)
    curvature_role = (
        _scaled_curvature_role(
            b2_numerator,
            cself2_numerator,
            ccross2_numerator,
        )
        if category == "curvature_overrun"
        else None
    )
    numerators = {
        "b2": b2_numerator,
        "cself2": cself2_numerator,
        "ccross2": ccross2_numerator,
        "c2": c2_numerator,
        "g2": g2_numerator,
    }
    integral = all(
        value % denominator == 0 for value in numerators.values()
    )
    integral_residual = bool(np.all(residual % denominator == 0))
    result = {
        "denominator": denominator,
        "delta_q": delta_q.tolist(),
        **{
            f"{name}_numerator": int(value)
            for name, value in numerators.items()
        },
        "integral_doubled_units": integral,
        "integral_count_residual_units": integral_residual,
        **{
            name: (
                int(value // denominator) if integral else None
            )
            for name, value in numerators.items()
        },
        "category": category,
        "curvature_role": curvature_role,
        "cross_label": (
            "cross_required_overrun"
            if curvature_role in {
                "cross_breaks_tie",
                "cross_decisive_overrun",
            }
            else curvature_role
        ),
    }
    if integral_residual:
        integer_reference = protocol.exact_doubled_decomposition(
            deltas,
            residual // denominator,
        )
        for name in ("delta_q", "b2", "cself2", "ccross2", "c2", "g2"):
            observed = result[name]
            expected = integer_reference[name]
            if name == "delta_q":
                expected = expected.tolist()
            if observed != expected:
                raise RuntimeError("scaled/integer exact decomposition 不一致")
    return result


def _validate_scaled_sequential_identity(
    *,
    copy: Mapping[str, Any],
    mutation_given_copy: Mapping[str, Any],
    full: Mapping[str, Any],
    interaction2_numerator: int,
) -> dict[str, Any]:
    denominator = int(copy["denominator"])
    if (
        mutation_given_copy["denominator"] != denominator
        or full["denominator"] != denominator
    ):
        raise ValueError("copy/mutation/full denominator 不一致")
    expected_bfull = (
        int(copy["b2_numerator"])
        + int(mutation_given_copy["b2_numerator"])
        + int(interaction2_numerator)
    )
    expected_cfull = (
        int(copy["c2_numerator"])
        + int(mutation_given_copy["c2_numerator"])
        + int(interaction2_numerator)
    )
    if (
        int(full["b2_numerator"]) != expected_bfull
        or int(full["c2_numerator"]) != expected_cfull
        or int(full["g2_numerator"])
        != int(copy["g2_numerator"])
        + int(mutation_given_copy["g2_numerator"])
    ):
        raise RuntimeError("copy/mutation/full exact scaled identity 失败")
    integral = interaction2_numerator % denominator == 0
    if (
        copy["integral_doubled_units"]
        and mutation_given_copy["integral_doubled_units"]
        and full["integral_doubled_units"]
        and integral
    ):
        protocol.validate_sequential_identity(
            bcopy2=int(copy["b2"]),
            ccopy2=int(copy["c2"]),
            bmutation_given_copy2=int(mutation_given_copy["b2"]),
            cmutation2=int(mutation_given_copy["c2"]),
            bfull2=int(full["b2"]),
            cfull2=int(full["c2"]),
            copy_mutation_interaction2=(
                interaction2_numerator // denominator
            ),
        )
    return {
        "denominator": denominator,
        "copy_mutation_interaction2_numerator": int(
            interaction2_numerator
        ),
        "copy_mutation_interaction2": (
            int(interaction2_numerator // denominator)
            if integral
            else None
        ),
        "integral_doubled_units": integral,
        "b_identity_verified": True,
        "c_identity_verified": True,
        "g_identity_verified": True,
    }


def _changed_row_indices(
    before: pd.DataFrame,
    after: pd.DataFrame,
    attribute_names: Sequence[str],
) -> np.ndarray:
    before_values = before[list(attribute_names)].reset_index(
        drop=True
    ).to_numpy()
    after_values = after[list(attribute_names)].reset_index(
        drop=True
    ).to_numpy()
    if before_values.shape != after_values.shape:
        raise ValueError("before/after table shape 不一致")
    return np.flatnonzero(np.any(before_values != after_values, axis=1))


def _row_deltas_for_changed_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    queries: list[dict[str, Any]],
    attribute_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = _changed_row_indices(before, after, attribute_names)
    if len(rows) == 0:
        return rows, np.zeros((0, len(queries)), dtype=np.int8)
    deltas = compute_row_query_deltas(
        before.iloc[rows].reset_index(drop=True),
        after.iloc[rows].reset_index(drop=True),
        queries,
    )
    if deltas.dtype.kind not in "iu" or np.any(np.abs(deltas) > 1):
        raise RuntimeError("row query deltas 不在精确 {-1,0,1} 中")
    return rows, deltas.astype(np.int8, copy=False)


def _copy_edit_log(
    current: pd.DataFrame,
    copy_table: pd.DataFrame,
    donors: pd.DataFrame,
    donor_indices: np.ndarray,
    attribute_names: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    edits: list[dict[str, Any]] = []
    cell_count = 0
    for row_index in range(len(current)):
        cells = []
        for attribute in attribute_names:
            before = current.at[row_index, attribute]
            after = copy_table.at[row_index, attribute]
            if before == after:
                continue
            donor_value = donors.at[row_index, attribute]
            if after != donor_value:
                raise RuntimeError("copy edit 不是 donor 对应块的值")
            cells.append({
                "attribute": attribute,
                "before": _json_value(before),
                "after": _json_value(after),
            })
        if cells:
            edits.append({
                "row_index": row_index,
                "donor_index": int(donor_indices[row_index]),
                "cells": cells,
            })
            cell_count += len(cells)
    return edits, cell_count


def reconstruct_pair_tables(
    current: pd.DataFrame,
    copy_edits: Sequence[Mapping[str, Any]],
    mutation_events: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct copy/full tables from the persisted sparse logs."""

    copy_table = current.reset_index(drop=True).copy(deep=True)
    for row in copy_edits:
        row_index = int(row["row_index"])
        for cell in row["cells"]:
            attribute = str(cell["attribute"])
            if copy_table.at[row_index, attribute] != cell["before"]:
                raise RuntimeError("copy sparse log before value 不匹配")
            copy_table.at[row_index, attribute] = cell["after"]
    full_table = copy_table.copy(deep=True)
    for event in mutation_events:
        row_index = int(event["row_index"])
        attribute = str(event["attribute"])
        if full_table.at[row_index, attribute] != event["before_copy"]:
            raise RuntimeError("mutation sparse log before value 不匹配")
        full_table.at[row_index, attribute] = event["sampled_value"]
    return copy_table, full_table


def replay_paired_update(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    direction_scores: np.ndarray,
    strength: float,
    update_seed: int,
    *,
    rho: float = protocol.RHO,
    eta: float = protocol.ETA,
    full_mu: float = protocol.FULL_PROPOSAL_MU,
) -> dict[str, Any]:
    """Replay the production independent update through mutation sampling."""

    current = current.reset_index(drop=True)
    donors = donors.reset_index(drop=True)
    n_records = len(current)
    attribute_names = schema.attribute_names()
    scores = np.asarray(direction_scores, dtype=float)
    if scores.shape != (n_records, len(attribute_names)):
        raise ValueError("direction_scores shape 不匹配")
    rng = np.random.default_rng(update_seed)
    initial_rng_sha = _rng_state_sha256(rng)
    participate = rng.random(n_records) < rho
    copy_table = current.copy(deep=True)
    copy_masks = np.zeros(
        (n_records, len(attribute_names)), dtype=bool
    )
    copy_probabilities = np.empty_like(scores, dtype=float)
    for attribute_index, attribute in enumerate(attribute_names):
        current_values = current[attribute].to_numpy()
        donor_values = donors[attribute].to_numpy()
        probabilities = tilted_copy_probabilities(
            eta,
            scores[:, attribute_index],
            strength,
        )
        copy_probabilities[:, attribute_index] = probabilities
        copy_roll = rng.random(n_records) < probabilities
        mask = participate & (current_values != donor_values) & copy_roll
        copy_masks[:, attribute_index] = mask
        if np.any(mask):
            values = copy_table[attribute].to_numpy().copy()
            values[mask] = donor_values[mask]
            copy_table[attribute] = values

    mutation_rolls = rng.random(n_records)
    pre_mutation_rng_sha = _rng_state_sha256(rng)
    mutate_rows = np.flatnonzero(
        participate & (mutation_rolls < full_mu)
    )
    full_table = copy_table.copy(deep=True)
    mutation_events: list[dict[str, Any]] = []
    for row_index_raw in mutate_rows:
        row_index = int(row_index_raw)
        attribute_index = int(rng.integers(0, len(attribute_names)))
        attribute = attribute_names[attribute_index]
        block = schema.get_block(attribute)
        if block.is_numeric():
            low, high = block.range
            sampled_value: Any = int(
                rng.integers(int(low), int(high) + 1)
            )
        else:
            value_index = int(rng.integers(0, len(block.values)))
            sampled_value = block.values[value_index]
        before_copy = full_table.at[row_index, attribute]
        full_table.at[row_index, attribute] = sampled_value
        mutation_events.append({
            "row_index": row_index,
            "attribute_index": attribute_index,
            "attribute": attribute,
            "before_copy": _json_value(before_copy),
            "sampled_value": _json_value(sampled_value),
            "changed": bool(before_copy != sampled_value),
            "overwrote_copied_cell": bool(
                copy_masks[row_index, attribute_index]
            ),
        })
    return {
        "copy_table": copy_table,
        "full_table": full_table,
        "participate": participate,
        "copy_masks": copy_masks,
        "copy_probabilities": copy_probabilities,
        "mutation_rolls": mutation_rolls,
        "mutation_events": mutation_events,
        "initial_rng_sha256": initial_rng_sha,
        "pre_mutation_rng_sha256": pre_mutation_rng_sha,
        "endpoint_rng_sha256": _rng_state_sha256(rng),
    }


def _assert_frames_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    message: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as error:
        raise RuntimeError(message) from error


def _mutation_failure_source(
    copy_g2_numerator: int,
    full_g2_numerator: int,
) -> str | None:
    if full_g2_numerator >= 0:
        return None
    return (
        "mutation_created_failure"
        if copy_g2_numerator >= 0
        else "copy_already_failed"
    )


def _generate_pair(
    *,
    dataset: str,
    seed: int,
    state_group: str,
    proposal_index: int,
    mode: str,
    current: pd.DataFrame,
    queries: list[dict[str, Any]],
    schema: Any,
    count_residual_numerators: np.ndarray,
    residual_denominator: int,
    residual_signal: np.ndarray,
    sampling_probabilities: Any,
    device: str,
    direction_reference_scale: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    current = current.reset_index(drop=True)
    state_identifier = protocol.state_id(
        dataset, seed, state_group, mode=mode
    )
    donor_seed = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "donor",
        mode=mode,
    )
    donor_rng = np.random.default_rng(donor_seed)
    donor_rng_initial = _rng_state_sha256(donor_rng)
    donor_indices = sample_donors(
        sampling_probabilities, donor_rng, device=device
    )
    donor_indices = np.asarray(donor_indices, dtype=np.int64)
    donor_rng_endpoint = _rng_state_sha256(donor_rng)
    donors = current.iloc[donor_indices].reset_index(drop=True)
    direction_scores = compute_copy_direction_scores(
        current,
        donors,
        schema,
        queries,
        residual_signal,
        batch_size=256,
        device=device,
    )
    if (
        not math.isfinite(direction_reference_scale)
        or direction_reference_scale <= 0.0
    ):
        raise RuntimeError("trajectory direction reference scale 无效")
    strength = protocol.TAU / direction_reference_scale

    update_seed = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "update",
        mode=mode,
    )
    copy_rng = np.random.default_rng(update_seed)
    full_rng = np.random.default_rng(update_seed)
    copy_table, copy_diagnostics = evolve_step(
        current,
        donors,
        schema,
        rho=protocol.RHO,
        eta=protocol.ETA,
        mu=protocol.COPY_ONLY_MU,
        rng=copy_rng,
        copy_direction_scores=direction_scores,
        copy_direction_strength=strength,
        direction_logit_clip=30.0,
        return_diagnostics=True,
    )
    full_table, full_diagnostics = evolve_step(
        current,
        donors,
        schema,
        rho=protocol.RHO,
        eta=protocol.ETA,
        mu=protocol.FULL_PROPOSAL_MU,
        rng=full_rng,
        copy_direction_scores=direction_scores,
        copy_direction_strength=strength,
        direction_logit_clip=30.0,
        return_diagnostics=True,
    )
    copy_rng_endpoint = _rng_state_sha256(copy_rng)
    full_rng_endpoint = _rng_state_sha256(full_rng)
    replay = replay_paired_update(
        current,
        donors,
        schema,
        direction_scores,
        strength,
        update_seed,
    )
    _assert_frames_equal(
        copy_table,
        replay["copy_table"],
        "copy-only production/replay table 不一致",
    )
    _assert_frames_equal(
        full_table,
        replay["full_table"],
        "full production/replay table 不一致",
    )
    if (
        copy_rng_endpoint != replay["pre_mutation_rng_sha256"]
        or full_rng_endpoint != replay["endpoint_rng_sha256"]
        or copy_diagnostics["participating_rows"]
        != int(np.sum(replay["participate"]))
        or full_diagnostics["participating_rows"]
        != int(np.sum(replay["participate"]))
        or copy_diagnostics["mutated_rows"] != 0
        or full_diagnostics["mutated_rows"]
        != len(replay["mutation_events"])
    ):
        raise RuntimeError("copy/full update RNG 或公开 diagnostics 配对失败")

    attribute_names = schema.attribute_names()
    copy_edits, copied_cell_count = _copy_edit_log(
        current,
        copy_table,
        donors,
        donor_indices,
        attribute_names,
    )
    reconstructed_copy, reconstructed_full = reconstruct_pair_tables(
        current, copy_edits, replay["mutation_events"]
    )
    _assert_frames_equal(
        copy_table,
        reconstructed_copy,
        "copy sparse edit reconstruction 失败",
    )
    _assert_frames_equal(
        full_table,
        reconstructed_full,
        "full sparse edit reconstruction 失败",
    )

    copy_rows, copy_row_deltas = _row_deltas_for_changed_rows(
        current, copy_table, queries, attribute_names
    )
    mutation_rows, mutation_row_deltas = _row_deltas_for_changed_rows(
        copy_table, full_table, queries, attribute_names
    )
    full_rows, full_row_deltas = _row_deltas_for_changed_rows(
        current, full_table, queries, attribute_names
    )
    copy_exact = exact_scaled_doubled_decomposition(
        copy_row_deltas,
        count_residual_numerators,
        residual_denominator,
    )
    copy_delta = np.asarray(copy_exact["delta_q"], dtype=np.int64)
    mutation_residual_numerators = (
        np.asarray(count_residual_numerators, dtype=np.int64)
        - residual_denominator * copy_delta
    )
    mutation_exact = exact_scaled_doubled_decomposition(
        mutation_row_deltas,
        mutation_residual_numerators,
        residual_denominator,
    )
    full_exact = exact_scaled_doubled_decomposition(
        full_row_deltas,
        count_residual_numerators,
        residual_denominator,
    )
    mutation_delta = np.asarray(
        mutation_exact["delta_q"], dtype=np.int64
    )
    full_delta = np.asarray(full_exact["delta_q"], dtype=np.int64)
    if not np.array_equal(full_delta, copy_delta + mutation_delta):
        raise RuntimeError("dfull != dcopy + dmutation")
    interaction_dot = int(np.dot(copy_delta, mutation_delta))
    interaction_numerator = (
        residual_denominator * 2 * interaction_dot
    )
    sequential = _validate_scaled_sequential_identity(
        copy=copy_exact,
        mutation_given_copy=mutation_exact,
        full=full_exact,
        interaction2_numerator=interaction_numerator,
    )
    copy_g = int(copy_exact["g2_numerator"])
    full_g = int(full_exact["g2_numerator"])
    mutation_source = _mutation_failure_source(copy_g, full_g)
    if (
        mutation_source is not None
        and copy_exact["integral_doubled_units"]
        and full_exact["integral_doubled_units"]
    ):
        expected_source = protocol.classify_mutation_failure(
            int(copy_exact["g2"]), int(full_exact["g2"])
        )
        if mutation_source != expected_source:
            raise RuntimeError("mutation failure source 分类不一致")

    mutation_events = replay["mutation_events"]
    pair = {
        "pair_id": _pair_id(state_identifier, proposal_index),
        "state_id": state_identifier,
        "dataset": dataset,
        "seed": int(seed),
        "state_group": state_group,
        "proposal_index": int(proposal_index),
        "retained_unconditionally": True,
        "rng": {
            "donor_address_uint64": int(donor_seed),
            "update_address_uint64": int(update_seed),
            "donor_initial_state_sha256": donor_rng_initial,
            "donor_endpoint_state_sha256": donor_rng_endpoint,
            "update_initial_state_sha256": replay[
                "initial_rng_sha256"
            ],
            "shared_pre_mutation_state_sha256": replay[
                "pre_mutation_rng_sha256"
            ],
            "copy_only_endpoint_state_sha256": copy_rng_endpoint,
            "full_endpoint_state_sha256": full_rng_endpoint,
            "mutation_rolls_sha256": _array_sha256(
                replay["mutation_rolls"]
            ),
        },
        "donor_indices_sha256": _array_sha256(donor_indices),
        "direction_scores_sha256": _array_sha256(direction_scores),
        "copy_probabilities_sha256": _array_sha256(
            replay["copy_probabilities"]
        ),
        "direction_reference_scale": float(
            direction_reference_scale
        ),
        "effective_direction_strength": float(strength),
        "participating_row_indices": np.flatnonzero(
            replay["participate"]
        ).tolist(),
        "copy_row_indices_by_attribute": {
            attribute: np.flatnonzero(
                replay["copy_masks"][:, attribute_index]
            ).tolist()
            for attribute_index, attribute in enumerate(attribute_names)
        },
        "copy_edits": copy_edits,
        "mutation_events": mutation_events,
        "work": {
            "participating_rows": int(np.sum(replay["participate"])),
            "copied_rows": len(copy_edits),
            "copied_cells": int(copied_cell_count),
            "mutation_rows": len(mutation_events),
            "mutation_changed_cells": sum(
                bool(event["changed"]) for event in mutation_events
            ),
            "mutation_overwrote_copied_cells": sum(
                bool(event["overwrote_copied_cell"])
                for event in mutation_events
            ),
            "copy_query_changed_rows": len(copy_rows),
            "mutation_query_changed_rows": len(mutation_rows),
            "full_query_changed_rows": len(full_rows),
        },
        "table_sha256": {
            "current": state_builder.fixed_inputs._frame_sha256(current),
            "copy_only": state_builder.fixed_inputs._frame_sha256(
                copy_table
            ),
            "full": state_builder.fixed_inputs._frame_sha256(full_table),
        },
        "exact": {
            "copy_only": copy_exact,
            "mutation_given_copy": mutation_exact,
            "full": full_exact,
            "sequential": sequential,
            "copy_full_gain_sign_transition": (
                f"{_sign(copy_g)}_to_{_sign(full_g)}"
            ),
            "mutation_failure_source": mutation_source,
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    return _json_value(pair)


def _source_target_residual_numerators(
    source_target: Sequence[Any],
    runtime_n_records: int,
    source_n_records: int,
    current_query_answers: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_target)
    q = np.asarray(current_query_answers)
    if (
        source.shape != q.shape
        or source.dtype.kind not in "iu"
        or q.dtype.kind not in "iu"
    ):
        raise ValueError("source target/current q 必须是匹配的整数向量")
    bound = (
        int(np.max(np.abs(source))) * int(runtime_n_records)
        + int(np.max(np.abs(q))) * int(source_n_records)
        if source.size
        else 0
    )
    if bound > np.iinfo(np.int64).max:
        raise OverflowError("exact scaled residual 存在 int64 溢出风险")
    return (
        source.astype(np.int64) * int(runtime_n_records)
        - q.astype(np.int64) * int(source_n_records)
    )


def _collect_state(
    state: dict[str, Any],
    trajectory: dict[str, Any],
    *,
    mode: str,
    queries: list[dict[str, Any]],
    schema: Any,
    source_target: Sequence[Any],
    runtime_target: np.ndarray,
) -> dict[str, Any]:
    dataset = state["dataset"]
    seed = int(state["seed"])
    state_group = state["state_group"]
    device = str(trajectory["runtime_device"])
    current = state_builder._snapshot_frame(state["snapshot"])
    n_records = len(current)
    q, residual_signal, fitness = evaluate_vectorized(
        current,
        queries,
        schema,
        target=runtime_target,
        n_records=n_records,
        batch_size=256,
        device=device,
        want_fitness=True,
        verbose=False,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
    )
    q = np.asarray(q)
    if q.dtype.kind not in "iu":
        raise RuntimeError("current q 必须是整数 counts")
    snapshot_q = np.asarray(state["snapshot"]["current_query_answers"])
    count_residual = runtime_target - q.astype(float)
    if (
        not np.array_equal(q.astype(float), snapshot_q.astype(float))
        or not np.array_equal(
            count_residual,
            np.asarray(state["current_count_residual"], dtype=float),
        )
        or not np.array_equal(
            np.asarray(residual_signal, dtype=float),
            np.asarray(
                state["snapshot"]["current_residual_signal"],
                dtype=float,
            ),
        )
        or state_builder.fixed_inputs._frame_sha256(current)
        != state["snapshot"]["current_table_sha256"]
    ):
        raise RuntimeError(f"{state['state_id']} current diagnostics 重算失败")

    source_n_records = int(protocol.DATASETS[dataset]["n_records"])
    residual_numerators = _source_target_residual_numerators(
        source_target,
        n_records,
        source_n_records,
        q,
    )
    if mode == "formal":
        if np.any(residual_numerators % source_n_records != 0):
            raise RuntimeError("formal count residual 未化为整数")
        if not np.array_equal(
            residual_numerators // source_n_records,
            np.asarray(state["current_count_residual"], dtype=np.int64),
        ):
            raise RuntimeError("formal exact count residual 身份失败")

    use_torch = device in ("cuda", "cpu")
    distances = pairwise_block_distance(
        current,
        current,
        schema,
        device=device,
        return_tensor=use_torch,
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
        winsorize_quantiles=tuple(
            SAMPLING_PARAMS["winsorize_quantiles"]
        ),
        exclude_self=SAMPLING_PARAMS["exclude_self"],
        scale_invariant=SAMPLING_PARAMS["scale_invariant"],
        scale_invariant_min_spread=SAMPLING_PARAMS[
            "scale_invariant_min_spread"
        ],
    )
    direction_reference_scale = float(
        trajectory["direction_reference_scale"]
    )
    pairs = []
    for proposal_index in range(
        protocol.proposals_per_state(dataset, mode=mode)
    ):
        pairs.append(_generate_pair(
            dataset=dataset,
            seed=seed,
            state_group=state_group,
            proposal_index=proposal_index,
            mode=mode,
            current=current,
            queries=queries,
            schema=schema,
            count_residual_numerators=residual_numerators,
            residual_denominator=source_n_records,
            residual_signal=np.asarray(residual_signal, dtype=float),
            sampling_probabilities=probabilities,
            device=device,
            direction_reference_scale=direction_reference_scale,
        ))
    del probabilities, distances
    return {
        "state_id": state["state_id"],
        "dataset": dataset,
        "seed": seed,
        "state_group": state_group,
        "source_state_scientific_sha256": protocol.canonical_sha256(
            state_builder._state_scientific_payload(state)
        ),
        "source_trajectory_scientific_sha256": (
            protocol.canonical_sha256(
                state_builder._trajectory_scientific_payload(trajectory)
            )
        ),
        "current_table_sha256": state["snapshot"][
            "current_table_sha256"
        ],
        "current_query_answers_sha256": _array_sha256(q),
        "count_residual_numerators_sha256": _array_sha256(
            residual_numerators
        ),
        "residual_denominator": source_n_records,
        "residual_signal_sha256": _array_sha256(residual_signal),
        "fitness_sha256": _array_sha256(fitness),
        "direction_reference_scale": direction_reference_scale,
        "runtime_device": device,
        "sampling_params": dict(SAMPLING_PARAMS),
        "pairs": pairs,
    }


def _state_result_scientific_payload(
    state_result: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in state_result.items()
        if key not in {"elapsed_sec_diagnostic_only", "pairs"}
    }
    payload["pairs"] = [
        {
            key: value
            for key, value in pair.items()
            if key != "elapsed_sec_diagnostic_only"
        }
        for pair in state_result["pairs"]
    ]
    return payload


def scientific_payload(collection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "proposal_collection_format": collection[
            "proposal_collection_format"
        ],
        "mode": collection["mode"],
        "artifact_scope": collection["artifact_scope"],
        "selected_seeds": collection["selected_seeds"],
        "protocol_sha256": collection["protocol_sha256"],
        "input_audit": collection["input_audit"],
        "runtime_targets": collection["runtime_targets"],
        "probe_boundary": collection["probe_boundary"],
        "states": [
            _state_result_scientific_payload(row)
            for row in collection["states"]
        ],
    }


def _validate_exact_result(
    exact: Mapping[str, Any],
    *,
    dataset: str,
    mode: str,
) -> None:
    denominator = exact.get("denominator")
    expected_denominator = int(protocol.DATASETS[dataset]["n_records"])
    if denominator != expected_denominator:
        raise RuntimeError("exact denominator 身份失败")
    query_count = int(protocol.DATASETS[dataset]["query_count"])
    delta = exact.get("delta_q")
    if (
        not isinstance(delta, list)
        or len(delta) != query_count
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in delta
        )
    ):
        raise RuntimeError("exact delta_q 结构失败")
    names = ("b2", "cself2", "ccross2", "c2", "g2")
    try:
        numerators = {
            name: exact[f"{name}_numerator"] for name in names
        }
    except KeyError as error:
        raise RuntimeError("exact numerator 缺失") from error
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in numerators.values()
    ):
        raise RuntimeError("exact numerator 必须是整数")
    if (
        numerators["g2"] != numerators["b2"] - numerators["c2"]
        or numerators["c2"]
        != numerators["cself2"] + numerators["ccross2"]
        or numerators["c2"] < 0
        or exact.get("category")
        != _scaled_full_category(numerators["b2"], numerators["c2"])
    ):
        raise RuntimeError("exact B/C/G 或 category 身份失败")
    expected_role = (
        _scaled_curvature_role(
            numerators["b2"],
            numerators["cself2"],
            numerators["ccross2"],
        )
        if exact["category"] == "curvature_overrun"
        else None
    )
    if exact.get("curvature_role") != expected_role:
        raise RuntimeError("curvature role 身份失败")
    expected_cross_label = (
        "cross_required_overrun"
        if expected_role in {"cross_breaks_tie", "cross_decisive_overrun"}
        else expected_role
    )
    if exact.get("cross_label") != expected_cross_label:
        raise RuntimeError("cross label 身份失败")
    integral = all(
        value % denominator == 0 for value in numerators.values()
    )
    if exact.get("integral_doubled_units") is not integral:
        raise RuntimeError("integral_doubled_units 身份失败")
    if not isinstance(exact.get("integral_count_residual_units"), bool):
        raise RuntimeError("integral_count_residual_units 身份缺失")
    for name in names:
        expected_value = (
            numerators[name] // denominator if integral else None
        )
        if exact.get(name) != expected_value:
            raise RuntimeError("integer exact unit 身份失败")
    if mode == "formal" and (
        not integral or exact.get("integral_count_residual_units") is not True
    ):
        raise RuntimeError("formal exact diagnostics/count residual 必须化为整数")


def _validate_pair_structure(
    pair: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    state_group: str,
    proposal_index: int,
    mode: str,
) -> None:
    state_identifier = protocol.state_id(
        dataset, seed, state_group, mode=mode
    )
    rng = pair.get("rng", {})
    expected_donor = protocol.proposal_address_seed(
        dataset, seed, state_group, proposal_index, "donor", mode=mode
    )
    expected_update = protocol.proposal_address_seed(
        dataset, seed, state_group, proposal_index, "update", mode=mode
    )
    expected_donor_initial = _rng_state_sha256(
        np.random.default_rng(expected_donor)
    )
    expected_update_initial = _rng_state_sha256(
        np.random.default_rng(expected_update)
    )
    if (
        pair.get("pair_id") != _pair_id(state_identifier, proposal_index)
        or pair.get("state_id") != state_identifier
        or pair.get("dataset") != dataset
        or pair.get("seed") != seed
        or pair.get("state_group") != state_group
        or pair.get("proposal_index") != proposal_index
        or pair.get("retained_unconditionally") is not True
        or rng.get("donor_address_uint64") != expected_donor
        or rng.get("update_address_uint64") != expected_update
        or rng.get("donor_initial_state_sha256")
        != expected_donor_initial
        or rng.get("update_initial_state_sha256")
        != expected_update_initial
        or any(
            not _is_sha256(rng.get(key))
            for key in (
                "donor_initial_state_sha256",
                "donor_endpoint_state_sha256",
                "update_initial_state_sha256",
                "shared_pre_mutation_state_sha256",
                "copy_only_endpoint_state_sha256",
                "full_endpoint_state_sha256",
                "mutation_rolls_sha256",
            )
        )
        or rng.get("shared_pre_mutation_state_sha256")
        != rng.get("copy_only_endpoint_state_sha256")
        or any(
            not _is_sha256(pair.get(key))
            for key in (
                "donor_indices_sha256",
                "direction_scores_sha256",
                "copy_probabilities_sha256",
            )
        )
        or any(
            not _is_sha256(value)
            for value in pair.get("table_sha256", {}).values()
        )
        or set(pair.get("table_sha256", {}))
        != {"current", "copy_only", "full"}
    ):
        raise RuntimeError("proposal pair identity/RNG/hash 结构失败")
    exact = pair.get("exact")
    if not isinstance(exact, dict):
        raise RuntimeError("proposal exact diagnostics 缺失")
    for leg in ("copy_only", "mutation_given_copy", "full"):
        _validate_exact_result(exact.get(leg, {}), dataset=dataset, mode=mode)
    copy = exact["copy_only"]
    mutation = exact["mutation_given_copy"]
    full = exact["full"]
    sequential = exact.get("sequential", {})
    interaction = sequential.get(
        "copy_mutation_interaction2_numerator"
    )
    if isinstance(interaction, bool) or not isinstance(interaction, int):
        raise RuntimeError("copy/mutation interaction 缺失")
    _validate_scaled_sequential_identity(
        copy=copy,
        mutation_given_copy=mutation,
        full=full,
        interaction2_numerator=interaction,
    )
    denominator = copy["denominator"]
    interaction_integral = interaction % denominator == 0
    if (
        sequential.get("denominator") != denominator
        or sequential.get("integral_doubled_units")
        is not interaction_integral
        or sequential.get("copy_mutation_interaction2")
        != (interaction // denominator if interaction_integral else None)
    ):
        raise RuntimeError("sequential interaction unit 身份失败")
    if not all(
        sequential.get(key) is True
        for key in (
            "b_identity_verified",
            "c_identity_verified",
            "g_identity_verified",
        )
    ):
        raise RuntimeError("sequential identity flags 失败")
    expected_transition = (
        f"{_sign(copy['g2_numerator'])}_to_"
        f"{_sign(full['g2_numerator'])}"
    )
    if (
        exact.get("copy_full_gain_sign_transition")
        != expected_transition
        or exact.get("mutation_failure_source")
        != _mutation_failure_source(
            copy["g2_numerator"], full["g2_numerator"]
        )
    ):
        raise RuntimeError("mutation transition/source 身份失败")
    if (
        copy["integral_doubled_units"]
        and full["integral_doubled_units"]
        and expected_transition
        != protocol.classify_mutation_sign_transition(
            copy["g2"], full["g2"]
        )
    ):
        raise RuntimeError("mutation sign transition integer 对拍失败")
    if not isinstance(pair.get("copy_edits"), list) or not isinstance(
        pair.get("mutation_events"), list
    ):
        raise RuntimeError("proposal sparse edit logs 缺失")


def _validate_collection_structure(
    collection: Mapping[str, Any],
    *,
    mode: str,
    artifact_scope: str,
    selected_seeds: Sequence[int],
) -> None:
    expected_format = (
        PROPOSAL_COLLECTION_FORMAT
        if artifact_scope == "full"
        else PROPOSAL_SHARD_FORMAT
    )
    selected_seeds = tuple(selected_seeds)
    expected_state_ids = _expected_state_ids(selected_seeds)
    expected_pair_ids = _expected_pair_ids(selected_seeds, mode=mode)
    manifest = collection.get("manifest", {})
    states = collection.get("states", [])
    if (
        collection.get("proposal_collection_format") != expected_format
        or collection.get("status") != "complete"
        or collection.get("mode") != mode
        or collection.get("artifact_scope") != artifact_scope
        or collection.get("selected_seeds") != list(selected_seeds)
        or collection.get("formal_result_valid") is not (mode == "formal")
        or collection.get("protocol") != protocol.frozen_protocol_manifest()
        or collection.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or collection.get("probe_boundary") != PROBE_BOUNDARY
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(selected_seeds)
        or manifest.get("state_group_order") != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_state_ids)
        or manifest.get("pair_count") != len(expected_pair_ids)
        or manifest.get("state_ids_in_fixed_order") != expected_state_ids
        or manifest.get("pair_ids_in_fixed_order") != expected_pair_ids
        or [row.get("state_id") for row in states] != expected_state_ids
    ):
        raise RuntimeError("Stage 6A proposal collection 结构/覆盖失败")
    if artifact_scope == "seed_shard" and manifest.get(
        "source_proposal_seed_shard_sha256"
    ) != {}:
        raise RuntimeError("proposal seed shard 不得递归绑定 shards")
    if artifact_scope == "full" and not isinstance(
        manifest.get("source_proposal_seed_shard_sha256"), dict
    ):
        raise RuntimeError("full proposal shard manifest 无效")

    pair_ids: list[str] = []
    all_addresses: set[int] = set()
    for state_result in states:
        dataset = state_result.get("dataset")
        seed = state_result.get("seed")
        state_group = state_result.get("state_group")
        expected_device = protocol.source_generator_params(
            dataset, seed, mode=mode
        )["device"] if dataset in protocol.DATASET_ORDER else None
        direction_scale = state_result.get("direction_reference_scale")
        if (
            dataset not in protocol.DATASET_ORDER
            or seed not in selected_seeds
            or state_group not in protocol.STATE_GROUPS
            or not _is_sha256(
                state_result.get("source_state_scientific_sha256")
            )
            or not _is_sha256(
                state_result.get("source_trajectory_scientific_sha256")
            )
            or not _is_sha256(state_result.get("current_table_sha256"))
            or state_result.get("residual_denominator")
            != protocol.DATASETS[dataset]["n_records"]
            or state_result.get("sampling_params") != SAMPLING_PARAMS
            or state_result.get("runtime_device") != expected_device
            or isinstance(direction_scale, bool)
            or not isinstance(direction_scale, (int, float))
            or not math.isfinite(direction_scale)
            or direction_scale <= 0.0
        ):
            raise RuntimeError("proposal state setup identity 失败")
        pairs = state_result.get("pairs", [])
        expected_count = protocol.proposals_per_state(dataset, mode=mode)
        if len(pairs) != expected_count:
            raise RuntimeError("proposal/state count 失败")
        for proposal_index, pair in enumerate(pairs):
            _validate_pair_structure(
                pair,
                dataset=dataset,
                seed=seed,
                state_group=state_group,
                proposal_index=proposal_index,
                mode=mode,
            )
            if (
                pair["table_sha256"]["current"]
                != state_result["current_table_sha256"]
                or pair.get("direction_reference_scale")
                != direction_scale
                or pair.get("effective_direction_strength")
                != protocol.TAU / direction_scale
            ):
                raise RuntimeError(
                    "pair current hash/direction scale 与 frozen state 不一致"
                )
            pair_ids.append(pair["pair_id"])
            donor_address = pair["rng"]["donor_address_uint64"]
            update_address = pair["rng"]["update_address_uint64"]
            if (
                donor_address == update_address
                or donor_address in all_addresses
                or update_address in all_addresses
            ):
                raise RuntimeError("proposal RNG address 重复")
            all_addresses.update((donor_address, update_address))
    if pair_ids != expected_pair_ids:
        raise RuntimeError("proposal pair fixed order 失败")

    observed_sha = protocol.canonical_sha256(
        scientific_payload(collection)
    )
    if collection.get("proposal_scientific_sha256") != observed_sha:
        raise RuntimeError("proposal collection scientific SHA-256 失败")
    forbidden = {
        "accepted",
        "rejected",
        "winner",
        "selected_proposal",
        "shared_label",
        "overall_result",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            overlap = forbidden.intersection(value)
            if overlap:
                raise RuntimeError(
                    "proposal collector 出现 gate/最终结论字段："
                    f"{sorted(overlap)}"
                )
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(collection)


def _load_state_library(
    path: str | Path,
    *,
    mode: str,
) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    library = state_builder._strict_load_json(resolved)
    if library.get("mode") != mode:
        raise RuntimeError("state library mode 与 proposal mode 不一致")
    scope = library.get("artifact_scope")
    if scope not in {"full", "seed_shard"}:
        raise RuntimeError("state library artifact scope 无效")
    seeds = library.get("selected_seeds")
    if not isinstance(seeds, list):
        raise RuntimeError("state library selected seeds 无效")
    state_builder._validate_library_structure(
        library,
        mode=mode,
        artifact_scope=scope,
        selected_seeds=tuple(seeds),
    )
    return resolved, library


def _select_collection_seeds(
    mode: str,
    library: Mapping[str, Any],
    shard_index: int | None,
) -> tuple[tuple[int, ...], str, str]:
    allowed = _mode_seeds(mode)
    available = tuple(library["selected_seeds"])
    if shard_index is not None:
        if not 0 <= shard_index < len(allowed):
            raise ValueError("--shard-index 超出冻结 seed 范围")
        selected = (allowed[shard_index],)
        if selected[0] not in available:
            raise RuntimeError("state library 不含请求的 proposal seed shard")
        return selected, "seed_shard", PROPOSAL_SHARD_FORMAT
    if available == allowed:
        return allowed, "full", PROPOSAL_COLLECTION_FORMAT
    if len(available) == 1 and available[0] in allowed:
        return available, "seed_shard", PROPOSAL_SHARD_FORMAT
    raise RuntimeError("state library seed 覆盖不是冻结 full 或单 seed shard")


def build_plan(mode: str) -> dict[str, Any]:
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    seeds = _mode_seeds(mode)
    pair_count = sum(
        len(seeds)
        * len(protocol.STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    return {
        "mode": "plan_only_no_state_read_no_proposal_generation",
        "requested_mode": mode,
        "formal_result_valid": False,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "proposal_collection_format": PROPOSAL_COLLECTION_FORMAT,
        "dataset_order": list(protocol.DATASET_ORDER),
        "seed_order": list(seeds),
        "state_group_order": list(protocol.STATE_GROUPS),
        "state_count": (
            len(protocol.DATASET_ORDER)
            * len(seeds)
            * len(protocol.STATE_GROUPS)
        ),
        "proposal_pair_count": pair_count,
        "proposal_table_result_count": 2 * pair_count,
        "seed_shards": [
            {
                "shard_index": index,
                "seed": seed,
                "state_count": (
                    len(protocol.DATASET_ORDER)
                    * len(protocol.STATE_GROUPS)
                ),
                "proposal_pair_count": sum(
                    len(protocol.STATE_GROUPS)
                    * protocol.proposals_per_state(dataset, mode=mode)
                    for dataset in protocol.DATASET_ORDER
                ),
            }
            for index, seed in enumerate(seeds)
        ],
        "generation_started": False,
        "formal_confirmation_consumed": False,
    }


def collect_proposals(
    mode: str,
    state_library_path: str | Path,
    output_path: str | Path,
    *,
    shard_index: int | None = None,
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Collect a full frozen proposal artifact or one seed shard."""

    protocol.require_formal_confirmation(mode, confirmed_protocol_sha256)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"proposal 输出已存在，不覆盖：{output}")
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    initial_state_path = Path(state_library_path).resolve()
    state_file_sha256 = protocol.file_sha256(initial_state_path)
    state_path, library = _load_state_library(
        state_library_path, mode=mode
    )
    if (
        state_path != initial_state_path
        or protocol.file_sha256(state_path) != state_file_sha256
    ):
        raise RuntimeError("source state library 在读取校验期间改变")
    seeds, artifact_scope, output_format = _select_collection_seeds(
        mode, library, shard_index
    )
    git = state_builder._git_identity(REPOSITORY_ROOT)
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=True,
    )
    if mode == "formal" and (
        library.get("formal_result_valid") is not True
        or library.get("git", {}).get("commit") != git["commit"]
        or library.get("git", {}).get(
            "worktree_clean_including_untracked"
        )
        is not True
    ):
        raise RuntimeError(
            "formal proposal 要求同一 clean execution commit 的正式状态库"
        )

    input_audit, runtime_inputs, runtime_targets = (
        state_builder._audit_runtime_inputs(REPOSITORY_ROOT, mode)
    )
    input_audit = _json_value(input_audit)
    runtime_targets = _json_value(runtime_targets)
    if (
        library["input_audit"] != input_audit
        or library["runtime_targets"] != runtime_targets
    ):
        raise RuntimeError("state library 与当前 measured inputs/targets 漂移")

    state_index = {row["state_id"]: row for row in library["states"]}
    trajectory_index = {
        (row["dataset"], row["seed"]): row
        for row in library["trajectories"]
    }
    states: list[dict[str, Any]] = []
    started = time.perf_counter()
    for dataset in protocol.DATASET_ORDER:
        dataset_config = protocol.DATASETS[dataset]
        queries = load_queries(
            str(REPOSITORY_ROOT / dataset_config["queries"])
        )
        schema = load_schema(
            str(REPOSITORY_ROOT / dataset_config["schema"])
        )
        if queries != runtime_inputs[dataset]["queries"]:
            raise RuntimeError(f"{dataset} ordered queries runtime 漂移")
        runtime_target = np.asarray(
            runtime_targets[dataset]["target_values"], dtype=float
        )
        source_target = runtime_inputs[dataset]["targets"]
        for seed in seeds:
            trajectory = trajectory_index.get((dataset, seed))
            if trajectory is None:
                raise RuntimeError("state library trajectory 覆盖缺失")
            for state_group in protocol.STATE_GROUPS:
                identifier = protocol.state_id(
                    dataset, seed, state_group, mode=mode
                )
                state = state_index.get(identifier)
                if state is None:
                    raise RuntimeError("state library frozen state 覆盖缺失")
                state_started = time.perf_counter()
                result = _collect_state(
                    state,
                    trajectory,
                    mode=mode,
                    queries=queries,
                    schema=schema,
                    source_target=source_target,
                    runtime_target=runtime_target,
                )
                result["elapsed_sec_diagnostic_only"] = (
                    time.perf_counter() - state_started
                )
                states.append(_json_value(result))
                print(
                    f"[Stage6A proposals {mode} {identifier}] "
                    f"pairs={len(result['pairs'])} "
                    f"elapsed={result['elapsed_sec_diagnostic_only']:.2f}s",
                    flush=True,
                )

    expected_ids = _expected_state_ids(seeds)
    if [row["state_id"] for row in states] != expected_ids:
        raise RuntimeError("proposal state fixed order 失败")
    pair_ids = [
        pair["pair_id"] for state in states for pair in state["pairs"]
    ]
    if protocol.file_sha256(state_path) != state_file_sha256:
        raise RuntimeError("source state library 在 proposal collection 期间改变")
    if mode == "formal" and state_builder._git_identity(
        REPOSITORY_ROOT
    ) != git:
        raise RuntimeError("formal proposal collection 期间 git identity 改变")
    source_ref = {
        "path_recorded_diagnostic_only": str(state_path),
        "file_sha256": state_file_sha256,
        "state_library_scientific_sha256": library[
            "state_library_scientific_sha256"
        ],
        "git_commit": library.get("git", {}).get("commit"),
    }
    collection = {
        "proposal_collection_format": output_format,
        "status": "complete",
        "mode": mode,
        "artifact_scope": artifact_scope,
        "selected_seeds": list(seeds),
        "formal_result_valid": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": _json_value(environment),
        "input_audit": input_audit,
        "runtime_targets": runtime_targets,
        "probe_boundary": dict(PROBE_BOUNDARY),
        "source_state_artifacts_diagnostic_only": [source_ref],
        "states": states,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(states),
            "pair_count": len(pair_ids),
            "proposal_table_result_count": 2 * len(pair_ids),
            "state_ids_in_fixed_order": expected_ids,
            "pair_ids_in_fixed_order": pair_ids,
            "source_proposal_seed_shard_sha256": {},
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    collection["proposal_scientific_sha256"] = (
        protocol.canonical_sha256(scientific_payload(collection))
    )
    _validate_collection_structure(
        collection,
        mode=mode,
        artifact_scope=artifact_scope,
        selected_seeds=seeds,
    )
    published = state_builder._exclusive_write_json(output, collection)
    print(f"Stage 6A proposal collection：{published}", flush=True)
    return published, collection


def aggregate_proposal_shards(
    mode: str,
    shard_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Combine exactly one frozen proposal shard for every mode seed."""

    protocol.require_formal_confirmation(mode, confirmed_protocol_sha256)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(f"proposal 聚合输出已存在，不覆盖：{output}")
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    git = state_builder._git_identity(REPOSITORY_ROOT)
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=False,
    )
    expected_input_audit, _, expected_runtime_targets = (
        state_builder._audit_runtime_inputs(REPOSITORY_ROOT, mode)
    )
    expected_input_audit = _json_value(expected_input_audit)
    expected_runtime_targets = _json_value(expected_runtime_targets)
    seeds = _mode_seeds(mode)
    indexed: dict[int, dict[str, Any]] = {}
    indexed_paths: dict[int, Path] = {}
    indexed_file_sha256: dict[int, str] = {}
    elapsed = 0.0
    source_state_artifacts = []
    source_environments = {}
    for raw_path in shard_paths:
        path = Path(raw_path).resolve()
        initial_file_sha256 = protocol.file_sha256(path)
        shard = state_builder._strict_load_json(path)
        shard_seeds = shard.get("selected_seeds")
        if (
            not isinstance(shard_seeds, list)
            or len(shard_seeds) != 1
            or shard_seeds[0] not in seeds
        ):
            raise RuntimeError(f"proposal seed shard seed 无效：{path}")
        seed = int(shard_seeds[0])
        if seed in indexed:
            raise RuntimeError(f"proposal seed shard 重复：{seed}")
        _validate_collection_structure(
            shard,
            mode=mode,
            artifact_scope="seed_shard",
            selected_seeds=(seed,),
        )
        if protocol.file_sha256(path) != initial_file_sha256:
            raise RuntimeError("proposal seed shard 在读取校验期间改变")
        if (
            shard["input_audit"] != expected_input_audit
            or shard["runtime_targets"] != expected_runtime_targets
            or (
                mode == "formal"
                and (
                    shard.get("formal_result_valid") is not True
                    or shard.get("git", {}).get("commit") != git["commit"]
                    or shard.get("git", {}).get(
                        "worktree_clean_including_untracked"
                    )
                    is not True
                )
            )
        ):
            raise RuntimeError(f"proposal seed shard input/git 身份失败：{path}")
        indexed[seed] = shard
        indexed_paths[seed] = path
        indexed_file_sha256[seed] = initial_file_sha256
        source_environments[str(seed)] = shard["environment"]
        source_state_artifacts.extend(
            shard.get("source_state_artifacts_diagnostic_only", [])
        )
        elapsed += float(shard["elapsed_sec_diagnostic_only"])
    if set(indexed) != set(seeds):
        raise RuntimeError("proposal shards 必须恰好覆盖全部冻结 seeds")
    if any(
        protocol.file_sha256(indexed_paths[seed])
        != indexed_file_sha256[seed]
        for seed in seeds
    ):
        raise RuntimeError("proposal seed shard 在 aggregation 期间改变")
    if mode == "formal" and state_builder._git_identity(
        REPOSITORY_ROOT
    ) != git:
        raise RuntimeError("formal proposal aggregation 期间 git identity 改变")

    state_index = {
        row["state_id"]: row
        for shard in indexed.values()
        for row in shard["states"]
    }
    expected_state_ids = _expected_state_ids(seeds)
    if set(state_index) != set(expected_state_ids):
        raise RuntimeError("proposal shards state 覆盖失败")
    states = [state_index[identifier] for identifier in expected_state_ids]
    pair_ids = [
        pair["pair_id"] for state in states for pair in state["pairs"]
    ]
    collection = {
        "proposal_collection_format": PROPOSAL_COLLECTION_FORMAT,
        "status": "complete",
        "mode": mode,
        "artifact_scope": "full",
        "selected_seeds": list(seeds),
        "formal_result_valid": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": git,
        "environment": {
            "aggregation": _json_value(environment),
            "source_seed_shards": source_environments,
        },
        "input_audit": expected_input_audit,
        "runtime_targets": expected_runtime_targets,
        "probe_boundary": dict(PROBE_BOUNDARY),
        "source_state_artifacts_diagnostic_only": (
            source_state_artifacts
        ),
        "states": states,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(states),
            "pair_count": len(pair_ids),
            "proposal_table_result_count": 2 * len(pair_ids),
            "state_ids_in_fixed_order": expected_state_ids,
            "pair_ids_in_fixed_order": pair_ids,
            "source_proposal_seed_shard_sha256": {
                str(seed): indexed_file_sha256[seed]
                for seed in seeds
            },
        },
        "elapsed_sec_diagnostic_only": elapsed,
    }
    collection["proposal_scientific_sha256"] = (
        protocol.canonical_sha256(scientific_payload(collection))
    )
    _validate_collection_structure(
        collection,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    published = state_builder._exclusive_write_json(output, collection)
    print(f"Stage 6A 聚合 proposal collection：{published}", flush=True)
    return published, collection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )

    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    collect_parser.add_argument("--state-library", required=True)
    collect_parser.add_argument("--output", required=True)
    collect_parser.add_argument("--shard-index", type=int)
    collect_parser.add_argument("--confirmed-protocol-sha256")
    collect_parser.add_argument("--confirmed-execution-commit")

    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--shards", nargs="+", required=True)
    aggregate_parser.add_argument("--confirmed-protocol-sha256")
    aggregate_parser.add_argument("--confirmed-execution-commit")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(
            build_plan(args.mode),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ))
        return
    if args.command == "aggregate":
        aggregate_proposal_shards(
            args.mode,
            args.shards,
            args.output,
            confirmed_protocol_sha256=args.confirmed_protocol_sha256,
            confirmed_execution_commit=args.confirmed_execution_commit,
        )
        return
    collect_proposals(
        args.mode,
        args.state_library,
        args.output,
        shard_index=args.shard_index,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
