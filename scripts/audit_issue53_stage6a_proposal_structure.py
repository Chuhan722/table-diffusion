#!/usr/bin/env python3
"""Result-blind structural audit for Issue #53 Stage 6A artifacts.

This auditor is an evaluator prerequisite.  It independently binds the state
library to the proposal collection, replays every frozen random address,
reconstructs both proposal tables from sparse edits, and recomputes exact
query-step identities.  It emits no B/C values, category counts, dominance
shares, dataset labels, or shared mechanism result.
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
from table_diffevo.queries import load_queries
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import load_schema
from table_diffevo.step_diagnostics import compute_row_query_deltas
from table_diffevo.vectorized_eval import evaluate_vectorized

if __package__:
    from scripts import build_issue53_stage6a_state_library as state_builder
    from scripts import issue53_stage6a_protocol as protocol
else:
    import build_issue53_stage6a_state_library as state_builder
    import issue53_stage6a_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STRUCTURAL_AUDIT_FORMAT = "issue53_stage6a_structural_audit_v1"
PROPOSAL_COLLECTION_FORMAT = "issue53_stage6a_proposal_collection_v1"
PROPOSAL_SHARD_FORMAT = "issue53_stage6a_proposal_seed_shard_v1"

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

AUDIT_BOUNDARY = {
    "result_blind": True,
    "evaluator_invoked": False,
    "mechanism_statistics_computed": False,
    "bc_values_emitted": False,
    "category_counts_emitted": False,
    "seed_shares_emitted": False,
    "dataset_labels_emitted": False,
    "shared_label_emitted": False,
    "new_generation_performed": False,
    "reference_table_read": False,
    "heldout_read": False,
    "offline_safety_read": False,
    "artifact_role": "evaluator_prerequisite_only",
}

PASS_CHECKS = {
    "frozen_protocol_identity_verified": True,
    "measured_input_identity_verified": True,
    "state_library_structure_verified": True,
    "proposal_collection_envelope_verified": True,
    "source_shard_coverage_and_hashes_verified": True,
    "state_trajectory_binding_verified": True,
    "all_rng_addresses_unique_and_replayed": True,
    "all_sparse_edits_reconstructed": True,
    "all_table_hashes_recomputed": True,
    "all_exact_vectors_and_identities_recomputed": True,
    "no_gate_boundary_verified": True,
    "complete_matrix_verified": True,
    "overall_pass": True,
}

FORBIDDEN_RESULT_OR_GATE_FIELDS = {
    "accepted",
    "rejected",
    "winner",
    "selected_proposal",
    "shared_label",
    "overall_result",
}


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


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: str, *, name: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{name} 必须是完整小写 SHA-256")
    return value


def _mode_seeds(mode: str) -> tuple[int, ...]:
    if mode == "formal":
        return protocol.FORMAL_SEEDS
    if mode == "smoke":
        return (protocol.SMOKE_SEED,)
    raise ValueError("mode 必须是 formal 或 smoke")


def _expected_state_ids(
    seeds: Sequence[int], *, mode: str
) -> list[str]:
    return [
        protocol.state_id(dataset, seed, group, mode=mode)
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for group in protocol.STATE_GROUPS
    ]


def _pair_id(state_id: str, proposal_index: int) -> str:
    return f"{state_id}__proposal_{proposal_index:04d}"


def _expected_pair_ids(
    seeds: Sequence[int], *, mode: str
) -> list[str]:
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


def _walk_forbidden_fields(value: Any) -> None:
    if isinstance(value, dict):
        overlap = FORBIDDEN_RESULT_OR_GATE_FIELDS.intersection(value)
        if overlap:
            raise RuntimeError(
                "proposal artifact 出现 gate/最终结论字段："
                f"{sorted(overlap)}"
            )
        for item in value.values():
            _walk_forbidden_fields(item)
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden_fields(item)


def _proposal_state_scientific_payload(
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


def _proposal_scientific_payload(
    collection: Mapping[str, Any],
) -> dict[str, Any]:
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
            _proposal_state_scientific_payload(row)
            for row in collection["states"]
        ],
    }


def _validate_proposal_envelope(
    collection: Mapping[str, Any],
    *,
    mode: str,
    artifact_scope: str,
    selected_seeds: Sequence[int],
) -> None:
    selected_seeds = tuple(selected_seeds)
    expected_format = (
        PROPOSAL_COLLECTION_FORMAT
        if artifact_scope == "full"
        else PROPOSAL_SHARD_FORMAT
    )
    expected_state_ids = _expected_state_ids(selected_seeds, mode=mode)
    expected_pair_ids = _expected_pair_ids(selected_seeds, mode=mode)
    states = collection.get("states")
    manifest = collection.get("manifest")
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
        or not isinstance(states, list)
        or not isinstance(manifest, dict)
        or [row.get("state_id") for row in states] != expected_state_ids
        or manifest.get("dataset_order")
        != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(selected_seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_state_ids)
        or manifest.get("pair_count") != len(expected_pair_ids)
        or manifest.get("proposal_table_result_count")
        != 2 * len(expected_pair_ids)
        or manifest.get("state_ids_in_fixed_order")
        != expected_state_ids
        or manifest.get("pair_ids_in_fixed_order")
        != expected_pair_ids
    ):
        raise RuntimeError("proposal collection envelope/覆盖身份失败")
    source_shards = manifest.get("source_proposal_seed_shard_sha256")
    if artifact_scope == "seed_shard" and source_shards != {}:
        raise RuntimeError("proposal seed shard 不得递归绑定 source shards")
    if artifact_scope == "full" and not isinstance(source_shards, dict):
        raise RuntimeError("proposal full source-shard manifest 无效")

    observed_pair_ids = []
    all_addresses: set[int] = set()
    for state_result in states:
        dataset = state_result.get("dataset")
        seed = state_result.get("seed")
        group = state_result.get("state_group")
        if (
            dataset not in protocol.DATASET_ORDER
            or seed not in selected_seeds
            or group not in protocol.STATE_GROUPS
            or state_result.get("state_id")
            != protocol.state_id(dataset, seed, group, mode=mode)
            or state_result.get("sampling_params") != SAMPLING_PARAMS
            or not _is_sha256(
                state_result.get("source_state_scientific_sha256")
            )
            or not _is_sha256(
                state_result.get("source_trajectory_scientific_sha256")
            )
            or not _is_sha256(state_result.get("current_table_sha256"))
        ):
            raise RuntimeError("proposal state envelope 身份失败")
        pairs = state_result.get("pairs")
        expected_count = protocol.proposals_per_state(dataset, mode=mode)
        if not isinstance(pairs, list) or len(pairs) != expected_count:
            raise RuntimeError("proposal state pair count 失败")
        for proposal_index, pair in enumerate(pairs):
            donor_address = protocol.proposal_address_seed(
                dataset,
                seed,
                group,
                proposal_index,
                "donor",
                mode=mode,
            )
            update_address = protocol.proposal_address_seed(
                dataset,
                seed,
                group,
                proposal_index,
                "update",
                mode=mode,
            )
            rng = pair.get("rng", {})
            expected_id = _pair_id(state_result["state_id"], proposal_index)
            if (
                pair.get("pair_id") != expected_id
                or pair.get("state_id") != state_result["state_id"]
                or pair.get("dataset") != dataset
                or pair.get("seed") != seed
                or pair.get("state_group") != group
                or pair.get("proposal_index") != proposal_index
                or pair.get("retained_unconditionally") is not True
                or rng.get("donor_address_uint64") != donor_address
                or rng.get("update_address_uint64") != update_address
                or donor_address == update_address
                or donor_address in all_addresses
                or update_address in all_addresses
            ):
                raise RuntimeError("proposal pair/address envelope 身份失败")
            all_addresses.update((donor_address, update_address))
            observed_pair_ids.append(expected_id)
    if observed_pair_ids != expected_pair_ids:
        raise RuntimeError("proposal pair fixed order 失败")
    observed_scientific_sha = protocol.canonical_sha256(
        _proposal_scientific_payload(collection)
    )
    if (
        collection.get("proposal_scientific_sha256")
        != observed_scientific_sha
    ):
        raise RuntimeError("proposal collection scientific SHA-256 失败")
    _walk_forbidden_fields(collection)


def _sign(value: int) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "zero"


def _classify_full(b2_numerator: int, c2_numerator: int) -> str:
    if c2_numerator < 0:
        raise ValueError("C2 numerator 必须非负")
    if c2_numerator == 0 and b2_numerator != 0:
        raise ValueError("C2=0 时 B2 必须为 0")
    gain = b2_numerator - c2_numerator
    if gain > 0:
        return "improving"
    if gain == 0:
        return "unchanged" if c2_numerator == 0 else "exact_balance"
    return (
        "direction_failure"
        if b2_numerator <= 0
        else "curvature_overrun"
    )


def _classify_curvature_role(
    b2_numerator: int,
    cself2_numerator: int,
    ccross2_numerator: int,
) -> str:
    if _classify_full(
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


def _exact_scaled_decomposition(
    row_query_deltas: np.ndarray,
    residual_numerators: np.ndarray,
    denominator: int,
) -> dict[str, Any]:
    raw_deltas = np.asarray(row_query_deltas)
    raw_residual = np.asarray(residual_numerators)
    if (
        raw_deltas.ndim != 2
        or raw_deltas.dtype.kind not in "iu"
        or raw_deltas.dtype.kind == "b"
        or raw_residual.shape != (raw_deltas.shape[1],)
        or raw_residual.dtype.kind not in "iu"
        or raw_residual.dtype.kind == "b"
    ):
        raise ValueError("exact audit 要求整数 row deltas/residual numerators")
    if (
        isinstance(denominator, (bool, np.bool_))
        or not isinstance(denominator, (int, np.integer))
        or denominator <= 0
    ):
        raise ValueError("exact denominator 必须是正整数")
    deltas = raw_deltas.astype(np.int64, copy=False)
    residual = raw_residual.astype(np.int64, copy=False)
    denominator = int(denominator)
    row_count, query_count = deltas.shape
    max_delta = max((abs(int(value)) for value in deltas.flat), default=0)
    max_residual = max(
        (abs(int(value)) for value in residual.flat), default=0
    )
    max_delta_q = row_count * max_delta
    bounds = (
        2 * query_count * max_residual * max_delta_q,
        denominator * query_count * max_delta_q * max_delta_q,
        denominator * row_count * query_count * max_delta * max_delta,
    )
    if any(bound > np.iinfo(np.int64).max for bound in bounds):
        raise OverflowError("structural audit exact B2/C2 int64 溢出风险")

    delta_q = deltas.sum(axis=0, dtype=np.int64)
    b2_numerator = 2 * int(np.dot(residual, delta_q))
    c2_numerator = denominator * int(np.dot(delta_q, delta_q))
    cself2_numerator = denominator * int(
        np.einsum("ij,ij->", deltas, deltas)
    )
    ccross2_numerator = c2_numerator - cself2_numerator
    g2_numerator = b2_numerator - c2_numerator
    category = _classify_full(b2_numerator, c2_numerator)
    curvature_role = (
        _classify_curvature_role(
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
    return {
        "denominator": denominator,
        "delta_q": delta_q.tolist(),
        **{
            f"{name}_numerator": int(value)
            for name, value in numerators.items()
        },
        "integral_doubled_units": integral,
        "integral_count_residual_units": integral_residual,
        **{
            name: int(value // denominator) if integral else None
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


def _sequential_identity(
    copy: Mapping[str, Any],
    mutation: Mapping[str, Any],
    full: Mapping[str, Any],
    interaction_numerator: int,
) -> dict[str, Any]:
    denominator = int(copy["denominator"])
    if (
        mutation["denominator"] != denominator
        or full["denominator"] != denominator
        or full["b2_numerator"]
        != copy["b2_numerator"]
        + mutation["b2_numerator"]
        + interaction_numerator
        or full["c2_numerator"]
        != copy["c2_numerator"]
        + mutation["c2_numerator"]
        + interaction_numerator
        or full["g2_numerator"]
        != copy["g2_numerator"] + mutation["g2_numerator"]
    ):
        raise RuntimeError("structural audit sequential identity 失败")
    integral = interaction_numerator % denominator == 0
    return {
        "denominator": denominator,
        "copy_mutation_interaction2_numerator": int(
            interaction_numerator
        ),
        "copy_mutation_interaction2": (
            int(interaction_numerator // denominator)
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
        raise RuntimeError("audit before/after table shape 不一致")
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
        raise RuntimeError("audit row query deltas 不在 {-1,0,1}")
    return rows, deltas.astype(np.int8, copy=False)


def _expected_copy_edits(
    current: pd.DataFrame,
    copy_table: pd.DataFrame,
    donors: pd.DataFrame,
    donor_indices: np.ndarray,
    attribute_names: Sequence[str],
) -> tuple[list[dict[str, Any]], int]:
    edits: list[dict[str, Any]] = []
    copied_cells = 0
    for row_index in range(len(current)):
        cells = []
        for attribute in attribute_names:
            before = current.at[row_index, attribute]
            after = copy_table.at[row_index, attribute]
            if before == after:
                continue
            if after != donors.at[row_index, attribute]:
                raise RuntimeError("audit replay copy 值不是 donor 块")
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
            copied_cells += len(cells)
    return edits, copied_cells


def _apply_artifact_sparse_logs(
    current: pd.DataFrame,
    copy_edits: Any,
    mutation_events: Any,
    attribute_names: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(copy_edits, list) or not isinstance(
        mutation_events, list
    ):
        raise RuntimeError("artifact sparse edit logs 必须是列表")
    allowed_attributes = set(attribute_names)
    copy_table = current.reset_index(drop=True).copy(deep=True)
    observed_copy_rows: list[int] = []
    for edit in copy_edits:
        if not isinstance(edit, dict) or set(edit) != {
            "row_index",
            "donor_index",
            "cells",
        }:
            raise RuntimeError("copy edit row schema 失败")
        row_index = edit["row_index"]
        cells = edit["cells"]
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < len(copy_table)
            or not isinstance(edit["donor_index"], int)
            or isinstance(edit["donor_index"], bool)
            or not isinstance(cells, list)
            or not cells
        ):
            raise RuntimeError("copy edit row identity 失败")
        observed_copy_rows.append(row_index)
        observed_attributes = []
        for cell in cells:
            if not isinstance(cell, dict) or set(cell) != {
                "attribute",
                "before",
                "after",
            }:
                raise RuntimeError("copy edit cell schema 失败")
            attribute = cell["attribute"]
            if (
                attribute not in allowed_attributes
                or attribute in observed_attributes
                or copy_table.at[row_index, attribute] != cell["before"]
                or cell["before"] == cell["after"]
            ):
                raise RuntimeError("copy edit cell identity/before value 失败")
            observed_attributes.append(attribute)
            copy_table.at[row_index, attribute] = cell["after"]
    if observed_copy_rows != sorted(set(observed_copy_rows)):
        raise RuntimeError("copy edit rows 必须严格递增且唯一")

    full_table = copy_table.copy(deep=True)
    observed_mutation_rows: list[int] = []
    for event in mutation_events:
        required = {
            "row_index",
            "attribute_index",
            "attribute",
            "before_copy",
            "sampled_value",
            "changed",
            "overwrote_copied_cell",
        }
        if not isinstance(event, dict) or set(event) != required:
            raise RuntimeError("mutation event schema 失败")
        row_index = event["row_index"]
        attribute_index = event["attribute_index"]
        attribute = event["attribute"]
        if (
            isinstance(row_index, bool)
            or not isinstance(row_index, int)
            or not 0 <= row_index < len(full_table)
            or isinstance(attribute_index, bool)
            or not isinstance(attribute_index, int)
            or not 0 <= attribute_index < len(attribute_names)
            or attribute != attribute_names[attribute_index]
            or not isinstance(event["changed"], bool)
            or not isinstance(event["overwrote_copied_cell"], bool)
            or full_table.at[row_index, attribute]
            != event["before_copy"]
            or event["changed"]
            != bool(event["before_copy"] != event["sampled_value"])
        ):
            raise RuntimeError("mutation event identity/before value 失败")
        observed_mutation_rows.append(row_index)
        full_table.at[row_index, attribute] = event["sampled_value"]
    if observed_mutation_rows != sorted(set(observed_mutation_rows)):
        raise RuntimeError("mutation rows 必须严格递增且唯一")
    return copy_table, full_table


def _replay_update(
    current: pd.DataFrame,
    donors: pd.DataFrame,
    schema: Any,
    direction_scores: np.ndarray,
    strength: float,
    update_seed: int,
) -> dict[str, Any]:
    current = current.reset_index(drop=True)
    donors = donors.reset_index(drop=True)
    attribute_names = schema.attribute_names()
    n_records = len(current)
    scores = np.asarray(direction_scores, dtype=float)
    if (
        scores.shape != (n_records, len(attribute_names))
        or not np.all(np.isfinite(scores))
        or not math.isfinite(strength)
        or strength < 0.0
    ):
        raise RuntimeError("audit replay direction input 无效")

    rng = np.random.default_rng(update_seed)
    initial_rng_sha = _rng_state_sha256(rng)
    participate = rng.random(n_records) < protocol.RHO
    copy_table = current.copy(deep=True)
    copy_masks = np.zeros(
        (n_records, len(attribute_names)), dtype=bool
    )
    copy_probabilities = np.empty_like(scores, dtype=float)
    for attribute_index, attribute in enumerate(attribute_names):
        current_values = current[attribute].to_numpy()
        donor_values = donors[attribute].to_numpy()
        probabilities = tilted_copy_probabilities(
            protocol.ETA,
            scores[:, attribute_index],
            strength,
            logit_clip=30.0,
        )
        copy_probabilities[:, attribute_index] = probabilities
        copy_roll = rng.random(n_records) < probabilities
        mask = (
            participate
            & (current_values != donor_values)
            & copy_roll
        )
        copy_masks[:, attribute_index] = mask
        if np.any(mask):
            values = copy_table[attribute].to_numpy().copy()
            values[mask] = donor_values[mask]
            copy_table[attribute] = values

    mutation_rolls = rng.random(n_records)
    pre_mutation_rng_sha = _rng_state_sha256(rng)
    mutate_rows = np.flatnonzero(
        participate & (mutation_rolls < protocol.FULL_PROPOSAL_MU)
    )
    full_table = copy_table.copy(deep=True)
    mutation_events = []
    for raw_row_index in mutate_rows:
        row_index = int(raw_row_index)
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


def _assert_frame_equal(
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


def _source_residual_numerators(
    source_target: Sequence[Any],
    runtime_n_records: int,
    source_n_records: int,
    current_query_answers: np.ndarray,
) -> np.ndarray:
    target = np.asarray(source_target)
    answers = np.asarray(current_query_answers)
    if (
        target.shape != answers.shape
        or target.dtype.kind not in "iu"
        or answers.dtype.kind not in "iu"
    ):
        raise RuntimeError("source target/current answers 必须是整数匹配向量")
    bound = max(
        (
            abs(int(target_value)) * runtime_n_records
            + abs(int(answer_value)) * source_n_records
            for target_value, answer_value in zip(target, answers)
        ),
        default=0,
    )
    if bound > np.iinfo(np.int64).max:
        raise OverflowError("structural audit residual int64 溢出风险")
    return (
        target.astype(np.int64) * int(runtime_n_records)
        - answers.astype(np.int64) * int(source_n_records)
    )


def _audit_pair(
    pair: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    state_group: str,
    proposal_index: int,
    mode: str,
    current: pd.DataFrame,
    queries: list[dict[str, Any]],
    schema: Any,
    residual_numerators: np.ndarray,
    residual_denominator: int,
    residual_signal: np.ndarray,
    sampling_probabilities: Any,
    device: str,
    direction_reference_scale: float,
) -> dict[str, Any]:
    pair_identifier = _pair_id(
        protocol.state_id(dataset, seed, state_group, mode=mode),
        proposal_index,
    )
    try:
        return _audit_pair_inner(
            pair,
            dataset=dataset,
            seed=seed,
            state_group=state_group,
            proposal_index=proposal_index,
            mode=mode,
            current=current,
            queries=queries,
            schema=schema,
            residual_numerators=residual_numerators,
            residual_denominator=residual_denominator,
            residual_signal=residual_signal,
            sampling_probabilities=sampling_probabilities,
            device=device,
            direction_reference_scale=direction_reference_scale,
        )
    except Exception as error:
        raise RuntimeError(
            f"structural audit pair 失败：{pair_identifier}"
        ) from error


def _audit_pair_inner(
    pair: Mapping[str, Any],
    *,
    dataset: str,
    seed: int,
    state_group: str,
    proposal_index: int,
    mode: str,
    current: pd.DataFrame,
    queries: list[dict[str, Any]],
    schema: Any,
    residual_numerators: np.ndarray,
    residual_denominator: int,
    residual_signal: np.ndarray,
    sampling_probabilities: Any,
    device: str,
    direction_reference_scale: float,
) -> dict[str, Any]:
    required_pair_keys = {
        "pair_id",
        "state_id",
        "dataset",
        "seed",
        "state_group",
        "proposal_index",
        "retained_unconditionally",
        "rng",
        "donor_indices_sha256",
        "direction_scores_sha256",
        "copy_probabilities_sha256",
        "direction_reference_scale",
        "effective_direction_strength",
        "participating_row_indices",
        "copy_row_indices_by_attribute",
        "copy_edits",
        "mutation_events",
        "work",
        "table_sha256",
        "exact",
        "elapsed_sec_diagnostic_only",
    }
    if not isinstance(pair, Mapping) or set(pair) != required_pair_keys:
        raise RuntimeError("pair schema 不是冻结 v1 schema")
    state_identifier = protocol.state_id(
        dataset, seed, state_group, mode=mode
    )
    expected_pair_id = _pair_id(state_identifier, proposal_index)
    if (
        pair["pair_id"] != expected_pair_id
        or pair["state_id"] != state_identifier
        or pair["dataset"] != dataset
        or pair["seed"] != seed
        or pair["state_group"] != state_group
        or pair["proposal_index"] != proposal_index
        or pair["retained_unconditionally"] is not True
        or pair["direction_reference_scale"]
        != direction_reference_scale
        or pair["effective_direction_strength"]
        != protocol.TAU / direction_reference_scale
    ):
        raise RuntimeError("pair identity/direction scale 漂移")

    donor_seed = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "donor",
        mode=mode,
    )
    update_seed = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "update",
        mode=mode,
    )
    donor_rng = np.random.default_rng(donor_seed)
    donor_initial_sha = _rng_state_sha256(donor_rng)
    donor_indices = np.asarray(
        sample_donors(
            sampling_probabilities, donor_rng, device=device
        ),
        dtype=np.int64,
    )
    donor_endpoint_sha = _rng_state_sha256(donor_rng)
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
    strength = protocol.TAU / direction_reference_scale
    replay = _replay_update(
        current,
        donors,
        schema,
        direction_scores,
        strength,
        update_seed,
    )
    rng = pair["rng"]
    required_rng_keys = {
        "donor_address_uint64",
        "update_address_uint64",
        "donor_initial_state_sha256",
        "donor_endpoint_state_sha256",
        "update_initial_state_sha256",
        "shared_pre_mutation_state_sha256",
        "copy_only_endpoint_state_sha256",
        "full_endpoint_state_sha256",
        "mutation_rolls_sha256",
    }
    if (
        not isinstance(rng, Mapping)
        or set(rng) != required_rng_keys
        or rng["donor_address_uint64"] != donor_seed
        or rng["update_address_uint64"] != update_seed
        or rng["donor_initial_state_sha256"] != donor_initial_sha
        or rng["donor_endpoint_state_sha256"] != donor_endpoint_sha
        or rng["update_initial_state_sha256"]
        != replay["initial_rng_sha256"]
        or rng["shared_pre_mutation_state_sha256"]
        != replay["pre_mutation_rng_sha256"]
        or rng["copy_only_endpoint_state_sha256"]
        != replay["pre_mutation_rng_sha256"]
        or rng["full_endpoint_state_sha256"]
        != replay["endpoint_rng_sha256"]
        or rng["mutation_rolls_sha256"]
        != _array_sha256(replay["mutation_rolls"])
        or pair["donor_indices_sha256"]
        != _array_sha256(donor_indices)
        or pair["direction_scores_sha256"]
        != _array_sha256(direction_scores)
        or pair["copy_probabilities_sha256"]
        != _array_sha256(replay["copy_probabilities"])
    ):
        raise RuntimeError("pair donor/update RNG replay 身份失败")

    attribute_names = schema.attribute_names()
    expected_participating = np.flatnonzero(
        replay["participate"]
    ).tolist()
    expected_copy_rows = {
        attribute: np.flatnonzero(
            replay["copy_masks"][:, attribute_index]
        ).tolist()
        for attribute_index, attribute in enumerate(attribute_names)
    }
    expected_copy_edits, copied_cells = _expected_copy_edits(
        current,
        replay["copy_table"],
        donors,
        donor_indices,
        attribute_names,
    )
    expected_mutation_events = replay["mutation_events"]
    if (
        pair["participating_row_indices"] != expected_participating
        or pair["copy_row_indices_by_attribute"] != expected_copy_rows
        or pair["copy_edits"] != expected_copy_edits
        or pair["mutation_events"] != expected_mutation_events
    ):
        raise RuntimeError("pair participation/copy/mutation identity 失败")

    artifact_copy, artifact_full = _apply_artifact_sparse_logs(
        current,
        pair["copy_edits"],
        pair["mutation_events"],
        attribute_names,
    )
    _assert_frame_equal(
        artifact_copy,
        replay["copy_table"],
        "artifact copy sparse reconstruction 与 RNG replay 不一致",
    )
    _assert_frame_equal(
        artifact_full,
        replay["full_table"],
        "artifact full sparse reconstruction 与 RNG replay 不一致",
    )

    copy_rows, copy_row_deltas = _row_deltas_for_changed_rows(
        current, artifact_copy, queries, attribute_names
    )
    mutation_rows, mutation_row_deltas = _row_deltas_for_changed_rows(
        artifact_copy, artifact_full, queries, attribute_names
    )
    full_rows, full_row_deltas = _row_deltas_for_changed_rows(
        current, artifact_full, queries, attribute_names
    )
    copy_exact = _exact_scaled_decomposition(
        copy_row_deltas,
        residual_numerators,
        residual_denominator,
    )
    copy_delta = np.asarray(copy_exact["delta_q"], dtype=np.int64)
    mutation_residual = (
        np.asarray(residual_numerators, dtype=np.int64)
        - residual_denominator * copy_delta
    )
    mutation_exact = _exact_scaled_decomposition(
        mutation_row_deltas,
        mutation_residual,
        residual_denominator,
    )
    full_exact = _exact_scaled_decomposition(
        full_row_deltas,
        residual_numerators,
        residual_denominator,
    )
    mutation_delta = np.asarray(
        mutation_exact["delta_q"], dtype=np.int64
    )
    full_delta = np.asarray(full_exact["delta_q"], dtype=np.int64)
    if not np.array_equal(full_delta, copy_delta + mutation_delta):
        raise RuntimeError("artifact dfull != dcopy + dmutation")
    max_copy_delta = max(
        (abs(int(value)) for value in copy_delta), default=0
    )
    max_mutation_delta = max(
        (abs(int(value)) for value in mutation_delta), default=0
    )
    interaction_bound = (
        residual_denominator
        * 2
        * len(copy_delta)
        * max_copy_delta
        * max_mutation_delta
    )
    if interaction_bound > np.iinfo(np.int64).max:
        raise OverflowError("copy/mutation interaction int64 溢出风险")
    interaction = (
        residual_denominator
        * 2
        * int(np.dot(copy_delta, mutation_delta))
    )
    sequential = _sequential_identity(
        copy_exact,
        mutation_exact,
        full_exact,
        interaction,
    )
    copy_gain = int(copy_exact["g2_numerator"])
    full_gain = int(full_exact["g2_numerator"])
    mutation_source = (
        None
        if full_gain >= 0
        else (
            "mutation_created_failure"
            if copy_gain >= 0
            else "copy_already_failed"
        )
    )
    expected_exact = {
        "copy_only": copy_exact,
        "mutation_given_copy": mutation_exact,
        "full": full_exact,
        "sequential": sequential,
        "copy_full_gain_sign_transition": (
            f"{_sign(copy_gain)}_to_{_sign(full_gain)}"
        ),
        "mutation_failure_source": mutation_source,
    }
    if pair["exact"] != expected_exact:
        raise RuntimeError("pair exact vectors/B/C/categories 身份失败")
    if mode == "formal" and not all(
        exact["integral_doubled_units"]
        and exact["integral_count_residual_units"]
        for exact in (copy_exact, mutation_exact, full_exact)
    ):
        raise RuntimeError("formal pair exact units 未约成整数")

    expected_work = {
        "participating_rows": len(expected_participating),
        "copied_rows": len(expected_copy_edits),
        "copied_cells": copied_cells,
        "mutation_rows": len(expected_mutation_events),
        "mutation_changed_cells": sum(
            event["changed"] for event in expected_mutation_events
        ),
        "mutation_overwrote_copied_cells": sum(
            event["overwrote_copied_cell"]
            for event in expected_mutation_events
        ),
        "copy_query_changed_rows": len(copy_rows),
        "mutation_query_changed_rows": len(mutation_rows),
        "full_query_changed_rows": len(full_rows),
    }
    expected_table_sha = {
        "current": state_builder.fixed_inputs._frame_sha256(current),
        "copy_only": state_builder.fixed_inputs._frame_sha256(
            artifact_copy
        ),
        "full": state_builder.fixed_inputs._frame_sha256(artifact_full),
    }
    if (
        pair["work"] != expected_work
        or pair["table_sha256"] != expected_table_sha
    ):
        raise RuntimeError("pair work/table hashes 身份失败")

    audit_identity = {
        "pair_id": expected_pair_id,
        "rng_identity_sha256": protocol.canonical_sha256(rng),
        "donor_indices_sha256": pair["donor_indices_sha256"],
        "direction_scores_sha256": pair["direction_scores_sha256"],
        "copy_probabilities_sha256": pair[
            "copy_probabilities_sha256"
        ],
        "sparse_edits_sha256": protocol.canonical_sha256({
            "copy_edits": pair["copy_edits"],
            "mutation_events": pair["mutation_events"],
        }),
        "table_sha256": expected_table_sha,
        "exact_identity_sha256": protocol.canonical_sha256(
            expected_exact
        ),
    }
    return {
        "pair_id": expected_pair_id,
        "pair_structural_audit_sha256": protocol.canonical_sha256(
            audit_identity
        ),
    }


def _audit_state(
    state: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    proposal_state: Mapping[str, Any],
    *,
    mode: str,
    queries: list[dict[str, Any]],
    schema: Any,
    source_target: Sequence[Any],
    runtime_target: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = state["dataset"]
    seed = int(state["seed"])
    state_group = state["state_group"]
    state_identifier = protocol.state_id(
        dataset, seed, state_group, mode=mode
    )
    required_state_keys = {
        "state_id",
        "dataset",
        "seed",
        "state_group",
        "source_state_scientific_sha256",
        "source_trajectory_scientific_sha256",
        "current_table_sha256",
        "current_query_answers_sha256",
        "count_residual_numerators_sha256",
        "residual_denominator",
        "residual_signal_sha256",
        "fitness_sha256",
        "direction_reference_scale",
        "runtime_device",
        "sampling_params",
        "pairs",
    }
    observed_state_keys = set(proposal_state)
    observed_state_keys.discard("elapsed_sec_diagnostic_only")
    if observed_state_keys != required_state_keys:
        raise RuntimeError(f"{state_identifier} proposal-state schema 漂移")
    if (
        proposal_state["state_id"] != state_identifier
        or proposal_state["dataset"] != dataset
        or proposal_state["seed"] != seed
        or proposal_state["state_group"] != state_group
        or proposal_state["source_state_scientific_sha256"]
        != protocol.canonical_sha256(
            state_builder._state_scientific_payload(dict(state))
        )
        or proposal_state["source_trajectory_scientific_sha256"]
        != protocol.canonical_sha256(
            state_builder._trajectory_scientific_payload(
                dict(trajectory)
            )
        )
        or proposal_state["sampling_params"] != SAMPLING_PARAMS
    ):
        raise RuntimeError(f"{state_identifier} state/trajectory 绑定失败")

    current = state_builder._snapshot_frame(dict(state["snapshot"]))
    n_records = len(current)
    device = str(trajectory["runtime_device"])
    direction_scale = float(trajectory["direction_reference_scale"])
    if (
        proposal_state["runtime_device"] != device
        or proposal_state["direction_reference_scale"] != direction_scale
        or not math.isfinite(direction_scale)
        or direction_scale <= 0.0
    ):
        raise RuntimeError(f"{state_identifier} runtime direction 身份失败")

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
    residual_signal = np.asarray(residual_signal, dtype=float)
    fitness = np.asarray(fitness, dtype=float)
    snapshot_q = np.asarray(state["snapshot"]["current_query_answers"])
    count_residual = runtime_target - q.astype(float)
    if (
        q.dtype.kind not in "iu"
        or not np.array_equal(q.astype(float), snapshot_q.astype(float))
        or not np.array_equal(
            count_residual,
            np.asarray(state["current_count_residual"], dtype=float),
        )
        or not np.array_equal(
            residual_signal,
            np.asarray(
                state["snapshot"]["current_residual_signal"],
                dtype=float,
            ),
        )
        or not np.all(np.isfinite(fitness))
    ):
        raise RuntimeError(f"{state_identifier} current query diagnostics 失败")
    source_n_records = int(protocol.DATASETS[dataset]["n_records"])
    residual_numerators = _source_residual_numerators(
        source_target,
        n_records,
        source_n_records,
        q,
    )
    if mode == "formal" and (
        np.any(residual_numerators % source_n_records != 0)
        or not np.array_equal(
            residual_numerators // source_n_records,
            np.asarray(state["current_count_residual"], dtype=np.int64),
        )
    ):
        raise RuntimeError(f"{state_identifier} formal count residual 失败")
    current_table_sha = state_builder.fixed_inputs._frame_sha256(current)
    if (
        proposal_state["current_table_sha256"] != current_table_sha
        or current_table_sha
        != state["snapshot"]["current_table_sha256"]
        or proposal_state["current_query_answers_sha256"]
        != _array_sha256(q)
        or proposal_state["count_residual_numerators_sha256"]
        != _array_sha256(residual_numerators)
        or proposal_state["residual_denominator"] != source_n_records
        or proposal_state["residual_signal_sha256"]
        != _array_sha256(residual_signal)
        or proposal_state["fitness_sha256"] != _array_sha256(fitness)
    ):
        raise RuntimeError(f"{state_identifier} current setup hashes 失败")

    use_torch = device in {"cuda", "cpu"}
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
    pairs = proposal_state["pairs"]
    expected_pair_count = protocol.proposals_per_state(
        dataset, mode=mode
    )
    if not isinstance(pairs, list) or len(pairs) != expected_pair_count:
        raise RuntimeError(f"{state_identifier} proposal count 失败")
    pair_audits = [
        _audit_pair(
            pair,
            dataset=dataset,
            seed=seed,
            state_group=state_group,
            proposal_index=proposal_index,
            mode=mode,
            current=current,
            queries=queries,
            schema=schema,
            residual_numerators=residual_numerators,
            residual_denominator=source_n_records,
            residual_signal=residual_signal,
            sampling_probabilities=probabilities,
            device=device,
            direction_reference_scale=direction_scale,
        )
        for proposal_index, pair in enumerate(pairs)
    ]
    del probabilities, distances
    pair_ids = [row["pair_id"] for row in pair_audits]
    return {
        "state_id": state_identifier,
        "source_state_scientific_sha256": proposal_state[
            "source_state_scientific_sha256"
        ],
        "source_trajectory_scientific_sha256": proposal_state[
            "source_trajectory_scientific_sha256"
        ],
        "current_table_sha256": current_table_sha,
        "pair_count": len(pair_audits),
        "pair_ids_sha256": protocol.canonical_sha256(pair_ids),
        "pair_structural_audits_sha256": protocol.canonical_sha256(
            pair_audits
        ),
        "all_pairs_structurally_valid": True,
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }


def _load_stable_json(
    path: str | Path,
    *,
    confirmed_sha256: str | None = None,
    confirmation_name: str = "artifact",
) -> tuple[Path, str, dict[str, Any]]:
    resolved = Path(path).resolve()
    initial_sha = protocol.file_sha256(resolved)
    if confirmed_sha256 is not None and initial_sha != confirmed_sha256:
        raise ValueError(
            f"{confirmation_name} SHA 与显式确认不一致："
            f"confirmed={confirmed_sha256}, observed={initial_sha}"
        )
    payload = state_builder._strict_load_json(resolved)
    if protocol.file_sha256(resolved) != initial_sha:
        raise RuntimeError(f"{confirmation_name} 在读取校验期间改变")
    return resolved, initial_sha, payload


def _audit_state_shards(
    full_library: Mapping[str, Any],
    shard_paths: Sequence[str | Path],
    *,
    mode: str,
    selected_seeds: Sequence[int],
) -> dict[str, str]:
    expected_hashes = full_library["manifest"][
        "source_seed_shard_sha256"
    ]
    if not expected_hashes:
        if shard_paths:
            raise RuntimeError("direct full state library 不接受额外 state shards")
        return {}
    expected_keys = {str(seed) for seed in selected_seeds}
    if set(expected_hashes) != expected_keys:
        raise RuntimeError("state full source-shard manifest 覆盖失败")
    indexed = {
        row["state_id"]: row for row in full_library["states"]
    }
    trajectory_index = {
        (row["dataset"], row["seed"]): row
        for row in full_library["trajectories"]
    }
    observed: dict[str, str] = {}
    for raw_path in shard_paths:
        path, file_sha, shard = _load_stable_json(
            raw_path, confirmation_name="state seed shard"
        )
        del path
        shard_seeds = shard.get("selected_seeds")
        if (
            not isinstance(shard_seeds, list)
            or len(shard_seeds) != 1
            or shard_seeds[0] not in selected_seeds
        ):
            raise RuntimeError("state seed shard seed 身份失败")
        seed = int(shard_seeds[0])
        key = str(seed)
        if key in observed or file_sha != expected_hashes.get(key):
            raise RuntimeError("state seed shard 重复或 file SHA 失败")
        state_builder._validate_library_structure(
            shard,
            mode=mode,
            artifact_scope="seed_shard",
            selected_seeds=(seed,),
        )
        expected_state_ids = _expected_state_ids((seed,), mode=mode)
        if any(
            shard_state != indexed[identifier]
            for identifier, shard_state in zip(
                expected_state_ids, shard["states"]
            )
        ):
            raise RuntimeError("state shard/full selected states 不一致")
        expected_trajectory_keys = [
            (dataset, seed) for dataset in protocol.DATASET_ORDER
        ]
        if any(
            shard_trajectory != trajectory_index[key_tuple]
            for key_tuple, shard_trajectory in zip(
                expected_trajectory_keys, shard["trajectories"]
            )
        ):
            raise RuntimeError("state shard/full trajectories 不一致")
        observed[key] = file_sha
    if set(observed) != expected_keys:
        raise RuntimeError("state seed shard paths 未完整覆盖 manifest")
    return {str(seed): observed[str(seed)] for seed in selected_seeds}


def _audit_proposal_shards(
    full_collection: Mapping[str, Any],
    shard_paths: Sequence[str | Path],
    *,
    mode: str,
    selected_seeds: Sequence[int],
) -> dict[str, str]:
    expected_hashes = full_collection["manifest"][
        "source_proposal_seed_shard_sha256"
    ]
    if not expected_hashes:
        if shard_paths:
            raise RuntimeError(
                "direct full proposal collection 不接受额外 proposal shards"
            )
        return {}
    expected_keys = {str(seed) for seed in selected_seeds}
    if set(expected_hashes) != expected_keys:
        raise RuntimeError("proposal full source-shard manifest 覆盖失败")
    full_state_index = {
        row["state_id"]: row for row in full_collection["states"]
    }
    observed: dict[str, str] = {}
    for raw_path in shard_paths:
        path, file_sha, shard = _load_stable_json(
            raw_path, confirmation_name="proposal seed shard"
        )
        del path
        shard_seeds = shard.get("selected_seeds")
        if (
            not isinstance(shard_seeds, list)
            or len(shard_seeds) != 1
            or shard_seeds[0] not in selected_seeds
        ):
            raise RuntimeError("proposal seed shard seed 身份失败")
        seed = int(shard_seeds[0])
        key = str(seed)
        if key in observed or file_sha != expected_hashes.get(key):
            raise RuntimeError("proposal seed shard 重复或 file SHA 失败")
        _validate_proposal_envelope(
            shard,
            mode=mode,
            artifact_scope="seed_shard",
            selected_seeds=(seed,),
        )
        expected_state_ids = _expected_state_ids((seed,), mode=mode)
        if any(
            shard_state != full_state_index[identifier]
            for identifier, shard_state in zip(
                expected_state_ids, shard["states"]
            )
        ):
            raise RuntimeError("proposal shard/full state results 不一致")
        observed[key] = file_sha
    if set(observed) != expected_keys:
        raise RuntimeError("proposal seed shard paths 未完整覆盖 manifest")
    return {str(seed): observed[str(seed)] for seed in selected_seeds}


def _load_full_artifacts(
    mode: str,
    state_library_path: str | Path,
    proposal_collection_path: str | Path,
    *,
    confirmed_state_library_sha256: str,
    confirmed_proposal_collection_sha256: str,
) -> tuple[
    Path,
    str,
    dict[str, Any],
    Path,
    str,
    dict[str, Any],
]:
    state_path, state_file_sha, state_library = _load_stable_json(
        state_library_path,
        confirmed_sha256=confirmed_state_library_sha256,
        confirmation_name="state library",
    )
    proposal_path, proposal_file_sha, proposal_collection = (
        _load_stable_json(
            proposal_collection_path,
            confirmed_sha256=confirmed_proposal_collection_sha256,
            confirmation_name="proposal collection",
        )
    )
    seeds = _mode_seeds(mode)
    state_builder._validate_library_structure(
        state_library,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    _validate_proposal_envelope(
        proposal_collection,
        mode=mode,
        artifact_scope="full",
        selected_seeds=seeds,
    )
    return (
        state_path,
        state_file_sha,
        state_library,
        proposal_path,
        proposal_file_sha,
        proposal_collection,
    )


def _state_audit_scientific_payload(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "elapsed_sec_diagnostic_only"
    }


def scientific_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "structural_audit_format": report["structural_audit_format"],
        "mode": report["mode"],
        "formal_result_valid": report["formal_result_valid"],
        "protocol_sha256": report["protocol_sha256"],
        "audit_git_commit": report["audit_git_commit"],
        "input_audit": report["input_audit"],
        "runtime_targets": report["runtime_targets"],
        "artifact_identity": report["artifact_identity"],
        "audit_boundary": report["audit_boundary"],
        "checks": report["checks"],
        "state_audits": [
            _state_audit_scientific_payload(row)
            for row in report["state_audits"]
        ],
        "manifest": report["manifest"],
    }


def _validate_structural_audit_report(
    report: Mapping[str, Any],
    *,
    mode: str,
    selected_seeds: Sequence[int],
) -> None:
    required_top_keys = {
        "structural_audit_format",
        "status",
        "mode",
        "formal_result_valid",
        "protocol",
        "protocol_sha256",
        "audit_git_commit",
        "git",
        "environment",
        "input_audit",
        "runtime_targets",
        "artifact_paths_diagnostic_only",
        "artifact_identity",
        "audit_boundary",
        "checks",
        "state_audits",
        "manifest",
        "elapsed_sec_diagnostic_only",
        "structural_audit_scientific_sha256",
    }
    expected_state_ids = _expected_state_ids(
        selected_seeds, mode=mode
    )
    expected_pair_count = len(
        _expected_pair_ids(selected_seeds, mode=mode)
    )
    state_audits = report.get("state_audits")
    manifest = report.get("manifest")
    if (
        set(report) != required_top_keys
        or report.get("structural_audit_format")
        != STRUCTURAL_AUDIT_FORMAT
        or report.get("status") != "complete"
        or report.get("mode") != mode
        or report.get("formal_result_valid") is not (mode == "formal")
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("audit_boundary") != AUDIT_BOUNDARY
        or report.get("checks") != PASS_CHECKS
        or not isinstance(state_audits, list)
        or not isinstance(manifest, dict)
        or [row.get("state_id") for row in state_audits]
        != expected_state_ids
        or manifest.get("dataset_order")
        != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(selected_seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_state_ids)
        or manifest.get("pair_count") != expected_pair_count
        or manifest.get("state_ids_in_fixed_order")
        != expected_state_ids
        or manifest.get("pair_ids_in_fixed_order_sha256")
        != protocol.canonical_sha256(
            _expected_pair_ids(selected_seeds, mode=mode)
        )
    ):
        raise RuntimeError("structural audit report 顶层/manifest 身份失败")

    required_state_keys = {
        "state_id",
        "source_state_scientific_sha256",
        "source_trajectory_scientific_sha256",
        "current_table_sha256",
        "pair_count",
        "pair_ids_sha256",
        "pair_structural_audits_sha256",
        "all_pairs_structurally_valid",
        "elapsed_sec_diagnostic_only",
    }
    for row in state_audits:
        state_id = row.get("state_id")
        dataset = next(
            (
                name
                for name in protocol.DATASET_ORDER
                if isinstance(state_id, str)
                and state_id.startswith(f"{name}__")
            ),
            None,
        )
        elapsed = row.get("elapsed_sec_diagnostic_only")
        if (
            set(row) != required_state_keys
            or dataset is None
            or row.get("pair_count")
            != protocol.proposals_per_state(dataset, mode=mode)
            or row.get("all_pairs_structurally_valid") is not True
            or any(
                not _is_sha256(row.get(key))
                for key in (
                    "source_state_scientific_sha256",
                    "source_trajectory_scientific_sha256",
                    "current_table_sha256",
                    "pair_ids_sha256",
                    "pair_structural_audits_sha256",
                )
            )
            or isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0.0
        ):
            raise RuntimeError("structural audit state row 身份失败")
    expected_state_audits_sha = protocol.canonical_sha256([
        _state_audit_scientific_payload(row) for row in state_audits
    ])
    if (
        manifest.get("state_structural_audits_sha256")
        != expected_state_audits_sha
        or sum(row["pair_count"] for row in state_audits)
        != expected_pair_count
    ):
        raise RuntimeError("structural audit state digest/count 失败")

    artifact = report.get("artifact_identity")
    required_artifact_keys = {
        "state_library_file_sha256",
        "state_library_scientific_sha256",
        "state_library_git_commit",
        "state_seed_shard_file_sha256",
        "proposal_collection_file_sha256",
        "proposal_collection_scientific_sha256",
        "proposal_collection_git_commit",
        "proposal_seed_shard_file_sha256",
    }
    if not isinstance(artifact, dict) or set(artifact) != required_artifact_keys:
        raise RuntimeError("structural audit artifact identity schema 失败")
    if any(
        not _is_sha256(artifact.get(key))
        for key in (
            "state_library_file_sha256",
            "state_library_scientific_sha256",
            "proposal_collection_file_sha256",
            "proposal_collection_scientific_sha256",
        )
    ):
        raise RuntimeError("structural audit artifact SHA 失败")
    for key in (
        "state_seed_shard_file_sha256",
        "proposal_seed_shard_file_sha256",
    ):
        shard_hashes = artifact.get(key)
        allowed_keys = {str(value) for value in selected_seeds}
        if (
            not isinstance(shard_hashes, dict)
            or frozenset(shard_hashes)
            not in {frozenset(), frozenset(allowed_keys)}
            or any(
                not _is_sha256(file_sha)
                for file_sha in shard_hashes.values()
            )
        ):
            raise RuntimeError("structural audit source shard SHA map 失败")
    if any(
        not isinstance(artifact.get(key), str)
        or len(artifact[key]) != 40
        for key in (
            "state_library_git_commit",
            "proposal_collection_git_commit",
        )
    ):
        raise RuntimeError("structural audit source artifact git commit 失败")
    if (
        report.get("audit_git_commit")
        != report.get("git", {}).get("commit")
        or not isinstance(report.get("audit_git_commit"), str)
        or len(report["audit_git_commit"]) != 40
    ):
        raise RuntimeError("structural audit git identity 失败")
    observed_scientific_sha = protocol.canonical_sha256(
        scientific_payload(report)
    )
    if (
        report.get("structural_audit_scientific_sha256")
        != observed_scientific_sha
    ):
        raise RuntimeError("structural audit scientific SHA-256 失败")
    if any(
        key in report for key in ("summary", "overall_result", "shared_label")
    ):
        raise RuntimeError("structural audit 不得输出机制统计或结论")


def _assert_source_paths_unchanged(
    state_path: Path,
    state_sha: str,
    proposal_path: Path,
    proposal_sha: str,
    state_shard_paths: Sequence[str | Path],
    state_shard_hashes: Mapping[str, str],
    proposal_shard_paths: Sequence[str | Path],
    proposal_shard_hashes: Mapping[str, str],
) -> None:
    if (
        protocol.file_sha256(state_path) != state_sha
        or protocol.file_sha256(proposal_path) != proposal_sha
    ):
        raise RuntimeError("full source artifact 在 structural audit 期间改变")
    observed_state_shards = sorted(
        protocol.file_sha256(Path(path).resolve())
        for path in state_shard_paths
    )
    observed_proposal_shards = sorted(
        protocol.file_sha256(Path(path).resolve())
        for path in proposal_shard_paths
    )
    if (
        observed_state_shards != sorted(state_shard_hashes.values())
        or observed_proposal_shards
        != sorted(proposal_shard_hashes.values())
    ):
        raise RuntimeError("source seed shard 在 structural audit 期间改变")


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
        "mode": "plan_only_no_artifact_read_no_audit_execution",
        "requested_mode": mode,
        "formal_result_valid": False,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "structural_audit_format": STRUCTURAL_AUDIT_FORMAT,
        "dataset_order": list(protocol.DATASET_ORDER),
        "seed_order": list(seeds),
        "state_group_order": list(protocol.STATE_GROUPS),
        "state_count": (
            len(protocol.DATASET_ORDER)
            * len(seeds)
            * len(protocol.STATE_GROUPS)
        ),
        "proposal_pair_count": pair_count,
        "audit_boundary": dict(AUDIT_BOUNDARY),
        "audit_started": False,
        "formal_confirmation_consumed": False,
    }


def audit_structure(
    mode: str,
    state_library_path: str | Path,
    proposal_collection_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_state_library_sha256: str,
    confirmed_proposal_collection_sha256: str,
    state_shard_paths: Sequence[str | Path] = (),
    proposal_shard_paths: Sequence[str | Path] = (),
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Audit all frozen state/proposal identities without evaluating results."""

    _require_sha256(
        confirmed_state_library_sha256, name="state library confirmation"
    )
    _require_sha256(
        confirmed_proposal_collection_sha256,
        name="proposal collection confirmation",
    )
    protocol.require_formal_confirmation(mode, confirmed_protocol_sha256)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(
            f"structural audit 输出已存在，不覆盖：{output}"
        )
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    git = state_builder._git_identity(REPOSITORY_ROOT)
    environment = state_builder._validate_formal_execution(
        mode,
        git,
        confirmed_execution_commit,
        require_cuda=True,
    )
    (
        state_path,
        state_file_sha,
        state_library,
        proposal_path,
        proposal_file_sha,
        proposal_collection,
    ) = _load_full_artifacts(
        mode,
        state_library_path,
        proposal_collection_path,
        confirmed_state_library_sha256=(
            confirmed_state_library_sha256
        ),
        confirmed_proposal_collection_sha256=(
            confirmed_proposal_collection_sha256
        ),
    )
    if mode == "formal" and (
        state_library.get("formal_result_valid") is not True
        or proposal_collection.get("formal_result_valid") is not True
        or state_library.get("git") != git
        or proposal_collection.get("git") != git
    ):
        raise RuntimeError(
            "formal structural audit 要求 state/proposal/current 同一 clean commit"
        )

    input_audit, runtime_inputs, runtime_targets = (
        state_builder._audit_runtime_inputs(REPOSITORY_ROOT, mode)
    )
    input_audit = _json_value(input_audit)
    runtime_targets = _json_value(runtime_targets)
    if (
        state_library["input_audit"] != input_audit
        or proposal_collection["input_audit"] != input_audit
        or state_library["runtime_targets"] != runtime_targets
        or proposal_collection["runtime_targets"] != runtime_targets
    ):
        raise RuntimeError("state/proposal/current measured inputs 身份失败")
    source_state_refs = proposal_collection.get(
        "source_state_artifacts_diagnostic_only"
    )
    if not isinstance(source_state_refs, list) or not source_state_refs:
        raise RuntimeError("proposal collection 缺少 source-state references")
    for reference in source_state_refs:
        if (
            not isinstance(reference, dict)
            or not _is_sha256(reference.get("file_sha256"))
            or not _is_sha256(
                reference.get("state_library_scientific_sha256")
            )
            or not isinstance(reference.get("git_commit"), str)
        ):
            raise RuntimeError("proposal source-state reference 无效")

    seeds = _mode_seeds(mode)
    state_shard_hashes = _audit_state_shards(
        state_library,
        state_shard_paths,
        mode=mode,
        selected_seeds=seeds,
    )
    proposal_shard_hashes = _audit_proposal_shards(
        proposal_collection,
        proposal_shard_paths,
        mode=mode,
        selected_seeds=seeds,
    )
    allowed_source_state_file_hashes = {
        state_file_sha,
        *state_shard_hashes.values(),
    }
    if any(
        reference["file_sha256"] not in allowed_source_state_file_hashes
        or (
            mode == "formal"
            and reference["git_commit"] != git["commit"]
        )
        for reference in source_state_refs
    ):
        raise RuntimeError(
            "proposal source-state references 未绑定显式审计的 state artifacts"
        )
    if mode == "formal":
        for raw_path in state_shard_paths:
            _, _, shard = _load_stable_json(
                raw_path, confirmation_name="formal state shard recheck"
            )
            if shard.get("git") != git:
                raise RuntimeError("formal state shard git identity 失败")
        for raw_path in proposal_shard_paths:
            _, _, shard = _load_stable_json(
                raw_path, confirmation_name="formal proposal shard recheck"
            )
            if shard.get("git") != git:
                raise RuntimeError("formal proposal shard git identity 失败")

    state_index = {
        row["state_id"]: row for row in state_library["states"]
    }
    trajectory_index = {
        (row["dataset"], row["seed"]): row
        for row in state_library["trajectories"]
    }
    proposal_state_index = {
        row["state_id"]: row for row in proposal_collection["states"]
    }
    expected_state_ids = _expected_state_ids(seeds, mode=mode)
    if (
        set(state_index) != set(expected_state_ids)
        or set(proposal_state_index) != set(expected_state_ids)
    ):
        raise RuntimeError("state/proposal full state identities 不完整")

    state_audits = []
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
            trajectory = trajectory_index[(dataset, seed)]
            for state_group in protocol.STATE_GROUPS:
                state_identifier = protocol.state_id(
                    dataset, seed, state_group, mode=mode
                )
                audited = _audit_state(
                    state_index[state_identifier],
                    trajectory,
                    proposal_state_index[state_identifier],
                    mode=mode,
                    queries=queries,
                    schema=schema,
                    source_target=source_target,
                    runtime_target=runtime_target,
                )
                state_audits.append(_json_value(audited))
                print(
                    f"[Stage6A structural audit {mode} "
                    f"{state_identifier}] pairs={audited['pair_count']} "
                    f"elapsed={audited['elapsed_sec_diagnostic_only']:.2f}s",
                    flush=True,
                )

    pair_count = sum(row["pair_count"] for row in state_audits)
    expected_pair_ids = _expected_pair_ids(seeds, mode=mode)
    if (
        [row["state_id"] for row in state_audits]
        != expected_state_ids
        or pair_count != len(expected_pair_ids)
        or not all(
            row["all_pairs_structurally_valid"] for row in state_audits
        )
    ):
        raise RuntimeError("structural audit state/pair coverage 失败")

    _assert_source_paths_unchanged(
        state_path,
        state_file_sha,
        proposal_path,
        proposal_file_sha,
        state_shard_paths,
        state_shard_hashes,
        proposal_shard_paths,
        proposal_shard_hashes,
    )
    if mode == "formal" and state_builder._git_identity(
        REPOSITORY_ROOT
    ) != git:
        raise RuntimeError("formal structural audit 期间 git identity 改变")

    artifact_identity = {
        "state_library_file_sha256": state_file_sha,
        "state_library_scientific_sha256": state_library[
            "state_library_scientific_sha256"
        ],
        "state_library_git_commit": state_library.get("git", {}).get(
            "commit"
        ),
        "state_seed_shard_file_sha256": state_shard_hashes,
        "proposal_collection_file_sha256": proposal_file_sha,
        "proposal_collection_scientific_sha256": proposal_collection[
            "proposal_scientific_sha256"
        ],
        "proposal_collection_git_commit": proposal_collection.get(
            "git", {}
        ).get("commit"),
        "proposal_seed_shard_file_sha256": proposal_shard_hashes,
    }
    checks = dict(PASS_CHECKS)
    report = {
        "structural_audit_format": STRUCTURAL_AUDIT_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "audit_git_commit": git["commit"],
        "git": git,
        "environment": _json_value(environment),
        "input_audit": input_audit,
        "runtime_targets": runtime_targets,
        "artifact_paths_diagnostic_only": {
            "state_library": str(state_path),
            "proposal_collection": str(proposal_path),
            "state_shards": [
                str(Path(path).resolve()) for path in state_shard_paths
            ],
            "proposal_shards": [
                str(Path(path).resolve())
                for path in proposal_shard_paths
            ],
        },
        "artifact_identity": artifact_identity,
        "audit_boundary": dict(AUDIT_BOUNDARY),
        "checks": checks,
        "state_audits": state_audits,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(state_audits),
            "pair_count": pair_count,
            "state_ids_in_fixed_order": expected_state_ids,
            "pair_ids_in_fixed_order_sha256": (
                protocol.canonical_sha256(expected_pair_ids)
            ),
            "state_structural_audits_sha256": (
                protocol.canonical_sha256([
                    _state_audit_scientific_payload(row)
                    for row in state_audits
                ])
            ),
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    report["structural_audit_scientific_sha256"] = (
        protocol.canonical_sha256(scientific_payload(report))
    )
    _validate_structural_audit_report(
        report,
        mode=mode,
        selected_seeds=seeds,
    )
    published = state_builder._exclusive_write_json(output, report)
    print(f"Stage 6A structural audit：{published}", flush=True)
    return published, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )

    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument(
        "--mode", choices=("smoke", "formal"), required=True
    )
    audit_parser.add_argument("--state-library", required=True)
    audit_parser.add_argument("--proposal-collection", required=True)
    audit_parser.add_argument("--state-shards", nargs="*", default=[])
    audit_parser.add_argument("--proposal-shards", nargs="*", default=[])
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument(
        "--confirmed-state-library-sha256", required=True
    )
    audit_parser.add_argument(
        "--confirmed-proposal-collection-sha256", required=True
    )
    audit_parser.add_argument("--confirmed-protocol-sha256")
    audit_parser.add_argument("--confirmed-execution-commit")
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
    audit_structure(
        args.mode,
        args.state_library,
        args.proposal_collection,
        args.output,
        confirmed_state_library_sha256=(
            args.confirmed_state_library_sha256
        ),
        confirmed_proposal_collection_sha256=(
            args.confirmed_proposal_collection_sha256
        ),
        state_shard_paths=args.state_shards,
        proposal_shard_paths=args.proposal_shards,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
