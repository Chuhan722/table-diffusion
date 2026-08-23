#!/usr/bin/env python3
"""Independently audit Stage 6A arithmetic and frozen evaluation results.

The auditor deliberately does not import the proposal collector, structural
auditor, or frozen evaluator.  Starting from frozen state tables, sparse edit
logs, and measured query definitions, it independently reconstructs every
copy/full table, reevaluates query indicators, recomputes exact B/C/G values,
rebuilds all descriptive strata and formal dominance decisions, and compares
those results with the frozen evaluator artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__:
    from scripts import issue53_stage6a_protocol as protocol
else:
    import issue53_stage6a_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATE_LIBRARY_FORMAT = "issue53_stage6a_state_library_v1"
PROPOSAL_COLLECTION_FORMAT = "issue53_stage6a_proposal_collection_v1"
STRUCTURAL_AUDIT_FORMAT = "issue53_stage6a_structural_audit_v1"
EVALUATION_FORMAT = "issue53_stage6a_frozen_evaluation_v1"
ARITHMETIC_AUDIT_FORMAT = "issue53_stage6a_independent_arithmetic_audit_v1"

LEG_NAMES = ("copy_only", "mutation_given_copy", "full")
EXACT_METRICS = ("b2", "c2", "g2", "cself2", "ccross2")
WORK_METRICS = (
    "participating_rows",
    "copied_rows",
    "copied_cells",
    "mutation_rows",
    "mutation_changed_cells",
    "mutation_overwrote_copied_cells",
    "copy_query_changed_rows",
    "mutation_query_changed_rows",
    "full_query_changed_rows",
)
SIGNS = ("negative", "zero", "positive")
SIGN_TRANSITIONS = tuple(
    f"{left}_to_{right}" for left in SIGNS for right in SIGNS
)
CATEGORY_TRANSITIONS = tuple(
    f"{left}_to_{right}"
    for left in protocol.FULL_GAIN_CATEGORIES
    for right in protocol.FULL_GAIN_CATEGORIES
)
PHASES = ("initial", "primary")

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

STRUCTURAL_AUDIT_BOUNDARY = {
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

STRUCTURAL_PASS_CHECKS = {
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

EVALUATION_BOUNDARY = {
    "post_hoc_only": True,
    "proposal_generation_run": False,
    "proposal_probability_modified": False,
    "proposal_feedback_to_generation": False,
    "proposal_acceptance_or_rejection": False,
    "proposal_retry_or_rollback": False,
    "proposal_selection_or_ranking": False,
    "structural_audit_required": True,
    "structural_audit_must_pass_and_bind_exact_collection": True,
    "initial_excluded_from_primary_dominance": True,
    "statistical_independence_unit": "source_trajectory_seed",
    "proposal_rows_are_not_independent_seeds": True,
    "equal_seed_weighting": True,
    "adaptive_thresholds_or_subgroups": False,
    "reference_or_held_out_read": False,
    "smoke_is_mechanism_evidence": False,
}

DESCRIPTIVE_STATISTICS_SPEC = {
    "exact_value_authority": "integer_numerator_over_positive_denominator",
    "reported_float_is_diagnostic_only": True,
    "mean": "exact arithmetic mean",
    "quartiles": (
        "Hyndman-Fan type 7 linear interpolation at p=0.25,0.50,0.75; "
        "interpolation is performed as an exact rational"
    ),
    "empty_distribution": "count_zero_and_all_statistics_null",
    "dataset_aggregate": (
        "proposal-level pooled distribution is descriptive only; the frozen "
        "dataset aggregate is the distribution of per-seed means/shares, "
        "with every seed weighted equally"
    ),
    "undefined_share": (
        "a seed with zero axis denominator is retained as undefined and is "
        "never silently converted to zero"
    ),
}

ARITHMETIC_AUDIT_BOUNDARY = {
    "collector_imported": False,
    "structural_auditor_imported": False,
    "frozen_evaluator_imported": False,
    "protocol_arithmetic_helpers_called": False,
    "protocol_classification_helpers_called": False,
    "queries_reevaluated_from_definitions": True,
    "tables_reconstructed_from_sparse_logs": True,
    "descriptive_statistics_recomputed": True,
    "dominance_and_shared_result_recomputed": True,
    "new_generation_performed": False,
    "proposal_probability_modified": False,
    "proposal_acceptance_or_rejection": False,
    "reference_table_read": False,
    "heldout_read": False,
    "offline_safety_read": False,
    "artifact_role": "final_independent_arithmetic_verification",
}

PASS_CHECKS = {
    "frozen_protocol_identity_verified": True,
    "measured_input_files_verified": True,
    "four_artifacts_cryptographically_bound": True,
    "structural_audit_pass_verified": True,
    "all_current_queries_independently_reevaluated": True,
    "all_sparse_tables_independently_reconstructed": True,
    "all_row_query_deltas_independently_reevaluated": True,
    "all_exact_bc_g_independently_recomputed": True,
    "all_sequential_identities_independently_recomputed": True,
    "all_proposal_exact_payloads_match": True,
    "all_evaluator_strata_match": True,
    "all_seed_dominance_results_match": True,
    "shared_result_matches": True,
    "no_gate_boundary_verified": True,
    "complete_matrix_verified": True,
    "overall_pass": True,
}

FORBIDDEN_PROPOSAL_FIELDS = {
    "accepted",
    "rejected",
    "winner",
    "selected_proposal",
    "shared_label",
    "overall_result",
}

INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min


def _mode_seeds(mode: str) -> tuple[int, ...]:
    if mode == "formal":
        return tuple(protocol.FORMAL_SEEDS)
    if mode == "smoke":
        return (protocol.SMOKE_SEED,)
    raise ValueError("mode 必须是 formal 或 smoke")


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
        frame.to_csv(index=False).encode("utf-8")
    ).hexdigest()


def _strict_load_json(path: str | Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"JSON 包含非标准数值：{value}")

    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise TypeError("artifact JSON 根必须是 object")
    return value


def _exclusive_write_json(path: str | Path, value: Any) -> Path:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"arithmetic audit 输出已存在，不覆盖：{output}"
        )
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
                f"arithmetic audit 输出已存在，不覆盖：{output}"
            ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _git_identity(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
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
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if mode == "smoke":
        return environment
    if mode != "formal":
        raise ValueError("mode 必须是 formal 或 smoke")
    if git.get("worktree_clean_including_untracked") is not True:
        raise RuntimeError("formal arithmetic audit 要求 clean worktree")
    if confirmed_execution_commit != git.get("commit"):
        raise PermissionError(
            "formal arithmetic audit 必须显式确认当前 execution commit"
        )
    return environment


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _expected_state_ids(seeds: Sequence[int], *, mode: str) -> list[str]:
    return [
        protocol.state_id(dataset, seed, group, mode=mode)
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for group in protocol.STATE_GROUPS
    ]


def _pair_id(state_id: str, proposal_index: int) -> str:
    return f"{state_id}__proposal_{proposal_index:04d}"


def _expected_pair_ids(seeds: Sequence[int], *, mode: str) -> list[str]:
    return [
        _pair_id(
            protocol.state_id(dataset, seed, group, mode=mode), index
        )
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for group in protocol.STATE_GROUPS
        for index in range(
            protocol.proposals_per_state(dataset, mode=mode)
        )
    ]


def _trajectory_scientific_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset",
        "seed",
        "runtime_n_records",
        "runtime_device",
        "runtime_target_sha256",
        "source_generator_params",
        "initial_table_sha256",
        "primary_rng_post_initialization_state_sha256",
        "terminal_table_sha256",
        "rounds_run",
        "candidate_evaluations",
        "termination_reason",
        "terminal_normalized_work",
        "terminal_squared_loss",
        "terminal_normalized_l1",
        "direction_reference_scale",
        "primary_rng_endpoint_sha256",
        "natural_work_snapshot_manifest",
        "selected_state_ids",
    )
    try:
        return {key: row[key] for key in keys}
    except KeyError as error:
        raise RuntimeError("state trajectory scientific payload 缺字段") from error


def _state_scientific_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "state_id",
        "dataset",
        "seed",
        "state_group",
        "target_fraction",
        "target_normalized_work",
        "selection_absolute_work_error",
        "source_snapshot_index",
        "current_count_residual",
        "snapshot",
    )
    try:
        return {key: row[key] for key in keys}
    except KeyError as error:
        raise RuntimeError("state scientific payload 缺字段") from error


def _state_library_scientific_payload(
    library: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "state_library_format": library["state_library_format"],
        "mode": library["mode"],
        "artifact_scope": library["artifact_scope"],
        "selected_seeds": library["selected_seeds"],
        "protocol_sha256": library["protocol_sha256"],
        "input_audit": library["input_audit"],
        "runtime_targets": library["runtime_targets"],
        "trajectories": [
            _trajectory_scientific_payload(row)
            for row in library["trajectories"]
        ],
        "states": [
            _state_scientific_payload(row) for row in library["states"]
        ],
    }


def _proposal_state_scientific_payload(
    state: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        key: value
        for key, value in state.items()
        if key not in {"elapsed_sec_diagnostic_only", "pairs"}
    }
    payload["pairs"] = [
        {
            key: value
            for key, value in pair.items()
            if key != "elapsed_sec_diagnostic_only"
        }
        for pair in state["pairs"]
    ]
    return payload


def _proposal_scientific_payload(
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "proposal_collection_format": proposal[
            "proposal_collection_format"
        ],
        "mode": proposal["mode"],
        "artifact_scope": proposal["artifact_scope"],
        "selected_seeds": proposal["selected_seeds"],
        "protocol_sha256": proposal["protocol_sha256"],
        "input_audit": proposal["input_audit"],
        "runtime_targets": proposal["runtime_targets"],
        "probe_boundary": proposal["probe_boundary"],
        "states": [
            _proposal_state_scientific_payload(row)
            for row in proposal["states"]
        ],
    }


def _state_audit_scientific_payload(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "elapsed_sec_diagnostic_only"
    }


def _structural_scientific_payload(
    report: Mapping[str, Any],
) -> dict[str, Any]:
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


def _evaluation_scientific_payload(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evaluation_format": report["evaluation_format"],
        "mode": report["mode"],
        "formal_result_valid": report["formal_result_valid"],
        "mechanism_evidence_emitted": report[
            "mechanism_evidence_emitted"
        ],
        "protocol_sha256": report["protocol_sha256"],
        "evaluator_git_commit": report["evaluator_git_commit"],
        "input_audit": report["input_audit"],
        "runtime_targets": report["runtime_targets"],
        "artifact_identity": report["artifact_identity"],
        "evaluation_boundary": report["evaluation_boundary"],
        "descriptive_statistics_spec": report[
            "descriptive_statistics_spec"
        ],
        "strata": report["strata"],
        "formal_dominance": report["formal_dominance"],
        "formal_overall_result": report["formal_overall_result"],
        "manifest": report["manifest"],
    }


def _load_measured_queries() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset in protocol.DATASET_ORDER:
        spec = protocol.DATASETS[dataset]
        observed_hashes = {
            key: _file_sha256(REPOSITORY_ROOT / spec[key])
            for key in ("schema", "queries", "marginals")
        }
        if observed_hashes != spec["input_sha256"]:
            raise RuntimeError(f"{dataset} measured input SHA-256 漂移")
        payload = _strict_load_json(REPOSITORY_ROOT / spec["queries"])
        queries = payload.get("queries")
        if not isinstance(queries, list) or len(queries) != spec[
            "query_count"
        ]:
            raise RuntimeError(f"{dataset} measured query count 失败")
        order_counts = Counter()
        source_target = []
        for query in queries:
            conditions = query.get("conditions")
            target = query.get("result")
            if (
                not isinstance(conditions, list)
                or isinstance(target, bool)
                or not isinstance(target, int)
            ):
                raise RuntimeError(f"{dataset} query schema/target 失败")
            order_counts[len(conditions)] += 1
            source_target.append(target)
        if dict(order_counts) != spec["query_order_counts"]:
            raise RuntimeError(f"{dataset} query order counts 漂移")
        if _canonical_sha256(source_target) != spec[
            "target_vector_sha256"
        ]:
            raise RuntimeError(f"{dataset} source target identity 漂移")
        result[dataset] = {
            "queries": queries,
            "source_target": np.asarray(source_target, dtype=np.int64),
            "source_n_records": int(spec["n_records"]),
            "observed_input_sha256": observed_hashes,
        }
    return result


def _validate_library_envelope(
    library: Mapping[str, Any], *, mode: str, seeds: Sequence[int]
) -> None:
    expected_state_ids = _expected_state_ids(seeds, mode=mode)
    expected_trajectories = [
        (dataset, seed)
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
    ]
    manifest = library.get("manifest", {})
    if (
        library.get("state_library_format") != STATE_LIBRARY_FORMAT
        or library.get("status") != "complete"
        or library.get("mode") != mode
        or library.get("artifact_scope") != "full"
        or library.get("selected_seeds") != list(seeds)
        or library.get("formal_result_valid") is not (mode == "formal")
        or library.get("protocol") != protocol.frozen_protocol_manifest()
        or library.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or [row.get("state_id") for row in library.get("states", [])]
        != expected_state_ids
        or [
            (row.get("dataset"), row.get("seed"))
            for row in library.get("trajectories", [])
        ]
        != expected_trajectories
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_state_ids)
        or manifest.get("state_ids_in_fixed_order") != expected_state_ids
        or not isinstance(manifest.get("source_seed_shard_sha256"), dict)
    ):
        raise RuntimeError("state library envelope/coverage 失败")
    if library.get("state_library_scientific_sha256") != _canonical_sha256(
        _state_library_scientific_payload(library)
    ):
        raise RuntimeError("state library scientific SHA-256 失败")


def _walk_forbidden_proposal_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        overlap = FORBIDDEN_PROPOSAL_FIELDS.intersection(value)
        if overlap:
            raise RuntimeError(
                f"proposal 出现 gate/result 字段：{sorted(overlap)}"
            )
        for item in value.values():
            _walk_forbidden_proposal_fields(item)
    elif isinstance(value, list):
        for item in value:
            _walk_forbidden_proposal_fields(item)


def _validate_proposal_envelope(
    proposal: Mapping[str, Any], *, mode: str, seeds: Sequence[int]
) -> None:
    expected_states = _expected_state_ids(seeds, mode=mode)
    expected_pairs = _expected_pair_ids(seeds, mode=mode)
    manifest = proposal.get("manifest", {})
    states = proposal.get("states", [])
    if (
        proposal.get("proposal_collection_format")
        != PROPOSAL_COLLECTION_FORMAT
        or proposal.get("status") != "complete"
        or proposal.get("mode") != mode
        or proposal.get("artifact_scope") != "full"
        or proposal.get("selected_seeds") != list(seeds)
        or proposal.get("formal_result_valid") is not (mode == "formal")
        or proposal.get("protocol") != protocol.frozen_protocol_manifest()
        or proposal.get("protocol_sha256")
        != protocol.FROZEN_PROTOCOL_SHA256
        or proposal.get("probe_boundary") != PROBE_BOUNDARY
        or [row.get("state_id") for row in states] != expected_states
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_states)
        or manifest.get("pair_count") != len(expected_pairs)
        or manifest.get("state_ids_in_fixed_order") != expected_states
        or manifest.get("pair_ids_in_fixed_order") != expected_pairs
        or not isinstance(
            manifest.get("source_proposal_seed_shard_sha256"), dict
        )
    ):
        raise RuntimeError("proposal collection envelope/coverage 失败")

    observed_pairs: list[str] = []
    addresses: set[int] = set()
    for state in states:
        dataset = state.get("dataset")
        seed = state.get("seed")
        group = state.get("state_group")
        pairs = state.get("pairs")
        if (
            dataset not in protocol.DATASET_ORDER
            or seed not in seeds
            or group not in protocol.STATE_GROUPS
            or not isinstance(pairs, list)
            or len(pairs)
            != protocol.proposals_per_state(dataset, mode=mode)
        ):
            raise RuntimeError("proposal state envelope 失败")
        state_id = protocol.state_id(dataset, seed, group, mode=mode)
        for index, pair in enumerate(pairs):
            expected_id = _pair_id(state_id, index)
            rng = pair.get("rng", {})
            donor = protocol.proposal_address_seed(
                dataset, seed, group, index, "donor", mode=mode
            )
            update = protocol.proposal_address_seed(
                dataset, seed, group, index, "update", mode=mode
            )
            if (
                pair.get("pair_id") != expected_id
                or pair.get("state_id") != state_id
                or pair.get("dataset") != dataset
                or pair.get("seed") != seed
                or pair.get("state_group") != group
                or pair.get("proposal_index") != index
                or pair.get("retained_unconditionally") is not True
                or rng.get("donor_address_uint64") != donor
                or rng.get("update_address_uint64") != update
                or donor == update
                or donor in addresses
                or update in addresses
            ):
                raise RuntimeError("proposal pair/address identity 失败")
            addresses.update((donor, update))
            observed_pairs.append(expected_id)
    if observed_pairs != expected_pairs:
        raise RuntimeError("proposal pair fixed order 失败")
    if proposal.get("proposal_scientific_sha256") != _canonical_sha256(
        _proposal_scientific_payload(proposal)
    ):
        raise RuntimeError("proposal scientific SHA-256 失败")
    _walk_forbidden_proposal_fields(proposal)


def _validate_structural_envelope(
    report: Mapping[str, Any],
    *,
    mode: str,
    seeds: Sequence[int],
) -> None:
    expected_states = _expected_state_ids(seeds, mode=mode)
    expected_pairs = _expected_pair_ids(seeds, mode=mode)
    state_rows = report.get("state_audits", [])
    manifest = report.get("manifest", {})
    if (
        report.get("structural_audit_format") != STRUCTURAL_AUDIT_FORMAT
        or report.get("status") != "complete"
        or report.get("mode") != mode
        or report.get("formal_result_valid") is not (mode == "formal")
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("audit_boundary") != STRUCTURAL_AUDIT_BOUNDARY
        or report.get("checks") != STRUCTURAL_PASS_CHECKS
        or [row.get("state_id") for row in state_rows] != expected_states
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_states)
        or manifest.get("pair_count") != len(expected_pairs)
        or manifest.get("state_ids_in_fixed_order") != expected_states
        or manifest.get("pair_ids_in_fixed_order_sha256")
        != _canonical_sha256(expected_pairs)
    ):
        raise RuntimeError("structural audit envelope/pass 失败")
    if report.get("structural_audit_scientific_sha256") != (
        _canonical_sha256(_structural_scientific_payload(report))
    ):
        raise RuntimeError("structural audit scientific SHA-256 失败")
    for row in state_rows:
        dataset = next(
            (
                name
                for name in protocol.DATASET_ORDER
                if row["state_id"].startswith(f"{name}__")
            ),
            None,
        )
        if (
            dataset is None
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
        ):
            raise RuntimeError("structural audit state pass row 失败")
    if manifest.get("state_structural_audits_sha256") != _canonical_sha256([
        _state_audit_scientific_payload(row) for row in state_rows
    ]):
        raise RuntimeError("structural audit state rows digest 失败")


def _validate_evaluation_envelope(
    report: Mapping[str, Any],
    *,
    mode: str,
    seeds: Sequence[int],
) -> None:
    expected_pairs = _expected_pair_ids(seeds, mode=mode)
    expected_state_count = len(_expected_state_ids(seeds, mode=mode))
    primary_count = sum(
        len(seeds)
        * len(protocol.PRIMARY_STATE_GROUPS)
        * protocol.proposals_per_state(dataset, mode=mode)
        for dataset in protocol.DATASET_ORDER
    )
    manifest = report.get("manifest", {})
    if (
        report.get("evaluation_format") != EVALUATION_FORMAT
        or report.get("status") != "complete"
        or report.get("mode") != mode
        or report.get("formal_result_valid") is not (mode == "formal")
        or report.get("mechanism_evidence_emitted") is not (mode == "formal")
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("evaluation_boundary") != EVALUATION_BOUNDARY
        or report.get("descriptive_statistics_spec")
        != DESCRIPTIVE_STATISTICS_SPEC
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("primary_state_group_order")
        != list(protocol.PRIMARY_STATE_GROUPS)
        or manifest.get("phase_order") != list(PHASES)
        or manifest.get("leg_order") != list(LEG_NAMES)
        or manifest.get("exact_metric_order") != list(EXACT_METRICS)
        or manifest.get("work_metric_order") != list(WORK_METRICS)
        or manifest.get("copy_full_category_transition_order")
        != list(CATEGORY_TRANSITIONS)
        or manifest.get("state_summary_count") != expected_state_count
        or manifest.get("pair_count") != len(expected_pairs)
        or manifest.get("primary_pair_count") != primary_count
        or manifest.get("initial_pair_count")
        != len(expected_pairs) - primary_count
        or manifest.get("source_pair_ids_in_fixed_order_sha256")
        != _canonical_sha256(expected_pairs)
        or not isinstance(report.get("strata"), dict)
    ):
        raise RuntimeError("frozen evaluation envelope/coverage 失败")
    if mode == "formal":
        if (
            not isinstance(report.get("formal_dominance"), dict)
            or report.get("formal_overall_result")
            not in protocol.ALLOWED_OVERALL_RESULTS
        ):
            raise RuntimeError("formal evaluation result envelope 失败")
    elif (
        report.get("formal_dominance") is not None
        or report.get("formal_overall_result") is not None
    ):
        raise RuntimeError("smoke evaluation 不得有 formal result")
    if report.get("evaluation_scientific_sha256") != _canonical_sha256(
        _evaluation_scientific_payload(report)
    ):
        raise RuntimeError("frozen evaluation scientific SHA-256 失败")


def _decode_snapshot_table(snapshot: Mapping[str, Any]) -> pd.DataFrame:
    columns = snapshot.get("table_columns")
    records = snapshot.get("table_records")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
        or not isinstance(records, list)
    ):
        raise RuntimeError("snapshot table payload 无效")
    return pd.DataFrame.from_records(records, columns=columns)


def _coerce_equal_value(series: pd.Series, value: Any) -> Any:
    if pd.api.types.is_numeric_dtype(series):
        try:
            return pd.to_numeric(value)
        except (TypeError, ValueError):
            return value
    return str(value)


def _condition_mask(
    frame: pd.DataFrame, condition: Mapping[str, Any]
) -> np.ndarray:
    attribute = condition.get("attribute")
    operator = condition.get("operator")
    if attribute not in frame.columns:
        raise RuntimeError("query condition attribute 不在 frozen table")
    series = frame[attribute]
    if operator == "==":
        if set(condition) != {"attribute", "operator", "value"}:
            raise RuntimeError("equality query condition schema 失败")
        value = _coerce_equal_value(series, condition["value"])
        mask = series == value
    elif operator == ">=":
        if set(condition) != {"attribute", "operator", "value"}:
            raise RuntimeError(">= query condition schema 失败")
        mask = series >= condition["value"]
    elif operator == "between":
        if set(condition) != {
            "attribute",
            "operator",
            "lower",
            "upper",
        }:
            raise RuntimeError("between query condition schema 失败")
        mask = (series >= condition["lower"]) & (
            series <= condition["upper"]
        )
    else:
        raise RuntimeError(f"未冻结 query operator：{operator!r}")
    result = np.asarray(mask, dtype=bool)
    if result.shape != (len(frame),):
        raise RuntimeError("query condition mask shape 失败")
    return result


def _query_mask(frame: pd.DataFrame, query: Mapping[str, Any]) -> np.ndarray:
    conditions = query.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise RuntimeError("query conditions 必须是非空列表")
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        if not isinstance(condition, Mapping):
            raise RuntimeError("query condition 必须是 object")
        mask &= _condition_mask(frame, condition)
    return mask


def _query_counts(
    frame: pd.DataFrame, queries: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    return np.asarray(
        [int(np.count_nonzero(_query_mask(frame, query))) for query in queries],
        dtype=np.int64,
    )


def _changed_rows(
    before: pd.DataFrame,
    after: pd.DataFrame,
    columns: Sequence[str],
) -> np.ndarray:
    before_values = before[list(columns)].reset_index(drop=True).to_numpy()
    after_values = after[list(columns)].reset_index(drop=True).to_numpy()
    if before_values.shape != after_values.shape:
        raise RuntimeError("before/after table shape 不一致")
    return np.flatnonzero(np.any(before_values != after_values, axis=1))


def _row_query_deltas(
    before: pd.DataFrame,
    after: pd.DataFrame,
    queries: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    rows = _changed_rows(before, after, columns)
    if len(rows) == 0:
        return rows, np.zeros((0, len(queries)), dtype=np.int8)
    before_rows = before.iloc[rows].reset_index(drop=True)
    after_rows = after.iloc[rows].reset_index(drop=True)
    deltas = np.empty((len(rows), len(queries)), dtype=np.int8)
    for index, query in enumerate(queries):
        deltas[:, index] = (
            _query_mask(after_rows, query).astype(np.int8)
            - _query_mask(before_rows, query).astype(np.int8)
        )
    if np.any(np.abs(deltas) > 1):
        raise RuntimeError("row query delta 不在 {-1,0,1}")
    return rows, deltas


def _apply_sparse_logs(
    current: pd.DataFrame,
    copy_edits: Any,
    mutation_events: Any,
    columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if not isinstance(copy_edits, list) or not isinstance(
        mutation_events, list
    ):
        raise RuntimeError("sparse edit logs 必须是列表")
    allowed = set(columns)
    copy_table = current.reset_index(drop=True).copy(deep=True)
    copy_rows: list[int] = []
    copied_positions: set[tuple[int, str]] = set()
    copied_cells = 0
    column_order = {name: index for index, name in enumerate(columns)}
    for edit in copy_edits:
        if not isinstance(edit, Mapping) or set(edit) != {
            "row_index",
            "donor_index",
            "cells",
        }:
            raise RuntimeError("copy edit row schema 失败")
        row = edit["row_index"]
        donor = edit["donor_index"]
        cells = edit["cells"]
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < len(copy_table)
            or isinstance(donor, bool)
            or not isinstance(donor, int)
            or not 0 <= donor < len(copy_table)
            or not isinstance(cells, list)
            or not cells
        ):
            raise RuntimeError("copy edit row identity 失败")
        copy_rows.append(row)
        seen_attributes: list[str] = []
        for cell in cells:
            if not isinstance(cell, Mapping) or set(cell) != {
                "attribute",
                "before",
                "after",
            }:
                raise RuntimeError("copy edit cell schema 失败")
            attribute = cell["attribute"]
            if (
                attribute not in allowed
                or attribute in seen_attributes
                or copy_table.at[row, attribute] != cell["before"]
                or cell["before"] == cell["after"]
            ):
                raise RuntimeError("copy edit before/after identity 失败")
            seen_attributes.append(attribute)
            copied_positions.add((row, attribute))
            copy_table.at[row, attribute] = cell["after"]
            copied_cells += 1
        if [column_order[value] for value in seen_attributes] != sorted(
            column_order[value] for value in seen_attributes
        ):
            raise RuntimeError("copy edit attributes 未按 schema 顺序")
    if copy_rows != sorted(set(copy_rows)):
        raise RuntimeError("copy edit rows 必须严格递增唯一")

    full_table = copy_table.copy(deep=True)
    mutation_rows: list[int] = []
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
        if not isinstance(event, Mapping) or set(event) != required:
            raise RuntimeError("mutation event schema 失败")
        row = event["row_index"]
        attribute_index = event["attribute_index"]
        attribute = event["attribute"]
        if (
            isinstance(row, bool)
            or not isinstance(row, int)
            or not 0 <= row < len(full_table)
            or isinstance(attribute_index, bool)
            or not isinstance(attribute_index, int)
            or not 0 <= attribute_index < len(columns)
            or attribute != columns[attribute_index]
            or not isinstance(event["changed"], bool)
            or not isinstance(event["overwrote_copied_cell"], bool)
            or full_table.at[row, attribute] != event["before_copy"]
            or event["changed"]
            != bool(event["before_copy"] != event["sampled_value"])
            or event["overwrote_copied_cell"]
            != ((row, attribute) in copied_positions)
        ):
            raise RuntimeError("mutation event identity 失败")
        mutation_rows.append(row)
        full_table.at[row, attribute] = event["sampled_value"]
    if mutation_rows != sorted(set(mutation_rows)):
        raise RuntimeError("mutation rows 必须严格递增唯一")
    return copy_table, full_table, copied_cells


def _sign(value: int) -> str:
    return "negative" if value < 0 else "positive" if value > 0 else "zero"


def _classify_gain(b2: int, c2: int) -> str:
    if c2 < 0 or (c2 == 0 and b2 != 0):
        raise RuntimeError("independent B/C identity 不可能")
    gain = b2 - c2
    if gain > 0:
        return "improving"
    if gain == 0:
        return "unchanged" if c2 == 0 else "exact_balance"
    return "direction_failure" if b2 <= 0 else "curvature_overrun"


def _curvature_role(b2: int, cself2: int, ccross2: int) -> str:
    if _classify_gain(b2, cself2 + ccross2) != "curvature_overrun":
        raise RuntimeError("self/cross role 仅对 curvature overrun 定义")
    without_cross = b2 - cself2
    if without_cross < 0:
        return "self_sufficient_overrun"
    if without_cross == 0:
        return "cross_breaks_tie"
    return "cross_decisive_overrun"


def _require_int64(name: str, value: int) -> int:
    value = int(value)
    if not INT64_MIN <= value <= INT64_MAX:
        raise OverflowError(f"{name} 超出 int64")
    return value


def _exact_decomposition(
    row_deltas: np.ndarray,
    residual_numerators: np.ndarray,
    denominator: int,
) -> dict[str, Any]:
    deltas = np.asarray(row_deltas)
    residual = np.asarray(residual_numerators)
    if (
        deltas.ndim != 2
        or deltas.dtype.kind not in "iu"
        or deltas.dtype.kind == "b"
        or residual.shape != (deltas.shape[1],)
        or residual.dtype.kind not in "iu"
        or residual.dtype.kind == "b"
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        raise RuntimeError("independent exact input schema 失败")
    delta_q = [
        _require_int64(
            "delta_q",
            sum(int(deltas[row, column]) for row in range(len(deltas))),
        )
        for column in range(deltas.shape[1])
    ]
    b2 = _require_int64(
        "B2 numerator",
        2 * sum(int(residual[index]) * delta_q[index]
                for index in range(len(delta_q))),
    )
    c2 = _require_int64(
        "C2 numerator",
        denominator * sum(value * value for value in delta_q),
    )
    cself2 = _require_int64(
        "Cself2 numerator",
        denominator
        * sum(int(value) * int(value) for value in deltas.flat),
    )
    ccross2 = _require_int64("Ccross2 numerator", c2 - cself2)
    g2 = _require_int64("G2 numerator", b2 - c2)
    category = _classify_gain(b2, c2)
    role = (
        _curvature_role(b2, cself2, ccross2)
        if category == "curvature_overrun"
        else None
    )
    values = {
        "b2": b2,
        "cself2": cself2,
        "ccross2": ccross2,
        "c2": c2,
        "g2": g2,
    }
    integral = all(value % denominator == 0 for value in values.values())
    return {
        "denominator": denominator,
        "delta_q": delta_q,
        **{f"{name}_numerator": value for name, value in values.items()},
        "integral_doubled_units": integral,
        "integral_count_residual_units": bool(
            np.all(residual % denominator == 0)
        ),
        **{
            name: value // denominator if integral else None
            for name, value in values.items()
        },
        "category": category,
        "curvature_role": role,
        "cross_label": (
            "cross_required_overrun"
            if role in {"cross_breaks_tie", "cross_decisive_overrun"}
            else role
        ),
    }


def _sequential_payload(
    copy: Mapping[str, Any],
    mutation: Mapping[str, Any],
    full: Mapping[str, Any],
) -> dict[str, Any]:
    denominator = int(copy["denominator"])
    copy_delta = [int(value) for value in copy["delta_q"]]
    mutation_delta = [int(value) for value in mutation["delta_q"]]
    if (
        mutation["denominator"] != denominator
        or full["denominator"] != denominator
        or [
            copy_delta[index] + mutation_delta[index]
            for index in range(len(copy_delta))
        ]
        != full["delta_q"]
    ):
        raise RuntimeError("independent dfull sequential identity 失败")
    interaction = _require_int64(
        "copy/mutation interaction",
        denominator
        * 2
        * sum(
            copy_delta[index] * mutation_delta[index]
            for index in range(len(copy_delta))
        ),
    )
    if (
        full["b2_numerator"]
        != copy["b2_numerator"] + mutation["b2_numerator"] + interaction
        or full["c2_numerator"]
        != copy["c2_numerator"] + mutation["c2_numerator"] + interaction
        or full["g2_numerator"]
        != copy["g2_numerator"] + mutation["g2_numerator"]
    ):
        raise RuntimeError("independent B/C/G sequential identity 失败")
    integral = interaction % denominator == 0
    return {
        "denominator": denominator,
        "copy_mutation_interaction2_numerator": interaction,
        "copy_mutation_interaction2": (
            interaction // denominator if integral else None
        ),
        "integral_doubled_units": integral,
        "b_identity_verified": True,
        "c_identity_verified": True,
        "g_identity_verified": True,
    }


def _compact_exact_leg(exact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "denominator": exact["denominator"],
        **{
            f"{metric}_numerator": exact[f"{metric}_numerator"]
            for metric in EXACT_METRICS
        },
        "category": exact["category"],
        "curvature_role": exact["curvature_role"],
        "cross_label": exact["cross_label"],
    }


def _audit_pair_arithmetic(
    pair: Mapping[str, Any],
    *,
    current: pd.DataFrame,
    queries: Sequence[Mapping[str, Any]],
    residual_numerators: np.ndarray,
    denominator: int,
    columns: Sequence[str],
    formal: bool,
) -> tuple[dict[str, Any], str]:
    copy_table, full_table, copied_cells = _apply_sparse_logs(
        current, pair.get("copy_edits"), pair.get("mutation_events"), columns
    )
    copy_rows, copy_deltas = _row_query_deltas(
        current, copy_table, queries, columns
    )
    mutation_rows, mutation_deltas = _row_query_deltas(
        copy_table, full_table, queries, columns
    )
    full_rows, full_deltas = _row_query_deltas(
        current, full_table, queries, columns
    )
    copy_exact = _exact_decomposition(
        copy_deltas, residual_numerators, denominator
    )
    copy_delta = np.asarray(copy_exact["delta_q"], dtype=np.int64)
    mutation_residual = np.asarray([
        _require_int64(
            "mutation residual numerator",
            int(residual_numerators[index])
            - denominator * int(copy_delta[index]),
        )
        for index in range(len(copy_delta))
    ], dtype=np.int64)
    mutation_exact = _exact_decomposition(
        mutation_deltas, mutation_residual, denominator
    )
    full_exact = _exact_decomposition(
        full_deltas, residual_numerators, denominator
    )
    sequential = _sequential_payload(copy_exact, mutation_exact, full_exact)
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
    exact = {
        "copy_only": copy_exact,
        "mutation_given_copy": mutation_exact,
        "full": full_exact,
        "sequential": sequential,
        "copy_full_gain_sign_transition": (
            f"{_sign(copy_gain)}_to_{_sign(full_gain)}"
        ),
        "mutation_failure_source": mutation_source,
    }
    if pair.get("exact") != exact:
        raise RuntimeError("proposal exact payload 与独立复算不一致")
    if formal and not all(
        row["integral_doubled_units"]
        and row["integral_count_residual_units"]
        for row in (copy_exact, mutation_exact, full_exact)
    ):
        raise RuntimeError("formal exact units 未约成整数")

    participating = pair.get("participating_row_indices")
    if (
        not isinstance(participating, list)
        or participating != sorted(set(participating))
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < len(current)
            for value in participating
        )
    ):
        raise RuntimeError("participating row identities 失败")
    copy_edits = pair["copy_edits"]
    mutation_events = pair["mutation_events"]
    work = {
        "participating_rows": len(participating),
        "copied_rows": len(copy_edits),
        "copied_cells": copied_cells,
        "mutation_rows": len(mutation_events),
        "mutation_changed_cells": sum(
            event["changed"] for event in mutation_events
        ),
        "mutation_overwrote_copied_cells": sum(
            event["overwrote_copied_cell"] for event in mutation_events
        ),
        "copy_query_changed_rows": len(copy_rows),
        "mutation_query_changed_rows": len(mutation_rows),
        "full_query_changed_rows": len(full_rows),
    }
    table_hashes = {
        "current": _frame_sha256(current),
        "copy_only": _frame_sha256(copy_table),
        "full": _frame_sha256(full_table),
    }
    if pair.get("work") != work or pair.get("table_sha256") != table_hashes:
        raise RuntimeError("proposal work/table hashes 与独立复算不一致")
    elapsed = pair.get("elapsed_sec_diagnostic_only")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(elapsed)
        or elapsed < 0.0
    ):
        raise RuntimeError("proposal wallclock 无效")
    compact = {
        "pair_id": pair["pair_id"],
        "dataset": pair["dataset"],
        "seed": pair["seed"],
        "state_group": pair["state_group"],
        "exact": {
            leg: _compact_exact_leg(exact[leg]) for leg in LEG_NAMES
        }
        | {
            "sequential": {
                "denominator": sequential["denominator"],
                "copy_mutation_interaction2_numerator": sequential[
                    "copy_mutation_interaction2_numerator"
                ],
            },
            "copy_full_gain_sign_transition": exact[
                "copy_full_gain_sign_transition"
            ],
            "mutation_failure_source": mutation_source,
        },
        "work": work,
        "elapsed_sec_diagnostic_only": float(elapsed),
    }
    identity = {
        "pair_id": pair["pair_id"],
        "current_query_residual_sha256": _canonical_sha256({
            "residual_numerators": residual_numerators.tolist(),
            "denominator": denominator,
        }),
        "copy_table_sha256": table_hashes["copy_only"],
        "full_table_sha256": table_hashes["full"],
        "exact_sha256": _canonical_sha256(exact),
        "work_sha256": _canonical_sha256(work),
    }
    return compact, _canonical_sha256(identity)


def _source_residual_numerators(
    source_target: np.ndarray,
    runtime_n_records: int,
    source_n_records: int,
    current_counts: np.ndarray,
) -> np.ndarray:
    target = np.asarray(source_target)
    counts = np.asarray(current_counts)
    if (
        target.shape != counts.shape
        or target.dtype.kind not in "iu"
        or counts.dtype.kind not in "iu"
    ):
        raise RuntimeError("source target/current counts 必须是整数匹配向量")
    values = [
        int(target[index]) * runtime_n_records
        - int(counts[index]) * source_n_records
        for index in range(len(target))
    ]
    for value in values:
        _require_int64("source residual numerator", value)
    return np.asarray(values, dtype=np.int64)


def _audit_state_arithmetic(
    state: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    proposal_state: Mapping[str, Any],
    *,
    mode: str,
    measured: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.perf_counter()
    dataset = state.get("dataset")
    seed = state.get("seed")
    group = state.get("state_group")
    if (
        dataset not in protocol.DATASET_ORDER
        or seed not in _mode_seeds(mode)
        or group not in protocol.STATE_GROUPS
    ):
        raise RuntimeError("state identity 不在冻结矩阵")
    state_id = protocol.state_id(dataset, seed, group, mode=mode)
    if (
        state.get("state_id") != state_id
        or trajectory.get("dataset") != dataset
        or trajectory.get("seed") != seed
        or proposal_state.get("state_id") != state_id
        or proposal_state.get("dataset") != dataset
        or proposal_state.get("seed") != seed
        or proposal_state.get("state_group") != group
        or proposal_state.get("source_state_scientific_sha256")
        != _canonical_sha256(_state_scientific_payload(state))
        or proposal_state.get("source_trajectory_scientific_sha256")
        != _canonical_sha256(_trajectory_scientific_payload(trajectory))
    ):
        raise RuntimeError(f"{state_id} state/trajectory/proposal 绑定失败")

    snapshot = state.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise RuntimeError(f"{state_id} snapshot 缺失")
    current = _decode_snapshot_table(snapshot)
    columns = list(current.columns)
    runtime_n = protocol.source_generator_params(
        dataset, seed, mode=mode
    )["n_records"]
    if len(current) != runtime_n:
        raise RuntimeError(f"{state_id} runtime table row count 失败")
    current_hash = _frame_sha256(current)
    if (
        snapshot.get("snapshot_format") != "natural_work_current_v1"
        or snapshot.get("current_table_sha256") != current_hash
        or proposal_state.get("current_table_sha256") != current_hash
    ):
        raise RuntimeError(f"{state_id} current table identity 失败")

    queries = measured["queries"]
    independent_q = _query_counts(current, queries)
    stored_q = np.asarray(snapshot.get("current_query_answers"))
    if (
        stored_q.shape != independent_q.shape
        or stored_q.dtype.kind not in "iuf"
        or not np.array_equal(stored_q.astype(float), independent_q.astype(float))
        or proposal_state.get("current_query_answers_sha256")
        != _array_sha256(independent_q)
    ):
        raise RuntimeError(f"{state_id} current query counts 独立复算失败")

    source_n = int(measured["source_n_records"])
    residual_numerators = _source_residual_numerators(
        measured["source_target"], len(current), source_n, independent_q
    )
    if (
        proposal_state.get("residual_denominator") != source_n
        or proposal_state.get("count_residual_numerators_sha256")
        != _array_sha256(residual_numerators)
    ):
        raise RuntimeError(f"{state_id} residual numerator identity 失败")
    runtime_target = (
        measured["source_target"].astype(float)
        * (len(current) / source_n)
    )
    expected_residual = runtime_target - independent_q.astype(float)
    stored_residual = np.asarray(state.get("current_count_residual"))
    if (
        stored_residual.shape != expected_residual.shape
        or stored_residual.dtype.kind not in "iuf"
        or not np.array_equal(stored_residual.astype(float), expected_residual)
    ):
        raise RuntimeError(f"{state_id} current count residual 失败")
    if mode == "formal" and (
        np.any(residual_numerators % source_n != 0)
        or not np.array_equal(
            residual_numerators // source_n,
            expected_residual.astype(np.int64),
        )
    ):
        raise RuntimeError(f"{state_id} formal residual units 失败")

    pairs = proposal_state.get("pairs")
    expected_pair_count = protocol.proposals_per_state(dataset, mode=mode)
    if not isinstance(pairs, list) or len(pairs) != expected_pair_count:
        raise RuntimeError(f"{state_id} pair count 失败")
    compact_pairs: list[dict[str, Any]] = []
    pair_hashes = []
    for index, pair in enumerate(pairs):
        if pair.get("pair_id") != _pair_id(state_id, index):
            raise RuntimeError(f"{state_id} pair fixed order 失败")
        compact, pair_hash = _audit_pair_arithmetic(
            pair,
            current=current,
            queries=queries,
            residual_numerators=residual_numerators,
            denominator=source_n,
            columns=columns,
            formal=mode == "formal",
        )
        compact_pairs.append(compact)
        pair_hashes.append(pair_hash)

    row = {
        "state_id": state_id,
        "source_state_scientific_sha256": _canonical_sha256(
            _state_scientific_payload(state)
        ),
        "source_trajectory_scientific_sha256": _canonical_sha256(
            _trajectory_scientific_payload(trajectory)
        ),
        "current_table_sha256": current_hash,
        "independent_current_query_answers_sha256": _array_sha256(
            independent_q
        ),
        "independent_residual_numerators_sha256": _array_sha256(
            residual_numerators
        ),
        "pair_count": len(compact_pairs),
        "pair_ids_sha256": _canonical_sha256([
            pair["pair_id"] for pair in compact_pairs
        ]),
        "pair_arithmetic_audits_sha256": _canonical_sha256(pair_hashes),
        "all_pairs_arithmetic_valid": True,
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    }
    return row, compact_pairs


def _fraction_payload(value: Fraction) -> dict[str, Any]:
    value = Fraction(value)
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "value_diagnostic_only": float(value),
    }


def _type7_fraction(
    ordered: Sequence[Fraction], probability: Fraction
) -> Fraction:
    if not ordered:
        raise RuntimeError("empty exact distribution 没有 quantile")
    location = Fraction(len(ordered) - 1, 1) * probability
    lower = location.numerator // location.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = location - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _fraction_summary(values: Sequence[Fraction]) -> dict[str, Any]:
    ordered = sorted(Fraction(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "q25": None,
            "median": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(ordered),
        "mean": _fraction_payload(
            sum(ordered, Fraction(0, 1)) / len(ordered)
        ),
        "q25": _fraction_payload(
            _type7_fraction(ordered, Fraction(1, 4))
        ),
        "median": _fraction_payload(
            _type7_fraction(ordered, Fraction(1, 2))
        ),
        "q75": _fraction_payload(
            _type7_fraction(ordered, Fraction(3, 4))
        ),
        "minimum": _fraction_payload(ordered[0]),
        "maximum": _fraction_payload(ordered[-1]),
    }


def _type7_float(ordered: Sequence[float], probability: Fraction) -> float:
    if not ordered:
        raise RuntimeError("empty float distribution 没有 quantile")
    location = Fraction(len(ordered) - 1, 1) * probability
    lower = location.numerator // location.denominator
    upper = min(lower + 1, len(ordered) - 1)
    weight = float(location - lower)
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


def _float_summary(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise RuntimeError("float summary 包含非有限数")
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "q25": None,
            "median": None,
            "q75": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered),
        "q25": _type7_float(ordered, Fraction(1, 4)),
        "median": _type7_float(ordered, Fraction(1, 2)),
        "q75": _type7_float(ordered, Fraction(3, 4)),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def _fixed_counts(
    values: Sequence[str | None], labels: Sequence[str]
) -> dict[str, int]:
    allowed = set(labels)
    unexpected = {value for value in values if value not in allowed | {None}}
    if unexpected:
        raise RuntimeError(f"出现未冻结 label：{sorted(unexpected)}")
    observed = Counter(value for value in values if value is not None)
    return {label: int(observed[label]) for label in labels}


def _pair_metric(
    pair: Mapping[str, Any], leg: str, metric: str
) -> Fraction:
    exact = pair["exact"][leg]
    return Fraction(
        exact[f"{metric}_numerator"], exact["denominator"]
    )


def _pair_interaction(pair: Mapping[str, Any]) -> Fraction:
    sequential = pair["exact"]["sequential"]
    return Fraction(
        sequential["copy_mutation_interaction2_numerator"],
        sequential["denominator"],
    )


def _category_transition(pair: Mapping[str, Any]) -> str:
    return (
        f"{pair['exact']['copy_only']['category']}_to_"
        f"{pair['exact']['full']['category']}"
    )


def _summarize_pairs(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    legs: dict[str, Any] = {}
    for leg in LEG_NAMES:
        legs[leg] = {
            "metrics": {
                metric: _fraction_summary([
                    _pair_metric(pair, leg, metric) for pair in pairs
                ])
                for metric in EXACT_METRICS
            },
            "category_counts": _fixed_counts(
                [pair["exact"][leg]["category"] for pair in pairs],
                protocol.FULL_GAIN_CATEGORIES,
            ),
            "curvature_role_counts": _fixed_counts(
                [pair["exact"][leg]["curvature_role"] for pair in pairs],
                protocol.CURVATURE_ROLE_CATEGORIES,
            ),
            "cross_label_counts": _fixed_counts(
                [pair["exact"][leg]["cross_label"] for pair in pairs],
                protocol.CROSS_LABELS,
            ),
        }
    return {
        "pair_count": len(pairs),
        "legs": legs,
        "sequential": {
            "copy_mutation_interaction2": _fraction_summary([
                _pair_interaction(pair) for pair in pairs
            ]),
            "copy_full_gain_sign_transition_counts": _fixed_counts(
                [
                    pair["exact"]["copy_full_gain_sign_transition"]
                    for pair in pairs
                ],
                SIGN_TRANSITIONS,
            ),
            "copy_full_category_transition_counts": _fixed_counts(
                [_category_transition(pair) for pair in pairs],
                CATEGORY_TRANSITIONS,
            ),
            "mutation_failure_source_counts": _fixed_counts(
                [
                    pair["exact"]["mutation_failure_source"]
                    for pair in pairs
                ],
                protocol.MUTATION_FAILURE_LABELS,
            ),
        },
        "work": {
            metric: _fraction_summary([
                Fraction(pair["work"][metric], 1) for pair in pairs
            ])
            for metric in WORK_METRICS
        },
        "wallclock_sec_diagnostic_only": _float_summary([
            pair["elapsed_sec_diagnostic_only"] for pair in pairs
        ]),
    }


def _seed_mean(
    pairs: Sequence[Mapping[str, Any]],
    extractor: Callable[[Mapping[str, Any]], Fraction],
) -> Fraction:
    if not pairs:
        raise RuntimeError("equal-seed stratum 为空")
    values = [extractor(pair) for pair in pairs]
    return sum(values, Fraction(0, 1)) / len(values)


def _equal_seed_means(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], Fraction],
) -> dict[str, Any]:
    rows = []
    means = []
    for seed, pairs in pairs_by_seed.items():
        mean = _seed_mean(pairs, extractor)
        means.append(mean)
        rows.append({
            "seed": seed,
            "proposal_count": len(pairs),
            "mean": _fraction_payload(mean),
        })
    return {
        "per_seed": rows,
        "equal_seed_distribution": _fraction_summary(means),
    }


def _equal_seed_float_means(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], float],
) -> dict[str, Any]:
    rows = []
    means = []
    for seed, pairs in pairs_by_seed.items():
        values = [float(extractor(pair)) for pair in pairs]
        if not values or any(not math.isfinite(value) for value in values):
            raise RuntimeError("equal-seed float stratum 无效")
        mean = sum(values) / len(values)
        means.append(mean)
        rows.append({
            "seed": seed,
            "proposal_count": len(pairs),
            "mean": mean,
        })
    return {
        "per_seed": rows,
        "equal_seed_distribution": _float_summary(means),
    }


def _equal_seed_shares(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
    extractor: Callable[[Mapping[str, Any]], str | None],
    labels: Sequence[str],
    selected: str,
) -> dict[str, Any]:
    rows = []
    shares = []
    for seed, pairs in pairs_by_seed.items():
        counts = _fixed_counts([extractor(pair) for pair in pairs], labels)
        denominator = sum(counts.values())
        share = Fraction(counts[selected], denominator) if denominator else None
        if share is not None:
            shares.append(share)
        rows.append({
            "seed": seed,
            "count": counts[selected],
            "denominator": denominator,
            "share": _fraction_payload(share) if share is not None else None,
        })
    return {
        "per_seed": rows,
        "defined_seed_count": len(shares),
        "undefined_seed_count": len(pairs_by_seed) - len(shares),
        "equal_seed_distribution": _fraction_summary(shares),
    }


def _equal_seed_aggregate(
    pairs_by_seed: Mapping[int, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    legs = {}
    for leg in LEG_NAMES:
        legs[leg] = {
            "metric_seed_means": {
                metric: _equal_seed_means(
                    pairs_by_seed,
                    lambda pair, leg=leg, metric=metric: _pair_metric(
                        pair, leg, metric
                    ),
                )
                for metric in EXACT_METRICS
            },
            "category_seed_shares": {
                label: _equal_seed_shares(
                    pairs_by_seed,
                    lambda pair, leg=leg: pair["exact"][leg]["category"],
                    protocol.FULL_GAIN_CATEGORIES,
                    label,
                )
                for label in protocol.FULL_GAIN_CATEGORIES
            },
            "curvature_role_seed_shares": {
                label: _equal_seed_shares(
                    pairs_by_seed,
                    lambda pair, leg=leg: pair["exact"][leg][
                        "curvature_role"
                    ],
                    protocol.CURVATURE_ROLE_CATEGORIES,
                    label,
                )
                for label in protocol.CURVATURE_ROLE_CATEGORIES
            },
            "cross_label_seed_shares": {
                label: _equal_seed_shares(
                    pairs_by_seed,
                    lambda pair, leg=leg: pair["exact"][leg]["cross_label"],
                    protocol.CROSS_LABELS,
                    label,
                )
                for label in protocol.CROSS_LABELS
            },
        }
    return {
        "seed_count": len(pairs_by_seed),
        "seed_order": list(pairs_by_seed),
        "seed_pair_counts": {
            str(seed): len(pairs) for seed, pairs in pairs_by_seed.items()
        },
        "legs": legs,
        "sequential": {
            "copy_mutation_interaction2_seed_means": _equal_seed_means(
                pairs_by_seed, _pair_interaction
            ),
            "copy_full_gain_sign_transition_seed_shares": {
                label: _equal_seed_shares(
                    pairs_by_seed,
                    lambda pair: pair["exact"][
                        "copy_full_gain_sign_transition"
                    ],
                    SIGN_TRANSITIONS,
                    label,
                )
                for label in SIGN_TRANSITIONS
            },
            "copy_full_category_transition_seed_shares": {
                label: _equal_seed_shares(
                    pairs_by_seed,
                    _category_transition,
                    CATEGORY_TRANSITIONS,
                    label,
                )
                for label in CATEGORY_TRANSITIONS
            },
            "mutation_failure_source_seed_shares": {
                label: _equal_seed_shares(
                    pairs_by_seed,
                    lambda pair: pair["exact"]["mutation_failure_source"],
                    protocol.MUTATION_FAILURE_LABELS,
                    label,
                )
                for label in protocol.MUTATION_FAILURE_LABELS
            },
        },
        "work_seed_means": {
            metric: _equal_seed_means(
                pairs_by_seed,
                lambda pair, metric=metric: Fraction(
                    pair["work"][metric], 1
                ),
            )
            for metric in WORK_METRICS
        },
        "wallclock_sec_diagnostic_only_seed_means": (
            _equal_seed_float_means(
                pairs_by_seed,
                lambda pair: pair["elapsed_sec_diagnostic_only"],
            )
        ),
    }


def _phase_groups(phase: str) -> tuple[str, ...]:
    if phase == "initial":
        return ("initial",)
    if phase == "primary":
        return tuple(protocol.PRIMARY_STATE_GROUPS)
    raise RuntimeError("未知 evaluator phase")


def _build_independent_strata(
    compact_pairs: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> dict[str, Any]:
    index: dict[tuple[str, int, str], list[Mapping[str, Any]]] = {}
    for pair in compact_pairs:
        key = (pair["dataset"], pair["seed"], pair["state_group"])
        index.setdefault(key, []).append(pair)
    dataset_seed_state = []
    dataset_seed_phase = []
    dataset_phase_equal_seed = []
    for dataset in protocol.DATASET_ORDER:
        for seed in seeds:
            for group in protocol.STATE_GROUPS:
                pairs = index.get((dataset, seed, group), [])
                expected = protocol.proposals_per_state(
                    dataset,
                    mode=("formal" if seed in protocol.FORMAL_SEEDS else "smoke"),
                )
                if len(pairs) != expected:
                    raise RuntimeError("independent strata pair coverage 失败")
                dataset_seed_state.append({
                    "dataset": dataset,
                    "seed": seed,
                    "state_group": group,
                    "phase": "initial" if group == "initial" else "primary",
                    "summary": _summarize_pairs(pairs),
                })
            for phase in PHASES:
                groups = _phase_groups(phase)
                pairs = [
                    pair
                    for group in groups
                    for pair in index[(dataset, seed, group)]
                ]
                dataset_seed_phase.append({
                    "dataset": dataset,
                    "seed": seed,
                    "phase": phase,
                    "state_groups": list(groups),
                    "summary": _summarize_pairs(pairs),
                })
        for phase in PHASES:
            groups = _phase_groups(phase)
            pairs_by_seed = {
                seed: [
                    pair
                    for group in groups
                    for pair in index[(dataset, seed, group)]
                ]
                for seed in seeds
            }
            pooled = [
                pair for rows in pairs_by_seed.values() for pair in rows
            ]
            dataset_phase_equal_seed.append({
                "dataset": dataset,
                "phase": phase,
                "state_groups": list(groups),
                "pooled_proposals_descriptive_only": _summarize_pairs(pooled),
                "equal_seed_aggregate": _equal_seed_aggregate(pairs_by_seed),
            })
    return {
        "dataset_seed_state": dataset_seed_state,
        "dataset_seed_phase": dataset_seed_phase,
        "dataset_phase_equal_seed": dataset_phase_equal_seed,
    }


def _axis_counts_by_seed(
    compact_pairs: Sequence[Mapping[str, Any]],
    dataset: str,
    labels: Sequence[str],
    extractor: Callable[[Mapping[str, Any]], str | None],
) -> dict[int, dict[str, int]]:
    result = {}
    for seed in protocol.FORMAL_SEEDS:
        rows = [
            pair
            for pair in compact_pairs
            if pair["dataset"] == dataset
            and pair["seed"] == seed
            and pair["state_group"] in protocol.PRIMARY_STATE_GROUPS
        ]
        result[seed] = _fixed_counts(
            [extractor(pair) for pair in rows], labels
        )
    return result


def _independent_dataset_dominance(
    counts_by_seed: Mapping[int, Mapping[str, int]],
    labels: Sequence[str],
) -> dict[str, Any]:
    labels = tuple(labels)
    if (
        len(labels) != 2
        or labels[0] == labels[1]
        or set(counts_by_seed) != set(protocol.FORMAL_SEEDS)
    ):
        raise RuntimeError("independent dominance axis/seed coverage 失败")
    denominators = {}
    seed_labels = {}
    shares: dict[int, dict[str, Fraction]] = {}
    support_counts = {label: 0 for label in labels}
    for seed in protocol.FORMAL_SEEDS:
        row = counts_by_seed[seed]
        if set(row) != set(labels) or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in row.values()
        ):
            raise RuntimeError("independent dominance counts 失败")
        denominator = sum(row.values())
        denominators[seed] = denominator
        shares[seed] = {
            label: (
                Fraction(row[label], denominator)
                if denominator
                else Fraction(0, 1)
            )
            for label in labels
        }
        if row[labels[0]] > row[labels[1]]:
            seed_label = labels[0]
        elif row[labels[1]] > row[labels[0]]:
            seed_label = labels[1]
        else:
            seed_label = None
        seed_labels[seed] = seed_label
        if seed_label is not None:
            support_counts[seed_label] += 1
    insufficient = [
        seed
        for seed in protocol.FORMAL_SEEDS
        if denominators[seed] < protocol.MIN_FAILURES_PER_SEED
    ]
    mean_shares = {
        label: sum(
            (shares[seed][label] for seed in protocol.FORMAL_SEEDS),
            Fraction(0, 1),
        )
        / len(protocol.FORMAL_SEEDS)
        for label in labels
    }
    selected = None
    if not insufficient:
        candidates = [
            label
            for label in labels
            if support_counts[label] >= protocol.REQUIRED_SEED_MAJORITY
            and mean_shares[label] > Fraction(1, 2)
        ]
        if len(candidates) > 1:
            raise RuntimeError("互斥 dominance axis 同时产生两个多数")
        selected = candidates[0] if candidates else None
    return {
        "status": (
            "insufficient_support"
            if insufficient
            else "supported" if selected is not None else "no_stable_label"
        ),
        "label": selected,
        "minimum_per_seed": protocol.MIN_FAILURES_PER_SEED,
        "insufficient_seeds": insufficient,
        "denominators": denominators,
        "seed_labels": seed_labels,
        "support_counts": support_counts,
        "mean_shares": {
            label: {
                "numerator": value.numerator,
                "denominator": value.denominator,
                "value": float(value),
            }
            for label, value in mean_shares.items()
        },
    }


def _dominance_with_counts(
    counts: Mapping[int, Mapping[str, int]], labels: Sequence[str]
) -> dict[str, Any]:
    return {
        "axis_labels": list(labels),
        "counts_by_seed": {
            str(seed): dict(counts[seed]) for seed in protocol.FORMAL_SEEDS
        },
        **_independent_dataset_dominance(counts, labels),
    }


def _independent_shared_geometry(
    results: Mapping[str, Mapping[str, Any]],
) -> str:
    if set(results) != set(protocol.DATASET_ORDER):
        raise RuntimeError("shared geometry dataset coverage 失败")
    rows = [results[dataset] for dataset in protocol.DATASET_ORDER]
    if any(row["status"] == "insufficient_support" for row in rows):
        return "insufficient_negative_support"
    labels = [
        row["label"] if row["status"] == "supported" else None
        for row in rows
    ]
    if labels == ["direction_failure", "direction_failure"]:
        return "shared_direction_failure"
    if labels == ["curvature_overrun", "curvature_overrun"]:
        return "shared_curvature_overrun"
    return "no_shared_failure_geometry"


def _shared_secondary(
    results: Mapping[str, Mapping[str, Any]],
    *,
    applicable: bool,
    reason: str,
) -> dict[str, Any]:
    if not applicable:
        return {
            "status": "not_applicable",
            "reason": reason,
            "shared_label": None,
        }
    rows = [results[dataset] for dataset in protocol.DATASET_ORDER]
    if any(row["status"] == "insufficient_support" for row in rows):
        return {
            "status": "insufficient_support",
            "reason": None,
            "shared_label": None,
        }
    labels = [
        row["label"] if row["status"] == "supported" else None
        for row in rows
    ]
    if labels[0] is not None and labels[0] == labels[1]:
        return {
            "status": "supported",
            "reason": None,
            "shared_label": labels[0],
        }
    return {
        "status": "no_shared_label",
        "reason": None,
        "shared_label": None,
    }


def _build_independent_formal_dominance(
    compact_pairs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], str]:
    dataset_results = {}
    geometry_results = {}
    cross_results = {}
    mutation_results = {}
    for dataset in protocol.DATASET_ORDER:
        geometry_counts = _axis_counts_by_seed(
            compact_pairs,
            dataset,
            protocol.GEOMETRY_LABELS,
            lambda pair: (
                pair["exact"]["full"]["category"]
                if pair["exact"]["full"]["category"]
                in protocol.GEOMETRY_LABELS
                else None
            ),
        )
        cross_counts = _axis_counts_by_seed(
            compact_pairs,
            dataset,
            protocol.CROSS_LABELS,
            lambda pair: pair["exact"]["full"]["cross_label"],
        )
        mutation_counts = _axis_counts_by_seed(
            compact_pairs,
            dataset,
            protocol.MUTATION_FAILURE_LABELS,
            lambda pair: pair["exact"]["mutation_failure_source"],
        )
        for seed in protocol.FORMAL_SEEDS:
            if sum(geometry_counts[seed].values()) != sum(
                mutation_counts[seed].values()
            ):
                raise RuntimeError("negative/mutation denominator 不一致")
            if sum(cross_counts[seed].values()) != geometry_counts[seed][
                "curvature_overrun"
            ]:
                raise RuntimeError("curvature/cross denominator 不一致")
        geometry = _dominance_with_counts(
            geometry_counts, protocol.GEOMETRY_LABELS
        )
        mutation = _dominance_with_counts(
            mutation_counts, protocol.MUTATION_FAILURE_LABELS
        )
        if geometry["status"] == "supported" and geometry["label"] == (
            "curvature_overrun"
        ):
            cross = _dominance_with_counts(
                cross_counts, protocol.CROSS_LABELS
            )
        else:
            cross = {
                "axis_labels": list(protocol.CROSS_LABELS),
                "counts_by_seed": {
                    str(seed): dict(cross_counts[seed])
                    for seed in protocol.FORMAL_SEEDS
                },
                "status": "not_applicable_without_dataset_curvature_support",
                "label": None,
                "minimum_per_seed": protocol.MIN_FAILURES_PER_SEED,
            }
        geometry_results[dataset] = geometry
        cross_results[dataset] = cross
        mutation_results[dataset] = mutation
        dataset_results[dataset] = {
            "geometry": geometry,
            "curvature_self_cross": cross,
            "mutation_failure_source": mutation,
        }
    overall = _independent_shared_geometry(geometry_results)
    result = {
        "primary_state_groups": list(protocol.PRIMARY_STATE_GROUPS),
        "initial_included": False,
        "minimum_failures_per_seed": protocol.MIN_FAILURES_PER_SEED,
        "required_same_seed_majorities": protocol.REQUIRED_SEED_MAJORITY,
        "dataset_rule": (
            "all five seed denominators >= 10; at least 4/5 strict seed "
            "majorities agree; that label's equal-seed mean share > 0.5"
        ),
        "dataset_results": dataset_results,
        "shared_geometry": overall,
        "shared_curvature_self_cross": _shared_secondary(
            cross_results,
            applicable=overall == "shared_curvature_overrun",
            reason="overall geometry is not shared_curvature_overrun",
        ),
        "shared_mutation_failure_source": _shared_secondary(
            mutation_results,
            applicable=True,
            reason="",
        ),
    }
    return _json_safe(result), overall


def _validate_artifact_chain(
    library: Mapping[str, Any],
    proposal: Mapping[str, Any],
    structural: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    *,
    mode: str,
    state_file_sha256: str,
    proposal_file_sha256: str,
    structural_file_sha256: str,
) -> None:
    if not (
        library.get("input_audit")
        == proposal.get("input_audit")
        == structural.get("input_audit")
        == evaluation.get("input_audit")
        and library.get("runtime_targets")
        == proposal.get("runtime_targets")
        == structural.get("runtime_targets")
        == evaluation.get("runtime_targets")
    ):
        raise RuntimeError("四 artifacts measured inputs/runtime targets 漂移")
    structural_identity = structural.get("artifact_identity", {})
    expected_structural_identity = {
        "state_library_file_sha256": state_file_sha256,
        "state_library_scientific_sha256": library.get(
            "state_library_scientific_sha256"
        ),
        "state_library_git_commit": library.get("git", {}).get("commit"),
        "state_seed_shard_file_sha256": library.get("manifest", {}).get(
            "source_seed_shard_sha256"
        ),
        "proposal_collection_file_sha256": proposal_file_sha256,
        "proposal_collection_scientific_sha256": proposal.get(
            "proposal_scientific_sha256"
        ),
        "proposal_collection_git_commit": proposal.get("git", {}).get(
            "commit"
        ),
        "proposal_seed_shard_file_sha256": proposal.get("manifest", {}).get(
            "source_proposal_seed_shard_sha256"
        ),
    }
    if structural_identity != expected_structural_identity:
        raise RuntimeError("structural audit 未精确绑定 state/proposal artifacts")
    evaluation_identity = evaluation.get("artifact_identity", {})
    expected_evaluation_identity = {
        "proposal_collection_file_sha256": proposal_file_sha256,
        "proposal_collection_scientific_sha256": proposal.get(
            "proposal_scientific_sha256"
        ),
        "proposal_collection_git_commit": proposal.get("git", {}).get(
            "commit"
        ),
        "structural_audit_file_sha256": structural_file_sha256,
        "structural_audit_scientific_sha256": structural.get(
            "structural_audit_scientific_sha256"
        ),
        "structural_audit_git_commit": structural.get("audit_git_commit"),
        "structural_audit_bound_artifact_identity": structural_identity,
    }
    if evaluation_identity != expected_evaluation_identity:
        raise RuntimeError("evaluation 未精确绑定 proposal/structural artifacts")
    if (
        structural.get("audit_git_commit")
        != structural.get("git", {}).get("commit")
        or evaluation.get("evaluator_git_commit")
        != evaluation.get("git", {}).get("commit")
    ):
        raise RuntimeError("audit/evaluator git identity 失败")

    proposal_states = proposal["states"]
    structural_states = structural["state_audits"]
    for proposal_state, structural_state in zip(
        proposal_states, structural_states
    ):
        if any(
            proposal_state.get(key) != structural_state.get(key)
            for key in (
                "state_id",
                "source_state_scientific_sha256",
                "source_trajectory_scientific_sha256",
                "current_table_sha256",
            )
        ):
            raise RuntimeError("structural state rows 未绑定 proposal states")
    if mode == "formal":
        commits = {
            library.get("git", {}).get("commit"),
            proposal.get("git", {}).get("commit"),
            structural.get("git", {}).get("commit"),
            evaluation.get("git", {}).get("commit"),
        }
        if len(commits) != 1 or any(
            artifact.get("git", {}).get("worktree_clean_including_untracked")
            is not True
            for artifact in (library, proposal, structural, evaluation)
        ):
            raise RuntimeError("formal 四 artifacts 必须来自同一 clean commit")


def _state_arithmetic_scientific_payload(
    row: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "elapsed_sec_diagnostic_only"
    }


def scientific_payload(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arithmetic_audit_format": report["arithmetic_audit_format"],
        "mode": report["mode"],
        "formal_result_valid": report["formal_result_valid"],
        "mechanism_evidence_validated": report[
            "mechanism_evidence_validated"
        ],
        "protocol_sha256": report["protocol_sha256"],
        "audit_git_commit": report["audit_git_commit"],
        "input_audit": report["input_audit"],
        "runtime_targets": report["runtime_targets"],
        "artifact_identity": report["artifact_identity"],
        "audit_boundary": report["audit_boundary"],
        "checks": report["checks"],
        "state_audits": [
            _state_arithmetic_scientific_payload(row)
            for row in report["state_audits"]
        ],
        "independent_recalculation": report[
            "independent_recalculation"
        ],
        "manifest": report["manifest"],
    }


def _validate_arithmetic_report(
    report: Mapping[str, Any], *, mode: str, seeds: Sequence[int]
) -> None:
    required_top_keys = {
        "arithmetic_audit_format",
        "status",
        "mode",
        "formal_result_valid",
        "mechanism_evidence_validated",
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
        "independent_recalculation",
        "manifest",
        "elapsed_sec_diagnostic_only",
        "arithmetic_audit_scientific_sha256",
    }
    required_state_keys = {
        "state_id",
        "source_state_scientific_sha256",
        "source_trajectory_scientific_sha256",
        "current_table_sha256",
        "independent_current_query_answers_sha256",
        "independent_residual_numerators_sha256",
        "pair_count",
        "pair_ids_sha256",
        "pair_arithmetic_audits_sha256",
        "all_pairs_arithmetic_valid",
        "elapsed_sec_diagnostic_only",
    }
    required_artifact_keys = {
        "state_library_file_sha256",
        "state_library_scientific_sha256",
        "state_library_git_commit",
        "proposal_collection_file_sha256",
        "proposal_collection_scientific_sha256",
        "proposal_collection_git_commit",
        "structural_audit_file_sha256",
        "structural_audit_scientific_sha256",
        "structural_audit_git_commit",
        "evaluation_file_sha256",
        "evaluation_scientific_sha256",
        "evaluation_git_commit",
    }
    expected_states = _expected_state_ids(seeds, mode=mode)
    expected_pairs = _expected_pair_ids(seeds, mode=mode)
    rows = report.get("state_audits", [])
    manifest = report.get("manifest", {})
    independent = report.get("independent_recalculation", {})
    artifact = report.get("artifact_identity", {})
    if (
        set(report) != required_top_keys
        or report.get("arithmetic_audit_format") != ARITHMETIC_AUDIT_FORMAT
        or report.get("status") != "complete"
        or report.get("mode") != mode
        or report.get("formal_result_valid") is not (mode == "formal")
        or report.get("mechanism_evidence_validated") is not (
            mode == "formal"
        )
        or report.get("protocol") != protocol.frozen_protocol_manifest()
        or report.get("protocol_sha256") != protocol.FROZEN_PROTOCOL_SHA256
        or report.get("audit_boundary") != ARITHMETIC_AUDIT_BOUNDARY
        or report.get("checks") != PASS_CHECKS
        or report.get("audit_git_commit")
        != report.get("git", {}).get("commit")
        or [row.get("state_id") for row in rows] != expected_states
        or any(
            set(row) != required_state_keys
            or row.get("all_pairs_arithmetic_valid") is not True
            or any(
                not _is_sha256(row.get(key))
                for key in (
                    "source_state_scientific_sha256",
                    "source_trajectory_scientific_sha256",
                    "current_table_sha256",
                    "independent_current_query_answers_sha256",
                    "independent_residual_numerators_sha256",
                    "pair_ids_sha256",
                    "pair_arithmetic_audits_sha256",
                )
            )
            for row in rows
        )
        or not isinstance(artifact, Mapping)
        or set(artifact) != required_artifact_keys
        or any(
            not _is_sha256(artifact.get(key))
            for key in required_artifact_keys
            if key.endswith("_sha256")
        )
        or manifest.get("dataset_order") != list(protocol.DATASET_ORDER)
        or manifest.get("seed_order") != list(seeds)
        or manifest.get("state_group_order")
        != list(protocol.STATE_GROUPS)
        or manifest.get("state_count") != len(expected_states)
        or manifest.get("pair_count") != len(expected_pairs)
        or manifest.get("state_ids_in_fixed_order") != expected_states
        or manifest.get("pair_ids_in_fixed_order_sha256")
        != _canonical_sha256(expected_pairs)
        or manifest.get("state_arithmetic_audits_sha256")
        != _canonical_sha256([
            _state_arithmetic_scientific_payload(row) for row in rows
        ])
        or sum(row.get("pair_count", 0) for row in rows)
        != len(expected_pairs)
        or independent.get("strata_match") is not True
        or independent.get("strata_sha256")
        != independent.get("evaluator_strata_sha256")
        or independent.get("formal_dominance_match") is not True
        or _canonical_sha256(independent.get("formal_dominance"))
        != independent.get("evaluator_formal_dominance_sha256")
        or independent.get("formal_overall_result_match") is not True
        or independent.get("formal_overall_result")
        != independent.get("evaluator_formal_overall_result")
    ):
        raise RuntimeError("independent arithmetic audit report 结构/结果失败")
    if mode == "formal":
        if independent.get("formal_overall_result") not in (
            protocol.ALLOWED_OVERALL_RESULTS
        ):
            raise RuntimeError("independent formal overall result 无效")
    elif (
        independent.get("formal_dominance") is not None
        or independent.get("formal_overall_result") is not None
    ):
        raise RuntimeError("smoke arithmetic audit 不得验证机制结论")
    if report.get("arithmetic_audit_scientific_sha256") != _canonical_sha256(
        scientific_payload(report)
    ):
        raise RuntimeError("arithmetic audit scientific SHA-256 失败")


def build_plan(mode: str) -> dict[str, Any]:
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    seeds = _mode_seeds(mode)
    pair_count = len(_expected_pair_ids(seeds, mode=mode))
    return {
        "mode": "plan_only_no_input_read_no_generation_no_arithmetic_audit",
        "requested_mode": mode,
        "formal_result_valid": False,
        "mechanism_evidence_validated": False,
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "arithmetic_audit_format": ARITHMETIC_AUDIT_FORMAT,
        "required_inputs": [
            "complete_state_library",
            "complete_proposal_collection",
            "passing_structural_audit",
            "complete_frozen_evaluation",
        ],
        "dataset_order": list(protocol.DATASET_ORDER),
        "seed_order": list(seeds),
        "state_group_order": list(protocol.STATE_GROUPS),
        "state_count": len(_expected_state_ids(seeds, mode=mode)),
        "pair_count": pair_count,
        "audit_boundary": dict(ARITHMETIC_AUDIT_BOUNDARY),
        "input_read_started": False,
        "generation_started": False,
        "formal_confirmation_consumed": False,
    }


def audit_arithmetic(
    mode: str,
    state_library_path: str | Path,
    proposal_collection_path: str | Path,
    structural_audit_path: str | Path,
    evaluation_path: str | Path,
    output_path: str | Path,
    *,
    confirmed_state_library_sha256: str,
    confirmed_proposal_collection_sha256: str,
    confirmed_structural_audit_sha256: str,
    confirmed_evaluation_sha256: str,
    confirmed_protocol_sha256: str | None = None,
    confirmed_execution_commit: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    protocol.require_formal_confirmation(mode, confirmed_protocol_sha256)
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(
            f"arithmetic audit 输出已存在，不覆盖：{output}"
        )
    paths = {
        "state_library": Path(state_library_path).resolve(),
        "proposal_collection": Path(proposal_collection_path).resolve(),
        "structural_audit": Path(structural_audit_path).resolve(),
        "evaluation": Path(evaluation_path).resolve(),
    }
    confirmations = {
        "state_library": confirmed_state_library_sha256,
        "proposal_collection": confirmed_proposal_collection_sha256,
        "structural_audit": confirmed_structural_audit_sha256,
        "evaluation": confirmed_evaluation_sha256,
    }
    file_hashes = {name: _file_sha256(path) for name, path in paths.items()}
    for name in paths:
        if confirmations[name] != file_hashes[name]:
            raise PermissionError(f"{name} 显式确认 SHA-256 不匹配")

    library = _strict_load_json(paths["state_library"])
    proposal = _strict_load_json(paths["proposal_collection"])
    structural = _strict_load_json(paths["structural_audit"])
    evaluation = _strict_load_json(paths["evaluation"])
    seeds = _mode_seeds(mode)
    _validate_library_envelope(library, mode=mode, seeds=seeds)
    _validate_proposal_envelope(proposal, mode=mode, seeds=seeds)
    _validate_structural_envelope(structural, mode=mode, seeds=seeds)
    _validate_evaluation_envelope(evaluation, mode=mode, seeds=seeds)
    _validate_artifact_chain(
        library,
        proposal,
        structural,
        evaluation,
        mode=mode,
        state_file_sha256=file_hashes["state_library"],
        proposal_file_sha256=file_hashes["proposal_collection"],
        structural_file_sha256=file_hashes["structural_audit"],
    )
    if any(_file_sha256(paths[name]) != file_hashes[name] for name in paths):
        raise RuntimeError("arithmetic audit 输入在读取/绑定期间改变")

    git = _git_identity(REPOSITORY_ROOT)
    environment = _validate_execution(
        mode, git, confirmed_execution_commit
    )
    if mode == "formal" and any(
        artifact.get("git", {}).get("commit") != git["commit"]
        for artifact in (library, proposal, structural, evaluation)
    ):
        raise RuntimeError("formal artifacts/current arithmetic auditor commit 漂移")

    measured_inputs = _load_measured_queries()
    for dataset in protocol.DATASET_ORDER:
        input_row = library.get("input_audit", {}).get(dataset, {})
        if (
            input_row.get("sha256")
            != measured_inputs[dataset]["observed_input_sha256"]
            or input_row.get("query_count")
            != protocol.DATASETS[dataset]["query_count"]
            or {
                int(order): count
                for order, count in input_row.get("order_counts", {}).items()
            }
            != protocol.DATASETS[dataset]["query_order_counts"]
            or input_row.get("target_vector_sha256")
            != protocol.DATASETS[dataset]["target_vector_sha256"]
            or input_row.get("query_identity_sha256")
            != protocol.DATASETS[dataset]["query_identity_sha256"]
        ):
            raise RuntimeError(f"{dataset} input audit 与 measured files 不一致")
        runtime_row = library["runtime_targets"][dataset]
        runtime_n = protocol.source_generator_params(
            dataset, seeds[0], mode=mode
        )["n_records"]
        expected_target = (
            measured_inputs[dataset]["source_target"].astype(float)
            * (
                runtime_n
                / measured_inputs[dataset]["source_n_records"]
            )
        ).tolist()
        if (
            runtime_row.get("runtime_n_records") != runtime_n
            or runtime_row.get("target_values") != expected_target
            or runtime_row.get("target_vector_sha256")
            != _canonical_sha256(expected_target)
            or runtime_row.get("source_target_vector_sha256")
            != protocol.DATASETS[dataset]["target_vector_sha256"]
        ):
            raise RuntimeError(f"{dataset} runtime target 独立复算失败")

    state_index = {row["state_id"]: row for row in library["states"]}
    trajectory_index = {
        (row["dataset"], row["seed"]): row
        for row in library["trajectories"]
    }
    proposal_index = {
        row["state_id"]: row for row in proposal["states"]
    }
    state_audits = []
    compact_pairs: list[dict[str, Any]] = []
    started = time.perf_counter()
    for state_id in _expected_state_ids(seeds, mode=mode):
        state = state_index[state_id]
        trajectory = trajectory_index[(state["dataset"], state["seed"])]
        state_audit, pairs = _audit_state_arithmetic(
            state,
            trajectory,
            proposal_index[state_id],
            mode=mode,
            measured=measured_inputs[state["dataset"]],
        )
        state_audits.append(_json_safe(state_audit))
        compact_pairs.extend(pairs)
        print(
            f"[Stage6A arithmetic {mode} {state_id}] "
            f"pairs={len(pairs)} "
            f"elapsed={state_audit['elapsed_sec_diagnostic_only']:.2f}s",
            flush=True,
        )

    independent_strata = _json_safe(
        _build_independent_strata(compact_pairs, seeds)
    )
    if independent_strata != evaluation["strata"]:
        raise RuntimeError("independent strata 与 frozen evaluator 不一致")
    if mode == "formal":
        independent_dominance, independent_overall = (
            _build_independent_formal_dominance(compact_pairs)
        )
    else:
        independent_dominance = None
        independent_overall = None
    if independent_dominance != evaluation.get("formal_dominance"):
        raise RuntimeError("independent dominance 与 frozen evaluator 不一致")
    if independent_overall != evaluation.get("formal_overall_result"):
        raise RuntimeError("independent shared result 与 frozen evaluator 不一致")

    if any(_file_sha256(paths[name]) != file_hashes[name] for name in paths):
        raise RuntimeError("arithmetic audit 输入在逐 pair 复算期间改变")
    if mode == "formal" and _git_identity(REPOSITORY_ROOT) != git:
        raise RuntimeError("formal arithmetic audit 期间 git identity 改变")

    artifact_identity = {
        "state_library_file_sha256": file_hashes["state_library"],
        "state_library_scientific_sha256": library[
            "state_library_scientific_sha256"
        ],
        "state_library_git_commit": library.get("git", {}).get("commit"),
        "proposal_collection_file_sha256": file_hashes[
            "proposal_collection"
        ],
        "proposal_collection_scientific_sha256": proposal[
            "proposal_scientific_sha256"
        ],
        "proposal_collection_git_commit": proposal.get("git", {}).get(
            "commit"
        ),
        "structural_audit_file_sha256": file_hashes["structural_audit"],
        "structural_audit_scientific_sha256": structural[
            "structural_audit_scientific_sha256"
        ],
        "structural_audit_git_commit": structural["audit_git_commit"],
        "evaluation_file_sha256": file_hashes["evaluation"],
        "evaluation_scientific_sha256": evaluation[
            "evaluation_scientific_sha256"
        ],
        "evaluation_git_commit": evaluation["evaluator_git_commit"],
    }
    independent_recalculation = {
        "strata_sha256": _canonical_sha256(independent_strata),
        "evaluator_strata_sha256": _canonical_sha256(evaluation["strata"]),
        "strata_match": True,
        "formal_dominance": independent_dominance,
        "evaluator_formal_dominance_sha256": _canonical_sha256(
            evaluation.get("formal_dominance")
        ),
        "formal_dominance_match": True,
        "formal_overall_result": independent_overall,
        "evaluator_formal_overall_result": evaluation.get(
            "formal_overall_result"
        ),
        "formal_overall_result_match": True,
    }
    expected_pairs = _expected_pair_ids(seeds, mode=mode)
    report = _json_safe({
        "arithmetic_audit_format": ARITHMETIC_AUDIT_FORMAT,
        "status": "complete",
        "mode": mode,
        "formal_result_valid": mode == "formal",
        "mechanism_evidence_validated": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "audit_git_commit": git["commit"],
        "git": git,
        "environment": environment,
        "input_audit": library["input_audit"],
        "runtime_targets": library["runtime_targets"],
        "artifact_paths_diagnostic_only": {
            name: str(path) for name, path in paths.items()
        },
        "artifact_identity": artifact_identity,
        "audit_boundary": dict(ARITHMETIC_AUDIT_BOUNDARY),
        "checks": dict(PASS_CHECKS),
        "state_audits": state_audits,
        "independent_recalculation": independent_recalculation,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "primary_state_group_order": list(
                protocol.PRIMARY_STATE_GROUPS
            ),
            "state_count": len(state_audits),
            "pair_count": len(compact_pairs),
            "state_ids_in_fixed_order": _expected_state_ids(
                seeds, mode=mode
            ),
            "pair_ids_in_fixed_order_sha256": _canonical_sha256(
                expected_pairs
            ),
            "state_arithmetic_audits_sha256": _canonical_sha256([
                _state_arithmetic_scientific_payload(row)
                for row in state_audits
            ]),
        },
        "elapsed_sec_diagnostic_only": time.perf_counter() - started,
    })
    report["arithmetic_audit_scientific_sha256"] = _canonical_sha256(
        scientific_payload(report)
    )
    _validate_arithmetic_report(report, mode=mode, seeds=seeds)
    if any(_file_sha256(paths[name]) != file_hashes[name] for name in paths):
        raise RuntimeError("arithmetic audit 输入在报告发布前改变")
    if mode == "formal" and _git_identity(REPOSITORY_ROOT) != git:
        raise RuntimeError("formal arithmetic audit 在报告发布前 git 漂移")
    published = _exclusive_write_json(output, report)
    print(f"Stage 6A independent arithmetic audit：{published}", flush=True)
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
    audit_parser.add_argument("--structural-audit", required=True)
    audit_parser.add_argument("--evaluation", required=True)
    audit_parser.add_argument("--output", required=True)
    audit_parser.add_argument(
        "--confirmed-state-library-sha256", required=True
    )
    audit_parser.add_argument(
        "--confirmed-proposal-collection-sha256", required=True
    )
    audit_parser.add_argument(
        "--confirmed-structural-audit-sha256", required=True
    )
    audit_parser.add_argument("--confirmed-evaluation-sha256", required=True)
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
    audit_arithmetic(
        args.mode,
        args.state_library,
        args.proposal_collection,
        args.structural_audit,
        args.evaluation,
        args.output,
        confirmed_state_library_sha256=args.confirmed_state_library_sha256,
        confirmed_proposal_collection_sha256=(
            args.confirmed_proposal_collection_sha256
        ),
        confirmed_structural_audit_sha256=(
            args.confirmed_structural_audit_sha256
        ),
        confirmed_evaluation_sha256=args.confirmed_evaluation_sha256,
        confirmed_protocol_sha256=args.confirmed_protocol_sha256,
        confirmed_execution_commit=args.confirmed_execution_commit,
    )


if __name__ == "__main__":
    main()
