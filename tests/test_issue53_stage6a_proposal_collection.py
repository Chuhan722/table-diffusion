from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import collect_issue53_stage6a_proposals as collector
from scripts import issue53_stage6a_protocol as protocol
from table_diffevo.schema import load_schema
from table_diffevo.update import evolve_step


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _toy_table(n_records: int = 64) -> pd.DataFrame:
    index = np.arange(n_records)
    return pd.DataFrame({
        "age": 18 + index % 83,
        "education": np.asarray(
            ["high_school", "vocational", "bachelor", "postgraduate"]
        )[index % 4],
        "employment": np.asarray([
            "unemployed",
            "student",
            "employed",
            "self_employed",
            "retired",
        ])[index % 5],
        "income": np.asarray(["low", "middle", "high"])[index % 3],
        "marital": np.asarray(["single", "married", "separated"])[
            index % 3
        ],
        "children": np.asarray(["0", "1", "2_plus"])[index % 3],
        "housing": np.asarray(["rent", "mortgage", "owned"])[index % 3],
        "vehicle": np.asarray(["0", "1", "2_plus"])[index % 3],
        "health": np.asarray(["poor", "fair", "good"])[index % 3],
        "region": np.asarray(["urban", "suburban", "rural"])[index % 3],
    })


def _toy_queries() -> list[dict]:
    return [
        {
            "conditions": [
                {"attribute": "age", "operator": ">=", "value": 40},
            ]
        },
        {
            "conditions": [
                {
                    "attribute": "employment",
                    "operator": "==",
                    "value": "employed",
                },
                {
                    "attribute": "region",
                    "operator": "==",
                    "value": "urban",
                },
            ]
        },
    ]


def _zero_exact(dataset: str, *, formal: bool = False) -> dict:
    query_count = int(protocol.DATASETS[dataset]["query_count"])
    denominator = int(protocol.DATASETS[dataset]["n_records"])
    return {
        "denominator": denominator,
        "delta_q": [0] * query_count,
        "b2_numerator": 0,
        "cself2_numerator": 0,
        "ccross2_numerator": 0,
        "c2_numerator": 0,
        "g2_numerator": 0,
        "integral_doubled_units": True,
        "integral_count_residual_units": formal,
        "b2": 0,
        "cself2": 0,
        "ccross2": 0,
        "c2": 0,
        "g2": 0,
        "category": "unchanged",
        "curvature_role": None,
        "cross_label": None,
    }


def _fake_pair(
    dataset: str,
    seed: int,
    state_group: str,
    proposal_index: int,
    *,
    mode: str,
) -> dict:
    state_id = protocol.state_id(
        dataset, seed, state_group, mode=mode
    )
    exact = _zero_exact(dataset, formal=mode == "formal")
    denominator = exact["denominator"]
    current_sha = _sha(state_id)
    donor_address = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "donor",
        mode=mode,
    )
    update_address = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "update",
        mode=mode,
    )
    return {
        "pair_id": collector._pair_id(state_id, proposal_index),
        "state_id": state_id,
        "dataset": dataset,
        "seed": seed,
        "state_group": state_group,
        "proposal_index": proposal_index,
        "retained_unconditionally": True,
        "rng": {
            "donor_address_uint64": donor_address,
            "update_address_uint64": update_address,
            "donor_initial_state_sha256": collector._rng_state_sha256(
                np.random.default_rng(donor_address)
            ),
            "donor_endpoint_state_sha256": _sha("donor-endpoint"),
            "update_initial_state_sha256": collector._rng_state_sha256(
                np.random.default_rng(update_address)
            ),
            "shared_pre_mutation_state_sha256": _sha("pre-mutation"),
            "copy_only_endpoint_state_sha256": _sha("pre-mutation"),
            "full_endpoint_state_sha256": _sha("full-endpoint"),
            "mutation_rolls_sha256": _sha("mutation-rolls"),
        },
        "donor_indices_sha256": _sha("donors"),
        "direction_scores_sha256": _sha("directions"),
        "copy_probabilities_sha256": _sha("probabilities"),
        "direction_reference_scale": 0.5,
        "effective_direction_strength": 4.0,
        "participating_row_indices": [],
        "copy_row_indices_by_attribute": {},
        "copy_edits": [],
        "mutation_events": [],
        "work": {
            "participating_rows": 0,
            "copied_rows": 0,
            "copied_cells": 0,
            "mutation_rows": 0,
            "mutation_changed_cells": 0,
            "mutation_overwrote_copied_cells": 0,
            "copy_query_changed_rows": 0,
            "mutation_query_changed_rows": 0,
            "full_query_changed_rows": 0,
        },
        "table_sha256": {
            "current": current_sha,
            "copy_only": current_sha,
            "full": current_sha,
        },
        "exact": {
            "copy_only": copy.deepcopy(exact),
            "mutation_given_copy": copy.deepcopy(exact),
            "full": copy.deepcopy(exact),
            "sequential": {
                "denominator": denominator,
                "copy_mutation_interaction2_numerator": 0,
                "copy_mutation_interaction2": 0,
                "integral_doubled_units": True,
                "b_identity_verified": True,
                "c_identity_verified": True,
                "g_identity_verified": True,
            },
            "copy_full_gain_sign_transition": "zero_to_zero",
            "mutation_failure_source": None,
        },
        "elapsed_sec_diagnostic_only": 0.01,
    }


def _fake_state_result(
    dataset: str,
    seed: int,
    state_group: str,
    *,
    mode: str,
) -> dict:
    state_id = protocol.state_id(
        dataset, seed, state_group, mode=mode
    )
    return {
        "state_id": state_id,
        "dataset": dataset,
        "seed": seed,
        "state_group": state_group,
        "source_state_scientific_sha256": _sha(f"science/{state_id}"),
        "source_trajectory_scientific_sha256": _sha(
            f"trajectory/{dataset}/{seed}"
        ),
        "current_table_sha256": _sha(state_id),
        "current_query_answers_sha256": _sha(f"q/{state_id}"),
        "count_residual_numerators_sha256": _sha(f"e/{state_id}"),
        "residual_denominator": protocol.DATASETS[dataset]["n_records"],
        "residual_signal_sha256": _sha(f"signal/{state_id}"),
        "fitness_sha256": _sha(f"fitness/{state_id}"),
        "direction_reference_scale": 0.5,
        "runtime_device": protocol.source_generator_params(
            dataset, seed, mode=mode
        )["device"],
        "sampling_params": copy.deepcopy(collector.SAMPLING_PARAMS),
        "pairs": [
            _fake_pair(
                dataset,
                seed,
                state_group,
                proposal_index,
                mode=mode,
            )
            for proposal_index in range(
                protocol.proposals_per_state(dataset, mode=mode)
            )
        ],
    }


def _fake_collection(
    *,
    mode: str = "smoke",
    scope: str = "seed_shard",
    input_audit: dict | None = None,
    runtime_targets: dict | None = None,
) -> dict:
    seeds = (
        protocol.FORMAL_SEEDS
        if mode == "formal" and scope == "full"
        else (
            (protocol.FORMAL_SEEDS[0],)
            if mode == "formal"
            else (protocol.SMOKE_SEED,)
        )
    )
    states = [
        _fake_state_result(
            dataset, seed, state_group, mode=mode
        )
        for dataset in protocol.DATASET_ORDER
        for seed in seeds
        for state_group in protocol.STATE_GROUPS
    ]
    pair_ids = [
        pair["pair_id"] for state in states for pair in state["pairs"]
    ]
    collection = {
        "proposal_collection_format": (
            collector.PROPOSAL_COLLECTION_FORMAT
            if scope == "full"
            else collector.PROPOSAL_SHARD_FORMAT
        ),
        "status": "complete",
        "mode": mode,
        "artifact_scope": scope,
        "selected_seeds": list(seeds),
        "formal_result_valid": mode == "formal",
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": {
            "commit": _sha("commit")[:40],
            "worktree_clean_including_untracked": mode == "formal",
            "status": [],
        },
        "environment": {"python": "test"},
        "input_audit": input_audit or {"fixture": True},
        "runtime_targets": runtime_targets or {"fixture": True},
        "probe_boundary": copy.deepcopy(collector.PROBE_BOUNDARY),
        "source_state_artifacts_diagnostic_only": [],
        "states": states,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": list(seeds),
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(states),
            "pair_count": len(pair_ids),
            "proposal_table_result_count": 2 * len(pair_ids),
            "state_ids_in_fixed_order": collector._expected_state_ids(seeds),
            "pair_ids_in_fixed_order": pair_ids,
            "source_proposal_seed_shard_sha256": {},
        },
        "elapsed_sec_diagnostic_only": 0.2,
    }
    collection["proposal_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(collection)
    )
    return collection


def test_exact_integer_decomposition_matches_protocol():
    deltas = np.asarray([[1, 0], [1, -1]], dtype=np.int8)
    residual = np.asarray([3, 2], dtype=np.int64)
    observed = collector.exact_scaled_doubled_decomposition(
        deltas, residual, 1
    )
    expected = protocol.exact_doubled_decomposition(deltas, residual)

    assert observed["delta_q"] == [2, -1]
    for name in ("b2", "cself2", "ccross2", "c2", "g2"):
        assert observed[name] == expected[name]
        assert observed[f"{name}_numerator"] == expected[name]
    assert observed["integral_count_residual_units"] is True


def test_exact_scaled_smoke_uses_rational_numerators():
    deltas = np.asarray([[1, 0], [0, -1]], dtype=np.int8)
    residual_numerators = np.asarray([5, 4], dtype=np.int64)
    observed = collector.exact_scaled_doubled_decomposition(
        deltas, residual_numerators, 3
    )

    assert observed["b2_numerator"] == 2
    assert observed["c2_numerator"] == 6
    assert observed["g2_numerator"] == -4
    assert observed["category"] == "curvature_overrun"
    assert observed["integral_count_residual_units"] is False
    assert observed["integral_doubled_units"] is False
    assert observed["b2"] is None


def test_exact_scaled_overflow_fails_closed():
    deltas = np.ones((2, 2), dtype=np.int64)
    residual = np.full(2, np.iinfo(np.int64).max, dtype=np.int64)
    with pytest.raises(OverflowError, match="溢出"):
        collector.exact_scaled_doubled_decomposition(
            deltas, residual, 2
        )


def test_scaled_sequential_identity_keeps_signed_interaction_separate():
    residual = np.asarray([4, -2], dtype=np.int64)
    copy = collector.exact_scaled_doubled_decomposition(
        np.asarray([[1, 0], [1, 0]], dtype=np.int8),
        residual,
        1,
    )
    mutation = collector.exact_scaled_doubled_decomposition(
        np.asarray([[-1, 1]], dtype=np.int8),
        residual - np.asarray(copy["delta_q"], dtype=np.int64),
        1,
    )
    full = collector.exact_scaled_doubled_decomposition(
        np.asarray([[1, 0], [0, 1]], dtype=np.int8),
        residual,
        1,
    )
    sequential = collector._validate_scaled_sequential_identity(
        copy=copy,
        mutation_given_copy=mutation,
        full=full,
        interaction2_numerator=-4,
    )
    assert sequential["copy_mutation_interaction2"] == -4
    assert full["g2"] == copy["g2"] + mutation["g2"]
    assert full["c2"] == copy["c2"] + mutation["c2"] - 4


def test_replay_matches_production_copy_and_full():
    schema = load_schema(
        str(REPOSITORY_ROOT / "configs/test_300x10/schema.yaml")
    )
    current = _toy_table(64)
    donors = current.iloc[np.roll(np.arange(len(current)), 7)].reset_index(
        drop=True
    )
    scores = np.zeros((len(current), len(schema.attribute_names())))
    update_seed = 71823
    replay = collector.replay_paired_update(
        current,
        donors,
        schema,
        scores,
        0.0,
        update_seed,
        rho=1.0,
        eta=0.5,
        full_mu=0.5,
    )

    copy_rng = np.random.default_rng(update_seed)
    full_rng = np.random.default_rng(update_seed)
    copy_table, copy_diagnostics = evolve_step(
        current,
        donors,
        schema,
        rho=1.0,
        eta=0.5,
        mu=0.0,
        rng=copy_rng,
        copy_direction_scores=scores,
        copy_direction_strength=0.0,
        return_diagnostics=True,
    )
    full_table, full_diagnostics = evolve_step(
        current,
        donors,
        schema,
        rho=1.0,
        eta=0.5,
        mu=0.5,
        rng=full_rng,
        copy_direction_scores=scores,
        copy_direction_strength=0.0,
        return_diagnostics=True,
    )
    pd.testing.assert_frame_equal(copy_table, replay["copy_table"])
    pd.testing.assert_frame_equal(full_table, replay["full_table"])
    assert collector._rng_state_sha256(copy_rng) == replay[
        "pre_mutation_rng_sha256"
    ]
    assert collector._rng_state_sha256(full_rng) == replay[
        "endpoint_rng_sha256"
    ]
    assert copy_diagnostics["mutated_rows"] == 0
    assert full_diagnostics["mutated_rows"] == len(
        replay["mutation_events"]
    )
    assert len(replay["mutation_events"]) > 0


def test_sparse_edit_logs_reconstruct_both_tables():
    current = pd.DataFrame({"a": [0, 1], "b": ["x", "y"]})
    copy_edits = [{
        "row_index": 0,
        "donor_index": 1,
        "cells": [{"attribute": "b", "before": "x", "after": "y"}],
    }]
    mutation_events = [{
        "row_index": 0,
        "attribute_index": 0,
        "attribute": "a",
        "before_copy": 0,
        "sampled_value": 4,
        "changed": True,
        "overwrote_copied_cell": False,
    }]
    copy_table, full_table = collector.reconstruct_pair_tables(
        current, copy_edits, mutation_events
    )
    assert copy_table.to_dict(orient="records") == [
        {"a": 0, "b": "y"},
        {"a": 1, "b": "y"},
    ]
    assert full_table.to_dict(orient="records") == [
        {"a": 4, "b": "y"},
        {"a": 1, "b": "y"},
    ]


def test_micro_pair_is_measured_and_exact_without_gate():
    schema = load_schema(
        str(REPOSITORY_ROOT / "configs/test_300x10/schema.yaml")
    )
    current = _toy_table(32)
    probabilities = np.full((len(current), len(current)), 1 / len(current))
    pair = collector._generate_pair(
        dataset="test_300x10",
        seed=protocol.SMOKE_SEED,
        state_group="initial",
        proposal_index=0,
        mode="smoke",
        current=current,
        queries=_toy_queries(),
        schema=schema,
        count_residual_numerators=np.asarray([30, -11], dtype=np.int64),
        residual_denominator=300,
        residual_signal=np.asarray([0.1, -0.2]),
        sampling_probabilities=probabilities,
        device="numpy",
        direction_reference_scale=0.5,
    )
    assert pair["retained_unconditionally"] is True
    assert "accepted" not in json.dumps(pair)
    assert pair["exact"]["sequential"]["g_identity_verified"] is True
    assert pair["rng"]["shared_pre_mutation_state_sha256"] == pair[
        "rng"
    ]["copy_only_endpoint_state_sha256"]


def test_collection_structure_rejects_gate_field():
    artifact = _fake_collection()
    collector._validate_collection_structure(
        artifact,
        mode="smoke",
        artifact_scope="seed_shard",
        selected_seeds=(protocol.SMOKE_SEED,),
    )
    artifact["states"][0]["pairs"][0]["accepted"] = True
    artifact["proposal_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(artifact)
    )
    with pytest.raises(RuntimeError, match="gate/最终结论字段"):
        collector._validate_collection_structure(
            artifact,
            mode="smoke",
            artifact_scope="seed_shard",
            selected_seeds=(protocol.SMOKE_SEED,),
        )


def test_scientific_sha_excludes_all_wallclock_diagnostics():
    left = _fake_collection()
    right = copy.deepcopy(left)
    right["elapsed_sec_diagnostic_only"] = 999.0
    right["states"][0]["elapsed_sec_diagnostic_only"] = 888.0
    right["states"][0]["pairs"][0][
        "elapsed_sec_diagnostic_only"
    ] = 777.0
    assert collector.scientific_payload(left) == collector.scientific_payload(
        right
    )


def test_collection_structure_rejects_sequential_tampering():
    artifact = _fake_collection()
    artifact["states"][0]["pairs"][0]["exact"]["sequential"][
        "copy_mutation_interaction2_numerator"
    ] = 1
    artifact["proposal_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(artifact)
    )
    with pytest.raises(RuntimeError, match="identity"):
        collector._validate_collection_structure(
            artifact,
            mode="smoke",
            artifact_scope="seed_shard",
            selected_seeds=(protocol.SMOKE_SEED,),
        )


def test_smoke_shard_aggregation_is_scientifically_identical(
    tmp_path, monkeypatch
):
    input_audit = {"fixture": "inputs"}
    runtime_targets = {"fixture": "targets"}
    shard = _fake_collection(
        input_audit=input_audit,
        runtime_targets=runtime_targets,
    )
    shard_path = tmp_path / "seed_9906.json"
    collector.state_builder._exclusive_write_json(shard_path, shard)
    monkeypatch.setattr(
        collector.state_builder,
        "_audit_runtime_inputs",
        lambda root, mode: (
            copy.deepcopy(input_audit),
            {"unused": True},
            copy.deepcopy(runtime_targets),
        ),
    )
    output_path = tmp_path / "aggregate.json"
    _, aggregate = collector.aggregate_proposal_shards(
        "smoke", [shard_path], output_path
    )
    expected_full = copy.deepcopy(shard)
    expected_full["proposal_collection_format"] = (
        collector.PROPOSAL_COLLECTION_FORMAT
    )
    expected_full["artifact_scope"] = "full"
    assert aggregate["proposal_scientific_sha256"] == (
        protocol.canonical_sha256(collector.scientific_payload(expected_full))
    )
    with pytest.raises(FileExistsError):
        collector.aggregate_proposal_shards(
            "smoke", [shard_path], output_path
        )


def test_plan_and_cli_expose_no_scientific_overrides():
    plan = collector.build_plan("formal")
    assert plan["proposal_pair_count"] == 5500
    assert plan["proposal_table_result_count"] == 11000
    parser = collector._build_parser()
    collect_parser = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
    ).choices["collect"]
    option_strings = {
        option
        for action in collect_parser._actions
        for option in action.option_strings
    }
    assert {
        "--rho",
        "--eta",
        "--mu",
        "--alpha",
        "--proposals-per-state",
        "--seed",
    }.isdisjoint(option_strings)


def test_source_target_residual_is_exact_for_scaled_smoke():
    observed = collector._source_target_residual_numerators(
        [3, 7],
        runtime_n_records=128,
        source_n_records=300,
        current_query_answers=np.asarray([1, 2], dtype=np.int64),
    )
    assert observed.tolist() == [84, 296]
