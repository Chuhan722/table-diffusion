from __future__ import annotations

import ast
import copy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import audit_issue53_stage6a_arithmetic as arithmetic
from scripts import audit_issue53_stage6a_proposal_structure as structural_tool
from scripts import build_issue53_stage6a_state_library as state_builder
from scripts import collect_issue53_stage6a_proposals as collector
from scripts import evaluate_issue53_stage6a_overshoot as evaluator
from scripts import issue53_stage6a_protocol as protocol
from table_diffevo.queries import evaluate_table, load_queries
from tests import test_issue53_stage6a_evaluator as evaluator_fixtures


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return __import__("hashlib").sha256(label.encode("utf-8")).hexdigest()


def _micro_pair_fixture() -> dict:
    current = pd.DataFrame({
        "a": [1, 2, 3],
        "b": ["x", "y", "x"],
    })
    queries = [
        {
            "conditions": [
                {"attribute": "a", "operator": ">=", "value": 2}
            ]
        },
        {
            "conditions": [
                {"attribute": "b", "operator": "==", "value": "x"}
            ]
        },
        {
            "conditions": [
                {
                    "attribute": "a",
                    "operator": "between",
                    "lower": 1,
                    "upper": 2,
                },
                {"attribute": "b", "operator": "==", "value": "x"},
            ]
        },
    ]
    copy_edits = [{
        "row_index": 0,
        "donor_index": 1,
        "cells": [{
            "attribute": "a",
            "before": 1,
            "after": 2,
        }],
    }]
    mutation_events = [{
        "row_index": 1,
        "attribute_index": 1,
        "attribute": "b",
        "before_copy": "y",
        "sampled_value": "x",
        "changed": True,
        "overwrote_copied_cell": False,
    }]
    copy_table, full_table, copied_cells = arithmetic._apply_sparse_logs(
        current, copy_edits, mutation_events, ["a", "b"]
    )
    copy_rows, copy_deltas = arithmetic._row_query_deltas(
        current, copy_table, queries, ["a", "b"]
    )
    mutation_rows, mutation_deltas = arithmetic._row_query_deltas(
        copy_table, full_table, queries, ["a", "b"]
    )
    full_rows, full_deltas = arithmetic._row_query_deltas(
        current, full_table, queries, ["a", "b"]
    )
    residual = np.asarray([3, 2, 1], dtype=np.int64)
    copy_exact = arithmetic._exact_decomposition(copy_deltas, residual, 1)
    mutation_residual = residual - np.asarray(
        copy_exact["delta_q"], dtype=np.int64
    )
    mutation_exact = arithmetic._exact_decomposition(
        mutation_deltas, mutation_residual, 1
    )
    full_exact = arithmetic._exact_decomposition(full_deltas, residual, 1)
    sequential = arithmetic._sequential_payload(
        copy_exact, mutation_exact, full_exact
    )
    exact = {
        "copy_only": copy_exact,
        "mutation_given_copy": mutation_exact,
        "full": full_exact,
        "sequential": sequential,
        "copy_full_gain_sign_transition": "positive_to_positive",
        "mutation_failure_source": None,
    }
    work = {
        "participating_rows": 2,
        "copied_rows": 1,
        "copied_cells": copied_cells,
        "mutation_rows": 1,
        "mutation_changed_cells": 1,
        "mutation_overwrote_copied_cells": 0,
        "copy_query_changed_rows": len(copy_rows),
        "mutation_query_changed_rows": len(mutation_rows),
        "full_query_changed_rows": len(full_rows),
    }
    pair = {
        "pair_id": "micro__proposal_0000",
        "dataset": "micro",
        "seed": 1,
        "state_group": "initial",
        "participating_row_indices": [0, 1],
        "copy_edits": copy_edits,
        "mutation_events": mutation_events,
        "work": work,
        "table_sha256": {
            "current": arithmetic._frame_sha256(current),
            "copy_only": arithmetic._frame_sha256(copy_table),
            "full": arithmetic._frame_sha256(full_table),
        },
        "exact": exact,
        "elapsed_sec_diagnostic_only": 0.01,
    }
    return {
        "current": current,
        "queries": queries,
        "residual": residual,
        "pair": pair,
        "copy_exact": copy_exact,
        "mutation_exact": mutation_exact,
        "full_exact": full_exact,
    }


def _test_frame(n_records: int) -> pd.DataFrame:
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


def _nltcs_frame(n_records: int) -> pd.DataFrame:
    index = np.arange(n_records)
    return pd.DataFrame({
        f"attr_{attribute}": (index + attribute) % 2
        for attribute in range(1, 17)
    })


def _trajectory(
    dataset: str,
    seed: int,
    state_ids: list[str],
    runtime_target_sha256: str,
) -> dict:
    return {
        "dataset": dataset,
        "seed": seed,
        "runtime_n_records": protocol.SMOKE_N_RECORDS,
        "runtime_device": "numpy",
        "runtime_target_sha256": runtime_target_sha256,
        "source_generator_params": {"fixture": True},
        "initial_table_sha256": _sha(f"initial/{dataset}"),
        "primary_rng_post_initialization_state_sha256": _sha(
            f"rng-initial/{dataset}"
        ),
        "terminal_table_sha256": _sha(f"terminal/{dataset}"),
        "rounds_run": 1,
        "candidate_evaluations": 1,
        "termination_reason": "resource_cap_reached",
        "terminal_normalized_work": 0.0,
        "terminal_squared_loss": 0.0,
        "terminal_normalized_l1": 0.0,
        "direction_reference_scale": 0.5,
        "primary_rng_endpoint_sha256": _sha(f"rng-end/{dataset}"),
        "natural_work_snapshot_manifest": {"fixture": True},
        "selected_state_ids": state_ids,
    }


def _build_mock_state_and_proposal() -> tuple[dict, dict]:
    seed = protocol.SMOKE_SEED
    proposal = evaluator_fixtures._smoke_collection()
    input_audit, _, runtime_targets = state_builder._audit_runtime_inputs(
        REPOSITORY_ROOT, "smoke"
    )
    proposal["input_audit"] = copy.deepcopy(input_audit)
    proposal["runtime_targets"] = copy.deepcopy(runtime_targets)
    measured = arithmetic._load_measured_queries()
    frames = {
        "test_300x10": _test_frame(protocol.SMOKE_N_RECORDS),
        "nltcs": _nltcs_frame(protocol.SMOKE_N_RECORDS),
    }
    trajectories = []
    states = []
    trajectory_index = {}
    for dataset in protocol.DATASET_ORDER:
        state_ids = [
            protocol.state_id(dataset, seed, group, mode="smoke")
            for group in protocol.STATE_GROUPS
        ]
        trajectory = _trajectory(
            dataset,
            seed,
            state_ids,
            runtime_targets[dataset]["target_vector_sha256"],
        )
        trajectories.append(trajectory)
        trajectory_index[dataset] = trajectory

    proposal_states = {row["state_id"]: row for row in proposal["states"]}
    for dataset in protocol.DATASET_ORDER:
        frame = frames[dataset]
        queries = measured[dataset]["queries"]
        q = arithmetic._query_counts(frame, queries)
        source_n = measured[dataset]["source_n_records"]
        residual_numerators = arithmetic._source_residual_numerators(
            measured[dataset]["source_target"],
            len(frame),
            source_n,
            q,
        )
        runtime_target = np.asarray(
            runtime_targets[dataset]["target_values"], dtype=float
        )
        current_residual = runtime_target - q.astype(float)
        current_hash = arithmetic._frame_sha256(frame)
        for state_index, group in enumerate(protocol.STATE_GROUPS):
            state_id = protocol.state_id(
                dataset, seed, group, mode="smoke"
            )
            snapshot = {
                "snapshot_format": "natural_work_current_v1",
                "table_columns": list(frame.columns),
                "table_records": frame.to_dict(orient="records"),
                "current_query_answers": q.tolist(),
                "current_residual_signal": [0.0] * len(q),
                "current_table_sha256": current_hash,
                "primary_rng_state_sha256": _sha(f"rng/{state_id}"),
                "factorized_gibbs_rng_state_sha256": None,
                "direction_reference_scale": 0.5,
                "current_squared_loss": 0.0,
                "current_normalized_l1": 0.0,
            }
            state = {
                "state_id": state_id,
                "dataset": dataset,
                "seed": seed,
                "state_group": group,
                "target_fraction": protocol.STATE_TARGET_FRACTIONS[group],
                "target_normalized_work": 0.0,
                "selection_absolute_work_error": 0.0,
                "source_snapshot_index": state_index,
                "current_count_residual": current_residual.tolist(),
                "snapshot": snapshot,
            }
            states.append(state)
            proposal_state = proposal_states[state_id]
            proposal_state["source_state_scientific_sha256"] = (
                arithmetic._canonical_sha256(
                    arithmetic._state_scientific_payload(state)
                )
            )
            proposal_state["source_trajectory_scientific_sha256"] = (
                arithmetic._canonical_sha256(
                    arithmetic._trajectory_scientific_payload(
                        trajectory_index[dataset]
                    )
                )
            )
            proposal_state["current_table_sha256"] = current_hash
            proposal_state["current_query_answers_sha256"] = (
                arithmetic._array_sha256(q)
            )
            proposal_state["count_residual_numerators_sha256"] = (
                arithmetic._array_sha256(residual_numerators)
            )
            proposal_state["residual_denominator"] = source_n
            residual_integral = bool(
                np.all(residual_numerators % source_n == 0)
            )
            for pair in proposal_state["pairs"]:
                pair["table_sha256"] = {
                    "current": current_hash,
                    "copy_only": current_hash,
                    "full": current_hash,
                }
                for leg in arithmetic.LEG_NAMES:
                    pair["exact"][leg][
                        "integral_count_residual_units"
                    ] = residual_integral

    proposal["proposal_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(proposal)
    )
    collector._validate_collection_structure(
        proposal,
        mode="smoke",
        artifact_scope="full",
        selected_seeds=(seed,),
    )
    state_ids = arithmetic._expected_state_ids((seed,), mode="smoke")
    library = {
        "state_library_format": arithmetic.STATE_LIBRARY_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "artifact_scope": "full",
        "selected_seeds": [seed],
        "formal_result_valid": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": {
            "commit": _sha("state-library-commit")[:40],
            "worktree_clean_including_untracked": False,
            "status": ["mock"],
        },
        "environment": {"fixture": True},
        "input_audit": copy.deepcopy(input_audit),
        "runtime_targets": copy.deepcopy(runtime_targets),
        "source_boundary": {"fixture": True},
        "trajectories": trajectories,
        "states": states,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": [seed],
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(states),
            "state_ids_in_fixed_order": state_ids,
            "source_seed_shard_sha256": {},
        },
        "elapsed_sec_diagnostic_only": 0.1,
    }
    library["state_library_scientific_sha256"] = (
        arithmetic._canonical_sha256(
            arithmetic._state_library_scientific_payload(library)
        )
    )
    arithmetic._validate_library_envelope(
        library, mode="smoke", seeds=(seed,)
    )
    return library, proposal


def _write_mock_artifact_chain(tmp_path: Path) -> dict[str, Path]:
    library, proposal = _build_mock_state_and_proposal()
    state_path = tmp_path / "state_library.json"
    proposal_path = tmp_path / "proposal.json"
    state_builder._exclusive_write_json(state_path, library)
    state_builder._exclusive_write_json(proposal_path, proposal)
    state_sha = protocol.file_sha256(state_path)
    proposal_sha = protocol.file_sha256(proposal_path)
    structural = evaluator_fixtures._structural_report(
        proposal, proposal_sha
    )
    structural["artifact_identity"].update({
        "state_library_file_sha256": state_sha,
        "state_library_scientific_sha256": library[
            "state_library_scientific_sha256"
        ],
        "state_library_git_commit": library["git"]["commit"],
        "state_seed_shard_file_sha256": {},
    })
    structural["structural_audit_scientific_sha256"] = (
        protocol.canonical_sha256(
            structural_tool.scientific_payload(structural)
        )
    )
    structural_tool._validate_structural_audit_report(
        structural,
        mode="smoke",
        selected_seeds=(protocol.SMOKE_SEED,),
    )
    structural_path = tmp_path / "structural.json"
    state_builder._exclusive_write_json(structural_path, structural)
    evaluation_path = tmp_path / "evaluation.json"
    evaluator.evaluate(
        "smoke",
        proposal_path,
        structural_path,
        evaluation_path,
        confirmed_proposal_collection_sha256=proposal_sha,
        confirmed_structural_audit_sha256=protocol.file_sha256(
            structural_path
        ),
    )
    return {
        "state_library": state_path,
        "proposal_collection": proposal_path,
        "structural_audit": structural_path,
        "evaluation": evaluation_path,
    }


def test_independent_query_sparse_and_exact_arithmetic_micro_oracle():
    fixture = _micro_pair_fixture()
    compact, audit_sha = arithmetic._audit_pair_arithmetic(
        fixture["pair"],
        current=fixture["current"],
        queries=fixture["queries"],
        residual_numerators=fixture["residual"],
        denominator=1,
        columns=["a", "b"],
        formal=True,
    )
    assert arithmetic._query_counts(
        fixture["current"], fixture["queries"]
    ).tolist() == [2, 2, 1]
    assert fixture["copy_exact"]["delta_q"] == [1, 0, 0]
    assert fixture["copy_exact"]["b2"] == 6
    assert fixture["copy_exact"]["c2"] == 1
    assert fixture["mutation_exact"]["delta_q"] == [0, 1, 1]
    assert fixture["mutation_exact"]["b2"] == 6
    assert fixture["mutation_exact"]["c2"] == 2
    assert fixture["full_exact"]["delta_q"] == [1, 1, 1]
    assert fixture["full_exact"]["b2"] == 12
    assert fixture["full_exact"]["c2"] == 3
    assert compact["work"]["full_query_changed_rows"] == 2
    assert arithmetic._is_sha256(audit_sha)


def test_independent_query_evaluator_matches_measured_workload_semantics():
    frames = {
        "test_300x10": _test_frame(128),
        "nltcs": _nltcs_frame(128),
    }
    for dataset in protocol.DATASET_ORDER:
        queries = load_queries(
            str(REPOSITORY_ROOT / protocol.DATASETS[dataset]["queries"])
        )
        expected = evaluate_table(frames[dataset], queries)
        observed = arithmetic._query_counts(frames[dataset], queries)
        np.testing.assert_array_equal(observed, expected)


def test_independent_pair_audit_rejects_sparse_and_exact_tampering():
    fixture = _micro_pair_fixture()
    sparse = copy.deepcopy(fixture["pair"])
    sparse["copy_edits"][0]["cells"][0]["before"] = 99
    with pytest.raises(RuntimeError, match="copy edit"):
        arithmetic._audit_pair_arithmetic(
            sparse,
            current=fixture["current"],
            queries=fixture["queries"],
            residual_numerators=fixture["residual"],
            denominator=1,
            columns=["a", "b"],
            formal=True,
        )
    exact = copy.deepcopy(fixture["pair"])
    exact["exact"]["full"]["g2_numerator"] += 1
    with pytest.raises(RuntimeError, match="exact payload"):
        arithmetic._audit_pair_arithmetic(
            exact,
            current=fixture["current"],
            queries=fixture["queries"],
            residual_numerators=fixture["residual"],
            denominator=1,
            columns=["a", "b"],
            formal=True,
        )


def test_independent_geometry_roles_and_int64_overflow_are_exact():
    assert arithmetic._classify_gain(-1, 1) == "direction_failure"
    assert arithmetic._classify_gain(3, 4) == "curvature_overrun"
    assert arithmetic._curvature_role(1, 2, 0) == (
        "self_sufficient_overrun"
    )
    assert arithmetic._curvature_role(2, 2, 1) == "cross_breaks_tie"
    assert arithmetic._curvature_role(3, 2, 2) == (
        "cross_decisive_overrun"
    )
    with pytest.raises(OverflowError, match="int64"):
        arithmetic._exact_decomposition(
            np.asarray([[1]], dtype=np.int8),
            np.asarray([np.iinfo(np.int64).max], dtype=np.int64),
            1,
        )


def test_independent_descriptive_statistics_match_evaluator_schema():
    rows = [
        evaluator_fixtures._minimal_summary_pair(1),
        evaluator_fixtures._minimal_summary_pair(3),
    ]
    assert arithmetic._summarize_pairs(rows) == evaluator._summarize_pairs(
        rows
    )
    pairs_by_seed = {
        1: rows,
        2: [evaluator_fixtures._minimal_summary_pair(2)],
    }
    assert arithmetic._equal_seed_aggregate(
        pairs_by_seed
    ) == evaluator._equal_seed_aggregate(pairs_by_seed)


def test_independent_dominance_matches_without_protocol_classifiers(
    monkeypatch,
):
    collection = evaluator_fixtures._formal_dominance_collection()
    compact = [
        {
            **pair,
            "dataset": state["dataset"],
            "seed": state["seed"],
            "state_group": state["state_group"],
        }
        for state in collection["states"]
        for pair in state["pairs"]
    ]
    expected, expected_overall = evaluator._build_formal_dominance(
        collection
    )
    monkeypatch.setattr(
        protocol,
        "classify_dataset_dominance",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protocol classifier called")
        ),
    )
    monkeypatch.setattr(
        protocol,
        "shared_geometry_result",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("protocol shared helper called")
        ),
    )
    observed, observed_overall = (
        arithmetic._build_independent_formal_dominance(compact)
    )
    assert observed_overall == expected_overall
    assert observed == arithmetic._json_safe(expected)
    insufficient_collection = (
        evaluator_fixtures._formal_dominance_collection(
            insufficient_last_seed=True
        )
    )
    insufficient_pairs = [
        {
            **pair,
            "dataset": state["dataset"],
            "seed": state["seed"],
            "state_group": state["state_group"],
        }
        for state in insufficient_collection["states"]
        for pair in state["pairs"]
    ]
    _, insufficient_overall = (
        arithmetic._build_independent_formal_dominance(
            insufficient_pairs
        )
    )
    assert insufficient_overall == "insufficient_negative_support"


def test_complete_mock_arithmetic_audit_binds_and_recomputes_everything(
    tmp_path,
):
    paths = _write_mock_artifact_chain(tmp_path)
    output = tmp_path / "arithmetic_audit.json"
    _, report = arithmetic.audit_arithmetic(
        "smoke",
        paths["state_library"],
        paths["proposal_collection"],
        paths["structural_audit"],
        paths["evaluation"],
        output,
        confirmed_state_library_sha256=protocol.file_sha256(
            paths["state_library"]
        ),
        confirmed_proposal_collection_sha256=protocol.file_sha256(
            paths["proposal_collection"]
        ),
        confirmed_structural_audit_sha256=protocol.file_sha256(
            paths["structural_audit"]
        ),
        confirmed_evaluation_sha256=protocol.file_sha256(
            paths["evaluation"]
        ),
    )
    assert output.exists()
    assert report["formal_result_valid"] is False
    assert report["mechanism_evidence_validated"] is False
    assert report["checks"] == arithmetic.PASS_CHECKS
    assert report["manifest"]["state_count"] == 10
    assert report["manifest"]["pair_count"] == 20
    assert report["independent_recalculation"]["strata_match"] is True
    assert report["independent_recalculation"]["formal_dominance"] is None
    changed = copy.deepcopy(report)
    changed["environment"] = {"different": True}
    changed["artifact_paths_diagnostic_only"] = {"different": True}
    changed["elapsed_sec_diagnostic_only"] = 99.0
    assert arithmetic._canonical_sha256(
        arithmetic.scientific_payload(changed)
    ) == report["arithmetic_audit_scientific_sha256"]


def test_artifact_chain_rejects_structural_state_file_mismatch(tmp_path):
    paths = _write_mock_artifact_chain(tmp_path)
    library = arithmetic._strict_load_json(paths["state_library"])
    proposal = arithmetic._strict_load_json(paths["proposal_collection"])
    structural = arithmetic._strict_load_json(paths["structural_audit"])
    evaluation = arithmetic._strict_load_json(paths["evaluation"])
    structural["artifact_identity"]["state_library_file_sha256"] = _sha(
        "wrong-state-file"
    )
    with pytest.raises(RuntimeError, match="未精确绑定"):
        arithmetic._validate_artifact_chain(
            library,
            proposal,
            structural,
            evaluation,
            mode="smoke",
            state_file_sha256=protocol.file_sha256(paths["state_library"]),
            proposal_file_sha256=protocol.file_sha256(
                paths["proposal_collection"]
            ),
            structural_file_sha256=protocol.file_sha256(
                paths["structural_audit"]
            ),
        )


def test_complete_audit_rejects_evaluator_strata_tampering(tmp_path):
    paths = _write_mock_artifact_chain(tmp_path)
    evaluation = arithmetic._strict_load_json(paths["evaluation"])
    evaluation["strata"]["dataset_seed_state"][0]["summary"][
        "pair_count"
    ] += 1
    evaluation["evaluation_scientific_sha256"] = protocol.canonical_sha256(
        evaluator.scientific_payload(evaluation)
    )
    tampered_path = tmp_path / "evaluation_tampered.json"
    state_builder._exclusive_write_json(tampered_path, evaluation)
    with pytest.raises(RuntimeError, match="strata"):
        arithmetic.audit_arithmetic(
            "smoke",
            paths["state_library"],
            paths["proposal_collection"],
            paths["structural_audit"],
            tampered_path,
            tmp_path / "arithmetic_should_not_exist.json",
            confirmed_state_library_sha256=protocol.file_sha256(
                paths["state_library"]
            ),
            confirmed_proposal_collection_sha256=protocol.file_sha256(
                paths["proposal_collection"]
            ),
            confirmed_structural_audit_sha256=protocol.file_sha256(
                paths["structural_audit"]
            ),
            confirmed_evaluation_sha256=protocol.file_sha256(
                tampered_path
            ),
        )


def test_arithmetic_audit_explicit_hash_and_exclusive_output(tmp_path):
    existing = tmp_path / "existing.json"
    state_builder._exclusive_write_json(existing, {"already": True})
    with pytest.raises(FileExistsError, match="不覆盖"):
        arithmetic.audit_arithmetic(
            "smoke",
            tmp_path / "not-read-state.json",
            tmp_path / "not-read-proposal.json",
            tmp_path / "not-read-structural.json",
            tmp_path / "not-read-evaluation.json",
            existing,
            confirmed_state_library_sha256=_sha("state"),
            confirmed_proposal_collection_sha256=_sha("proposal"),
            confirmed_structural_audit_sha256=_sha("structural"),
            confirmed_evaluation_sha256=_sha("evaluation"),
        )


def test_formal_confirmation_happens_before_any_artifact_read(tmp_path):
    with pytest.raises(PermissionError, match="SHA-256"):
        arithmetic.audit_arithmetic(
            "formal",
            tmp_path / "not-read-state.json",
            tmp_path / "not-read-proposal.json",
            tmp_path / "not-read-structural.json",
            tmp_path / "not-read-evaluation.json",
            tmp_path / "not-written.json",
            confirmed_state_library_sha256=_sha("state"),
            confirmed_proposal_collection_sha256=_sha("proposal"),
            confirmed_structural_audit_sha256=_sha("structural"),
            confirmed_evaluation_sha256=_sha("evaluation"),
        )


def test_plan_and_import_graph_are_read_free_and_independent():
    smoke = arithmetic.build_plan("smoke")
    formal = arithmetic.build_plan("formal")
    assert smoke["pair_count"] == 20
    assert formal["pair_count"] == 5500
    assert smoke["input_read_started"] is False
    assert smoke["generation_started"] is False
    assert smoke["audit_boundary"]["frozen_evaluator_imported"] is False
    tree = ast.parse(inspect.getsource(arithmetic))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any(
        fragment in module
        for module in imported_modules
        for fragment in (
            "collect_issue53_stage6a_proposals",
            "audit_issue53_stage6a_proposal_structure",
            "evaluate_issue53_stage6a_overshoot",
        )
    )
