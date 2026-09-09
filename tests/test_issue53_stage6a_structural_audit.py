from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import audit_issue53_stage6a_proposal_structure as auditor
from scripts import collect_issue53_stage6a_proposals as collector
from scripts import issue53_stage6a_protocol as protocol
from table_diffevo.schema import load_schema


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _toy_table(n_records: int) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def micro_pair_fixture():
    schema = load_schema(
        str(REPOSITORY_ROOT / "configs/test_300x10/schema.yaml")
    )
    current = _toy_table(512)
    queries = _toy_queries()
    probabilities = np.full(
        (len(current), len(current)), 1.0 / len(current)
    )
    residual_numerators = np.asarray([71, -43], dtype=np.int64)
    residual_signal = np.asarray([0.25, -0.125], dtype=float)
    pair = collector._generate_pair(
        dataset="test_300x10",
        seed=protocol.SMOKE_SEED,
        state_group="initial",
        proposal_index=0,
        mode="smoke",
        current=current,
        queries=queries,
        schema=schema,
        count_residual_numerators=residual_numerators,
        residual_denominator=300,
        residual_signal=residual_signal,
        sampling_probabilities=probabilities,
        device="numpy",
        direction_reference_scale=0.5,
    )
    assert pair["copy_edits"]
    return {
        "schema": schema,
        "current": current,
        "queries": queries,
        "probabilities": probabilities,
        "residual_numerators": residual_numerators,
        "residual_signal": residual_signal,
        "pair": pair,
    }


def _audit_micro_pair(fixture, pair):
    return auditor._audit_pair(
        pair,
        dataset="test_300x10",
        seed=protocol.SMOKE_SEED,
        state_group="initial",
        proposal_index=0,
        mode="smoke",
        current=fixture["current"],
        queries=fixture["queries"],
        schema=fixture["schema"],
        residual_numerators=fixture["residual_numerators"],
        residual_denominator=300,
        residual_signal=fixture["residual_signal"],
        sampling_probabilities=fixture["probabilities"],
        device="numpy",
        direction_reference_scale=0.5,
    )


def _envelope_collection() -> dict:
    seed = protocol.SMOKE_SEED
    states = []
    pair_ids = []
    for dataset in protocol.DATASET_ORDER:
        for state_group in protocol.STATE_GROUPS:
            state_id = protocol.state_id(
                dataset, seed, state_group, mode="smoke"
            )
            pairs = []
            for proposal_index in range(
                protocol.proposals_per_state(dataset, mode="smoke")
            ):
                pair_id = auditor._pair_id(state_id, proposal_index)
                pair_ids.append(pair_id)
                pairs.append({
                    "pair_id": pair_id,
                    "state_id": state_id,
                    "dataset": dataset,
                    "seed": seed,
                    "state_group": state_group,
                    "proposal_index": proposal_index,
                    "retained_unconditionally": True,
                    "rng": {
                        "donor_address_uint64": (
                            protocol.proposal_address_seed(
                                dataset,
                                seed,
                                state_group,
                                proposal_index,
                                "donor",
                                mode="smoke",
                            )
                        ),
                        "update_address_uint64": (
                            protocol.proposal_address_seed(
                                dataset,
                                seed,
                                state_group,
                                proposal_index,
                                "update",
                                mode="smoke",
                            )
                        ),
                    },
                })
            states.append({
                "state_id": state_id,
                "dataset": dataset,
                "seed": seed,
                "state_group": state_group,
                "source_state_scientific_sha256": _sha(
                    f"state/{state_id}"
                ),
                "source_trajectory_scientific_sha256": _sha(
                    f"trajectory/{dataset}/{seed}"
                ),
                "current_table_sha256": _sha(f"table/{state_id}"),
                "sampling_params": copy.deepcopy(auditor.SAMPLING_PARAMS),
                "pairs": pairs,
            })
    state_ids = auditor._expected_state_ids((seed,), mode="smoke")
    collection = {
        "proposal_collection_format": auditor.PROPOSAL_COLLECTION_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "artifact_scope": "full",
        "selected_seeds": [seed],
        "formal_result_valid": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "input_audit": {"fixture": True},
        "runtime_targets": {"fixture": True},
        "probe_boundary": copy.deepcopy(auditor.PROBE_BOUNDARY),
        "states": states,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": [seed],
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(states),
            "pair_count": len(pair_ids),
            "proposal_table_result_count": 2 * len(pair_ids),
            "state_ids_in_fixed_order": state_ids,
            "pair_ids_in_fixed_order": pair_ids,
            "source_proposal_seed_shard_sha256": {},
        },
    }
    collection["proposal_scientific_sha256"] = protocol.canonical_sha256(
        auditor._proposal_scientific_payload(collection)
    )
    return collection


def _mock_full_artifacts(input_audit, runtime_targets):
    seed = protocol.SMOKE_SEED
    states = []
    trajectories = []
    proposal_states = []
    for dataset in protocol.DATASET_ORDER:
        trajectories.append({
            "dataset": dataset,
            "seed": seed,
            "runtime_device": "numpy",
            "direction_reference_scale": 0.5,
        })
        for state_group in protocol.STATE_GROUPS:
            state_id = protocol.state_id(
                dataset, seed, state_group, mode="smoke"
            )
            states.append({
                "state_id": state_id,
                "dataset": dataset,
                "seed": seed,
                "state_group": state_group,
            })
            proposal_states.append({
                "state_id": state_id,
                "dataset": dataset,
                "seed": seed,
                "state_group": state_group,
            })
    state_library = {
        "git": {"commit": _sha("state-commit")[:40]},
        "input_audit": copy.deepcopy(input_audit),
        "runtime_targets": copy.deepcopy(runtime_targets),
        "states": states,
        "trajectories": trajectories,
        "state_library_scientific_sha256": _sha("state-science"),
        "manifest": {"source_seed_shard_sha256": {}},
    }
    proposal_collection = {
        "git": {"commit": _sha("proposal-commit")[:40]},
        "input_audit": copy.deepcopy(input_audit),
        "runtime_targets": copy.deepcopy(runtime_targets),
        "states": proposal_states,
        "proposal_scientific_sha256": _sha("proposal-science"),
        "manifest": {"source_proposal_seed_shard_sha256": {}},
        "source_state_artifacts_diagnostic_only": [{
            "file_sha256": _sha("state-file"),
            "state_library_scientific_sha256": _sha("state-science"),
            "git_commit": _sha("state-commit")[:40],
        }],
    }
    return state_library, proposal_collection


def test_independent_exact_scaled_arithmetic_matches_frozen_units():
    deltas = np.asarray([[1, 0], [1, -1]], dtype=np.int8)
    residual = np.asarray([3, 2], dtype=np.int64)
    observed = auditor._exact_scaled_decomposition(deltas, residual, 1)
    assert observed["delta_q"] == [2, -1]
    assert observed["b2"] == 8
    assert observed["c2"] == 5
    assert observed["g2"] == 3
    assert observed["integral_count_residual_units"] is True


def test_pair_audit_replays_rng_sparse_tables_and_exact_geometry(
    micro_pair_fixture,
):
    result = _audit_micro_pair(
        micro_pair_fixture, micro_pair_fixture["pair"]
    )
    assert result["pair_id"].endswith("proposal_0000")
    assert auditor._is_sha256(result["pair_structural_audit_sha256"])
    assert set(result) == {"pair_id", "pair_structural_audit_sha256"}


def test_pair_audit_rejects_sparse_edit_tampering(micro_pair_fixture):
    pair = copy.deepcopy(micro_pair_fixture["pair"])
    pair["copy_edits"][0]["cells"][0]["after"] = (
        pair["copy_edits"][0]["cells"][0]["before"]
    )
    with pytest.raises(RuntimeError, match="structural audit pair"):
        _audit_micro_pair(micro_pair_fixture, pair)


def test_pair_audit_rejects_rng_endpoint_tampering(micro_pair_fixture):
    pair = copy.deepcopy(micro_pair_fixture["pair"])
    pair["rng"]["full_endpoint_state_sha256"] = _sha("tampered")
    with pytest.raises(RuntimeError, match="structural audit pair"):
        _audit_micro_pair(micro_pair_fixture, pair)


def test_pair_audit_rejects_exact_identity_tampering(micro_pair_fixture):
    pair = copy.deepcopy(micro_pair_fixture["pair"])
    pair["exact"]["full"]["g2_numerator"] += 1
    with pytest.raises(RuntimeError, match="structural audit pair"):
        _audit_micro_pair(micro_pair_fixture, pair)


def test_independent_update_replay_covers_mutation_sparse_path():
    schema = load_schema(
        str(REPOSITORY_ROOT / "configs/test_300x10/schema.yaml")
    )
    current = _toy_table(512)
    donors = current.iloc[np.roll(np.arange(len(current)), 11)].reset_index(
        drop=True
    )
    scores = np.zeros((len(current), len(schema.attribute_names())))
    replay = auditor._replay_update(
        current,
        donors,
        schema,
        scores,
        0.0,
        0,
    )
    assert replay["mutation_events"]
    edits, _ = auditor._expected_copy_edits(
        current,
        replay["copy_table"],
        donors,
        np.roll(np.arange(len(current)), 11),
        schema.attribute_names(),
    )
    copy_table, full_table = auditor._apply_artifact_sparse_logs(
        current,
        edits,
        replay["mutation_events"],
        schema.attribute_names(),
    )
    pd.testing.assert_frame_equal(copy_table, replay["copy_table"])
    pd.testing.assert_frame_equal(full_table, replay["full_table"])


def test_proposal_envelope_is_result_blind_and_rejects_gate_fields():
    collection = _envelope_collection()
    auditor._validate_proposal_envelope(
        collection,
        mode="smoke",
        artifact_scope="full",
        selected_seeds=(protocol.SMOKE_SEED,),
    )
    collection["states"][0]["pairs"][0]["accepted"] = True
    collection["proposal_scientific_sha256"] = protocol.canonical_sha256(
        auditor._proposal_scientific_payload(collection)
    )
    with pytest.raises(RuntimeError, match="gate/最终结论字段"):
        auditor._validate_proposal_envelope(
            collection,
            mode="smoke",
            artifact_scope="full",
            selected_seeds=(protocol.SMOKE_SEED,),
        )


def test_proposal_source_shard_hash_and_content_are_reaudited(tmp_path):
    full = _envelope_collection()
    shard = copy.deepcopy(full)
    shard["proposal_collection_format"] = auditor.PROPOSAL_SHARD_FORMAT
    shard["artifact_scope"] = "seed_shard"
    shard["proposal_scientific_sha256"] = protocol.canonical_sha256(
        auditor._proposal_scientific_payload(shard)
    )
    shard_path = tmp_path / "proposal_seed_9906.json"
    auditor.state_builder._exclusive_write_json(shard_path, shard)
    shard_sha = protocol.file_sha256(shard_path)
    full["manifest"]["source_proposal_seed_shard_sha256"] = {
        str(protocol.SMOKE_SEED): shard_sha
    }
    observed = auditor._audit_proposal_shards(
        full,
        [shard_path],
        mode="smoke",
        selected_seeds=(protocol.SMOKE_SEED,),
    )
    assert observed == {str(protocol.SMOKE_SEED): shard_sha}


def test_missing_manifest_shards_fail_closed():
    full = {
        "manifest": {
            "source_seed_shard_sha256": {
                str(protocol.SMOKE_SEED): _sha("missing")
            }
        },
        "states": [],
        "trajectories": [],
    }
    with pytest.raises(RuntimeError, match="未完整覆盖"):
        auditor._audit_state_shards(
            full,
            (),
            mode="smoke",
            selected_seeds=(protocol.SMOKE_SEED,),
        )


def test_stable_loader_requires_explicit_file_sha(tmp_path):
    path = tmp_path / "artifact.json"
    auditor.state_builder._exclusive_write_json(path, {"ok": True})
    with pytest.raises(ValueError, match="显式确认不一致"):
        auditor._load_stable_json(
            path,
            confirmed_sha256=_sha("wrong"),
            confirmation_name="fixture",
        )
    _, observed_sha, payload = auditor._load_stable_json(
        path,
        confirmed_sha256=protocol.file_sha256(path),
        confirmation_name="fixture",
    )
    assert observed_sha == protocol.file_sha256(path)
    assert payload == {"ok": True}


def test_audit_report_is_result_blind_exclusive_and_wallclock_free(
    tmp_path, monkeypatch
):
    input_audit = {"fixture": "inputs"}
    runtime_targets = {
        dataset: {"target_values": []}
        for dataset in protocol.DATASET_ORDER
    }
    runtime_inputs = {
        dataset: {"queries": [], "targets": []}
        for dataset in protocol.DATASET_ORDER
    }
    state_library, proposal_collection = _mock_full_artifacts(
        input_audit, runtime_targets
    )
    state_path = tmp_path / "state.json"
    proposal_path = tmp_path / "proposal.json"
    auditor.state_builder._exclusive_write_json(state_path, {})
    auditor.state_builder._exclusive_write_json(proposal_path, {})
    state_sha = protocol.file_sha256(state_path)
    proposal_sha = protocol.file_sha256(proposal_path)
    proposal_collection["source_state_artifacts_diagnostic_only"][0][
        "file_sha256"
    ] = state_sha

    monkeypatch.setattr(
        auditor,
        "_load_full_artifacts",
        lambda *args, **kwargs: (
            state_path,
            state_sha,
            copy.deepcopy(state_library),
            proposal_path,
            proposal_sha,
            copy.deepcopy(proposal_collection),
        ),
    )
    monkeypatch.setattr(
        auditor.state_builder,
        "_audit_runtime_inputs",
        lambda root, mode: (
            copy.deepcopy(input_audit),
            copy.deepcopy(runtime_inputs),
            copy.deepcopy(runtime_targets),
        ),
    )
    monkeypatch.setattr(auditor, "load_queries", lambda path: [])
    monkeypatch.setattr(auditor, "load_schema", lambda path: object())

    def fake_audit_state(state, trajectory, proposal_state, **kwargs):
        del trajectory, proposal_state, kwargs
        return {
            "state_id": state["state_id"],
            "source_state_scientific_sha256": _sha(
                f"state/{state['state_id']}"
            ),
            "source_trajectory_scientific_sha256": _sha(
                f"trajectory/{state['dataset']}/{state['seed']}"
            ),
            "current_table_sha256": _sha(f"table/{state['state_id']}"),
            "pair_count": 2,
            "pair_ids_sha256": _sha(f"pairs/{state['state_id']}"),
            "pair_structural_audits_sha256": _sha(
                f"audits/{state['state_id']}"
            ),
            "all_pairs_structurally_valid": True,
            "elapsed_sec_diagnostic_only": 1.0,
        }

    monkeypatch.setattr(auditor, "_audit_state", fake_audit_state)
    output = tmp_path / "structural_audit.json"
    _, report = auditor.audit_structure(
        "smoke",
        state_path,
        proposal_path,
        output,
        confirmed_state_library_sha256=state_sha,
        confirmed_proposal_collection_sha256=proposal_sha,
    )
    assert report["checks"]["overall_pass"] is True
    assert report["audit_boundary"] == auditor.AUDIT_BOUNDARY
    assert "summary" not in report
    assert "overall_result" not in report
    assert "shared_label" not in report
    assert all(
        set(row)
        == {
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
        for row in report["state_audits"]
    )
    changed = copy.deepcopy(report)
    changed["elapsed_sec_diagnostic_only"] = 999.0
    changed["artifact_paths_diagnostic_only"]["state_library"] = "/tmp/x"
    changed["state_audits"][0]["elapsed_sec_diagnostic_only"] = 888.0
    assert auditor.scientific_payload(changed) == auditor.scientific_payload(
        report
    )
    tampered = copy.deepcopy(report)
    tampered["checks"]["all_sparse_edits_reconstructed"] = False
    tampered["structural_audit_scientific_sha256"] = (
        protocol.canonical_sha256(auditor.scientific_payload(tampered))
    )
    with pytest.raises(RuntimeError, match="顶层/manifest"):
        auditor._validate_structural_audit_report(
            tampered,
            mode="smoke",
            selected_seeds=(protocol.SMOKE_SEED,),
        )
    with pytest.raises(FileExistsError):
        auditor.audit_structure(
            "smoke",
            state_path,
            proposal_path,
            output,
            confirmed_state_library_sha256=state_sha,
            confirmed_proposal_collection_sha256=proposal_sha,
        )


def test_formal_confirmation_fails_before_artifact_audit(
    tmp_path, monkeypatch
):
    called = []
    monkeypatch.setattr(
        auditor,
        "_load_full_artifacts",
        lambda *args, **kwargs: called.append(True),
    )
    with pytest.raises(PermissionError):
        auditor.audit_structure(
            "formal",
            tmp_path / "missing-state",
            tmp_path / "missing-proposal",
            tmp_path / "output",
            confirmed_state_library_sha256=_sha("state"),
            confirmed_proposal_collection_sha256=_sha("proposal"),
        )
    assert called == []


def test_plan_and_cli_have_no_scientific_or_result_overrides():
    plan = auditor.build_plan("formal")
    assert plan["proposal_pair_count"] == 5500
    assert plan["audit_boundary"]["shared_label_emitted"] is False
    parser = auditor._build_parser()
    audit_parser = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None)
    ).choices["audit"]
    option_strings = {
        option
        for action in audit_parser._actions
        for option in action.option_strings
    }
    assert {
        "--seed",
        "--dataset",
        "--rho",
        "--eta",
        "--mu",
        "--alpha",
        "--category",
        "--minimum-failures",
    }.isdisjoint(option_strings)
