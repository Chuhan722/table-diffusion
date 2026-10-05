from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import audit_issue53_stage6a_proposal_structure as auditor
from scripts import build_issue53_stage6a_state_library as state_builder
from scripts import collect_issue53_stage6a_proposals as collector
from scripts import evaluate_issue53_stage6a_overshoot as evaluator
from scripts import issue53_stage6a_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _zero_exact(dataset: str) -> dict:
    denominator = int(protocol.DATASETS[dataset]["n_records"])
    query_count = int(protocol.DATASETS[dataset]["query_count"])
    return {
        "denominator": denominator,
        "delta_q": [0] * query_count,
        "b2_numerator": 0,
        "cself2_numerator": 0,
        "ccross2_numerator": 0,
        "c2_numerator": 0,
        "g2_numerator": 0,
        "integral_doubled_units": True,
        "integral_count_residual_units": False,
        "b2": 0,
        "cself2": 0,
        "ccross2": 0,
        "c2": 0,
        "g2": 0,
        "category": "unchanged",
        "curvature_role": None,
        "cross_label": None,
    }


def _smoke_pair(
    dataset: str,
    state_group: str,
    proposal_index: int,
) -> dict:
    seed = protocol.SMOKE_SEED
    state_id = protocol.state_id(
        dataset, seed, state_group, mode="smoke"
    )
    donor_address = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "donor",
        mode="smoke",
    )
    update_address = protocol.proposal_address_seed(
        dataset,
        seed,
        state_group,
        proposal_index,
        "update",
        mode="smoke",
    )
    exact = _zero_exact(dataset)
    current_sha = _sha(f"table/{state_id}")
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
            "donor_endpoint_state_sha256": _sha(
                f"donor-end/{state_id}/{proposal_index}"
            ),
            "update_initial_state_sha256": collector._rng_state_sha256(
                np.random.default_rng(update_address)
            ),
            "shared_pre_mutation_state_sha256": _sha(
                f"pre-mutation/{state_id}/{proposal_index}"
            ),
            "copy_only_endpoint_state_sha256": _sha(
                f"pre-mutation/{state_id}/{proposal_index}"
            ),
            "full_endpoint_state_sha256": _sha(
                f"full-end/{state_id}/{proposal_index}"
            ),
            "mutation_rolls_sha256": _sha(
                f"mutation-rolls/{state_id}/{proposal_index}"
            ),
        },
        "donor_indices_sha256": _sha(f"donors/{state_id}/{proposal_index}"),
        "direction_scores_sha256": _sha(
            f"direction/{state_id}/{proposal_index}"
        ),
        "copy_probabilities_sha256": _sha(
            f"copy-prob/{state_id}/{proposal_index}"
        ),
        "direction_reference_scale": 0.5,
        "effective_direction_strength": 4.0,
        "participating_row_indices": [],
        "copy_row_indices_by_attribute": {},
        "copy_edits": [],
        "mutation_events": [],
        "work": {metric: 0 for metric in evaluator.WORK_METRICS},
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
                "denominator": exact["denominator"],
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
        "elapsed_sec_diagnostic_only": 0.01 + proposal_index * 0.01,
    }


def _smoke_collection() -> dict:
    seed = protocol.SMOKE_SEED
    states = []
    pair_ids = []
    for dataset in protocol.DATASET_ORDER:
        for state_group in protocol.STATE_GROUPS:
            state_id = protocol.state_id(
                dataset, seed, state_group, mode="smoke"
            )
            pairs = [
                _smoke_pair(dataset, state_group, proposal_index)
                for proposal_index in range(
                    protocol.proposals_per_state(dataset, mode="smoke")
                )
            ]
            pair_ids.extend(pair["pair_id"] for pair in pairs)
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
                "current_query_answers_sha256": _sha(f"q/{state_id}"),
                "count_residual_numerators_sha256": _sha(
                    f"residual/{state_id}"
                ),
                "residual_denominator": protocol.DATASETS[dataset][
                    "n_records"
                ],
                "residual_signal_sha256": _sha(f"signal/{state_id}"),
                "fitness_sha256": _sha(f"fitness/{state_id}"),
                "direction_reference_scale": 0.5,
                "runtime_device": protocol.source_generator_params(
                    dataset, seed, mode="smoke"
                )["device"],
                "sampling_params": copy.deepcopy(collector.SAMPLING_PARAMS),
                "pairs": pairs,
                "elapsed_sec_diagnostic_only": 0.1,
            })
    collection = {
        "proposal_collection_format": collector.PROPOSAL_COLLECTION_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "artifact_scope": "full",
        "selected_seeds": [seed],
        "formal_result_valid": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "git": {
            "commit": _sha("proposal-commit")[:40],
            "worktree_clean_including_untracked": False,
            "status": ["mock"],
        },
        "environment": {"fixture": True},
        "input_audit": {"fixture": True},
        "runtime_targets": {"fixture": True},
        "probe_boundary": copy.deepcopy(collector.PROBE_BOUNDARY),
        "source_state_artifacts_diagnostic_only": [],
        "states": states,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": [seed],
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(states),
            "pair_count": len(pair_ids),
            "proposal_table_result_count": 2 * len(pair_ids),
            "state_ids_in_fixed_order": collector._expected_state_ids(
                (seed,)
            ),
            "pair_ids_in_fixed_order": pair_ids,
            "source_proposal_seed_shard_sha256": {},
        },
        "elapsed_sec_diagnostic_only": 1.0,
    }
    collection["proposal_scientific_sha256"] = protocol.canonical_sha256(
        collector.scientific_payload(collection)
    )
    collector._validate_collection_structure(
        collection,
        mode="smoke",
        artifact_scope="full",
        selected_seeds=(seed,),
    )
    return collection


def _structural_report(
    proposal: dict,
    proposal_file_sha256: str,
    *,
    bound_proposal_file_sha256: str | None = None,
) -> dict:
    seed = protocol.SMOKE_SEED
    state_audits = []
    for state in proposal["states"]:
        state_audits.append({
            "state_id": state["state_id"],
            "source_state_scientific_sha256": state[
                "source_state_scientific_sha256"
            ],
            "source_trajectory_scientific_sha256": state[
                "source_trajectory_scientific_sha256"
            ],
            "current_table_sha256": state["current_table_sha256"],
            "pair_count": len(state["pairs"]),
            "pair_ids_sha256": protocol.canonical_sha256([
                pair["pair_id"] for pair in state["pairs"]
            ]),
            "pair_structural_audits_sha256": _sha(
                f"pair-audits/{state['state_id']}"
            ),
            "all_pairs_structurally_valid": True,
            "elapsed_sec_diagnostic_only": 0.1,
        })
    report = {
        "structural_audit_format": auditor.STRUCTURAL_AUDIT_FORMAT,
        "status": "complete",
        "mode": "smoke",
        "formal_result_valid": False,
        "protocol": protocol.frozen_protocol_manifest(),
        "protocol_sha256": protocol.FROZEN_PROTOCOL_SHA256,
        "audit_git_commit": _sha("audit-commit")[:40],
        "git": {
            "commit": _sha("audit-commit")[:40],
            "worktree_clean_including_untracked": False,
            "status": ["mock"],
        },
        "environment": {"fixture": True},
        "input_audit": copy.deepcopy(proposal["input_audit"]),
        "runtime_targets": copy.deepcopy(proposal["runtime_targets"]),
        "artifact_paths_diagnostic_only": {},
        "artifact_identity": {
            "state_library_file_sha256": _sha("state-file"),
            "state_library_scientific_sha256": _sha("state-science"),
            "state_library_git_commit": _sha("state-commit")[:40],
            "state_seed_shard_file_sha256": {},
            "proposal_collection_file_sha256": (
                bound_proposal_file_sha256 or proposal_file_sha256
            ),
            "proposal_collection_scientific_sha256": proposal[
                "proposal_scientific_sha256"
            ],
            "proposal_collection_git_commit": proposal["git"]["commit"],
            "proposal_seed_shard_file_sha256": {},
        },
        "audit_boundary": copy.deepcopy(auditor.AUDIT_BOUNDARY),
        "checks": copy.deepcopy(auditor.PASS_CHECKS),
        "state_audits": state_audits,
        "manifest": {
            "dataset_order": list(protocol.DATASET_ORDER),
            "seed_order": [seed],
            "state_group_order": list(protocol.STATE_GROUPS),
            "state_count": len(state_audits),
            "pair_count": proposal["manifest"]["pair_count"],
            "state_ids_in_fixed_order": auditor._expected_state_ids(
                (seed,), mode="smoke"
            ),
            "pair_ids_in_fixed_order_sha256": protocol.canonical_sha256(
                proposal["manifest"]["pair_ids_in_fixed_order"]
            ),
            "state_structural_audits_sha256": protocol.canonical_sha256([
                auditor._state_audit_scientific_payload(row)
                for row in state_audits
            ]),
        },
        "elapsed_sec_diagnostic_only": 1.0,
    }
    report["structural_audit_scientific_sha256"] = (
        protocol.canonical_sha256(auditor.scientific_payload(report))
    )
    auditor._validate_structural_audit_report(
        report, mode="smoke", selected_seeds=(seed,)
    )
    return report


def _write_bound_smoke_artifacts(
    tmp_path: Path, *, wrong_binding: bool = False
) -> tuple[Path, Path]:
    proposal = _smoke_collection()
    proposal_path = tmp_path / "proposal.json"
    state_builder._exclusive_write_json(proposal_path, proposal)
    proposal_sha = protocol.file_sha256(proposal_path)
    report = _structural_report(
        proposal,
        proposal_sha,
        bound_proposal_file_sha256=(
            _sha("wrong-proposal-file") if wrong_binding else None
        ),
    )
    audit_path = tmp_path / "structural_audit.json"
    state_builder._exclusive_write_json(audit_path, report)
    return proposal_path, audit_path


def _minimal_summary_pair(
    value: int,
    *,
    category: str = "improving",
    transition: str = "positive_to_positive",
) -> dict:
    exact_leg = {
        "denominator": 2,
        "b2_numerator": value,
        "c2_numerator": 0,
        "g2_numerator": value,
        "cself2_numerator": 0,
        "ccross2_numerator": 0,
        "category": category,
        "curvature_role": None,
        "cross_label": None,
    }
    return {
        "exact": {
            "copy_only": copy.deepcopy(exact_leg),
            "mutation_given_copy": copy.deepcopy(exact_leg),
            "full": copy.deepcopy(exact_leg),
            "sequential": {
                "denominator": 2,
                "copy_mutation_interaction2_numerator": value,
            },
            "copy_full_gain_sign_transition": transition,
            "mutation_failure_source": None,
        },
        "work": {metric: abs(value) for metric in evaluator.WORK_METRICS},
        "elapsed_sec_diagnostic_only": float(abs(value)),
    }


def _formal_dominance_collection(
    *, insufficient_last_seed: bool = False
) -> dict:
    states = []
    for dataset in protocol.DATASET_ORDER:
        for seed_index, seed in enumerate(protocol.FORMAL_SEEDS):
            for state_group in protocol.STATE_GROUPS:
                pairs = []
                if state_group == "initial":
                    pairs = [
                        {
                            "exact": {
                                "full": {
                                    "category": "direction_failure",
                                    "cross_label": None,
                                },
                                "mutation_failure_source": (
                                    "copy_already_failed"
                                ),
                            }
                        }
                        for _ in range(100)
                    ]
                elif state_group == "work_q25":
                    total = (
                        9
                        if insufficient_last_seed and seed_index == 4
                        else 30
                    )
                    curvature_count = (
                        18 if seed_index < 4 else 12
                    )
                    curvature_count = min(curvature_count, total)
                    created_count = (
                        18 if seed_index < 4 else 12
                    )
                    created_count = min(created_count, total)
                    cross_required_count = (
                        11 if seed_index < 4 else 5
                    )
                    cross_required_count = min(
                        cross_required_count, curvature_count
                    )
                    for index in range(total):
                        curvature = index < curvature_count
                        pairs.append({
                            "exact": {
                                "full": {
                                    "category": (
                                        "curvature_overrun"
                                        if curvature
                                        else "direction_failure"
                                    ),
                                    "cross_label": (
                                        "cross_required_overrun"
                                        if curvature
                                        and index < cross_required_count
                                        else (
                                            "self_sufficient_overrun"
                                            if curvature
                                            else None
                                        )
                                    ),
                                },
                                "mutation_failure_source": (
                                    "mutation_created_failure"
                                    if index < created_count
                                    else "copy_already_failed"
                                ),
                            }
                        })
                states.append({
                    "dataset": dataset,
                    "seed": seed,
                    "state_group": state_group,
                    "pairs": pairs,
                })
    return {"states": states}


def test_exact_type7_statistics_are_predeclared_and_rational():
    summary = evaluator._exact_summary([
        Fraction(0), Fraction(1), Fraction(2), Fraction(3)
    ])
    assert summary["mean"]["numerator"] == 3
    assert summary["mean"]["denominator"] == 2
    assert summary["q25"]["numerator"] == 3
    assert summary["q25"]["denominator"] == 4
    assert summary["median"]["numerator"] == 3
    assert summary["median"]["denominator"] == 2
    assert summary["q75"]["numerator"] == 9
    assert summary["q75"]["denominator"] == 4
    assert evaluator._exact_summary([])["mean"] is None


def test_pair_summary_reports_all_legs_counts_work_and_wallclock():
    rows = [
        _minimal_summary_pair(1),
        _minimal_summary_pair(
            3,
            category="direction_failure",
            transition="positive_to_negative",
        ),
    ]
    summary = evaluator._summarize_pairs(rows)
    assert summary["pair_count"] == 2
    assert summary["legs"]["full"]["metrics"]["b2"]["mean"] == {
        "numerator": 1,
        "denominator": 1,
        "value_diagnostic_only": 1.0,
    }
    assert summary["legs"]["full"]["category_counts"] == {
        label: int(label in {"improving", "direction_failure"})
        for label in protocol.FULL_GAIN_CATEGORIES
    }
    assert summary["sequential"][
        "copy_full_gain_sign_transition_counts"
    ]["positive_to_negative"] == 1
    assert summary["sequential"][
        "copy_full_category_transition_counts"
    ]["improving_to_improving"] == 1
    assert summary["sequential"][
        "copy_full_category_transition_counts"
    ]["direction_failure_to_direction_failure"] == 1
    assert summary["work"]["participating_rows"]["maximum"][
        "numerator"
    ] == 3
    assert summary["wallclock_sec_diagnostic_only"]["median"] == 2.0


def test_equal_seed_aggregate_does_not_pool_proposal_counts():
    pairs_by_seed = {
        1: [_minimal_summary_pair(2)] * 100,
        2: [_minimal_summary_pair(-2)],
    }
    aggregate = evaluator._equal_seed_aggregate(pairs_by_seed)
    b2 = aggregate["legs"]["full"]["metric_seed_means"]["b2"]
    assert b2["equal_seed_distribution"]["mean"]["numerator"] == 0
    assert [row["proposal_count"] for row in b2["per_seed"]] == [100, 1]


def test_formal_dominance_uses_primary_only_and_reports_secondary_axes():
    dominance, overall = evaluator._build_formal_dominance(
        _formal_dominance_collection()
    )
    assert overall == "shared_curvature_overrun"
    assert dominance["initial_included"] is False
    for dataset in protocol.DATASET_ORDER:
        result = dominance["dataset_results"][dataset]
        assert result["geometry"]["label"] == "curvature_overrun"
        assert result["curvature_self_cross"]["label"] == (
            "cross_required_overrun"
        )
        assert result["mutation_failure_source"]["label"] == (
            "mutation_created_failure"
        )
    assert dominance["shared_curvature_self_cross"] == {
        "status": "supported",
        "reason": None,
        "shared_label": "cross_required_overrun",
    }
    assert dominance["shared_mutation_failure_source"]["shared_label"] == (
        "mutation_created_failure"
    )


def test_formal_dominance_fails_closed_on_one_seed_below_ten():
    dominance, overall = evaluator._build_formal_dominance(
        _formal_dominance_collection(insufficient_last_seed=True)
    )
    assert overall == "insufficient_negative_support"
    for dataset in protocol.DATASET_ORDER:
        assert dominance["dataset_results"][dataset]["geometry"][
            "status"
        ] == "insufficient_support"


def test_smoke_evaluation_requires_exact_passing_audit_binding(tmp_path):
    proposal_path, audit_path = _write_bound_smoke_artifacts(tmp_path)
    output_path = tmp_path / "evaluation.json"
    _, report = evaluator.evaluate(
        "smoke",
        proposal_path,
        audit_path,
        output_path,
        confirmed_proposal_collection_sha256=protocol.file_sha256(
            proposal_path
        ),
        confirmed_structural_audit_sha256=protocol.file_sha256(audit_path),
    )
    assert output_path.exists()
    assert report["formal_result_valid"] is False
    assert report["mechanism_evidence_emitted"] is False
    assert report["formal_dominance"] is None
    assert report["formal_overall_result"] is None
    assert report["manifest"]["pair_count"] == 20
    assert len(report["strata"]["dataset_seed_state"]) == 10
    assert report["evaluation_boundary"][
        "proposal_probability_modified"
    ] is False


def test_evaluation_rejects_valid_audit_bound_to_another_file(tmp_path):
    proposal_path, audit_path = _write_bound_smoke_artifacts(
        tmp_path, wrong_binding=True
    )
    with pytest.raises(RuntimeError, match="未精确绑定"):
        evaluator.evaluate(
            "smoke",
            proposal_path,
            audit_path,
            tmp_path / "evaluation.json",
            confirmed_proposal_collection_sha256=protocol.file_sha256(
                proposal_path
            ),
            confirmed_structural_audit_sha256=protocol.file_sha256(
                audit_path
            ),
        )


def test_evaluation_requires_explicit_file_hashes_and_exclusive_output(
    tmp_path,
):
    proposal_path, audit_path = _write_bound_smoke_artifacts(tmp_path)
    with pytest.raises(PermissionError, match="proposal collection"):
        evaluator.evaluate(
            "smoke",
            proposal_path,
            audit_path,
            tmp_path / "wrong-hash.json",
            confirmed_proposal_collection_sha256=_sha("wrong"),
            confirmed_structural_audit_sha256=protocol.file_sha256(
                audit_path
            ),
        )
    existing = tmp_path / "existing.json"
    state_builder._exclusive_write_json(existing, {"already": True})
    with pytest.raises(FileExistsError, match="不覆盖"):
        evaluator.evaluate(
            "smoke",
            proposal_path,
            audit_path,
            existing,
            confirmed_proposal_collection_sha256=protocol.file_sha256(
                proposal_path
            ),
            confirmed_structural_audit_sha256=protocol.file_sha256(
                audit_path
            ),
        )


def test_scientific_sha_excludes_paths_environment_and_top_wallclock(
    tmp_path,
):
    proposal_path, audit_path = _write_bound_smoke_artifacts(tmp_path)
    _, report = evaluator.evaluate(
        "smoke",
        proposal_path,
        audit_path,
        tmp_path / "evaluation.json",
        confirmed_proposal_collection_sha256=protocol.file_sha256(
            proposal_path
        ),
        confirmed_structural_audit_sha256=protocol.file_sha256(audit_path),
    )
    changed = copy.deepcopy(report)
    changed["environment"] = {"different": True}
    changed["artifact_paths_diagnostic_only"] = {"different": True}
    changed["elapsed_sec_diagnostic_only"] = 999.0
    assert protocol.canonical_sha256(
        evaluator.scientific_payload(changed)
    ) == report["evaluation_scientific_sha256"]
    changed["strata"]["dataset_seed_state"][0]["summary"][
        "pair_count"
    ] += 1
    assert protocol.canonical_sha256(
        evaluator.scientific_payload(changed)
    ) != report["evaluation_scientific_sha256"]


def test_plan_is_read_free_gate_free_and_has_correct_pair_count():
    smoke = evaluator.build_plan("smoke")
    formal = evaluator.build_plan("formal")
    assert smoke["pair_count"] == 20
    assert formal["pair_count"] == 5500
    assert smoke["generation_started"] is False
    assert smoke["input_read_started"] is False
    assert smoke["evaluation_boundary"][
        "proposal_acceptance_or_rejection"
    ] is False


def test_formal_evaluation_requires_protocol_confirmation_before_input_read(
    tmp_path,
):
    with pytest.raises(PermissionError, match="SHA-256"):
        evaluator.evaluate(
            "formal",
            tmp_path / "not-read-proposal.json",
            tmp_path / "not-read-audit.json",
            tmp_path / "not-written.json",
            confirmed_proposal_collection_sha256=_sha("proposal"),
            confirmed_structural_audit_sha256=_sha("audit"),
        )
