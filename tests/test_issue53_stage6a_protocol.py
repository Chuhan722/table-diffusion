from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts import issue53_stage6a_protocol as protocol
from scripts import issue53_stage5_protocol as stage5


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_document_manifest_and_inputs_are_bound():
    assert protocol.file_sha256(
        REPOSITORY_ROOT / protocol.PROTOCOL_DOC
    ) == protocol.PROTOCOL_DOC_SHA256
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    assert set(protocol.verify_input_file_identities(REPOSITORY_ROOT)) == {
        "test_300x10",
        "nltcs",
    }
    json.dumps(
        protocol.frozen_protocol_manifest(),
        ensure_ascii=False,
        allow_nan=False,
    )


def test_formal_matrix_is_fresh_fixed_and_complete():
    assert protocol.FORMAL_SEEDS == tuple(range(348, 353))
    assert all(
        seed > protocol.EXCLUDED_ISSUE53_SEED_RANGE[1]
        for seed in protocol.FORMAL_SEEDS
    )
    assert protocol.STATE_GROUPS == (
        "initial",
        "work_q25",
        "work_q50",
        "work_q75",
        "terminal",
    )
    assert len(protocol.expected_state_ids()) == 50
    assert len(set(protocol.expected_state_ids())) == 50
    assert protocol.proposals_per_state("test_300x10") == 200
    assert protocol.proposals_per_state("nltcs") == 20
    manifest = protocol.frozen_protocol_manifest()
    assert manifest["source_trajectory_count"] == 10
    assert manifest["state_count"] == 50
    assert manifest["proposal_pair_count"] == 5500
    assert manifest["proposal_table_result_count"] == 11000


def test_source_trajectory_is_independent_terminal_current_and_gate_free():
    for dataset in protocol.DATASET_ORDER:
        params = protocol.source_generator_params(
            dataset, protocol.FORMAL_SEEDS[0]
        )
        assert params["n_records"] == protocol.DATASETS[dataset][
            "n_records"
        ]
        assert params["device"] == protocol.DATASETS[dataset]["device"]
        assert params["tol"] == float("inf")
        assert params["max_retries"] == 0
        assert params["factorized_gibbs_sweeps"] == 0
        assert params["mu"] == 0.01
        assert params["fixed_alpha"] == 16.0
        assert params["diffusion_direction_strength"] == 2.0
        assert params["residual_geometry"] == "relative"
        assert params["residual_geometry_floor"] == 8.0
        assert params["record_natural_work_snapshots"] is True
        assert params["return_final_table"] is True
        assert (
            params["inner_early_stopping_patience_ticks"]
            == protocol.PATIENCE_TICKS
        )
    no_gate = protocol.frozen_protocol_manifest()["no_gate_contract"]
    assert no_gate == {
        "tolerance": "positive_infinity",
        "max_retries": 0,
        "one_finite_proposal_unconditionally_applied": True,
        "terminal_current": True,
        "proposal_result_filtering": False,
        "retry_after_bc": False,
        "rollback": False,
        "best_or_shadow_selection": False,
        "probe_feedback_to_source_trajectory": False,
    }


def test_source_scientific_parameters_match_stage5_independent_reference():
    for dataset in protocol.DATASET_ORDER:
        previous = stage5.generator_params(
            dataset,
            stage5.FORMAL_SEEDS[0],
            stage5.ARM_INDEPENDENT,
        )
        current = protocol.source_generator_params(
            dataset, protocol.FORMAL_SEEDS[0]
        )
        for extra_key in ("n_records", "device", "init_method"):
            current.pop(extra_key)
        for observational_key in (
            "seed",
            "log_every",
            "record_natural_work_snapshots",
        ):
            previous.pop(observational_key)
            current.pop(observational_key)
        assert current == previous


def test_smoke_is_small_nonformal_and_cannot_take_formal_seed():
    params = protocol.source_generator_params(
        "nltcs", protocol.SMOKE_SEED, mode="smoke"
    )
    assert params["n_records"] == 128
    assert params["device"] == "numpy"
    assert params["n_rounds"] == 500
    assert params["candidate_budget"] == 500
    assert protocol.proposals_per_state("nltcs", mode="smoke") == 2
    with pytest.raises(ValueError, match="冻结矩阵"):
        protocol.source_generator_params(
            "nltcs", protocol.FORMAL_SEEDS[0], mode="smoke"
        )
    with pytest.raises(ValueError, match="冻结矩阵"):
        protocol.source_generator_params(
            "nltcs", protocol.SMOKE_SEED, mode="formal"
        )


def test_formal_execution_requires_exact_protocol_confirmation():
    protocol.require_formal_confirmation("smoke", None)
    with pytest.raises(PermissionError, match="单独显式授权"):
        protocol.require_formal_confirmation("formal", None)
    with pytest.raises(PermissionError, match="单独显式授权"):
        protocol.require_formal_confirmation("formal", "wrong")
    protocol.require_formal_confirmation(
        "formal", protocol.FROZEN_PROTOCOL_SHA256
    )


def test_proposal_addresses_are_deterministic_unique_and_fail_closed():
    observed = set()
    for dataset in protocol.DATASET_ORDER:
        for seed in protocol.FORMAL_SEEDS:
            for group in protocol.STATE_GROUPS:
                for proposal_index in range(
                    protocol.proposals_per_state(dataset)
                ):
                    for stream in protocol.PROPOSAL_STREAMS:
                        address = protocol.proposal_address_seed(
                            dataset,
                            seed,
                            group,
                            proposal_index,
                            stream,
                        )
                        assert 0 <= address < 2**64
                        assert address not in observed
                        observed.add(address)
                        assert address == protocol.proposal_address_seed(
                            dataset,
                            seed,
                            group,
                            proposal_index,
                            stream,
                        )
    assert len(observed) == 2 * 5500
    with pytest.raises(ValueError, match="proposal_index"):
        protocol.proposal_address_seed(
            "nltcs",
            protocol.FORMAL_SEEDS[0],
            "initial",
            20,
            "donor",
        )
    with pytest.raises(ValueError, match="RNG stream"):
        protocol.proposal_address_seed(
            "nltcs",
            protocol.FORMAL_SEEDS[0],
            "initial",
            0,
            "mutation",
        )


@pytest.mark.parametrize(
    ("b2", "c2", "expected"),
    [
        (3, 2, "improving"),
        (0, 0, "unchanged"),
        (2, 2, "exact_balance"),
        (0, 1, "direction_failure"),
        (-1, 1, "direction_failure"),
        (1, 2, "curvature_overrun"),
    ],
)
def test_full_proposal_categories_are_exact(b2, c2, expected):
    assert protocol.classify_full_proposal(b2, c2) == expected


def test_full_proposal_categories_reject_impossible_or_inexact_values():
    with pytest.raises(ValueError, match="C2 必须非负"):
        protocol.classify_full_proposal(0, -1)
    with pytest.raises(ValueError, match="B2 也必须为零"):
        protocol.classify_full_proposal(1, 0)
    with pytest.raises(ValueError, match="精确整数"):
        protocol.classify_full_proposal(1.0, 2)
    with pytest.raises(ValueError, match="精确整数"):
        protocol.classify_full_proposal(True, 1)


@pytest.mark.parametrize(
    ("b2", "cself2", "ccross2", "expected"),
    [
        (3, 5, -1, "self_sufficient_overrun"),
        (3, 3, 1, "cross_breaks_tie"),
        (3, 2, 2, "cross_decisive_overrun"),
    ],
)
def test_curvature_role_uses_counterfactual_without_cross(
    b2, cself2, ccross2, expected
):
    assert (
        protocol.classify_curvature_role(b2, cself2, ccross2)
        == expected
    )


def test_curvature_role_rejects_non_overrun_and_negative_total_c():
    with pytest.raises(ValueError, match="只对 curvature_overrun"):
        protocol.classify_curvature_role(5, 2, 1)
    with pytest.raises(ValueError, match="必须非负"):
        protocol.classify_curvature_role(1, 2, -3)


def test_mutation_failure_and_sign_transition_are_sequential():
    assert (
        protocol.classify_mutation_failure(1, -1)
        == "mutation_created_failure"
    )
    assert (
        protocol.classify_mutation_failure(-3, -1)
        == "copy_already_failed"
    )
    assert (
        protocol.classify_mutation_sign_transition(-1, 2)
        == "negative_to_positive"
    )
    assert (
        protocol.classify_mutation_sign_transition(0, 0)
        == "zero_to_zero"
    )
    with pytest.raises(ValueError, match="full G2<0"):
        protocol.classify_mutation_failure(-1, 0)


def test_exact_row_decomposition_keeps_signed_cross_cancellation():
    row_deltas = np.asarray(
        [
            [1, 0],
            [-1, 1],
            [0, -1],
        ],
        dtype=np.int8,
    )
    residual = np.asarray([3, -2], dtype=np.int64)
    result = protocol.exact_doubled_decomposition(
        row_deltas, residual
    )
    np.testing.assert_array_equal(result["delta_q"], [0, 0])
    assert result == {
        "delta_q": result["delta_q"],
        "b2": 0,
        "cself2": 4,
        "ccross2": -4,
        "c2": 0,
        "g2": 0,
        "category": "unchanged",
    }


def test_exact_row_decomposition_computes_positive_gain_without_float():
    row_deltas = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
    residual = np.asarray([2, 1], dtype=np.int64)
    result = protocol.exact_doubled_decomposition(
        row_deltas, residual
    )
    assert result["b2"] == 6
    assert result["cself2"] == 2
    assert result["ccross2"] == 0
    assert result["c2"] == 2
    assert result["g2"] == 4
    assert result["category"] == "improving"


def test_exact_row_decomposition_rejects_float_inputs():
    with pytest.raises(ValueError, match="整数二维"):
        protocol.exact_doubled_decomposition(
            np.asarray([[1.0]]), np.asarray([1], dtype=np.int64)
        )
    with pytest.raises(ValueError, match="整数一维"):
        protocol.exact_doubled_decomposition(
            np.asarray([[1]], dtype=np.int8), np.asarray([1.0])
        )


def test_sequential_copy_mutation_identity_is_exact():
    result = protocol.validate_sequential_identity(
        bcopy2=6,
        ccopy2=1,
        bmutation_given_copy2=4,
        cmutation2=1,
        bfull2=10,
        cfull2=2,
        copy_mutation_interaction2=0,
    )
    assert result == {
        "gcopy2": 5,
        "gmutation_given_copy2": 3,
        "gfull2": 8,
    }
    with pytest.raises(ValueError, match="恒等式失败"):
        protocol.validate_sequential_identity(
            bcopy2=6,
            ccopy2=1,
            bmutation_given_copy2=4,
            cmutation2=1,
            bfull2=10,
            cfull2=3,
            copy_mutation_interaction2=0,
        )


def _counts(left_right_by_seed):
    return {
        seed: {
            "direction_failure": left,
            "curvature_overrun": right,
        }
        for seed, (left, right) in zip(
            protocol.FORMAL_SEEDS, left_right_by_seed
        )
    }


def test_dataset_dominance_requires_minimum_4_of_5_and_equal_seed_mean():
    supported = protocol.classify_dataset_dominance(
        _counts([(6, 4), (6, 4), (6, 4), (6, 4), (4, 6)]),
        protocol.GEOMETRY_LABELS,
    )
    assert supported["status"] == "supported"
    assert supported["label"] == "direction_failure"
    assert supported["support_counts"]["direction_failure"] == 4
    assert supported["mean_shares"]["direction_failure"]["value"] == 0.56

    insufficient = protocol.classify_dataset_dominance(
        _counts([(5, 4), (6, 4), (6, 4), (6, 4), (6, 4)]),
        protocol.GEOMETRY_LABELS,
    )
    assert insufficient["status"] == "insufficient_support"
    assert insufficient["label"] is None
    assert insufficient["insufficient_seeds"] == [348]

    only_three_seed_majorities = protocol.classify_dataset_dominance(
        _counts([(6, 4), (6, 4), (6, 4), (4, 6), (4, 6)]),
        protocol.GEOMETRY_LABELS,
    )
    assert only_three_seed_majorities["status"] == "no_stable_label"

    mean_share_fails = protocol.classify_dataset_dominance(
        _counts(
            [
                (6, 4),
                (6, 4),
                (6, 4),
                (6, 4),
                (0, 100),
            ]
        ),
        protocol.GEOMETRY_LABELS,
    )
    assert mean_share_fails["support_counts"]["direction_failure"] == 4
    assert mean_share_fails["status"] == "no_stable_label"


def test_dataset_dominance_rejects_seed_or_axis_drift():
    missing_seed = _counts([(6, 4)] * 5)
    missing_seed.pop(protocol.FORMAL_SEEDS[-1])
    with pytest.raises(ValueError, match="五个正式 seeds"):
        protocol.classify_dataset_dominance(
            missing_seed, protocol.GEOMETRY_LABELS
        )
    bad_axis = _counts([(6, 4)] * 5)
    bad_axis[348]["extra"] = 1
    with pytest.raises(ValueError, match="两个 axis labels"):
        protocol.classify_dataset_dominance(
            bad_axis, protocol.GEOMETRY_LABELS
        )


def test_shared_geometry_requires_same_supported_label_on_both_datasets():
    direction = protocol.classify_dataset_dominance(
        _counts([(6, 4)] * 5), protocol.GEOMETRY_LABELS
    )
    curvature = protocol.classify_dataset_dominance(
        _counts([(4, 6)] * 5), protocol.GEOMETRY_LABELS
    )
    insufficient = protocol.classify_dataset_dominance(
        _counts([(5, 4)] * 5), protocol.GEOMETRY_LABELS
    )
    assert protocol.shared_geometry_result(
        {"test_300x10": direction, "nltcs": direction}
    ) == "shared_direction_failure"
    assert protocol.shared_geometry_result(
        {"test_300x10": curvature, "nltcs": curvature}
    ) == "shared_curvature_overrun"
    assert protocol.shared_geometry_result(
        {"test_300x10": direction, "nltcs": curvature}
    ) == "no_shared_failure_geometry"
    assert protocol.shared_geometry_result(
        {"test_300x10": direction, "nltcs": insufficient}
    ) == "insufficient_negative_support"
