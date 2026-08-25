"""第 6B-1 阶段冻结协议身份和判定顺序测试。"""

from pathlib import Path

import pytest

from scripts import issue53_stage6b1_protocol as protocol


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_document_manifest_and_source_artifacts_are_bound():
    assert protocol.file_sha256(
        REPOSITORY_ROOT / protocol.PROTOCOL_DOC
    ) == protocol.PROTOCOL_DOC_SHA256
    assert protocol.protocol_sha256() == protocol.FROZEN_PROTOCOL_SHA256
    protocol.assert_frozen_protocol_identity(REPOSITORY_ROOT)
    assert set(protocol.assert_source_artifact_identities(
        REPOSITORY_ROOT, "formal"
    )) == {
        "state_library",
        "proposal_collection",
        "structural_audit",
        "frozen_evaluation",
        "independent_arithmetic_audit",
    }
    protocol.assert_source_artifact_identities(REPOSITORY_ROOT, "smoke")
    assert protocol.PARENT_STAGE6B1_PROTOCOL_SHA256 == (
        "6087598eda6080711f532f07566be28680f86ea2059c7afe09085b1b7e30b0dc"
    )
    for binding in protocol.PARENT_STAGE6B1_SMOKE_ARTIFACTS.values():
        assert protocol.file_sha256(
            REPOSITORY_ROOT / binding["path"]
        ) == binding["sha256"]


def test_frozen_matrices_and_no_gate_contract_are_exact():
    formal = protocol.build_plan("formal")
    smoke = protocol.build_plan("smoke")
    assert formal["datasets"] == ["test_300x10"]
    assert formal["pair_count"] == 5000
    assert formal["arm_leg_record_count"] == 30000
    assert smoke["datasets"] == ["test_300x10"]
    assert smoke["pair_count"] == 10
    assert smoke["arm_leg_record_count"] == 60
    assert protocol.ARMS == (
        "independent_b_s0",
        "factor_b_s8",
        "gap_l1_global_s8",
    )
    assert protocol.GIBBS_SWEEPS == 8
    assert protocol.NO_GATE_CONTRACT == {
        "one_final_mask_per_address_and_arm": True,
        "all_addressed_results_retained": True,
        "post_proposal_acceptance": False,
        "proposal_rejection": False,
        "proposal_retry": False,
        "proposal_rollback": False,
        "winner_or_best_selection": False,
        "result_feedback_to_source_states": False,
        "result_driven_resampling": False,
    }


def test_gibbs_streams_are_deterministic_unique_and_domain_isolated():
    observed = set()
    for mode in ("smoke", "formal"):
        seeds = protocol.mode_seeds(mode)
        # 完整遍历会同时验证 11000 个派生地址没有碰撞。
        for dataset in protocol.DATASET_ORDER:
            for seed in seeds:
                for group in protocol.STATE_GROUPS:
                    for proposal_index in range(
                        protocol.proposals_per_state(dataset, mode=mode)
                    ):
                        factor = protocol.gibbs_address_seed(
                            dataset,
                            seed,
                            group,
                            proposal_index,
                            protocol.ARM_FACTOR,
                            mode=mode,
                        )
                        gap = protocol.gibbs_address_seed(
                            dataset,
                            seed,
                            group,
                            proposal_index,
                            protocol.ARM_GAP_L1,
                            mode=mode,
                        )
                        assert factor != gap
                        assert factor not in observed
                        observed.add(factor)
                        assert gap not in observed
                        observed.add(gap)
                        assert factor == protocol.gibbs_address_seed(
                            dataset,
                            seed,
                            group,
                            proposal_index,
                            protocol.ARM_FACTOR,
                            mode=mode,
                        )
    with pytest.raises(ValueError, match="两个吉布斯组"):
        protocol.gibbs_address_seed(
            "nltcs", 9906, "initial", 0,
            protocol.ARM_INDEPENDENT, mode="smoke"
        )


def test_stage1_reuses_original_stage6b1_gibbs_address_domain():
    assert protocol.RNG_DOMAIN_VERSION == (
        "issue53-stage6b1-gap-l1-fixed-state-screen-v1"
    )
    assert protocol.gibbs_address_seed(
        "test_300x10",
        348,
        "initial",
        0,
        protocol.ARM_FACTOR,
        mode="formal",
    ) == 8925589475095895860
    assert protocol.gibbs_address_seed(
        "test_300x10",
        348,
        "initial",
        0,
        protocol.ARM_GAP_L1,
        mode="formal",
    ) == 16072142247567796561
    # 这两个快照同时与已通过双审计的原第 6B-1 阶段
    # 小规模落盘产物一致，防止分阶段修正重抽随机流。
    assert protocol.gibbs_address_seed(
        "test_300x10",
        9906,
        "terminal",
        1,
        protocol.ARM_FACTOR,
        mode="smoke",
    ) == 16938597047447038544
    assert protocol.gibbs_address_seed(
        "test_300x10",
        9906,
        "terminal",
        1,
        protocol.ARM_GAP_L1,
        mode="smoke",
    ) == 1282527255084910333


def test_every_run_requires_separate_exact_confirmation():
    for mode in ("smoke", "formal"):
        with pytest.raises(PermissionError, match="单独授权"):
            protocol.require_run_confirmation(mode, None)
        with pytest.raises(PermissionError, match="单独授权"):
            protocol.require_run_confirmation(mode, "wrong")
        protocol.require_run_confirmation(
            mode, protocol.FROZEN_PROTOCOL_SHA256
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"execution_label": "execution_invalid"}, "execution_invalid"),
        ({"stable_gap_error_gain": False}, "no_stable_gap_error_gain"),
        ({"transition_support": False}, "insufficient_transition_support"),
        (
            {"harm_suppression": False},
            "gap_gain_without_harm_suppression",
        ),
        (
            {"good_step_retention": False},
            "gap_gain_with_good_step_suppression",
        ),
        (
            {"initial_safety": False},
            "gap_mechanism_with_transition_risk",
        ),
        (
            {"full_safety": False},
            "gap_mechanism_with_transition_risk",
        ),
        ({}, "gap_kernel_development_supported"),
    ],
)
def test_dataset_classification_obeys_frozen_precedence(kwargs, expected):
    values = {
        "execution_label": None,
        "stable_gap_error_gain": True,
        "transition_support": True,
        "harm_suppression": True,
        "good_step_retention": True,
        "initial_safety": True,
        "full_safety": True,
    }
    values.update(kwargs)
    assert protocol.classify_dataset(**values) == expected


def test_stage1_resource_decision_is_frozen():
    supported = "gap_kernel_development_supported"
    failed = "no_stable_gap_error_gain"
    assert protocol.classify_stage1({
        "test_300x10": supported
    }) == "advance_to_nltcs_gpu_protocol"
    assert protocol.classify_stage1({
        "test_300x10": failed
    }) == "stop_before_nltcs_no_test300_support"
    assert protocol.classify_stage1({
        "test_300x10": "calibration_unsupported"
    }) == "stage1_inconclusive_or_invalid"
    with pytest.raises(ValueError, match="恰好覆盖"):
        protocol.classify_stage1({
            "test_300x10": supported,
            "nltcs": supported,
        })
