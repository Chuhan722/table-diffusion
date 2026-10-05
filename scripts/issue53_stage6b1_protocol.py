"""Issue #53 第 6B-1 阶段结果盲冻结协议的可执行身份。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

if __package__:
    from scripts import issue53_stage6a_protocol as stage6a
else:
    import issue53_stage6a_protocol as stage6a


PROTOCOL_VERSION = "issue53-stage6b1a-test300-gap-l1-fixed-state-screen-v1"
RNG_DOMAIN_VERSION = "issue53-stage6b1-gap-l1-fixed-state-screen-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_Stage6B1剩余缺口感知绝对误差Gibbs固定状态筛查结果前协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "150f4cd84eefc3e0764ac739a11e4ccd1a1ca7e1a4619f08d07cb9b380bdaad0"
)
PROTOCOL_DOC_COMMIT = "3926ef76dd4dafea8da481eb21753beacb5d1f95"

# 不纳入 manifest，避免自指；在 manifest 定稿后填入。
FROZEN_PROTOCOL_SHA256 = (
    "fbeedb1abdb2be23f71b61c1fa7614f06db31ccec35c84eb02f1f27021aa487e"
)

UPSTREAM_EXECUTION_COMMIT = "3775413e0bc4684e30803c0b43c752ba57cba6a6"
UPSTREAM_PROTOCOL_DOC_SHA256 = stage6a.PROTOCOL_DOC_SHA256
UPSTREAM_PROTOCOL_SHA256 = stage6a.FROZEN_PROTOCOL_SHA256
PARENT_STAGE6B1_EXECUTION_COMMIT = (
    "1eb743a43132892e5d4e18c4bf4345e017f220fd"
)
PARENT_STAGE6B1_PROTOCOL_DOC_SHA256 = (
    "0eacb06c5d2a34d58ecbf78029da6c46accbee0385e0b24a60c41a5fe7b9f8a3"
)
PARENT_STAGE6B1_PROTOCOL_SHA256 = (
    "6087598eda6080711f532f07566be28680f86ea2059c7afe09085b1b7e30b0dc"
)

DATASET_ORDER = ("test_300x10",)
FORMAL_SEEDS = stage6a.FORMAL_SEEDS
SMOKE_SEED = stage6a.SMOKE_SEED
STATE_GROUPS = stage6a.STATE_GROUPS
PRIMARY_STATE_GROUPS = stage6a.PRIMARY_STATE_GROUPS
DATASETS = stage6a.DATASETS

ARM_INDEPENDENT = "independent_b_s0"
ARM_FACTOR = "factor_b_s8"
ARM_GAP_L1 = "gap_l1_global_s8"
ARMS = (ARM_INDEPENDENT, ARM_FACTOR, ARM_GAP_L1)
MEASUREMENT_LEGS = ("copy_only", "full")

RHO = 0.01
ETA = 0.5
MU = 0.01
B_STRENGTH = 2.0
GAP_L1_STRENGTH = 2.0
GAP_L1_FLOOR = 8.0
GAP_L1_TOLERANCE_RADIUS = 0.0
GIBBS_SWEEPS = 8
LOGIT_CLIP = 30.0

SEED_STABILITY_MINIMUM = 4
INITIAL_AND_FULL_NONINFERIOR_RATIO = 1.05
RETENTION_EQUAL_SEED_MINIMUM = 0.95
RETENTION_PER_SEED_MINIMUM = 0.90

OUTPUT_DIR = Path("outputs/issue53_stage6b1a_test300_gap_l1_screen_v1")
SMOKE_OUTPUT_DIR = Path(
    "outputs/issue53_stage6b1a_test300_gap_l1_screen_smoke_v1"
)
CALIBRATION_FILENAME = "calibration_manifest.json"
COLLECTION_FILENAME = "screen_collection.json"
STRUCTURAL_AUDIT_FILENAME = "structural_audit.json"
EVALUATION_FILENAME = "frozen_evaluation.json"
ARITHMETIC_AUDIT_FILENAME = "independent_arithmetic_audit.json"

PARENT_STAGE6B1_SMOKE_ARTIFACTS = {
    "calibration": {
        "path": Path(
            "outputs/issue53_stage6b1_gap_l1_screen_smoke_v1/"
            "calibration_manifest.json"
        ),
        "sha256": (
            "2588692c090f0c7f6282c1c63fc98106b8e17600260bd4f030b51becf13b2d92"
        ),
    },
    "collection": {
        "path": Path(
            "outputs/issue53_stage6b1_gap_l1_screen_smoke_v1/"
            "screen_collection.json"
        ),
        "sha256": (
            "7502866bf0c2fade2f895e8b072fa4b095d7d1f37cc6ac5b5e43d9fe4f573cda"
        ),
    },
    "structural_audit": {
        "path": Path(
            "outputs/issue53_stage6b1_gap_l1_screen_smoke_v1/"
            "structural_audit.json"
        ),
        "sha256": (
            "315676f60fe52ef864d0a53ab6e661bf9576d2f706aedec4d733c54f358add0f"
        ),
    },
    "frozen_evaluation": {
        "path": Path(
            "outputs/issue53_stage6b1_gap_l1_screen_smoke_v1/"
            "frozen_evaluation.json"
        ),
        "sha256": (
            "591b952c1248609672b520a6ee5f79fa0c2b11b7ee44231d85333c927181e5dd"
        ),
    },
    "independent_arithmetic_audit": {
        "path": Path(
            "outputs/issue53_stage6b1_gap_l1_screen_smoke_v1/"
            "independent_arithmetic_audit.json"
        ),
        "sha256": (
            "edfc4cdb28e396a311aa5bc3c30d19ea4d85f3766edbda6a4a6d392bbef320c8"
        ),
    },
}

SOURCE_ARTIFACTS = {
    "formal": {
        "state_library": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_v1/state_library.json"
            ),
            "sha256": "f454710014d6cd3e7c6d8ffc6327bc4f753007664c0cc917968e875a87e6d0a0",
        },
        "proposal_collection": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_v1/proposal_collection.json"
            ),
            "sha256": "1ccc406fb2bc385f927a00f8674f87b72cc6883b88b619b05fc002876a4c5960",
        },
        "structural_audit": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_v1/structural_audit.json"
            ),
            "sha256": "5faeb05ae0a8d1445eec279354c00e8b416dad6f0ff90976d0d03ce35bb38c6d",
        },
        "frozen_evaluation": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_v1/frozen_evaluation.json"
            ),
            "sha256": "972252c23b7a0edb08a712184a5f54d6e5d75c72625459dea4d4a5b9cc200b47",
        },
        "independent_arithmetic_audit": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_v1/independent_arithmetic_audit.json"
            ),
            "sha256": "7425ab85d7db96ba2fa74200eb2428759b29129b3e31223d8489ecb3d3b2beab",
        },
    },
    "smoke": {
        "state_library": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_smoke_v1/state_library.json"
            ),
            "sha256": "cc693b6906b50f374a89126adde920b64785be635a37c2ebea57965e29cf60f0",
        },
        "proposal_collection": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_smoke_v1/proposal_collection.json"
            ),
            "sha256": "b87aac8dc59683cc8daa8287dba10e1bea3d8c407fe7dd658b1dd1bb0b0d6ea2",
        },
        "structural_audit": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_smoke_v1/structural_audit.json"
            ),
            "sha256": "d8cf229a8e5e7c2a5a77034f1a083c7cc4f8ad9ca2fa5c7be5a3ffa18f28ed27",
        },
        "frozen_evaluation": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_smoke_v1/frozen_evaluation.json"
            ),
            "sha256": "f4fd6d8e44e33b1287092912a1837b757bd80b7b5f95a27eac4e4a76bac0aeff",
        },
        "independent_arithmetic_audit": {
            "path": Path(
                "outputs/issue53_stage6a_overshoot_diagnosis_smoke_v1/independent_arithmetic_audit.json"
            ),
            "sha256": "0cade5091d91e2fd88a9d171b560d86a9ecfa77db9ad3c9c1b1f61209be6a297",
        },
    },
}

NO_GATE_CONTRACT = {
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

EXECUTION_FAILURE_LABELS = (
    "execution_invalid",
    "calibration_unsupported",
    "outside_frozen_soft_scale",
)
DATASET_RESULT_LABELS = (
    *EXECUTION_FAILURE_LABELS,
    "no_stable_gap_error_gain",
    "insufficient_transition_support",
    "gap_gain_without_harm_suppression",
    "gap_gain_with_good_step_suppression",
    "gap_mechanism_with_transition_risk",
    "gap_kernel_development_supported",
)
STAGE1_DECISION_LABELS = (
    "advance_to_nltcs_gpu_protocol",
    "stop_before_nltcs_no_test300_support",
    "stage1_inconclusive_or_invalid",
)


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, float) and math.isinf(value):
        return "positive_infinity"
    return value


def mode_seeds(mode: str) -> tuple[int, ...]:
    if mode == "formal":
        return FORMAL_SEEDS
    if mode == "smoke":
        return (SMOKE_SEED,)
    raise ValueError("mode 必须是 formal 或 smoke")


def proposals_per_state(dataset: str, *, mode: str) -> int:
    return stage6a.proposals_per_state(dataset, mode=mode)


def state_id(dataset: str, seed: int, group: str, *, mode: str) -> str:
    return stage6a.state_id(dataset, seed, group, mode=mode)


def pair_id(
    dataset: str,
    seed: int,
    group: str,
    proposal_index: int,
    *,
    mode: str,
) -> str:
    return (
        f"{state_id(dataset, seed, group, mode=mode)}"
        f"__proposal_{proposal_index:04d}"
    )


def expected_pair_ids(mode: str) -> tuple[str, ...]:
    return tuple(
        pair_id(dataset, seed, group, proposal_index, mode=mode)
        for dataset in DATASET_ORDER
        for seed in mode_seeds(mode)
        for group in STATE_GROUPS
        for proposal_index in range(
            proposals_per_state(dataset, mode=mode)
        )
    )


def gibbs_address_seed(
    dataset: str,
    seed: int,
    group: str,
    proposal_index: int,
    arm: str,
    *,
    mode: str,
) -> int:
    """派生互相隔离且不占用主随机流的 64 位吉布斯地址。"""

    if arm not in (ARM_FACTOR, ARM_GAP_L1):
        raise ValueError("arm 必须是两个吉布斯组之一")
    # 先借上游地址校验完整冻结坐标边界。
    stage6a.proposal_address_seed(
        dataset, seed, group, proposal_index, "update", mode=mode
    )
    payload = _strict_json_bytes({
        "domain": RNG_DOMAIN_VERSION,
        "stream": (
            "factor_b_gibbs_rng" if arm == ARM_FACTOR
            else "gap_l1_gibbs_rng"
        ),
        "mode": mode,
        "dataset": dataset,
        "source_seed": seed,
        "state_group": group,
        "proposal_index": proposal_index,
    })
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _source_manifest(mode: str) -> dict[str, Any]:
    return {
        name: {
            "path": str(binding["path"]),
            "sha256": binding["sha256"],
        }
        for name, binding in SOURCE_ARTIFACTS[mode].items()
    }


def frozen_protocol_manifest() -> dict[str, Any]:
    formal_pairs = len(expected_pair_ids("formal"))
    smoke_pairs = len(expected_pair_ids("smoke"))
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6B-1A_test300_gap_l1_fixed_state_development_screen",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
            "commit": PROTOCOL_DOC_COMMIT,
        },
        "upstream_stage6a": {
            "execution_commit": UPSTREAM_EXECUTION_COMMIT,
            "protocol_doc_sha256": UPSTREAM_PROTOCOL_DOC_SHA256,
            "protocol_sha256": UPSTREAM_PROTOCOL_SHA256,
            "formal_artifacts": _source_manifest("formal"),
            "smoke_artifacts": _source_manifest("smoke"),
        },
        "parent_stage6b1": {
            "execution_commit": PARENT_STAGE6B1_EXECUTION_COMMIT,
            "protocol_doc_sha256": PARENT_STAGE6B1_PROTOCOL_DOC_SHA256,
            "protocol_sha256": PARENT_STAGE6B1_PROTOCOL_SHA256,
            "smoke_artifacts": {
                name: {
                    "path": str(binding["path"]),
                    "sha256": binding["sha256"],
                }
                for name, binding in PARENT_STAGE6B1_SMOKE_ARTIFACTS.items()
            },
        },
        "dataset_order": list(DATASET_ORDER),
        "gibbs_rng_domain_version": RNG_DOMAIN_VERSION,
        "execution_scope": {
            "stage": "test300_first",
            "only_formal_dataset": "test_300x10",
            "nltcs_dataset_runtime_constructed": False,
            "cross_dataset_claim_allowed": False,
            "stage1_result_may_only_control_stage2_protocol_preparation": True,
            "stage2_nltcs_authorized": False,
            "stage2_physical_gpu": 1,
            "stage2_gap_kernel_cpu_path_allowed": False,
            "stage2_gap_replay_audit_cpu_path_allowed": False,
            "stage2_exact_rational_final_check_may_use_cpu": True,
            "stage2_parameters_may_change_from_stage1_result": False,
        },
        "formal_seeds": list(FORMAL_SEEDS),
        "smoke_seed": SMOKE_SEED,
        "state_groups": list(STATE_GROUPS),
        "primary_state_groups": list(PRIMARY_STATE_GROUPS),
        "arms": list(ARMS),
        "measurement_legs": list(MEASUREMENT_LEGS),
        "fixed_parameters": {
            "rho": RHO,
            "eta": ETA,
            "mu": MU,
            "b_strength": B_STRENGTH,
            "gap_l1_strength": GAP_L1_STRENGTH,
            "gap_l1_floor": GAP_L1_FLOOR,
            "gap_l1_tolerance_radius": GAP_L1_TOLERANCE_RADIUS,
            "gibbs_sweeps": GIBBS_SWEEPS,
            "logit_clip": [-LOGIT_CLIP, LOGIT_CLIP],
            "outer_order": "donor_then_participation_unchanged",
            "mutation_position": "after_copy",
            "source_donor_replay_backend": "per_state_stage6a_runtime_device",
            "new_kernel_backend": "numpy_float64_cpu",
        },
        "gap_l1_objective": (
            "mean_j(abs(y_j-(q_j+D_j(M)))/max(y_j,8))"
        ),
        "gap_l1_condition_score": "E(g=0)-E(g=1)",
        "gap_l1_condition_logit": (
            "clip(logit(eta)+2*score/R,-30,30)"
        ),
        "calibration": {
            "scope": "one_R_per_dataset_and_source_seed",
            "state_group": "initial",
            "address_search": "ascending_frozen_proposal_index",
            "scores": "all_differing_row_attribute_isolated_E0_minus_E1",
            "zero_scores_excluded": True,
            "zero_score_authority": (
                "exact_scaled_integer_rational_query_counts"
            ),
            "nonzero_score_magnitudes": "numpy_float64_cpu",
            "signs_retained": True,
            "scale": "stable_RMS_of_nonzero_scores",
            "fixed_across_five_states_and_all_addresses": True,
            "no_fallback_scale": True,
        },
        "random_scan": {
            "active_domain": "participating_and_current_not_equal_donor",
            "coordinate_sampling": "uniform_with_replacement",
            "microsteps": "8*K",
            "k_zero_consumes_no_gibbs_rng": True,
            "factor_stream": "sha256_domain_derived_uint64_little_endian",
            "gap_l1_stream": "sha256_domain_derived_uint64_little_endian",
            "streams_disjoint_by_named_domain": True,
            "microstep_trace_format": (
                "issue53_gap_l1_microstep_trace_le_v1"
            ),
        },
        "no_gate_contract": dict(NO_GATE_CONTRACT),
        "formal_matrix": {
            "state_count": 25,
            "pair_count": formal_pairs,
            "arm_leg_record_count": formal_pairs * len(ARMS) * 2,
            "dataset_effect_claim": "test_300x10_only",
            "cross_dataset_claim": None,
        },
        "smoke_matrix": {
            "state_count": 5,
            "pair_count": smoke_pairs,
            "arm_leg_record_count": smoke_pairs * len(ARMS) * 2,
            "formal_result_valid": False,
            "mechanism_evidence_emitted": False,
            "artifact_role": "pipeline_smoke_only",
            "final_screen_classification": None,
        },
        "thresholds": {
            "strict_mean_gap_improvement": True,
            "strict_mean_harm_reduction": True,
            "seed_stability_minimum_of_five": SEED_STABILITY_MINIMUM,
            "initial_noninferior_ratio": INITIAL_AND_FULL_NONINFERIOR_RATIO,
            "full_noninferior_ratio": INITIAL_AND_FULL_NONINFERIOR_RATIO,
            "retention_equal_seed_minimum": RETENTION_EQUAL_SEED_MINIMUM,
            "retention_per_seed_minimum": RETENTION_PER_SEED_MINIMUM,
        },
        "execution_failures": list(EXECUTION_FAILURE_LABELS),
        "dataset_result_labels": list(DATASET_RESULT_LABELS),
        "stage1_decision_labels": list(STAGE1_DECISION_LABELS),
        "stage1_decision_rule": {
            "gap_kernel_development_supported": (
                "advance_to_nltcs_gpu_protocol"
            ),
            "other_valid_dataset_label": (
                "stop_before_nltcs_no_test300_support"
            ),
            "execution_failure_label": "stage1_inconclusive_or_invalid",
        },
        "result_blind_pipeline": [
            "calibration_manifest",
            "three_arm_collection",
            "structural_audit",
            "frozen_evaluation",
            "independent_arithmetic_audit",
        ],
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> None:
    root = Path(repository_root)
    # 同时验证第 6A 阶段协议和 schema/query/marginal 文件身份；第 6B-1
    # 阶段只复用这些输入，不允许当前工作树中的同名文件静默漂移。
    stage6a.assert_frozen_protocol_identity(root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("第 6B-1 阶段协议文档身份漂移")
    if protocol_sha256() != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("第 6B-1 阶段协议 manifest 身份漂移")
    for name, binding in PARENT_STAGE6B1_SMOKE_ARTIFACTS.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"原第 6B-1 阶段小规模产物漂移：{name}")


def assert_source_artifact_identities(
    repository_root: str | Path,
    mode: str,
) -> dict[str, str]:
    root = Path(repository_root)
    mode_seeds(mode)
    observed = {}
    for name, binding in SOURCE_ARTIFACTS[mode].items():
        digest = file_sha256(root / binding["path"])
        if digest != binding["sha256"]:
            raise RuntimeError(f"第 6A 阶段来源产物漂移：{name}")
        observed[name] = digest
    return observed


def require_run_confirmation(mode: str, confirmed_sha256: str | None) -> None:
    mode_seeds(mode)
    if confirmed_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError(
            f"{mode} 运行需要用户单独授权并精确确认冻结协议 SHA-256"
        )


def classify_dataset(
    *,
    execution_label: str | None,
    stable_gap_error_gain: bool,
    transition_support: bool,
    harm_suppression: bool,
    good_step_retention: bool,
    initial_safety: bool,
    full_safety: bool,
) -> str:
    if execution_label is not None:
        if execution_label not in EXECUTION_FAILURE_LABELS:
            raise ValueError("未知执行资格标签")
        return execution_label
    if not stable_gap_error_gain:
        return "no_stable_gap_error_gain"
    if not transition_support:
        return "insufficient_transition_support"
    if not harm_suppression:
        return "gap_gain_without_harm_suppression"
    if not good_step_retention:
        return "gap_gain_with_good_step_suppression"
    if not initial_safety or not full_safety:
        return "gap_mechanism_with_transition_risk"
    return "gap_kernel_development_supported"


def classify_stage1(labels: Mapping[str, str]) -> str:
    if set(labels) != set(DATASET_ORDER):
        raise ValueError("第一阶段标签必须恰好覆盖小数据集")
    if any(label not in DATASET_RESULT_LABELS for label in labels.values()):
        raise ValueError("包含未知数据集标签")
    if any(label in EXECUTION_FAILURE_LABELS for label in labels.values()):
        return "stage1_inconclusive_or_invalid"
    if labels["test_300x10"] == "gap_kernel_development_supported":
        return "advance_to_nltcs_gpu_protocol"
    return "stop_before_nltcs_no_test300_support"


def build_plan(mode: str) -> dict[str, Any]:
    pairs = len(expected_pair_ids(mode))
    return {
        "mode": "plan_only_no_source_read_no_generation",
        "requested_mode": mode,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "datasets": list(DATASET_ORDER),
        "seeds": list(mode_seeds(mode)),
        "state_groups": list(STATE_GROUPS),
        "arms": list(ARMS),
        "pair_count": pairs,
        "arm_leg_record_count": pairs * len(ARMS) * len(MEASUREMENT_LEGS),
        "generation_started": False,
        "confirmation_consumed": False,
    }
