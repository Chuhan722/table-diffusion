"""Issue #53 第 6B-1B 阶段 NLTCS 显卡差量协议的可执行身份。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

if __package__:
    from scripts import issue53_stage6b1_protocol as upstream
    from scripts.check_issue53_stage6b1b_nltcs_gpu_environment import (
        EXPECTED_GPU_NAME,
        EXPECTED_GPU_UUID,
        EXPECTED_PHYSICAL_GPU,
        validate_gpu_environment,
    )
else:
    import issue53_stage6b1_protocol as upstream
    from check_issue53_stage6b1b_nltcs_gpu_environment import (
        EXPECTED_GPU_NAME,
        EXPECTED_GPU_UUID,
        EXPECTED_PHYSICAL_GPU,
        validate_gpu_environment,
    )


stage6a = upstream.stage6a

PROTOCOL_VERSION = "issue53-stage6b1b-nltcs-gap-l1-gpu-screen-v1"
RNG_DOMAIN_VERSION = upstream.RNG_DOMAIN_VERSION
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_Stage6B1B_NLTCS显卡固定状态筛查差量协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "fd97fd9049465217005e58728aa0c9d3b05b281022240a20aafe61d3c38947e4"
)
PROTOCOL_DOC_COMMIT = "55ab7a982d1a808b7c9ac87d8876eadf9ea45069"

# 不纳入 manifest（执行清单），避免自指；清单定稿后填入。
FROZEN_PROTOCOL_SHA256 = (
    "754fbf02fdaeacb716fae0be90ee1b3d8fd27f1b8dd0c8faa1c73a11ddc63387"
)

UPSTREAM_STAGE6B1A_EXECUTION_COMMIT = (
    "a6c32da1a7a5e8815510f4aaaba364c6032263a0"
)
UPSTREAM_STAGE6B1A_PROTOCOL_DOC_SHA256 = upstream.PROTOCOL_DOC_SHA256
UPSTREAM_STAGE6B1A_PROTOCOL_SHA256 = upstream.FROZEN_PROTOCOL_SHA256
UPSTREAM_STAGE6B1A_RESULTS = {
    "frozen_evaluation": {
        "path": Path(
            "outputs/issue53_stage6b1a_test300_gap_l1_screen_v1/"
            "frozen_evaluation.json"
        ),
        "sha256": (
            "7811b8a834babb5167097a4ff0f9b6c2caecea98abdad821cd692a73ca4e54e4"
        ),
    },
    "independent_arithmetic_audit": {
        "path": Path(
            "outputs/issue53_stage6b1a_test300_gap_l1_screen_v1/"
            "independent_arithmetic_audit.json"
        ),
        "sha256": (
            "6813fc56fd49d5e726c72029de7f25f6456ff8558e3df051a1d11b99e7380ed3"
        ),
    },
}

DATASET_ORDER = ("nltcs",)
FORMAL_SEEDS = upstream.FORMAL_SEEDS
SMOKE_SEED = upstream.SMOKE_SEED
STATE_GROUPS = upstream.STATE_GROUPS
PRIMARY_STATE_GROUPS = upstream.PRIMARY_STATE_GROUPS
DATASETS = upstream.DATASETS

ARM_INDEPENDENT = upstream.ARM_INDEPENDENT
ARM_FACTOR = upstream.ARM_FACTOR
ARM_GAP_L1 = upstream.ARM_GAP_L1
ARMS = upstream.ARMS
MEASUREMENT_LEGS = upstream.MEASUREMENT_LEGS

RHO = upstream.RHO
ETA = upstream.ETA
MU = upstream.MU
B_STRENGTH = upstream.B_STRENGTH
GAP_L1_STRENGTH = upstream.GAP_L1_STRENGTH
GAP_L1_FLOOR = upstream.GAP_L1_FLOOR
GAP_L1_TOLERANCE_RADIUS = upstream.GAP_L1_TOLERANCE_RADIUS
GIBBS_SWEEPS = upstream.GIBBS_SWEEPS
LOGIT_CLIP = upstream.LOGIT_CLIP

SEED_STABILITY_MINIMUM = upstream.SEED_STABILITY_MINIMUM
INITIAL_AND_FULL_NONINFERIOR_RATIO = (
    upstream.INITIAL_AND_FULL_NONINFERIOR_RATIO
)
RETENTION_EQUAL_SEED_MINIMUM = upstream.RETENTION_EQUAL_SEED_MINIMUM
RETENTION_PER_SEED_MINIMUM = upstream.RETENTION_PER_SEED_MINIMUM

OUTPUT_DIR = Path("outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_screen_v1")
SMOKE_OUTPUT_DIR = Path(
    "outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_screen_smoke_v1"
)
CALIBRATION_FILENAME = upstream.CALIBRATION_FILENAME
COLLECTION_FILENAME = upstream.COLLECTION_FILENAME
STRUCTURAL_AUDIT_FILENAME = upstream.STRUCTURAL_AUDIT_FILENAME
EVALUATION_FILENAME = upstream.EVALUATION_FILENAME
ARITHMETIC_AUDIT_FILENAME = upstream.ARITHMETIC_AUDIT_FILENAME

SOURCE_ARTIFACTS = upstream.SOURCE_ARTIFACTS
NO_GATE_CONTRACT = upstream.NO_GATE_CONTRACT
EXECUTION_FAILURE_LABELS = upstream.EXECUTION_FAILURE_LABELS
DATASET_RESULT_LABELS = upstream.DATASET_RESULT_LABELS
FINAL_SCREEN_LABELS = (
    "shared_development_support",
    "dataset_dependent_development_support",
    "inconclusive_or_invalid_screen",
)

GAP_L1_DEVICE = "cuda"
NEW_KERNEL_BACKEND = "torch_cuda_float64"

canonical_sha256 = upstream.canonical_sha256
file_sha256 = upstream.file_sha256
mode_seeds = upstream.mode_seeds
proposals_per_state = upstream.proposals_per_state
state_id = upstream.state_id
pair_id = upstream.pair_id
gibbs_address_seed = upstream.gibbs_address_seed
classify_dataset = upstream.classify_dataset


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


def _artifact_manifest(bindings: Mapping[str, Mapping[str, Any]]) -> dict:
    return {
        name: {
            "path": str(binding["path"]),
            "sha256": binding["sha256"],
        }
        for name, binding in bindings.items()
    }


def frozen_protocol_manifest() -> dict[str, Any]:
    formal_pairs = len(expected_pair_ids("formal"))
    smoke_pairs = len(expected_pair_ids("smoke"))
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6B-1B_nltcs_gap_l1_gpu_fixed_state_screen",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
            "commit": PROTOCOL_DOC_COMMIT,
        },
        "inherits_stage6b1a_without_parameter_changes": {
            "execution_commit": UPSTREAM_STAGE6B1A_EXECUTION_COMMIT,
            "protocol_doc_sha256": UPSTREAM_STAGE6B1A_PROTOCOL_DOC_SHA256,
            "protocol_sha256": UPSTREAM_STAGE6B1A_PROTOCOL_SHA256,
            "eligibility_artifacts": _artifact_manifest(
                UPSTREAM_STAGE6B1A_RESULTS
            ),
            "dataset_classification": "gap_kernel_development_supported",
            "resource_decision": "advance_to_nltcs_gpu_protocol",
        },
        "source_stage6a_artifacts": {
            mode: _artifact_manifest(bindings)
            for mode, bindings in SOURCE_ARTIFACTS.items()
        },
        "dataset_order": list(DATASET_ORDER),
        "formal_seeds": list(FORMAL_SEEDS),
        "smoke_seed": SMOKE_SEED,
        "state_groups": list(STATE_GROUPS),
        "primary_state_groups": list(PRIMARY_STATE_GROUPS),
        "arms": list(ARMS),
        "measurement_legs": list(MEASUREMENT_LEGS),
        "inherited_method_identity": {
            "rho": RHO,
            "eta": ETA,
            "mu": MU,
            "b_strength": B_STRENGTH,
            "gap_l1_strength": GAP_L1_STRENGTH,
            "gap_l1_floor": GAP_L1_FLOOR,
            "gibbs_sweeps": GIBBS_SWEEPS,
            "logit_clip": [-LOGIT_CLIP, LOGIT_CLIP],
            "rng_domain_version": RNG_DOMAIN_VERSION,
            "no_gate_contract": dict(NO_GATE_CONTRACT),
        },
        "gpu_delta": {
            "physical_gpu": int(EXPECTED_PHYSICAL_GPU),
            "physical_gpu_uuid": EXPECTED_GPU_UUID,
            "gpu_name": EXPECTED_GPU_NAME,
            "logical_device": "cuda:0",
            "visible_device_count": 1,
            "cuda_visible_devices_exact": EXPECTED_PHYSICAL_GPU,
            "new_kernel_backend": NEW_KERNEL_BACKEND,
            "condition_float_dtype": "float64",
            "query_count_dtype": "int64",
            "switch_dtype": "bool",
            "deterministic_algorithms_required": True,
            "automatic_cpu_fallback": False,
            "random_tape_order": "coordinate_then_uniform_per_microstep",
            "random_tape_transfer": "one_complete_address",
            "formal_address_batch_size": 1,
            "sequential_microstep_dependency_preserved": True,
            "existing_cuda_outer_paths_reused": True,
            "public_generator_or_default_kernel_changed": False,
            "independent_cuda_replay_imports_production_gap_math": False,
        },
        "formal_matrix": {
            "state_count": 25,
            "pair_count": formal_pairs,
            "arm_leg_record_count": formal_pairs * len(ARMS) * 2,
        },
        "smoke_matrix": {
            "state_count": 5,
            "pair_count": smoke_pairs,
            "arm_leg_record_count": smoke_pairs * len(ARMS) * 2,
            "artifact_role": "pipeline_smoke_only",
            "formal_result_valid": False,
            "mechanism_evidence_emitted": False,
            "final_screen_classification": None,
        },
        "final_screen_labels": list(FINAL_SCREEN_LABELS),
        "result_blind_pipeline": [
            "calibration_manifest",
            "three_arm_collection",
            "structural_audit",
            "frozen_evaluation",
            "independent_cuda_arithmetic_audit",
        ],
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> None:
    root = Path(repository_root)
    stage6a.assert_frozen_protocol_identity(root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("第 6B-1B 阶段差量协议文档身份漂移")
    if (
        file_sha256(root / upstream.PROTOCOL_DOC)
        != UPSTREAM_STAGE6B1A_PROTOCOL_DOC_SHA256
        or upstream.protocol_sha256() != UPSTREAM_STAGE6B1A_PROTOCOL_SHA256
    ):
        raise RuntimeError("第 6B-1A 阶段上游协议身份漂移")
    if protocol_sha256() != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("第 6B-1B 阶段执行清单身份漂移")
    for name, binding in UPSTREAM_STAGE6B1A_RESULTS.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"第 6B-1A 阶段进入资格产物漂移：{name}")


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
            f"{mode} 运行需要用户单独授权并精确确认冻结执行清单"
        )


def validate_runtime_environment(mode: str) -> dict[str, Any]:
    mode_seeds(mode)
    return {"stage6b1b_gpu": validate_gpu_environment()}


def classify_stage1(labels: Mapping[str, str]) -> str:
    """保留旧管线调用名，执行第 6B-1B 两数据合并规则。"""

    if set(labels) != {"nltcs"}:
        raise ValueError("第二阶段标签必须恰好覆盖 NLTCS")
    label = labels["nltcs"]
    if label not in DATASET_RESULT_LABELS:
        raise ValueError("包含未知 NLTCS 数据集标签")
    if label in EXECUTION_FAILURE_LABELS:
        return "inconclusive_or_invalid_screen"
    if label == "gap_kernel_development_supported":
        return "shared_development_support"
    return "dataset_dependent_development_support"


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
        "new_kernel_backend": NEW_KERNEL_BACKEND,
        "physical_gpu": int(EXPECTED_PHYSICAL_GPU),
        "source_read_started": False,
        "generation_started": False,
        "confirmation_consumed": False,
    }
