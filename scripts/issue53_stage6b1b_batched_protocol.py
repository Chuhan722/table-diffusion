"""Issue #53 第 6B-1B 阶段 NLTCS 显卡批量执行差量协议。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

if __package__:
    from scripts import issue53_stage6b1b_protocol as base
else:
    import issue53_stage6b1b_protocol as base


PROTOCOL_VERSION = (
    "issue53-stage6b1b-nltcs-gap-l1-gpu-batched-execution-v2"
)
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_Stage6B1B_NLTCS显卡批量执行差量协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "6fdcefb5db152c9ec7b3f507cbbe14750d76d85c6da92d51f6e3865b72531474"
)
PROTOCOL_DOC_COMMIT = "fd6031043f606fbc2073b55abf6b1714ed99cb0e"

# 不纳入 manifest（执行清单），避免自指；清单定稿后填入。
FROZEN_PROTOCOL_SHA256 = (
    "ecdd49ada43f4da578f901538a25339e6fce9b99b6718e09ee91af9e1fa17d00"
)

BASE_PROTOCOL_VERSION = base.PROTOCOL_VERSION
BASE_PROTOCOL_DOC_SHA256 = base.PROTOCOL_DOC_SHA256
BASE_PROTOCOL_SHA256 = base.FROZEN_PROTOCOL_SHA256

BATCH_EXECUTION_COMMIT = "104d2c8c09e86f70e5eb91e445ba8c181ead376d"
BATCH_EXECUTION_SOURCES = {
    "production_batched_gap_kernel": {
        "path": Path("src/table_diffevo/gap_l1_diffusion.py"),
        "sha256": (
            "8c0fd1f8233edf8c756d9036e0ebce7289f45c5f7c5ece100fd7acc585d83b47"
        ),
    },
    "independent_batched_arithmetic_audit": {
        "path": Path("scripts/issue53_stage6b1b_independent_cuda.py"),
        "sha256": (
            "d02e6698eb421c8c2ad64c6b4f41ed003c337b873c0bc2ff9b7865bfac2f897b"
        ),
    },
}

PRODUCTION_BATCH_EXECUTION_FORMAT = (
    "issue53_gap_l1_batched_cuda_float64_v2"
)
PRODUCTION_BATCH_BACKEND = "torch_cuda_float64_batched"
INDEPENDENT_BATCH_AUDIT_FORMAT = (
    "issue53_gap_l1_independent_batched_audit_v1"
)
INDEPENDENT_BATCH_BACKEND = "independent_torch_cuda_float64_batched"
MICROSTEP_TRACE_FORMAT = "issue53_gap_l1_microstep_trace_le_v1"

FORMAL_ADDRESS_BATCH_SIZE = 20
SMOKE_ADDRESS_BATCH_SIZE = 2
FORMAL_STATE_BATCH_COUNT = 25
SMOKE_STATE_BATCH_COUNT = 5

OUTPUT_DIR = Path(
    "outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_v2"
)
SMOKE_OUTPUT_DIR = Path(
    "outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_smoke_v2"
)
BATCHED_GAP_EXECUTION = True
PIPELINE_WIRING_AVAILABLE = True
COLLECTION_GAP_BACKEND = PRODUCTION_BATCH_BACKEND
CALIBRATION_GAP_BACKEND = base.NEW_KERNEL_BACKEND

canonical_sha256 = base.canonical_sha256
file_sha256 = base.file_sha256
mode_seeds = base.mode_seeds
expected_pair_ids = base.expected_pair_ids


def __getattr__(name: str) -> Any:
    """未被批量差量覆盖的协议字段严格继承旧协议。"""

    return getattr(base, name)


def address_batch_size(mode: str) -> int:
    mode_seeds(mode)
    return (
        FORMAL_ADDRESS_BATCH_SIZE
        if mode == "formal"
        else SMOKE_ADDRESS_BATCH_SIZE
    )


def assert_stage_output_path(
    repository_root: str | Path,
    mode: str,
    stage: str,
    output_path: str | Path,
) -> None:
    mode_seeds(mode)
    filenames = {
        "calibration": base.CALIBRATION_FILENAME,
        "collection": base.COLLECTION_FILENAME,
        "structural_audit": base.STRUCTURAL_AUDIT_FILENAME,
        "evaluation": base.EVALUATION_FILENAME,
        "arithmetic_audit": base.ARITHMETIC_AUDIT_FILENAME,
    }
    if stage not in filenames:
        raise ValueError(f"未知批量管线阶段：{stage}")
    directory = OUTPUT_DIR if mode == "formal" else SMOKE_OUTPUT_DIR
    expected = (Path(repository_root) / directory / filenames[stage]).resolve()
    if Path(output_path).resolve() != expected:
        raise ValueError(f"批量协议 {stage} 输出必须使用全新冻结目录")


def _artifact_manifest(bindings: dict[str, dict[str, Any]]) -> dict:
    return {
        name: {
            "path": str(binding["path"]),
            "sha256": binding["sha256"],
        }
        for name, binding in bindings.items()
    }


def frozen_protocol_manifest() -> dict[str, Any]:
    base_manifest = base.frozen_protocol_manifest()
    formal_pairs = len(expected_pair_ids("formal"))
    smoke_pairs = len(expected_pair_ids("smoke"))
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "6B-1B_nltcs_gap_l1_gpu_batched_execution_delta",
        "protocol_document": {
            "path": str(PROTOCOL_DOC),
            "sha256": PROTOCOL_DOC_SHA256,
            "commit": PROTOCOL_DOC_COMMIT,
        },
        "inherits_frozen_single_address_protocol": {
            "contract_version": BASE_PROTOCOL_VERSION,
            "protocol_document_sha256": BASE_PROTOCOL_DOC_SHA256,
            "protocol_sha256": BASE_PROTOCOL_SHA256,
            "method_parameters_changed": False,
            "queries_addresses_arms_metrics_or_thresholds_changed": False,
            "no_gate_contract": dict(base.NO_GATE_CONTRACT),
        },
        "batch_execution_implementation": {
            "commit": BATCH_EXECUTION_COMMIT,
            "sources": _artifact_manifest(BATCH_EXECUTION_SOURCES),
            "production_format": PRODUCTION_BATCH_EXECUTION_FORMAT,
            "production_backend": PRODUCTION_BATCH_BACKEND,
            "independent_audit_format": INDEPENDENT_BATCH_AUDIT_FORMAT,
            "independent_audit_backend": INDEPENDENT_BATCH_BACKEND,
            "microstep_trace_format": MICROSTEP_TRACE_FORMAT,
            "production_imports_independent_batch_arithmetic": False,
            "independent_imports_production_batch_arithmetic": False,
            "artificial_prototype_is_formal_entry": False,
        },
        "fixed_batch_partition": {
            "unit": "one_frozen_state_gap_l1_arm",
            "address_order": "proposal_index_ascending",
            "formal_address_batch_size": FORMAL_ADDRESS_BATCH_SIZE,
            "formal_state_batch_count": FORMAL_STATE_BATCH_COUNT,
            "smoke_address_batch_size": SMOKE_ADDRESS_BATCH_SIZE,
            "smoke_state_batch_count": SMOKE_STATE_BATCH_COUNT,
            "cross_state_batching": False,
            "sort_or_group_by_k_donor_or_result": False,
            "adaptive_batch_resize": False,
            "automatic_single_address_fallback": False,
            "automatic_cpu_fallback": False,
        },
        "per_address_execution_identity": {
            "random_tape_order": "coordinate_then_uniform_per_microstep",
            "random_draws": "n_sweeps_times_k",
            "k_zero_random_draws": 0,
            "strict_sequential_dependency": True,
            "variable_length_padding": "batch_maximum_microsteps",
            "padding_changes_state": False,
            "padding_consumes_rng": False,
            "padding_emits_trace": False,
            "output_order_matches_input": True,
            "initial_and_endpoint_rng_sha256_exact": True,
            "query_count_dtype": "int64",
            "switch_dtype": "bool",
            "condition_float_dtype": "float64",
            "deterministic_algorithms_required": True,
        },
        "execution_equivalence": {
            "legacy_single_trace_sha256_must_match": False,
            "legacy_single_float_absolute_tolerance": 1e-12,
            "legacy_single_sampling_flip_tolerance": 0,
            "legacy_single_discrete_outputs_exact": True,
            "production_vs_independent_batch_float_tolerance": 0.0,
            "production_vs_independent_batch_exact_fields": [
                "valid_step_mask",
                "coordinates",
                "random_rolls",
                "initial_rng_sha256",
                "endpoint_rng_sha256",
                "seven_float64_microstep_values",
                "three_boolean_microstep_values",
                "per_address_trace_sha256",
                "final_mask",
                "final_query_counts",
                "final_table",
            ],
            "nonfinite_value_tolerance": 0,
            "exact_zero_or_one_probability_tolerance": 0,
            "full_recount_mismatch_tolerance": 0,
        },
        "no_gate_execution": {
            "no_gate": True,
            "acceptance_or_rejection": False,
            "candidate_rollback": False,
            "winner_selection": False,
            "result_dependent_address_filtering": False,
            "freeze_address_or_batch_inside_noise_range": False,
        },
        "formal_matrix": {
            "state_count": 25,
            "pair_count": formal_pairs,
            "gap_l1_batch_count": FORMAL_STATE_BATCH_COUNT,
            "arm_leg_record_count": base_manifest["formal_matrix"][
                "arm_leg_record_count"
            ],
        },
        "smoke_matrix": {
            "state_count": 5,
            "pair_count": smoke_pairs,
            "gap_l1_batch_count": SMOKE_STATE_BATCH_COUNT,
            "arm_leg_record_count": base_manifest["smoke_matrix"][
                "arm_leg_record_count"
            ],
            "artifact_role": "pipeline_smoke_only",
            "formal_result_valid": False,
            "mechanism_evidence_emitted": False,
            "final_screen_classification": None,
        },
        "output_isolation": {
            "formal_output_dir": str(OUTPUT_DIR),
            "smoke_output_dir": str(SMOKE_OUTPUT_DIR),
            "legacy_interrupted_output_dir": str(base.OUTPUT_DIR),
            "resume_or_copy_legacy_interrupted_output": False,
            "fresh_calibration_required": True,
        },
        "authorization_boundary": {
            "formal_pipeline_wired": False,
            "smoke_run_authorized": False,
            "formal_500_address_run_authorized": False,
            "public_generator_or_default_kernel_changed": False,
        },
    }


def protocol_sha256() -> str:
    return canonical_sha256(frozen_protocol_manifest())


def assert_frozen_protocol_identity(repository_root: str | Path) -> None:
    root = Path(repository_root)
    base.assert_frozen_protocol_identity(root)
    if file_sha256(root / PROTOCOL_DOC) != PROTOCOL_DOC_SHA256:
        raise RuntimeError("第 6B-1B 阶段批量差量协议文档身份漂移")
    for name, binding in BATCH_EXECUTION_SOURCES.items():
        if file_sha256(root / binding["path"]) != binding["sha256"]:
            raise RuntimeError(f"第 6B-1B 阶段批量执行来源漂移：{name}")
    if protocol_sha256() != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError("第 6B-1B 阶段批量执行清单身份漂移")


def require_run_confirmation(mode: str, confirmed_sha256: str | None) -> None:
    mode_seeds(mode)
    if confirmed_sha256 != FROZEN_PROTOCOL_SHA256:
        raise PermissionError(
            f"{mode} 运行需要用户单独授权并精确确认批量冻结执行清单"
        )


def build_plan(mode: str) -> dict[str, Any]:
    seeds = mode_seeds(mode)
    address_batch_size = (
        FORMAL_ADDRESS_BATCH_SIZE
        if mode == "formal"
        else SMOKE_ADDRESS_BATCH_SIZE
    )
    state_batch_count = (
        FORMAL_STATE_BATCH_COUNT
        if mode == "formal"
        else SMOKE_STATE_BATCH_COUNT
    )
    pairs = len(expected_pair_ids(mode))
    return {
        "mode": "plan_only_no_source_read_no_generation",
        "requested_mode": mode,
        "protocol_sha256": FROZEN_PROTOCOL_SHA256,
        "datasets": list(base.DATASET_ORDER),
        "seeds": list(seeds),
        "state_groups": list(base.STATE_GROUPS),
        "arms": list(base.ARMS),
        "pair_count": pairs,
        "arm_leg_record_count": (
            pairs * len(base.ARMS) * len(base.MEASUREMENT_LEGS)
        ),
        "gap_l1_address_batch_size": address_batch_size,
        "gap_l1_state_batch_count": state_batch_count,
        "production_batch_backend": PRODUCTION_BATCH_BACKEND,
        "calibration_gap_backend": CALIBRATION_GAP_BACKEND,
        "pipeline_wiring_available": PIPELINE_WIRING_AVAILABLE,
        "formal_pipeline_wired_at_protocol_freeze": False,
        "source_read_started": False,
        "generation_started": False,
        "confirmation_consumed": False,
    }
