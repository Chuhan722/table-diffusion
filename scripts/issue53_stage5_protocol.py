"""Result-blind frozen protocol for Issue #53 Stage 5 kernel A/B."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "issue53-stage5-kernel-ab-v1"
PROTOCOL_DOC = Path(
    "docs/设计/Issue53_Stage5同温度独立核与factor核外层公平比较结果前冻结协议.md"
)
PROTOCOL_DOC_SHA256 = (
    "4d0ffb8ebf77006becea00849eef452174559aea62faeb86dd70c227e3fc7fab"
)
PROTOCOL_DOC_COMMIT = "8fcc65f12e9891c98f8ca5e4e8224a9ae47b28b1"

# Filled only after the manifest below is frozen.  It is deliberately not part
# of the manifest, so the identity has no self-reference.
FROZEN_PROTOCOL_SHA256 = (
    "1d447be0fb0ce9a2c7707abd2e259ed6ea41edbf3780f30426e320b1bad94f1c"
)

STAGE4_PROTOCOL_SHA256 = (
    "6a2834db4cb75fffbaac3330bbb1923fa2e864572ca05ac901c3142ecd443680"
)
STAGE4_EXECUTION_COMMIT = "e7cbcc0beaf6718f7f4ad148a8ee07f8cc9a089f"
STAGE4_STATE_LIBRARY_SHA256 = (
    "3c7475e89d693bd2240846bb78dec6b7a8d2abc14a71beea25fd6be3e2a02561"
)
STAGE4_MIXING_REPORT_SHA256 = (
    "15751180b96c6a466f7096a63935d93eb60f47b836f2aa9462a1842ee58b7fa5"
)
STAGE4_AUDIT_SHA256 = (
    "fde7929a39cb039d26dd56b91063fa4151a639ea0e483ba8b9303e912a42ed6d"
)

OUTPUT_DIR = Path("outputs/issue53_stage5_kernel_ab_v1")
SMOKE_OUTPUT_DIR = Path("outputs/issue53_stage5_kernel_ab_smoke_v1")
COLLECTION_REPORT = "collection_report.json"
EVALUATION_REPORT = "evaluation_report.json"
AUDIT_REPORT = "independent_audit.json"

DATASET_ORDER = ("test_300x10", "nltcs")
ARM_INDEPENDENT = "independent_s0"
ARM_FACTOR = "factor_random_scan_s8"
ARMS = (ARM_INDEPENDENT, ARM_FACTOR)
FORMAL_SEEDS = tuple(range(338, 348))
SMOKE_SEED = 9905
EXCLUDED_SEED_RANGES = ((313, 317), (318, 322), (323, 327), (328, 332), (333, 337))

TAU = 2.0
FACTOR_SWEEPS = 8
FIXED_ALPHA = 16.0
PATIENCE_TICKS = 6
ROUND_CAP = 6000
CANDIDATE_BUDGET = 6000
SMOKE_ROUND_CAP = 3

NORMAL_REASONS = ("fit_target_reached", "early_stopped")
RESOURCE_CAP_REASON = "resource_cap_reached"
STABLE_WIN_MINIMUM = 8
LOWER_RISK_RATIO_MAX = 1.05
HIGHER_QUALITY_RATIO_MIN = 0.95
OUTER_COMPUTE_RATIO_MAX = 1.05

REFERENCE_PATHS = {
    "test_300x10": Path("data/test_300x10/test_300x10.csv"),
    "nltcs": Path("data/nltcs/nltcs.train.data"),
}
REFERENCE_SHA256 = {
    "test_300x10": (
        "c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb"
    ),
    "nltcs": (
        "e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30"
    ),
}

TEST_GROUP_ORDER = (
    "one_way_safety",
    "common_unseen_2way",
    "fixed_heldout_3way",
    "fixed_heldout_4way",
)
TEST_GROUP_COUNTS = {
    "one_way_safety": 25,
    "common_unseen_2way": 521,
    "fixed_heldout_3way": 512,
    "fixed_heldout_4way": 512,
}
TEST_GROUP_IDENTITIES = {
    "one_way_safety": (
        "b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c"
    ),
    "common_unseen_2way": (
        "fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2"
    ),
    "fixed_heldout_3way": (
        "d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a"
    ),
    "fixed_heldout_4way": (
        "2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171"
    ),
}
NLTCS_GROUP_COUNTS = {
    "one_way_safety": 32,
    "unmeasured_3way": 3958,
    "all_4way": 29120,
}
NLTCS_GROUP_IDENTITIES = {
    "one_way_safety": (
        "bbc8fc5d1b1ed0e5cd318a2168fe3887297b1c6aa33634736d0c693e96785c13"
    ),
    "unmeasured_3way": (
        "9c43437d6366e3cce0438fdf79e104d70ebabc112db9236b3feef5220b5eb588"
    ),
    "all_4way": (
        "1b92f8d80e775cffd637450d3d5015c78d43f7d9a870faf1603c99c88ec5d408"
    ),
}

DATASETS = {
    "test_300x10": {
        "schema": Path("configs/test_300x10/schema.yaml"),
        "queries": Path("configs/test_300x10/measured_50query_30_15_5.json"),
        "marginals": Path("configs/test_300x10/init_marginals.json"),
        "n_records": 300,
        "device": "numpy",
        "query_count": 50,
        "order_counts": {2: 30, 3: 15, 4: 5},
        "max_factor_order": 4,
        "query_identity_sha256": (
            "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
        ),
        "target_vector_sha256": (
            "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
        ),
        "sha256": {
            "schema": (
                "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792"
            ),
            "queries": (
                "708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14"
            ),
            "marginals": (
                "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4"
            ),
        },
    },
    "nltcs": {
        "schema": Path("configs/nltcs/schema.yaml"),
        "queries": Path("configs/nltcs/measured_1000query.json"),
        "marginals": Path("configs/nltcs/init_marginals.json"),
        "n_records": 16181,
        "device": "cuda",
        "query_count": 1001,
        "order_counts": {2: 479, 3: 522},
        "max_factor_order": 3,
        "query_identity_sha256": (
            "48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429"
        ),
        "target_vector_sha256": (
            "f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d"
        ),
        "sha256": {
            "schema": (
                "5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4"
            ),
            "queries": (
                "b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f"
            ),
            "marginals": (
                "a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e"
            ),
        },
    },
}


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_seed(seed: int, mode: str) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed 必须是冻结的整数 seed")
    allowed = FORMAL_SEEDS if mode == "formal" else (SMOKE_SEED,)
    if mode not in {"formal", "smoke"} or seed not in allowed:
        raise ValueError(f"seed 不在 {mode!r} 冻结矩阵中：{seed!r}")


def arm_order_for_seed(seed: int, *, mode: str = "formal") -> tuple[str, str]:
    _validate_seed(seed, mode)
    if seed % 2 == 0:
        return (ARM_INDEPENDENT, ARM_FACTOR)
    return (ARM_FACTOR, ARM_INDEPENDENT)


def case_order_for_seed(
    seed: int, *, mode: str = "formal"
) -> tuple[tuple[str, str], ...]:
    arms = arm_order_for_seed(seed, mode=mode)
    return tuple((dataset, arm) for dataset in DATASET_ORDER for arm in arms)


def generator_params(
    dataset: str,
    seed: int,
    arm: str,
    *,
    mode: str = "formal",
) -> dict[str, Any]:
    """Return the exact frozen generator arguments for one trajectory."""

    if dataset not in DATASETS:
        raise ValueError(f"dataset 不在冻结矩阵中：{dataset!r}")
    _validate_seed(seed, mode)
    if arm not in ARMS:
        raise ValueError(f"arm 不在冻结矩阵中：{arm!r}")
    cap = ROUND_CAP if mode == "formal" else SMOKE_ROUND_CAP
    return {
        "n_rounds": cap,
        "seed": seed,
        "beta": 1.0,
        "h": 0.8,
        "rho": 0.01,
        "eta": 0.5,
        "mu": 0.01,
        "tol": float("inf"),
        "eval_method": "vectorized",
        "batch_size": 256,
        "log_every": 100,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "alpha_min": FIXED_ALPHA,
        "alpha_max": FIXED_ALPHA,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": TAU,
        "diffusion_direction_normalization": "initial_rms",
        "diffusion_direction_logit_clip": 30.0,
        "factorized_gibbs_sweeps": (
            0 if arm == ARM_INDEPENDENT else FACTOR_SWEEPS
        ),
        "factorized_gibbs_max_order": DATASETS[dataset]["max_factor_order"],
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": arm == ARM_FACTOR,
        "candidate_budget": cap,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "return_final_table": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": FIXED_ALPHA,
        "record_transition_clocks": True,
        "record_stationarity_trace": False,
        "record_natural_work_snapshots": False,
        "stop_on_exact_residual": True,
        "inner_early_stopping_patience_ticks": PATIENCE_TICKS,
    }


def _dataset_manifest(name: str) -> dict[str, Any]:
    return _jsonable(DATASETS[name])


def _common_generator_manifest() -> dict[str, Any]:
    params = generator_params(
        DATASET_ORDER[0], FORMAL_SEEDS[0], ARM_INDEPENDENT
    ).copy()
    for key in (
        "seed",
        "factorized_gibbs_sweeps",
        "factorized_gibbs_max_order",
        "factorized_gibbs_use_compiled_workload",
    ):
        params.pop(key)
    return _jsonable(params)


def frozen_protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "5_same_tau_outer_kernel_ab",
        "purpose": "factor_random_scan_s8_vs_independent_s0_at_tau2",
        "protocol_doc": str(PROTOCOL_DOC),
        "protocol_doc_sha256": PROTOCOL_DOC_SHA256,
        "protocol_doc_commit": PROTOCOL_DOC_COMMIT,
        "upstream_stage4": {
            "protocol_sha256": STAGE4_PROTOCOL_SHA256,
            "execution_commit": STAGE4_EXECUTION_COMMIT,
            "state_library_sha256": STAGE4_STATE_LIBRARY_SHA256,
            "mixing_report_sha256": STAGE4_MIXING_REPORT_SHA256,
            "independent_audit_sha256": STAGE4_AUDIT_SHA256,
            "result": "qualified_random_scan_s8",
        },
        "dataset_order": list(DATASET_ORDER),
        "datasets": {
            name: _dataset_manifest(name) for name in DATASET_ORDER
        },
        "arms": {
            ARM_INDEPENDENT: {
                "factorized_gibbs_sweeps": 0,
                "factorized_gibbs_use_compiled_workload": False,
            },
            ARM_FACTOR: {
                "factorized_gibbs_sweeps": FACTOR_SWEEPS,
                "factorized_gibbs_use_compiled_workload": True,
            },
        },
        "arm_order_is_seed_counterbalanced": True,
        "even_seed_arm_order": list(ARMS),
        "odd_seed_arm_order": list(reversed(ARMS)),
        "formal_seeds": list(FORMAL_SEEDS),
        "excluded_seed_ranges_inclusive": [
            list(pair) for pair in EXCLUDED_SEED_RANGES
        ],
        "smoke": {
            "seed": SMOKE_SEED,
            "round_cap": SMOKE_ROUND_CAP,
            "formal_result_valid": False,
        },
        "formal_case_count": len(FORMAL_SEEDS) * len(DATASET_ORDER) * len(ARMS),
        "paired_dataset_seed_count": len(FORMAL_SEEDS) * len(DATASET_ORDER),
        "common_generator": _common_generator_manifest(),
        "dataset_max_factor_order": {
            name: DATASETS[name]["max_factor_order"] for name in DATASET_ORDER
        },
        "factor_gibbs_logit_clip": 30.0,
        "output_identity": "terminal_current",
        "initial_state_pairing": {
            "same_initial_table_sha256": True,
            "same_primary_rng_post_initialization_sha256": True,
            "same_positive_direction_reference_scale": True,
            "factor_extra_randomness_uses_derived_rng_only": True,
        },
        "collection_information_boundary": {
            "raw_reference_data_accessed": False,
            "offline_classification_emitted": False,
            "partial_matrix_comparison_emitted": False,
        },
        "offline_evaluation": {
            "reference_paths": _jsonable(REFERENCE_PATHS),
            "reference_sha256": REFERENCE_SHA256,
            "test_groups": {
                "order": list(TEST_GROUP_ORDER),
                "counts": TEST_GROUP_COUNTS,
                "identity_sha256": TEST_GROUP_IDENTITIES,
            },
            "nltcs_groups": {
                "counts": NLTCS_GROUP_COUNTS,
                "identity_sha256": NLTCS_GROUP_IDENTITIES,
            },
        },
        "frozen_gates": {
            "stable_factor_measured_gain": {
                "aggregate_count_error_sum_strictly_lower": True,
                "paired_strict_wins_minimum": STABLE_WIN_MINIMUM,
                "paired_seed_count": len(FORMAL_SEEDS),
            },
            "lower_is_better_ratio_max": LOWER_RISK_RATIO_MAX,
            "higher_is_better_ratio_min": HIGHER_QUALITY_RATIO_MIN,
            "valid_row_rate_required": 1.0,
            "outer_compute_ratio_max": OUTER_COMPUTE_RATIO_MAX,
            "direction_clip_hits_required": 0,
            "factor_conditional_clip_hits_required": 0,
        },
        "wall_clock_is_diagnostic_only": True,
        "cross_dataset_weighted_score_allowed": False,
        "parameter_retuning_allowed": False,
        "formal_generation_started": False,
    }


def protocol_sha256() -> str:
    return hashlib.sha256(_strict_json_bytes(frozen_protocol_manifest())).hexdigest()


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "Stage 5 protocol 身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    """Describe the exact formal matrix without touching inputs or RNG."""

    protocol_sha = assert_frozen_protocol_identity()
    return {
        "mode": "plan_only_no_input_result_or_rng_access_no_generator_import",
        "protocol_sha256": protocol_sha,
        "protocol": frozen_protocol_manifest(),
        "formal_shards": [
            {
                "shard_index": index,
                "seed": seed,
                "case_order": [
                    {"dataset": dataset, "arm": arm}
                    for dataset, arm in case_order_for_seed(seed)
                ],
            }
            for index, seed in enumerate(FORMAL_SEEDS)
        ],
        "output_dir": str(OUTPUT_DIR),
        "scientific_overrides_allowed": False,
        "generation_started": False,
    }
