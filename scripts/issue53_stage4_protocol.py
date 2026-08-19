"""Frozen, result-blind protocol for Issue #53 Stage 4 Gibbs qualification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROTOCOL_VERSION = "issue53-stage4-factor-gibbs-qualification-v1"
STATE_LIBRARY_FORMAT = "issue53_stage4_state_library_v1"
REPORT_FORMAT = "issue53_stage4_mixing_report_v1"
AUDIT_FORMAT = "issue53_stage4_mixing_audit_v1"

DATASET_ORDER = ("test_300x10", "nltcs")
STATE_GROUPS = (
    "initial",
    "work_q25",
    "work_q50",
    "work_q75",
    "terminal",
)
ACTIVE_WIDTH_GROUPS = (
    "active_width_1_4",
    "active_width_5_8",
    "active_width_9_12",
    "active_width_13_16",
)
CANDIDATE_SWEEPS = (8, 16, 32)
QUALIFICATION_SEEDS = (333, 334, 335, 336, 337)
DEVELOPMENT_SEEDS = (323, 324, 325, 326, 327)
SMOKE_SEEDS = (9904,)

RHO = 0.01
ETA = 0.5
MU = 0.01
TAU = 2.0
FIXED_ALPHA = 16.0
PATIENCE_TICKS = 6
RESOURCE_CAP = 6000
GIBBS_LOGIT_CLIP = 30.0
ENERGY_TOLERANCE = 1e-10
TVD_THRESHOLD = 0.05
RECOVERY_THRESHOLD = 0.80
TVD_MONOTONIC_TOLERANCE = 1e-12
PROBABILITY_SUM_TOLERANCE = 1e-12

DATASETS = {
    "test_300x10": {
        "schema": "configs/test_300x10/schema.yaml",
        "queries": "configs/test_300x10/measured_50query_30_15_5.json",
        "marginals": "configs/test_300x10/init_marginals.json",
        "n_records": 300,
        "device": "numpy",
        "query_count": 50,
        "query_order_counts": {"2": 30, "3": 15, "4": 5},
        "query_identity_sha256": (
            "602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1"
        ),
        "target_vector_sha256": (
            "e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d"
        ),
        "input_sha256": {
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
        "max_factor_order": 4,
        "max_active_attributes": 10,
        "qualification_proposals_per_state": 200,
    },
    "nltcs": {
        "schema": "configs/nltcs/schema.yaml",
        "queries": "configs/nltcs/measured_1000query.json",
        "marginals": "configs/nltcs/init_marginals.json",
        "n_records": 16181,
        "device": "cuda",
        "query_count": 1001,
        "query_order_counts": {"2": 479, "3": 522},
        "query_identity_sha256": (
            "48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429"
        ),
        "target_vector_sha256": (
            "f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d"
        ),
        "input_sha256": {
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
        "max_factor_order": 3,
        "max_active_attributes": 16,
        "qualification_proposals_per_state": 4,
    },
}

MODE_CONFIG = {
    "development": {
        "seeds": DEVELOPMENT_SEEDS,
        "formal_result_valid": False,
        "runtime_n_records": None,
        "runtime_device": None,
        "round_cap": RESOURCE_CAP,
        "candidate_budget": RESOURCE_CAP,
        "proposals_per_state": {
            name: DATASETS[name]["qualification_proposals_per_state"]
            for name in DATASET_ORDER
        },
        "pipeline_only_runtime_override": False,
    },
    "qualification": {
        "seeds": QUALIFICATION_SEEDS,
        "formal_result_valid": True,
        "runtime_n_records": None,
        "runtime_device": None,
        "round_cap": RESOURCE_CAP,
        "candidate_budget": RESOURCE_CAP,
        "proposals_per_state": {
            name: DATASETS[name]["qualification_proposals_per_state"]
            for name in DATASET_ORDER
        },
        "pipeline_only_runtime_override": False,
    },
    # The smoke mode exercises both real public workloads but reduces only the
    # synthetic population, device and external budget.  It can never support
    # a scientific qualification claim.
    "smoke": {
        "seeds": SMOKE_SEEDS,
        "formal_result_valid": False,
        "runtime_n_records": 128,
        "runtime_device": "numpy",
        "round_cap": 500,
        "candidate_budget": 500,
        "proposals_per_state": {"test_300x10": 8, "nltcs": 8},
        "pipeline_only_runtime_override": True,
    },
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_protocol(name: str, mode: str) -> dict:
    source = DATASETS[name]
    config = MODE_CONFIG[mode]
    return {
        **source,
        "runtime_n_records": (
            source["n_records"]
            if config["runtime_n_records"] is None
            else config["runtime_n_records"]
        ),
        "runtime_device": (
            source["device"]
            if config["runtime_device"] is None
            else config["runtime_device"]
        ),
        "proposals_per_state": config["proposals_per_state"][name],
    }


def stage4_protocol(mode: str) -> dict:
    if mode not in MODE_CONFIG:
        raise ValueError("mode 必须是 smoke、development 或 qualification")
    config = MODE_CONFIG[mode]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue": 53,
        "stage": "4_factor_gibbs_qualification",
        "mode": mode,
        "formal_result_valid": config["formal_result_valid"],
        "dataset_order": list(DATASET_ORDER),
        "datasets": {
            name: _dataset_protocol(name, mode) for name in DATASET_ORDER
        },
        "source_kernel": "independent_directional_mask",
        "candidate_kernel": "factor_random_scan_gibbs_with_replacement",
        "source_temperature": TAU,
        "evaluation_temperature": TAU,
        "source_sweeps": 0,
        "fixed_alpha": FIXED_ALPHA,
        "seeds": list(config["seeds"]),
        "stage5_seed_reuse_forbidden": True,
        "state_groups": list(STATE_GROUPS),
        "state_selection_rule": (
            "initial and terminal exact; choose three distinct recorded "
            "natural-work states minimizing total absolute distance to "
            "25%, 50%, 75% of terminal normalized work; lexicographic "
            "earlier-index tie break"
        ),
        "expected_states_per_dataset": len(config["seeds"]) * len(STATE_GROUPS),
        "rho": RHO,
        "eta": ETA,
        "trajectory_mu": MU,
        "probe_mu": 0.0,
        "direction_strength": TAU,
        "direction_normalization": "initial_rms",
        "direction_logit_clip": GIBBS_LOGIT_CLIP,
        "gibbs_logit_clip": GIBBS_LOGIT_CLIP,
        "selection_scale_invariant": True,
        "selection_scale_invariant_min_spread": 1e-3,
        "residual_geometry": "relative",
        "residual_geometry_floor": 8.0,
        "initialization": "marginal",
        "fixed_generation_acceptance_tolerance": "positive_infinity",
        "max_retries": 0,
        "patience_ticks": PATIENCE_TICKS,
        "resource_cap_rounds": config["round_cap"],
        "candidate_budget": config["candidate_budget"],
        "terminal_current": True,
        "candidate_sweeps": list(CANDIDATE_SWEEPS),
        "sweeps_hard_cap": max(CANDIDATE_SWEEPS),
        "shared_sweep_selection": (
            "execute both datasets at 8, then 16, then 32; stop at first "
            "single sweep passing both datasets; 32 failure is unqualified"
        ),
        "invalid_stop_rule": (
            "stop immediately at the first structurally invalid or "
            "incomplete attempt; later sweeps cannot restore qualification"
        ),
        "higher_sweeps_forbidden": True,
        "shared_condition_rule": (
            "per state/proposal freeze donor draw, participation rows, "
            "independent initial mask, exact-joint outcome and random-scan "
            "tape address across 8/16/32"
        ),
        "tvd_threshold": TVD_THRESHOLD,
        "gap_recovery_threshold": RECOVERY_THRESHOLD,
        "conditional_clip_hit_count_required": 0,
        "energy_tolerance": ENERGY_TOLERANCE,
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        "tvd_monotonic_tolerance": TVD_MONOTONIC_TOLERANCE,
        "production_exact_tape_replay_required": True,
        "required_stage_groups": list(STATE_GROUPS),
        "active_width_groups": list(ACTIVE_WIDTH_GROUPS),
        "active_width_group_rule": "gate every nonempty group",
        "cost_hard_gate": "sweeps <= 32",
        "wall_time_is_diagnostic_only": True,
        "allowed_results": [
            "qualified_random_scan_s8",
            "qualified_random_scan_s16",
            "qualified_random_scan_s32",
            "unqualified_at_s32",
            "invalid_or_incomplete",
        ],
        "qualified_next_stage": (
            "Stage 5 same-tau independent versus factor comparison on fresh seeds"
        ),
        "unqualified_next_stage": (
            "freeze a separate random-permutation Gibbs protocol"
        ),
        "pipeline_only_runtime_override": config[
            "pipeline_only_runtime_override"
        ],
    }


def protocol_sha256(mode: str) -> str:
    return canonical_sha256(stage4_protocol(mode))


def require_qualification_confirmation(
    mode: str,
    confirmed_protocol_sha256: str | None,
) -> None:
    """Fail closed before any qualification-seed trajectory or probe runs."""
    if mode != "qualification":
        return
    expected = protocol_sha256(mode)
    if confirmed_protocol_sha256 != expected:
        raise PermissionError(
            "qualification 模式尚未获显式授权；需传入当前冻结协议 SHA256："
            f"{expected}"
        )
