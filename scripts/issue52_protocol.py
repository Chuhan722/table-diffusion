"""Issue #52 low-temperature frontier and long-horizon frozen protocol."""


PROTOCOL_VERSION = 1
DATASET = "test_300x10"
INDEPENDENT_TEMPERATURES = (1.0, 2.0, 3.0, 4.0, 5.0)
RHO = 0.01
ETA = 0.5
TRAJECTORY_MU = 0.01
LOGIT_CLIP = 30.0
DEVICE = "numpy"
SOURCE_SWEEPS = 0
MAX_WORKERS = 8
CLEAR_DESCENT_RELATIVE_THRESHOLD = 0.05
EXPECTED_INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "queries": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
}
FORMAL_STAGE_T_ARTIFACT_SHA256 = {
    "report": "abf7f6e25d86d518ea5255d0c1414e5fb7b606f9c14876ddfc3c952499dcc665",
    "audit": "70253a5e6d115bea1bc37463499702ef3a1e421cbcbfc869d7cd6504408e9705",
}
FORMAL_STAGE_A_STATE_LIBRARY_ARTIFACT_SHA256 = {
    "library": (
        "4945f52a644e059a710decf66659b82fa"
        "693c4443d799ca2e3948a1ca61d33fc"
    ),
    "audit": (
        "837a90851795dcc05d4f15f88fea17658"
        "2dcc92ebf9227cadbe40ae21e161545"
    ),
}
FORMAL_STAGE_A_STATE_LIBRARY_PROTOCOL_SHA256 = (
    "918350fa3d1358363c35d222d424d4f7221e39dc04f9af93d422dcd8d0eef2dc"
)
FORMAL_STAGE_A_STATE_LIBRARY_SCIENTIFIC_SHA256 = (
    "bd18e9691427a2023af7dfad9bcc7fbdb5c2ce882a2a8a84a8a1ca4fd6dd2462"
)
FORMAL_STAGE_A_MIXING_ARTIFACT_SHA256 = {
    "report": (
        "52654a455d42a0899194878789aa4690"
        "b3527c5482a6e4c2d5475b59df835b09"
    ),
    "audit": (
        "6b4b76af01dc2ed3b267f8047e8edefe"
        "f84e5fc1049d7aa5974af6ab8286a741"
    ),
}
FORMAL_STAGE_A_MIXING_PROTOCOL_SHA256 = (
    "e0259b1c614aa49ed79f4d7dec61829ae0b46233e77a14e127b39af93c62e17d"
)
FORMAL_STAGE_A_MIXING_SCIENTIFIC_SHA256 = (
    "56d5a470a891e99c985503514d7092e385805d1b6900cb46e1da4995fb0c2e0d"
)
FORMAL_STAGE_A_MIXING_SELECTION = {
    "minimal_sufficient_sweeps": {
        "tau_1": 8,
        "tau_2": 8,
        "tau_3": 16,
        "tau_4": None,
        "tau_5": None,
    },
    "qualified_temperatures": [1.0, 2.0, 3.0],
    "unqualified_temperatures": [4.0, 5.0],
}

STAGE_A_EVALUATION_TEMPERATURES = INDEPENDENT_TEMPERATURES
STAGE_A_CANDIDATE_SWEEPS = (8, 16, 32)
STAGE_A_TVD_THRESHOLD = 0.05
STAGE_A_RECOVERY_THRESHOLD = 0.80
STAGE_A_ENERGY_TOLERANCE = 1e-10
STAGE_A_TVD_MONOTONIC_TOLERANCE = 1e-12
STAGE_A_PROBABILITY_SUM_TOLERANCE = 1e-12
STAGE_A_MAX_FACTOR_ORDER = 3
STAGE_A_MAX_ACTIVE_ATTRIBUTES = 12
STAGE_A_MODE_CONFIG = {
    "formal": {"proposals_per_state": 200},
    "smoke": {"proposals_per_state": 2},
}

MODE_CONFIG = {
    "formal": {
        "stage_t_seeds": tuple(range(200, 210)),
        "state_library_seeds": (200, 201, 202),
        "rounds": 3000,
        "trend_checkpoints": (500, 1000, 1500, 2000, 2500, 3000),
        "snapshot_rounds": (0, 1000, 2000, 3000),
        "late_window_size": 500,
    },
    # Pipeline-only identities, disjoint from development/confirmation seeds.
    "smoke": {
        "stage_t_seeds": (9903,),
        "state_library_seeds": (9903,),
        "rounds": 12,
        "trend_checkpoints": (2, 4, 6, 8, 10, 12),
        "snapshot_rounds": (0, 4, 8, 12),
        "late_window_size": 2,
    },
}


def stage_t_protocol(mode):
    if mode not in MODE_CONFIG:
        raise ValueError("mode 必须是 smoke 或 formal")
    config = MODE_CONFIG[mode]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue": 52,
        "stage": "T",
        "mode": mode,
        "dataset": DATASET,
        "kernel": "independent",
        "stage_t_seeds": list(config["stage_t_seeds"]),
        "state_library_seeds": list(config["state_library_seeds"]),
        "source_temperatures": list(INDEPENDENT_TEMPERATURES),
        "source_sweeps": SOURCE_SWEEPS,
        "rounds": int(config["rounds"]),
        "trend_checkpoints": list(config["trend_checkpoints"]),
        "snapshot_rounds": list(config["snapshot_rounds"]),
        "late_window_size": int(config["late_window_size"]),
        "primary_metric": "rounds_2501_3000_current_loss_mean"
        if mode == "formal" else "final_smoke_window_current_loss_mean",
        "clear_descent_relative_threshold": (
            CLEAR_DESCENT_RELATIVE_THRESHOLD
        ),
        "clear_descent_is_diagnostic_only": True,
        "rho": RHO,
        "eta": ETA,
        "trajectory_mu": TRAJECTORY_MU,
        "logit_clip": LOGIT_CLIP,
        "device": DEVICE,
        "max_workers_allowed": MAX_WORKERS,
        "worker_count_is_nonscientific": True,
        "current_state_primary": True,
        "generation_acceptance": False,
        "best_state_diagnostic_only": True,
    }


def stage_a_state_library_protocol(mode):
    """Freeze the deterministic Stage A state-library materialization."""
    stage_t = stage_t_protocol(mode)
    state_seeds = stage_t["state_library_seeds"]
    temperatures = stage_t["source_temperatures"]
    snapshot_rounds = stage_t["snapshot_rounds"]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue": 52,
        "stage": "A_state_library",
        "mode": mode,
        "dataset": DATASET,
        "source_stage": "T",
        "source_report_format": "issue52_stage_t_report_v1",
        "source_audit_format": "issue52_stage_t_audit_v1",
        "expected_source_sha256": (
            dict(FORMAL_STAGE_T_ARTIFACT_SHA256)
            if mode == "formal" else None
        ),
        "state_library_format": "issue52_stage_a_state_library_v1",
        "state_library_seeds": list(state_seeds),
        "source_temperatures": list(temperatures),
        "source_sweeps": SOURCE_SWEEPS,
        "source_rounds": stage_t["rounds"],
        "snapshot_rounds": list(snapshot_rounds),
        "round0_dedup_rule": "one shared initial state per seed",
        "canonical_round0_source_temperature": temperatures[0],
        "state_order": "seed_then_round_then_source_temperature",
        "expected_raw_snapshot_count": (
            len(state_seeds) * len(temperatures) * len(snapshot_rounds)
        ),
        "expected_unique_state_count": (
            len(state_seeds)
            * (1 + len(temperatures) * (len(snapshot_rounds) - 1))
        ),
        "current_state_only": True,
        "query_loss_recomputed": True,
        "device": DEVICE,
    }


def _stage_a_required_groups(mode):
    """Return the fixed 16 groups represented by that mode's state library."""
    library_protocol = stage_a_state_library_protocol(mode)
    groups = ["initial"]
    for state_round in library_protocol["snapshot_rounds"][1:]:
        for temperature in library_protocol["source_temperatures"]:
            tau = f"{temperature:g}".replace(".", "p")
            groups.append(f"round_{state_round}_source_tau_{tau}")
    return groups


def stage_a_mixing_protocol(mode):
    """Freeze Stage A factor-Gibbs mixing qualification before any result."""
    if mode not in STAGE_A_MODE_CONFIG:
        raise ValueError("mode 必须是 smoke 或 formal")
    library_protocol = stage_a_state_library_protocol(mode)
    frozen_artifact_hashes = (
        dict(FORMAL_STAGE_A_STATE_LIBRARY_ARTIFACT_SHA256)
        if mode == "formal" else None
    )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue": 52,
        "stage": "A_factor_mixing_qualification",
        "mode": mode,
        "dataset": DATASET,
        "state_library_mode": mode,
        "state_library_format": "issue52_stage_a_state_library_v1",
        "state_library_audit_format": (
            "issue52_stage_a_state_library_audit_v1"
        ),
        "expected_state_library_artifact_sha256": frozen_artifact_hashes,
        "expected_state_library_protocol_sha256": (
            FORMAL_STAGE_A_STATE_LIBRARY_PROTOCOL_SHA256
            if mode == "formal" else None
        ),
        "expected_state_library_scientific_sha256": (
            FORMAL_STAGE_A_STATE_LIBRARY_SCIENTIFIC_SHA256
            if mode == "formal" else None
        ),
        "expected_state_count": library_protocol[
            "expected_unique_state_count"
        ],
        "state_library_seeds": list(
            library_protocol["state_library_seeds"]
        ),
        "source_temperatures": list(
            library_protocol["source_temperatures"]
        ),
        "source_snapshot_rounds": list(
            library_protocol["snapshot_rounds"]
        ),
        "required_state_groups": _stage_a_required_groups(mode),
        "expected_state_group_count": 16,
        "states_per_group": len(library_protocol["state_library_seeds"]),
        "evaluation_temperatures": list(
            STAGE_A_EVALUATION_TEMPERATURES
        ),
        "candidate_sweeps": list(STAGE_A_CANDIDATE_SWEEPS),
        "sweeps_hard_cap": max(STAGE_A_CANDIDATE_SWEEPS),
        "selection_rule": (
            "for each tau execute 8 then 16 then 32; stop immediately at "
            "the first sweep passing every required state group; if 32 "
            "fails, mark that tau unqualified"
        ),
        "later_sweeps_forbidden_after_first_pass": True,
        "proposals_per_state": int(
            STAGE_A_MODE_CONFIG[mode]["proposals_per_state"]
        ),
        "rho": RHO,
        "eta": ETA,
        "probe_mu": 0.0,
        "max_factor_order": STAGE_A_MAX_FACTOR_ORDER,
        "max_active_attributes": STAGE_A_MAX_ACTIVE_ATTRIBUTES,
        "logit_clip": LOGIT_CLIP,
        "device": DEVICE,
        "tvd_threshold": STAGE_A_TVD_THRESHOLD,
        "recovery_threshold": STAGE_A_RECOVERY_THRESHOLD,
        "energy_tolerance": STAGE_A_ENERGY_TOLERANCE,
        "tvd_monotonic_tolerance": (
            STAGE_A_TVD_MONOTONIC_TOLERANCE
        ),
        "probability_sum_tolerance": (
            STAGE_A_PROBABILITY_SUM_TOLERANCE
        ),
        "zero_clip_hits_required": True,
        "production_exact_tape_replay_required": True,
        "shared_condition_rule": (
            "within each tau, every attempted sweep must reproduce exactly "
            "the same state identities, donor/participation conditions, "
            "baseline mask outcomes and exact-joint outcomes under the "
            "probe address tape"
        ),
        "worker_count_is_nonscientific": True,
        "max_workers_allowed": MAX_WORKERS,
        "input_sha256": dict(EXPECTED_INPUT_SHA256),
    }


def stage_b_protocol(mode):
    """Freeze the Stage B factor long-horizon comparison and selection."""
    stage_t = stage_t_protocol(mode)
    if mode not in MODE_CONFIG:
        raise ValueError("mode 必须是 smoke 或 formal")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "issue": 52,
        "stage": "B_factor_long_horizon",
        "mode": mode,
        "dataset": DATASET,
        "stage_t_report_format": "issue52_stage_t_report_v1",
        "stage_t_audit_format": "issue52_stage_t_audit_v1",
        "expected_stage_t_artifact_sha256": (
            dict(FORMAL_STAGE_T_ARTIFACT_SHA256)
            if mode == "formal" else None
        ),
        "stage_a_report_format": "issue52_stage_a_mixing_report_v1",
        "stage_a_audit_format": "issue52_stage_a_mixing_audit_v1",
        "expected_stage_a_artifact_sha256": (
            dict(FORMAL_STAGE_A_MIXING_ARTIFACT_SHA256)
            if mode == "formal" else None
        ),
        "expected_stage_a_protocol_sha256": (
            FORMAL_STAGE_A_MIXING_PROTOCOL_SHA256
            if mode == "formal" else None
        ),
        "expected_stage_a_scientific_sha256": (
            FORMAL_STAGE_A_MIXING_SCIENTIFIC_SHA256
            if mode == "formal" else None
        ),
        "eligible_factor_source": (
            "bound_stage_a_minimal_sufficient_sweeps"
        ),
        "expected_formal_stage_a_selection": (
            {
                "minimal_sufficient_sweeps": dict(
                    FORMAL_STAGE_A_MIXING_SELECTION[
                        "minimal_sufficient_sweeps"
                    ]
                ),
                "qualified_temperatures": list(
                    FORMAL_STAGE_A_MIXING_SELECTION[
                        "qualified_temperatures"
                    ]
                ),
                "unqualified_temperatures": list(
                    FORMAL_STAGE_A_MIXING_SELECTION[
                        "unqualified_temperatures"
                    ]
                ),
            }
            if mode == "formal" else None
        ),
        "stage_b_seeds": list(stage_t["stage_t_seeds"]),
        "rounds": stage_t["rounds"],
        "trend_checkpoints": list(stage_t["trend_checkpoints"]),
        "late_window_size": stage_t["late_window_size"],
        "primary_metric": stage_t["primary_metric"],
        "clear_descent_relative_threshold": (
            CLEAR_DESCENT_RELATIVE_THRESHOLD
        ),
        "clear_descent_is_diagnostic_only": True,
        "factor_builder": "compiled_batch",
        "factor_builder_changes_science": False,
        "i_star_source": "bound_stage_t_late_mean_ranking",
        "expected_formal_i_star": (
            "independent_tau_5" if mode == "formal" else None
        ),
        "g_star_rule": (
            "lowest factor late-window current-loss mean; exact ties use "
            "fewer sweeps then lower tau"
        ),
        "candidate_gate": (
            "G* point estimate must be lower than both the same-tau "
            "independent and I*; otherwise no_factor_candidate"
        ),
        "ranking_tie_breakers": [
            "late_window_current_loss_mean",
            "fewer_sweeps",
            "lower_temperature",
        ],
        "no_reselection": True,
        "rho": RHO,
        "eta": ETA,
        "trajectory_mu": TRAJECTORY_MU,
        "logit_clip": LOGIT_CLIP,
        "device": DEVICE,
        "max_workers_allowed": MAX_WORKERS,
        "worker_count_is_nonscientific": True,
        "current_state_primary": True,
        "generation_acceptance": False,
        "best_state_diagnostic_only": True,
        "snapshots_disabled": True,
        "input_sha256": dict(EXPECTED_INPUT_SHA256),
    }
