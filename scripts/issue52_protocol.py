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
