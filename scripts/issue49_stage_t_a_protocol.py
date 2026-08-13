"""Issue #49 Stage T/A/B 冻结协议的唯一运行配置。"""

import math

PROTOCOL_VERSION = 2
DATASET = "test_300x10"
TEMPERATURES = (4.0, 5.0, 6.0, 7.0, 8.0)
SWEEPS = (0, 8, 16, 32)
CANDIDATE_SWEEPS = (8, 16, 32)
RHO = 0.01
ETA = 0.5
TRAJECTORY_MU = 0.01
PROBE_MU = 0.0
LOGIT_CLIP = 30.0
MAX_FACTOR_ORDER = 3
MAX_ACTIVE_ATTRIBUTES = 12
TVD_THRESHOLD = 0.05
RECOVERY_THRESHOLD = 0.80
ENERGY_TOLERANCE = 1e-10
TVD_MONOTONIC_TOLERANCE = 1e-12
PROBABILITY_SUM_TOLERANCE = 1e-12
DEVICE = "numpy"
EXPECTED_INPUT_SHA256 = {
    "schema": "58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792",
    "queries": "7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88",
    "marginals": "1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4",
}
MODE_CONFIG = {
    "formal": {
        "stage_t_seeds": tuple(range(10)),
        "state_library_seeds": (0, 1, 2),
        "rounds": 1000,
        "snapshot_rounds": (0, 500, 1000),
        "proposals_per_state": 200,
    },
    "smoke": {
        "stage_t_seeds": (99,),
        "state_library_seeds": (99,),
        "rounds": 12,
        "snapshot_rounds": (0, 6, 12),
        "proposals_per_state": 2,
    },
}

STAGE_B_FACTOR_BUILDER = "legacy_rowwise"
STAGE_B_MINIMUM_PAIRED_WIN_FRACTION = 0.60
STAGE_B_SELF_REVIEW_GROUPS = ("global", "initial", "mid", "late")
STAGE_B_RANKING_METRICS = (
    "late_window_current_loss_mean",
    "late_window_current_loss_median",
    "current_loss_auc_mean",
    "final_current_loss_mean",
    "fewer_sweeps",
    "lower_temperature",
)
STAGE_B_MODE_CONFIG = {
    "formal": {
        "seeds": tuple(range(100, 110)),
        "rounds": 1000,
        "snapshot_rounds": (0, 500, 1000),
    },
    "smoke": {
        "seeds": (99,),
        "rounds": 12,
        "snapshot_rounds": (0, 6, 12),
    },
}

CONFIRMATION_INCUMBENT_TEMPERATURE = 8.0
CONFIRMATION_MINIMUM_PAIRED_WIN_FRACTION = 0.60
CONFIRMATION_CONFIDENCE_LEVEL = 0.95
CONFIRMATION_T_CRITICAL_95_DF9 = 2.2621571627409915
CONFIRMATION_MODE_CONFIG = {
    "formal": {
        "seeds": tuple(range(110, 120)),
        "rounds": 1000,
        "snapshot_rounds": (0, 500, 1000),
    },
    "smoke": {
        "seeds": (99,),
        "rounds": 12,
        "snapshot_rounds": (0, 6, 12),
    },
}


def protocol(mode):
    if mode not in MODE_CONFIG:
        raise ValueError("mode 必须是 smoke 或 formal")
    mode_config = MODE_CONFIG[mode]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "dataset": DATASET,
        "stage_t_seeds": list(mode_config["stage_t_seeds"]),
        "state_library_seeds": list(
            mode_config["state_library_seeds"]
        ),
        "rounds": int(mode_config["rounds"]),
        "snapshot_rounds": list(mode_config["snapshot_rounds"]),
        "source_temperatures": list(TEMPERATURES),
        "source_sweeps": 0,
        "evaluation_temperatures": list(TEMPERATURES),
        "sweeps": list(SWEEPS),
        "candidate_sweeps": list(CANDIDATE_SWEEPS),
        "proposals_per_state": int(
            mode_config["proposals_per_state"]
        ),
        "rho": RHO,
        "eta": ETA,
        "trajectory_mu": TRAJECTORY_MU,
        "probe_mu": PROBE_MU,
        "max_factor_order": MAX_FACTOR_ORDER,
        "max_active_attributes": MAX_ACTIVE_ATTRIBUTES,
        "logit_clip": LOGIT_CLIP,
        "device": DEVICE,
        "tvd_threshold": TVD_THRESHOLD,
        "recovery_threshold": RECOVERY_THRESHOLD,
        "energy_tolerance": ENERGY_TOLERANCE,
        "tvd_monotonic_tolerance": TVD_MONOTONIC_TOLERANCE,
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
    }


def stage_b_protocol(mode):
    """返回 Stage B 的冻结运行与选择配置。"""
    if mode not in STAGE_B_MODE_CONFIG:
        raise ValueError("mode 必须是 smoke 或 formal")
    mode_config = STAGE_B_MODE_CONFIG[mode]
    seeds = list(mode_config["seeds"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "dataset": DATASET,
        "stage_b_seeds": seeds,
        "rounds": int(mode_config["rounds"]),
        "snapshot_rounds": list(mode_config["snapshot_rounds"]),
        "late_window_size": 250,
        "independent_temperatures": list(TEMPERATURES),
        "factor_builder": STAGE_B_FACTOR_BUILDER,
        "rho": RHO,
        "eta": ETA,
        "trajectory_mu": TRAJECTORY_MU,
        "max_factor_order": MAX_FACTOR_ORDER,
        "logit_clip": LOGIT_CLIP,
        "device": DEVICE,
        "minimum_paired_win_fraction": (
            STAGE_B_MINIMUM_PAIRED_WIN_FRACTION
        ),
        "minimum_paired_wins": int(math.ceil(
            STAGE_B_MINIMUM_PAIRED_WIN_FRACTION * len(seeds)
        )),
        "ranking_tie_breakers": list(STAGE_B_RANKING_METRICS),
        "self_review_required_groups": list(
            STAGE_B_SELF_REVIEW_GROUPS
        ),
        "self_review_proposals_per_state": int(
            MODE_CONFIG[mode]["proposals_per_state"]
        ),
        "self_review_probe_mu": PROBE_MU,
        "self_review_max_active_attributes": MAX_ACTIVE_ATTRIBUTES,
        "self_review_tvd_threshold": TVD_THRESHOLD,
        "self_review_recovery_threshold": RECOVERY_THRESHOLD,
        "energy_tolerance": ENERGY_TOLERANCE,
        "tvd_monotonic_tolerance": TVD_MONOTONIC_TOLERANCE,
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
    }


def confirmation_protocol(mode):
    """返回唯一候选最终确认阶段的冻结配置。"""
    if mode not in CONFIRMATION_MODE_CONFIG:
        raise ValueError("mode 必须是 smoke 或 formal")
    mode_config = CONFIRMATION_MODE_CONFIG[mode]
    seeds = list(mode_config["seeds"])
    return {
        "protocol_version": PROTOCOL_VERSION,
        "mode": mode,
        "dataset": DATASET,
        "confirmation_seeds": seeds,
        "rounds": int(mode_config["rounds"]),
        "snapshot_rounds": list(mode_config["snapshot_rounds"]),
        "late_window_size": 250,
        "independent_temperatures": list(TEMPERATURES),
        "factor_builder": STAGE_B_FACTOR_BUILDER,
        "rho": RHO,
        "eta": ETA,
        "trajectory_mu": TRAJECTORY_MU,
        "max_factor_order": MAX_FACTOR_ORDER,
        "logit_clip": LOGIT_CLIP,
        "device": DEVICE,
        "minimum_paired_win_fraction": (
            CONFIRMATION_MINIMUM_PAIRED_WIN_FRACTION
        ),
        "minimum_paired_wins": int(math.ceil(
            CONFIRMATION_MINIMUM_PAIRED_WIN_FRACTION * len(seeds)
        )),
        "paired_confidence_level": CONFIRMATION_CONFIDENCE_LEVEL,
        "paired_t_critical_95_df9": CONFIRMATION_T_CRITICAL_95_DF9,
        "formal_paired_sample_size": 10,
        "incumbent_independent_temperature": (
            CONFIRMATION_INCUMBENT_TEMPERATURE
        ),
        "self_review_required_groups": list(
            STAGE_B_SELF_REVIEW_GROUPS
        ),
        "self_review_proposals_per_state": int(
            MODE_CONFIG[mode]["proposals_per_state"]
        ),
        "self_review_probe_mu": PROBE_MU,
        "self_review_max_active_attributes": MAX_ACTIVE_ATTRIBUTES,
        "self_review_tvd_threshold": TVD_THRESHOLD,
        "self_review_recovery_threshold": RECOVERY_THRESHOLD,
        "energy_tolerance": ENERGY_TOLERANCE,
        "tvd_monotonic_tolerance": TVD_MONOTONIC_TOLERANCE,
        "probability_sum_tolerance": PROBABILITY_SUM_TOLERANCE,
        "frozen_candidate_only": True,
        "runner_up_forbidden": True,
    }
