#!/usr/bin/env python3
"""Run the fitness-only floor-phase eta-cooling development screen (v5).

This is a *delta executor* over the frozen v3 T=6000 screen
(``scripts/run_fitness_only_schedule_v3_t6000.py``).  The rho schedule is
kept verbatim from v3 (hold H=900 / geometric descent D=600 / floor 0.001);
the changes are::

    eta schedule : constant 0.5 -> three-phase blind cooling
                   (hold H_eta=1500 / geometric descent D_eta=600 /
                    floor 0.25) — starts exactly when rho lands on its
                   floor, so the only difference vs v3 is the floor-phase
                   per-step copy magnitude
    n_rounds     : 6000 -> 9000 (budget only; halved per-step repair
                   magnitude gets extra floor time, windows aligned to v4)

Mechanism pairing with v4: v4 halved the participation *frequency*
(rho floor 0.001 -> 0.0005) and the nltcs oscillation baseline did not
narrow (1.1639 -> 1.1754) while the small table froze
(quality_regression_under_lower_floor).  The v5 screen tests exactly one
pre-frozen hypothesis (jump magnitude): the steady-state band width is
dominated by the per-participating-row copy magnitude (expected changed
attributes = differing attributes x eta), so halving eta on the floor
narrows the band enough to reach the original drift target <= 1.100
without hurting measured/held-out quality.  Criteria, labels and the
descending morphology rule were frozen before this run in
``docs/设计/FitnessOnly地板段Eta降温v5结果前协议.md``.

Falsifiable audit: eta only replaces the threshold of the independent
copy switch (``rng.random(n) < eta``) and never changes the random-stream
consumption order; the v5 eta schedule holds at exactly 0.5 for
t in [0, 1500] (1501 rounds — the hold phase plus the t=1500 boundary
where the anneal progress is exactly zero) and diverges from t=1501.
Because both schedules depend only on the round index, every v5 arm must
match the corresponding v3 arm bit-for-bit over those 1501 rounds.  Any
mismatch invalidates the whole run (``schedule_blindness_violated``).

This remains a single-development-seed screen: no formal quality claim,
no promotion gate, never overwrites an existing output.  v5 is *not* a
retake of v2/v3/v4 — their unsupported/rejected/quality-regression
records stand unchanged.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2
from scripts import run_fitness_only_schedule_v3_t6000 as v3
from scripts import run_fitness_only_floor00005_v4 as v4
from table_diffevo.fitness_only import FitnessOnlyConfig


PROTOCOL_VERSION = "fitness-only-eta-cooling-v5"
OUTPUT_DIR = Path("outputs/fitness_only_eta_cooling_dev_seed9908_v5")
DESIGN_PROTOCOL_DOC = (
    "docs/设计/FitnessOnly地板段Eta降温v5结果前协议.md"
)

N_ROUNDS = 9000
ETA_BASE = 0.5
ETA_ANNEAL_START_ROUND = 1500
ETA_ANNEAL_ROUNDS = 600
ETA_ANNEAL_END = 0.25
# v5 与 v3 的全部时间表在 t∈[0,1500]（含 t=1500，eta progress 恰为 0）
# 逐点相同，自 t=1501 起 eta 几何降温而分叉；rho 时间表 v3 原样。
PREFIX_ROUNDS = ETA_ANNEAL_START_ROUND + 1  # 1501

FROZEN_PROTOCOL_SHA256 = (
    "05863ca27710fe52794a6106c74b341962b170b1edb964312d09571dc8e6fe27"
)

# v3 参照产物（只读）与冻结参照数值：与 v4 同一组，直接复用其冻结常数。
V3_OUTPUT_DIR = v4.V3_OUTPUT_DIR
V3_REFERENCE_ARTIFACT_SHA256 = v4.V3_REFERENCE_ARTIFACT_SHA256
V3_REFERENCE = v4.V3_REFERENCE

# 判据与结果标签（协议 §4 字面数值，生成前冻结）。
VERDICT_CRITERIA = {
    "primary_drift": {
        "dataset": "nltcs",
        "arm": "residual",
        "drift_ratio_max": 1.100,
        "note": "v2_v3_v4_original_target_not_relaxed",
    },
    "quality_gate_measured_normalized_l1_max": {
        "nltcs": 0.000197,
        "test_300x10": 0.002640,
    },
    "quality_gate_rule": "1.10x_v3_reference_no_regression_both_must_pass",
    "morphology": {
        "dataset": "nltcs",
        "arm": "residual",
        "tail_window": [8000, 9000],
        "prev_window": [7000, 8000],
        "last_best_update_min_round": 8000,
        "descending_definition": (
            "mean(loss_history[8000:9000]) < mean(loss_history[7000:8000])"
            " AND last_running_best_update_round >= 8000"
        ),
    },
    "verdict_labels": {
        "quality_gate_failed": "quality_regression_under_eta_cooling",
        "quality_pass_drift_pass": "eta_cooling_supported",
        "quality_pass_drift_fail_descending": (
            "eta_cooling_budget_insufficient"
        ),
        "quality_pass_drift_fail_not_descending": "eta_cooling_rejected",
        "prefix_audit_failed": "schedule_blindness_violated",
    },
    "rule": (
        "labels_mutually_exclusive; "
        "post_hoc_constant_or_threshold_retuning_forbidden; "
        "no_further_eta_lowering_if_rejected; "
        "v2_unsupported_v3_rejected_v4_quality_regression_records_stand"
    ),
}

FLOOR_SEGMENT_ROUNDS = 500  # 观察项：eta 地板段分段均值窗口宽度。
ETA_FLOOR_START = ETA_ANNEAL_START_ROUND + ETA_ANNEAL_ROUNDS  # 2100


def _json_protocol_manifest() -> dict[str, Any]:
    manifest = v3._json_protocol_manifest()
    manifest["contract_version"] = PROTOCOL_VERSION
    manifest["purpose"] = (
        "development_eta_cooling_screen_only_no_formal_quality_claim"
    )
    manifest["inherits"] = {
        "base_contract_version": v3.PROTOCOL_VERSION,
        "base_protocol_sha256": v3.FROZEN_PROTOCOL_SHA256,
        "v4_contract_version": v4.PROTOCOL_VERSION,
        "v4_protocol_sha256": v4.FROZEN_PROTOCOL_SHA256,
        "v2_contract_version": v2.PROTOCOL_VERSION,
        "v2_protocol_sha256": v2.FROZEN_PROTOCOL_SHA256,
        "v1_contract_version": v1.PROTOCOL_VERSION,
        "v1_protocol_sha256": v1.FROZEN_PROTOCOL_SHA256,
        "only_change": (
            "floor_phase_eta_cooling_05_to_025_and_budget_9000_"
            "rho_schedule_v3_verbatim"
        ),
        "design_protocol_doc": DESIGN_PROTOCOL_DOC,
    }
    manifest["n_rounds"] = N_ROUNDS
    manifest["generation_config"]["eta_anneal_start_round"] = (
        ETA_ANNEAL_START_ROUND
    )
    manifest["generation_config"]["eta_anneal_rounds"] = ETA_ANNEAL_ROUNDS
    manifest["generation_config"]["eta_anneal_end"] = ETA_ANNEAL_END
    manifest["eta_schedule"] = {
        "kind": "three_phase_blind_time_schedule",
        "formula": (
            "eta_t = eta0 * (floor/eta0) ** "
            "min(1, max(0, (t - H_eta) / D_eta))"
        ),
        "eta0": ETA_BASE,
        "hold_rounds_H_eta": ETA_ANNEAL_START_ROUND,
        "descent_rounds_D_eta": ETA_ANNEAL_ROUNDS,
        "floor": ETA_ANNEAL_END,
        "floor_ratio": 0.5,
        "staggered_after_rho_floor_landing": True,
        "threshold_only_no_random_stream_change": True,
        "depends_only_on_round_index": True,
        "reads_no_residual_loss_or_candidate_evaluation": True,
        "shared_by_both_arms": True,
        "formula_contains_no_total_round_count": True,
        "constants_frozen_before_any_v5_result": True,
        "per_dataset_tuning_forbidden": True,
    }
    del manifest["truncation_hypothesis"]
    manifest["jump_magnitude_hypothesis"] = {
        "statement": (
            "v4_showed_participation_frequency_halving_does_not_narrow_"
            "nltcs_band_1p1639_to_1p1754;"
            "steady_state_band_width_is_dominated_by_per_row_copy_"
            "magnitude_diff_attrs_times_eta;"
            "halving_floor_phase_eta_to_025_narrows_band_to_reach_"
            "drift_1p100"
        ),
        "changes": {
            "eta_schedule": {
                "from": "constant_0.5",
                "to": (
                    "three_phase_hold1500_descent600_floor025"
                ),
            },
            "n_rounds": {"from": v3.N_ROUNDS, "to": N_ROUNDS},
        },
        "rho_schedule_v3_verbatim": True,
        "design_choice": (
            "staggered_cooling_eta_descends_only_after_rho_floor_landing"
        ),
        "expected_participation_rows_per_round_floor": {
            "nltcs": 16.181,
            "test_300x10": 0.3,
        },
        "expected_copy_magnitude_ratio_on_floor": 0.5,
        "v2_unsupported_v3_rejected_v4_quality_regression_records_stand": (
            True
        ),
    }
    manifest["prefix_consistency_audit"] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "prefix_definition": (
            "rho_schedule_v3_verbatim_all_rounds;"
            "eta_hold_phase_plus_t1500_boundary_where_progress_is_exactly_"
            "zero;schedules_diverge_from_t1501"
        ),
        "compared_fields_bit_for_bit": [
            "loss_history[:1501]",
            "rho_schedule_history[:1501]",
            "initial_table_sha256",
            "primary_rng_post_initialization_state_sha256",
        ],
        "final_rng_state_expected_to_differ": True,
        "failure_label": "schedule_blindness_violated",
        "failure_consequence": (
            "run_invalid_no_quality_interpretation_architecture_must_be_"
            "fixed_first"
        ),
        "v3_reference_output_dir": str(V3_OUTPUT_DIR),
        "v3_reference_artifact_sha256": V3_REFERENCE_ARTIFACT_SHA256,
    }
    manifest["baseline_v3_reference"] = V3_REFERENCE
    manifest["baseline_v4_record"] = {
        "output_dir": str(v4.OUTPUT_DIR),
        "protocol_sha256": v4.FROZEN_PROTOCOL_SHA256,
        "verdict": "quality_regression_under_lower_floor",
        "nltcs_drift_ratio": 1.1753874591212854,
        "nltcs_measured_normalized_l1_mean": 0.0001647817604804194,
        "test_measured_normalized_l1_mean": 0.0027333333333333333,
        "test_drift_ratio": 3.1363636363636362,
        "note": (
            "frequency_halving_did_not_narrow_band_and_froze_small_table;"
            "record_stands_not_a_retake"
        ),
    }
    manifest["support_criteria"] = VERDICT_CRITERIA
    return manifest


def protocol_sha256() -> str:
    return v1._sha256_bytes(v1._strict_json_bytes(_json_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "fitness-only v5 eta-cooling 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    """Build a plan without reading any input file or result artifact."""

    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "protocol": _json_protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "trajectory_count": len(v1.DATASETS) * len(v1.ARMS),
        "generation_started": False,
    }


def _fitness_config(spec: Mapping[str, Any]) -> FitnessOnlyConfig:
    """v3 config plus floor-phase eta cooling and its budget — the v5 delta.

    Deliberately built on the *v3* config (rho floor 0.001), not v4.
    """

    return dataclasses.replace(
        v3._fitness_config(spec),
        eta_anneal_start_round=ETA_ANNEAL_START_ROUND,
        eta_anneal_rounds=ETA_ANNEAL_ROUNDS,
        eta_anneal_end=ETA_ANNEAL_END,
        n_rounds=N_ROUNDS,
    )


def _source_snapshot(root: Path) -> dict[str, Any]:
    tracked_sources = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/fitness_only.py"),
        Path("scripts/run_fitness_only_attribution.py"),
        Path("scripts/run_fitness_only_schedule_v2.py"),
        Path("scripts/run_fitness_only_schedule_v3_t6000.py"),
        Path("scripts/run_fitness_only_floor00005_v4.py"),
    )
    runner_path = Path(__file__).resolve().relative_to(root)
    paths = (*tracked_sources, runner_path)
    return {
        "git_commit": v1._git_text(root, "rev-parse", "HEAD"),
        "source_sha256": {
            str(path): v1._sha256_file(root / path) for path in paths
        },
        "git_status_porcelain": v1._git_text(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
        ),
    }


def _expected_rho_schedule(n_rounds: int) -> list[float]:
    """Recompute the v3-verbatim rho schedule with the same float expression."""

    rho0 = 0.01
    values = []
    for t in range(n_rounds):
        progress = min(
            1.0,
            max(
                0.0,
                (t - v2.RHO_ANNEAL_START_ROUND) / v2.RHO_ANNEAL_ROUNDS,
            ),
        )
        values.append(rho0 * (v2.RHO_ANNEAL_END / rho0) ** progress)
    return values


def _expected_eta_schedule(n_rounds: int) -> list[float]:
    """Recompute the frozen v5 eta schedule with the same float expression."""

    values = []
    for t in range(n_rounds):
        progress = min(
            1.0,
            max(
                0.0,
                (t - ETA_ANNEAL_START_ROUND) / ETA_ANNEAL_ROUNDS,
            ),
        )
        values.append(ETA_BASE * (ETA_ANNEAL_END / ETA_BASE) ** progress)
    return values


def _audit_schedule(staging: Path) -> dict[str, Any]:
    """Verify every arm followed both frozen schedules over all 9000 rounds."""

    expected_rho = _expected_rho_schedule(N_ROUNDS)
    expected_eta = _expected_eta_schedule(N_ROUNDS)
    rho_boundary_rounds = [
        0,
        v2.RHO_ANNEAL_START_ROUND - 1,
        v2.RHO_ANNEAL_START_ROUND,
        v2.RHO_ANNEAL_START_ROUND + 1,
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS - 1,
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS,
        N_ROUNDS - 1,
    ]
    eta_boundary_rounds = [
        0,
        ETA_ANNEAL_START_ROUND - 1,
        ETA_ANNEAL_START_ROUND,
        ETA_ANNEAL_START_ROUND + 1,
        ETA_FLOOR_START - 1,
        ETA_FLOOR_START,
        N_ROUNDS - 1,
    ]
    audit: dict[str, Any] = {
        "formula_checked_per_round": True,
        "rho_boundary_rounds": rho_boundary_rounds,
        "expected_rho_boundary_values": [
            expected_rho[t] for t in rho_boundary_rounds
        ],
        "eta_boundary_rounds": eta_boundary_rounds,
        "expected_eta_boundary_values": [
            expected_eta[t] for t in eta_boundary_rounds
        ],
        "arms": {},
    }
    for name in v1.DATASETS:
        audit["arms"][name] = {}
        for arm in v1.ARMS:
            diagnostics = v1._load_json_object(
                staging / "generation" / name / f"{arm}_diagnostics.json"
            )
            rho_history = diagnostics.get("rho_schedule_history")
            if rho_history != expected_rho:
                raise RuntimeError(
                    f"{name}/{arm} rho_schedule_history 偏离 v3 原样时间表"
                )
            eta_history = diagnostics.get("eta_schedule_history")
            if eta_history != expected_eta:
                raise RuntimeError(
                    f"{name}/{arm} eta_schedule_history 偏离预冻结时间表"
                )
            audit["arms"][name][arm] = {
                "rho_schedule_matches_v3_verbatim_formula": True,
                "eta_schedule_matches_frozen_formula": True,
                "rounds": len(rho_history),
                "rho_boundary_values": [
                    rho_history[t] for t in rho_boundary_rounds
                ],
                "eta_boundary_values": [
                    eta_history[t] for t in eta_boundary_rounds
                ],
                "output_squared_loss": float(
                    diagnostics["output_squared_loss"]
                ),
                "best_loss_diagnostic_only": float(
                    diagnostics["best_loss_diagnostic_only"]
                ),
            }
    return audit


def _prefix_audit(
    v5_diagnostics: Mapping[str, Mapping[str, Any]],
    v3_diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bit-for-bit 1501-round prefix comparison against v3."""

    audit: dict[str, Any] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "arms": {},
    }
    passed = True
    for name in v1.DATASETS:
        audit["arms"][name] = {}
        for arm in v1.ARMS:
            new = v5_diagnostics[name][arm]
            old = v3_diagnostics[name][arm]
            loss_prefix = list(new["loss_history"][:PREFIX_ROUNDS])
            rho_prefix = list(new["rho_schedule_history"][:PREFIX_ROUNDS])
            old_loss = list(old["loss_history"][:PREFIX_ROUNDS])
            old_rho = list(old["rho_schedule_history"][:PREFIX_ROUNDS])
            checks = {
                "loss_history_prefix_identical": loss_prefix == old_loss,
                "rho_schedule_history_prefix_identical": (
                    rho_prefix == old_rho
                ),
                "initial_table_sha256_identical": (
                    new["initial_table_sha256"]
                    == old["initial_table_sha256"]
                ),
                "primary_rng_post_initialization_state_identical": (
                    new["primary_rng_post_initialization_state_sha256"]
                    == old["primary_rng_post_initialization_state_sha256"]
                ),
            }
            detail: dict[str, Any] = dict(checks)
            if not checks["loss_history_prefix_identical"]:
                detail["loss_history_first_mismatch_round"] = (
                    v3._first_mismatch_index(loss_prefix, old_loss)
                )
            if not checks["rho_schedule_history_prefix_identical"]:
                detail["rho_schedule_first_mismatch_round"] = (
                    v3._first_mismatch_index(rho_prefix, old_rho)
                )
            arm_passed = all(checks.values())
            detail["passed"] = arm_passed
            passed = passed and arm_passed
            audit["arms"][name][arm] = detail
    audit["passed"] = passed
    audit["final_rng_state_expected_to_differ"] = True
    audit["failure_label"] = (
        None if passed
        else VERDICT_CRITERIA["verdict_labels"]["prefix_audit_failed"]
    )
    return audit


def _floor_morphology(loss_history: Sequence[float]) -> dict[str, Any]:
    """Frozen descending rule plus observation-only eta-floor-phase shape."""

    spec = VERDICT_CRITERIA["morphology"]
    tail_lo, tail_hi = spec["tail_window"]
    prev_lo, prev_hi = spec["prev_window"]
    tail = loss_history[tail_lo:tail_hi]
    prev = loss_history[prev_lo:prev_hi]
    tail_mean = sum(tail) / len(tail)
    prev_mean = sum(prev) / len(prev)
    last_best_update = v3._last_running_best_update_round(loss_history)
    descending = (
        tail_mean < prev_mean
        and last_best_update >= spec["last_best_update_min_round"]
    )

    segment_means = []
    for lo in range(ETA_FLOOR_START, len(loss_history), FLOOR_SEGMENT_ROUNDS):
        segment = loss_history[lo:lo + FLOOR_SEGMENT_ROUNDS]
        segment_means.append({
            "rounds": [lo, lo + len(segment)],
            "mean_loss": sum(segment) / len(segment),
        })
    floor_best_updates = 0
    best = (
        min(loss_history[:ETA_FLOOR_START])
        if ETA_FLOOR_START else float("inf")
    )
    for value in loss_history[ETA_FLOOR_START:]:
        if value < best:
            best = value
            floor_best_updates += 1
    return {
        "tail_window": [tail_lo, tail_hi],
        "prev_window": [prev_lo, prev_hi],
        "tail_mean_loss": tail_mean,
        "prev_mean_loss": prev_mean,
        "tail_below_prev": tail_mean < prev_mean,
        "last_running_best_update_round": last_best_update,
        "last_best_update_min_round": spec["last_best_update_min_round"],
        "descending": descending,
        "eta_floor_start_round": ETA_FLOOR_START,
        "observation_floor_segment_means": segment_means,
        "observation_floor_running_best_updates": floor_best_updates,
    }


def _verdict_evaluation(
    schedule_audit: Mapping[str, Any],
    quality_by_dataset: Mapping[str, Mapping[str, Any]],
    nltcs_residual_loss_history: Sequence[float],
) -> dict[str, Any]:
    """Apply the pre-frozen v5 criteria and mutually exclusive labels."""

    per_dataset: dict[str, Any] = {}
    for name in v1.DATASETS:
        arm = schedule_audit["arms"][name]["residual"]
        best = arm["best_loss_diagnostic_only"]
        output = arm["output_squared_loss"]
        measured_l1 = float(
            quality_by_dataset[name]["residual"]["measured"][
                "normalized_l1_mean"
            ]
        )
        reference = V3_REFERENCE["residual_arm"][name]
        per_dataset[name] = {
            "best_loss_diagnostic_only": best,
            "output_squared_loss": output,
            "drift_ratio": output / best if best > 0 else float("inf"),
            "drift_ratio_v3": reference["drift_ratio"],
            "measured_normalized_l1_mean": measured_l1,
            "measured_normalized_l1_v3": (
                reference["measured_normalized_l1_mean"]
            ),
        }

    drift_spec = VERDICT_CRITERIA["primary_drift"]
    drift_ratio = per_dataset[drift_spec["dataset"]]["drift_ratio"]
    drift_pass = drift_ratio <= drift_spec["drift_ratio_max"]

    l1_max = VERDICT_CRITERIA["quality_gate_measured_normalized_l1_max"]
    quality_gate = {}
    quality_pass = True
    for name in v1.DATASETS:
        observed = per_dataset[name]["measured_normalized_l1_mean"]
        gate_pass = observed <= l1_max[name]
        quality_gate[name] = {
            "measured_normalized_l1_mean": observed,
            "max_allowed": l1_max[name],
            "pass": gate_pass,
        }
        quality_pass = quality_pass and gate_pass

    morphology = _floor_morphology(nltcs_residual_loss_history)
    labels = VERDICT_CRITERIA["verdict_labels"]
    if not quality_pass:
        verdict = labels["quality_gate_failed"]
    elif drift_pass:
        verdict = labels["quality_pass_drift_pass"]
    elif morphology["descending"]:
        verdict = labels["quality_pass_drift_fail_descending"]
    else:
        verdict = labels["quality_pass_drift_fail_not_descending"]

    return {
        "criteria": VERDICT_CRITERIA,
        "per_dataset_residual_arm": per_dataset,
        "primary_drift": {
            "dataset": drift_spec["dataset"],
            "drift_ratio": drift_ratio,
            "drift_ratio_max_allowed": drift_spec["drift_ratio_max"],
            "pass": drift_pass,
        },
        "quality_gate": {
            "per_dataset": quality_gate,
            "pass": quality_pass,
        },
        "nltcs_floor_morphology": morphology,
        "primary_verdict": verdict,
        "development_only": True,
        "formal_claim_allowed": False,
    }


def _heldout_observations(
    paired_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the no-hard-gate observation items next to v3 references."""

    observations: dict[str, Any] = {
        "interpretation": "observation_only_no_hard_gate",
    }
    for name in v1.DATASETS:
        delta = paired_comparison[name]["delta_residual_minus_equal"]
        observations[name] = {
            "heldout_3way_normalized_l1_mean_delta": delta[
                "heldout_3way_normalized_l1_mean"
            ],
            "heldout_4way_normalized_l1_mean_delta": delta[
                "heldout_4way_normalized_l1_mean"
            ],
            "v3_reference": dict(
                V3_REFERENCE["heldout_paired_delta_residual_minus_equal"][
                    name
                ]
            ),
        }
    return observations


def run(confirmed_protocol_sha256: str) -> Path:
    expected_protocol = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected_protocol:
        raise ValueError(
            "protocol SHA-256 确认值不一致："
            f"expected={expected_protocol}, "
            f"received={confirmed_protocol_sha256}"
        )
    root = v1._repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")

    v3_artifact_audit = v4._verify_v3_reference_artifacts(root)
    source_before = _source_snapshot(root)
    generation_audits = v1._audit_generation_inputs(root)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        generation_results = v1._run_generation(
            root,
            staging,
            generation_audits,
            config_factory=_fitness_config,
        )
        source_after_generation = _source_snapshot(root)
        v1._audit_source_unchanged(source_before, source_after_generation)
        schedule_audit = _audit_schedule(staging)

        v5_diagnostics: dict[str, dict[str, Any]] = {}
        for name in v1.DATASETS:
            v5_diagnostics[name] = {}
            for arm in v1.ARMS:
                v5_diagnostics[name][arm] = v1._load_json_object(
                    staging / "generation" / name / f"{arm}_diagnostics.json"
                )
        v3_diagnostics = v4._load_v3_reference_diagnostics(root)
        prefix_audit = _prefix_audit(v5_diagnostics, v3_diagnostics)

        report: dict[str, Any] = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected_protocol,
            "protocol": _json_protocol_manifest(),
            "source_before_generation": source_before,
            "source_after_generation": source_after_generation,
            "v3_reference_artifact_audit": v3_artifact_audit,
            "generation_inputs": {
                name: {
                    key: value
                    for key, value in audit.items()
                    if key not in {"queries", "targets"}
                }
                for name, audit in generation_audits.items()
            },
            "generation": generation_results,
            "schedule_audit": schedule_audit,
            "prefix_consistency_audit": prefix_audit,
        }
        summary_common = {
            "development_only": True,
            "formal_claim_allowed": False,
            "only_change_vs_v3": (
                "floor_phase_eta_cooling_05_to_025_and_budget_9000_"
                "rho_schedule_v3_verbatim"
            ),
            "all_generation_terminal_current": True,
            "all_generation_fixed_rounds": all(
                generation_results[name]["arms"][arm]["rounds_run"]
                == N_ROUNDS
                for name in v1.DATASETS
                for arm in v1.ARMS
            ),
            "all_generation_unconditional": True,
            "schedule_audit_passed": True,
            "parameter_retuning_performed": False,
            "v2_unsupported_v3_rejected_v4_quality_regression_records_stand": (
                True
            ),
        }

        if not prefix_audit["passed"]:
            # 时间表盲性被证伪：运行无效。保留生成产物供架构诊断，但不
            # 打开 held-out 答案与参考表（不产生任何可解读的质量结果）。
            report["quality"] = None
            report["paired_quality_comparison"] = None
            report["heldout_observations"] = None
            report["support_evaluation"] = None
            report["summary"] = {
                **summary_common,
                "prefix_audit_passed": False,
                "primary_verdict": prefix_audit["failure_label"],
                "quality_interpretation_forbidden": True,
                "reference_loaded_after_generation": False,
            }
            report["completed_at_unix"] = time.time()
            v1._write_json(staging / "report.json", report)
            os.replace(staging, destination)
            return destination / "report.json"

        # Explicit phase boundary: held-out answers and raw reference tables
        # are opened only after every generation trajectory has finished and
        # the prefix audit has passed.
        quality_by_dataset: dict[str, dict[str, Any]] = {}
        for name in v1.DATASETS:
            quality_by_dataset[name] = {}
            for arm in v1.ARMS:
                quality_by_dataset[name][arm] = v1._evaluate_one(
                    root,
                    name,
                    arm,
                    generation_audits[name],
                    generation_results[name]["arms"][arm],
                    staging,
                )
        paired_comparison = v1._paired_quality_comparison(quality_by_dataset)
        support_evaluation = _verdict_evaluation(
            schedule_audit,
            quality_by_dataset,
            v5_diagnostics["nltcs"]["residual"]["loss_history"],
        )

        report["quality"] = quality_by_dataset
        report["paired_quality_comparison"] = paired_comparison
        report["heldout_observations"] = _heldout_observations(
            paired_comparison
        )
        report["support_evaluation"] = support_evaluation
        report["summary"] = {
            **summary_common,
            "prefix_audit_passed": True,
            "primary_verdict": support_evaluation["primary_verdict"],
            "reference_loaded_after_generation": True,
        }
        report["completed_at_unix"] = time.time()
        v1._write_json(staging / "report.json", report)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "report.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--confirm-protocol-sha", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    report_path = run(args.confirm_protocol_sha)
    report = v1._load_json_object(report_path)
    verdict = report["summary"]["primary_verdict"]
    print(f"fitness-only eta-cooling v5 report -> {report_path}")
    print(f"primary_verdict = {verdict}")


if __name__ == "__main__":
    main()
