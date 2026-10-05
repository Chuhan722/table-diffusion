#!/usr/bin/env python3
"""Run the fitness-only lower-floor (0.0005) development screen (v4).

This is a *delta executor* over the frozen v3 T=6000 screen
(``scripts/run_fitness_only_schedule_v3_t6000.py``), which inherits the v2
schedule screen and the v1 attribution screen.  The changes are::

    rho_anneal_end : 0.001 -> 0.0005   (the single schedule-constant change)
    n_rounds       : 6000  -> 9000     (budget only; repair speed halves on
                                        the colder floor, so the floor phase
                                        gets ~2x the v3 steady-state time
                                        plus margin)

The three-phase shape is unchanged (hold H=900 / geometric descent D=600 /
open-ended floor).  v3 established that drift ~1.16 on the 0.001 floor is a
*steady-state oscillation baseline* (chain plateaued from ~r4400), not a
budget artifact.  The v4 screen tests exactly one pre-frozen hypothesis
(lower floor): halving the per-round participation (nltcs ~8 rows/round)
narrows the oscillation band enough to reach the original drift target
<= 1.100 without hurting measured/held-out quality.  Criteria, labels and
the descending morphology rule were frozen before this run in
``docs/设计/FitnessOnly地板0.0005降温v4结果前协议.md``.

Falsifiable audit: the v4 and v3 rho schedules coincide pointwise for
t in [0, 900] (901 rounds — the hold phase plus the t=900 boundary where
the anneal progress is exactly zero) and diverge from t=901.  Because the
schedule depends only on the round index, every v4 arm must match the
corresponding v3 arm bit-for-bit over those 901 rounds.  Any mismatch
invalidates the whole run (``schedule_blindness_violated``).

This remains a single-development-seed screen: no formal quality claim, no
promotion gate, never overwrites an existing output.  v4 is *not* a retake
of v2/v3 — their unsupported/rejected records stand unchanged.
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
from table_diffevo.fitness_only import FitnessOnlyConfig


PROTOCOL_VERSION = "fitness-only-lower-floor-00005-v4"
OUTPUT_DIR = Path("outputs/fitness_only_floor00005_dev_seed9908_v4")
DESIGN_PROTOCOL_DOC = (
    "docs/设计/FitnessOnly地板0.0005降温v4结果前协议.md"
)

N_ROUNDS = 9000
RHO_ANNEAL_END = 0.0005
# v4 与 v3 的时间表在 t∈[0,900]（含 t=900，progress 恰为 0）逐点相同，
# 自 t=901 起几何降温底数不同而分叉。
PREFIX_ROUNDS = v2.RHO_ANNEAL_START_ROUND + 1  # 901

FROZEN_PROTOCOL_SHA256 = (
    "3002afe55683f08c9b87ac5e8a59b0f45ca938406cc09e8287d1ddfb31d84cd2"
)

# v3 参照产物（只读）。身份在协议 §3 预冻结；运行前 fail-closed 核对。
V3_OUTPUT_DIR = v3.OUTPUT_DIR
V3_REFERENCE_ARTIFACT_SHA256 = {
    "report.json": (
        "0a613bb4bdb695f53469d5b7bb132725eb4a3df66f9404be707f81280d14dd48"
    ),
    "generation/nltcs/residual_diagnostics.json": (
        "d400be58270b713cd254e08f4dd9434adff0b7f2e07e1acbfee6850c9845b453"
    ),
    "generation/nltcs/equal_diagnostics.json": (
        "336f0cf45a3dea486a79dea183ef495ba4b16678a7f1ee8523ce500911869c1d"
    ),
    "generation/test_300x10/residual_diagnostics.json": (
        "c73ce211dd21545365065e645ae897ccfeb3f21a256af28e44959d6a80a2722e"
    ),
    "generation/test_300x10/equal_diagnostics.json": (
        "ead9f446f72d786444f14c960a8ac52fd339ab0dc8df9622e8703c29e3e40956"
    ),
}

# v3 冻结参照（residual 臂精确值，来自哈希钉死的 v3 report；只读常数）。
V3_REFERENCE = {
    "output_dir": str(V3_OUTPUT_DIR),
    "protocol_sha256": v3.FROZEN_PROTOCOL_SHA256,
    "verdict": "truncation_hypothesis_rejected",
    "residual_arm": {
        "nltcs": {
            "best_loss_diagnostic_only": 7146.5,
            "output_squared_loss": 8317.5,
            "drift_ratio": 1.1638564332190582,
            "measured_normalized_l1_mean": 0.00017922871887398185,
        },
        "test_300x10": {
            "best_loss_diagnostic_only": 12.5,
            "output_squared_loss": 28.0,
            "drift_ratio": 2.2400000000000002,
            "measured_normalized_l1_mean": 0.0024000000000000002,
        },
    },
    "heldout_paired_delta_residual_minus_equal": {
        "nltcs": {
            "heldout_3way_normalized_l1_mean": -0.061454333979667514,
            "heldout_4way_normalized_l1_mean": -0.037263032260058095,
        },
        "test_300x10": {
            "heldout_3way_normalized_l1_mean": 0.0033398437500000017,
            "heldout_4way_normalized_l1_mean": 0.00099609375000000045,
        },
    },
}

# 判据与结果标签（协议 §4 字面数值，生成前冻结）。
VERDICT_CRITERIA = {
    "primary_drift": {
        "dataset": "nltcs",
        "arm": "residual",
        "drift_ratio_max": 1.100,
        "note": "v2_v3_original_target_not_relaxed",
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
        "quality_gate_failed": "quality_regression_under_lower_floor",
        "quality_pass_drift_pass": "lower_floor_supported",
        "quality_pass_drift_fail_descending": (
            "lower_floor_budget_insufficient"
        ),
        "quality_pass_drift_fail_not_descending": "lower_floor_rejected",
        "prefix_audit_failed": "schedule_blindness_violated",
    },
    "rule": (
        "labels_mutually_exclusive; "
        "post_hoc_constant_or_threshold_retuning_forbidden; "
        "no_further_floor_lowering_if_rejected; "
        "v2_unsupported_and_v3_rejected_records_stand"
    ),
}

FLOOR_SEGMENT_ROUNDS = 500  # 观察项：地板段分段均值窗口宽度。


def _json_protocol_manifest() -> dict[str, Any]:
    manifest = v3._json_protocol_manifest()
    manifest["contract_version"] = PROTOCOL_VERSION
    manifest["purpose"] = (
        "development_lower_floor_screen_only_no_formal_quality_claim"
    )
    manifest["inherits"] = {
        "base_contract_version": v3.PROTOCOL_VERSION,
        "base_protocol_sha256": v3.FROZEN_PROTOCOL_SHA256,
        "v2_contract_version": v2.PROTOCOL_VERSION,
        "v2_protocol_sha256": v2.FROZEN_PROTOCOL_SHA256,
        "v1_contract_version": v1.PROTOCOL_VERSION,
        "v1_protocol_sha256": v1.FROZEN_PROTOCOL_SHA256,
        "only_change": "rho_anneal_end_0001_to_00005_and_budget_9000",
        "design_protocol_doc": DESIGN_PROTOCOL_DOC,
    }
    manifest["n_rounds"] = N_ROUNDS
    manifest["generation_config"]["rho_anneal_end"] = RHO_ANNEAL_END
    manifest["rho_schedule"] = {
        "kind": "three_phase_blind_time_schedule",
        "formula": (
            "rho_t = rho0 * (floor/rho0) ** min(1, max(0, (t - H) / D))"
        ),
        "hold_rounds_H": v2.RHO_ANNEAL_START_ROUND,
        "descent_rounds_D": v2.RHO_ANNEAL_ROUNDS,
        "floor": RHO_ANNEAL_END,
        "floor_ratio": 0.05,
        "depends_only_on_round_index": True,
        "reads_no_residual_loss_or_candidate_evaluation": True,
        "shared_by_both_arms": True,
        "formula_contains_no_total_round_count": True,
        "constants_frozen_before_any_v4_result": True,
        "per_dataset_tuning_forbidden": True,
    }
    del manifest["truncation_hypothesis"]
    manifest["lower_floor_hypothesis"] = {
        "statement": (
            "nltcs_v3_drift_baseline_1p16_is_floor_temperature_oscillation;"
            "halving_floor_to_00005_narrows_band_to_reach_drift_1p100"
        ),
        "changes": {
            "rho_anneal_end": {"from": v2.RHO_ANNEAL_END, "to": RHO_ANNEAL_END},
            "n_rounds": {"from": v3.N_ROUNDS, "to": N_ROUNDS},
        },
        "hold_and_descent_rounds_unchanged": True,
        "design_choice": "three_phase_direct_endpoint_not_four_phase",
        "expected_participation_rows_per_round_floor": {
            "nltcs": 8.0905,
            "test_300x10": 0.15,
        },
        "v2_unsupported_and_v3_rejected_records_stand": True,
    }
    manifest["prefix_consistency_audit"] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "prefix_definition": (
            "hold_phase_plus_t900_boundary_where_progress_is_exactly_zero;"
            "schedules_diverge_from_t901"
        ),
        "compared_fields_bit_for_bit": [
            "loss_history[:901]",
            "rho_schedule_history[:901]",
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
    manifest["support_criteria"] = VERDICT_CRITERIA
    return manifest


def protocol_sha256() -> str:
    return v1._sha256_bytes(v1._strict_json_bytes(_json_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "fitness-only v4 lower-floor 协议身份漂移："
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
    """v3 config plus the lower floor and its budget — the v4 delta."""

    return dataclasses.replace(
        v3._fitness_config(spec),
        rho_anneal_end=RHO_ANNEAL_END,
        n_rounds=N_ROUNDS,
    )


def _source_snapshot(root: Path) -> dict[str, Any]:
    tracked_sources = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/fitness_only.py"),
        Path("scripts/run_fitness_only_attribution.py"),
        Path("scripts/run_fitness_only_schedule_v2.py"),
        Path("scripts/run_fitness_only_schedule_v3_t6000.py"),
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


def _expected_schedule(n_rounds: int) -> list[float]:
    """Recompute the frozen v4 schedule with the same float expression."""

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
        values.append(rho0 * (RHO_ANNEAL_END / rho0) ** progress)
    return values


def _audit_schedule(staging: Path) -> dict[str, Any]:
    """Verify every arm followed the frozen schedule over all 9000 rounds."""

    expected = _expected_schedule(N_ROUNDS)
    boundary_rounds = [
        0,
        v2.RHO_ANNEAL_START_ROUND - 1,
        v2.RHO_ANNEAL_START_ROUND,
        v2.RHO_ANNEAL_START_ROUND + 1,
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS - 1,
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS,
        N_ROUNDS - 1,
    ]
    audit: dict[str, Any] = {
        "formula_checked_per_round": True,
        "boundary_rounds": boundary_rounds,
        "expected_boundary_values": [expected[t] for t in boundary_rounds],
        "arms": {},
    }
    for name in v1.DATASETS:
        audit["arms"][name] = {}
        for arm in v1.ARMS:
            diagnostics = v1._load_json_object(
                staging / "generation" / name / f"{arm}_diagnostics.json"
            )
            history = diagnostics.get("rho_schedule_history")
            if history != expected:
                raise RuntimeError(
                    f"{name}/{arm} rho_schedule_history 偏离预冻结时间表"
                )
            audit["arms"][name][arm] = {
                "schedule_matches_frozen_formula": True,
                "rounds": len(history),
                "boundary_values": [history[t] for t in boundary_rounds],
                "output_squared_loss": float(
                    diagnostics["output_squared_loss"]
                ),
                "best_loss_diagnostic_only": float(
                    diagnostics["best_loss_diagnostic_only"]
                ),
            }
    return audit


def _verify_v3_reference_artifacts(root: Path) -> dict[str, Any]:
    """Fail-closed identity check of the read-only v3 reference artifacts.

    Runs *before* generation (hash-only, no content parsing) so a missing or
    tampered reference aborts the run without wasting the GPU budget.
    """

    base = root / V3_OUTPUT_DIR
    observed: dict[str, str] = {}
    for rel_path, expected in V3_REFERENCE_ARTIFACT_SHA256.items():
        path = base / rel_path
        if not path.is_file():
            raise RuntimeError(f"v3 参照产物缺失：{path}")
        digest = v1._sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                "v3 参照产物身份漂移："
                f"{rel_path} expected={expected}, observed={digest}"
            )
        observed[rel_path] = digest
    return {
        "v3_reference_output_dir": str(V3_OUTPUT_DIR),
        "artifact_sha256_verified": observed,
        "verified_before_generation": True,
    }


def _load_v3_reference_diagnostics(root: Path) -> dict[str, dict[str, Any]]:
    """Load the hash-verified v3 diagnostics for the prefix audit."""

    diagnostics: dict[str, dict[str, Any]] = {}
    for name in v1.DATASETS:
        diagnostics[name] = {}
        for arm in v1.ARMS:
            diagnostics[name][arm] = v1._load_json_object(
                root
                / V3_OUTPUT_DIR
                / "generation"
                / name
                / f"{arm}_diagnostics.json"
            )
    return diagnostics


def _prefix_audit(
    v4_diagnostics: Mapping[str, Mapping[str, Any]],
    v3_diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bit-for-bit 901-round prefix comparison against v3."""

    audit: dict[str, Any] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "arms": {},
    }
    passed = True
    for name in v1.DATASETS:
        audit["arms"][name] = {}
        for arm in v1.ARMS:
            new = v4_diagnostics[name][arm]
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
    """Frozen descending rule plus observation-only floor-phase shape."""

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

    floor_start = v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS  # 1500
    segment_means = []
    for lo in range(floor_start, len(loss_history), FLOOR_SEGMENT_ROUNDS):
        segment = loss_history[lo:lo + FLOOR_SEGMENT_ROUNDS]
        segment_means.append({
            "rounds": [lo, lo + len(segment)],
            "mean_loss": sum(segment) / len(segment),
        })
    floor_best_updates = 0
    best = min(loss_history[:floor_start]) if floor_start else float("inf")
    for value in loss_history[floor_start:]:
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
        "observation_floor_segment_means": segment_means,
        "observation_floor_running_best_updates": floor_best_updates,
    }


def _verdict_evaluation(
    schedule_audit: Mapping[str, Any],
    quality_by_dataset: Mapping[str, Mapping[str, Any]],
    nltcs_residual_loss_history: Sequence[float],
) -> dict[str, Any]:
    """Apply the pre-frozen v4 criteria and mutually exclusive labels."""

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

    v3_artifact_audit = _verify_v3_reference_artifacts(root)
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

        v4_diagnostics: dict[str, dict[str, Any]] = {}
        for name in v1.DATASETS:
            v4_diagnostics[name] = {}
            for arm in v1.ARMS:
                v4_diagnostics[name][arm] = v1._load_json_object(
                    staging / "generation" / name / f"{arm}_diagnostics.json"
                )
        v3_diagnostics = _load_v3_reference_diagnostics(root)
        prefix_audit = _prefix_audit(v4_diagnostics, v3_diagnostics)

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
                "rho_anneal_end_0001_to_00005_and_budget_9000"
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
            "v2_unsupported_and_v3_rejected_records_stand": True,
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
            v4_diagnostics["nltcs"]["residual"]["loss_history"],
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
    print(f"fitness-only lower-floor v4 report -> {report_path}")
    print(f"primary_verdict = {verdict}")


if __name__ == "__main__":
    main()
