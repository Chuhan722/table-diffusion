#!/usr/bin/env python3
"""Run the fitness-only T=6000 truncation-hypothesis development screen (v3).

This is a *delta executor* over the frozen v2 schedule screen
(``scripts/run_fitness_only_schedule_v2.py``), which itself is a delta over
the frozen v1 attribution screen.  The one and only change is the total
round budget::

    n_rounds : 3000 -> 6000

The three-phase rho schedule (hold 900 / descent 600 / floor 0.001) is
inherited verbatim — its formula contains no total round count, so the
floor phase simply extends from 1500 to 4500 rounds without touching any
schedule definition.

The v3 screen tests exactly one pre-frozen hypothesis (truncation): nltcs
drift failed in v2 because the fixed budget cut the chain while it was
still descending.  Verdict labels, quality gates and the descending
morphology rule were frozen before this run in
``docs/设计/FitnessOnly时间表T6000截断假说v3结果前协议.md``.

New falsifiable audit: because trajectories must never read the total
round count (horizon invariance), the first 3000 rounds of every v3 arm
must equal the corresponding v2 arm bit-for-bit.  Any mismatch invalidates
the whole run (``horizon_invariance_violated``) and no quality result may
be interpreted.

This remains a single-development-seed screen: no formal quality claim, no
promotion gate, never overwrites an existing output.  v3 is *not* a retake
of v2 — the v2 ``schedule_dev_unsupported`` record stands unchanged.
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
from table_diffevo.fitness_only import FitnessOnlyConfig


PROTOCOL_VERSION = "fitness-only-schedule-t6000-truncation-v3"
OUTPUT_DIR = Path("outputs/fitness_only_schedule_T6000_dev_seed9908_v3")
DESIGN_PROTOCOL_DOC = (
    "docs/设计/FitnessOnly时间表T6000截断假说v3结果前协议.md"
)

N_ROUNDS = 6000
PREFIX_ROUNDS = v1.N_ROUNDS  # 3000 — the audited bit-identical prefix.

FROZEN_PROTOCOL_SHA256 = (
    "cd834439b90d6ffa7e7789c583019b92d07b3a763a4cf0a6357f30a0df8553b9"
)

# v2 参照产物（只读）。身份在协议 §3 预冻结；运行前 fail-closed 核对。
V2_OUTPUT_DIR = v2.OUTPUT_DIR
V2_REFERENCE_ARTIFACT_SHA256 = {
    "report.json": (
        "4f6d8e8ce6903955b619d95272af8e5ed31b5476767bf621afc852e616ac3476"
    ),
    "generation/nltcs/residual_diagnostics.json": (
        "21a42c2e8b8fd4e39fa828bd58b7aa074884fddb21c766c8f6c5d224a46fb8b5"
    ),
    "generation/nltcs/equal_diagnostics.json": (
        "e61f455a27b6fce94c515f51461408e3675849219ba40a53a15e231a346e4f41"
    ),
    "generation/test_300x10/residual_diagnostics.json": (
        "6b709c1450790d0e1baea25ec6fe00d405e7f868e782b524a35bbef2442c191d"
    ),
    "generation/test_300x10/equal_diagnostics.json": (
        "96cc6baaa8463ffa42070a47cb3cd0e9d8e403b1ff63258269c071f6eec69208"
    ),
}

# v2 冻结参照（residual 臂精确值，来自哈希钉死的 v2 report；只读常数）。
V2_REFERENCE = {
    "output_dir": str(V2_OUTPUT_DIR),
    "protocol_sha256": v2.FROZEN_PROTOCOL_SHA256,
    "verdict": "schedule_dev_unsupported",
    "residual_arm": {
        "nltcs": {
            "best_loss_diagnostic_only": 8409.0,
            "output_squared_loss": 10143.5,
            "drift_ratio": 1.206267094779403,
            "measured_normalized_l1_mean": 0.00018243915407255127,
        },
        "test_300x10": {
            "best_loss_diagnostic_only": 19.5,
            "output_squared_loss": 35.0,
            "drift_ratio": 1.794871794871795,
            "measured_normalized_l1_mean": 0.002533333333333333,
        },
    },
    "heldout_paired_delta_residual_minus_equal": {
        "nltcs": {
            "heldout_3way_normalized_l1_mean": -0.061482820321673556,
            "heldout_4way_normalized_l1_mean": -0.03737142520548792,
        },
        "test_300x10": {
            "heldout_3way_normalized_l1_mean": 0.0028125000000000025,
            "heldout_4way_normalized_l1_mean": 0.0005664062500000001,
        },
    },
}

# 判据与结果标签（协议 §4 字面数值，生成前冻结）。
VERDICT_CRITERIA = {
    "primary_drift": {
        "dataset": "nltcs",
        "arm": "residual",
        "drift_ratio_max": 1.100,
        "note": "v2_original_target_not_relaxed",
    },
    "quality_gate_measured_normalized_l1_max": {
        "nltcs": 0.000200,
        "test_300x10": 0.002786,
    },
    "quality_gate_rule": "1.10x_v2_reference_no_regression_both_must_pass",
    "morphology": {
        "dataset": "nltcs",
        "arm": "residual",
        "tail_window": [5000, 6000],
        "prev_window": [4000, 5000],
        "last_best_update_min_round": 5000,
        "descending_definition": (
            "mean(loss_history[5000:6000]) < mean(loss_history[4000:5000])"
            " AND last_running_best_update_round >= 5000"
        ),
    },
    "verdict_labels": {
        "quality_gate_failed": "quality_regression_under_extended_budget",
        "quality_pass_drift_pass": "truncation_hypothesis_supported",
        "quality_pass_drift_fail_descending": "budget_still_insufficient",
        "quality_pass_drift_fail_not_descending": (
            "truncation_hypothesis_rejected"
        ),
        "prefix_audit_failed": "horizon_invariance_violated",
    },
    "rule": (
        "labels_mutually_exclusive; "
        "post_hoc_constant_or_threshold_retuning_forbidden; "
        "v2_unsupported_record_stands_v3_is_not_a_retake"
    ),
}

FLOOR_SEGMENT_ROUNDS = 500  # 观察项：地板段分段均值窗口宽度。


def _json_protocol_manifest() -> dict[str, Any]:
    manifest = v2._json_protocol_manifest()
    manifest["contract_version"] = PROTOCOL_VERSION
    manifest["purpose"] = (
        "development_truncation_hypothesis_screen_only"
        "_no_formal_quality_claim"
    )
    manifest["inherits"] = {
        "base_contract_version": v2.PROTOCOL_VERSION,
        "base_protocol_sha256": v2.FROZEN_PROTOCOL_SHA256,
        "v1_contract_version": v1.PROTOCOL_VERSION,
        "v1_protocol_sha256": v1.FROZEN_PROTOCOL_SHA256,
        "only_change": "total_rounds_3000_to_6000",
        "design_protocol_doc": DESIGN_PROTOCOL_DOC,
    }
    manifest["n_rounds"] = N_ROUNDS
    manifest["truncation_hypothesis"] = {
        "statement": (
            "nltcs_v2_drift_failure_caused_by_fixed_budget_truncation"
            "_while_chain_still_descending"
        ),
        "only_change": {"n_rounds": {"from": v1.N_ROUNDS, "to": N_ROUNDS}},
        "schedule_constants_unchanged": True,
        "schedule_formula_contains_no_total_round_count": True,
        "floor_phase_rounds": {"from": 1500, "to": 4500},
        "v2_unsupported_record_stands": True,
    }
    manifest["prefix_consistency_audit"] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "compared_fields_bit_for_bit": [
            "loss_history[:3000]",
            "rho_schedule_history[:3000]",
            "initial_table_sha256",
            "primary_rng_post_initialization_state_sha256",
        ],
        "final_rng_state_expected_to_differ": True,
        "failure_label": "horizon_invariance_violated",
        "failure_consequence": (
            "run_invalid_no_quality_interpretation_architecture_must_be_"
            "fixed_first"
        ),
        "v2_reference_output_dir": str(V2_OUTPUT_DIR),
        "v2_reference_artifact_sha256": V2_REFERENCE_ARTIFACT_SHA256,
    }
    manifest["baseline_v2_reference"] = V2_REFERENCE
    manifest["support_criteria"] = VERDICT_CRITERIA
    return manifest


def protocol_sha256() -> str:
    return v1._sha256_bytes(v1._strict_json_bytes(_json_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "fitness-only v3 T6000 协议身份漂移："
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
    """v2 config plus the extended budget — the single delta of this screen."""

    return dataclasses.replace(v2._fitness_config(spec), n_rounds=N_ROUNDS)


def _source_snapshot(root: Path) -> dict[str, Any]:
    tracked_sources = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/fitness_only.py"),
        Path("scripts/run_fitness_only_attribution.py"),
        Path("scripts/run_fitness_only_schedule_v2.py"),
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


def _audit_schedule(staging: Path) -> dict[str, Any]:
    """Verify every arm followed the frozen schedule over all 6000 rounds."""

    expected = v2._expected_schedule(N_ROUNDS)
    boundary_rounds = [
        0,
        v2.RHO_ANNEAL_START_ROUND - 1,
        v2.RHO_ANNEAL_START_ROUND,
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS - 1,
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS,
        PREFIX_ROUNDS - 1,
        PREFIX_ROUNDS,
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


def _verify_v2_reference_artifacts(root: Path) -> dict[str, Any]:
    """Fail-closed identity check of the read-only v2 reference artifacts.

    Runs *before* generation (hash-only, no content parsing) so a missing or
    tampered reference aborts the run without wasting the GPU budget.
    """

    base = root / V2_OUTPUT_DIR
    observed: dict[str, str] = {}
    for rel_path, expected in V2_REFERENCE_ARTIFACT_SHA256.items():
        path = base / rel_path
        if not path.is_file():
            raise RuntimeError(f"v2 参照产物缺失：{path}")
        digest = v1._sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                "v2 参照产物身份漂移："
                f"{rel_path} expected={expected}, observed={digest}"
            )
        observed[rel_path] = digest
    return {
        "v2_reference_output_dir": str(V2_OUTPUT_DIR),
        "artifact_sha256_verified": observed,
        "verified_before_generation": True,
    }


def _load_v2_reference_diagnostics(root: Path) -> dict[str, dict[str, Any]]:
    """Load the hash-verified v2 diagnostics for the prefix audit."""

    diagnostics: dict[str, dict[str, Any]] = {}
    for name in v1.DATASETS:
        diagnostics[name] = {}
        for arm in v1.ARMS:
            diagnostics[name][arm] = v1._load_json_object(
                root
                / V2_OUTPUT_DIR
                / "generation"
                / name
                / f"{arm}_diagnostics.json"
            )
    return diagnostics


def _first_mismatch_index(
    left: Sequence[Any],
    right: Sequence[Any],
) -> int | None:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _prefix_audit(
    v3_diagnostics: Mapping[str, Mapping[str, Any]],
    v2_diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bit-for-bit prefix comparison against v2 (falsifiable horizon check)."""

    audit: dict[str, Any] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "arms": {},
    }
    passed = True
    for name in v1.DATASETS:
        audit["arms"][name] = {}
        for arm in v1.ARMS:
            new = v3_diagnostics[name][arm]
            old = v2_diagnostics[name][arm]
            loss_prefix = list(new["loss_history"][:PREFIX_ROUNDS])
            rho_prefix = list(new["rho_schedule_history"][:PREFIX_ROUNDS])
            old_loss = list(old["loss_history"])
            old_rho = list(old["rho_schedule_history"])
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
                    _first_mismatch_index(loss_prefix, old_loss)
                )
            if not checks["rho_schedule_history_prefix_identical"]:
                detail["rho_schedule_first_mismatch_round"] = (
                    _first_mismatch_index(rho_prefix, old_rho)
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


def _last_running_best_update_round(loss_history: Sequence[float]) -> int:
    best = float("inf")
    last_update = -1
    for round_index, value in enumerate(loss_history):
        if value < best:
            best = value
            last_update = round_index
    return last_update


def _floor_morphology(loss_history: Sequence[float]) -> dict[str, Any]:
    """Frozen descending rule plus observation-only floor-phase shape."""

    spec = VERDICT_CRITERIA["morphology"]
    tail_lo, tail_hi = spec["tail_window"]
    prev_lo, prev_hi = spec["prev_window"]
    tail = loss_history[tail_lo:tail_hi]
    prev = loss_history[prev_lo:prev_hi]
    tail_mean = sum(tail) / len(tail)
    prev_mean = sum(prev) / len(prev)
    last_best_update = _last_running_best_update_round(loss_history)
    descending = (
        tail_mean < prev_mean
        and last_best_update >= spec["last_best_update_min_round"]
    )

    floor_start = (
        v2.RHO_ANNEAL_START_ROUND + v2.RHO_ANNEAL_ROUNDS  # 1500
    )
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
    """Apply the pre-frozen v3 criteria and mutually exclusive labels."""

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
        reference = V2_REFERENCE["residual_arm"][name]
        per_dataset[name] = {
            "best_loss_diagnostic_only": best,
            "output_squared_loss": output,
            "drift_ratio": output / best if best > 0 else float("inf"),
            "drift_ratio_v2": reference["drift_ratio"],
            "measured_normalized_l1_mean": measured_l1,
            "measured_normalized_l1_v2": (
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
    """Record the no-hard-gate observation items next to v2 references."""

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
            "v2_reference": dict(
                V2_REFERENCE["heldout_paired_delta_residual_minus_equal"][
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

    v2_artifact_audit = _verify_v2_reference_artifacts(root)
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

        v3_diagnostics: dict[str, dict[str, Any]] = {}
        for name in v1.DATASETS:
            v3_diagnostics[name] = {}
            for arm in v1.ARMS:
                v3_diagnostics[name][arm] = v1._load_json_object(
                    staging / "generation" / name / f"{arm}_diagnostics.json"
                )
        v2_diagnostics = _load_v2_reference_diagnostics(root)
        prefix_audit = _prefix_audit(v3_diagnostics, v2_diagnostics)

        report: dict[str, Any] = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected_protocol,
            "protocol": _json_protocol_manifest(),
            "source_before_generation": source_before,
            "source_after_generation": source_after_generation,
            "v2_reference_artifact_audit": v2_artifact_audit,
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
            "only_change_vs_v2": "total_rounds_3000_to_6000",
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
            "v2_unsupported_record_stands": True,
        }

        if not prefix_audit["passed"]:
            # 视界不变被证伪：运行无效。保留生成产物供架构诊断，但不打开
            # held-out 答案与参考表（不产生任何可解读的质量结果）。
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
            v3_diagnostics["nltcs"]["residual"]["loss_history"],
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
    print(f"fitness-only schedule v3 T6000 report -> {report_path}")
    print(f"primary_verdict = {verdict}")


if __name__ == "__main__":
    main()
