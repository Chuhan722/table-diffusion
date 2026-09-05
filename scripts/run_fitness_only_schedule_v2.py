#!/usr/bin/env python3
"""Run the fitness-only three-phase rho schedule development screen (v2).

This is a *delta executor* over the frozen v1 attribution screen
(``scripts/run_fitness_only_attribution.py``).  The one and only change is
the pre-frozen three-phase rho schedule::

    t <  900         : rho_t = 0.01                        (hold)
    900 <= t < 1500  : rho_t = 0.01 * 0.1 ** ((t-900)/600)  (geometric descent)
    t >= 1500        : rho_t = 0.001                        (floor, open-ended)

Everything else — datasets, frozen inputs and hashes, seed, fixed rounds,
paired arms, evaluation pipeline and phase boundary — is inherited verbatim
from the v1 module.  The schedule constants were frozen before any v2 result
in ``docs/设计/FitnessOnly三段式rho时间表v2开发屏结果前协议.md`` and must not
be tuned per dataset or after seeing results.

This remains a single-development-seed screen: no formal quality claim, no
promotion gate, never overwrites an existing output.
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
from typing import Any, Mapping

from scripts import run_fitness_only_attribution as v1
from table_diffevo.fitness_only import FitnessOnlyConfig


PROTOCOL_VERSION = "fitness-only-schedule-development-v2"
OUTPUT_DIR = Path("outputs/fitness_only_schedule_dev_seed9908_v2")
DESIGN_PROTOCOL_DOC = (
    "docs/设计/FitnessOnly三段式rho时间表v2开发屏结果前协议.md"
)

RHO_ANNEAL_START_ROUND = 900
RHO_ANNEAL_ROUNDS = 600
RHO_ANNEAL_END = 0.001

FROZEN_PROTOCOL_SHA256 = (
    "bb7d19cb806a501a4cce6b139166a53cf8fe27f73140d951c15e4c3bb05a1531"
)

# v1 冻结参照（residual 臂，来自已批准协议第 3 节；只读常数，不读 v1 文件）。
V1_REFERENCE = {
    "output_dir": "outputs/fitness_only_attribution_dev_seed9908_v1",
    "protocol_sha256": v1.FROZEN_PROTOCOL_SHA256,
    "residual_arm": {
        "test_300x10": {
            "best_loss_diagnostic_only": 15.0,
            "output_squared_loss": 50.0,
            "drift_ratio": 3.333,
            "measured_normalized_l1_mean": 0.003200,
        },
        "nltcs": {
            "best_loss_diagnostic_only": 14870.0,
            "output_squared_loss": 17847.5,
            "drift_ratio": 1.2002,
            "measured_normalized_l1_mean": 0.000261,
        },
    },
    "heldout_paired_delta_test_300x10": {
        "heldout_3way_normalized_l1_mean": 0.005632,
        "heldout_4way_normalized_l1_mean": 0.001875,
    },
}

# 支持判据（已批准协议第 6 节字面数值，生成前冻结）。
SUPPORT_CRITERIA = {
    "drift_ratio_max": {"test_300x10": 2.167, "nltcs": 1.100},
    "measured_normalized_l1_max": {
        "test_300x10": 0.003520,
        "nltcs": 0.000287,
    },
    "rule": (
        "both_datasets_residual_arm_must_pass_both_criteria; "
        "failure_label=schedule_dev_unsupported; "
        "post_hoc_constant_retuning_forbidden"
    ),
}


def _json_protocol_manifest() -> dict[str, Any]:
    manifest = v1._json_protocol_manifest()
    manifest["contract_version"] = PROTOCOL_VERSION
    manifest["purpose"] = (
        "development_schedule_screen_only_no_formal_quality_claim"
    )
    manifest["inherits"] = {
        "base_contract_version": v1.PROTOCOL_VERSION,
        "base_protocol_sha256": v1.FROZEN_PROTOCOL_SHA256,
        "only_change": "pre_frozen_three_phase_rho_schedule",
        "design_protocol_doc": DESIGN_PROTOCOL_DOC,
    }
    manifest["generation_config"]["rho_anneal_end"] = RHO_ANNEAL_END
    manifest["generation_config"]["rho_anneal_rounds"] = RHO_ANNEAL_ROUNDS
    manifest["generation_config"]["rho_anneal_start_round"] = (
        RHO_ANNEAL_START_ROUND
    )
    manifest["rho_schedule"] = {
        "kind": "three_phase_blind_time_schedule",
        "formula": (
            "rho_t = rho0 * (floor/rho0) ** min(1, max(0, (t - H) / D))"
        ),
        "hold_rounds_H": RHO_ANNEAL_START_ROUND,
        "descent_rounds_D": RHO_ANNEAL_ROUNDS,
        "floor": RHO_ANNEAL_END,
        "floor_ratio": 0.1,
        "depends_only_on_round_index": True,
        "reads_no_residual_loss_or_candidate_evaluation": True,
        "shared_by_both_arms": True,
        "formula_contains_no_total_round_count": True,
        "constants_frozen_before_any_v2_result": True,
        "per_dataset_tuning_forbidden": True,
    }
    manifest["baseline_v1_reference"] = V1_REFERENCE
    manifest["support_criteria"] = SUPPORT_CRITERIA
    return manifest


def protocol_sha256() -> str:
    return v1._sha256_bytes(v1._strict_json_bytes(_json_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "fitness-only v2 时间表协议身份漂移："
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
    """v1 config plus the frozen schedule — the single delta of this screen."""

    return dataclasses.replace(
        v1._fitness_config(spec),
        rho_anneal_start_round=RHO_ANNEAL_START_ROUND,
        rho_anneal_rounds=RHO_ANNEAL_ROUNDS,
        rho_anneal_end=RHO_ANNEAL_END,
    )


def _source_snapshot(root: Path) -> dict[str, Any]:
    tracked_sources = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/fitness_only.py"),
        Path("scripts/run_fitness_only_attribution.py"),
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
    """Recompute the frozen schedule with the same float expression."""

    rho0 = 0.01
    values = []
    for t in range(n_rounds):
        progress = min(
            1.0,
            max(0.0, (t - RHO_ANNEAL_START_ROUND) / RHO_ANNEAL_ROUNDS),
        )
        values.append(rho0 * (RHO_ANNEAL_END / rho0) ** progress)
    return values


def _audit_schedule(staging: Path) -> dict[str, Any]:
    """Verify every arm followed the frozen schedule bit-for-bit."""

    expected = _expected_schedule(v1.N_ROUNDS)
    boundary_rounds = [
        0,
        RHO_ANNEAL_START_ROUND - 1,
        RHO_ANNEAL_START_ROUND,
        RHO_ANNEAL_START_ROUND + RHO_ANNEAL_ROUNDS - 1,
        RHO_ANNEAL_START_ROUND + RHO_ANNEAL_ROUNDS,
        v1.N_ROUNDS - 1,
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


def _support_evaluation(
    schedule_audit: Mapping[str, Any],
    quality_by_dataset: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the pre-frozen support criteria to the residual arm."""

    per_dataset: dict[str, Any] = {}
    all_pass = True
    for name in v1.DATASETS:
        arm = schedule_audit["arms"][name]["residual"]
        best = arm["best_loss_diagnostic_only"]
        output = arm["output_squared_loss"]
        drift_ratio = output / best if best > 0 else float("inf")
        measured_l1 = float(
            quality_by_dataset[name]["residual"]["measured"][
                "normalized_l1_mean"
            ]
        )
        drift_max = SUPPORT_CRITERIA["drift_ratio_max"][name]
        l1_max = SUPPORT_CRITERIA["measured_normalized_l1_max"][name]
        drift_pass = drift_ratio <= drift_max
        l1_pass = measured_l1 <= l1_max
        all_pass = all_pass and drift_pass and l1_pass
        per_dataset[name] = {
            "best_loss_diagnostic_only": best,
            "output_squared_loss": output,
            "drift_ratio": drift_ratio,
            "drift_ratio_max_allowed": drift_max,
            "drift_ratio_v1": (
                V1_REFERENCE["residual_arm"][name]["drift_ratio"]
            ),
            "drift_criterion_pass": drift_pass,
            "measured_normalized_l1_mean": measured_l1,
            "measured_normalized_l1_max_allowed": l1_max,
            "measured_normalized_l1_v1": (
                V1_REFERENCE["residual_arm"][name][
                    "measured_normalized_l1_mean"
                ]
            ),
            "measured_criterion_pass": l1_pass,
        }
    return {
        "criteria": SUPPORT_CRITERIA,
        "per_dataset_residual_arm": per_dataset,
        "verdict": (
            "schedule_dev_supported" if all_pass
            else "schedule_dev_unsupported"
        ),
        "development_only": True,
        "formal_claim_allowed": False,
    }


def _heldout_observations(
    paired_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Record the no-hard-gate observation items next to v1 references."""

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
        }
    observations["test_300x10"]["v1_reference"] = dict(
        V1_REFERENCE["heldout_paired_delta_test_300x10"]
    )
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

        # Explicit phase boundary: held-out answers and raw reference tables
        # are opened only after every generation trajectory has finished.
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
        support_evaluation = _support_evaluation(
            schedule_audit,
            quality_by_dataset,
        )

        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected_protocol,
            "protocol": _json_protocol_manifest(),
            "source_before_generation": source_before,
            "source_after_generation": source_after_generation,
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
            "quality": quality_by_dataset,
            "paired_quality_comparison": paired_comparison,
            "heldout_observations": _heldout_observations(paired_comparison),
            "support_evaluation": support_evaluation,
            "summary": {
                "development_only": True,
                "formal_claim_allowed": False,
                "only_change_vs_v1": "pre_frozen_three_phase_rho_schedule",
                "all_generation_terminal_current": True,
                "all_generation_fixed_rounds": all(
                    generation_results[name]["arms"][arm]["rounds_run"]
                    == v1.N_ROUNDS
                    for name in v1.DATASETS
                    for arm in v1.ARMS
                ),
                "all_generation_unconditional": True,
                "schedule_audit_passed": True,
                "support_verdict": support_evaluation["verdict"],
                "parameter_retuning_performed": False,
                "reference_loaded_after_generation": True,
            },
            "completed_at_unix": time.time(),
        }
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
    print(f"fitness-only schedule v2 report -> {report_path}")


if __name__ == "__main__":
    main()
