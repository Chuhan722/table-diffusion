#!/usr/bin/env python3
"""Run the fitness-only multiplicative-weights aggregation screen (v6).

This is a *delta executor* over the frozen v3 T=6000 screen
(``scripts/run_fitness_only_schedule_v3_t6000.py``).  Every v3 constant is
kept verbatim (rho three-phase H=900/D=600/floor=0.001, eta constant 0.5,
alpha=16, mu=0.01, relative geometry floor 8, T=6000, seed 9908); the single
change is the fitness *aggregation*::

    query weights : all-ones (equal aggregation) -> multiplicative weights
                    (MW, wrong-answer-notebook mechanism) — at the end of
                    each round t >= 1500 the measured residual updates
                    w via rel=|eps|/mean|eps|, s=min(rel, 8),
                    w <- clip(w*exp(0.002*s)/mean(.), 1/8, 8); the weights
                    enter only the pre-existing weights port of the fitness
                    aggregation (F = sum_j w_j*eps_j*(a_j - p_j))

Lesion evidence (read-only audit of the frozen v3 terminal tables,
2026-09-04): equal aggregation sacrifices rare queries to the total score —
test_300x10 residual arm carries 7 queries with >= 2x debt (equal arm: 2);
nltcs rare-quartile average debt is 2.99x with 21 queries above 8x
(max 29.92x).  The v6 screen tests exactly one pre-frozen hypothesis:
compound-interest attention on persistently indebted queries makes
high-order structure more faithful, improving the *test held-out 3/4-way
paired deltas* — the only indicator where residual currently loses to the
equal control (+0.0033/+0.0010).  Criteria, labels and follow-up actions
were frozen before this run in
``docs/设计/FitnessOnly乘性权重聚合v6结果前协议.md``.

Falsifiable audit: the MW update reads only the current-round measured
residual, is pure deterministic numpy (never consumes the primary RNG), and
w stays trivially all-ones before round 1500 — so the residual arm must
match the v3 residual arm bit-for-bit over the first 1501 rounds
(t in [0, 1500]); the weight statistics must be exactly all-ones over the
first 1500 entries.  Any mismatch invalidates the whole run
(``schedule_blindness_violated``).

Single-arm design: the equal control arm is *not* re-run.  Its fitness is
identically zero, so query weights cannot affect it; the paired held-out
deltas are computed against the hash-frozen v3 equal terminal tables
(protocol §3.4).  This remains a single-development-seed screen: no formal
quality claim, no promotion gate, never overwrites an existing output.
v6 is *not* a retake of v2/v3/v4/v5 — their records stand unchanged.
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

import pandas as pd

from scripts import run_fitness_only_attribution as v1
from scripts import run_fitness_only_schedule_v2 as v2
from scripts import run_fitness_only_schedule_v3_t6000 as v3
from scripts import run_fitness_only_floor00005_v4 as v4
from scripts import run_fitness_only_eta_cooling_v5 as v5
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_fitness_only_evolution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.schema import load_schema


PROTOCOL_VERSION = "fitness-only-mw-v6"
OUTPUT_DIR = Path("outputs/fitness_only_mw_dev_seed9908_v6")
DESIGN_PROTOCOL_DOC = (
    "docs/设计/FitnessOnly乘性权重聚合v6结果前协议.md"
)

N_ROUNDS = 6000
MW_START_ROUND = 1500
MW_ETA = 0.002
MW_SIGNAL_CAP = 8.0
MW_WEIGHT_CAP = 8.0
# t∈[0,1500]（含 t=1500——该轮评价仍用平凡权重，轮末才首次更新）与 v3
# 逐位相同，自 t=1501 起 fitness 聚合分叉；rho/eta 时间表 v3 原样。
PREFIX_ROUNDS = MW_START_ROUND + 1  # 1501

FROZEN_PROTOCOL_SHA256 = (
    "b3cc455ca19b4e3f510e2c2a09e465499bd828a3a85b4c969baca089f0da8717"
)

RESIDUAL_ARM = "residual"
EQUAL_ARM = "equal"

# v3 参照产物（只读）与冻结参照数值：与 v4/v5 同一组，直接复用。
V3_OUTPUT_DIR = v4.V3_OUTPUT_DIR
V3_REFERENCE_ARTIFACT_SHA256 = v4.V3_REFERENCE_ARTIFACT_SHA256
V3_REFERENCE = v4.V3_REFERENCE

# 判据与结果标签（协议 §4 字面数值，生成前冻结）。
VERDICT_CRITERIA = {
    "primary_heldout_paired_delta": {
        "dataset": "test_300x10",
        "delta_definition": "v6_residual_minus_v3_equal_lower_is_better",
        "v3_reference_delta": {
            "heldout_3way_normalized_l1_mean": 0.0033398437500000017,
            "heldout_4way_normalized_l1_mean": 0.00099609375000000045,
        },
        "supported_rule": "both_deltas_strictly_negative",
        "partial_rule": (
            "not_supported_and_at_least_one_delta_improved_ge_50pct_"
            "and_neither_delta_worse_than_v3"
        ),
        "improved_ge_50pct_definition": "delta <= 0.5 * v3_reference_delta",
        "not_worse_definition": "delta <= v3_reference_delta",
    },
    "guard_nltcs_heldout_paired_delta_max": {
        "heldout_3way_normalized_l1_mean": -0.050,
        "heldout_4way_normalized_l1_mean": -0.030,
    },
    "guard_measured_normalized_l1_max": {
        "nltcs": 0.000197,
        "test_300x10": 0.002640,
    },
    "guard_rule": (
        "any_guard_violation_overrides_primary_verdict_to_"
        "quality_regression_under_mw"
    ),
    "verdict_labels": {
        "supported": "mw_supported",
        "partial": "mw_partial_improvement",
        "rejected": "mw_rejected",
        "guard_failed": "quality_regression_under_mw",
        "prefix_audit_failed": "schedule_blindness_violated",
    },
    "rule": (
        "labels_mutually_exclusive; "
        "post_hoc_constant_or_threshold_retuning_forbidden; "
        "no_parameter_sweep_no_multi_seed_cherry_pick; "
        "equal_arm_not_rerun_v3_frozen_artifacts_reused; "
        "v2_v3_v4_v5_records_stand"
    ),
}


def _json_protocol_manifest() -> dict[str, Any]:
    manifest = v3._json_protocol_manifest()
    manifest["contract_version"] = PROTOCOL_VERSION
    manifest["purpose"] = (
        "development_mw_aggregation_screen_only_no_formal_quality_claim"
    )
    manifest["inherits"] = {
        "base_contract_version": v3.PROTOCOL_VERSION,
        "base_protocol_sha256": v3.FROZEN_PROTOCOL_SHA256,
        "v5_contract_version": v5.PROTOCOL_VERSION,
        "v5_protocol_sha256": v5.FROZEN_PROTOCOL_SHA256,
        "v4_contract_version": v4.PROTOCOL_VERSION,
        "v4_protocol_sha256": v4.FROZEN_PROTOCOL_SHA256,
        "v2_contract_version": v2.PROTOCOL_VERSION,
        "v2_protocol_sha256": v2.FROZEN_PROTOCOL_SHA256,
        "v1_contract_version": v1.PROTOCOL_VERSION,
        "v1_protocol_sha256": v1.FROZEN_PROTOCOL_SHA256,
        "only_change": (
            "fitness_aggregation_equal_to_multiplicative_weights_"
            "rho_eta_schedules_and_all_v3_constants_verbatim"
        ),
        "design_protocol_doc": DESIGN_PROTOCOL_DOC,
    }
    manifest["generation_config"]["mw_query_weight_eta"] = MW_ETA
    manifest["generation_config"]["mw_signal_cap"] = MW_SIGNAL_CAP
    manifest["generation_config"]["mw_weight_cap"] = MW_WEIGHT_CAP
    manifest["generation_config"]["mw_start_round"] = MW_START_ROUND
    del manifest["truncation_hypothesis"]
    manifest["mw_aggregation_hypothesis"] = {
        "statement": (
            "equal_aggregation_sacrifices_rare_queries_to_total_score;"
            "v3_terminal_audit_test_residual_7_queries_ge_2x_debt_vs_"
            "equal_2_and_nltcs_rare_quartile_2p99x_with_21_above_8x_"
            "max_29p92x;"
            "compound_interest_attention_on_indebted_queries_improves_"
            "test_heldout_3_4way_paired_deltas"
        ),
        "changes": {
            "fitness_aggregation": {
                "from": "equal_query_weights_all_ones",
                "to": (
                    "multiplicative_weights_start1500_eta0002_"
                    "signal_cap8_weight_cap8"
                ),
            },
        },
        "rho_eta_schedules_v3_verbatim": True,
        "n_rounds_unchanged_same_budget_comparison": True,
        "update_rule": (
            "rel=abs(eps)/mean_abs_eps; s=min(rel, signal_cap); "
            "w<-w*exp(eta_mw*s); w<-w/mean(w); "
            "w<-clip(w, 1/weight_cap, weight_cap); "
            "skip_round_if_mean_abs_eps_is_zero"
        ),
        "state_not_action_no_gating": (
            "weights_score_query_debt_states_never_candidate_moves;"
            "residual_enters_evolution_only_via_fitness_softmax_"
            "donor_selection;kernel_rho_eta_mu_unconditional_update_"
            "untouched"
        ),
        "fail_closed_red_lines": (
            "update_reads_only_current_round_measured_residual;"
            "weights_enter_only_fitness_weights_port;"
            "incompatible_with_residual_directed_diffusion;"
            "equal_arm_never_receives_weights"
        ),
        "academic_lineage": "MWEM_Hardt2012_Hedge_AdaBoost",
        "v2_v3_v4_v5_records_stand": True,
    }
    manifest["single_arm_design"] = {
        "arms_run": [RESIDUAL_ARM],
        "equal_arm_not_rerun_reason": (
            "equal_arm_fitness_identically_zero_weights_cannot_affect_it;"
            "paired_deltas_computed_against_hash_frozen_v3_equal_"
            "terminal_tables"
        ),
        "v3_equal_terminal_table_frame_sha256_source": (
            "v3_report_json_generation_section_hash_pinned_by_"
            "artifact_sha256"
        ),
    }
    manifest["prefix_consistency_audit"] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "prefix_definition": (
            "rho_eta_schedules_v3_verbatim_all_rounds;"
            "weights_trivially_all_ones_before_round_1500_including_"
            "the_t1500_evaluation_first_update_at_end_of_t1500;"
            "fitness_aggregation_diverges_from_t1501"
        ),
        "compared_fields_bit_for_bit": [
            "loss_history[:1501]",
            "rho_schedule_history[:1501]",
            "current_state_metrics_history[:1502]",
            "initial_table_sha256",
            "primary_rng_post_initialization_state_sha256",
        ],
        "weight_stats_all_ones_rounds": MW_START_ROUND,
        "final_rng_state_expected_to_differ": True,
        "failure_label": "schedule_blindness_violated",
        "failure_consequence": (
            "run_invalid_no_quality_interpretation_architecture_must_be_"
            "fixed_first"
        ),
        "v3_reference_output_dir": str(V3_OUTPUT_DIR),
        "v3_reference_artifact_sha256": V3_REFERENCE_ARTIFACT_SHA256,
        "audited_arm": RESIDUAL_ARM,
    }
    manifest["baseline_v3_reference"] = V3_REFERENCE
    manifest["baseline_v4_record"] = {
        "output_dir": str(v4.OUTPUT_DIR),
        "protocol_sha256": v4.FROZEN_PROTOCOL_SHA256,
        "verdict": "quality_regression_under_lower_floor",
        "note": "temperature_route_frequency_leg_closed_record_stands",
    }
    manifest["baseline_v5_record"] = {
        "output_dir": str(v5.OUTPUT_DIR),
        "protocol_sha256": v5.FROZEN_PROTOCOL_SHA256,
        "verdict": "quality_regression_under_eta_cooling",
        "note": (
            "temperature_route_magnitude_leg_closed_v3_config_is_the_"
            "empirical_quality_optimum_record_stands"
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
            "fitness-only v6 mw-aggregation 协议身份漂移："
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
        "trajectory_count": len(v1.DATASETS),
        "arms_run": [RESIDUAL_ARM],
        "generation_started": False,
    }


def _fitness_config(spec: Mapping[str, Any]) -> FitnessOnlyConfig:
    """v3 config plus the MW aggregation constants — the v6 delta."""

    return dataclasses.replace(
        v3._fitness_config(spec),
        mw_query_weight_eta=MW_ETA,
        mw_signal_cap=MW_SIGNAL_CAP,
        mw_weight_cap=MW_WEIGHT_CAP,
        mw_start_round=MW_START_ROUND,
    )


def _source_snapshot(root: Path) -> dict[str, Any]:
    tracked_sources = (
        Path("src/table_diffevo/evolution.py"),
        Path("src/table_diffevo/fitness_only.py"),
        Path("scripts/run_fitness_only_attribution.py"),
        Path("scripts/run_fitness_only_schedule_v2.py"),
        Path("scripts/run_fitness_only_schedule_v3_t6000.py"),
        Path("scripts/run_fitness_only_floor00005_v4.py"),
        Path("scripts/run_fitness_only_eta_cooling_v5.py"),
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


def _run_generation_residual_only(
    root: Path,
    staging: Path,
    audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Run only the residual arm per dataset (equal arm reused from v3)."""

    import numpy as np

    results: dict[str, Any] = {}
    for name, spec in v1.DATASETS.items():
        schema = load_schema(str(root / spec["schema"]))
        marginals = load_marginals(str(root / spec["marginals"]))
        config = _fitness_config(spec)
        started = time.perf_counter()
        table, diagnostics = run_fitness_only_evolution(
            np.asarray(audits[name]["targets"], dtype=float),
            list(audits[name]["queries"]),
            schema,
            int(spec["n_records"]),
            config=config,
            fitness_mode=RESIDUAL_ARM,
            marginals=marginals,
        )
        elapsed = time.perf_counter() - started
        dataset_dir = staging / "generation" / name
        dataset_dir.mkdir(parents=True, exist_ok=False)
        table = table.reset_index(drop=True)
        table_path = dataset_dir / f"{RESIDUAL_ARM}_terminal_current.csv"
        table.to_csv(table_path, index=False)
        v1._write_json(
            dataset_dir / f"{RESIDUAL_ARM}_diagnostics.json",
            diagnostics,
        )
        arm_result = {
            "terminal_table_sha256": v1._frame_sha256(table),
            "rounds_run": int(diagnostics["rounds_run"]),
            "candidate_evaluation_count": int(
                diagnostics["candidate_evaluation_count"]
            ),
            "termination_reason": diagnostics["termination_reason"],
            "output_table_identity": diagnostics["output_table_identity"],
            "final_current_squared_loss": float(
                diagnostics["final_current_squared_loss"]
            ),
            "final_current_normalized_l1": float(
                diagnostics["final_current_normalized_l1"]
            ),
            "best_loss_diagnostic_only": float(
                diagnostics["best_loss_diagnostic_only"]
            ),
            "elapsed_sec": float(diagnostics["elapsed_sec"]),
        }
        results[name] = {
            "dataset": name,
            "seed": v1.SEED,
            "n_records": int(spec["n_records"]),
            "query_count": int(audits[name]["query_count"]),
            "device": spec["device"],
            "elapsed_sec_wall": float(elapsed),
            "arms_run": [RESIDUAL_ARM],
            "equal_arm_reused_from": str(V3_OUTPUT_DIR),
            "arms": {RESIDUAL_ARM: arm_result},
        }
        print(
            f"[{name}] residual-arm generation complete in {elapsed:.1f}s",
            flush=True,
        )
    return results


def _expected_eta_constant_schedule(n_rounds: int) -> list[float]:
    """v3-verbatim constant eta schedule (no anneal in v6)."""

    return [0.5] * n_rounds


def _audit_schedule(staging: Path) -> dict[str, Any]:
    """Verify the residual arm followed the frozen v3-verbatim schedules
    and the MW weight contract (all-ones before round 1500, caps after)."""

    expected_rho = v5._expected_rho_schedule(N_ROUNDS)
    expected_eta = _expected_eta_constant_schedule(N_ROUNDS)
    rho_boundary_rounds = [
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
        "rho_boundary_rounds": rho_boundary_rounds,
        "expected_rho_boundary_values": [
            expected_rho[t] for t in rho_boundary_rounds
        ],
        "eta_constant_expected": 0.5,
        "arms": {},
    }
    for name in v1.DATASETS:
        diagnostics = v1._load_json_object(
            staging
            / "generation"
            / name
            / f"{RESIDUAL_ARM}_diagnostics.json"
        )
        rho_history = diagnostics.get("rho_schedule_history")
        if rho_history != expected_rho:
            raise RuntimeError(
                f"{name}/{RESIDUAL_ARM} rho_schedule_history 偏离 v3 "
                "原样时间表"
            )
        eta_history = diagnostics.get("eta_schedule_history")
        if eta_history != expected_eta:
            raise RuntimeError(
                f"{name}/{RESIDUAL_ARM} eta_schedule_history 偏离 v3 "
                "恒 0.5 时间表"
            )
        stats = diagnostics.get("mw_weight_stats_history")
        if not isinstance(stats, list) or len(stats) != N_ROUNDS:
            raise RuntimeError(
                f"{name}/{RESIDUAL_ARM} mw_weight_stats_history 长度异常"
            )
        for row in stats[:MW_START_ROUND]:
            if (
                row["max"] != 1.0
                or row["min"] != 1.0
                or row["mean"] != 1.0
            ):
                raise RuntimeError(
                    f"{name}/{RESIDUAL_ARM} MW 权重在 t={row['round']} "
                    "(< mw_start_round) 已非平凡——保温段合同被破坏"
                )
        eps = 1e-12
        for row in stats[MW_START_ROUND:]:
            if (
                row["max"] > MW_WEIGHT_CAP + eps
                or row["min"] < 1.0 / MW_WEIGHT_CAP - eps
            ):
                raise RuntimeError(
                    f"{name}/{RESIDUAL_ARM} MW 权重在 t={row['round']} "
                    "越出 [1/cap, cap] 围栏"
                )
        snapshots = diagnostics.get("mw_weight_snapshot_history")
        if not isinstance(snapshots, list) or not snapshots:
            raise RuntimeError(
                f"{name}/{RESIDUAL_ARM} mw_weight_snapshot_history 缺失"
            )
        terminal_stats = stats[-1]
        audit["arms"][name] = {
            RESIDUAL_ARM: {
                "rho_schedule_matches_v3_verbatim_formula": True,
                "eta_schedule_matches_v3_constant": True,
                "rounds": len(rho_history),
                "mw_weight_all_ones_before_start_round": True,
                "mw_weight_within_caps_after_start_round": True,
                "mw_weight_terminal_stats": dict(terminal_stats),
                "mw_weight_snapshot_count": len(snapshots),
                "output_squared_loss": float(
                    diagnostics["output_squared_loss"]
                ),
                "best_loss_diagnostic_only": float(
                    diagnostics["best_loss_diagnostic_only"]
                ),
            },
        }
    return audit


def _prefix_audit(
    v6_diagnostics: Mapping[str, Mapping[str, Any]],
    v3_diagnostics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bit-for-bit 1501-round residual-arm prefix comparison against v3."""

    state_prefix_len = PREFIX_ROUNDS + 1  # initial 条目 + 前 1501 轮
    audit: dict[str, Any] = {
        "prefix_rounds": PREFIX_ROUNDS,
        "current_state_metrics_prefix_entries": state_prefix_len,
        "audited_arm": RESIDUAL_ARM,
        "arms": {},
    }
    passed = True
    for name in v1.DATASETS:
        new = v6_diagnostics[name][RESIDUAL_ARM]
        old = v3_diagnostics[name][RESIDUAL_ARM]
        loss_prefix = list(new["loss_history"][:PREFIX_ROUNDS])
        rho_prefix = list(new["rho_schedule_history"][:PREFIX_ROUNDS])
        state_prefix = list(
            new["current_state_metrics_history"][:state_prefix_len]
        )
        old_loss = list(old["loss_history"][:PREFIX_ROUNDS])
        old_rho = list(old["rho_schedule_history"][:PREFIX_ROUNDS])
        old_state = list(
            old["current_state_metrics_history"][:state_prefix_len]
        )
        weight_stats = new["mw_weight_stats_history"]
        weights_all_ones = all(
            row["max"] == 1.0 and row["min"] == 1.0 and row["mean"] == 1.0
            for row in weight_stats[:MW_START_ROUND]
        )
        checks = {
            "loss_history_prefix_identical": loss_prefix == old_loss,
            "rho_schedule_history_prefix_identical": (
                rho_prefix == old_rho
            ),
            "current_state_metrics_history_prefix_identical": (
                state_prefix == old_state
            ),
            "initial_table_sha256_identical": (
                new["initial_table_sha256"] == old["initial_table_sha256"]
            ),
            "primary_rng_post_initialization_state_identical": (
                new["primary_rng_post_initialization_state_sha256"]
                == old["primary_rng_post_initialization_state_sha256"]
            ),
            "mw_weights_all_ones_before_start_round": weights_all_ones,
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
        if not checks["current_state_metrics_history_prefix_identical"]:
            detail["current_state_metrics_first_mismatch_entry"] = (
                v3._first_mismatch_index(state_prefix, old_state)
            )
        arm_passed = all(checks.values())
        detail["passed"] = arm_passed
        passed = passed and arm_passed
        audit["arms"][name] = {RESIDUAL_ARM: detail}
    audit["passed"] = passed
    audit["final_rng_state_expected_to_differ"] = True
    audit["failure_label"] = (
        None if passed
        else VERDICT_CRITERIA["verdict_labels"]["prefix_audit_failed"]
    )
    return audit


def _load_v3_report(root: Path) -> dict[str, Any]:
    """Load the hash-pinned v3 report (verified by artifact SHA audit)."""

    return v1._load_json_object(root / V3_OUTPUT_DIR / "report.json")


def _stage_v3_equal_tables(
    root: Path,
    staging: Path,
    v3_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy the frozen v3 equal terminal tables into staging for evaluation.

    The frame SHA of each copied table must match the value recorded in the
    hash-pinned v3 report — this closes the identity chain from
    V3_REFERENCE_ARTIFACT_SHA256 (report.json file hash) to the actual
    tables evaluated here.
    """

    staged: dict[str, Any] = {}
    for name in v1.DATASETS:
        source = (
            root
            / V3_OUTPUT_DIR
            / "generation"
            / name
            / f"{EQUAL_ARM}_terminal_current.csv"
        )
        if not source.is_file():
            raise RuntimeError(f"v3 equal 终态表缺失：{source}")
        expected_sha = v3_report["generation"][name]["arms"][EQUAL_ARM][
            "terminal_table_sha256"
        ]
        table = pd.read_csv(source)
        observed_sha = v1._frame_sha256(table)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"{name} v3 equal 终态表 frame SHA 漂移："
                f"expected={expected_sha}, observed={observed_sha}"
            )
        destination = (
            staging / "generation" / name / f"{EQUAL_ARM}_terminal_current.csv"
        )
        shutil.copyfile(source, destination)
        staged[name] = {
            "source": str(source.relative_to(root)),
            "terminal_table_sha256": observed_sha,
            "reused_not_rerun": True,
        }
    return staged


def _verdict_evaluation(
    schedule_audit: Mapping[str, Any],
    quality_by_dataset: Mapping[str, Mapping[str, Any]],
    paired_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-frozen v6 criteria and mutually exclusive labels."""

    labels = VERDICT_CRITERIA["verdict_labels"]

    per_dataset: dict[str, Any] = {}
    for name in v1.DATASETS:
        arm = schedule_audit["arms"][name][RESIDUAL_ARM]
        best = arm["best_loss_diagnostic_only"]
        output = arm["output_squared_loss"]
        measured_l1 = float(
            quality_by_dataset[name][RESIDUAL_ARM]["measured"][
                "normalized_l1_mean"
            ]
        )
        reference = V3_REFERENCE["residual_arm"][name]
        per_dataset[name] = {
            "best_loss_diagnostic_only": best,
            "output_squared_loss": output,
            "drift_ratio_observation_only": (
                output / best if best > 0 else float("inf")
            ),
            "drift_ratio_v3": reference["drift_ratio"],
            "measured_normalized_l1_mean": measured_l1,
            "measured_normalized_l1_v3": (
                reference["measured_normalized_l1_mean"]
            ),
        }

    # 护栏 1：nltcs held-out 配对差保住 v3 优势（含余量）。
    guard_nltcs_max = VERDICT_CRITERIA["guard_nltcs_heldout_paired_delta_max"]
    nltcs_delta = paired_comparison["nltcs"]["delta_residual_minus_equal"]
    guard_nltcs = {}
    guard_nltcs_pass = True
    for key, max_allowed in guard_nltcs_max.items():
        observed = float(nltcs_delta[key])
        item_pass = observed <= max_allowed
        guard_nltcs[key] = {
            "observed_delta": observed,
            "max_allowed": max_allowed,
            "pass": item_pass,
        }
        guard_nltcs_pass = guard_nltcs_pass and item_pass

    # 护栏 2：measured L1 双数据集 1.10×v3 阈值（与 v4/v5 同阈值）。
    l1_max = VERDICT_CRITERIA["guard_measured_normalized_l1_max"]
    guard_l1 = {}
    guard_l1_pass = True
    for name in v1.DATASETS:
        observed = per_dataset[name]["measured_normalized_l1_mean"]
        item_pass = observed <= l1_max[name]
        guard_l1[name] = {
            "measured_normalized_l1_mean": observed,
            "max_allowed": l1_max[name],
            "pass": item_pass,
        }
        guard_l1_pass = guard_l1_pass and item_pass

    guards_pass = guard_nltcs_pass and guard_l1_pass

    # 主判据：test held-out 配对差（v6 residual − v3 equal）。
    primary_spec = VERDICT_CRITERIA["primary_heldout_paired_delta"]
    reference_delta = primary_spec["v3_reference_delta"]
    test_delta = paired_comparison["test_300x10"][
        "delta_residual_minus_equal"
    ]
    primary_detail = {}
    both_negative = True
    any_improved_50 = False
    none_worse = True
    for key, v3_value in reference_delta.items():
        observed = float(test_delta[key])
        negative = observed < 0.0
        improved_50 = observed <= 0.5 * v3_value
        not_worse = observed <= v3_value
        primary_detail[key] = {
            "observed_delta": observed,
            "v3_reference_delta": v3_value,
            "strictly_negative": negative,
            "improved_ge_50pct": improved_50,
            "not_worse_than_v3": not_worse,
        }
        both_negative = both_negative and negative
        any_improved_50 = any_improved_50 or improved_50
        none_worse = none_worse and not_worse

    if not guards_pass:
        verdict = labels["guard_failed"]
    elif both_negative:
        verdict = labels["supported"]
    elif any_improved_50 and none_worse:
        verdict = labels["partial"]
    else:
        verdict = labels["rejected"]

    return {
        "criteria": VERDICT_CRITERIA,
        "per_dataset_residual_arm": per_dataset,
        "guard_nltcs_heldout_paired_delta": {
            "per_metric": guard_nltcs,
            "pass": guard_nltcs_pass,
        },
        "guard_measured_normalized_l1": {
            "per_dataset": guard_l1,
            "pass": guard_l1_pass,
        },
        "guards_pass": guards_pass,
        "primary_test_heldout_paired_delta": {
            "per_metric": primary_detail,
            "both_strictly_negative": both_negative,
            "any_improved_ge_50pct": any_improved_50,
            "none_worse_than_v3": none_worse,
        },
        "primary_verdict": verdict,
        "development_only": True,
        "formal_claim_allowed": False,
    }


def _heldout_observations(
    paired_comparison: Mapping[str, Any],
) -> dict[str, Any]:
    """Record paired deltas next to the frozen v3 references."""

    observations: dict[str, Any] = {
        "delta_definition": "v6_residual_minus_v3_equal_lower_is_better",
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
        generation_results = _run_generation_residual_only(
            root,
            staging,
            generation_audits,
        )
        source_after_generation = _source_snapshot(root)
        v1._audit_source_unchanged(source_before, source_after_generation)
        schedule_audit = _audit_schedule(staging)

        v6_diagnostics: dict[str, dict[str, Any]] = {}
        for name in v1.DATASETS:
            v6_diagnostics[name] = {
                RESIDUAL_ARM: v1._load_json_object(
                    staging
                    / "generation"
                    / name
                    / f"{RESIDUAL_ARM}_diagnostics.json"
                ),
            }
        v3_diagnostics = v4._load_v3_reference_diagnostics(root)
        prefix_audit = _prefix_audit(v6_diagnostics, v3_diagnostics)

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
                "fitness_aggregation_equal_to_multiplicative_weights_"
                "rho_eta_schedules_and_all_v3_constants_verbatim"
            ),
            "arms_run": [RESIDUAL_ARM],
            "equal_arm_reused_from_v3": True,
            "all_generation_terminal_current": True,
            "all_generation_fixed_rounds": all(
                generation_results[name]["arms"][RESIDUAL_ARM]["rounds_run"]
                == N_ROUNDS
                for name in v1.DATASETS
            ),
            "all_generation_unconditional": True,
            "schedule_audit_passed": True,
            "parameter_retuning_performed": False,
            "v2_v3_v4_v5_records_stand": True,
        }

        if not prefix_audit["passed"]:
            # 时间表盲性被证伪：运行无效。保留生成产物供架构诊断，但不
            # 打开 held-out 答案与参考表（不产生任何可解读的质量结果）。
            report["v3_equal_tables_staged"] = None
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

        # Explicit phase boundary: held-out answers, raw reference tables
        # and the v3 report (equal-arm identity source) are opened only
        # after every generation trajectory has finished and the prefix
        # audit has passed.
        v3_report = _load_v3_report(root)
        staged_equal = _stage_v3_equal_tables(root, staging, v3_report)
        quality_by_dataset: dict[str, dict[str, Any]] = {}
        for name in v1.DATASETS:
            quality_by_dataset[name] = {
                RESIDUAL_ARM: v1._evaluate_one(
                    root,
                    name,
                    RESIDUAL_ARM,
                    generation_audits[name],
                    generation_results[name]["arms"][RESIDUAL_ARM],
                    staging,
                ),
                EQUAL_ARM: v1._evaluate_one(
                    root,
                    name,
                    EQUAL_ARM,
                    generation_audits[name],
                    v3_report["generation"][name]["arms"][EQUAL_ARM],
                    staging,
                ),
            }
        paired_comparison = v1._paired_quality_comparison(quality_by_dataset)
        support_evaluation = _verdict_evaluation(
            schedule_audit,
            quality_by_dataset,
            paired_comparison,
        )

        report["v3_equal_tables_staged"] = staged_equal
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
    print(f"fitness-only mw-aggregation v6 report -> {report_path}")
    print(f"primary_verdict = {verdict}")


if __name__ == "__main__":
    main()
