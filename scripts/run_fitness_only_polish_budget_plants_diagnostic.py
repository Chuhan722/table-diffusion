#!/usr/bin/env python3
"""Polish-density budget diagnostic on plants (kappa=1, single residual arm).

The one-way-in-pool diagnostic healed the plants one-way collapse but left a
truncated optimisation: at round 6000 the residual arm loss was still falling
by ~19% per 500 rounds.  Root cause: the v3 budget copied the nltcs absolute
round count, giving plants a per-cell polish density of only kappa~=0.23
(sum of rho_t / n_attributes), while the nltcs frozen schedule realises
kappa=0.991~=1.  This protocol sets the budget by the polish-density
principle: M = the attribute count (69) is the per-row polish target; the
minimal T whose relative schedule (anneal start at 15% T, anneal length
10% T, rho 0.01 -> 0.001 geometric) accumulates sum(rho_t) >= M solves to
T = 26126; the budget cap is that solve rounded up to the nearest thousand,
C = 27000 (anneal 4050 + 2700), realising kappa=1.033 at the cap.

Termination follows the amended fitness-only contract: gate-free inner
A/B/C early stopping (patience 6 natural work ticks, engine default, no
tuning) under the fixed budget cap.  The equal blind arm is not rerun: the
one-way-in-pool reports on both plants and nltcs proved byte-identical equal
trajectories (equal_table_identity_match=true), so its reference numbers are
reused from the pinned one-way-in-pool report.  Diagnostic only: single
development seed, no gate, no formal claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from scripts import run_fitness_only_attribution as base
from scripts import (
    run_fitness_only_oneway_pool_plants_diagnostic as oneway,
)
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_fitness_only_evolution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.schema import load_schema


PROTOCOL_VERSION = "fitness-only-polish-budget-plants-diagnostic-v1"
OUTPUT_DIR = Path("outputs/fitness_only_polish_budget_plants_seed9908_v1")
FROZEN_PROTOCOL_SHA256 = (
    "4b7c97105ab7f19824defb0b2cc7213b5f6c2dcb38d577befe91f3354a63874e"
)
SEED = 9908
DATASET = "plants"
N_RECORDS = 17412
N_ATTRIBUTES = 69

# 打磨密度定标：M = 属性数（列数）为每行打磨次数目标；
# 解出满足 sum(rho_t) >= M 的最小轮数，再向上取整到千作为预算上限（保险丝）。
POLISH_TARGET_PER_ROW = float(N_ATTRIBUTES)
RHO_BASE = 0.01
RHO_ANNEAL_END = 0.001
ANNEAL_START_FRACTION = 0.15
ANNEAL_LENGTH_FRACTION = 0.10
SOLVED_MINIMAL_ROUNDS = 26126
CAP_ROUNDING_UNIT = 1000
N_ROUNDS = 27000
RHO_ANNEAL_START_ROUND = 4050
RHO_ANNEAL_ROUNDS = 2700

EARLY_STOPPING_PATIENCE_TICKS = 6  # 引擎默认值，零调参。

ARM = "residual"

FLOOR_SEGMENT_ROUNDS = 500

REFERENCE_SHA256 = {
    "oneway_pool_report": (
        "30fe3cd233bb72b51d8218769bdcb651f4376b32ebb122e14fb99c64133a72ec"
    ),
    "frozen_report": (
        "26a8934270e9042d05e90d0a77ba1859aebd34a48ca1082291dbee9330d95565"
    ),
    "pgm_report": (
        "fbe0c88d793fde6ed3e6b919933b7bedf2d239cb6ea606f3865eadad9f026105"
    ),
}
ONEWAY_POOL_REPORT_PATH = Path(
    "outputs/fitness_only_oneway_pool_plants_seed9908_v1/report.json"
)
FROZEN_REPORT_PATH = Path(
    "outputs/fitness_only_plants_diagnostic_seed9908_v1/report.json"
)
PGM_REPORT_PATH = Path("outputs/baseline_pgm_plants_v1/report.json")

SCHEMA_PATH = oneway.SCHEMA_PATH
MARGINALS_PATH = oneway.MARGINALS_PATH


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} 顶层必须是对象")
    return payload


def _relative_schedule(n_rounds: int) -> tuple[int, int]:
    start = round(ANNEAL_START_FRACTION * n_rounds)
    rounds = max(1, round(ANNEAL_LENGTH_FRACTION * n_rounds))
    return start, rounds


def _expected_rho_schedule(
    n_rounds: int,
    anneal_start: int,
    anneal_rounds: int,
) -> list[float]:
    values = []
    for t in range(n_rounds):
        progress = min(
            1.0,
            max(0.0, (t - anneal_start) / anneal_rounds),
        )
        values.append(RHO_BASE * (RHO_ANNEAL_END / RHO_BASE) ** progress)
    return values


def _polish_sum(n_rounds: int) -> float:
    start, rounds = _relative_schedule(n_rounds)
    return float(
        sum(_expected_rho_schedule(n_rounds, start, rounds))
    )


def assert_polish_budget_identity() -> dict[str, Any]:
    """预算三重自检：解的最小性、取整规则、上限打磨量，常数漂移即失败。"""

    start, rounds = _relative_schedule(N_ROUNDS)
    if (start, rounds) != (RHO_ANNEAL_START_ROUND, RHO_ANNEAL_ROUNDS):
        raise RuntimeError(
            "相对时刻表常数漂移："
            f"expected=({RHO_ANNEAL_START_ROUND}, {RHO_ANNEAL_ROUNDS}), "
            f"observed=({start}, {rounds})"
        )
    at_solved = _polish_sum(SOLVED_MINIMAL_ROUNDS)
    below_solved = _polish_sum(SOLVED_MINIMAL_ROUNDS - 1)
    if not at_solved >= POLISH_TARGET_PER_ROW:
        raise RuntimeError(
            f"解不足：polish({SOLVED_MINIMAL_ROUNDS})={at_solved} < "
            f"{POLISH_TARGET_PER_ROW}"
        )
    if not below_solved < POLISH_TARGET_PER_ROW:
        raise RuntimeError(
            f"解非最小：polish({SOLVED_MINIMAL_ROUNDS - 1})="
            f"{below_solved} >= {POLISH_TARGET_PER_ROW}"
        )
    expected_cap = (
        math.ceil(SOLVED_MINIMAL_ROUNDS / CAP_ROUNDING_UNIT)
        * CAP_ROUNDING_UNIT
    )
    if N_ROUNDS != expected_cap:
        raise RuntimeError(
            f"上限取整规则漂移：expected={expected_cap}, observed={N_ROUNDS}"
        )
    at_cap = _polish_sum(N_ROUNDS)
    if not at_cap >= POLISH_TARGET_PER_ROW:
        raise RuntimeError(
            f"上限打磨量不足：polish({N_ROUNDS})={at_cap} < "
            f"{POLISH_TARGET_PER_ROW}"
        )
    return {
        "polish_target_per_row": POLISH_TARGET_PER_ROW,
        "solved_minimal_rounds": SOLVED_MINIMAL_ROUNDS,
        "polish_at_solved": at_solved,
        "polish_below_solved": below_solved,
        "cap_rounding_unit": CAP_ROUNDING_UNIT,
        "budget_cap_rounds": N_ROUNDS,
        "polish_at_cap": at_cap,
        "kappa_at_cap": at_cap / POLISH_TARGET_PER_ROW,
        "minimality_and_rounding": "pass",
    }


def _protocol_manifest() -> dict[str, Any]:
    return {
        "contract_version": PROTOCOL_VERSION,
        "purpose": "polish_density_budget_hypothesis_diagnostic_only",
        "hypothesis": (
            "the plants budget was copied from the nltcs absolute round"
            " count, leaving per-cell polish density kappa~=0.23 while the"
            " nltcs frozen schedule realises kappa=0.991~=1; raising the"
            " budget cap to the polish-density solve (minimal T with"
            " sum(rho_t) >= M = 69) rounded up to the nearest thousand,"
            " under the same relative schedule, lifts the observed loss"
            " truncation and substantially closes the measured and"
            " held-out gaps to the PGM baseline"
        ),
        "seed": SEED,
        "dataset": DATASET,
        "n_records": N_RECORDS,
        "n_attributes": N_ATTRIBUTES,
        "arms": [ARM],
        "polish_density_principle": {
            "definition": (
                "kappa = sum_t(rho_t) / M where M is the attribute"
                " (column) count; expected number of modifications per row"
                " equals sum_t(rho_t), so kappa=1 means each row is"
                " polished once per attribute on average"
            ),
            "target_kappa": 1.0,
            "polish_target_per_row": POLISH_TARGET_PER_ROW,
            "budget_rule": (
                "solve the minimal T such that the relative schedule"
                " accumulates sum(rho_t) >= M, then round the cap up to"
                " the nearest thousand rounds; the relative schedule is"
                " anchored on the rounded cap; verified by"
                " assert_polish_budget_identity"
            ),
            "solved_minimal_rounds": SOLVED_MINIMAL_ROUNDS,
            "cap_rounding_unit": CAP_ROUNDING_UNIT,
            "relative_schedule": {
                "anneal_start_fraction": ANNEAL_START_FRACTION,
                "anneal_length_fraction": ANNEAL_LENGTH_FRACTION,
                "source": (
                    "nltcs frozen v3 anneal points 900/600 at T=6000"
                    " restated as fractions"
                ),
            },
            "historical_calibration": (
                "nltcs frozen v3 schedule realises sum(rho_t)=15.85 over"
                " d=16 attributes, kappa=0.991; the same principle"
                " back-solves nltcs T=6057 vs frozen 6000 (0.9% deviation)"
            ),
        },
        "budget": {
            "n_rounds": N_ROUNDS,
            "rho": RHO_BASE,
            "rho_anneal_start_round": RHO_ANNEAL_START_ROUND,
            "rho_anneal_rounds": RHO_ANNEAL_ROUNDS,
            "rho_anneal_end": RHO_ANNEAL_END,
        },
        "early_stopping": {
            "mechanism": "inner_early_stopping_a_b_c",
            "patience_ticks": EARLY_STOPPING_PATIENCE_TICKS,
            "patience_source": "engine DEFAULT_PATIENCE_TICKS, no tuning",
            "stop_on_exact_residual": True,
            "semantics": (
                "observe-only terminal-current stopping: no proposal"
                " control, no state rejection, no historical-best output;"
                " one work tick = cumulative participating rows reaching"
                " n_records (each row modified once on average); stop"
                " after 6 consecutive ticks without a strict best-loss"
                " improvement; the fixed budget above remains the hard cap"
            ),
            "contract_amendment": (
                "fitness-only contract amended to allow the paired flags"
                " stop_on_exact_residual + inner_early_stopping as a unit;"
                " audit accepts early_stopped/fit_target_reached with"
                " rounds_run <= budget and candidate count == rounds_run"
            ),
        },
        "equal_arm_cut": {
            "rerun": False,
            "justification": (
                "equal_table_identity_match=true in both the plants and"
                " nltcs one-way-in-pool reports: the blind arm trajectory"
                " is independent of the query pool and budget treatment;"
                " its reference numbers are reused from the pinned"
                " one-way-in-pool plants report"
            ),
        },
        "generation_input": {
            "schema": str(SCHEMA_PATH),
            "marginals": str(MARGINALS_PATH),
            "measured": str(oneway.MEASURED_PATH),
            "expected_measured_count": oneway.EXPECTED_QUERY_COUNT,
            "expected_one_way_count": oneway.EXPECTED_ONE_WAY_COUNT,
            "expected_pool_count": oneway.EXPECTED_POOL_COUNT,
            "pool_construction": (
                "identical to the one-way-in-pool protocol: original"
                " measured workload plus every one-way cell as an ordinary"
                " query"
            ),
            "input_sha256": {
                key: oneway.INPUT_SHA256[key]
                for key in ("schema", "marginals", "measured")
            },
        },
        "shared_generation_config": {
            "configuration_source": (
                "v3 frozen configuration verbatim except the predeclared"
                " budget and termination amendments listed here"
            ),
            "changed_relative_to_v3": [
                "n_rounds",
                "rho_anneal_start_round",
                "rho_anneal_rounds",
                "inner_early_stopping_patience_ticks",
                "stop_on_exact_residual",
            ],
            "unchanged": {
                "init_method": "marginal",
                "eval_method": "vectorized",
                "device": "cuda",
                "rho": RHO_BASE,
                "rho_anneal_end": RHO_ANNEAL_END,
                "eta": 0.5,
                "mu": 0.01,
                "distance_mode": "geometric",
                "selection_scale_invariant": True,
                "selection_scale_invariant_min_spread": 1e-3,
                "alpha_schedule_mode": "fixed",
                "fixed_alpha": 16.0,
                "lambda_param": 0.5,
                "delta": 0.05,
                "winsorize_quantiles": [0.01, 0.99],
                "residual_geometry": "relative",
                "residual_geometry_floor": 8.0,
                "exclude_self": True,
                "eta_anneal": None,
                "mw_query_weights": None,
            },
        },
        "evaluation_grouping": {
            "measured_metric_uses_only_original_measured_queries": True,
            "one_way_safety_same_construction_as_frozen_reports": True,
            "heldout_frozen_untouched": True,
            "one_way_now_in_evolution_pool": True,
        },
        "comparison_reference": {
            "oneway_pool_report": str(ONEWAY_POOL_REPORT_PATH),
            "oneway_pool_report_sha256": (
                REFERENCE_SHA256["oneway_pool_report"]
            ),
            "frozen_report": str(FROZEN_REPORT_PATH),
            "frozen_report_sha256": REFERENCE_SHA256["frozen_report"],
            "pgm_report": str(PGM_REPORT_PATH),
            "pgm_report_sha256": REFERENCE_SHA256["pgm_report"],
            "primary_contrast": (
                "new residual (kappa=1 budget) minus one-way-in-pool"
                " residual (truncated budget): isolates the budget effect"
                " with pool, seed and schedule shape held fixed"
            ),
        },
        "observation_points": {
            "O1_truncation_lifted": (
                "either early stopping fires before the cap (loss plateau"
                " reached within budget) or the cap is hit while the loss"
                " still descends; report realized rounds, realized polish"
                " sum, and terminal loss segment slope"
            ),
            "O2_measured_gap": (
                "measured mean vs PGM: one-way-in-pool residual was"
                " 0.003634 vs PGM 0.000779 (4.7x); record the new ratio"
            ),
            "O3_heldout_gap": (
                "heldout 3/4-way mean vs PGM: was 0.031725/0.023505 vs"
                " 0.015229/0.012866 (~2x); record the new ratios"
            ),
            "O4_one_way_held": (
                "one-way safety stays at or below the one-way-in-pool"
                " level 0.004230 (healing must not regress)"
            ),
            "O5_no_crowding": (
                "measured mean not worse than the one-way-in-pool residual"
                " 0.003634 (long budget must not crowd out measured)"
            ),
        },
        "phase_boundary": {
            "heldout_and_reference_loaded_after_generation": True,
            "quality_used_online": False,
            "parameter_retuning_allowed": False,
            "output_overwrite_allowed": False,
        },
        "interpretation": {
            "diagnostic_only": True,
            "formal_claim_allowed": False,
            "question": (
                "with the polish-density budget kappa=1 on plants, does"
                " the loss truncation lift, and how much of the measured"
                " and held-out gap to the PGM baseline closes without any"
                " other mechanism change?"
            ),
        },
    }


def protocol_sha256() -> str:
    return _sha256_bytes(_strict_json_bytes(_protocol_manifest()))


def assert_frozen_protocol_identity() -> str:
    observed = protocol_sha256()
    if observed != FROZEN_PROTOCOL_SHA256:
        raise RuntimeError(
            "polish-budget plants diagnostic 协议身份漂移："
            f"expected={FROZEN_PROTOCOL_SHA256}, observed={observed}"
        )
    return observed


def build_plan() -> dict[str, Any]:
    return {
        "mode": "plan_only_no_input_or_result_read_no_generation",
        "protocol_sha256": assert_frozen_protocol_identity(),
        "polish_budget_identity": assert_polish_budget_identity(),
        "protocol": _protocol_manifest(),
        "output_dir": str(OUTPUT_DIR),
        "trajectory_count": 1,
        "generation_started": False,
    }


def _config() -> FitnessOnlyConfig:
    return FitnessOnlyConfig(
        n_rounds=N_ROUNDS,
        seed=SEED,
        device="cuda",
        eval_method="vectorized",
        batch_size=256,
        init_method="marginal",
        log_every=100,
        rho=RHO_BASE,
        eta=0.5,
        mu=0.01,
        lambda_param=0.5,
        fixed_alpha=16.0,
        delta=0.05,
        winsorize_quantiles=(0.01, 0.99),
        selection_scale_invariant_min_spread=1e-3,
        residual_geometry="relative",
        residual_geometry_floor=8.0,
        exclude_self=True,
        record_transition_clocks=False,
        rho_anneal_start_round=RHO_ANNEAL_START_ROUND,
        rho_anneal_rounds=RHO_ANNEAL_ROUNDS,
        rho_anneal_end=RHO_ANNEAL_END,
        inner_early_stopping_patience_ticks=(
            EARLY_STOPPING_PATIENCE_TICKS
        ),
    )


def _audit_rho_schedule(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    observed = [float(x) for x in diagnostics["rho_schedule_history"]]
    rounds_run = int(diagnostics["rounds_run"])
    if len(observed) != rounds_run:
        raise RuntimeError("rho schedule 长度不等于实际轮数")
    expected_full = _expected_rho_schedule(
        N_ROUNDS,
        RHO_ANNEAL_START_ROUND,
        RHO_ANNEAL_ROUNDS,
    )
    expected = expected_full[:rounds_run]
    max_abs = max(
        abs(a - b) for a, b in zip(observed, expected)
    )
    if max_abs > 1e-12:
        raise RuntimeError(f"rho schedule 数值漂移: max_abs={max_abs}")
    return {
        "length": len(observed),
        "budget_length": N_ROUNDS,
        "max_abs_deviation": max_abs,
        "realized_polish_sum": float(sum(observed)),
        "budget_polish_sum": float(sum(expected_full)),
        "verdict": "pass",
    }


def _floor_morphology(loss_history: Sequence[float]) -> dict[str, Any]:
    floor_start = RHO_ANNEAL_START_ROUND + RHO_ANNEAL_ROUNDS
    floor = [float(x) for x in loss_history[floor_start:]]
    segments = []
    for begin in range(0, len(floor), FLOOR_SEGMENT_ROUNDS):
        chunk = floor[begin:begin + FLOOR_SEGMENT_ROUNDS]
        if chunk:
            segments.append(float(np.mean(chunk)))
    best_index = int(np.argmin(loss_history))
    return {
        "floor_start_round": floor_start,
        "segment_rounds": FLOOR_SEGMENT_ROUNDS,
        "segment_means": segments,
        "best_round_index": best_index,
        "last_round_index": len(loss_history) - 1,
    }


def _early_stopping_summary(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    inner = diagnostics["inner_early_stopping"]
    if not inner.get("enabled"):
        raise RuntimeError("inner early stopping 诊断缺失")
    decision = inner.get("last_decision") or {}
    return {
        "enabled": True,
        "patience_ticks": int(inner["patience_ticks"]),
        "termination_reason": diagnostics["termination_reason"],
        "stopped_early": bool(diagnostics["stopped_early"]),
        "completed_work_ticks": decision.get("completed_work_ticks"),
        "consecutive_no_progress_ticks": decision.get(
            "consecutive_no_progress_ticks"
        ),
        "best_state_index_diagnostic_only": decision.get(
            "best_state_index_diagnostic_only"
        ),
        "normalized_work": decision.get("normalized_work"),
    }


def _run_generation(
    root: Path,
    staging: Path,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    schema = load_schema(str(root / SCHEMA_PATH))
    marginals = load_marginals(str(root / MARGINALS_PATH))
    started = time.perf_counter()
    table, diagnostics = run_fitness_only_evolution(
        np.asarray(audit["pool_targets"], dtype=float),
        list(audit["pool_queries"]),
        schema,
        N_RECORDS,
        config=_config(),
        fitness_mode=ARM,
        marginals=marginals,
    )
    elapsed = time.perf_counter() - started
    generation_dir = staging / "generation"
    generation_dir.mkdir(parents=True, exist_ok=False)
    table = table.reset_index(drop=True)
    table.to_csv(
        generation_dir / f"{ARM}_terminal_current.csv",
        index=False,
    )
    base._write_json(
        generation_dir / f"{ARM}_diagnostics.json",
        diagnostics,
    )
    best_loss = float(diagnostics["best_loss_diagnostic_only"])
    final_loss = float(diagnostics["final_current_squared_loss"])
    arm_result = {
        "terminal_table_sha256": base._frame_sha256(table),
        "rounds_run": int(diagnostics["rounds_run"]),
        "budget_rounds": N_ROUNDS,
        "candidate_evaluation_count": int(
            diagnostics["candidate_evaluation_count"]
        ),
        "termination_reason": diagnostics["termination_reason"],
        "output_table_identity": diagnostics["output_table_identity"],
        "final_current_squared_loss": final_loss,
        "final_current_normalized_l1": float(
            diagnostics["final_current_normalized_l1"]
        ),
        "best_loss_diagnostic_only": best_loss,
        "drift_ratio_final_over_best_observation_only": (
            final_loss / best_loss if best_loss > 0 else None
        ),
        "rho_schedule_audit": _audit_rho_schedule(diagnostics),
        "early_stopping": _early_stopping_summary(diagnostics),
        "floor_morphology_observation_only": _floor_morphology(
            diagnostics["loss_history"]
        ),
        "elapsed_sec": float(diagnostics["elapsed_sec"]),
    }
    return {
        "dataset": DATASET,
        "seed": SEED,
        "device": "cuda",
        "n_records": N_RECORDS,
        "measured_query_count": int(audit["query_count"]),
        "one_way_count": int(audit["one_way_count"]),
        "pool_query_count": int(audit["pool_query_count"]),
        "elapsed_sec_wall": float(elapsed),
        "arms": {ARM: arm_result},
    }


def _load_reference_reports(root: Path) -> dict[str, Any]:
    paths = {
        "oneway_pool_report": root / ONEWAY_POOL_REPORT_PATH,
        "frozen_report": root / FROZEN_REPORT_PATH,
        "pgm_report": root / PGM_REPORT_PATH,
    }
    for key, path in paths.items():
        observed = _sha256_file(path)
        if observed != REFERENCE_SHA256[key]:
            raise RuntimeError(
                f"{key} SHA 漂移: expected={REFERENCE_SHA256[key]}, "
                f"observed={observed}"
            )
    oneway_pool = _load_json_object(paths["oneway_pool_report"])
    frozen = _load_json_object(paths["frozen_report"])
    pgm = _load_json_object(paths["pgm_report"])
    return {
        "reference_sha256": dict(REFERENCE_SHA256),
        "oneway_pool_residual": oneway._metric_snapshot(
            oneway_pool["quality"]["residual"]
        ),
        "oneway_pool_equal_context_only": oneway._metric_snapshot(
            oneway_pool["quality"]["equal"]
        ),
        "frozen_residual": oneway._metric_snapshot(
            frozen["quality"]["residual"]
        ),
        "pgm": oneway._metric_snapshot(pgm["quality"]["pgm"]),
    }


def _comparison(
    quality: Mapping[str, Any],
    references: Mapping[str, Any],
) -> dict[str, Any]:
    new_snapshot = oneway._metric_snapshot(quality)

    def deltas(right: Mapping[str, Any]) -> dict:
        return {
            name: {
                "delta_mean": float(
                    new_snapshot[name]["mean"] - right[name]["mean"]
                ),
                "delta_median": float(
                    new_snapshot[name]["median"] - right[name]["median"]
                ),
                "left_mean": float(new_snapshot[name]["mean"]),
                "right_mean": float(right[name]["mean"]),
            }
            for name in new_snapshot
        }

    return {
        "new_residual_snapshot": new_snapshot,
        "new_residual_minus_oneway_pool_residual": deltas(
            references["oneway_pool_residual"]
        ),
        "new_residual_minus_frozen_residual": deltas(
            references["frozen_residual"]
        ),
        "new_residual_minus_pgm": deltas(references["pgm"]),
        "interpretation": (
            "diagnostic_only_lower_is_better_no_promotion_gate"
        ),
    }


def run(confirmed_protocol_sha256: str) -> Path:
    expected = assert_frozen_protocol_identity()
    if confirmed_protocol_sha256 != expected:
        raise ValueError(
            "polish-budget plants diagnostic protocol SHA-256 确认值不一致"
        )
    budget_identity = assert_polish_budget_identity()
    root = _repo_root()
    destination = root / OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"输出已存在，不覆盖：{destination}")
    audit = oneway._audit_generation_inputs(root)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-",
        dir=str(destination.parent),
    ))
    try:
        generation = _run_generation(root, staging, audit)
        heldout_queries, heldout_targets, reference = (
            oneway._load_heldout_and_reference(root)
        )
        quality = {
            ARM: oneway._evaluate_arm(
                root,
                staging,
                ARM,
                audit,
                generation["arms"][ARM],
                heldout_queries,
                heldout_targets,
                reference,
            ),
        }
        references = _load_reference_reports(root)
        report = {
            "contract_version": PROTOCOL_VERSION,
            "protocol_sha256": expected,
            "protocol": _protocol_manifest(),
            "polish_budget_identity": budget_identity,
            "generation_inputs": {
                key: value
                for key, value in audit.items()
                if key not in {
                    "queries",
                    "targets",
                    "one_way_queries",
                    "one_way_targets",
                    "pool_queries",
                    "pool_targets",
                }
            },
            "generation": generation,
            "quality": quality,
            "reference_report": references,
            "comparison": _comparison(quality[ARM], references),
            "summary": {
                "diagnostic_only": True,
                "formal_claim_allowed": False,
                "all_terminal_current": True,
                "budget_rounds": N_ROUNDS,
                "rounds_run": generation["arms"][ARM]["rounds_run"],
                "early_stopping_enabled": True,
                "termination_reason": (
                    generation["arms"][ARM]["termination_reason"]
                ),
                "equal_arm_rerun": False,
                "all_unconditional": True,
                "parameter_retuning_performed": False,
                "heldout_and_reference_loaded_after_generation": True,
            },
            "completed_at_unix": time.time(),
        }
        base._write_json(staging / "report.json", report)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "report.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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
    print(
        "polish-budget plants diagnostic report -> "
        f"{run(args.confirm_protocol_sha)}"
    )


if __name__ == "__main__":
    main()
