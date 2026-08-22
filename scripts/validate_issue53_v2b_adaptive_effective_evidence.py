#!/usr/bin/env python
"""Run the frozen artificial validation for Issue #53 V2b.

The formal entry has no scientific command-line knobs.  It generates only
stationary artificial AR(1) trajectories, evaluates the pre-registered V2b
checkpoints, and writes non-overwriting JSON artifacts.  It never reads a
project dataset, a saved real trajectory, or generator output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import inspect
import json
from math import isqrt
from pathlib import Path
import subprocess
import time
from typing import Any, Dict, Sequence

import numpy as np

from table_diffevo.adaptive_effective_evidence import (
    V2B_ADAPTIVE_CHECKPOINTS,
    V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION,
    V2B_RESOURCE_ROUND_CAP,
    V2B_SCALE_RATIO_LIMIT,
    compute_v2b_adaptive_checkpoint_evidence,
    compute_v2b_adaptive_trajectory_evidence,
    summarize_v2b_adaptive_checkpoint_decisions,
    v2b_scale_ratio_is_acceptable,
)
from table_diffevo.effective_evidence import (
    compute_v2_effective_round_evidence_for_batch,
)


ARTIFICIAL_PROTOCOL_VERSION = (
    "issue53-v2b-adaptive-effective-evidence-artificial-v1"
)
REPORT_FORMAT = "issue53_v2b_adaptive_effective_evidence_report_v1"
SEED_NAMESPACE = (53, 2, 2)
REPEAT_COUNT = 2000
MAX_TRAJECTORY_LENGTH = 2048
CHECKPOINTS = V2B_ADAPTIVE_CHECKPOINTS
CONFIDENCE_MULTIPLIER = 1.96
COVERAGE_LOWER = 0.925
COVERAGE_UPPER = 0.975
LONG_RUN_VARIANCE_RATIO_LOWER = 0.80
LONG_RUN_VARIANCE_RATIO_UPPER = 1.25
MAIN_READY_COUNT_MINIMUM = 1850
IID_RESOURCE_MEDIAN_MAXIMUM = 512.0
PHI_0P5_RESOURCE_MEDIAN_MAXIMUM = 1024.0
MAIN_POOLED_RESOURCE_MEAN_MAXIMUM = 1536.0
SLOW_RELEASE_COUNT_MINIMUM = 1000

PROTOCOL_DOCUMENT = Path(
    "docs/设计/Issue53_V2b自适应有效证据人工验收协议.md"
)
DESIGN_DOCUMENT = Path(
    "docs/设计/Issue53_V2b自适应有效证据设计稿.md"
)
V2_CORE_MODULE = Path("src/table_diffevo/effective_evidence.py")
V2B_CORE_MODULE = Path(
    "src/table_diffevo/adaptive_effective_evidence.py"
)
RUNNER_MODULE = Path(
    "scripts/validate_issue53_v2b_adaptive_effective_evidence.py"
)
AUDITOR_MODULE = Path(
    "scripts/audit_issue53_v2b_adaptive_effective_evidence.py"
)
V2_CORE_TEST_MODULE = Path("tests/test_effective_evidence.py")
V2B_CORE_TEST_MODULE = Path("tests/test_adaptive_effective_evidence.py")
RUNNER_TEST_MODULE = Path(
    "tests/test_issue53_v2b_adaptive_effective_evidence_artificial.py"
)

FORBIDDEN_DECISION_FIELDS = {
    "stable",
    "converged",
    "qualified",
    "stop",
    "stop_round",
    "threshold",
    "quality_pass",
}


@dataclass(frozen=True)
class ArtificialFamily:
    """One stationary Gaussian AR(1) family with known LRV."""

    code: int
    name: str
    phi: float
    role: str

    @property
    def theoretical_long_run_variance(self) -> float:
        return (1.0 + self.phi) / (1.0 - self.phi)

    @property
    def theoretical_raw_ess_ratio(self) -> float:
        return 1.0 / self.theoretical_long_run_variance


FAMILIES = (
    ArtificialFamily(0, "iid", 0.0, "main"),
    ArtificialFamily(1, "ar1_phi_0p5", 0.5, "main"),
    ArtificialFamily(2, "ar1_phi_0p8", 0.8, "main"),
    ArtificialFamily(3, "ar1_phi_m0p5", -0.5, "negative_control"),
    ArtificialFamily(4, "ar1_phi_0p95", 0.95, "slow_pressure"),
)
MAIN_FAMILY_NAMES = ("iid", "ar1_phi_0p5", "ar1_phi_0p8")
NEGATIVE_FAMILY_NAME = "ar1_phi_m0p5"
SLOW_FAMILY_NAME = "ar1_phi_0p95"


def _strict_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_strict_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        handle.write("\n")


def frozen_protocol() -> Dict[str, Any]:
    """Return every scientific choice used by the formal V2b run."""

    protocol = {
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "evidence_contract_version": (
            V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        "scope": {
            "scalar_trajectory_only": True,
            "reads_project_dataset": False,
            "reads_saved_real_trajectory": False,
            "runs_generator": False,
            "uses_gpu": False,
            "consumes_privacy_budget": False,
            "stationarity_decision_present": False,
            "convergence_decision_present": False,
            "stopping_decision_present": False,
        },
        "randomness": {
            "library": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "seed_sequence_prefix": list(SEED_NAMESPACE),
            "seed_sequence_suffix": ["family_code", "repeat_index"],
            "repeat_index_start": 0,
            "repeat_index_end_inclusive": REPEAT_COUNT - 1,
            "repeat_count_per_family": REPEAT_COUNT,
        },
        "trajectory": {
            "maximum_length": MAX_TRAJECTORY_LENGTH,
            "checkpoints": list(CHECKPOINTS),
            "initial_draw": "standard_normal_stationary_marginal",
            "innovation_draw": "independent_standard_normal",
            "recurrence": (
                "x[t]=phi*x[t-1]+sqrt(1-phi^2)*epsilon[t]"
            ),
            "burn_in": 0,
            "marginal_mean": 0.0,
            "marginal_variance": 1.0,
        },
        "families": [
            {
                **asdict(family),
                "theoretical_long_run_variance": (
                    family.theoretical_long_run_variance
                ),
                "theoretical_raw_ess_ratio": (
                    family.theoretical_raw_ess_ratio
                ),
            }
            for family in FAMILIES
        ],
        "adaptive_rule": {
            "short_batch": "floor(sqrt(n))",
            "long_batch": "2*floor(sqrt(n))",
            "formal_inflation_floor": 1.0,
            "scale_ratio_inclusive_maximum": V2B_SCALE_RATIO_LIMIT,
            "official_inflation": "max(short_inflation,long_inflation)",
            "first_ready_only_for_primary_metrics": True,
            "resource_cap_is_automatic_pass": False,
            "resource_failure_reason": (
                "resource_cap_without_multiscale_evidence"
            ),
        },
        "interval": {
            "form": "sample_mean_plus_or_minus_multiplier_times_mcse",
            "multiplier": CONFIDENCE_MULTIPLIER,
            "true_mean": 0.0,
            "mcse_is_maximum_scale_conservative_value": True,
        },
        "main_acceptance": {
            "families": list(MAIN_FAMILY_NAMES),
            "first_ready_count_minimum_per_family": (
                MAIN_READY_COUNT_MINIMUM
            ),
            "coverage_inclusive_lower": COVERAGE_LOWER,
            "coverage_inclusive_upper": COVERAGE_UPPER,
            "median_official_lrv_ratio_inclusive_lower": (
                LONG_RUN_VARIANCE_RATIO_LOWER
            ),
            "median_official_lrv_ratio_inclusive_upper": (
                LONG_RUN_VARIANCE_RATIO_UPPER
            ),
            "median_official_ess_ratio_strict_order": list(
                MAIN_FAMILY_NAMES
            ),
        },
        "cost_acceptance": {
            "not_ready_resource_round_count": V2B_RESOURCE_ROUND_CAP,
            "iid_resource_median_inclusive_maximum": (
                IID_RESOURCE_MEDIAN_MAXIMUM
            ),
            "phi_0p5_resource_median_inclusive_maximum": (
                PHI_0P5_RESOURCE_MEDIAN_MAXIMUM
            ),
            "main_equal_weight_pooled_resource_mean_inclusive_maximum": (
                MAIN_POOLED_RESOURCE_MEAN_MAXIMUM
            ),
            "minimum_pooled_saving_fraction": 0.25,
        },
        "negative_control_acceptance": {
            "family": NEGATIVE_FAMILY_NAME,
            "every_checkpoint_short_raw_ess_ratio_median_above": 1.0,
            "every_checkpoint_long_raw_ess_ratio_median_above": 1.0,
            "every_formal_ess_ratio_at_most": 1.0,
            "formal_mcse_not_below_iid_standard_error": True,
        },
        "slow_pressure_acceptance": {
            "family": SLOW_FAMILY_NAME,
            "complete_rejection_count": 0,
            "safe_release_count_minimum": SLOW_RELEASE_COUNT_MINIMUM,
            "safe_release_coverage_inclusive_lower": COVERAGE_LOWER,
            "safe_release_coverage_inclusive_upper": COVERAGE_UPPER,
            "safe_release_median_lrv_ratio_inclusive_lower": (
                LONG_RUN_VARIANCE_RATIO_LOWER
            ),
            "safe_release_median_lrv_ratio_inclusive_upper": (
                LONG_RUN_VARIANCE_RATIO_UPPER
            ),
            "intermediate_release_action": "candidate_failed",
        },
        "global_acceptance": {
            "core_failure_count_must_equal": 0,
            "nonfinite_output_count_must_equal": 0,
            "contract_violation_count_must_equal": 0,
            "formal_ess_cap_violation_count_must_equal": 0,
            "mcse_floor_violation_count_must_equal": 0,
            "trajectory_identity_violation_count_must_equal": 0,
            "fixed_boundary_checks_must_pass": True,
        },
        "post_result_retuning_allowed": False,
    }
    _strict_json_bytes(protocol)
    return protocol


def build_plan() -> Dict[str, Any]:
    protocol = frozen_protocol()
    plan = {
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "mode": "plan_only_no_artificial_draws",
        "protocol_sha256": _sha256_json(protocol),
        "family_count": len(FAMILIES),
        "trajectory_count": len(FAMILIES) * REPEAT_COUNT,
        "checkpoint_classification_count": (
            len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS)
        ),
        "scale_evaluation_count": (
            len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS) * 2
        ),
        "maximum_artificial_scalar_count": (
            len(FAMILIES) * REPEAT_COUNT * MAX_TRAJECTORY_LENGTH
        ),
        "real_data_accessed": False,
        "generation_started": False,
        "execution_started": False,
        "protocol": protocol,
    }
    _strict_json_bytes(plan)
    return plan


def generate_artificial_trajectory(
    family: ArtificialFamily,
    *,
    repeat_index: int,
    maximum_length: int,
    seed_namespace: Sequence[int] = SEED_NAMESPACE,
) -> np.ndarray:
    """Generate one stationary trajectory; formal wrapper fixes all inputs."""

    if family not in FAMILIES:
        raise ValueError("family must be one of the frozen protocol families")
    if isinstance(repeat_index, bool) or not isinstance(
        repeat_index, (int, np.integer)
    ) or repeat_index < 0:
        raise ValueError("repeat_index must be a nonnegative integer")
    if isinstance(maximum_length, bool) or not isinstance(
        maximum_length, (int, np.integer)
    ) or maximum_length < 2:
        raise ValueError("maximum_length must be an integer of at least two")
    normalized_namespace = tuple(seed_namespace)
    if (
        not normalized_namespace
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in normalized_namespace
        )
    ):
        raise ValueError("seed_namespace must contain only integers")

    seed = np.random.SeedSequence([
        *(int(value) for value in normalized_namespace),
        family.code,
        int(repeat_index),
    ])
    rng = np.random.Generator(np.random.PCG64(seed))
    values = rng.standard_normal(int(maximum_length))
    if family.phi != 0.0:
        innovation_scale = float(np.sqrt(1.0 - family.phi**2))
        for position in range(1, len(values)):
            values[position] = (
                family.phi * values[position - 1]
                + innovation_scale * values[position]
            )
    return values


def _finite_optional(values: Sequence[float | None]) -> bool:
    present = [value for value in values if value is not None]
    return bool(not present or np.all(np.isfinite(present)))


def checkpoint_record(
    family: ArtificialFamily,
    repeat_index: int,
    values: np.ndarray,
    checkpoint_result,
) -> Dict[str, Any]:
    """Flatten and independently check one project-core result."""

    n = checkpoint_result.actual_round_count
    prefix = np.asarray(values[:n], dtype=np.float64)
    short_raw_ess_ratio = (
        1.0 / checkpoint_result.short_raw_correlation_inflation
        if checkpoint_result.short_raw_correlation_inflation is not None
        else None
    )
    long_raw_ess_ratio = (
        1.0 / checkpoint_result.long_raw_correlation_inflation
        if checkpoint_result.long_raw_correlation_inflation is not None
        else None
    )
    short_formal_ess_ratio = (
        1.0
        / checkpoint_result.short_conservative_correlation_inflation
        if checkpoint_result.short_conservative_correlation_inflation
        is not None
        else None
    )
    long_formal_ess_ratio = (
        1.0
        / checkpoint_result.long_conservative_correlation_inflation
        if checkpoint_result.long_conservative_correlation_inflation
        is not None
        else None
    )
    official_ess_ratio = (
        checkpoint_result.effective_round_count / n
        if checkpoint_result.effective_round_count is not None
        else None
    )

    fields = set(asdict(checkpoint_result))
    structural_checks = {
        "round_count": n in CHECKPOINTS,
        "short_batch": (
            checkpoint_result.short_batch_round_count == isqrt(n)
        ),
        "long_batch": (
            checkpoint_result.long_batch_round_count == 2 * isqrt(n)
        ),
        "stationarity_marker": (
            checkpoint_result.stationarity_not_assessed is True
        ),
        "contract_version": (
            checkpoint_result.contract_version
            == V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        "forbidden_fields_absent": FORBIDDEN_DECISION_FIELDS.isdisjoint(
            fields
        ),
        "short_state_consistent": (
            checkpoint_result.short_numerically_estimable
            == (checkpoint_result.short_reason is None)
        ),
        "long_state_consistent": (
            checkpoint_result.long_numerically_estimable
            == (checkpoint_result.long_reason is None)
        ),
    }

    numeric_fields = (
        checkpoint_result.single_round_variance,
        checkpoint_result.short_long_run_variance,
        checkpoint_result.long_long_run_variance,
        checkpoint_result.short_raw_correlation_inflation,
        checkpoint_result.long_raw_correlation_inflation,
        checkpoint_result.short_conservative_correlation_inflation,
        checkpoint_result.long_conservative_correlation_inflation,
        checkpoint_result.scale_ratio,
        checkpoint_result.official_correlation_inflation,
        checkpoint_result.official_long_run_variance,
        checkpoint_result.effective_round_count,
        checkpoint_result.mcse,
    )
    nonfinite_output = not _finite_optional(numeric_fields)
    contract_violation = not all(structural_checks.values())
    formal_ess_cap_violation = False
    mcse_floor_violation = False

    both_scales_estimable = bool(
        checkpoint_result.short_numerically_estimable
        and checkpoint_result.long_numerically_estimable
    )
    if both_scales_estimable:
        required = numeric_fields
        if any(value is None for value in required):
            contract_violation = True
        else:
            short_inflation = float(
                checkpoint_result.short_conservative_correlation_inflation
            )
            long_inflation = float(
                checkpoint_result.long_conservative_correlation_inflation
            )
            expected_official = max(short_inflation, long_inflation)
            expected_ratio = max(short_inflation, long_inflation) / min(
                short_inflation, long_inflation
            )
            expected_lrv = float(
                checkpoint_result.single_round_variance
                * expected_official
            )
            expected_ess = float(n / expected_official)
            expected_mcse = float(np.sqrt(expected_lrv / n))
            expected_short_raw = float(
                checkpoint_result.short_long_run_variance
                / checkpoint_result.single_round_variance
            )
            expected_long_raw = float(
                checkpoint_result.long_long_run_variance
                / checkpoint_result.single_round_variance
            )
            contract_violation = bool(
                contract_violation
                or not np.isclose(
                    checkpoint_result.short_raw_correlation_inflation,
                    expected_short_raw,
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    checkpoint_result.long_raw_correlation_inflation,
                    expected_long_raw,
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    short_inflation,
                    max(1.0, expected_short_raw),
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    long_inflation,
                    max(1.0, expected_long_raw),
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    checkpoint_result.scale_ratio,
                    expected_ratio,
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    checkpoint_result.effective_round_count,
                    expected_ess,
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    checkpoint_result.official_correlation_inflation,
                    expected_official,
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    checkpoint_result.official_long_run_variance,
                    expected_lrv,
                    rtol=1e-12,
                    atol=1e-12,
                )
                or not np.isclose(
                    checkpoint_result.mcse,
                    expected_mcse,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
        if checkpoint_result.adaptive_numerically_estimable:
            contract_violation = bool(
                contract_violation
                or checkpoint_result.reason is not None
                or checkpoint_result.scale_ratio > V2B_SCALE_RATIO_LIMIT
            )
        else:
            contract_violation = bool(
                contract_violation
                or checkpoint_result.reason != "multiscale_disagreement"
                or checkpoint_result.scale_ratio <= V2B_SCALE_RATIO_LIMIT
            )
    else:
        contract_violation = bool(
            contract_violation
            or checkpoint_result.reason != "core_not_estimable"
        )

    if checkpoint_result.effective_round_count is not None:
        formal_ess_cap_violation = bool(
            checkpoint_result.effective_round_count > n
        )
    if checkpoint_result.mcse is not None:
        ordinary_standard_error = float(
            np.sqrt(checkpoint_result.single_round_variance / n)
        )
        mcse_floor_violation = bool(
            checkpoint_result.mcse < ordinary_standard_error
        )

    return {
        "family_code": family.code,
        "family": family.name,
        "repeat_index": int(repeat_index),
        "n": n,
        "sample_mean": float(np.mean(prefix)),
        "short_batch": checkpoint_result.short_batch_round_count,
        "long_batch": checkpoint_result.long_batch_round_count,
        "short_raw_ess_ratio": short_raw_ess_ratio,
        "long_raw_ess_ratio": long_raw_ess_ratio,
        "short_formal_ess_ratio": short_formal_ess_ratio,
        "long_formal_ess_ratio": long_formal_ess_ratio,
        "official_ess_ratio": official_ess_ratio,
        "scale_ratio": checkpoint_result.scale_ratio,
        "official_inflation": (
            checkpoint_result.official_correlation_inflation
        ),
        "official_lrv": checkpoint_result.official_long_run_variance,
        "official_mcse": checkpoint_result.mcse,
        "adaptive_numerically_estimable": (
            checkpoint_result.adaptive_numerically_estimable
        ),
        "reason": checkpoint_result.reason,
        "short_reason": checkpoint_result.short_reason,
        "long_reason": checkpoint_result.long_reason,
        "nonfinite_output": nonfinite_output,
        "contract_violation": contract_violation,
        "formal_ess_cap_violation": formal_ess_cap_violation,
        "mcse_floor_violation": mcse_floor_violation,
    }


def trajectory_record(
    family: ArtificialFamily,
    repeat_index: int,
    values: np.ndarray,
    trajectory_result,
    checkpoint_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Record first-ready metrics and verify first-ready identity."""

    decisions = [
        bool(row["adaptive_numerically_estimable"])
        for row in checkpoint_rows
    ]
    independent_summary = summarize_v2b_adaptive_checkpoint_decisions(
        decisions
    )
    identity_violation = bool(
        trajectory_result.first_adaptive_numerically_estimable_round
        != independent_summary.first_adaptive_numerically_estimable_round
        or trajectory_result.resource_round_count
        != independent_summary.resource_round_count
        or trajectory_result.reason != independent_summary.reason
        or tuple(row["n"] for row in checkpoint_rows) != CHECKPOINTS
        or trajectory_result.adaptive_numerically_estimable
        != (
            independent_summary.first_adaptive_numerically_estimable_round
            is not None
        )
        or trajectory_result.stationarity_not_assessed is not True
        or trajectory_result.contract_version
        != V2B_ADAPTIVE_EFFECTIVE_EVIDENCE_RESEARCH_CONTRACT_VERSION
        or not FORBIDDEN_DECISION_FIELDS.isdisjoint(
            asdict(trajectory_result)
        )
    )

    first_round = independent_summary.first_adaptive_numerically_estimable_round
    if first_round is None:
        first_coverage = None
        first_lrv_ratio = None
        first_ess_ratio = None
        first_sample_mean = None
        first_mcse = None
        first_scale_ratio = None
    else:
        first_row = next(
            row for row in checkpoint_rows if row["n"] == first_round
        )
        first_sample_mean = float(np.mean(values[:first_round]))
        first_mcse = float(first_row["official_mcse"])
        first_coverage = bool(
            abs(first_sample_mean)
            <= CONFIDENCE_MULTIPLIER * first_mcse
        )
        first_lrv_ratio = float(
            first_row["official_lrv"]
            / family.theoretical_long_run_variance
        )
        first_ess_ratio = float(first_row["official_ess_ratio"])
        first_scale_ratio = float(first_row["scale_ratio"])

    return {
        "family_code": family.code,
        "family": family.name,
        "phi": family.phi,
        "repeat_index": int(repeat_index),
        "first_adaptive_numerically_estimable_round": first_round,
        "resource_round_count": independent_summary.resource_round_count,
        "reason": independent_summary.reason,
        "first_ready_coverage": first_coverage,
        "first_ready_official_lrv_ratio": first_lrv_ratio,
        "first_ready_official_ess_ratio": first_ess_ratio,
        "first_ready_sample_mean": first_sample_mean,
        "first_ready_mcse": first_mcse,
        "first_ready_scale_ratio": first_scale_ratio,
        "trajectory_identity_violation": identity_violation,
    }


def _new_checkpoint_accumulator() -> Dict[str, Any]:
    return {
        "adaptive_count": 0,
        "core_failure_count": 0,
        "nonfinite_output_count": 0,
        "contract_violation_count": 0,
        "formal_ess_cap_violation_count": 0,
        "mcse_floor_violation_count": 0,
        "short_raw_ess_ratios": [],
        "long_raw_ess_ratios": [],
        "short_formal_ess_ratios": [],
        "long_formal_ess_ratios": [],
        "official_ess_ratios": [],
        "scale_ratios": [],
    }


def _append_checkpoint_record(
    accumulator: Dict[str, Any],
    row: Dict[str, Any],
) -> None:
    accumulator["adaptive_count"] += int(
        row["adaptive_numerically_estimable"]
    )
    accumulator["core_failure_count"] += int(
        row["reason"] == "core_not_estimable"
    )
    for field in (
        "nonfinite_output",
        "contract_violation",
        "formal_ess_cap_violation",
        "mcse_floor_violation",
    ):
        accumulator[f"{field}_count"] += int(row[field])
    for output_name, row_name in (
        ("short_raw_ess_ratios", "short_raw_ess_ratio"),
        ("long_raw_ess_ratios", "long_raw_ess_ratio"),
        ("short_formal_ess_ratios", "short_formal_ess_ratio"),
        ("long_formal_ess_ratios", "long_formal_ess_ratio"),
        ("official_ess_ratios", "official_ess_ratio"),
        ("scale_ratios", "scale_ratio"),
    ):
        if row[row_name] is not None:
            accumulator[output_name].append(float(row[row_name]))


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_checkpoint(
    family: ArtificialFamily,
    n: int,
    accumulator: Dict[str, Any],
    *,
    repeat_count: int,
) -> Dict[str, Any]:
    row = {
        "family_code": family.code,
        "family": family.name,
        "role": family.role,
        "phi": family.phi,
        "n": n,
        "repeat_count": int(repeat_count),
        "adaptive_count": accumulator["adaptive_count"],
        "adaptive_rate": accumulator["adaptive_count"] / repeat_count,
        "core_failure_count": accumulator["core_failure_count"],
        "nonfinite_output_count": accumulator["nonfinite_output_count"],
        "contract_violation_count": (
            accumulator["contract_violation_count"]
        ),
        "formal_ess_cap_violation_count": (
            accumulator["formal_ess_cap_violation_count"]
        ),
        "mcse_floor_violation_count": (
            accumulator["mcse_floor_violation_count"]
        ),
        "short_raw_ess_ratio_median": _median(
            accumulator["short_raw_ess_ratios"]
        ),
        "long_raw_ess_ratio_median": _median(
            accumulator["long_raw_ess_ratios"]
        ),
        "short_formal_ess_ratio_median": _median(
            accumulator["short_formal_ess_ratios"]
        ),
        "long_formal_ess_ratio_median": _median(
            accumulator["long_formal_ess_ratios"]
        ),
        "official_ess_ratio_median": _median(
            accumulator["official_ess_ratios"]
        ),
        "short_formal_ess_ratio_maximum": (
            float(max(accumulator["short_formal_ess_ratios"]))
            if accumulator["short_formal_ess_ratios"]
            else None
        ),
        "long_formal_ess_ratio_maximum": (
            float(max(accumulator["long_formal_ess_ratios"]))
            if accumulator["long_formal_ess_ratios"]
            else None
        ),
        "official_ess_ratio_maximum": (
            float(max(accumulator["official_ess_ratios"]))
            if accumulator["official_ess_ratios"]
            else None
        ),
        "scale_ratio_median": _median(accumulator["scale_ratios"]),
    }
    _strict_json_bytes(row)
    return row


def summarize_family(
    family: ArtificialFamily,
    trajectory_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    if not trajectory_rows:
        raise ValueError("trajectory_rows must not be empty")
    ready = [
        row
        for row in trajectory_rows
        if row["first_adaptive_numerically_estimable_round"] is not None
    ]
    resources = np.asarray(
        [row["resource_round_count"] for row in trajectory_rows],
        dtype=np.float64,
    )
    ready_count = len(ready)
    coverage_count = sum(
        row["first_ready_coverage"] is True for row in ready
    )
    coverage = coverage_count / ready_count if ready_count else None
    lrv_ratios = [
        row["first_ready_official_lrv_ratio"] for row in ready
    ]
    ess_ratios = [
        row["first_ready_official_ess_ratio"] for row in ready
    ]
    row = {
        "family_code": family.code,
        "family": family.name,
        "role": family.role,
        "phi": family.phi,
        "repeat_count": len(trajectory_rows),
        "theoretical_long_run_variance": (
            family.theoretical_long_run_variance
        ),
        "theoretical_raw_ess_ratio": family.theoretical_raw_ess_ratio,
        "first_ready_count": ready_count,
        "first_ready_rate": ready_count / len(trajectory_rows),
        "first_ready_coverage_count": coverage_count,
        "first_ready_coverage": coverage,
        "first_ready_official_lrv_ratio_median": _median(lrv_ratios),
        "first_ready_official_ess_ratio_median": _median(ess_ratios),
        "resource_round_count_sum": int(np.sum(resources)),
        "resource_round_count_minimum": float(np.min(resources)),
        "resource_round_count_q25": float(np.quantile(resources, 0.25)),
        "resource_round_count_median": float(np.median(resources)),
        "resource_round_count_mean": float(np.mean(resources)),
        "resource_round_count_q95": float(np.quantile(resources, 0.95)),
        "resource_round_count_maximum": float(np.max(resources)),
        "first_ready_at_cap_count": sum(
            row["first_adaptive_numerically_estimable_round"]
            == V2B_RESOURCE_ROUND_CAP
            for row in trajectory_rows
        ),
        "not_ready_at_cap_count": len(trajectory_rows) - ready_count,
        "trajectory_identity_violation_count": sum(
            row["trajectory_identity_violation"] for row in trajectory_rows
        ),
    }
    _strict_json_bytes(row)
    return row


def collect_matrix(
    *,
    families: Sequence[ArtificialFamily],
    repeat_count: int,
    maximum_length: int,
    checkpoints: Sequence[int],
    seed_namespace: Sequence[int],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Execute a matrix; the formal wrapper supplies only frozen values."""

    if tuple(checkpoints) != CHECKPOINTS:
        raise ValueError("checkpoints must equal the frozen V2b schedule")
    if maximum_length != MAX_TRAJECTORY_LENGTH:
        raise ValueError("maximum_length must equal the frozen V2b cap")
    if isinstance(repeat_count, bool) or not isinstance(
        repeat_count, (int, np.integer)
    ) or repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if not families or any(family not in FAMILIES for family in families):
        raise ValueError("families must contain frozen protocol families")
    if len(set(families)) != len(families):
        raise ValueError("families must not contain duplicates")
    trajectory_rows: list[Dict[str, Any]] = []
    checkpoint_summaries: list[Dict[str, Any]] = []
    family_summaries: list[Dict[str, Any]] = []

    round_indices = np.arange(1, maximum_length + 1, dtype=np.int64)
    for family in families:
        accumulators = {
            n: _new_checkpoint_accumulator() for n in checkpoints
        }
        current_family_rows = []
        for repeat_index in range(repeat_count):
            values = generate_artificial_trajectory(
                family,
                repeat_index=repeat_index,
                maximum_length=maximum_length,
                seed_namespace=seed_namespace,
            )
            values_before = values.copy()
            round_indices_before = round_indices.copy()
            result = compute_v2b_adaptive_trajectory_evidence(
                round_indices,
                values,
            )
            checkpoint_rows = [
                checkpoint_record(
                    family,
                    repeat_index,
                    values,
                    checkpoint_result,
                )
                for checkpoint_result in result.checkpoint_evidence
            ]
            for row in checkpoint_rows:
                _append_checkpoint_record(accumulators[row["n"]], row)
            first_row = trajectory_record(
                family,
                repeat_index,
                values,
                result,
                checkpoint_rows,
            )
            first_row["trajectory_identity_violation"] = bool(
                first_row["trajectory_identity_violation"]
                or not np.array_equal(values, values_before)
                or not np.array_equal(round_indices, round_indices_before)
            )
            current_family_rows.append(first_row)
            trajectory_rows.append(first_row)

        checkpoint_summaries.extend(
            summarize_checkpoint(
                family,
                n,
                accumulators[n],
                repeat_count=repeat_count,
            )
            for n in checkpoints
        )
        family_summaries.append(
            summarize_family(family, current_family_rows)
        )

    return trajectory_rows, checkpoint_summaries, family_summaries


def run_fixed_boundary_checks() -> Dict[str, Any]:
    """Repeat protocol boundaries inside the future formal entry."""

    checks: Dict[str, bool] = {}
    indices = np.arange(1, 257, dtype=np.int64)
    positions = np.arange(256, dtype=np.float64)

    short = compute_v2_effective_round_evidence_for_batch(
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
        batch_round_count=1,
    )
    long = compute_v2_effective_round_evidence_for_batch(
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
        batch_round_count=2,
    )
    checks["two_hand_checked_obm_values"] = bool(
        np.isclose(short.long_run_variance, 5.0 / 3.0)
        and np.isclose(long.long_run_variance, 8.0 / 3.0)
    )
    checks["scale_ratio_exact_boundary_passes"] = (
        v2b_scale_ratio_is_acceptable(1.0, 1.25)
    )
    checks["scale_ratio_nextafter_boundary_fails"] = not (
        v2b_scale_ratio_is_acceptable(
            1.0,
            np.nextafter(1.25, np.inf),
        )
    )

    one_bad_scale_values = np.where(
        (np.arange(256) // 16) % 2 == 0,
        1.0,
        -1.0,
    )
    one_bad_scale = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        one_bad_scale_values,
    )
    checks["one_bad_scale_fails_closed"] = bool(
        one_bad_scale.short_numerically_estimable
        and not one_bad_scale.long_numerically_estimable
        and one_bad_scale.reason == "core_not_estimable"
    )

    long_risk = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        np.linspace(0.0, 1.0, 256),
    )
    short_risk = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        np.sin(0.15 * positions),
    )
    checks["long_scale_maximum_is_official"] = bool(
        long_risk.long_conservative_correlation_inflation
        > long_risk.short_conservative_correlation_inflation
        and long_risk.official_correlation_inflation
        == long_risk.long_conservative_correlation_inflation
    )
    checks["short_scale_maximum_is_official"] = bool(
        short_risk.short_conservative_correlation_inflation
        > short_risk.long_conservative_correlation_inflation
        and short_risk.official_correlation_inflation
        == short_risk.short_conservative_correlation_inflation
    )

    try:
        compute_v2b_adaptive_checkpoint_evidence(
            np.arange(1, 258),
            np.linspace(0.0, 1.0, 257),
        )
    except ValueError:
        checks["noncheckpoint_classifier_rejected"] = True
    else:
        checks["noncheckpoint_classifier_rejected"] = False
    scale_only = compute_v2_effective_round_evidence_for_batch(
        np.arange(1, 258),
        np.linspace(0.0, 1.0, 257),
        batch_round_count=16,
    )
    checks["noncheckpoint_scale_core_still_computes"] = bool(
        scale_only.numerically_estimable
    )

    first_at_cap = summarize_v2b_adaptive_checkpoint_decisions(
        [False] * 14 + [True]
    )
    never_ready = summarize_v2b_adaptive_checkpoint_decisions(
        [False] * 15
    )
    checks["first_ready_at_cap_distinguished"] = bool(
        first_at_cap.first_adaptive_numerically_estimable_round == 2048
        and first_at_cap.reason is None
        and never_ready.first_adaptive_numerically_estimable_round is None
        and never_ready.reason
        == "resource_cap_without_multiscale_evidence"
    )

    base_values = np.sin(0.07 * positions) + 0.002 * positions
    base = compute_v2b_adaptive_checkpoint_evidence(indices, base_values)
    shifted = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        base_values + 13.0,
    )
    scaled = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        base_values * 7.0,
    )
    checks["shift_invariance"] = bool(
        np.isclose(base.scale_ratio, shifted.scale_ratio, rtol=1e-10)
        and np.isclose(base.mcse, shifted.mcse, rtol=1e-10)
    )
    checks["positive_scale_invariance"] = bool(
        np.isclose(base.scale_ratio, scaled.scale_ratio, rtol=1e-10)
        and np.isclose(scaled.mcse, 7.0 * base.mcse, rtol=1e-10)
    )

    constant = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        np.ones(256),
    )
    checks["constant_fails_closed"] = bool(
        constant.reason == "core_not_estimable"
        and constant.short_reason == "zero_round_variance"
        and constant.long_reason == "zero_round_variance"
    )
    periodic_values = np.where(np.arange(256) % 2 == 0, 1.0, -1.0)
    periodic = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        periodic_values,
    )
    checks["periodic_fails_closed"] = bool(
        periodic.reason == "core_not_estimable"
    )
    spike_values = np.zeros(256)
    spike_values[128] = 1.0
    spike = compute_v2b_adaptive_checkpoint_evidence(indices, spike_values)
    checks["single_spike_finite_and_capped"] = bool(
        spike.effective_round_count is not None
        and spike.effective_round_count <= 256
        and np.all(np.isfinite([
            spike.official_long_run_variance,
            spike.effective_round_count,
            spike.mcse,
        ]))
    )
    trend = compute_v2b_adaptive_checkpoint_evidence(
        indices,
        np.linspace(0.0, 1.0, 256),
    )
    checks["trend_has_no_decision_fields"] = bool(
        trend.stationarity_not_assessed
        and FORBIDDEN_DECISION_FIELDS.isdisjoint(asdict(trend))
    )

    invalid_inputs = (
        (np.concatenate([np.arange(1, 128), np.arange(129, 258)]), positions),
        (indices, np.resize([False, True], 256)),
        (indices, np.full(256, np.nan)),
    )
    invalid_rejections = []
    for invalid_indices, invalid_values in invalid_inputs:
        try:
            compute_v2b_adaptive_checkpoint_evidence(
                invalid_indices,
                invalid_values,
            )
        except ValueError:
            invalid_rejections.append(True)
        else:
            invalid_rejections.append(False)
    checks["invalid_inputs_rejected"] = all(invalid_rejections)

    result = {"passed": all(checks.values()), "checks": checks}
    _strict_json_bytes(result)
    return result


def build_acceptance_gates(
    trajectory_rows: Sequence[Dict[str, Any]],
    checkpoint_summaries: Sequence[Dict[str, Any]],
    family_summaries: Sequence[Dict[str, Any]],
    boundary_checks: Dict[str, Any],
) -> Dict[str, Any]:
    """Apply every frozen safety, cost, and pressure gate."""

    by_family = {row["family"]: row for row in family_summaries}
    by_cell = {
        (row["family"], row["n"]): row for row in checkpoint_summaries
    }
    if set(by_family) != {family.name for family in FAMILIES}:
        raise ValueError("family summaries do not match the frozen families")
    expected_cells = {
        (family.name, n) for family in FAMILIES for n in CHECKPOINTS
    }
    if set(by_cell) != expected_cells or len(checkpoint_summaries) != len(
        expected_cells
    ):
        raise ValueError("checkpoint summaries do not match the frozen matrix")
    repeat_counts = {row["repeat_count"] for row in family_summaries}
    if len(repeat_counts) != 1 or next(iter(repeat_counts)) < 1:
        raise ValueError("family summaries must share a positive repeat count")
    matrix_repeat_count = int(next(iter(repeat_counts)))
    if any(
        row["repeat_count"] != matrix_repeat_count
        for row in checkpoint_summaries
    ):
        raise ValueError("checkpoint summaries have an inconsistent repeat count")
    expected_identities = {
        (family.code, family.name, repeat_index)
        for family in FAMILIES
        for repeat_index in range(matrix_repeat_count)
    }
    actual_identities = [
        (row["family_code"], row["family"], row["repeat_index"])
        for row in trajectory_rows
    ]
    if (
        len(actual_identities) != len(expected_identities)
        or set(actual_identities) != expected_identities
    ):
        raise ValueError("trajectory rows do not match the frozen matrix identity")
    for family in FAMILIES:
        actual_summary = summarize_family(
            family,
            [row for row in trajectory_rows if row["family"] == family.name],
        )
        if actual_summary != by_family[family.name]:
            raise ValueError("family summary does not match trajectory rows")

    main_family_gates = {}
    for name in MAIN_FAMILY_NAMES:
        summary = by_family[name]
        coverage = summary["first_ready_coverage"]
        lrv_ratio = summary["first_ready_official_lrv_ratio_median"]
        main_family_gates[name] = {
            "ready_count": (
                summary["first_ready_count"] >= MAIN_READY_COUNT_MINIMUM
            ),
            "coverage": bool(
                coverage is not None
                and COVERAGE_LOWER <= coverage <= COVERAGE_UPPER
            ),
            "lrv_ratio": bool(
                lrv_ratio is not None
                and LONG_RUN_VARIANCE_RATIO_LOWER
                <= lrv_ratio
                <= LONG_RUN_VARIANCE_RATIO_UPPER
            ),
        }
    main_ess_medians = [
        by_family[name]["first_ready_official_ess_ratio_median"]
        for name in MAIN_FAMILY_NAMES
    ]
    main_ess_ordering = bool(
        all(value is not None for value in main_ess_medians)
        and main_ess_medians[0] > main_ess_medians[1] > main_ess_medians[2]
    )

    main_rows = [
        row for row in trajectory_rows if row["family"] in MAIN_FAMILY_NAMES
    ]
    pooled_resource_mean = float(np.mean([
        row["resource_round_count"] for row in main_rows
    ]))
    cost_gates = {
        "iid_resource_median": (
            by_family["iid"]["resource_round_count_median"]
            <= IID_RESOURCE_MEDIAN_MAXIMUM
        ),
        "phi_0p5_resource_median": (
            by_family["ar1_phi_0p5"]["resource_round_count_median"]
            <= PHI_0P5_RESOURCE_MEDIAN_MAXIMUM
        ),
        "main_pooled_resource_mean": (
            pooled_resource_mean <= MAIN_POOLED_RESOURCE_MEAN_MAXIMUM
        ),
    }

    negative_cells = [
        by_cell[(NEGATIVE_FAMILY_NAME, n)] for n in CHECKPOINTS
    ]
    negative_gates = {
        "short_raw_ess_medians_above_one": all(
            cell["short_raw_ess_ratio_median"] is not None
            and cell["short_raw_ess_ratio_median"] > 1.0
            for cell in negative_cells
        ),
        "long_raw_ess_medians_above_one": all(
            cell["long_raw_ess_ratio_median"] is not None
            and cell["long_raw_ess_ratio_median"] > 1.0
            for cell in negative_cells
        ),
        "formal_ess_ratios_at_most_one": all(
            cell["short_formal_ess_ratio_maximum"] is not None
            and cell["short_formal_ess_ratio_maximum"] <= 1.0
            and cell["long_formal_ess_ratio_maximum"] is not None
            and cell["long_formal_ess_ratio_maximum"] <= 1.0
            and cell["official_ess_ratio_maximum"] is not None
            and cell["official_ess_ratio_maximum"] <= 1.0
            for cell in negative_cells
        ),
        "all_outputs_estimable_and_finite": all(
            cell["core_failure_count"] == 0
            and cell["nonfinite_output_count"] == 0
            for cell in negative_cells
        ),
        "mcse_floor": all(
            cell["mcse_floor_violation_count"] == 0
            for cell in negative_cells
        ),
    }

    slow = by_family[SLOW_FAMILY_NAME]
    slow_count = slow["first_ready_count"]
    if slow_count == 0:
        slow_branch = "complete_rejection"
        slow_pass = True
    elif slow_count >= SLOW_RELEASE_COUNT_MINIMUM:
        slow_branch = "validated_release"
        slow_coverage = slow["first_ready_coverage"]
        slow_lrv = slow["first_ready_official_lrv_ratio_median"]
        slow_pass = bool(
            slow_coverage is not None
            and COVERAGE_LOWER <= slow_coverage <= COVERAGE_UPPER
            and slow_lrv is not None
            and LONG_RUN_VARIANCE_RATIO_LOWER
            <= slow_lrv
            <= LONG_RUN_VARIANCE_RATIO_UPPER
        )
    else:
        slow_branch = "unsafe_sparse_release"
        slow_pass = False
    slow_pressure = {
        "first_ready_count": slow_count,
        "branch": slow_branch,
        "passed": slow_pass,
    }

    checkpoint_violation_fields = (
        "core_failure_count",
        "nonfinite_output_count",
        "contract_violation_count",
        "formal_ess_cap_violation_count",
        "mcse_floor_violation_count",
    )
    global_gates = {
        field: sum(row[field] for row in checkpoint_summaries) == 0
        for field in checkpoint_violation_fields
    }
    global_gates["trajectory_identity_violation_count"] = (
        sum(
            row["trajectory_identity_violation"] for row in trajectory_rows
        )
        == 0
    )
    global_gates["fixed_boundary_checks"] = bool(
        boundary_checks["passed"]
    )

    failed_gates = []
    for family_name, family_gates in main_family_gates.items():
        failed_gates.extend(
            f"main.{family_name}.{name}"
            for name, passed in family_gates.items()
            if not passed
        )
    if not main_ess_ordering:
        failed_gates.append("main.ess_ordering")
    failed_gates.extend(
        f"cost.{name}" for name, passed in cost_gates.items() if not passed
    )
    failed_gates.extend(
        f"negative.{name}"
        for name, passed in negative_gates.items()
        if not passed
    )
    if not slow_pass:
        failed_gates.append("slow_pressure")
    failed_gates.extend(
        f"global.{name}"
        for name, passed in global_gates.items()
        if not passed
    )

    result = {
        "main_family_gates": main_family_gates,
        "main_ess_ratio_medians": dict(
            zip(MAIN_FAMILY_NAMES, main_ess_medians)
        ),
        "main_ess_ordering_pass": main_ess_ordering,
        "cost_gates": {
            **cost_gates,
            "main_pooled_resource_mean_value": pooled_resource_mean,
        },
        "negative_control_gates": negative_gates,
        "slow_pressure_gate": slow_pressure,
        "global_gates": global_gates,
        "failed_gates": failed_gates,
        "status": (
            "candidate_supported" if not failed_gates else "candidate_failed"
        ),
    }
    _strict_json_bytes(result)
    return result


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def build_execution_manifest(root: Path) -> Dict[str, Any]:
    status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "V2b artificial protocol requires a clean worktree before draws"
        )
    source_paths = (
        PROTOCOL_DOCUMENT,
        DESIGN_DOCUMENT,
        V2_CORE_MODULE,
        V2B_CORE_MODULE,
        RUNNER_MODULE,
        AUDITOR_MODULE,
        V2_CORE_TEST_MODULE,
        V2B_CORE_TEST_MODULE,
        RUNNER_TEST_MODULE,
    )
    missing = [str(path) for path in source_paths if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"V2b protocol source files are missing: {missing}")

    protocol = frozen_protocol()
    manifest = {
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": True,
        "protocol_sha256": _sha256_json(protocol),
        "source_sha256": {
            str(path): _sha256_file(root / path) for path in source_paths
        },
        "environment": {
            "python_version": __import__("sys").version,
            "numpy_version": np.__version__,
            "platform": __import__("platform").platform(),
            "machine": __import__("platform").machine(),
            "processor": __import__("platform").processor(),
        },
        "protocol": protocol,
    }
    _strict_json_bytes(manifest)
    return manifest


def scientific_payload(
    *,
    trajectory_rows: Sequence[Dict[str, Any]],
    checkpoint_summaries: Sequence[Dict[str, Any]],
    family_summaries: Sequence[Dict[str, Any]],
    boundary_checks: Dict[str, Any],
    acceptance: Dict[str, Any],
) -> Dict[str, Any]:
    payload = {
        "trajectory_first_ready_rows": list(trajectory_rows),
        "checkpoint_summaries": list(checkpoint_summaries),
        "family_summaries": list(family_summaries),
        "boundary_checks": boundary_checks,
        "acceptance": acceptance,
    }
    _strict_json_bytes(payload)
    return payload


def run_artificial_protocol(
    output_dir: Path,
) -> tuple[Path, Path, Dict[str, Any]]:
    """Execute exactly the frozen formal matrix and write new artifacts."""

    if set(inspect.signature(run_artificial_protocol).parameters) != {
        "output_dir"
    }:
        raise RuntimeError("formal V2b runner gained an unexpected override")
    root = _repo_root()
    manifest = build_execution_manifest(root)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "protocol_manifest.json"
    report_path = output_dir / "adaptive_evidence_report.json"
    _write_json_exclusive(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)

    started = time.perf_counter()
    boundary_checks = run_fixed_boundary_checks()
    trajectory_rows, checkpoint_summaries, family_summaries = collect_matrix(
        families=FAMILIES,
        repeat_count=REPEAT_COUNT,
        maximum_length=MAX_TRAJECTORY_LENGTH,
        checkpoints=CHECKPOINTS,
        seed_namespace=SEED_NAMESPACE,
    )
    acceptance = build_acceptance_gates(
        trajectory_rows,
        checkpoint_summaries,
        family_summaries,
        boundary_checks,
    )
    payload = scientific_payload(
        trajectory_rows=trajectory_rows,
        checkpoint_summaries=checkpoint_summaries,
        family_summaries=family_summaries,
        boundary_checks=boundary_checks,
        acceptance=acceptance,
    )
    elapsed = float(time.perf_counter() - started)
    report = {
        "report_format": REPORT_FORMAT,
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "status": acceptance["status"],
        "manifest_path": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": manifest["protocol_sha256"],
        "git_commit": manifest["git_commit"],
        "protocol": manifest["protocol"],
        "execution": {
            "elapsed_sec": elapsed,
            "family_count": len(FAMILIES),
            "trajectory_count": len(trajectory_rows),
            "checkpoint_classification_count": (
                len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS)
            ),
            "scale_evaluation_count": (
                len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS) * 2
            ),
            "real_data_accessed": False,
            "generator_run": False,
            "privacy_budget_consumed": False,
        },
        "scientific_payload": payload,
        "scientific_result_sha256": _sha256_json(payload),
        "audit_required_before_interpretation": True,
    }
    _strict_json_bytes(report)
    _write_json_exclusive(report_path, report)
    return manifest_path, report_path, report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    manifest_path, report_path, report = run_artificial_protocol(
        Path(args.output_dir)
    )
    print("\n===== Issue #53 V2b Artificial Evidence =====")
    print(f"status={report['status']}")
    print(f"scientific_sha256={report['scientific_result_sha256']}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
