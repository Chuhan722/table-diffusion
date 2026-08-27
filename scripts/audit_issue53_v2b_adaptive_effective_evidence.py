#!/usr/bin/env python
"""Independent replay audit for the frozen Issue #53 V2b experiment.

This module deliberately does not import the formal runner or either project
effective-evidence module.  It owns a second AR(1) generator, a direct NumPy
implementation of the overlapping-batch formula, and independent aggregation
and acceptance logic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isqrt
from pathlib import Path
import subprocess
from typing import Any, Dict, Sequence

import numpy as np


ARTIFICIAL_PROTOCOL_VERSION = (
    "issue53-v2b-adaptive-effective-evidence-artificial-v1"
)
EVIDENCE_CONTRACT_VERSION = (
    "issue53-v2b-adaptive-effective-round-evidence-research-v1"
)
REPORT_FORMAT = "issue53_v2b_adaptive_effective_evidence_report_v1"
AUDIT_FORMAT = "issue53_v2b_adaptive_effective_evidence_audit_v1"
SEED_NAMESPACE = (53, 2, 2)
REPEAT_COUNT = 2000
MAX_TRAJECTORY_LENGTH = 2048
CHECKPOINTS = tuple(range(256, 2048 + 1, 128))
SCALE_RATIO_LIMIT = 1.25
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
V2B_CORE_MODULE = Path("src/table_diffevo/adaptive_effective_evidence.py")
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
SOURCE_PATHS = (
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
class AuditFamily:
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
    AuditFamily(0, "iid", 0.0, "main"),
    AuditFamily(1, "ar1_phi_0p5", 0.5, "main"),
    AuditFamily(2, "ar1_phi_0p8", 0.8, "main"),
    AuditFamily(3, "ar1_phi_m0p5", -0.5, "negative_control"),
    AuditFamily(4, "ar1_phi_0p95", 0.95, "slow_pressure"),
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


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _load_json_strict(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )


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


def expected_protocol() -> Dict[str, Any]:
    """Reconstruct the protocol without consulting the runner."""

    protocol = {
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "evidence_contract_version": EVIDENCE_CONTRACT_VERSION,
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
            "scale_ratio_inclusive_maximum": SCALE_RATIO_LIMIT,
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
            "not_ready_resource_round_count": MAX_TRAJECTORY_LENGTH,
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


def generate_artificial_trajectory_independent(
    family: AuditFamily,
    *,
    repeat_index: int,
    maximum_length: int,
    seed_namespace: Sequence[int] = SEED_NAMESPACE,
) -> np.ndarray:
    if family not in FAMILIES:
        raise ValueError("family must be one of the frozen audit families")
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


def _normalize_inputs(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(tuple(round_indices))
    raw_values = tuple(values)
    if indices.ndim != 1 or len(indices) < 2:
        raise ValueError("round_indices must be a one-dimensional sequence")
    if any(isinstance(value, (bool, np.bool_)) for value in indices):
        raise ValueError("round_indices must not contain bool")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("round_indices must contain integers")
    normalized_indices = np.asarray(indices, dtype=np.int64)
    if normalized_indices[0] < 1 or not np.all(
        np.diff(normalized_indices) == 1
    ):
        raise ValueError("round_indices must be positive and contiguous")
    if any(isinstance(value, (bool, np.bool_)) for value in raw_values):
        raise ValueError("values must not contain bool")
    normalized_values = np.asarray(raw_values)
    if normalized_values.ndim != 1 or len(normalized_values) != len(indices):
        raise ValueError("values must be a matching one-dimensional sequence")
    if not (
        np.issubdtype(normalized_values.dtype, np.integer)
        or np.issubdtype(normalized_values.dtype, np.floating)
    ):
        raise ValueError("values must contain real numbers")
    normalized_values = np.asarray(normalized_values, dtype=np.float64)
    if not np.all(np.isfinite(normalized_values)):
        raise ValueError("values must be finite")
    return normalized_indices.copy(), normalized_values.copy()


def _independent_obm(values: np.ndarray, batch: int) -> Dict[str, Any]:
    n = len(values)
    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)):
        raise ValueError("batch must be an integer")
    batch = int(batch)
    if not 1 <= batch < n:
        raise ValueError("batch must be positive and smaller than n")
    overlapping_count = n - batch + 1
    with np.errstate(over="ignore", invalid="ignore"):
        variance = float(np.var(values, ddof=1))
    if not np.isfinite(variance):
        return {
            "batch": batch,
            "variance": None,
            "lrv": None,
            "raw_inflation": None,
            "formal_inflation": None,
            "estimable": False,
            "reason": "nonfinite_computation",
        }
    if variance == 0.0:
        return {
            "batch": batch,
            "variance": 0.0,
            "lrv": None,
            "raw_inflation": None,
            "formal_inflation": None,
            "estimable": False,
            "reason": "zero_round_variance",
        }

    with np.errstate(over="ignore", invalid="ignore"):
        overall_mean = float(np.mean(values))
        centered = values - overall_mean
        cumulative = np.concatenate((
            np.asarray([0.0]),
            np.cumsum(centered, dtype=np.float64),
        ))
        deviations = (
            cumulative[batch:] - cumulative[:-batch]
        ) / batch
        squared_sum = float(
            np.sum(np.square(deviations), dtype=np.float64)
        )
        lrv = float(
            n
            * batch
            / ((n - batch) * overlapping_count)
            * squared_sum
        )
    if not np.isfinite(overall_mean) or not np.isfinite(lrv):
        return {
            "batch": batch,
            "variance": variance,
            "lrv": None,
            "raw_inflation": None,
            "formal_inflation": None,
            "estimable": False,
            "reason": "nonfinite_computation",
        }
    if lrv == 0.0:
        return {
            "batch": batch,
            "variance": variance,
            "lrv": 0.0,
            "raw_inflation": None,
            "formal_inflation": None,
            "estimable": False,
            "reason": "degenerate_long_run_variance",
        }

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        raw_inflation = float(lrv / variance)
        formal_inflation = max(1.0, raw_inflation)
        derived = (
            raw_inflation,
            formal_inflation,
            n / raw_inflation,
            n / formal_inflation,
            np.sqrt(variance * formal_inflation / n),
        )
    if raw_inflation <= 0.0 or not np.all(np.isfinite(derived)):
        return {
            "batch": batch,
            "variance": variance,
            "lrv": lrv,
            "raw_inflation": None,
            "formal_inflation": None,
            "estimable": False,
            "reason": "nonfinite_computation",
        }
    return {
        "batch": batch,
        "variance": variance,
        "lrv": lrv,
        "raw_inflation": raw_inflation,
        "formal_inflation": formal_inflation,
        "estimable": True,
        "reason": None,
    }


def independent_checkpoint(
    round_indices: Sequence[int] | np.ndarray,
    values: Sequence[float] | np.ndarray,
) -> Dict[str, Any]:
    indices, normalized_values = _normalize_inputs(round_indices, values)
    n = len(indices)
    if n not in CHECKPOINTS:
        raise ValueError("round count must be a frozen checkpoint")
    short_batch = isqrt(n)
    long_batch = 2 * short_batch
    short = _independent_obm(normalized_values, short_batch)
    long = _independent_obm(normalized_values, long_batch)
    result: Dict[str, Any] = {
        "n": n,
        "short_batch": short_batch,
        "long_batch": long_batch,
        "variance": (
            short["variance"]
            if short["variance"] is not None
            else long["variance"]
        ),
        "short_lrv": short["lrv"],
        "long_lrv": long["lrv"],
        "short_raw_inflation": short["raw_inflation"],
        "long_raw_inflation": long["raw_inflation"],
        "short_formal_inflation": short["formal_inflation"],
        "long_formal_inflation": long["formal_inflation"],
        "scale_ratio": None,
        "official_inflation": None,
        "official_lrv": None,
        "official_ess": None,
        "official_mcse": None,
        "short_estimable": short["estimable"],
        "long_estimable": long["estimable"],
        "short_reason": short["reason"],
        "long_reason": long["reason"],
        "adaptive_numerically_estimable": False,
        "reason": "core_not_estimable",
        "stationarity_not_assessed": True,
        "contract_version": EVIDENCE_CONTRACT_VERSION,
    }
    if not short["estimable"] or not long["estimable"]:
        return result

    short_inflation = float(short["formal_inflation"])
    long_inflation = float(long["formal_inflation"])
    scale_ratio = max(short_inflation, long_inflation) / min(
        short_inflation, long_inflation
    )
    official_inflation = max(short_inflation, long_inflation)
    variance = float(short["variance"])
    official_lrv = float(variance * official_inflation)
    official_ess = float(n / official_inflation)
    official_mcse = float(np.sqrt(official_lrv / n))
    official_values = (
        scale_ratio,
        official_inflation,
        official_lrv,
        official_ess,
        official_mcse,
    )
    if not np.all(np.isfinite(official_values)):
        result["reason"] = "nonfinite_computation"
        return result

    adaptive = scale_ratio <= SCALE_RATIO_LIMIT
    result.update({
        "scale_ratio": scale_ratio,
        "official_inflation": official_inflation,
        "official_lrv": official_lrv,
        "official_ess": official_ess,
        "official_mcse": official_mcse,
        "adaptive_numerically_estimable": adaptive,
        "reason": None if adaptive else "multiscale_disagreement",
    })
    return result


def _finite_optional(values: Sequence[float | None]) -> bool:
    present = [value for value in values if value is not None]
    return bool(not present or np.all(np.isfinite(present)))


def _checkpoint_row(
    family: AuditFamily,
    repeat_index: int,
    values: np.ndarray,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    n = int(result["n"])
    numeric_fields = (
        result["variance"],
        result["short_lrv"],
        result["long_lrv"],
        result["short_raw_inflation"],
        result["long_raw_inflation"],
        result["short_formal_inflation"],
        result["long_formal_inflation"],
        result["scale_ratio"],
        result["official_inflation"],
        result["official_lrv"],
        result["official_ess"],
        result["official_mcse"],
    )
    short_raw_ess_ratio = (
        1.0 / result["short_raw_inflation"]
        if result["short_raw_inflation"] is not None
        else None
    )
    long_raw_ess_ratio = (
        1.0 / result["long_raw_inflation"]
        if result["long_raw_inflation"] is not None
        else None
    )
    short_formal_ess_ratio = (
        1.0 / result["short_formal_inflation"]
        if result["short_formal_inflation"] is not None
        else None
    )
    long_formal_ess_ratio = (
        1.0 / result["long_formal_inflation"]
        if result["long_formal_inflation"] is not None
        else None
    )
    official_ess_ratio = (
        result["official_ess"] / n
        if result["official_ess"] is not None
        else None
    )
    formal_ess_cap_violation = bool(
        result["official_ess"] is not None
        and result["official_ess"] > n
    )
    mcse_floor_violation = bool(
        result["official_mcse"] is not None
        and result["official_mcse"] < np.sqrt(result["variance"] / n)
    )
    return {
        "family_code": family.code,
        "family": family.name,
        "repeat_index": int(repeat_index),
        "n": n,
        "sample_mean": float(np.mean(values[:n])),
        "short_batch": result["short_batch"],
        "long_batch": result["long_batch"],
        "short_raw_ess_ratio": short_raw_ess_ratio,
        "long_raw_ess_ratio": long_raw_ess_ratio,
        "short_formal_ess_ratio": short_formal_ess_ratio,
        "long_formal_ess_ratio": long_formal_ess_ratio,
        "official_ess_ratio": official_ess_ratio,
        "scale_ratio": result["scale_ratio"],
        "official_inflation": result["official_inflation"],
        "official_lrv": result["official_lrv"],
        "official_mcse": result["official_mcse"],
        "adaptive_numerically_estimable": (
            result["adaptive_numerically_estimable"]
        ),
        "reason": result["reason"],
        "short_reason": result["short_reason"],
        "long_reason": result["long_reason"],
        "nonfinite_output": not _finite_optional(numeric_fields),
        "contract_violation": False,
        "formal_ess_cap_violation": formal_ess_cap_violation,
        "mcse_floor_violation": mcse_floor_violation,
    }


def _trajectory_row(
    family: AuditFamily,
    repeat_index: int,
    values: np.ndarray,
    checkpoint_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    first_row = next(
        (
            row
            for row in checkpoint_rows
            if row["adaptive_numerically_estimable"]
        ),
        None,
    )
    if first_row is None:
        first_round = None
        resource_round_count = MAX_TRAJECTORY_LENGTH
        reason = "resource_cap_without_multiscale_evidence"
        first_coverage = None
        first_lrv_ratio = None
        first_ess_ratio = None
        first_sample_mean = None
        first_mcse = None
        first_scale_ratio = None
    else:
        first_round = int(first_row["n"])
        resource_round_count = first_round
        reason = None
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
    identity_violation = bool(
        tuple(row["n"] for row in checkpoint_rows) != CHECKPOINTS
        or first_round not in (None, *CHECKPOINTS)
    )
    return {
        "family_code": family.code,
        "family": family.name,
        "phi": family.phi,
        "repeat_index": int(repeat_index),
        "first_adaptive_numerically_estimable_round": first_round,
        "resource_round_count": resource_round_count,
        "reason": reason,
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


def _append_checkpoint_row(
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


def _checkpoint_summary(
    family: AuditFamily,
    n: int,
    accumulator: Dict[str, Any],
    *,
    repeat_count: int,
) -> Dict[str, Any]:
    return {
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


def _family_summary(
    family: AuditFamily,
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
    return {
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
        "first_ready_coverage": (
            coverage_count / ready_count if ready_count else None
        ),
        "first_ready_official_lrv_ratio_median": _median([
            row["first_ready_official_lrv_ratio"] for row in ready
        ]),
        "first_ready_official_ess_ratio_median": _median([
            row["first_ready_official_ess_ratio"] for row in ready
        ]),
        "resource_round_count_sum": int(np.sum(resources)),
        "resource_round_count_minimum": float(np.min(resources)),
        "resource_round_count_q25": float(np.quantile(resources, 0.25)),
        "resource_round_count_median": float(np.median(resources)),
        "resource_round_count_mean": float(np.mean(resources)),
        "resource_round_count_q95": float(np.quantile(resources, 0.95)),
        "resource_round_count_maximum": float(np.max(resources)),
        "first_ready_at_cap_count": sum(
            row["first_adaptive_numerically_estimable_round"]
            == MAX_TRAJECTORY_LENGTH
            for row in trajectory_rows
        ),
        "not_ready_at_cap_count": len(trajectory_rows) - ready_count,
        "trajectory_identity_violation_count": sum(
            row["trajectory_identity_violation"] for row in trajectory_rows
        ),
    }


def collect_matrix_independent(
    *,
    families: Sequence[AuditFamily],
    repeat_count: int,
    maximum_length: int,
    checkpoints: Sequence[int],
    seed_namespace: Sequence[int],
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], list[Dict[str, Any]]]:
    """Replay a matrix with audit-owned generation and mathematics."""

    if tuple(checkpoints) != CHECKPOINTS:
        raise ValueError("checkpoints must equal the frozen schedule")
    if maximum_length != MAX_TRAJECTORY_LENGTH:
        raise ValueError("maximum_length must equal the frozen cap")
    if isinstance(repeat_count, bool) or not isinstance(
        repeat_count, (int, np.integer)
    ) or repeat_count < 1:
        raise ValueError("repeat_count must be a positive integer")
    if not families or any(family not in FAMILIES for family in families):
        raise ValueError("families must contain frozen audit families")
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
            values = generate_artificial_trajectory_independent(
                family,
                repeat_index=repeat_index,
                maximum_length=maximum_length,
                seed_namespace=seed_namespace,
            )
            checkpoint_rows = []
            for n in checkpoints:
                result = independent_checkpoint(
                    round_indices[:n],
                    values[:n],
                )
                row = _checkpoint_row(
                    family,
                    repeat_index,
                    values,
                    result,
                )
                checkpoint_rows.append(row)
                _append_checkpoint_row(accumulators[n], row)
            first_row = _trajectory_row(
                family,
                repeat_index,
                values,
                checkpoint_rows,
            )
            current_family_rows.append(first_row)
            trajectory_rows.append(first_row)
        checkpoint_summaries.extend(
            _checkpoint_summary(
                family,
                n,
                accumulators[n],
                repeat_count=repeat_count,
            )
            for n in checkpoints
        )
        family_summaries.append(
            _family_summary(family, current_family_rows)
        )
    return trajectory_rows, checkpoint_summaries, family_summaries


def run_fixed_boundary_checks_independent() -> Dict[str, Any]:
    checks: Dict[str, bool] = {}
    indices = np.arange(1, 257, dtype=np.int64)
    positions = np.arange(256, dtype=np.float64)

    short = _independent_obm(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        1,
    )
    long = _independent_obm(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        2,
    )
    checks["two_hand_checked_obm_values"] = bool(
        np.isclose(short["lrv"], 5.0 / 3.0)
        and np.isclose(long["lrv"], 8.0 / 3.0)
    )
    checks["scale_ratio_exact_boundary_passes"] = bool(
        max(1.0, 1.25) / min(1.0, 1.25) <= SCALE_RATIO_LIMIT
    )
    checks["scale_ratio_nextafter_boundary_fails"] = not bool(
        max(1.0, np.nextafter(1.25, np.inf))
        / min(1.0, np.nextafter(1.25, np.inf))
        <= SCALE_RATIO_LIMIT
    )

    one_bad_values = np.where(
        (np.arange(256) // 16) % 2 == 0,
        1.0,
        -1.0,
    )
    one_bad = independent_checkpoint(indices, one_bad_values)
    checks["one_bad_scale_fails_closed"] = bool(
        one_bad["short_estimable"]
        and not one_bad["long_estimable"]
        and one_bad["reason"] == "core_not_estimable"
    )

    long_risk = independent_checkpoint(
        indices,
        np.linspace(0.0, 1.0, 256),
    )
    short_risk = independent_checkpoint(
        indices,
        np.sin(0.15 * positions),
    )
    checks["long_scale_maximum_is_official"] = bool(
        long_risk["long_formal_inflation"]
        > long_risk["short_formal_inflation"]
        and long_risk["official_inflation"]
        == long_risk["long_formal_inflation"]
    )
    checks["short_scale_maximum_is_official"] = bool(
        short_risk["short_formal_inflation"]
        > short_risk["long_formal_inflation"]
        and short_risk["official_inflation"]
        == short_risk["short_formal_inflation"]
    )

    try:
        independent_checkpoint(
            np.arange(1, 258),
            np.linspace(0.0, 1.0, 257),
        )
    except ValueError:
        checks["noncheckpoint_classifier_rejected"] = True
    else:
        checks["noncheckpoint_classifier_rejected"] = False
    scale_only = _independent_obm(
        np.linspace(0.0, 1.0, 257),
        16,
    )
    checks["noncheckpoint_scale_core_still_computes"] = bool(
        scale_only["estimable"]
    )

    first_at_cap = next(
        (n for n, decision in zip(CHECKPOINTS, [False] * 14 + [True])
         if decision),
        None,
    )
    never_ready = next(
        (n for n, decision in zip(CHECKPOINTS, [False] * 15)
         if decision),
        None,
    )
    checks["first_ready_at_cap_distinguished"] = bool(
        first_at_cap == 2048 and never_ready is None
    )

    base_values = np.sin(0.07 * positions) + 0.002 * positions
    base = independent_checkpoint(indices, base_values)
    shifted = independent_checkpoint(indices, base_values + 13.0)
    scaled = independent_checkpoint(indices, base_values * 7.0)
    checks["shift_invariance"] = bool(
        np.isclose(base["scale_ratio"], shifted["scale_ratio"], rtol=1e-10)
        and np.isclose(
            base["official_mcse"],
            shifted["official_mcse"],
            rtol=1e-10,
        )
    )
    checks["positive_scale_invariance"] = bool(
        np.isclose(base["scale_ratio"], scaled["scale_ratio"], rtol=1e-10)
        and np.isclose(
            scaled["official_mcse"],
            7.0 * base["official_mcse"],
            rtol=1e-10,
        )
    )

    constant = independent_checkpoint(indices, np.ones(256))
    checks["constant_fails_closed"] = bool(
        constant["reason"] == "core_not_estimable"
        and constant["short_reason"] == "zero_round_variance"
        and constant["long_reason"] == "zero_round_variance"
    )
    periodic = independent_checkpoint(
        indices,
        np.where(np.arange(256) % 2 == 0, 1.0, -1.0),
    )
    checks["periodic_fails_closed"] = bool(
        periodic["reason"] == "core_not_estimable"
    )
    spike_values = np.zeros(256)
    spike_values[128] = 1.0
    spike = independent_checkpoint(indices, spike_values)
    checks["single_spike_finite_and_capped"] = bool(
        spike["official_ess"] is not None
        and spike["official_ess"] <= 256
        and np.all(np.isfinite([
            spike["official_lrv"],
            spike["official_ess"],
            spike["official_mcse"],
        ]))
    )
    trend = independent_checkpoint(
        indices,
        np.linspace(0.0, 1.0, 256),
    )
    checks["trend_has_no_decision_fields"] = bool(
        trend["stationarity_not_assessed"]
        and FORBIDDEN_DECISION_FIELDS.isdisjoint(trend)
    )

    invalid_inputs = (
        (np.concatenate([np.arange(1, 128), np.arange(129, 258)]), positions),
        (indices, np.resize([False, True], 256)),
        (indices, np.full(256, np.nan)),
    )
    invalid_rejections = []
    for invalid_indices, invalid_values in invalid_inputs:
        try:
            independent_checkpoint(invalid_indices, invalid_values)
        except ValueError:
            invalid_rejections.append(True)
        else:
            invalid_rejections.append(False)
    checks["invalid_inputs_rejected"] = all(invalid_rejections)
    result = {"passed": all(checks.values()), "checks": checks}
    _strict_json_bytes(result)
    return result


def build_acceptance_gates_independent(
    trajectory_rows: Sequence[Dict[str, Any]],
    checkpoint_summaries: Sequence[Dict[str, Any]],
    family_summaries: Sequence[Dict[str, Any]],
    boundary_checks: Dict[str, Any],
) -> Dict[str, Any]:
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
        actual_summary = _family_summary(
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


def scientific_payload_independent(
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


def _assert_exact(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{label} does not match the frozen audit contract")


def _resolve_sibling_manifest(
    report_file: Path,
    recorded_manifest_path: Any,
) -> Path:
    """Resolve the bound sibling while accepting the legacy absolute metadata."""

    if not isinstance(recorded_manifest_path, str):
        raise ValueError("report manifest path must be a string")
    recorded = Path(recorded_manifest_path)
    if recorded.is_absolute():
        if recorded.name != "protocol_manifest.json":
            raise ValueError(
                "legacy absolute manifest path must name protocol_manifest.json"
            )
    elif recorded.parts != ("protocol_manifest.json",):
        raise ValueError(
            "report manifest path must be the portable sibling filename"
        )
    manifest_file = (report_file.parent / "protocol_manifest.json").resolve(
        strict=True
    )
    if manifest_file.parent != report_file.parent:
        raise ValueError("resolved manifest must remain a sibling artifact")
    return manifest_file


def _validate_report_and_manifest(
    report_path: Path,
) -> tuple[Dict[str, Any], Dict[str, Any], Path, Path]:
    report_file = report_path.resolve(strict=True)
    if not report_file.is_file():
        raise ValueError("report_path must identify a regular file")
    report = _load_json_strict(report_file)
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")

    expected_report_keys = {
        "report_format",
        "contract_version",
        "status",
        "manifest_path",
        "manifest_sha256",
        "protocol_sha256",
        "git_commit",
        "protocol",
        "execution",
        "scientific_payload",
        "scientific_result_sha256",
        "audit_required_before_interpretation",
    }
    _assert_exact("report schema", set(report), expected_report_keys)
    _assert_exact("report format", report["report_format"], REPORT_FORMAT)
    _assert_exact(
        "report contract version",
        report["contract_version"],
        ARTIFICIAL_PROTOCOL_VERSION,
    )
    _assert_exact(
        "audit requirement",
        report["audit_required_before_interpretation"],
        True,
    )

    protocol = expected_protocol()
    protocol_sha256 = _sha256_json(protocol)
    _assert_exact("report protocol", report["protocol"], protocol)
    _assert_exact(
        "report protocol SHA-256",
        report["protocol_sha256"],
        protocol_sha256,
    )
    _assert_exact(
        "report scientific SHA-256",
        report["scientific_result_sha256"],
        _sha256_json(report["scientific_payload"]),
    )

    manifest_file = _resolve_sibling_manifest(
        report_file,
        report["manifest_path"],
    )
    _assert_exact(
        "report manifest SHA-256",
        report["manifest_sha256"],
        _sha256_file(manifest_file),
    )
    manifest = _load_json_strict(manifest_file)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a JSON object")
    expected_manifest_keys = {
        "contract_version",
        "created_at_utc",
        "git_commit",
        "git_worktree_clean_including_untracked",
        "protocol_sha256",
        "source_sha256",
        "environment",
        "protocol",
    }
    _assert_exact("manifest schema", set(manifest), expected_manifest_keys)
    _assert_exact(
        "manifest contract version",
        manifest["contract_version"],
        ARTIFICIAL_PROTOCOL_VERSION,
    )
    _assert_exact("manifest protocol", manifest["protocol"], protocol)
    _assert_exact(
        "manifest protocol SHA-256",
        manifest["protocol_sha256"],
        protocol_sha256,
    )
    _assert_exact(
        "manifest clean-worktree precondition",
        manifest["git_worktree_clean_including_untracked"],
        True,
    )
    _assert_exact(
        "report/manifest commit",
        report["git_commit"],
        manifest["git_commit"],
    )

    root = _repo_root()
    current_commit = _git_text(root, "rev-parse", "HEAD")
    _assert_exact("current Git commit", current_commit, manifest["git_commit"])
    expected_source_keys = {str(path) for path in SOURCE_PATHS}
    source_hashes = manifest["source_sha256"]
    if not isinstance(source_hashes, dict):
        raise ValueError("manifest source_sha256 must be an object")
    _assert_exact("manifest source set", set(source_hashes), expected_source_keys)
    for path in SOURCE_PATHS:
        source = root / path
        if not source.is_file():
            raise ValueError(f"bound source is missing: {path}")
        _assert_exact(
            f"current source SHA-256 for {path}",
            _sha256_file(source),
            source_hashes[str(path)],
        )

    expected_execution = {
        "family_count": len(FAMILIES),
        "trajectory_count": len(FAMILIES) * REPEAT_COUNT,
        "checkpoint_classification_count": (
            len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS)
        ),
        "scale_evaluation_count": (
            len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS) * 2
        ),
        "real_data_accessed": False,
        "generator_run": False,
        "privacy_budget_consumed": False,
    }
    execution = report["execution"]
    if not isinstance(execution, dict):
        raise ValueError("execution must be an object")
    _assert_exact(
        "execution schema",
        set(execution),
        {"elapsed_sec", *expected_execution},
    )
    for key, expected in expected_execution.items():
        _assert_exact(f"execution.{key}", execution[key], expected)
    elapsed = execution["elapsed_sec"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not np.isfinite(elapsed)
        or elapsed < 0.0
    ):
        raise ValueError("execution.elapsed_sec must be finite and nonnegative")

    payload = report["scientific_payload"]
    if not isinstance(payload, dict):
        raise ValueError("scientific_payload must be an object")
    _assert_exact(
        "scientific payload schema",
        set(payload),
        {
            "trajectory_first_ready_rows",
            "checkpoint_summaries",
            "family_summaries",
            "boundary_checks",
            "acceptance",
        },
    )
    _assert_exact(
        "trajectory row count",
        len(payload["trajectory_first_ready_rows"]),
        len(FAMILIES) * REPEAT_COUNT,
    )
    _assert_exact(
        "checkpoint summary count",
        len(payload["checkpoint_summaries"]),
        len(FAMILIES) * len(CHECKPOINTS),
    )
    _assert_exact(
        "family summary count",
        len(payload["family_summaries"]),
        len(FAMILIES),
    )
    _assert_exact(
        "report status/acceptance status",
        report["status"],
        payload["acceptance"]["status"],
    )
    return report, manifest, report_file, manifest_file


def _mismatch_paths(
    actual: Any,
    expected: Any,
    *,
    path: str = "$",
    limit: int = 100,
) -> list[str]:
    mismatches: list[str] = []

    def visit(left: Any, right: Any, current: str) -> None:
        if len(mismatches) >= limit:
            return
        if type(left) is not type(right):
            mismatches.append(current)
            return
        if isinstance(left, dict):
            if set(left) != set(right):
                mismatches.append(current)
                return
            for key in sorted(left):
                visit(left[key], right[key], f"{current}.{key}")
            return
        if isinstance(left, list):
            if len(left) != len(right):
                mismatches.append(current)
                return
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{current}[{index}]")
            return
        if left != right:
            mismatches.append(current)

    visit(actual, expected, path)
    return mismatches


def audit_artificial_report(
    report_path: Path,
    audit_path: Path,
) -> tuple[Path, Dict[str, Any]]:
    """Fully replay a formal report and write a non-overwriting audit."""

    audit_file = Path(audit_path).resolve()
    if audit_file.exists():
        raise FileExistsError(f"audit output already exists: {audit_file}")
    report, manifest, report_file, manifest_file = (
        _validate_report_and_manifest(Path(report_path))
    )
    if audit_file in {report_file, manifest_file}:
        raise ValueError("audit output must not overwrite an input artifact")
    if not audit_file.parent.is_dir():
        raise ValueError("audit output parent directory must already exist")

    boundary_checks = run_fixed_boundary_checks_independent()
    trajectory_rows, checkpoint_summaries, family_summaries = (
        collect_matrix_independent(
            families=FAMILIES,
            repeat_count=REPEAT_COUNT,
            maximum_length=MAX_TRAJECTORY_LENGTH,
            checkpoints=CHECKPOINTS,
            seed_namespace=SEED_NAMESPACE,
        )
    )
    acceptance = build_acceptance_gates_independent(
        trajectory_rows,
        checkpoint_summaries,
        family_summaries,
        boundary_checks,
    )
    recomputed_payload = scientific_payload_independent(
        trajectory_rows=trajectory_rows,
        checkpoint_summaries=checkpoint_summaries,
        family_summaries=family_summaries,
        boundary_checks=boundary_checks,
        acceptance=acceptance,
    )
    recorded_payload = report["scientific_payload"]
    mismatches = _mismatch_paths(recorded_payload, recomputed_payload)
    checks = {
        "independent_boundary_checks_pass": boundary_checks["passed"],
        "scientific_payload_exact": not mismatches,
        "scientific_sha256_exact": (
            report["scientific_result_sha256"]
            == _sha256_json(recomputed_payload)
        ),
        "acceptance_status_exact": report["status"] == acceptance["status"],
    }
    passed = all(checks.values())
    audit = {
        "audit_format": AUDIT_FORMAT,
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if passed else "failed",
        "passed": passed,
        "report_path": report_file.name,
        "report_sha256": _sha256_file(report_file),
        "manifest_path": manifest_file.name,
        "manifest_sha256": _sha256_file(manifest_file),
        "git_commit": manifest["git_commit"],
        "protocol_sha256": manifest["protocol_sha256"],
        "recorded_scientific_result_sha256": (
            report["scientific_result_sha256"]
        ),
        "recomputed_scientific_result_sha256": (
            _sha256_json(recomputed_payload)
        ),
        "checks": checks,
        "mismatch_count_at_least": len(mismatches),
        "first_mismatch_paths": mismatches,
        "replay_counts": {
            "family_count": len(FAMILIES),
            "trajectory_count": len(trajectory_rows),
            "checkpoint_classification_count": (
                len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS)
            ),
            "scale_evaluation_count": (
                len(FAMILIES) * REPEAT_COUNT * len(CHECKPOINTS) * 2
            ),
        },
        "independence": {
            "imports_formal_runner": False,
            "imports_project_v2_core": False,
            "imports_project_v2b_core": False,
            "regenerates_artificial_trajectories": True,
            "recomputes_obm_directly": True,
        },
    }
    _strict_json_bytes(audit)
    _write_json_exclusive(audit_file, audit)
    return audit_file, audit


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output, audit = audit_artificial_report(
        Path(args.report),
        Path(args.output),
    )
    print("\n===== Issue #53 V2b Independent Audit =====")
    print(f"passed={audit['passed']}")
    print(f"status={audit['status']}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
