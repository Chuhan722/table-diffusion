#!/usr/bin/env python
"""Run the preregistered artificial validation for Issue #53 V2.

The formal entry point has no scientific command-line knobs.  It generates
only stationary artificial trajectories with known long-run variance and
calls :func:`compute_v2_effective_round_evidence` directly.  It never reads a
project dataset, a saved trajectory, or a generator output.
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

from table_diffevo.effective_evidence import (
    V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION,
    compute_v2_effective_round_evidence,
)


ARTIFICIAL_PROTOCOL_VERSION = (
    "issue53-v2-effective-evidence-artificial-validation-v1"
)
SEED_NAMESPACE = (53, 2)
REPEAT_COUNT = 2000
MAX_TRAJECTORY_LENGTH = 4096
OBSERVATION_LENGTHS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
CONFIDENCE_MULTIPLIER = 1.96
COVERAGE_LOWER = 0.925
COVERAGE_UPPER = 0.975
LONG_RUN_VARIANCE_RATIO_LOWER = 0.80
LONG_RUN_VARIANCE_RATIO_UPPER = 1.25

PROTOCOL_DOCUMENT = Path(
    "docs/设计/Issue53_V2人工轨迹验收协议.md"
)
DESIGN_DOCUMENT = Path(
    "docs/设计/Issue53_V2有效证据计数器设计稿.md"
)
CORE_MODULE = Path("src/table_diffevo/effective_evidence.py")
RUNNER_MODULE = Path("scripts/validate_issue53_v2_effective_evidence.py")
TEST_MODULE = Path("tests/test_effective_evidence.py")
RUNNER_TEST_MODULE = Path(
    "tests/test_issue53_v2_effective_evidence_artificial.py"
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
    """One stationary Gaussian AR(1) family with known evidence ratio."""

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
    ArtificialFamily(0, "iid", 0.0, "positive_history_selection"),
    ArtificialFamily(1, "ar1_phi_0p5", 0.5, "positive_history_selection"),
    ArtificialFamily(2, "ar1_phi_0p8", 0.8, "positive_history_selection"),
    ArtificialFamily(3, "ar1_phi_m0p5", -0.5, "negative_control"),
)
POSITIVE_SELECTION_FAMILY_NAMES = tuple(
    family.name
    for family in FAMILIES
    if family.role == "positive_history_selection"
)
NEGATIVE_CONTROL_FAMILY_NAME = "ar1_phi_m0p5"


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
    """Return every scientific choice used by the artificial run."""

    protocol = {
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "evidence_contract_version": (
            V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION
        ),
        "scope": {
            "scalar_trajectory_only": True,
            "reads_project_dataset": False,
            "reads_saved_real_trajectory": False,
            "runs_generator": False,
            "uses_gpu": False,
            "consumes_privacy_budget": False,
            "stationarity_decision_present": False,
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
            "observation_prefix_lengths": list(OBSERVATION_LENGTHS),
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
        "interval": {
            "form": "sample_mean_plus_or_minus_multiplier_times_mcse",
            "multiplier": CONFIDENCE_MULTIPLIER,
            "mcse_is_official_conservative_value": True,
            "true_mean": 0.0,
        },
        "positive_family_acceptance": {
            "coverage_inclusive_lower": COVERAGE_LOWER,
            "coverage_inclusive_upper": COVERAGE_UPPER,
            "median_long_run_variance_ratio_inclusive_lower": (
                LONG_RUN_VARIANCE_RATIO_LOWER
            ),
            "median_long_run_variance_ratio_inclusive_upper": (
                LONG_RUN_VARIANCE_RATIO_UPPER
            ),
            "median_formal_ess_ratio_strict_order": list(
                POSITIVE_SELECTION_FAMILY_NAMES
            ),
            "numerical_failure_count_must_equal": 0,
            "nonfinite_output_count_must_equal": 0,
            "contract_violation_count_must_equal": 0,
        },
        "negative_control_acceptance": {
            "median_raw_ess_ratio_strictly_greater_than": 1.0,
            "every_formal_ess_ratio_at_most": 1.0,
            "numerical_failure_count_must_equal": 0,
            "nonfinite_output_count_must_equal": 0,
            "contract_violation_count_must_equal": 0,
        },
        "minimum_history_selection": {
            "rule": (
                "smallest_length_passing_itself_and_every_larger_length"
            ),
            "no_passing_length_action": "candidate_failed",
            "selected_value_is_a_stopping_round": False,
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
        "evidence_evaluation_count": (
            len(FAMILIES) * REPEAT_COUNT * len(OBSERVATION_LENGTHS)
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


def generate_family_trajectories(
    family: ArtificialFamily,
    *,
    repeat_count: int,
    maximum_length: int,
) -> np.ndarray:
    """Generate stationary trajectories with one independent seed per row."""

    if family not in FAMILIES:
        raise ValueError("family must be one of the frozen protocol families")
    if isinstance(repeat_count, bool) or not isinstance(
        repeat_count, (int, np.integer)
    ) or repeat_count <= 0:
        raise ValueError("repeat_count must be a positive integer")
    if isinstance(maximum_length, bool) or not isinstance(
        maximum_length, (int, np.integer)
    ) or maximum_length < 2:
        raise ValueError("maximum_length must be an integer of at least two")

    trajectories = np.empty(
        (int(repeat_count), int(maximum_length)),
        dtype=np.float64,
    )
    for repeat_index in range(int(repeat_count)):
        seed_sequence = np.random.SeedSequence([
            *SEED_NAMESPACE,
            family.code,
            repeat_index,
        ])
        rng = np.random.Generator(np.random.PCG64(seed_sequence))
        trajectories[repeat_index] = rng.standard_normal(maximum_length)

    if family.phi != 0.0:
        innovation_scale = float(np.sqrt(1.0 - family.phi**2))
        for position in range(1, int(maximum_length)):
            trajectories[:, position] = (
                family.phi * trajectories[:, position - 1]
                + innovation_scale * trajectories[:, position]
            )
    return trajectories


def _new_accumulator() -> Dict[str, Any]:
    return {
        "coverage_count": 0,
        "numerical_failure_count": 0,
        "nonfinite_output_count": 0,
        "contract_violation_count": 0,
        "formal_ess_cap_violation_count": 0,
        "long_run_variance_ratios": [],
        "raw_ess_ratios": [],
        "formal_ess_ratios": [],
    }


def _collect_family(
    family: ArtificialFamily,
) -> list[Dict[str, Any]]:
    trajectories = generate_family_trajectories(
        family,
        repeat_count=REPEAT_COUNT,
        maximum_length=MAX_TRAJECTORY_LENGTH,
    )
    round_indices = {
        length: np.arange(1, length + 1, dtype=np.int64)
        for length in OBSERVATION_LENGTHS
    }
    accumulators = {
        length: _new_accumulator() for length in OBSERVATION_LENGTHS
    }

    for repeat_index in range(REPEAT_COUNT):
        trajectory = trajectories[repeat_index]
        for length in OBSERVATION_LENGTHS:
            values = trajectory[:length]
            result = compute_v2_effective_round_evidence(
                round_indices[length],
                values,
            )
            accumulator = accumulators[length]
            if not result.numerically_estimable:
                accumulator["numerical_failure_count"] += 1
                continue

            expected_batch_round_count = isqrt(length)
            if (
                result.actual_round_count != length
                or result.batch_round_count != expected_batch_round_count
                or result.overlapping_batch_count
                != length - expected_batch_round_count + 1
                or result.reason is not None
                or result.stationarity_not_assessed is not True
                or result.contract_version
                != V2_EFFECTIVE_ROUND_EVIDENCE_RESEARCH_CONTRACT_VERSION
            ):
                accumulator["contract_violation_count"] += 1
                continue

            numeric_values = (
                result.single_round_variance,
                result.long_run_variance,
                result.raw_correlation_inflation,
                result.conservative_correlation_inflation,
                result.raw_effective_round_count,
                result.effective_round_count,
                result.mcse,
            )
            if any(value is None for value in numeric_values) or not np.all(
                np.isfinite(numeric_values)
            ):
                accumulator["nonfinite_output_count"] += 1
                continue

            if result.effective_round_count > length:
                accumulator["formal_ess_cap_violation_count"] += 1
            sample_mean = float(np.mean(values))
            if abs(sample_mean) <= CONFIDENCE_MULTIPLIER * result.mcse:
                accumulator["coverage_count"] += 1
            accumulator["long_run_variance_ratios"].append(
                result.long_run_variance
                / family.theoretical_long_run_variance
            )
            accumulator["raw_ess_ratios"].append(
                result.raw_effective_round_count / length
            )
            accumulator["formal_ess_ratios"].append(
                result.effective_round_count / length
            )

    return [
        _summarize_cell(family, length, accumulators[length])
        for length in OBSERVATION_LENGTHS
    ]


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _summarize_cell(
    family: ArtificialFamily,
    length: int,
    accumulator: Dict[str, Any],
) -> Dict[str, Any]:
    lrv_ratio_median = _median(
        accumulator["long_run_variance_ratios"]
    )
    raw_ess_ratio_median = _median(accumulator["raw_ess_ratios"])
    formal_ess_ratio_median = _median(accumulator["formal_ess_ratios"])
    coverage = accumulator["coverage_count"] / REPEAT_COUNT
    common_numerical_pass = (
        accumulator["numerical_failure_count"] == 0
        and accumulator["nonfinite_output_count"] == 0
        and accumulator["contract_violation_count"] == 0
        and accumulator["formal_ess_cap_violation_count"] == 0
    )

    if family.role == "positive_history_selection":
        coverage_pass: bool | None = (
            COVERAGE_LOWER <= coverage <= COVERAGE_UPPER
        )
        lrv_ratio_pass: bool | None = (
            lrv_ratio_median is not None
            and LONG_RUN_VARIANCE_RATIO_LOWER
            <= lrv_ratio_median
            <= LONG_RUN_VARIANCE_RATIO_UPPER
        )
        individual_acceptance_pass: bool | None = (
            common_numerical_pass and coverage_pass and lrv_ratio_pass
        )
        negative_raw_ess_pass: bool | None = None
    else:
        coverage_pass = None
        lrv_ratio_pass = None
        individual_acceptance_pass = None
        negative_raw_ess_pass = (
            common_numerical_pass
            and raw_ess_ratio_median is not None
            and raw_ess_ratio_median > 1.0
        )

    summary = {
        "family_code": family.code,
        "family": family.name,
        "role": family.role,
        "phi": family.phi,
        "length": length,
        "repeat_count": REPEAT_COUNT,
        "theoretical_long_run_variance": (
            family.theoretical_long_run_variance
        ),
        "theoretical_raw_ess_ratio": family.theoretical_raw_ess_ratio,
        "coverage_count": accumulator["coverage_count"],
        "coverage": coverage,
        "long_run_variance_ratio_median": lrv_ratio_median,
        "raw_ess_ratio_median": raw_ess_ratio_median,
        "formal_ess_ratio_median": formal_ess_ratio_median,
        "formal_ess_ratio_maximum": (
            float(max(accumulator["formal_ess_ratios"]))
            if accumulator["formal_ess_ratios"]
            else None
        ),
        "numerical_failure_count": (
            accumulator["numerical_failure_count"]
        ),
        "nonfinite_output_count": accumulator["nonfinite_output_count"],
        "contract_violation_count": accumulator[
            "contract_violation_count"
        ],
        "formal_ess_cap_violation_count": (
            accumulator["formal_ess_cap_violation_count"]
        ),
        "coverage_pass": coverage_pass,
        "long_run_variance_ratio_pass": lrv_ratio_pass,
        "individual_acceptance_pass": individual_acceptance_pass,
        "negative_raw_ess_pass": negative_raw_ess_pass,
    }
    _strict_json_bytes(summary)
    return summary


def build_length_decisions(
    cell_summaries: Sequence[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    by_identity = {
        (cell["family"], cell["length"]): cell for cell in cell_summaries
    }
    expected_count = len(FAMILIES) * len(OBSERVATION_LENGTHS)
    if len(by_identity) != expected_count:
        raise ValueError("cell summaries do not match the frozen matrix")

    decisions = []
    for length in OBSERVATION_LENGTHS:
        positive_cells = [
            by_identity[(family_name, length)]
            for family_name in POSITIVE_SELECTION_FAMILY_NAMES
        ]
        ess_medians = [
            cell["formal_ess_ratio_median"] for cell in positive_cells
        ]
        ordering_pass = (
            all(value is not None for value in ess_medians)
            and ess_medians[0] > ess_medians[1] > ess_medians[2]
        )
        positive_pass = (
            all(cell["individual_acceptance_pass"] for cell in positive_cells)
            and ordering_pass
        )
        negative_cell = by_identity[(NEGATIVE_CONTROL_FAMILY_NAME, length)]
        negative_pass = bool(negative_cell["negative_raw_ess_pass"])
        decisions.append({
            "length": length,
            "positive_individual_cells_pass": all(
                cell["individual_acceptance_pass"]
                for cell in positive_cells
            ),
            "positive_ess_ordering_pass": ordering_pass,
            "positive_history_acceptance_pass": positive_pass,
            "negative_control_pass": negative_pass,
        })
    return decisions


def select_minimum_history(
    length_decisions: Sequence[Dict[str, Any]],
) -> int | None:
    """Select the first positive-family pass with no later regression."""

    if [decision.get("length") for decision in length_decisions] != list(
        OBSERVATION_LENGTHS
    ):
        raise ValueError("length decisions must use the frozen ordered grid")
    for index, decision in enumerate(length_decisions):
        if all(
            later["positive_history_acceptance_pass"]
            for later in length_decisions[index:]
        ):
            return int(decision["length"])
    return None


def run_fixed_boundary_checks() -> Dict[str, Any]:
    """Re-run deterministic fail-closed checks inside the formal entry."""

    checks: Dict[str, bool] = {}

    try:
        compute_v2_effective_round_evidence([1, 3], [0.0, 1.0])
    except ValueError:
        checks["round_gap_rejected"] = True
    else:
        checks["round_gap_rejected"] = False

    try:
        compute_v2_effective_round_evidence([1, 1], [0.0, 1.0])
    except ValueError:
        checks["duplicate_round_rejected"] = True
    else:
        checks["duplicate_round_rejected"] = False

    try:
        compute_v2_effective_round_evidence([2, 1], [0.0, 1.0])
    except ValueError:
        checks["reordered_round_rejected"] = True
    else:
        checks["reordered_round_rejected"] = False

    try:
        compute_v2_effective_round_evidence([1, 2], [0.0, np.nan])
    except ValueError:
        checks["nonfinite_value_rejected"] = True
    else:
        checks["nonfinite_value_rejected"] = False

    try:
        compute_v2_effective_round_evidence([1, 2], [False, True])
    except ValueError:
        checks["boolean_value_rejected"] = True
    else:
        checks["boolean_value_rejected"] = False

    constant = compute_v2_effective_round_evidence(
        np.arange(1, 33),
        np.ones(32),
    )
    checks["constant_fails_closed"] = (
        not constant.numerically_estimable
        and constant.reason == "zero_round_variance"
    )

    periodic_values = np.where(np.arange(256) % 2 == 0, 1.0, -1.0)
    periodic = compute_v2_effective_round_evidence(
        np.arange(1, 257),
        periodic_values,
    )
    checks["periodic_batch_coupling_fails_closed"] = (
        not periodic.numerically_estimable
        and periodic.reason == "degenerate_long_run_variance"
    )

    positions = np.arange(64, dtype=np.float64)
    base_values = np.sin(0.37 * positions) + 0.01 * positions
    indices = np.arange(101, 165)
    base = compute_v2_effective_round_evidence(indices, base_values)
    shifted = compute_v2_effective_round_evidence(indices, base_values + 13.0)
    scaled = compute_v2_effective_round_evidence(indices, base_values * 7.0)
    checks["shift_invariance"] = bool(
        base.numerically_estimable
        and shifted.numerically_estimable
        and np.isclose(
            shifted.effective_round_count,
            base.effective_round_count,
            rtol=1e-10,
            atol=1e-10,
        )
        and np.isclose(
            shifted.mcse,
            base.mcse,
            rtol=1e-10,
            atol=1e-10,
        )
    )
    checks["positive_scale_invariance"] = bool(
        scaled.numerically_estimable
        and np.isclose(
            scaled.effective_round_count,
            base.effective_round_count,
            rtol=1e-10,
            atol=1e-10,
        )
        and np.isclose(
            scaled.mcse,
            7.0 * base.mcse,
            rtol=1e-10,
            atol=1e-10,
        )
    )

    spike_values = np.zeros(256)
    spike_values[128] = 1.0
    spike = compute_v2_effective_round_evidence(
        np.arange(1, 257),
        spike_values,
    )
    checks["single_spike_is_finite_and_capped"] = bool(
        spike.numerically_estimable
        and spike.effective_round_count <= spike.actual_round_count
        and np.all(np.isfinite([
            spike.long_run_variance,
            spike.raw_effective_round_count,
            spike.effective_round_count,
            spike.mcse,
        ]))
    )

    trend = compute_v2_effective_round_evidence(
        np.arange(1, 65),
        np.linspace(0.0, 1.0, 64),
    )
    checks["trend_has_no_decision_fields"] = bool(
        trend.stationarity_not_assessed
        and FORBIDDEN_DECISION_FIELDS.isdisjoint(asdict(trend))
    )

    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "deferred_until_history_selection": [
            "less_than_selected_history_returns_insufficient_history"
        ],
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
            "artificial protocol requires a clean worktree before any draws"
        )
    source_paths = (
        PROTOCOL_DOCUMENT,
        DESIGN_DOCUMENT,
        CORE_MODULE,
        RUNNER_MODULE,
        TEST_MODULE,
        RUNNER_TEST_MODULE,
    )
    missing = [str(path) for path in source_paths if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"protocol source files are missing: {missing}")

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
        },
        "protocol": protocol,
    }
    _strict_json_bytes(manifest)
    return manifest


def run_artificial_protocol(output_dir: Path) -> tuple[Path, Path, Dict[str, Any]]:
    """Execute exactly the frozen matrix and write non-overwriting results."""

    if set(inspect.signature(run_artificial_protocol).parameters) != {
        "output_dir"
    }:
        raise RuntimeError("formal runner gained an unexpected override")
    root = _repo_root()
    manifest = build_execution_manifest(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "protocol_manifest.json"
    report_path = output_dir / "artificial_evidence_report.json"
    _write_json_exclusive(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)

    started = time.perf_counter()
    boundary_checks = run_fixed_boundary_checks()
    cell_summaries = []
    for family in FAMILIES:
        cell_summaries.extend(_collect_family(family))
    length_decisions = build_length_decisions(cell_summaries)
    positive_history_candidate = select_minimum_history(length_decisions)
    negative_control_all_lengths_pass = all(
        decision["negative_control_pass"] for decision in length_decisions
    )

    if not boundary_checks["passed"]:
        candidate_status = "candidate_failed_fixed_boundary_checks"
    elif positive_history_candidate is None:
        candidate_status = "candidate_failed_no_supported_history_floor"
    elif not negative_control_all_lengths_pass:
        candidate_status = "candidate_failed_negative_control"
    else:
        candidate_status = "candidate_supported"
    selected_minimum_history = (
        positive_history_candidate
        if candidate_status == "candidate_supported"
        else None
    )

    scientific_result = {
        "contract_version": ARTIFICIAL_PROTOCOL_VERSION,
        "protocol_sha256": manifest["protocol_sha256"],
        "boundary_checks": boundary_checks,
        "cell_summaries": cell_summaries,
        "length_decisions": length_decisions,
        "selection": {
            "candidate_status": candidate_status,
            "positive_history_candidate": positive_history_candidate,
            "negative_control_all_lengths_pass": (
                negative_control_all_lengths_pass
            ),
            "minimum_history_round_count": selected_minimum_history,
            "minimum_history_is_a_stopping_round": False,
            "stationarity_assessed": False,
            "real_data_accessed": False,
        },
    }
    report = {
        **scientific_result,
        "manifest_sha256": manifest_sha256,
        "scientific_result_sha256": _sha256_json(scientific_result),
        "protocol_execution_complete": True,
        "protocol_execution_valid": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json_exclusive(report_path, report)
    return manifest_path, report_path, report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "plan":
        print(json.dumps(build_plan(), ensure_ascii=False, indent=2))
        return
    manifest_path, report_path, report = run_artificial_protocol(
        arguments.output_dir
    )
    print(json.dumps({
        "manifest_path": str(manifest_path),
        "report_path": str(report_path),
        "candidate_status": report["selection"]["candidate_status"],
        "minimum_history_round_count": report["selection"][
            "minimum_history_round_count"
        ],
        "scientific_result_sha256": report["scientific_result_sha256"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
