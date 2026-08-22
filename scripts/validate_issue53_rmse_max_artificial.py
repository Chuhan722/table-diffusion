#!/usr/bin/env python
"""Validate the frozen Issue #53 exact-count RMSE+max fit target.

The ``plan`` command is read-only and does not instantiate any formal seed.
The ``run`` command has no scientific overrides and is reserved for a later,
explicitly authorized execution.  It runs only the three frozen artificial
binary workloads and never reads a project dataset or privacy parameter.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from table_diffevo.evolution import run_evolution
from table_diffevo.generator import init_synthetic_table
from table_diffevo.inner_fit_target import (
    QueryFitAssessment,
    QueryFitThresholds,
    assess_query_fit,
)
from table_diffevo.metrics import compute_normalized_l1
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema
from table_diffevo.stationarity import (
    StationarityTrace,
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)

CONTRACT_VERSION = "issue53-rmse-max-artificial-v1"
REPORT_FORMAT = "issue53_rmse_max_artificial_report_v1"
EXPECTED_PROTOCOL_DOCUMENT_SHA256 = (
    "012d2507f7b3a79a7fb566047a7f2dae3dbf2e9d77e30dbeb5da44bcfbff6245"
)
RESOURCE_WORK_LIMIT = 20.0
MINIMUM_TAIL_WORK = 10.0
RHOS = (1.0, 0.25)
ROUNDS_BY_RHO = {1.0: 40, 0.25: 160}

PROTOCOL_DOCUMENT = Path("docs/设计/Issue53_RMSEMax全新人工验证协议.md")
DESIGN_DOCUMENT = Path("docs/设计/Issue53_无噪声生成内层停止与best输出契约设计稿.md")
FIT_TARGET_MODULE = Path("src/table_diffevo/inner_fit_target.py")
EVOLUTION_MODULE = Path("src/table_diffevo/evolution.py")
GENERATOR_MODULE = Path("src/table_diffevo/generator.py")
QUERY_MODULE = Path("src/table_diffevo/queries.py")
STATIONARITY_MODULE = Path("src/table_diffevo/stationarity.py")
RUNNER_MODULE = Path("scripts/validate_issue53_rmse_max_artificial.py")
RUNNER_TEST_MODULE = Path("tests/test_issue53_rmse_max_artificial.py")
SOURCE_PATHS = (
    PROTOCOL_DOCUMENT,
    DESIGN_DOCUMENT,
    FIT_TARGET_MODULE,
    EVOLUTION_MODULE,
    GENERATOR_MODULE,
    QUERY_MODULE,
    STATIONARITY_MODULE,
    RUNNER_MODULE,
    RUNNER_TEST_MODULE,
)


@dataclass(frozen=True)
class ArtificialFamily:
    """One frozen feasible binary table workload."""

    name: str
    attributes: tuple[str, ...]
    n_records: int
    zero_row_count: int
    one_row_count: int
    query_attribute_groups: tuple[tuple[str, ...], ...]
    seeds: tuple[int, int]

    @property
    def query_count(self) -> int:
        return len(self.query_attribute_groups)

    @property
    def target(self) -> tuple[float, ...]:
        return (float(self.one_row_count),) * self.query_count


FAMILIES = (
    ArtificialFamily(
        name="marginal_skew",
        attributes=("a", "b", "c"),
        n_records=24,
        zero_row_count=18,
        one_row_count=6,
        query_attribute_groups=(("a",), ("b",), ("c",)),
        seeds=(20260901, 20260902),
    ),
    ArtificialFamily(
        name="ring_pair",
        attributes=("a", "b", "c", "d", "e"),
        n_records=32,
        zero_row_count=16,
        one_row_count=16,
        query_attribute_groups=(
            ("a",),
            ("b",),
            ("c",),
            ("d",),
            ("e",),
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "a"),
        ),
        seeds=(20260911, 20260912),
    ),
    ArtificialFamily(
        name="nested_overlap",
        attributes=("a", "b", "c", "d", "e", "f"),
        n_records=64,
        zero_row_count=32,
        one_row_count=32,
        query_attribute_groups=(
            ("a",),
            ("b",),
            ("c",),
            ("d",),
            ("e",),
            ("f",),
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "f"),
            ("a", "b", "c"),
            ("a", "b", "c", "d"),
            ("a", "b", "c", "d", "e"),
            ("a", "b", "c", "d", "e", "f"),
        ),
        seeds=(20260921, 20260922),
    ),
)


@dataclass(frozen=True)
class MatrixCase:
    """One formal family/seed/rho trajectory identity."""

    family: str
    seed: int
    rho: float
    n_rounds: int

    @property
    def identity(self) -> tuple[str, int, float]:
        return self.family, self.seed, self.rho

    @property
    def rho_label(self) -> str:
        return "1" if self.rho == 1.0 else "0p25"


@dataclass
class ArtificialProblem:
    """Materialized public artificial inputs and preflight-only reference."""

    family: ArtificialFamily
    schema: Schema
    queries: list[dict[str, Any]]
    target: np.ndarray
    reference_table: pd.DataFrame


@dataclass(frozen=True)
class QueryFitReplayState:
    """The complete selector-visible projection for one current table."""

    state_index: int
    round_index: int
    count_errors: tuple[float, ...]
    cumulative_participating_rows: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.state_index, "state_index"),
            (self.round_index, "round_index"),
            (
                self.cumulative_participating_rows,
                "cumulative_participating_rows",
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.count_errors, tuple) or not self.count_errors:
            raise ValueError("count_errors must be a nonempty tuple")
        for value in self.count_errors:
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError("count_errors must contain finite real values")


@dataclass(frozen=True)
class QueryFitReplayDecision:
    """Frozen quality/cap decision derived only from replay states."""

    initial_assessment: QueryFitAssessment
    resource_boundary_state_index: int
    resource_boundary_round_index: int
    resource_boundary_work: float
    first_qualified_state_index: int | None
    first_qualified_round_index: int | None
    first_qualified_work: float | None
    selected_state_index: int
    selected_round_index: int
    selected_work: float
    selected_assessment: QueryFitAssessment
    termination_reason: str
    fit_target_reached: bool
    qualified_by_resource_boundary: bool
    prefix_minimum_loss_state_index: int
    selected_is_prefix_minimum_loss: bool
    full_trace_minimum_loss: float
    post_selected_minimum_loss: float | None
    post_selected_strict_improvement: bool
    full_work: float
    tail_work_after_resource_boundary: float


@dataclass(frozen=True)
class ExtractedTrace:
    """Auditable trace fields kept outside the minimal selector projection."""

    replay_states: tuple[QueryFitReplayState, ...]
    table_sha256_by_state: tuple[str, ...]
    primary_rng_sha256_by_state: tuple[str, ...]
    query_answers_by_state: tuple[tuple[float, ...], ...]


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


def _table_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _rng_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(
        rng.bit_generator.state,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json_exclusive(path: Path, value: Any) -> None:
    _strict_json_bytes(value)
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


def _family_by_name(name: str) -> ArtificialFamily:
    matches = [family for family in FAMILIES if family.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown frozen family: {name!r}")
    return matches[0]


def formal_cases() -> tuple[MatrixCase, ...]:
    """List the 12 formal identities without constructing any RNG."""

    cases = tuple(
        MatrixCase(
            family=family.name,
            seed=seed,
            rho=rho,
            n_rounds=ROUNDS_BY_RHO[rho],
        )
        for family in FAMILIES
        for seed in family.seeds
        for rho in RHOS
    )
    identities = [case.identity for case in cases]
    if len(cases) != 12 or len(set(identities)) != 12:
        raise RuntimeError("formal matrix identity is not exactly 12 unique cases")
    return cases


def _query(attributes: Sequence[str]) -> dict[str, Any]:
    return {
        "conditions": [
            {"attribute": name, "operator": "==", "value": 1} for name in attributes
        ]
    }


def build_artificial_problem(family: ArtificialFamily) -> ArtificialProblem:
    """Build and independently preflight one public feasible workload."""

    if family not in FAMILIES:
        raise ValueError("family must be one of the frozen families")
    if family.zero_row_count + family.one_row_count != family.n_records:
        raise RuntimeError("reference row multiplicities do not sum to N")
    schema = Schema(
        [
            AttributeBlock(
                name=name,
                type="categorical",
                description=name,
                values=[0, 1],
            )
            for name in family.attributes
        ]
    )
    queries = [_query(group) for group in family.query_attribute_groups]
    zero_rows = np.zeros(
        (family.zero_row_count, len(family.attributes)),
        dtype=int,
    )
    one_rows = np.ones(
        (family.one_row_count, len(family.attributes)),
        dtype=int,
    )
    reference_table = pd.DataFrame(
        np.vstack([zero_rows, one_rows]),
        columns=list(family.attributes),
    )
    target = np.asarray(family.target, dtype=np.float64)
    independently_evaluated = evaluate_table(reference_table, queries)
    if not np.array_equal(independently_evaluated, target):
        raise RuntimeError("explicit reference table does not match target")
    if len(queries) != family.query_count or len(target) != family.query_count:
        raise RuntimeError("query/target count does not match frozen family")
    return ArtificialProblem(
        family=family,
        schema=schema,
        queries=queries,
        target=target,
        reference_table=reference_table,
    )


def _frozen_generator_parameters() -> dict[str, Any]:
    """Return a JSON-safe statement of every fixed generator choice."""

    return {
        "init_method": "random",
        "device": "numpy",
        "eval_method": "vectorized",
        "batch_size": 256,
        "beta": 1.0,
        "h": 0.8,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "delta": 0.05,
        "winsorize_quantiles": [0.01, 0.99],
        "exclude_self": True,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 6.0,
        "eta": 0.45,
        "mu": 0.02,
        "tol": "positive_infinity",
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 0.8,
        "diffusion_direction_normalization": "fixed",
        "diffusion_direction_reference_scale": 1.25,
        "diffusion_direction_logit_clip": 9.0,
        "residual_geometry": "absolute",
        "residual_geometry_floor_inactive": 8.0,
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": None,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "rho_anneal_rounds": None,
        "selection_scale_invariant": False,
        "horizon_invariant": True,
        "stop_on_exact_residual": False,
        "record_transition_clocks": True,
        "record_stationarity_trace": True,
        "log_every": 100_000,
    }


def frozen_protocol() -> dict[str, Any]:
    """Return every scientific choice of the accepted result-before protocol."""

    thresholds = QueryFitThresholds.exact_integer_counts()
    protocol = {
        "contract_version": CONTRACT_VERSION,
        "protocol_document_sha256": EXPECTED_PROTOCOL_DOCUMENT_SHA256,
        "scope": {
            "fixed_exact_count_workload": True,
            "sigma": 0.0,
            "runs_project_generator": True,
            "reads_project_dataset": False,
            "reads_real_reference_table": False,
            "uses_gpu": False,
            "consumes_privacy_budget": False,
            "outer_selection_present": False,
            "online_generator_integration_present": False,
            "convergence_claim_present": False,
        },
        "fit_target": {
            "count_rmse_inclusive_maximum": thresholds.count_rmse_limit,
            "per_query_absolute_count_error_inclusive_maximum": (
                thresholds.max_abs_count_error_limit
            ),
            "both_required_on_same_current_checkpoint": True,
            "calibration_source": thresholds.calibration_source,
        },
        "resource": {
            "normalized_work": "cumulative_applied_participating_rows/N",
            "inclusive_boundary_work": RESOURCE_WORK_LIMIT,
            "boundary_state": ("first_real_state_with_normalized_work_at_least_20"),
            "minimum_post_boundary_tail_work": MINIMUM_TAIL_WORK,
            "priority": [
                "exact_residual",
                "fit_target_reached",
                "resource_cap_reached",
            ],
        },
        "selection": {
            "qualified": "first_current_checkpoint_satisfying_both_limits",
            "resource_fallback": ("minimum_squared_loss_through_boundary_earliest_tie"),
            "normal_return_is_first_qualified_current_not_loss_only_best": True,
            "normalized_l1_used_for_selection": False,
        },
        "families": [
            {
                "name": family.name,
                "attributes": list(family.attributes),
                "n_records": family.n_records,
                "query_count": family.query_count,
                "query_attribute_groups": [
                    list(group) for group in family.query_attribute_groups
                ],
                "reference_zero_rows": family.zero_row_count,
                "reference_one_rows": family.one_row_count,
                "target": list(family.target),
                "seeds": list(family.seeds),
            }
            for family in FAMILIES
        ],
        "rho": list(RHOS),
        "rounds_by_rho": {
            "1.0": ROUNDS_BY_RHO[1.0],
            "0.25": ROUNDS_BY_RHO[0.25],
        },
        "case_count": len(formal_cases()),
        "generator": _frozen_generator_parameters(),
        "acceptance": {
            "valid_case_count": 12,
            "qualified_by_boundary_count": 12,
            "all_cases_required": True,
            "post_result_threshold_retuning_allowed": False,
            "post_result_seed_or_family_replacement_allowed": False,
        },
        "formal_execution_requires_new_authorization": True,
    }
    _strict_json_bytes(protocol)
    return protocol


def _verify_protocol_document(root: Path) -> None:
    path = root / PROTOCOL_DOCUMENT
    if not path.is_file():
        raise RuntimeError(f"protocol document is missing: {path}")
    observed = _sha256_file(path)
    if observed != EXPECTED_PROTOCOL_DOCUMENT_SHA256:
        raise RuntimeError(
            "protocol document SHA-256 drifted: "
            f"expected {EXPECTED_PROTOCOL_DOCUMENT_SHA256}, got {observed}"
        )


def build_plan() -> dict[str, Any]:
    """Describe the frozen formal matrix without generating any table."""

    root = _repo_root()
    _verify_protocol_document(root)
    protocol = frozen_protocol()
    cases = [asdict(case) for case in formal_cases()]
    plan = {
        "contract_version": CONTRACT_VERSION,
        "mode": "plan_only_no_formal_rng_instantiation",
        "protocol_sha256": _sha256_json(protocol),
        "case_count": len(cases),
        "families": len(FAMILIES),
        "full_round_count": sum(case["n_rounds"] for case in cases),
        "formal_seed_values_listed_not_instantiated": True,
        "real_data_accessed": False,
        "generation_started": False,
        "execution_started": False,
        "cases": cases,
        "protocol": protocol,
    }
    _strict_json_bytes(plan)
    return plan


def _validate_replay_states(
    states: Sequence[QueryFitReplayState],
    *,
    n_records: int,
) -> tuple[QueryFitReplayState, ...]:
    if isinstance(n_records, bool) or not isinstance(n_records, int):
        raise TypeError("n_records must be a positive integer")
    if n_records <= 0:
        raise ValueError("n_records must be a positive integer")
    normalized = tuple(states)
    if not normalized:
        raise ValueError("states must not be empty")
    query_count = len(normalized[0].count_errors)
    previous_rows = 0
    for expected_index, state in enumerate(normalized):
        if not isinstance(state, QueryFitReplayState):
            raise TypeError("states must contain QueryFitReplayState values")
        if state.state_index != expected_index:
            raise ValueError("state_index must start at zero and be contiguous")
        if state.round_index != expected_index:
            raise ValueError("round_index must match the real state index")
        if len(state.count_errors) != query_count:
            raise ValueError("all count-error vectors must have equal length")
        rows = state.cumulative_participating_rows
        if expected_index == 0 and rows != 0:
            raise ValueError("the initial state must have zero participating work")
        if rows < previous_rows:
            raise ValueError("cumulative participating rows must not decrease")
        if expected_index > 0 and rows - previous_rows > n_records:
            raise ValueError("one real round cannot contribute more than N rows")
        previous_rows = rows
    return normalized


def replay_query_fit_states(
    states: Sequence[QueryFitReplayState],
    *,
    n_records: int,
) -> QueryFitReplayDecision:
    """Select a checkpoint using only same-state errors and applied work."""

    states_tuple = _validate_replay_states(states, n_records=n_records)
    thresholds = QueryFitThresholds.exact_integer_counts()
    assessments = tuple(
        assess_query_fit(state.count_errors, thresholds) for state in states_tuple
    )
    boundary_position = next(
        (
            position
            for position, state in enumerate(states_tuple)
            if state.cumulative_participating_rows / n_records >= RESOURCE_WORK_LIMIT
        ),
        None,
    )
    if boundary_position is None:
        raise ValueError("full trace does not reach the frozen resource boundary")
    first_qualified_position = next(
        (
            position
            for position, assessment in enumerate(assessments)
            if assessment.fit_target_reached
        ),
        None,
    )
    qualified_by_boundary = bool(
        first_qualified_position is not None
        and first_qualified_position <= boundary_position
    )
    prefix_minimum_position = min(
        range(boundary_position + 1),
        key=lambda position: (
            assessments[position].squared_loss,
            states_tuple[position].state_index,
        ),
    )
    selected_position = (
        int(first_qualified_position)
        if qualified_by_boundary
        else prefix_minimum_position
    )
    selected_assessment = assessments[selected_position]
    if qualified_by_boundary and selected_assessment.exact_residual:
        termination_reason = "exact_residual"
    elif qualified_by_boundary:
        termination_reason = "fit_target_reached"
    else:
        termination_reason = "resource_cap_reached"

    post_selected = assessments[selected_position + 1 :]
    post_selected_minimum = (
        min(row.squared_loss for row in post_selected) if post_selected else None
    )
    selected_state = states_tuple[selected_position]
    boundary_state = states_tuple[boundary_position]
    first_state = (
        states_tuple[first_qualified_position]
        if first_qualified_position is not None
        else None
    )
    full_work = states_tuple[-1].cumulative_participating_rows / n_records
    decision = QueryFitReplayDecision(
        initial_assessment=assessments[0],
        resource_boundary_state_index=boundary_state.state_index,
        resource_boundary_round_index=boundary_state.round_index,
        resource_boundary_work=(
            boundary_state.cumulative_participating_rows / n_records
        ),
        first_qualified_state_index=(
            first_state.state_index if first_state is not None else None
        ),
        first_qualified_round_index=(
            first_state.round_index if first_state is not None else None
        ),
        first_qualified_work=(
            first_state.cumulative_participating_rows / n_records
            if first_state is not None
            else None
        ),
        selected_state_index=selected_state.state_index,
        selected_round_index=selected_state.round_index,
        selected_work=(selected_state.cumulative_participating_rows / n_records),
        selected_assessment=selected_assessment,
        termination_reason=termination_reason,
        fit_target_reached=qualified_by_boundary,
        qualified_by_resource_boundary=qualified_by_boundary,
        prefix_minimum_loss_state_index=(
            states_tuple[prefix_minimum_position].state_index
        ),
        selected_is_prefix_minimum_loss=(selected_position == prefix_minimum_position),
        full_trace_minimum_loss=min(row.squared_loss for row in assessments),
        post_selected_minimum_loss=post_selected_minimum,
        post_selected_strict_improvement=bool(
            post_selected_minimum is not None
            and post_selected_minimum < selected_assessment.squared_loss
        ),
        full_work=full_work,
        tail_work_after_resource_boundary=(
            states_tuple[-1].cumulative_participating_rows
            - boundary_state.cumulative_participating_rows
        )
        / n_records,
    )
    if decision.fit_target_reached != (decision.selected_assessment.fit_target_reached):
        raise RuntimeError("selected checkpoint contradicts termination status")
    return decision


def extract_replay_trace(
    *,
    trace: StationarityTrace,
    transition_clocks: Sequence[dict[str, Any]],
    target: Sequence[float],
    queries: Sequence[dict[str, Any]],
) -> ExtractedTrace:
    """Build the L1-free selector projection and cross-check both clocks."""

    if not isinstance(trace, StationarityTrace):
        raise TypeError("trace must be a StationarityTrace")
    trace.validate()
    target_array = np.asarray(target, dtype=np.float64)
    if target_array.ndim != 1 or len(target_array) != trace.query_count:
        raise ValueError("target shape does not match the trace")
    if not np.all(np.isfinite(target_array)):
        raise ValueError("target must be finite")
    if not np.all(target_array == np.floor(target_array)):
        raise ValueError("the exact-count target must contain integers")
    if trace.query_identity_sha256 != ordered_query_identity_sha256(queries):
        raise ValueError("query identity does not match the trace")
    if trace.target_identity_sha256 != target_answer_identity_sha256(target_array):
        raise ValueError("target identity does not match the trace")
    if trace.termination_reason != "max_rounds":
        raise ValueError("the full evidence trace must terminate at max_rounds")
    clocks = tuple(transition_clocks)
    if len(clocks) != trace.state_count - 1:
        raise ValueError("transition clocks do not align with trace states")

    states: list[QueryFitReplayState] = []
    table_hashes: list[str] = []
    rng_hashes: list[str] = []
    answers_by_state: list[tuple[float, ...]] = []
    cumulative_rows = 0
    thresholds = QueryFitThresholds.exact_integer_counts()
    for position, (observation, answers) in enumerate(
        zip(
            trace.observations,
            trace.measured_query_answers,
        )
    ):
        answers_array = np.asarray(answers, dtype=np.float64)
        if not np.all(answers_array == np.floor(answers_array)):
            raise ValueError("exact query answers must contain integers")
        if position == 0:
            if observation["applied_participating_row_count"] != 0:
                raise ValueError("initial state cannot contain applied work")
        else:
            clock = clocks[position - 1]
            if clock.get("state_index") != position or clock.get("round") != position:
                raise ValueError("transition clock state identity is invalid")
            attempts = clock.get("attempts")
            if (
                clock.get("accepted_attempt") != 1
                or not isinstance(attempts, list)
                or len(attempts) != 1
            ):
                raise ValueError("formal no-gate trace must accept one attempt")
            participating_rows = attempts[0].get("participating_rows")
            if (
                isinstance(participating_rows, bool)
                or not isinstance(participating_rows, int)
                or not 0 <= participating_rows <= trace.n_records
            ):
                raise ValueError("transition participating rows are invalid")
            if (
                observation["proposal_attempt_count"] != 1
                or observation["proposal_accepted"] is not True
                or observation["applied_attempt_index"] != 1
                or observation["applied_participating_row_count"] != participating_rows
            ):
                raise ValueError("trace observation and transition clock differ")
            if (
                clock.get("post_current_table_sha256")
                != observation["current_table_sha256"]
            ):
                raise ValueError("table identity differs across trace clocks")
            cumulative_rows += participating_rows

        errors = target_array - answers_array
        assessment = assess_query_fit(errors, thresholds)
        if assessment.squared_loss != observation["current_squared_loss"]:
            raise ValueError("trace loss does not match same-state count errors")
        states.append(
            QueryFitReplayState(
                state_index=position,
                round_index=position,
                count_errors=tuple(float(value) for value in errors),
                cumulative_participating_rows=cumulative_rows,
            )
        )
        table_hashes.append(observation["current_table_sha256"])
        rng_hashes.append(observation["primary_rng_state_sha256"])
        answers_by_state.append(tuple(float(value) for value in answers_array))

    return ExtractedTrace(
        replay_states=tuple(states),
        table_sha256_by_state=tuple(table_hashes),
        primary_rng_sha256_by_state=tuple(rng_hashes),
        query_answers_by_state=tuple(answers_by_state),
    )


def _generator_kwargs(
    problem: ArtificialProblem,
    case: MatrixCase,
    *,
    n_rounds: int,
    return_final_table: bool,
) -> dict[str, Any]:
    """Translate the JSON-safe frozen statement into generator arguments."""

    return {
        "target": problem.target,
        "queries": problem.queries,
        "schema": problem.schema,
        "n_records": problem.family.n_records,
        "n_rounds": n_rounds,
        "seed": case.seed,
        "beta": 1.0,
        "h": 0.8,
        "rho": case.rho,
        "eta": 0.45,
        "mu": 0.02,
        "tol": float("inf"),
        "device": "numpy",
        "eval_method": "vectorized",
        "batch_size": 256,
        "init_method": "random",
        "log_every": 100_000,
        "distance_mode": "geometric",
        "lambda_param": 0.5,
        "delta": 0.05,
        "winsorize_quantiles": (0.01, 0.99),
        "exclude_self": True,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 0.8,
        "diffusion_direction_normalization": "fixed",
        "factorized_gibbs_sweeps": 0,
        "factorized_gibbs_max_order": 3,
        "factorized_gibbs_logit_clip": 30.0,
        "factorized_gibbs_use_compiled_workload": False,
        "candidate_budget": None,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "rho_anneal_rounds": None,
        "selection_scale_invariant": False,
        "residual_geometry": "absolute",
        "residual_geometry_floor": 8.0,
        "return_final_table": return_final_table,
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 6.0,
        "diffusion_direction_reference_scale": 1.25,
        "diffusion_direction_logit_clip": 9.0,
        "record_transition_clocks": True,
        "record_stationarity_trace": True,
        "stop_on_exact_residual": False,
        "horizon_invariant": True,
    }


def _validate_full_diagnostics(
    diagnostics: dict[str, Any],
    *,
    case: MatrixCase,
) -> None:
    if diagnostics.get("termination_reason") != "max_rounds":
        raise RuntimeError("full trace did not terminate at max_rounds")
    if diagnostics.get("rounds_run") != case.n_rounds:
        raise RuntimeError("full trace did not run the frozen horizon")
    if diagnostics.get("accept_history") != [True] * case.n_rounds:
        raise RuntimeError("formal no-gate trace contains a rejected proposal")
    if diagnostics.get("proposal_attempts_history") != [1] * case.n_rounds:
        raise RuntimeError("formal no-gate trace contains retry attempts")
    if diagnostics.get("accepted_attempt_history") != [1] * case.n_rounds:
        raise RuntimeError("formal no-gate trace has an invalid accepted attempt")
    if diagnostics.get("candidate_evaluation_count") != case.n_rounds:
        raise RuntimeError("candidate evaluation count differs from horizon")
    if len(diagnostics.get("transition_clock_history", [])) != case.n_rounds:
        raise RuntimeError("transition clock count differs from horizon")
    trace = diagnostics.get("stationarity_trace")
    if not isinstance(trace, StationarityTrace):
        raise TypeError("formal trace did not record query-answer evidence")
    if trace.state_count != case.n_rounds + 1:
        raise RuntimeError("stationarity state count differs from horizon")
    params = diagnostics.get("params", {})
    expected = {
        "n_rounds": case.n_rounds,
        "seed": case.seed,
        "rho": case.rho,
        "eta": 0.45,
        "mu": 0.02,
        "tol": float("inf"),
        "device": "numpy",
        "eval_method": "vectorized",
        "init_method": "random",
        "distance_mode": "geometric",
        "alpha_schedule_mode": "fixed",
        "fixed_alpha": 6.0,
        "max_retries": 0,
        "residual_directed_diffusion": True,
        "diffusion_direction_strength": 0.8,
        "diffusion_direction_normalization": "fixed",
        "diffusion_direction_reference_scale": 1.25,
        "diffusion_direction_logit_clip": 9.0,
        "factorized_gibbs_sweeps": 0,
        "candidate_budget": None,
        "residual_self_cooling": None,
        "rho_anneal_end": None,
        "selection_scale_invariant": False,
        "residual_geometry": "absolute",
        "record_transition_clocks": True,
        "record_stationarity_trace": True,
        "stop_on_exact_residual": False,
        "horizon_invariant": True,
    }
    mismatches = {
        key: (params.get(key), value)
        for key, value in expected.items()
        if params.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"formal generator parameter drift: {mismatches}")


def _materialize_selected_checkpoint(
    *,
    problem: ArtificialProblem,
    case: MatrixCase,
    full_diagnostics: dict[str, Any],
    extracted: ExtractedTrace,
    decision: QueryFitReplayDecision,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_state = decision.selected_state_index
    if selected_state == 0:
        rng = np.random.default_rng(case.seed)
        selected_table = init_synthetic_table(
            problem.family.n_records,
            problem.schema,
            rng,
        )
        replay_checks = {
            "initial_table_sha256": (
                _table_sha256(selected_table)
                == full_diagnostics["initial_table_sha256"]
            ),
            "initial_rng_sha256": (
                _rng_state_sha256(rng)
                == full_diagnostics["primary_rng_post_initialization_state_sha256"]
            ),
            "selected_state_rng_sha256": (
                _rng_state_sha256(rng) == extracted.primary_rng_sha256_by_state[0]
            ),
        }
    else:
        _, prefix = run_evolution(
            **_generator_kwargs(
                problem,
                case,
                n_rounds=selected_state,
                return_final_table=True,
            )
        )
        selected_table = prefix.pop("final_table")
        prefix_trace = prefix["stationarity_trace"]
        full_trace = full_diagnostics["stationarity_trace"]
        replay_checks = {
            "prefix_termination": (
                prefix["termination_reason"] == "max_rounds"
                and prefix["rounds_run"] == selected_state
            ),
            "prefix_metrics": (
                prefix["current_state_metrics_history"]
                == full_diagnostics["current_state_metrics_history"][
                    : selected_state + 1
                ]
            ),
            "prefix_clocks": (
                prefix["transition_clock_history"]
                == full_diagnostics["transition_clock_history"][:selected_state]
            ),
            "prefix_observations": (
                prefix_trace.observations
                == full_trace.observations[: selected_state + 1]
            ),
            "prefix_query_answers": np.array_equal(
                prefix_trace.measured_query_answers,
                full_trace.measured_query_answers[: selected_state + 1],
            ),
            "selected_state_rng_sha256": (
                prefix["primary_rng_state_sha256"]
                == extracted.primary_rng_sha256_by_state[selected_state]
            ),
        }

    selected_hash = _table_sha256(selected_table)
    selected_answers = evaluate_table(selected_table, problem.queries)
    selected_errors = problem.target - selected_answers
    independent_assessment = assess_query_fit(
        selected_errors,
        QueryFitThresholds.exact_integer_counts(),
    )
    replay_checks.update(
        {
            "selected_table_sha256": (
                selected_hash == extracted.table_sha256_by_state[selected_state]
            ),
            "selected_query_answers": (
                tuple(float(value) for value in selected_answers)
                == extracted.query_answers_by_state[selected_state]
            ),
            "selected_count_errors": (
                tuple(float(value) for value in selected_errors)
                == extracted.replay_states[selected_state].count_errors
            ),
            "selected_assessment": (
                independent_assessment == decision.selected_assessment
            ),
        }
    )
    if not all(replay_checks.values()):
        failed = [key for key, value in replay_checks.items() if not value]
        raise RuntimeError(f"selected checkpoint replay failed: {failed}")

    offline_l1 = compute_normalized_l1(
        problem.target,
        selected_answers,
        problem.family.n_records,
    )
    l1_bound_check = bool(
        not decision.fit_target_reached or offline_l1 <= 1.0 / problem.family.n_records
    )
    if not l1_bound_check:
        raise RuntimeError("post-selection L1 violates the RMSE-implied bound")
    details = {
        "selected_table_sha256": selected_hash,
        "selected_query_answers": [float(value) for value in selected_answers],
        "offline_normalized_l1": offline_l1,
        "offline_l1_bound": (
            1.0 / problem.family.n_records if decision.fit_target_reached else None
        ),
        "offline_l1_bound_check": l1_bound_check,
        "checkpoint_replay_checks": replay_checks,
    }
    _strict_json_bytes(details)
    return selected_table, details


def _assessment_dict(assessment: QueryFitAssessment) -> dict[str, Any]:
    return {
        "squared_loss": assessment.squared_loss,
        "count_rmse": assessment.count_rmse,
        "max_abs_count_error": assessment.max_abs_count_error,
        "per_query_abs_count_errors": list(assessment.per_query_abs_count_errors),
        "rmse_within_limit": assessment.rmse_within_limit,
        "every_query_within_limit": assessment.every_query_within_limit,
        "fit_target_reached": assessment.fit_target_reached,
        "exact_residual": assessment.exact_residual,
        "calibration_source": assessment.calibration_source,
    }


def _execute_case(
    case: MatrixCase,
    *,
    table_output_dir: Path,
) -> tuple[dict[str, Any], float]:
    family = _family_by_name(case.family)
    if case.seed not in family.seeds:
        raise RuntimeError("case seed does not belong to the frozen family")
    if case.rho not in RHOS or case.n_rounds != ROUNDS_BY_RHO[case.rho]:
        raise RuntimeError("case rho/horizon differs from the frozen matrix")
    problem = build_artificial_problem(family)
    started = time.perf_counter()
    _, diagnostics = run_evolution(
        **_generator_kwargs(
            problem,
            case,
            n_rounds=case.n_rounds,
            return_final_table=False,
        )
    )
    _validate_full_diagnostics(diagnostics, case=case)
    extracted = extract_replay_trace(
        trace=diagnostics["stationarity_trace"],
        transition_clocks=diagnostics["transition_clock_history"],
        target=problem.target,
        queries=problem.queries,
    )
    decision = replay_query_fit_states(
        extracted.replay_states,
        n_records=family.n_records,
    )
    if decision.tail_work_after_resource_boundary < MINIMUM_TAIL_WORK:
        raise RuntimeError("full trace has insufficient post-boundary tail work")
    selected_table, selected_details = _materialize_selected_checkpoint(
        problem=problem,
        case=case,
        full_diagnostics=diagnostics,
        extracted=extracted,
        decision=decision,
    )

    filename = f"{case.family}_seed{case.seed}_rho{case.rho_label}_selected.csv"
    table_path = table_output_dir / filename
    selected_table.to_csv(table_path, index=False, mode="x")
    artifact_sha256 = _sha256_file(table_path)
    if artifact_sha256 != selected_details["selected_table_sha256"]:
        raise RuntimeError("selected table artifact SHA-256 is inconsistent")

    full_trace = diagnostics["stationarity_trace"]
    query_identity = ordered_query_identity_sha256(problem.queries)
    target_identity = target_answer_identity_sha256(problem.target)
    row = {
        "family": family.name,
        "n_records": family.n_records,
        "query_count": family.query_count,
        "seed": case.seed,
        "rho": case.rho,
        "n_rounds": case.n_rounds,
        "initial": _assessment_dict(decision.initial_assessment),
        "resource_boundary_state_index": (decision.resource_boundary_state_index),
        "resource_boundary_round_index": (decision.resource_boundary_round_index),
        "resource_boundary_work": decision.resource_boundary_work,
        "first_qualified_state_index": (decision.first_qualified_state_index),
        "first_qualified_round_index": (decision.first_qualified_round_index),
        "first_qualified_work": decision.first_qualified_work,
        "selected_state_index": decision.selected_state_index,
        "selected_round_index": decision.selected_round_index,
        "selected_work": decision.selected_work,
        "selected_assessment": _assessment_dict(decision.selected_assessment),
        "termination_reason": decision.termination_reason,
        "fit_target_reached": decision.fit_target_reached,
        "qualified_by_resource_boundary": (decision.qualified_by_resource_boundary),
        "prefix_minimum_loss_state_index": (decision.prefix_minimum_loss_state_index),
        "selected_is_prefix_minimum_loss": (decision.selected_is_prefix_minimum_loss),
        "full_trace_minimum_loss": decision.full_trace_minimum_loss,
        "post_selected_minimum_loss": (decision.post_selected_minimum_loss),
        "post_selected_strict_improvement": (decision.post_selected_strict_improvement),
        "full_work": decision.full_work,
        "tail_work_after_resource_boundary": (
            decision.tail_work_after_resource_boundary
        ),
        "candidate_evaluation_count": diagnostics["candidate_evaluation_count"],
        "selected_table_artifact": {
            "filename": filename,
            "sha256": artifact_sha256,
        },
        **selected_details,
        "identity": {
            "query_sha256": query_identity,
            "target_sha256": target_identity,
            "trace_query_sha256": full_trace.query_identity_sha256,
            "trace_target_sha256": full_trace.target_identity_sha256,
            "initial_table_sha256": diagnostics["initial_table_sha256"],
            "generator_parameter_sha256": _sha256_json(
                {
                    **_frozen_generator_parameters(),
                    "family": family.name,
                    "seed": case.seed,
                    "rho": case.rho,
                    "n_rounds": case.n_rounds,
                    "n_records": family.n_records,
                }
            ),
        },
        "validity": {
            "reference_target_preflight": True,
            "full_horizon_max_rounds": True,
            "one_accepted_attempt_per_round": True,
            "query_and_transition_clocks_aligned": True,
            "finite_same_checkpoint_metrics": True,
            "resource_boundary_reached": True,
            "minimum_tail_work_reached": True,
            "selected_checkpoint_replayed": True,
            "l1_computed_only_after_selection": True,
        },
    }
    elapsed = float(time.perf_counter() - started)
    _strict_json_bytes(row)
    return row, elapsed


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def summarize(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
        first_work = [
            row["first_qualified_work"]
            for row in selected
            if row["first_qualified_work"] is not None
        ]
        return {
            "case_count": len(selected),
            "qualified_anywhere_count": len(first_work),
            "qualified_by_resource_boundary_count": sum(
                row["qualified_by_resource_boundary"] for row in selected
            ),
            "first_qualified_work_values": first_work,
            "first_qualified_work_median": (
                float(statistics.median(first_work)) if first_work else None
            ),
        }

    by_family = {
        family.name: summarize([row for row in rows if row["family"] == family.name])
        for family in FAMILIES
    }
    by_rho = {
        str(rho): summarize([row for row in rows if row["rho"] == rho]) for rho in RHOS
    }
    result = {
        "overall": summarize(rows),
        "by_family": by_family,
        "by_rho": by_rho,
    }
    _strict_json_bytes(result)
    return result


def _matrix_acceptance(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected = {case.identity for case in formal_cases()}
    actual = [(row["family"], row["seed"], row["rho"]) for row in rows]
    identity_pass = len(actual) == 12 and set(actual) == expected
    validity_pass = bool(
        identity_pass
        and all(
            all(row["validity"].values())
            and row["tail_work_after_resource_boundary"] >= MINIMUM_TAIL_WORK
            for row in rows
        )
    )
    qualified_count = sum(row["qualified_by_resource_boundary"] for row in rows)
    scientific_pass = bool(validity_pass and qualified_count == 12)
    result = {
        "matrix_identity_pass": identity_pass,
        "execution_validity_pass": validity_pass,
        "qualified_by_resource_boundary_count": qualified_count,
        "required_qualified_count": 12,
        "scientific_pass": scientific_pass,
        "status": ("candidate_supported" if scientific_pass else "candidate_failed")
        if validity_pass
        else "execution_invalid",
        "post_result_retuning_allowed": False,
    }
    _strict_json_bytes(result)
    return result


def build_execution_manifest(root: Path) -> dict[str, Any]:
    """Fail closed before any formal generator call or RNG construction."""

    if set(inspect.signature(run_artificial_protocol).parameters) != {"output_dir"}:
        raise RuntimeError("formal runner gained an unexpected override")
    _verify_protocol_document(root)
    status = _git_text(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise RuntimeError(
            "RMSE+max formal protocol requires a clean worktree before RNG"
        )
    missing = [str(path) for path in SOURCE_PATHS if not (root / path).is_file()]
    if missing:
        raise RuntimeError(f"formal protocol source files are missing: {missing}")
    protocol = frozen_protocol()
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_text(root, "rev-parse", "HEAD"),
        "git_worktree_clean_including_untracked": True,
        "protocol_sha256": _sha256_json(protocol),
        "protocol_document_sha256": EXPECTED_PROTOCOL_DOCUMENT_SHA256,
        "source_sha256": {
            str(path): _sha256_file(root / path) for path in SOURCE_PATHS
        },
        "environment": {
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "execution_started": False,
        "formal_rng_instantiated": False,
        "protocol": protocol,
    }
    _strict_json_bytes(manifest)
    return manifest


def run_artificial_protocol(
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Execute exactly the frozen 12-case matrix after later authorization."""

    root = _repo_root()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    manifest = build_execution_manifest(root)
    output_dir.mkdir(parents=True, exist_ok=False)
    table_output_dir = output_dir / "selected_tables"
    table_output_dir.mkdir()
    manifest_path = output_dir / "protocol_manifest.json"
    report_path = output_dir / "rmse_max_evidence_report.json"
    _write_json_exclusive(manifest_path, manifest)
    manifest_sha256 = _sha256_file(manifest_path)

    total_started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    elapsed_by_case: list[dict[str, Any]] = []
    for case in formal_cases():
        row, elapsed = _execute_case(
            case,
            table_output_dir=table_output_dir,
        )
        rows.append(row)
        elapsed_by_case.append(
            {
                "family": case.family,
                "seed": case.seed,
                "rho": case.rho,
                "elapsed_sec": elapsed,
            }
        )
    aggregate = _aggregate_rows(rows)
    acceptance = _matrix_acceptance(rows)
    scientific_payload = {
        "case_rows": rows,
        "aggregate": aggregate,
        "acceptance": acceptance,
    }
    report = {
        "report_format": REPORT_FORMAT,
        "contract_version": CONTRACT_VERSION,
        "status": acceptance["status"],
        "manifest_path": manifest_path.name,
        "manifest_sha256": manifest_sha256,
        "protocol_sha256": manifest["protocol_sha256"],
        "protocol_document_sha256": EXPECTED_PROTOCOL_DOCUMENT_SHA256,
        "git_commit": manifest["git_commit"],
        "protocol": manifest["protocol"],
        "execution": {
            "elapsed_sec": float(time.perf_counter() - total_started),
            "case_elapsed": elapsed_by_case,
            "case_count": len(rows),
            "full_round_count": sum(case.n_rounds for case in formal_cases()),
            "real_data_accessed": False,
            "privacy_budget_consumed": False,
            "formal_artificial_generation": True,
        },
        "scientific_payload": scientific_payload,
        "scientific_result_sha256": _sha256_json(scientific_payload),
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
    manifest_path, report_path, report = run_artificial_protocol(Path(args.output_dir))
    print("\n===== Issue #53 RMSE+max Artificial Evidence =====")
    print(f"status={report['status']}")
    print(f"scientific_sha256={report['scientific_result_sha256']}")
    print(f"manifest={manifest_path}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
