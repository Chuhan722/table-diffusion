"""Deterministic contracts for the frozen Issue #53 RMSE+max runner."""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import validate_issue53_rmse_max_artificial as runner
from table_diffevo.stationarity import (
    StationarityTrace,
    build_stationarity_observation,
    ordered_query_identity_sha256,
    target_answer_identity_sha256,
)


def _unqualified_errors(query_count: int = 10) -> tuple[float, ...]:
    # RMSE is below one for m=10, but the single-query guard rejects 3.
    return (3.0,) + (0.0,) * (query_count - 1)


def _qualified_errors(query_count: int = 10) -> tuple[float, ...]:
    return (1.0,) * query_count


def _states(
    overrides: dict[int, tuple[float, ...]] | None = None,
    *,
    total_rounds: int = 30,
    n_records: int = 10,
    default_errors: tuple[float, ...] | None = None,
    increments: list[int] | None = None,
) -> tuple[runner.QueryFitReplayState, ...]:
    overrides = overrides or {}
    default_errors = default_errors or _unqualified_errors()
    increments = increments or [n_records] * total_rounds
    if len(increments) != total_rounds:
        raise ValueError("increments length must equal total_rounds")
    cumulative = 0
    rows = [
        runner.QueryFitReplayState(
            state_index=0,
            round_index=0,
            count_errors=overrides.get(0, default_errors),
            cumulative_participating_rows=0,
        )
    ]
    for index, increment in enumerate(increments, start=1):
        cumulative += increment
        rows.append(
            runner.QueryFitReplayState(
                state_index=index,
                round_index=index,
                count_errors=overrides.get(index, default_errors),
                cumulative_participating_rows=cumulative,
            )
        )
    return tuple(rows)


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False).encode("utf-8")).hexdigest()


def _identity_hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _trace_fixture(
    errors_by_state: list[tuple[float, ...]],
    *,
    n_records: int = 10,
) -> tuple[
    StationarityTrace,
    list[dict],
    np.ndarray,
    list[dict],
]:
    query_count = len(errors_by_state[0])
    target = np.full(query_count, 5.0)
    queries = [
        {"conditions": [{"attribute": "x", "operator": "==", "value": 1}]}
        for _ in range(query_count)
    ]
    observations = []
    answers = []
    clocks = []
    previous_frame = pd.DataFrame({"x": np.zeros(n_records, dtype=int)})
    previous_answers = target - np.asarray(errors_by_state[0])
    observations.append(
        build_stationarity_observation(
            frame=previous_frame,
            target=target,
            current_query_answers=previous_answers,
            n_records=n_records,
            squared_loss=float(
                0.5
                * np.dot(
                    np.asarray(errors_by_state[0]),
                    np.asarray(errors_by_state[0]),
                )
            ),
            state_index=0,
            round_index=0,
            phase="initial",
            proposal_attempt_count=0,
            proposal_accepted=False,
            applied_attempt_index=0,
            attempted_participating_row_count=0,
            applied_participating_row_count=0,
            actual_changed_row_count=0,
            actual_changed_cell_count=0,
            actual_changed_query_count=0,
            normalized_query_l1_movement_mean=0.0,
            gibbs_microstep_count_attempted=0,
            gibbs_microstep_count_applied=0,
            candidate_evaluation_count_cumulative=0,
            current_table_sha256=_frame_hash(previous_frame),
            primary_rng_state_sha256=_identity_hash("rng-0"),
            factorized_gibbs_rng_state_sha256=None,
        )
    )
    answers.append(previous_answers)

    for state_index, error_values in enumerate(
        errors_by_state[1:],
        start=1,
    ):
        current_frame = previous_frame.copy()
        current_frame["x"] = 1 - current_frame["x"]
        current_answers = target - np.asarray(error_values)
        delta = current_answers - previous_answers
        table_hash = _frame_hash(current_frame)
        observations.append(
            build_stationarity_observation(
                frame=current_frame,
                target=target,
                current_query_answers=current_answers,
                n_records=n_records,
                squared_loss=float(
                    0.5
                    * np.dot(
                        np.asarray(error_values),
                        np.asarray(error_values),
                    )
                ),
                state_index=state_index,
                round_index=state_index,
                phase="post_round",
                proposal_attempt_count=1,
                proposal_accepted=True,
                applied_attempt_index=1,
                attempted_participating_row_count=n_records,
                applied_participating_row_count=n_records,
                actual_changed_row_count=n_records,
                actual_changed_cell_count=n_records,
                actual_changed_query_count=int(np.count_nonzero(delta)),
                normalized_query_l1_movement_mean=float(
                    np.mean(np.abs(delta)) / n_records
                ),
                gibbs_microstep_count_attempted=0,
                gibbs_microstep_count_applied=0,
                candidate_evaluation_count_cumulative=state_index,
                current_table_sha256=table_hash,
                primary_rng_state_sha256=_identity_hash(f"rng-{state_index}"),
                factorized_gibbs_rng_state_sha256=None,
            )
        )
        clocks.append(
            {
                "state_index": state_index,
                "round": state_index,
                "accepted_attempt": 1,
                "attempts": [{"participating_rows": n_records}],
                "post_current_table_sha256": table_hash,
            }
        )
        answers.append(current_answers)
        previous_frame = current_frame
        previous_answers = current_answers

    trace = StationarityTrace(
        n_records=n_records,
        query_identity_sha256=ordered_query_identity_sha256(queries),
        target_identity_sha256=target_answer_identity_sha256(target),
        observations=observations,
        measured_query_answers=np.stack(answers),
        termination_reason="max_rounds",
    )
    return trace, clocks, target, queries


def _forbidden_rng(*args, **kwargs):
    raise AssertionError("formal seed/RNG must not be instantiated in tests")


def test_plan_is_exact_and_instantiates_no_rng_or_generator(monkeypatch) -> None:
    monkeypatch.setattr(runner.np.random, "default_rng", _forbidden_rng)
    monkeypatch.setattr(runner, "run_evolution", _forbidden_rng)
    monkeypatch.setattr(runner, "init_synthetic_table", _forbidden_rng)

    plan = runner.build_plan()

    assert plan["mode"] == "plan_only_no_formal_rng_instantiation"
    assert plan["case_count"] == 12
    assert plan["families"] == 3
    assert plan["full_round_count"] == 1200
    assert plan["formal_seed_values_listed_not_instantiated"] is True
    assert plan["real_data_accessed"] is False
    assert plan["generation_started"] is False
    assert plan["execution_started"] is False


def test_formal_matrix_identity_and_pairing_are_frozen() -> None:
    cases = runner.formal_cases()

    assert len(cases) == 12
    assert len({case.identity for case in cases}) == 12
    assert {case.rho for case in cases} == {1.0, 0.25}
    assert all(case.n_rounds == runner.ROUNDS_BY_RHO[case.rho] for case in cases)
    for family in runner.FAMILIES:
        family_cases = [case for case in cases if case.family == family.name]
        assert len(family_cases) == 4
        assert {case.seed for case in family_cases} == set(family.seeds)
        assert {(case.seed, case.rho) for case in family_cases} == {
            (seed, rho) for seed in family.seeds for rho in runner.RHOS
        }


def test_artificial_references_independently_match_frozen_targets() -> None:
    observed = {}
    for family in runner.FAMILIES:
        problem = runner.build_artificial_problem(family)
        independently_evaluated = runner.evaluate_table(
            problem.reference_table,
            problem.queries,
        )
        np.testing.assert_array_equal(
            independently_evaluated,
            problem.target,
        )
        assert len(problem.reference_table) == family.n_records
        assert len(problem.queries) == family.query_count
        assert problem.schema.attribute_names() == list(family.attributes)
        observed[family.name] = (
            family.n_records,
            family.query_count,
            tuple(problem.target),
        )

    assert observed == {
        "marginal_skew": (24, 3, (6.0, 6.0, 6.0)),
        "ring_pair": (32, 10, (16.0,) * 10),
        "nested_overlap": (64, 15, (32.0,) * 15),
    }


def test_protocol_document_and_frozen_protocol_identity_match() -> None:
    root = Path(runner.__file__).resolve().parents[1]

    assert (
        runner._sha256_file(root / runner.PROTOCOL_DOCUMENT)
        == runner.EXPECTED_PROTOCOL_DOCUMENT_SHA256
    )
    protocol = runner.frozen_protocol()
    assert protocol["fit_target"] == {
        "count_rmse_inclusive_maximum": 1.0,
        "per_query_absolute_count_error_inclusive_maximum": 2.0,
        "both_required_on_same_current_checkpoint": True,
        "calibration_source": "exact_integer_counts",
    }
    assert protocol["acceptance"]["valid_case_count"] == 12
    assert protocol["acceptance"]["qualified_by_boundary_count"] == 12
    assert protocol["formal_execution_requires_new_authorization"] is True


def test_formal_cli_has_no_scientific_override() -> None:
    assert set(inspect.signature(runner.run_artificial_protocol).parameters) == {
        "output_dir"
    }
    parsed = runner._build_parser().parse_args(["run", "--output-dir", "new-output"])
    assert vars(parsed) == {
        "command": "run",
        "output_dir": "new-output",
    }
    with pytest.raises(SystemExit):
        runner._build_parser().parse_args(
            [
                "run",
                "--output-dir",
                "new-output",
                "--seed",
                "1",
            ]
        )


def test_dirty_worktree_rejected_before_rng_or_generator(
    monkeypatch,
) -> None:
    root = Path(runner.__file__).resolve().parents[1]

    def fake_git_text(_root, *arguments):
        if arguments[:2] == ("status", "--porcelain"):
            return "?? dirty"
        raise AssertionError("dirty preflight must stop before rev-parse")

    monkeypatch.setattr(runner, "_git_text", fake_git_text)
    monkeypatch.setattr(runner.np.random, "default_rng", _forbidden_rng)
    monkeypatch.setattr(runner, "run_evolution", _forbidden_rng)

    with pytest.raises(RuntimeError, match="clean worktree"):
        runner.build_execution_manifest(root)


def test_existing_output_rejected_before_manifest_or_rng(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "build_execution_manifest", _forbidden_rng)
    monkeypatch.setattr(runner.np.random, "default_rng", _forbidden_rng)
    monkeypatch.setattr(runner, "run_evolution", _forbidden_rng)

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_artificial_protocol(tmp_path)


def test_selector_projection_has_no_l1_reference_family_or_noise_fields() -> None:
    assert {field.name for field in fields(runner.QueryFitReplayState)} == {
        "state_index",
        "round_index",
        "count_errors",
        "cumulative_participating_rows",
    }
    assert set(inspect.signature(runner.replay_query_fit_states).parameters) == {
        "states",
        "n_records",
    }


def test_first_qualified_current_wins_even_if_its_loss_is_not_prefix_best() -> None:
    states = _states({5: _qualified_errors()})

    decision = runner.replay_query_fit_states(states, n_records=10)

    assert decision.termination_reason == "fit_target_reached"
    assert decision.fit_target_reached is True
    assert decision.first_qualified_state_index == 5
    assert decision.selected_state_index == 5
    assert decision.selected_assessment.squared_loss == 5.0
    assert decision.prefix_minimum_loss_state_index == 0
    assert decision.selected_is_prefix_minimum_loss is False
    assert decision.initial_assessment.squared_loss == 4.5
    assert decision.initial_assessment.rmse_within_limit is True
    assert decision.initial_assessment.every_query_within_limit is False


def test_exact_initial_state_has_highest_termination_priority() -> None:
    states = _states({0: (0.0,) * 10})

    decision = runner.replay_query_fit_states(states, n_records=10)

    assert decision.termination_reason == "exact_residual"
    assert decision.first_qualified_state_index == 0
    assert decision.selected_state_index == 0
    assert decision.selected_work == 0.0
    assert decision.selected_assessment.exact_residual is True


def test_nonexact_initial_fit_target_is_a_valid_zero_work_result() -> None:
    states = _states({0: _qualified_errors()})

    decision = runner.replay_query_fit_states(states, n_records=10)

    assert decision.termination_reason == "fit_target_reached"
    assert decision.selected_state_index == 0
    assert decision.selected_work == 0.0
    assert decision.selected_assessment.exact_residual is False


def test_resource_boundary_state_is_assessed_before_cap() -> None:
    states = _states({20: _qualified_errors()})

    decision = runner.replay_query_fit_states(states, n_records=10)

    assert decision.resource_boundary_state_index == 20
    assert decision.resource_boundary_work == 20.0
    assert decision.first_qualified_state_index == 20
    assert decision.qualified_by_resource_boundary is True
    assert decision.termination_reason == "fit_target_reached"
    assert decision.selected_state_index == 20


def test_atomic_boundary_overshoot_still_allows_same_state_quality() -> None:
    increments = [4] * 19 + [3, 2] + [4] * 10
    states = _states(
        {21: _qualified_errors()},
        total_rounds=len(increments),
        n_records=4,
        increments=increments,
    )

    decision = runner.replay_query_fit_states(states, n_records=4)

    assert decision.resource_boundary_state_index == 21
    assert decision.resource_boundary_work == 20.25
    assert decision.first_qualified_state_index == 21
    assert decision.termination_reason == "fit_target_reached"
    assert decision.tail_work_after_resource_boundary == 10.0


def test_late_qualification_fails_and_returns_earliest_minimum_loss() -> None:
    worse = (4.0,) + (0.0,) * 9
    states = _states(
        {
            0: worse,
            7: _unqualified_errors(),
            9: _unqualified_errors(),
            21: _qualified_errors(),
        },
        default_errors=worse,
    )

    decision = runner.replay_query_fit_states(states, n_records=10)

    assert decision.resource_boundary_state_index == 20
    assert decision.first_qualified_state_index == 21
    assert decision.qualified_by_resource_boundary is False
    assert decision.termination_reason == "resource_cap_reached"
    assert decision.fit_target_reached is False
    assert decision.prefix_minimum_loss_state_index == 7
    assert decision.selected_state_index == 7
    assert decision.selected_assessment.fit_target_reached is False


def test_never_qualified_is_fail_closed_and_reports_tail_improvement() -> None:
    worse = (5.0,) + (0.0,) * 9
    states = _states(
        {
            8: (4.0,) + (0.0,) * 9,
            25: _unqualified_errors(),
        },
        default_errors=worse,
    )

    decision = runner.replay_query_fit_states(states, n_records=10)

    assert decision.first_qualified_state_index is None
    assert decision.termination_reason == "resource_cap_reached"
    assert decision.selected_state_index == 8
    assert decision.post_selected_minimum_loss == 4.5
    assert decision.post_selected_strict_improvement is True
    assert decision.full_trace_minimum_loss == 4.5


@pytest.mark.parametrize(
    "states,n_records,error",
    [
        (_states(total_rounds=19), 10, "resource boundary"),
        (
            (
                runner.QueryFitReplayState(0, 0, (3.0,), 0),
                runner.QueryFitReplayState(2, 2, (3.0,), 10),
            ),
            10,
            "state_index",
        ),
        (
            (
                runner.QueryFitReplayState(0, 0, (3.0,), 0),
                runner.QueryFitReplayState(1, 1, (3.0,), 11),
            ),
            10,
            "more than N",
        ),
        (
            (
                runner.QueryFitReplayState(0, 0, (3.0,), 0),
                runner.QueryFitReplayState(1, 1, (3.0, 0.0), 10),
            ),
            10,
            "equal length",
        ),
    ],
)
def test_replay_rejects_invalid_state_or_work_contract(
    states,
    n_records,
    error,
) -> None:
    with pytest.raises(ValueError, match=error):
        runner.replay_query_fit_states(states, n_records=n_records)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "state_index": True,
            "round_index": 0,
            "count_errors": (1.0,),
            "cumulative_participating_rows": 0,
        },
        {
            "state_index": 0,
            "round_index": 0,
            "count_errors": (),
            "cumulative_participating_rows": 0,
        },
        {
            "state_index": 0,
            "round_index": 0,
            "count_errors": (float("nan"),),
            "cumulative_participating_rows": 0,
        },
        {
            "state_index": 0,
            "round_index": 0,
            "count_errors": (True,),
            "cumulative_participating_rows": 0,
        },
    ],
)
def test_replay_state_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        runner.QueryFitReplayState(**kwargs)


def test_trace_extraction_ignores_l1_but_checks_same_state_loss() -> None:
    error_rows = [_unqualified_errors()] * 31
    error_rows[5] = _qualified_errors()
    trace, clocks, target, queries = _trace_fixture(error_rows)
    poisoned_l1_trace = copy.deepcopy(trace)
    for index, observation in enumerate(poisoned_l1_trace.observations):
        observation["current_normalized_l1"] = 1000.0 + index

    original = runner.extract_replay_trace(
        trace=trace,
        transition_clocks=clocks,
        target=target,
        queries=queries,
    )
    poisoned = runner.extract_replay_trace(
        trace=poisoned_l1_trace,
        transition_clocks=clocks,
        target=target,
        queries=queries,
    )

    assert original == poisoned
    assert runner.replay_query_fit_states(
        original.replay_states,
        n_records=10,
    ) == runner.replay_query_fit_states(
        poisoned.replay_states,
        n_records=10,
    )

    wrong_loss = copy.deepcopy(trace)
    wrong_loss.observations[7]["current_squared_loss"] += 0.5
    with pytest.raises(ValueError, match="same-state count errors"):
        runner.extract_replay_trace(
            trace=wrong_loss,
            transition_clocks=clocks,
            target=target,
            queries=queries,
        )


def test_trace_extraction_rejects_clock_and_identity_mismatch() -> None:
    trace, clocks, target, queries = _trace_fixture([_unqualified_errors()] * 31)
    wrong_clock = copy.deepcopy(clocks)
    wrong_clock[3]["attempts"][0]["participating_rows"] = 9
    with pytest.raises(ValueError, match="transition clock differ"):
        runner.extract_replay_trace(
            trace=trace,
            transition_clocks=wrong_clock,
            target=target,
            queries=queries,
        )

    wrong_queries = copy.deepcopy(queries)
    wrong_queries[0]["conditions"][0]["value"] = 0
    with pytest.raises(ValueError, match="query identity"):
        runner.extract_replay_trace(
            trace=trace,
            transition_clocks=clocks,
            target=target,
            queries=wrong_queries,
        )


def test_nonformal_smoke_wires_generator_trace_and_checkpoint_replay() -> None:
    test_seed = 999_053_001
    assert test_seed not in {
        seed for family in runner.FAMILIES for seed in family.seeds
    }
    family = runner.FAMILIES[0]
    problem = runner.build_artificial_problem(family)
    case = runner.MatrixCase(
        family=family.name,
        seed=test_seed,
        rho=1.0,
        n_rounds=40,
    )

    _, diagnostics = runner.run_evolution(
        **runner._generator_kwargs(
            problem,
            case,
            n_rounds=case.n_rounds,
            return_final_table=False,
        )
    )
    runner._validate_full_diagnostics(diagnostics, case=case)
    extracted = runner.extract_replay_trace(
        trace=diagnostics["stationarity_trace"],
        transition_clocks=diagnostics["transition_clock_history"],
        target=problem.target,
        queries=problem.queries,
    )
    decision = runner.replay_query_fit_states(
        extracted.replay_states,
        n_records=family.n_records,
    )
    selected_table, details = runner._materialize_selected_checkpoint(
        problem=problem,
        case=case,
        full_diagnostics=diagnostics,
        extracted=extracted,
        decision=decision,
    )

    assert decision.resource_boundary_state_index == 20
    assert decision.resource_boundary_work == 20.0
    assert decision.full_work == 40.0
    assert decision.tail_work_after_resource_boundary == 20.0
    assert len(selected_table) == family.n_records
    assert all(details["checkpoint_replay_checks"].values())
    assert details["offline_l1_bound_check"] is True


def _fake_case_rows(*, qualified: bool) -> list[dict]:
    validity = {
        "reference_target_preflight": True,
        "full_horizon_max_rounds": True,
        "one_accepted_attempt_per_round": True,
        "query_and_transition_clocks_aligned": True,
        "finite_same_checkpoint_metrics": True,
        "resource_boundary_reached": True,
        "minimum_tail_work_reached": True,
        "selected_checkpoint_replayed": True,
        "l1_computed_only_after_selection": True,
    }
    return [
        {
            "family": case.family,
            "seed": case.seed,
            "rho": case.rho,
            "qualified_by_resource_boundary": qualified,
            "tail_work_after_resource_boundary": 10.0,
            "validity": dict(validity),
        }
        for case in runner.formal_cases()
    ]


def test_matrix_acceptance_requires_all_twelve_cases() -> None:
    passing = _fake_case_rows(qualified=True)
    supported = runner._matrix_acceptance(passing)

    assert supported["matrix_identity_pass"] is True
    assert supported["execution_validity_pass"] is True
    assert supported["qualified_by_resource_boundary_count"] == 12
    assert supported["status"] == "candidate_supported"

    failing = copy.deepcopy(passing)
    failing[-1]["qualified_by_resource_boundary"] = False
    rejected = runner._matrix_acceptance(failing)
    assert rejected["qualified_by_resource_boundary_count"] == 11
    assert rejected["scientific_pass"] is False
    assert rejected["status"] == "candidate_failed"
    assert rejected["post_result_retuning_allowed"] is False


def test_matrix_identity_or_validity_failure_is_not_scientific_failure() -> None:
    rows = _fake_case_rows(qualified=True)[:-1]
    result = runner._matrix_acceptance(rows)
    assert result["matrix_identity_pass"] is False
    assert result["execution_validity_pass"] is False
    assert result["status"] == "execution_invalid"

    rows = _fake_case_rows(qualified=True)
    rows[0]["validity"]["selected_checkpoint_replayed"] = False
    result = runner._matrix_acceptance(rows)
    assert result["matrix_identity_pass"] is True
    assert result["execution_validity_pass"] is False
    assert result["status"] == "execution_invalid"


def test_strict_json_rejects_nonfinite_output() -> None:
    with pytest.raises(ValueError):
        runner._strict_json_bytes({"bad": float("nan")})
