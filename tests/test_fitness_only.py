"""fitness-only 单通道残差合同与配对归因测试。"""

import numpy as np
import pandas as pd
import pytest

import table_diffevo.evolution as evolution_module
from table_diffevo.evolution import run_evolution
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    build_fitness_only_kwargs,
    run_fitness_only_evolution,
    run_paired_fitness_only_attribution,
)
from table_diffevo.objective import compute_loss
from table_diffevo.queries import evaluate_table
from table_diffevo.schema import AttributeBlock, Schema


def _problem():
    schema = Schema([
        AttributeBlock(
            name="a",
            type="categorical",
            description="a",
            values=["0", "1"],
        ),
        AttributeBlock(
            name="b",
            type="categorical",
            description="b",
            values=["0", "1"],
        ),
        AttributeBlock(
            name="c",
            type="categorical",
            description="c",
            values=["0", "1"],
        ),
    ])
    queries = [
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
        ]},
        {"conditions": [
            {"attribute": "b", "operator": "==", "value": "1"},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
            {"attribute": "b", "operator": "==", "value": "1"},
        ]},
        {"conditions": [
            {"attribute": "a", "operator": "==", "value": "1"},
            {"attribute": "c", "operator": "==", "value": "1"},
        ]},
    ]
    target = np.asarray([8.0, 3.0, 3.0, 6.0])
    return schema, queries, target


def _config(**overrides):
    values = {
        "n_rounds": 7,
        "seed": 23,
        "init_method": "random",
        "log_every": 100,
        "rho": 0.5,
        "eta": 0.5,
        "mu": 0.01,
    }
    values.update(overrides)
    return FitnessOnlyConfig(**values)


def test_residual_arm_is_blind_unconditional_terminal_current():
    schema, queries, target = _problem()
    config = _config()

    table, diagnostics = run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode="residual",
    )

    contract = diagnostics["fitness_only_contract"]
    assert contract == {
        "enabled": True,
        "fitness_mode": "residual",
        "residual_driving_channels": ["fitness"],
        "donor_selection": "residual_fitness_and_structure",
        "transition_kernel": "blind_independent",
        "proposal_transition": "unconditional",
        "termination_rule": "fixed_n_rounds",
        "output_identity": "terminal_current",
    }
    assert diagnostics["output_table_identity"] == "terminal_current"
    assert diagnostics["rounds_run"] == config.n_rounds
    assert diagnostics["candidate_evaluation_count"] == config.n_rounds
    assert diagnostics["termination_reason"] == "max_rounds"
    assert diagnostics["accept_history"] == [True] * config.n_rounds
    assert diagnostics["direction_evaluation_count"] == 0
    assert diagnostics["factorized_gibbs_microsteps"] == 0
    assert diagnostics["gap_l1_microsteps"] == 0
    assert diagnostics["params"]["residual_directed_diffusion"] is False
    assert diagnostics["params"]["stop_on_exact_residual"] is False
    assert diagnostics["params"]["fitness_only_mode"] == "residual"
    final_q = evaluate_table(table, queries)
    assert diagnostics["output_squared_loss"] == compute_loss(target, final_q)


def test_equal_fitness_trajectory_is_invariant_to_target():
    """target 可改变只读指标，但不能改变 equal 对照的任何状态转移。"""

    schema, queries, target = _problem()
    other_target = np.asarray([1.0, 10.0, 0.0, 2.0])
    config = _config(n_rounds=9)

    first, first_diag = run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode="equal",
    )
    second, second_diag = run_fitness_only_evolution(
        other_target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode="equal",
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_diag["initial_table_sha256"] == second_diag[
        "initial_table_sha256"
    ]
    assert first_diag["primary_rng_state_sha256"] == second_diag[
        "primary_rng_state_sha256"
    ]
    assert first_diag["donor_distance_history"] == second_diag[
        "donor_distance_history"
    ]
    assert first_diag["donor_fitness_history"] == [0.0] * config.n_rounds
    assert second_diag["donor_fitness_history"] == [0.0] * config.n_rounds
    assert first_diag["final_current_squared_loss"] != second_diag[
        "final_current_squared_loss"
    ]


def test_fixed_budget_returns_last_state_not_historical_best(monkeypatch):
    schema, queries, _ = _problem()
    target = np.asarray([12.0, 12.0, 12.0, 12.0])
    proposals = [
        pd.DataFrame({"a": ["1"] * 12, "b": ["1"] * 12, "c": ["1"] * 12}),
        pd.DataFrame({"a": ["0"] * 12, "b": ["0"] * 12, "c": ["0"] * 12}),
    ]

    def deterministic_step(*args, **kwargs):
        return proposals.pop(0).copy()

    monkeypatch.setattr(evolution_module, "evolve_step", deterministic_step)
    table, diagnostics = run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=_config(n_rounds=2, seed=7),
        fitness_mode="residual",
    )

    assert diagnostics["best_loss_diagnostic_only"] == 0.0
    assert diagnostics["final_current_squared_loss"] > 0.0
    assert diagnostics["output_squared_loss"] == diagnostics[
        "final_current_squared_loss"
    ]
    pd.testing.assert_frame_equal(table, pd.DataFrame({
        "a": ["0"] * 12,
        "b": ["0"] * 12,
        "c": ["0"] * 12,
    }))


def test_paired_runner_locks_initialization_and_random_addresses():
    schema, queries, target = _problem()
    config = _config(n_rounds=6, seed=91)

    runs, pairing = run_paired_fitness_only_attribution(
        target,
        queries,
        schema,
        12,
        config=config,
    )

    assert set(runs) == {"residual", "equal"}
    assert pairing["paired"] is True
    assert pairing["only_treatment_difference"] == "row_fitness"
    assert pairing["shared_structure_distance"] is True
    assert pairing["shared_blind_independent_kernel"] is True
    assert pairing["fixed_rounds"] == config.n_rounds
    for mode in ("residual", "equal"):
        assert runs[mode][1]["fitness_only_contract"]["fitness_mode"] == mode


@pytest.mark.parametrize(
    "overrides",
    [
        {"horizon_invariant": False},
        {"selection_scale_invariant": False},
        {"alpha_schedule_mode": "legacy_linear_horizon", "fixed_alpha": None},
        {"residual_directed_diffusion": True},
        {"tol": 0.0},
        {"max_retries": 1},
        {"residual_self_cooling": 1.0},
        {"rho_anneal_end": 0.001},
        {"stop_on_exact_residual": True},
        {"candidate_budget": 2},
    ],
)
def test_core_contract_rejects_forbidden_control_paths(overrides):
    schema, queries, target = _problem()
    config = _config(n_rounds=3)
    kwargs = build_fitness_only_kwargs(
        config,
        fitness_mode="residual",
    )
    kwargs.update(overrides)

    with pytest.raises(ValueError):
        run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=12,
            **kwargs,
        )


def test_wrapper_rejects_missing_or_irrelevant_marginals():
    with pytest.raises(ValueError, match="显式提供冻结的 marginals"):
        build_fitness_only_kwargs(
            FitnessOnlyConfig(n_rounds=2, seed=1),
            fitness_mode="residual",
        )
    with pytest.raises(ValueError, match="不允许提供 marginals"):
        build_fitness_only_kwargs(
            FitnessOnlyConfig(
                n_rounds=2,
                seed=1,
                init_method="random",
            ),
            fitness_mode="residual",
            marginals={"unexpected": True},
        )


@pytest.mark.parametrize("bad", ["uniform", "", 1, True])
def test_invalid_fitness_mode_rejected(bad):
    schema, queries, target = _problem()
    config = _config(n_rounds=2)
    with pytest.raises(ValueError, match="fitness_mode"):
        run_fitness_only_evolution(
            target,
            queries,
            schema,
            12,
            config=config,
            fitness_mode=bad,
        )
