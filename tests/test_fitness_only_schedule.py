"""预冻结三段式 rho 时间表（保温→几何降温→地板）的专项合同测试。"""

import numpy as np
import pandas as pd
import pytest

import table_diffevo.fitness_only as fitness_only_module
from table_diffevo.evolution import run_evolution
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    build_fitness_only_kwargs,
    run_fitness_only_evolution,
    run_paired_fitness_only_attribution,
)
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
        "n_rounds": 12,
        "seed": 23,
        "init_method": "random",
        "log_every": 100,
        "rho": 0.5,
        "eta": 0.5,
        "mu": 0.01,
        "rho_anneal_start_round": 4,
        "rho_anneal_rounds": 4,
        "rho_anneal_end": 0.05,
    }
    values.update(overrides)
    return FitnessOnlyConfig(**values)


def _expected_schedule(config):
    expected = []
    for t in range(config.n_rounds):
        if config.rho_anneal_end is None:
            expected.append(config.rho)
            continue
        progress = min(
            1.0,
            max(
                0.0,
                (t - config.rho_anneal_start_round)
                / config.rho_anneal_rounds,
            ),
        )
        expected.append(
            config.rho * (config.rho_anneal_end / config.rho) ** progress
        )
    return expected


def test_three_phase_schedule_matches_frozen_formula_per_round():
    """逐轮核对：保温段恒 rho、降温段几何插值、地板段恒 rho_anneal_end。"""

    schema, queries, target = _problem()
    config = _config()

    _, diagnostics = run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode="residual",
    )

    history = diagnostics["rho_schedule_history"]
    assert history == _expected_schedule(config)
    # 三段边界值显式核对（H=4, D=4, T=12）。
    assert history[0] == 0.5
    assert history[3] == 0.5
    assert history[4] == 0.5
    assert history[5] == pytest.approx(0.5 * 0.1 ** 0.25)
    assert history[7] == pytest.approx(0.5 * 0.1 ** 0.75)
    assert history[8] == pytest.approx(0.05)
    assert history[11] == history[8]
    assert diagnostics["params"]["rho_anneal_start_round"] == 4
    assert diagnostics["params"]["rho_anneal_rounds"] == 4
    assert diagnostics["params"]["rho_anneal_end"] == 0.05


def test_schedule_is_pure_time_driven_not_a_gate():
    """时间表只依赖轮次：equal 对照换 target 后轨迹与时间表逐位不变。"""

    schema, queries, target = _problem()
    other_target = np.asarray([1.0, 10.0, 0.0, 2.0])
    config = _config()

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
    assert first_diag["rho_schedule_history"] == second_diag[
        "rho_schedule_history"
    ]
    assert first_diag["primary_rng_state_sha256"] == second_diag[
        "primary_rng_state_sha256"
    ]


def test_paired_arms_share_identical_schedule():
    schema, queries, target = _problem()
    config = _config(seed=91)

    runs, pairing = run_paired_fitness_only_attribution(
        target,
        queries,
        schema,
        12,
        config=config,
    )

    residual_history = runs["residual"][1]["rho_schedule_history"]
    equal_history = runs["equal"][1]["rho_schedule_history"]
    assert residual_history == equal_history
    assert residual_history == _expected_schedule(config)
    assert pairing["shared_rho_schedule"] == {
        "rho_anneal_start_round": 4,
        "rho_anneal_rounds": 4,
        "rho_anneal_end": 0.05,
    }


def test_disabled_schedule_keeps_constant_rho_and_none_kwargs():
    """不启用时间表时 kwargs 全为 None，rho 全程恒定（legacy 行为）。"""

    schema, queries, target = _problem()
    config = _config(
        rho_anneal_start_round=None,
        rho_anneal_rounds=None,
        rho_anneal_end=None,
    )

    kwargs = build_fitness_only_kwargs(config, fitness_mode="residual")
    assert kwargs["rho_anneal_start_round"] is None
    assert kwargs["rho_anneal_rounds"] is None
    assert kwargs["rho_anneal_end"] is None

    _, diagnostics = run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode="residual",
    )
    assert diagnostics["rho_schedule_history"] == [0.5] * config.n_rounds


def test_start_round_zero_matches_legacy_two_phase_bitwise():
    """start=0 必须与既有两段式（start 缺省）逐位一致，保证向后兼容。"""

    schema, queries, target = _problem()
    config = _config(rho_anneal_start_round=0)
    kwargs_with_start = build_fitness_only_kwargs(
        config, fitness_mode="residual"
    )
    kwargs_legacy = dict(kwargs_with_start)
    kwargs_legacy["rho_anneal_start_round"] = None

    table_a, diag_a = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=12,
        **kwargs_with_start,
    )
    table_b, diag_b = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=12,
        **kwargs_legacy,
    )

    pd.testing.assert_frame_equal(table_a, table_b)
    assert diag_a["rho_schedule_history"] == diag_b["rho_schedule_history"]
    assert diag_a["primary_rng_state_sha256"] == diag_b[
        "primary_rng_state_sha256"
    ]


def test_evolution_accepts_absolute_schedule_under_contracts():
    """horizon_invariant + fitness-only 合同下，绝对轮数时间表应放行。"""

    schema, queries, target = _problem()
    config = _config(n_rounds=6)
    kwargs = build_fitness_only_kwargs(config, fitness_mode="residual")
    assert kwargs["horizon_invariant"] is True

    _, diagnostics = run_evolution(
        target=target,
        queries=queries,
        schema=schema,
        n_records=12,
        **kwargs,
    )
    assert diagnostics["rho_schedule_history"][0] == 0.5


def test_full_length_anneal_still_rejected_under_contracts():
    """全程式退火（公式含总轮数）在两道合同下都必须继续 fail closed。"""

    schema, queries, target = _problem()
    config = _config(
        n_rounds=3,
        rho_anneal_start_round=None,
        rho_anneal_rounds=None,
        rho_anneal_end=None,
    )
    kwargs = build_fitness_only_kwargs(config, fitness_mode="residual")
    kwargs["rho_anneal_end"] = 0.05
    kwargs["rho_anneal_rounds"] = None
    kwargs["rho_anneal_start_round"] = None

    with pytest.raises(ValueError, match="绝对轮数"):
        run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=12,
            **kwargs,
        )


def test_evolution_rejects_start_round_without_rounds():
    schema, queries, target = _problem()
    config = _config(
        n_rounds=3,
        rho_anneal_start_round=None,
        rho_anneal_rounds=None,
        rho_anneal_end=None,
    )
    kwargs = build_fitness_only_kwargs(config, fitness_mode="residual")
    kwargs["rho_anneal_start_round"] = 4

    with pytest.raises(ValueError, match="rho_anneal_start_round"):
        run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=12,
            **kwargs,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"rho_anneal_start_round": None},
        {"rho_anneal_rounds": None},
        {"rho_anneal_end": None},
        {"rho_anneal_start_round": -1},
        {"rho_anneal_start_round": True},
        {"rho_anneal_rounds": 0},
        {"rho_anneal_rounds": 4.0},
        {"rho_anneal_end": 0.0},
        {"rho_anneal_end": 0.6},
        {"rho_anneal_end": float("nan")},
    ],
)
def test_config_rejects_partial_or_invalid_schedule(overrides):
    config = _config(**overrides)
    with pytest.raises(ValueError, match="fitness-only 配置验证失败"):
        build_fitness_only_kwargs(config, fitness_mode="residual")


def test_post_run_audit_fails_closed_on_tampered_schedule(monkeypatch):
    """若主循环未来偏离预冻结公式，运行后审计必须拒绝结果。"""

    schema, queries, target = _problem()
    config = _config()
    real_run_evolution = fitness_only_module.run_evolution

    def tampered_run_evolution(*args, **kwargs):
        table, diagnostics = real_run_evolution(*args, **kwargs)
        diagnostics["rho_schedule_history"] = list(
            diagnostics["rho_schedule_history"]
        )
        diagnostics["rho_schedule_history"][6] *= 1.5
        return table, diagnostics

    monkeypatch.setattr(
        fitness_only_module, "run_evolution", tampered_run_evolution
    )

    with pytest.raises(RuntimeError, match="预冻结时间表公式不一致"):
        run_fitness_only_evolution(
            target,
            queries,
            schema,
            12,
            config=config,
            fitness_mode="residual",
        )
