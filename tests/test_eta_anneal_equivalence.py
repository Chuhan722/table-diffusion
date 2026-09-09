"""eta 退火（盲复制强度时间表）的等价性与合同测试。

v5 结果前协议 §5.2 的等价性合同：eta 退火三参数全 None 时，代码必须与
改动前行为逐位一致。由于旧版 fitness_only.py 不在 git 历史中，这里用
自洽等价来钉死该性质（三条互相独立的证据）：

1. 关闭（None）与"启用但保温段覆盖全程"两条轨迹逐位相同——启用路径在
   eta_t == eta 时与关闭路径产生完全相同的随机流与状态序列；
2. 中途降温的轨迹在保温段与关闭轨迹逐位相同、降温开始后分叉——这正是
   v5 前缀一致审计（对 v3 冻结产物）依赖的性质；
3. eta_schedule_history 逐轮等于冻结公式，关闭时恒为 eta。
"""

import hashlib

import numpy as np
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_fitness_only_evolution,
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
        "n_rounds": 60,
        "seed": 977,
        "init_method": "random",
        "log_every": 1000,
        "rho": 0.5,
        "eta": 0.5,
        "mu": 0.01,
    }
    values.update(overrides)
    return FitnessOnlyConfig(**values)


def _run(config):
    schema, queries, target = _problem()
    return run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode="residual",
    )


def _table_sha(table):
    payload = table.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_eta_schedule(n_rounds, eta, start, rounds, end):
    values = []
    for t in range(n_rounds):
        progress = min(1.0, max(0.0, (t - start) / rounds))
        values.append(eta * (end / eta) ** progress)
    return values


def test_disabled_and_all_warm_enabled_trajectories_are_bit_identical():
    """关闭 vs 保温段覆盖全程：轨迹、终表、RNG 终态逐位一致。"""

    table_off, diag_off = _run(_config())
    table_warm, diag_warm = _run(_config(
        eta_anneal_start_round=200,   # > n_rounds，全程保温
        eta_anneal_rounds=50,
        eta_anneal_end=0.25,
    ))

    assert diag_warm["loss_history"] == diag_off["loss_history"]
    assert diag_warm["rho_schedule_history"] == (
        diag_off["rho_schedule_history"]
    )
    assert _table_sha(table_warm) == _table_sha(table_off)
    assert diag_warm["primary_rng_state_sha256"] == (
        diag_off["primary_rng_state_sha256"]
    )
    assert diag_warm["eta_schedule_history"] == [0.5] * 60
    assert diag_off["eta_schedule_history"] == [0.5] * 60


def test_midway_cooling_preserves_prefix_and_forks_afterwards():
    """中途降温：保温段逐位同关闭轨迹，降温开始后分叉（v5 审计性质）。"""

    _, diag_off = _run(_config())
    _, diag_cool = _run(_config(
        eta_anneal_start_round=20,
        eta_anneal_rounds=10,
        eta_anneal_end=0.25,
    ))

    # t=20 时 progress=0，eta_t = 0.5 * (0.5)**0.0 == 0.5 浮点精确，
    # 因此前缀一致窗口是 [0, 20] 共 21 轮。
    assert diag_cool["loss_history"][:21] == diag_off["loss_history"][:21]
    assert diag_cool["eta_schedule_history"][:21] == [0.5] * 21
    assert diag_cool["loss_history"] != diag_off["loss_history"]
    assert diag_cool["eta_schedule_history"][21] < 0.5
    assert diag_cool["eta_schedule_history"][-1] == pytest.approx(0.25)


def test_eta_schedule_history_matches_frozen_formula():
    _, diag = _run(_config(
        eta_anneal_start_round=15,
        eta_anneal_rounds=12,
        eta_anneal_end=0.25,
    ))
    expected = _expected_eta_schedule(60, 0.5, 15, 12, 0.25)
    assert diag["eta_schedule_history"] == expected
    assert diag["params"]["eta_anneal_end"] == 0.25
    assert diag["params"]["eta_anneal_rounds"] == 12
    assert diag["params"]["eta_anneal_start_round"] == 15


def test_config_rejects_partial_eta_schedule():
    with pytest.raises(ValueError, match="eta 三段式时间表"):
        _config(eta_anneal_end=0.25).validate()
    with pytest.raises(ValueError, match="eta 三段式时间表"):
        _config(
            eta_anneal_start_round=10, eta_anneal_rounds=5,
        ).validate()


def test_config_rejects_eta_end_out_of_range():
    with pytest.raises(ValueError, match="eta_anneal_end 必须位于"):
        _config(
            eta_anneal_start_round=10,
            eta_anneal_rounds=5,
            eta_anneal_end=0.75,
        ).validate()
    with pytest.raises(ValueError, match="eta_anneal_end 必须位于"):
        _config(
            eta_anneal_start_round=10,
            eta_anneal_rounds=5,
            eta_anneal_end=0.0,
        ).validate()


def test_run_evolution_rejects_eta_anneal_with_directed_diffusion():
    schema, queries, target = _problem()
    with pytest.raises(ValueError, match="residual_directed_diffusion"):
        run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=12,
            n_rounds=5,
            seed=1,
            init_method="random",
            residual_directed_diffusion=True,
            diffusion_direction_strength=1.0,
            eta_anneal_end=0.25,
            eta_anneal_rounds=3,
            eta_anneal_start_round=1,
        )


def test_run_evolution_rejects_eta_end_without_rounds_under_horizon_invariant():
    schema, queries, target = _problem()
    with pytest.raises(ValueError, match="eta_anneal_end"):
        run_evolution(
            target=target,
            queries=queries,
            schema=schema,
            n_records=12,
            n_rounds=5,
            seed=1,
            init_method="random",
            tol=float("inf"),
            max_retries=0,
            horizon_invariant=True,
            eta_anneal_end=0.25,
        )
