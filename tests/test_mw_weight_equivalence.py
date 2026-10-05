"""MW 查询权重（乘性权重聚合，v6）的等价性与合同测试。

v6 结果前协议 §3 的等价性合同：MW 四参数全 None 时，代码必须与改动前
行为逐位一致。由于 fitness_only.py 不在 git 历史中，这里用自洽等价钉死
该性质（互相独立的证据）：

1. 关闭（None）与"启用但 mw_start_round > n_rounds（全程休眠）"两条
   轨迹逐位相同——休眠期权重保持平凡 None，评价路径与关闭状态一致；
2. 中途开启的轨迹在开启前逐位同关闭轨迹、开启后分叉——这正是 v6 前缀
   一致审计（对 v3 冻结产物，1501 轮）依赖的性质；
3. 权重不变量逐轮成立：clip 围栏 [1/cap, cap]、无顶格时归一化均值为 1、
   休眠期恒为全 1；
4. MW 更新纯确定性（不消耗 RNG）：同配置重跑逐位一致，且 RNG 终态与
   关闭轨迹的休眠等价保证已由证据 1 覆盖；
5. fail-closed：部分提供、equal 臂、非 fitness-only、方向场组合、常数
   越域全部拒绝。
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


def _mw_overrides(**extra):
    values = {
        "mw_query_weight_eta": 0.5,
        "mw_signal_cap": 8.0,
        "mw_weight_cap": 8.0,
        "mw_start_round": 0,
    }
    values.update(extra)
    return values


def _run(config, fitness_mode="residual"):
    schema, queries, target = _problem()
    return run_fitness_only_evolution(
        target,
        queries,
        schema,
        12,
        config=config,
        fitness_mode=fitness_mode,
    )


def _table_sha(table):
    payload = table.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_disabled_and_dormant_enabled_trajectories_are_bit_identical():
    """关闭 vs 全程休眠（start_round > n_rounds）：逐位一致。"""

    table_off, diag_off = _run(_config())
    table_dormant, diag_dormant = _run(_config(
        **_mw_overrides(mw_start_round=200)   # > n_rounds，全程休眠
    ))

    assert diag_dormant["loss_history"] == diag_off["loss_history"]
    assert _table_sha(table_dormant) == _table_sha(table_off)
    assert diag_dormant["primary_rng_state_sha256"] == (
        diag_off["primary_rng_state_sha256"]
    )
    # 关闭时不记录任何 MW 历史；休眠时逐轮记录平凡权重（全 1）。
    assert diag_off["mw_weight_stats_history"] == []
    assert diag_off["mw_weight_snapshot_history"] == []
    stats = diag_dormant["mw_weight_stats_history"]
    assert len(stats) == 60
    assert all(
        row["max"] == 1.0 and row["min"] == 1.0 and row["mean"] == 1.0
        for row in stats
    )


def test_midway_activation_preserves_prefix_and_forks_afterwards():
    """中途开启：开启前逐位同关闭轨迹，开启后分叉（v6 审计性质）。

    t = start_round 的轮末才执行首次更新，本轮评价仍用平凡权重，因此
    前缀一致窗口是 [0, start_round] 共 start_round + 1 轮。
    """

    _, diag_off = _run(_config())
    _, diag_mw = _run(_config(**_mw_overrides(mw_start_round=20)))

    assert diag_mw["loss_history"][:21] == diag_off["loss_history"][:21]
    assert diag_mw["loss_history"] != diag_off["loss_history"]
    stats = diag_mw["mw_weight_stats_history"]
    assert len(stats) == 60
    # 开启前平凡；开启轮末已出现非平凡权重。
    assert all(row["max"] == 1.0 for row in stats[:20])
    assert stats[20]["max"] > 1.0 or stats[20]["min"] < 1.0


def test_weight_invariants_hold_every_round():
    """clip 围栏、无顶格时的归一化均值、快照维度逐轮成立。"""

    _, diag = _run(_config(**_mw_overrides()))

    stats = diag["mw_weight_stats_history"]
    assert len(stats) == 60
    for row in stats:
        assert row["max"] <= 8.0 + 1e-12
        assert row["min"] >= 1.0 / 8.0 - 1e-12
        assert row["at_upper_cap"] >= 0
        assert row["at_lower_cap"] >= 0
        if row["at_upper_cap"] == 0 and row["at_lower_cap"] == 0:
            assert row["mean"] == pytest.approx(1.0)
        else:
            # clip 只能把归一化后的均值往下切或抬高，围栏内浮动。
            assert 1.0 / 8.0 <= row["mean"] <= 8.0

    snapshots = diag["mw_weight_snapshot_history"]
    assert snapshots[0]["round"] == 0
    assert snapshots[-1]["round"] == 59
    for snap in snapshots:
        assert len(snap["weights"]) == 4      # 查询数
    # 快照与同轮 stats 相互印证。
    by_round = {row["round"]: row for row in stats}
    for snap in snapshots:
        w = np.asarray(snap["weights"])
        row = by_round[snap["round"]]
        assert float(w.max()) == pytest.approx(row["max"])
        assert float(w.min()) == pytest.approx(row["min"])
        assert float(w.mean()) == pytest.approx(row["mean"])


def test_mw_update_is_deterministic_across_reruns():
    """MW 更新纯确定性：同配置重跑逐位一致（不引入额外随机源）。"""

    table_a, diag_a = _run(_config(**_mw_overrides()))
    table_b, diag_b = _run(_config(**_mw_overrides()))

    assert diag_a["loss_history"] == diag_b["loss_history"]
    assert _table_sha(table_a) == _table_sha(table_b)
    assert diag_a["primary_rng_state_sha256"] == (
        diag_b["primary_rng_state_sha256"]
    )
    assert diag_a["mw_weight_stats_history"] == (
        diag_b["mw_weight_stats_history"]
    )
    assert diag_a["mw_weight_snapshot_history"] == (
        diag_b["mw_weight_snapshot_history"]
    )


def test_params_recorded_in_generation_config():
    _, diag = _run(_config(**_mw_overrides(mw_start_round=20)))
    assert diag["params"]["mw_query_weight_eta"] == 0.5
    assert diag["params"]["mw_signal_cap"] == 8.0
    assert diag["params"]["mw_weight_cap"] == 8.0
    assert diag["params"]["mw_start_round"] == 20

    _, diag_off = _run(_config())
    assert diag_off["params"]["mw_query_weight_eta"] is None
    assert diag_off["params"]["mw_start_round"] is None


def test_config_rejects_partial_mw_params():
    with pytest.raises(ValueError, match="MW 查询权重"):
        _config(mw_query_weight_eta=0.002).validate()
    with pytest.raises(ValueError, match="MW 查询权重"):
        _config(
            mw_signal_cap=8.0, mw_weight_cap=8.0, mw_start_round=0,
        ).validate()


def test_config_rejects_out_of_domain_constants():
    with pytest.raises(ValueError, match="mw_query_weight_eta"):
        _config(**_mw_overrides(mw_query_weight_eta=0.0)).validate()
    with pytest.raises(ValueError, match="mw_signal_cap"):
        _config(**_mw_overrides(mw_signal_cap=-1.0)).validate()
    with pytest.raises(ValueError, match="mw_weight_cap"):
        _config(**_mw_overrides(mw_weight_cap=1.0)).validate()
    with pytest.raises(ValueError, match="mw_start_round"):
        _config(**_mw_overrides(mw_start_round=-1)).validate()


def test_equal_arm_rejects_mw_params():
    """equal 臂 fitness 恒 0，权重无意义——fail-closed 拒绝。"""

    with pytest.raises(ValueError, match="residual"):
        _run(_config(**_mw_overrides()), fitness_mode="equal")


def test_run_evolution_rejects_mw_outside_fitness_only_residual():
    """非 fitness-only 路径禁 MW（评价缓存与权重状态不同步）。"""

    schema, queries, target = _problem()
    with pytest.raises(ValueError, match="fitness_only_mode"):
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
            mw_query_weight_eta=0.002,
            mw_signal_cap=8.0,
            mw_weight_cap=8.0,
            mw_start_round=0,
        )


def test_run_evolution_rejects_mw_with_directed_diffusion():
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
            tol=float("inf"),
            max_retries=0,
            fitness_only_mode="residual",
            residual_directed_diffusion=True,
            diffusion_direction_strength=1.0,
            mw_query_weight_eta=0.002,
            mw_signal_cap=8.0,
            mw_weight_cap=8.0,
            mw_start_round=0,
        )


def test_run_evolution_rejects_partial_mw_params():
    schema, queries, target = _problem()
    with pytest.raises(ValueError, match="全部提供或全部为 None"):
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
            fitness_only_mode="residual",
            mw_query_weight_eta=0.002,
        )
