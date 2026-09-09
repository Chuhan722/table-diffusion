"""先抽签后选供体（lottery_first_donor_selection）的等价与合同测试。

换位优化只改变计算顺序（先抽参与签，再只对中签行算供体），不改变任何
分布语义。等价性合同分两档：
- numpy / torch-cpu：同种子下开启与关闭逐位一致——轨迹 loss、终表、
  主 RNG 终态全部相同；
- torch-cuda：数值等价（随机流终态逐位一致，loss 轨迹数值接近）。
  float32 行归约顺序随矩阵形状变化，(P,N) 子集与 (N,N) 全表在最后一位
  舍入上偶有差异，属既有 Stage 6 "numpy 逐位 + cuda 数值等价" 惯例。
诊断口径合同：开启后供体诊断只统计中签行
（donor_diagnostics_scope="participants_only"），零中签轮记 None。
守卫合同：与重试、方向倾斜、gap/gibbs 核的组合 fail-closed 拒绝。
"""

import hashlib

import numpy as np
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    build_fitness_only_kwargs,
    run_fitness_only_evolution,
    run_paired_fitness_only_attribution,
)
from table_diffevo.sampling import compute_sampling_probs, sample_donors
from table_diffevo.schema import AttributeBlock, Schema
from table_diffevo.update import sample_update_random_plan


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


def _assert_bit_identical(off, on):
    table_off, diag_off = off
    table_on, diag_on = on
    assert diag_on["loss_history"] == diag_off["loss_history"]
    assert _table_sha(table_on) == _table_sha(table_off)
    assert diag_on["primary_rng_state_sha256"] == (
        diag_off["primary_rng_state_sha256"]
    )
    assert diag_on["initial_table_sha256"] == (
        diag_off["initial_table_sha256"]
    )
    assert diag_on["rho_schedule_history"] == diag_off["rho_schedule_history"]


def test_numpy_bit_identical():
    """numpy 路径：关闭 vs 开启逐位一致。"""

    off = _run(_config())
    on = _run(_config(lottery_first_donor_selection=True))
    _assert_bit_identical(off, on)


def test_numpy_bit_identical_with_rho_anneal():
    """带 rho 三段式退火（复刻正式跑结构）时同样逐位一致。"""

    kwargs = dict(
        n_rounds=80,
        rho=0.5,
        rho_anneal_start_round=20,
        rho_anneal_rounds=30,
        rho_anneal_end=0.05,
    )
    off = _run(_config(**kwargs))
    on = _run(_config(lottery_first_donor_selection=True, **kwargs))
    _assert_bit_identical(off, on)


def test_cpu_torch_bit_identical():
    """torch cpu 路径：关闭 vs 开启逐位一致。"""

    pytest.importorskip("torch")
    off = _run(_config(device="cpu"))
    on = _run(_config(device="cpu", lottery_first_donor_selection=True))
    _assert_bit_identical(off, on)


def test_cuda_numerically_equivalent():
    """torch cuda 路径：不变量合同（不对 loss 差设阈值）。

    CUDA float32 的行归约（标准化均值/方差、softmax 求和、cumsum）切块
    方式随矩阵形状变化：(P, N) 子集与 (N, N) 全表的加法顺序不同，概率
    在最后一位舍入上偶有差异，抽签恰好压中边界时个别行换供体。此后两条
    轨迹按混沌动力学指数分离——loss 逐轮相对差**随轮数增长无上界**（外部
    审查在 nltcs 全规模 400 轮实测最大相对差 1.57%），因此任何 loss 阈值
    合同都只在特定形状/轮数/GPU 上偶然成立，不可承诺。CUDA 合同只断言
    可证明的不变量（分布语义相同的平行轨迹）：

    1. 初始表逐位一致；
    2. 主 RNG 终态逐位一致（随机流消耗的槽位序列相同）；
    3. rho 退火时间表逐轮一致；
    4. **每轮参与行（中签行）集合逐位一致**——中签由主随机流与 rho 决定，
       与供体选择无关，两态必须逐位相同（经 sample_update_random_plan
       的 participate 掩码逐轮 SHA 对账）；
    5. loss 轨迹长度一致且逐轮有限。

    numpy 路径才是逐位合同（见 test_numpy_bitwise_identical）。
    """

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA 不可用")

    from table_diffevo import update as update_module

    original_plan = update_module.sample_update_random_plan

    def _run_with_participation_trace(config):
        trace = []

        def spy(*args, **kwargs):
            plan = original_plan(*args, **kwargs)
            mask = np.asarray(plan.participate, dtype=bool)
            trace.append(
                hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
            )
            return plan

        update_module.sample_update_random_plan = spy
        try:
            table, diag = _run(config)
        finally:
            update_module.sample_update_random_plan = original_plan
        return table, diag, trace

    table_off, diag_off, trace_off = _run_with_participation_trace(
        _config(device="cuda")
    )
    table_on, diag_on, trace_on = _run_with_participation_trace(
        _config(device="cuda", lottery_first_donor_selection=True)
    )
    assert diag_on["initial_table_sha256"] == (
        diag_off["initial_table_sha256"]
    )
    assert diag_on["primary_rng_state_sha256"] == (
        diag_off["primary_rng_state_sha256"]
    )
    assert diag_on["rho_schedule_history"] == diag_off["rho_schedule_history"]
    assert len(trace_on) == len(trace_off) > 0
    assert trace_on == trace_off
    assert len(diag_on["loss_history"]) == len(diag_off["loss_history"])
    for loss_off, loss_on in zip(
        diag_off["loss_history"], diag_on["loss_history"]
    ):
        assert np.isfinite(loss_off) and np.isfinite(loss_on)


def test_zero_participation_rounds_none_diagnostics_and_equivalence():
    """极小 rho 下出现零中签轮：诊断记 None，且与关闭路径逐位一致。"""

    kwargs = dict(n_rounds=40, rho=0.02, seed=1234)
    off = _run(_config(**kwargs))
    on = _run(_config(lottery_first_donor_selection=True, **kwargs))
    _assert_bit_identical(off, on)

    diag_on = on[1]
    donor_fitness = diag_on["donor_fitness_history"]
    assert len(donor_fitness) == diag_on["rounds_run"]
    # rho=0.02、N=12 时 (1-0.02)^12 ≈ 0.78，40 轮几乎必然出现零中签轮
    assert any(value is None for value in donor_fitness), (
        "预期存在零中签轮（None）；若此断言失败请调小 rho 或换种子"
    )
    for hist_key in (
        "donor_fitness_history",
        "donor_distance_history",
        "donor_self_rate_history",
        "donor_top_share_history",
        "row_max_prob_mean_history",
        "row_max_prob_max_history",
        "effective_donors_mean_history",
    ):
        values = diag_on[hist_key]
        assert len(values) == diag_on["rounds_run"]
        assert all(
            value is None or isinstance(value, float) for value in values
        )
    # 零中签轮的位置在所有供体诊断序列上对齐
    none_mask = [value is None for value in donor_fitness]
    assert [
        value is None for value in diag_on["donor_distance_history"]
    ] == none_mask
    assert [
        value is None for value in diag_on["donor_self_rate_history"]
    ] == none_mask


def test_equal_arm_and_pairing_with_lottery():
    """equal 对照 + 配对审计在 lottery 口径（None 轮）下仍通过。"""

    schema, queries, target = _problem()
    config = _config(
        n_rounds=40, rho=0.02, seed=1234,
        lottery_first_donor_selection=True,
    )
    runs, pairing = run_paired_fitness_only_attribution(
        target, queries, schema, 12, config=config,
    )
    assert pairing["paired"] is True
    equal_diag = runs["equal"][1]
    assert all(
        value in (0.0, None)
        for value in equal_diag["donor_fitness_history"]
    )


def test_diagnostics_scope_markers():
    """run_config 必须携带开关与口径标记。"""

    _, diag_off = _run(_config(n_rounds=5))
    _, diag_on = _run(_config(
        n_rounds=5, lottery_first_donor_selection=True,
    ))
    assert diag_off["params"]["lottery_first_donor_selection"] is False
    assert diag_off["params"]["donor_diagnostics_scope"] == "all_rows"
    assert diag_on["params"]["lottery_first_donor_selection"] is True
    assert diag_on["params"]["donor_diagnostics_scope"] == (
        "participants_only"
    )


def test_guards_fail_closed():
    """与重试/方向倾斜/gap/gibbs 的组合必须在运行前报错。"""

    schema, queries, target = _problem()
    base = dict(
        target=target,
        queries=queries,
        schema=schema,
        n_records=12,
        n_rounds=2,
        seed=0,
        lottery_first_donor_selection=True,
    )
    with pytest.raises(ValueError, match="lottery_first_donor_selection"):
        run_evolution(**base, max_retries=1)
    with pytest.raises(ValueError, match="lottery_first_donor_selection"):
        run_evolution(**base, residual_directed_diffusion=True)
    with pytest.raises(ValueError, match="lottery_first_donor_selection"):
        run_evolution(**base, gap_l1_sweeps=1)
    with pytest.raises(ValueError, match="lottery_first_donor_selection"):
        run_evolution(**base, factorized_gibbs_sweeps=1)
    non_bool = dict(base)
    non_bool["lottery_first_donor_selection"] = "yes"
    with pytest.raises(ValueError, match="lottery_first_donor_selection"):
        run_evolution(**non_bool)


def test_config_validation_and_kwargs_passthrough():
    """配置字段类型校验 + kwargs 透传。"""

    with pytest.raises(ValueError, match="lottery_first_donor_selection"):
        _config(lottery_first_donor_selection="yes").validate()
    kwargs = build_fitness_only_kwargs(
        _config(lottery_first_donor_selection=True),
        fitness_mode="residual",
    )
    assert kwargs["lottery_first_donor_selection"] is True
    kwargs_off = build_fitness_only_kwargs(
        _config(), fitness_mode="residual",
    )
    assert kwargs_off["lottery_first_donor_selection"] is False


def test_compute_sampling_probs_subset_rows_bit_identical():
    """子集概率与全表对应行逐位相等（numpy）。"""

    rng = np.random.default_rng(0)
    n = 50
    fitness = rng.normal(size=n)
    dist = rng.random((n, n))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    kwargs = dict(
        distance_mode="geometric", lambda_param=0.5, alpha=16.0,
        delta=0.05, winsorize_quantiles=(0.01, 0.99), exclude_self=True,
        scale_invariant=True, scale_invariant_min_spread=1e-3,
    )
    full = compute_sampling_probs(fitness, dist, **kwargs)
    sub_idx = np.array([0, 7, 23, 49])
    sub = compute_sampling_probs(
        fitness, dist[sub_idx], self_indices=sub_idx, **kwargs
    )
    assert np.array_equal(full[sub_idx], sub)


def test_sample_donors_external_uniforms_match_internal_draws():
    """预抽均匀数路径与内部现场抽逐位一致，且子集对齐全表。"""

    rng = np.random.default_rng(0)
    n = 50
    probs = rng.random((n, n))
    probs /= probs.sum(axis=1, keepdims=True)

    internal = sample_donors(probs, np.random.default_rng(7))
    uniforms = np.random.default_rng(7).uniform(size=n)
    external = sample_donors(probs, rng=None, uniforms=uniforms)
    assert np.array_equal(internal, external)

    sub_idx = np.array([3, 11, 42])
    sub = sample_donors(
        probs[sub_idx], rng=None, uniforms=uniforms[sub_idx]
    )
    assert np.array_equal(internal[sub_idx], sub)

    with pytest.raises(ValueError, match="uniforms"):
        sample_donors(probs, rng=None, uniforms=uniforms[:10])
    with pytest.raises(ValueError, match="uniforms"):
        sample_donors(probs, rng=None, uniforms=uniforms + 1.0)


def test_sample_update_random_plan_external_participate_bit_identical():
    """外部参与签（同流同槽位）与内部抽取的计划逐位一致。"""

    import pandas as pd

    schema, _, _ = _problem()
    rng_table = np.random.default_rng(5)
    frame = pd.DataFrame({
        name: rng_table.integers(0, 2, size=20).astype(str)
        for name in schema.attribute_names()
    })
    donors = frame.sample(
        frac=1.0, random_state=3
    ).reset_index(drop=True)

    rng_a = np.random.default_rng(11)
    plan_a = sample_update_random_plan(
        frame, donors, schema, rho=0.3, eta=0.5, mu=0.05, rng=rng_a,
    )
    rng_b = np.random.default_rng(11)
    participate = rng_b.random(len(frame)) < 0.3
    plan_b = sample_update_random_plan(
        frame, donors, schema, rho=0.3, eta=0.5, mu=0.05, rng=rng_b,
        participate=participate,
    )
    assert np.array_equal(plan_a.participate, plan_b.participate)
    assert np.array_equal(plan_a.initial_copy_mask, plan_b.initial_copy_mask)
    assert plan_a.mutation_events == plan_b.mutation_events
    assert (
        rng_a.bit_generator.state == rng_b.bit_generator.state
    )

    with pytest.raises(ValueError, match="participate"):
        sample_update_random_plan(
            frame, donors, schema, rho=0.3, eta=0.5, mu=0.05,
            rng=np.random.default_rng(0),
            participate=np.ones(3, dtype=bool),
        )
    with pytest.raises(ValueError, match="participate"):
        sample_update_random_plan(
            frame, donors, schema, rho=0.3, eta=0.5, mu=0.05,
            rng=np.random.default_rng(0),
            participate=np.ones(len(frame), dtype=float),
        )
