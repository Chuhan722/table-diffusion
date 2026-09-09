"""残差引导值分布核（value guidance kernel）合同测试。

机制：被 ρ 点中的参与行，每个属性不再走"η 硬币二选一 + 独立变异事件"，
而是从全值域按 p(v) ∝ base(v)·exp(λ_t·gain(v)) 抽签。base 是旧核的
分布化改写（(1−μc)[η·δ_donor+(1−η)·δ_self] + μc·uniform，μc=μ/A）；
gain 是把该格改成 v 对加权残差账本的一阶增益（与 fitness 同源）；
λ_t 按绝对轮数线性升温（warmup 前 0，warmup 后恒 λ_max）。

合同：
1. gain 恒等式：ValueGainComputer 与暴力逐格参考实现逐位一致
   （gain_a[i,v] = Σ_{j∈J_a,u_j=v} wr_j·M_other_j(i)）。
2. λ=0 分布：value_guided_probabilities 逐位等于旧核每格边缘分布
   （解析断言，不跑采样）；λ>0 向 gain 大的值倾斜且行和恒 1。
3. drop_donor：供体质量并给自值（供体价值对照臂）。
4. fail-closed：非 == / >2-way / halfspace / 数值属性 / 值域外取值在
   构造时抛错；与 tilt / directed / gap / gibbs / MW / eta 退火 /
   max_retries / legacy 评估 / equal 臂互斥；warmup 参数必须成对；
   drop_donor 需要 strength>0。
5. 主循环接线：λ 调度公式核验、gain 每接受轮重算一次、诊断与 params
   回显、lottery 组合跑通、关闭时零痕迹（enabled=False、空历史、
   evolve_step 不收新键）。
6. fitness-only 臂：配置校验镜像、参数透传、运行后 λ 调度审计。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.fitness_only import (
    FitnessOnlyConfig,
    run_fitness_only_evolution,
)
from table_diffevo.marginals import load_marginals
from table_diffevo.objective import compute_residual
from table_diffevo.queries import eval_query_mask
from table_diffevo.schema import load_schema
from table_diffevo.update import (
    evolve_step,
    sample_value_guided_plan,
    value_guided_probabilities,
)
from table_diffevo.vectorized_eval import ValueGainComputer

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def nltcs_setup():
    schema = load_schema(str(ROOT / "configs/nltcs/schema.yaml"))
    marginals = load_marginals(str(ROOT / "configs/nltcs/init_marginals.json"))
    payload = json.loads(
        (ROOT / "configs/nltcs/all2way_issue53_v1.json").read_text()
    )
    queries = list(payload["queries"])[:120]
    target = np.array([float(q["result"]) for q in queries]) / 16181 * 800
    return schema, marginals, queries, target


def _random_table(schema, n, seed):
    rng = np.random.default_rng(seed)
    data = {}
    for name in schema.attribute_names():
        block = schema.get_block(name)
        data[name] = rng.choice(block.values, size=n)
    return pd.DataFrame(data)


def _run_kwargs(schema, marginals, queries, target, **overrides):
    kwargs = dict(
        target=target, queries=queries, schema=schema, n_records=800,
        n_rounds=30, seed=333, rho=0.05, eta=0.5, mu=0.01,
        marginals=marginals, init_method="marginal",
        eval_method="vectorized", device="numpy", log_every=1000,
        distance_mode="geometric", selection_scale_invariant=True,
        residual_geometry="relative", residual_geometry_floor=8.0,
        fitness_only_mode="residual", alpha_schedule_mode="fixed",
        fixed_alpha=16.0, exclude_self=True, stop_on_exact_residual=False,
        horizon_invariant=True, tol=float("inf"),
    )
    kwargs.update(overrides)
    return kwargs


# ---------------------------------------------------------------- gain 恒等式


class TestValueGainComputer:
    def test_gain_matches_bruteforce(self, nltcs_setup):
        schema, _, queries, target = nltcs_setup
        queries = queries[:60]
        target = target[:60]
        df = _random_table(schema, 150, seed=7)
        n_records = len(df)

        computer = ValueGainComputer(
            queries, schema,
            weights=np.ones(len(queries)),
            target=np.asarray(target, dtype=float),
            residual_geometry="relative", residual_geometry_floor=8.0,
        )
        q = np.array(
            [int(eval_query_mask(df, qr).sum()) for qr in queries]
        )
        gains = computer.compute(df, q, n_records)

        wr = compute_residual(
            np.asarray(target, dtype=float), q, n_records,
            geometry="relative", geometry_floor=8.0,
        )
        # 暴力参考：对每个查询条件，other 掩码 × wr 散射到 (attr, value)
        for name in schema.attribute_names():
            block = schema.get_block(name)
            expected = np.zeros((n_records, len(block.values)))
            for j, query in enumerate(queries):
                conds = query["conditions"]
                for pos, c in enumerate(conds):
                    if c["attribute"] != name:
                        continue
                    v_idx = [str(v) for v in block.values].index(
                        str(c["value"])
                    )
                    if len(conds) == 1:
                        other_mask = np.ones(n_records, dtype=float)
                    else:
                        oc = conds[1 - pos]
                        other_mask = (
                            df[oc["attribute"]].to_numpy() == oc["value"]
                        ).astype(float)
                    expected[:, v_idx] += wr[j] * other_mask
            np.testing.assert_allclose(
                gains[name], expected, rtol=0, atol=1e-12,
                err_msg=f"属性 {name} 的 gain 与暴力参考不一致",
            )

    def test_rows_subset_matches_full(self, nltcs_setup):
        schema, _, queries, target = nltcs_setup
        df = _random_table(schema, 120, seed=13)
        computer = ValueGainComputer(
            queries, schema,
            weights=np.ones(len(queries)),
            target=np.asarray(target, dtype=float),
            residual_geometry="relative", residual_geometry_floor=8.0,
        )
        q = np.array(
            [int(eval_query_mask(df, qr).sum()) for qr in queries]
        )
        full = computer.compute(df, q, len(df))
        rows = np.array([3, 17, 44, 99, 100])
        sub = computer.compute(df, q, len(df), rows=rows)
        for name in schema.attribute_names():
            assert sub[name].shape == (len(rows), full[name].shape[1])
            np.testing.assert_array_equal(sub[name], full[name][rows])

    def test_wr_scale_formula_and_failsafe(self, nltcs_setup):
        schema, _, queries, target = nltcs_setup
        computer = ValueGainComputer(
            queries, schema,
            weights=np.ones(len(queries)),
            target=np.asarray(target, dtype=float),
            residual_geometry="relative", residual_geometry_floor=8.0,
        )
        n = 800
        q = np.asarray(target, dtype=float) * 0.9  # 人造欠账
        expected_wr = compute_residual(
            np.asarray(target, dtype=float), q, n,
            geometry="relative", geometry_floor=8.0,
        )
        expected = max(float(np.abs(expected_wr).max()), 1e-12)
        assert computer.wr_scale(q, n) == expected
        # 账全平：|wr| 全 0 → 落底 fail-safe
        q_exact = np.asarray(target, dtype=float)
        assert computer.wr_scale(q_exact, n) == 1e-12

    def test_domains_expose_schema_values(self, nltcs_setup):
        schema, _, queries, target = nltcs_setup
        computer = ValueGainComputer(
            queries, schema,
            weights=np.ones(len(queries)),
            target=np.asarray(target, dtype=float),
        )
        for name in schema.attribute_names():
            assert computer.domains[name] == list(
                schema.get_block(name).values
            )

    def test_fail_closed_structures(self, nltcs_setup):
        schema, _, queries, target = nltcs_setup
        ones = np.ones(3)
        tgt = np.zeros(3)
        base = [dict(q) for q in queries[:3]]

        bad = [dict(q) for q in base]
        bad[0] = {"type": "halfspace", "halfspace": {"attributes": []}}
        with pytest.raises(ValueError, match="halfspace"):
            ValueGainComputer(bad, schema, weights=ones, target=tgt)

        bad = [dict(q) for q in base]
        bad[1] = {
            "conditions": [
                {"attribute": "adl_eating", "operator": ">=", "value": 1},
            ]
        }
        with pytest.raises(ValueError, match="非 == 算子"):
            ValueGainComputer(bad, schema, weights=ones, target=tgt)

        bad = [dict(q) for q in base]
        names = schema.attribute_names()
        bad[2] = {
            "conditions": [
                {"attribute": names[k], "operator": "==", "value": 1}
                for k in range(3)
            ]
        }
        with pytest.raises(ValueError, match="只支持 ≤2-way"):
            ValueGainComputer(bad, schema, weights=ones, target=tgt)

        bad = [dict(q) for q in base]
        bad[0] = {
            "conditions": [
                {"attribute": names[0], "operator": "==", "value": 99},
            ]
        }
        with pytest.raises(ValueError, match="不在属性"):
            ValueGainComputer(bad, schema, weights=ones, target=tgt)


# ---------------------------------------------------------------- 分布合同


class TestValueGuidedProbabilities:
    def _setup(self, seed=0, n=64, v=4):
        rng = np.random.default_rng(seed)
        cur = rng.integers(0, v, size=n)
        don = rng.integers(0, v, size=n)
        gains = rng.normal(size=(n, v))
        return cur, don, gains

    def test_lambda_zero_matches_legacy_cellwise_marginal(self):
        """λ=0 时逐位等于旧核每格边缘分布（独立推导公式，解析断言）。

        旧核参与行属性 a 的每格边缘：先 η 硬币（donor≠self 时以 η 抄，
        否则保持），再以 μ/A 概率整格重抽均匀：
        P(v) = (1−μc)·[η·1{v=donor} + (1−η)·1{v=self}] + μc/V
        （donor==self 时两点质量合并，公式自动成立）。
        """
        cur, don, gains = self._setup(seed=1)
        n, v = gains.shape
        eta, mu_cell = 0.5, 0.01 / 16
        p = value_guided_probabilities(
            cur, don, gains, eta=eta, mu_cell=mu_cell,
            guidance_strength=0.0,
        )
        expected = np.full((n, v), mu_cell / v)
        for i in range(n):
            expected[i, don[i]] += (1 - mu_cell) * eta
            expected[i, cur[i]] += (1 - mu_cell) * (1 - eta)
        np.testing.assert_array_equal(p, expected)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-15)

    def test_guidance_tilts_towards_gain(self):
        cur, don, gains = self._setup(seed=2)
        p0 = value_guided_probabilities(
            cur, don, gains, eta=0.5, mu_cell=0.01, guidance_strength=0.0,
        )
        p4 = value_guided_probabilities(
            cur, don, gains, eta=0.5, mu_cell=0.01, guidance_strength=4.0,
        )
        np.testing.assert_allclose(p4.sum(axis=1), 1.0, atol=1e-12)
        best = gains.argmax(axis=1)
        rows = np.arange(len(cur))
        # 行内最大 gain 的值：λ>0 的概率不低于 λ=0（重加权只会抬升最大者）
        assert np.all(p4[rows, best] >= p0[rows, best] - 1e-12)
        assert p4[rows, best].mean() > p0[rows, best].mean()

    def test_drop_donor_moves_mass_to_self(self):
        cur, don, gains = self._setup(seed=3)
        n, v = gains.shape
        eta, mu_cell = 0.5, 0.02
        p = value_guided_probabilities(
            cur, don, gains, eta=eta, mu_cell=mu_cell,
            guidance_strength=0.0, drop_donor=True,
        )
        expected = np.full((n, v), mu_cell / v)
        rows = np.arange(n)
        expected[rows, cur] += 1 - mu_cell
        np.testing.assert_array_equal(p, expected)


# ---------------------------------------------------------------- 抽样核


class TestSampleValueGuidedPlan:
    def _tiny(self, nltcs_setup, n=50, seed=11):
        schema, _, queries, target = nltcs_setup
        current = _random_table(schema, n, seed=seed)
        donors = _random_table(schema, n, seed=seed + 1)
        computer = ValueGainComputer(
            queries, schema,
            weights=np.ones(len(queries)),
            target=np.asarray(target, dtype=float),
        )
        q = np.array(
            [int(eval_query_mask(current, qr).sum()) for qr in queries]
        )
        gains = computer.compute(current, q, n)
        return schema, current, donors, gains, computer.domains

    def test_nonparticipants_unchanged_and_values_legal(self, nltcs_setup):
        schema, current, donors, gains, domains = self._tiny(nltcs_setup)
        rng = np.random.default_rng(5)
        plan = sample_value_guided_plan(
            current, donors, schema,
            rho=0.3, eta=0.5, mu=0.01, rng=rng,
            value_gains=gains, value_domains=domains,
            guidance_strength=2.0,
        )
        outside = ~plan.participate
        for name in schema.attribute_names():
            col = plan.new_columns[name]
            legal = set(schema.get_block(name).values)
            assert set(col.tolist()) <= legal
            np.testing.assert_array_equal(
                col[outside], current[name].to_numpy()[outside],
            )
        assert not plan.changed_mask[outside].any()

    def test_external_participate_respected(self, nltcs_setup):
        schema, current, donors, gains, domains = self._tiny(nltcs_setup)
        participate = np.zeros(len(current), dtype=bool)
        participate[:7] = True
        plan = sample_value_guided_plan(
            current, donors, schema,
            rho=0.3, eta=0.5, mu=0.01,
            rng=np.random.default_rng(6),
            value_gains=gains, value_domains=domains,
            guidance_strength=0.0, participate=participate,
        )
        np.testing.assert_array_equal(plan.participate, participate)

    def test_callable_gains_matches_dict(self, nltcs_setup):
        schema, current, donors, gains, domains = self._tiny(nltcs_setup)
        calls: list = []

        def gain_fn(rows):
            calls.append(np.asarray(rows).copy())
            return {a: gains[a][rows] for a in gains}

        kwargs = dict(
            rho=0.4, eta=0.5, mu=0.01,
            value_domains=domains, guidance_strength=2.0,
        )
        plan_dict = sample_value_guided_plan(
            current, donors, schema,
            rng=np.random.default_rng(21), value_gains=gains, **kwargs,
        )
        plan_fn = sample_value_guided_plan(
            current, donors, schema,
            rng=np.random.default_rng(21), value_gains=gain_fn, **kwargs,
        )
        # 同 seed 随机流一致（gain 来源不进随机流）→ 结果逐位一致
        np.testing.assert_array_equal(plan_fn.participate, plan_dict.participate)
        np.testing.assert_array_equal(plan_fn.changed_mask, plan_dict.changed_mask)
        for name in schema.attribute_names():
            np.testing.assert_array_equal(
                plan_fn.new_columns[name], plan_dict.new_columns[name]
            )
        assert len(calls) == 1
        np.testing.assert_array_equal(
            calls[0], np.flatnonzero(plan_dict.participate)
        )

    def test_empty_participation_short_circuits(self, nltcs_setup):
        schema, current, donors, gains, domains = self._tiny(nltcs_setup)

        def must_not_call(rows):
            raise AssertionError("参与行为空时不得调用 gain 回调")

        rng = np.random.default_rng(31)
        rng_ref = np.random.default_rng(31)
        plan = sample_value_guided_plan(
            current, donors, schema,
            rho=0.0, eta=0.5, mu=0.01, rng=rng,
            value_gains=must_not_call, value_domains=domains,
            guidance_strength=2.0,
        )
        assert not plan.participate.any()
        assert not plan.changed_mask.any()
        for name in schema.attribute_names():
            np.testing.assert_array_equal(
                plan.new_columns[name], current[name].to_numpy()
            )
        # 只消费 participate 一条带，之后 rng 状态与参考对齐
        rng_ref.random(len(current))
        assert rng.random() == rng_ref.random()

    def test_evolve_step_branch_and_diagnostics(self, nltcs_setup):
        schema, current, donors, gains, domains = self._tiny(nltcs_setup)
        next_table, diag = evolve_step(
            current, donors, schema,
            rho=0.5, eta=0.5, mu=0.01,
            rng=np.random.default_rng(7),
            return_diagnostics=True,
            value_gains=gains, value_domains=domains,
            value_guidance_strength=3.0,
        )
        assert len(next_table) == len(current)
        assert diag["mutated_rows"] == 0
        assert diag["value_guided_changed_cells"] >= 0
        assert diag["participating_rows"] >= 0

    def test_evolve_step_mutual_exclusion(self, nltcs_setup):
        schema, current, donors, gains, domains = self._tiny(nltcs_setup)
        with pytest.raises(ValueError, match="互斥"):
            evolve_step(
                current, donors, schema,
                rng=np.random.default_rng(8),
                copy_direction_scores=np.zeros(
                    (len(current), len(schema.attribute_names()))
                ),
                copy_direction_strength=1.0,
                value_gains=gains, value_domains=domains,
            )
        with pytest.raises(ValueError, match="同时提供"):
            evolve_step(
                current, donors, schema,
                rng=np.random.default_rng(8),
                value_gains=gains,
            )
        with pytest.raises(ValueError, match="只在提供 value_gains"):
            evolve_step(
                current, donors, schema,
                rng=np.random.default_rng(8),
                value_guidance_strength=1.0,
            )


# ---------------------------------------------------------------- 主循环接线


class TestRunEvolutionWiring:
    def test_smoke_lambda_schedule_and_diagnostics(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target,
            value_guidance_strength=4.0,
            value_guidance_warmup_start_round=5,
            value_guidance_warmup_rounds=10,
        ))
        vg = diagnostics["value_guidance"]
        assert vg["enabled"] is True
        assert vg["strength"] == 4.0
        lam = vg["lambda_history"]
        assert len(lam) == 30
        for t, actual in enumerate(lam):
            expected = 4.0 * min(1.0, max(0.0, (t - 5) / 10))
            assert actual == expected, f"第 {t} 轮 λ 调度不符"
        assert lam[0] == 0.0 and lam[-1] == 4.0
        # gain 每接受轮重算一次：次数 = 接受轮数 + 首轮
        accepted = sum(diagnostics["accept_history"])
        assert vg["gain_recompute_count"] <= accepted + 1
        assert vg["structure_compile_elapsed_sec"] >= 0.0
        params = diagnostics["params"]
        assert params["value_guidance_strength"] == 4.0
        assert params["value_guidance_warmup_start_round"] == 5
        assert params["value_guidance_warmup_rounds"] == 10
        assert params["value_guidance_drop_donor"] is False

    def test_constant_lambda_without_warmup(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target,
            n_rounds=8, value_guidance_strength=2.5,
        ))
        assert diagnostics["value_guidance"]["lambda_history"] == [2.5] * 8

    def test_drop_donor_arm_runs(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target,
            n_rounds=8, value_guidance_strength=2.0,
            value_guidance_drop_donor=True,
        ))
        assert diagnostics["value_guidance"]["drop_donor"] is True
        assert diagnostics["params"]["value_guidance_drop_donor"] is True

    def test_lottery_combination_runs(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target,
            n_rounds=8, value_guidance_strength=2.0,
            lottery_first_donor_selection=True,
        ))
        assert diagnostics["value_guidance"]["enabled"] is True
        assert diagnostics["params"]["lottery_first_donor_selection"] is True

    def test_adaptive_scale_wiring(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target,
            n_rounds=12, value_guidance_strength=4.0,
            value_guidance_warmup_start_round=2,
            value_guidance_warmup_rounds=6,
            value_guidance_adaptive_scale=True,
        ))
        vg = diagnostics["value_guidance"]
        assert vg["adaptive_scale"] is True
        scales = vg["scale_history"]
        assert len(scales) == 12
        assert all(np.isfinite(s) and s > 0.0 for s in scales)
        # lambda_history 仍记调度值（λ_eff = λ_t/scale_t 由两者复算）
        for t, actual in enumerate(vg["lambda_history"]):
            assert actual == 4.0 * min(1.0, max(0.0, (t - 2) / 6))
        assert diagnostics["params"]["value_guidance_adaptive_scale"] is True

    def test_adaptive_scale_off_leaves_no_scale_history(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target,
            n_rounds=6, value_guidance_strength=2.0,
        ))
        vg = diagnostics["value_guidance"]
        assert vg["adaptive_scale"] is False
        assert vg["scale_history"] == []
        assert diagnostics["params"]["value_guidance_adaptive_scale"] is False

    def test_disabled_leaves_no_trace(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        _, diagnostics = run_evolution(**_run_kwargs(
            schema, marginals, queries, target, n_rounds=5,
        ))
        vg = diagnostics["value_guidance"]
        assert vg["enabled"] is False
        assert vg["lambda_history"] == []
        assert vg["scale_history"] == []
        assert vg["gain_recompute_count"] == 0
        assert diagnostics["params"]["value_guidance_strength"] == 0.0

    def test_fail_closed_combinations(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(
            schema, marginals, queries, target,
            value_guidance_strength=2.0,
        )
        for overrides, match in [
            (dict(block_score_tilt_strength=1.0), "block_score_tilt"),
            (dict(residual_directed_diffusion=True,
                  diffusion_direction_normalization="fixed",
                  diffusion_direction_reference_scale=1.0),
             "residual_directed_diffusion"),
            (dict(mw_query_weight_eta=0.1, mw_signal_cap=2.0,
                  mw_weight_cap=8.0), "MW"),
            (dict(eta_anneal_end=0.1, eta_anneal_rounds=5,
                  eta_anneal_start_round=0), "eta 退火"),
            (dict(fitness_only_mode="equal"), "equal"),
            (dict(eval_method="legacy"), "vectorized"),
            (dict(value_guidance_warmup_start_round=5), "成对"),
            (dict(value_guidance_warmup_rounds=5), "成对"),
        ]:
            kwargs = dict(base)
            kwargs.update(overrides)
            with pytest.raises(ValueError, match=match):
                run_evolution(**kwargs)
        no_strength = _run_kwargs(
            schema, marginals, queries, target,
            value_guidance_drop_donor=True,
        )
        with pytest.raises(ValueError, match="strength>0"):
            run_evolution(**no_strength)
        no_strength_adaptive = _run_kwargs(
            schema, marginals, queries, target,
            value_guidance_adaptive_scale=True,
        )
        with pytest.raises(ValueError, match="strength>0"):
            run_evolution(**no_strength_adaptive)


# ---------------------------------------------------------------- fitness-only 臂


class TestFitnessOnlyArm:
    def test_config_validation_mirror(self):
        FitnessOnlyConfig(
            n_rounds=10, seed=1, value_guidance_strength=4.0,
            value_guidance_warmup_start_round=5,
            value_guidance_warmup_rounds=10,
        ).validate()
        FitnessOnlyConfig(
            n_rounds=10, seed=1, value_guidance_strength=4.0,
            lottery_first_donor_selection=True,
        ).validate()
        with pytest.raises(ValueError, match="非负有限"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, value_guidance_strength=-1.0,
            ).validate()
        with pytest.raises(ValueError, match="vectorized"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, value_guidance_strength=1.0,
                eval_method="legacy",
            ).validate()
        with pytest.raises(ValueError, match="分科倾斜互斥"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, value_guidance_strength=1.0,
                block_score_tilt_strength=0.5,
            ).validate()
        with pytest.raises(ValueError, match="成对"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, value_guidance_strength=1.0,
                value_guidance_warmup_start_round=5,
            ).validate()
        with pytest.raises(ValueError, match="strength>0"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, value_guidance_drop_donor=True,
            ).validate()
        with pytest.raises(ValueError, match="strength>0"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, value_guidance_adaptive_scale=True,
            ).validate()
        FitnessOnlyConfig(
            n_rounds=10, seed=1, value_guidance_strength=4.0,
            value_guidance_adaptive_scale=True,
        ).validate()

    def test_run_with_guidance_passes_audit(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        config = FitnessOnlyConfig(
            n_rounds=15, seed=42, rho=0.05,
            value_guidance_strength=3.0,
            value_guidance_warmup_start_round=3,
            value_guidance_warmup_rounds=6,
        )
        table, diagnostics = run_fitness_only_evolution(
            target, queries, schema, 800,
            config=config, fitness_mode="residual", marginals=marginals,
        )
        vg = diagnostics["value_guidance"]
        assert vg["enabled"] is True
        assert len(vg["lambda_history"]) == 15
        assert vg["lambda_history"][-1] == 3.0
        assert len(table) == 800

    def test_run_with_adaptive_scale_passes_audit(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        config = FitnessOnlyConfig(
            n_rounds=12, seed=42, rho=0.05,
            value_guidance_strength=3.0,
            value_guidance_adaptive_scale=True,
        )
        _, diagnostics = run_fitness_only_evolution(
            target, queries, schema, 800,
            config=config, fitness_mode="residual", marginals=marginals,
        )
        vg = diagnostics["value_guidance"]
        assert vg["adaptive_scale"] is True
        assert len(vg["scale_history"]) == 12
        assert all(s > 0.0 for s in vg["scale_history"])

    def test_run_without_guidance_unchanged(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        config = FitnessOnlyConfig(n_rounds=8, seed=42, rho=0.05)
        _, diagnostics = run_fitness_only_evolution(
            target, queries, schema, 800,
            config=config, fitness_mode="residual", marginals=marginals,
        )
        assert diagnostics["value_guidance"]["enabled"] is False
