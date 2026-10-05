"""分科倾斜复制（block-tilted copy）合同测试。

机制：评估阶段顺手输出分科状态分 s_a(z)（行级 fitness 的按属性分解，
纯状态量、零动作预演）；选完供体后取逐 (行, 科) 分差
Δ = s_a(donor) − s_a(recipient)，除以首轮全表 RMS（开局标定一次），
按 logit(p) = logit(η) + strength·Δ/scale 连续倾斜复制硬币，并硬夹
到 [lo, hi] 带内。

合同：
1. 分科分正确性：与逐查询参考实现一致（numpy 数学等价、cuda 数值等价）；
   2-way 查询同时归属两科；默认调用仍返回三元组（旧调用点零影响）。
2. 等价性：strength=0 与不传参数的历史路径逐位一致（表 + loss 轨迹）。
3. 倾斜生效：正分差抬高复制率、负分差压低；夹带把概率硬限在带内。
4. fail-closed：与 residual_directed_diffusion / gap_l1 /
   factorized_gibbs / MW / eta 退火 / legacy 评估 / equal 臂全部互斥；
   bounds 必须含 η；bounds 只允许与倾斜路径联用。
5. fitness-only 臂：配置校验、参数透传、运行后审计。
"""
import hashlib
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
from table_diffevo.generator import init_synthetic_table
from table_diffevo.queries import eval_query_mask
from table_diffevo.objective import compute_residual
from table_diffevo.schema import load_schema
from table_diffevo.update import sample_update_random_plan
from table_diffevo.vectorized_eval import (
    _query_attribute_matrix,
    evaluate_vectorized,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def nltcs_setup():
    schema = load_schema(str(ROOT / "configs/nltcs/schema.yaml"))
    marginals = load_marginals(str(ROOT / "configs/nltcs/init_marginals.json"))
    payload = json.loads(
        (ROOT / "configs/nltcs/all2way_issue53_v1.json").read_text()
    )
    queries = list(payload["queries"])[:120]
    # 缩到小表规模：目标计数按行数比例缩放，保持残差几何量级合理
    target = np.array([float(q["result"]) for q in queries]) / 16181 * 800
    return schema, marginals, queries, target


def _table_digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


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


# ---------------------------------------------------------------- 分科分正确性


class TestBlockScores:
    def _reference_block_scores(self, S, queries, attr_names, target, N):
        """逐查询散射的参考实现（与向量化实现独立）。"""
        q_ref = np.array(
            [eval_query_mask(S, q).sum() for q in queries], dtype=float
        )
        resid = compute_residual(
            target, q_ref, N, geometry="relative", geometry_floor=8.0
        )
        p = q_ref / N
        s_ref = np.zeros((N, len(attr_names)))
        for j, query in enumerate(queries):
            mask = eval_query_mask(S, query).astype(float) - p[j]
            for name in {c["attribute"] for c in query["conditions"]}:
                s_ref[:, attr_names.index(name)] += resid[j] * mask
        return q_ref, s_ref

    def test_matches_reference_numpy(self, nltcs_setup):
        schema, marginals, queries, _ = nltcs_setup
        rng = np.random.default_rng(7)
        N = 500
        S = init_synthetic_table(N, schema, rng, marginals=marginals)
        target = np.array([float(q["result"]) for q in queries]) / 16181 * N
        attr_names = schema.attribute_names()
        q_ref, s_ref = self._reference_block_scores(
            S, queries, attr_names, target, N
        )
        q, _, _, s = evaluate_vectorized(
            S, queries, schema, target=target, n_records=N, device="numpy",
            residual_geometry="relative", residual_geometry_floor=8.0,
            want_block_scores=True, verbose=False,
        )
        assert np.array_equal(q, q_ref.astype(int))
        assert s.shape == (N, len(attr_names))
        # 数学等价（浮点求和顺序差异在 1e-12 内）
        np.testing.assert_allclose(s, s_ref, atol=1e-12, rtol=0)

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available(), reason="需要 CUDA"
    )
    def test_matches_reference_cuda(self, nltcs_setup):
        schema, marginals, queries, _ = nltcs_setup
        rng = np.random.default_rng(7)
        N = 500
        S = init_synthetic_table(N, schema, rng, marginals=marginals)
        target = np.array([float(q["result"]) for q in queries]) / 16181 * N
        attr_names = schema.attribute_names()
        _, s_ref = self._reference_block_scores(
            S, queries, attr_names, target, N
        )
        _, _, _, s = evaluate_vectorized(
            S, queries, schema, target=target, n_records=N, device="cuda",
            residual_geometry="relative", residual_geometry_floor=8.0,
            want_block_scores=True, verbose=False,
        )
        np.testing.assert_allclose(s, s_ref, atol=1e-6, rtol=0)

    def test_two_way_membership_and_unknown_attr(self, nltcs_setup):
        schema, _, queries, _ = nltcs_setup
        attr_names = schema.attribute_names()
        E = _query_attribute_matrix(queries, attr_names)
        orders = np.array([
            len({c["attribute"] for c in q["conditions"]}) for q in queries
        ], dtype=float)
        # 2-way 查询恰好归属两科、1-way 归属一科
        assert np.array_equal(E.sum(axis=1), orders)
        with pytest.raises(ValueError, match="未知属性"):
            _query_attribute_matrix(
                [{"conditions": [
                    {"attribute": "不存在", "operator": "==", "value": 1}
                ]}],
                attr_names,
            )

    def test_halfspace_membership(self, nltcs_setup):
        schema, _, _, _ = nltcs_setup
        attr_names = schema.attribute_names()
        hs = {
            "id": "HS1", "type": "halfspace",
            "halfspace": {
                "attributes": [attr_names[0], attr_names[3]],
                "weights": [1, -1], "theta": 0,
            },
            "result": 1.0,
        }
        E = _query_attribute_matrix([hs], attr_names)
        expected = np.zeros(len(attr_names))
        expected[[0, 3]] = 1.0
        assert np.array_equal(E[0], expected)

    def test_requires_fitness(self, nltcs_setup):
        schema, marginals, queries, _ = nltcs_setup
        rng = np.random.default_rng(7)
        S = init_synthetic_table(100, schema, rng, marginals=marginals)
        with pytest.raises(ValueError, match="want_fitness"):
            evaluate_vectorized(
                S, queries, schema, want_fitness=False,
                want_block_scores=True, verbose=False,
            )

    def test_default_return_stays_triple(self, nltcs_setup):
        schema, marginals, queries, _ = nltcs_setup
        rng = np.random.default_rng(7)
        N = 100
        S = init_synthetic_table(N, schema, rng, marginals=marginals)
        target = np.array([float(q["result"]) for q in queries]) / 16181 * N
        out = evaluate_vectorized(
            S, queries, schema, target=target, n_records=N,
            residual_geometry="relative", verbose=False,
        )
        assert len(out) == 3


# ---------------------------------------------------------------- 等价与生效


class TestTiltEquivalenceAndEffect:
    def test_strength_zero_bitwise_equivalent(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(schema, marginals, queries, target)
        t_head, d_head = run_evolution(**base)
        t_off, d_off = run_evolution(**base, block_score_tilt_strength=0.0)
        assert _table_digest(t_head) == _table_digest(t_off)
        assert d_head["loss_history"] == d_off["loss_history"]
        assert d_off["block_score_tilt"]["enabled"] is False
        assert d_off["block_score_tilt"]["reference_scale"] is None

    def test_tilt_changes_transitions_and_records_diagnostics(
        self, nltcs_setup
    ):
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(schema, marginals, queries, target)
        t_off, _ = run_evolution(**base, block_score_tilt_strength=0.0)
        t_on, d_on = run_evolution(
            **base,
            block_score_tilt_strength=2.0,
            block_score_tilt_bounds=(0.2, 0.8),
        )
        assert _table_digest(t_on) != _table_digest(t_off)
        bt = d_on["block_score_tilt"]
        assert bt["enabled"] is True
        assert bt["strength"] == 2.0
        assert bt["bounds"] == (0.2, 0.8)
        # 参考尺度首轮标定且为正
        assert bt["reference_scale"] is not None
        assert bt["reference_scale"] > 0.0
        # 逐轮诊断与轮数对齐
        n_rounds = base["n_rounds"]
        assert len(bt["mean_abs_scaled_delta_history"]) == n_rounds
        assert len(bt["clip_lo_rate_history"]) == n_rounds
        assert len(bt["clip_hi_rate_history"]) == n_rounds
        assert all(
            0.0 <= v <= 1.0
            for v in bt["clip_lo_rate_history"] + bt["clip_hi_rate_history"]
        )
        # params 回显
        assert d_on["params"]["block_score_tilt_strength"] == 2.0
        assert d_on["params"]["block_score_tilt_bounds"] == (0.2, 0.8)

    def test_lottery_combo_zero_strength_bitwise_equivalent(
        self, nltcs_setup
    ):
        """lottery×tilt 兼容合同：strength=0 与 lottery 旧路径逐位一致。"""
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(
            schema, marginals, queries, target,
            lottery_first_donor_selection=True,
        )
        t_head, d_head = run_evolution(**base)
        t_off, d_off = run_evolution(**base, block_score_tilt_strength=0.0)
        assert _table_digest(t_head) == _table_digest(t_off)
        assert d_head["loss_history"] == d_off["loss_history"]

    def test_lottery_combo_tilt_active(self, nltcs_setup):
        """lottery×tilt：跑通、标定成功、participants_only 诊断口径。"""
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(
            schema, marginals, queries, target,
            lottery_first_donor_selection=True,
        )
        t_off, _ = run_evolution(**base)
        t_on, d_on = run_evolution(
            **base,
            block_score_tilt_strength=2.0,
            block_score_tilt_bounds=(0.2, 0.8),
        )
        bt = d_on["block_score_tilt"]
        assert bt["enabled"] is True
        assert bt["reference_scale"] is not None
        assert bt["reference_scale"] > 0.0
        assert bt["diagnostics_scope"] == "participants_only"
        assert _table_digest(t_on) != _table_digest(t_off)
        n_rounds = base["n_rounds"]
        assert len(bt["mean_abs_scaled_delta_history"]) == n_rounds
        # 标定尺度与全表路径同量级（同一 seed、同一初始表；中签行是
        # 均匀抽样，RMS 期望一致，容忍 3x 带宽）
        _, d_full = run_evolution(
            **_run_kwargs(schema, marginals, queries, target),
            block_score_tilt_strength=2.0,
            block_score_tilt_bounds=(0.2, 0.8),
        )
        full_scale = d_full["block_score_tilt"]["reference_scale"]
        assert full_scale is not None
        ratio = bt["reference_scale"] / full_scale
        assert 1 / 3 < ratio < 3


# ---------------------------------------------------------------- update 夹带


class TestCopyProbabilityBounds:
    def _toy(self, n=4000):
        schema = load_schema(str(ROOT / "configs/nltcs/schema.yaml"))
        attr_names = schema.attribute_names()
        rng = np.random.default_rng(11)
        current = pd.DataFrame(
            {a: np.zeros(n, dtype=int) for a in attr_names}
        )
        donors = pd.DataFrame(
            {a: np.ones(n, dtype=int) for a in attr_names}
        )
        return schema, attr_names, current, donors

    def test_bounds_clip_effective(self):
        schema, attr_names, current, donors = self._toy()
        n = len(current)
        # 巨大正分差：无夹带时概率≈1，夹带后应压回 hi=0.7
        scores = np.full((n, len(attr_names)), 50.0)
        plan = sample_update_random_plan(
            current, donors, schema, rho=1.0, eta=0.5, mu=0.0,
            rng=np.random.default_rng(5),
            copy_direction_scores=scores, copy_direction_strength=1.0,
            copy_probability_bounds=(0.3, 0.7),
        )
        rate = plan.initial_copy_mask.mean()
        assert abs(rate - 0.7) < 0.02, f"复制率 {rate} 应≈夹带上限 0.7"
        # 巨大负分差 → 压回 lo=0.3
        plan_neg = sample_update_random_plan(
            current, donors, schema, rho=1.0, eta=0.5, mu=0.0,
            rng=np.random.default_rng(6),
            copy_direction_scores=-scores, copy_direction_strength=1.0,
            copy_probability_bounds=(0.3, 0.7),
        )
        rate_neg = plan_neg.initial_copy_mask.mean()
        assert abs(rate_neg - 0.3) < 0.02, f"复制率 {rate_neg} 应≈夹带下限 0.3"

    def test_zero_scores_stay_neutral_inside_bounds(self):
        schema, attr_names, current, donors = self._toy()
        n = len(current)
        plan = sample_update_random_plan(
            current, donors, schema, rho=1.0, eta=0.5, mu=0.0,
            rng=np.random.default_rng(7),
            copy_direction_scores=np.zeros((n, len(attr_names))),
            copy_direction_strength=1.0,
            copy_probability_bounds=(0.3, 0.7),
        )
        rate = plan.initial_copy_mask.mean()
        assert abs(rate - 0.5) < 0.02, f"零分差复制率 {rate} 应≈η=0.5"

    def test_bounds_without_tilt_rejected(self):
        schema, attr_names, current, donors = self._toy(n=50)
        with pytest.raises(ValueError, match="倾斜路径"):
            sample_update_random_plan(
                current, donors, schema, rho=0.5, eta=0.5, mu=0.0,
                rng=np.random.default_rng(8),
                copy_probability_bounds=(0.3, 0.7),
            )

    def test_bounds_must_contain_eta(self):
        schema, attr_names, current, donors = self._toy(n=50)
        scores = np.ones((50, len(attr_names)))
        with pytest.raises(ValueError, match="lo ≤ eta ≤ hi"):
            sample_update_random_plan(
                current, donors, schema, rho=0.5, eta=0.5, mu=0.0,
                rng=np.random.default_rng(9),
                copy_direction_scores=scores, copy_direction_strength=1.0,
                copy_probability_bounds=(0.6, 0.9),
            )


# ---------------------------------------------------------------- fail-closed


class TestMutualExclusions:
    @pytest.mark.parametrize("bad_overrides,match", [
        ({"eval_method": "legacy"}, "vectorized"),
        ({"max_retries": 1}, "max_retries"),
        (
            {
                "mw_query_weight_eta": 0.1,
                "mw_signal_cap": 1.0,
                "mw_weight_cap": 8.0,
                "mw_start_round": 0,
            },
            "MW",
        ),
        (
            {
                "eta_anneal_end": 0.4,
                "eta_anneal_rounds": 5,
                "eta_anneal_start_round": 0,
            },
            "eta 退火",
        ),
        ({"fitness_only_mode": "equal"}, "equal"),
    ])
    def test_rejected_combinations(
        self, nltcs_setup, bad_overrides, match
    ):
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(
            schema, marginals, queries, target,
            block_score_tilt_strength=1.0,
            **bad_overrides,
        )
        with pytest.raises(ValueError, match=match):
            run_evolution(**base)

    def test_negative_strength_rejected(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(
            schema, marginals, queries, target,
            block_score_tilt_strength=-1.0,
        )
        with pytest.raises(ValueError, match="非负有限"):
            run_evolution(**base)

    def test_bounds_not_containing_eta_rejected(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        base = _run_kwargs(
            schema, marginals, queries, target,
            block_score_tilt_strength=1.0,
            block_score_tilt_bounds=(0.6, 0.9),
        )
        with pytest.raises(ValueError, match="lo ≤ eta ≤ hi"):
            run_evolution(**base)


# ---------------------------------------------------------------- fitness-only 臂


class TestFitnessOnlyArm:
    def test_config_validation(self):
        ok = FitnessOnlyConfig(
            n_rounds=10, seed=1, block_score_tilt_strength=0.5,
            block_score_tilt_bounds=(0.3, 0.7),
        )
        ok.validate()
        with pytest.raises(ValueError, match="非负有限"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, block_score_tilt_strength=-0.5,
            ).validate()
        # lottery×tilt 兼容化后：组合配置合法（子集口径分差 + 顺延标定）
        FitnessOnlyConfig(
            n_rounds=10, seed=1, block_score_tilt_strength=0.5,
            lottery_first_donor_selection=True,
        ).validate()
        with pytest.raises(ValueError, match="lo ≤ eta ≤ hi"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, block_score_tilt_strength=0.5,
                block_score_tilt_bounds=(0.6, 0.9),
            ).validate()
        with pytest.raises(ValueError, match="vectorized"):
            FitnessOnlyConfig(
                n_rounds=10, seed=1, block_score_tilt_strength=0.5,
                eval_method="legacy",
            ).validate()

    def test_run_with_tilt_passes_audit(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        config = FitnessOnlyConfig(
            n_rounds=15, seed=42, rho=0.05,
            block_score_tilt_strength=0.5,
            block_score_tilt_bounds=(0.3, 0.7),
        )
        table, diagnostics = run_fitness_only_evolution(
            target, queries, schema, 800,
            config=config, fitness_mode="residual", marginals=marginals,
        )
        bt = diagnostics["block_score_tilt"]
        assert bt["enabled"] is True
        assert bt["reference_scale"] is not None
        assert len(table) == 800

    def test_run_without_tilt_unchanged_and_audited(self, nltcs_setup):
        schema, marginals, queries, target = nltcs_setup
        config = FitnessOnlyConfig(n_rounds=10, seed=42, rho=0.05)
        _, diagnostics = run_fitness_only_evolution(
            target, queries, schema, 800,
            config=config, fitness_mode="residual", marginals=marginals,
        )
        assert diagnostics["block_score_tilt"]["enabled"] is False
