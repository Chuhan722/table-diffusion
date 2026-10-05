"""
半空间不可微查询（type == "halfspace"）引擎扩展测试

对应设计稿：docs/设计/半空间不可微查询能力线设计稿.md 第 3 节（第一档，保正确）。

核心验证：
1. eval_halfspace_mask 与朴素逐行计算逐位一致（含 row-sum 特例）
2. 与合取查询混合评价：evaluate_table 计数一致
3. 向量化路径：halfspace 进回退组，计数/残差/fitness 与旧路径
   （evaluate_table + compute_fitness）逐位一致；分块大小不影响结果
4. 指纹：项顺序不变性、int/float 权重归一、θ 区分度、重复属性拒绝、
   measured/heldout 互斥断言可用
5. fail-closed：字段缺失/长度不齐/未知属性/非数值列/非有限数值一律 ValueError
6. 方向势能路径：halfspace 回退贡献正确、verbose 提示不崩溃
"""
import numpy as np
import pandas as pd
import pytest

from table_diffevo.schema import load_schema
from table_diffevo.queries import (
    eval_halfspace_mask,
    eval_query_mask,
    evaluate_table,
    load_data,
)
from table_diffevo.objective import compute_residual
from table_diffevo.fitness import compute_fitness
from table_diffevo.quality import query_fingerprint, validate_query_partition
from table_diffevo.vectorized_eval import (
    evaluate_directional_potential,
    evaluate_vectorized,
)

NLTCS_SCHEMA = "configs/nltcs/schema.yaml"
NLTCS_DATA = "data/nltcs/nltcs.csv"


def _hs(attrs, weights, theta, qid="HS0001"):
    """构造一条半空间查询（测试用）。"""
    return {
        "id": qid,
        "type": "halfspace",
        "expression": f"halfspace(theta={theta})",
        "halfspace": {
            "attributes": list(attrs),
            "weights": list(weights),
            "theta": theta,
        },
    }


def _conj(attr, value, qid="CJ0001"):
    """构造一条单条件合取查询（测试用）。"""
    return {
        "id": qid,
        "type": "single",
        "expression": f"{attr}=={value}",
        "conditions": [{"attribute": attr, "operator": "==", "value": value}],
    }


def _naive_halfspace_count(df, attrs, weights, theta):
    """朴素逐行参考实现：sum_k w_k * x_k >= theta 的行数。"""
    count = 0
    for _, row in df.iterrows():
        proj = sum(w * float(row[a]) for a, w in zip(attrs, weights))
        if proj >= theta:
            count += 1
    return count


@pytest.fixture(scope="module")
def nltcs_head():
    schema = load_schema(NLTCS_SCHEMA)
    df = load_data(NLTCS_DATA).head(500).reset_index(drop=True)
    return schema, df


class TestHalfspaceMask:
    """掩码与朴素逐行计算一致"""

    def test_mask_matches_naive_random_halfspaces(self, nltcs_head):
        _, df = nltcs_head
        rng = np.random.default_rng(20260905)
        columns = list(df.columns)
        for _ in range(10):
            k = int(rng.integers(2, 8))
            attrs = list(rng.choice(columns, size=k, replace=False))
            weights = [int(w) for w in rng.choice([-1, 1], size=k)]
            theta = int(rng.integers(-2, 5))
            query = _hs(attrs, weights, theta)
            mask = eval_query_mask(df, query)
            assert mask.dtype == bool and mask.shape == (len(df),)
            assert int(mask.sum()) == _naive_halfspace_count(
                df, attrs, weights, theta
            )

    def test_rowsum_special_case(self, nltcs_head):
        _, df = nltcs_head
        columns = list(df.columns)
        theta = int(df.sum(axis=1).median())
        query = _hs(columns, [1] * len(columns), theta)
        mask = eval_halfspace_mask(df, query)
        expected = (df.sum(axis=1).to_numpy() >= theta)
        np.testing.assert_array_equal(mask, expected)
        assert 0 < int(mask.sum()) < len(df), "row-sum 中位数阈值不应退化"

    def test_mixed_evaluate_table(self, nltcs_head):
        _, df = nltcs_head
        queries = [
            _conj("attr_1", 1, "CJ0001"),
            _hs(["attr_1", "attr_2", "attr_3"], [1, 1, 1], 2, "HS0001"),
            _conj("attr_2", 0, "CJ0002"),
        ]
        counts = evaluate_table(df, queries)
        assert counts[0] == int((df["attr_1"] == 1).sum())
        assert counts[1] == _naive_halfspace_count(
            df, ["attr_1", "attr_2", "attr_3"], [1, 1, 1], 2
        )
        assert counts[2] == int((df["attr_2"] == 0).sum())


class TestHalfspaceFingerprint:
    """指纹语义：顺序不变、数值归一、θ 区分、重复拒绝、互斥断言可用"""

    def test_term_order_invariant(self):
        a = _hs(["attr_1", "attr_5"], [1, -1], 3)
        b = _hs(["attr_5", "attr_1"], [-1, 1], 3)
        assert query_fingerprint(a) == query_fingerprint(b)

    def test_int_float_weight_normalized(self):
        a = _hs(["attr_1", "attr_5"], [1, -1], 3)
        b = _hs(["attr_1", "attr_5"], [1.0, -1.0], 3.0)
        assert query_fingerprint(a) == query_fingerprint(b)

    def test_theta_and_weight_distinguish(self):
        base = _hs(["attr_1", "attr_5"], [1, -1], 3)
        other_theta = _hs(["attr_1", "attr_5"], [1, -1], 4)
        other_weight = _hs(["attr_1", "attr_5"], [1, 1], 3)
        assert query_fingerprint(base) != query_fingerprint(other_theta)
        assert query_fingerprint(base) != query_fingerprint(other_weight)

    def test_duplicate_attribute_rejected(self):
        query = _hs(["attr_1", "attr_1"], [1, 1], 2)
        with pytest.raises(ValueError, match="重复属性"):
            query_fingerprint(query)

    def test_no_collision_with_conjunction(self):
        conj = _conj("attr_1", 1)
        hs = _hs(["attr_1"], [1], 1)
        assert query_fingerprint(conj) != query_fingerprint(hs)

    def test_partition_disjointness_assertion(self):
        measured = [_hs(["attr_1", "attr_2"], [1, 1], t, f"M{t}") for t in (1, 2)]
        heldout = [_hs(["attr_1", "attr_2"], [1, 1], t, f"H{t}") for t in (3, 4)]
        summary = validate_query_partition(measured, heldout)
        assert summary["overlap_count"] == 0
        with pytest.raises(ValueError, match="不相交"):
            validate_query_partition(measured, measured)


class TestHalfspaceFailClosed:
    """fail-closed 校验"""

    def test_missing_halfspace_field(self, nltcs_head):
        _, df = nltcs_head
        with pytest.raises(ValueError, match="halfspace 字典字段"):
            eval_query_mask(df, {"type": "halfspace", "id": "HS_BAD"})

    def test_length_mismatch(self, nltcs_head):
        _, df = nltcs_head
        with pytest.raises(ValueError, match="等长"):
            eval_halfspace_mask(df, _hs(["attr_1", "attr_2"], [1], 1))

    def test_duplicate_attribute(self, nltcs_head):
        _, df = nltcs_head
        with pytest.raises(ValueError, match="重复属性"):
            eval_halfspace_mask(df, _hs(["attr_1", "attr_1"], [1, 1], 1))

    def test_unknown_attribute(self, nltcs_head):
        _, df = nltcs_head
        with pytest.raises(ValueError, match="不在表中"):
            eval_halfspace_mask(df, _hs(["attr_999"], [1], 1))

    def test_non_numeric_column(self):
        df = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2]})
        with pytest.raises(ValueError, match="数值列"):
            eval_halfspace_mask(df, _hs(["cat"], [1], 1))

    def test_non_finite_values(self, nltcs_head):
        _, df = nltcs_head
        with pytest.raises(ValueError, match="theta"):
            eval_halfspace_mask(df, _hs(["attr_1"], [1], float("nan")))
        with pytest.raises(ValueError, match="权重"):
            eval_halfspace_mask(df, _hs(["attr_1"], [float("inf")], 1))
        with pytest.raises(ValueError, match="theta"):
            eval_halfspace_mask(df, _hs(["attr_1"], [1], True))


class TestVectorizedFallback:
    """向量化路径：halfspace 回退组与旧路径逐位一致"""

    def _mixed_pool(self, df):
        rng = np.random.default_rng(9908)
        columns = list(df.columns)
        pool = [_conj(a, 1, f"CJ_{a}") for a in columns[:6]]
        for i in range(4):
            k = int(rng.integers(3, 9))
            attrs = list(rng.choice(columns, size=k, replace=False))
            weights = [int(w) for w in rng.choice([-1, 1], size=k)]
            theta = int(rng.integers(0, 3))
            pool.append(_hs(attrs, weights, theta, f"HS{i:04d}"))
        pool.append(_hs(columns, [1] * len(columns), 5, "HS_ROWSUM"))
        target = evaluate_table(df, pool).astype(float) + rng.integers(
            -30, 31, size=len(pool)
        )
        return pool, np.clip(target, 0, len(df))

    def test_counts_residual_fitness_match_legacy(self, nltcs_head, capsys):
        schema, df = nltcs_head
        pool, target = self._mixed_pool(df)
        n = len(df)

        q_new, residual_new, fitness_new = evaluate_vectorized(
            df, pool, schema, target=target, n_records=n,
            want_fitness=True, device="numpy", verbose=True,
        )
        printed = capsys.readouterr().out
        assert "type=halfspace" in printed, "回退提示应标注半空间类型"

        q_old = evaluate_table(df, pool)
        residual_old = compute_residual(target, q_old, n)
        fitness_old = compute_fitness(df, pool, residual_old, q_old)

        np.testing.assert_array_equal(q_new, q_old)
        np.testing.assert_allclose(residual_new, residual_old, atol=1e-12)
        np.testing.assert_allclose(fitness_new, fitness_old, atol=1e-12)

    def test_batch_size_invariance(self, nltcs_head):
        schema, df = nltcs_head
        pool, target = self._mixed_pool(df)
        results = [
            evaluate_vectorized(
                df, pool, schema, target=target, n_records=len(df),
                want_fitness=True, device="numpy", verbose=False,
                batch_size=bs,
            )
            for bs in (3, 256)
        ]
        np.testing.assert_array_equal(results[0][0], results[1][0])
        np.testing.assert_allclose(results[0][2], results[1][2], atol=1e-12)

    def test_counts_only_mode(self, nltcs_head):
        schema, df = nltcs_head
        pool, _ = self._mixed_pool(df)
        q_new, residual, fitness = evaluate_vectorized(
            df, pool, schema, want_fitness=False, device="numpy",
            verbose=False,
        )
        np.testing.assert_array_equal(q_new, evaluate_table(df, pool))
        assert residual is None and fitness is None

    def test_directional_potential_fallback(self, nltcs_head, capsys):
        schema, df = nltcs_head
        pool, target = self._mixed_pool(df)
        n = len(df)
        residual = compute_residual(target, evaluate_table(df, pool), n)
        potential = evaluate_directional_potential(
            df, pool, schema, residual, device="numpy", verbose=True,
        )
        assert "type=halfspace" in capsys.readouterr().out
        expected = np.zeros(n, dtype=float)
        for j, query in enumerate(pool):
            expected += eval_query_mask(df, query).astype(float) * residual[j]
        np.testing.assert_allclose(potential, expected, atol=1e-12)
