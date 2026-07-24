"""
marginals 模块测试：造箱、确定性填充、每列打乱、复现性、回退、端到端。

设计见 docs/设计/初始化设计_1way边缘确定性初始化.md。
"""
import numpy as np
import pandas as pd
import pytest

from table_diffevo.schema import Schema, AttributeBlock, load_schema
from table_diffevo.queries import load_queries, eval_query
from table_diffevo.marginals import (
    derive_bins_from_queries,
    init_from_marginals,
    _largest_remainder_quota,
    _cut_points_from_condition,
)


# ---------- 测试用小 schema ----------

def _toy_schema():
    """1 个数值属性 age[18,100] + 1 个类别属性 color。"""
    return Schema([
        AttributeBlock(name="age", type="numeric", description="", range=[18, 100]),
        AttributeBlock(name="color", type="categorical", description="",
                       values=["red", "green", "blue"]),
    ])


def _q(qid, conditions):
    return {"id": qid, "type": "x", "conditions": conditions, "result": 0}


# ========== 1. 造箱：切点与端点语义 ==========

class TestCutPoints:
    def test_ge(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": ">=", "value": 65}) == [65]

    def test_gt(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": ">", "value": 65}) == [66]

    def test_le(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": "<=", "value": 65}) == [66]

    def test_lt(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": "<", "value": 65}) == [65]

    def test_between(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": "between", "lower": 25, "upper": 34}
        ) == [25, 35]

    def test_eq(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": "==", "value": 40}) == [40, 41]

    def test_unknown_op_returns_none(self):
        assert _cut_points_from_condition(
            {"attribute": "age", "operator": "in", "value": [1, 2]}) is None


class TestDeriveBins:
    def test_bins_cover_domain_and_are_contiguous(self):
        """箱必须铺满整个域、无缝无叠。"""
        schema = _toy_schema()
        queries = [_q("a", [{"attribute": "age", "operator": ">=", "value": 65}])]
        bins = derive_bins_from_queries(queries, schema, verbose=False)["age"]
        # 覆盖 [18,100]
        assert bins[0][0] == 18
        assert bins[-1][1] == 100
        # 相邻无缝：下一箱 lo == 上一箱 hi + 1
        for (lo1, hi1), (lo2, hi2) in zip(bins, bins[1:]):
            assert lo2 == hi1 + 1

    def test_between_and_ge_produce_expected_bins(self):
        """18-24 / 25-34 / >=65 三条 → 切点 {18,25,35,65,101} → 4 个箱。"""
        schema = _toy_schema()
        queries = [
            _q("a", [{"attribute": "age", "operator": "between", "lower": 18, "upper": 24}]),
            _q("b", [{"attribute": "age", "operator": "between", "lower": 25, "upper": 34}]),
            _q("c", [{"attribute": "age", "operator": ">=", "value": 65}]),
        ]
        bins = derive_bins_from_queries(queries, schema, verbose=False)["age"]
        assert bins == [(18, 24), (25, 34), (35, 64), (65, 100)]

    def test_numeric_attr_without_query_is_omitted(self):
        """数值属性无查询覆盖 → 不在返回值里（运行时回退随机）。"""
        schema = _toy_schema()
        queries = [_q("a", [{"attribute": "color", "operator": "==", "value": "red"}])]
        bins = derive_bins_from_queries(queries, schema, verbose=False)
        assert "age" not in bins

    def test_unknown_operator_skipped_not_crash(self):
        """白名单外算子跳过该条件，其余照常造箱，不报错。"""
        schema = _toy_schema()
        queries = [
            _q("a", [{"attribute": "age", "operator": "in", "value": [1, 2]}]),
            _q("b", [{"attribute": "age", "operator": ">=", "value": 50}]),
        ]
        bins = derive_bins_from_queries(queries, schema, verbose=False)["age"]
        # in 被跳过，只有 >=50 生效 → 两个箱
        assert bins == [(18, 49), (50, 100)]

    def test_out_of_domain_cut_clipped(self):
        """域外切点裁剪到域内，不产生非法箱。"""
        schema = _toy_schema()
        queries = [_q("a", [{"attribute": "age", "operator": ">=", "value": 200}])]
        bins = derive_bins_from_queries(queries, schema, verbose=False)["age"]
        # 切点 200 裁到 101 → 与域端点重合 → 单个箱 [18,100]
        assert bins == [(18, 100)]


# ========== 2. 凑 N：最大余数法 ==========

class TestLargestRemainderQuota:
    def test_integer_counts_sum_to_n_unchanged(self):
        """无噪时计数是和为 N 的整数 → 原样返回。"""
        counts = np.array([24, 57, 81, 138])  # 和=300
        quota = _largest_remainder_quota(counts, 300)
        assert quota.sum() == 300
        assert list(quota) == [24, 57, 81, 138]

    def test_fractional_counts_round_to_exact_n(self):
        """带小数计数 → 凑成正好 N。"""
        counts = np.array([10.4, 20.4, 69.2])  # 和=100
        quota = _largest_remainder_quota(counts, 100)
        assert quota.sum() == 100
        assert quota.dtype.kind in "iu"

    def test_counts_not_summing_to_n_rescaled(self):
        """计数总量 != N（加噪可能出现）→ 归一化到 N。"""
        counts = np.array([1.0, 1.0, 2.0])  # 和=4，但要凑 100
        quota = _largest_remainder_quota(counts, 100)
        assert quota.sum() == 100
        # 比例大致 1:1:2
        assert quota[2] > quota[0]

    def test_negative_counts_clipped(self):
        """负计数（加噪可能出现）截到 0。"""
        counts = np.array([-5.0, 50.0, 55.0])
        quota = _largest_remainder_quota(counts, 100)
        assert quota.sum() == 100
        assert quota[0] == 0

    def test_all_zero_counts_uniform(self):
        """全 0（无信息）→ 均摊，和仍为 N。"""
        counts = np.array([0.0, 0.0, 0.0])
        quota = _largest_remainder_quota(counts, 10)
        assert quota.sum() == 10


# ========== 3. 确定性填充 + 每列打乱 + 复现性 ==========

def _toy_marginals():
    """age 4 箱 + color 3 值，N=300。"""
    return {
        "dataset": "toy", "n_records": 300, "queries_source": "x",
        "attributes": {
            "age": {"type": "numeric",
                    "bins": [[18, 24], [25, 34], [35, 64], [65, 100]],
                    "counts": [24, 57, 81, 138]},
            "color": {"type": "categorical",
                      "values": ["red", "green", "blue"],
                      "counts": [100, 100, 100]},
        },
    }


class TestInitFromMarginals:
    def test_shape_and_columns(self):
        schema = _toy_schema()
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, _toy_marginals(), rng)
        assert S.shape == (300, 2)
        assert list(S.columns) == ["age", "color"]

    def test_categorical_marginal_exact(self):
        """类别列每个值的计数精确等于目标。"""
        schema = _toy_schema()
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, _toy_marginals(), rng)
        vc = S["color"].value_counts()
        assert vc["red"] == 100 and vc["green"] == 100 and vc["blue"] == 100

    def test_numeric_bin_marginal_exact(self):
        """数值列每个箱的人数精确等于目标（箱内均摊不影响箱计数）。"""
        schema = _toy_schema()
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, _toy_marginals(), rng)
        age = S["age"]
        assert ((age >= 18) & (age <= 24)).sum() == 24
        assert ((age >= 25) & (age <= 34)).sum() == 57
        assert ((age >= 35) & (age <= 64)).sum() == 81
        assert ((age >= 65) & (age <= 100)).sum() == 138

    def test_values_within_domain(self):
        schema = _toy_schema()
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, _toy_marginals(), rng)
        assert S["age"].between(18, 100).all()
        assert S["color"].isin(["red", "green", "blue"]).all()

    def test_reproducible(self):
        """同 seed → 完全相同的 S_0。"""
        schema = _toy_schema()
        S1 = init_from_marginals(300, schema, _toy_marginals(), np.random.default_rng(7))
        S2 = init_from_marginals(300, schema, _toy_marginals(), np.random.default_rng(7))
        pd.testing.assert_frame_equal(S1, S2)

    def test_columns_shuffled_independently(self):
        """每列独立打乱：列不应是块状，且列间无人为关联。"""
        schema = _toy_schema()
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, _toy_marginals(), rng)
        # 若未打乱，前 24 行 age 全在 [18,24]；打乱后应被打散
        head_ages = S["age"].iloc[:24]
        assert not ((head_ages >= 18) & (head_ages <= 24)).all()


# ========== 4. 回退：无边缘信息的属性走随机 ==========

class TestFallback:
    def test_missing_attr_falls_back_to_random(self):
        """marginals 里没有的属性 → 随机填充，仍落在合法域，不报错。"""
        schema = _toy_schema()
        marg = {"attributes": {  # 只给 color，不给 age
            "color": {"type": "categorical",
                      "values": ["red", "green", "blue"],
                      "counts": [100, 100, 100]},
        }}
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, marg, rng)
        assert S["age"].between(18, 100).all()          # 随机但合法
        assert S["color"].value_counts()["red"] == 100  # color 仍精确

    def test_empty_marginals_all_random_but_legal(self):
        """完全没有 attributes → 全列随机，全部合法。"""
        schema = _toy_schema()
        rng = np.random.default_rng(0)
        S = init_from_marginals(300, schema, {"attributes": {}}, rng)
        assert S.shape == (300, 2)
        assert S["age"].between(18, 100).all()
        assert S["color"].isin(["red", "green", "blue"]).all()


# ========== 5. 真实数据端到端（test_300x10）==========

class TestRealDataEndToEnd:
    def test_bins_align_with_age_queries(self):
        """test_300x10 的 age 查询阈值应生成对齐的箱。"""
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        bins = derive_bins_from_queries(queries, schema, verbose=False)
        assert "age" in bins
        # 覆盖全域、无缝
        assert bins["age"][0][0] == 18
        assert bins["age"][-1][1] == 100

    def test_init_hits_1way_age_queries(self):
        """按边缘初始化后，单属性 age 查询应精确命中目标（1-way 已烤进初始化）。"""
        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        df = pd.read_csv("data/test_300x10/test_300x10.csv")
        n = len(df)

        # 现场造边缘（等价于 build_marginals 的逻辑，但不落盘）
        bins = derive_bins_from_queries(queries, schema, verbose=False)
        attributes = {}
        for attr in schema.attributes:
            if attr.is_numeric():
                if attr.name not in bins:
                    continue
                counts = [int(((df[attr.name] >= lo) & (df[attr.name] <= hi)).sum())
                          for lo, hi in bins[attr.name]]
                attributes[attr.name] = {"type": "numeric",
                                         "bins": [[lo, hi] for lo, hi in bins[attr.name]],
                                         "counts": counts}
            else:
                counts = [int((df[attr.name].astype(str) == str(v)).sum())
                          for v in attr.values]
                attributes[attr.name] = {"type": "categorical",
                                         "values": list(attr.values), "counts": counts}
        marg = {"attributes": attributes}

        S = init_from_marginals(n, schema, marg, np.random.default_rng(0))

        # 单属性 age 查询（S01-S04）应精确命中
        for q in queries:
            conds = q["conditions"]
            if len(conds) == 1 and conds[0]["attribute"] == "age":
                assert eval_query(S, q) == q["result"], q["id"]

    def test_init_beats_random_on_loss(self):
        """按边缘初始化的起点 loss 应远低于纯随机初始化。"""
        from table_diffevo.generator import init_synthetic_table
        from table_diffevo.objective import compute_loss
        from table_diffevo.queries import evaluate_table

        schema = load_schema("configs/test_300x10/schema.yaml")
        queries = load_queries("configs/test_300x10/measured_50query.json")
        df = pd.read_csv("data/test_300x10/test_300x10.csv")
        n = len(df)
        target = np.array([q["result"] for q in queries])

        bins = derive_bins_from_queries(queries, schema, verbose=False)
        attributes = {}
        for attr in schema.attributes:
            if attr.is_numeric():
                if attr.name not in bins:
                    continue
                counts = [int(((df[attr.name] >= lo) & (df[attr.name] <= hi)).sum())
                          for lo, hi in bins[attr.name]]
                attributes[attr.name] = {"type": "numeric",
                                         "bins": [[lo, hi] for lo, hi in bins[attr.name]],
                                         "counts": counts}
            else:
                counts = [int((df[attr.name].astype(str) == str(v)).sum())
                          for v in attr.values]
                attributes[attr.name] = {"type": "categorical",
                                         "values": list(attr.values), "counts": counts}
        marg = {"attributes": attributes}

        S_marg = init_from_marginals(n, schema, marg, np.random.default_rng(0))
        S_rand = init_synthetic_table(n, schema, np.random.default_rng(0))

        loss_marg = compute_loss(target, evaluate_table(S_marg, queries))
        loss_rand = compute_loss(target, evaluate_table(S_rand, queries))
        assert loss_marg < loss_rand

