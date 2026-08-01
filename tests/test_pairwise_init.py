"""二阶最大熵初始化：边缘恢复、IPF、抽样复现与主循环接入。"""

import json

import numpy as np
import pandas as pd
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.pairwise_init import init_from_pairwise_maxent
from table_diffevo.schema import AttributeBlock, Schema


def _binary_schema(n_attributes=3):
    return Schema([
        AttributeBlock(
            name=f"x{i}", type="categorical", description="", values=[0, 1]
        )
        for i in range(n_attributes)
    ])


def _all_pair_queries(df):
    queries = []
    target = []
    names = list(df.columns)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            for value_i in (0, 1):
                for value_j in (0, 1):
                    queries.append({
                        "conditions": [
                            {"attribute": first, "operator": "==", "value": value_i},
                            {"attribute": second, "operator": "==", "value": value_j},
                        ]
                    })
                    target.append(int(
                        ((df[first] == value_i) & (df[second] == value_j)).sum()
                    ))
    return queries, np.asarray(target)


class TestPairwiseMaxent:
    def test_complete_pair_marginals_converge(self):
        """一致的完整二阶边缘应收敛，抽样边缘应接近目标。"""
        rng = np.random.default_rng(12)
        source = pd.DataFrame(
            rng.choice(2, size=(20_000, 3), p=[0.6, 0.4]),
            columns=["x0", "x1", "x2"],
        )
        # 人为加入相关性，避免测试退化成独立分布。
        source.loc[:9_999, "x2"] = source.loc[:9_999, "x0"].to_numpy()
        queries, target = _all_pair_queries(source)

        synthetic, diag = init_from_pairwise_maxent(
            len(source), _binary_schema(), queries, target,
            rng=np.random.default_rng(3),
        )

        assert diag["converged"] is True
        assert diag["usable_pairs"] == 3
        assert diag["max_pair_error"] <= 1e-8
        assert synthetic.shape == source.shape
        for query, expected in zip(queries, target / len(source)):
            mask = np.ones(len(synthetic), dtype=bool)
            for condition in query["conditions"]:
                mask &= (
                    synthetic[condition["attribute"]].to_numpy()
                    == condition["value"]
                )
            assert abs(mask.mean() - expected) < 0.02

    def test_one_missing_cell_is_reconstructed_from_total(self):
        """每个属性对允许缺一个单元，使用公开 N 补齐。"""
        schema = _binary_schema(2)
        queries = [
            {"conditions": [
                {"attribute": "x0", "operator": "==", "value": 0},
                {"attribute": "x1", "operator": "==", "value": 0},
            ]},
            {"conditions": [
                {"attribute": "x0", "operator": "==", "value": 0},
                {"attribute": "x1", "operator": "==", "value": 1},
            ]},
            {"conditions": [
                {"attribute": "x0", "operator": "==", "value": 1},
                {"attribute": "x1", "operator": "==", "value": 0},
            ]},
        ]
        target = np.array([4_000, 1_000, 2_000])

        synthetic, diag = init_from_pairwise_maxent(
            10_000, schema, queries, target, rng=np.random.default_rng(4)
        )

        assert diag["reconstructed_cells"] == 1
        assert diag["converged"] is True
        missing_frequency = ((synthetic["x0"] == 1) & (synthetic["x1"] == 1)).mean()
        assert missing_frequency == pytest.approx(0.3, abs=0.02)

    def test_same_seed_is_reproducible(self):
        source = pd.DataFrame({
            "x0": [0, 0, 1, 1] * 25,
            "x1": [0, 1, 0, 1] * 25,
        })
        queries, target = _all_pair_queries(source)
        args = (len(source), _binary_schema(2), queries, target)

        first, first_diag = init_from_pairwise_maxent(
            *args, rng=np.random.default_rng(8)
        )
        second, second_diag = init_from_pairwise_maxent(
            *args, rng=np.random.default_rng(8)
        )

        pd.testing.assert_frame_equal(first, second)
        # 计时字段允许不同，其余诊断应完全复现。
        first_diag.pop("elapsed_sec")
        second_diag.pop("elapsed_sec")
        assert first_diag == second_diag
        json.dumps(first_diag, ensure_ascii=False)

    def test_numeric_schema_fails_clearly(self):
        schema = Schema([
            AttributeBlock(name="age", type="numeric", description="", range=[0, 10]),
            AttributeBlock(name="x", type="categorical", description="", values=[0, 1]),
        ])
        with pytest.raises(ValueError, match="仅支持全部为 categorical"):
            init_from_pairwise_maxent(
                100, schema, [], np.array([]), rng=np.random.default_rng(0)
            )

    def test_state_space_limit_fails_before_enumeration(self):
        schema = _binary_schema(4)  # 16 个联合状态
        with pytest.raises(ValueError, match="状态空间.*超过上限"):
            init_from_pairwise_maxent(
                100, schema, [], np.array([]),
                rng=np.random.default_rng(0), max_states=8,
            )

    def test_insufficient_pair_coverage_fails_clearly(self):
        schema = _binary_schema(2)
        one_query = [{"conditions": [
            {"attribute": "x0", "operator": "==", "value": 0},
            {"attribute": "x1", "operator": "==", "value": 0},
        ]}]
        with pytest.raises(ValueError, match="没有可用的完整二阶等值边缘"):
            init_from_pairwise_maxent(
                100, schema, one_query, np.array([50]),
                rng=np.random.default_rng(0),
            )


class TestEvolutionIntegration:
    def test_pairwise_maxent_is_available_as_init_method(self):
        source = pd.DataFrame({
            "x0": [0, 0, 1, 1] * 25,
            "x1": [0, 1, 0, 1] * 25,
        })
        queries, target = _all_pair_queries(source)

        best, diag = run_evolution(
            target, queries, _binary_schema(2),
            n_records=100, n_rounds=2, seed=5,
            init_method="pairwise_maxent", log_every=10,
        )

        assert best.shape == (100, 2)
        assert diag["params"]["init_method"] == "pairwise_maxent"
        assert diag["initialization"]["method"] == "pairwise_maxent"
        assert diag["initialization"]["converged"] is True

    def test_unknown_init_method_message_lists_pairwise_option(self):
        with pytest.raises(ValueError, match="pairwise_maxent"):
            run_evolution(
                np.array([]), [], _binary_schema(2), n_records=10,
                init_method="unknown",
            )
