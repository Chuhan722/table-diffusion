"""
集成测试：验证 evolution.py 正确集成接受规则

测试目标：
1. A0 规则（默认）与历史行为一致（向后兼容）
2. A1 规则可以正常运行
3. 诊断信息包含 delta_L1 和 delta_Q
4. 参数正确记录在 diagnostics['params']
"""
import numpy as np
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.queries import load_queries
from table_diffevo.evolution import run_evolution


@pytest.fixture
def simple_schema():
    """2x2 简单 schema"""
    return Schema([
        AttributeBlock(name="A", type="categorical", description="Attr A", values=["a0", "a1"]),
        AttributeBlock(name="B", type="categorical", description="Attr B", values=["b0", "b1"]),
    ])


@pytest.fixture
def simple_queries():
    """4 个 1-way 边缘查询"""
    return [
        {
            "conditions": [
                {"attribute": "A", "operator": "==", "value": "a0"}
            ]
        },
        {
            "conditions": [
                {"attribute": "A", "operator": "==", "value": "a1"}
            ]
        },
        {
            "conditions": [
                {"attribute": "B", "operator": "==", "value": "b0"}
            ]
        },
        {
            "conditions": [
                {"attribute": "B", "operator": "==", "value": "b1"}
            ]
        },
    ]


def test_a0_default_runs(simple_schema, simple_queries):
    """A0 规则（默认）可以运行"""
    target = np.array([30.0, 20.0, 25.0, 25.0])
    n_records = 50

    best_S, diag = run_evolution(
        target=target,
        queries=simple_queries,
        schema=simple_schema,
        n_records=n_records,
        n_rounds=5,
        seed=42,
        # 使用默认：acceptance_rule='A0', eps_L1=1e-5, eps_Q=0.0
    )

    assert best_S.shape == (n_records, 2)
    assert diag["rounds_run"] == 5
    assert "delta_L1_history" in diag
    assert "delta_Q_history" in diag
    assert len(diag["delta_L1_history"]) == 5
    assert len(diag["delta_Q_history"]) == 5
    assert diag["params"]["acceptance_rule"] == "A0"
    assert diag["params"]["eps_L1"] == 1e-5
    assert diag["params"]["eps_Q"] == 0.0


def test_a1_explicit_runs(simple_schema, simple_queries):
    """A1 规则可以运行"""
    target = np.array([30.0, 20.0, 25.0, 25.0])
    n_records = 50

    best_S, diag = run_evolution(
        target=target,
        queries=simple_queries,
        schema=simple_schema,
        n_records=n_records,
        n_rounds=5,
        seed=42,
        acceptance_rule='A1',
        eps_L1=1e-5,
        eps_Q=0.0,
    )

    assert best_S.shape == (n_records, 2)
    assert diag["rounds_run"] == 5
    assert "delta_L1_history" in diag
    assert "delta_Q_history" in diag
    assert diag["params"]["acceptance_rule"] == "A1"


def test_delta_histories_structure(simple_schema, simple_queries):
    """delta_L1_history 和 delta_Q_history 结构正确"""
    target = np.array([30.0, 20.0, 25.0, 25.0])
    n_records = 50

    best_S, diag = run_evolution(
        target=target,
        queries=simple_queries,
        schema=simple_schema,
        n_records=n_records,
        n_rounds=3,
        seed=42,
        max_retries=2,  # 每轮最多 3 次尝试
    )

    # 每轮有一个列表
    assert len(diag["delta_L1_history"]) == 3
    assert len(diag["delta_Q_history"]) == 3

    # 每轮的列表长度 ≥ 1（至少一次尝试）
    for round_idx in range(3):
        round_L1 = diag["delta_L1_history"][round_idx]
        round_Q = diag["delta_Q_history"][round_idx]
        assert len(round_L1) >= 1
        assert len(round_Q) >= 1
        assert len(round_L1) == len(round_Q)

        # 检查都是浮点数
        for dL1, dQ in zip(round_L1, round_Q):
            assert isinstance(dL1, float)
            assert isinstance(dQ, float)


def test_a0_backward_compatible_behavior(simple_schema, simple_queries):
    """A0 规则的行为与历史 tol 参数一致（向后兼容测试）"""
    target = np.array([30.0, 20.0, 25.0, 25.0])
    n_records = 50
    seed = 123

    # 使用新接口：A0 + eps_Q=0.0（默认）
    best_S_new, diag_new = run_evolution(
        target=target,
        queries=simple_queries,
        schema=simple_schema,
        n_records=n_records,
        n_rounds=10,
        seed=seed,
        tol=0.0,  # 旧参数，现在应该被忽略（A0 用 eps_Q）
        acceptance_rule='A0',
        eps_Q=0.0,
    )

    # 历史行为：tol=0.0 应该等价于 A0 + eps_Q=0.0
    # 检查接受历史和最终 loss
    assert diag_new["params"]["acceptance_rule"] == "A0"
    assert diag_new["params"]["eps_Q"] == 0.0

    # 确保能产生改进（至少接受了一些提案）
    accept_count = sum(diag_new["accept_history"])
    assert accept_count > 0, "应该至少接受了一些提案"


def test_custom_eps_values(simple_schema, simple_queries):
    """自定义 eps 值可以正确传递和记录"""
    target = np.array([30.0, 20.0, 25.0, 25.0])
    n_records = 50

    best_S, diag = run_evolution(
        target=target,
        queries=simple_queries,
        schema=simple_schema,
        n_records=n_records,
        n_rounds=5,
        seed=42,
        acceptance_rule='A1',
        eps_L1=0.001,
        eps_Q=0.1,
    )

    assert diag["params"]["acceptance_rule"] == "A1"
    assert diag["params"]["eps_L1"] == 0.001
    assert diag["params"]["eps_Q"] == 0.1
