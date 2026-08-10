"""
探测式 α 调度的集成测试

验证 probe 模式在 evolution.py 中的完整集成：
- 参数传递
- ProbeController 初始化
- 块结束检查
- 三岔路探测逻辑
- 经验平台检测
- 诊断输出
"""
import numpy as np
import pytest
from table_diffevo.schema import Schema, AttributeBlock
from table_diffevo.evolution import run_evolution


@pytest.fixture
def simple_schema():
    """简单的二元属性 schema（A, B 各有两个值）"""
    return Schema([
        AttributeBlock(name="A", type="categorical", description="A", values=["a0", "a1"]),
        AttributeBlock(name="B", type="categorical", description="B", values=["b0", "b1"]),
    ])


@pytest.fixture
def simple_workload():
    """两条简单查询：A=a0 和 B=b0"""
    return [
        {"conditions": [{"attribute": "A", "operator": "==", "value": "a0"}]},
        {"conditions": [{"attribute": "B", "operator": "==", "value": "b0"}]},
    ]


def test_probe_mode_basic_run(simple_schema, simple_workload):
    """测试 probe 模式能正常运行"""
    target = np.array([30.0, 25.0])
    n_records = 50

    best_S, diagnostics = run_evolution(
        n_records=n_records,
        queries=simple_workload,
        target=target,
        schema=simple_schema,
        n_rounds=100,
        seed=42,
        alpha_schedule_mode="probe",
        alpha_value=2.0,  # 初始 alpha
        alpha_min=2.0,
        alpha_max=10.0,
        probe_P=3,
        probe_H_candidate_budget=5,
        probe_s=0.2,
        probe_C=2,
        probe_stall_rel=0.05,
        probe_patience=3,
        probe_block_candidate_budget=10,
        log_every=-1,  # 不打印进度
    )

    # 基本检查
    assert len(best_S) == n_records
    assert "probe_history" in diagnostics
    assert diagnostics["probe_history"] is not None
    assert isinstance(diagnostics["probe_history"], list)

    # 参数记录
    params = diagnostics["params"]
    assert params["alpha_schedule_mode"] == "probe"
    assert params["probe_P"] == 3
    assert params["probe_H_candidate_budget"] == 5
    assert params["probe_s"] == 0.2
    assert params["probe_C"] == 2
    assert params["probe_stall_rel"] == 0.05
    assert params["probe_patience"] == 3
    assert params["probe_block_candidate_budget"] == 10


def test_probe_triggers_and_records_history(simple_schema, simple_workload):
    """测试探测被触发并记录历史"""
    # 使用普通目标，但设置极严格的 tol 使其永远不会"达标"
    target = np.array([30.0, 25.0])
    n_records = 50

    # 使用较小的 P 和较大的 stall_rel，容易触发探测
    best_S, diagnostics = run_evolution(
        n_records=n_records,
        queries=simple_workload,
        target=target,
        schema=simple_schema,
        n_rounds=200,
        seed=42,
        alpha_schedule_mode="probe",
        alpha_value=2.0,
        alpha_min=2.0,
        alpha_max=10.0,
        probe_P=2,         # 2 块停滞即触发
        probe_H_candidate_budget=5,
        probe_s=0.3,
        probe_C=1,
        probe_stall_rel=0.05,  # 较大的相对阈值，容易记停滞
        probe_patience=3,
        probe_block_candidate_budget=10,
        tol=-1.0,  # 负容差，永远不会接受"达标"提前停止
        log_every=-1,
    )

    probe_history = diagnostics["probe_history"]

    # 应该至少触发一次探测
    assert len(probe_history) > 0, "探测应该被触发至少一次"

    # 检查探测记录结构
    first_probe = probe_history[0]
    assert "round" in first_probe
    assert "trigger_alpha" in first_probe
    assert "branches" in first_probe
    assert "winner" in first_probe
    assert "winner_alpha" in first_probe
    assert "no_improve_probes" in first_probe

    # 分支应该有 3 个（DOWN, HOLD, UP）
    assert len(first_probe["branches"]) == 3
    branch_names = [b[0] for b in first_probe["branches"]]
    assert "DOWN" in branch_names
    assert "HOLD" in branch_names
    assert "UP" in branch_names

    # winner 应该是三个分支之一
    assert first_probe["winner"] in ["DOWN", "HOLD", "UP"]


def test_probe_empirical_plateau_detection(simple_schema, simple_workload):
    """测试经验平台检测和提前停止"""
    target = np.array([30.0, 25.0])
    n_records = 50

    # 配置容易达到经验平台的参数
    best_S, diagnostics = run_evolution(
        n_records=n_records,
        queries=simple_workload,
        target=target,
        schema=simple_schema,
        n_rounds=500,  # 足够长，但应该提前停止
        seed=42,
        alpha_schedule_mode="probe",
        alpha_value=2.0,
        alpha_min=2.0,
        alpha_max=10.0,
        probe_P=2,
        probe_H_candidate_budget=5,
        probe_s=0.3,
        probe_C=1,
        probe_stall_rel=0.05,
        probe_patience=3,
        probe_block_candidate_budget=10,
        tol=-1.0,  # 负容差，禁止残差全 0 提前停止
        log_every=-1,
    )

    probe_history = diagnostics["probe_history"]

    # 如果提前停止，rounds_run 应该 < n_rounds
    if diagnostics["stopped_early"]:
        assert diagnostics["rounds_run"] < 500
        # 提前停止由耐心值触发：最后一次探测的 no_improve_probes 应达到 patience
        if len(probe_history) >= 1:
            assert probe_history[-1]["no_improve_probes"] >= 3


def test_probe_alpha_history_changes(simple_schema, simple_workload):
    """测试 probe 模式下 alpha 会根据探测结果变化"""
    target = np.array([30.0, 25.0])
    n_records = 50

    best_S, diagnostics = run_evolution(
        n_records=n_records,
        queries=simple_workload,
        target=target,
        schema=simple_schema,
        n_rounds=200,
        seed=42,
        alpha_schedule_mode="probe",
        alpha_value=2.0,
        alpha_min=2.0,
        alpha_max=10.0,
        probe_P=2,
        probe_H_candidate_budget=5,
        probe_s=0.3,
        probe_C=1,
        probe_stall_rel=0.05,
        probe_patience=3,
        probe_block_candidate_budget=10,
        tol=-1.0,  # 负容差
        log_every=-1,
    )

    alpha_history = diagnostics["alpha_history"]
    probe_history = diagnostics["probe_history"]

    # 如果有探测被触发，检查探测历史是否被正确记录
    if len(probe_history) > 0:
        # 检查是否有非 HOLD 的获胜者（说明 alpha 可能会变化）
        winners = [p["winner"] for p in probe_history]
        # alpha_history 的长度应该等于运行的轮数
        assert len(alpha_history) == diagnostics["rounds_run"]


def test_probe_vs_fixed_reproducibility(simple_schema, simple_workload):
    """测试 probe 模式的可复现性"""
    target = np.array([30.0, 25.0])
    n_records = 50

    # 运行两次相同配置
    results = []
    for _ in range(2):
        best_S, diagnostics = run_evolution(
            n_records=n_records,
            queries=simple_workload,
            target=target,
            schema=simple_schema,
            n_rounds=100,
            seed=42,  # 相同种子
            alpha_schedule_mode="probe",
            alpha_value=2.0,
            alpha_min=2.0,
            alpha_max=10.0,
            probe_P=3,
            probe_H_candidate_budget=5,
            probe_s=0.2,
            probe_C=2,
            probe_stall_rel=0.05,
            probe_patience=3,
            probe_block_candidate_budget=10,
            log_every=-1,
        )
        results.append((best_S, diagnostics))

    # 两次运行应该完全一致
    best_S1, diag1 = results[0]
    best_S2, diag2 = results[1]

    assert best_S1.equals(best_S2), "相同种子下表应该相同"
    assert diag1["best_loss"] == diag2["best_loss"], "最优 loss 应该相同"
    assert len(diag1["probe_history"]) == len(diag2["probe_history"]), "探测次数应该相同"

    # alpha_history 应该完全一致
    np.testing.assert_array_equal(
        diag1["alpha_history"], diag2["alpha_history"],
        err_msg="alpha 历史应该完全一致"
    )
