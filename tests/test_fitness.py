"""
测试适应度计算

验证：
1. temp.md 四状态例子（00/01/10/11 的适应度符合直觉）
2. 残差为零时所有记录适应度为零
3. 方向正确（偏低查询鼓励满足记录，偏高查询抑制满足记录）
4. 权重影响适应度大小
5. 与真实数据集成（50个查询）
"""
import numpy as np
import pandas as pd
import pytest
from table_diffevo.fitness import compute_fitness


def test_temp_md_four_state_example():
    """
    temp.md 第七节的四状态例子（锚定测试）

    数据：4条记录 00, 01, 10, 11
    查询：
        q1: A=1，目标3，当前2（偏低，需增加满足q1的记录）
        q2: B=1，目标1，当前2（偏高，需减少满足q2的记录）
        q3: A=1 AND B=1，目标1，当前1（达标，中性）

    期望适应度方向：
        10（满足q1不满足q2）：正（最该增加）
        01（满足q2不满足q1）：负（最该减少）
        00（都不满足）：0（中性）
        11（都满足）：0（中性）
    """
    # 构造数据
    df = pd.DataFrame({
        "A": ["0", "0", "1", "1"],
        "B": ["0", "1", "0", "1"]
    })

    # 构造查询
    queries = [
        {"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]},
        {"id": "q2", "conditions": [{"attribute": "B", "operator": "==", "value": "1"}]},
        {"id": "q3", "conditions": [
            {"attribute": "A", "operator": "==", "value": "1"},
            {"attribute": "B", "operator": "==", "value": "1"}
        ]},
    ]

    # 当前答案
    current_answer = np.array([2, 2, 1])  # q1=2, q2=2, q3=1

    # 目标
    target = np.array([3, 1, 1])

    # 残差（比例残差，N=4）
    residual = (target - current_answer) / 4  # [1/4, -1/4, 0]

    # 计算适应度
    fitness = compute_fitness(df, queries, residual, current_answer)

    # 验证方向正确（这是核心）
    assert fitness[0] == 0.0,  "00 应该是中性"
    assert fitness[1] < 0.0,   "01 应该是负（最该减少）"
    assert fitness[2] > 0.0,   "10 应该是正（最该增加）"
    assert fitness[3] == 0.0,  "11 应该是中性"

    # 验证相对大小
    assert fitness[2] == -fitness[1], "10 和 01 应该大小相等、方向相反"

    # 验证具体数值（手算验证）
    # 记录10: q1贡献=0.25×(1-0.5)=0.125, q2贡献=-0.25×(0-0.5)=0.125, 总=0.25
    # 记录01: q1贡献=0.25×(0-0.5)=-0.125, q2贡献=-0.25×(1-0.5)=-0.125, 总=-0.25
    np.testing.assert_allclose(fitness, np.array([0.0, -0.25, 0.25, 0.0]), atol=1e-10)


def test_zero_residual_zero_fitness():
    """测试残差全为零时，所有记录适应度为零"""
    df = pd.DataFrame({
        "A": ["0", "1", "0", "1"],
        "B": ["0", "0", "1", "1"]
    })

    queries = [
        {"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]},
        {"id": "q2", "conditions": [{"attribute": "B", "operator": "==", "value": "1"}]},
    ]

    current_answer = np.array([2, 2])
    residual = np.array([0.0, 0.0])  # 残差全为零

    fitness = compute_fitness(df, queries, residual, current_answer)

    # 所有记录适应度应为零
    np.testing.assert_array_equal(fitness, np.zeros(4))


def test_direction_low_query():
    """测试偏低查询（ε>0）鼓励满足记录"""
    df = pd.DataFrame({
        "A": ["0", "1"]  # 记录0不满足，记录1满足
    })

    queries = [{"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]}]

    current_answer = np.array([1])  # 1条满足
    residual = np.array([0.5])  # 偏低，需要增加

    fitness = compute_fitness(df, queries, residual, current_answer)

    # 记录1满足查询且查询偏低，应该是正适应度
    assert fitness[1] > fitness[0]
    # 且记录1应该是正值（鼓励增加）
    assert fitness[1] > 0


def test_direction_high_query():
    """测试偏高查询（ε<0）抑制满足记录"""
    df = pd.DataFrame({
        "A": ["0", "1"]
    })

    queries = [{"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]}]

    current_answer = np.array([1])
    residual = np.array([-0.5])  # 偏高，需要减少

    fitness = compute_fitness(df, queries, residual, current_answer)

    # 记录1满足查询但查询偏高，应该是负适应度（抑制）
    assert fitness[1] < 0
    # 记录0不满足，应该比记录1高（相对更值得保留）
    assert fitness[0] > fitness[1]


def test_weights_affect_fitness():
    """测试权重影响适应度大小"""
    df = pd.DataFrame({
        "A": ["0", "1"]
    })

    queries = [{"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]}]

    current_answer = np.array([1])
    residual = np.array([0.5])

    # 权重为1
    fitness_w1 = compute_fitness(df, queries, residual, current_answer, weights=np.array([1.0]))

    # 权重为2
    fitness_w2 = compute_fitness(df, queries, residual, current_answer, weights=np.array([2.0]))

    # 权重加倍，适应度应该加倍
    np.testing.assert_allclose(fitness_w2, fitness_w1 * 2)


def test_multiple_queries_cumulative():
    """测试多个查询的贡献累加"""
    df = pd.DataFrame({
        "A": ["0", "1"],
        "B": ["0", "1"]
    })

    queries = [
        {"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]},
        {"id": "q2", "conditions": [{"attribute": "B", "operator": "==", "value": "1"}]},
    ]

    current_answer = np.array([1, 1])
    residual = np.array([0.25, 0.25])  # 两个查询都偏低

    fitness = compute_fitness(df, queries, residual, current_answer)

    # 记录1同时满足两个查询，应该得到两倍的正贡献
    # 记录0两个都不满足，应该是负贡献
    assert fitness[1] > fitness[0]
    assert fitness[1] > 0


def test_input_validation():
    """测试输入校验"""
    df = pd.DataFrame({"A": ["0", "1"]})
    queries = [{"id": "q1", "conditions": [{"attribute": "A", "operator": "==", "value": "1"}]}]
    current_answer = np.array([1])

    # residual 长度不匹配
    with pytest.raises(ValueError, match="residual 长度.*不一致"):
        compute_fitness(df, queries, np.array([0.1, 0.2]), current_answer)

    # current_answer 长度不匹配
    with pytest.raises(ValueError, match="current_answer 长度.*不一致"):
        compute_fitness(df, queries, np.array([0.1]), np.array([1, 2]))

    # weights 长度不匹配
    with pytest.raises(ValueError, match="weights 长度.*不一致"):
        compute_fitness(df, queries, np.array([0.1]), current_answer, weights=np.array([1, 2]))


def test_integration_with_real_data():
    """
    集成测试：在真实数据上算适应度

    场景：原数据已达标（残差=0），所有记录适应度应为0
    """
    from table_diffevo.queries import load_queries, load_data, evaluate_table
    from table_diffevo.objective import compute_residual

    df = load_data("data/test_300x10/test_300x10.csv")
    queries = load_queries("configs/test_300x10/measured_50query.json")
    target = np.array([q["result"] for q in queries], dtype=int)

    current_answer = evaluate_table(df, queries)
    residual = compute_residual(target, current_answer, n_records=len(df))

    fitness = compute_fitness(df, queries, residual, current_answer)

    # 原数据上残差=0，所有记录适应度应为0
    np.testing.assert_allclose(fitness, np.zeros(len(df)), atol=1e-10)


def test_integration_with_perturbed_data():
    """
    集成测试：在扰动数据上验证适应度方向

    场景：人工扰动数据，使某些查询偏离目标，验证适应度方向正确
    """
    from table_diffevo.queries import load_queries, load_data, evaluate_table
    from table_diffevo.objective import compute_residual

    df = load_data("data/test_300x10/test_300x10.csv")
    queries = load_queries("configs/test_300x10/measured_50query.json")
    target = np.array([q["result"] for q in queries], dtype=int)

    # 扰动：把前10条记录的 age 都改成 20（减少 age>=65 的满足数）
    df_perturbed = df.copy()
    df_perturbed.loc[:9, "age"] = 20

    current_answer = evaluate_table(df_perturbed, queries)
    residual = compute_residual(target, current_answer, n_records=len(df_perturbed))

    fitness = compute_fitness(df_perturbed, queries, residual, current_answer)

    # 至少应该有一些记录的适应度非零（因为有扰动）
    assert np.any(fitness != 0)
    # 适应度向量应该是有限值
    assert np.all(np.isfinite(fitness))
