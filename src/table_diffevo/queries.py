"""
查询评价器

核心功能：在给定的表上评价查询，返回计数向量。

设计原则：
- 查询评价器只依赖"表+查询定义"，不绑定任何特定数据源
- 可用于评价原始数据（仅验证用）或合成表（算法运行时）
- 符合"铁律 6"：生成器运行时不访问原始数据

复杂度：
- 单个查询：O(N)，N 为表的行数
- 50 个查询：O(50N) ≈ O(N)
- 对于 5 万行数据，单轮评价约 20-40ms，完全可接受
"""
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def load_queries(path: str) -> List[Dict[str, Any]]:
    """
    加载查询定义文件。

    Parameters
    ----------
    path : str
        查询 JSON 文件路径（如 configs/measured_50query.json）

    Returns
    -------
    List[Dict]
        查询列表，每个查询包含 id, type, conditions, result 等字段

    Notes
    -----
    不在此处转换 value 类型。== 算子的 value 在评价时由 eval_condition
    按数据列的实际类型动态对齐（见 _coerce_to_column_type），
    从而适配不同数据集（字符串列或整数列）。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return data["queries"]


def load_data(path: str) -> pd.DataFrame:
    """
    加载数据表。

    Parameters
    ----------
    path : str
        CSV 文件路径

    Returns
    -------
    pd.DataFrame
        数据表

    Notes
    -----
    【警告】此函数可读取原始数据，仅用于：
    1. 验证查询评价器正确性（测试阶段）
    2. 后续生成器运行时不应调用此函数读取原始数据
    """
    return pd.read_csv(path)


def _coerce_to_column_type(column: pd.Series, value: Any) -> Any:
    """
    把查询的 value 转成 column 的实际类型，用于 == 比较。

    不同数据集的列类型不同：
    - test_300x10 的类别列是字符串（如 children="2_plus"）
    - nltcs 的属性列是整数（0/1）

    若强行用固定类型比较（如整数列 == 字符串"1"），pandas 判为不相等，
    导致查询恒为 0。此函数按列的实际 dtype 对齐 value 类型。

    Parameters
    ----------
    column : pd.Series
        数据表中被比较的列
    value : Any
        查询条件里的比较值（可能是 int / str）

    Returns
    -------
    Any
        转换成与列类型匹配的 value；无法转换时原样返回，交由 pandas 比较
    """
    if pd.api.types.is_numeric_dtype(column):
        # 数值列：把 value 转成数值（字符串 "1" → 1）
        try:
            return pd.to_numeric(value)
        except (ValueError, TypeError):
            return value
    else:
        # 非数值列（字符串等）：把 value 转成字符串
        return str(value)


def eval_condition(df: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
    """
    评价单个查询条件，返回布尔掩码。

    Parameters
    ----------
    df : pd.DataFrame
        数据表
    condition : dict
        单个条件，包含 attribute, operator, value/lower/upper 等字段

    Returns
    -------
    pd.Series (bool)
        布尔掩码，长度等于表的行数，True 表示该行满足条件

    Raises
    ------
    ValueError
        不支持的操作符

    Examples
    --------
    >>> condition = {"attribute": "age", "operator": "between", "lower": 25, "upper": 34}
    >>> mask = eval_condition(df, condition)
    >>> mask.sum()  # 满足条件的记录数
    57
    """
    attr = condition["attribute"]
    op = condition["operator"]

    if op == "==":
        # 等值查询：把 value 转成该列的实际类型再比较
        # （不同数据集列类型不同：test_300x10 的类别列是字符串，
        #  nltcs 的属性列是整数；按数据类型对齐避免"整数==字符串"永不相等）
        value = _coerce_to_column_type(df[attr], condition["value"])
        return df[attr] == value

    elif op == ">=":
        # 大于等于查询（用于 age 列）
        return df[attr] >= condition["value"]

    elif op == "between":
        # 区间查询（用于 age 列）
        lower = condition["lower"]
        upper = condition["upper"]
        return (df[attr] >= lower) & (df[attr] <= upper)

    else:
        raise ValueError(f"不支持的操作符: {op}")


def eval_halfspace_mask(df: pd.DataFrame, query: Dict[str, Any]) -> np.ndarray:
    """
    评价单个半空间查询（w·x ≥ θ），返回布尔掩码（不求和）。

    半空间查询是不可微查询的代表：硬阈值阶跃使梯度方法必须用有偏松弛，
    而本引擎只需要行级布尔掩码，可以精确处理。row-sum（权重全 1）是特例。

    查询格式（与合取查询的 conditions 结构不同，用独立的 halfspace 字段）::

        {
          "id": "HS0001",
          "type": "halfspace",
          "halfspace": {
            "attributes": ["attr_1", "attr_5"],   # 参与属性，不允许重复
            "weights": [1, -1],                    # 与 attributes 等长
            "theta": 3                             # 阈值，w·x >= theta 计数
          },
          "result": 1234
        }

    Parameters
    ----------
    df : pd.DataFrame
        数据表（参与属性必须是数值列）
    query : dict
        查询定义，type 必须是 "halfspace" 且包含 halfspace 字段

    Returns
    -------
    np.ndarray (bool), shape (N,)
        布尔掩码，True 表示该行满足 w·x ≥ θ

    Notes
    -----
    投影用 float64 累加。整数权重 × 整数数据（如二值 0/1 列）在
    float64 下精确（远小于 2^53），掩码结果与整数运算逐位一致。
    校验 fail-closed：字段缺失、长度不一致、重复属性、非数值列、
    非有限数值一律 ValueError。
    """
    spec = query.get("halfspace")
    if not isinstance(spec, dict):
        raise ValueError("halfspace 查询必须包含 halfspace 字典字段")
    attrs = spec.get("attributes")
    weights = spec.get("weights")
    theta = spec.get("theta")
    if not isinstance(attrs, list) or not attrs:
        raise ValueError("halfspace.attributes 必须是非空列表")
    if not isinstance(weights, list) or len(weights) != len(attrs):
        raise ValueError("halfspace.weights 必须是与 attributes 等长的列表")
    if len(set(attrs)) != len(attrs):
        raise ValueError("halfspace.attributes 不允许重复属性")
    if (
        isinstance(theta, bool)
        or not isinstance(theta, (int, float))
        or not np.isfinite(theta)
    ):
        raise ValueError("halfspace.theta 必须是有限数值")

    projection = np.zeros(len(df), dtype=float)
    for attr, weight in zip(attrs, weights):
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not np.isfinite(weight)
        ):
            raise ValueError(
                f"halfspace 权重必须是有限数值，属性 {attr!r} 的权重为 {weight!r}"
            )
        if attr not in df.columns:
            raise ValueError(f"halfspace 属性 {attr!r} 不在表中")
        column = df[attr]
        if not pd.api.types.is_numeric_dtype(column):
            raise ValueError(f"halfspace 只支持数值列，属性 {attr!r} 不是数值列")
        projection += float(weight) * column.to_numpy(dtype=float)

    return projection >= float(theta)


def eval_query_mask(df: pd.DataFrame, query: Dict[str, Any]) -> np.ndarray:
    """
    评价单个查询，返回布尔掩码（不求和）。

    Parameters
    ----------
    df : pd.DataFrame
        数据表
    query : dict
        查询定义。合取查询包含 conditions 列表；type == "halfspace" 的
        半空间查询包含 halfspace 字段（见 eval_halfspace_mask）。

    Returns
    -------
    np.ndarray (bool), shape (N,)
        布尔掩码，True 表示该记录满足查询

    Notes
    -----
    此函数是查询评价的核心底层函数：
    - evaluate_table 内部调用它来计算计数向量
    - fitness.py 调用它来计算适应度（逐查询累加，不存矩阵）

    通过统一的底层函数，确保计数和适应度基于同一套掩码逻辑，永远一致。
    半空间查询在此分派到 eval_halfspace_mask，因此 fitness 与计数对
    半空间查询同样天然一致，无需任何机制改动。

    Examples
    --------
    >>> mask = eval_query_mask(df, {"conditions": [...]})
    >>> mask.sum()  # 满足该查询的记录数
    >>> mask.astype(float)  # 转成 0/1 用于适应度计算
    """
    # 半空间查询：独立结构，分派到专用掩码函数
    if query.get("type") == "halfspace":
        return eval_halfspace_mask(df, query)

    # 初始掩码：全为 True
    mask = pd.Series([True] * len(df), index=df.index)

    # 逐个条件求交集（AND）
    for condition in query["conditions"]:
        mask &= eval_condition(df, condition)

    return mask.to_numpy()


def eval_query(df: pd.DataFrame, query: Dict[str, Any]) -> int:
    """
    评价单个查询，返回满足该查询的记录数。

    【内部实现已改为调用 eval_query_mask】

    Parameters
    ----------
    df : pd.DataFrame
        数据表
    query : dict
        查询定义，包含 conditions 列表

    Returns
    -------
    int
        满足该查询所有条件的记录数

    Notes
    -----
    多个条件通过逻辑与（AND）合并。
    """
    mask = eval_query_mask(df, query)
    return int(mask.sum())


def evaluate_table(df: pd.DataFrame, queries: List[Dict[str, Any]]) -> np.ndarray:
    """
    在给定的表上评价所有查询，返回计数向量。

    这是查询评价器的主入口，生成器运行时调用此函数。

    Parameters
    ----------
    df : pd.DataFrame
        数据表（可以是合成表）
    queries : List[Dict]
        查询列表

    Returns
    -------
    np.ndarray (shape: (n_queries,), dtype: int)
        计数向量，每个元素是对应查询的满足记录数

    Examples
    --------
    >>> queries = load_queries("configs/measured_50query.json")
    >>> synthetic_table = ...  # 生成的合成表
    >>> current_answer = evaluate_table(synthetic_table, queries)
    >>> residual = target - current_answer
    """
    counts = np.array([eval_query(df, q) for q in queries], dtype=int)
    return counts


def verify_evaluator(data_path: str, query_path: str, verbose: bool = True) -> bool:
    """
    【仅测试用】验证查询评价器在原始数据上的正确性。

    将评价器计算的结果与查询文件中的 result 字段逐一对比。

    Parameters
    ----------
    data_path : str
        原始数据路径（如 data/test_300x10/test_300x10.csv）
    query_path : str
        查询文件路径（如 configs/measured_50query.json）
    verbose : bool
        是否打印详细对比信息

    Returns
    -------
    bool
        全部匹配返回 True，有任何不匹配返回 False

    Notes
    -----
    【警告】此函数访问原始数据，仅用于验证阶段。
    验证通过后，后续代码不应再调用此函数。
    """
    df = load_data(data_path)
    queries = load_queries(query_path)

    computed = evaluate_table(df, queries)
    expected = np.array([q["result"] for q in queries], dtype=int)

    all_match = True
    mismatches = []

    for i, (comp, exp) in enumerate(zip(computed, expected)):
        query_id = queries[i]["id"]
        expression = queries[i]["expression"]

        if comp != exp:
            all_match = False
            mismatches.append({
                "query_id": query_id,
                "expression": expression,
                "expected": exp,
                "computed": comp,
                "diff": comp - exp
            })
            if verbose:
                print(f"❌ {query_id}: {expression}")
                print(f"   期望: {exp}, 实际: {comp}, 差值: {comp - exp}")

    if all_match:
        if verbose:
            print(f"✅ 全部 {len(queries)} 个查询验证通过")
            print(f"   查询评价器正确，可用于评价合成表")
        return True
    else:
        if verbose:
            print(f"\n❌ 发现 {len(mismatches)} 个不匹配")
            print(f"   查询评价器逻辑可能有误，需要修正")
        return False
