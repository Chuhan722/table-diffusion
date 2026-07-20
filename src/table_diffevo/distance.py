"""
记录距离计算

距离衡量"两条记录有多相似"，用于限制学习范围：
让每条记录主要向"相似的高适应度记录"学习，而不是全表都学最高分那一条，
从而保持多样性、避免全表坍缩。

## 归一化 Hamming 距离（完整方案 6.3）

逐块比较，落在 [0, 1] 区间：

- **数值块**（age）：归一化绝对差 |age_x − age_z| / (max − min)
- **类别块**（其余 9 个）：相同为 0，不同为 1
- 10 个块等权重求和，除以 10 → 总距离

## 向量化实现（应对大规模）

不用双重循环，而是用 NumPy 广播一次性算出 (N, M) 距离矩阵：
- N 条当前记录 vs M 条参考记录
- 玩具阶段 M=N=300（全对全）
- 大规模时 M=512（固定参考池）

## 接口设计（区分当前表和参考表）

    pairwise_block_distance(rows, donor_rows, schema)

- rows: 当前记录（N 条）
- donor_rows: 参考记录（M 条）
- 返回: (N, M) 距离矩阵

**调用场景：**
- 全对全：`pairwise_block_distance(df, df, schema)` → (N, N)
- 小池子：`pairwise_block_distance(df, pool, schema)` → (N, M)

一套代码，两种用法。
"""
from typing import Optional
import numpy as np
import pandas as pd
from table_diffevo.schema import Schema


def pairwise_block_distance(
    rows: pd.DataFrame,
    donor_rows: pd.DataFrame,
    schema: Schema,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    计算记录间的归一化 Hamming 距离（逐块，等权重）。

    Parameters
    ----------
    rows : pd.DataFrame, shape (N, n_attributes)
        当前记录表
    donor_rows : pd.DataFrame, shape (M, n_attributes)
        参考记录表
    schema : Schema
        属性 schema 定义
    weights : np.ndarray or None, shape (n_blocks,)
        块权重，默认等权重（全为 1）

    Returns
    -------
    np.ndarray, shape (N, M), dtype float
        距离矩阵，distances[i, j] = rows第i条和donor_rows第j条的距离
        所有值在 [0, 1] 区间

    Raises
    ------
    ValueError
        rows 或 donor_rows 缺少 schema 定义的属性

    Notes
    -----
    **内存占用：** (N, M) 矩阵
    - 玩具阶段（300×300）：9 万 × 8 字节 ≈ 0.7 MB
    - 小池子（5万×512）：2560 万 × 8 字节 ≈ 200 MB（可接受）
    - 全对全（5万×5万）：25 亿 × 8 字节 ≈ 20 GB（不可接受，用小池子）

    Examples
    --------
    >>> from table_diffevo.schema import load_schema
    >>> from table_diffevo.queries import load_data
    >>>
    >>> df = load_data("data/test_300x10.csv")
    >>> schema = load_schema("configs/schema.yaml")
    >>>
    >>> # 全对全
    >>> distances = pairwise_block_distance(df, df, schema)
    >>> distances.shape
    (300, 300)
    >>> distances[0, 0]  # 自己和自己距离为 0
    0.0
    >>>
    >>> # 小池子
    >>> pool = df.sample(100)
    >>> distances = pairwise_block_distance(df, pool, schema)
    >>> distances.shape
    (300, 100)
    """
    N = len(rows)
    M = len(donor_rows)
    n_blocks = schema.n_blocks()

    if weights is None:
        weights = np.ones(n_blocks)
    elif len(weights) != n_blocks:
        raise ValueError(
            f"weights 长度 ({len(weights)}) 与块数 ({n_blocks}) 不一致"
        )

    # 验证属性完整性
    required_attrs = set(schema.attribute_names())
    if not required_attrs.issubset(rows.columns):
        missing = required_attrs - set(rows.columns)
        raise ValueError(f"rows 缺少属性: {missing}")
    if not required_attrs.issubset(donor_rows.columns):
        missing = required_attrs - set(donor_rows.columns)
        raise ValueError(f"donor_rows 缺少属性: {missing}")

    # 初始化距离矩阵
    total_distance = np.zeros((N, M), dtype=float)

    # 逐块计算距离
    for block_idx, attr in enumerate(schema.attributes):
        weight = weights[block_idx]

        if attr.is_numeric():
            # 数值块：归一化绝对差
            values_current = rows[attr.name].values  # (N,)
            values_donors = donor_rows[attr.name].values  # (M,)

            # 广播成 (N, M)
            diff = np.abs(values_current[:, None] - values_donors[None, :])

            # 归一化
            range_min, range_max = attr.range
            block_distance = diff / (range_max - range_min)

        else:
            # 类别块：相同为 0，不同为 1
            values_current = rows[attr.name].values  # (N,)
            values_donors = donor_rows[attr.name].values  # (M,)

            # 转成 object 数组（解决 pandas StringArray 广播问题）
            if hasattr(values_current, 'to_numpy'):
                values_current = values_current.to_numpy(dtype=object)
            if hasattr(values_donors, 'to_numpy'):
                values_donors = values_donors.to_numpy(dtype=object)

            # 广播比较
            block_distance = (values_current[:, None] != values_donors[None, :]).astype(float)

        # 加权累加
        total_distance += weight * block_distance

    # 归一化：除以权重总和
    total_distance /= weights.sum()

    return total_distance
