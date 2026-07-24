"""
合成表初始化

为扩散演化主循环生成起点 S_0。

## 两种初始化方法（通过 marginals 参数切换）

**方法 1：纯随机（marginals=None，默认，向后兼容）**
- 每个格子独立从"合法取值"抽样
- 类别块：schema 合法值集合上均匀抽样
- 数值块：schema 合法范围 [min, max] 内均匀整数（含端点）

**方法 2：按 1-way 边缘确定性初始化（marginals=非 None）**
- 每列按边缘计数确定性填配额 + 数值列箱内均摊 + 每列独立打乱
- 设计详见 docs/设计/初始化设计_1way边缘确定性初始化.md
- 由 marginals.init_from_marginals 实现

## 与源数据的关系

**保持一致：**
- 记录条数 N（与源数据相同）
- 属性/列：列名、列数、每列类型（数值/类别）

**不要求一致：**
- 取值范围只需落在 schema 合法域内，不必复刻源数据的实际 min/max
  例如 age 合法域 [18,100]，即使源数据实际只有 25~70，合成表也可出现 18 或 100

**与严格 DP 的关系：**
- 只使用公开 schema 的合法域，不读源数据的真实取值范围（那属于隐私）

## 抽样口径

与 update.py 的 _sample_legal_value 一致：均匀分布。
"""
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from table_diffevo.schema import Schema


def init_synthetic_table(
    n_records: int,
    schema: Schema,
    rng: Optional[np.random.Generator] = None,
    marginals: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    生成初始合成表 S_0。

    Parameters
    ----------
    n_records : int
        记录条数 N（与源数据一致）
    schema : Schema
        属性 schema 定义，提供列结构和合法取值域
    rng : np.random.Generator or None
        随机数生成器。推荐显式传入 np.random.default_rng(seed) 保证复现
    marginals : Dict or None, default None
        1-way 边缘测量（load_marginals 的返回值）。
        - None（默认）：纯随机初始化（方法 1，向后兼容）
        - 非 None：按边缘确定性初始化（方法 2）

    Returns
    -------
    pd.DataFrame, shape (n_records, n_attributes)
        初始合成表，列顺序与 schema.attribute_names() 一致
        - 类别列：取值来自 schema 合法值集合
        - 数值列：取值为 schema 合法范围内的整数（含端点）

    Raises
    ------
    ValueError
        n_records <= 0

    Notes
    -----
    **复现性（铁律 5）：** 使用固定种子的 rng 保证结果可复现。

    **列结构与源数据一致，取值只需合法：** 不复刻源数据实际取值范围。

    **向后兼容：** marginals=None 时行为与旧版完全一致（纯随机）。

    Examples
    --------
    >>> from table_diffevo.schema import load_schema
    >>> schema = load_schema("configs/schema.yaml")
    >>> rng = np.random.default_rng(42)
    >>> # 方法 1：纯随机
    >>> s0 = init_synthetic_table(300, schema, rng)
    >>> s0.shape
    (300, 10)
    >>> # 方法 2：按边缘初始化
    >>> from table_diffevo.marginals import load_marginals
    >>> marg = load_marginals("configs/init_marginals.json")
    >>> s0 = init_synthetic_table(300, schema, rng, marginals=marg)
    >>> s0["age"].between(18, 100).all()
    True
    """
    if n_records <= 0:
        raise ValueError(f"n_records 必须 > 0，得到 {n_records}")

    if rng is None:
        rng = np.random.default_rng()

    # 分派：有 marginals → 按边缘初始化；否则纯随机
    if marginals is not None:
        from table_diffevo.marginals import init_from_marginals
        return init_from_marginals(n_records, schema, marginals, rng)

    # 以下是纯随机路径（原逻辑保持不变）
    columns = {}
    for attr in schema.attributes:
        if attr.is_numeric():
            low, high = attr.range
            # 合法范围内均匀整数，含端点
            columns[attr.name] = rng.integers(
                int(low), int(high) + 1, size=n_records
            )
        else:
            # 类别：合法值集合上均匀抽样
            idx = rng.integers(0, len(attr.values), size=n_records)
            values = np.asarray(attr.values)  # 让 numpy 自动推断类型
            columns[attr.name] = values[idx]

    # 保持列顺序与 schema 一致
    return pd.DataFrame(columns, columns=schema.attribute_names())
