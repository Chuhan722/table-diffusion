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

    pairwise_block_distance(rows, donor_rows, schema, device='cuda')

- rows: 当前记录（N 条）
- donor_rows: 参考记录（M 条）
- device: 计算设备 ('cuda'=GPU, 'numpy'=原NumPy, 'cpu'=PyTorch CPU)
- 返回: (N, M) 距离矩阵（NumPy array）

**调用场景：**
- 全对全：`pairwise_block_distance(df, df, schema)` → (N, N)
- 小池子：`pairwise_block_distance(df, pool, schema)` → (N, M)

一套代码，两种用法。

## GPU 加速（可选）

当 device='cuda' 时，使用 PyTorch 在 GPU 上计算距离矩阵（20-50x 加速）。
内部实现自动处理 DataFrame → tensor → GPU 计算 → NumPy 的转换，
外部调用无感知，接口完全一致。随机操作仍在 CPU（NumPy），确保可复现性。
"""
from typing import Optional, Literal
import numpy as np
import pandas as pd
from table_diffevo.schema import Schema


def pairwise_block_distance(
    rows: pd.DataFrame,
    donor_rows: pd.DataFrame,
    schema: Schema,
    weights: Optional[np.ndarray] = None,
    device: Literal['cuda', 'cpu', 'numpy'] = 'numpy',
    return_tensor: bool = False,
):
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
    device : {'cuda', 'cpu', 'numpy'}, default 'numpy'
        计算设备：
        - 'cuda': PyTorch GPU 加速（需要 CUDA 可用）
        - 'cpu': PyTorch CPU（用于调试 torch 实现）
        - 'numpy': 原始 NumPy 实现（默认，兼容性最好）
    return_tensor : bool, default False
        是否直接返回留在设备上的 torch.Tensor（仅 torch 路径有效）：
        - False（默认）：返回 NumPy array（保持原行为，numpy 路径始终如此）
        - True：torch 路径下不做 .cpu().numpy()，直接返回 GPU 上的 tensor，
          供下游采样在同一设备上接力，避免 GPU→CPU 搬运。
          numpy 路径忽略此参数（始终返回 array）。

    Returns
    -------
    np.ndarray 或 torch.Tensor, shape (N, M)
        距离矩阵，distances[i, j] = rows第i条和donor_rows第j条的距离
        所有值在 [0, 1] 区间。
        return_tensor=True 且 torch 路径时返回 GPU tensor，否则返回 NumPy array。

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
    - nltcs（16K×16K）：262M × 8 字节 ≈ 2 GB（CPU 可行，GPU 更快）

    **GPU 加速：**
    - 当 device='cuda' 时，自动检测 GPU 可用性，不可用则降级到 'cpu'
    - 内部转换：DataFrame → torch.tensor → GPU 计算 → NumPy array
    - 外部调用无感知，接口完全一致
    - 数值误差 < 1e-6（float32 精度）

    Examples
    --------
    >>> from table_diffevo.schema import load_schema
    >>> from table_diffevo.queries import load_data
    >>>
    >>> df = load_data("data/test_300x10/test_300x10.csv")
    >>> schema = load_schema("configs/schema.yaml")
    >>>
    >>> # 原始 NumPy 实现
    >>> distances = pairwise_block_distance(df, df, schema, device='numpy')
    >>> distances.shape
    (300, 300)
    >>>
    >>> # GPU 加速
    >>> distances_gpu = pairwise_block_distance(df, df, schema, device='cuda')
    >>> np.allclose(distances, distances_gpu, atol=1e-6)  # 数值接近
    True
    """
    # 根据 device 选择实现
    if device == 'numpy':
        # numpy 路径不支持返回 tensor，始终返回 array（保持原行为）
        return _pairwise_distance_numpy(rows, donor_rows, schema, weights)
    elif device in ('cuda', 'cpu'):
        return _pairwise_distance_torch(
            rows, donor_rows, schema, weights, device, return_tensor
        )
    else:
        raise ValueError(f"Unknown device: {device}. Choose from 'cuda', 'cpu', 'numpy'.")
def _pairwise_distance_numpy(
    rows: pd.DataFrame,
    donor_rows: pd.DataFrame,
    schema: Schema,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    原始 NumPy 实现（保留作为参考和兼容性保证）。
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


def _pairwise_distance_torch(
    rows: pd.DataFrame,
    donor_rows: pd.DataFrame,
    schema: Schema,
    weights: Optional[np.ndarray] = None,
    device: str = 'cuda',
    return_tensor: bool = False,
):
    """
    PyTorch GPU 实现（20-50x 加速）。

    内部流程：
    1. DataFrame → torch.tensor
    2. 数据移到 GPU
    3. 逐块计算距离（torch 操作，GPU 并行）
    4. 转回 NumPy array（CPU）；若 return_tensor=True 则保留在设备上直接返回 tensor
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Use device='numpy' or install PyTorch: "
            "pip install torch"
        )

    # 设备检测
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = 'cpu'

    device = torch.device(device)

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

    # 初始化距离矩阵（在 GPU 上）
    total_distance = torch.zeros((N, M), dtype=torch.float32, device=device)

    # 转换权重到 GPU
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # 逐块计算距离
    for block_idx, attr in enumerate(schema.attributes):
        weight = weights_tensor[block_idx]

        if attr.is_numeric():
            # 数值块：归一化绝对差
            values_current = torch.tensor(
                rows[attr.name].values, dtype=torch.float32, device=device
            )  # (N,)
            values_donors = torch.tensor(
                donor_rows[attr.name].values, dtype=torch.float32, device=device
            )  # (M,)

            # 广播成 (N, M)
            diff = torch.abs(values_current[:, None] - values_donors[None, :])

            # 归一化
            range_min, range_max = attr.range
            block_distance = diff / (range_max - range_min)

        else:
            # 类别块：相同为 0，不同为 1
            # 字符串无法直接转 tensor，需要先映射为整数
            values_current = rows[attr.name].values  # (N,)
            values_donors = donor_rows[attr.name].values  # (M,)

            # 创建值到整数的映射
            unique_vals = list(set(values_current) | set(values_donors))
            val_to_int = {v: i for i, v in enumerate(unique_vals)}

            # 映射为整数
            current_ints = np.array([val_to_int[v] for v in values_current])
            donor_ints = np.array([val_to_int[v] for v in values_donors])

            # 转为 tensor
            values_current_t = torch.tensor(current_ints, dtype=torch.int32, device=device)
            values_donors_t = torch.tensor(donor_ints, dtype=torch.int32, device=device)

            # 广播比较
            block_distance = (values_current_t[:, None] != values_donors_t[None, :]).float()

        # 加权累加
        total_distance += weight * block_distance

    # 归一化：除以权重总和
    total_distance /= weights_tensor.sum()

    # 若要求返回 tensor，保留在设备上（供下游采样接力，避免 GPU→CPU 搬运）
    if return_tensor:
        return total_distance

    # 否则转回 NumPy（CPU），保持原行为
    return total_distance.cpu().numpy()
