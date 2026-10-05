"""
向量化 + 分块查询评价（性能加速路径）

## 为什么需要这个模块

原始 queries.py 的 evaluate_table / compute_fitness 用 Python 循环逐个查询
调用 pandas（df[attr] == value），1000 个查询就是 1000 次 pandas 调用。
实测 nltcs 稳态单轮 3.87s，其中 93% 花在这三处 pandas 评价上
（fitness 31% + 当前表评价 31% + 提案评价 31%）。

本模块把"逐查询 pandas"换成"批量矩阵运算"，并合并计数与 fitness 的重复计算：
- 表转整数/数值矩阵 X (N × 属性数)，摆脱 pandas 逐列开销
- 一次算出"哪些行满足哪些查询"的掩码矩阵，计数和 fitness 都从它派生
- 分块（batch）计算：一次只算 (N, batch) 一竖条，边算边派生、算完即释放，
  内存不随查询数爆掉（保住原设计"内存与查询数无关"的性质）

## 与旧代码的关系（务必理解）

**本模块不替代、不修改 queries.py 的任何函数。** 旧的 evaluate_table /
eval_query_mask / eval_condition / compute_fitness 全部保留：
1. 作为正确性基准——测试用它们的结果当"标准答案"对拍
2. 作为 legacy 开关路径——evolution.py 可切回旧路径应急/对拍
3. 作为算子回退兜底——本模块遇到不认识的算子时，调旧 evaluate_table 来算

## 算子白名单与回退（可扩展性 + 安全性）

快路径只认识 VECTORIZED_OPS = {'==', '>=', 'between'}（当前全部算子）。
某查询若含白名单外的新算子，整条查询进"回退组"，走旧 evaluate_table 慢路径，
并打印提醒。这样将来加新算子时：
- 什么都不做 → 新算子自动走慢路径，结果一定正确（只是不加速）
- 想让它也快 → 把算子加进白名单 + 补一段向量化实现，其余不用动

半空间查询（type == "halfspace"，见 queries.eval_halfspace_mask）不是合取
条件结构，整条固定走回退组，由旧 eval_query_mask 精确评价（计数与 fitness
贡献都对）。若将来成为性能瓶颈，可按同一机制补向量化（本质一次矩阵乘）。

## 权重与噪声接口（照原样保留）

- fitness 的查询权重 w_j：通过 weights 参数传入，默认全 1。
  加权残差 wr = w * residual，fitness = M @ wr − (wr·p)，权重天然融入矩阵乘法。
- 残差的噪声 σ/κ：本模块不碰残差计算，残差由 objective.compute_residual 算好后
  传入，σ/κ 的语义完全由那里负责。本模块只用算好的 residual。

## 可复现性

- 计数是整数比较，numpy 与 cuda 路径逐位精确（可复现无忧）
- fitness 是浮点矩阵乘法，cuda 用 float32、numpy 用 float64，二者有极小数值差
  （既有性质，与 GPU 采样一致）；保证的是"同 device 同种子可复现"
- 本模块不做任何随机操作
"""
from typing import List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd

from table_diffevo.schema import Schema
from table_diffevo.queries import evaluate_table
from table_diffevo.objective import compute_residual


# 快路径支持的算子白名单。新算子若不在此集合，整条查询走旧 evaluate_table 回退。
VECTORIZED_OPS = {"==", ">=", "between"}

# 算子的整数编码（编译成紧凑数组用，避免循环里比字符串）
_OP_EQ = 0
_OP_GE = 1
_OP_BETWEEN = 2
_OP_CODE = {"==": _OP_EQ, ">=": _OP_GE, "between": _OP_BETWEEN}

# 每条查询最多允许的条件数（single/double/triple → 1/2/3）。padding 到此宽度。
_MAX_CONDS = 3


def _encode_table(
    df: pd.DataFrame, schema: Schema
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Dict[Any, int]]]:
    """
    把表编码成数值矩阵 X (N × 属性数)，摆脱 pandas 逐列开销。

    编码规则（与旧 eval_condition 的类型对齐语义保持一致）：
    - 数值列（is_numeric，如 age）：直接存原始数值
    - 类别列：
        - 若列本身是数值 dtype（如 nltcs 的 0/1 整数列）：直接存原始数值
        - 若列是字符串（如 toy 的 children="2_plus"）：按"值→整数"映射编码

    Returns
    -------
    X : np.ndarray, shape (N, A)，dtype float64
        编码后的数值矩阵，列顺序 = schema.attribute_names()
    col_index : Dict[str, int]
        属性名 → X 的列下标
    cat_maps : Dict[str, Dict[value, int]]
        字符串类别列的"值→整数"映射（查询值也要按此映射，才能对齐编码）。
        数值列不在此字典中。

    Notes
    -----
    只有字符串类别列需要映射；数值列（含 nltcs 的整数 0/1 列）保持原值，
    这样查询里的整数比较值无需转换，语义与旧路径的 _coerce_to_column_type 一致。
    """
    names = schema.attribute_names()
    A = len(names)
    N = len(df)
    X = np.empty((N, A), dtype=float)
    col_index: Dict[str, int] = {}
    cat_maps: Dict[str, Dict[Any, int]] = {}

    for j, name in enumerate(names):
        col_index[name] = j
        col = df[name]
        if pd.api.types.is_numeric_dtype(col):
            # 数值列（含 nltcs 整数类别列）：直接存原值
            X[:, j] = col.to_numpy(dtype=float)
        else:
            # 字符串类别列：建立"值→整数"映射并编码
            values = col.to_numpy()
            uniq = {}
            codes = np.empty(N, dtype=float)
            for i, v in enumerate(values):
                if v not in uniq:
                    uniq[v] = len(uniq)
                codes[i] = uniq[v]
            X[:, j] = codes
            cat_maps[name] = uniq

    return X, col_index, cat_maps


# 哨兵编码：== 比较值在字符串类别列里不存在时用它，保证该条件恒为 False
# （所有真实类别编码都 >= 0，用 -1 永不匹配，与旧路径 df[attr]==缺失值 → 全 False 一致）
_MISSING_CODE = -1.0


def _compile_queries(
    queries: List[Dict[str, Any]],
    col_index: Dict[str, int],
    cat_maps: Dict[str, Dict[Any, int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    把查询编译成紧凑的定长数组，并按算子白名单分成快路径组与回退组。

    每条快路径查询的条件补齐（padding）到 _MAX_CONDS 个槽位。

    Returns
    -------
    fast_cols : np.ndarray (F, _MAX_CONDS), int
        快路径查询各条件作用的列下标（padding 槽填 0，配合 valid 掩码忽略）
    fast_ops : np.ndarray (F, _MAX_CONDS), int
        各条件算子编码（_OP_EQ/_OP_GE/_OP_BETWEEN；padding 槽填 0）
    fast_lo : np.ndarray (F, _MAX_CONDS), float
        比较值下界：== 和 >= 用它；between 的 lower 也用它
    fast_hi : np.ndarray (F, _MAX_CONDS), float
        比较值上界：仅 between 的 upper 用它；其余算子填 0（不参与）
    fast_valid : np.ndarray (F, _MAX_CONDS), bool
        该槽位是否为真实条件（True）还是 padding（False）
    fast_orig_idx : List[int]
        快路径组各查询在原 queries 列表中的下标（用于把结果拼回原顺序）

    其中 F = 快路径查询数。回退组由调用方通过"不在 fast_orig_idx 里的下标"识别。
    """
    n = len(queries)
    fast_cols, fast_ops = [], []
    fast_lo, fast_hi, fast_valid = [], [], []
    fast_orig_idx: List[int] = []

    for qi in range(n):
        if queries[qi].get("type") == "halfspace":
            # 半空间查询不是合取条件结构，整条进回退组（旧 eval_query_mask 精确评价）
            continue
        conds = queries[qi]["conditions"]
        # 判断整条查询是否全部条件都可向量化
        all_ops_ok = all(c["operator"] in VECTORIZED_OPS for c in conds)
        if not all_ops_ok or len(conds) > _MAX_CONDS:
            # 进回退组（含未知算子，或条件数超出 padding 宽度的极端情况）
            continue

        cols_row = [0] * _MAX_CONDS
        ops_row = [0] * _MAX_CONDS
        lo_row = [0.0] * _MAX_CONDS
        hi_row = [0.0] * _MAX_CONDS
        valid_row = [False] * _MAX_CONDS

        for k, c in enumerate(conds):
            attr = c["attribute"]
            op = c["operator"]
            cols_row[k] = col_index[attr]
            ops_row[k] = _OP_CODE[op]
            valid_row[k] = True
            if op == "between":
                lo_row[k] = float(c["lower"])
                hi_row[k] = float(c["upper"])
            elif op == ">=":
                lo_row[k] = float(c["value"])
            else:  # ==
                lo_row[k] = _encode_eq_value(attr, c["value"], cat_maps)

        fast_cols.append(cols_row)
        fast_ops.append(ops_row)
        fast_lo.append(lo_row)
        fast_hi.append(hi_row)
        fast_valid.append(valid_row)
        fast_orig_idx.append(qi)

    return (
        np.array(fast_cols, dtype=np.intp).reshape(-1, _MAX_CONDS),
        np.array(fast_ops, dtype=np.intp).reshape(-1, _MAX_CONDS),
        np.array(fast_lo, dtype=float).reshape(-1, _MAX_CONDS),
        np.array(fast_hi, dtype=float).reshape(-1, _MAX_CONDS),
        np.array(fast_valid, dtype=bool).reshape(-1, _MAX_CONDS),
        fast_orig_idx,
    )


def _encode_eq_value(attr: str, value: Any, cat_maps: Dict[str, Dict[Any, int]]) -> float:
    """
    把 == 的比较值编码成与 _encode_table 一致的数值。

    - 数值列（attr 不在 cat_maps）：直接转 float（与旧路径 pd.to_numeric 对齐）
    - 字符串类别列（attr 在 cat_maps）：按 str(value) 查映射；
      查不到（该类别不在表中）→ 返回 _MISSING_CODE，使条件恒 False，
      与旧路径 df[attr] == 缺失值 → 全 False 完全一致
    """
    if attr in cat_maps:
        # 字符串类别列：旧路径 _coerce_to_column_type 会把 value 转成 str 再比较
        key = str(value)
        return float(cat_maps[attr].get(key, _MISSING_CODE))
    else:
        # 数值列：直接数值比较
        return float(value)


def _batch_masks_numpy(
    X: np.ndarray,
    cols: np.ndarray,
    ops: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """
    NumPy：算一批查询的掩码矩阵 (N, B)。B = 这批的查询数。

    对每个条件槽 k（最多 _MAX_CONDS 个），用广播算 (N, B) 的比较结果，
    再按 AND 累积。padding 槽（valid=False）视为恒 True，不影响 AND。

    Parameters
    ----------
    X : (N, A)  编码后的表矩阵
    cols/ops/lo/hi/valid : (B, _MAX_CONDS)  这批查询的编译数组

    Returns
    -------
    mask : (N, B) bool
    """
    N = X.shape[0]
    B = cols.shape[0]
    mask = np.ones((N, B), dtype=bool)

    for k in range(cols.shape[1]):
        col_k = cols[:, k]          # (B,)
        op_k = ops[:, k]            # (B,)
        lo_k = lo[:, k]             # (B,)
        hi_k = hi[:, k]             # (B,)
        valid_k = valid[:, k]       # (B,)

        # 取每个查询该槽对应的列数据 → (N, B)
        vals = X[:, col_k]          # 花式索引，(N, B)

        # 按算子分别算比较结果（对整批一次算完）
        eq_res = vals == lo_k[None, :]
        ge_res = vals >= lo_k[None, :]
        bt_res = (vals >= lo_k[None, :]) & (vals <= hi_k[None, :])

        # 按每个查询的算子选择对应结果
        cond = np.where(
            op_k[None, :] == _OP_EQ, eq_res,
            np.where(op_k[None, :] == _OP_GE, ge_res, bt_res),
        )

        # padding 槽（valid=False）当作 True，不参与 AND
        cond = np.where(valid_k[None, :], cond, True)
        mask &= cond

    return mask


def _batch_masks_torch(X_t, cols_t, ops_t, lo_t, hi_t, valid_t, torch):
    """
    PyTorch：算一批查询的掩码矩阵 (N, B)，逻辑与 _batch_masks_numpy 逐行对应。

    所有张量已在目标设备上。X_t 为 float32，比较值同为 float32。
    整数比较（本项目查询值都是整数，编码后也是整数值的 float）在 float32 下精确。
    """
    N = X_t.shape[0]
    B = cols_t.shape[0]
    mask = torch.ones((N, B), dtype=torch.bool, device=X_t.device)

    for k in range(cols_t.shape[1]):
        col_k = cols_t[:, k]        # (B,)
        op_k = ops_t[:, k]          # (B,)
        lo_k = lo_t[:, k]           # (B,)
        hi_k = hi_t[:, k]           # (B,)
        valid_k = valid_t[:, k]     # (B,)

        vals = X_t[:, col_k]        # (N, B)

        eq_res = vals == lo_k.unsqueeze(0)
        ge_res = vals >= lo_k.unsqueeze(0)
        bt_res = (vals >= lo_k.unsqueeze(0)) & (vals <= hi_k.unsqueeze(0))

        cond = torch.where(
            (op_k == _OP_EQ).unsqueeze(0), eq_res,
            torch.where((op_k == _OP_GE).unsqueeze(0), ge_res, bt_res),
        )
        cond = torch.where(valid_k.unsqueeze(0), cond, torch.ones_like(cond))
        mask &= cond

    return mask


def evaluate_conditions_vectorized(
    df: pd.DataFrame,
    conditions: List[Dict[str, Any]],
    schema: Schema,
    *,
    device: Literal["numpy", "cuda", "cpu"] = "numpy",
    return_tensor: bool = False,
    float64: bool = False,
):
    """一次评价一组独立条件，供增量查询后端复用。

    与 :func:`evaluate_vectorized` 不同，这里每一列是一个条件而不是一条
    合取查询。表编码和条件编码完全复用本模块的现有规则；CUDA 路径不会在
    设备不可用时静默回退，便于结果前协议严格绑定运行设备。
    """

    if device not in ("numpy", "cuda", "cpu"):
        raise ValueError("device 必须是 'numpy'、'cuda' 或 'cpu'")
    if not isinstance(return_tensor, bool) or not isinstance(float64, bool):
        raise ValueError("return_tensor 和 float64 必须是布尔值")

    X, col_index, cat_maps = _encode_table(df, schema)
    cols = []
    ops = []
    lo = []
    hi = []
    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise ValueError(f"conditions[{index}] 必须是字典")
        attribute = condition.get("attribute")
        operator = condition.get("operator")
        if attribute not in col_index or operator not in VECTORIZED_OPS:
            raise ValueError(f"不支持的独立条件：{condition!r}")
        cols.append(col_index[attribute])
        ops.append(_OP_CODE[operator])
        if operator == "between":
            if "lower" not in condition or "upper" not in condition:
                raise ValueError("between 条件缺少 lower/upper")
            lo.append(float(condition["lower"]))
            hi.append(float(condition["upper"]))
        elif operator == ">=":
            if "value" not in condition:
                raise ValueError(">= 条件缺少 value")
            lo.append(float(condition["value"]))
            hi.append(0.0)
        else:
            if "value" not in condition:
                raise ValueError("== 条件缺少 value")
            lo.append(_encode_eq_value(attribute, condition["value"], cat_maps))
            hi.append(0.0)

    cols_np = np.asarray(cols, dtype=np.intp)
    ops_np = np.asarray(ops, dtype=np.intp)
    lo_np = np.asarray(lo, dtype=np.float64)
    hi_np = np.asarray(hi, dtype=np.float64)
    if device == "numpy":
        values = X[:, cols_np]
        result = np.where(
            ops_np[None, :] == _OP_EQ,
            values == lo_np[None, :],
            np.where(
                ops_np[None, :] == _OP_GE,
                values >= lo_np[None, :],
                (values >= lo_np[None, :]) & (values <= hi_np[None, :]),
            ),
        )
        return result if return_tensor else np.asarray(result, dtype=bool)

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("条件 CUDA 路径需要 PyTorch") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("请求 CUDA 条件评价但 CUDA 不可用")
    target_device = torch.device(device)
    dtype = torch.float64 if float64 else torch.float32
    X_t = torch.as_tensor(X, dtype=dtype, device=target_device)
    cols_t = torch.as_tensor(cols_np, dtype=torch.long, device=target_device)
    ops_t = torch.as_tensor(ops_np, dtype=torch.long, device=target_device)
    lo_t = torch.as_tensor(lo_np, dtype=dtype, device=target_device)
    hi_t = torch.as_tensor(hi_np, dtype=dtype, device=target_device)
    values = X_t[:, cols_t]
    result_t = torch.where(
        (ops_t == _OP_EQ).unsqueeze(0),
        values == lo_t.unsqueeze(0),
        torch.where(
            (ops_t == _OP_GE).unsqueeze(0),
            values >= lo_t.unsqueeze(0),
            (values >= lo_t.unsqueeze(0)) & (values <= hi_t.unsqueeze(0)),
        ),
    )
    if return_tensor:
        return result_t
    return result_t.cpu().numpy().astype(bool, copy=False)


def _query_attribute_matrix(
    queries: List[Dict[str, Any]],
    attr_names: List[str],
) -> np.ndarray:
    """查询→属性归属矩阵 E，shape (m, A)，float。

    E[j, a] = 1 当且仅当查询 j 涉及属性 a：合取查询取 conditions 里出现的
    属性集合（2-way 查询同时归属两科）；halfspace 查询取
    ``query["halfspace"]["attributes"]``。未知属性 fail-closed 抛错。
    """
    attr_index = {name: i for i, name in enumerate(attr_names)}
    E = np.zeros((len(queries), len(attr_names)), dtype=float)
    for j, query in enumerate(queries):
        if query.get("type") == "halfspace":
            spec = query.get("halfspace")
            if not isinstance(spec, dict) or "attributes" not in spec:
                raise ValueError(
                    f"查询 {j} 是 halfspace 但缺少 halfspace.attributes 字段"
                )
            involved = spec["attributes"]
        else:
            involved = {c["attribute"] for c in query["conditions"]}
        for name in involved:
            if name not in attr_index:
                raise ValueError(
                    f"查询 {j} 涉及未知属性 {name!r}（不在 schema 属性表中）"
                )
            E[j, attr_index[name]] = 1.0
    return E


def evaluate_vectorized(
    df: pd.DataFrame,
    queries: List[Dict[str, Any]],
    schema: Schema,
    target: Optional[np.ndarray] = None,
    n_records: Optional[int] = None,
    sigma: Optional[np.ndarray] = None,
    kappa: float = 1.0,
    weights: Optional[np.ndarray] = None,
    batch_size: int = 256,
    device: Literal["numpy", "cuda", "cpu"] = "numpy",
    want_fitness: bool = True,
    verbose: bool = True,
    residual_geometry: str = "absolute",
    residual_geometry_floor: float = 8.0,
    want_block_scores: bool = False,
) -> Tuple[np.ndarray, ...]:
    """
    向量化 + 分块评价：一次掩码扫描同时拿到计数 q、残差 ε 和 fitness。

    这是本模块的主入口，替代 evolution.py 里"evaluate_table + compute_fitness"
    的重复计算：计数、残差、fitness 都从同一批掩码派生，省掉一次评价。

    **为什么能一次扫描**：分块是按"查询（列）"切的，每个查询的计数在它所在的批
    里就已完整（所有 N 行都参与）。而某查询的残差只依赖它自己的计数，所以能在
    同一批里顺手算出残差、累加 fitness，不需要"先得到全部计数再回头算 fitness"。

    Parameters
    ----------
    df : pd.DataFrame
        待评价的表（合成表）
    queries : List[Dict]
        查询定义列表（长度 m）
    schema : Schema
        属性 schema（用于编码列类型）
    target : np.ndarray or None, shape (m,)
        目标计数 y。want_fitness=True 时必需（用于算残差）。
    n_records : int or None
        记录总数 N（算残差要除以它）。默认取 len(df)。
    sigma : np.ndarray or None, shape (m,)
        各查询噪声标准差（DP 阶段用）。None=无噪声（σ=0）。语义与
        objective.compute_residual 完全一致，本模块只是转调它。
    kappa : float, default 1.0
        噪声容忍系数（配合 sigma）。
    weights : np.ndarray or None, shape (m,)
        查询权重 w_j，默认全 1。加权残差 wr = w * residual 融入 fitness。
    batch_size : int, default 256
        分块大小：一次算多少个查询的掩码。内存峰值 ∝ N × batch_size。
    device : {'numpy', 'cuda', 'cpu'}, default 'numpy'
        计算设备。numpy 用 float64；cuda/cpu 用 torch float32。
    want_fitness : bool, default True
        是否同时算残差和 fitness。False 时只算计数（如评价 proposal 只需计数），
        返回的 residual 和 fitness 均为 None。
    verbose : bool, default True
        回退组非空时是否打印提醒。
    residual_geometry : str, default "absolute"
        残差几何，透传给 objective.compute_residual（"absolute" 现状口径 /
        "sqrt_relative" 平方根标准化 / "relative" 相对残差口径），影响
        返回的 residual 与 fitness。
    residual_geometry_floor : float, default 8.0
        sqrt_relative/relative 几何的分母下限，透传给
        objective.compute_residual。
    want_block_scores : bool, default False
        是否同时输出分科状态分（block scores）。True 要求 want_fitness=True，
        返回值变为四元组 ``(q, residual, fitness, block_scores)``；False 保持
        历史三元组，所有旧调用点不受影响。

    Returns
    -------
    q : np.ndarray, shape (m,), int
        计数向量。整数，与旧 evaluate_table 逐位相同。
    residual : np.ndarray, shape (m,), float 或 None
        比例残差 ε（want_fitness=False 时为 None）。与 compute_residual 一致。
    fitness : np.ndarray, shape (N,), float 或 None
        每条记录的适应度（want_fitness=False 时为 None）。
    block_scores : np.ndarray, shape (N, A), float（仅 want_block_scores=True）
        分科状态分 s_a(z) = Σ_{j 涉及属性 a} wr_j·(M[z,j] − p_j)。
        行 fitness 的按属性分解（2-way 查询同时计入两科，故
        Σ_a s_a ≠ fitness）。纯状态量：只用当前表的掩码与残差，不预演编辑。

    Notes
    -----
    **计数正确性**：整数比较，numpy/cuda 与旧路径逐元素相同。
    **残差**：内部调用 objective.compute_residual（σ/κ 语义完全一致）。
    **fitness 公式**：fitness = M @ wr − (wr·p)，p = q/N，wr = w*residual。
    与旧 compute_fitness 的逐查询累加数学等价（numpy 路径逐位一致）。
    **分科分公式**：block_scores = M @ (wr ⊙ E) − (wr⊙p) @ E，E 为查询→属性
    归属矩阵（_query_attribute_matrix），与 fitness 同一批掩码顺手算出。
    **回退组**：含未向量化算子的查询走旧 evaluate_table，计数按原顺序填回；
    这些查询的 fitness / 分科分贡献也用旧掩码逻辑补上（见实现）。
    """
    m = len(queries)
    N = len(df)
    if n_records is None:
        n_records = N

    if not isinstance(want_block_scores, (bool, np.bool_)):
        raise ValueError("want_block_scores 必须是布尔值")
    if want_block_scores and not want_fitness:
        raise ValueError(
            "want_block_scores=True 需要 want_fitness=True（分科分依赖加权残差）"
        )

    if want_fitness:
        if target is None:
            raise ValueError("want_fitness=True 时必须提供 target")
        if len(target) != m:
            raise ValueError(f"target 长度 ({len(target)}) 与查询数 ({m}) 不一致")
        if weights is None:
            weights = np.ones(m)
        elif len(weights) != m:
            raise ValueError(f"weights 长度 ({len(weights)}) 与查询数 ({m}) 不一致")

    # 编码表 + 编译查询（分快路径组 / 回退组）
    X, col_index, cat_maps = _encode_table(df, schema)
    fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig = _compile_queries(
        queries, col_index, cat_maps
    )

    # 回退组 = 不在快路径组里的查询下标
    fast_set = set(fast_orig)
    fallback_idx = [qi for qi in range(m) if qi not in fast_set]
    if fallback_idx and verbose:
        reasons = sorted({
            c["operator"]
            for qi in fallback_idx
            for c in queries[qi].get("conditions", [])
            if c["operator"] not in VECTORIZED_OPS
        })
        if any(queries[qi].get("type") == "halfspace" for qi in fallback_idx):
            reasons.append("type=halfspace")
        print(
            f"提示：{len(fallback_idx)} 个查询含未向量化算子/类型 {reasons}，"
            f"已走慢路径（旧 evaluate_table）。如需加速请补向量化实现。"
        )

    q = np.zeros(m, dtype=np.int64)
    # 全量 wr（加权残差）在扫描中逐批填入；扫描后用它算 fitness 常数项
    wr_full = np.zeros(m, dtype=float) if want_fitness else None
    fitness_accum = np.zeros(N, dtype=float) if want_fitness else None
    # 分科分：E (m, A) 归属矩阵 + (N, A) 累加器（M @ (wr ⊙ E) 逐批累加）
    if want_block_scores:
        attr_names = schema.attribute_names()
        query_attr_matrix = _query_attribute_matrix(queries, attr_names)
        block_accum = np.zeros((N, len(attr_names)), dtype=float)
    else:
        query_attr_matrix = None
        block_accum = None

    # 每批算残差用的闭包：counts 完整 → residual 只依赖自身 count，可在批内算
    def _batch_wr(orig):
        """给定这批查询的原始下标，算它们的加权残差 wr（依赖已填好的 q[orig]）。"""
        sig = None if sigma is None else np.asarray(sigma)[orig]
        r = compute_residual(
            np.asarray(target)[orig], q[orig], n_records, sigma=sig, kappa=kappa,
            geometry=residual_geometry, geometry_floor=residual_geometry_floor,
        )
        return np.asarray(weights)[orig] * r

    if device in ("cuda", "cpu"):
        _run_batches_torch(
            X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
            q, wr_full, fitness_accum, batch_size, device, want_fitness, _batch_wr,
            query_attr_matrix=query_attr_matrix, block_accum=block_accum,
        )
    else:
        _run_batches_numpy(
            X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
            q, wr_full, fitness_accum, batch_size, want_fitness, _batch_wr,
            query_attr_matrix=query_attr_matrix, block_accum=block_accum,
        )

    # 回退组：用旧 evaluate_table 逐查询算，计数填回原位置；fitness 贡献用旧掩码补
    if fallback_idx:
        _handle_fallback(
            df, queries, fallback_idx, q, wr_full, fitness_accum,
            want_fitness, N, n_records, target, weights, sigma, kappa,
            residual_geometry, residual_geometry_floor,
            query_attr_matrix=query_attr_matrix, block_accum=block_accum,
        )

    residual = fitness = None
    block_scores = None
    if want_fitness:
        # fitness = M @ wr − (wr·p)，常数项对所有记录相同
        p = q / N
        const = float(np.dot(wr_full, p))
        fitness = fitness_accum - const
        # 残差整体重算一次（与 compute_residual 完全一致，返回给主循环用）
        sig = None if sigma is None else np.asarray(sigma)
        residual = compute_residual(
            np.asarray(target), q, n_records, sigma=sig, kappa=kappa,
            geometry=residual_geometry, geometry_floor=residual_geometry_floor,
        )
        if want_block_scores:
            # 分科常数项：每科减去 Σ_{j∋a} wr_j·p_j（按归属矩阵散射）
            block_scores = block_accum - (wr_full * p) @ query_attr_matrix

    if want_block_scores:
        return q, residual, fitness, block_scores
    return q, residual, fitness


class ValueGainComputer:
    """残差引导值分布核（value guidance）的逐格增益计算器。

    对每条记录 i、每个属性 a、每个候选值 v，计算"把 i 的 a 改成 v"
    对加权残差账本的一阶增益（只保留随 v 变化的部分）：

        gain_a[i, v] = Σ_{j ∈ J_a, u_j = v} wr_j · M_other_j(i)

    其中 J_a = 涉及属性 a 的合取查询集合；u_j = 查询 j 对 a 要求的值；
    M_other_j(i) = 查询 j 除 a 外其余条件在记录 i 上的合取掩码
    （1-way 查询为全 1）；wr_j = 加权残差（与 fitness 完全同源：
    weights ⊙ compute_residual，欠账为正 → gain 越大越该补）。

    实现（矩阵分解，避免逐查询扫描）：
    - 1-way 查询 (a==u)：other 掩码 = 全 1 列；
    - 2-way 查询 (a==u ∧ b==w)：other 掩码 = 单条件掩码 (X_b==w)。
    于是全属性拼接的增益矩阵 G (N, D) = B @ W：
    - B (N, K+1)：K 个去重单条件掩码列 + 1 列常量 1；
    - W (K+1, D)：散射矩阵，W[k, dest] += wr_j，每轮按当前 wr 重填；
    - D = Σ_a |V_a|（全属性值域拼接宽度）。
    查询结构（基底列注册、散射条目、值域布局）在构造时解析一次；
    每轮 compute() 只需重编码表 + 重算 B + 重填 W + 一次矩阵乘。

    fail-closed 限制（构造时抛错，不静默降级）：
    - 只支持全 == 条件的合取查询，且条件数 ≤ 2（考卷为 1-way/2-way）；
    - halfspace 查询不支持；
    - 全部属性必须是 categorical 且 values 非空（数值属性无有限值域）；
    - 查询里出现的值必须在对应属性的 schema 值域内。

    注意：字符串类别列的编码映射（cat_maps）按表内出现顺序建立、
    跨轮不稳定，所以条件值在每轮 compute() 里用当轮映射重新编码，
    与 evaluate_vectorized 的掩码语义逐位同源。
    """

    _CONST = -1  # 基底条目哨兵：表示 other 掩码 = 全 1 列

    def __init__(
        self,
        queries: List[Dict[str, Any]],
        schema: Schema,
        weights: np.ndarray,
        target: np.ndarray,
        *,
        sigma: Optional[np.ndarray] = None,
        kappa: float = 1.0,
        residual_geometry: str = "absolute",
        residual_geometry_floor: float = 8.0,
    ):
        self._schema = schema
        self._weights = np.asarray(weights, dtype=float)
        self._target = np.asarray(target, dtype=float)
        self._sigma = None if sigma is None else np.asarray(sigma, dtype=float)
        self._kappa = kappa
        self._geometry = residual_geometry
        self._geometry_floor = residual_geometry_floor

        if len(self._weights) != len(queries) or len(self._target) != len(queries):
            raise ValueError(
                f"weights/target 长度须等于查询数：weights={len(self._weights)} "
                f"target={len(self._target)} queries={len(queries)}"
            )

        # —— 值域布局：全属性 categorical，值域拼接成 D 列 ——
        attr_names = schema.attribute_names()
        self._domains: Dict[str, list] = {}
        self._dest_offset: Dict[str, int] = {}
        value_pos: Dict[str, Dict[str, int]] = {}
        offset = 0
        for name in attr_names:
            block = schema.get_block(name)
            if not block.is_categorical() or not block.values:
                raise ValueError(
                    f"value guidance 只支持有限值域的 categorical 属性，"
                    f"属性 {name!r} type={block.type!r} values={block.values!r}"
                )
            self._domains[name] = list(block.values)
            self._dest_offset[name] = offset
            # 值匹配按 str 对齐（与 _encode_eq_value 的 str(value) 语义一致）
            value_pos[name] = {str(v): k for k, v in enumerate(block.values)}
            offset += len(block.values)
        self._d_total = offset

        # —— 解析查询：注册基底掩码列 + 散射条目 ——
        basis_index: Dict[Tuple[str, str], int] = {}
        self._basis_conds: List[Tuple[str, Any]] = []  # (attr, 原始值)，每轮重编码
        entry_query: List[int] = []
        entry_basis: List[int] = []
        entry_dest: List[int] = []
        for j, query in enumerate(queries):
            if query.get("type") == "halfspace" or "halfspace" in query:
                raise ValueError(f"查询 {j} 是 halfspace，value guidance 不支持")
            conds = query.get("conditions")
            if not conds:
                raise ValueError(f"查询 {j} 缺少 conditions")
            if len(conds) > 2:
                raise ValueError(
                    f"查询 {j} 有 {len(conds)} 个条件，value guidance 首版只支持 ≤2-way"
                )
            for c in conds:
                if c.get("operator") != "==":
                    raise ValueError(
                        f"查询 {j} 含非 == 算子 {c.get('operator')!r}，value guidance 不支持"
                    )
                if c["attribute"] not in self._domains:
                    raise ValueError(f"查询 {j} 涉及未知属性 {c['attribute']!r}")
            # 每个条件轮流当"目标属性"，其余条件构成 other 掩码
            for pos, c in enumerate(conds):
                a, u = c["attribute"], c["value"]
                if str(u) not in value_pos[a]:
                    raise ValueError(
                        f"查询 {j} 条件值 {u!r} 不在属性 {a!r} 的 schema 值域 "
                        f"{self._domains[a]!r} 内"
                    )
                dest = self._dest_offset[a] + value_pos[a][str(u)]
                if len(conds) == 1:
                    basis = self._CONST
                else:
                    other = conds[1 - pos]
                    key = (other["attribute"], str(other["value"]))
                    if key not in basis_index:
                        basis_index[key] = len(self._basis_conds)
                        self._basis_conds.append(
                            (other["attribute"], other["value"])
                        )
                    basis = basis_index[key]
                entry_query.append(j)
                entry_basis.append(basis)
                entry_dest.append(dest)

        k = len(self._basis_conds)
        eb = np.asarray(entry_basis, dtype=np.int64)
        eb[eb == self._CONST] = k  # 常量列排在末尾
        self._entry_query = np.asarray(entry_query, dtype=np.int64)
        self._entry_basis = eb
        self._entry_dest = np.asarray(entry_dest, dtype=np.int64)
        self._n_basis = k

    @property
    def domains(self) -> Dict[str, list]:
        """属性 → 候选值列表（schema 原始值，抽样后直接写回表）。"""
        return self._domains

    def wr_scale(
        self,
        q: np.ndarray,
        n_records: int,
        *,
        percentile: float = 100.0,
        scale_floor: float = 1e-12,
    ) -> float:
        """当轮加权残差的典型尺度（自适应 λ 的归一化分母）。

        scale = max(p{percentile}(|wr|), scale_floor)，默认 percentile=100
        即 max(|wr|)（最欠账的账）。用 max 保证 λ_eff·gain 有界：
        gain ≤ max|wr|·每格覆盖账数，故 λ_eff·gain ≤ λ·覆盖数（~16），
        不会随账本变平发散（p90 会：末期九成账全平 → 分母坍缩 →
        λ_eff 爆炸 → 引导退化成确定性贪心，实测比无引导还差）。
        wr 与 compute() 填 W 用的加权残差完全同源（同几何/同 floor/
        同 weights）。账全平时 |wr| 全 0 → 返回 scale_floor（此时
        gain 也全 0，λ/scale 再大也不产生倾斜，fail-safe）。
        """
        r = compute_residual(
            self._target, np.asarray(q), n_records,
            sigma=self._sigma, kappa=self._kappa,
            geometry=self._geometry, geometry_floor=self._geometry_floor,
        )
        wr = self._weights * r
        return max(float(np.percentile(np.abs(wr), percentile)), scale_floor)

    def compute(
        self,
        df: pd.DataFrame,
        q: np.ndarray,
        n_records: int,
        rows: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """按当前表与当前计数账本算逐格增益。

        Parameters
        ----------
        df : 当前合成表（掩码按它重算）
        q : (m,) 当前查询计数（与 evaluate_vectorized 返回的 q 同源）
        n_records : 残差比例口径的分母 N
        rows : (K,) 行下标或 None
            非 None 时只算这些行的增益（gain 逐行独立，子集结果与全表
            结果的对应行逐位一致）；None 时算全表。残差 wr 只依赖 q，
            与行子集无关。

        Returns
        -------
        gains : Dict[attr, np.ndarray (K, V_a)]
            每属性的逐行逐候选值增益（K=len(rows) 或全表 N）；
            欠账方向为正。

        注意：rows 子集编码使用子集自建的 cat_maps（按子集内出现顺序），
        与全表映射可能不同，但掩码语义只要求行编码与条件编码同映射，
        子集内自洽即语义正确。
        """
        if rows is not None:
            rows = np.asarray(rows)
            df = df.iloc[rows]
        X, col_index, cat_maps = _encode_table(df, self._schema)
        n = X.shape[0]

        # 基底掩码矩阵 B (N, K+1)：条件值用当轮 cat_maps 编码（映射跨轮不稳定）
        B = np.empty((n, self._n_basis + 1), dtype=float)
        for k, (attr, value) in enumerate(self._basis_conds):
            code = _encode_eq_value(attr, value, cat_maps)
            B[:, k] = X[:, col_index[attr]] == code
        B[:, self._n_basis] = 1.0

        # 加权残差（与 evaluate_vectorized 的 _batch_wr 同源）
        r = compute_residual(
            self._target, np.asarray(q), n_records,
            sigma=self._sigma, kappa=self._kappa,
            geometry=self._geometry, geometry_floor=self._geometry_floor,
        )
        wr = self._weights * r

        # 散射矩阵 W (K+1, D) 重填 + 一次矩阵乘
        W = np.zeros((self._n_basis + 1, self._d_total), dtype=float)
        np.add.at(W, (self._entry_basis, self._entry_dest), wr[self._entry_query])
        G = B @ W

        return {
            name: G[:, off:off + len(self._domains[name])]
            for name, off in self._dest_offset.items()
        }


def evaluate_directional_potential(
    df: pd.DataFrame,
    queries: List[Dict[str, Any]],
    schema: Schema,
    residual: np.ndarray,
    weights: Optional[np.ndarray] = None,
    batch_size: int = 256,
    device: Literal["numpy", "cuda", "cpu"] = "numpy",
    verbose: bool = True,
) -> np.ndarray:
    """按固定残差场计算每条记录的方向势能。

    对记录 ``x`` 返回

    ``potential(x) = sum_j weights[j] * residual[j] * a_j(x)``。

    这里不重新计算查询计数或残差，也不减去种群中心项。对于同一残差场下的
    局部转移 ``x -> x_prime``，中心项会严格相消，因此
    ``potential(x_prime) - potential(x)`` 正好是该转移的比例残差一阶方向量。
    函数不执行阈值筛选或随机操作，只为扩散转移核提供连续方向信号。
    """
    residual = np.asarray(residual)
    m = len(queries)
    if residual.shape != (m,):
        raise ValueError(
            f"residual 必须是长度与 queries 一致的一维数组，"
            f"得到 shape {residual.shape}，期望 ({m},)"
        )
    if residual.dtype.kind not in "iuf" or not np.all(np.isfinite(residual)):
        raise ValueError("residual 必须是有限数值数组")

    if weights is None:
        weights_array = np.ones(m, dtype=float)
    else:
        weights_array = np.asarray(weights)
        if weights_array.shape != (m,):
            raise ValueError(
                f"weights 必须是长度与 queries 一致的一维数组，"
                f"得到 shape {weights_array.shape}，期望 ({m},)"
            )
        if (
            weights_array.dtype.kind not in "iuf"
            or not np.all(np.isfinite(weights_array))
        ):
            raise ValueError("weights 必须是有限数值数组")
        weights_array = weights_array.astype(float, copy=False)

    if isinstance(batch_size, bool) or not isinstance(
        batch_size, (int, np.integer)
    ) or batch_size <= 0:
        raise ValueError(f"batch_size 必须是正整数，得到 {batch_size!r}")
    if device not in ("numpy", "cuda", "cpu"):
        raise ValueError(
            f"device 必须是 'numpy'、'cuda' 或 'cpu'，得到 {device!r}"
        )

    N = len(df)
    if N == 0 or m == 0:
        return np.zeros(N, dtype=float)

    weighted_residual = residual.astype(float, copy=False) * weights_array
    X, col_index, cat_maps = _encode_table(df, schema)
    fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig = (
        _compile_queries(queries, col_index, cat_maps)
    )
    fast_set = set(fast_orig)
    fallback_idx = [qi for qi in range(m) if qi not in fast_set]
    if fallback_idx and verbose:
        reasons = sorted({
            c["operator"]
            for qi in fallback_idx
            for c in queries[qi].get("conditions", [])
            if c["operator"] not in VECTORIZED_OPS
        })
        if any(queries[qi].get("type") == "halfspace" for qi in fallback_idx):
            reasons.append("type=halfspace")
        print(
            f"提示：{len(fallback_idx)} 个方向查询含未向量化算子/类型 {reasons}，"
            "已走慢路径。"
        )

    potential = np.zeros(N, dtype=float)
    fast_orig_array = np.asarray(fast_orig, dtype=np.intp)
    if device in ("cuda", "cpu"):
        potential += _directional_potential_torch(
            X,
            fast_cols,
            fast_ops,
            fast_lo,
            fast_hi,
            fast_valid,
            fast_orig_array,
            weighted_residual,
            int(batch_size),
            device,
        )
    else:
        for start in range(0, len(fast_orig), int(batch_size)):
            end = min(start + int(batch_size), len(fast_orig))
            mask = _batch_masks_numpy(
                X,
                fast_cols[start:end],
                fast_ops[start:end],
                fast_lo[start:end],
                fast_hi[start:end],
                fast_valid[start:end],
            )
            orig = fast_orig_array[start:end]
            potential += mask.astype(float) @ weighted_residual[orig]

    if fallback_idx:
        from table_diffevo.queries import eval_query_mask

        for qi in fallback_idx:
            potential += (
                eval_query_mask(df, queries[qi]).astype(float)
                * weighted_residual[qi]
            )

    return potential


def _directional_potential_torch(
    X,
    fast_cols,
    fast_ops,
    fast_lo,
    fast_hi,
    fast_valid,
    fast_orig,
    weighted_residual,
    batch_size,
    device,
):
    """在 torch 设备上累加固定残差方向势能，只回传长度 N 的结果。

    掩码比较在 float32 上进行（0/1 语义精确），但残差加权累加必须用
    float64：势能是全部查询项的和，float32 累加会在大残差场下产生
    ~1e-9 量级的绝对误差，破坏与 float64 参考路径的方向恒等。
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Use device='numpy' or install PyTorch."
        )
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    dev = torch.device(device)

    X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)
    cols_t = torch.as_tensor(fast_cols, dtype=torch.long, device=dev)
    ops_t = torch.as_tensor(fast_ops, dtype=torch.long, device=dev)
    lo_t = torch.as_tensor(fast_lo, dtype=torch.float32, device=dev)
    hi_t = torch.as_tensor(fast_hi, dtype=torch.float32, device=dev)
    valid_t = torch.as_tensor(fast_valid, dtype=torch.bool, device=dev)
    potential_t = torch.zeros(X_t.shape[0], dtype=torch.float64, device=dev)

    for start in range(0, len(fast_orig), batch_size):
        end = min(start + batch_size, len(fast_orig))
        mask = _batch_masks_torch(
            X_t,
            cols_t[start:end],
            ops_t[start:end],
            lo_t[start:end],
            hi_t[start:end],
            valid_t[start:end],
            torch,
        )
        orig = fast_orig[start:end]
        wr_t = torch.as_tensor(
            weighted_residual[orig], dtype=torch.float64, device=dev
        )
        potential_t += mask.to(torch.float64) @ wr_t

    return potential_t.cpu().numpy().astype(float)


def _run_batches_numpy(
    X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
    q, wr_full, fitness_accum, batch_size, want_fitness, batch_wr,
    query_attr_matrix=None, block_accum=None,
):
    """NumPy：对快路径组分块算掩码，边算边派生计数、残差与 fitness 第一项。

    query_attr_matrix / block_accum 非 None 时同批顺手累加分科分第一项
    M @ (wr ⊙ E)（见 evaluate_vectorized 的分科分公式）。
    """
    F = fast_cols.shape[0]
    fast_orig_arr = np.asarray(fast_orig, dtype=np.intp)
    for start in range(0, F, batch_size):
        end = min(start + batch_size, F)
        mask = _batch_masks_numpy(
            X,
            fast_cols[start:end], fast_ops[start:end],
            fast_lo[start:end], fast_hi[start:end], fast_valid[start:end],
        )  # (N, b)
        orig = fast_orig_arr[start:end]
        # 计数：每列有几个 True（这批查询计数在批内即完整）
        q[orig] = mask.sum(axis=0)
        # fitness 第一项：M @ wr（残差在批内算，只累加这批查询）
        if want_fitness:
            wr_b = batch_wr(orig)
            wr_full[orig] = wr_b
            mask_f = mask.astype(float)
            fitness_accum += mask_f @ wr_b
            if block_accum is not None:
                # (N, b) @ (b, A)：这批查询的加权残差按属性归属散射
                block_accum += mask_f @ (
                    wr_b[:, None] * query_attr_matrix[orig]
                )


def _run_batches_torch(
    X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
    q, wr_full, fitness_accum, batch_size, device, want_fitness, batch_wr,
    query_attr_matrix=None, block_accum=None,
):
    """PyTorch：对快路径组分块算掩码，计数与 fitness 在设备上算，只回传小结果。

    残差在批内算：先把这批计数搬回 CPU 填 q，用 batch_wr 算加权残差（numpy），
    再把 wr 搬到设备做 M @ wr。计数是整数、精确；残差用 compute_residual 保证语义一致。
    query_attr_matrix / block_accum 非 None 时分科分第一项也在设备上累加，
    最后一次性搬回。
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch not installed. Use device='numpy' or install PyTorch."
        )
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    dev = torch.device(device)

    X_t = torch.as_tensor(X, dtype=torch.float32, device=dev)
    cols_t = torch.as_tensor(fast_cols, dtype=torch.long, device=dev)
    ops_t = torch.as_tensor(fast_ops, dtype=torch.long, device=dev)
    lo_t = torch.as_tensor(fast_lo, dtype=torch.float32, device=dev)
    hi_t = torch.as_tensor(fast_hi, dtype=torch.float32, device=dev)
    valid_t = torch.as_tensor(fast_valid, dtype=torch.bool, device=dev)
    fitness_accum_t = (
        torch.zeros(X_t.shape[0], dtype=torch.float32, device=dev)
        if want_fitness else None
    )
    if block_accum is not None:
        attr_matrix_t = torch.as_tensor(
            query_attr_matrix, dtype=torch.float32, device=dev
        )
        block_accum_t = torch.zeros(
            (X_t.shape[0], attr_matrix_t.shape[1]),
            dtype=torch.float32, device=dev,
        )
    else:
        attr_matrix_t = block_accum_t = None

    F = fast_cols.shape[0]
    fast_orig_arr = np.asarray(fast_orig, dtype=np.intp)
    for start in range(0, F, batch_size):
        end = min(start + batch_size, F)
        mask = _batch_masks_torch(
            X_t,
            cols_t[start:end], ops_t[start:end],
            lo_t[start:end], hi_t[start:end], valid_t[start:end],
            torch,
        )  # (N, b)
        orig = fast_orig_arr[start:end]
        # 计数回传 CPU 填入 q（整数，精确）
        q[orig] = mask.sum(dim=0).cpu().numpy()
        if want_fitness:
            wr_b = batch_wr(orig)                       # numpy，依赖已填的 q[orig]
            wr_full[orig] = wr_b
            wr_bt = torch.as_tensor(wr_b, dtype=torch.float32, device=dev)
            mask_f = mask.float()
            fitness_accum_t += mask_f @ wr_bt
            if block_accum_t is not None:
                orig_t = torch.as_tensor(orig, dtype=torch.long, device=dev)
                block_accum_t += mask_f @ (
                    wr_bt[:, None] * attr_matrix_t[orig_t]
                )

    if want_fitness:
        # 只把 (N,) 的 fitness 累加项搬回 CPU
        fitness_accum += fitness_accum_t.cpu().numpy().astype(float)
        if block_accum_t is not None:
            block_accum += block_accum_t.cpu().numpy().astype(float)


def _handle_fallback(
    df, queries, fallback_idx, q, wr_full, fitness_accum,
    want_fitness, N, n_records, target, weights, sigma, kappa,
    residual_geometry="absolute", residual_geometry_floor=8.0,
    query_attr_matrix=None, block_accum=None,
):
    """
    回退组：用旧 evaluate_table 逐查询算（保证正确），计数填回原位置。

    fitness 的第一项 M @ wr 也要包含回退组的贡献，用旧 eval_query_mask 补上。
    残差用 compute_residual 逐查询算（σ/κ 语义一致）。
    block_accum 非 None 时分科分第一项同样按归属散射补上。
    """
    from table_diffevo.queries import eval_query_mask

    for qi in fallback_idx:
        mask = eval_query_mask(df, queries[qi])  # (N,) bool，旧逻辑
        q[qi] = int(mask.sum())
        if want_fitness:
            sig = None if sigma is None else np.asarray(sigma)[qi:qi+1]
            r = compute_residual(
                np.asarray(target)[qi:qi+1], q[qi:qi+1], n_records,
                sigma=sig, kappa=kappa,
                geometry=residual_geometry,
                geometry_floor=residual_geometry_floor,
            )[0]
            wr_qi = float(np.asarray(weights)[qi]) * r
            wr_full[qi] = wr_qi
            mask_f = mask.astype(float)
            fitness_accum += mask_f * wr_qi
            if block_accum is not None:
                block_accum += np.outer(
                    mask_f, wr_qi * query_attr_matrix[qi]
                )
