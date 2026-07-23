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
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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

    Returns
    -------
    q : np.ndarray, shape (m,), int
        计数向量。整数，与旧 evaluate_table 逐位相同。
    residual : np.ndarray, shape (m,), float 或 None
        比例残差 ε（want_fitness=False 时为 None）。与 compute_residual 一致。
    fitness : np.ndarray, shape (N,), float 或 None
        每条记录的适应度（want_fitness=False 时为 None）。

    Notes
    -----
    **计数正确性**：整数比较，numpy/cuda 与旧路径逐元素相同。
    **残差**：内部调用 objective.compute_residual（σ/κ 语义完全一致）。
    **fitness 公式**：fitness = M @ wr − (wr·p)，p = q/N，wr = w*residual。
    与旧 compute_fitness 的逐查询累加数学等价（numpy 路径逐位一致）。
    **回退组**：含未向量化算子的查询走旧 evaluate_table，计数按原顺序填回；
    这些查询的 fitness 贡献也用旧掩码逻辑补上（见实现）。
    """
    m = len(queries)
    N = len(df)
    if n_records is None:
        n_records = N

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
        bad_ops = sorted({
            c["operator"]
            for qi in fallback_idx
            for c in queries[qi]["conditions"]
            if c["operator"] not in VECTORIZED_OPS
        })
        print(
            f"提示：{len(fallback_idx)} 个查询含未向量化算子 {bad_ops}，"
            f"已走慢路径（旧 evaluate_table）。如需加速请补向量化实现。"
        )

    q = np.zeros(m, dtype=np.int64)
    # 全量 wr（加权残差）在扫描中逐批填入；扫描后用它算 fitness 常数项
    wr_full = np.zeros(m, dtype=float) if want_fitness else None
    fitness_accum = np.zeros(N, dtype=float) if want_fitness else None

    # 每批算残差用的闭包：counts 完整 → residual 只依赖自身 count，可在批内算
    def _batch_wr(orig):
        """给定这批查询的原始下标，算它们的加权残差 wr（依赖已填好的 q[orig]）。"""
        sig = None if sigma is None else np.asarray(sigma)[orig]
        r = compute_residual(
            np.asarray(target)[orig], q[orig], n_records, sigma=sig, kappa=kappa
        )
        return np.asarray(weights)[orig] * r

    if device in ("cuda", "cpu"):
        _run_batches_torch(
            X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
            q, wr_full, fitness_accum, batch_size, device, want_fitness, _batch_wr,
        )
    else:
        _run_batches_numpy(
            X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
            q, wr_full, fitness_accum, batch_size, want_fitness, _batch_wr,
        )

    # 回退组：用旧 evaluate_table 逐查询算，计数填回原位置；fitness 贡献用旧掩码补
    if fallback_idx:
        _handle_fallback(
            df, queries, fallback_idx, q, wr_full, fitness_accum,
            want_fitness, N, n_records, target, weights, sigma, kappa,
        )

    residual = fitness = None
    if want_fitness:
        # fitness = M @ wr − (wr·p)，常数项对所有记录相同
        p = q / N
        const = float(np.dot(wr_full, p))
        fitness = fitness_accum - const
        # 残差整体重算一次（与 compute_residual 完全一致，返回给主循环用）
        sig = None if sigma is None else np.asarray(sigma)
        residual = compute_residual(
            np.asarray(target), q, n_records, sigma=sig, kappa=kappa
        )

    return q, residual, fitness


def _run_batches_numpy(
    X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
    q, wr_full, fitness_accum, batch_size, want_fitness, batch_wr,
):
    """NumPy：对快路径组分块算掩码，边算边派生计数、残差与 fitness 第一项。"""
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
            fitness_accum += mask.astype(float) @ wr_b


def _run_batches_torch(
    X, fast_cols, fast_ops, fast_lo, fast_hi, fast_valid, fast_orig,
    q, wr_full, fitness_accum, batch_size, device, want_fitness, batch_wr,
):
    """PyTorch：对快路径组分块算掩码，计数与 fitness 在设备上算，只回传小结果。

    残差在批内算：先把这批计数搬回 CPU 填 q，用 batch_wr 算加权残差（numpy），
    再把 wr 搬到设备做 M @ wr。计数是整数、精确；残差用 compute_residual 保证语义一致。
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
            fitness_accum_t += mask.float() @ wr_bt

    if want_fitness:
        # 只把 (N,) 的 fitness 累加项搬回 CPU
        fitness_accum += fitness_accum_t.cpu().numpy().astype(float)


def _handle_fallback(
    df, queries, fallback_idx, q, wr_full, fitness_accum,
    want_fitness, N, n_records, target, weights, sigma, kappa,
):
    """
    回退组：用旧 evaluate_table 逐查询算（保证正确），计数填回原位置。

    fitness 的第一项 M @ wr 也要包含回退组的贡献，用旧 eval_query_mask 补上。
    残差用 compute_residual 逐查询算（σ/κ 语义一致）。
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
            )[0]
            wr_qi = float(np.asarray(weights)[qi]) * r
            wr_full[qi] = wr_qi
            fitness_accum += mask.astype(float) * wr_qi
