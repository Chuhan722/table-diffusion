# 查询评价器重构记录

## 改动时间
2025年（commit 1986ec3）

## 改动原因
为了支持适应度计算，需要获取"每条记录对每个查询的贡献"（掩码），而不仅仅是求和后的计数。同时要避免在大规模查询场景下构建完整的 (N × m) 贡献矩阵（可能几GB）。

## 具体改动

### 1. 新增函数：`eval_query_mask`

**功能：** 返回单个查询的布尔掩码（不求和）

**签名：**
```python
def eval_query_mask(df: pd.DataFrame, query: Dict[str, Any]) -> np.ndarray
```

**返回：** 长度 N 的布尔数组，True 表示该记录满足查询

**用途：**
- `evaluate_table` 内部调用它，再求和得到计数
- `fitness.py` 逐查询调用它，累加适应度（不建矩阵，内存 O(N)）

### 2. 重构函数：`eval_query`

**改动前：** 内部自己算掩码，再求和
```python
def eval_query(df, query):
    mask = pd.Series([True] * len(df), index=df.index)
    for condition in query["conditions"]:
        mask &= eval_condition(df, condition)
    return int(mask.sum())
```

**改动后：** 调用 `eval_query_mask`，再求和
```python
def eval_query(df, query):
    mask = eval_query_mask(df, query)
    return int(mask.sum())
```

**对外行为：** 完全不变（输入输出一致）

### 3. `evaluate_table` 无改动

内部实现改为调用 `eval_query_mask`，但对外接口和行为完全不变。

## 设计原理

**逐查询累加 vs 建矩阵：**

| 方案 | 内存 | 说明 |
|------|------|------|
| 建完整矩阵 | O(N × m) | 5万×5万 = 2.5GB，不可接受 |
| 逐查询累加 | O(N) | 每次只存一个查询的掩码，用完就丢 |

**代价：** 掩码被计算两次（evaluate_table 一次，fitness 一次），但在查询量很大时，多花计算时间（几秒）换来内存从 GB 降到 KB 是值得的。

**正确性保证：** 两处都调用同一个底层函数 `eval_query_mask`，相同输入 → 相同掩码 → 计数与适应度永远一致。

## 测试验证

**已有测试全部通过：** 9 个查询测试在重构后仍然通过，证明对外行为未变。

**新增测试：**
- `test_eval_query_mask_basic`：验证掩码返回正确
- `test_eval_query_mask_multiple_conditions`：验证多条件（AND）
- `test_eval_query_uses_eval_query_mask`：验证重构后计数一致

## 影响范围

**受影响的模块：**
- ✅ `queries.py`：内部重构，对外无影响
- ✅ `fitness.py`：依赖新函数 `eval_query_mask`
- ❌ 其他模块：无影响（objective.py / utils.py 等只调用 `evaluate_table`）

**向后兼容：** 完全兼容，现有代码无需修改。
