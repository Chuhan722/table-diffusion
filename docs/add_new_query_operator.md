# 如何为向量化查询评价添加新算子

## 概述

当前向量化快路径支持三种查询算子：`==`、`>=`、`between`。

**如果你的新查询用到了这三种之外的算子**（如 `!=`、`<`、`in` 等），会发生什么：
- 含新算子的查询**自动走旧 `evaluate_table` 慢路径**（保证算对，只是慢）
- 终端打印提醒："N 个查询含未向量化算子 {新算子}，已走慢路径"

**如果你希望新算子也走快路径**（向量化加速），按本文档流程补实现。

---

## 前置要求

- 新算子的**语义**已经明确（比如 `!=` 就是"不等于"）
- 新算子能用**逐元素广播比较**实现（NumPy/PyTorch 支持的操作）
- 熟悉 Python、NumPy、pytest 基础

---

## 完整流程（以 `!=` 为例）

### 第 1 步：在旧路径定义新算子的语义

**文件**：`src/table_diffevo/queries.py`

在 `eval_condition` 函数里加一个 `elif` 分支：

```python
def eval_condition(df: pd.DataFrame, condition: Dict[str, Any]) -> pd.Series:
    attr = condition["attribute"]
    op = condition["operator"]

    if op == "==":
        value = _coerce_to_column_type(df[attr], condition["value"])
        return df[attr] == value

    elif op == ">=":
        return df[attr] >= condition["value"]

    elif op == "between":
        lower = condition["lower"]
        upper = condition["upper"]
        return (df[attr] >= lower) & (df[attr] <= upper)

    elif op == "!=":  # 新增：不等于算子
        value = _coerce_to_column_type(df[attr], condition["value"])
        return df[attr] != value

    else:
        raise ValueError(f"不支持的操作符: {op}")
```

**为什么这步最重要**：
- 旧路径（`eval_condition` + `evaluate_table`）是**正确性基准**
- 向量化只是它的加速镜像，必须先在这里定义"什么是正确的"
- 测试会用旧路径的结果当标准答案，对比向量化算得对不对

---

### 第 2 步：把新算子加进白名单

**文件**：`src/table_diffevo/vectorized_eval.py`

在文件开头找到白名单和编码定义，加入新算子：

```python
# 快路径支持的算子白名单。新算子若不在此集合，整条查询走旧 evaluate_table 回退。
VECTORIZED_OPS = {"==", ">=", "between", "!="}  # 加了 !=

# 算子的整数编码（编译成紧凑数组用，避免循环里比字符串）
_OP_EQ = 0
_OP_GE = 1
_OP_BETWEEN = 2
_OP_NE = 3  # 新增
_OP_CODE = {"==": _OP_EQ, ">=": _OP_GE, "between": _OP_BETWEEN, "!=": _OP_NE}
```

---

### 第 3 步：在查询编译里处理新算子的值

**文件**：`src/table_diffevo/vectorized_eval.py`

找到 `_compile_queries` 函数，在处理条件值的循环里加分支：

```python
def _compile_queries(...):
    ...
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
        elif op in ("==", "!="):  # 改这里：两种算子都用 lo 存 value
            lo_row[k] = _encode_eq_value(attr, c["value"], cat_maps)
```

**解释**：
- `!=` 和 `==` 一样，都有一个 `value` 字段
- 把 value 编码后存进 `lo_row[k]`（低位数组），向量化时会用到
- `between` 特殊，有 `lower` 和 `upper` 两个值，分别存 `lo` 和 `hi`

---

### 第 4 步：在掩码计算里加广播比较

需要改**两个函数**（NumPy 版和 Torch 版），逻辑完全一样。

#### 4a. NumPy 版本

**文件**：`src/table_diffevo/vectorized_eval.py`  
**函数**：`_batch_masks_numpy`

```python
def _batch_masks_numpy(X, cols, ops, lo, hi, valid):
    N = X.shape[0]
    B = cols.shape[0]
    mask = np.ones((N, B), dtype=bool)

    for k in range(cols.shape[1]):
        col_k = cols[:, k]          # (B,)
        op_k = ops[:, k]            # (B,)
        lo_k = lo[:, k]             # (B,)
        hi_k = hi[:, k]             # (B,)
        valid_k = valid[:, k]       # (B,)

        vals = X[:, col_k]          # (N, B)

        # 按算子分别算比较结果（对整批一次算完）
        eq_res = vals == lo_k[None, :]
        ge_res = vals >= lo_k[None, :]
        bt_res = (vals >= lo_k[None, :]) & (vals <= hi_k[None, :])
        ne_res = vals != lo_k[None, :]  # 新增：不等于

        # 按每个查询的算子选择对应结果
        cond = np.where(
            op_k[None, :] == _OP_EQ, eq_res,
            np.where(op_k[None, :] == _OP_GE, ge_res,
            np.where(op_k[None, :] == _OP_NE, ne_res, bt_res)),  # 新增分支
        )

        # padding 槽（valid=False）当作 True，不参与 AND
        cond = np.where(valid_k[None, :], cond, True)
        mask &= cond

    return mask
```

#### 4b. PyTorch 版本

**文件**：`src/table_diffevo/vectorized_eval.py`  
**函数**：`_batch_masks_torch`

```python
def _batch_masks_torch(X_t, cols_t, ops_t, lo_t, hi_t, valid_t, torch):
    N = X_t.shape[0]
    B = cols_t.shape[0]
    mask = torch.ones((N, B), dtype=torch.bool, device=X_t.device)

    for k in range(cols_t.shape[1]):
        col_k = cols_t[:, k]
        op_k = ops_t[:, k]
        lo_k = lo_t[:, k]
        hi_k = hi_t[:, k]
        valid_k = valid_t[:, k]

        vals = X_t[:, col_k]        # (N, B)

        eq_res = vals == lo_k.unsqueeze(0)
        ge_res = vals >= lo_k.unsqueeze(0)
        bt_res = (vals >= lo_k.unsqueeze(0)) & (vals <= hi_k.unsqueeze(0))
        ne_res = vals != lo_k.unsqueeze(0)  # 新增：不等于

        cond = torch.where(
            (op_k == _OP_EQ).unsqueeze(0), eq_res,
            torch.where((op_k == _OP_GE).unsqueeze(0), ge_res,
            torch.where((op_k == _OP_NE).unsqueeze(0), ne_res, bt_res)),  # 新增分支
        )
        cond = torch.where(valid_k.unsqueeze(0), cond, torch.ones_like(cond))
        mask &= cond

    return mask
```

**关键点**：
- `ne_res = vals != lo_k[...]`：广播比较，一次算完这批所有查询的"不等于"掩码
- `np.where` / `torch.where` 嵌套：根据算子编码选择对应的比较结果
- 两个版本逻辑完全一样，只是 NumPy 用 `[None, :]`，PyTorch 用 `.unsqueeze(0)`

---

### 第 5 步：写测试验证新算子

**文件**：`tests/test_vectorized_eval.py`

加一个测试类，验证新算子的向量化结果与旧路径一致：

```python
class TestNewOperatorNE:
    """测试 != 算子向量化正确性"""

    def test_ne_operator_matches_legacy(self):
        """!= 算子：向量化计数与旧 evaluate_table 逐元素相同"""
        # 构造一些含 != 的查询
        queries = [
            {
                "id": "NE1",
                "type": "single",
                "conditions": [
                    {"attribute": "children", "operator": "!=", "value": "2_plus"}
                ],
                "result": 999,  # 随便填，测试只对比计数
            },
            {
                "id": "NE2",
                "type": "single",
                "conditions": [
                    {"attribute": "age", "operator": "!=", "value": 35}
                ],
                "result": 999,
            },
            {
                "id": "NE3",
                "type": "double",
                "conditions": [
                    {"attribute": "age", "operator": "!=", "value": 25},
                    {"attribute": "children", "operator": "==", "value": "0"},
                ],
                "result": 999,
            },
        ]

        schema = load_schema("configs/test_300x10/schema.yaml")
        df = load_data("data/test_300x10/test_300x10.csv")

        # 旧路径（正确性基准）
        q_old = evaluate_table(df, queries)

        # 新路径（向量化）
        q_new, _, _ = evaluate_vectorized(
            df, queries, schema, batch_size=16, device="numpy",
            want_fitness=False, verbose=False,
        )

        # 必须逐元素相同（整数，精确）
        np.testing.assert_array_equal(q_old, q_new)

    @pytest.mark.parametrize("device", _devices())
    def test_ne_with_fitness_all_devices(self, device):
        """!= 算子：含 fitness 路径也正确（numpy/cuda）"""
        queries = [
            {
                "id": "NE_FIT",
                "type": "single",
                "conditions": [
                    {"attribute": "occupation", "operator": "!=", "value": "tech_support"}
                ],
                "result": 999,
            }
        ]
        schema = load_schema("configs/test_300x10/schema.yaml")
        df = load_data("data/test_300x10/test_300x10.csv")
        target = np.array([q["result"] for q in queries], dtype=float)

        q_old = evaluate_table(df, queries)
        resid = compute_residual(target, q_old, len(df))
        fit_old = compute_fitness(df, queries, resid, q_old)

        q_new, _, fit_new = evaluate_vectorized(
            df, queries, schema, target=target, n_records=len(df),
            batch_size=16, device=device, want_fitness=True, verbose=False,
        )

        np.testing.assert_array_equal(q_old, q_new)
        atol = 1e-9 if device == "numpy" else 1e-3
        np.testing.assert_allclose(fit_old, fit_new, atol=atol)
```

**解释**：
- 第一个测试：构造含 `!=` 的查询，对比新旧路径计数是否逐元素相同
- 第二个测试：验证 fitness 路径也对（numpy 逐位，cuda float32 允许极小差）
- `queries` 里的 `result` 字段随便填（测试不用它，只对比实际计数）

---

### 第 6 步：运行测试

```bash
# 只跑你的新测试
python -m pytest tests/test_vectorized_eval.py::TestNewOperatorNE -v

# 全套测试不能破
python -m pytest tests/test_vectorized_eval.py -q
python -m pytest -q
```

全部通过 → 新算子补完 ✅

---

## 改动清单（5 个文件）

| 步骤 | 文件 | 改什么 | 关键点 |
|------|------|--------|--------|
| 1 | `src/table_diffevo/queries.py` | `eval_condition` 加 `elif op == "!="` 分支 | 定义"什么是正确的" |
| 2 | `src/table_diffevo/vectorized_eval.py` | 白名单加 `"!="`，编码加 `_OP_NE` | 告诉向量化"认识"这个算子 |
| 3 | `src/table_diffevo/vectorized_eval.py` | `_compile_queries` 里 `!=` 和 `==` 一样处理 | 把 value 编码存进数组 |
| 4a | `src/table_diffevo/vectorized_eval.py` | `_batch_masks_numpy` 加 `ne_res` 和分支 | NumPy 广播比较 |
| 4b | `src/table_diffevo/vectorized_eval.py` | `_batch_masks_torch` 加 `ne_res` 和分支 | PyTorch 广播比较 |
| 5 | `tests/test_vectorized_eval.py` | 加测试类 `TestNewOperatorNE` | 锁住"新旧一致" |

---

## 关键原则

1. **旧路径是真相**：`eval_condition` 定义算子语义，向量化只是它的加速镜像
2. **测试是保险**：用旧路径结果当标准答案，验证向量化没算错
3. **计数必须精确**：整数比较，numpy/cuda 必须逐元素相同（`np.testing.assert_array_equal`）
4. **fitness 允许小差**：numpy 逐位一致（atol=1e-9），cuda float32 允许极小差（atol=1e-3）

---

## 常见问题

### Q1: 我的新算子不是简单比较，能向量化吗？

**看情况**：
- 如果能用 NumPy/PyTorch 的逐元素操作实现（如 `&`、`|`、`~`、数学函数），可以
- 如果需要复杂逻辑（循环、条件分支、字符串操作），难度大，建议留在回退组

### Q2: 新算子要用两个比较值怎么办？

参考 `between` 的做法：
- `lo_row[k]` 存 `lower`，`hi_row[k]` 存 `upper`
- 掩码计算时：`bt_res = (vals >= lo_k) & (vals <= hi_k)`

### Q3: 我改了代码，测试报"计数不相等"怎么办？

**常见原因**：
- 忘了在 `_batch_masks_torch` 里也加分支（只改了 numpy 版）
- `np.where` 嵌套层级错了，算子选择逻辑不对
- 编码时 `_encode_eq_value` 用错了（数值列直接 float，字符串列要查映射）

**调试技巧**：
```python
# 在测试里打印具体哪些查询算错了
for i, (old, new) in enumerate(zip(q_old, q_new)):
    if old != new:
        print(f"查询 {i}: 旧={old}, 新={new}, 差={new-old}")
```

### Q4: 测试通过后，还需要做什么？

- 提交前运行一次完整测试套件：`python -m pytest -q`
- 在 `PROJECT_STATUS.md` 记录你加的算子（可选，但推荐）
- git commit 时写清楚改了什么

---

## 需要帮助？

如果按这个流程操作遇到问题，检查：
1. 旧路径（`eval_condition`）能否正确处理你的新算子？
2. 白名单和编码是否加对了？
3. NumPy 和 Torch 两个版本都改了吗？
4. 测试里构造的查询是否合法（属性名、value 类型对不对）？

如果还不清楚，可以参考已有的三种算子（`==`/`>=`/`between`）的实现作为模板。
