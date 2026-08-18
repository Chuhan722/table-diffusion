# Issue #53：`test_300x10` 分阶 held-out 只读诊断结果

## 结论

原 50 条测量查询的确放大了 `relative` 的 1-way 劣势，但跨数据反转**不只是查询设计造成的**：在全部 531 条未测量 2-way、冻结的 512 条 3-way 和 512 条 4-way 上，`relative` 的平均绝对计数误差都略高于 `absolute`。

平方根中间方法 `sqrt_relative` 没有形成统一赢家：它在未测量 2-way 和 4-way 上最好，在冻结 3-way 上最差；测量内 1-way 几乎追平 `absolute`，但 2-way 更差。因此当前结果不支持直接实现 `order_aware_relative`，也不支持把平方根方法设成跨 workload 默认值。

## 固定身份与信息流

本分析绑定：

```text
protocol commit  d427db68b927375a58e87ea8b172476e1ed5dcbd
analysis commit  219bf74ea753823058c0b2842d7c90a543d47079
source report    241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143
result report    cb88a5bbbd6de494fd97f60ca3984dfe53fe714379978137ae69436773feff24
```

- 只评价原 P=6 三臂、seed 310–312 的 9 张 `test_300x10/terminal_current.csv`，没有生成新表。
- 先根据公开域和 measured 语义指纹冻结未测量 2-way 身份，再读取原表附加答案。
- 公开标准 2-way cell 共 548 条，与 measured 精确重叠 17 条；其余 531 条全部纳入，没有按答案或误差抽样。
- 既有 3/4-way held-out 确定性身份重建一致，各 512 条，与 measured 重叠为 0。
- 本诊断明确记录 `raw_reference_data_accessed=true`、`privacy_budget_consumed=false`。
- 六组分别报告，报告中没有跨组总分。

未测量 2-way 查询身份 SHA-256 为：

```text
7d88a2db88a4576bb54bed341a3a8ccfbfc11f368662ad3b513e8fa863b5647f
```

## 六组结果

下表单位为每条查询的平均绝对计数误差，均在 3 个 paired seed 上汇总。每行只在本组内比较，不跨行加权。

| 查询组 | 条数 | `absolute` | `sqrt_relative` | `relative` | 本组最低 |
|---|---:|---:|---:|---:|---|
| measured 1-way | 25 | **0.6933** | 0.7333 | 1.3733 | `absolute` |
| measured 2-way | 20 | **0.8500** | 1.0833 | 0.9000 | `absolute` |
| measured 3-way | 5 | 0.9333 | 0.9333 | **0.8000** | `relative`，仅 5 条 |
| 全部未测量 2-way | 531 | 7.0929 | **6.8763** | 7.1620 | `sqrt_relative` |
| 冻结 held-out 3-way | 512 | **4.1296** | 4.2637 | 4.2435 | `absolute` |
| 冻结 held-out 4-way | 512 | 1.7760 | **1.7650** | 1.8737 | `sqrt_relative` |

### `relative` 对 `absolute`

| 查询组 | 平均误差差（计数） | paired seed 更好 / 更差 / 平局 |
|---|---:|---:|
| measured 1-way | +0.6800 | 0 / 3 / 0 |
| measured 2-way | +0.0500 | 1 / 2 / 0 |
| measured 3-way | -0.1333 | 1 / 1 / 1 |
| 全部未测量 2-way | +0.0691 | 1 / 2 / 0 |
| 冻结 held-out 3-way | +0.1139 | 1 / 2 / 0 |
| 冻结 held-out 4-way | +0.0977 | 1 / 2 / 0 |

三个未测量组的均值同向为正，按结果前固定解释属于 `supports_unmeasured_joint_query_weakness`。不过差值只有每 query 0.069–0.114 条记录，且每组都只是 2/3 seed 偏 `absolute`；这是描述性弱证据，不能声称稳定显著劣化。

### `sqrt_relative` 对 `absolute`

| 查询组 | 平均误差差（计数） | paired seed 更好 / 更差 / 平局 |
|---|---:|---:|
| measured 1-way | +0.0400 | 1 / 2 / 0 |
| measured 2-way | +0.2333 | 0 / 3 / 0 |
| measured 3-way | 0.0000 | 1 / 1 / 1 |
| 全部未测量 2-way | -0.2166 | 2 / 1 / 0 |
| 冻结 held-out 3-way | +0.1341 | 1 / 2 / 0 |
| 冻结 held-out 4-way | -0.0111 | 2 / 1 / 0 |

方向在查询阶数间反转，固定分类为 `mixed_no_universal_winner`。未测量 2-way 的改善最大，但 seed 312 反向；4-way 的均值差几乎为零。

## 对 test 查询设计的回答

现有 test 测量 workload 有结构偏重：25/50 是由初始化边缘分布直接给出的 1-way；`relative` 在这里相对 `absolute` 每 query 多 0.68 条误差，远大于三个未测量组各自的差值。因此只看原 50 条 aggregate 会明显夸大 `relative` 的劣势。

但把全部未测量 2-way 和冻结 3/4-way 补齐后，`relative` 仍没有在任何一个未测量组的均值上胜过 `absolute`。所以准确结论是：**原查询设计放大了问题，但没有凭空制造问题。**

## 下一步

1. 不实现先前候选 `order_aware_relative`：其 order>=2 部分采用 `relative`，而当前三个未测量联合查询组都没有支持这一替换。
2. 不继续扫 gamma/floor，也不依据这 3 个已见 seed 调混合系数。
3. 若要继续 residual 板块，先冻结一轮 fresh-seed、`test_300x10` 专用复核，仍比较三臂并把未测量 2/3/4-way 分开；重点检验当前只有 2:1 的小差值是否可复现。
4. fresh seeds 若仍为 mixed，就停止寻找单一跨数据 residual，把 `absolute`、`sqrt_relative`、`relative` 记录为 workload 级 Pareto 选择，并先完成 Issue #53 尚缺的跨方案质量—成本门禁，再进入 donor/alpha。

本结果是已见主实验后的 development mechanism diagnostic，不是 fresh selection evidence。
