# Issue #53 残差几何查询级诊断口径

日期：2026-08-18

## 1. 身份与用途

本诊断发生在平方根残差 P=6 两数据三臂聚合结果已经可见之后，固定读取：

```text
source report SHA  241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143
source cases       2 datasets × 3 arms × 3 seeds = 18
source outputs     terminal-current tables
```

它只用于解释 `test_300x10` 与 `nltcs` 为何偏好相反，并为下一候选提供设计信息。它是明确的
result-aware development diagnostic，不是结果前验证、显著性检验、canonical 选择证据或新的正式实验。
不生成新表、不读取 raw reference table、不消耗隐私预算，也不允许修改分箱阈值或输入路径。

固定入口：

```text
scripts/analyze_issue53_residual_geometry_queries.py
```

## 2. 查询频率分箱

目标计数为 `y`，表记录数为 `N`，沿用既有 relative floor=8 和 Issue #57 的 rare `p<0.05` 定义；
另以固定 `p=0.20` 区分中等与常见查询：

```text
zero         y == 0
below_floor  0 < y < 8
rare         y >= 8 and y/N < 0.05
medium       0.05 <= y/N < 0.20
common       y/N >= 0.20
```

这些阈值在读取 query-level 三臂误差之前固定，不根据输赢移动。空分箱保留计数 0，但不产生汇总行。

## 3. 查询结构

查询阶数定义为条件中不同属性的数量，不依赖查询文件中的文字标签。

结构重叠只看 workload 属性集合。对查询属性集合 `A_j`，计算它与全部其他查询属性集合的平均
Jaccard：

```text
overlap_j = mean_{k != j} |A_j ∩ A_k| / |A_j ∪ A_k|
```

每个数据集只根据该输入 workload 的 overlap 分布计算 q25/q75：`<=q25` 为 low，`>=q75` 为 high，
其余为 middle。阈值完全不读取生成结果；报告必须记录实际 q25/q75。

## 4. 固定汇总

每张终态表重新评价同一 measured workload，并逐 query/seed/arm 记录 terminal answer、signed error 和
absolute error。每个分层固定报告：

- 三臂平均绝对计数误差、归一化误差、对整个数据集 mean L1 的贡献及该臂总误差占比；
- exact-match rate；
- 每个 query/seed 上最小绝对误差的 fractional win rate，平局在获胜臂间平分 1 票；
- `sqrt−absolute`、`relative−sqrt`、`relative−absolute` 的配对平均误差差和 better/tie/worse 数量。

固定拆解维度：overall、frequency、order、overlap、frequency×order、frequency×overlap。另输出完整
`query_seed_errors.csv` 和按三 seed 平均的 `query_summary.csv`。各臂 overall L1 必须与 source report
精确复算一致，否则 fail closed。

## 5. 解释边界

查询之间高度相关，三个 seed 也不足以支持把 query 当独立样本做常规显著性检验，因此本诊断不报告
p-value。若结果显示稳定频率交叉点，只能据此提出 selector/双尺度组合；候选公式和门禁必须在新 seed
结果前另行冻结。若频率效应被 order/overlap 分层明显解释，则不得把简单频率阈值直接实现为默认规则。
