# Issue #53 B+C 问题一：A/R 双通道单种子开发筛查结果前协议

## 1. 目的与结论边界

R8 和平方根开发筛查已表明，把所有查询压成一套固定分母会在稀有与常见
查询之间产生折中。本阶段只检验一个新候选：使用彼此可比的绝对 A 和相对 R
两个通道，并让较差通道决定 C 能量，能否同时保留常见查询收益与稀有查询
保护。

候选是在读取 R8/平方根结果后提出的，因此仍是开发筛查，不是独立确认。
通过只表示值得另立新种子确认协议；不得据此声称全面优越，不得修改公共默认。

## 2. 唯一候选公式

记 `e_j=|target_j-current_j|`，表行数为 `N`，查询数为 `J`，
`P={j:target_j>0}`：

```text
A = sum_all(e_j) / (N*J)

S = sum_{k in P}(1/target_k)
R = [sum_{j in P}(e_j/target_j) / S] / N    (P nonempty)
R = 0                                        (P empty)

E = max(A,R)
```

配置身份固定为：

```text
gap_l1_weighting = "dual_abs_relative_max"
gap_l1_max_weight_ratio = null
```

`target=0` 只进入 A，不进入 R，不单设 Z。若无正目标则退化为 `E=A`。
`floor` 仅因公共 API 兼容而保留，不进入候选公式。没有可调 floor、指数、
混合系数、权重比上限或数据集专用尺度。

每个条件微步必须先在关闭/打开两侧各自计算 `E_b=max(A_b,R_b)`，再取
`score=E_0-E_1`。切换点不加 epsilon、滞回或随机平局。

## 3. 唯一算法差量与不变量

只改 C 核的查询误差聚合。以下均沿用已完成的平方根筛查：

- B 残差方向及 `residual_geometry="relative"` / `residual_geometry_floor=8`;
- `gap_l1_sweeps=8`，每轮精确 `8*K` 个微步；
- `strength=2`、`eta=0.5`、logit clip `30`，以本候选孤立分数 RMS 一次性定尺；
- 数据、查询、目标、边际初始化、种子、供体、参与行、后置突变与 RNG 域；
- `tol=+inf`、`max_retries=0`，唯一提案无条件接续；
- P=6 自然工作量停止、6000 轮/候选上限、固定检查点和 terminal-current 输出。

`legacy_relative`、`bounded_relative`、`sqrt_target_relative` 的公式、轨迹和
诊断必须保持兼容。

## 4. 任务矩阵与旧产物复用

只新增两条候选轨迹：

| 顺序 | 数据集 | 种子 | 方法 |
|---:|---|---:|---|
| 1 | `test_300x10` | 9908 | `gap_dual_abs_relative_max_s8` |
| 2 | `nltcs` | 9908 | `gap_dual_abs_relative_max_s8` |

不重跑 legacy、R8 或平方根。比较复用之前已完整采集并独立审计的六条轨迹。
复用成立的前提是输入身份一致，且旧三种核模式的回归与冻结轨迹通过。

种子 9908 已被之前的开发筛查使用；继续使用只为做配对机理诊断，不得并入
后续正式胜率或据本轮结果挑确认种子。

## 5. 结果前目标权重审计

在生成前只读取冻结查询 JSON 中已附着的整数目标，不读取参考表、旧终表或
候选输出。独立计算 A 的等权、R 的正目标逆权重及归一化常数，并对每条查询
保存原序号、ID、目标、冻结箱、A 单位误差影响、R 原始/归一化权重与 R
单位误差影响。

整体预期：

| 数据集 | `J` | `N` | 零/正目标数 | `sum(1/target)` | R 内正权重跨度 |
|---|---:|---:|---:|---:|---:|
| `test_300x10` | 50 | 300 | 3 / 47 | 5.243603286242844 | 88 |
| `nltcs` | 1001 | 16181 | 0 / 1001 | 3.8332971794614985 | 1091 |

这个跨度是 R 通道内部优先级，不是整个 `max(A,R)` 目标的单一权重比上限。
查询数、目标向量 SHA、分箱成员 SHA 或上述汇总任一不符即失败，不得生成。

## 6. 生成有效性

两条候选轨迹必须全部完成后才能形成 collection。任一情况统一为
`execution_invalid`，不得解释质量：

- 提交、协议、runner、输入、参数或旧基线哈希不符；
- 不是唯一 A/R `max` 候选，或出现非空 `max_weight_ratio` / smoothing / floor 应用；
- 正目标数、逆目标归一化常数、R 权重跨度或零目标策略与审计不符；
- 不是精确 `8*K` 微步，出现非有限条件值、精确 0/1 概率或 logit 裁剪；
- RNG、初始化、检查点、停止或 terminal-current 契约漂移；
- 读取参考表/未测查询、提前比较部分 collection、重跑旧臂或选检查点。

collection 只包含生成有效性与转移审计，不发布质量指标。完整 collection 哈希
固定后必须先停止；离线评价需要用户后续单独授权。

## 7. 结果前通过门

执行有效后，以 legacy 为主安全基线、平方根为最近的稀有查询对照，按以下顺序分类：

1. NLTCS 正式 measured normalized L1 与常见箱均必须严格低于 legacy，否则
   `common_mechanism_not_retained`；
2. NLTCS 稀有箱必须不超过 `1.25*legacy`，且严格低于平方根，否则
   `rare_query_protection_not_recovered`；
3. `test_300x10` 正式 measured normalized L1 不得超过 `1.05*legacy`，两数据集
   one-way safety 各自不得超过 `1.05*legacy`，否则
   `measured_or_one_way_safety_risk`；
4. 全部满足才为 `advance_to_fresh_seed_confirmation`。

所有固定检查点与其他分箱必须完整报告，但不得事后加门、选最好检查点或更换终表。

## 8. 执行边界

结果前允许完成源码、测试、独立目标权重审计、执行接线、只读 plan 和环境
preflight。哈希匹配不替代用户对真实 GPU 生成的单独授权。

不得自动启动两条轨迹、读取不存在的候选质量、运行离线评价/独立审计、
扩大种子、搜索参数、修改默认、推送或操作 PR #69。
