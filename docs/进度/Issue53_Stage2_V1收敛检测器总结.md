# Issue #53 Stage 2：V1 收敛检测器方法与正式验证总结

## 结论先行

V1 已完成从轨迹契约、无阈值量程、development 校准、协议冻结到独立 validation
回放的完整研究闭环。正式结果为：

```text
does_not_support_frozen_detector_on_validation
```

因此，V1 **不能接入在线自动停止**。这个结果否决的是当前这套检测与验证协议，
不是在证明 factorized Gibbs 核没有收敛，也不是在比较 independent 与 factorized
Gibbs 的最终生成质量。

正式失败来自 20 条 validation 轨迹中的 1 条：
`test_300x10 / seed 220 / factorized_gibbs` 在已经给出候选停止点后出现连续 6 次
不稳定检查，触发了预先冻结的“四连败持续再漂移”门禁。事后只读诊断显示该轨迹的
长期 L1 没有持续恶化，更像一次随后恢复的局部波动。因此，下一版需要重新设计统计
证据和状态语义，而不能用这批 validation seeds 回调 V1 阈值。

## 1. 这项工作回答什么

目标是为无门控、持续运动的 current-state 生成过程建立一个只看单条运行轨迹的停止
判据。检测器不读取：

- 初始表内容或 seed；
- dataset、kernel 名称；
- `alpha/rho/eta/mu/tau/s0/clip` 等生成参数；
- 历史 best table；
- held-out 质量或源数据。

检测器只回答“当前轨迹是否进入稳定且仍有运动的区域”。它不回答这个区域的 L1 是否
足够低。外层稳定性与稳态质量仍是两个独立问题。

## 2. Stage 2A：统一轨迹与离线回放

### 2.1 状态语义

- `S0` 只用于复现审计，不进入收敛窗口；
- 每个真实完成的 `post_round` 记录一个 current state；
- 被拒绝的 proposal 记录为 self-transition，实际运动量为 0；
- proposal 前终止时不虚构新状态；
- 最终 current table 与历史 best diagnostic 明确分离。

每个状态记录 measured query vector、current normalized L1、current squared loss、
唯一行比例、归一化行熵、实际改行/改单元格/改查询数量、查询空间运动量，以及表和
RNG 哈希。attempted work 与 applied movement 分开记录，避免把“尝试过很多提案”误当成
“状态真的在运动”。

轨迹标量使用严格 JSON，查询向量矩阵使用禁止 pickle 的压缩 NPZ。保存、加载、哈希、
shape、dtype、未知字段和终止原因均采用 fail-closed 校验。开启轨迹观测不允许改变表、
已有历史、查询评价次数或 RNG 终点。

### 2.2 V1 窗口证据

V1 在每个检查点读取三个相邻、互不重叠、等长的 `post_round` 窗口 A/B/C。对三个窗口
的所有两两组合取最坏差异：

| 分组 | 证据 | 含义 |
|---|---|---|
| 查询稳定性 | `query_mean_shift` | 各窗口查询均值向量之差先对查询取 mean，再在三个窗口对中取最大 |
| 查询稳定性 | `query_p95_shift` | 各窗口查询均值向量之差先取线性 P95，再在三个窗口对中取最大 |
| L1 稳定性 | `l1_mean_shift` | 三个窗口 current L1 均值的最大差 |
| L1 稳定性 | `l1_p90_minus_p10_shift` | 三个窗口内 L1 的 P90-P10 宽度之最大差 |
| 结构稳定性 | `unique_row_rate_shift` | 三个窗口平均唯一行比例的最大差 |
| 结构稳定性 | `normalized_row_entropy_shift` | 三个窗口平均归一化行熵的最大差 |
| 运动护栏 | `minimum_observed_active_round_rate` | 三个窗口中“实际改行数大于 0”的轮次比例最小值 |
| 运动护栏 | `minimum_observed_mean_changed_row_fraction` | 三个窗口中平均改行比例的最小值 |

六项稳定性证据必须全部不超过上限，两项运动证据必须全部不低于下限。连续两个检查
同时满足“稳定且运动充分”才给出 `stationary_qualified`；稳定但运动不足连续达到
`stall_patience_checks` 才给出 `stalled`。达到预算仍未合格时返回 `horizon_reached`。

这只是可操作的候选停止定义，不是对马尔可夫链平稳分布或严格混合时间的数学证明。

## 3. Stage 2B：结果前冻结的实验协议

### 3.1 固定生成配置

四个实验 cell 为：

```text
2 datasets: test_300x10, nltcs
2 kernels:  independent, factorized_gibbs (8 sweeps, max order 3)
```

统一固定 exact target、no gate、marginal init、scale-invariant donor、`alpha=16`、
`rho=0.01`、`eta=0.5`、`mu=0.01`、`tau=2`，以及 independent/factor 两侧
`logit clip=30`。每个 dataset×seed 的两个核共享 `S0`、初始化后 RNG 和由 independent
参考预检得到的固定 `s0`。

这些参数只定义校准参考过程，不代表已经选出最优 donor、alpha 或 kernel。

### 3.2 Development 量程与阈值

- development seeds：`200..202`；
- 轨迹：2 datasets × 2 kernels × 3 seeds = 12 条；
- 每条完整运行 8000 轮，不在线早停；
- 窗口固定为 `W=400`；
- 阈值只使用完整落在 `6001..8000` 的检查终点 `7200/7600/8000`；
- 每个 cell 为 3 seeds × 3 检查 = 9 行，四格合计 36 行；
- 六项稳定性先取每格线性 P95，再取四格最大值作为公共上限；
- 两项运动性先取每格线性 P05，再取四格最小值作为公共下限；
- 不添加人工倍数，不舍入。

冻结的 V1 配置为：

| 参数 | 数值 |
|---|---:|
| `window_size` | 400 |
| `query_mean_shift_tolerance` | 0.0022331666666666017 |
| `query_p95_shift_tolerance` | 0.005488583333333529 |
| `l1_mean_shift_tolerance` | 0.0004866000000000001 |
| `l1_p90_minus_p10_shift_tolerance` | 0.00044000000000000034 |
| `unique_row_rate_tolerance` | 0.05588666666666668 |
| `normalized_row_entropy_tolerance` | 0.019247834404109442 |
| `minimum_active_round_rate` | 0.8625 |
| `minimum_mean_changed_row_fraction` | 0.005748717631790372 |
| `stall_patience_checks` | 4 |

候选停止后的完整 8000 轮仍继续审计。若出现连续 4 次稳定性失败，则记为 persistent
redrift（持续再漂移）；单次或短暂异常不直接否决。

### 3.3 封存 validation

- validation seeds：`220..224`，在阈值冻结前封存；
- 轨迹：2 datasets × 2 kernels × 5 seeds = 20 条；
- 每条跑满 8000 轮，共 160000 轮；
- 采集阶段不得执行 detector replay 或读取部分分类；
- 正式回放必须同时确认冻结协议 SHA 和 collection manifest SHA；
- 不允许阈值覆盖，不重跑生成器，不使用 query-max 补充规则；
- 硬门禁：20/20 合格、0 stalled、0 persistent redrift；任一失败即否决 V1。

## 4. 实验结果

### 4.1 Development

| 项目 | 结果 |
|---|---:|
| 合格轨迹 | 12/12 |
| stalled | 0 |
| persistent redrift | 0 |
| 候选停止轮次 | 11 条为 2000；1 条为 2400 |
| development 分类 | `candidate_supported_on_development` |

这一步只能说明 V1 拟合了 development 轨迹，不能作为最终证据。

### 4.2 正式 validation

| 项目 | 结果 |
|---|---:|
| 完整采集 | 20/20，每条 8000 轮 |
| 曾达到 `stationary_qualified` | 20/20 |
| stalled | 0 |
| 候选停止轮次 | 18 条为 2000；2 条为 2400 |
| persistent redrift | 1 |
| 正式分类 | `does_not_support_frozen_detector_on_validation` |

唯一失败轨迹为 `test_300x10 / seed 220 / factorized_gibbs`：

- 候选停止轮次：2000；
- 5600、6000、6400、6800：L1 `P90-P10` 宽度变化超过 `0.00044`；
- 7200、7600：L1 窗口均值变化超过 `0.0004866`；
- 连续不稳定 streak 为 6，超过冻结门槛 4；
- 两项运动护栏全程通过；
- 8000 轮恢复稳定，但冻结协议不允许用恢复撤销已经触发的门禁。

### 4.3 失败诊断

只读诊断没有发现公式实现、输入哈希、clip、配对或异构 GPU 问题。失败轨迹的 L1 为：

| 区间 | current normalized L1 均值 |
|---|---:|
| 候选证据区间 801..2000 | 0.00295911 |
| 候选后 2001..8000 | 0.00291260 |
| 最后 2000 轮 | 0.00292033 |
| 最后一个 400 轮块 | 0.00287450 |

候选后的块均值斜率近零且略向下，不支持“loss 持续向坏处漂移”。V1 更可能存在以下
协议问题：

1. 每个 cell 只有 9 行晚期证据估计单检查 P95，但 validation 对 298 个候选后检查、
   六项稳定性指标施加了长期四连败门禁；单检查校准与全过程误报风险没有对齐。
2. 相邻检查复用三分之二窗口，连续通过或连续失败不是独立证据。
3. 小表离散波动明显高于 nltcs；统一绝对阈值在不同 `N` 和查询数下不具备统一尺度语义。
4. development 与 validation 共 32 条轨迹全部在 2000 或 2400 轮合格，候选规则接近
   固定 burn-in，数据自适应性证据不足。
5. “候选后长期绝不出现一次局部波动”和“已经进入稳定分布”并不是同一个目标。

所以，正式结论不能外推为 factorized Gibbs 未收敛或效果更差。

## 5. Query-max 补充规则的边界

人工反例证明，V1 的 query mean + query P95 可能漏掉少数查询单独漂移。为此实现了
独立版本的 query-max 证据与 replay，development 自动导出的公共上限为
`0.007808333333333567`。它在 12/12 development 轨迹上合格，仅把一条候选停止从
2000 推迟到 2400。

但 query-max：

- 没有进入冻结的 V1 validation 协议；
- 没有使用新的 validation seeds 做正式验证；
- 不能直接处理本次导致失败的 L1 均值/宽度波动；
- 还没有解决查询数量变化时极值统计量尺度变化的问题。

因此它只能作为已实现的 development 候选保留，不能用于挽救 V1 的正式结论。

## 6. 可复现身份与产物

| 对象 | SHA-256 |
|---|---|
| 冻结 validation 协议 | `7c6d345dc559298dafd4a28eb5a2c1f08742133f660bbbef67b0347c726e8921` |
| validation collection manifest | `cdb58df5d6ebcc0ea0892ace2244889448cb62e3ba7a4174259fe4c3c5fd4e92` |
| validation replay manifest | `bb4de0d6cfee9257eb3f4c2045ed1011b55e36bbf5c2f72b712c5b952e96b324` |
| validation report | `dcccefff9ae2237f0be3298ef53e3d9df2dbb621537a5c278ca6f41c91c306b7` |
| trajectory results CSV | `ff945cf8b29ed86a316e643210934fbc37a66ac01b121813f4bad4f0eaefaaff` |
| full replay checks CSV | `3c3e21a9914e532693b18672667bb20aa63fffcf7fbafb19e87a234748bdf51e` |

主要入口：

- `src/table_diffevo/stationarity.py`：轨迹、无阈值证据与版本化 replay；
- `scripts/collect_issue53_stage2b_range_finding.py`：development 长轨迹采集；
- `scripts/analyze_issue53_stage2b_range_finding.py`：无阈值量程报告；
- `scripts/calibrate_issue53_stage2b_detector.py`：V1 development 校准；
- `scripts/issue53_stage2b_validation_protocol.py`：冻结协议；
- `scripts/collect_issue53_stage2b_validation.py`：封存 validation 采集；
- `scripts/replay_issue53_stage2b_validation.py`：唯一正式 V1 解封回放；
- `scripts/analyze_issue53_stage2b_query_max_range.py` 与
  `scripts/calibrate_issue53_stage2b_query_max_detector.py`：query-max development 补充。

正式长轨迹和报告位于 gitignored `outputs/`，不提交大型生成产物；代码、协议常量、
结果摘要和产物哈希进入版本控制。

## 7. 冻结后的处理

- 保留 V1 实现、协议、失败结果和诊断，作为可复现的负结果；
- 不把 V1 配置接入在线主循环；
- validation seeds `220..224` 对后续 detector 配置作废；
- 不根据这批 validation 结果调阈值；
- V2 另行设计、另行冻结，并使用全新的 validation seeds；
- V2 的具体证据公式、阈值和状态机不属于本总结，也不进入本 PR。
