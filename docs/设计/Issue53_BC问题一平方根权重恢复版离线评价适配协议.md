# Issue #53 B+C 问题一：平方根权重恢复版离线评价适配协议

## 1. 目的

平方根权重的两条开发轨迹已完整生成，并按采集后勘误协议无重跑
收口。本适配器只回答结果前协议已经写定的问题：平方根候选是否同时
保留 NLTCS 常见查询收益、恢复稀有查询保护，并通过 test 与 one-way
安全门。

这仍是种子 `9908` 的开发筛查。即使全部通过，也只能进入新种子确认，
不能声称全面优于旧 B+C，不能修改公共默认。

## 2. 冻结输入

候选 collection：

```text
path    outputs/issue53_gap_weight_sqrt_target_screen_seed9908_v1/collection_report.json
SHA-256 50f0c3561fa9614abc85eacb014566e3354e987a647784f27453419ea3e882f4
```

其生成后恢复协议 SHA-256 为
`0f6420a0501fbedf792720716b54735d141aafef90261e37560a6a312a4642ac`。
恢复 loader 必须先完成冻结字节、状态评价次数勘误、任务、停止、
平方根权重、`8*K`、附件哈希与监控限制复核。

不重跑 legacy/R8，只复用已独立审计通过的四个冻结基线产物：

```text
collection        35e0b5c594dd8b502198b58690ca720449d2283702817acae843b11968bd3eb8
evaluation        548b611b3762cd5489fcbd97d787ea9e91f987c5a94bed0b73e6782f6108bc4c
metrics CSV       b6d1efff24202a74003cee01eb05568db64b071aab3a47a86e3cafa4bc27b660
independent audit 9e9e29c39cadc756f29dd67cc1bb17eeccccace32a167225383fca81e5ce33e6
```

任一哈希、任务身份、查询/参考身份或审计 pass 身份漂移都立即失败。

## 3. 指标算术与仅有的适配

候选两条的终表、固定检查点、measured normalized L1、目标频率箱、
查询阶数、未测查询组、合法行和参考支持全部直接委托给已冻结的 R8
evaluator `_evaluate_cases`。适配器不重写终表或检查点指标公式。

候选/基线比值的零分母规则直接委托给同一 R8 evaluator 的
`_ratio_record`：基线为 0 时，两者都为 0 记为 1；只有候选非零则记为正无穷。

分类只调用平方根结果前科学协议已冻结的 `classify_screen`。不允许
修改指标、目标箱、查询组、比值规则、门槛、判定顺序、终表或检查点选择。

## 4. 六道结果门与判定顺序

执行身份先于质量。若候选或基线的完整性、有效行率、权重护栏失败，
统一为 `execution_invalid`，不解释质量。执行有效后按以下先后顺序：

1. NLTCS measured normalized L1：平方根 / legacy `< 1`。
2. NLTCS `target_2001_16181` 常见箱：平方根 / legacy `< 1`。
3. NLTCS `target_0_50` 稀有箱：平方根 / legacy `<= 1.25`。
4. 同一稀有箱：平方根 / R8 `< 1`。
5. `test_300x10` measured normalized L1：平方根 / legacy `<= 1.05`。
6. `test_300x10` 与 NLTCS 的 `one_way_safety`：各自平方根 / legacy
   `<= 1.05`。

第 1 或 2 道失败为 `common_mechanism_not_retained`；第 3 或 4 道失败为
`rare_query_protection_not_recovered`；第 5 或 6 道失败为
`measured_or_one_way_safety_risk`；全部通过才是
`advance_to_fresh_seed_confirmation`。

其余目标箱、查询阶数、未测查询组与所有已到达的固定检查点必须完整
发布，但只作诊断，不得事后增加门槛或改用最好检查点。

## 5. 输出与信息边界

真正评价后在候选 collection 目录中原子新建：

- `evaluation_report.json`；
- `screen_metrics.csv`（只含两条候选的检查点指标，旧基线 CSV 按哈希复用）。

不覆盖任何已有文件。评价报告必须继承原 runner GPU monitor 未落盘的
限制，不得将执行监控表述为完整。

在适配器冻结且用户再次明确指示之前，只允许核对文件 SHA-256、源码
委托、确认门、输出不存在性和不读取质量的 plan/preflight。禁止读取
参考表、评价候选终表/检查点、形成新方法比较或发布评价文件。

真正评价需要同时确认完整适配协议 SHA-256 与候选 collection SHA-256，
并要求已提交的干净工作树。评价完成后先报告 evaluation SHA-256 并停止。
评价不自动授权独立审计、新种子确认、调参、生成器重跑、公共默认修改、
推送或 PR 操作。
