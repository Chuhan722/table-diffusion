# Issue #53 B+C 问题一：R8 恢复版离线评价适配协议

## 1. 目的与唯一适配点

R8 四条轨迹已经生成完成并按采集后勘误协议无重跑收口。原离线 evaluator
的指标、分箱、查询组、比值、判定优先级和原子发布逻辑均未出错；唯一不兼容
点是它在评价前仍通过旧 collector 断言：

```text
state_evaluation_count == applied_rounds + 1
```

恢复 collection 中保留的生成器真实契约是：

```text
state_evaluation_count == max(1, applied_rounds)
```

本适配器只把原 evaluator 的 collection loader 替换为已冻结的结果盲恢复
loader。恢复 loader 完成字节清单、计数勘误、任务、配对、停止、权重、`8*K`、
附件哈希和监控限制复核后，适配器直接调用原 evaluator 的计算函数。

## 2. 冻结输入与实现继承

- collection 报告 SHA-256 固定为
  `35e0b5c594dd8b502198b58690ca720449d2283702817acae843b11968bd3eb8`。
- 采集后恢复协议 SHA-256 固定为
  `252f7bb9dd018a258e4eafe40a9b84a0afb7846e1bb3c1a30e22138cbb8290d0`。
- 原 evaluator 源码 SHA-256 固定为
  `287779a7e66466d3ed2e04862d2aded7ce3e8f084277de72893600e71c6f7445`。
- 科学协议、任务矩阵、数据输入、目标分箱、查询组、门槛与判定顺序继续由原
  R8 执行协议和科学协议冻结；本协议不复制或修改它们。

适配器必须直接复用原 evaluator 的 `_evaluate_cases`、
`_summary_and_decision`、`_csv_rows` 与 `_publish`。不得在适配器中另写指标
公式、比值规则、分类器、CSV 字段或发布逻辑。

## 3. 结果前信息边界

接线与测试阶段只允许：

- 核对 collection 文件 SHA-256；
- 运行恢复 loader 的结构与哈希检查；
- 使用人工桩验证函数委托、报告字段、确认门与原子发布入口；
- 运行不读取参考表、不计算真实质量指标的 plan。

禁止读取原始参考表、评价终表或检查点、计算 L1/分箱/查询阶数、形成两臂
比较、发布评价文件、修改参数或运行生成器。正确的 collection/适配协议哈希
本身不构成评价授权。

## 4. 真正评价与后续边界

真正评价必须在代码提交且工作树干净后，同时确认完整适配协议 SHA-256 与
上述 collection SHA-256，并得到用户在接线完成后的明确指示。评价器才可读取
参考表并原子写入既有文件名：

- `evaluation_report.json`
- `screen_metrics.csv`

评价完成后先报告评价报告 SHA-256 并停止。评价不自动授权独立审计、五种子
确认、R4/R16 调参、公共默认修改、推送或 PR 操作。原 runner GPU monitor
样本未落盘的限制必须继续传递到评价报告，不能改写为监控完整。
