# Issue #53 B+C 问题一：R8 恢复版独立审计适配协议

## 1. 目的

主 evaluator 已对恢复后的四轨迹 collection 发布评价。独立审计的目标不是
再做一次实验，而是从相同终表、检查点、查询和参考表出发，使用原独立 auditor
中不导入 R8 evaluator 算术的实现重新计算全部指标，再与评价报告及 CSV 逐项
比较。

本适配协议不冻结任何预期 L1、比值或分类结果。它只冻结待审计产物的字节
身份、原独立 auditor 的源码身份以及恢复契约的唯一适配点。

## 2. 冻结输入

- collection SHA-256：
  `35e0b5c594dd8b502198b58690ca720449d2283702817acae843b11968bd3eb8`
- evaluation SHA-256：
  `548b611b3762cd5489fcbd97d787ea9e91f987c5a94bed0b73e6782f6108bc4c`
- `screen_metrics.csv` SHA-256：
  `b6d1efff24202a74003cee01eb05568db64b071aab3a47a86e3cafa4bc27b660`
- 恢复版评价适配协议 SHA-256：
  `35d93e9e77c5d3e28085bc20d76852bec63eb128b21f8c64d316675e609e4744`
- 原独立 auditor 源码 SHA-256：
  `7a7f991fd9ef066771c58c36f1b22afa13f78ab27dd021cb8c13bb2ddbd4ad03`

## 3. 唯一契约适配

原独立 auditor 的 collection 入口仍包含旧
`state_evaluation_count == applied_rounds + 1` 校验，且只接受原评价报告顶层
契约。恢复版入口只做两项外壳适配：

1. collection 结构校验交给冻结恢复 loader，接受生成器真实的
   `state_evaluation_count == max(1, applied_rounds)`；
2. 评价报告顶层身份校验改为恢复版 evaluator 的冻结字段。

随后必须直接调用原独立 auditor 的以下实现：

- `_recompute_cases`
- `_summary_independent`
- `_csv_rows_independent`
- `_assert_equal`
- `_audit_csv`

适配器不得导入主 R8 evaluator，不得复制或修改指标公式、查询分箱、未测查询
组、比值规则、分类优先级、CSV 行生成或浮点比较容差。

## 4. 审计通过条件

只有下列项目全部一致才可发布 `pass=true`：

- 四个终表及所有实际到达检查点的指标；
- 两种 gap 代理、目标频率箱、查询阶数和全部未测查询组；
- 两个数据集的完整 summary；
- execution validity 与冻结分类；
- CSV 列、行顺序、行值、行数及 SHA-256；
- collection/evaluation/提交/源码/输入身份；
- 原 runner 监控证据不完整的披露。

任一项不一致必须失败且不得写通过报告。

## 5. 执行与后续边界

用户在理解独立审计用途后已明确指示继续。适配器仍须先完成源码冻结、测试与
干净提交，再同时确认完整审计协议、collection 和 evaluation SHA 才可运行。

审计会读取参考表并产生 `independent_audit.json`，但不会重跑生成、修改评价
产物、调参或消耗新的隐私预算。审计完成后只报告是否逐项一致及审计 SHA，
不得自动进入五种子确认、新核实验、推送或 PR 操作。
