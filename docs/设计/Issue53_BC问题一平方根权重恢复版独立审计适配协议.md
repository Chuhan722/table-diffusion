# Issue #53 B+C 问题一：平方根权重恢复版独立审计适配协议

## 1. 目的

平方根候选的主离线评价已经发布。独立审计不是再跑一次生成实验，而是从同一
候选终表、检查点、冻结查询和参考表出发，使用不导入主 evaluator 算术的原
独立 auditor 重新计算候选的全部指标，再独立重建候选相对 legacy/R8 的汇总、
六道门和分类，最后逐项核对评价报告与 CSV。

本协议不把已观察到的 L1、比值或分类值写成预期答案。审计能通过的唯一方式
是独立复算与已发布产物一致。

## 2. 冻结输入

- 平方根候选 collection SHA-256：
  `50f0c3561fa9614abc85eacb014566e3354e987a647784f27453419ea3e882f4`
- 平方根候选 evaluation SHA-256：
  `e6d8110f521e366130949b4b4d3ac460b34eb7109d3e4a8f5518d96e92df3fad`
- 平方根候选 `screen_metrics.csv` SHA-256：
  `b5dd01a653544bb7a3a98317ba0e708d2e89a10b70679c57c6046cf339b072e6`
- 平方根恢复版评价适配协议 SHA-256：
  `1c0ec85b711356a98cd8540af8c8b5deedd7820b5a8f76f0e5a6f55f5101489d`
- 原独立 auditor 源码 SHA-256：
  `7a7f991fd9ef066771c58c36f1b22afa13f78ab27dd021cb8c13bb2ddbd4ad03`

legacy/R8 基线仍绑定已经独立审计通过的四项冻结产物：collection、evaluation、
metrics CSV 和 independent audit。审计不得把主平方根 evaluator 当作独立
算术来源。

## 3. 独立复算路径

### 3.1 候选 case

恢复 loader 先验证两条候选 collection 及勘误契约。随后只在调用域内把原
R8 独立 auditor 的协议读取面绑定到平方根协议，并直接使用它的独立实现：

- `_recompute_cases`：重读终表、检查点、查询与参考表；
- `_nested`、`_independent_ratio`、`_ratio_value`：读取指标并处理零分母；
- `_csv_rows_independent`、`_audit_csv`：独立生成并逐行核对候选 CSV；
- `_assert_equal`：严格比较结构及近机器精度浮点。

不得导入 `evaluate_issue53_gap_weight_sqrt_screen_recovered.py`，也不得导入原
R8 主 evaluator。终表、检查点、目标频率箱、查询阶数、未测查询组和两个代理
指标的算术均不得修改。

### 3.2 基线与新汇总

legacy/R8 四条基线已由既有 independent audit 从原始终表独立复算通过，因此
本次只按冻结哈希加载它们，不重跑生成，也不重复消耗计算或隐私预算。

平方根评价新增了“一条候选同时对比两个基线”的展示结构，原 R8 独立 auditor
没有这一顶层结构。因此本适配器必须独立完成以下接线：

1. 对每项候选指标分别生成 `vs_legacy` 与 `vs_r8`；
2. 独立重建两个数据集的分箱、查询阶数、未测查询组、检查点曲线与成本汇总；
3. 按结果前冻结的阈值逐条重算六道门；
4. 在适配器内按冻结优先级显式展开分类，不调用科学协议的分类函数，也不调用
   主 evaluator 的 `_summary_and_decision`。

## 4. 审计通过条件

只有下列项目全部一致才允许发布 `pass=true`：

- 两个候选终表及全部实际到达检查点的指标；
- 候选的两个代理、目标频率箱、查询阶数和全部未测查询组；
- 候选相对 legacy/R8 的完整 summary；
- 六道门、execution validity、分类及分类优先级；
- 候选 CSV 的列、行顺序、行值、行数与 SHA-256；
- collection、evaluation、基线、提交、源码、查询和参考身份；
- 原 runner 监控样本未落盘这一限制继续披露。

任一项不一致都必须失败，且不得写出通过报告。

## 5. 执行与后续边界

适配器必须先完成源码冻结、相关测试、干净提交和只读预检。正式审计还要再次
同时确认审计适配协议、collection 和 evaluation 的完整 SHA-256；匹配哈希本身
不代表已经获得执行授权。

正式审计会读取参考表并新建候选目录下的 `independent_audit.json`，但不会重跑
生成、改写 evaluation/CSV、调参、扩大种子或消耗新的隐私预算。审计结束后只
报告能否逐项复现及审计 SHA，不自动进入新核实验、修改默认、推送或 PR 操作。
