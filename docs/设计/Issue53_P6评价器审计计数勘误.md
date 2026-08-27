# Issue #53：P=6 评价器审计计数勘误

> 状态：erratum 1，2026-08-17；在原始 collection 完成、首次 evaluator fail closed 后登记。
>
> 本勘误只修复 artifact 审计中的 `state_evaluation_count` 语义和因此产生的 evaluator/collection
> commit 绑定问题。不修改 P、family、seed、rho、C、generator、停止规则、shadow 检查点、证据门禁、
> 质量/计算阈值、聚合公式、classification 或回退规则，也不重新采集。

## 1. 绑定的唯一原始 collection

```text
protocol SHA-256:
759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317

collection Git commit:
34b477acff11adabfc22b6eb9c14e4fb3939b7a1

collection manifest SHA-256:
aa4b34f80cbe72546c6a085845d205e988e04ccdeb0ee843ec135fbfa3505133
```

erratum evaluator 必须同时匹配上述两项 collection 身份。它不得用于另一批重跑结果。

## 2. 首次评价如何失败

冻结 evaluator 在逐 case online diagnostics 审计阶段抛出：

```text
RuntimeError: online current metrics terminal 身份不一致
```

失败发生在 B shadow checkpoint、证据聚合、质量/计算门禁和 classification 之前，没有生成
`p6_evaluation_report.json`。

原断言把两个不同计数混为一谈：

```text
错误：state_evaluation_count == rounds_run + 1
```

`current_state_metrics_history` 确实包含 S0 和每轮后的 terminal-current 状态，因此长度为
`rounds_run+1`。但 `state_evaluation_count` 只统计完整 current state evaluation。S0 计一次；每轮
proposal 的 query/loss 已由 candidate evaluation 得到，已接受 proposal 只会让 cache 失效，并在下一轮
开始时才增加新的完整 state evaluation。因此冻结无门控路径的真实关系是：

```text
rounds_run == 0 -> state_evaluation_count == 1
rounds_run > 0  -> state_evaluation_count == rounds_run

即：state_evaluation_count == max(1, rounds_run)
```

该修正不使用任何 loss/L1/shadow 数值，也不改变结果判定。

## 3. 已暴露范围

为定位错误，只读诊断解析了 12 个 case manifest 的 online 部分及 online diagnostics，并在控制台打印：

- 全部 12 条 termination reason，均为 `early_stopped`；
- stop state；
- terminal loss/L1 及其与末状态 diagnostics 的相等性。

没有访问或输出 shadow checkpoint 字段，没有计算 `delta_L1`、saving、coverage、family median、质量/
计算门禁或 classification。后续报告必须保留该暴露声明，不能再把整个评价过程描述为完全未见。

## 4. 允许的修订范围

只允许：

1. 把上述计数断言改为 `max(1, rounds_run)`；
2. 把假 B artifact 的计数改成真实语义并增加回归；
3. 让 evaluator commit 可以不同于 collection commit，但仅限下述逐文件身份检查全部通过之后；
4. 在正式报告中同时记录 collection/evaluator commit、本勘误、暴露范围和未修改项。

原 collection execution manifest 记录的所有 collector 路径必须与 erratum commit 当前文件 SHA-256
逐项完全一致：

```text
scripts/issue53_p6_unseen_protocol.py
scripts/collect_issue53_p6_unseen.py
src/table_diffevo/evolution.py
src/table_diffevo/inner_early_stopping.py
src/table_diffevo/stationarity.py
docs/设计/Issue53_P6未见轨迹质量计算验收协议.md
```

任一文件漂移都必须 fail closed。只有 evaluator、测试、`PROJECT_STATUS.md` 和本勘误文档可以形成新的
审计修订 commit。

## 5. 明确禁止

- 不覆盖、手改或重新生成原始 artifacts；
- 不重跑 20260819—20260821；
- 不改 protocol SHA 或验收阈值；
- 不因已知“12 条均为 B”改变 coverage、质量、计算或 family conflict 规则；
- 不在修复提交时顺带运行正式 evaluator；
- 不在未来评价完成后自动启动 P=12/P=4 回退。

erratum 修复、测试与提交完成后必须再次停止。只有用户另行授权，才能用修订 evaluator 对上述唯一原始
collection 进行只读评价。
