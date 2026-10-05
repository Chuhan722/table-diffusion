# Issue #53 Stage 6E 生成后收口勘误恢复协议

## 1. 事实与勘误范围

Stage 6E 的 30 条冻结轨迹已在生成提交
`04904b2cdf1ae2125a5ff7d347f4888506ed7cac` 上全部生成完成。旧收集器随后在写出
`shard_report.json` 之前，错误地断言
`state_evaluation_count == applied_rounds + 1`，因此没有发布 collection。

生成器的真实计数语义是：逻辑状态包含初态和每轮后状态，共
`applied_rounds + 1` 个；但终点状态已经作为最后一次候选评价被计数，实际状态评价调用次数为
`max(1, applied_rounds)`。30 条暂存结果全部符合后一语义，且除该旧断言外的身份、停止、无门控、数值和附件哈希护栏均通过。

本协议只勘误收口校验，不修改原 Stage 6E 协议、生成器、收集器或任何 case 文件。

## 2. 冻结输入

- 唯一输入是 `configs/issue53_stage6e_recovery_inventory.json` 中冻结的一个暂存目录、一个暂存清单哈希和 30 个 case 清单哈希。
- 每个 case 必须恰好包含 `case_manifest.json`、`terminal_current.csv`、`checkpoint_query_answers.json` 和 `transition_audit.json`。
- 必须逐一复核 90 个附件哈希、30 个生成参数清单、任务顺序、三核配对身份、停止元数据和转移时钟。
- 收口阶段不得读取原始 reference，不得解释检查点质量字段，不得调用 GPU 或生成器，不得重写 case 文件。

## 3. 唯一允许的校验修正

旧断言：`state_evaluation_count == applied_rounds + 1`。

勘误断言：`state_evaluation_count == max(1, applied_rounds)`。

同时保留并报告 `logical_state_count = applied_rounds + 1`。除这一处外，不允许放宽任何身份、停止、数值、无门控或完整性条件。

## 4. 证据限制

原进程在写分片报告前退出，父进程内存中的 GPU 监控样本没有持久化。恢复报告必须将其标为不可重建的执行监控证据缺失；冻结硬件/软件配置只能标记为预期环境，不能冒充运行时观测。该限制不改变已冻结 case 字节，也不影响离线质量指标复算。

## 5. 执行与发布顺序

1. 恢复代码、测试、文档和输入清单全部提交，且工作树干净。
2. 用户已明确允许继续后，恢复器仍要求传入完整恢复协议 SHA-256。
3. 恢复器只读复核暂存 case，原子收口单分片，再生成明确标注为 recovered/erratum 的完整 collection。
4. 只有确认完整 collection 报告 SHA-256 后，评价器才可读取 reference 并发布 30 条指标与完整 L1 CSV。
5. 独立审计器重新读取终表、检查点向量和 reference，独立复算全部指标、汇总、分类与 CSV；审计通过前不得把结果写入汇报材料。

## 6. 结果解释边界

恢复不新增实验样本、不补跑或筛除轨迹、不改阈值、不选择历史最佳检查点，也不消耗新的隐私预算。运行时间使用 case 内原始 `elapsed_sec`；由于两个进程共享同一张 RTX 4090，并行采集下的每轮时间只作为本次正式运行诊断，不与单进程基准作严格等价比较。
