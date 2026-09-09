# Issue #53 B+C 问题一：A/R 双通道筛查采集后勘误恢复协议

日期：2026-09-02

## 1. 事故边界

用户确认执行冻结协议
`4ccf7bbe953d2523fca76d6a7e0ef1410e8773ebdb9a81fd50f8ed8f230b9386`
后，提交 `a3ba71fa2e84512fc6c8ba1bc818ba03e908cc71` 在物理 GPU 1 上完成
两条候选轨迹。采集器在两个 case 文件均写入 staging 后、发布 shard 报告前失败：

```text
RuntimeError: Stage 6E case 行身份漂移：
seed_9908__gap_dual_abs_relative_max_s8__test_300x10
```

源 staging 为
`.local_rtx4090.staging-m3bnqwn9`。正式 shard 与 collection 均未发布。

## 2. 唯一勘误

失败来自沿用的 Stage 6E 发布校验器要求：

```text
state_evaluation_count == applied_rounds + 1
```

当前生成器的实际诊断契约，以及此前 R8、平方根筛查已经分别冻结并恢复过的
契约是：

```text
state_evaluation_count == max(1, applied_rounds)
```

本次两个 case 分别为 `1505 == max(1,1505)` 与
`2805 == max(1,2805)`。把只供旧校验器使用的内存代理行改为
`applied_rounds+1` 后，两条行的所有旧 Stage 6E 护栏和 A/R 专属护栏均通过。

恢复过程只允许这一项代理修正；冻结 case manifest 中的原始行不得改写，终态表、
检查点查询答案与逐轮转移审计不得改写。

## 3. 冻结证据

恢复协议逐字节冻结以下内容：

- 源执行协议、源 collector、源生成提交与源 staging manifest；
- 两个 case 各四个文件的文件名、大小和 SHA-256；
- A/R 权重模式、`max` 聚合、零目标策略、无 floor、正目标数量、逆目标归一化常数、
  正目标权重比、逐轮 `8*K`、随机流、终态身份和停止身份；
- 监督进程在源运行期间观察到的物理 GPU 1 样本及其证据限制。

源 collector 在 shard 报告写出前失败，因此其内部 GPU 样本没有持久化。恢复报告
必须明确标记 `execution_monitoring_evidence_complete=false`，监督样本不得冒充源
runner 样本。这一限制不改变两个 case 的逐字节身份。

## 4. 信息与执行边界

恢复适配器只允许：

1. 结果盲验证 staging、八个 case 文件及全部结构/数值护栏；
2. 生成恢复 shard 报告并原子关闭源 staging；
3. 复制已验证 case，生成正式 collection 与恢复证据；
4. 报告 collection SHA-256。

恢复适配器禁止调用生成器、删除或重跑 case、访问 raw reference、解释候选质量、
比较方法、调参或覆盖已有正式输出。质量评价只能在完整恢复 collection 固定后，
按单独冻结的评价适配协议进行。

