# Issue #53 B+C 问题一：平方根查询权重筛查执行接线协议

## 1. 作用

本文件只把已经冻结的平方根查询权重科学协议接到 Stage 6D 的原子 case 产物/恢复收口设施和 Stage 6E 的 P=6 自动停止设施。执行层不得改变候选公式、两个任务、开发种子、生成参数、停止规则、目标权重审计或质量通过门。

本接线完成后只允许运行只读 `plan` 与 `preflight`。真实 `collect` 仍需用户在看到完整执行协议 SHA-256、提交身份、环境检查和预计耗时后再次明确授权。

## 2. 冻结任务

执行顺序固定为：

```text
seed_9908__gap_sqrt_target_s8__test_300x10
seed_9908__gap_sqrt_target_s8__nltcs
```

两条任务位于同一 RTX 4090 分片，最多两个 spawn worker。只生成候选，不重跑 legacy/R8，也不在生成阶段读取既有基线、原始参考表或未测查询。

## 3. 候选权重诊断硬检查

每个实际执行 C 核扫描的轮次必须同时满足：

```text
kernel                         gap_l1_global_random_scan_sqrt_target_relative
gap_l1_weighting               sqrt_target_relative
gap_l1_max_weight_ratio        null
gap_l1_smoothing_count         null
gap_l1_target_count_quantum    1.0
gap_l1_actual_weight_ratio     与结果前目标权重审计逐位相同
gibbs_microsteps               8 * active_switches_k
conditional_error_evaluations  2 * gibbs_microsteps
clip_hit_count                 0
nonfinite_condition_count      0
exact_zero_or_one_probability_count 0
```

`test_300x10` 的固定实际跨度为 `9.38083151964686`，NLTCS 为 `33.03028912982749`。没有实际扫描、模式/跨度/计数单位漂移或任何数值护栏失败都使 case 无效。

## 4. 复用的执行设施

- Stage 6D：输入哈希、查询/目标身份、边际初始化、GPU 身份、spawn 执行、case 原子发布、断点续跑、文件字节数和 SHA-256 收口。
- Stage 6E：P=6 自然工作量自动停止、A/B/C 终止优先级、terminal current、检查点答案、转移审计与 GPU 监控。
- R8 collector 只作为代码结构参考，不复用其 legacy/R8 双臂判断、四 case 报告或质量评价。

执行 adapter 必须把基础设施临时绑定到本协议，并在调用结束后恢复原模块对象，避免污染其他实验入口。

## 5. collection 边界

新输出固定使用全新目录：

```text
outputs/issue53_gap_weight_sqrt_target_screen_seed9908_v1
outputs/issue53_gap_weight_sqrt_target_screen_seed9908_v1_shards
```

已有目标或临时目录一律拒绝覆盖。恢复只允许按已验证 case manifest 的文件哈希跳过完整 case，不允许仅凭文件名续跑。

collection 必须恰好包含两条候选 case，顺序与冻结任务一致。它只报告执行身份、停止、资源、C 核权重和产物哈希，不报告候选 L1、分箱、查询阶数、未测查询或与 legacy/R8 的排序。

完整 collection 固定并报告 SHA-256 后立即停止。既有基线直到后续单独授权的离线评价才可读取和合并。

## 6. 环境与有效性

沿用 R8 筛查的冻结软件与本机物理 GPU 1：

```text
CUDA_VISIBLE_DEVICES=1
物理 UUID  GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02
型号       NVIDIA GeForce RTX 4090
显存       24564 MiB
```

preflight 必须确认源码/协议/目标审计/输入/参数哈希、干净提交、输出目录不存在、唯一可见且空闲的指定 GPU、确定性算法和两任务完整性。任何一项失败都不得启动生成。

## 7. 明确禁止

本接线阶段不得：

- 自动运行 `collect`；
- 为缩短耗时调整轮数、P、worker、权重或数据集顺序；
- 读取或发布候选质量；
- 自动评价、独立审计、新种子确认或参数搜索；
- 覆盖 R8 既有产物；
- 修改公共默认、推送或操作 PR #69。
