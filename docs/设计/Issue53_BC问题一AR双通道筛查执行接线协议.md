# Issue #53 B+C 问题一：A/R 双通道筛查执行接线协议

## 1. 性质与授权边界

本文只把已冻结的 A/R 双通道科学协议接到已验证的 Stage 6D 原子 case/断点
续跑基础设施和 Stage 6E P=6 自动停止适配器。接线完成、协议哈希匹配或 preflight
通过都不授权真实生成。

本阶段允许 `plan` 和 `preflight`；禁止 `collect`、评价、独立审计、新种子、调参、
推送与 PR #69 操作。`collect` 必须同时收到用户后续明确授权和完整执行协议
SHA-256。

## 2. 任务与分片

唯一任务集按以下顺序执行：

```text
seed_9908__gap_dual_abs_relative_max_s8__test_300x10
seed_9908__gap_dual_abs_relative_max_s8__nltcs
```

两条轨迹固定分配到当前双 RTX 4090 主机的物理 GPU 1，最多两个 worker，
multiprocessing 使用 `spawn`。运行前要求 GPU 1 空闲；输出目录必须不存在，不允许
覆盖。

## 3. 候选专用诊断门

对每个实际执行 C 扫描的轮次，collector 必须验证：

- `kernel = gap_l1_global_random_scan_dual_abs_relative_max`;
- `gap_l1_weighting = dual_abs_relative_max`;
- `gap_l1_channel_aggregation = max`;
- `gap_l1_zero_target_policy = absolute_channel_only`;
- `gap_l1_floor_applied = false`;
- `gap_l1_max_weight_ratio = null` 且 smoothing 为 null；
- 正目标查询数、`sum(1/target)` 和 R 内正权重跨度与独立目标审计精确一致；
- 扫描微步数为 `8*K`，无非有限值、精确 0/1 概率或 logit 裁剪。

若已应用至少一轮却从未实际执行 A/R 扫描，collection 失败关闭。

## 4. 信息流与输出

collector 只读 schema、measured queries/targets 和边际初始化文件，不读参考原表、
legacy/R8/平方根终表或任何评价产物。单 case 使用既有临时目录后原子发布；
已完整且哈希通过的 case 可断点续用，不重跑。两条都完整后才合并唯一
`collection_report.json`。

collection 不计算或发布 measured L1、分箱质量、one-way safety、方法排名或
通过分类。收口后先报告 collection SHA-256 并停止。

## 5. 只读命令与真实执行命令边界

当前只允许：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
  /home/chuhan/projects/table-diffusion/.conda/bin/python3.11 \
  -m scripts.run_issue53_gap_weight_dual_ar_screen plan

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
  /home/chuhan/projects/table-diffusion/.conda/bin/python3.11 \
  -m scripts.run_issue53_gap_weight_dual_ar_screen preflight
```

真实执行命令只做冻结记录，本阶段不得运行：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
  /home/chuhan/projects/table-diffusion/.conda/bin/python3.11 \
  -m scripts.run_issue53_gap_weight_dual_ar_screen collect \
  --confirm-protocol-sha <FROZEN_EXECUTION_PROTOCOL_SHA256>
```
