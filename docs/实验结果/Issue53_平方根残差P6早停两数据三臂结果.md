# Issue #53 平方根残差 P=6 早停两数据三臂结果

日期：2026-08-17

## 1. 结论

在同一套 P=6 natural-work 早停、terminal-current 输出和其余固定生成参数下，三种残差几何没有一个
跨数据统一最优：

- `test_300x10`：`absolute` 的 terminal measured normalized L1 在 3/3 个配对 seed 中最好，且平均
  work 也最低；它同时支配 `sqrt_relative` 和 `relative`。
- `nltcs`：`relative` 的 L1 在 3/3 个配对 seed 中最好；`sqrt_relative` 的精度居中，但平均 work
  明显更低。该数据上的质量—计算 Pareto 前沿是 `sqrt_relative` 与 `relative`，`absolute` 被
  `sqrt_relative` 同时以质量和 work 支配。

因此，平方根残差是 `nltcs` 上有价值的中间折中，但不是可直接设为全局默认的统一答案。固定归一化
指数从 0（absolute）改到 0.5（sqrt）再到 1（relative），只是在两个数据的相反偏好之间移动，没有
消除跨数据反转。

## 2. 冻结身份与运行边界

```text
execution commit  fe8fb797a718bf0e9a89668d46fbd5726c1c3082
protocol SHA      7e7b5e08f9d934031257cbd98b6a857f7ba1dcb4cf1f97077d48f781a4e2585f
report SHA        241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143
datasets          test_300x10, nltcs
arms              absolute, sqrt_relative, relative
seeds             310, 311, 312
cases             18
server            root@10.8.176.53:6006, RTX A6000 GPU 0
execution         one visible GPU, one worker, all seed shards serial
```

除 seed 和 `residual_geometry` 外，矩阵固定复用 PR #63 的两数据早停配置：P=6、C=6000、rho=0.01、
scale-invariant donor、fixed alpha=16、direction `initial_rms`、eta=0.5、mu=0.01、Gibbs sweeps=0、
`tol=inf`、无重试、terminal-current 输出。平方根残差固定为：

```text
sign(raw) * magnitude / sqrt(max(target, 8)) / n_records
```

运行后没有按结果改参数。18/18 cases 均由 `early_stopped` 结束，无 case 触及资源上限。在线停止不读取
L1；本次没有访问 raw reference table、没有消耗隐私预算，也不作 canonical、held-out、收敛或真实数据
正式验证声明。

## 3. 三 seed 聚合结果

| 数据 | 残差几何 | mean terminal L1 | median terminal L1 | 配对 seed 胜数 | mean work | mean rounds |
|---|---|---:|---:|---:|---:|---:|
| `test_300x10` | `absolute` | 0.0026000 | 0.0024667 | 3/3 | 13.3356 | 1319.7 |
| `test_300x10` | `sqrt_relative` | 0.0029778 | 0.0028000 | 0/3 | 14.0044 | 1396.0 |
| `test_300x10` | `relative` | 0.0037556 | 0.0038000 | 0/3 | 15.6689 | 1550.0 |
| `nltcs` | `absolute` | 0.0011321723 | 0.0011256897 | 0/3 | 17.6697 | 1768.7 |
| `nltcs` | `sqrt_relative` | 0.0004930488 | 0.0005186088 | 0/3 | 15.3349 | 1534.3 |
| `nltcs` | `relative` | 0.0003474679 | 0.0002898035 | 3/3 | 23.0065 | 2301.3 |

### `test_300x10`

相对 `absolute`，`sqrt_relative` 的平均 L1 高 14.53%，平均 work 高 5.02%；`relative` 的平均 L1
高 44.44%，平均 work 高 17.50%。所以当前协议下 `absolute` 同时更准、更省，不存在质量—计算取舍。

### `nltcs`

相对 `absolute`，`sqrt_relative` 的平均 L1 低 56.45%，平均 work 低 13.21%，因此
`sqrt_relative` 严格支配 `absolute`。相对 `sqrt_relative`，`relative` 的平均 L1 再低 29.53%，
但平均 work 高 50.03%；两者构成真实的精度—计算取舍。

## 4. 各 seed 原始摘要

| 数据 | seed | `absolute` L1 / work | `sqrt_relative` L1 / work | `relative` L1 / work |
|---|---:|---:|---:|---:|
| `test_300x10` | 310 | 0.0029333 / 11.0000 | 0.0034667 / 15.0000 | 0.0036000 / 17.0000 |
| `test_300x10` | 311 | 0.0024000 / 17.0067 | 0.0026667 / 12.0033 | 0.0038000 / 17.0067 |
| `test_300x10` | 312 | 0.0024667 / 12.0000 | 0.0028000 / 15.0100 | 0.0038667 / 13.0000 |
| `nltcs` | 310 | 0.0011194541 / 14.0036 | 0.0005186088 / 17.0022 | 0.0004665627 / 20.0075 |
| `nltcs` | 311 | 0.0011513732 / 18.0018 | 0.0004171714 / 14.0012 | 0.0002860374 / 26.0072 |
| `nltcs` | 312 | 0.0011256897 / 21.0036 | 0.0005433662 / 15.0013 | 0.0002898035 / 23.0048 |

完整 raw case、终态表和聚合报告位于被 gitignore 的：

```text
outputs/issue53_sqrt_residual_earlystop_comparison_v1/
```

本地与远端共 40 个文件已逐文件比较 SHA-256，内容完全一致。远端运行树在结束后保持 clean，GPU 0
已回到 `3 MiB / 0%`；GPU 3 的既有无关任务未触碰。

## 5. 后续查询级诊断（已完成）

已使用这 18 个 terminal artifacts 完成固定的 result-aware 查询级误差分解，完整结果见：

```text
docs/实验结果/Issue53_残差几何查询级诊断结果.md
report SHA 876b7cc2f75ddf315800dd36853ca617fbbbbbf6258bc908709bec49c251e48b
```

诊断否定了简单频率分流：nltcs 的 relative 在所有频率档都改善。更强的解释是 test 有 25 条由
marginal 初始化精确满足的 1-way 查询，而 nltcs measured workload 没有 1-way。下一候选改为结果前
设计 order-aware geometry：1-way absolute、order>=2 relative；必须先解决两块尺度对齐，再用 fresh
seeds 验证，不能把本诊断当成选择证据。
