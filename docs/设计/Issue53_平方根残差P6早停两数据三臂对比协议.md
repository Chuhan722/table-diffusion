# Issue #53 平方根残差 P=6 早停两数据三臂对比协议

> 状态：结果前 development 协议。本文写成时尚未实现正式 runner、运行新 seed 或查看本矩阵结果。
> 既有 PR #59 absolute/relative 固定 2000 轮结果和 PR #63 relative seed 200 smoke 已经看过，
> 因此本轮只能回答新的 P=6 配对三臂问题，不包装成完全未见或 canonical 正式验证。

## 1. 问题

在 PR #63 的同一 P=6 自然工作早停下，比较以下三种残差几何在 `test_300x10` 和 `nltcs` 的
terminal-current measured quality 与工作量：

```text
absolute
sqrt_relative
relative
```

本轮只作逐数据集 development 描述：哪个 arm 的三 seed 平均 terminal normalized L1 最低，以及
该质量对应多少 rounds/normalized work。不同数据的赢家可以不同；本轮不据此选择 canonical 默认值。

## 2. 新候选的唯一公式

容忍后的计数残差幅度记为 `magnitude`，目标计数为 `y`，表记录数为 `N`，固定 `floor=8`：

```text
absolute:
  epsilon = sign(raw) * magnitude / N

sqrt_relative:
  epsilon = sign(raw) * magnitude / sqrt(max(y, 8)) / N

relative:
  epsilon = sign(raw) * magnitude / max(y, 8) / N
```

`sqrt_relative` 是固定的中间强度，不暴露指数、不扫描 gamma、不根据数据集切换公式。噪声容忍仍在
标准化之前；当前 sigma=0。三种几何的残差零点相同。

## 3. 冻结矩阵

```text
datasets = [test_300x10, nltcs]
arms     = [absolute, sqrt_relative, relative]
seeds    = [310, 311, 312]
cases    = 2 * 3 * 3 = 18
```

三个 seed 在当前仓库、Issue #53 和本地工作笔记中没有生成实验身份记录。每个 dataset/seed 的三臂
使用相同初始化 seed，形成配对比较。seed、arm、dataset 不允许通过正式命令覆盖。

## 4. 共享生成配置

完全复用 PR #63 两数据 P=6 smoke 配置，仅 seed 和 residual arm 按冻结矩阵变化：

```text
rho = 0.01
P = 6 natural-work ticks
C = 6000 rounds / 6000 candidates
tol = +inf
max_retries = 0
residual floor = 8
scale-invariant donor
fixed alpha = 16
relative-direction strength = 2
direction normalization = initial_rms
eta = 0.5
mu = 0.01
factorized Gibbs sweeps = 0
output = terminal current
```

不得同时修改 P、C、rho、alpha、donor、eta、mu、direction normalization、Gibbs 或初始化方法。
absolute 路径忽略 floor；另外两臂使用同一个 floor=8。

## 5. 在线与离线边界

在线停止只读取 measured-query squared loss、历史 best 刷新和 applied participating rows；不读取
normalized L1，不按 residual arm 更换 P/C，也不读取未来状态。

停止后只离线复算：

- terminal-current measured normalized L1；
- terminal-current squared loss；
- rounds、candidate evaluations、normalized work；
- A/B/C termination reason；
- elapsed wall time（只作成本描述，不参与确定性身份）。

本轮不读取原始 reference table，不计算 held-out/支持集/多样性，不形成最终质量资格结论。

## 6. 汇总规则

每个数据集分别报告：

1. 每 arm 的三条原始结果；
2. terminal normalized L1 的 mean/median；
3. 每 seed 的 L1 配对赢家和 3 seed 胜数；
4. squared loss、rounds、normalized work 的 mean/median；
5. A/B/C 数量与 resource-cap cases。

“本数据上 L1 最好”只指三 seed 平均 terminal normalized L1 最低。若 L1 最优 arm 使用更多 work，
必须并列写成质量—计算取舍，不能合成未预注册的单一分数。若任何 arm 出现 C，只报告
resource-capped，不把它与正常 A/B 终点包装成相同完成身份。

## 7. 执行与审计

- runner 只有 result-blind `plan`、固定 seed shard 执行和只读 `aggregate`；没有科学参数覆盖；
- 完整 protocol manifest 计算 SHA-256，正式执行必须确认完整 SHA；
- 每个 shard 要求包含 untracked 在内的 clean worktree，并只暴露一张 GPU；
- 正式运行使用 `root@10.8.176.53:6006` 的 RTX A6000；只暴露一张空闲卡、一个 worker，三个 seed
  shard 串行执行，不并占多张 GPU，GPU 3 的既有任务不触碰；
- 每个 case 保存 terminal table、结果 JSON 和 SHA；aggregate 校验 18/18 后原子生成总报告；
- 结果出现后不增加 seed、不改 floor/P/C、不重跑选择性 case。

## 8. 允许结论

允许：

```text
在当前两数据、P=6、三 fresh paired seeds 的 development 对比中，
逐数据描述哪种 residual arm 的 terminal measured L1 最低及其工作量代价。
```

不允许声称 canonical residual 已选定、平方根方法跨数据普遍最好、P=6 全局最优、held-out/多样性
通过、算法收敛或结果可外推到带噪/DP 阶段。
