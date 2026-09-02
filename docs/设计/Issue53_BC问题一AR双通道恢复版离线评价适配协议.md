# Issue #53 B+C 问题一：A/R 双通道恢复版离线评价适配协议

日期：2026-09-02

## 1. 固定输入

候选输入只允许使用经计数勘误恢复、完整验证后的两 case collection：

```text
collection SHA-256
3b53fcac678924205a4bd1038ec15f404b5245b09894d2b065d28794e615a790

recovery protocol SHA-256
2132a4392740114398064b62bfc604cb97b3226e4b758d654dbd6a71dcc8c4b3
```

候选地址固定为开发种子 9908 的 `test_300x10` 与 NLTCS，臂固定为
`gap_dual_abs_relative_max_s8`。不得选择历史最好表或某个检查点代替
`terminal_current`。

比较基线固定复用此前已独立审计的六条 case：两个数据集上的 legacy、R8 和
平方根臂。collection、evaluation、CSV 与 independent audit 均逐文件绑定
结果前协议已经冻结的 SHA-256；不重跑基线。

## 2. 评价算术

候选终表、固定检查点、measured queries、target 分箱、query order 与 offline
query groups 全部复用冻结 R8 evaluator 的同一算术、查询集合、reference 和
零分母比例规则。

A/R 检查点中的 `gap_e` 不是 legacy proxy，适配器按冻结候选公式独立复算并
只用于身份校验：

```text
A = sum_all |y-q| / (N*J)
R = sum_positive(|y-q|/y) / (N*sum_positive(1/y))
E = max(A,R)
```

这不改变 normalized L1、分箱、one-way 或任何决策指标的算术。

## 3. 结果前筛查门

执行有效后按既定顺序分类：

1. NLTCS measured normalized L1 必须严格低于 legacy；
2. NLTCS 常见箱绝对计数误差均值必须严格低于 legacy；
3. NLTCS 稀有箱必须不超过 `1.25*legacy`；
4. NLTCS 稀有箱必须严格低于平方根臂；
5. `test_300x10` measured normalized L1 不得超过 `1.05*legacy`；
6. 两数据集 one-way safety 各自不得超过 `1.05*legacy`。

分类优先级固定为：

```text
execution_invalid
common_mechanism_not_retained
rare_query_protection_not_recovered
measured_or_one_way_safety_risk
advance_to_fresh_seed_confirmation
```

R8 作为完整展示对照保留，但不替代已冻结的“稀有箱 vs sqrt”判定门。

## 4. 发布与边界

评价必须在干净提交、评价报告与 CSV 均不存在时启动；只有 adapter SHA 与
collection SHA 同时确认后，才允许读取候选终表、reference 和冻结基线质量。
报告与候选 CSV 原子发布且不得覆盖。

本评价是读取 R8/sqrt 开发结果后提出的新候选的单种子开发筛查，不是独立确认。
评价完成后不得自动扩种子、调参、运行独立审计、推送或操作 PR #69。

