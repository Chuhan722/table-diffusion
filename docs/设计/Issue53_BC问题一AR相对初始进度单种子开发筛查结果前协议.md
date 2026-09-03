# Issue 53：A/R 相对初始进度单种子开发筛查结果前协议

## 1. 目的与结论边界

本实验只检验：将 C 核从 `max(A,R)` 改为
`max(A/A_init,R/R_init)`，能否在不牺牲稀有查询的前提下，解除绝对通道 A 对
相对通道 R 的量级遮蔽，并改善上一版在常见查询上的结果。

这是复用 seed 9908 的开发筛查，只能决定是否值得进入未见新种子确认；不能据此
宣称正式优越，不能修改公开默认值。

## 2. 唯一候选

```text
arm                         gap_dual_abs_relative_progress_max_s8
gap_l1_weighting            dual_abs_relative_progress_max
gap_l1_sweeps               8
gap_l1_max_weight_ratio     null
A/R aggregation             max(A/A_init, R/R_init)
channel reference source    initial_current_before_round_1
zero-target policy           absolute_channel_only
```

`A_init`、`R_init` 只在 round 1 前从初始合成表查询计数计算一次，随后对所有轮次、
所有地址、孤立分数定尺和微步冻结。不得按轮重算，不得加 epsilon，不得另调比例。
完整边界见《Issue53_BC问题一AR相对初始进度max设计冻结.md》。

## 3. 冻结任务矩阵

只生成两条候选轨迹：

| 顺序 | 数据集 | seed | arm |
|---:|---|---:|---|
| 1 | `test_300x10` | 9908 | `gap_dual_abs_relative_progress_max_s8` |
| 2 | `nltcs` | 9908 | `gap_dual_abs_relative_progress_max_s8` |

轮数上限、候选预算、P6 自动停止、固定检查点、B 残差方向、初始化、donor 选择、
参与率、突变、随机流和无门控接续全部继承上一版 A/R 筛查，不重跑历史臂。

因为 seed、初始化及生成输入不变，round 0 冻结参照应为：

| 数据集 | A_init | R_init |
|---|---:|---:|
| `test_300x10` | 0.023733333333333332 | 0.011553523550340313 |
| `nltcs` | 0.063173770793819 | 0.03520355244885822 |

任何不相等都视为身份或接线漂移，而不是重新估计新参数。

## 4. 运行有效性

两条轨迹必须同时满足：

- 生成期间不读取参考表或未见查询，只使用冻结 schema、测量查询目标和 marginals；
- 每轮只有一个候选且无条件成为下一 current，不接受/拒绝、不重试、不回滚；
- 每个有效扫描严格执行 `8*K` 个微步，所有条件概率有限且严格位于 `(0,1)`；
- 新模式、初始参照、A/R 权重常数、零目标策略和最终通道诊断完整；
- 每个数据集完整记录 R 主导候选侧的次数，且该次数允许为 0；
- 固定检查点和 terminal current 完整，输入、代码、协议和产物 SHA-256 可核验；
- 两条候选全部采集结束后才允许单独启动质量评价。

上述采集完整性任一项失败，分类首先为 `execution_invalid`。在采集有效的前提
下，只要任一数据集的 R 主导候选侧次数为 0，就单独分类为
`relative_channel_not_activated`；这是设计没有实际发挥作用的科学结果，不是
执行错误。

## 5. 质量比较基线与门槛

主比较基线是同 seed、同任务的上一版 `gap_dual_abs_relative_max_s8` terminal
current；历史 legacy、R8、sqrt 仅保留上下文，不因本实验重跑。

通过开发筛查必须同时满足：

1. `nltcs` 全部测量查询 normalized L1 相对上一版严格 `< 1.0`；
2. `nltcs` 常见目标分箱 normalized L1 相对上一版严格 `< 1.0`；
3. `nltcs` 稀有目标分箱 normalized L1 相对上一版 `<= 1.05`；
4. `test_300x10` 全部测量查询 normalized L1 相对上一版 `<= 1.05`；
5. 两个数据集的 1-way 评价相对上一版各自 `<= 1.05`。

分类顺序冻结为：

1. `execution_invalid`；
2. `relative_channel_not_activated`；
3. `common_query_improvement_not_recovered`；
4. `rare_query_safety_risk`；
5. `measured_or_one_way_safety_risk`；
6. `advance_to_fresh_seed_confirmation`。

门槛在生成前冻结；不得看到候选结果后修改。

## 6. 信息流与授权边界

- 采集阶段不能读取参考表、未见查询或历史质量排名；
- 评价阶段只有在两条候选完整封存后才读取冻结参考评价输入；
- 不允许逐检查点挑赢家，主结论使用 P6 terminal current；
- 不自动搜索 A/R 系数、epsilon、稀有阈值或新分母；
- 本协议冻结与 preflight 不授权真实 GPU 生成；启动采集仍需用户后续明确确认
  完整执行协议 SHA-256。
