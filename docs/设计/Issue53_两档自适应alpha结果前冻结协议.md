# Issue #53：两档自适应 α 结果前冻结协议

> 状态：结果前设计协议。本文写成时只完成了既有固定 α 结果复核、控制器语义设计和新随机种子审计；
> 尚未实现自适应状态机、运行 smoke（小型检查）或正式轨迹，也没有读取本矩阵的任何新结果。

## 1. 本次只回答什么

既有固定 α 响应实验已经证明：α 是稳定的 donor concentration（供体集中度）控制量——α 越小，
供体选择越分散；α 越大，供体选择越集中。但 α=`12/16/24` 没有一个点同时在两套数据上通过
“已测拟合、离线安全、多样性、计算量”的完整门禁。

本实验只回答一个新问题：

> 以固定 `α=16` 为正常档时，在运行真实停滞后临时切到 `α=12` 两个自然工作刻度，再恢复
> `α=16`，能否比一直使用 `α=16` 更稳定地改善终态质量，并保住离线安全、多样性和计算量？

同时保留固定 `α=12` 作为机制对照，用于区分“只在停滞时短暂放宽供体选择”和“从头到尾都使用
低 α”。它不是本轮主要基准，也不是重新进行固定 α 搜索。

本实验不允许声称：

- α=12 或 α=16 是所有数据、查询集合或带噪阶段的全局最优固定值；
- 两档状态机已经覆盖连续 α、更多档位或其他触发信号；
- donor concentration（供体集中度）本身越高或越低就必然越好；
- 本次使用的数据是从未参与开发的外部测试集；
- 一次 5-seed 开发实验就是收敛证明或公共 API 默认值变更依据。

## 2. 为什么先试简单的 16→12→16

尺度不变供体选择使用：

```text
z_ij     = (log_A_ij - row_mean_i) / max(row_std_i, 1e-3)
logit_ij = α * z_ij
prob_ij  = softmax(logit_i)_j
```

固定响应实验中，两套数据都稳定表现为：从 α=16 降到 α=12 后，有效供体比例上升，单行最大供体
概率下降。也就是说，α=12 可以作为一个方向已知的“短暂扩大探索”动作。

但两套数据的有效供体比例绝对量级差异很大，而且固定 α 的质量取舍不同。因此第一版不反推“目标
供体比例”，也不逐轮连续调 α。供体集中度只用于验证切换是否真的让选择变分散，不进入控制决策。

本轮只使用两个已经实测过的档位：

```text
normal alpha（正常档） = 16
escape alpha（探索档） = 12
```

不使用 α=24，不增加 14、18、20 等新点，也不根据结果追加档位。

## 3. 控制器的精确定义

### 3.1 自然工作刻度与进展

继续复用现有 P=6 早停定义：

```text
normalized_work = cumulative participating rows / N
completed_work_ticks = floor(cumulative participating rows / N)
```

其中 `N` 是当前合成表行数。一次 post-round（轮后）观察最多跨过一个自然工作刻度。

`progress`（进展）只指当前损失严格小于此前历史最好损失。一个自然工作刻度内只要出现过至少一次
严格新最好，该刻度就算有进展；否则算一个连续无进展刻度。相等、数值抖动、供体集中度改变、α 切换
或当前损失从较差状态回到旧最好，都不算新进展。

### 3.2 三个冻结参数

```text
stall trigger（停滞触发）       = 连续 2 个无进展自然工作刻度
escape duration（探索持续）     = 恰好 2 个自然工作刻度
early-stop patience（早停耐心） = 连续 6 个无进展自然工作刻度
```

每条轨迹从 α=16 开始。某个 progress epoch（进展阶段）第一次达到连续两个无进展刻度时，安排一次
α=12 探索段。一个 progress epoch 是两次严格新最好之间的区间；只有严格新最好才能开始新阶段。

### 3.3 无歧义的轮次顺序

每轮必须按以下顺序执行：

1. 在本轮开始前，根据此前已经完成的观察选择本轮 α；
2. 用该 α 完成本轮 donor 抽样和状态更新；
3. 计算本轮 current loss（当前损失）；
4. 把本轮参与更新行数和当前损失交给现有早停器；
5. 先按现有 A→B→C 优先级判断终止，再为下一轮更新 α 控制器状态。

因此，若使用 α=16 的某一轮恰好完成第二个连续无进展刻度，这一轮仍使用 α=16；从下一轮起才使用
α=12。α=12 持续到两个后续自然工作刻度都完成；完成第二个探索刻度的那一轮仍使用 α=12，下一轮
恢复 α=16。

可写成以下正常无改善时间线：

```text
完成无进展 tick 1       下一轮仍用 α16
完成无进展 tick 2       触发；下一轮开始用 α12
完成探索 tick 1         继续用 α12
完成探索 tick 2         下一轮恢复 α16
完成无进展 tick 5       继续 α16
完成无进展 tick 6       B / early_stopped（正常早停）
```

这里的 tick 编号是同一连续无改善阶段里的早停计数，不是生成轮数。

### 3.4 探索段与 P=6 的关系

- α 切换本身绝不清零、暂停或延长 P=6；
- 同一个没有新最好的 progress epoch 最多触发一次探索段；
- 如果探索段一直没有新最好，连续无进展计数照常从 2 走到 6，并按现有 B 条件停止；
- 如果探索段期间出现严格新最好，P=6 立即按现有规则清零，但已经开始的探索段仍完成恰好两个
  自然工作刻度，不能因结果好坏提前缩短或延长；
- 新最好会创建新的 progress epoch。当前探索段结束并回到 α=16 后，这个新阶段未来再次连续停滞
  两个刻度时，可以触发新的一次探索段；
- 探索段不能重叠。新阶段即使在当前探索段中已经开始，也只能等当前探索段结束后再满足下一次触发；
- 若完成第二个无进展刻度的同一轮产生了严格新最好，则该刻度属于“有进展”，不触发探索；
- 若 A/`fit_target_reached`（精确达到目标）、B/`early_stopped` 或 C/`resource_cap_reached`
  已经决定停止，就不再安排任何未来 α。

### 3.5 控制器允许读取什么

控制器只允许读取现有 post-round 早停决定中的：

```text
best_updated
work_tick_completed
completed_tick_had_progress
completed_work_ticks
consecutive_no_progress_ticks
termination_reason
```

它不能读取 held-out（留出）查询、原始参考表、离线误差、未来轮次、`n_rounds`、供体比例目标、墙钟
时间或其他轨迹的结果。

## 4. 前缀不变与随机数契约

自适应调度必须 horizon invariant（预算前缀不变）：在相同输入、seed 和状态机配置下，只要两个
`n_rounds` 上限都尚未终止或触顶，它们共同前缀中的 α、donor、提案、当前表、损失、自然工作刻度和
触发事件必须逐位一致。`n_rounds` 只能作为外部 C 上限，不能参与 α 计算。

控制器本身不得创建随机数生成器，也不得额外抽取、跳过或预取随机数。每轮只把已经确定的 α 交给
现有 donor 概率计算；随机流消费顺序继续由原生成过程决定。

固定 α=16 与自适应臂在第一次触发前必须有完全相同的：

- 初始表 SHA-256；
- 初始化后主随机数状态 SHA-256；
- 每轮 α、donor、提案、当前表和损失；
- 早停与自然工作状态。

如果某条自适应轨迹从未触发探索，它必须与同 seed 的固定 α=16 轨迹直到终止逐位一致。

## 5. 分支、基线与新种子审计

本协议写在既有固定 α 工作树，不另建分支：

```text
worktree       = /home/chuhan/projects/table-diffusion-issue53-fixed-alpha
branch         = research/issue53-fixed-alpha
protocol base  = 4f1af7f8b2d05ebabd0962eb768fa7b7e1ab141d
PR #65 base    = a2bc496da223ef49a5a1e8a8e5ac6f60252ab62b
```

正式种子冻结为：

```text
seeds = [328, 329, 330, 331, 332]
```

审计范围包括 `/home/chuhan/projects` 下当前项目工作树、可见的忽略产物、`工作笔记.md`、
`PR工作记录.md`，以及本仓库所有本地 Git refs 的完整历史。直接 seed 字段、命令行形式、文件名形式、
显式列表和 seed 范围都已检查：328–332 没有被用作既有生成实验身份。历史中同样的数字会作为查询
结果、重复编号、指标或 SHA-256 片段出现，这些不构成随机种子复用。

协议提交后，328–332 即被本矩阵占用；无论正式运行是否完成，都不得把它们重新标为 fresh seeds。

## 6. 冻结输入与查询身份

### 6.1 `test_300x10`

生成必须继续使用更新后的高阶查询集合，不得退回含 1-way 已测查询的旧 workload：

| 输入 | SHA-256 |
|---|---|
| `configs/test_300x10/schema.yaml` | `58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792` |
| `configs/test_300x10/init_marginals.json` | `1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4` |
| `configs/test_300x10/measured_50query_30_15_5.json` | `708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14` |
| `configs/test_300x10/issue53_query_workload_ab_v1.json` | `a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36` |
| `configs/test_300x10/heldout_issue53_v1.json` | `300bffea1f3d9105ad8f1840d50a900616115659065efec35b3c02f7a38cc1e0` |
| 离线参考表 `data/test_300x10/test_300x10.csv` | `c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb` |

测量查询语义身份为
`602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1`，固定构成为：

```text
30 × 2-way + 15 × 3-way + 5 × 4-way = 50 queries
1-way measured queries = 0
```

分阶语义身份分别为：

```text
2-way  324263282febe4b0b4045e6806196ce00138ac98d4ca7147b922d2761ed745ce
3-way  b9200d3908def6d250973871319443c1ce3984baccc0e0e8fcf68cc6b9c081ac
4-way  394f9dddd68c38638d81c10f0c3f06d7f2159cafec2c4a772d5f2e856cecbdc6
```

### 6.2 `nltcs`

| 输入 | SHA-256 |
|---|---|
| `configs/nltcs/schema.yaml` | `5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4` |
| `configs/nltcs/init_marginals.json` | `a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e` |
| `configs/nltcs/measured_1000query.json` | `b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f` |
| 离线参考表 `data/nltcs/nltcs.train.data` | `e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30` |

测量查询共 1001 条，没有 1-way，语义身份为
`48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429`：

```text
479 × 2-way + 522 × 3-way = 1001 queries
1-way measured queries = 0
```

## 7. 冻结实验矩阵

```text
datasets = [test_300x10, nltcs]
arms     = [fixed_alpha_16, fixed_alpha_12, adaptive_alpha_16_12]
seeds    = [328, 329, 330, 331, 332]
cases    = 2 × 3 × 5 = 30
```

三臂角色固定为：

| 实验臂 | 角色 |
|---|---|
| `fixed_alpha_16` | 主要基准：当前正常档 |
| `fixed_alpha_12` | 机制对照：从头到尾使用低档 |
| `adaptive_alpha_16_12` | 候选：只在冻结停滞条件下短暂使用低档 |

每个“数据集 × seed”的三臂必须使用完全相同的边缘初始化随机流，三张初始表 SHA-256 和初始化后
主随机数状态 SHA-256 必须一致。每个 seed 片段的执行顺序固定为：

```text
test/fixed16 -> test/fixed12 -> test/adaptive
-> nltcs/fixed16 -> nltcs/fixed12 -> nltcs/adaptive
```

同数据、同 seed 的三臂必须在同一硬件后端执行并串行完成；不同 seed 片段可以分流。若跨机器采集，
必须先用非正式短前缀确认相同提交和相同后端的初态、初始化后随机数、终表、损失及轨迹签名逐位一致，
才允许合并正式结果。墙钟时间只描述环境，不进入科学门禁。

## 8. 除实验臂外全部固定

```text
n_rounds / candidate_budget          = 6000 / 6000
rho                                  = 0.01
P / inner patience                   = 6 natural-work ticks
distance mode                        = geometric
beta / h                             = 1.0 / 0.8
lambda / delta                       = 0.5 / 0.05
winsorize quantiles                  = 0.01 / 0.99
exclude self                         = true
selection scale invariant            = true
selection scale invariant min spread = 1e-3
residual geometry / floor            = relative / 8
residual directed diffusion          = true
direction strength / normalization   = 2 / initial_rms
direction logit clip                 = 30
eta / mu                             = 0.5 / 0.01
factorized Gibbs sweeps               = 0
tol / max retries                    = +inf / 0
stop on exact residual               = true
output                               = terminal current
test backend                         = numpy
nltcs backend                        = cuda
```

固定臂只分别使用恒定 α=16 或 α=12。自适应臂只使用第 3 节定义的两档状态机。不能为某个臂单独
修改 P、资源上限、rho、残差、方向强度、初始化、Gibbs、输出状态或查询集合。

## 9. 离线评价组与信息隔离

公共离线组完整复用固定 α 响应实验的语义身份。

`test_300x10`：

| 查询组 | 数量 | 查询语义身份 SHA-256 |
|---|---:|---|
| `one_way_safety` | 25 | `b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c` |
| `common_unseen_2way` | 521 | `fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2` |
| `fixed_heldout_3way` | 512 | `d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a` |
| `fixed_heldout_4way` | 512 | `2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171` |

`nltcs`：

| 查询组 | 数量 | 查询语义身份 SHA-256 |
|---|---:|---|
| `one_way_safety` | 32 | `bbc8fc5d1b1ed0e5cd318a2168fe3887297b1c6aa33634736d0c693e96785c13` |
| `unmeasured_3way` | 3958 | `9c43437d6366e3cce0438fdf79e104d70ebabc112db9236b3feef5220b5eb588` |
| `all_4way` | 29120 | `1b92f8d80e775cffd637450d3d5015c78d43f7d9a870faf1603c99c88ec5d408` |

正式生成只能读取 schema、测量查询与答案、初始化边缘以及在线早停/控制器状态。离线参考表和上述安全
查询答案不能参与初始化、α 切换、供体选择、停止、重试或 case 调度。必须先物化并审计 30/30 张终态
表，再读取固定 SHA-256 的参考表进行只读评价。

## 10. 固定报告内容

每个 case 保存既有固定 α 报告的全部内容：

- 终态 measured normalized L1（已测归一化 L1）、平方损失和分阶测量误差；
- 各离线查询组的 mean、median、p90、max 绝对计数误差与归一化 L1；
- 轮数、候选评估数、归一化工作量、停止原因和描述性墙钟时间；
- 终态表 SHA-256、schema 有效性、唯一行比例、有效唯一行比例、属性支持和参考支持；
- 逐行最大供体概率、有效供体数/比例和全局最高供体占比。

自适应臂还必须逐条保存：

- 每轮实际使用的 α；
- 每次触发所在的 completed work tick（已完成工作刻度）和 progress epoch 编号；
- 每个探索段开始/结束的状态索引、轮次、累计参与行数和自然工作量；
- 每个探索段是否出现严格新最好、第一次新最好发生在探索段内还是恢复 α=16 后；
- 每条轨迹的探索段次数、α=12 总轮数和 α=12 总自然工作量；
- 正常档、探索档及恢复后窗口的供体集中度诊断。

供体集中度和“探索段是否刷新最好”只解释机制，不单独构成质量通过门槛。不能因为探索段没有产生即时
新最好就删除该 seed、延长探索段或补跑另一组参数。

所有比较均为同数据、同 seed 的配对比较。报告五个逐 seed 原始值和差值、better/tie/worse
（更好/相同/更差）计数、均值差与 95% 配对区间；区间只作描述，不增加事后显著性门槛。不同数据、
不同查询组、质量与计算量不能压成一个总分。

## 11. 结果前冻结的判定规则

### 11.1 执行资格

30/30 个 case 都必须以 A/`fit_target_reached` 或 B/`early_stopped` 正常结束，并通过输入、初态、
状态机、前缀和终表审计。任一 case 以 C/`resource_cap_reached` 结束，本矩阵记为
`inconclusive_resource_cap`（资源上限导致无法判断），不提高上限、不追加 seed。

自适应轨迹可以自然地触发零次、一次或多次探索段，不能为增加触发次数而重跑。每个数据集若 5 条
自适应轨迹全部零触发，额外记录 `adaptive_not_exercised`（自适应动作未被实际触发）；这不改变原始
结果，但禁止声称 α=12 探索机制得到验证。

### 11.2 自适应臂相对固定 α=16 的主要门禁

在每个数据集上分别判定。稳定测量改善必须同时满足：

1. 自适应臂五 seed 平均终态 measured normalized L1 严格低于固定 α=16；
2. 五个配对 seed 中至少 `4/5` 的该指标严格低于固定 α=16。

离线安全要求每个独立安全项的五 seed mean 不高于固定 α=16 的 `1.05` 倍：

- `test_300x10`：一阶安全、公共未见二维、固定未见三维、固定未见四维；
- `nltcs`：一阶安全、未测量三维、全部四维、分箱联合 TVD（总变差距离）；
- 若固定 α=16 某项为零，自适应臂对应项也必须为零。

多样性、有效性和计算门槛为：

```text
adaptive mean unique_row_rate
    >= 0.95 × fixed16 mean unique_row_rate

adaptive mean effective_unique_row_ratio
    >= 0.95 × fixed16 mean effective_unique_row_ratio

all adaptive valid_row_rate = 1

adaptive mean normalized_work
    <= 1.05 × fixed16 mean normalized_work
```

按以下固定顺序分类：

- 全部门槛通过：`supported_adaptive_escape`（两档自适应探索得到完整支持）；
- 测量改善、离线安全、多样性和有效性通过，但计算失败：
  `quality_supported_with_compute_tradeoff`（质量支持但有计算代价）；
- 测量改善通过，但离线安全、多样性或有效性失败：
  `measured_gain_with_quality_or_diversity_risk`（已测改善但有离线质量或多样性风险）；
- 稳定测量改善不通过：`no_stable_measured_gain`（没有稳定已测改善）。

### 11.3 固定 α=12 如何区分机制

固定 α=12 使用与第 11.2 节完全相同的门槛，相对同 seed 固定 α=16 独立分类。然后每个数据集按
以下四种组合解释：

| 自适应完整通过 | 固定 α=12 完整通过 | 冻结解释 |
|---|---|---|
| 是 | 否 | `supports_timed_escape_beyond_always_low_alpha`：支持定时探索，而不是一直低 α |
| 是 | 是 | `adaptive_and_always_low_both_supported`：两者都可行，不能把收益归因于切换时机 |
| 否 | 是 | `supports_always_low_not_adaptive`：支持一直低 α，不支持当前自适应时序 |
| 否 | 否 | `no_supported_alpha12_strategy`：两种使用 α=12 的方式都未获完整支持 |

自适应与固定 α=12 的逐 seed 差值仍完整报告，但不增加一套事后直接胜负门槛。只有
`supported_adaptive_escape` 才算自适应相对 α=16 获得主要支持；固定 α=12 对照只约束如何解释这种
支持，不能替代主要门禁。

### 11.4 两套数据如何汇总

两个数据集分别给出第 11.2 和 11.3 节分类，不做跨数据加权总分：

- 自适应在两套数据都完整通过，才可称为 `shared_adaptive_support`（两数据共同支持）；
- 只在一套数据完整通过，称为 `dataset_dependent_adaptive_response`（自适应响应依赖数据）；
- 两套数据都不完整通过，称为 `no_shared_adaptive_support`（没有跨数据共同支持）。

即使两套数据都通过，是否进入更多数据、带噪设置或公共 API，仍需用户另行决定。

## 12. 实现前测试与不可变流程

协议提交后，最小实现至少覆盖以下定向测试：

1. 第二个连续无进展自然工作刻度结束后才触发；
2. α=12 精确覆盖后续两个自然工作刻度，再恢复 α=16；
3. α 切换不清零或暂停 P=6，同一无改善阶段不重复触发；
4. 严格新最好才重置 P=6，并允许未来新阶段再次触发；
5. 探索段中的新最好不提前结束当前两刻度探索；
6. A→B→C 终止优先级不变，终止后不安排未来 α；
7. 不同 `n_rounds` 上限的共同前缀逐位一致；
8. 未触发轨迹与固定 α=16 逐位一致，控制器不额外消费 RNG（随机数）；
9. NumPy 与 CUDA 后端的状态机事件语义一致；
10. test 查询严格为 30/15/5 且没有 1-way，nltcs 查询严格为 479/522 且没有 1-way；
11. plan（计划）模式固定 30 个 case、输入 SHA-256、种子和三臂身份，禁止科学参数覆盖；
12. 评价器严格实现第 11 节分类，离线参考不能进入生成或控制器。

正式流程固定为：

1. 在干净提交上实现状态机、采集器、评价器和测试；
2. 只做 plan、定向测试和不用于选参数的短前缀 smoke；
3. 将实现与本协议 SHA-256 绑定并提交；
4. 用户确认后才启动 30 条正式 fresh-seed 轨迹；
5. 30/30 终态全部物化并审计后，才加载离线参考并执行冻结分类；
6. 结果出来后先报告逐 seed 原始数据、机制事件和分类，等待用户决定下一步。

结果出现后不得增加 seed、调整 2/2/6 三个刻度参数、改变 α 档位、删掉未触发轨迹、只重跑科学结果
不理想的 case 或修改 5% 风险带。基础设施中断可以用相同提交和 case 身份原样重放，但必须保留失败
记录并说明。

## 13. 当前停止点

本协议提交只冻结设计，不授权实现、运行 GPU 实验、读取新结果、push（推送）、创建或更新 PR，
也不授权自行 review（审查）、approve（批准）或 merge（合并）任何 PR。下一步必须先向用户说明本次
冻结内容，并等待用户明确同意后再实现。
