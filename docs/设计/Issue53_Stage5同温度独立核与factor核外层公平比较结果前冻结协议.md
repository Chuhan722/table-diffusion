# Issue #53：Stage 5 同温度 independent / factor 外层公平比较结果前冻结协议

> 状态：**用户已于 2026-08-22 确认，结果前冻结**。
>
> 日期：2026-08-22。
>
> 本文件在任何 Stage 5 smoke 或正式 seed 运行前冻结。独立 Stage 5 本地分支已从 Draft PR #67
> head 建立；当前尚未实现 runner、evaluator 或 auditor，没有实例化正式 seed，也没有读取任何
> Stage 5 结果。后续 protocol module 必须绑定本文 SHA-256，正式 collector 另行绑定 clean
> implementation / execution commit。

## 1. 只回答一个问题

Stage 4 已经回答：在冻结的 `tau=2` factor 目标分布上，random-scan Gibbs 以 8 sweeps 能否达到
条件正确性与混合资格。

Stage 5 只回答：

> 在当前冻结的完整外层生成流程中，保持相同温度、供体、残差、初始化、停止和输出协议时，
> 已取得资格的 factor Gibbs 8-sweep 核能否相对 independent 核稳定改善 terminal-current
> 生成质量，同时保持离线安全、支持集、多样性和有效性；它需要多少额外计算成本？

Stage 5 不做以下事情：

- 不选择最佳 `tau`，两臂都固定 `tau=2`；
- 不扫描 Gibbs sweeps，factor 臂只使用 Stage 4 已取得资格的 8 sweeps；
- 不重新选择 residual、floor、donor、alpha、rho、eta、mu、P 或资源上限；
- 不加入接受门、回滚、best-checkpoint 输出、shadow 选择或外层择优；
- 不把 factor 核升级为公共默认；
- 不外推到其他数据、带噪、隐私预算或 DP；
- 不回答 P4 行内/跨行过冲的因果来源，该问题仍属于后续 Stage 6。

## 2. 上游资格与科学边界

Stage 5 必须绑定并继承以下 Stage 4 正式身份：

```text
Stage 4 protocol SHA-256
6a2834db4cb75fffbaac3330bbb1923fa2e864572ca05ac901c3142ecd443680

Stage 4 clean execution commit
e7cbcc0beaf6718f7f4ad148a8ee07f8cc9a089f

Stage 4 result
qualified_random_scan_s8

state library SHA-256
3c7475e89d693bd2240846bb78dec6b7a8d2abc14a71beea25fd6be3e2a02561

mixing report SHA-256
15751180b96c6a466f7096a63935d93eb60f47b836f2aa9462a1842ee58b7fa5

independent audit SHA-256
fde7929a39cb039d26dd56b91063fa4151a639ea0e483ba8b9303e912a42ed6d
```

当前 Draft PR #67 head 为
`e463cdeab89baba2115c71a6c03318a4a921920b`。它只是拟议的 Stage 5 分支起点，不作为未来正式
执行提交的替代。最终 collector 必须绑定届时实际的 clean implementation commit；若下层 PR 同步导致
代码树变化，必须重新做实现审计与测试，不能继续沿用旧执行身份。

Stage 4 的结论只证明 8-sweep 内层核在冻结状态分布上的资格。Stage 5 不得把该资格预写成外层质量优势。

## 3. 唯一实验变量：外层提案核

| 身份 | independent 臂 | factor 臂 |
|---|---|---|
| 名称 | `independent_s0` | `factor_random_scan_s8` |
| `factorized_gibbs_sweeps` | 0 | 8 |
| Gibbs 初态 | 不适用 | 同轮 independent directional mask |
| Gibbs 扫描 | 不适用 | random scan with replacement |
| Gibbs RNG | 不实例化 | 从同一正式 seed 独立派生 |
| compiled factor workload | false | true |
| test 最大因子阶数 | 参数不生效 | 4 |
| nltcs 最大因子阶数 | 参数不生效 | 3 |
| Gibbs logit clip | 参数不生效 | 30 |

最大因子阶数由 measured workload 的最高查询阶数直接决定，不是待调超参数：

- `test_300x10` 的 measured workload 含 2/3/4-way，因此为 4；
- `nltcs` 的 measured workload 只含 2/3-way，因此为 3；
- held-out 4-way 只允许在生成完成后评价，不得进入 nltcs 的 factor energy。

factor 臂使用已经验证轨迹、查询向量和 RNG 精确等价的 compiled workload 路径，以避免把已知的
rowwise 构造开销当成 Gibbs 科学成本。正式 smoke 必须在当前父代码上再次验证 compiled 与 rowwise
的非计时结果等价。一次性编译时间和运行内 factor-build 时间仍分别报告，不隐藏实现成本。

## 4. 数据集、measured workload 与输入身份

### 4.1 `test_300x10`

```text
schema             configs/test_300x10/schema.yaml
measured queries   configs/test_300x10/measured_50query_30_15_5.json
marginals          configs/test_300x10/init_marginals.json
N                  300
backend            numpy
measured identity  602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1
target identity    e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d
workload           30 × 2-way + 15 × 3-way + 5 × 4-way
```

输入文件 SHA-256：

```text
schema      58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792
queries     708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14
marginals   1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4
```

### 4.2 `nltcs`

```text
schema             configs/nltcs/schema.yaml
measured queries   configs/nltcs/measured_1000query.json
marginals          configs/nltcs/init_marginals.json
N                  16181
backend            cuda
measured identity  48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429
target identity    f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d
workload           479 × 2-way + 522 × 3-way = 1001 queries
```

历史文件名写作 `1000query`，正式身份按文件内完整 1001 条 ordered queries 与 SHA-256 使用，不删减。

输入文件 SHA-256：

```text
schema      5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4
queries     b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f
marginals   a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e
```

## 5. 除核之外全部固定

两臂的公共生成身份固定为：

```text
initialization                           marginal
n_rounds / candidate_budget             6000 / 6000
beta / h                                1.0 / 0.8
rho                                     0.01
eta / mu                                0.5 / 0.01
tol                                     positive infinity
max_retries                             0
distance_mode                           geometric
lambda_param / delta                    0.5 / 0.05
winsorize_quantiles                     0.01 / 0.99
exclude_self                            true
selection_scale_invariant               true
selection_scale_invariant_min_spread    1e-3
residual_geometry / floor               relative / 8
residual_directed_diffusion             true
direction strength / temperature        2.0
direction normalization                 initial_rms
direction logit clip                    30
alpha_schedule_mode                     fixed
fixed_alpha                             16
residual_self_cooling                   off
rho_anneal                              off
stop_on_exact_residual                  true
inner patience                          P=6 completed natural-work ticks
output                                  terminal current
record_transition_clocks                true
record_stationarity_trace               false
record_natural_work_snapshots           false
```

在线停止规则保持：

```text
natural work = cumulative applied participating rows / N
work tick    = floor(natural work)

A：current squared loss == 0
B：连续 6 个已完成 work tick 没有严格历史 best 刷新
C：6000 raw rounds 或 6000 candidate evaluations
优先级：A > B > C
```

`tol=+inf`、`max_retries=0` 意味着每轮唯一有限 proposal 都成为下一张 current table。历史 best 只参与
进展诊断，不参与状态更新或输出。normalized L1、held-out、参考支持、多样性、墙钟和未来状态都不得进入
停止、核选择或输出选择。

## 6. 正式 seed、矩阵与执行顺序

正式 paired seeds 固定为：

```text
338, 339, 340, 341, 342, 343, 344, 345, 346, 347
```

这些 seed 尚未在当前 Issue #53 的正式生成矩阵中使用。既有阶段的 `313..317`、`318..322`、
`323..327`、`328..332`、Stage 4 qualification `333..337` 均不得复用。

正式矩阵：

```text
2 datasets × 2 kernels × 10 paired seeds = 40 trajectories
```

同一 dataset/seed 的两臂必须在同一 host、同一后端、同一设备型号上串行执行。为平衡进程冷热、缓存和
执行顺序的描述性墙钟偏差，顺序固定为：

```text
偶数 seed：independent_s0 -> factor_random_scan_s8
奇数 seed：factor_random_scan_s8 -> independent_s0
```

每个 seed shard 内数据集顺序固定为 `test_300x10 -> nltcs`。不同 seed shard 可以并行，但每个 shard
必须显式且仅暴露一张 CUDA GPU；同一物理 GPU 不同时运行两个正式 shard。协议绑定 runtime
`device=cuda` 和设备型号，不绑定物理卡号。

formal collector 不得接受 seed、臂、数据集、顺序、P、C 或生成参数覆盖项。

非正式 pipeline smoke 只允许 seed `9905`，使用相同真实输入、两个数据集、两个核，每 case 最多 3 轮，
`formal_result_valid=false`。Stage 4 smoke seed `9904` 不复用。smoke 只能验证管线、配对、compiled
等价、审计和落盘，不能产生方法证据或修改正式规则。

## 7. 配对身份与公平性门禁

每个 dataset/seed 的两臂必须逐项通过：

1. schema、ordered measured queries、target 和 marginals 身份一致；
2. marginal 初始化后的表逐字节一致，initial table SHA-256 一致；
3. 初始化后 primary RNG state SHA-256 一致；
4. `initial_rms` 得到相同、正有限的 direction reference scale；
5. factor 的独立初始 mask 使用与 independent 路径相同的算法和 primary RNG 地址；
6. factor 的额外 Gibbs 随机量只来自独立派生 RNG，不改变 primary RNG 的共同随机流；
7. 两臂都恰好一 candidate/round、无重试、无接受门；
8. 输出都为 A/B/C 触发状态的 terminal current table；
9. 终表、diagnostics、参数、RNG 端点和 artifact SHA-256 可独立复算。

状态一旦分叉，数据依赖的 donor 概率与实际 donor 可以不同。这是核改变后完整闭环的真实方法效应，
不得强行锁死 donor 或用同一落地 proposal 替换真实闭环。配对保证的是共同起点与共同随机地址，不宣称
分叉后两臂仍走相同状态路径。

两臂按同一停止协议自然结束，不强制相同 raw rounds 或相同墙钟。natural work 只衡量外层实际参与行，
不把 8-sweep Gibbs 的内部微步伪装成免费工作；内部成本由独立计数和墙钟完整披露。

## 8. 生成与离线评价的信息隔离

### 8.1 collection 阶段

正式生成只允许读取：

- schema；
- measured ordered queries 与 measured target；
- marginal 初始化文件；
- 冻结 protocol、生成代码和在线 A/B/C 状态。

不得读取：

- 原始 reference table；
- offline safety / held-out 查询答案；
- reference support、binned joint TVD、多样性比较或任何离线分类；
- 其他正式 seed 的部分质量结果。

collector 只允许发布逐 case 原始 artifact、终表、身份与完整集合 manifest，不生成 factor/independent
胜负、跨 seed 均值、held-out 指标或最终分类。40/40 完成并通过 collection audit 前，不运行 evaluator，
也不按部分结果决定是否继续剩余 case。

基础设施中断且未生成有效 case artifact 时，可以从同一 clean commit 按 manifest 恢复。若已读取正式
结果后发现科学代码、协议或评价身份错误，本批 seed 作废；不得修正后仍把同一批结果称为未见正式证据。

### 8.2 evaluation 阶段

只有完整 collection manifest 被显式确认后，离线 evaluator 才能读取：

```text
test reference   data/test_300x10/test_300x10.csv
SHA-256          c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb

nltcs reference  data/nltcs/nltcs.train.data
SHA-256          e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30
```

evaluator 必须只读复算 40 张终表，不能调用 generator、改变表、补跑 seed、改变查询或覆盖 collection。
独立 auditor 再从原始终表和 manifest 复算全部主值、配对差、门禁和分类。

## 9. 固定评价内容

### 9.1 measured 主质量

每个 case 报告：

- terminal measured absolute count error sum；
- terminal measured normalized L1 mean、median、P90、max；
- terminal current squared loss；
- 按 measured 2/3/4-way 分阶的相同误差统计；
- terminal current 与 `best_loss_diagnostic_only` 的差距；
- historical best 只读诊断，不保存为候选输出、不参与分类选表。

主胜负以整数 count error sum 复算，避免浮点 tie。normalized L1 用于跨数据解释，不改变同数据配对顺序。

### 9.2 离线查询安全

`test_300x10`：

| 组 | 数量 | 查询语义身份 SHA-256 |
|---|---:|---|
| `one_way_safety` | 25 | `b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c` |
| `common_unseen_2way` | 521 | `fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2` |
| `fixed_heldout_3way` | 512 | `d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a` |
| `fixed_heldout_4way` | 512 | `2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171` |

`nltcs`：

| 组 | 数量 | 查询语义身份 SHA-256 |
|---|---:|---|
| `one_way_safety` | 32 | `bbc8fc5d1b1ed0e5cd318a2168fe3887297b1c6aa33634736d0c693e96785c13` |
| `unmeasured_3way` | 3958 | `9c43437d6366e3cce0438fdf79e104d70ebabc112db9236b3feef5220b5eb588` |
| `all_4way` | 29120 | `1b92f8d80e775cffd637450d3d5015c78d43f7d9a870faf1603c99c88ec5d408` |

nltcs 另报告既有冻结口径的 `binned_joint.tvd`。每组均报告 normalized L1 mean、median、P90、max
和 squared-loss diagnostic。不同组不合成总分。

### 9.3 支持集、多样性与有效性

每个 case 复用现有离线质量契约，报告：

- `synthetic_mass_in_reference_support`；
- `reference_mass_covered`；
- unique row count / rate；
- empirical row entropy、effective unique rows / ratio；
- attribute effective support ratio mean / minimum 及逐属性值；
- valid row count / rate、invalid row count 及逐属性合法率。

reference support 只在完整 collection 后计算，不进入生成。

### 9.4 terminal-current 行为

报告：

- `fit_target_reached` / `early_stopped` / `resource_cap_reached`；
- rounds、candidate evaluations、normalized work、completed work ticks；
- 最后一次严格新 best 到终止之间的 natural-work 距离；
- terminal squared loss 与 best squared loss diagnostic 的差；
- terminal measured L1 与 best-squared-loss 状态的 measured L1 diagnostic 差；
- current table SHA、状态索引与 A/B/C 输出身份。

这些量解释两种核在 P=6 下的 terminal-current 行为，不允许改成 best 输出。

### 9.5 成本

每个 case 至少报告：

- normalized work、cumulative applied participating rows；
- raw rounds、candidate evaluations、state/query evaluations；
- direction evaluations 与耗时；
- factor workload compile、factor build、Gibbs sampling 耗时；
- Gibbs microsteps；
- conditional-logit evaluated / clipped count；
- 完整 generator elapsed time 与 case wall-clock time；
- runtime、设备型号与执行顺序。

墙钟不进入硬科学门禁，因为它受设备、进程冷热与系统负载影响；必须保留逐 seed 配对值、均值、中位数、
比值和顺序分层，不能只发布一个总和。

## 10. 结果前冻结的判定规则

所有判断都在 `test_300x10` 和 `nltcs` 上分别执行，不做跨数据、跨查询组或质量—成本加权总分。

### 10.1 执行资格

以下条件必须全部满足：

1. 40/40 个 case 和 20/20 个 dataset/seed 配对完整；
2. 全部 case 由 A/`fit_target_reached` 或 B/`early_stopped` 正常结束；
3. 0 个 C/`resource_cap_reached`；
4. 全部输入、初态、RNG、参数、terminal-current、artifact 和配对身份审计通过；
5. 全部数值有限、表结构合法；
6. 两臂 direction clip hit 均为 0；factor conditional Gibbs clip hit 为 0；
7. factor microsteps 与已应用/尝试参与身份一致，compiled 等价门禁通过。

任一 case 触及 C，整个矩阵分类为 `inconclusive_resource_cap`：不提高 C、不追加 seed、不把已完成 case
当作正式胜负。结构或身份失效则为 `execution_invalid`。若出现 clip，则标记
`outside_stage4_qualified_unclipped_regime`，不得给出 factor 支持结论，也不事后提高 clip。

### 10.2 稳定 measured 主改善

对每个数据集的 10 个 paired seeds：

```text
factor 的 10-seed aggregate measured count-error sum
    < independent 的 aggregate measured count-error sum

且 factor 至少 8/10 paired seeds 的 measured count-error sum 严格更低
```

严格相等记 tie，不算任一方胜。95% paired interval 只作描述，不增加或替代上述门槛。

同时反向计算：若 independent aggregate 更低且至少 8/10 严格胜，记录
`stable_independent_measured_advantage=true`。若两侧都没有达到稳定门槛，记录 mixed，而不是追加 seed。

### 10.3 factor 离线安全门禁

在 10-seed mean 上，factor 的每个独立风险项必须满足：

```text
lower-is-better 指标：factor <= 1.05 × independent
higher-is-better 指标：factor >= 0.95 × independent
```

lower-is-better：

- test 四个 offline query groups 的 normalized L1 mean；
- nltcs 三个 offline query groups 的 normalized L1 mean；
- nltcs binned joint TVD。

higher-is-better：

- `synthetic_mass_in_reference_support`；
- `reference_mass_covered`；
- unique row rate；
- effective unique row ratio；
- attribute effective support ratio mean / minimum。

若 independent 的 lower-is-better baseline 为 0，factor 必须也为 0。若 higher-is-better baseline 为 0，
该项只报告、不形成无意义的比例通过证据。所有 factor case 必须 `valid_row_rate=1`；任一无效行直接
使 validity gate 失败。

每个安全项独立列出逐 seed 值和比例，不允许一个组的改善抵消另一个组的恶化。

### 10.4 外层计算门禁

外层计算非劣要求同时满足：

```text
factor mean normalized work
    <= 1.05 × independent mean normalized work

factor mean candidate_evaluation_count
    <= 1.05 × independent mean candidate_evaluation_count
```

Gibbs microsteps和 conditional-logit evaluations 没有 independent 同量纲对照，必须作为 factor 的额外内部
成本原样报告。墙钟同样只报告，不设任意硬门槛。因此“外层计算通过”不等于 Gibbs 免费，也不等于已经
达到产品部署成本要求。

### 10.5 每数据集固定分类

按以下顺序分类：

1. 执行资格失败：`execution_invalid`、`inconclusive_resource_cap` 或
   `outside_stage4_qualified_unclipped_regime`；
2. factor 稳定 measured 改善失败：`no_stable_factor_gain`，并附
   `stable_independent_measured_advantage` 或 `mixed_no_stable_kernel_winner`；
3. factor 稳定 measured 改善通过，但 offline safety、support、diversity 或 validity 任一失败：
   `factor_measured_gain_with_quality_or_diversity_risk`；
4. 质量、安全、支持、多样性和有效性全部通过，但外层计算门禁失败：
   `factor_quality_supported_with_outer_compute_tradeoff`；
5. 上述门禁全部通过：`factor_quality_supported_outer_efficient`。

第 5 类仍必须同时发布 factor/independent 墙钟比、Gibbs 微步和 conditional evaluations，不能写成
“factor 无额外成本”。第 4 类也不自动否定 factor，只说明需要用户在质量收益与成本间另作选择。

### 10.6 跨数据汇总

- 两数据都为 `factor_quality_supported_outer_efficient`：
  `shared_factor_support_at_tau2`；
- 两数据质量均获支持，但至少一套存在 outer compute tradeoff：
  `shared_factor_quality_support_with_compute_tradeoff`；
- 只有一套数据获得 factor 质量支持：
  `dataset_dependent_kernel_response`；
- 两套都没有 factor 稳定质量支持：
  `no_shared_factor_support`；
- 任一数据执行无资格：
  `inconclusive_or_invalid_stage5`。

不做跨数据加权投票。跨数据结果不能自动修改公共默认。

## 11. 结果后的唯一解释边界

### 11.1 factor 两数据获支持

只允许写作：

> 在当前 higher-order workload、relative/floor=8、scale-invariant donor、fixed α16、P=6、
> `tau=2` 的冻结身份下，8-sweep factor Gibbs 获得外层质量支持；其实际计算代价见正式报告。

下一步可单独讨论：是否先做独立确认、Stage 6 P4 机制诊断，或只对已支持的 factor 核设计新的 tau
结果前协议。不得用本结果宣称最佳 tau。

### 11.2 dataset-dependent

不得设置共享 kernel 默认，也不得通过跨数据平均掩盖差异。先报告哪套数据支持、哪套不支持及风险/成本
来源，再决定是否值得做数据依赖机制诊断。

### 11.3 factor 未获支持

保留 independent 作为当前 development reference。不得在同一批 seed 上改 sweeps、tau、rho 或 P
把 factor 调到通过；若未来重开 factor，必须提出新的单变量问题与全新结果前协议。

### 11.4 tau 的位置

Stage 5 全程固定 `tau=2`。只有核比较结束后，才知道后续 tau 调优应花在 factor、independent 还是分数据
身份上。任何 tau 网格、连续优化或核×tau 联合搜索都不属于本协议。

## 12. 冻结后的实施顺序

按以下互相隔离的步骤执行，每一步都需要对应授权：

1. 已从经审计的 #67 head `e463cdeab89baba2115c71a6c03318a4a921920b` 建立独立本地
   Stage 5 worktree/branch；
2. 本协议作为独立结果前提交移入 `docs/设计/`，protocol module 绑定其文件 SHA-256；
3. 实现只读 `plan`、固定 protocol module 和确定性契约测试；`plan` 不实例化 RNG、不导入生成器、
   不读取 formal results；
4. 实现 collector、离线 evaluator、独立 auditor 与 artifact 原子落盘；命令行不开放科学覆盖参数；
5. 运行普通测试和 seed `9905` 的 pipeline smoke，确认 compiled 等价、配对、RNG、terminal-current、
   信息隔离与审计；
6. 根据 smoke 报告实际 GPU、单 case 计时、正式总预算与运行安排，取得用户单独正式运行授权；
7. 从 clean execution commit 完成 40-case collection，不查看或发布部分矩阵结论；
8. 只读审计 collection，向用户报告完整性与 collection SHA；
9. 用户另行确认后运行离线 evaluator；
10. 独立 auditor 复算全部指标和冻结分类，再向用户报告结果。

任何一步都不自动 push、建 PR、评论 Issue、请求 review、修改公共默认或启动下一步。用户没有明确说
“推”就不 push；不催 reviewer。

## 13. 已冻结的关键选择

用户确认冻结的三个关键选择为：

1. 正式使用 `338..347` 共 10 个 paired seeds，而不是 5 个；
2. factor 正式臂显式使用已验证轨迹等价的 compiled workload 路径，并单独报告编译/构造成本；
3. 墙钟只作完整配对诊断，不设任意硬阈值；科学分类使用质量、安全、支持、多样性、有效性和外层
   normalized work / query evaluations，Gibbs 内部成本始终原样披露。

本文内容若在任何 Stage 5 结果暴露后发生科学性修改，必须升级协议版本并使用全新正式 seed；不得用
修改后的协议追认已经看过的结果。本协议本身不授权 push、smoke 或正式实验。
