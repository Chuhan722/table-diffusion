# Issue #53：Stage 6A 无门控 proposal 过冲来源诊断结果前冻结协议

> 状态：用户已于 2026-08-23 授权继续到结果前协议；正式 collection 尚未运行。
>
> 日期：2026-08-23。
>
> 本阶段只诊断 independent 无门控基线的 proposal 几何，不设计或验证新生成方法。本文必须在
> Stage 6A 正式 seed、smoke 和正式 proposal collection 之前冻结。任何正式运行仍需用户再次
> 明确授权。

## 1. 只回答一个问题

Stage 5 已经表明：同温度 factor Gibbs 的 aggregate 质量较好，但没有形成逐 seed 稳定支配，
因此不能据此更改默认核。Stage 6A 只回答：

> 对当前 independent 无门控基线，在全新自然工作状态上，无条件产生的下一表为什么会改善、
> 持平或恶化？负 gain 主要来自线性方向 B 非正，还是 B 为正但二次代价 C 超过 B；C 的行内
> 自身项、跨行交叉项和后置 mutation 分别扮演什么角色？

Stage 6A 不做以下事情：

- 不比较 independent 与 factor Gibbs；
- 不扫描 tau、alpha、rho、eta、mu、P、seed 数或状态位置；
- 不训练、不选择、不验证任何新 kernel；
- 不在 proposal 生成后接受、拒绝、回滚、重试或选择较好分支；
- 不输出 best checkpoint，不改变 terminal-current 契约；
- 不读取 held-out、offline safety、reference support、多样性或真实参考表来生成或分类 proposal；
- 不把本阶段的局部 frozen-state 诊断解释成长期生成质量改善；
- 不自动授权 Stage 6B。

## 2. 无门控边界

本协议中的无门控含义固定为：

1. 来源 independent 轨迹继续使用 tol=positive infinity、max_retries=0；
2. 每轮唯一有限 proposal 无条件成为下一张 current table；
3. frozen-state probe 对每个随机地址都保留结果，不按 B、C、G、loss 或任何离线指标删样、补样或重抽；
4. mu=0 的 copy-only 表只是对同一次复制随机量的诊断性反事实，mu=0.01 的 full 表才对应当前
   production independent proposal；两者之间不择优；
5. probe 结果不反馈进来源轨迹，也不改变后续 proposal 地址；
6. B、C、G 的分类只发生在两张表都产生之后，只能用于解释；
7. 将来若研究采样前的条件 B/C，只能另立协议，以连续软概率修改实现；本协议不实现该方法。

因此，配对 copy-only/full 不是两个生成臂，更不是 proposal guard。它相当于在同一个冻结状态上
观察同一份复制随机量在“关闭 mutation”和“保持 production mutation”时的两个测量值。

## 3. 精确 B、C、G 口径

对冻结 current state：

    q0 = current measured query count vector
    y  = frozen measured target count vector
    e0 = y - q0

对 full proposal：

    d  = q_full - q0
    B  = e0 dot d
    C  = 1/2 * ||d||^2
    G  = B - C

G 与平方 workload loss 的真实变化严格一致：

    G = L(current) - L(full)
    L(q) = 1/2 * ||y - q||^2

### 3.1 使用二倍整数做所有分类

两个正式数据集的 target、current count 和 proposal count 都是整数。为消除浮点阈值，正式分类只使用：

    B2 = 2 * e0 dot d
    C2 = d dot d
    G2 = B2 - C2

B2、C2、G2 必须以有符号 int64 精确计算，并在计算前验证保守上界不会溢出。浮点 B、C、G
只可作为可读输出，不能用于类别或最终决定。

full proposal 的一级类别固定为：

| 条件 | 类别 |
|---|---|
| G2 > 0 | improving |
| G2 = 0 且 C2 = 0 | unchanged |
| G2 = 0 且 C2 > 0 | exact_balance |
| G2 < 0 且 B2 <= 0 | direction_failure |
| G2 < 0 且 B2 > 0 | curvature_overrun |

不使用经验 epsilon，也不把 exact_balance 合并进正或负 proposal。

## 4. 行自身项与跨行项

令第 i 行对全部 measured queries 的计数变化为 d_i，则：

    d = sum_i d_i
    Cself  = 1/2 * sum_i ||d_i||^2
    Ccross = sum_{i<j} d_i dot d_j
    C = Cself + Ccross

正式输出继续使用二倍整数：

    Cself2  = sum_i ||d_i||^2
    Ccross2 = C2 - Cself2

Cself2 必须非负；Ccross2 可以为正、零或负。不得把负 Ccross 当成无效值，它表示不同记录的查询变化
互相抵消。

对 curvature_overrun proposal，冻结的跨行作用类别为：

| 条件 | 类别 |
|---|---|
| B2 - Cself2 < 0 | self_sufficient_overrun |
| B2 - Cself2 = 0 | cross_breaks_tie |
| B2 - Cself2 > 0 | cross_decisive_overrun |

后两类合并称为 cross_required_overrun：没有实际 Ccross 时，该 proposal 不会严格恶化。
该口径只做代数归因，不声称逐行独立更新在生产中真实发生。

## 5. copy 与 mutation 的顺序归因

对同一 frozen state、donors、参与行和属性复制随机量，生成：

    q_copy = mu=0 的 copy-only query counts
    q_full = mu=0.01 的 production full query counts
    dc     = q_copy - q0
    dm     = q_full - q_copy
    ec     = e0 - dc

顺序 gain 固定为：

    Bcopy2 = 2 * e0 dot dc
    Ccopy2 = dc dot dc
    Gcopy2 = Bcopy2 - Ccopy2

    Bmutation_given_copy2 = 2 * ec dot dm
    Cmutation2            = dm dot dm
    Gmutation_given_copy2 = Bmutation_given_copy2 - Cmutation2

必须逐 proposal 验证：

    Gfull2 = Gcopy2 + Gmutation_given_copy2

同时记录复制与 mutation 的二次交互：

    Ccopy_mutation_interaction2 = 2 * dc dot dm
    Cfull2 = Ccopy2 + Cmutation2 + Ccopy_mutation_interaction2

不得把 Cfull-Ccopy 直接命名为“mutation 的 C”，因为该差值包含复制与 mutation 的有符号交互。

对 full 负 proposal，mutation 来源类别固定为：

| 条件 | 类别 |
|---|---|
| Gcopy2 >= 0 且 Gfull2 < 0 | mutation_created_failure |
| Gcopy2 < 0 且 Gfull2 < 0 | copy_already_failed |

另行完整报告 mutation_rescued_failure、positive_weakened、positive_strengthened、no_gain_change 等
转移表，但这些描述性类别不改变主归因规则。

## 6. 数据、workload 与输入身份

### 6.1 test_300x10

    schema             configs/test_300x10/schema.yaml
    measured queries   configs/test_300x10/measured_50query_30_15_5.json
    marginals          configs/test_300x10/init_marginals.json
    N                  300
    device             numpy
    measured queries   30 x 2-way + 15 x 3-way + 5 x 4-way

身份：

    schema SHA-256      58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792
    queries SHA-256     708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14
    marginals SHA-256   1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4
    query identity      602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1
    target identity     e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d

### 6.2 nltcs

    schema             configs/nltcs/schema.yaml
    measured queries   configs/nltcs/measured_1000query.json
    marginals          configs/nltcs/init_marginals.json
    N                  16181
    device             cuda
    measured queries   479 x 2-way + 522 x 3-way = 1001

身份：

    schema SHA-256      5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4
    queries SHA-256     b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f
    marginals SHA-256   a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e
    query identity      48fd2802ed25efa6b2a0736de2fc8234452001787bb7a07e768c25eb4fad9429
    target identity     f1b7f3b67b4e2f791c69e0b4d49693c9e84f18b004a1f2ece1053514fe05174d

历史文件名虽然写 1000query，正式身份按文件中的 1001 条 ordered queries 使用。

## 7. 来源轨迹

Stage 6A 来源轨迹只使用 Stage 5 冻结的 independent_s0 development reference：

上游证据身份固定为：

    Stage 5 protocol document SHA-256  4d0ffb8ebf77006becea00849eef452174559aea62faeb86dd70c227e3fc7fab
    Stage 5 protocol manifest SHA-256  1d447be0fb0ce9a2c7707abd2e259ed6ea41edbf3780f30426e320b1bad94f1c
    Stage 5 execution commit           a5455ca1574a45acbbbe68abe3100a97e4b976ba
    Stage 5 collection SHA-256         375377849aaec401ec6e2dcd29ed850f180c4204c17ed785f67fba2f8f6c506d
    Stage 5 evaluation SHA-256         8368c58d462a9f4b540f32c0d815faf93c436a7b23fef7fa852cda013a8f94f5
    Stage 5 independent audit SHA-256  c14fb651466e641b4bf75d0b5966aed4b4a2dcd8f588da2cf5862aaa035e34bb

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
    factorized_gibbs_sweeps                 0
    alpha_schedule_mode                     fixed
    fixed_alpha                             16
    residual_self_cooling                   off
    rho_anneal                              off
    stop_on_exact_residual                  true
    inner patience                          P=6 completed natural-work ticks
    output                                  terminal current
    record_transition_clocks                true
    record_natural_work_snapshots           true

在线停止仍为 exact fit、连续六个已完成 natural-work tick 未刷新历史 best、6000 raw rounds/candidates，
优先级与 Stage 5 相同。历史 best 只控制进展停止，不参与 proposal 接受或输出选择。

## 8. 全新 seeds、状态和固定规模

正式来源 seeds 固定为：

    348, 349, 350, 351, 352

这些 seeds 与 Issue #53 既有 313..347 全部分离，不复用 Stage 4 的 333..337 或 Stage 5 的
338..347。正式矩阵为：

    2 datasets x 5 seeds = 10 independent source trajectories

每条来源轨迹固定选五个 current states：

    initial, work_q25, work_q50, work_q75, terminal

initial 与 terminal 精确选取。三个内部状态从已记录 natural-work snapshots 中选择三个互异状态，
使其到 terminal normalized work 的 25%、50%、75% 目标的总绝对误差最小；并列时按更早 snapshot
index 的字典序决胜。不得看到 B/C 结果后移动状态。

每个 frozen state 的 proposal 数：

| dataset | proposals/state | states | seeds | proposal pairs |
|---|---:|---:|---:|---:|
| test_300x10 | 200 | 5 | 5 | 5000 |
| nltcs | 20 | 5 | 5 | 500 |
| total | - | 50 | - | 5500 |

每组 pair 产生 copy-only 与 full 两张测量表，因此共 11000 个表结果。它们都只属于 frozen-state
诊断，不串成新的闭环轨迹。

nltcs 使用 20 而不是 Stage 4 的 4 proposals/state，是因为本阶段以 proposal 为归因单位，并要求每个
seed 在四个 post-initial 状态上形成可检查的负 proposal 分母；这项增加在结果前固定，不得运行后调整。

## 9. frozen proposal 的共同随机条件

每个 dataset/seed/state/proposal index 使用域隔离 SeedSequence 地址生成两个流：

1. donor stream：只用于 donor indices；
2. update stream：copy-only 与 full 从同一初态各自实例化。

两侧必须具有完全相同的：

- frozen current table、q0、e0、fitness 与 donor sampling probabilities；
- donor indices 和 donors；
- fixed trajectory initial_rms direction reference scale；
- per-row/per-attribute direction scores；
- participation rolls；
- 每个属性的 copy rolls；
- mutation Bernoulli rolls。

两侧唯一有意差异是 mutation threshold：

- copy-only：mu=0；
- full：mu=0.01。

full 在命中 mutation 后继续消耗属性和值的随机量；这不允许反向改变已经共同完成的 donor、
participation 或 copy 随机量。每个 pair 必须验证 copy-only 与 full 在 mutation 之前的重放身份。

probe 的 rho=0.01、eta=0.5、tau=2、fixed alpha=16、相对残差 floor=8 与来源轨迹一致。不得根据
当前 loss、state group 或之前 proposal 的结果改变这些值。

## 10. 正式汇总与稳定来源规则

统计独立单位是 source trajectory seed，不把 5500 个 proposal 或参与行伪装成独立 seeds。

primary analysis 固定使用四个 post-initial groups：

    work_q25, work_q50, work_q75, terminal

initial 单独完整报告，作为早期高残差对照，不进入稳定来源门槛。

### 10.1 负 proposal 几何

对每个 dataset/seed 合并四个 primary groups：

- 负 proposal 数必须至少为 10；否则该 dataset 记 insufficient_negative_support；
- 计算 direction_failure 与 curvature_overrun 在全部 G2<0 中的占比；
- seed label 只给严格多数的一类，50% 精确平局记 no_seed_majority。

dataset 支持某个 geometry label 当且仅当：

1. 五个 seeds 都满足最少 10 个负 proposal；
2. 至少 4/5 seeds 对同一 label 为严格多数；
3. 五个 seed-level shares 的等权算术平均中，该 label 也严格超过 50%。

否则为 no_stable_dataset_geometry。

跨数据 shared geometry 只有在 test_300x10 与 nltcs 支持相同 label 时成立。允许的总体结果为：

    invalid_or_incomplete
    insufficient_negative_support
    no_shared_failure_geometry
    shared_direction_failure
    shared_curvature_overrun

### 10.2 curvature 的 self/cross 来源

仅在某 dataset 已支持 curvature_overrun 时评价。每个 seed 至少需要 10 个 curvature_overrun；
否则为 insufficient_curvature_support。

对每个 seed，将 cross_breaks_tie 与 cross_decisive_overrun 合并为 cross_required_overrun，
与 self_sufficient_overrun 做严格多数。dataset 与跨数据稳定规则继续使用 4/5 seeds 加 seed-level
shares 等权平均 strict majority，不修改门槛。

### 10.3 mutation 来源

对全部 full 负 proposal，比较 mutation_created_failure 与 copy_already_failed。最少分母、
seed 严格多数、4/5 seeds 和 pooled strict majority 的规则与 10.1 完全相同。

mutation 的顺序 gain 正负、copy/full 类别转移、mutation rows、覆盖已复制 cell 的数量和
Ccopy_mutation_interaction2 分布全部报告，但不另造可调阈值。

### 10.4 分层报告

必须无选择地报告：

- dataset x seed x state group；
- 五 seed 等权 dataset aggregate；
- initial 与 primary post-initial 分开；
- full、copy-only、mutation-given-copy；
- B2、C2、G2 与 Cself2、Ccross2 的 count、均值、中位数、四分位、最小和最大；
- 所有一级类别、cross 类别、mutation 转移类别的原始 counts；
- changed rows/cells、participating rows、mutated rows和墙钟。

不得在结果出现后新增“更好看”的 state、seed、ratio、截尾规则或条件子组。

## 11. collection、artifact 与 independent audit

正式运行必须分成：

1. state-library collection；
2. frozen proposal collection；
3. result-blind structural audit；
4. frozen evaluator；
5. independent arithmetic audit。

state library 必须保存全部 50 个 current tables、q0、e0、state identities、source trajectory
参数、initial table/RNG/terminal hashes 与自然工作选择清单。

proposal artifact 必须保存或以排他压缩 shard 保存足以独立重建每个 pair 的：

- RNG 地址、donor indices hash 与 update RNG endpoints；
- participation/copy/mutation 身份；
- current-to-copy 与 copy-to-full 的稀疏 edit log；
- dc、dm、full d 的 exact int64 vectors 或等价可独立恢复表示；
- B2/C2/G2、row self/cross、sequential mutation identities；
- copy/full table hashes与所有类别。

正式 collector 只发布逐 proposal 原始量、身份和完整 manifest，不输出最终 shared label。全部 shards
完整且 structural audit 通过后，frozen evaluator 才能读取它们。

independent auditor 不调用 evaluator 的分类函数，必须从 state tables、稀疏 edits 和 query 定义
重新计算 q、row deltas、B2/C2/G2、所有恒等式、类别、seed majorities 与 shared label。

任何缺 shard、重复 proposal 地址、非有限值、int64 溢出风险、恒等式失败、RNG 配对失败、输入漂移、
dirty formal worktree 或无门控身份失败，都得到 invalid_or_incomplete；不得跳过坏行后继续分类。

## 12. 开发、smoke 与正式授权

已经查看过的 Stage 4 state library 只允许做开发接线和性能估计，必须标记：

    formal_result_valid = false
    artifact_role       = development_only

它不能进入 Stage 6A 的 348..352 正式统计。

未来 pipeline smoke 固定使用 seed 9906、两数据、每条来源最多 500 rounds、缩小到 128 records、
每状态 2 proposal pairs，只验证五状态选择、RNG 配对、稀疏 edit 重建、精确恒等式、排他落盘和 audit。
smoke 不产生机制证据。

正式 collector 必须 fail closed：只有 clean implementation commit、本文与 protocol manifest 身份
匹配，并显式传入当前冻结 protocol SHA-256 时才允许实例化 348..352。本文冻结不等于正式运行授权；
运行 smoke 或 formal collection 都必须等用户再次明确同意。

## 13. Stage 6A 之后

Stage 6A 的 shared labels 只决定后续讨论从哪里开始：

- shared_direction_failure：优先讨论采样前方向信号；
- shared_curvature_overrun：优先讨论采样前 B-C 软能量，并结合 self/cross 子标签确定粒度；
- no_shared_failure_geometry：不冻结共享新核，先解释 workload 差异；
- mutation_created_failure 若稳定：任何后续方法都必须显式讨论 mutation 的采样前处理；
- invalid/insufficient：停止，不补 seed、不追加 proposal、不改最少分母。

任何后续 kernel 都必须另立 Stage 6B 结果前协议，并继续满足：

- 只在抽样前连续修改概率；
- 所有有限条件下 0/1 outcome 都保留正概率；
- proposal 生成后无接受、拒绝、回滚、重试、shadow best 或 winner selection；
- Stage 6A 的 348..352 只能作开发依据，不能作为 Stage 6B 正式效果 seeds。
