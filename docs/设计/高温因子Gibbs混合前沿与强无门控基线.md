# 高温因子 Gibbs 混合前沿与强无门控基线

> 状态：正式实验前协议 v2 与工具收口中，尚未运行任何正式结果
> 日期：2026-08-13
> 关联：Issue #49、#10、#14、#24、#27、#30、#43、#44

## 1. 协议修订历史与当前边界

### 1.1 v1 预注册

提交 `7cc66ee` 预注册了 `tau=4/8` 的阶段 A：在无门控独立方向轨迹的初始、中期和
晚期 current state 上，比较 `sweeps=0/8/16/32`，为每个温度选择最小充分 Gibbs
sweeps。随后依次完成了 current-state 快照、外部快照探针、原始条件 logit、条件概率与熵、
以及五类状态库的开发实现。

### 1.2 触发修订的开发 smoke

正式实验开始前，使用 seed 8、12 轮、快照 `0/6/12`、每状态 2 个 proposal 的短流程
验证了状态库、精确传播、汇总、自动分类和独立审计。该 smoke 只用于检查管线，不是正式
效果证据。

短流程中：

```text
tau=4：最大原始条件 |logit|=15.09，零 clip 命中
tau=8：最大原始条件 |logit|=30.18，4/6468 个条件超过 clip=30
```

这说明固定 `clip=30` 的实际随机扫描核与未截断联合 oracle 在少数 `tau=8` 条件上不再是
同一目标，不能继续用原 v1 规则给 `tau=8` 选择 sweeps。与此同时，历史无门控长期温度前沿
只测试过 `tau=0/1/2/4/8`，没有测试 `5/6/7`；“tau=8 最强”只表示它在旧离散网格中最好，
不是所有温度中的全局最优。

### 1.3 v2 修订

由于尚未运行任何正式 Stage A 结果，本协议在正式运行前一次性修订为：

```text
tau 网格：4、5、6、7、8
logit clip：固定 30，不提高、不关闭
sweeps 网格：0、8、16、32，不追加 64
数值资格与混合充分性分开判断，并且逐 tau 独立分类
```

本次修订后不得根据结果追加小数温度、`tau>8`、其他 sweeps、删除失败状态或放宽门槛。
如果整个冻结网格失败，应保留失败结论并另立新协议。

## 2. 整条研究链要回答什么

本研究分成四个相互独立的问题：

1. **独立温度前沿 T：**在当前实现上，`tau=4/5/6/7/8` 的独立方向无门控长期轨迹
   分别表现如何？
2. **数值资格 A0：**每个 tau 在固定 `clip=30` 下是否从未需要裁剪，从而可以与未截断
   联合 oracle 做严格同目标比较？
3. **混合前沿 A1：**对 A0 合格的 tau，`8/16/32` 中最少多少 sweeps 能充分逼近同温度
   联合 oracle？
4. **长期效果 B：**通过 A1 的因子 Gibbs 是否在 1000 轮无门控动力学中优于同 tau 的
   独立方向核，并能否在新 seeds 上确认？

最终目标是冻结一个唯一的小表强无门控基线。它可能是 `independent tau=X`，也可能是
`factor Gibbs tau=X, sweeps=Y`。本协议不预设因子 Gibbs 必须获胜。

选择性软曲率、gate-only 因果对照和 nltcs 迁移均不属于本协议；只有强无门控基线完成新
seeds 确认后，才分别另立协议。

## 3. 冻结输入与共同参数

### 3.1 小表输入

只使用以下公开配置和精确 target：

```text
schema：configs/test_300x10/schema.yaml
queries/target：configs/test_300x10/measured_50query.json
marginals：configs/test_300x10/init_marginals.json
N：300
属性数：10
查询数：50
```

冻结 SHA-256：

```text
schema：58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792
queries：7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88
marginals：1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4
```

不读取真实 CSV，不评价真实联合 TVD，不属于 DP 实验。

### 3.2 共同动力学

除非某阶段另有说明，固定：

```text
tau：4、5、6、7、8
rho=0.01
eta=0.5
mu=0.01（长期轨迹）；冻结 probe mutation=0
donor distance：geometric
donor alpha：完整 1000 轮内按 2→10 调度
normalization：initial_rms
generation acceptance：关闭
主终点：current state，不使用 best 回滚
logit clip：30
设备：NumPy / CPU
```

`clip=30` 是所有正式方法的一部分，用于在 float64 中保留严格双向支持；它不是待搜索的效果
参数。原始 `|logit|>=30` 记为触及数值边界。正式分类不会修改 clip 来迁就某个 tau。

## 4. 阶段 T：独立方向无门控温度前沿

### 4.1 配置

```text
seeds：0、1、2、3、4、5、6、7、8、9
每个 seed 的 tau：4、5、6、7、8
总轨迹数：10 × 5 = 50
rounds：1000
核：独立方向核
factor Gibbs sweeps：0
```

同一 seed 的五条轨迹必须从完全相同的 marginal 初始表和主 RNG 起点出发。首轮温度影响
mask 前得到的正有限 `s0` 必须在五条轨迹间精确一致；之后各轨迹固定自己的该值，不得按状态
重新定标。

### 4.2 报告与用途

完整保留 50 条轨迹，不按结果删 seed。至少报告：

- final current loss；
- rounds 751～1000 的 late-window current loss 均值；
- 全轨迹 current-loss AUC；
- 正/负 proposal 比例及正负 gain 幅度；
- 负方向复制概率、条件熵和改动规模；
- 原始独立方向 logit 的最大绝对值与 `|logit|>=30` 数量；
- 每条轨迹是否跑满 1000 轮及终点表/RNG 哈希。

历史 commit `31afd2a` 的 CUDA 结果只作背景参照；本次固定 commit、NumPy/CPU 的完整重跑是
后续阶段的权威输入。阶段 T 不根据 loss 删除任何 tau，五个 tau 全部进入状态库与 A0。

## 5. 共同 current-state 状态库

### 5.1 状态来源

从阶段 T 的开发 seeds `0/1/2` 五条独立方向轨迹中保存：

- round 0：每个 seed 的共同 marginal 初始 current state，只保留一份；
- round 500：每个 source tau 分别保存 current state；
- round 1000：每个 source tau 分别保存 current state。

每个 seed 共 `1 + 5 × 2 = 11` 个状态，三个 seeds 合计：

```text
3 × 11 = 33 个 seed-state
```

状态族固定为：

```text
initial
mid/source-tau4、5、6、7、8
late/source-tau4、5、6、7、8
```

即 11 个状态族。即使某个 source tau 后来未通过 A0，它产生的状态也不得从共同状态库删除；
这些状态仍用于检查其他 evaluation tau 的跨轨迹稳健性。

### 5.2 状态身份门禁

每个快照记录并重载核对：

- 完整表及 SHA-256；
- seed、source tau、source sweeps、总轮数和 state round；
- current loss 与只作诊断的 best-so-far loss；
- round 对应的 donor alpha；
- 固定 `s0` 及其发现轮次；
- 主 RNG 与 Gibbs RNG 状态哈希；
- `state_kind=current`。

同 seed 五条来源轨迹的初始表、初始 loss、主 RNG、正有限 `s0` 和 alpha 日程必须精确对齐；
所有来源轨迹必须跑满 1000 轮。任一身份门禁失败，正式 A0/A1 不开始。

## 6. 阶段 A0：逐 tau 数值资格

### 6.1 冻结条件

在每个 33 个状态上生成 200 组地址化、可重放的冻结 proposal 条件。固定并在所有
evaluation tau 间共享：

- donor；
- recipient 参与行；
- 活跃属性及完整 mask 枚举；
- 稀疏因子能量；
- 基础 Bernoulli 初始核；
- 后续 A1 使用的随机 tape。

A0 只构建因子并检查原始条件 logit，不根据 proposal gain、loss 或某个 tau 的结果更换状态、
donor 或条件。

### 6.2 独立资格规则

每个 evaluation tau 单独得到资格状态。只有同时满足以下条件才记为
`eligible_for_mixing`：

1. 阶段 T 中该 tau 的 10 条独立来源轨迹全部跑满且所有活跃独立方向原始
   `|logit|<30`；
2. 在全部 33 个状态、每状态 200 个冻结 proposal 条件上，该 evaluation tau 的所有因子
   Gibbs 原始条件 `|logit|<30`；
3. 所有 logit、条件概率、因子能量和 `s0` 有限；
4. 所有条件概率在 float64 中严格位于 `(0,1)`。

只要出现一次 `|logit|>=30`，该 tau 记为 `out_of_numerical_domain`，保留完整命中数量、
最大值、seed-state 和条件身份，但不进入 A1，不提供最小 sweeps。一个 tau 失败不影响其他
tau；全体失败则阶段 A 在 A0 结束。

不得根据 A0 结果提高或关闭 clip、增加新 tau、将失败 tau 四舍五入为其他 tau，或删除造成
命中的状态与 proposal。

## 7. 阶段 A1：最小充分 Gibbs sweeps

### 7.1 网格与耦合

只对 A0 合格的 evaluation tau，在完全相同的 33 个状态和冻结条件上检查：

```text
sweeps：0、8、16、32
候选 sweeps：8、16、32
joint oracle：同 tau 的未截断联合 mask 分布
最大活跃属性数：12
最大查询因子阶数：3
```

由于 A0 要求所有相关原始条件 `|logit|<30`，A1 中 `clip=30` 实际不改变条件核，因此随机
扫描 Gibbs 与未截断 joint oracle 具有同一目标。若重放时出现任何 A0 未记录的 clip 命中，
身份或确定性门禁失败，A1 停止。

对每个固定 seed-state-proposal：

- `sweeps=0` 是同温度独立方向分布，只用于测量起始差距；
- 每个 sweep 是 `k` 个随机、有放回的坐标微步，`k` 为该行 active bit 数；
- `8/16/32` 使用同一条累计精确传播路径；
- donor、参与行、基础核和 Gumbel tape 在所有配置间共享；
- 随机源地址化，增加配置不得错位其他 proposal；
- TVD 使用有限状态精确传播，不用有限 Monte Carlo 估计。

### 7.2 正确性门禁

每个 tau 独立检查：

1. 稀疏因子能量与完整 mask oracle 最大误差不超过 `1e-10`；
2. one-hot 因子方向与独立方向最大误差不超过 `1e-10`；
3. 所有传播分布有限、非负、归一，概率和误差不超过 `1e-12`；
4. TVD 随累计微步的最大反向变化不超过 `1e-12`；
5. 所有 33×200 个 proposal 条件完整且每个必需汇总组都有活跃 recipient 行；
6. 正式 `random_scan_gibbs_mask` 在冻结小因子上与精确传播语义对拍通过，并单独记录生产
   采样墙钟和微步数，不能用精确传播耗时冒充生产成本。

任一正确性门禁失败，只停止相应 tau 的分类；失败原因和原始数据必须保留。

### 7.3 混合指标与汇总

主要指标：

- 到同温度 joint oracle 的 TVD；
- `KL(candidate || oracle)`；
- 相对 0 sweep 的期望方向差距恢复比例；
- mask 熵、条件熵、到参考 Bernoulli 核的 KL；
- 负方向完整 mask 质量及 oracle 值。

proposal gain、线性收益、二次惩罚、改变单元格/记录数和正负 gain 比例只作诊断，不参与
sweeps 选择。

至少保留：

1. 每个 seed-state；
2. initial 加 10 个 mid/late source 状态族；
3. initial / mid / late 三阶段；
4. 五个 source tau 分层；
5. 全部状态全局汇总。

kernel 指标按活跃 recipient 行加权；不得把 recipient 行或 proposal 当作独立 seed 做显著性
推断。

### 7.4 最小 sweeps 规则

对某个合格 tau，候选必须在以下 12 个必需汇总中同时通过：

```text
global
initial
mid/source-tau4、5、6、7、8
late/source-tau4、5、6、7、8
```

每个汇总均要求：

```text
TVD <= 0.05
期望方向差距恢复率 >= 80%
```

在 `8、16、32` 中选择第一个通过的最小值。32 仍失败则记为
`not_sufficient_through_32`，不追加 64。阶段 A 不根据 proposal loss 在不同 tau 间选冠军；
所有得到最小充分 sweeps 的 tau 都进入阶段 B。

## 8. 阶段 B：长期无门控效果与唯一候选

### 8.1 开发对照

阶段 B 使用与阶段 T/A 不重叠的开发 seeds：

```text
seeds：100～109
rounds：1000
设备：NumPy / CPU
```

固定运行：

- `independent tau=4/5/6/7/8, sweeps=0` 五个独立候选；
- 对每个通过 A1 的 tau，运行 `factor Gibbs tau=X, sweeps=Y`，其中 Y 是 A1 自动选出的
  最小充分值。

同 seed 各配置从相同初始表和主 RNG 起点出发；Gibbs 只增加独立 Gibbs RNG，不得错位
donor、独立初始 mask 或 mutation 主随机流。所有 proposal 无条件成为下一 current state。

主选择指标是每个 seed 在 rounds 751～1000 的平均 current loss，再在 10 个 seeds 上取均值；
越低越好。secondary 指标包括 final current loss、全轨迹 AUC、均值/中位数/最差 seed、
正负 gain 幅度、熵、反向概率、改动规模和墙钟。

### 8.2 数值与自身状态复查

所有长期轨迹继续记录原始 logit 和 clip 命中。任何配置只要在 Stage B 实际轨迹中出现一次
`|logit|>=30`，结果仍保留为实际 clipped 算法的描述，但不得晋级为本协议认证基线。

factor 配置先按冻结的长期指标和同温度比较规则确定一个预备冠军 `G0`，不读取确认 seeds。
只对 `G0` 自己访问的 round `0/500/1000` current states，使用冻结的 A1 规则重做数值资格
与混合检查。自身状态复查的必需汇总固定为 `global/initial/mid/late` 四组，每组继续要求
`TVD<=0.05` 且恢复率 `>=80%`；数值、概率、传播和生产采样器正确性门禁不变。该检查失败则
Stage B 记为未产生认证 factor 候选，不改 sweeps、不改 clip，也不递补 factor 第二名。

### 8.3 冻结唯一候选

先确定 Stage B 的最强独立候选 `I*`：五个 independent tau 中，阶段 T/A0 与 Stage B
长期轨迹均零 clip 命中者按主指标取最低值；主指标精确相等时，依次按 late-window 中位数、
AUC 均值、final-current 均值、较少 sweeps、较低 tau 决胜。若五个独立配置均触及边界，
本协议不产生认证基线。

factor 配置要进入预备冠军 `G0` 排名，必须：

1. A0/A1 通过；
2. Stage B 全轨迹零 clip 命中；
3. 相对同 tau independent 的 10-seed 配对 late-window 差值均值和中位数均小于 0，且至少
   6/10 seeds 改善；
4. 完整保留全部 10 个 seeds。

满足条件的 factor 配置中，以主指标最低者为 `G0`；若主指标精确相等，依次按 late-window
中位数、AUC 均值、final-current 均值、较少 sweeps、较低 tau 决胜。`G0` 通过自身访问状态
复查后才成为 `G*`，否则本阶段没有 `G*`，且不递补。

若没有合格 `G*`，唯一候选为 `I*`；若存在 `G*`，则在 `G*` 与 `I*` 中选择主指标更低者。
二者主指标精确相等时，仍按 late-window 中位数、AUC 均值、final-current 均值、较少 sweeps、
较低 tau 决胜。

该规则只产生一个唯一候选；冻结后不得因确认结果不好而换用第二名。

## 9. 新 seeds 最终确认

最终确认只在唯一候选冻结后读取：

```text
seeds：110～119
rounds：1000
```

为避免温度选择在新 seeds 上失去参照，确认阶段固定重跑五个 independent tau。若唯一候选是
factor Gibbs，再额外运行该唯一 factor 配置；其同 tau independent 已包含在五个独立臂中。

确认成功要求：

1. 冻结候选在保持零 clip 的确认臂中仍具有最低的 10-seed 平均 late-window current loss；
2. 若候选为 factor Gibbs，它相对同 tau independent 和 Stage B 冻结的 `I*` 均需保持配对
   均值与中位数小于 0、至少 6/10 seeds 改善，并且 95% 配对 t 区间上界小于 0；若两者是
   同一 independent 配置，只计算一次；
3. 若候选为 independent 且不是 `tau=8`，它相对旧网格 incumbent `independent tau=8`
   的配对均值与中位数小于 0、至少 6/10 seeds 改善，且 95% 配对 t 区间上界小于 0；若候选
   就是 `tau=8`，只要求它在合格确认臂中仍为最低；
4. 候选的正式身份、概率、clip 和自身状态混合复查门禁通过；
5. 不删除失败 seed，不用 best checkpoint 替代 current state。

确认失败即记录未确认，不在 seeds `110～119` 上重新选 tau、递补 runner-up、调 sweeps 或修改
规则。成功后才把该配置称为当前 `test_300x10` 强无门控基线；跨 workload 泛化仍需 nltcs
独立验证。

## 10. 正式身份、审计与停止规则

正式运行前必须：

- Python 3.11+ 隔离环境和依赖版本完整记录；
- 代码、完整协议、runner、audit 和测试先形成一个干净的 tools-freeze commit；
- 使用非正式 seed 99、12 轮、快照 `0/6/12`、11 状态、每状态 2 proposals 完成五温度
  短 smoke；smoke 的数值合格与否不作为正式效果证据；
- 只在 clean worktree 的固定 tools-freeze commit 上依次运行全部正式阶段；阶段之间不得修改
  产生结果的代码、协议或参数；
- 输入哈希、协议身份、commit、命令、环境、状态身份和原始逐状态结果完整保存；
- 输出排他原子创建，不覆盖旧结果；
- 独立 audit 不调用主 runner 的汇总/分类实现，重新核对身份、重算汇总与自动决策；
- 每个阶段在 Issue #49 留下 commit、命令、报告 SHA-256、audit SHA-256 和停止/继续决定。

如果发现影响正式结果的代码 bug：旧结果立即标记无效，先修复并重新评审，然后从受影响的最早
阶段完整重跑；不得把修复前后的 run 拼接成一批。

停止规则：

```text
A0 全部 tau 不合格：整条协议停止，不运行 A1/B
A0 至少一个 tau 合格但 A1 无 factor：Stage B 只运行五个 independent，不运行 factor
Stage B 无 factor 成功：允许 independent I* 进入确认
最终确认失败：记录失败，不试 runner-up
```

## 11. 单 PR 的提交与运行边界

Issue #49 只使用一个 PR，按以下顺序形成证据链：

1. 先提交 tools-freeze commit，包含：

   - 本 v2 预注册与 Issue #49 修订；
   - 五温度独立前沿、33 状态库、A0/A1、Stage B 和确认 runner；
   - 原始 logit、概率、混合、成本和轨迹诊断；
   - 自动决策、排他落盘和独立 audit；
   - 全部专项测试、五温度 seed-99 smoke 和旧功能回归。

2. 在该 commit 和 clean worktree 上依次运行正式 T/A、Stage B 与最终确认；每个阶段先完成
   audit，并在 Issue #49 记录运行 commit、命令、报告哈希、审计哈希与停止/继续决定。
3. 再用后续只增加正式结果、审计产物、图表和结论的 commit 收口；tools-freeze commit 与这些
   结果 commit 统一进入同一个 PR。

结果 commit 不得修改产生结果的算法、参数网格、seeds、门槛或选择规则。若正式运行暴露代码
bug，必须先让受影响结果失效，修复并重新冻结工具，然后从最早受影响阶段完整重跑；不得一边
保留旧结果一边调整算法解释。

## 12. 当前实现检查点与待办

当前分支已经完成：

```text
d3fefe3  无门控 current-state 快照
bf21eed  mixing 探针读取外部快照并固定 s0/alpha
7a53d1e  原始条件 logit 与 clip 诊断
7f45969  条件概率、最小双向概率与条件熵
7868bcc  v1 的 tau=4/8 五类状态库
```

旧 v1 五状态短流程报告 SHA-256：

```text
4f39f713eb55e6fc6e3463a2c45daf6d1d7b8562e2021d589a5b9087f3f66cfb
```

它继续作为协议修订触发记录保留，不会改写为 v2 结果。

当前工作树已经实现五温度 Stage T、33 状态库、逐 tau A0/A1、Stage B 唯一候选、最终确认，
以及三个阶段各自的独立 auditor、排他原子输出与身份/哈希门禁。seed-99 的端到端 v2 smoke 和
85 项相关测试已通过；这些 smoke 只验证管线，不能作为效果结论。

接下来的收口顺序：

1. 完成全部 diff、CLI、哈希边界、停止条件和单 PR 文档审查；
2. 运行全仓库 pytest、Python 编译检查与 `git diff --check`；
3. 建立唯一 tools-freeze commit，在此之前不运行正式 seeds；
4. 在该固定 commit 上先运行并审计正式 T/A，再严格按停止规则决定 Stage B 与最终确认；
5. 只把正式产物、哈希、图表与结论作为后续 commit 加入同一个 PR。
