# Issue #53：A+B+C 三条件自然工作时钟组合停止设计稿

> 状态：用户已确认 A 只表示“进入测量噪声允许范围”，不再把固定 RMSE/MAX 工程容差作为正式 A；
> `累计参与记录数 / N` 继续作为 B 的自然工作时钟。B 已从“证明优化停滞”改为“计算—质量权衡下
> 的早停”，终止原因统一为 `early_stopped`。历史 best loss 可以作为 B 的只读 progress 信号，但
> 不得影响 proposal、current 状态、残差、回滚或输出选表。A、B、C 任一出口都只返回触发时的
> terminal current table/current loss；其中 A、B 是正常完成路径，C 只是外部资源强制截断，不能
> 表示质量合格或收敛。方法的准确身份是“无门控残差引导扩散核 + 状态相关早停器”。
> 固定六 tick 只保留为已知六条轨迹上的 development 基线，不是最终耐心值。本文仍是设计记录，
> 不是结果前协议；当前已完成纯状态机与 `run_evolution` 的 opt-in 接线，但不冻结早停耐心、质量
> 损失容限、C 的数值或实验 seed，也未授权正式实验。

## 1. 这一版只解决什么

一次内层生成的输入是固定 schema、固定 measured workload 及其已发布答案。生成器执行无门控、
残差引导的随机扩散更新；停止层只读观察已生成的状态，自动决定何时结束本次内层并返回一张合成表。

真正目标是：

```text
不为每个数据集人工试一个固定轮数；
结果已经足够拟合时及时结束；
尚未拟合达标时，允许按统一的计算—质量取舍提前结束；
其余异常情况最终由统一资源护栏兜底。
```

本稿不研究：

- 外层查询选择、私有测量、加噪、sigma、accountant 或隐私预算；
- rho、alpha、residual geometry、eta、mu 或 Gibbs 配置选优；
- 把 true L1、held-out、支持集或多样性反馈给在线停止与选点；这些量只允许在输出身份固定后用于
  离线验收；
- 马尔可夫链严格平稳、混合时间或全局最优证明；
- 新人工 family、正式 seed、实验次数或结果门禁。

## 2. 三个条件的职责必须分开

| 条件 | 回答的问题 | 结束含义 |
|---|---|---|
| A：拟合/噪声一致范围 | 当前这张表是否已经达到预定拟合精度？ | 正常质量完成 |
| B：早停 | 尚未达到 A 时，是否按事先冻结的计算—质量取舍提前结束？ | 近似输出，不表示拟合达标或停滞获证 |
| C：外部资源上限 | A、B 都没有结束时，是否必须停止继续消耗计算？ | 强制截断，不是正常质量完成 |

三种终止都必须物化触发时的 terminal current 表，但必须保留不同原因。不能把 B 或 C 写成 A，也
不能因为它们仍能返回表就把它们表述为拟合质量通过。B 主动接受有限的潜在后续改善来节省计算；
它不再承担“未来不会改善”或“链已经停滞”的证明责任。C 只说明外部资源已经耗尽；C 返回的数据
可以被调用方接收，但该次运行必须标为 resource-limited，不能计入 A/B 正常完成。

终止原因保持为三个主类：

```text
fit_target_reached      # A
early_stopped           # B
resource_cap_reached    # C；具体来源另记为 candidate_budget 或 max_rounds
```

所有 measured residual 恰好为零时自然满足 A，不再另设一个停止条件或决策分支；如有需要，只作为
诊断字段记录，不改变 `fit_target_reached` 的分类。

## 3. 信息边界与无门控身份

### 3.1 停止层可以读取

```text
当前 checkpoint 的 measured-query count-error vector
由同一向量得到的内部 squared loss
历史 best loss 与 best checkpoint 身份
本轮已应用 proposal 的 participating row count
累计 candidate evaluations / Gibbs microsteps / raw rounds
```

### 3.2 停止层不得读取

```text
真实数据答案、reference table、true/normalized evaluation L1
held-out、支持集、多样性或下游任务指标
dataset 名到阈值/耐心值的映射
未来尚未生成的轨迹尾部
隐私预算或 accountant
```

未来带噪阶段允许外层把已经根据发布噪声换算好的数值拟合阈值传给 A；内层不直接读取 sigma，也不
自行访问未发布信息。

### 3.3 无门控与统一 terminal-current 输出

每个 proposal 一旦由生成核产生，就正常成为下一张 current table。普通有限 loss 上升不能触发
拒绝、回滚、缩步重试或立即停止。停止器不改变 proposal、RNG 或已经发生的状态前缀。

历史 best 只按内部 squared loss 排序：

```text
loss 更低       → 更新 best
loss 相同       → 保留更早 checkpoint
current loss 上升 → 正常继续，不改变 best
```

best 只是只读进度诊断，可以驱动 B 的耐心计时器，但不能影响 proposal、下一张 current 表、残差、
拒绝、回滚或输出选表。A、B 或 C 触发时统一固定为：

```text
selected_table = terminal_current_table
selected_loss = terminal_current_loss
selected_state = terminal_state
```

即使历史上存在 loss 更低的表，也不得在 B 或 C 之后回头选它，更不得等待 current 再次回到 best
附近才停止。后两种做法都会变成有利状态择时。算法身份准确写为：

```text
无门控残差引导扩散核 + 状态相关早停器 + 终止状态输出
```

## 4. 保留的自然工作时钟

原始 round 随 rho 改变：rho=1 与 rho=0.01 的一轮不是相同更新机会。本稿保留已经实现并审计过的
applied participating-row 时钟：

```text
cumulative_participating_rows
  = 截至当前状态，所有已应用 proposal 的参与记录数总和

normalized_work
  = cumulative_participating_rows / N

work_tick
  = floor(normalized_work)
```

`work_tick` 每增加 1，表示累计获得约一次等效全表记录参与机会。边界使用第一个真实 post-round
状态使累计值越过整数，不插值、不伪造恰好位于整数处的表。单轮 participating rows 不超过 N，
所以一个真实状态最多新增一个 work tick。

这一时钟自动吸收 rho 的轮数差异：

```text
rho=1     约 1 raw round / work tick
rho=0.25  约 4 raw rounds / work tick
rho=0.01  约 100 raw rounds / work tick
```

它不是墙钟成本的完整替代。factor Gibbs 的 factor build、microsteps 和采样成本，及每轮固定的查询/
fitness/donor 计算仍需另报。相同 normalized work 只表示记录更新机会可比，不表示两种核墙钟相同。

为了防止 participating rows 长期为零但每轮仍消耗计算，`candidate_budget` 或 `max_rounds_guard` 继续
作为 C 的独立故障保险；它们不能冒充 work tick、拟合达标或优化停滞。

## 5. 条件 A：进入测量噪声允许范围

A 只保留一个简单语义：当前合成表对已测量查询的残差已经进入外层测量噪声允许的范围。它必须在
S0 和每个真实 post-round current table 上立即检查；残差、判断结果与返回表必须来自同一个 current
checkpoint。首次达到时立即结束并返回该表。

当前生成研究没有加入测量噪声，噪声为 0，因此允许范围也为 0：只有所有 measured residual 都为零
时才会触发 A。它仍统一记作 `fit_target_reached`，不再另造 exact 停止分支。当前阶段 A 可能很少触发，
不能为了让它更容易触发而人为发明 floor。

未来加入外层私有测量后，由外层把根据已发布噪声事先换算好的容许范围传给 A；内层不读取真实答案、
隐私预算或 accountant，也不自己估计 sigma。A 的同-checkpoint 契约与返回语义保持不变。

此前 `query-count RMSE<=1 AND MAX<=2` 的纯接口和正式负结果保留为研究证据，但该固定工程容差不再
作为正式 A，也不接入组合停止流程。

## 6. 条件 B：基于自然 work tick 的早停

B 只在 A 尚未触发时工作。它是节省计算的近似出口，不是收敛、平稳或未来无改善判定。在线仍不能
使用 true L1 或未来轨迹，只能在每个 work tick 结算刚完成的自然工作区间是否曾刷新历史 best：

```text
本 tick 内出现过严格更低 best loss → progress=true，连续无进展计数清零
本 tick 内 best loss 完全未刷新      → progress=false，连续无进展计数加一
```

当前精确整数 target、无查询权重阶段，loss 是整数或半整数，任意严格 best 改善至少为 0.5。因此当前
不需要另设按数据集变化的浮点 min-change 百分比。带噪 target 可能为非整数，这一离散性质将失效；
未来 B 的 meaningful-improvement 口径必须另立设计，不能静默复用。

早停发生后返回停止当下最后一张 current table 及其 current loss，并标记：

```text
termination_reason = early_stopped
fit_target_reached = false
```

停止后的完整参考轨迹若出现更低 loss，只形成“早停后续变化”，不再自动证明规则失败。规则是否
可接受，不能由某个任意 C 下的单一 terminal 终点决定；应固定 B 的正式输出后，在仅供研究的影子
续跑中观察增加不同工作量还能带来多少持续收益。历史 best 可并列报告，但不能参与任何输出选表。

### 6.1 已尝试的自适应耐心候选

固定 3+3 已由旧反事实矩阵否决。用户将 A 恢复为噪声范围后，当前零噪声 A 不再能用 RMSE/MAX
提前覆盖旧反例，因此不能原样恢复 3+3。随后只作为 development 候选尝试了以下简单自适应规则：

```text
base_gap = 3 work ticks
longest_gap = max(base_gap, 历史相邻 progress tick 的最大间隔)
patience = 2 * longest_gap

当前 tick 内刷新 best → 更新 longest_gap，idle_ticks 清零
当前 tick 内未刷新 best → idle_ticks 加一
idle_ticks >= patience   → early_stopped
```

初始 best 视作 work tick 0 的参考点；同一真实状态先更新 best，再结算 tick，因此停止边界恰好出现
新 best 时不会误停。该规则想表达“至少等 6 tick；若本次运行历史上曾经较慢才进步，就自动延长等待”，
不使用 dataset 名、raw round 或 loss 百分比。

### 6.2 已知六条 development 轨迹回放：候选失败

2026-08-17 在固定 3+3 已使用过的六条完整人工轨迹上，只作已知反例回放。因为规则是在看过这些
轨迹后提出的，该回放只能检查已知漏洞，不能提供新候选的独立通过证据。生成器仍完整运行，规则仅
事后读取 loss 和 applied participating rows；未写输出文件、未读取真实数据、未使用 GPU。

| seed | rho | 原因 | stop state | stop work | stop best | tail best | patience |
|---:|---:|---|---:|---:|---:|---:|---:|
| 20260816 | 1.0 | B | 18 | 18.0000 | 3.0 | 2.0 | 12 |
| 20260817 | 1.0 | B | 22 | 22.0000 | 3.0 | 2.0 | 8 |
| 20260818 | 1.0 | B | 8 | 8.0000 | 3.0 | 1.5 | 6 |
| 20260816 | 0.25 | B | 27 | 7.0625 | 3.0 | 1.0 | 6 |
| 20260817 | 0.25 | B | 64 | 16.0000 | 0.5 | 0.5 | 6 |
| 20260818 | 0.25 | A | 11 | 2.8125 | 0.0 | 0.0 | — |

五条由 B 停止的轨迹中，四条在尾部出现严格更低 best；它们正是固定 3+3 已发现的四个反例。
自适应规则只把前两条的停止从 work 12/20 推迟到 18/22，没有解决“下一次改善可能比历史任何一次
间隔更晚”的根本漏洞。因此该候选不能证明停滞，也不以该身份实现或接入。旧结果没有回答它作为
早停器的质量—成本取舍是否可接受；该新问题必须先定义 regret 口径，再作为新候选重新立项，不能
事后把旧负结果改写为通过。

保留下来的结论是：自然 work tick 与 best-so-far 观察口径仍合理；固定 3+3 和上述最长间隔两倍
均不能用作停滞证明。当前尚未选择任何早停耐心值或质量损失容限。

### 6.3 固定六 tick 的 terminal-current 早停开发诊断

用户确认 B 可以用历史 best 刷新作为只读 progress 信号，但 B 输出必须是停止当下的 current 表。
因此在同一六条已知轨迹上，以固定六个无 progress tick 作为唯一 development 基线，重新物化停止
terminal 表；没有比较其他 patience，也没有把已知轨迹当独立验证。

五条由 B 早停、一条先由零噪声 A 精确命中。B 的汇总结果为：平均节省 68.47% normalized work；
terminal loss 和 L1 均为 1 条优于、4 条差于完整参考终点。更关键的是 5/5 B 停止状态的 current loss
都高于当时历史 best，current-minus-best loss 平均为 3.8、中位为 4.0；最明显一条 best=3、最终却以
current=12 输出。

这不按旧“任何尾部 improvement 都失败”的苛刻口径否决早停，但说明 best-stagnation 只回答“最近
是否产生新低”，而 terminal current 本身仍会随机波动。该风险不能靠返回 best 或等待一个有利
current 状态解决，只能通过结果前冻结的质量—计算验收口径评价。完整结果、逐条指标和复现入口位于：
`docs/实验结果/Issue53_TerminalCurrent早停开发诊断.md`。

### 6.4 B 后相对继续收益曲线 development 诊断

取消单一参考终点后，已在相同六条已知轨迹上实现相对检查点诊断。令固定六 tick development 基线在
`tau` 触发，只对五条 B 轨迹观察 `tau+6`、`tau+12`、`tau+24` 的第一个真实 current 状态；B 在
`tau` 的正式输出先冻结，影子续跑不改变该身份。缺失检查点不使用轨迹末尾补齐。

结果呈明显非单调：`+6` 时 1/5 更好、4/5 更差；`+12` 时 0/5 更好、1/5 相同、4/5 更差；`+24`
只有 3/5 可观察，三条都更好，另两条右删失。loss 与离线 L1 的方向计数一致。该结果不能支持挑选
`+24`，因为可观察样本既小又受旧 horizon 截断；它说明选择任意一个远端 terminal 都可能给出不同
甚至相反结论，验证应使用事前冻结的多检查点曲线和右删失规则。

该步只检查协议和实现，不验证 P=6。完整数值、逐条表、身份核对与复现命令仍位于上述结果文档。

## 7. 条件 C：外部资源强制截断

旧设计的候选值为：

```text
max_normalized_work = 20
```

正式 RMSE+max 矩阵已经证明 work=20 不是所有 workload 的拟合或稳定点。因此 20 不能再作为默认
质量终点、B 的参考终点或收敛证据。

C 不是由 loss、数据集名称或观测结果推导的科学参数，而是调用方在运行前根据可用计算资源传入的
外部上限，例如最大 normalized work、candidate evaluations 或墙钟。当前不存在一个理论推出的
通用最佳数值，本文不重新选择 20，也不硬编码新的默认 C。

C 触发时只返回最后一个已经应用 proposal 后的 terminal current 表，并标记
`resource_cap_reached`、`inner_complete=false`。这表示该次运行被资源截断，不表示 A 或 B 成立。
`candidate_budget` 与 `max_rounds_guard` 同理。正式研究中若 C 截断了所需观察范围，该轨迹只能记为
right-censored/证据不足；不能把 C 的 terminal 当成真值，也不能据此宣称 B 通过。

## 8. 单一状态的固定处理顺序

### 8.1 初始表 S0

```text
1. 用 S0 的同一 count-error vector 计算 fit assessment 和 squared loss；
2. 将 S0 建立为初始 best；
3. 若进入噪声允许范围，按 A/fit_target_reached 返回 S0；
4. 否则进入生成，累计工作量仍为 0。
```

### 8.2 每个已经应用的 post-round 状态

先完整记录事实，再按优先级作决定：

```text
1. 对当前表只评价一次 measured-query answer/count-error；
2. 从同一向量得到 current loss 与 A 的 fit assessment；
3. current loss 严格更低时更新历史 best 及其表/状态/RNG 身份；
4. 累加本轮 applied participating rows，结算是否新增 work tick；
5. 若新增 tick，结算该 tick 是否出现过 best improvement；
6. 按 fit_target_reached > early_stopped > resource_cap_reached
   的顺序选择终止原因；C 的具体来源另作诊断，不另造质量类别。
```

这样可以同时保证：

- 终止状态的 applied work 不丢失；
- A 在同一状态与 B/C 同时满足时优先；
- 第六个空 tick 边界若恰好刷新 best，不会误报停滞；
- 资源边界状态先获得一次真实 fit 与 best 评价，不被截成伪中间表；
- 停止观察不会额外评价查询或消费 RNG。

## 9. 返回表与诊断语义

| 终止原因 | 返回表 | fit_target_reached | inner_complete |
|---|---|---:|---:|
| `fit_target_reached` | 第一张达标 current 表 | true | true |
| `early_stopped` | 停止当下的 terminal current 表 | false | true |
| `resource_cap_reached`（来源为 candidate budget 或 max rounds） | 截断当下的 terminal current 表 | false | false |

`inner_complete=true` 只表示本轮内层按 A 或 B 的定义正常返回。对 B，它不表示拟合质量、真实 L1、
优化停滞、全局最优或理论收敛通过。C 的 `inner_complete=false` 明确表示资源截断。调用方仍可取得
该张 terminal current 表，但不能把它混入正常完成结果。

至少保留：

```text
selected table/state/round/work/loss identity
terminal table/state/round/work/loss identity
termination_reason / fit_target_reached / inner_complete
current fit assessment（仅 A 或终端诊断）
best-loss history at work ticks
cumulative participating rows / completed work ticks
candidate evaluations / Gibbs microsteps / elapsed time
```

true evaluation L1 只能在 selected table 已固定后离线计算，不能反向改变选点或分类。

对 A、B、C，selected identity 都必须与各自 terminal identity 完全相同；best identity 只能是诊断
字段。区别只在停止原因和完成状态，不在于事后选择不同输出表。

## 10. A 简化后，旧 3+3 反例重新直接约束 B

此前只有把固定 RMSE/MAX 当作 A 时，四条 3+3 反例才会在 B 前被 A 覆盖。用户现已取消该工程容差，
恢复“只在测量噪声范围内停止”的 A；当前零噪声下，除完全命中外，旧四条都仍需由 B/C 处理。

因此旧 3+3 负结果现在直接适用：固定六个无进展 tick 不能证明停滞。6.2 的回放进一步表明，仅用
过去 best 刷新间隔放大耐心也不能预测未来更迟的改善。不得再把这两种候选解释为收敛或停滞验证
通过；若以后把其中之一作为早停候选，必须使用新的质量—成本问题、名称、协议和未见验证轨迹。

## 11. B 是主要实践停止方式，不再要求先证明收敛

若只保留 A+C，当前零噪声 A 很难触发，大多数运行就只能一直走到外部 C；而 terminal current 随
运行会波动，增加轮数不保证质量单调改善。因此 A+C 不是更可靠的质量方案，只是一个结果依赖资源
上限的固定预算方案。

B 现在明确承担大多数运行的正常实践停止：连续 `P` 个自然 work tick 没有历史 best 刷新就立即
`early_stopped`，输出当前 terminal 表。它主动用可能错过的未来改善换计算成本，不需要先证明未来
永不改善，也不需要在停止后增加 terminal-readiness、有利状态等待或回到 best。

当前统一实现候选使用：

```text
P = 6 work ticks（可配置的 development 默认值）
A > B > C
```

这里的 6 不是理论收敛常数或正式生产结论，但可以先把简单机制正确实现并接线；不再把复杂的质量—
成本验收协议当作实现 B 的前置障碍。已完成的 `tau+P/2P/4P` continuation 曲线只说明 current 后续
变化非单调，提醒正式评价不能挑任意 C 终点；这些影子状态不会进入在线运行逻辑。

未来实验再按结果前固定的未见 seed/family 方案，报告 A/B/C 触发比例、terminal current loss、
离线 true L1、normalized work、candidate evaluations 与墙钟。B 不需要支配每一个更晚 checkpoint；
评价问题只是它是否提供了合理的整体计算—质量取舍。正式协议仍需在看正式结果前冻结，但不阻止
当前完成纯状态机和接线。

## 12. 当前暂停点

已经确认：

1. A 只表示进入测量噪声允许范围；当前噪声为 0 时只有 residual 全零才能触发；
2. `累计 applied participating rows / N` 的整数增长继续作为 B 的自然 work tick；
3. current loss 普通上升不拒绝状态；B 结束时返回停止当下最后一张 current table/current loss，
   不返回历史 best；
4. B 可由历史 best 是否刷新驱动计时，但到达早停边界后立即输出 current；不再等待 current 回到
   best 附近，不回滚，也不增加 terminal-readiness 输出门控；
5. C 只由外部资源限制触发，返回 terminal current 并标记 `inner_complete=false`；C 不是质量条件，
   也不能作为 B 的单一参考终点；
6. 方法身份是“无门控残差引导扩散核 + 状态相关早停器”，不声称整个算法没有状态相关停止；
7. 固定六 tick 现作为可配置的 development 默认值，不是最终生产耐心或收敛结论。

独立状态机已新增于 `src/table_diffevo/inner_early_stopping.py`。它与用于保存旧 3+3 负证据的
`inner_stopping.py` 隔离；新实现只观察 current loss、best 刷新、参与行时钟和调用方传入的外部 C
标志。终止优先级固定 A>B>C，A/B/C 的 terminal output state/loss 都等于触发时 current；A/B 的
`inner_complete=true`，C 为 false。C 不在配置里硬编码数值。

纯边界测试位于 `tests/test_inner_early_stopping.py`，覆盖 P=6、可配置 P、自然时钟、边界刷新 best、
current 高于 best 仍按 terminal 输出、A/B/C 同时触发优先级、初始/外部 C、非法输入与生命周期。

状态机现已通过 opt-in 参数 `inner_early_stopping_patience_ticks` 接入 `run_evolution`：默认 `None`
完整保留 legacy 历史-best 主返回；传入正整数后，A/B/C 的主返回与 `final_table` 都固定为 terminal
current。C 当前由 `candidate_budget` 或 `n_rounds` 触发，统一使用 `resource_cap_reached`，具体来源只
作诊断。新模式 fail-closed 要求 `tol=+inf` 且 `max_retries=0`，避免出现“名义无门控、实际拒绝
上升 proposal”的配置；状态观察只发生在 proposal 已成为 current 之后，不额外消费 RNG。

接线后又在相同六条已知人工轨迹上做了在线/离线一致性审计：在线 5 条由 B 停止、1 条由 A 停止、
0 条由 C 截断，termination reason、stop state/work、terminal 表 SHA-256、current metrics、transition
clocks、accept history、candidate evaluations 与主 RNG 前缀逐条完全一致。5 条 B 的 terminal current
仍全部高于当时历史 best，证明在线实现没有偷偷回到 best 表。该审计只验证接线，不验证或调定 P=6、
C 或输出质量。

纯状态机与基础接线测试合计 42 项通过；六轨迹开发逻辑/在线审计测试共 15 项通过；相关新旧回归
255 项通过、1 项按原条件跳过；全库单测 1549 项通过、7 项按原条件跳过。没有增加 seed、读取真实
数据或使用 GPU，也没有接外层选择、加噪、隐私预算与 DP。

未见轨迹的结果前协议现已冻结在
`docs/设计/Issue53_P6未见轨迹质量计算验收协议.md`：首轮固定 P=6，使用两个新公开人工 family、
3 个首轮 seed 与 rho=1/0.25，共 12 cases；B 后只看相对 `+6/+12 work`，C 只作预期 60 work 的
外部护栏。通过门禁固定为 A/B 正常结束覆盖、B/检查点证据完整性、normalized L1 中位非劣与大退化
尾部、以及至少 30% 的中位 work 节省。

失败分流也已结果前写死：质量单独失败只允许一次 P=12；计算单独失败只允许一次 P=4；两者同时
失败或 family 方向冲突则停止调 P、重新设计 B；C/覆盖不足只算证据不足。唯一回退使用另一组已
冻结 seed，不得复用首轮结果或继续扫描其他 P。当前只完成协议，没有实现 family/manifest，也没有
运行任何新 seed。下一步若获授权，只实现确定性 manifest 与纯协议测试，完成后再次停止；不得自动
开始 12-case 验证。
