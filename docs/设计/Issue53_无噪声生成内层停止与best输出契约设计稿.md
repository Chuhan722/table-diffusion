# Issue #53：无噪声生成内层停止与 best 输出契约设计稿

> 状态：三窗直接停止已被影子反例否定；3+3 候选又被固定人工反事实尾部矩阵否定。
> RMSE-only 候选通过旧 6 轨迹小矩阵；用户随后接受平衡候选 `RMSE<=1 AND max error<=2`。纯判定接口及
> 未来外部噪声阈值接口已完成。全新 m/N/workload 结果前协议草案已写，但尚待用户审查；
> 未实现 runner、未运行矩阵，不得在线接入。
> 独立 3+3 停止器仍只保留为已否定原型与回归证据。
> 未运行真实数据、GPU 或外层 DP，也未修改生成主循环。

## 1. 这一小步解决什么

当前只研究一次完整的生成过程：

```text
固定 schema + 固定 workload + 精确查询答案
→ 初始化合成表
→ 无门控残差引导扩散
→ 在线判断是否已经没有继续优化的必要
→ 返回本次运行见过的最好合成表
```

本设计不包含：

- 外层查询选择；
- 私有测量、加噪、sigma、隐私预算或 accountant；
- 从上一轮外层合成表继续运行；
- DP 噪声可信域或噪声地板；
- residual geometry、alpha 或 Gibbs 参数的选优。

未来外层流程只保留为路线记录，必须等本内层全部定型后另立专题。

## 2. 算法身份：无门控更新，best 输出

“无门控”只约束每一步状态转移：

```text
proposal 生成后直接成为下一张 current table
```

不得因为 proposal 的内部 loss 变坏而拒绝、回滚、缩步重试或 top-k 筛选。对应现有主循环
约束为：

```text
tol = positive infinity
max_retries = 0
residual_self_cooling = off
rho annealing = off
alpha = fixed
rho / eta / mu = fixed during one run
factor Gibbs sweeps / tau / s0 / clip = fixed during one run
```

停止器是只读观察者。它可以记录历史 checkpoint 并在停止时返回 best，但不能改变已经执行的随机
转移。因而本方法应准确描述为：

```text
无门控残差引导更新核 + best-checkpoint 输出
```

而不是 current-state 稳态采样器。

## 3. 主拟合指标：生成器内部平方 loss

设：

```text
m        = workload 查询数
N        = 合成表记录数
target_j = 查询 j 的精确目标计数
q_j(S)   = 合成表 S 上查询 j 的计数
```

生成器内部唯一允许参与 best 和停止决策的指标是现有平方 loss：

```text
raw_squared_error(S)
  = sum_j (target_j - q_j(S))^2

loss(S)
  = 0.5 * raw_squared_error(S)

normalized_fit_loss(S)
  = raw_squared_error(S) / (m * N^2)
```

best 和停滞比较直接使用 loss；normalized_fit_loss 只用于把同一内部目标换成跨 N、m 可读的报告
尺度，不改变 checkpoint 排序。理由是：

1. residual、fitness、proposal gain 和现有 best 都以这一内部目标为动力学口径；
2. AIM、Private-GSD 一类 select-measure-generate 方法也区分“拟合已发布测量的内部 loss”和最终
   对真实答案的评价；
3. 当前无噪声阶段 target 是生成器明确收到的精确测量，未来 DP 阶段才会替换成已发布的带噪测量；
4. 无论 target 当前是否精确，生成器都不应读取最终评价 L1 来选择 checkpoint 或决定何时停止；
5. 现有 best_loss 可直接复用，不必为停止器再建立第二套优化目标。

normalized L1、held-out、支持集、多样性和参考表指标全部属于生成结束后的评价层。即使当前精确
target 使 measured L1 在数学上可计算，它也不得进入 stopper、best、参数控制或资源上限。旧
current-state L1 轨迹可以为兼容性保留，但新的生产停止入口不得读取它；正式评价只在返回表已经
物化后进行。

### 3.1 best checkpoint 的唯一排序

初始表 S0 也有资格成为 best。每个真实 post-round 状态只按内部 loss 比较：

```text
loss 更小：替换 best
loss 相同：保留更早出现的 checkpoint
```

因此 best 身份可写为：

```text
best_key = (loss, first_seen_state_index)
```

只有 loss 严格下降才替换 best 并记为一次进展；相等不替换、不重置停滞计数。这与当前代码只在
`current_loss < best_loss` 时更新 best 的语义一致。

current loss 允许自由上升或下降；普通有限上升不会触发拒绝、回滚或立即停止。判断进展时不比较
相邻 current loss，只判断是否刷新历史 best：

```text
current_loss: 100 -> 110 -> 105 -> 95
best_loss:    100 -> 100 -> 100 -> 95   # 95 才是新进展

current_loss: 100 -> 110 -> 105 -> 101
best_loss:    100 -> 100 -> 100 -> 100  # 没有新进展
```

因此临时上坡仍可帮助无门控核离开当前区域；若长期没有找到低于历史 best 的状态，连续窗口规则会
停止运行，并返回历史 best 而不是较差的 terminal current。

## 4. 不再设置浮点“改善率阈值”

当前无噪声、无查询权重阶段有一个可直接利用的离散性质：

```text
target_j 和 q_j(S) 都是整数
→ raw_squared_error 是非负整数
→ loss 是非负整数或半整数
→ 任意严格改善至少让 loss 减少 0.5
```

所以本设计不再引入 0.1%、0.01% 或按数据集定标的 min_change_rate。统一规则是：

```text
一个窗口内 best_loss 至少减少 0.5 → 有进展
一个窗口内 best_loss 完全不变     → 无进展
```

这等价于把“实质改善阈值”冻结为一个由问题离散结构决定的最小单位，而不是人为浮点超参数。

该规则只适用于当前精确整数 target。未来带噪 target 可能是非整数，届时内部仍使用 noisy loss，
但“最小改善为 0.5”的离散性质不再成立，必须另立 DP 设计，不能静默复用本阈值。

## 5. 标准工作量：参与记录的等效全表扫描

原始 round 的含义随 rho 改变：rho=0.01 的一轮和 rho=0.1 的一轮不是同等更新机会。因此使用
现有 transition clock 中已经记录的 participating_rows 定义：

```text
cumulative_participating_rows
  = 到当前状态为止，所有已应用 proposal 的参与记录数之和

normalized_work
  = cumulative_participating_rows / N
```

一个标准工作窗口固定为：

```text
window_work = 1.0
```

也就是累计 N 次记录参与事件，解释为“一次等效全表扫描”。窗口边界取第一次满足下面条件的真实
post-round 状态：

```text
normalized_work >= 1, 2, 3, ...
```

不插值、不伪造状态。单轮最多贡献 N 个参与事件，因此不会一次跨过多个完整窗口。

### 5.1 为什么用 participating rows，而不是 changed rows

- participating rows 表示核实际获得了多少次尝试修改记录的机会；
- changed rows 会受 donor 是否相同、eta、Gibbs mask 和 mutation 影响；
- 若核获得机会却长期不改变或不改善，这本身正是 optimization_stalled 应捕获的行为；
- 使用 changed rows 会让弱核因为“改不动”反而获得更多运行时间。

### 5.2 independent 与 factor Gibbs 的成本边界

停止器对两种核都只读取 normalized_work，不读取 kernel 名、sweeps 或 tau。factor 核的额外成本必须
继续单独报告：

```text
candidate evaluations
Gibbs microsteps
factor build / sampling wall time
total wall time
```

相同 normalized_work 表示相同数量的记录更新机会，不表示相同计算成本。质量和成本必须同时报告。

## 6. 双条件正常停止

### 6.1 条件 A：exact_residual

```text
best_loss == 0
```

等价于所有 measured 查询精确命中。检查位置包括：

1. 初始状态 S0；
2. 每个已经应用的 post-round 状态。

若 proposal 在某轮后首次达到 0，必须在该 post-round 立即停止，不能为了下一轮 proposal 前检查而
多记一个没有执行 proposal 的伪轮次。此时当前表就是新 best。

### 6.2 条件 B：optimization_stalled

每个完整工作窗口结束时，比较该窗口起点和终点的 best_loss：

```text
best_loss 下降至少 0.5 → 本窗口 progress=true，连续无进展计数清零
best_loss 完全不变     → 本窗口 progress=false，连续无进展计数加一
```

只要窗口内任一状态刷新 best，就立即打断此前连续无进展序列；current loss 后来再次上升不会抹掉
这次进展。current loss 从更高位置下降、但仍未低于历史 best，不算进展。

停滞采用两个长度相同、语义不同的确认块：

```text
stall_block_windows = 3
required_no_progress_windows = 2 * stall_block_windows = 6
```

连续前三个空窗口只产生 stall candidate，不能停止；随后再连续三个空窗口仍没有任何 best loss 改善，
才输出 optimization_stalled。确认期间任何严格新 best 都立即取消 candidate，并把连续计数清零。

选择 3 的机制依据不是数据集结果：在固定 rho、逐记录独立参与的近似下，经过 k 次等效扫描后，
某条记录从未获得参与机会的概率约为 exp(-k)。k=3 时约为 4.98%，即约 95% 的记录至少获得过一次
参与机会，因此只能作为“候选”依据。最初把这三窗直接当停止条件，已被固定影子反例否定：state 3
候选后，state 6 才出现更低 best。第二个三窗块用于捕获这种延迟改善；六次等效扫描后从未参与的近似
概率约为 exp(-6)=0.25%。这仍只是工作覆盖率护栏，不是“未来不可能再改善”的统计证明，所以状态名
必须是 stalled，不能写 converged 或 stationary。

以 rho=0.01 为例，三窗候选约在连续无进展 300 个原始轮后出现，正式停滞最早约在连续无进展 600
轮后确认。这不是每个数据必须预跑的固定轮数：条件 A 可以更早停止，任何新 best 都会让两阶段时钟
从该改善之后重新开始。

每个真实 post-round 的固定处理顺序为：先用已计算的 current loss 更新 best，再累计本轮实际参与
记录并结算可能到达的窗口边界，最后按 `exact_residual > optimization_stalled > resource_cap_reached`
选择终止原因。这样第六个空窗口的边界若恰好刷新 best，不会被误判停滞；若同一状态精确命中，
exact_residual 优先。

## 7. 资源护栏

建议第一版冻结：

```text
max_normalized_work = 20
```

理由是它只复用当前正式 2000-round、rho=0.01 实验的大致既有计算上界：

```text
2000 rounds * 0.01 expected participation ≈ 20 equivalent scans
```

它不来自哪个数据集的质量终点，也不是正常停止标准。大多数正常运行应由 exact_residual 或
optimization_stalled 更早结束；达到 20 只输出：

```text
termination_reason = resource_cap_reached
inner_complete = false
```

触及护栏前最后一个已应用状态仍须先参与 best 更新，然后返回 best checkpoint。现有 n_rounds 或
candidate_budget 可以保留为更外层的故障保险，但若它们先触发，必须使用独立原因并同样标记
inner_complete=false；不得把任何资源上限改名为 converged。

用户已同意在当前独立状态机中把 max_normalized_work=20 作为统一工程护栏。以后若进入正式比较，
必须对全部数据集、residual geometry、alpha 和 kernel 使用同一个值，不允许按结果分别修改。

## 8. 输出契约

主返回值：

```text
best_table
```

至少同时记录：

```text
termination_reason
inner_complete
best_state_index
best_round_index
best_normalized_work
best_loss
best_normalized_fit_loss
terminal_state_index
terminal_round_index
terminal_normalized_work
terminal_loss
terminal_normalized_fit_loss
completed_work_windows
consecutive_no_progress_windows
candidate_evaluation_count
factorized_gibbs_microsteps
elapsed_sec
```

终止语义唯一为：

| termination_reason | inner_complete | 准确含义 |
|---|---:|---|
| exact_residual | true | measured workload 精确命中 |
| optimization_stalled | true | 三窗候选加三窗确认期间均未找到更低内部 loss |
| resource_cap_reached | false | 只耗尽工程资源，不能声明正常完成 |
| candidate_budget | false | 更外层候选评价保险先触发 |
| max_rounds_guard | false | 更外层原始轮数保险先触发 |

即使 inner_complete=true，也只表示本次内层可结束：

- exact_residual 证明内部 measured workload loss 为 0；
- optimization_stalled 不证明全局最优、held-out 质量、平稳性或未来永不改善；
- 最终方法质量仍必须用 measured、held-out、支持集、多样性和成本共同评价。

## 9. 与已有实现的复用和必须修改处

可以直接复用：

- initial current-state 评价；
- current_state_metrics_history；
- participating_rows 与 transition clocks；
- candidate_evaluation_count、Gibbs microsteps 和墙钟；
- fixed alpha / fixed s0 / prefix invariance 门禁；
- 现有 best 表深拷贝方式；
- 离线 quality evaluator。

后续实现必须有意修改：

1. 保留现有 squared best_loss 主排序，但补齐 best 的 state/round/work 身份；
2. exact_residual 改为 initial 和真实 post-round 检查，避免下一轮伪计数；
3. 在线累计 participating rows 并建立整数 work-window 边界；
4. 增加“3 窗候选 + 3 窗确认”的连续无进展状态机；
5. Issue #53 新入口返回 best，而不是 reference_process 强制的 final current；
6. stopper API 不接收 normalized L1、held-out 或任何最终评价结果；
7. 观测不得增加查询评价次数、消费 RNG 或改变停止前的轨迹前缀。

旧 V1/V2/V2b/V2c detector 不进入该状态机。它们只保留轨迹持久化、审计和反例价值。

## 10. 实现前人工边界审查

后续实现前必须先用纯构造序列覆盖以下行为，不读取 test_300x10、nltcs 或 validation：

| 构造情形 | 预期行为 |
|---|---|
| S0 已经 loss=0 | 0 proposal，exact_residual，返回 S0 |
| 某个 post-round 首次降到 0 | 该状态立即停止，不多记伪轮次 |
| 连续三窗口 best loss 不变 | 只进入 stall candidate，不停止 |
| candidate 后再连续三窗口 best 不变 | 第六窗口边界 optimization_stalled |
| 两个空窗口后第三窗改善 | 连续计数清零，不停止 |
| 前五个空窗口后第六窗改善 | candidate 与确认计数清零，不停止 |
| 改善后再连续六个空窗口 | 在第六个窗口结束时停止 |
| loss 只下降 0.5 | 仍是精确有效进展，计数清零 |
| current loss 上下波动、best 不变 | 仍按 best 正确停滞，不被 current 波动欺骗 |
| 两条相同 work/best-loss 序列在结束后离线算出的 L1 不同 | 停止分类仍相同；生成器与 stopper 均不读取 L1 |
| 相同 work/best 序列但 N、rho 不同 | 得到相同工作窗口分类 |
| independent/factor 标签互换 | 停止结果不变 |
| normalized_work 先到 20 | resource_cap_reached，先更新边界状态 best |
| candidate budget 先到 | candidate_budget，不能冒充正常完成 |
| exact 与工作窗口同一状态触发 | exact_residual 优先 |
| 第六空窗与资源护栏同一状态触发 | optimization_stalled 优先，保留正常停滞语义 |
| loss 为 NaN、无穷或负数 | 立即报运行错误，不当作普通上升或正常终止 |
| best 与 terminal 不同 | 返回 best，两个身份和指标均可独立复算 |

### 10.1 已完成的只读离线影子回放

`tests/test_inner_stopping_shadow_replay.py` 用固定 seed 在 CPU 上运行现有、未修改的无门控残差引导核：
人工 16×3 二元 schema、6 个固定精确计数查询、rho=1、固定 alpha，共采集 25 个 post-round；新停止器
不参与在线控制。生成完成后，回放材料只保留 state/round、current squared loss 和 applied
participating rows，明确删除 normalized L1 与表内容，再离线送入停止器。

最初的三窗直接停止规则在该轨迹上得到：

```text
state 0..3 current_loss = 5.0, 19.5, 9.5, 10.5
state 3 candidate best = 5.0
state 6 later best     = 3.0
```

因此原三窗规则会在 state 3 错过 state 6 的真实后续改善，已明确否定，不得接入。修订为 3+3 后，
state 3 只进入确认期；state 6 的新 best=3.0 在第六个边界先更新 best、取消 candidate 并清零计数。
随后 state 7..12 连续六窗没有更低 best，修订规则在 state 12 输出 optimization_stalled 并返回 state 6。
反事实 state 13..25 也没有低于 3.0 的结果。

影子逐状态复算 prefix minimum、累计参与行、窗口数和 first-seen best，并核验 best 表哈希。现有生成器
继续跑满 25 轮，回放前后的完整诊断、候选评价数和 RNG 哈希完全不变，证明停止预测没有反向影响
轨迹。

该测试只证明现有诊断能够无歧义驱动停止器和恢复 best 身份，不证明这条人工轨迹的质量、收敛或
真实数据效果。rho=1 只是让每个小测试轮恰好形成一个工作窗口，不是生产参数建议。预检查还确认，
legacy 非残差引导配置不会仅因 record_transition_clocks 记录 initial table 哈希；当前目标残差引导核
已有该哈希，因此本测试可完整验证 best 身份。该 legacy 诊断差异不扩展本阶段范围。当前只有这一条
固定反例通过修订规则，尚不能据此宣布 3+3 普遍安全；在线接入前还必须做固定人工反事实尾部矩阵。

### 10.2 固定人工反事实尾部矩阵（结果前冻结）

矩阵继续使用上述 16×3 人工二元问题与完全相同的无门控核参数，只系统改变 seed 和参与比例：

```text
seed = 20260816, 20260817, 20260818
rho  = 1.0  时完整运行 40 轮
rho  = 0.25 时完整运行 160 轮
```

共 6 条轨迹；两个 rho 尺度的期望工作量都是 40 次等效全表扫描。原生成器始终完整跑完，
停止器只在事后读取 loss 和 applied participating rows；其预测停止点之后的原轨迹就是反事实尾部。
停止器资源护栏保持 20 次扫描，不根据矩阵结果改参数。

矩阵在运行前固定三个通过条件：

1. 每条预测停止点之后至少保留 10 次等效扫描的尾部，否则证据不足；
2. 任一 `resource_cap_reached` 都只是未解决，不能冒充正常完成；
3. 任一 `optimization_stalled` 之后若出现严格更低 loss，3+3 候选即被否定，不得在线接入。

本矩阵是小规模反例搜索，即使全部通过也不证明收敛或未来永不改善；只有失败方向的结论是确定的。

### 10.3 矩阵结果：3+3 候选被否定

首次执行前，固定测试源文件 SHA-256 为
`4506aedca5bca7fc415e539e3b7862ac1e96f452c454c8254eab9b44a34cb411`。三个证据门禁中，尾部长度与
无资源护栏冒充均通过，但核心“停止后无严格改善”门禁失败：

| seed | rho | 预测停止 state | 停止 work | 停止时 best | 尾部 best |
|---:|---:|---:|---:|---:|---:|
| 20260816 | 1.0 | 12 | 12.0000 | 3.0 | 2.0 |
| 20260817 | 1.0 | 20 | 20.0000 | 3.0 | 2.0 |
| 20260818 | 1.0 | 8 | 8.0000 | 3.0 | 1.5 |
| 20260816 | 0.25 | 27 | 7.0625 | 3.0 | 1.0 |

即 6 条固定轨迹中有 4 条提前停止，另 2 条在本有限尾部内未找到反例。原影子轨迹
`seed=20260816, rho=1.0` 在 state 13..25 确实没有低于 3.0，但延长至 state 40 后已出现 2.0；
这说明原来的有限短尾部结论不能支持在线停止。

结论是 3+3 不得接入 `run_evolution`。本步不事后增加窗口、更换 seed 或删除失败轨迹；
因为任意固定耐心值都只能声明“已停滞”，不能从无门控随机轨迹本身证明“继续跑不会再改善”。

### 10.4 一条记录查询 RMSE 目标的固定离线验证（结果前冻结）

3+3 反例否定的是“已收敛”，不是“已达到事先定义的拟合质量”。新候选只把现有 squared loss
换成可读的查询计数均方根误差，不引入新优化指标：

```text
m = 查询数
query_count_RMSE = sqrt(2 * best_loss / m)
候选合格线：query_count_RMSE <= 1 条记录
等价线：best_loss <= m / 2
```

在当前无噪声整数计数问题中，一条记录是最自然的非零误差尺度。它表示查询计数误差的均方根不超过 1，
不保证每个查询都单独不超过 1。现有归一化 residual 的 RMSE 等于 `query_count_RMSE / N`；
由均方根不小于平均绝对值，该条件还能保证选定表的离线 normalized L1 不超过 `1/N`。

验证不新增 seed，原样复用 10.2 的 6 条完整轨迹：`20260816..20260818` 与
`rho in {1.0, 0.25}`。选点材料明确删除 L1，只保留 state 与 current squared loss；事后定位历史
best 首次满足合格线的状态。再依赖 `horizon_invariant=true` 重放到该固定前缀，物化当时的 best table；
只在表已选定后独立复算查询答案、loss、residual RMSE 和 normalized L1。

运行前固定三个门禁：

1. 6/6 轨迹都必须在 20 次等效扫描之前首次达到合格线；
2. 物化 checkpoint 必须与原轨迹前缀、表哈希和独立复算 loss 逐值一致；
3. 后算 normalized L1 必须不超过 `1/16 = 0.0625`，但不得反向影响选点。

本小矩阵通过只能说候选没有立即被已知人工轨迹否定，不代表真实数据或其他 workload 已验证，
也不把“达标”改称为“收敛”。

### 10.5 一条记录查询 RMSE 小矩阵结果：候选暂时可行

首次执行前，固定测试源文件 SHA-256 为
`e0d4a28f160bdf8732ce4776c4bcb4732484cb8f52abb6b206033a7610759d9d`。三项门禁全部通过：6/6 轨迹都在 20 次
等效扫描前达标，前缀、表哈希和 loss 复算一致，且只在 checkpoint 选定后计算的 L1 均低于 0.0625。

| seed | rho | first state | work | loss | 计数 RMSE | 后算 L1 | 最大单查询误差 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260816 | 1.0 | 6 | 6.0000 | 3.0 | 1.0000 | 0.041667 | 2 |
| 20260817 | 1.0 | 14 | 14.0000 | 3.0 | 1.0000 | 0.041667 | 2 |
| 20260818 | 1.0 | 2 | 2.0000 | 3.0 | 1.0000 | 0.041667 | 2 |
| 20260816 | 0.25 | 2 | 0.5625 | 3.0 | 1.0000 | 0.041667 | 2 |
| 20260817 | 0.25 | 2 | 0.7500 | 3.0 | 1.0000 | 0.041667 | 2 |
| 20260818 | 0.25 | 9 | 2.5625 | 1.5 | 0.7071 | 0.031250 | 1 |

首次达标工作量范围为 0.5625..14，明显早于 20 次资源护栏。完整尾部仍会将 best loss 继续改善到
0..2，因此该条件的正确语义是“已达到预定拟合质量”，不是“已收敛或后续不会改善”。

边界也很明确：前 5 条轨迹在计数 RMSE=1 时都有一个查询绝对误差为 2。所以它保证的是 workload 整体均方根误差，
不是逐查询的最大误差。当前只能说“同一 6-query、16-record 人工问题的 6 条轨迹暂时可行”；
是否接受这个平均语义，以及不同 m/N/workload 下是否仍可行，必须在在线接入前另行确认。

### 10.6 单查询护栏与未来噪声接口

用户指出只看整体 RMSE 可能遮蔽少数很差的查询。若直接要求每个查询计数误差都不超过 1，
在原 6 条完整轨迹上虽然最终 6/6 都能达到，但只有 3/6 在 20 次等效扫描前达到；平均达标工作量从
RMSE 条件的 4.3125 上升到 19.6458，约为 4.6 倍。因此 `max error <= 1` 在当前效率目标下偏严。

当前待新样本验证的平衡候选固定为两项同时成立：

```text
query_count_RMSE <= 1
max_j abs(count_error_j) <= 2
```

第二项能拒绝“100 个查询中 99 个为 0、1 个误差为 10”这类 RMSE 恰好等于 1 的尖峰。计数误差为整数时，
2 是从过严的 1 向上的最小放宽。但这个 2 是在看过旧 6 条轨迹后提出的，旧轨迹只能说它不会增加已知成本，
不能当作新候选的独立通过证据。

新增纯模块 `src/table_diffevo/inner_fit_target.py`，尚未被主循环引用。`QueryFitThresholds.exact_integer_counts()`
唯一生成当前 `RMSE=1 / max=2` 候选；`assess_query_fit` 只评估一个明确 checkpoint 的同一条
count-error 向量，同时返回 squared loss、RMSE、最大误差、两个门禁和 exact residual。返回表、loss 与最大误差
必须绑定同一 checkpoint，不得把不同表的指标拼接成一次达标。

为未来带噪测量只保留数值阈值接口，不在当前内层实现噪声公式。
`QueryFitThresholds.external_noise_calibrated(...)` 可接收外层从已发布噪声事先换算的：

- 一个全局 count-RMSE limit；
- 一个统一 max-error limit，或与 workload 等长的逐查询 limit 向量。

纯模块刻意不接收 sigma、真实答案、reference table、L1、隐私预算或 accountant。未来“如何从噪声得到阈值”必须另立
外层设计；当前只保证内层接口无需破坏就能接收均匀或异质噪声阈值。

### 10.7 全新 RMSE + max 人工验证协议与固定 runner

结果前协议已由用户确认并单独写入
`docs/设计/Issue53_RMSEMax全新人工验证协议.md`。它固定三个全新二元人工 family，
分别覆盖 `N=24,m=3` 的偏态边缘、`N=32,m=10` 的环形二阶重叠，以及
`N=64,m=15` 的高阶嵌套包含。每类两个全新 seed，与 `rho in {1.0,0.25}` 配对，
共 12 条完整无门控轨迹。

协议补齐了一个必须在在线接入前解决的输出语义：正常达标时，loss、RMSE、MAX 和
返回表必须来自同一张首次达标的 current table；不能用它决定停止后，改为返回另一张
虽然 loss 更低但 MAX 未达标的历史 best。只有 20 次等效扫描护栏用尽且仍无达标表时，
才返回历史最低 loss 表作为明确未达标的 fail-closed 兜底。

候选的唯一科学门禁是 12/12 轨迹都在第一个跨过 20 次扫描的真实状态及之前同时达到
`RMSE<=1` 和 `MAX<=2`。任一轨迹迟到或从未达标，候选即失败；不事后改 2、改资源护栏或换 seed。
固定 runner `scripts/validate_issue53_rmse_max_artificial.py` 与确定性契约测试已经实现。runner 只暴露
只读 `plan` 和无科学覆盖项的 `run --output-dir`；正式运行前检查协议哈希、干净工作树、源文件和
全新输出目录。选点纯回放层只接收同状态 count-error 与实际 applied work，L1 只在选点后计算。

只读 plan 已验证 12 个身份和 1200 个小型人工轮次的固定计划，但没有构造正式 RNG 或调用生成器。
29 项新契约测试中只有一条使用非正式 seed 的 40 轮内存内接线 smoke，不属于正式矩阵。
**12 条正式轨迹仍未运行，6 个正式 seed 仍未实例化；正式运行需要用户再次明确授权。**

## 11. 验收不变式

未来代码验收至少包括：

1. 关闭新停止器时，表、loss、RNG、候选数与当前冻结轨迹完全一致；
2. 把资源上限延长时，原上限前的轨迹、best 历史和工作窗口逐位前缀一致；
3. 开启停止器不会额外评价同一张表或额外消费 RNG；
4. best_loss 与 normalized_fit_loss 可由返回 best_table 独立复算；
5. participating rows 总和与 transition clocks 精确一致；
6. termination reason、inner_complete 和返回表语义 fail closed；
7. API 不接受 dataset 名、held-out 答案、reference table、kernel-specific 阈值映射；
8. sigma、噪声地板、隐私预算与外层 orchestration 不进入本次实现。

## 12. 已确认边界与本步授权

用户已确认：

1. L1 只在生成选点结束后离线评价，不进入在线达标或资源兜底选点；
2. 无门控轨迹允许 current loss 上升；正常质量完成看同一 current checkpoint 的 RMSE+max，
   资源用尽兜底才比较历史 squared loss；
3. 一个窗口等于 1 次等效全表参与扫描；
4. 原“连续 3 窗直接停止”已被影子反例否定，“3 窗候选 + 3 窗确认”也已被固定矩阵否定；
5. 精确整数阶段不设置浮点改善率阈值，任意严格 loss 改善至少为 0.5；
6. max_normalized_work=20 只作工程护栏，触及后不得称正常完成。
7. 当前待新矩阵验证的达标候选为整体计数 RMSE 不超过 1，且最大单查询计数误差不超过 2；
8. 未来加噪时由外层从已发布噪声换算阈值，内层评估接口不直接读取 sigma、隐私预算或真实答案。
9. 当前授权止于固定 runner、只读 plan 与契约测试；正式 12 轨迹执行必须再次授权。

当前已实现不读取表、查询、L1 或 RNG 的 3+3 纯逻辑状态机，完成纯构造边界测试、一条未修改生成器
的只读离线回放和 6 条固定人工反事实尾部矩阵。矩阵否定了 3+3，所以该状态机只保留为反例证据，
不得在线接入。新的 RMSE+max 单-checkpoint 纯评估接口、已确认的结果前协议、固定 runner 和契约测试
均已完成；正式 12 轨迹矩阵仍未运行，也未接入主循环。当前不运行真实数据，不推进外层 DP，
不事后调整阈值、seed、family 或资源护栏。下一步若继续，必须由用户另行授权一次正式矩阵执行。
