# Issue #53：RMSE + max 无噪声达标候选的全新人工验证协议

> 状态：结果前协议已由用户确认；当前只授权实现固定 runner、只读 plan 与确定性契约测试。
> 正式 RNG 尚未实例化，12 条矩阵轨迹尚未运行，也不得修改或接入 `run_evolution`。

## 1. 这一小步只回答什么

当前已接受、但是在旧 6 条轨迹之后才形成的无噪声候选是：

```text
查询计数 RMSE <= 1 条记录
AND
每个查询的绝对计数误差 <= 2 条记录
```

这个协议只做一次小型的全新反例搜索：它在不同 `N`、查询数和查询重叠结构下，
能否让固定的无门控生成核在统一资源护栏前返回一张同时满足两个误差条件的表。

它不回答：

- 生成链是否收敛、平稳或已到全局最优；
- 继续运行是否还会得到更低 loss；
- 真实数据、held-out 或支持集质量；
- relative residual geometry、floor、alpha 或 Gibbs 参数的选优；
- 外层查询选择、测量加噪、sigma、隐私预算或 accountant。

即使 12/12 全部通过，结论也只是“这个达标候选没有被本次全新人工矩阵立即否定”，
不得写成“证明收敛”或“对任意数据通用”。

## 2. 达标必须绑定同一张表

对任一明确的当前 checkpoint `S_k`，记：

```text
e_j(S_k) = 目标计数_j - S_k 上的查询计数_j

RMSE(S_k) = sqrt(所有 e_j(S_k)^2 的平均值)
MAX(S_k)  = 所有 abs(e_j(S_k)) 中的最大值
```

两个边界都是包含等号的：

```text
qualified(S_k) = [RMSE(S_k) <= 1] AND [MAX(S_k) <= 2]
```

`loss = 0.5 * sum_j e_j^2`、RMSE、MAX 和返回表必须来自同一个 `S_k`。禁止把表 A 的
历史最低 loss 与表 B 的最大误差拼在一起宣布达标。正式评估必须调用已有的
`QueryFitThresholds.exact_integer_counts()` 和 `assess_query_fit(...)`，不在 runner 里重写第二套公式。

### 2.1 正常达标时返回什么

对初始表 `S_0` 和每个真实 post-round 当前表，按状态顺序评估。第一个达标状态记为：

```text
K = 第一个使 qualified(S_K)=true 的状态
```

正常结束时返回 `S_K` 本身，不是自动改成前缀中另一张只有 loss 更低的表。原因是：

```text
历史最低 loss 的表，可能把误差集中在一个查询上，因而 MAX > 2；
稍高 loss 的当前表，反而可能同时满足 RMSE <= 1 和 MAX <= 2。
```

如果用 `S_K` 决定停止，却返回另一张未达标的历史 best，那么返回值就违反了停止理由。
因此后续接入时应使用不含混的名称 `selected_table`；`best_loss_table` 只作资源用尽时的兜底。

终止优先级固定为：

```text
exact_residual > fit_target_reached > resource_cap_reached
```

exact residual 本来就同时满足 1/2 条件，但保留更精确的终止原因。

### 2.2 资源用尽但仍未达标时返回什么

资源护栏仍是 20 次等效全表扫描：

```text
work(S_k) = 到 S_k 为止的累计 applied participating rows / N
B = 第一个使 work(S_B) >= 20 的真实状态
```

每轮先完整应用 proposal、评估 `S_k`，再检查资源边界。所以如果 `S_B` 首次达标，
它仍按 `fit_target_reached` 正常返回。因为一轮最多参与 `N` 行，`S_B` 的实际 work 只可能在
`[20, 21]` 之内；不伪造 work=20 的中间表。

只有到 `S_B` 仍没有任何达标表时，才：

```text
termination_reason = resource_cap_reached
fit_target_reached = false
selected_table = S_0..S_B 中 loss 最低、并列时最早的表
```

这只是 fail-closed 兜底，不得宣布“结果已合格”。被旧反例否定的 3+3 停滞规则不进入本协议。

## 3. 三类全新人工问题

三类问题全部只含二元类别属性，且有显式可行的参考表。参考表只在 preflight 中独立计算一次
固定 target；生成器和达标判定器只获得 schema、查询和 target 计数，不得获得参考表。

| family | 属性数 | N | 查询数 m | 查询结构 | 主要压力 |
|---|---:|---:|---:|---|---|
| `marginal_skew` | 3 | 24 | 3 | 3 个单属性边缘 | 小 m、低目标计数 |
| `ring_pair` | 5 | 32 | 10 | 5 个边缘 + 5 个环形相邻对 | 中等 m、重叠二阶约束 |
| `nested_overlap` | 6 | 64 | 15 | 6 个边缘 + 5 个相邻对 + 4 个嵌套前缀 | 大一些的 N/m、高阶包含约束 |

### 3.1 `marginal_skew`

schema 属性为 `a,b,c`。查询分别计数 `a=1`、`b=1`、`c=1`。参考表为：

```text
18 行 000
 6 行 111
```

因此固定 target 向量为 `[6, 6, 6]`。它与随机初始化的单属性期望计数 12 明显不同，
用来防止只在“目标天然接近随机初始化表”的玩具上通过。

### 3.2 `ring_pair`

schema 属性为 `a,b,c,d,e`。查询为：

```text
5 个边缘：a=1, b=1, c=1, d=1, e=1
5 个环形对：(a,b), (b,c), (c,d), (d,e), (e,a) 都等于 1
```

参考表是 16 行 `00000` 和 16 行 `11111`，所以 10 个 target 全部是 16。环形对让同一属性
同时出现在多个查询中，用来检查两个误差条件在重叠 workload 上是否仍能同时达到。

### 3.3 `nested_overlap`

schema 属性为 `a,b,c,d,e,f`。查询为：

```text
6 个边缘：每个属性等于 1
5 个相邻对：(a,b), (b,c), (c,d), (d,e), (e,f) 都等于 1
4 个前缀合取：(a,b,c)、(a,b,c,d)、(a,b,c,d,e)、(a,b,c,d,e,f) 都等于 1
```

参考表是 32 行 `000000` 和 32 行 `111111`，所以 15 个 target 全部是 32。这一类同时增加
`N`、`m` 和包含关系，专门寻找“平均误差已小，但某个高阶查询仍集中超标”的反例。

这三类是小型压力矩阵，不是用来分离估计 `N`、`m` 或查询结构的单独因果效应。

## 4. 全新 seed 和固定轨迹数

每个 family 使用两个未出现在旧 6 轨迹证据中的 seed：

```text
marginal_skew:  20260901, 20260902
ring_pair:      20260911, 20260912
nested_overlap: 20260921, 20260922
```

每个 `(family, seed)` 都与 `rho in {1.0, 0.25}` 配对，一共：

```text
3 families * 2 seeds * 2 rho = 12 条轨迹
```

同一 family/seed 的两个 rho 从同一随机初始化表开始，用来检查判定是否绑定实际参与工作量，
而不是将原始 round 误当为可直接比较的成本。

## 5. 固定的无门控生成轨迹

为了保留资源边界后的尾部，生成器在采集时始终跑满固定 horizon，达标判定只在完整轨迹上
事后回放，不影响生成轨迹或 RNG：

```text
rho=1.0  -> 40 rounds
rho=0.25 -> 160 rounds
```

两者期望都是 40 次等效全表扫描。正式运行前固定的共同参数为：

```text
init_method = random
device = numpy
eval_method = vectorized
beta = 1.0
h = 0.8
distance_mode = geometric
lambda = 0.5
exclude_self = true
alpha_schedule_mode = fixed
fixed_alpha = 6.0
rho = 1.0 or 0.25
eta = 0.45
mu = 0.02
tol = positive infinity
max_retries = 0
residual_directed_diffusion = true
diffusion_direction_strength = 0.8
diffusion_direction_normalization = fixed
diffusion_direction_reference_scale = 1.25
diffusion_direction_logit_clip = 9.0
residual_geometry = absolute
factorized_gibbs_sweeps = 0
candidate_budget = none
residual_self_cooling = off
rho_annealing = off
selection_scale_invariant = false
horizon_invariant = true
stop_on_exact_residual = false
record_transition_clocks = true
record_stationarity_trace = true
```

`tol=+infinity` 与 `max_retries=0` 保证有限 proposal 直接成为下一张 current table，不因 loss 上升拒绝、
回滚或重试。本次只使用 0-sweep independent 核是为了保持矩阵小且不混入 Gibbs 配置选择；
这不能作为该达标候选已在所有 Gibbs、alpha 或 residual geometry 配置上验证的证据。

## 6. 判定器能看到和不能看到的东西

完整轨迹采集完成后，先建立最小回放投影。每个状态只向达标判定层提供：

```text
state_index
该状态自身的 count-error vector
到该状态的累计 applied participating rows
```

达标判定层不得接收：

```text
normalized L1
参考表或任何真实行
held-out 答案
dataset/family 名称到阈值的映射
sigma、privacy budget 或 accountant
完整尾部中未来状态的任何指标
```

虽然现有诊断采集器可能同时记录 L1，但选点投影必须将其删除，并用测试证明替换全部离线 L1 值
不会改变首次达标状态和分类。

## 7. 结果前固定的门禁

### 7.1 先验证执行是否有效

以下任一项失败都属于“执行无效”，不是候选的科学通过或失败：

1. 矩阵不是精确的 12 个唯一 `(family, seed, rho)` 组合；
2. schema、查询、N、显式参考表和固定 target 的独立评价不一致；
3. 任一完整轨迹没有跨过 20 次扫描资源边界；
4. 任一轨迹在边界状态之后不足 10 次实际等效扫描的尾部；
5. state、query-answer trace、transition clock 数量不对齐，applied participating rows 不能独立复算；
6. 任一 count error、loss、RMSE、MAX 或 work 为非有限数；
7. 完整轨迹不是 `max_rounds` 结束，或存在 proposal 拒绝/重试；
8. 完整轨迹的输入、参数、源码和结果前协议身份无法用 SHA-256 绑定。

如果发生上述情况，必须停止并回到用户讨论；不得悄悄延长 horizon、换 seed 或删除病例。

### 7.2 候选的唯一科学通过条件

对每条轨迹，先求第一个资源边界状态 `B`，再求第一个达标状态 `K`。唯一通过门禁是：

```text
12/12 轨迹都存在 K，且 K <= B
```

即每条轨迹都要在资源护栏的边界状态及之前，用同一张当前表同时达到 `RMSE<=1`
和 `MAX<=2`。初始表若已达标，按 work=0 的合法结果如实记录，不为了让问题看起来更难而换 seed。

任意一条轨迹只在 `B` 之后达标，或到完整 horizon 仍未达标，候选就在该矩阵上失败。
失败后固定动作是保留负结果、停止接入讨论，不在同一结果上改 `2`、改 20、换 seed 或删 family。

### 7.3 同 checkpoint 物化与独立复算

对 12 条轨迹各自的选定状态：

- 若 `K<=B`，选定 `S_K`；
- 否则选定 `S_0..S_B` 中 loss 最低、并列时最早的未达标兜底表。

使用 `horizon_invariant=true` 重放到选定状态，物化该状态的 **final current table**，不能用
`run_evolution` 默认返回的历史 best 冒充当前表。对 `S_0` 必须有专门的初始化重放路径。

物化后必须逐项匹配：

```text
完整轨迹中该 state 的 table SHA-256
完整轨迹的前缀诊断和 RNG 身份
独立计算的 query answers 和 count-error vector
assess_query_fit 的 loss / RMSE / MAX / 两个子门禁 / 合并分类
```

任一不匹配属于实现或证据链失败，不得用近似容差掩盖。

## 8. L1 只能在选点后报告

只有 `selected_table` 已由第 7 节的过程完全固定后，才可以离线计算：

```text
offline normalized L1 = 平均 abs(count error) / N
```

它不能改变 `K`、`B`、选定表或通过/失败。对于已达标的表，`RMSE<=1` 在数学上已经保证：

```text
offline normalized L1 <= 1/N
```

所以这个上界只作选点后的独立一致性检查；如果违反，说明实现或数据身份有错，
不是为候选另外增加一个可调阈值。

## 9. 必须固定的结果字段

每条轨迹至少报告：

```text
family, N, m, seed, rho, n_rounds
initial loss / RMSE / MAX / qualified
resource-boundary state and actual work
first-qualified state, round and actual work（若存在）
selected state and termination reason
selected loss / RMSE / MAX / per-query absolute errors
selected checkpoint 是否也是当时的 prefix minimum-loss checkpoint
选点后 offline normalized L1
完整尾部的 minimum loss 与相对 selected loss 的后续改善
full work, tail work, candidate evaluations, elapsed time
table/query/target/parameter/protocol/source SHA-256
```

“首次达标表是否恰好也是前缀最低 loss”必须报告，但不参与通过条件；它用来检查后续 API 中
`selected_table` 与 `best_loss_table` 是否需要同时保留。完整尾部后续改善也只作报告；出现改善不会
否定“已达到预定质量”，只是再次防止把达标误叫成收敛。

聚合报告至少按 family 和 rho 分别给出达标数、资源边界前达标数、首次达标 work 的原始值和中位数。
不用只报一个总平均来掩盖单条失败。

## 10. 执行顺序与失败后动作

顺序固定为：

```text
1. 用户审查并确认本协议（已完成）
2. 实现固定 runner 和纯确定性契约测试，但不运行 12 条正式轨迹
3. 审查 runner、测试、协议 SHA-256 和只读 plan
4. 用户再次授权后，只运行一次固定 12 轨迹矩阵
5. 冻结完整结果和分类，再讨论是否有资格进入在线接入设计
```

第 1 步已经完成，当前只执行第 2、3 步；未经第 4 步确认，不实例化正式 seed。
若候选科学门禁失败，本版结束；不使用本次结果修补阈值或重跑一个更有利的矩阵。
