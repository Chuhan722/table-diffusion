# Issue #53：P=6 未见轨迹质量—计算验收协议

> 状态：v1，结果前冻结，2026-08-17。
>
> 本协议只评价当前无门控残差引导生成核中的 terminal-current A/B/C 停止层。它不选择 rho、alpha、
> residual geometry、eta、mu 或 Gibbs，不实现外层查询选择、加噪、隐私预算与 DP。协议写完前没有
> 运行下述 family 或 seed；看见结果后不得修改本页阈值并仍把同一批结果称为验证。

## 1. 只回答一个问题

固定全局候选 `P=6` 后，B 是否能在不同公开人工 family 和两种更新尺度下，以可接受的 terminal-current
质量代价换取足够计算节省。

本协议不要求 B 证明未来永不改善，也不把 C 的 terminal 当作质量参考。在线输出仍固定为：

```text
A/B/C output = 触发状态的 terminal current table
historical best = 只读 progress/诊断，不参与选表
```

## 2. 冻结的在线规则

```text
natural work = cumulative applied participating rows / N
work tick = floor(natural work)

A：current squared loss == 0
B：连续 P=6 个已完成 work tick 没有严格历史 best 刷新
C：外部 raw-round/candidate-evaluation 护栏
优先级：A > B > C
```

启用在线停止时固定 `tol=+inf`、`max_retries=0`，每个有限 proposal 都成为 current。L1、reference
table、影子续跑和未来状态不得进入在线判断。

## 3. 两个未见公开人工 family

两个 family 均只由下述公开多重集产生精确整数 target。reference multiset 只用于在运行前计算固定
query target；生成器只接收 schema、ordered queries 与 target，不接收 reference table。

### 3.1 U1：`binary_chain_4`

```text
N = 32
schema = a,b,c,d，均为 categorical {0,1}

reference state (a,b,c,d) : count
0000:6  0001:2  0010:2  0011:2
1100:2  1101:2  1110:2  1111:6
0101:2  1010:2  0110:2  1001:2
```

ordered queries 与 target：

```text
a=1                         16
b=1                         16
c=1                         16
d=1                         16
a=1,b=1                     12
b=1,c=1                     10
c=1,d=1                      8
a=1,d=1                     10
a=1,b=1,c=1                  8
b=1,c=1,d=1                  6
a=1,b=1,c=1,d=1              6
```

### 3.2 U2：`mixed_2x3x2`

```text
N = 36
schema = x categorical {0,1}; y categorical {0,1,2}; z categorical {0,1}

reference state (x,y,z) : count
000:6  001:2  010:3  011:1  020:1  021:5
100:2  101:4  110:1  111:5  120:4  121:2
```

ordered queries 与 target：

```text
x=1                         18
y=0                         14
y=1                         10
y=2                         12
z=1                         19
x=1,y=0                      6
x=1,y=1                      6
x=1,y=2                      6
y=0,z=1                      6
y=1,z=1                      6
y=2,z=1                      7
x=1,z=1                     11
x=1,y=0,z=1                  4
x=1,y=1,z=1                  5
x=1,y=2,z=1                  2
```

实现 family 时必须用纯测试复算 `N`、ordered target 和查询顺序；任何定义修正都必须发生在首次运行
这些 seed 之前，并在状态记录中留下原因。

## 4. 冻结的 case 矩阵

首轮验证 seed：

```text
20260819, 20260820, 20260821
```

每个 family 对每个 seed 分别运行：

```text
rho = 1.0
rho = 0.25
```

总数为 `2 family × 3 seed × 2 rho = 12 cases`。已经看过的 20260816—20260818 不得计入验证。

除 family、N、seed、rho 外，生成参数固定为：

```text
init_method = random
eta = 0.45
mu = 0.02
distance_mode = geometric
alpha_schedule_mode = fixed
fixed_alpha = 6.0
residual_directed_diffusion = true
diffusion_direction_strength = 0.8
diffusion_direction_normalization = fixed
diffusion_direction_reference_scale = 1.25
diffusion_direction_logit_clip = 9.0
factorized_gibbs_sweeps = 0
residual_self_cooling = off
tol = +inf
max_retries = 0
device = numpy
P = 6
```

这些参数不是本协议认可的全局最优值；固定它们只是隔离停止规则，禁止在看到本批结果后同时调整
生成核和 P。

## 5. C 只作防挂死护栏

C 以“预期 60 normalized work”的统一外部计算额度换算，不按 family 或结果变化：

```text
rho=1.0  -> n_rounds=60，candidate_budget=60
rho=0.25 -> n_rounds=240，candidate_budget=240
```

因为 participating rows 是随机量，实际 terminal work 不必恰好等于 60。该数字不是拟合点、质量终点
或 B 的参考答案。触发 C 的 case 标记 `inner_complete=false`，只计为资源截断；其 terminal 质量不得
拿来判 B 通过。

## 6. 影子续跑与离线指标

对 A：当前已精确达到零噪声 target，固定 A 输出，不做 B 后续比较。

对 B：先冻结在线 B 的 terminal current 表与 RNG 身份，再由只读影子副本沿同一 RNG 前缀定位：

```text
tau + 6 normalized work
tau + 12 normalized work
```

每个检查点取第一个真实 current 状态使累计 work 达到或越过目标，不插值、不返回 best。若 C 前没有
观察到检查点，则记为 right-censored，不用 C terminal 或最后可用状态补齐。

只有表身份固定后才离线计算 ordered measured-query normalized L1：

```text
delta_L1(k) = L1_at_B - L1_at_tau_plus_k
```

`delta_L1>0` 表示 B 提前停止的质量更差；`delta_L1<0` 表示 B 的 terminal current 反而更好。current
squared loss 同时报告，但因两个 family 的查询数不同，不作为跨 family 的硬验收阈值。另报告 stop
work、raw rounds、candidate evaluations 与墙钟；墙钟只作诊断，不作为跨机器硬门禁。

## 7. P=6 的固定通过条件

以下条件必须同时满足：

1. **正常结束覆盖率**：12 cases 中至少 10 条由 A 或 B 结束；C 不得超过 2 条。
2. **B 证据完整性**：至少观察到 6 条 B；在 `+6` 与 `+12` 各自至少 80% 的 B cases 有可用检查点。
   因第 4 条还要求逐 family 中位数，所以每个检查点还必须在两个 family 中分别至少有一条可用 B；
   任一 family 完全右删失时其中位数无定义，同样只能判“证据不足”，不能算通过或质量失败。
3. **聚合质量非劣**：在 `+6` 和 `+12` 两个检查点分别计算，`delta_L1` 中位数都不得超过 `0.01`。
4. **大退化尾部**：在每个检查点，`delta_L1>0.02` 的比例都不得超过 25%；并且两个 family 各自的
   `delta_L1` 中位数都不得超过 `0.02`，防止总体均值掩盖单一 family 失败。
5. **计算收益**：对具有 `+12` 检查点的 B cases，定义
   `saving_12 = 12 / (stop_work + 12)`；其中位数必须至少为 30%。

`0.01/0.02` 分别表示平均每个查询相差不超过数据规模的 1%/2%，不依赖 N 的绝对大小。它们是本轮
结果前冻结的工程非劣界，不是理论收敛常数或未来带噪阶段的通用阈值。

满足全部条件，只能写作：

```text
P=6 在当前两个人工 family 的未见轨迹上通过 development 质量—计算验收。
```

不得写作“算法收敛”“P=6 全局最优”“对真实数据已验证”或“未来带噪阶段自动成立”。

## 8. 不通过后的唯一分流

首轮结果只能进入以下一个分支，不允许看完结果后任意扫描 P：

### 8.1 只有质量条件失败，计算收益与证据完整性通过

唯一允许的新候选是 `P=12`。较大 P 只是多等待的候选，不保证 terminal current 单调改善。

### 8.2 只有计算收益失败，质量条件与证据完整性通过

唯一允许的新候选是 `P=4`。较小 P 只是更早停止的候选，不能预设质量仍会通过。

### 8.3 质量与计算同时失败，或两个 family 需要相反方向

当前 best-refresh patience B 判为结构性不合格；停止调整 P，返回重新设计 B 的在线信号。不得继续
试 5、7、8、9 等值。

“两个 family 需要相反方向”在看结果前固定解释为：逐 family 分别用第 7 节的 `0.02` family 质量界
与 `30%` 计算收益界作方向诊断；质量失败但计算通过指向增大 P，质量通过但计算失败指向减小 P。若
一个 family 指向增大、另一个指向减小，直接拒绝当前 B，不允许用全局聚合结果选择单一回退。该逐
family 方向只用于识别此冲突，不替代第 7 节的全局质量和计算门禁。

### 8.4 C 太多、B 少于 6 条或检查点覆盖不足

结论固定为“证据不足/资源截断”，不调整 P。先单独审查 C 与观察范围；不得把删失当作质量失败。

## 9. 唯一一次 P 回退的独立数据

若且仅若进入 8.1 或 8.2，使用预先冻结的第二批 seed：

```text
20260822, 20260823, 20260824
```

family、rho、生成参数、C、检查点和第 7 节门禁全部不变，只替换 seed 和按分支唯一确定的 P。首轮
12 cases 从此只算 development 证据；不得拿它们重新评价新 P。

第二批若通过，则只接受该唯一回退 P 的当前阶段 development 结论；若仍不通过或证据不足，停止
P 调整并重新设计 B。协议不允许第三批 P 值或在相同 seed 上反复调到通过。

## 10. 实施与停止边界

确定性 manifest 与纯校验已于 2026-08-17 实现：

```text
scripts/issue53_p6_unseen_protocol.py
tests/test_issue53_p6_unseen_protocol.py

protocol SHA-256:
759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317

U1 family SHA-256:
c47200c0b68c6c3bcf4818b7b9322f85666584eaa1459d94a19d216642f447ee

U2 family SHA-256:
db3af48d083e1e4905a16362b63ba4bbbe7c55045efd3ae6e6a580f82a58bbab
```

协议入口只支持 `--mode plan`，不导入生成器，也没有运行模式。12 项纯测试逐项锁定 U1/U2 的 schema、
reference state/count、N、查询顺序和 target，首轮 12-case 顺序与 case ID，C 的 60/240 映射，两条互斥
回退分支及其独立 seed，验收阈值、失败分流和上述协议哈希。manifest 每次重建为新对象；未知 rho、
未知回退分支或内容/身份不一致均 fail closed。

本步只运行上述纯测试，没有执行 generator、12 cases 或单条轨迹，没有预览 loss/L1，也没有读取真实
数据或使用 GPU。

受协议 SHA 强制约束的原始采集入口已于 2026-08-17 实现：

```text
scripts/collect_issue53_p6_unseen.py
tests/test_collect_issue53_p6_unseen.py
collection contract = issue53-p6-unseen-primary-collection-v1
```

`plan` 只打印固定 12 cases 与最坏开销，不实例化 RNG；`collect` 必须显式传入上述 protocol SHA、要求
包含 untracked 在内的干净工作树，并拒绝覆盖已有输出目录。命令行没有 family、seed、rho、P、C、
检查点或阈值覆盖参数。正式 execution manifest 会记录 Git commit、协议全文、关键源码和本页的文件
哈希以及 Python/NumPy/Pandas 环境。

每条 case 先运行在线 terminal-current A/B/C。A/C 只保存 terminal table 和在线诊断；只有 B 才从同一
S0/seed 重新执行关闭停止器的只读影子轨迹到固定 C，并要求 online 与 shadow 在 B terminal 之前的
current metrics、transition clocks、接受序列、table/query/RNG 身份和 candidate count 全部一致。然后
只按第 6 节定位 +6/+12 的第一个真实状态；不插值，未观察到则右删失。reference multiset 不传入
generator。

采集入口只原子保存逐 case 原始 artifact 和集合 manifest，不计算 `delta_L1`、聚合门禁、通过/失败或
回退分支，控制台也不打印单条 loss/L1。最坏上限为 24 次 generator 调用、3600 raw rounds；实际只对
B 增加 shadow。8 项假执行器契约测试覆盖 SHA 前置失败、干净树门禁、固定 kwargs、12-case 编排、
terminal 落盘、B 前缀审计、+6/+12 定位与右删失；没有调用真实 generator。

受同一协议 SHA 约束的只读审计与唯一分流判定器已于 2026-08-17 实现：

```text
scripts/evaluate_issue53_p6_unseen.py
tests/test_evaluate_issue53_p6_unseen.py
evaluation contract = issue53-p6-unseen-evaluation-v1
```

`plan` 不读取 collection；`evaluate` 只有 collection 路径与完整 protocol SHA 两个输入，不开放阈值或
分流覆盖参数。正式判定要求包含 untracked 在内的干净工作树、与 collection 完全相同的 Git commit
和 Python/NumPy/Pandas runtime，并拒绝覆盖已有报告。它先逐级核对 collection/execution/case
manifest、源文件和 artifact SHA、冻结 case/family/query/target 身份，重新读取 terminal CSV 并复算
query vector、squared loss 与 normalized L1。对 B 还会从保存的 current-loss/自然工作时钟重新执行
A/B/C 状态机，核对无门控单次 proposal
直接生效、历史 best、RNG/candidate 前缀、shadow trace/summary 和 +6/+12 检查点；A/C 不允许夹带
shadow。

判定顺序固定为“artifact 完整性 -> 证据完整性 -> 质量/计算 -> 唯一分流”。证据不足时不计算质量或
计算通过/失败，右删失值不补齐；证据完整后才可能得到 P=6 当前 development 支持、唯一 P=12、唯一
P=4 或拒绝 B 四类行动。报告明确限制为两个人工 family 的 development 结论，禁止声称收敛，且最多
一次回退、禁止第三个 P 和结果后重调阈值。

22 项 evaluator 假证据/假 artifact 测试覆盖五种分类、证据不足、逐 family 方向冲突、`>0.02` 严格
边界、右删失不插补、A 优先级、A/B/C 离线复算、完整 A/B artifact 审计、路径逃逸、文件篡改、错误
SHA、dirty tree、runtime 漂移和报告拒绝覆盖。协议、collector、evaluator 合计 42 项测试通过；全部
使用伪造矩阵或假执行器，没有调用真实 generator。

提交前的 Issue #53 当前链路相关回归为 `99 passed`；全仓 CPU 回归为 `1591 passed, 7 skipped`，仅有
两条既有 residual-geometry 错误路径 NumPy warning，零失败。上述状态机、接线、协议、collector、
evaluator、文档和测试作为同一冻结提交落盘，使正式 clean-tree 门禁可以满足。

当前仍未执行 12 cases，没有预览任何未见 loss/L1。下一步仍需用户单独明确确认完整 protocol SHA，
才允许执行 primary collection；采集完成后还要另行授权 evaluate。不得改 P、阈值、family、seed 或
C，也不得把提交、采集和判定自动串成一步。
