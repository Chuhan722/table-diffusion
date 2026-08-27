# Issue #53 V2b：自适应有效证据人工验收协议

> 状态：用户已于 2026-08-16 接受为结果前固定协议；V2b 研究核心、固定 runner、独立 auditor 与
> 测试均已实现并通过回归。尚未生成正式协议 seed，尚未运行正式人工矩阵。
> 本协议只验收标量 ESS/MCSE 的自适应数值门禁，不读取真实数据，不判断平稳或收敛，不决定生成停止。

## 1. 这次必须回答什么

V2 已证明统一固定下限在既有人工压力范围内需要 2048 round。V2b 不修改该结论，而是检验下面这个
新候选：

```text
在 256..2048 的固定检查点顺序查看短、长两种批长，
能否让弱相关轨迹更早获得可靠 ESS/MCSE，
同时不让强相关轨迹因为某次随机尺度一致而过早放行？
```

本协议必须同时回答两类问题：

```text
安全：第一次通过时，正式 MCSE 是否仍有正确覆盖率？
成本：把未通过轨迹也按 2048 计入后，是否仍有明确 round 节省？
```

安全门禁失败时，不能用节省轮数补偿；成本门禁失败时，也不能把“数值上没出错”包装成自适应方案
成功。

## 2. 唯一候选规则

本协议只验收已获用户接受的 V2b 设计，不并行比较其他规则：

```text
checkpoints = [
  256, 384, 512, 640, 768, 896, 1024, 1152,
  1280, 1408, 1536, 1664, 1792, 1920, 2048,
]

short_batch = floor(sqrt(n))
long_batch  = 2 * short_batch

short_inflation = max(1, short_LRV / sample_variance)
long_inflation  = max(1, long_LRV  / sample_variance)

scale_ratio = max(short_inflation, long_inflation)
              / min(short_inflation, long_inflation)

通过当前检查点 = 两个尺度均可计算，并且 scale_ratio <= 1.25

official_inflation = max(short_inflation, long_inflation)
official_ESS        = n / official_inflation
official_MCSE       = sqrt(sample_variance * official_inflation / n)
```

每条轨迹只把日程中第一次通过的位置记为：

```text
first_adaptive_numerically_estimable_round
```

若 15 个检查点都不通过，该字段为 `null`，原因为
`resource_cap_without_multiscale_evidence`。不得把 2048 本身当作自动通过条件。

禁止在正式运行中增加检查点、跳过不利检查点、改成连续逐轮检查、把 `1.25` 改成其他阈值、平均两个
尺度，或在结果后选择更有利的尺度。

## 3. 固定随机协议

```text
随机数生成器：NumPy Generator + PCG64
每类重复次数：2000
每条轨迹长度：2048
每条轨迹检查次数：15
family 数：5
总轨迹数：10000
总检查点分类数：150000
总重叠批均值尺度计算数：300000
```

每条 2048-round 轨迹只生成一次，15 个检查点读取同一条轨迹的连续前缀，不为不同检查点另抽随机
数。即使某条轨迹提前通过，正式人工 runner 仍生成完整 2048 前缀并计算全部检查点，便于审计后续
状态；主判定始终只使用第一次通过的位置。

全新 seed 唯一固定为：

```text
SeedSequence([53, 2, 2, family_code, repeat_index])
repeat_index = 0, 1, ..., 1999
```

第三个整数 `2` 表示 V2b 新协议；该五元组与 V2 的
`SeedSequence([53, 2, family_code, repeat_index])` 不同。禁止使用 Python `hash()`、时间戳或当前
进程身份派生正式 seed。

固定 `family_code`：

```text
0 = iid，等价于 AR(1) phi=0
1 = AR(1), phi=0.5
2 = AR(1), phi=0.8
3 = AR(1), phi=-0.5
4 = AR(1), phi=0.95
```

## 4. 人工轨迹与理论值

五类轨迹都从平稳分布启动，真实均值为 0、单 round 方差为 1：

```text
x[0] 从标准正态分布抽取
x[t] = phi*x[t-1] + sqrt(1-phi*phi)*epsilon[t]
epsilon[t] 独立服从标准正态分布
```

不设置 burn-in，也不挑选初态。理论对照固定为：

| family | 理论相关膨胀 | 理论 raw ESS/n | 角色 |
|---|---:|---:|---|
| iid | 1 | 1 | 正式主判定 |
| `phi=0.5` | 3 | 1/3 | 正式主判定 |
| `phi=0.8` | 9 | 1/9 | 正式主判定最强相关 |
| `phi=-0.5` | 1/3 | 3 | 负相关保守边界 |
| `phi=0.95` | 39 | 1/39 | 超出 V2 正式范围的慢相关压力 |

`phi=0.95` 不要求必须在 2048 前通过；它专门检查两个尺度会不会一起漏掉更慢相关。安全拒绝是允许
结果，少量随机放行后不验收不是允许结果。

## 5. 每条轨迹记录什么

每个 family、repeat 和 checkpoint 必须记录或可由固定 seed 独立重算：

```text
n
short_batch / long_batch
sample_mean / sample_variance
short_LRV / long_LRV
short_raw_inflation / long_raw_inflation
short_inflation / long_inflation
scale_ratio
official_inflation / official_ESS / official_MCSE
adaptive_numerically_estimable / reason
stationarity_not_assessed
```

每条轨迹另记录：

```text
first_adaptive_numerically_estimable_round，或 null
resource_round_count
first_ready_coverage，若未通过则为 null
first_ready_official_LRV_ratio，若未通过则为 null
```

其中：

```text
resource_round_count = first_adaptive_numerically_estimable_round
                       （若始终未通过，则固定为 2048）

first_ready_coverage = 真均值 0 是否落在
                       sample_mean ± 1.96 * official_MCSE

first_ready_official_LRV_ratio
  = sample_variance * official_inflation / 理论长期方差
```

`resource_round_count=2048` 可能表示恰好在 2048 首次通过，也可能表示到上限仍未通过；报告必须用
first-ready 字段和 reason 区分，不能混为一类。

## 6. 正式主判定：安全门禁

正式主判定 family 固定为 `iid / phi=0.5 / phi=0.8`。每类 2000 条轨迹都必须分别满足以下条件：

### 6.1 足够多轨迹能在上限内给出数值证据

```text
first-ready 非 null 的轨迹数 >= 1850 / 2000
```

即至少 `92.5%`。该门禁防止候选通过“几乎全部拒绝报告”来获得虚假的安全性；它是本协议结果前
冻结的实用性要求，不是批均值理论常数。

### 6.2 第一次通过时的 MCSE 覆盖率

只在该 family 的 first-ready 非 null 轨迹中计算：

```text
覆盖率 = first_ready_coverage=true 的数量 / first-ready 非 null 数量
要求：92.5% <= 覆盖率 <= 97.5%
```

不能使用 2048 固定前缀覆盖率替代，不能把未通过轨迹计作覆盖，也不能在后续更好看的检查点重新
计算主指标。

### 6.3 第一次通过时的长期方差量级

该 family 所有 first-ready 非 null 轨迹的
`first_ready_official_LRV_ratio` 中位数必须满足：

```text
0.80 <= 中位数 <= 1.25
```

该范围沿用 V2 已在结果前冻结的乘法容差。正式值采用两个尺度的最大不确定性，因此不得改用较小
尺度、两个尺度平均或 raw 值来通过此门禁。

### 6.4 相关越强，单位 round 的正式证据不能越多

分别在三类轨迹的 first-ready 位置计算 `official_ESS/n` 中位数，必须严格保持：

```text
iid > phi=0.5 > phi=0.8
```

三类 first-ready 轮数可以不同；排序比较的是各自单位 round 的正式证据比例，而不是 ESS 总数。

## 7. 正式主判定：成本门禁

成本计算必须把 first-ready 为 `null` 的轨迹按完整 2048 round 计入，禁止只对提前通过者求平均。

三个固定门禁为：

```text
iid 的 resource_round_count 中位数       <= 512
phi=0.5 的 resource_round_count 中位数   <= 1024
iid/phi=0.5/phi=0.8 共 6000 条等权合并后：
  resource_round_count 平均值             <= 1536
```

前两条要求弱相关和中等相关不能仍普遍拖到上限；第三条要求在固定三类人工压力等权口径下，相对所有
轨迹固定跑 2048，平均至少节省 25% outer rounds。该等权人工口径不声称代表未来真实数据分布，只用
来判断 V2b 是否完成了本次“避免所有轨迹统一 2048”的设计目标。

墙钟时间、每类 first-ready 的最小值、四分位数、中位数、均值、95% 分位数、2048 首次通过比例和
到上限仍未通过比例全部报告，但不得用墙钟或某个辅助分位数翻转上述固定门禁。

## 8. 负相关控制

`phi=-0.5` 不参与主成本聚合，也不要求正式 ESS 利用负相关获得超过 n 的收益。必须满足：

- 15 个检查点的 short/long raw ESS/n 中位数都大于 1；
- 每条轨迹、每个检查点的 short/long/official 正式 ESS/n 都不得大于 1；
- 所有 LRV、ESS、MCSE 和 scale ratio 均为有限数；
- first-ready 只能是固定日程成员或 `null`；
- 正式 MCSE 始终不小于普通独立样本标准误。

负相关 first-ready 分布只报告，不进入主成本通过条件。

## 9. `phi=0.95` 慢相关压力门禁

记 `K95` 为 2000 条 `phi=0.95` 中 first-ready 非 null 的数量。只允许以下两个安全分支：

### 9.1 完全拒绝分支

```text
K95 == 0
```

说明在当前 2048 资源上限内一致 fail closed。该分支通过慢相关压力门禁，但不能声称 V2b 支持
`phi=0.95`。

### 9.2 可验收放行分支

```text
K95 >= 1000
并且 first-ready 覆盖率位于 92.5%..97.5%
并且 first-ready official LRV ratio 中位数位于 0.80..1.25
```

这里要求至少半数轨迹放行，保证覆盖率不是根据极少数偶然样本计算。该分支通过时，只能说在本人工
压力中安全支持，不能外推到更强相关或非平稳过程。

下面的中间状态固定判为候选失败：

```text
1 <= K95 < 1000
```

原因是同一种慢相关过程只偶发放行，既没有一致 fail closed，也没有足够样本证明放行安全。不得删除
这些轨迹、把它们降为纯观察项或在结果后降低 1000 门槛。

## 10. 全局数值与契约门禁

五类、全部重复、全部检查点必须同时满足：

- 人工连续高斯轨迹上 `core_not_estimable` 总数必须为 0；慢相关的安全拒绝只能来自两个可计算尺度
  的 `multiscale_disagreement`，不能用数学核心失效冒充安全拒绝；
- 没有未声明异常、`NaN`、Infinity 或负方差；
- `short_batch=floor(sqrt(n))`，`long_batch=2*short_batch`，且二者均小于 n；
- `official_inflation` 逐次等于 short/long 正式膨胀的最大值；
- `official_ESS <= n`，正式 MCSE 与同一个 official inflation 对拍；
- 输入轨迹和 round 身份未被函数修改；
- schema 固定 `stationarity_not_assessed=true`；
- 不存在 stable、converged、qualified、stop、stop_round、quality_pass、threshold 或同义字段；
- first-ready 只由第一次 `scale_ratio <= 1.25` 且两尺度可计算的位置产生；
- 到 2048 未通过时必须保留 `resource_cap_without_multiscale_evidence`。

任一违规都使正式候选失败，不能用总体比例掩盖。

## 11. 固定确定性边界测试

实现后、生成任何正式随机数之前，至少必须通过：

| 输入或边界 | 必须结果 |
|---|---|
| 手工短数组 + 指定短/长批长 | 两个 OBM 长期方差均可独立手算复核 |
| `scale_ratio == 1.25` | 当前检查点通过 |
| `scale_ratio = nextafter(1.25, +inf)` | 当前检查点不通过 |
| 两尺度任一不可估计 | 整体 fail closed，不能用另一尺度单独通过 |
| short 风险较大 | official inflation 精确取 short |
| long 风险较大 | official inflation 精确取 long |
| round 数不在固定检查点 | 顺序分类器拒绝，研究尺度核心仍只读计算 |
| 2048 首次相容 | first-ready=2048，不能误记为资源失败 |
| 2048 仍不相容 | first-ready=null，固定资源失败 reason |
| 整体平移 | 两尺度 inflation、scale ratio、ESS 不变，MCSE 不变 |
| 整体乘正数 | inflation、scale ratio、ESS 不变，MCSE 同比例变化 |
| 完全常数 | `zero_round_variance`，整体 fail closed |
| 周期偶合 | 任一尺度伪零时整体 fail closed |
| 单点尖峰 | 所有正式值有限且 ESS 不超过 n |
| 线性趋势 | 仍无稳定、收敛、质量或停止字段 |
| 输入含缺号、重复、乱序、bool、NaN、Infinity | 拒绝输入 |

平移、缩放与手工公式对拍使用相对/绝对容差 `1e-10`。阈值边界必须使用 `numpy.nextafter`，不另加
隐藏 epsilon。

若实现需要把现有 V2 的 OBM 公式抽成可指定批长的内部 helper，现有
`compute_v2_effective_round_evidence` 的公开行为、25 项确定性测试以及已归档 V2 人工结果必须保持
不变；V2b 不得静默重写 V2。

## 12. 结果状态

只有第 6、7、8、9、10、11 节全部通过，才允许：

```text
status = candidate_supported
```

任一科学门禁失败统一为：

```text
status = candidate_failed
failed_gates = [全部失败项]
```

入口、环境、上游哈希、任务完整性或独立审计失败时，不得产生科学结论：

```text
status = run_invalid
```

`candidate_supported` 也只支持“V2b 标量自适应数值门禁可进入下一研究层”，不支持真实轨迹平稳、
生成收敛、质量合格、在线停止或默认参数修改。

## 13. 固定入口与产物要求

未来实现入口固定建议为：

```text
scripts/validate_issue53_v2b_adaptive_effective_evidence.py plan
scripts/validate_issue53_v2b_adaptive_effective_evidence.py run --output-dir <全新目录>
```

`plan` 只打印固定矩阵和判定规则，不实例化 RNG。`run` 只允许指定一个不存在的新输出目录；seed、
重复次数、family、检查点、批长、阈值、覆盖率、成本门禁均不可通过 CLI 或环境变量覆盖。

正式运行前必须满足包含 untracked 在内的干净工作树。manifest 至少绑定：

```text
Git commit
本设计稿与本协议 SHA-256
V2/V2b 数学核心、runner、测试 SHA-256
Python / NumPy / OS / CPU 环境
固定 seed、family、检查点和全部门禁
```

正式产物至少包括：

```text
protocol_manifest.json
每条轨迹 first-ready 记录
family × checkpoint 聚合表
安全/成本/压力/边界门禁逐项结果
scientific result SHA-256
完整墙钟与任务计数
```

还必须提供一个不导入正式 runner 的独立 auditor：重新生成固定轨迹，独立重算两个 OBM 尺度、
first-ready、覆盖率、长期方差比、成本门禁、压力分支与最终 status。

## 14. 正确执行顺序

```text
用户审查本协议
→ 实现独立 V2b 研究核心与确定性测试
→ 实现固定 runner 与独立 auditor
→ Issue #53 相关回归和全仓回归
→ 干净预运行 commit 锁定代码与协议
→ plan 只读核对
→ 一次正式人工矩阵
→ 独立 audit
→ 结果文档与用户审查
```

本协议本身不授权正式人工运行或读取真实轨迹；本轮由用户另行授权的工具实现与测试已经完成。正式
人工结果产生前，不得读取 nltcs/test 的真实 development/validation 轨迹，不得接入生成器，不得
修改 residual geometry、alpha、rho、mu、eta、Gibbs 核或隐私预算，也不得创建在线停止器。

当前执行位置停在“实现与全仓回归完成”之后、“干净预运行 commit”之前。只读 `plan` 已核对为
5 个 family、10000 条轨迹、150000 次检查点分类与 300000 次尺度估计，且明确
`generation_started=false`；这次核对没有实例化正式 RNG。
