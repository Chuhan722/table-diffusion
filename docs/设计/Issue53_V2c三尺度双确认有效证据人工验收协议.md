# Issue #53 V2c：三尺度双确认有效证据人工验收协议

> 状态：用户于 2026-08-16 接受为 V2c 唯一结果前人工验收协议；V2c 唯一候选设计已由 commit
> `c274221` 冻结。本次只授权实现独立研究核心与确定性测试，不授权 runner、auditor 或任何实验。
> 尚未生成 `[53,2,3,...]` seed，尚未读取或运行任何真实数据。

## 1. 本次必须回答什么

V2b 正式失败已经证明：两个批长的一次相容不足以保证长期方差可靠。V2c 只检验一个新候选：

```text
同一前缀的 b、2b、4b 三个 OBM 尺度均相容，
并且相邻两个固定检查点连续相容，
是否足以让弱/中等相关轨迹较早获得可靠 ESS/MCSE，
同时对强慢相关轨迹安全放行或一致 fail closed？
```

必须同时验收：

```text
安全：第一次取得数值资格时，正式 MCSE 覆盖率和 LRV 量级是否正确？
成本：把始终未取得资格的轨迹按 2048 计入后，是否仍节省 round？
压力：phi=0.95 是否会因三个尺度共同低估或顺序偶然而被不安全放行？
```

安全失败不能由成本补偿。候选成功也只支持标量 ESS/MCSE 数值门禁，不支持平稳、收敛、质量或
生成停止结论。

## 2. 唯一候选公式

固定检查点为：

```text
256, 384, 512, 640, 768, 896, 1024, 1152,
1280, 1408, 1536, 1664, 1792, 1920, 2048
```

每个检查点 `n` 的三个批长固定为：

| n | b1 | b2 | b3 |
|---:|---:|---:|---:|
| 256 | 16 | 32 | 64 |
| 384 | 19 | 38 | 76 |
| 512 | 22 | 44 | 88 |
| 640 | 25 | 50 | 100 |
| 768 | 27 | 54 | 108 |
| 896 | 29 | 58 | 116 |
| 1024 | 32 | 64 | 128 |
| 1152 | 33 | 66 | 132 |
| 1280 | 35 | 70 | 140 |
| 1408 | 37 | 74 | 148 |
| 1536 | 39 | 78 | 156 |
| 1664 | 40 | 80 | 160 |
| 1792 | 42 | 84 | 168 |
| 1920 | 43 | 86 | 172 |
| 2048 | 45 | 90 | 180 |

即：

```text
b1 = floor(sqrt(n))
b2 = 2*b1
b3 = 4*b1
```

每个尺度使用现有 V2 的同一重叠批均值公式：

```text
raw_inflation_j    = LRV_j / sample_variance
formal_inflation_j = max(1, raw_inflation_j)
```

三个尺度均可计算时：

```text
scale_ratio = max(formal_inflation_1,
                  formal_inflation_2,
                  formal_inflation_3)
              / min(formal_inflation_1,
                    formal_inflation_2,
                    formal_inflation_3)

three_scale_compatible = scale_ratio <= 1.25

official_inflation = max(formal_inflation_1,
                         formal_inflation_2,
                         formal_inflation_3)
official_LRV       = sample_variance * official_inflation
official_ESS       = n / official_inflation
official_MCSE      = sqrt(official_LRV / n)
```

任一尺度不可计算时，`three_scale_compatible=false`。不得删除第三尺度、平均三个尺度、只选更好看的
两个尺度，或在结果后修改 `1.25`。

## 3. 双确认状态与 reason

设固定日程第 `k` 个检查点的三尺度相容状态为 `C_k`。当前数值资格状态固定为：

```text
A_0 = false
A_k = C_(k-1) AND C_k，k >= 1
```

公开字段继续使用：

```text
adaptive_numerically_estimable = A_k
stationarity_not_assessed = true
```

不得新增 stable、converged、qualified、stop、stop_round、threshold、quality_pass 或同义决策字段。

当前检查点未取得资格时，reason 只能是：

```text
core_not_estimable
  三个尺度至少一个不可计算

multiscale_disagreement
  三个尺度均可计算，但当前 scale_ratio > 1.25

nonfinite_computation
  三个尺度各自可计算，但合并 scale ratio、official LRV、ESS 或 MCSE 时发生数值溢出；整体
  fail closed

awaiting_consecutive_multiscale_evidence
  当前三个尺度相容，但这是第一个检查点，或前一个检查点不相容
```

取得资格时 reason 必须为 null。后续检查点可以重新变为 false；本状态可回撤，历史 first-ready 不得
覆盖当前状态。

每条完整轨迹按顺序记录第一次 `A_k=true` 的检查点：

```text
first_adaptive_numerically_estimable_round
```

若始终没有连续两个相容检查点，该字段为 null，轨迹 reason 固定为：

```text
resource_cap_without_consecutive_multiscale_evidence
```

`resource_round_count` 等于 first-ready；若 first-ready 为 null，则固定为 2048。恰好在 2048 首次
取得资格与到 2048 仍无资格必须严格区分。

## 4. 固定随机协议

```text
随机数生成器：NumPy Generator + PCG64
每类重复次数：2000
每条轨迹长度：2048
每条轨迹检查次数：15
每个检查点尺度数：3
family 数：5
总轨迹数：10000
总检查点分类数：150000
总 OBM 尺度计算数：450000
总人工标量生成数：20480000
```

每条轨迹只生成一次，15 个检查点读取同一条轨迹的连续前缀。即使较早取得资格，正式 runner 仍
计算全部 15 个检查点，以审计后续资格回撤；主安全与成本指标只使用第一次取得资格的位置。

正式 seed 唯一固定为：

```text
SeedSequence([53, 2, 3, family_code, repeat_index])
repeat_index = 0, 1, ..., 1999
```

该 namespace 不得与 V2b 的 `[53,2,2,...]` 混用。禁止使用 Python `hash()`、时间、进程身份、环境
变量或结果内容派生 seed。

## 5. 人工 family 与理论值

固定 family 不增不减：

| code | family | phi | 理论 LRV | 理论 raw ESS/n | 角色 |
|---:|---|---:|---:|---:|---|
| 0 | iid | 0 | 1 | 1 | 正式主判定 |
| 1 | AR(1) | 0.5 | 3 | 1/3 | 正式主判定 |
| 2 | AR(1) | 0.8 | 9 | 1/9 | 正式主判定最强相关 |
| 3 | AR(1) | -0.5 | 1/3 | 3 | 负相关控制 |
| 4 | AR(1) | 0.95 | 39 | 1/39 | 慢相关压力 |

所有轨迹从平稳边缘分布启动：

```text
x[0] ~ Normal(0,1)
x[t] = phi*x[t-1] + sqrt(1-phi^2)*epsilon[t]
epsilon[t] 独立服从 Normal(0,1)
burn-in = 0
```

真实均值为 0，单 round 方差为 1。不得挑初态、丢弃不利轨迹或为不同检查点重新抽样。

## 6. 每条轨迹必须记录什么

每个 family、repeat、checkpoint 必须记录或可由固定 seed 独立重算：

```text
n
b1 / b2 / b3
sample_mean / sample_variance
三个 LRV
三个 raw inflation
三个 formal inflation
scale_ratio
official inflation / LRV / ESS / MCSE
three_scale_compatible
adaptive_numerically_estimable
reason
stationarity_not_assessed
```

每条轨迹另记录：

```text
first_adaptive_numerically_estimable_round，或 null
resource_round_count
first_ready_coverage，或 null
first_ready_official_LRV_ratio，或 null
first_ready_official_ESS_ratio，或 null
first_ready 后是否曾再次 three_scale_compatible=false
first_ready 后不相容的检查点数量
2048 时的 three_scale_compatible 与 adaptive_numerically_estimable
```

其中：

```text
first_ready_coverage
  = abs(first_ready_sample_mean) <= 1.96 * first_ready_official_MCSE

first_ready_official_LRV_ratio
  = first_ready_official_LRV / theoretical_LRV
```

后续回撤只作预注册诊断，不是额外成功门禁，也不能用后续更好看的检查点替换 first-ready 主指标。

## 7. 主 family 安全门禁

正式主 family 固定为 iid、`phi=0.5`、`phi=0.8`。每类 2000 条必须分别满足：

### 7.1 数值资格利用率

```text
first-ready 非 null 数量 >= 1850
```

### 7.2 第一次取得资格时的 MCSE 覆盖率

只在 first-ready 非 null 的轨迹中计算：

```text
92.5% <= coverage <= 97.5%
```

未取得资格的轨迹不计作覆盖，不能用 2048 固定前缀或后续检查点覆盖率替换。

### 7.3 第一次取得资格时的长期方差量级

```text
0.80 <= median(first_ready_official_LRV_ratio) <= 1.25
```

### 7.4 相关越强，单位 round 的正式证据越少

三类 first-ready 的 `official_ESS/n` 中位数必须严格保持：

```text
iid > phi=0.5 > phi=0.8
```

任一安全门禁失败，候选失败。

## 8. 成本门禁

first-ready 为 null 的轨迹按完整 2048 round 计入。固定门禁保持 V2b 原值：

```text
iid resource_round_count 中位数               <= 512
phi=0.5 resource_round_count 中位数           <= 1024
iid/phi=0.5/phi=0.8 共 6000 条等权 pooled mean <= 1536
```

由于最早资格从 256 推迟到 384，这些门禁没有相应放宽。最小值、四分位数、中位数、均值、P95、
2048 首次取得资格数量和到上限仍无资格数量全部报告。墙钟只报告，不翻转门禁。

## 9. 负相关控制

`phi=-0.5` 不参与主成本聚合。必须满足：

- 15 个检查点的 b1/b2/b3 raw ESS/n 中位数均大于 1；
- 每条轨迹、每个检查点的三个 scale 与 official 正式 ESS/n 均不大于 1；
- 所有可声明的 LRV、inflation、ESS、MCSE 和 scale ratio 有限；
- 正式 MCSE 不小于普通独立样本标准误；
- first-ready 只能是 384..2048 的固定检查点或 null；
- 三尺度 compatible 与双确认逻辑逐项一致。

负相关 first-ready 分布与后续回撤只报告，不进入主成本门禁。

## 10. `phi=0.95` 慢相关压力

记 `K95` 为 2000 条慢相关轨迹中 first-ready 非 null 的数量。只允许两个安全分支：

### 10.1 完全拒绝

```text
K95 == 0
```

该分支只说明在 2048 上限内一致没有取得数值资格，不能声称支持 `phi=0.95`。

### 10.2 可验收确认

```text
K95 >= 1000
92.5% <= first-ready coverage <= 97.5%
0.80 <= median(first-ready official LRV ratio) <= 1.25
```

下面状态固定失败：

```text
1 <= K95 < 1000
```

`K95>=1000` 但覆盖率或 LRV 比失败也固定失败。不得因 V2b 中 `phi=0.95` 不利而删除、降级或降低
1000 门槛。

## 11. 后续回撤诊断

每个 family 必须报告：

```text
first-ready 后至少一次 three_scale_compatible=false 的轨迹数/比例
first-ready 后不相容检查点数的分布
2048 时 three_scale_compatible=true 的数量
2048 时 adaptive_numerically_estimable=true 的数量
```

这些指标用于回答双确认后仍有多少回撤。当前协议不为它们另设阈值，因为 first-ready 覆盖率已经
直接验收取得资格时的 MCSE，而公开数值状态本身可回撤。不得在结果后根据回撤率增加第三次确认或
把报告项临时升级为成功门禁。

## 12. 全局数值与契约门禁

五类、全部重复、全部检查点必须满足：

- `core_not_estimable` 总数为 0；
- 没有未声明异常、NaN、Infinity、负方差或非正 inflation；
- b1/b2/b3 精确等于结果前表格，且三者均小于 n；
- 每个 raw inflation 精确对应同尺度 `LRV/sample_variance`；
- 每个 formal inflation 精确等于 `max(1,raw)`；
- scale ratio 精确等于三个 formal inflation 的 max/min；
- official inflation 精确等于三个 formal inflation 的最大值；
- official LRV、ESS、MCSE 使用同一个 official inflation；
- 三个 scale 和 official ESS 均不超过 n；
- `C_k` 与 `A_k=C_(k-1) AND C_k` 逐检查点对拍；
- 第一个检查点 A 必为 false，first-ready 最早只能为 384；
- first-ready、resource count、当前状态、reason 与完整 15 点序列一致；
- 输入 round 身份和值未被修改；
- schema 固定 `stationarity_not_assessed=true`，无越权决策字段。

任一违规使候选失败。不得用总体违规比例掩盖。

## 13. 固定确定性边界测试

实现后、生成任何正式随机数之前，至少必须通过：

| 边界 | 必须结果 |
|---|---|
| 手工短数组 + 三个指定批长 | 三个 OBM LRV 均与独立手算一致 |
| 三尺度 ratio 恰好 1.25 | compatible=true |
| ratio=`nextafter(1.25,+inf)` | compatible=false |
| 任一尺度不可估计 | 三尺度整体 fail closed |
| b1/b2/b3 分别为最大风险 | official inflation 分别精确取对应尺度 |
| 非固定 checkpoint | 顺序分类器拒绝；显式批长 V2 核心仍只读可算 |
| `T,T` | 首次资格在第二个检查点 |
| `T,F,T` | 三点均无资格 |
| `T,F,T,T` | 首次资格在第四点 |
| 1920/2048 连续为 T | first-ready 可为 2048 |
| 2048 仍未双确认 | first-ready=null + 固定资源失败 reason |
| first-ready 后又 F | 当前资格回撤，历史 first-ready 不改变 |
| 整体平移 | inflation、ratio、ESS、MCSE 不变 |
| 整体乘正数 | inflation、ratio、ESS 不变，MCSE 同比例变化 |
| 完全常数 | `zero_round_variance`，整体 fail closed |
| 周期偶合 | 任一尺度伪零时整体 fail closed |
| 单点尖峰 | 正式值有限且 ESS 不超过 n |
| 线性趋势 | 无稳定、收敛、质量或停止字段 |
| 有限超大值导致溢出 | `nonfinite_computation`，整体 fail closed |
| 缺号、重复、乱序、bool、NaN、Infinity | 拒绝输入 |
| 原 V2/V2b 回归 | 公开行为和归档结果不变 |

平移、缩放和公式对拍使用相对/绝对容差 `1e-10`。`1.25` 边界使用 `numpy.nextafter`，不增加隐藏
epsilon。

## 14. 结果状态

只有第 7 至 13 节全部通过，才允许：

```text
status = candidate_supported
```

任一科学门禁失败：

```text
status = candidate_failed
failed_gates = [全部失败项]
```

入口、环境、上游哈希、任务完整性或独立审计失败时，不得产生科学结论：

```text
status = run_invalid
```

结果后禁止调整阈值、检查点、尺度、确认次数、family、重复数或门禁并复用本批 seed。若 V2c 正式
失败，本路线停止继续增加 V2d。

## 15. 固定入口、manifest 与独立审计

未来入口固定建议为：

```text
scripts/validate_issue53_v2c_three_scale_effective_evidence.py plan
scripts/validate_issue53_v2c_three_scale_effective_evidence.py run --output-dir <全新目录>
```

`plan` 只打印固定矩阵，不实例化 RNG。`run` 只允许一个不存在的新输出目录；不能由 CLI 或环境变量
覆盖任何科学参数。

正式运行前必须是包含 untracked 在内的干净工作树。manifest 至少绑定：

```text
Git commit
V2b 正式负结果 scientific SHA
V2c 设计稿与本协议 SHA-256
V2 显式批长核心、V2b 归档核心、V2c 核心 SHA-256
runner、独立 auditor、全部相关测试 SHA-256
Python / NumPy / OS / CPU 环境
固定 seed、family、检查点、三尺度、双确认和全部门禁
```

正式输出至少包括：

```text
protocol_manifest.json
每条轨迹 first-ready 与后续回撤记录
family × checkpoint 三尺度聚合
安全/成本/压力/契约/边界门禁结果
scientific result SHA-256
完整墙钟与任务计数
```

report 对 manifest 的绑定必须使用同目录相对身份与 SHA-256；不得把某台机器的绝对本地路径作为
换机器后无法复验的必要条件。正式结果提交时，原字节 manifest/report/audit 必须显式纳入版本控制，
或上传到长期可访问的 Release 并记录链接与 SHA-256；不能只留下 gitignored 本地路径。

必须提供不导入正式 runner、V2b 核心或 V2c 核心的独立 auditor：重新生成固定轨迹，直接实现三个
OBM 尺度，重算相容序列、双确认、first-ready、覆盖率、LRV 比、成本、回撤和最终 status，并与完整
科学 payload 精确对拍。正式结果在 audit `passed=true` 前不得解释。

## 16. 正确执行顺序与当前停止点

```text
用户审查本协议
→ 实现独立 V2c 研究核心与确定性测试
→ 实现固定 runner 与独立 auditor
→ Issue #53 相关回归与全仓回归
→ 干净预运行 commit
→ plan 只读核对
→ 一次正式 10000 轨迹矩阵
→ 独立 audit
→ 结果归档与用户审查
```

用户已审查并确认本协议准确落实“三尺度 + 连续两次”设计，当前只授权进入“独立 V2c 研究核心与
确定性测试”一步。完成并交用户审查前，不得继续实现固定 runner 或独立 auditor，不得生成正式
seed，也不得运行人工矩阵。

正式人工结果产生前，不得读取 `test_300x10`、`nltcs`、退休 validation seeds 或其他真实轨迹；
不得接表格生成器、在线停止器、DP accountant，也不得修改 residual geometry、alpha、rho、mu、
eta 或 Gibbs 核。
