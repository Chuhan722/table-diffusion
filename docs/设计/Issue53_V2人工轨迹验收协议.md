# Issue #53 V2：有效证据计数器人工轨迹验收协议

> 状态：研究版纯函数与固定边界测试已完成；人工重复实验尚未执行，历史下限尚未产生。
> 本协议只验证标量 ESS/MCSE 计数器；不读取真实数据，不判断收敛，不决定停止轮数。

## 1. 这一步要回答什么

只回答两个问题：

```text
1. 重叠批均值 + b=floor(sqrt(n)) 能否把相关 round 的重复信息识别出来？
2. 至少积累多少 round 后，这个数值估计器才允许开始报告 ESS/MCSE？
```

这里选出的最少历史长度只控制 `numerically_estimable`。它不是生成过程的正常停止轮数，也不能单独
产生 `stable`、`converged` 或 `stop`。

## 2. 本轮固定范围

- 只审查设计稿中的唯一候选：重叠批均值，`b=floor(sqrt(n))`；
- 不同时比较多个估计器后挑结果最好者；
- 不读取 nltcs、development、validation、历史正式实验或 GPU 输出；
- 全部轨迹由 CPU 人工生成，不消耗隐私预算；
- 本轮不修改 residual geometry、alpha、Gibbs 核或生成器。

文献把 `floor(sqrt(n))` 作为常见的简单默认方案，但也明确指出批长对有限样本结果很重要，并存在
需要额外估计常数的其他最优化方案。因此本协议只把它当作候选，不把它写成“理论最优”。

## 3. 固定随机协议

```text
随机数生成器：NumPy Generator + PCG64
每类重复次数：2000
轨迹最大长度：4096
观察前缀长度：16, 32, 64, 128, 256, 512, 1024, 2048, 4096
```

每条最长轨迹只生成一次，然后读取上述前缀。这样不同长度的比较来自同一条逐渐增长的历史，不为
每个长度另抽一批更有利的随机数。

每条轨迹的种子按下面的整数元组建立，禁止使用 Python `hash()`：

```text
SeedSequence([53, 2, family_code, repeat_index])
repeat_index = 0, 1, ..., 1999
```

固定 `family_code`：

```text
0 = 独立白噪声
1 = AR(1), phi=0.5
2 = AR(1), phi=0.8
3 = AR(1), phi=-0.5
```

运行结果必须记录协议版本、Git commit、Python/NumPy 版本和完整参数，确保能够复现。

## 4. 随机轨迹

### 4.1 统一生成方式

四类轨迹都保持真实均值为 0、单 round 方差为 1：

```text
x[0] 从标准正态分布抽取
x[t] = phi*x[t-1] + sqrt(1-phi*phi)*epsilon[t]
epsilon[t] 独立服从标准正态分布
```

独立白噪声等价于 `phi=0`。因为初始值已经从平稳分布抽取，所以不设置也不挑选 burn-in。

理论对照值固定为：

| 轨迹 | 理论相关膨胀倍数 | 理论 raw ESS/n |
|---|---:|---:|
| 独立白噪声 | 1 | 1 |
| `phi=0.5` | 3 | 1/3 |
| `phi=0.8` | 9 | 1/9 |
| `phi=-0.5` | 1/3 | 3 |

`phi=0.8` 是本协议预先声明的正相关压力边界，不代表已经覆盖任意更慢、更非平稳的真实轨迹。

### 4.2 每个长度计算什么

对每个 family、每次重复和每个观察长度，记录：

```text
长期方差估计值
raw ESS/n
正式保守 ESS/n
MCSE
真实均值 0 是否落入 sample_mean ± 1.96*MCSE
是否产生 NaN 或 Infinity
```

这里的 `MCSE` 是正式保守值，与正式 ESS 使用同一个“相关膨胀倍数至少为 1”的规则；原始长期方差
仍单独记录，负相关不会让正式 MCSE 暗中声称超过实际 round 数的证据。

## 5. 随机轨迹通过条件

对独立白噪声、`phi=0.5` 和 `phi=0.8`，某个观察长度只有同时满足下面三条才算合格：

1. 2000 次重复中，`sample_mean ± 1.96*MCSE` 覆盖真实均值 0 的比例位于
   `92.5%..97.5%`；
2. 2000 个“估计长期方差 / 理论长期方差”的中位数位于 `0.80..1.25`；
3. 同一长度下三类轨迹的 ESS 比例中位数必须保持：独立白噪声最高，`phi=0.5` 居中，
   `phi=0.8` 最低。

这些范围在执行前固定：覆盖率目标围绕 95%，长期方差容许最多约 1.25 倍的乘法偏差。不得看完结果
后放宽。

负相关 `phi=-0.5` 不参与最少历史长度选择，只检查：

- `raw ESS/n` 的中位数大于 1；
- 每一次正式保守 `ESS/n` 都不得大于 1；
- MCSE、长期方差和两种 ESS 均为有限数。

所有随机轨迹的全部长度、全部重复都不得产生未声明的异常或非有限输出。

## 6. 最少历史长度如何产生

按观察长度从小到大检查。候选长度 `n0` 必须满足：

```text
n0 自己合格，并且列表中所有大于 n0 的长度也都合格。
```

第一个满足该条件的 `n0`，才冻结为 `insufficient_history` 的共同下限。这样不会因为某一个长度偶然
好看就提前放行。

如果一直到 4096 都不存在这样的 `n0`：

```text
本版 b=floor(sqrt(n)) 候选失败；不产生历史下限；不得实现正式计数器。
```

此时若要换批长或估计器，必须先写新版本协议和新 seed 命名空间，不能在本轮结果上反复调参挑最好
方案。

## 7. 固定边界测试

下面是不需要随机重复的接口和退化行为测试：

| 输入 | 必须结果 |
|---|---|
| round 身份缺号、重复或乱序 | 拒绝输入 |
| 数值含 `NaN`、`Infinity` 或布尔值 | 拒绝输入 |
| 全部为同一个常数 | `zero_round_variance` |
| 长度 256 的 `1,-1,1,-1,...` | 命中伪零长期方差时返回 `degenerate_long_run_variance` |
| 一条轨迹整体加 13 | ESS 不变，MCSE 不变 |
| 一条轨迹整体乘 7 | ESS 不变，MCSE 乘 7 |
| 中间只有一个尖峰 | 所有数值有限，正式 ESS 不超过实际 round 数 |
| 持续线性趋势 | 输出仍声明 `stationarity_not_assessed=true`，且不存在收敛/停止字段 |
| 少于最终历史下限 | `insufficient_history` |

平移、缩放比较使用相对与绝对容差 `1e-10`。周期序列产生精确零长期方差时必须 fail closed；若得到
有限正值，则不事后增加“接近零”阈值。正式 ESS 与 MCSE 已统一使用相关膨胀至少为 1 的保守规则，
因此极小原始长期方差只能留在 raw 诊断中，不能增加正式证据。

## 8. 一次执行的产物

执行前先实现一个研究版纯函数和第 7 节中当时可执行的固定边界测试。研究版纯函数只实现已经写死
的数学公式，不设置 `insufficient_history` 下限、不接生成器，也不形成正式生产接口；人工协议将
直接调用这个函数，避免另写一份可能不一致的实验计算代码。

只有第 6 节选出最终历史下限后，才补上“少于最终历史下限”的测试，并冻结带
`insufficient_history` 语义的正式接口。

执行时只允许生成以下只读结果：

```text
人工协议 manifest
每个 family × length 的聚合指标
固定边界测试结果
自动选出的 n0，或明确的 candidate_failed
```

正确顺序固定为：

```text
协议审查
→ 研究版纯函数 + 固定边界测试
→ 人工重复实验
→ 结果审查
→ 若通过，再冻结历史下限和正式接口
```

人工结果不能直接修改协议、默认值或生成器。若结果失败，只能报告候选失败并重新设计下一版。

## 9. 固定执行入口

```text
scripts/validate_issue53_v2_effective_evidence.py plan
scripts/validate_issue53_v2_effective_evidence.py run --output-dir <新目录>
```

`plan` 只打印固定矩阵，不生成随机数。`run` 的唯一参数是新输出目录；重复次数、seed、轨迹类型、
长度、容差和选择规则都不能从命令行覆盖。执行前必须是包含全部 untracked 检查在内的干净 Git
工作树；manifest 绑定 Git commit、协议 SHA-256、实现/测试/两份设计文档 SHA-256 和环境版本。

当前冻结协议 SHA-256 为：

```text
79c88437c3ae720f6938fdb2fa56b31b198734a4a39f6dd596d75e16a1690e22
```

## 10. 方法依据

- Flegal, J. M. & Jones, G. L., *Batch means and spectral variance estimators in Markov chain Monte
  Carlo*, https://arxiv.org/abs/0811.1729
- Liu, Y., Vats, D. & Flegal, J. M., *Batch size selection for variance estimators in MCMC*,
  https://arxiv.org/abs/1804.05975
- Vats, D., Flegal, J. M. & Jones, G. L., *Multivariate output analysis for Markov chain Monte Carlo*,
  https://arxiv.org/abs/1512.07713
