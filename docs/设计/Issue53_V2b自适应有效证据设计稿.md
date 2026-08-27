# Issue #53 V2b：自适应有效证据设计稿

> 状态：用户已于 2026-08-16 接受为人工验收的唯一候选设计；研究核心、固定 runner、独立 auditor
> 与测试均已实现，但尚未生成正式协议 seed，尚未运行正式人工矩阵。
> 本稿只研究“已有多少连续 outer round 后，ESS/MCSE 的数值估计可以进入下一层判断”。它不判断
> 平稳、收敛、质量或停止，也不改变 V2 已归档的 2048 结论。

## 1. 一句话设计

不再把 V2 最坏人工条件得到的 `2048` 当作所有轨迹的统一起点，而是在预先固定的检查点，对同一段
残差历史使用短、长两种批长估计时间相关性：

```text
两种尺度给出的正式不确定性接近 → 当前 ESS/MCSE 可以进入后续判断
两种尺度仍明显不一致             → 历史还不够，继续积累
到 2048 仍不一致                 → 只报告证据不足，不强行通过
```

“根据数据自适应”只体现在每条轨迹第一次满足尺度一致性的检查点可能不同。检查日程和判定规则必须
在看新结果之前固定，不能由结果临时挑选。

## 2. 为什么另立 V2b

V2 的固定人工协议只检查了：

```text
16, 32, 64, 128, 256, 512, 1024, 2048, 4096
```

其中 `AR(1), phi=0.8` 在 1024 round 的正式 MCSE 覆盖率为 `92.30%`，低于预注册下限
`92.50%`；2048 round 为 `93.70%`，通过。因此 V2 正确得到共同历史候选 2048。

这个结论说明：若对所有允许的人工相关性使用**同一个固定下限**，现有 V2 候选不能小于 2048。
它没有说明每条具体轨迹都需要 2048，也没有证明 2048 是统计学上限。

V2b 是新的顺序候选设计，不回改 V2：

- V2 结果继续作为固定下限路线的有效记录；
- V2b 增加第二个观察尺度和顺序检查，因此必须使用新协议、新 seed 命名空间重新验收；
- 若 V2b 验收失败，只能报告候选失败，不能回头调整本稿阈值后复用同一批结果。

## 3. 严格范围

V2b 输入仍然只是一条连续、有限的一维 post-round 标量轨迹，例如
`current_normalized_l1`。round 身份、数值类型和 fail-closed 边界沿用 V2。

V2b 只允许回答：

```text
当前两个时间尺度给出的保守 ESS/MCSE 是否已经相容？
当前应报告的更保守 ESS/MCSE 是多少？
首次相容发生在哪个预定检查点？（只用于人工验收记录）
```

V2b 仍然不允许回答：

```text
轨迹是否平稳？
残差是否足够小或不再改善？
生成是否收敛？
生成器是否应该停止？
合成数据质量是否通过？
```

因此本层通过以后，也只能把 ESS/MCSE 交给后续趋势、运动和质量层；不能单独结束生成过程。

## 4. 固定检查日程

第一版唯一候选日程为：

```text
首次检查：256 outer rounds
检查间隔：128 outer rounds
研究上限：2048 outer rounds

检查点：256, 384, 512, 640, 768, 896, 1024, 1152,
        1280, 1408, 1536, 1664, 1792, 1920, 2048
```

三个数的含义分别是：

- `256`：既有 V2 结果中，中等正相关 `phi=0.5` 第一次通过固定长度覆盖验收的位置；128 及以前
  已有欠覆盖证据。这里把既有 V2 当作 development 依据，而不是把 256 宣称为理论下限；
- `128`：让在线判断最多晚发现 127 round，同时把总检查次数限制为 15 次。它是首版计算分辨率，
  不是统计常数；重复查看造成的偏差必须在新协议中按“第一次通过”整体检验；
- `2048`：既有 V2 中最强正式压力 `phi=0.8` 第一次通过的档位，因此适合作为首版资源上限。
  它不是保证通过的最大需要量，也不是收敛轮数。

如果 2048 时仍未通过，唯一合法结果是：

```text
adaptive_numerically_estimable = false
reason = resource_cap_without_multiscale_evidence
```

调用方可以在未来另行批准更高资源上限，但不得把“到达 2048”本身改写成通过证据。

## 5. 每个检查点如何计算

设当前已经观察 `n` 个连续 outer round。V2b 在同一段历史上使用两个批长：

```text
short_batch_round_count = floor(sqrt(n))
long_batch_round_count  = 2 * short_batch_round_count
```

例如在 512 round 时：

```text
短批长 = 22 round
长批长 = 44 round
```

两个尺度分别沿用 V2 的重叠批均值公式，得到：

```text
short_long_run_variance
long_long_run_variance
```

再用同一个单 round 样本方差计算两种正式相关膨胀：

```text
short_inflation = max(1, short_long_run_variance / single_round_variance)
long_inflation  = max(1, long_long_run_variance  / single_round_variance)
```

这里的 `max(1, ...)` 沿用 V2 的正式保守口径：负相关可以保留 raw 诊断，但正式 ESS 不得超过实际
round 数，正式 MCSE 也不得小于把现有 round 当作独立样本时的普通标准误。

尺度差异唯一写成下面的无量纲比值：

```text
scale_ratio = max(short_inflation, long_inflation)
              / min(short_inflation, long_inflation)
```

最终报告不平均两个尺度，而是取更保守者：

```text
official_inflation = max(short_inflation, long_inflation)
effective_round_count = n / official_inflation
mcse = sqrt(single_round_variance * official_inflation / n)
```

取最大值的原因很直接：如果长尺度暴露出更多慢相关，平均会把刚发现的风险再次稀释；取最大值则不会
比任一已检查尺度更乐观。

## 6. 唯一尺度一致候选

V2b 的唯一待验收门禁为：

```text
两个尺度都能有限、非退化地计算
并且
scale_ratio <= 1.25
```

也就是两个正式相关膨胀相差不超过 1.25 倍。选择 `1.25` 的理由不是声称存在一个 25% 的理论定理，
而是复用 V2 在结果产生前已经冻结过的长期方差乘法容差 `0.80..1.25`，避免看完 V2 结果后另外挑
一个更有利的新数字。它只是一个可被新人工协议证伪的唯一候选。

禁止同时运行 `1.10/1.25/1.50` 后挑最好者。如果 `1.25` 在全新顺序验收中欠覆盖或几乎不能节省
round，V2b 就失败；下一版必须另写设计、另用 seed，不能在同一结果上微调。

## 7. 检查状态

每个检查点只能产生下面三类数值状态：

```text
1. core_not_estimable
   某个尺度命中 zero_round_variance、degenerate_long_run_variance
   或 nonfinite_computation；fail closed。

2. multiscale_disagreement
   两个尺度可计算，但 scale_ratio > 1.25；继续积累历史。

3. adaptive_numerically_estimable
   两个尺度可计算且 scale_ratio <= 1.25；允许把更保守 ESS/MCSE
   交给后续判断层。
```

人工协议会记录每条轨迹第一次进入第 3 类的检查点，名称固定为
`first_adaptive_numerically_estimable_round`。该字段只描述数值门禁第一次通过的位置，不得缩写成
`stop_round`。

实现 schema 继续固定：

```text
stationarity_not_assessed = true
```

并拒绝 `stable / converged / qualified / stop / quality_pass` 及其同义输出。

## 8. 两个已知漏洞如何处理

### 8.1 两个尺度可能一起看错

短、长尺度接近不等于数学上证明已经捕获全部相关性。如果真实相关持续时间远大于长批长，两个估计
可能一起偏低并错误地显得一致。

因此本稿只能把尺度一致当候选证据，不能把它写成理论保证。新人工协议除复现 V2 的
`iid/phi=0.5/phi=0.8/phi=-0.5` 外，还必须加入更慢的 `phi=0.95` 压力边界：

- 若它到 2048 仍不相容，允许安全返回证据不足；
- 若它提前相容，则必须单独检查“第一次相容时”的 MCSE 覆盖率，不能因为通过门禁就免检。

非平稳趋势不属于本层能证明的对象；边界测试只要求它绝不产生稳定、收敛或停止字段，后续趋势层
仍负责拒绝非平稳过程。

### 8.2 多次查看可能偶然提前通过

固定长度分别覆盖良好，不代表从 15 次检查中挑“第一次好看”的时刻仍然覆盖良好。因此新协议的
主验收单位必须是整条顺序过程：

```text
生成一条最长 2048 的人工轨迹
→ 按固定日程依次检查
→ 记录第一次通过，或记录到上限仍未通过
→ 只在这个第一次通过的位置检查 MCSE 覆盖率
```

不能先分别验收 15 个固定长度，再据此推断顺序规则安全。

## 9. 计算成本

每次检查只处理最多 2048 个残差标量，不读取整张表，也不重新运行扩散或 Gibbs 核。15 个检查点
从头重算时，每个尺度累计读取：

```text
256 + 384 + ... + 2048 = 17280 个标量
```

两个尺度约为 34560 个标量级操作；实现时还可复用累计和。与生成 128 个新 outer round 相比，这个
诊断成本预期可以忽略，但正式人工 runner 仍应记录墙钟时间，不把“预期便宜”当结果。

## 10. 人工协议已经冻结什么

用户已接受本稿及配套人工协议；协议已经在结果前冻结：

1. 全新且不与 V2 重叠的 seed 命名空间、重复次数和人工 family；
2. 本稿 15 个检查点、两个批长、`1.25` 门禁和取最大不确定性的唯一计算规则；
3. `iid/phi=0.5/phi=0.8` 在**第一次通过时**的 MCSE 覆盖率门槛；
4. 每类首次通过轮数的分布、到 2048 仍未通过的比例，以及必须达到的计算节省标准；
5. `phi=0.95` 提前通过时的安全门禁和不通过时的 fail-closed 解释；
6. 固定边界、非有限、负相关正式 ESS 上限、schema 越权与重复检查审计；
7. 若候选失败，禁止复用结果调阈值、改检查间隔或删除不利 family。

研究函数、固定 runner、独立 auditor 与测试已经完成。下一步仍必须先把实现与协议绑定到干净的
预运行 commit，再只读核对 `plan`；未完成这两步并再次获得用户授权前，不运行正式人工矩阵。当前
仍不得读取真实 development/validation、接入生成器或创建停止器。

## 11. 方法边界

批均值文献支持使用增长的批长估计相关序列的长期方差，并强调有限样本结果依赖批长；精度驱动的
顺序停止文献也要求一致的长期方差估计。但“比较 `b` 与 `2b`、比值不超过 1.25”不是下列文献中的
现成定理，而是本项目为降低统一 2048 成本提出、必须由全新顺序覆盖实验审查的候选工程规则。

- Flegal, J. M. & Jones, G. L., *Batch means and spectral variance estimators in Markov chain
  Monte Carlo*, https://arxiv.org/abs/0811.1729
- Liu, Y., Vats, D. & Flegal, J. M., *Batch size selection for variance estimators in MCMC*,
  https://arxiv.org/abs/1804.05975
- Vats, D., Flegal, J. M. & Jones, G. L., *Multivariate output analysis for Markov chain Monte
  Carlo*, https://arxiv.org/abs/1512.07713
