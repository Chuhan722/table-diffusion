# Issue #53 V2：有效证据计数器设计稿

> 状态：研究版纯函数与固定边界测试已实现；人工重复协议尚未执行，正式协议尚未冻结。
> 本稿只定义一个标量轨迹的只读证据计数器；没有读取 development/validation，也不包含稳定、
> 收敛或停止判定。

## 1. 这一小步解决什么

输入一段连续的外层生成轨迹，回答：

```text
实际观察了多少个外层 round？
这些相关 round 相当于多少个独立样本（ESS）？
当前均值估计还有多大随机误差（MCSE）？
现有历史是否只在数值上足以估计这些量？
```

本计数器不回答：

```text
是否稳定？
是否收敛？
是否应该停止？
应该在哪一轮停止？
质量是否合格？
```

历史 `100` 轮小块和 `12` 小块候选继续作为已归档的 V2 development 反例证据，但不进入本计数器
的生产规则。

## 2. “一轮”的唯一含义

这里的一轮只指一次完整的**外层 current-state 更新**：

```text
读取当前表 → 选择/构造本轮更新 → 得到下一张当前表 = 1 outer round
```

factorized Gibbs 的 `sweeps` 是一轮内部生成候选行的核参数，不是这里统计的 round。内层 Gibbs
混合资格与外层 current-state 证据必须分别研究。

## 3. 运行轮数边界

计数器不接收“必须跑多少轮”的目标，也不知道 `max_rounds`。它可以在任意检查点只读当前已有的
确认轨迹并重新计算证据。

未来 runner 才具有下面的控制语义：

```text
证据满足 → 正常停止，qualified=true
达到 max_rounds 仍不满足 → 安全退出，qualified=false
```

因此 `max_rounds` 只是资源保险，不是收敛证据；本稿不为它设数值。

## 4. 输入契约

第一版只接受一组连续 round 身份和一个同长度标量序列：

```text
round_indices = [t1, t2, ..., tn]
values        = [x1, x2, ..., xn]
```

要求：

- `round_indices` 必须严格逐一递增且相邻差为 1；每个 `xi` 对应同位置的 post-round 观测，initial
  state 不进入序列；
- 序列必须一维、有限、非布尔，顺序不可重排；
- 不接受 dataset 名、kernel 名或 dataset→参数映射；
- 调用者必须声明序列身份，例如 `current_normalized_l1` 或某个查询的有符号残差；
- 第一版不同时处理多查询，也不读取表、查询目标或任何 validation 身份。

## 5. 输出契约

建议第一版只输出：

```text
actual_round_count                 输入的实际 round 数 n
batch_round_count                  滑动批长 b
overlapping_batch_count            重叠批次数 n-b+1（不是独立证据数）
single_round_variance              单 round 样本方差
long_run_variance                  考虑时间相关后的长期方差估计
raw_correlation_inflation          原始相关膨胀倍数
conservative_correlation_inflation 保守相关膨胀倍数，至少为1
raw_effective_round_count          原始 ESS 诊断值
effective_round_count              正式保守 ESS，不得超过实际 round 数
mcse                               当前均值的 Monte Carlo 标准误
numerically_estimable              是否满足数值估计前提
reason                             无法估计时的版本化原因
stationarity_not_assessed          固定为true，声明本层未判断平稳性
contract_version                   输出契约版本
```

输出 schema 必须拒绝以下字段及同义词：

```text
stable / converged / qualified / stop / stop_round / threshold / quality_pass
```

`numerically_estimable=true` 只表示数学量可以计算，不表示轨迹已经平稳。固定字段
`stationarity_not_assessed=true` 防止调用方把 ESS 误当收敛证据。

## 6. 第一版计算规则

### 6.1 重叠批长

```text
n = actual_round_count
b = floor(sqrt(n))
```

从每个可能起点取一个长度为 `b` 的连续滑动批次：

```text
batch_1 = values[1 : b]
batch_2 = values[2 : b+1]
...
batch_(n-b+1) = values[n-b+1 : n]
```

每批计算一个平均值。重叠批次使用全部输入 round，不丢尾部；这些批次高度重叠，所以
`overlapping_batch_count` 只是估计器内部数量，绝不能当作独立样本量。

选择 `floor(sqrt(n))` 的原因是：当历史增长时，批长和等价非重叠批次数都会增长。它不是某个
数据集的经验轮数，也不代表已经证明有限样本最优；是否保留该规则只由后述人工覆盖率审查决定。

### 6.2 长期方差

用程序式符号表示：

```text
overall_mean = mean(values)
batch_mean[i] = mean(batch_i)

long_run_variance
  = n*b / ((n-b)*(n-b+1))
    * sum((batch_mean[i] - overall_mean)^2)
```

这是重叠批均值长期方差估计。它的作用是把相邻 round 的重复信息计入均值不确定性。

### 6.3 ESS 与 MCSE

```text
single_round_variance = sample_variance(values)

raw_correlation_inflation
  = long_run_variance / single_round_variance

conservative_correlation_inflation
  = max(1, raw_correlation_inflation)

raw_effective_round_count
  = n / raw_correlation_inflation

effective_round_count
  = n / conservative_correlation_inflation

mcse
  = sqrt(single_round_variance
         * conservative_correlation_inflation / n)
```

正式 ESS 与正式 MCSE 使用同一个保守相关膨胀倍数，因此
`effective_round_count <= actual_round_count`，同时正式 MCSE 不会小于把现有 round 当作独立样本时
的普通标准误。负相关理论上可能使原始 ESS 大于实际 round 数，但本阶段不借此声称获得“超额证据”；
原始 ESS 与长期方差只保留作诊断。

### 6.4 必须 fail closed 的边界

- 输入含 `NaN/Infinity/bool` 或不连续：拒绝；
- `single_round_variance == 0`：`numerically_estimable=false`，原因为
  `zero_round_variance`，不得把完全冻结序列记为高 ESS；
- `long_run_variance == 0` 但单 round 方差非零：`numerically_estimable=false`，原因为
  `degenerate_long_run_variance`，防止周期序列与批长偶合后假装独立；
- 历史过短：`numerically_estimable=false`，原因为 `insufficient_history`；
- “历史过短”的精确共同下限尚未冻结。它必须根据人工轨迹的方差估计覆盖率预注册，不能根据
  nltcs/test 的结果选择，也不是未来的正常停止轮数。

## 7. 人工轨迹验收矩阵

实现前另行冻结人工 seed、长度、重复次数和数值容差；下面只冻结必须覆盖的行为，不提前读取真实
development。

| 人工轨迹 | 必须看到的行为 |
|---|---|
| 独立白噪声 | 随长度增加，保守 `ESS/n` 接近 1，MCSE 区间覆盖预设真均值 |
| 正相关 AR 序列 | 相关越强，`ESS/n` 单调下降；不得把 1000 round 计成约 1000 份独立证据 |
| 同序列整体平移/乘正数 | ESS 比例不变，MCSE 按比例变化 |
| 完全常数 | `zero_round_variance`，不得产生稳定/停止结论 |
| 持续线性趋势 | 即使能给出数值诊断，schema 中仍不存在稳定/停止字段；后续趋势层必须负责拒绝 |
| 单点尖峰 | 不得出现非有限输出或异常增大的 ESS |
| 周期序列 | 不得因批长与周期偶合得到伪零长期方差；命中时 fail closed |
| 负相关序列 | 可记录 raw ESS 大于 n，但正式保守 ESS 必须截在 n 以内 |
| 历史过短 | 明确返回 `insufficient_history`，不得补默认 ESS |

验收重点不是让每条有限轨迹精确命中理论 ESS，而是检查：估计随样本增长趋向正确、MCSE 的重复
覆盖率达到预注册目标、相关越强时证据不会反而增多、所有退化情况均 fail closed。

## 8. 与后续工作的边界

该标量计数器通过人工审查后，后续仍须按顺序单独完成：

```text
标量有效证据计数器
→ 多查询层与 query-count 修正
→ 趋势等价判断
→ 非冻结运动护栏
→ 质量标准
→ 重复检查错误控制
→ development 只读回放
→ 完整规则冻结
→ 全新 validation
→ 在线 runner 与 max_rounds
```

在标量计数器阶段，不修改 residual geometry、alpha、rho、mu、eta 或 Gibbs 配置；不运行 GPU 或
生成实验；不读取退休 validation seeds `220..224`；不创建停止器。

## 9. 本稿待用户审查的三个决定

1. 是否接受“第一小步只做标量 ESS/MCSE，不做收敛判断”的边界；
2. 是否接受重叠批均值与 `b=floor(sqrt(n))` 作为进入人工覆盖率审查的唯一候选，不并列试多个方法
   后挑结果最好者；
3. 是否接受先用人工重复覆盖率冻结 `insufficient_history` 下限，再实现正式接口，且该下限只表示
   估计器可用，不表示生成过程收敛。

## 10. 方法依据

- Flegal, J. M. & Jones, G. L.，*Batch means and spectral variance estimators in Markov chain Monte
  Carlo*：批均值/重叠批均值长期方差估计及一致性条件，https://arxiv.org/abs/0811.1729
- Vats, D., Flegal, J. M. & Jones, G. L.，*Multivariate output analysis for Markov chain Monte
  Carlo*：ESS、Monte Carlo 误差与精度驱动停止的关系，DOI `10.1093/biomet/asz002`。
