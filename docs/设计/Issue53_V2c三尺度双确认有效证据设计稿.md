# Issue #53 V2c：三尺度双确认有效证据设计稿

> 状态：用户于 2026-08-16 接受为 V2c 人工验收的唯一候选设计。
> 当前只冻结设计，不实现代码、不生成新 seed、不运行人工或真实数据。

## 1. 为什么需要 V2c

V2b v1 的正式结果为 `candidate_failed`，独立审计通过。失败不是计算成本，而是安全性：

```text
phi=0.8  first-ready 覆盖率 90.39%，LRV 比中位数 0.7856
phi=0.95 first-ready 覆盖率 77.13%，LRV 比中位数 0.4210
```

它暴露两个明确问题：

1. `b` 与 `2b` 可能一起低估长期相关，却因为彼此接近而错误放行；
2. 15 次顺序查看可能抓住某一次偶然相容，并把它记作 first-ready。

V2c 只针对这两个问题各增加一道直接防线，不修改 V2/V2b 已归档结果。

## 2. 唯一候选规则

检查点保持不变：

```text
256, 384, 512, 640, 768, 896, 1024, 1152,
1280, 1408, 1536, 1664, 1792, 1920, 2048
```

在当前检查点 `n`，使用三个批长：

```text
b1 = floor(sqrt(n))
b2 = 2 * b1
b3 = 4 * b1
```

最小检查点 256 时三者为 16、32、64，均严格小于 `n`。每个尺度继续使用完全相同的 V2 重叠批
均值公式：

```text
raw_inflation_j    = LRV_j / sample_variance
formal_inflation_j = max(1, raw_inflation_j)
```

当前检查点的三尺度相容状态定义为：

```text
scale_ratio = max(formal_inflation_1,
                  formal_inflation_2,
                  formal_inflation_3)
              / min(formal_inflation_1,
                    formal_inflation_2,
                    formal_inflation_3)

compatible_at_n = 三个尺度均可计算，并且 scale_ratio <= 1.25
```

`1.25` 沿用 V2b，不根据 V2b 结果调小或调大。第三个更长尺度只负责增加观察范围，不能删除不利
尺度或对三个尺度求平均。

正式不确定性始终取当前三个尺度的最大值：

```text
official_inflation_n = max(formal_inflation_1,
                           formal_inflation_2,
                           formal_inflation_3)
official_LRV_n       = sample_variance_n * official_inflation_n
official_ESS_n       = n / official_inflation_n
official_MCSE_n      = sqrt(official_LRV_n / n)
```

## 3. 连续两次确认

单个检查点相容还不取得数值资格。设固定日程中当前检查点为 `n_k`：

```text
confirmed_at_n_k = compatible_at_n_(k-1) AND compatible_at_n_k
```

因此：

- 256 只能形成第一次候选观察，不能取得资格；
- 最早可能取得资格的位置是 384；
- 若出现 `通过 → 不通过 → 通过`，第三个位置仍不取得资格，必须等待下一个检查点再次通过；
- 2048 若要取得资格，1920 与 2048 必须连续相容；
- 若全程没有连续两个相容检查点，first-confirmed 为 null，资源计数固定为 2048。

人工协议仍记录第一次 `confirmed_at_n=true` 的位置，用来检查顺序选择后的覆盖率和资源成本。但
该字段只表示当时 ESS/MCSE 获得数值资格，不是永久状态：后续检查点可以重新变为不相容，调用方
必须按当前最近两个检查点重新计算，不能把历史确认改写成 stable、converged 或 stop。公开 schema
继续使用 V2b 已有的 `adaptive_numerically_estimable` 与
`first_adaptive_numerically_estimable_round`；`confirmed` 只作为本稿解释“两次相容”的内部记号，
不新增容易被误读成收敛确认的公开字段。

## 4. 两道新防线分别解决什么

```text
第三个 4b 尺度     → 尝试暴露 b 与 2b 一起漏掉的更慢相关
连续两个检查点确认 → 降低一次随机相容被永久选中的概率
```

二者都不是理论保证。特别慢的相关过程仍可能让三个尺度一起偏低，因此 V2c 仍必须使用全新人工协议
验证 `phi=0.95`；不能因为规则看起来更保守就跳过正式覆盖实验。

## 5. 下一份人工协议保持什么不变

为避免同时改太多东西，V2c 人工协议建议原样保留：

- 五个 family：iid、`phi=0.5/0.8/-0.5/0.95`；
- 每类 2000 条、每条 2048 round，从平稳分布启动；
- 主 family first-confirmed 数不少于 1850 / 2000；
- first-confirmed 覆盖率位于 `92.5%..97.5%`；
- official LRV / 理论 LRV 中位数位于 `0.80..1.25`；
- official ESS/n 中位数严格保持 `iid > 0.5 > 0.8`；
- iid 资源中位数不超过 512，`phi=0.5` 不超过 1024，三个主 family pooled mean 不超过 1536；
- `phi=0.95` 仍只允许完全拒绝，或至少 1000 条安全确认；中间状态失败；
- 负相关 formal ESS 不超过 `n`，MCSE 不低于独立样本标准误；
- 2048 是资源上限，不是自动通过条件。

新协议必须使用全新 namespace：

```text
SeedSequence([53, 2, 3, family_code, repeat_index])
```

其中第三个版本号 `3` 表示 V2c。不得复用 V2b 的 `[53,2,2,...]` 正式结果来判断 V2c 是否通过。

## 6. V2c 必须新增的报告项

除了 V2b 已有输出，必须增加：

```text
b1 / b2 / b3
三个 raw/formal inflation
三尺度 scale_ratio
three_scale_compatible_at_checkpoint
adaptive_numerically_estimable_at_checkpoint（连续两次相容）
first_adaptive_numerically_estimable_round
first-ready 后再次不相容的轨迹数和比例
```

最后一项只描述数值资格是否会回撤，不能替代 first-confirmed 覆盖率门禁，也不能据此挑选更好看的
确认位置。

## 7. 必须先通过的确定性边界

实现前至少冻结以下测试：

1. 三个指定批长的 OBM 都能与手算值对拍；
2. 三尺度 ratio 恰好 1.25 通过，`nextafter(1.25,+inf)` 不通过；
3. 任一尺度不可计算，当前检查点整体 fail closed；
4. official inflation 精确等于三个尺度最大值；
5. 决策序列 `T,T` 首次确认在第二点；
6. `T,F,T` 不确认，`T,F,T,T` 只在第四点确认；
7. 1920/2048 连续通过与到上限仍未确认严格区分；
8. 平移、正缩放、常数、周期、尖峰、趋势和非法输入继续沿用 V2b 边界；
9. 不产生 stable、converged、qualified、stop、threshold 或质量字段；
10. 原 V2 与 V2b 的公开行为和归档结果不被修改。

这里内部可使用 `confirmed` 解释双确认公式，但公开 schema 继续使用
`adaptive_numerically_estimable`，并固定 `stationarity_not_assessed=true`；不得新增或缩写成生成
停止、质量合格或收敛确认。

## 8. 成功与失败后的去向

若 V2c 使用全新协议得到 `candidate_supported`，它也只说明三尺度双确认的标量 ESS/MCSE 数值门禁
可进入下一研究层。之后仍需单独设计真实轨迹 development/validation，不能直接宣称
`test_300x10` 或 `nltcs` 收敛。

若 V2c 再次 `candidate_failed`，本路线停止增加尺度、连续次数或隐藏门禁，不继续堆叠 V2d。当前
研究改回 V2 已支持的统一 2048 数值资格下限；2048 仍不是生成停止轮数。

## 9. 当前明确不做什么

- 不实现 V2c 核心、runner 或 auditor；
- 不生成 `[53,2,3,...]` 随机数；
- 不回放 V2b 正式 seed 调 `1.25`；
- 不读取 `test_300x10`、`nltcs` 或退休 validation seeds；
- 不接扩散生成器、在线停止器、DP 预算或 Gibbs/alpha/rho 等生成参数。

用户已确认“三尺度 + 连续两次”作为 V2c 唯一候选。下一步只写结果前人工协议；协议再次经用户
审查前，不实现代码或运行实验。
