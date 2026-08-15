# Issue #49 高温因子 Gibbs 强无门控基线正式结果

> 状态：正式运行与独立审计均完成；唯一候选未通过最终确认
> 日期：2026-08-14
> 运行提交：`eda31f7340e7659d7cd5dd4048e5564f1927708b`
> 关联 Issue：[#49](https://github.com/Chuhan722/table-diffusion/issues/49)

## 1. 最终结论

本协议在干净的固定提交上完整运行了 Stage T/A、Stage B 和最终确认，三个阶段的正式身份与
独立审计全部通过。Stage B 冻结的唯一候选为：

```text
factor Gibbs, tau=4, sweeps=32
```

该候选在最终确认集上是所有数值合格配置中 late-window current loss 点估计最低的配置，且
相对同温度独立核 `tau=4` 的优势通过了全部配对门槛；但它相对 Stage B 冻结的最强独立基线
`I*=independent tau=5` 的 95% 配对 t 区间上界为正，未通过预注册确认门槛。因此最终状态是：

```text
status    = not_confirmed_no_reselection
confirmed = false
```

这不是实现、数值资格或 Gibbs 混合失败。准确结论是：因子 Gibbs 对同温度独立核有稳定优势，
相对更强的 `tau=5` 独立核也有正向点估计，但本次 10 个确认 seeds 不足以把该优势确认为严格
小于 0。协议按预注册规则停止，没有追加 seeds、递补配置、调整 tau/sweeps/clip 或改写门槛。

## 2. 正式身份与规模

所有正式结果都来自同一个 clean commit：

```text
commit          eda31f7340e7659d7cd5dd4048e5564f1927708b
Python          3.11.15
NumPy           2.4.6
pandas          3.0.5
device          NumPy / CPU
rounds          1000
primary metric  rounds 751..1000 current-loss mean（越低越好）
logit clip      30
```

输入 SHA-256：

```text
schema     58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792
queries    7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88
marginals  1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4
```

| 阶段 | seeds | 轨迹数 | 墙钟 | 正式身份 | 独立审计 |
|---|---:|---:|---:|---|---|
| Stage T/A | `0..9` | 50 | 1476.39 秒 | 通过 | 通过 |
| Stage B | `100..109` | 60 | 1482.42 秒 | 通过 | 通过 |
| 最终确认 | `110..119` | 60 | 1475.49 秒 | 通过 | 通过 |

总运行墙钟约 73.9 分钟。tools-freeze 前的回归结果为 `884 passed, 7 skipped`，相关专项为
`85 passed`，同时通过 Python 编译检查与 `git diff --check`。

## 3. Stage T 与 A0：温度效果和数值资格

Stage T 的独立方向无门控结果：

| tau | late mean | late median | late std | final mean | AUC mean | Stage T clip hits |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 80.0420 | 80.4010 | 7.3808 | 76.60 | 204208.900 | 0 |
| 5 | 74.2666 | 73.1380 | 8.2787 | 67.90 | 186668.050 | 0 |
| 6 | 71.4506 | 68.8130 | 7.2537 | 70.75 | 177475.975 | 5 |
| 7 | 64.2092 | 64.0050 | 4.2337 | 56.85 | 165831.975 | 32 |
| 8 | 55.6606 | 53.9980 | 4.9593 | 49.20 | 156305.250 | 147 |

固定 1000 轮下，高温点估计更低；但 A0 要求来源独立轨迹和冻结因子条件都满足原始
`|logit|<30`，所以效果排名不能脱离数值资格解释：

| tau | 来源最大 `|logit|` / hits | 因子最大 `|logit|` / hits | A0 状态 |
|---:|---:|---:|---|
| 4 | 21.9947 / 0 | 19.4137 / 0 | `eligible_for_mixing` |
| 5 | 27.4934 / 0 | 24.2671 / 0 | `eligible_for_mixing` |
| 6 | 32.9921 / 5 | 29.1206 / 0 | `out_of_numerical_domain` |
| 7 | 38.4908 / 32 | 33.9740 / 44 | `out_of_numerical_domain` |
| 8 | 43.9895 / 147 | 38.8274 / 766 | `out_of_numerical_domain` |

因此只有 `tau=4/5` 能进入 A1。`tau=6/7/8` 的 loss 继续作为固定 clipped 实现的趋势诊断
保留，但不能参与本协议的认证排名。

## 4. A1：最小充分 Gibbs sweeps

每个候选必须在 global、initial 和十个来源温度的 mid/late 状态组上同时满足：

```text
TVD <= 0.05
expected-direction gap recovery >= 0.80
```

实际瓶颈是共同 initial 状态：

| tau | sweeps | global TVD / recovery | initial TVD / recovery | 全组结果 |
|---:|---:|---:|---:|---|
| 4 | 8 | 0.00681 / 0.93507 | 0.07205 / 0.90987 | 失败 |
| 4 | 16 | 0.00499 / 0.95030 | 0.05615 / 0.92989 | 失败 |
| 4 | 32 | 0.00383 / 0.96160 | 0.04321 / 0.94578 | 通过 |
| 5 | 8 | 0.00820 / 0.93149 | 0.08243 / 0.90137 | 失败 |
| 5 | 16 | 0.00616 / 0.94420 | 0.06877 / 0.91741 | 失败 |
| 5 | 32 | 0.00521 / 0.95155 | 0.05873 / 0.92814 | 失败 |

所以 `tau=4` 的最小充分值为 `32 sweeps`；`tau=5` 记为
`not_sufficient_through_32`，按冻结协议不追加 64。两个适用温度的生产 Gibbs 与精确随机 tape
重放全部一致，分别完成 58,923 个生产采样器对拍且零 mismatch。

## 5. Stage B：开发集选择

| 配置 | late mean | median | std | final mean | AUC mean | clip hits | 资格 |
|---|---:|---:|---:|---:|---:|---:|---|
| independent tau=4 | 78.1254 | 77.308 | 9.3293 | 69.10 | 211547.500 | 0 | 合格 |
| independent tau=5 | 73.6942 | 74.352 | 9.2121 | 65.90 | 187754.800 | 0 | 合格 |
| independent tau=6 | 67.6162 | 67.067 | 5.3902 | 65.50 | 175624.650 | 0 | A0 不合格 |
| independent tau=7 | 60.6716 | 62.001 | 4.1086 | 61.05 | 169659.275 | 7 | 不合格 |
| independent tau=8 | 57.9290 | 59.892 | 5.4955 | 55.80 | 161191.150 | 130 | 不合格 |
| factor tau=4, sweeps=32 | 66.8170 | 65.871 | 6.9663 | 61.65 | 149958.725 | 0 | 合格 |

冻结选择为：

```text
I* = independent_tau_5
G0 = G* = factor_tau_4_sweeps_32
unique candidate = factor_tau_4_sweeps_32
```

因子候选相对同温度 independent tau=4 的 paired candidate-minus-baseline：

```text
mean difference    -11.3084
median difference  -10.7820
wins / ties / loss 9 / 0 / 1
```

候选自身访问状态复查通过；initial 组最接近门槛，`TVD=0.04969`、恢复率 `0.94817`，其余
global/mid/late 也全部通过。最大原始 `|logit|=21.8201`，clip hits 为 0。

## 6. 最终确认

| 配置 | late mean | median | std | final mean | AUC mean | clip hits | 资格 |
|---|---:|---:|---:|---:|---:|---:|---|
| independent tau=4 | 84.8986 | 88.126 | 10.7110 | 68.65 | 217501.775 | 0 | 合格 |
| independent tau=5 | 71.4562 | 69.286 | 11.4798 | 63.50 | 192536.600 | 0 | 合格 |
| independent tau=6 | 67.5086 | 66.686 | 6.2799 | 64.05 | 183711.475 | 0 | Stage B 未认证 |
| independent tau=7 | 61.7218 | 60.651 | 6.6635 | 62.60 | 172795.000 | 38 | 不合格 |
| independent tau=8 | 56.4668 | 55.480 | 4.5492 | 53.20 | 165561.400 | 124 | 不合格 |
| factor tau=4, sweeps=32 | 64.8426 | 63.415 | 4.5600 | 61.90 | 151868.650 | 0 | 合格 |

候选在合格臂中的点估计最低。正式配对比较中，负数表示候选更好：

| 对照 | mean diff | median diff | wins / ties / losses | 95% paired t CI | 结果 |
|---|---:|---:|---:|---:|---|
| independent tau=4 | -20.0560 | -22.9750 | 10 / 0 / 0 | [-26.0481, -14.0639] | 通过 |
| independent tau=5 (`I*`) | -6.6136 | -7.6440 | 8 / 0 / 2 | [-14.4794, 1.2522] | **失败** |

相对 `I*` 的两个负向 seed 是 112（差值 `+3.154`）和 115（差值 `+13.244`）。均值、中位数、
至少 6/10 wins、身份、数值、概率、混合和完整 seeds 均通过；唯一失败项是 95% 区间上界
`1.2522` 没有严格小于 0。

最终自身状态复查继续通过：最大原始 `|logit|=21.6971`、零 clip；global/initial TVD 分别为
`0.01405/0.04223`，恢复率分别为 `0.95489/0.95174`，mid/late TVD 接近 0。

## 7. 结论边界与后续研究动机

本次正式结果只回答固定 `1000 rounds` 预算下的 `tau=4..8` 前沿，不能解释为已经达到长期
平稳分布。确认集候选最后五个 50-round 窗口的平均 current loss 仍从 `71.14` 降至
`56.90`，说明 1000 轮终点仍含明显的外层时间效应。

### 7.1 与历史实验的关系：后续不是重复扫参

项目以前已经分别研究过温度和 Gibbs sweeps，但这些实验回答的是不同的局部问题：

| 历史实验 | 已覆盖内容 | 已得到的证据 | 仍不能回答的问题 |
|---|---|---|---|
| 独立核无门控温度前沿 | `tau=0/1/2/4/8`，10 paired seeds，1000 轮 | 固定预算下高温下降更快，旧网格中 tau=8 最低 | 缺 tau=3/5 的统一比较；没有长期收敛判定 |
| 冻结状态 factor 混合 | `tau=1/2 × sweeps=0/1/2/4/8`，3 seeds、2 个状态 | tau=1/2 在 8 sweeps 时 TVD 为 0.00315/0.02479 | 只检查少量冻结状态，不是长期动力学或独立确认 |
| factor tau=2 长期无门控 | `sweeps=0` 对 8，1000 轮，先 10 后追加 20 seeds | factor 对同 tau 独立核有明显方向性优势 | 追加 seeds 是看过首批结果后决定，不能作为一次性确认；也没有 tau=3/4 的同协议比较 |
| 本次 Issue #49 | 独立 `tau=4..8`；A0 合格的 factor `tau=4/5 × sweeps=8/16/32` | tau=4 需要 32 sweeps；factor 对同 tau 稳定更好 | 固定 1000 轮仍未收敛；相对 I*=tau5 未正式确认；没有低温统一前沿 |

因此后续研究不会重新无差别扫描已经回答过的组合。历史结果只用来缩小并预注册新设计：

- `tau=1/2` 不重新搜索 `0/1/2/4/8 sweeps`，只先在新的严格 current-state 状态库上复验
  历史上两个温度共同通过门槛的候选 `sweeps=8`；只有复验失败时，才按事前规则检查现有的
  16/32，硬上限仍为 32；
- `tau=3` 是此前 factor 混合与长期实验都没有覆盖的新温度，需要首次检查；
- `tau=4` 固定继承本次正式选择的 `sweeps=32`，只在新增低温来源状态上复验资格，不重新选择；
- 新长期实验把有效的 independent `tau=1..5` 与合格 factor 候选放在同一实现、同一状态语义、
  同一批配对 seeds 和同一 horizon 下比较；
- 新增的核心变量是外层时间：用预注册检查点区分“1000 轮内下降更快”和“长期水平更低”；
- 最终使用一次性冻结的新确认 seeds，不与历史 10+20 seeds 或 Issue #49 seeds 合并成伪扩样。

所以后续工作的新增贡献是：**补齐 tau=3、统一此前分散的温度/混合证据、识别长期 horizon，
并完成一次性独立确认**，而不是重复证明“tau=1/2 的 8 sweeps 在旧冻结状态上能混合”。

### 7.2 后续协议边界

因此本结果支持但不完成以下后续问题：

1. 在独立新协议中补齐 `tau=1/2/3`，与当前有效温度 `4/5` 形成低温到数值上界的前沿；
2. factor 只使用已有 `8/16/32` sweeps，并设置 32 为硬上限，不追加 64/128；
3. 用开发 seeds 的预注册检查点确定合理长期 horizon，区分“固定预算下降更快”和“长期水平更低”；
4. 正式实验前先实现按 seed/config 的多进程 CPU 并行，并验证串行/并行科学字段一致；
5. 冻结唯一配置和 horizon 后，才使用全新确认 seeds；nltcs、选择性曲率和 gate-only 仍在其后。

这些内容不是对 Issue #49 协议的事后追加，也不改变本次 `not_confirmed_no_reselection` 结论；
应在当前结果进入 PR 后另立 Issue 和预注册。

## 8. 产物、校验和与公开记录

原始 JSON 总计约 257 MiB，保留在本地 `outputs/issue49_high_temperature_factor_gibbs/` 并由
`.gitignore` 排除，不把大文件强行加入 Git。权威身份如下：

| 阶段 | report SHA-256 | audit SHA-256 | protocol SHA-256 |
|---|---|---|---|
| Stage T/A | `b160ed589f75f3874dc10519134204a3f3f77377220d9ddd983b8faccdcee3f9` | `368255136d1aed9117eb97d51379bd3bfbcf63f8d49d2f490867c1753c483c44` | `6328b014a7e85ad53e9f4511a87bc226ea9ab81ff5688ff4a01b2518259d6199` |
| Stage B | `a9312ededfc6e5a893adef7f7f6318fd3216367271f8b90915d8a31c25e930a6` | `44a7224568da58d2720877ad994f05ac5dbb382e9028cd1e6e366422143bb8c6` | `17b87af803c86056e43e63b4f41e1dc2dddf54886133afc9bfe22d135c33352e` |
| 最终确认 | `3b4f5961242948adcf560377348ec7d014f56b8cc4e200f075aeba2830625c61` | `d6bb9eeaff03b134c25055367f8d2b6c916b3e260dce6524020208b79937e15b` | `63ba988fc4bf30ed72f41d4eb229d5742b8a7759490ea889972ed26fab99376c` |

Stage T/A 状态库 SHA-256：

```text
7803e5be77b3711f4ad57bb1dc915995decf69d15dbb4b397c8a482a4d27776e
```

Issue #49 运行记录：

- [tools-freeze 身份与测试](https://github.com/Chuhan722/table-diffusion/issues/49#issuecomment-5281787230)
- [Stage T/A 正式结果](https://github.com/Chuhan722/table-diffusion/issues/49#issuecomment-5282090727)
- [Stage B 正式结果](https://github.com/Chuhan722/table-diffusion/issues/49#issuecomment-5282395861)
- [最终确认与停止决定](https://github.com/Chuhan722/table-diffusion/issues/49#issuecomment-5282816288)
