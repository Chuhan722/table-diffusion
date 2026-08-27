# Issue #53 Stage 5：同温度 independent 与 factor 核外层公平比较正式结果

日期：2026-08-27（正式运行与独立审计已完成）

## 1. 正式结论

在冻结的 `tau=2`（温度为 2）、fixed α=16（固定抽样强度 16）、relative residual
floor=8（相对残差且参考下限为 8）、scale-invariant donor（尺度不变供体选择）、P=6
terminal-current early stopping（输出当前终态的提前停止）以及无门控条件下，正式比较：

- `independent_s0`（独立核，0 次 Gibbs 扫描）；
- `factor_random_scan_s8`（因子随机扫描 Gibbs 核，8 次扫描）。

factor（因子核）在 `test_300x10` 与 `nltcs` 上的 measured aggregate（已测查询总绝对计数误差）
分别下降 8.35% 和 9.80%，但两套数据都只有 6/10 个配对种子严格胜出，没有达到结果前冻结的
8/10 稳定门槛。因此正式分类为：

```text
test_300x10                         no_stable_factor_gain（无稳定因子核增益）
nltcs                               no_stable_factor_gain（无稳定因子核增益）
两数据细分                           mixed_no_stable_kernel_winner（混合结果，无稳定胜出核）
跨数据结论                           no_shared_factor_support（无共享的因子核支持证据）
kernel_default_changed              false（不改变默认核）
tau_selected                        null（不据此选择温度）
additional_seed_requested           false（不追加种子）
```

Stage 4 的资格结论不变：因子核的 8 次扫描在冻结条件分布上混合充分。Stage 5 进一步说明，内层核合格
不等于完整外层生成稳定更好。当前继续保留 independent（独立核）作为 development reference
（开发参考核），不把 factor（因子核）升级为共享默认。

## 2. 正式身份与冻结边界

```text
worktree               /home/chuhan/projects/table-diffusion-issue53-stage5-kernel-ab
branch                 research/issue53-stage5-kernel-ab
execution commit       a5455ca1574a45acbbbe68abe3100a97e4b976ba
execution dirty        false
formal_result_valid    true
seeds                  338..347
case count             2 datasets × 2 arms × 10 seeds = 40
protocol document SHA  4d0ffb8ebf77006becea00849eef452174559aea62faeb86dd70c227e3fc7fab
protocol manifest SHA  1d447be0fb0ce9a2c7707abd2e259ed6ea41edbf3780f30426e320b1bad94f1c
```

两臂共享完全相同的外层生成身份：边际初始化、`tau=2`、fixed α=16、`rho=0.01`、`eta=0.5`、
`mu=0.01`、相对残差 floor=8、尺度不变供体、P=6、6000 轮资源上限、`tol=+inf`、零重试、
terminal-current（当前终态）输出和无门控接受语义。唯一科学变量是条件核：独立核与 8-sweep
factor random-scan Gibbs（8 次扫描的因子随机扫描 Gibbs 核）。

正式 fresh paired seeds（全新配对种子）固定为 `338..347`；Stage 4 使用过的 `313..337` 全部排除。
偶数种子先运行独立核，奇数种子先运行因子核，以平衡执行顺序；两臂绑定同一初表、初始化后主随机数
状态、方向尺度和公共随机数前缀，Gibbs 额外随机量使用独立派生随机流。

## 3. 采集、评价与审计资格

正式 collection（采集）结果：

- 10/10 shards（分片）、40/40 cases（案例）和 20/20 数据集—种子配对完整；
- 40/40 案例全部由 P=6 正常 `early_stopped`（提前停止）；
- 0 个 `resource_cap_reached`（触及资源上限），没有案例达到 6000 轮；
- 40/40 终态表 `valid_row_rate=1`（有效行比例为 1）；
- direction logit clip（方向对数几率裁剪）与 factor conditional logit clip
  （因子条件对数几率裁剪）均为 0；
- 采集阶段不读取 reference（源数据参考答案），不产生局部胜负或离线质量分类；
- 没有调参、追加种子、消耗隐私预算或进行部分结果比较。

Evaluator（评价器）先冻结查询身份并显式绑定完整采集 SHA，之后才读取 reference；它不调用生成器、
不追加种子、不修改终态表。Independent auditor（独立审计器）不信任评价器派生值，从终态 CSV、
diagnostics（诊断）和各层 manifest（清单）重新计算 40 个案例指标、全部配对差、门禁与最终分类。

独立审计结果为：

```text
overall_pass                         true
collection audit                     pass
evaluation audit                     pass
recomputed cases                     40 / 40
paired differences / gates / class   全部精确复现
```

## 4. 已测查询主结果

冻结胜出规则要求 factor（因子核）同时满足：

1. 10 个种子的总绝对计数误差严格低于 independent（独立核）；
2. 至少 8/10 个配对种子严格胜出。

正式结果：

| 数据集 | 独立核总误差 | 因子核总误差 | 因子核相对变化 | 配对胜/平/负 | 配对差 95% t 区间 | 分类 |
|---|---:|---:|---:|---:|---:|---|
| `test_300x10` | 407 | 373 | -8.35% | 6/1/3 | [-7.5329, 0.7329] | 无稳定因子核增益 |
| `nltcs` | 54,759 | 49,393 | -9.80% | 6/0/4 | [-1722.6011, 649.4011] | 无稳定因子核增益 |

逐种子值如下。`Δ=F-I`（因子核减独立核），负数表示因子核更好：

| seed（种子） | test I | test F | test Δ | nltcs I | nltcs F | nltcs Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 338 | 32 | 30 | -2 | 6493 | 4416 | -2077 |
| 339 | 47 | 43 | -4 | 4559 | 4847 | +288 |
| 340 | 46 | 32 | -14 | 5216 | 4969 | -247 |
| 341 | 35 | 36 | +1 | 4274 | 5281 | +1007 |
| 342 | 39 | 41 | +2 | 8692 | 4851 | -3841 |
| 343 | 41 | 34 | -7 | 4517 | 6138 | +1621 |
| 344 | 44 | 41 | -3 | 4849 | 4423 | -426 |
| 345 | 32 | 32 | 0 | 5060 | 5782 | +722 |
| 346 | 38 | 42 | +4 | 6264 | 4299 | -1965 |
| 347 | 53 | 42 | -11 | 4835 | 4387 | -448 |

按查询阶数汇总时，平均方向全部是因子核更低：

| 数据集与阶数 | 独立核 | 因子核 |
|---|---:|---:|
| test 2-way（二阶） | 27.7 | 26.9 |
| test 3-way（三阶） | 10.2 | 8.9 |
| test 4-way（四阶） | 2.8 | 1.5 |
| nltcs 2-way（二阶） | 2746.0 | 2409.3 |
| nltcs 3-way（三阶） | 2729.9 | 2530.0 |

所以主门禁失败点不是总误差方向，而是不同种子之间不够稳定；不能只看总和下降就宣称因子核胜出。

## 5. 离线质量与支持集结果

### 5.1 nltcs

nltcs 的 offline safety（离线安全质量）、reference support（参考支持集）、diversity（多样性）、
attribute support（属性支持）和 validity（有效性）全部通过。

| 指标（越低越好） | 独立核 | 因子核 | F/I | 通过 |
|---|---:|---:|---:|---|
| one-way safety（一阶安全误差） | 0.000258405 | 0.000238706 | 0.92377 | 是 |
| unmeasured 3-way（未测三阶误差） | 0.000600429 | 0.000584994 | 0.97429 | 是 |
| all 4-way（全部四阶误差） | 0.000884218 | 0.000888474 | 1.00481 | 是 |
| binned joint TVD（分箱联合总变差） | 0.237618194 | 0.237624374 | 1.00003 | 是 |

参考支持集也略有改善：合成概率质量位于参考支持集中的比例从 `0.922860` 到 `0.923429`，参考概率
质量被覆盖比例从 `0.887485` 到 `0.889055`。

### 5.2 test_300x10

test 的四组冻结离线安全指标均超过 `1.05×` 上限，因此均失败：

| 指标（越低越好） | 独立核 | 因子核 | F/I | 配对胜/负 | 通过 |
|---|---:|---:|---:|---:|---|
| one-way safety（一阶安全误差） | 0.043200 | 0.048013 | 1.11142 | 3/7 | 否 |
| common unseen 2-way（共同未测二阶） | 0.031088 | 0.034353 | 1.10500 | 2/8 | 否 |
| fixed held-out 3-way（固定留出三阶） | 0.015709 | 0.017171 | 1.09304 | 2/8 | 否 |
| fixed held-out 4-way（固定留出四阶） | 0.006138 | 0.006492 | 1.05759 | 3/7 | 否 |

test 的精确参考支持基线本身很小，但冻结比例门仍失败：合成概率质量位于参考支持集中的比例从
`0.002667` 降至 `0.001333`，参考概率质量被覆盖比例从 `0.001667` 降至 `0.001000`。多样性、
属性支持和有效性仍通过：
unique-row rate（唯一行比例）为 `0.724667 → 0.704667`，effective-unique ratio（有效唯一比例）为
`0.646397 → 0.623693`，两臂所有生成行都有效。

## 6. 外层工作量与运行时间

| 数据集 | I 轮数/归一化工作量 | F 轮数/归一化工作量 | F/I 外层工作量 | I 墙钟 | F 墙钟 | F/I 墙钟 |
|---|---:|---:|---:|---:|---:|---:|
| test | 1237.9 / 12.4053 | 1419.3 / 14.2053 | 1.1451 | 31.25 s | 40.20 s | 1.2864 |
| nltcs | 2423.1 / 24.2049 | 2262.2 / 22.6047 | 0.9339 | 569.44 s | 900.32 s | 1.5811 |

- test 的因子核外层工作量增加约 14.5%，超过 1.05 门槛；
- nltcs 的因子核外层工作量减少约 6.6%，通过该门槛；
- 因子核墙钟时间在 test 9/10、nltcs 10/10 个种子上更慢；
- 墙钟时间按冻结协议只作诊断，不进入硬分类。

因子核内部平均成本：test 的 factor build（因子结构构建）为 3.27 秒、Gibbs sample（Gibbs 抽样）
为 0.54 秒、79,810 个微步；nltcs 分别为 325.61 秒、35.02 秒、3,974,094 个微步。nltcs 的主要
额外时间来自逐轮 factor build，而不是 Gibbs 抽样本身。

## 7. 产物身份

只读复核的 SHA-256：

```text
pipeline smoke report
  d5dba1c41bd8f5c422fce4ee28cf2f4b5990878102104cbda1c6efedec2c8f24
collection report
  375377849aaec401ec6e2dcd29ed850f180c4204c17ed785f67fba2f8f6c506d
evaluation report
  8368c58d462a9f4b540f32c0d815faf93c436a7b23fef7fa852cda013a8f94f5
independent audit
  c14fb651466e641b4bf75d0b5966aed4b4a2dcd8f588da2cf5862aaa035e34bb
test_300x10 reference
  c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb
nltcs reference
  e547a7aedad1dd2f7177030881ab1b92c7e24ae5464c71a0f1f89daecaf52b30
```

约 210 MiB 的正式输出继续保留在 gitignored（被 Git 忽略）的
`outputs/issue53_stage5_kernel_ab_v1/`，不提交原始终态表。结果通过协议、smoke（冒烟验证）、采集、
评价和独立审计 SHA 串联绑定。

## 8. 收口边界

本结果不支持以下动作：

- 不把 factor（因子核）改为共享默认；
- 不根据同一批结果选择或调整 tau（温度）；
- 不为追求 8/10 门槛追加种子；
- 不用跨数据加权总分掩盖 test 的质量与计算门失败；
- 不否定 Stage 4 的内层混合资格，也不把内层资格误写成外层优势。

若未来重新研究 factor（因子核），需要先提出新的单变量问题并冻结新的结果前协议；本批 `338..347`
不能反复用于调参。Stage 5 到此完成科学、产物与独立审计闭环。
