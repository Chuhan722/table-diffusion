# Issue #53 V2b：自适应有效证据人工验收结果

> 正式结论：`candidate_failed`；独立审计 `passed=true`。
> 该结果只否决 V2b v1 的“双尺度一次相容即可取得 ESS/MCSE 数值资格”规则，不判断生成收敛，
> 不读取 `test_300x10` 或 `nltcs`，也不否决无门控扩散核本身。

## 1. 冻结身份

```text
Git commit
  1ad9340b6bd93ce7e998d5bc73d28d48f82231d6

protocol SHA-256
  a7dde6b7867e215c9147131f085eaa47b47e04495b5d1bed37355f95a69dd33f

scientific result SHA-256
  abd39f88da0408b5341374b1019ddb61df50fa591ec745d10bba27e504dbdb12

正式输出目录
  outputs/issue53_v2b_adaptive_effective_evidence_1ad9340/
```

正式入口在包含 untracked 在内的干净工作树上运行。矩阵固定为 5 个 family、每类 2000 条、每条
2048 round，共 10000 条人工轨迹、150000 次检查点分类与 300000 次尺度估计。runner 墙钟为
`88.0692 sec`。没有读取项目数据集、没有运行表格生成器、没有使用 GPU、没有消耗隐私预算。

独立 auditor 未导入正式 runner，也未导入项目 V2/V2b 数学核心；它重新生成固定 AR(1) 轨迹并
独立重算 OBM、first-ready、覆盖率、长期方差比、成本和全部门禁。审计结果为：

```text
passed                         true
scientific_payload_exact       true
scientific_sha256_exact        true
acceptance_status_exact        true
independent_boundary_checks    true
mismatch_count                 0
```

因此下面的 `candidate_failed` 是有效科学负结果，不是入口、哈希、任务缺失或实现漂移。

## 2. 正式失败项

预注册失败项恰好为：

```text
main.ar1_phi_0p8.coverage
main.ar1_phi_0p8.lrv_ratio
slow_pressure
```

主 family 的 first-ready 安全指标为：

| family | first-ready 数 | 覆盖率 | 覆盖门禁 | official LRV / 理论 LRV 中位数 | LRV 门禁 |
|---|---:|---:|---|---:|---|
| iid | 2000 / 2000 | 95.75% | 通过 | 1.0766 | 通过 |
| `phi=0.5` | 2000 / 2000 | 93.65% | 通过 | 0.9319 | 通过 |
| `phi=0.8` | 1997 / 2000 | **90.39%** | **失败** | **0.7856** | **失败** |

冻结要求是覆盖率位于 `92.5%..97.5%`、LRV 比中位数位于 `0.80..1.25`。`phi=0.8` 虽然几乎
全部轨迹最终出现过一次尺度相容，但第一次相容时的不确定性仍偏小；覆盖率比下限低约 2.11 个
百分点，LRV 比也低于下限。因此不能把它解释为边界误差后放行。

`phi=0.95` 的慢相关压力结果为：

```text
first-ready 数                         1373 / 2000
协议分支                               validated_release（因为 1373 >= 1000）
first-ready 覆盖率                     77.13%
official LRV / 理论 LRV 中位数         0.4210
压力门禁                               失败
```

这不是允许的“完全拒绝”分支，因为有 1373 条轨迹被放行；也不是安全放行，因为其正式长期方差典型
值只达到真实值的约 42.1%，覆盖率只有约 77.1%。该项是明确且幅度很大的安全失败。

## 3. 成本为什么通过但不能挽救候选

三个主 family 的资源统计为：

| family | 资源中位数 | 资源均值 | P95 | 未在上限内取得资格 |
|---|---:|---:|---:|---:|
| iid | 256 | 277.9 | 384 | 0 |
| `phi=0.5` | 256 | 351.6 | 768 | 0 |
| `phi=0.8` | 256 | 399.0 | 896 | 3 |

三类等权 pooled resource mean 为 `342.8053`，远低于冻结上限 1536；iid 与 `phi=0.5` 的资源
中位数门禁也都通过。单位 round 的正式 ESS 中位数仍保持：

```text
iid 0.9946 > phi=0.5 0.3541 > phi=0.8 0.1370
```

但低成本主要来自过早放行。协议明确规定安全门禁不能由成本补偿，所以成本全通过不能改变正式失败。

## 4. 失败机制

V2b v1 检查的是两个批长给出的相关膨胀是否接近：

```text
b = floor(sqrt(n))
2b
scale_ratio <= 1.25
```

结果验证了设计稿事先写明的漏洞：**两个尺度可以一起低估长期相关，并因此错误地显得相容。**

- `phi=0.8` 在 256 round 已有 1132 / 2000 条通过当前检查点；其 first-ready 中位数也是 256。
  这批过早选择使 first-ready 覆盖率降到 90.39%。
- `phi=0.95` 的真实相关膨胀为 39。短、长批长在早期都不足以看到完整相关时间，1373 条轨迹至少
  偶然相容过一次，但 first-ready 的正式 LRV 中位数只是真值的 42.1%。
- `phi=0.95` 在 2048 检查点本身只有 819 / 2000 条相容，而整段日程累计 first-ready 为 1373；
  即 554 条曾被一次性放行的轨迹到 2048 时已经不再相容。这是“一次通过即取得资格”不够稳定的
  描述性证据，不是新增正式门禁。

因此失败同时暴露两个问题：尺度的共同向下偏差，以及多次查看后抓住一次偶然相容。不能只把
`1.25` 改小或删掉 `phi=0.95` 来挽救同一候选。

## 5. 其余安全与契约结果

下面各项全部通过：

- 负相关 `phi=-0.5` 的 short/long raw ESS 比中位数在 15 个检查点均大于 1；
- short、long、official 正式 ESS 均未超过实际 round 数；
- 正式 MCSE 均未低于独立样本标准误；
- `core_not_estimable`、非有限输出、契约违规、ESS cap 违规、MCSE floor 违规均为 0；
- 轨迹身份违规为 0；
- 固定边界检查 16 / 16 通过；
- first-ready 的主 family 利用率门禁与 ESS 强弱排序通过。

这些结果说明 V2/V2b 数学实现、formal floor、输入身份和审计链本身工作正常；候选失败集中在
“尺度相容是否足以代表长期方差可靠”这一科学假设。

## 6. 正式结论与下一步边界

V2b v1 必须永久记录为：

```text
candidate_failed
```

禁止使用本批结果事后修改 `1.25`、检查间隔、first-ready 规则、family 或门禁后重新宣称通过，也
不得直接把该门禁接入 `test_300x10`、`nltcs` 或在线生成过程。

在新候选获得单独设计和全新 seed 协议之前，上一版 V2 人工验收支持的 `2048` 仍只是当前可用的
统一 ESS/MCSE 数值资格下限；它依然不是收敛轮数或停止条件。

若继续研究自适应版本，必须另立 V2c，并同时处理：

1. 两个批长共同低估时，不能仅凭相对接近放行；
2. 不能把某一个检查点的偶然相容永久记作取得资格。

具体新规则必须先讨论、再冻结新协议和新 seed；本结果本身不选择 V2c 公式。

## 7. 产物哈希

```text
protocol_manifest.json
  size    6,952 bytes
  SHA-256 1aae6b7a37a64a123a5585cb55d5bcb1454fa139367c0a7b473173d5e85e2bb7

adaptive_evidence_report.json
  size    6,117,401 bytes
  SHA-256 1a3f9e7eb483d0319167b4e9433519e5dd04fd7740aeeeba12192b1d6cbd104c

independent_audit.json
  size    1,739 bytes
  SHA-256 df0a2b46da135c21b6119f618551f4249fd68c49f9ccaaae7146d87d4aabbcb2
```

上述目录虽然属于通常被忽略的 `outputs/`，但本次三份冻结产物将由正式结果提交显式纳入版本控制；
后续普通运行输出仍保持忽略。归档保留原始 JSON 字节与上述 SHA-256，不复制、不改写正式产物。
