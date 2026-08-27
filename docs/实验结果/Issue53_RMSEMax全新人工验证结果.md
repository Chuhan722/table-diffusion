# Issue #53：RMSE + max 全新人工验证结果

> 正式结论：`candidate_failed`；执行有效性门禁全部通过。
> 只有最简单的 `marginal_skew` 4/4 在标准化工作量 20 及之前达到
> `query-count RMSE <= 1 AND max absolute count error <= 2`；
> `ring_pair` 与 `nested_overlap` 合计 0/8 按时达标。该结果否决当前 v1
> “固定生成核 + 工作量 20 + RMSE/max 目标”的在线接入资格，不是执行故障，
> 不证明生成链收敛，也不否决 RMSE/max 作为已达标状态的描述性质量指标。

## 1. 冻结身份与执行边界

```text
正式生成 commit
  898b76c2a8e60093888bfe05ffce74b89a124c5e

protocol SHA-256
  cb1224ac797191b74aa40f7baadfab08928b5cb25414971fe8ee091a297d433a

protocol document SHA-256
  012d2507f7b3a79a7fb566047a7f2dae3dbf2e9d77e30dbeb5da44bcfbff6245

scientific result SHA-256
  a29fa02edf7492ab171da50a61eb532a33ce1b47db3f318f3ec2912ff448da49

正式输出目录
  outputs/issue53_rmse_max_artificial_898b76c/
```

正式入口从包含 untracked 在内的干净工作树运行，输出目录事先不存在。矩阵为
3 family × 2 全新 seed × 2 rho，共 12 条小型人工轨迹；`rho=1` 跑 40 rounds，
`rho=0.25` 跑 160 rounds，总计 1200 个生成轮次。实际 case 计算墙钟为
`14.441969 sec`。运行使用 NumPy/CPU，没有读取项目真实数据、真实参考表，没有使用
GPU，也没有消耗隐私预算。

正式 preflight 绑定了 Git commit、结果前协议及 9 个证据源文件的 SHA-256；每条
case 的完整轨迹均为 `max_rounds` 结束、每轮一次已应用 proposal、无拒绝或重试，
查询答案与 transition clock 对齐，资源边界和至少 10 次等效扫描尾部均存在。12 张
选中表都完成同 checkpoint 前缀重放，表、RNG、查询答案、count error、loss、RMSE、
MAX 和时钟逐项对拍。全部 validity 与 checkpoint replay 字段均为 true。

独立只读字节核验结果：

```text
scientific payload canonical SHA 与报告一致   true
protocol manifest 文件 SHA 与报告一致         true
12 张 selected table 文件 SHA 与报告一致      12/12
```

因此下面的失败是有效科学负结果，不是身份漂移、文件损坏或 runner 故障。

## 2. 冻结主门禁结果

唯一科学通过条件预先固定为：12/12 都在第一个 `work>=20` 的真实边界状态及
之前，同时达到 RMSE 和 MAX 两项限制。正式结果为：

```text
execution_validity_pass                true
matrix_identity_pass                   true
qualified_by_resource_boundary_count   4
required_qualified_count               12
scientific_pass                        false
status                                 candidate_failed
```

按 family 汇总：

| family | case 数 | work<=20 达标 | 完整 horizon 曾达标 | 首次达标 work |
|---|---:|---:|---:|---|
| `marginal_skew` | 4 | **4** | 4 | 2.0417、3、3.0417、9 |
| `ring_pair` | 4 | **0** | 2 | 22.9688、39.0312；另 2 条从未达标 |
| `nested_overlap` | 4 | **0** | 2 | 21.4219、35.5938；另 2 条从未达标 |
| 总计 | 12 | **4** | 8 | — |

按 rho 汇总：

| rho | case 数 | work<=20 达标 | 完整 horizon 曾达标 |
|---:|---:|---:|---:|
| 1.0 | 6 | 2 | 2 |
| 0.25 | 6 | 2 | 6 |

所以结果不是某一个 seed 的偶然失败：两个包含联合结构的 family 在两个 seed 和两个
rho 下都没有一条按时通过。较小 rho 的四条复杂 case 最终能够达标，但全部晚于冻结的
工作量 20；rho=1 的四条复杂 case 到完整工作量约 40 仍未达标。

## 3. 逐 case 结果

`selected` 对达标 case 是第一张合格 current table；对未按时达标 case 是资源边界及
之前 squared loss 最低、并列最早的兜底表。

| family | seed | rho | work<=20 | 首次达标 work | selected work | selected RMSE | selected MAX | selected loss |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| marginal_skew | 20260901 | 1.0 | 是 | 3.0000 | 3.0000 | 1.0000 | 1 | 1.5 |
| marginal_skew | 20260901 | 0.25 | 是 | 2.0417 | 2.0417 | 0.5774 | 1 | 0.5 |
| marginal_skew | 20260902 | 1.0 | 是 | 9.0000 | 9.0000 | 0.5774 | 1 | 0.5 |
| marginal_skew | 20260902 | 0.25 | 是 | 3.0417 | 3.0417 | 1.0000 | 1 | 1.5 |
| ring_pair | 20260911 | 1.0 | 否 | 从未 | 3.0000 | 3.9497 | 5 | 78.0 |
| ring_pair | 20260911 | 0.25 | 否 | 39.0312 | 19.7812 | 3.5777 | 6 | 64.0 |
| ring_pair | 20260912 | 1.0 | 否 | 从未 | 17.0000 | 3.7283 | 6 | 69.5 |
| ring_pair | 20260912 | 0.25 | 否 | 22.9688 | 19.3438 | 2.5690 | 4 | 33.0 |
| nested_overlap | 20260921 | 1.0 | 否 | 从未 | 15.0000 | 10.3409 | 19 | 802.0 |
| nested_overlap | 20260921 | 0.25 | 否 | 21.4219 | 20.0000 | 1.0954 | 3 | 9.0 |
| nested_overlap | 20260922 | 1.0 | 否 | 从未 | 3.0000 | 10.6301 | 20 | 847.5 |
| nested_overlap | 20260922 | 0.25 | 否 | 35.5938 | 19.8281 | 1.1547 | 2 | 10.0 |

8 条失败 case 中有 6 条在 selected checkpoint 之后出现严格更低 loss；全部 12 条中为
9 条。特别是四条复杂 `rho=0.25` case 都在工作量 20 后继续改善并最终达标。这说明
工作量 20 不是生成动力学的普遍稳定点，也不能把 `resource_cap_reached` 写成“已经收敛”。

## 4. 这个负结果说明什么

本次正式结果支持以下有限结论：

1. RMSE/max 两项接口和同 checkpoint 返回契约工作正确；简单一维偏态问题可以较早满足。
2. 在冻结的无门控生成核下，工作量 20 不能保证包含二阶环或高阶嵌套关系的问题达到同一
   质量目标。
3. 当前 v1 不能作为跨 workload 的正常在线停止方案接入 `run_evolution`。若照此接入，
   8/12 case 会以 `resource_cap_reached / fit_target_reached=false` 返回未达标兜底表。
4. 失败不能单独归因于 RMSE<=1、MAX<=2、资源 20、固定 alpha、absolute residual 或
   0-sweep independent 核中的某一个因素；本矩阵只否决它们组成的冻结候选，不能事后从同一
   结果挑一个参数修改后宣称修复。

`QueryFitThresholds.exact_integer_counts()` 和 `assess_query_fit(...)` 可以继续保留为纯
质量评估工具，因为它们正确回答“当前这张表是否达到所定义的误差标准”。但正式矩阵没有支持
把该目标与工作量 20 组合成可直接接入的通用自动停止规则。

## 5. 冻结失败动作与下一步边界

结果前协议已经规定：任意 case 迟到或从未达标即失败，失败后保留负结果并停止接入讨论；
不得在同一结果上把 MAX 2 改大、把资源 20 延长、替换 seed/family 或重跑更有利的矩阵。

因此本版结束时必须保持：

```text
candidate_failed
online integration = forbidden
real dataset run = not authorized
post-result retuning/re-run = forbidden
```

下一步不是把规则接到 nltcs/plants/test_300x10，也不是恢复已否定的 3+3、V2b/V2c 或固定
2048 轮。若继续 Issue #53，应先与用户重新讨论一个新的问题：是把跨数据统一资源上限仅作为
“未达标也必须返回”的工程失败出口，还是先改进生成核使复杂关系更可靠地吸收测量。任何新候选
都要另立结果前协议和全新证据，不能复用本矩阵做事后调参证明。

## 6. 产物哈希

```text
protocol_manifest.json
  size    8,406 bytes
  SHA-256 cc49b278276846879d3fc44767451742fe701400a69eed1eff4d92390f7f144c

rmse_max_evidence_report.json
  size    63,906 bytes
  SHA-256 92f1588a735bced793dc9a6086c304a1b56a6f99b2d2e08354d31834183ec29f

scientific payload canonical SHA-256
  a29fa02edf7492ab171da50a61eb532a33ce1b47db3f318f3ec2912ff448da49
```

报告另逐文件绑定 12 张 `selected_tables/*.csv` 的 SHA-256；独立字节复核为 12/12
一致。`outputs/` 默认被忽略，本次 manifest、完整报告和 12 张 selected table 由正式负结果
归档提交显式纳入版本控制。后续不得覆盖、改写或重复运行同一正式矩阵。
