# Issue #53 V2c：三尺度双确认有效证据人工验收结果

> 正式结论：`candidate_failed`；独立审计 `passed=true`。
> V2c 改善了 V2b 在主相关 family 上的条件安全性，但未达到冻结的取得资格数量要求，慢相关压力也
> 仍不安全。该结论只否决 V2c v1 的自适应数值资格规则，不判断生成收敛，不读取真实数据，也不
> 否决无门控残差引导扩散核本身。

## 1. 冻结身份

```text
Git commit
  f9db5d6fb4af9bccf36c5fad3c1c2565eb8b57c5

protocol SHA-256
  a9930b440f3483d0bb2e6ad8d3bbf4cd8db097b2d85deed69216a475679cbc04

scientific result SHA-256
  77c054980c4da46bc385fbadd8dbe79e968b6dfada704fadd5eac1329983de0c

正式输出目录
  outputs/issue53_v2c_three_scale_effective_evidence_f9db5d6/
```

正式入口从包含 untracked 在内的干净工作树运行，且只运行一次。固定矩阵为 5 个 family、每类
2000 条、每条 2048 round，共 10000 条人工轨迹、150000 次检查点分类与 450000 次尺度估计；
runner 墙钟为 `128.3000 sec`。没有读取项目数据、运行表格生成器、使用 GPU 或消耗隐私预算。

独立 auditor 未导入正式 runner，也未导入项目 V2/V2b/V2c 数学核心；它重新生成全部固定 PCG64
轨迹，独立重算三尺度 OBM、相邻双确认、first-ready、回撤诊断、family 汇总和全部门禁。审计结果：

```text
passed                         true
scientific_payload_exact       true
scientific_sha256_exact        true
acceptance_status_exact        true
independent_boundary_checks    true
mismatch_count                 0
```

因此下面的 `candidate_failed` 是有效科学负结果，不是实现漂移、文件篡改或 runner 损坏。

## 2. 正式失败项

预注册失败项恰好为：

```text
main.ar1_phi_0p5.ready_count
main.ar1_phi_0p8.ready_count
slow_pressure
```

主 family 的 first-ready 指标为：

| family | first-ready 数 | 数量门禁（至少 1850） | 条件覆盖率 | 条件 LRV 比中位数 | 资源中位数 |
|---|---:|---|---:|---:|---:|
| iid | 1963 / 2000 | 通过 | 95.26% | 1.0653 | 384 |
| `phi=0.5` | **1646 / 2000** | **失败** | 94.41% | 0.9920 | 1024 |
| `phi=0.8` | **1631 / 2000** | **失败** | 93.44% | 0.8853 | 1024 |

三个 family 的条件覆盖率都位于冻结区间 `92.5%..97.5%`，条件 LRV 比中位数也都位于
`0.80..1.25`。因此 V2c 没有重复 V2b 在 `phi=0.8` 上“取得资格后仍明显低估不确定性”的同一种
主门禁失败；它失败在可用性：`phi=0.5` 与 `phi=0.8` 分别只有 82.30% 和 81.55% 的轨迹曾取得
资格，均低于预注册的 92.5% 数量下限。安全指标不能只在被规则挑中的较容易子集上通过而忽略大量
未放行轨迹，所以不能事后删除 ready-count 门禁。

`phi=0.95` 慢相关压力结果为：

```text
first-ready 数                         606 / 2000
协议分支                               unsafe_sparse_release
first-ready 条件覆盖率                 84.49%
official LRV / 理论 LRV 中位数         0.5316
压力门禁                               失败
```

它既不是允许的完全拒绝（必须 0 条放行），也没有达到安全放行分支要求的至少 1000 条；而且已放行
子集的覆盖率和 LRV 比仍远低于安全下限。这说明第三尺度和相邻双确认减少了危险放行，却没有把慢相关
规则变成可靠的自适应资格判据。

## 3. V2c 相比 V2b 改善了什么、代价是什么

同类人工协议下的关键变化为：

| 指标 | V2b | V2c |
|---|---:|---:|
| `phi=0.8` first-ready 数 | 1997 | 1631 |
| `phi=0.8` 条件覆盖率 | 90.39% | 93.44% |
| `phi=0.8` 条件 LRV 比中位数 | 0.7856 | 0.8853 |
| `phi=0.95` first-ready 数 | 1373 | 606 |
| `phi=0.95` 条件覆盖率 | 77.13% | 84.49% |
| `phi=0.95` 条件 LRV 比中位数 | 0.4210 | 0.5316 |

第三尺度取最大风险与相邻双确认确实向正确方向移动：`phi=0.8` 的条件安全门禁由失败变为通过，慢
相关的危险放行数量也减少。但改善来自显著延后或拒绝，而不是得到一个同时安全、可用的自适应信号。
主 family 等权 pooled resource mean 从 V2b 的 `342.8053` 上升到 `958.0587`；成本仍在冻结上限
1536 内，却不能补偿两个主 family 的取得资格数量失败。

## 4. 回撤诊断说明规则仍不稳定

V2c 预先要求记录 first-ready 后再次不相容，但不把它临时加成新门禁。正式描述性结果为：

| family | 曾 first-ready | 此后至少一次不相容 | 比例 | 2048 当前连续双确认通过 |
|---|---:|---:|---:|---:|
| iid | 1963 | 712 | 36.27% | 1652 |
| `phi=0.5` | 1646 | 1117 | 67.86% | 994 |
| `phi=0.8` | 1631 | 1118 | 68.55% | 950 |
| `phi=0.95` | 606 | 355 | 58.58% | 328 |

即使 iid，2048 时当前双确认状态也只有 1652 / 2000；对 `phi=0.5/0.8` 则不足一半。三尺度比例
仍是随有限样本明显抖动的量，相邻两点通过只能减少偶然放行，不能把它变成稳定的自适应证据。若用
first-ready 永久放行，会保留大量后来回撤的轨迹；若坚持使用当前状态，又会在同一轨迹上频繁撤销
数值资格。两种解释都不适合作为简单可靠的内层生成轮数判据。

## 5. 成本、负控制与全局契约

成本门禁全部通过：iid 资源中位数为 384，不超过 512；`phi=0.5` 为 1024，恰好不超过 1024；
三个主 family pooled mean 为 `958.0587`，对应约 53.2% 的上限内节省。正式 ESS 中位数顺序也正确：

```text
iid 0.9761 > phi=0.5 0.3326 > phi=0.8 0.1244
```

负相关 `phi=-0.5` 的三个 raw ESS 尺度、formal ESS cap 和 MCSE floor 全部通过。全局
`core_not_estimable`、非有限输出、契约违规、ESS cap 违规、MCSE floor 违规和轨迹身份违规均为 0；
固定边界检查 22 / 22 通过。它们说明数学实现、输入身份和审计链工作正常，但不能覆盖正式的主
family 可用性与慢相关安全失败。

## 6. 正式结论与下一步边界

V2c v1 必须永久记录为：

```text
candidate_failed
```

冻结协议已预先规定失败动作：`no_v2d_return_to_v2_fixed_2048`。因此不能根据本结果继续增加第四
尺度、第三次确认、调整 1.25、检查点或 family，再以同一路线宣称通过；也不能把 V2c 门控接入
`test_300x10`、`nltcs` 或真实生成过程。

当前可保留的是 V2 人工验收支持的统一 `2048 outer rounds` ESS/MCSE 数值资格下限。它只是固定
资源下限，不是收敛证明、自动停止条件或生成质量保证。Issue #53 后续若继续，应回到无门控残差
引导扩散核本身：固定资源下比较完整生成结果，而不是继续从同一条三尺度门控路线追加 V2d。

## 7. 产物哈希

```text
protocol_manifest.json
  size    9,188 bytes
  SHA-256 daecf32977f98f28108d2de29b9ca3a51c96554f75e87b5748a62eef7e014ba9

three_scale_evidence_report.json
  size    8,413,656 bytes
  SHA-256 1404c1ba49e13259c047d55fa1d68e77ae5d954874a59aa5606e831fd7d7eaaf

independent_audit.json
  size    1,559 bytes
  SHA-256 1a2c652c091779f74eadb4a3f548f62aa98096d1900df81ca8165423efca6b04
```

上述目录通常由 `outputs/` 规则忽略；本次三份冻结产物由正式结果提交显式归档，保留原始 JSON
字节和上述 SHA-256。后续不得改写、覆盖或重复运行同一正式矩阵。
