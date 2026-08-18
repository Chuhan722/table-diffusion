# Issue #53：`test_300x10` 残差几何 fresh-seed 确认协议

## 1. 目的与结论边界

本实验只确认一个问题：上一轮 seed 310–312 的分阶 held-out 小差值，在全新随机轨迹上是否稳定到足以支持一个 `test_300x10` 统一残差几何候选。

- 比较 `absolute`、`sqrt_relative`、`relative` 三臂。
- 只跑 `test_300x10`；不重跑 nltcs，不据此作跨数据 canonical 声明。
- 查询阶数分别报告，禁止把 measured/未测量 2/3/4-way 汇成总分。
- 本协议在查看 seed 313–317 的任何生成结果前冻结；不得按结果增加 seed、改门禁、调 floor/gamma/rho/alpha/P。
- 这是 PR 前的 fresh-seed 确认实验，不修改或合并等待外部审查的 PR #63。

## 2. 固定生成矩阵

```text
dataset  = test_300x10
arms     = [absolute, sqrt_relative, relative]
seeds    = [313, 314, 315, 316, 317]
cases    = 15
```

除 seed 和 `residual_geometry` 外，完全复用已冻结的 P=6 三臂实验：

```text
rho                                  = 0.01
P / inner patience                   = 6 natural-work ticks
n_rounds / candidate_budget          = 6000 / 6000
selection_scale_invariant            = true
selection_scale_invariant_min_spread = 1e-3
alpha_schedule_mode / fixed_alpha    = fixed / 16
diffusion direction normalization    = initial_rms
eta / mu                             = 0.5 / 0.01
factorized_gibbs_sweeps              = 0
tol                                  = +inf
max_retries                          = 0
output                               = terminal_current
```

三臂定义保持不变：

```text
absolute      = sign(raw) * magnitude / n_records
sqrt_relative = sign(raw) * magnitude / sqrt(max(target, 8)) / n_records
relative      = sign(raw) * magnitude / max(target, 8) / n_records
```

在线停止仍只看 generation-visible squared loss 的自然工作时钟，不读 held-out、raw reference 或离线 L1。

## 3. 固定评价查询

评价器只在 15 条生成轨迹全部物化后运行。查询身份固定为：

| 查询组 | 数量 | 角色 | 身份 SHA-256 |
|---|---:|---|---|
| `measured_1way` | 25 | 1-way 安全门禁 | `b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c` |
| `measured_2way` | 20 | 次要描述 | `ea558bd958af3fa996925b159657973ff0d6a0dc873efbc0e0d41856f9e6887e` |
| `measured_3way` | 5 | 次要描述 | `cb2a96159985cf0a241e82ef6ea90475910e98bfee311c9778c33696bcd5aea2` |
| `unmeasured_2way_all` | 531 | primary | `7d88a2db88a4576bb54bed341a3a8ccfbfc11f368662ad3b513e8fa863b5647f` |
| `heldout_3way_512` | 512 | primary | `d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a` |
| `heldout_4way_512` | 512 | primary | `2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171` |

未测量 2-way 继续按公开 marginal 域枚举全部 548 个标准 cell，排除与 measured 语义完全相同的 17 条后保留全部 531 条。身份必须在加载 raw reference 前复算为上表 SHA；随后才能读取固定 SHA 的原表附加答案。3/4-way 使用既有 result-blind 冻结文件，不重新抽样。

## 4. 固定指标与配对方式

每个查询组、每个 arm 分别报告：

- 5 个 paired seed 各自的 mean absolute count error；
- 全部 query×seed 的 mean、median、p90、max absolute count error；
- mean 除以 300 的 normalized L1；
- exact-match rate；
- 相对 `absolute` 的 query×seed better/tie/worse 和 paired-seed better/tie/worse。

生成成本分别报告 terminal measured L1、squared loss、rounds、normalized work、elapsed 和 A/B/C。质量与成本不标量化。

## 5. 结果前冻结的判定规则

先做执行资格门禁：15/15 必须由 A/`fit_target_reached` 或 B/`early_stopped` 正常结束；只要有一个 C/`resource_cap_reached`，本轮统一候选判定为 `inconclusive_resource_cap`。

对 `sqrt_relative`、`relative` 分别相对 `absolute` 计算：

### 5.1 未测量联合查询门禁

候选必须同时满足：

1. `unmeasured_2way_all`、`heldout_3way_512`、`heldout_4way_512` 三组的总体 mean delta 都 `<= 0`；
2. 至少一个 primary 组 mean delta `< 0`，并且该组 paired-seed candidate-better 数量 `>= 4/5`。

通过记为 `unseen_pareto_pass=true`。这里使用严格零边界，不设事后 non-inferiority margin。

### 5.2 measured 1-way 安全门禁

`measured_1way` 总体 mean delta 必须 `<= 0`，通过记为 `measured_1way_safety_pass=true`。measured 2/3-way 只描述，不参与统一候选门禁。

### 5.3 分类

- 执行资格、未测量门禁和 1-way 安全门禁全通过：`supports_unified_test_candidate`。
- 未测量门禁通过但 1-way 安全失败：`unseen_gain_with_measured_1way_tradeoff`。
- 其他正常完成组合：`mixed_no_unified_test_candidate`。
- 两个候选都不获得 `supports_unified_test_candidate`：保留 `absolute` 作为 test 参考，不再用本批结果调新公式；把三种 geometry 记录为 workload 级 Pareto/负证据，下一板块先冻结跨方案质量—成本门禁。

该分类只决定是否存在值得继续讨论的 test 候选，不自动改默认值，不自动选择跨数据 canonical 配置。

## 6. 执行与信息流

正式采集使用用户指定的服务器 `root@10.8.176.53:6006`：

```text
server              = Cardiff_VM_6 / RTX A6000 server
generator device    = numpy
CUDA_VISIBLE_DEVICES= empty
worker_count        = 5
parallel unit       = one seed shard
within-shard order  = absolute -> sqrt_relative -> relative, serial
```

并行只缩短墙钟，不共享 RNG 或状态；每条轨迹 seed、输入与参数独立。采集阶段不得读取 raw reference。评价阶段必须审计 collection protocol SHA、执行 commit、15 张终态表 SHA、六组查询身份和所有固定输入 SHA，随后才允许读取 reference；报告必须记录 `raw_reference_data_accessed=true`、`privacy_budget_consumed=false`。

## 7. 产物与 PR 边界

```text
outputs/issue53_test_residual_geometry_confirmation_v1/
  seed_313 ... seed_317/      # 原始终态表与 shard manifest
  collection_report.json     # 15-case 采集聚合
  evaluation_report.json     # 六组独立评价与冻结分类
  query_seed_errors.csv
```

输出目录不覆盖；正式采集与评价入口不暴露 seed、arm、阈值或科学参数覆盖。结果归档、相关测试和工作树审计通过后，可以推送当前研究分支并创建依赖 PR #63 head 分支的 stacked PR；不得自行 review 或 merge。
