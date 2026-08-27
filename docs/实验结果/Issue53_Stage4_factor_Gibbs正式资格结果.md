# Issue #53 Stage 4：factor random-scan Gibbs 正式资格结果

日期：2026-08-22（正式运行于 2026-08-20 完成）

## 1. 结论

在冻结的 `test_300x10` / `nltcs` higher-order workload、`tau=2`、fixed α16、relative residual
floor=8、scale-invariant donor、P=6、无门控、terminal-current 身份下，factor random-scan Gibbs
的最小充分档正式确定为：

```text
qualified_random_scan_s8
```

qualification seeds `333..337` 形成两数据各 5 条独立来源轨迹；每条轨迹固定选择
`initial/work_q25/work_q50/work_q75/terminal` 五个状态，共 50 个状态。冻结 runner 按共享
`8 -> 16 -> 32` 顺序执行；sweeps=8 时两个数据集已经同时 valid 且通过全部性能门槛，因此依协议停止，
没有运行 16/32。

本结论只证明当前冻结提议分布上的条件核正确性与 8 sweeps 混合充分性。它不证明 factor 外层生成优于
same-tau independent，不把 factor 核升级为公共默认，也不外推到其他 tau、workload、参数或罕见宽
active set。

## 2. 正式身份

```text
worktree               /home/chuhan/projects/table-diffusion-issue53-factor-gibbs-stage4
branch                 research/issue53-factor-gibbs-stage4
execution commit       e7cbcc0beaf6718f7f4ad148a8ee07f8cc9a089f
execution dirty        false
mode                   qualification
formal_result_valid    true
seeds                  [333, 334, 335, 336, 337]
protocol SHA-256       6a2834db4cb75fffbaac3330bbb1923fa2e864572ca05ac901c3142ecd443680
execution scientific   c992dcdab9b0fc8dc2b6816c15317a035ed61adaed6b081bcf3a6d817816f4de
```

固定科学身份：

```text
source kernel          independent directional mask, sweeps=0
candidate kernel       factor random-scan Gibbs with replacement
temperature            tau=2（source/evaluation 相同）
candidate sweeps       [8, 16, 32]，hard cap=32
alpha                  fixed 16
rho / eta / mu         0.01 / 0.5 / 0.01（probe mu=0）
residual               relative, floor=8
donor                  scale-invariant
direction scale        initial_rms
direction/Gibbs clip   30
termination            P=6 natural-work ticks，C=6000 raw rounds
output                 terminal current
acceptance             no gate，no retry
initialization         marginal
```

`test_300x10` workload 为 30 条 2-way、15 条 3-way、5 条 4-way；`nltcs` 为 479 条 2-way、
522 条 3-way。held-out 不进入 Gibbs energy。

## 3. 来源轨迹与状态库

10/10 条来源轨迹均由 `early_stopped` 正常结束，没有触及 6000 轮资源上限：

| dataset | seed | rounds | terminal normalized work | recorded natural-work states |
|---|---:|---:|---:|---:|
| test_300x10 | 333 | 1397 | 14.0000 | 15 |
| test_300x10 | 334 | 1758 | 18.0000 | 19 |
| test_300x10 | 335 | 1372 | 14.0067 | 15 |
| test_300x10 | 336 | 1487 | 15.0100 | 16 |
| test_300x10 | 337 | 1209 | 12.0133 | 13 |
| nltcs | 333 | 1895 | 19.0044 | 20 |
| nltcs | 334 | 2502 | 25.0040 | 26 |
| nltcs | 335 | 2505 | 25.0032 | 26 |
| nltcs | 336 | 3400 | 34.0031 | 35 |
| nltcs | 337 | 2196 | 22.0040 | 23 |

每条轨迹的五个资格状态均按冻结 natural-work 最近点与更早索引 tie-break 规则选择；状态库 manifest
固定 dataset、seed、stage group 顺序并绑定五个 seed shard。全部 binding gates 为 true。

## 4. 正式 mixing 结果

冻结门槛为 global、五个 stage groups 与所有非空 active-width groups 同时满足 TVD `<=0.05`、
gap recovery `>=0.80`，并要求零 conditional clip hit。sweeps=8 的正式结果为：

| dataset | global TVD | gap recovery | valid | passed |
|---|---:|---:|---|---|
| test_300x10 | 0.0015438274 | 98.7275% | true | true |
| nltcs | 0.0000104018 | 99.9486% | true | true |

结果不是擦线通过：

- 50 个状态中最差单状态为 `test_300x10 seed=337 initial`：TVD 0.0133081、gap recovery
  97.6778%，仍明显优于 0.05 / 80% 门槛。
- test 最差 gated stage 是 initial（TVD 0.0068328）；非空 width groups 中最高 TVD 在 5–8
  （0.0054873），最低 recovery 在 1–4（98.6644%）。
- nltcs 最差 gated stage TVD 为 0.0000442；最差非空 width group 是 5–8
  （TVD 0.0001573、recovery 99.8834%）。
- test 的 width 9–12/13–16、nltcs 的 width 13–16 在本批提议中为空，依冻结规则不 gate；nltcs
  width 9–12 只有 1 个 participating active row，因此不外推为所有罕见宽 active set 的普遍保证。

## 5. 数值正确性与生产语义

| dataset | max energy error | tolerance ratio | clip hits | exact tape replay |
|---|---:|---:|---:|---:|
| test_300x10 | 2.60e-18 | 2.60e-8 | 0 | 0 / 13243 mismatch |
| nltcs | 1.39e-17 | 1.39e-7 | 0 | 0 / 12407 mismatch |

- 两数据各 8 项 validity gates 全 true；概率均有限、非负、归一化，最大概率和误差
  `<=8.88e-16`。
- 50 个状态的最大 raw conditional logit 绝对值为 11.8063，距 clip=30 有充足余量。
- production exact-tape replay 共 25,650 次比较，零失配。
- 修复后的 torch oracle 势能累加继续在 float64 路径上工作；正式 fresh seeds 的能量恒等误差保持
  ulp 级，没有复现 development v1/v2 的 float32 结构失败。

## 6. 独立审计与产物身份

独立 auditor 不导入 Stage 4 builder/runner，重新绑定状态库、协议、状态顺序、共享条件、分组、门禁与
停止规则，重算得到：

```text
status                              complete
passed                              true
result                              qualified_random_scan_s8
selected_minimal_sufficient_sweeps  8
attempted_sweeps                    [8]
recomputed state count              50
```

本地 ignored 归档及只读复核的文件 SHA-256：

```text
outputs/issue53_stage4_qualification_v1/state_library.json
  3c7475e89d693bd2240846bb78dec6b7a8d2abc14a71beea25fd6be3e2a02561
outputs/issue53_stage4_qualification_v1/mixing_report.json
  15751180b96c6a466f7096a63935d93eb60f47b836f2aa9462a1842ee58b7fa5
outputs/issue53_stage4_qualification_v1/mixing_audit.json
  fde7929a39cb039d26dd56b91063fa4151a639ea0e483ba8b9303e912a42ed6d
```

状态库 scientific SHA 为
`c9788506fde41cdf3767f1b4de48d90304e7ce482a60f88754e21107e95ceb5a`。auditor 内部记录的
report SHA 与上表一致。三个产物继续保留在 ignored 本地归档；本结果文档不把约 171MB 状态库或
约 2.1MB report 强行纳入 Git，也不改写任何正式产物。

## 7. 执行备注

1. 第一次 launch 通过 commit/protocol/clean-tree preflight 并进入 seed 333 start，但没有生成 shard
   JSON；保留的 `serial_pipeline_launch1_failed.log` 与 0-byte launch 日志如实记录该技术重启。
   随后完整串行运行生成五个 complete shard，并由同一 clean commit 聚合、mixing、audit。无结果的
   首次启动不污染正式结论。
2. qualification 脚本固定 `CUDA_VISIBLE_DEVICES=0`，development v3 记录使用物理 GPU 1。两张均为
   24 GiB RTX 4090；冻结协议绑定 runtime `device="cuda"`，不绑定物理卡号。正式 report/auditor
   判定身份有效。后续仍遵守单卡串行纪律，具体物理卡按运行时授权。

## 8. 与 development v3 的一致性

| dataset | development v3 TVD / recovery | qualification TVD / recovery |
|---|---:|---:|
| test_300x10 | 0.0011614 / 98.91% | 0.0015438 / 98.73% |
| nltcs | 0.00001085 / 99.9526% | 0.00001040 / 99.9486% |

正式 fresh seeds 没有复现结构失败或明显的 development-seed 过拟合信号。test 略弱但仍有大幅门槛
余量；nltcs 几乎逐指标重现 development v3。性能保持同水平、能量恒等继续在 ulp 级通过，完成了
float32 根因诊断、最小修复、development 与 qualification 的科学闭环。

## 9. 下一步边界

Stage 4 到此正式收口。下一研究阶段是 Stage 5 same-tau independent vs factor 外层公平比较：先写
结果前协议，固定 independent sweeps=0 与 factor sweeps=8，使用不得复用 `323..327`、`333..337`
的全新 paired seeds，再比较 measured/held-out 质量、terminal-current、支持集、多样性、validity、
normalized work、Gibbs 微步、查询评价次数与墙钟。

本文不授权 Stage 5 实现或实验，不授权 push、建 PR、Issue 评论、公共默认值修改，也不解决当前
PR #65 的堆叠冲突。
