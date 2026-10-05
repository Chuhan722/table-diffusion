# FitnessOnly 地板段 Eta 降温 v5 结果前协议（草稿待审）

状态：**草稿，等待用户审查**。本协议起草时未修改任何代码、未运行任何新
实验。协议获批后，实现与正式运行仍需分别单独授权。

## 1. 目的与研究问题

v3（预算加倍）判定 `truncation_hypothesis_rejected`、v4（地板减半）判定
`quality_regression_under_lower_floor`，且 v4 中 nltcs 本底 1.1639→1.1754
未收窄：**降低每轮参与行数（频率）不能收窄稳态震荡带**。据此提出配对
机制假说——带宽由**单次改动幅度**主导：一行被选中后，每个与 donor 不同
的属性以 eta=0.5 概率被整批复制，单次跳变 ≈ 差异属性数 × eta。

本 v5 屏检验唯一假说（**跳变幅度假说**）：

> 保持 rho 时间表为 v3 原样（floor=0.001，修复频率不降），仅在 rho
> 落地后对 eta 做同构几何降温 0.5 → 0.25（单步幅度减半），nltcs 稳态
> 震荡带将收窄，终态输出漂移可达原目标 drift ≤ 1.100，且不损失
> 测量/held-out 质量。

与 v4 构成机制拆解配对：v4 降频率（无效）vs v5 降幅度——无论结果如何，
归因证据链完整。v5 不加剧 v4 的小表冻结机理（行照常参与，每步变小），
但小步修复变慢是新风险，由预算与形态判定兜住。

本屏仍是单开发 seed 的 development 实验，不产生正式效果声明，
promotion gate 保持关闭（`formal_claim_allowed=false`）。

## 2. 变更定义与不变量

```text
机制新增   ：evolution.py 增加 eta 三段式退火（eta_anneal_start_round /
             eta_anneal_rounds / eta_anneal_end），公式与 rho 退火完全
             同构、纯轮数驱动、不读残差、不读 n_rounds；三参数全 None
             时逐位退化为现行为（见 §5 等价性合同）
eta 时间表 ：H_eta=1500、D_eta=600、end=0.25 ——
             t≤1500 恒 0.5；1500<t<2100 按 0.5·(0.5)^((t-1500)/600)
             几何降至 0.25；t≥2100 恒 0.25
rho 时间表 ：v3 原样逐字不变（H=900、D=600、end=0.001）
预算       ：n_rounds 9000（运行长度非时间表参数；eta 减半后单步修复
             量减半，比照 v4 处理给足地板段余量；形态窗口与 v4 对齐）
不变       ：seed 9908、两数据集、两臂配对、冻结输入与 SHA-256、
             rho0=0.01、eta0=0.5、mu=0.01、fixed alpha=16、
             relative residual floor=8、geometric scale-invariant
             donor、冻结 marginal 初始化、tol=+inf、
             horizon_invariant=True、terminal current 输出、
             固定轮数无条件走满
参与量     ：nltcs 地板段仍 ≈16.2 行/轮、test 仍 ≈0.3 行/轮（频率
             不变）；每参与行被复制属性期望减半（幅度减半）
输出目录   ：outputs/fitness_only_eta_cooling_dev_seed9908_v5/
             （独立，不覆盖 v1–v4）
```

## 3. 保温段前缀一致审计（时间表盲性的可证伪检验）

v5 的 rho 时间表与 v3 逐字相同；eta 时间表在 **t∈[0,1500]（含
t=1500，共 1501 轮）恒为 0.5**（t=1500 时 progress=0，`0.5·(0.5)^0.0
= 0.5` 浮点精确），自 t=1501 起分叉。eta 只作为
`rng.random(n) < eta` 的阈值，不改变随机数消费顺序。因此 v5 各臂前
1501 轮必须与 v3 对应臂**逐位一致**。生成完成后 fail-closed 核对：

- `loss_history[:1501]` 与 v3 对应臂逐位相等；
- `rho_schedule_history[:1501]` 与 v3 对应臂逐位相等；
- `initial_table_sha256`、`primary_rng_post_initialization_state_sha256`
  与 v3 对应臂相等（终态 RNG 不同属预期）。

任一不一致 → 运行判为无效（`schedule_blindness_violated`），说明 eta
退火实现改变了随机流或主循环存在隐藏依赖，必须先修架构，不得解读任何
质量结果（产物保留供诊断，不打开 held-out）。

v3 参照产物只读，其身份预先冻结（SHA-256，与 v4 协议同一组）：

```text
v3 report.json                    0a613bb4bdb695f53469d5b7bb132725eb4a3df66f9404be707f81280d14dd48
v3 nltcs/residual_diagnostics     d400be58270b713cd254e08f4dd9434adff0b7f2e07e1acbfee6850c9845b453
v3 nltcs/equal_diagnostics        336f0cf45a3dea486a79dea183ef495ba4b16678a7f1ee8523ce500911869c1d
v3 test/residual_diagnostics      c73ce211dd21545365065e645ae897ccfeb3f21a256af28e44959d6a80a2722e
v3 test/equal_diagnostics         ead9f446f72d786444f14c960a8ac52fd339ab0dc8df9622e8703c29e3e40956
```

## 4. 判据与结果标签（生成前冻结）

v3 参照数值（residual 臂，v3 冻结报告）：

```text
nltcs : best 7146.5   output 8317.5   drift 1.1639   measured L1 0.000179
test  : best 12.5     output 28.0     drift 2.2400   measured L1 0.002400
```

**主判据（drift）：**

1. nltcs residual 臂 `drift_ratio = output_squared_loss /
   best_loss_diagnostic_only` ≤ **1.100**（沿 v2/v3/v4 原目标，不放宽）。

**质量门（不倒退，两条都须满足，阈值与 v4 相同=1.10×v3 字面值）：**

2. nltcs residual 臂 measured normalized L1 ≤ **0.000197**；
3. test residual 臂 measured normalized L1 ≤ **0.002640**。

**形态判定（用于三分支归因，定义生成前冻结，窗口与 v4 对齐）：**

nltcs residual 臂记 `descending = (mean(loss_history[8000:9000]) <
mean(loss_history[7000:8000])) AND (running-best 最后刷新轮次 ≥ 8000)`。

**结果标签（primary_verdict，互斥）：**

```text
质量门任一失败                     → quality_regression_under_eta_cooling
质量门通过 且 判据1通过            → eta_cooling_supported
质量门通过 且 判据1失败 且 descending
                                   → eta_cooling_budget_insufficient
质量门通过 且 判据1失败 且 非 descending
                                   → eta_cooling_rejected
（另：§3 前缀审计失败              → schedule_blindness_violated，
  运行无效，优先于以上全部标签）
```

**观察项（无硬门）：**

- test residual 臂 drift_ratio（v3 为 2.2400）；
- 两数据集 held-out 3/4-way 配对差值相对 v3 的变化
  （v3 参照：nltcs -0.061454/-0.037263，test +0.003340/+0.000996）；
- nltcs eta 地板段 [2100,9000) 分段均值轨迹（500 轮/段）、best 出现
  轮次、running-best 刷新次数、稳态到达的描述性估计；
- 震荡带宽度的描述性对比（v3/v4/v5 地板稳态段的 loss 标准差与
  极差，衡量"幅度减半→带宽减半"的机制预测）；
- equal 臂在 eta 降温下的行为。

**审计（沿用 v4 全部项并适配）：**

- rho 时间表逐轮公式核对（9000 轮，v3 公式原样）；
- eta 时间表逐轮公式核对（9000 轮，含 0.5 保温/几何降/0.25 地板边界，
  与 diagnostics 新字段 `eta_schedule_history` 逐位比对）；
- 两臂初始表哈希、初始化后 RNG、最终 RNG、轮数、候选评价数配对一致；
- §3 保温段前缀一致审计（v3 参照产物哈希预核对在生成前完成）；
- 运行期间源码不变、固定轮数走满、terminal current 输出。

## 5. 实现范围（协议获批后、运行授权前完成）

1. **核心代码改动**（本屏与 v4 的关键差异——v4 零核心改动，v5 需要）：
   - `evolution.py`：`run_evolution` 新增 `eta_anneal_end` /
     `eta_anneal_rounds` / `eta_anneal_start_round` 三参数，验证逻辑、
     每轮 `eta_t` 计算、`eta_schedule_history` 诊断字段均与 rho 退火
     同构；`eta_t` 传入现行 update 调用位置，不改变随机数消费顺序；
     fitness-only fail-closed 合同放行 eta 退火（纯轮数驱动，同 rho
     退火待遇）；
   - `fitness_only.py`：`FitnessOnlyConfig` 新增对应三字段（默认
     None）+ validate + 传递。
2. **等价性合同（fail-closed）**：eta 退火三参数全 None 时，新代码与
   改动前在同 seed 小型冒烟配置下产出逐位相同轨迹（loss_history、
   最终 RNG 状态哈希）；v1–v4 既有专项测试全部保持通过（回归零容忍）。
3. 新建 delta 执行器 `scripts/run_fitness_only_eta_cooling_v5.py`：
   继承 v4/v3/v2/v1 执行器链，配置变更 = `eta_anneal_start_round=1500`
   + `eta_anneal_rounds=600` + `eta_anneal_end=0.25` + `n_rounds=9000`
   （rho 三参数用 v3 值）；实现 §3 前缀审计（1501 轮）与 §4 标签判定；
   协议清单含本文档全部冻结数值，SHA-256 钉死。
4. 专项测试（全部通过后才可申请运行授权）：plan 只读、身份
   fail-closed、不覆盖输出、config delta 仅 {eta_anneal_start_round,
   eta_anneal_rounds, eta_anneal_end, n_rounds}、双时间表边界（rho 三
   段 v3 原样；eta 0.5 保温/几何降/0.25 地板/t=1500 精确 0.5）、前缀
   审计逻辑（伪造不一致必须拒绝）、标签判定逻辑（四标签+违规标签
   各一例）、v3 参照哈希核对、eta 退火等价性（§5.2）。

## 6. 成本预算

生成：nltcs ≈ v4 实测 3561.4s（9000 轮两臂，eta 变化不改变每轮候选
评价量）≈ 59 分钟、test ≈ 230s；评价照旧；总 ≈ 65-70 分钟。GPU 用
nvidia-smi 选空闲卡（CUDA_VISIBLE_DEVICES 注入，协议内 device 字段
不变）。

## 7. 失败处理与后续动作（结果前绑定）

- `eta_cooling_supported` → 幅度假说成立且达标：方法定稿为
  "rho 三段式 H=900/D=600/floor=0.001 + eta 三段式 H=1500/D=600/
  floor=0.25"；下一步讨论 5-seed 确认屏协议与 fitness 聚合缺口优先级。
- `eta_cooling_budget_insufficient` → 小步修复慢、链仍在降但 9000 轮
  不够；**不得**连续加倍预算刷结果；评估现实性后另行讨论。
- `eta_cooling_rejected` → 幅度减半未收窄本底——频率（v4）与幅度
  （v5）双双排除，本底另有主因（如 mu 突变或 donor 池波动）；停止
  参数下探（不做 eta=0.1），回 B 路线（接受本底、修订叙事），v3/v4/
  v5 三次冻结实验构成完整机制排除证据链。
- `quality_regression_under_eta_cooling` → 小步损伤质量，停下诊断，
  不得进入任何后续屏。
- `schedule_blindness_violated` → 运行无效，修架构优先，产物保留；
  若为 eta 退火实现改变随机流所致，先修实现再议重跑（新协议）。
- 任何标签下：不得结果后修改 H/D/地板/判据阈值重跑冒充同一实验；
  产物一律保留。

## 8. 禁止事项

- 结果后调整时间表常数或判据阈值；
- 把单 seed 开发结果表述为正式效果结论或打开 promotion gate；
- 复用本屏结果直接进入 5-seed 确认而不另行冻结确认协议；
- 把本屏当作 v2/v3/v4 的"重考"——v2 unsupported、v3 rejected、
  v4 quality_regression 记录保留不变，v5 检验的是新假说（跳变幅度），
  四者在笔记与后续叙事中并列呈现；
- eta 退火机制进入核心代码后，任何非本协议路径不得默认启用（默认
  None=现行为，由等价性测试钉死）。
