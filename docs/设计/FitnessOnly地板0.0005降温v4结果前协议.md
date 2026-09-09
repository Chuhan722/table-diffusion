# FitnessOnly 地板 0.0005 降温 v4 结果前协议（草稿待审）

状态：**草稿，等待用户审查**。本协议起草时未修改任何代码、未运行任何新
实验。协议获批后，实现与正式运行仍需分别单独授权。

## 1. 目的与研究问题

v3（T=6000）按冻结判据判定 `truncation_hypothesis_rejected`：nltcs 链在
约 r4400 到达稳态后横盘（尾窗均值 8389 > 前窗 8115，best 7146.5@r4432），
drift 1.1639 未达 ≤1.100。结论：**drift ≈1.16 是地板温度 rho=0.001 下
nltcs 的波动本底**，与预算无关。

本底幅度由每轮扰动行数决定（nltcs 地板段 ≈16 行/轮）。本 v4 屏检验唯一
假说（**更低地板假说**）：

> 把地板从 0.001 降至 0.0005（nltcs ≈8 行/轮），稳态震荡带将收窄，
> 终态输出漂移可达原目标 drift ≤ 1.100，且不损失测量/held-out 质量。

设计选择（已与用户确认）：采用**三段式直接改终点**（方案 A），而非四段
式二次降温——若成功，最终方法即"三段式 H=900/D=600/floor=0.0005"，
叙事最简；代价是降温段斜率同步变化，轨迹自 t=901 起与 v2/v3 分叉。

本屏仍是单开发 seed 的 development 实验，不产生正式效果声明，
promotion gate 保持关闭（`formal_claim_allowed=false`）。

## 2. 变更定义与不变量

```text
时间表变更 ：rho_anneal_end 0.001 → 0.0005（唯一时间表常数变更；
             H=900、D=600 逐字不变）
预算       ：n_rounds 9000（运行长度非时间表参数；地板段修复速度
             减半，地板段给 7500 轮 ≈ v3 实测稳态用时 2900 轮的
             2 倍再加 ~30% 余量）
不变       ：seed 9908、两数据集、两臂配对、冻结输入与 SHA-256、
             rho0=0.01、eta=0.5、mu=0.01、fixed alpha=16、
             relative residual floor=8、geometric scale-invariant
             donor、冻结 marginal 初始化、tol=+inf、
             horizon_invariant=True、terminal current 输出、
             固定轮数无条件走满
时间表     ：t≤900 恒 0.01；900<t<1500 按 0.01·(0.05)^((t-900)/600)
             几何降至 0.0005；t≥1500 恒 0.0005
参与行期望 ：nltcs 162→8 行/轮；test 3→0.15 行/轮（地板段约 7 轮
             才动 1 行，量子化加剧为已知现象，见 §4 观察项）
输出目录   ：outputs/fitness_only_floor00005_dev_seed9908_v4/
             （独立，不覆盖 v1/v2/v3）
```

## 3. 保温段前缀一致审计（时间表盲性的可证伪检验）

时间表仅依赖轮次索引，且 v4 与 v3 的 rho 时间表在 **t∈[0,900]（含
t=900，共 901 轮）逐点相同**（保温段 0.01；t=900 两者 progress=0）；
自 t=901 起几何降温底数不同而分叉。因此 v4 各臂前 901 轮必须与 v3
对应臂**逐位一致**。生成完成后 fail-closed 核对：

- `loss_history[:901]` 与 v3 对应臂逐位相等；
- `rho_schedule_history[:901]` 与 v3 对应臂逐位相等；
- `initial_table_sha256`、`primary_rng_post_initialization_state_sha256`
  与 v3 对应臂相等（终态 RNG 不同属预期）。

任一不一致 → 运行判为无效（`schedule_blindness_violated`），说明主循环
存在对时间表未来值或总轮数的隐藏依赖，必须先修架构，不得解读任何质量
结果（产物保留供诊断，不打开 held-out）。

v3 参照产物只读，其身份预先冻结（SHA-256）：

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
   best_loss_diagnostic_only` ≤ **1.100**（沿 v2/v3 原目标，不放宽）。

**质量门（不倒退，两条都须满足）：**

2. nltcs residual 臂 measured normalized L1 ≤ **0.000197**（1.10×v3）；
3. test residual 臂 measured normalized L1 ≤ **0.002640**（1.10×v3）。

**形态判定（用于三分支归因，定义生成前冻结）：**

nltcs residual 臂记 `descending = (mean(loss_history[8000:9000]) <
mean(loss_history[7000:8000])) AND (running-best 最后刷新轮次 ≥ 8000)`。

**结果标签（primary_verdict，互斥）：**

```text
质量门任一失败                     → quality_regression_under_lower_floor
质量门通过 且 判据1通过            → lower_floor_supported
质量门通过 且 判据1失败 且 descending
                                   → lower_floor_budget_insufficient
质量门通过 且 判据1失败 且 非 descending
                                   → lower_floor_rejected
（另：§3 前缀审计失败              → schedule_blindness_violated，
  运行无效，优先于以上全部标签）
```

**观察项（无硬门）：**

- test residual 臂 drift_ratio（v3 为 2.2400；地板 0.15 行/轮量子化
  加剧为预期现象，不设门）；
- 两数据集 held-out 3/4-way 配对差值相对 v3 的变化
  （v3 参照：nltcs -0.061454/-0.037263，test +0.003340/+0.000996）；
- nltcs 地板段 [1500,9000) 分段均值轨迹（500 轮/段）、best 出现轮次、
  running-best 刷新次数、稳态到达的描述性估计；
- equal 臂在更低地板下的行为。

**审计（沿用 v3 全部项并适配）：**

- 时间表逐轮公式核对（9000 轮，两臂两数据集，含 0.0005 新地板边界）；
- 两臂初始表哈希、初始化后 RNG、最终 RNG、轮数、候选评价数配对一致；
- §3 保温段前缀一致审计（v3 参照产物哈希预核对在生成前完成）；
- 运行期间源码不变、固定轮数走满、terminal current 输出。

## 5. 实现范围（协议获批后、运行授权前完成）

1. 新建 delta 执行器 `scripts/run_fitness_only_floor00005_v4.py`：
   继承 v3/v2/v1 执行器链，配置变更 = `rho_anneal_end=0.0005` +
   `n_rounds=9000`；实现 §3 前缀审计（901 轮）与 §4 标签判定；协议
   清单含本文档全部冻结数值，SHA-256 钉死。
2. 不修改 `evolution.py` / `fitness_only.py` / v1 / v2 / v3 执行器
   （三段式机制已支持任意 end 值，无核心代码改动）。
3. 专项测试（全部通过后才可申请运行授权）：plan 只读、身份 fail-closed、
   不覆盖输出、config delta 仅 {rho_anneal_end, n_rounds}、时间表边界
   （0.01 保温/几何降/0.0005 地板/前 901 轮与 v3 公式逐位同）、前缀
   审计逻辑（伪造不一致必须拒绝）、标签判定逻辑（四标签+违规标签
   各一例）、v3 参照哈希核对。

## 6. 成本预算

生成：nltcs ≈ 9/6 × v3 实测 2366.5s ≈ 59 分钟（两臂合计）、
test ≈ 227s；评价照旧；总 ≈ 65-70 分钟。GPU 用 nvidia-smi 选空闲卡
（CUDA_VISIBLE_DEVICES 注入，协议内 device 字段不变）。

## 7. 失败处理与后续动作（结果前绑定）

- `lower_floor_supported` → 时间表定稿为 H=900/D=600/floor=0.0005；
  下一步讨论 5-seed 确认屏协议与 fitness 聚合缺口的优先级。
- `lower_floor_budget_insufficient` → 链仍在降但 9000 轮不够；**不得**
  连续加倍预算刷结果；评估行次/时间预算现实性后，接受慢收敛现实或
  回 B 路线，另行讨论。
- `lower_floor_rejected` → 0.0005 的本底仍宽于目标——地板下探收益
  递减实锤；停止继续下探（不做 0.00025），回 B 路线（接受本底、
  修订方法叙事，drift 如实报告为终态输出的已知代价）。
- `quality_regression_under_lower_floor` → 低温冻结损伤质量，重大反常，
  停下诊断，不得进入任何后续屏。
- `schedule_blindness_violated` → 运行无效，修架构优先，产物保留。
- 任何标签下：不得结果后修改 H/D/地板/判据阈值重跑冒充同一实验；
  产物一律保留。

## 8. 禁止事项

- 结果后调整时间表常数或判据阈值；
- 把单 seed 开发结果表述为正式效果结论或打开 promotion gate；
- 复用本屏结果直接进入 5-seed 确认而不另行冻结确认协议；
- 把本屏当作 v2/v3 的"重考"——v2 unsupported、v3 rejected 记录保留
  不变，v4 检验的是新假说（更低地板），三者在笔记与后续叙事中并列
  呈现。
