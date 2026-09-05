# FitnessOnly 三段式 rho 时间表 v2 开发屏结果前协议（草稿待审）

状态：**草稿，等待用户审查**。本协议起草时未修改任何代码、未运行任何新
实验。协议获批后，实现与正式运行仍需分别单独授权。

## 1. 目的与研究问题

v1 开发归因屏（`outputs/fitness_only_attribution_dev_seed9908_v1`）证明
了 fitness-only 架构的流程假设，但暴露两个问题：终点漂移（terminal
current 明显差于轨迹历史最好）与 test 的 held-out 退化。本 v2 开发屏只
检验一个干预：**预冻结的三段式 rho 时间表**能否——

1. 显著收窄终点漂移（主判据）；
2. 不损害 measured 质量（主判据）；
3. 顺带缓解 test 的 held-out 退化（观察项，不设硬门）。

本屏仍是单开发 seed 的 development 实验，不产生正式效果声明，
promotion gate 保持关闭（`formal_claim_allowed=false`）。

## 2. 机制定义：三段式 rho 时间表

记 `rho0 = 0.01`、地板比 `r_floor = 0.1`、保温轮数 `H`、降温轮数 `D`。
第 `t` 轮（t 从 0 起）的参与率：

```text
t <  H        ：rho_t = rho0                          （保温段）
H ≤ t < H+D   ：rho_t = rho0 * r_floor^((t-H)/D)      （几何降温段）
t ≥ H+D       ：rho_t = rho0 * r_floor = 0.001        （地板段，开口）
```

性质与边界：

- 时间表**只依赖轮次 t**，不读取残差、loss、候选评价或任何结果量；
  不是门控，不引入接受/拒绝，不破坏"残差只通过 fitness 进入"的归因。
- 公式中**不出现总轮数 T**。地板段没有自己的长度：本实验中被固定
  T=3000 截断（地板段 = 1500 轮）；未来换成早停终止时，时间表逐字
  不变，仅替换终止规则。
- 变异只作用于参与行（`update.py` 的
  `mutate_mask = participate & (rng < mu)`），因此 rho 是唯一全局
  温度旋钮；eta、mu 保持恒定不变。
- residual/equal 两臂共用同一时间表；两臂唯一处理差异仍是行适应度。

## 3. 常数取值与推导规则（以 v1 冻结产物为设计依据）

通用性原则：三个常数用归一化单位定义，**一次冻结、全数据集共用、禁止
按数据集调整**，地位等同于既有全局常数 `alpha=16、rho0=0.01、floor=8`。

推导规则（在看任何 v2 结果之前声明）：

1. **保温段 H**：v1 residual 臂 `loss_history` 的 running-best 达到总
   降幅 99% 的轮次为 test `r323`（每行约 3.2 次参与）、nltcs `r591`
   （每行约 5.9 次参与）。取最差值 5.9 次/行，乘安全系数 1.5，向上取
   整为 **9 次/行**，即 `H = 9 / rho0 = 900` 轮。
2. **降温段 D**：降一个数量级（`r_floor = 0.1`）的着陆坡，取
   **D = 600** 轮（降温段每行约 2.3 次参与）。工程选择，敏感度低。
3. **地板比 r_floor**：**0.1**（地板 rho = 0.001）。地板保留非零修复
   能力：若未来某数据集保温段不够用，拟合在降温/地板段以较慢速度继续
   完成，时间表不会锁死未拟合的链；数据集之间的收敛时长差异由未来的
   停止规则吸收，不由时间表吸收。

v1 参照数值（字段 `best_loss_diagnostic_only` / `output_squared_loss`，
residual 臂）：

```text
test_300x10 : best 15.0    terminal 50.0      drift_ratio 3.333
nltcs       : best 14870.0 terminal 17847.5   drift_ratio 1.2002
measured normalized L1 : test 0.003200，nltcs 0.000261
```

## 4. 实现范围（协议获批后、运行授权前完成）

1. `evolution.py`：为现有纯时间驱动退火增加**起始偏移**能力（如新参数
   `rho_anneal_start_round`，默认 None 完全向后兼容；语义：退火进度按
   `min(1, max(0, t - start) / D)` 计算）。不修改任何 legacy 默认行为。
2. `fitness_only.py` 合同修订：放行纯时间驱动时间表参数
   （`rho_anneal_end`、`rho_anneal_rounds`、`rho_anneal_start_round`），
   由入口按本协议第 2 节公式统一构造；**继续钉死**残差驱动机制：
   `residual_self_cooling=None`、`self_cooling_stop_ratio=None`，以及
   v1 合同全部既有禁止项（Gibbs、gap/C、方向核、重试、早停、历史最好
   选表、candidate budget 等）。
3. 专项测试（新增，全部须通过后才可申请运行授权）：
   - 三段边界逐轮正确：`rho_t` 在 t=0、H-1、H、H+D-1、H+D、T-1 与公式
     一致；`rho_schedule_history` 与公式全程逐位一致；
   - 时间表残差无关：换 target 后 equal 臂生成轨迹完全不变；
   - 两臂时间表一致且随机地址配对通过（沿用 v1 配对审计）；
   - legacy 路径（fitness_only_mode=None 或未启用时间表）逐位不变。

## 5. 实验矩阵与共同条件

除 rho 时间表外，一切与 v1 完全一致：

```text
数据集    ：test_300x10（workload B，50 查询）、nltcs（1001 查询）
臂        ：residual / equal（配对，共用时间表）
seed      ：9908（开发 seed，与 v1 相同）
轮数      ：固定 T=3000，无条件走满，返回 terminal current
共同条件  ：relative residual floor=8、eta=0.5、mu=0.01、fixed alpha=16、
            geometric scale-invariant donor（lambda=0.5、min_spread=1e-3）、
            冻结 1-way marginal 初始化、tol=+inf、horizon_invariant=True
输入      ：与 v1 相同的冻结输入文件与 SHA-256
输出目录  ：outputs/fitness_only_schedule_dev_seed9908_v2/（独立，不覆盖 v1）
```

不引入 B+one-way 覆盖变体：本屏唯一变更必须是时间表，覆盖/聚合属于
后续独立工作。

## 6. 评价与判据（生成前冻结）

离线评价与 v1 同一套：measured normalized L1、held-out 2/3/4-way、
1-way safety、TVD、支持集覆盖、多样性、查询频段分箱；只取 terminal
current。

**支持判据（两条都满足才算"时间表获得开发支持"）：**

1. 漂移收窄：两数据集 residual 臂
   `drift_ratio = output_squared_loss / best_loss_diagnostic_only`
   的超额部分至少减半，即
   `drift_ratio_v2 ≤ 1 + (drift_ratio_v1 - 1) / 2`：
   test ≤ 2.167，nltcs ≤ 1.100。
2. 质量不倒退：两数据集 residual 臂 measured normalized L1
   ≤ 1.10 × v1 参照（test ≤ 0.003520，nltcs ≤ 0.000287）。

**观察项（记录并解释，不设硬门）：**

- test held-out 3-way / 4-way 配对差值（residual − equal）相对 v1
  （+0.005632 / +0.001875）是否收窄；
- donor 结构距离轨迹、unique row rate、支持质量的变化；
- equal 臂在时间表下的行为（作为无信号对照的描述性记录）。

**审计（沿用 v1 全部项并新增）：**

- 两臂初始表哈希、初始化后 RNG、最终 RNG、轮数、候选评价数配对一致；
- `direction_evaluation_count=0`、全部 proposal 无条件接续、
  `output_table_identity=terminal_current`；
- 新增：`rho_schedule_history` 与第 2 节公式逐轮核对（两臂、两数据集）。

**失败处理：** 任一支持判据未达成即记录为
`schedule_dev_unsupported`，不得结果后修改 H/D/r_floor 重跑充当同一
实验；若要换常数，必须另立新版本协议并保留本次失败产物。

## 7. 禁止事项

- 结果后调整 H、D、r_floor 或按数据集使用不同常数；
- 任何残差/loss 驱动的降温、早停、历史最好选表、接受门；
- 把单 seed 开发结果表述为正式效果结论或据此打开 promotion gate；
- 复用本屏结果直接进入 5-seed 确认而不另行冻结确认协议。

## 8. 与后续阶段的关系

- 本屏支持 → 时间表常数并入 fitness-only 方法定义，再讨论 fitness
  聚合（覆盖/组保护）与 5-seed 确认的顺序；
- 本屏不支持 → 保留产物，回到调度设计（如两段式对照或常数修订），
  另立协议；
- 未来早停阶段：时间表原样沿用，终止规则单独设计与消融；停止规则读取
  （带噪）拟合 loss 属于其合法职责，与时间表互不混用。
