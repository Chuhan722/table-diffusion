# FitnessOnly 时间表 T=6000 截断假说 v3 结果前协议（草稿待审）

状态：**草稿，等待用户审查**。本协议起草时未修改任何代码、未运行任何新
实验。协议获批后，实现与正式运行仍需分别单独授权。

## 1. 目的与研究问题

v2 开发屏（`outputs/fitness_only_schedule_dev_seed9908_v2`）按冻结判据
判定 `schedule_dev_unsupported`：nltcs 的 drift_ratio 1.2063 未达 ≤1.100。
但结果后只读形态诊断显示 nltcs 在地板段**仍处于非稳态下降**（分段均值
11382→9978→9640，running-best 最后刷新在第 2582 轮），提示失败原因可能
是 **T=3000 预算截断**，而非时间表无效。

本 v3 屏只检验一个假说（**截断假说**）：

> nltcs 在 v2 的漂移未收窄，是因为固定预算在链仍在下降时强制截断；
> 给足预算后，同一时间表下漂移将自然收窄至原目标。

唯一变更：**总轮数 T=3000 → 6000**。时间表三常数（H=900、D=600、地板
0.001）与全部生成条件逐字不变。时间表公式不含 T，因此本延长**不需要
修改任何时间表定义**——这是 v2 协议第 2 节"公式中不出现总轮数"性质
的直接兑现。

本屏仍是单开发 seed 的 development 实验，不产生正式效果声明，
promotion gate 保持关闭（`formal_claim_allowed=false`）。

## 2. 变更定义与不变量

```text
变更     ：n_rounds 3000 → 6000（唯一变更）
不变     ：seed 9908、两数据集、两臂配对、冻结输入与 SHA-256、
           三段式时间表（H=900、D=600、地板 0.001）、
           rho0=0.01、eta=0.5、mu=0.01、fixed alpha=16、
           relative residual floor=8、geometric scale-invariant donor、
           冻结 marginal 初始化、tol=+inf、horizon_invariant=True、
           terminal current 输出、固定轮数无条件走满
时间表   ：t<900 恒 0.01；900≤t<1500 几何降至 0.001；t≥1500 恒 0.001
           （t≥1500 的地板段由 1500 轮延长为 4500 轮，公式逐字不变）
输出目录 ：outputs/fitness_only_schedule_T6000_dev_seed9908_v3/
           （独立，不覆盖 v1/v2）
```

## 3. 前 3000 轮前缀一致审计（视界不变的可证伪检验）

horizon_invariant 设计承诺：轨迹不读取总轮数 T。因此 v3（T=6000）的前
3000 轮必须与 v2（T=3000）**逐位一致**。生成完成后（相位边界之后）,
对每个数据集、每条臂 fail-closed 核对：

- `loss_history[:3000]` 与 v2 对应臂逐位相等；
- `rho_schedule_history[:3000]` 与 v2 对应臂逐位相等；
- `initial_table_sha256`、`primary_rng_post_initialization_state_sha256`
  与 v2 对应臂相等（最终 RNG 状态因多跑 3000 轮而不同，属预期）。

任一不一致 → 运行判为无效（`horizon_invariance_violated`），说明主循环
存在对 T 的隐藏依赖，必须先修架构，不得解读任何质量结果。

v2 参照产物只读，其身份预先冻结（SHA-256）：

```text
v2 report.json                    4f6d8e8ce6903955b619d95272af8e5ed31b5476767bf621afc852e616ac3476
v2 nltcs/residual_diagnostics     21a42c2e8b8fd4e39fa828bd58b7aa074884fddb21c766c8f6c5d224a46fb8b5
v2 nltcs/equal_diagnostics        e61f455a27b6fce94c515f51461408e3675849219ba40a53a15e231a346e4f41
v2 test/residual_diagnostics      6b709c1450790d0e1baea25ec6fe00d405e7f868e782b524a35bbef2442c191d
v2 test/equal_diagnostics         96cc6baaa8463ffa42070a47cb3cd0e9d8e403b1ff63258269c071f6eec69208
```

## 4. 判据与结果标签（生成前冻结）

v2 参照数值（residual 臂，v2 冻结报告）：

```text
nltcs : best 8409.0   output 10143.5   drift 1.2063   measured L1 0.000182
test  : best 19.5     output 35.0      drift 1.7949   measured L1 0.002533
```

**主判据（drift）：**

1. nltcs residual 臂 `drift_ratio = output_squared_loss /
   best_loss_diagnostic_only` ≤ **1.100**（沿用 v2 对 nltcs 的原目标，
   即相对 v1 超额减半；不因 v2 失败而放宽）。

**质量门（不倒退，两条都须满足）：**

2. nltcs residual 臂 measured normalized L1 ≤ **0.000200**（1.10×v2）；
3. test residual 臂 measured normalized L1 ≤ **0.002786**（1.10×v2）。

**形态判定（用于三分支归因，定义生成前冻结）：**

nltcs residual 臂记 `descending = (mean(loss_history[5000:6000]) <
mean(loss_history[4000:5000])) AND (running-best 最后刷新轮次 ≥ 5000)`。

**结果标签（primary_verdict，互斥）：**

```text
质量门任一失败                    → quality_regression_under_extended_budget
质量门通过 且 判据1通过           → truncation_hypothesis_supported
质量门通过 且 判据1失败 且 descending
                                  → budget_still_insufficient
质量门通过 且 判据1失败 且 非 descending
                                  → truncation_hypothesis_rejected
```

三分支的含义与后续动作（§7）绑定，避免结果后解读争议。

**观察项（无硬门）：**

- test residual 臂 drift_ratio（v2 为 1.7949；小表低 loss 区量子化波动
  已知，不设门）；
- 两数据集 held-out 3/4-way 配对差值相对 v2 的变化；
- nltcs 地板段 [1500,6000) 分段均值轨迹、best 出现轮次、
  running-best 刷新次数（用于描述收敛形态）；
- equal 臂在延长预算下的行为。

**审计（沿用 v2 全部项并新增）：**

- 时间表逐轮公式核对（6000 轮，两臂两数据集）；
- 两臂初始表哈希、初始化后 RNG、最终 RNG、轮数、候选评价数配对一致；
- §3 前缀一致审计；
- 运行期间源码不变、固定轮数走满、terminal current 输出。

## 5. 实现范围（协议获批后、运行授权前完成）

1. 新建 delta 执行器 `scripts/run_fitness_only_schedule_v3_t6000.py`：
   继承 v2 执行器全部内容（其自身继承 v1），唯一配置变更
   `n_rounds=6000`；新增 §3 前缀审计与 §4 标签判定；协议清单含本文档
   全部冻结数值，SHA-256 钉死。
2. 不修改 `evolution.py` / `fitness_only.py` / v1 / v2 执行器
   （若前缀审计失败暴露架构问题则另行处理，不在本协议内）。
3. 专项测试（全部通过后才可申请运行授权）：plan 只读、身份 fail-closed、
   不覆盖输出、config delta 仅 n_rounds、前缀审计逻辑（伪造不一致数据
   必须拒绝）、标签判定逻辑（四标签各一例）、v2 参照哈希核对。

## 6. 成本预算

生成：nltcs ≈ 2×20 分钟、test ≈ 2×75 秒；评价照旧；总 ≈ 45 分钟。
GPU 单卡（与 v2 相同 device 配置）。

## 7. 失败处理与后续动作（结果前绑定）

- `truncation_hypothesis_supported` → 截断假说成立；时间表+充足预算
  达成全部原目标。下一步：以本结果为靶子设计 DP 兼容早停（停止规则
  另立协议），并讨论 fitness 聚合缺口的优先级。
- `budget_still_insufficient` → 假说未被证伪但 6000 仍不够；**不得**
  连续加倍预算刷结果，转入早停设计（停止规则本来就是吸收数据集时长
  差异的正解）。
- `truncation_hypothesis_rejected` → nltcs 漂移为地板温度下的波动本底；
  回到 B/C 路线（接受本底修订方法叙事，或另立地板修订协议）。
- `quality_regression_under_extended_budget` → 长预算伤质量，重大反常，
  停下诊断，不得进入任何后续屏。
- 任何标签下：不得结果后修改 H/D/地板/判据阈值重跑冒充同一实验；
  产物一律保留。

## 8. 禁止事项

- 结果后调整时间表常数或判据阈值；
- 把单 seed 开发结果表述为正式效果结论或打开 promotion gate；
- 复用本屏结果直接进入 5-seed 确认而不另行冻结确认协议；
- 把本屏当作 v2 的"重考"——v2 的 unsupported 记录保留不变，v3 检验
  的是新假说（截断），两者在笔记与后续叙事中并列呈现。
