# Issue #53 Stage 4 能量恒等门禁混合容差修订协议

> 状态：结果后协议修订 **v1，已于 2026-08-20 经用户确认冻结**。
> 本修订在查看 development seeds `323..327` 的 sweep=8 mixing 结果之后提出，因此**不冒充预注册**；
> 它只修改 `exact_factor_energy` 结构门禁的数值容差公式与相应报告/审计字段，不修改任何科学参数、
> 混合性能门槛、状态选择或停止规则。qualification seeds `333..337` 从未运行、从未被读取。

## 一、修订模板字段

- **修改前**：`exact_factor_energy` 门禁为纯绝对容差
  `max_abs_error <= ENERGY_TOLERANCE = 1e-10`，其中 `max_abs_error` 是 probe 内全部
  逐 mask 能量元素 `|E_factor - E_oracle|` 的最大值（`scripts/issue53_stage4_protocol.py:42`、
  `scripts/run_issue53_stage4_mixing.py`、`scripts/audit_issue53_stage4_mixing.py`）。
- **修改后**：逐元素混合容差
  `|E_factor - E_oracle| <= ENERGY_ATOL + ENERGY_RTOL * max(|E_factor|, |E_oracle|)`，
  冻结 `ENERGY_ATOL = 1e-10`、`ENERGY_RTOL = 1e-12`；门禁量为逐元素
  `ratio = |E_factor - E_oracle| / (ENERGY_ATOL + ENERGY_RTOL * max(|E_factor|, |E_oracle|))`
  的最大值，要求 `ratio_max <= 1.0`。
- **原因**：纯绝对容差不具尺度稳健性；development 显示 nltcs initial 状态能量约 `1e7` 量级，
  double 精度在该量级的 1 ulp 约 `1.86e-9`，`1e-10` 在该尺度物理上不可满足，门禁实际测的是
  浮点求和顺序而不是实现正确性（详见第二节）。
- **影响的 Stage / 既有证据**：仅 Stage 4。协议语义变化产生新 protocol SHA，development
  `323..327` 的状态库与 mixing 必须在新 clean commit 上整体重跑；2026-08-19 的
  `invalid_or_incomplete` 报告与审计**永久归档、不改写、不删除**。不影响 PR #63/#65/#66、
  residual geometry、α 板块与早停板块的任何既有证据。
- **是否已查看相关正式结果**：是（development `323..327`，本文明确引用其数值）；
  否（qualification `333..337`，从未运行）。

## 二、为什么绝对容差是协议设计缺陷

development sweep=8 的逐状态证据（`outputs/issue53_stage4_development_v1/mixing_report.json`，
report SHA `7268800e37238a733c483ca755572344f35b84d5226adc015ce5139082a76d9c`）：

| 状态组 | state squared loss 量级 | `exact_energy_max_error` 范围 | 对 1e-10 判定 |
|---|---:|---:|---|
| nltcs initial ×5 | ~6.9e8 | 3.68e-9 ~ 4.75e-9 | 全部失败 |
| nltcs 非 initial ×20 | 1.6e4 ~ 1.2e6 | 3.3e-12 ~ 4.4e-11 | 全部通过 |
| test 全部 ×25 | 17 ~ 2373 | ~1e-19 ~ 3.5e-18 | 全部通过 |

- 误差与能量量级严格同步：能量越大误差越大，跨 9 个数量级保持一致比例（相对误差恒为
  `~1e-16` 量级）。
- double 尾数精度 `eps = 2^-52 ≈ 2.22e-16`；量级 `1e7` 的数其 1 ulp
  `= 2^23 × eps ≈ 1.86e-9`。观察到的 3.68e-9~4.75e-9 相当于 **2~3 ulp**，是稀疏因子逐因子
  求和与 oracle 全表路径**求和顺序不同**导致的正常浮点舍入。
- 同一实现（同一 commit `6d102af`）在小能量状态误差低至 1e-19，且 production tape replay
  25448 次比对零失配、概率归一误差 ~5.5e-16——排除实现错误。
- 结论：`1e-10` 纯绝对容差隐含"能量为 O(1)"假设，对 initial 状态（1-way 初始化后高阶残差
  天然最大）不可满足。这是量尺缺陷，需按协议修订处理，不能删除门禁，也不能事后放宽到刚好
  覆盖观察值。

## 三、冻结的新判定规则

### 3.1 公式

对 probe 内每一个被比较的能量元素（逐行、逐 proposal、逐 mask）：

```text
scale     = max(|E_factor|, |E_oracle|)
allowed   = ENERGY_ATOL + ENERGY_RTOL * scale
ratio     = |E_factor - E_oracle| / allowed
门禁      = 数据集内全部元素 ratio 的最大值 <= 1.0
```

```text
ENERGY_ATOL = 1e-10
ENERGY_RTOL = 1e-12
```

判定层级不变：probe 逐元素取最不利值 → 状态级 → 数据集级；任一数据集失败即该 sweep 尝试
`valid=false`，invalid-stop 规则原样保留（首个结构性 invalid 立即停止，更高 sweeps 不得恢复资格）。

### 3.2 语义性质

- **近零保护与旧规则等严**：`scale → 0` 时 `allowed → 1e-10`，与修改前完全一致；
- **大尺度按有效数字判定**：`scale = 1e7` 时 `allowed ≈ 1e-5`，要求两条路径约 12 位有效数字
  一致；
- 公式对 `E_factor`、`E_oracle` 对称，不偏向任一路径。

### 3.3 常量依据（第一性原理，非观察值反推）

- `ENERGY_RTOL = 1e-12 ≈ eps(2.22e-16) × 求和深度上界 O(10^2) × 安全系数 O(10^2)`。
  推导只使用浮点精度与因子求和规模（development 观测 mean_factor_count ≈ 5.8/行、单因子表
  最多几十项），不使用任何观察误差值。
- 判别力校验：浮点噪声相对误差 ~1e-16（比门槛低约 4 个数量级）；真实实现错误（漏因子、错表项）
  产生的相对误差 ≥ 1e-6（比门槛高约 6 个数量级）。门槛落在两者之间的空档中央，两侧各留
  千倍以上余量。
- **明确禁止**：不得改成纯绝对 `1e-8`；不得以 `323..327` 观察到的最大误差 `4.746e-9`
  或其任何函数设置常量；本协议冻结后，若未来任何 seed 触发本门禁，只能按失败记录，
  不得再次修改容差后重跑同批 seeds。

## 四、报告与审计字段变更

### 4.1 probe（`scripts/probe_factorized_gibbs_mixing.py`）

`factor_diagnostics` 保留旧字段并新增：

| 字段 | 语义 |
|---|---|
| `exact_energy_max_error` | 保留：逐元素绝对误差最大值（继续报告） |
| `exact_energy_max_relative_error` | 新增：`scale > 0` 元素上 `abs_diff / scale` 的最大值（报告用） |
| `exact_energy_tolerance_ratio_max` | 新增：门禁量，逐元素 `ratio` 最大值 |
| `exact_energy_worst_case` | 新增：`ratio` 最大元素处的 `{abs_diff, scale}`，供 auditor 独立重算 |
| `energy_atol` / `energy_rtol` | 新增（实施时补充，2026-08-20）：probe 实际使用的常量自描述；auditor 断言与冻结协议常量逐位相等 |

### 4.2 runner（`scripts/run_issue53_stage4_mixing.py`）

- 数据集级门禁改为 `max(exact_energy_tolerance_ratio_max) <= 1.0`；
- `numerical_diagnostics` 同时输出绝对最大误差、相对最大误差与 ratio 最大值三个数。

### 4.3 独立 auditor（`scripts/audit_issue53_stage4_mixing.py`）

- 继续不导入 builder/runner；
- 从每个状态记录的 `exact_energy_worst_case.{abs_diff, scale}` 用协议常量**独立重算** `ratio`，
  与 probe 记录的 `exact_energy_tolerance_ratio_max` 比对一致后再聚合重算门禁；
- 聚合结论必须与 runner 报告一致，否则审计失败。

### 4.4 协议常量（`scripts/issue53_stage4_protocol.py`）

- 移除 `ENERGY_TOLERANCE`，新增 `ENERGY_ATOL`、`ENERGY_RTOL`；
- protocol dict 以显式结构记录：

```text
"energy_identity_gate": {
    "rule": "mixed_absolute_relative",
    "formula": "abs(E_factor - E_oracle) <= atol + rtol * max(abs(E_factor), abs(E_oracle))",
    "atol": 1e-10,
    "rtol": 1e-12
}
```

- protocol SHA 必然改变；state library / 分片 / mixing / audit 的 fail-closed SHA 绑定链原样
  生效，旧协议状态库不可静默复用。

## 五、新增人工测试（实现时一并交付）

1. **跨尺度恒等通过**：构造能量约 `1e-3`、`1`、`1e8` 三个量级的因子模型，恒等校验在新公式下
   全部通过；
2. **注错必抓（fail-closed）**：
   - 大尺度（~1e7）注入相对 `1e-6` 扰动 → 门禁必须失败；
   - 近零尺度注入绝对 `2e-10` 扰动 → 门禁必须失败（验证 atol 保护未变松）；
3. **runner/auditor 公式一致**：对同一份人工报告，两者给出相同判定；构造 `worst_case` 分量与
   记录 ratio 不一致的篡改样例，auditor 必须报错；
4. **协议 SHA 换代**：新常量进入 protocol dict 后，旧 development 状态库的 SHA 绑定必须失败，
   防止静默复用。

## 六、生效流程与重跑计划

1. 用户确认本协议 → 实施代码修改 → 定向测试与相关回归通过 → 本地 clean commit
   （产生新 protocol SHA）；
2. 在新 commit 上重新采集 development `323..327` 状态库（沿用一种子一分片机制；预计 test 约
   3 分钟、nltcs 约 56 分钟）；
3. 重跑共享 `8→16→32` runner 与独立 audit；
4. 旧 `outputs/issue53_stage4_development_v1/` 整目录归档保留（`invalid_or_incomplete`），
   不删除、不改写；新产物写入独立目录 `outputs/issue53_stage4_development_v2/`；
5. development 性能与结构门禁全部通过、审计一致后，**另行单独请求** qualification
   `333..337` 授权；本协议不包含该授权，也不授权任何 push、PR 或 Issue 操作。

## 七、本修订不改变的事项（不变量清单）

- sweeps 候选 `8/16/32`、共享停止规则、invalid-stop 规则；
- TVD 门槛 0.05、gap recovery 门槛 0.80、零 conditional clip 命中要求、clip=30；
- `tau=2`、`fixed alpha=16`、`rho=0.01`、`eta=0.5`、`mu`、`s0`、residual geometry
  `relative/floor=8`、scale-invariant donor、P=6、C=6000、no-gate、terminal-current、
  marginal 初始化；
- development/qualification seeds 与两级分离纪律；状态选择规则（initial/q25/q50/q75/terminal）；
- 其他全部数值容差：`probability_sum_tolerance = 1e-12`、`tvd_monotonic_tolerance = 1e-12`
  维持不变（development 观测远低于阈值且无尺度问题）；
- `one_hot_direction_max_error` 维持 report-only 诊断，不新增门禁；
- 公共 API 与主生成路径代码零改动。
