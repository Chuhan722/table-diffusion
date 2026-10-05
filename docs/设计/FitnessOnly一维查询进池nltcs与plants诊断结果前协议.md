# FitnessOnly 一维查询进池 nltcs+plants 诊断结果前协议

状态：**设计方向已获用户口头批准（2026-09-05 下午）**。本协议起草时未运行
任何新实验。正式运行需用户单独授权（预定当晚，等待 4090 空闲）。

## 1. 目的与研究问题

PGM 基线校准（`outputs/baseline_pgm_nltcs_v1`、`outputs/baseline_pgm_plants_v1`）
给出两个事实：

- nltcs（密覆盖，2-way 100%）：我方 residual 臂全面优于 PGM；
- plants（稀疏覆盖，2-way 4.9%）：PGM 全面优于我方两臂，其中
  **one_way_safety 差约 70×**（PGM 0.000589 vs 我方 R 0.040966）——
  一阶边缘只在初始化用过一次，fitness 看不见它，演化把它牺牲了。

本诊断只检验一个假说（**one-way 进池假说**）：

> 把一阶边缘的全部格子以普通查询身份加入演化查询池（fitness 与残差
> 引导都可见），残差机制会自动守住一阶边缘（守住时 ε≈0 自动沉默、
> 被破坏时自动回修），从而在同预算下改善 plants 的 one-way 与 heldout
> 泛化，且不损害 nltcs 已有优势。

机制论证（预注册，防事后合理化）：残差引导按 |ε| 分配注意力。one-way
守住时其残差是围绕零的随机噪声，138 项合力只有 √138 量级的波动，无法
掩盖任何系统性大残差；one-way 系统性劣化时合力报警是**期望行为**
（一阶是分布地基）。外层接入后新查询由选择阶段定义保证是全场最大残差，
不会被守住的旧池淹没。本诊断不实现任何权重、门控或判据（现役 fitness-only
合同无接受门，tol=+inf）。

两个数据集回答两个不同问题：

- **plants（主战场）**：同预算下 one-way 与 heldout 能追回多少？与 PGM
  的剩余差距即"泛化机制缺陷"的份额；
- **nltcs（回归保险）**：在已经全胜 PGM 的数据集上，进池是否引入劣化。

本诊断为单开发 seed 的 development 实验，`diagnostic_only=true`、
`formal_claim_allowed=false`，无晋级门。

## 2. 变更定义与不变量

```text
变更     ：演化查询池 = measured ∪ one-way 全格（唯一变更）
           nltcs  1001 + 32  = 1033（16 属性 × 2 值）
           plants  980 + 138 = 1118（69 属性 × 2 值）
           one-way 目标 = init_marginals 冻结计数（无噪声阶段真值）
不变     ：v3 冻结配置逐字不动——seed 9908、T=6000、三段式时间表
           （H=900、D=600、地板 0.001）、rho0=0.01、eta=0.5、mu=0.01、
           fixed alpha=16、relative residual floor=8、
           geometric scale-invariant donor、冻结 marginal 初始化、
           tol=+inf、无接受门/重试/冷却/早停、terminal current 输出、
           residual/equal 两臂严格配对（同初始表、同 rng 地址）
机制     ：零改动、零新超参——one-way 格走与 measured 完全相同的
           fitness/残差通路
评价口径 ：完全不变（见第 4 节），保证与全部冻结报告可比
输出     ：outputs/fitness_only_oneway_pool_nltcs_seed9908_v1/
           outputs/fitness_only_oneway_pool_plants_seed9908_v1/
           （独立目录，fail-closed 不覆盖）
```

## 3. 演化查询池构造（确定性）

按 `init_marginals.json` 的 `attributes` 原始顺序逐属性、逐 value 展开：

- 每格一条查询：`id=OW####`（4 位序号）、`type="single"`、
  `conditions=[{attribute, operator:"==", value}]`、`result=count`；
- 断言：one-way 格数 nltcs=32 / plants=138；每属性 counts 总和 = N；
- 断言：measured 查询全部 ≥2 条件、one-way 全部 1 条件，两组
  `query_fingerprint` 集合无交、并集无重复；
- 池 targets 向量与池指纹 SHA-256 写入报告（`pool_identity_sha256`、
  `pool_target_vector_sha256`）。

## 4. 评价口径不变声明（可比性关键）

评估只用三个**互斥**查询组，与冻结报告逐字同口径：

- `measured`：仅原 measured 查询（nltcs 1001 / plants 980）——one-way
  格**不计入** measured 指标；
- `heldout`：冻结 heldout 文件（512×3way + 512×4way），不参与演化；
- `one_way_safety`：与冻结报告同一构造（marginals 全格）。本诊断中该组
  **首次同时也在演化池内**，报告以
  `one_way_now_in_evolution_pool=true` 显式披露。

`validate_query_partition` 三组两两互斥照旧执行。评价函数复用
`run_fitness_only_attribution` 的 `_grouped_error_metrics` 与
`evaluate_quality_snapshot`，逐字不改。

## 5. 对照与冻结引用（不重跑，SHA 钉死）

```text
nltcs 输入 ：schema 5765de90… marginals a5e63ea8… measured b34eb2d5…
             heldout a025b5b4… reference 7d185b8a…
nltcs 对照 ：v3 报告 0a613bb4bdb695f53469d5b7bb132725eb4a3df66f9404be707f81280d14dd48
             （布局 quality.nltcs.{residual,equal}）
             PGM 报告 bd349bc3dcaf4751cd34d3fdeb2bcd485858be0594bd0e5c149fba9d9e59b3b1
plants 输入：schema 7eaf404b… marginals 4e302b18… measured f93c2d97…
             heldout d6540176… reference c7b5cf1e…
plants 对照：冻结诊断报告 26a8934270e9042d05e90d0a77ba1859aebd34a48ca1082291dbee9330d95565
             （布局 quality.{residual,equal}）
             PGM 报告 fbe0c88d793fde6ed3e6b919933b7bedf2d239cb6ea606f3865eadad9f026105
```

对照报告运行时按 SHA 校验后只读提取，聚合入本报告 `comparison` 节。

## 6. 预注册观察点（结果读法，无晋级门）

plants 主问题（对照冻结 R 0.040966/0.039262/0.027691/0.005249）：

- **P1 one-way 自愈**：new-R 的 one_way_safety 相对 0.040966 的改善
  幅度——机制有效性的直接证据；与 PGM 0.000589 的剩余差距；
- **P2 heldout 追赶**：heldout_3way/4way 相对 0.039262/0.027691 的
  变化；与 PGM 0.015229/0.012866 的剩余差距 = 泛化机制缺陷的份额；
- **P3 淹没实证**：measured 相对 0.005249 的变化。同预算下池
  +14%，每题算力摊薄，轻微劣化（<~15% 相对）按摊薄解释；大幅劣化
  （>~50%）视为 one-way 挤占信号，回到加权设计讨论；
- **P4 盲臂对照**：R vs E 的 heldout 均值劣势（+0.0061/+0.0052）是否
  收窄或翻转。

nltcs 回归问题（对照 v3 R 0.000179/0.000419/0.000769/0.000112）：

- **N1**：measured/heldout/one-way 各指标的方向与相对幅度，重点看
  是否出现实质劣化（预期：不劣化或轻微摊薄）。

配对结构观察点（两数据集通用）：

- **E1 equal 臂身份**：equal 臂 fitness 恒 0，演化轨迹理论上与查询池
  无关。预注册预期：new-E 的 terminal_table_sha256 与冻结 old-E
  **逐字节一致**。报告记录 `equal_table_identity_match`；不一致不
  fail-closed（记录后照常落盘），但按"查询池向盲臂泄漏"线索另行调查。

## 7. fail-closed 与运行纪律

- 运行前：协议 SHA `--confirm-protocol-sha256` 逐字确认；全部输入与
  对照报告 SHA 校验；池构造断言（第 3 节）全过才开跑；
- 运行中：rho 时间表逐轮审计；配对审计（shared_initial_table_sha256、
  shared_primary_rng_endpoint_sha256）；
- 落盘：staging 目录原子 rename；已存在目标目录立即失败不覆盖；
- 解释纪律：单 seed、diagnostic_only，所有数字仅作方向性诊断，
  不作正式效果声明，不触发任何晋级。

## 8. 运行预算与顺序

- plants 双臂：单臂约 70 分钟 + 池 14% 开销 ≈ 2.7 小时（GPU）；
- nltcs 双臂：约 40–60 分钟（GPU）；
- 顺序：可并行（两张 4090 各一），或先 nltcs 后 plants；
- PGM/我方冻结臂一律不重跑。
