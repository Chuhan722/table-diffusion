# FitnessOnly 乘性权重聚合 v6 结果前协议

状态：结果前冻结（本文件写定并经用户批准后，判据与动作不得因结果修改）
日期：2026-09-04
前置：v3（T6000，`outputs/fitness_only_schedule_T6000_dev_seed9908_v3`，冻结）、
v4（floor0.0005，quality_regression，冻结）、v5（eta 降温，quality_regression，
冻结）。温度路线已由 v3/v4/v5 三重实验关闭；本协议开启聚合路线（fitness
内部的查询权重设计），与温度路线正交。

## 1. 假说（本实验要检验什么）

**病灶（v3 冻结产物查账证据，2026-09-04 只读分析）**：等权总分制 fitness
把稀有查询牺牲给总分——

- test_300x10：residual 臂终态欠账倍数（|ε|/mean|ε|）超 2 倍的查询 7 个，
  而 equal 对照臂只有 2 个（等权引导的取舍比无引导更极端）；
- nltcs：目标计数最小四分位查询的平均欠账 2.99 倍于全体平均；欠账超
  8 倍者 21 个，最重 29.92 倍；
- 既有病灶记录：R8 实验固定检查点证明"常见/稀有取舍早于停止出现，
  根因是权重目标对稀有查询关注不足"（工作笔记 L12072-12143）。

**假说**：给 fitness 的查询权重引入乘性自适应（MW，错题本机制——持续
欠账的查询权重复利上涨），可让 measured 稀有查询获得持续注意力 →
高阶结构更真实 → **test held-out 3/4-way 配对差改善乃至转负**（当前
+0.0033/+0.0010，residual 劣于 equal——本方法唯一输给对照的指标）。

**机制边界（为什么这不是门控）**：MW 评价的是**状态**（哪些查询仍欠账），
不是**动作**（某个候选变更好不好）；它只改变"残差→适应度"的聚合映射，
残差进入演化的唯一入口（fitness→softmax 选 donor）不变，核（rho/eta/mu、
无条件更新）一字不动。学术出身：MWEM（Hardt et al. 2012，DP 合成数据
奠基算法）与 Hedge/AdaBoost 的乘性权重更新，本方法将其嫁接到选择级。

## 2. 变更定义（唯一 delta）

基线：v3 配置逐字不变（rho 三段式 H=900/D=600/floor=0.001、eta 恒 0.5、
alpha=16、mu=0.01、relative 几何 floor=8、T=6000、seed 9908、marginal
初始化、winsorize (0.01,0.99)、scale-invariant 选择）。

新增 MW 查询权重（仅 residual 臂生效；equal 臂 fitness 恒 0，权重无作用）：

- 权重状态 `w ∈ R^m`，初始全 1；
- **每轮时序**：轮初用当前 w 计算 fitness（F = Σ_j w_j·ε_j·(a_j(z)−p_j)，
  经由 compute_fitness / evaluate_vectorized 已预留的 weights 口）→
  donor 选择与盲核更新照旧 → **轮末**若 t ≥ MW_START_ROUND 则更新 w：
  1. `rel_j = |ε_j| / mean(|ε|)`（当轮已算好的残差；mean(|ε|)==0 时
     跳过本轮更新）；
  2. `s_j = min(rel_j, MW_SIGNAL_CAP)`（信号封顶，防重尾单轮暴涨）；
  3. `w_j ← w_j · exp(MW_ETA · s_j)`（乘性复利）；
  4. `w ← w / mean(w)`（归一化：把"全体上涨"翻译为相对涨跌；达标查询
     自动回落，无显式惩罚规则）；
  5. `w ← clip(w, 1/MW_WEIGHT_CAP, MW_WEIGHT_CAP)`（活动围栏：防独裁、
     防归零失守。clip 后 mean(w) 允许偏离 1——选择尺度不变
     （fitness_only.py 强制 selection_scale_invariant=True）使整体缩放
     不影响任何行为，下一轮步骤 4 重新锚定，偏差不累积）。

**冻结常数**（依据：v3 终态查账 + 分层速度量纲推算，见 §1 与下）：

| 常数 | 值 | 依据 |
|---|---|---|
| MW_START_ROUND | 1500 | rho 落地时刻；欠账信号此时才分化；复用 v5 的 1501 轮前缀审计基建 |
| MW_ETA | 0.002 | 满格欠账户相对平均户的对数增速 ≈ (8−1)×0.002=0.014/轮 → 约 ln(8)/0.014 ≈ 150 轮完成分层（占地板段 4500 轮的 3%）；单轮最大乘数 exp(0.016)≈1.6%，量子化噪声不可能单轮掀动权重 |
| MW_SIGNAL_CAP | 8 | nltcs 欠账重尾（max 29.9）直接入指数会失控；封顶后欠 30 倍与欠 8 倍同速上涨 |
| MW_WEIGHT_CAP | 8 | 用户拍板（2026-09-04）：保守起步；21 个 nltcs 真欠账户全部顶到 8——彼此失去区分度但都获最高注意力；饱和情况由权重诊断记录，不足下轮另议 |

**t < 1500 时 w 恒等于全 1**：该段与 v3 的计算逐字相同（等价性合同）。

**n_rounds = 6000（与 v3 相同）**：本实验检验引导质量而非预算；同预算
对照最公平。

## 3. 审计与合同

1. **前缀一致性审计（硬门）**：residual 臂前 1501 轮（t∈[0,1500]）的
   loss_history、rho_schedule_history、current_state_metrics_history 与
   v3 residual 臂逐位一致；权重历史前 1500 轮恒全 1。失败 →
   `schedule_blindness_violated`，运行无效（先修架构，不得解读质量）。
2. **等价性合同（实现级，运行前测试锁死）**：MW 三参数全 None（关闭）
   时与现行为逐位一致（loss/表 SHA/primary_rng_state_sha256）；MW 更新
   不消耗主 RNG（纯确定性计算，不得改变随机流）。
3. **fail-closed 红线**：
   - w 的更新只读当轮 measured 残差 ε；禁止读 loss 变化方向、接受历史、
     held-out、任何结果级信号；
   - w 只进 compute_fitness / evaluate_vectorized 的 weights 口；禁止
     进入 rho/eta/mu/核概率/接受决策的任何路径；
   - 与 residual_directed_diffusion 组合 → ValueError（方向核下权重
     语义未冻结）；
   - equal 臂强制 w 恒 None（fitness 恒 0，传权重即为架构错误）。
4. **equal 臂复用 v3 冻结产物**：equal 臂代码路径不触 fitness 权重
   （evolution.py equal 分支直接返回零 fitness，不调用 compute_fitness），
   MW 对其无任何影响；v6 只运行 residual 臂，held-out 配对差用
   v3 equal 臂产物计算。v3 参照产物 5 文件 SHA 沿用 v4/v5 runner 的
   V3_REFERENCE_ARTIFACT_SHA256 冻结清单，运行前逐一校验。

## 4. 判据（结果前冻结）

记 v3 参照：test held-out 配对差 3way +0.0033398437500000017 /
4way +0.00099609375000000045；nltcs -0.061454333979667514 /
-0.037263032260058095；nltcs L1 0.00017922871887398185；test L1
0.0024000000000000002。

- **主判据（test held-out 配对差，residual−equal，负=residual 更好）**：
  - 3way 与 4way **均转负** → `mw_supported`；
  - 未双转负，但至少一项改善 ≥50% 且两项均不恶化 →
    `mw_partial_improvement`；
  - 其余 → `mw_rejected`。
- **护栏（质量门，任何违反直接判 `quality_regression_under_mw`，
  覆盖主判据）**：
  - nltcs held-out 配对差 3way ≤ −0.050 且 4way ≤ −0.030（v3 优势
    保留余量）；
  - nltcs measured L1 ≤ 0.000197、test measured L1 ≤ 0.002640
    （1.10×v3，与 v4/v5 同阈值）。
- **观察项（不进判定）**：两臂 drift、权重诊断（顶到 8/沉到 1/8 的
  查询数、权重分层稳定性、稀有四分位欠账倍数 vs v3 终态查账）。

## 5. 实现范围

- `src/table_diffevo/evolution.py`：`run_diffusion_evolution` 新增
  `mw_query_weight_eta / mw_signal_cap / mw_weight_cap / mw_start_round`
  四参数（全 None=关闭=现行为）；参数验证（全提供或全 None；正有限；
  cap>1；start_round 非负整数）；轮初传 w 进两条评价路径；轮末更新 w；
  诊断新增 `mw_weight_snapshot_history`（每 100 轮 + 首末轮的权重快照）
  与 `mw_weight_stats_history`（每轮 max/min/mean/顶格数/沉底数）；
  generation_config 输出四参数。
- `src/table_diffevo/fitness_only.py`：FitnessOnlyConfig 四字段 + validate
  同构块 + build_fitness_only_kwargs 传递。
- `tests/test_mw_weight_equivalence.py`：关闭=逐位一致；start_round 大于
  n_rounds 时=逐位一致；中途开启在开启前逐位一致、开启后分叉；更新公式
  逐轮匹配；非法组合 fail-closed。
- `scripts/run_fitness_only_mw_v6.py`：v6 delta 执行器，继承 v1-v5 链
  （复用 V3_REFERENCE 常数、前缀审计、评价管线）；只跑 residual 臂；
  报告含判定、审计、查账对比。
- `tests/test_fitness_only_mw_v6_runner.py`：runner 专项测试。
- 输出：`outputs/fitness_only_mw_dev_seed9908_v6/`（独立目录，不覆盖）。

## 6. 成本与运行

单臂 6000 轮：nltcs ≈ 20 分钟 + test ≈ 2 分钟 + 评价 ≈ 5 分钟，
总 ≈ 30 分钟。GPU 按空闲卡选择。运行前跑全量回归测试。

## 7. 失败处理与后续动作（结果前绑定）

- `mw_supported` → 聚合定稿"等权→MW"；方法主张更新为"残差经 MW 聚合
  进入选择级"；下一步起草 5-seed 确认屏协议（v3+MW 配置）。
- `mw_partial_improvement` → 不改常数不重跑；先只读诊断权重轨迹
  （分层是否稳定、是否饱和不足/过度），与用户讨论是否一次性剂量修订
  （新协议 v7，禁止扫参）。
- `mw_rejected` → 等权聚合定稿（v3 配置）；MW 记录为已排除方向；
  5-seed 确认屏用 v3 配置；test held-out 短板如实报告为 future work。
- `quality_regression_under_mw` → 停止；只读诊断权重轨迹找退化机理
  （独裁/震荡/稀有过冲）；回 v3 配置，不做剂量下探。
- `schedule_blindness_violated` → 运行无效，修架构后另立协议。

## 8. 禁止事项

- 禁止结果后调整 MW_ETA / SIGNAL_CAP / WEIGHT_CAP / START_ROUND 重跑
  （任何剂量修订=新协议）；
- 禁止扫参、禁止多 seed 择优；
- 禁止 held-out 进入权重更新或任何运行时决策（只做离线判据）；
- 禁止重跑或修改 v2/v3/v4/v5 冻结产物；equal 臂不重跑（§3.4 论证）；
- 单 seed 开发屏结果不作正式声明；
- v6 记录与 v2-v5 并列保留，无论判定如何不得删改。
