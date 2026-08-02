# 三阶因子随机扫描 Gibbs 扩散

## 1. 研究问题

精确联合属性块扩散核已经证明：在相同冻结状态、donor、参与记录和随机耦合下，
完整 hybrid 方向相对独立单块相加能给出更好的原始 proposal。但是直接枚举一条
recipient 与 donor 之间的全部 `2^k` 个复制 mask，只能作为小表 oracle，不能作为
宽表生成器。

本阶段只回答一个更小的问题：当 workload 中每个合取查询至多涉及三个不同属性时，
能否把精确联合方向写成稀疏低阶因子，并用有限步随机扫描 Gibbs 在不做方向筛选、
top-k 或整代接受的前提下逼近精确联合核。

本阶段仍使用固定 workload 的精确统计量，是无噪声方法原型，不构成 DP 实现。

## 2. 稀疏因子表示

固定当前记录 `x`、donor `y` 和残差 `r`。只给 `x` 与 `y` 取值不同的属性编号，
复制 mask 记为 `M`。第 `j` 个查询只依赖它涉及的活跃属性集合 `S_j`，因此中心化
方向能量可以写为

\[
U(M)=\sum_j \phi_j(M_{S_j}),\qquad U(0)=0.
\]

其中 `phi_j` 是至多 `2^|S_j|` 项的局部真值表。当前 `test_300x10` workload 的查询
至多涉及三个不同属性，所以每个因子最多只有 8 项。相同 scope 的查询可以相加为
一个因子；这不会采用加性近似，而是精确保留二阶、三阶交互。

在因子已经构造后：

- 存储量为所有非零局部因子表大小之和，不随 `2^k` 增长；
- 一次条件更新只访问包含被更新属性的因子；
- 精确枚举只保留在测试和小表混合诊断中，不进入候选生成路径。

## 3. 联合目标与随机扫描更新

以历史独立复制核为参考分布

\[
q_0(M)=\prod_g \eta^{M_g}(1-\eta)^{1-M_g},
\]

联合扩散目标为

\[
q_\beta(M)\propto q_0(M)\exp\{\beta U(M)\}.
\]

随机扫描 Gibbs 的一个微步先在全部活跃属性中均匀、带放回地选择 `g`，再按完整
条件分布重采样该 bit：

\[
\Pr(M_g=1\mid M_{-g})=
\sigma\!\left(\operatorname{logit}(\eta)+
\beta[U(1,M_{-g})-U(0,M_{-g})]\right).
\]

一个 sweep 定义为 `k` 个随机坐标微步。候选核先按现有独立定向核抽取初始 mask，
然后执行给定 sweep 数。于是：

- 0 sweep 精确等于现有独立核，是明确 baseline；
- 有限 `beta` 下每个 bit 的条件概率严格位于 `(0, 1)`，保留完整支持集；
- 不截断 logit 时，随机扫描链以精确联合 `q_beta` 为平稳分布并满足细致平衡；
- 该更新不要求每一步方向为正，也不保证单次 proposal 降低 loss；方向性来自连续
  Gibbs 倾斜，而不是接受筛选；
- sweep 数控制计算量和对联合核的逼近程度，不应在生成时用真实数据或 loss 自适应
  选择。

## 4. 阶段 A：冻结状态混合实验

### 4.1 假设

至多三阶的稀疏因子能够数值精确复现完整 hybrid 能量；从独立核出发的少量随机扫描
Gibbs sweep 能显著缩小到精确联合核的分布距离，并恢复大部分精确联合核的期望方向
和原始 proposal 收益。

### 4.2 固定实验协议

- 数据配置：`test_300x10`，只使用 50 个预定义查询的精确 target；
- 状态：初始化状态与历史算法运行 100 轮后的状态；
- 种子：`0, 1, 2`，状态内使用配对 donor、参与记录和 Gumbel 随机量；
- 每个种子和状态 200 个冻结 proposal；
- `rho=0.01`、`eta=0.5`、`mu=0`；
- 温度：`tau=1, 2`，沿用冻结状态第一次非零单块方向 RMS 归一化；
- 唯一变量：Gibbs sweep 数 `0, 1, 2, 4, 8`；
- 环境：`qdte` Conda 环境、NumPy/CPU 路径；正式输出记录 Python、NumPy、平台、
  commit 和完整参数；
- oracle：同一强度下完整 hybrid 的精确联合 Gibbs 分布；
- 不执行变异、整代接受、重试、best checkpoint 或真实 train/test 评价；
- 设备、commit、完整命令和原始输出随正式实验记录。

这里的同温度比较回答“有限步链能否实现所定义的联合扩散核”。它不等价于此前控制
KL 后的机制归因；两类结论必须分开陈述。

### 4.3 指标

对每条参与的 recipient-donor 对报告：

- 稀疏能量与完整 hybrid 枚举的最大绝对误差；
- 到联合 oracle 的 TVD 与 `KL(candidate || oracle)`；
- 到参考核的 KL、mask 熵、精确方向期望以及负方向概率质量；
- 因子数、最高活跃阶数和每个 sweep 的运行成本。

对每个完整 proposal 另外报告 workload loss 的原始 gain、线性 gain、二次惩罚、
改变单元格数，以及候选相对 0 sweep 和 oracle 的配对差值。训练目标只用于 proposal
结束后的离线评价，不参与 Gibbs 更新或 sweep 选择。

### 4.4 预先约定的判断规则

先做单种子小样本冒烟，只有语义与数值校验通过才运行冻结协议。正式结果按以下顺序
判断：

1. 全部稀疏能量误差不超过 `1e-10`，否则实现不成立；
2. TVD 应随微步不增；若出现超过 `1e-12` 的反向变化，先视为实现或统计错误；
3. 8 sweep 后，`tau=1,2` 的全局平均 TVD 均不高于 `0.05`，且精确方向期望与
   oracle 的差距相对 0 sweep 至少缩小 80%；
4. 原始 proposal gain 用于确认分布逼近是否转化为目标改善，但不以单个种子或单一
   p 值宣称通用最优。

只有前三项通过，才把 4 或 8 sweep 中成本更低且接近 oracle 的配置带入阶段 B。

### 4.5 已验证结果

正式实验在 commit `a1681e7`、Conda `qdte`、NumPy/CPU 上按上述协议运行。6 个
seed-state 共 1200 个冻结 proposal、3568 条活跃参与记录。稀疏因子与完整 hybrid
方向的最大绝对误差为 `1.11e-16`，one-hot 方向与现有单块方向的最大误差为
`3.12e-17`；最高活跃因子阶数为 3。记录级 TVD 快照最大反向变化仅
`3.61e-16`，属于 float64 舍入误差。

按活跃参与记录数加权的混合结果为：

| tau | sweep | 到 oracle TVD | `KL(candidate||oracle)` | 方向差距恢复 | proposal 平均 gain |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0.26031 | 0.33343 | 0% | 28.37 |
| 1 | 1 | 0.10672 | 0.06194 | 59.42% | 36.04 |
| 1 | 2 | 0.05561 | 0.02077 | 78.88% | 38.13 |
| 1 | 4 | 0.01876 | 0.00355 | 93.16% | 39.35 |
| 1 | 8 | **0.00315** | **0.00023** | **98.97%** | **40.13** |
| 1 | oracle | 0 | 0 | 100% | 40.20 |
| 2 | 0 | 0.39194 | 1.02682 | 0% | 37.86 |
| 2 | 1 | 0.19443 | 0.29205 | 54.49% | 48.46 |
| 2 | 2 | 0.12078 | 0.14489 | 73.32% | 51.78 |
| 2 | 4 | 0.06039 | 0.05632 | 88.09% | 54.55 |
| 2 | 8 | **0.02479** | **0.01868** | **95.66%** | **55.79** |
| 2 | oracle | 0 | 0 | 100% | 56.50 |

8 sweep 相对 0 sweep 把 `tau=1/2` 的平均 proposal gain 分别提高 41.47% 和
47.37%，并恢复 oracle 相对独立核 gain 缺口的 99.43% 和 96.20%。六个
“种子 × 状态”组合的平均差值全部为正；`tau=1/2` 的逐 proposal 配对分别为
`575/523/102` 和 `769/366/65`（胜/平/负）。这些比较是同温度结果，不是匹配 KL
结果；更强联合倾斜和交互结构都包含在所定义的联合核内。

8 sweep 是唯一同时让两个温度的加权 TVD 低于 0.05、方向差距恢复超过 80% 的预设
候选。因此阶段 B 固定 `tau=2, 8 sweep`；`tau=2` 在阶段 A 的原始 proposal gain
更高，比较基线也固定为同温度的 0 sweep，不把温度差混入因子 Gibbs 归因。

正式原始输出：
`outputs/factorized_gibbs/frozen_3seed_2state_200p_tau1_2_sweep0_8_a1681e7.json`。
输出不提交 Git。

## 5. 阶段 B：关闭整代接受的动力学

阶段 B 单独比较 0 sweep 独立核与阶段 A 选出的固定 Gibbs sweep 数。每个原始
proposal 都无条件成为下一状态，最终当前表而非历史 best 表是主终点。至少使用 10 个
配对种子、固定轮数，同时记录最终 workload loss、逐轮正负 gain、支持集、墙钟和
因子/Gibbs 成本。

阶段 B 仍保留 donor 的适应度与距离抽样。因此它检验的是“去掉整代接受后，联合
扩散算子能否形成稳定下降动力学”，不能解释为完全没有其他定向信息的自由扩散。

### 5.1 正式协议

- 假设：同温度下，8 sweep 因子 Gibbs 的最终当前 workload loss 低于 0 sweep
  独立核；
- baseline：`initial_rms, tau=2, sweep=0`；
- candidate：`initial_rms, tau=2, sweep=8`；
- 唯一算法变量：sweep 数，其余均固定；
- 数据配置：`test_300x10`、1-way marginal 初始化、精确 50-query target；
- 10 个配对种子 `0..9`，每个运行 1000 轮；
- `rho=0.01`、`eta=0.5`、`mu=0.01`，geometric donor、`alpha=2→10`；
- 每个 proposal 无条件进入下一状态，不使用 loss gate、重试或 checkpoint 回滚；
- 设备：`gsd` Conda 环境、CUDA 卡 0；
- 主 RNG 在两侧保持现有抽取顺序，额外 Gibbs 微步使用独立 RNG。逐种子记录主 RNG
  端点，要求全部一致；
- 主终点：1000 轮后的最终当前 loss；同时报告 best 仅作诊断、每轮 gain 正负幅度、
  支持集唯一状态数、改变单元格、方向/因子/Gibbs 墙钟和因子工作量；
- 相同轮数回答算法动力学，不冒充相同墙钟比较。

正式运行前，0 sweep 已用种子 0、10 轮与历史无接受脚本对拍：最终 loss 均为
`1462`，最终 CSV SHA-256 均为
`1f6890ac1c68a9627d018f0f642f6a06e343a838462530e4781a68b75f487b07`。

### 5.2 首批 10 种子结果与边界

commit `e586d62`、Conda `gsd`、CUDA 卡 0 的首批正式结果为：

| 指标 | 0 sweep | 8 sweep | 配对结果 |
|------|--------:|--------:|---------:|
| 最终当前 loss（预设主指标） | 114.20 ± 46.25 | **91.75 ± 14.82** | -22.45，7/0/3，`p=0.128` |
| best loss（仅诊断） | 73.55 ± 18.41 | **63.80 ± 11.16** | -9.75，7/0/3 |
| 全轨迹平均 loss（事后诊断） | 292.52 | **222.75** | -69.77，10/0/0 |
| 最后 250 轮平均 loss（事后诊断） | 122.18 | **103.52** | -18.67，8/0/2 |
| 正收益事件比例 | 50.58% | 50.38% | -0.20 个百分点 |
| 正/负收益平均幅度 | 9.84 / -7.07 | 9.72 / **-6.82** | — |
| 最终唯一状态数 | 287.1 | 283.9 | -1.11% |
| 墙钟/种子 | 15.50s | 25.00s | +61.28% |

全部运行都跑满 1000 轮，主 RNG 端点 10/10 对齐；0 sweep 的最终 loss 均值
`114.20` 也精确复现此前相同 `tau=2` 的历史无接受实验。candidate 的因子构造、
Gibbs 采样每种子分别耗时 8.23s 和 0.445s；增量成本主要来自当前 Pandas 因子构造，
不是 Gibbs 微步本身。

主指标均值改善 19.66%，但首批只有 7/10 胜且配对区间仍跨 0，不能据此宣称最终
当前 loss 已稳定改善。全轨迹与末段均值更支持“下降更快、较长时间处于较低 loss
区域”，但这两个汇总是在看到主终点波动后分析的，只作为事后机制诊断。

### 5.3 顺序扩样协议

由于首批预设主指标不确定，保持 commit、算法配置、轮数和设备不变，追加完全不重叠
的种子 `10..29`，不删除任何结果：

- 最终当前 loss 继续作为主指标；
- 最后 250 轮平均当前 loss 在追加运行前固定为稳定性副指标；
- 同时保留全轨迹平均 loss、best、支持集和成本，但不替代主指标；
- 首批 10、追加 20 和合计 30 分开报告；
- 扩样决定发生在观察首批结果之后，所以合计 30 种子的 p 值和区间只作描述，不能
  冒充一次性固定样本量的确认性检验。

首批输出：
`outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_10seed_e586d62.json`。

### 5.4 追加 20 种子与顺序合计结果

追加运行在 commit `102aa24` 上完成，种子 `10..29` 全部跑满 1000 轮，20/20 主
RNG 对齐。它单独给出：

| 指标 | 0 sweep | 8 sweep | 配对结果 |
|------|--------:|--------:|---------:|
| 最终当前 loss（主指标） | 102.35 ± 28.25 | **89.03 ± 22.92** | -13.33，15/0/5 |
| 最后 250 轮平均 loss（预设副指标） | 121.94 ± 14.75 | **101.64 ± 13.25** | -20.29，17/0/3 |
| 全轨迹平均 loss | 299.16 ± 34.87 | **221.27 ± 21.13** | -77.89，20/0/0 |

追加 20 种子复现了首批方向，说明结果不是只由首批 seed 4 的大改善驱动。最终单点
仍有 5 个失败种子和较大波动；末 250 轮与全轨迹均值更稳定地支持“更快进入并更久
处于低 loss 区域”。

首批与追加合计 30 种子的顺序描述性汇总为：

| 指标 | 0 sweep | 8 sweep | 相对变化 | 胜/平/负 |
|------|--------:|--------:|---------:|---------:|
| 最终当前 loss | 106.30 ± 34.92 | **89.93 ± 20.35** | **-15.40%** | 22/0/8 |
| 最后 100 轮平均 loss | 114.52 ± 21.76 | **92.81 ± 15.10** | **-18.95%** | 22/0/8 |
| 最后 250 轮平均 loss | 122.02 ± 16.17 | **102.27 ± 12.75** | **-16.19%** | 25/0/5 |
| 全轨迹平均 loss | 296.94 ± 32.35 | **221.76 ± 19.38** | **-25.32%** | 30/0/0 |
| best loss（仅诊断） | 76.85 ± 14.03 | **61.20 ± 12.04** | **-20.36%** | 24/0/6 |
| 正收益事件比例 | 50.64% | 50.06% | -0.58 个百分点 | 11/0/19 |
| 正收益平均幅度 | 9.79 | 9.69 | -1.07% | 11/0/19 |
| 负收益平均幅度 | -7.08 | **-6.73** | 绝对值 -4.98% | 24/0/6 |
| 最终唯一状态数 | 285.47 | 284.07 | -0.49% | 10/2/18 |

改善没有来自“正收益事件变多”：正收益比例略降，正收益幅度近似不变，而负收益的
绝对幅度缩小约 5%。这与连续联合扩散重新分配转移质量、减轻反向步幅的机制一致，
也没有依赖 generation acceptance。最终唯一状态数只下降 0.49%，当前小表没有
明显支持集塌缩信号；但这不是高维支持集或真实分布多样性的充分证明。

合计同轮数墙钟为 `15.50→24.97s/种子`（+61.12%）。candidate 中因子构造平均
8.22s，真正 8-sweep Gibbs 抽样仅 0.444s；当前工程瓶颈是反复用 Pandas 评价并构造
查询局部因子，而不是 Gibbs 混合本身。因子预编译与批量构造属于独立性能目标，
不应在本方法 PR 中顺手混入。

因为追加样本量是在观察首批结果后决定的，合计 30 种子的配对区间和 p 值只作
描述；不能把它们描述为一次性固定 30 种子的确认性检验。更稳妥的结论是：冻结
oracle、独立追加种子和无接受长轨迹共同支持低阶联合 Gibbs 改善扩散动力，但尚未
验证标准接受闭环、nltcs、真实离线 TVD 或跨 workload 泛化。

追加与汇总输出：

- `outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_seed10_29_102aa24.json`
- `outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_sequential_30seed_summary_final.json`

## 6. 当前边界

- 本方案不改变默认生成器，也不声称已经找到宽表上的最优混合器；
- 查询阶数高于护栏时应明确拒绝，不能静默截断交互；
- 随机扫描可能在强耦合、低温或更宽的因子图上混合缓慢；阶段 A 只给当前 workload
  的有限状态证据；
- 固定 sweep 数必须是公开算法参数，不能依据私有训练表、测试表或完整联合 TVD
  在线调节；
- 当前只验证无接受动力学，没有把 candidate 接入 `run_evolution` 默认闭环，也没有
  依据小表结果改变默认温度或 sweep；
- 当前因子构造比独立核明显更慢；未来性能优化必须单独证明能量、采样轨迹和结果
  等价，并报告墙钟与内存；
- 将来进入 select-measure 与加噪 pipeline 后，查询选择、测量、预算核算和带噪
  一致性仍需单独设计与验证。

## 7. 复现记录

冻结混合实验使用 Conda `qdte`（Python 3.11.15、NumPy 2.4.6）和 NumPy/CPU。
无接受动力学使用 Conda `gsd`（Python 3.9.21、NumPy 1.26.4）与
`CUDA_VISIBLE_DEVICES=0`，设备为 NVIDIA GeForce RTX 4090 24 GiB。实验开始前
GPU 0 空闲。核心命令为：

```bash
PYTHONPATH=src conda run -n qdte python \
  scripts/probe_factorized_gibbs_mixing.py \
  --seeds 0 1 2 --state-rounds 0 100 --proposals 200 \
  --temperatures 1 2 --sweeps 0 1 2 4 8 --device numpy \
  --output outputs/factorized_gibbs/frozen_3seed_2state_200p_tau1_2_sweep0_8_a1681e7.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd python \
  scripts/compare_factorized_gibbs_unfiltered.py \
  --rounds 1000 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --temperature 2 --sweeps 8 --device cuda \
  --output outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_10seed_e586d62.json

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd python \
  scripts/compare_factorized_gibbs_unfiltered.py \
  --rounds 1000 --seeds 10 11 12 13 14 15 16 17 18 19 \
  20 21 22 23 24 25 26 27 28 29 \
  --temperature 2 --sweeps 8 --device cuda \
  --output outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_seed10_29_102aa24.json

PYTHONPATH=src conda run -n qdte python \
  scripts/analyze_factorized_gibbs_sequential.py \
  --initial outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_10seed_e586d62.json \
  --extension outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_seed10_29_102aa24.json \
  --output outputs/factorized_gibbs/unfiltered_tau2_sweep8_1000r_sequential_30seed_summary_final.json
```

`outputs/` 保持 Git 忽略；原始逐 proposal、逐轮和全部失败种子均保留，没有按结果
筛种子。冻结实验与动力学分别对应算法 commit `a1681e7` 和 `e586d62`；追加运行只在
`102aa24` 增加预注册说明和只读报告字段，没有改变更新算法。

最终深审门禁结果：

- `tests/test_factorized_diffusion.py`：34 passed；
- 因子、联合 oracle、方向、更新和集成相关测试：181 passed, 1 skipped；
- `gsd` 环境完整 CPU/torch/CUDA 测试：418 passed；
- `qdte` 环境既有演化测试：25 passed, 1 skipped；
- 三个本阶段实验/分析脚本及核心模块均通过 `py_compile`，`git diff --check` 通过；
- 正式配置 10 轮 endpoint 回归中，0 sweep 与历史 baseline 的最终表哈希均为
  `1f6890ac1c68a9627d018f0f642f6a06e343a838462530e4781a68b75f487b07`，主 RNG、
  初始 loss 和首轮方向尺度全部对齐。
