# 高温因子 Gibbs 混合前沿与强无门控基线

> 状态：阶段 A 预注册，尚未实现新探针、尚未运行高温结果
> 日期：2026-08-13
> 关联：Issue #49、#10、#14、#24、#27、#30、#43、#44

## 1. 本阶段要回答什么

已有结果分别说明：

- 独立方向核在 `test_300x10` 的无门控 1000 轮动力学中，已测试范围内
  `tau=8` 最强；
- 因子 Gibbs 在 `tau=2` 下能够改善独立核，但尚未与 `tau=4/8` 的强方向基线结合；
- 固定 `gamma=1` 曲率长期失败，因此在建立强因子基线前不继续加入新曲率。

阶段 A 只回答：

> 在真实无门控高温轨迹访问的初始、中期和晚期状态上，`tau=4/8` 分别需要多少
> random-scan Gibbs sweeps，才能充分逼近同温度的联合 mask oracle？

本阶段不选择最终温度，不判断长期终点，也不实现选择性软曲率。阶段 A 只为每个温度
选择最小充分 sweeps；温度与核的长期效果由后续阶段 B 单独比较。

## 2. 现有代码审计

### 2.1 可以直接复用的部分

- `src/table_diffevo/factorized_diffusion.py`
  - 已实现最高三阶稀疏 mask 能量；
  - `random_scan_gibbs_mask` 支持任意非负微步数；
  - `propagate_random_scan_distribution` 能在小状态空间精确传播随机扫描分布；
  - `n_steps=0` 精确退化到初始独立 mask，且不消费 Gibbs RNG；
  - 条件 logit 默认使用 `[-30, 30]` 数值护栏。
- `scripts/probe_factorized_gibbs_mixing.py`
  - 温度和 sweeps 已是命令行参数，没有把 `tau=2` 或 8 sweeps 写死；
  - 能对每条活跃 recipient-donor 完整枚举 mask，并核对稀疏因子能量；
  - 能精确报告 TVD、KL、mask 熵、负方向质量和期望方向差距；
  - donor、参与行和 Gumbel 随机量在所有候选间配对；
  - proposal 只作离线评价，不执行 mutation、generation acceptance 或 best 回滚。
- `tests/test_factorized_diffusion.py`
  - 已覆盖平稳分布、精确传播、随机扫描采样器、schema 顺序等变性、数值护栏、
    0-sweep 回归和独立 Gibbs RNG 语义。

### 2.2 不能直接沿用的部分

现有 mixing 脚本是 Issue #14 的历史协议，不能直接把参数改成 `tau=4/8` 后当作
Issue #49 的正式实验：

1. 非初始状态来自历史有门控 `run_evolution`，且返回的是 best 表，不是强无门控
   轨迹的 current state；
2. 每个冻结状态会重新估计方向 RMS `s0`，会在晚期重新放大微小残差，不符合实际
   动力学中“首个非零尺度固定后全程复用”的语义；
3. 非初始状态的 donor `alpha` 一律设为 10，没有使用同一条 1000 轮轨迹的实际阶段值；
4. 精确传播使用截断后的条件核，但比较目标仍是未截断联合 Gibbs 分布；高温下若命中
   logit clip，两者不再是同一个目标，现有脚本没有检测这一点；
5. 只有逐状态结果，没有跨状态的活跃行加权汇总、阶段分层汇总和自动选择最小充分
   sweeps 的正式判断；
6. 没有输入文件哈希、工作树洁净门禁、正式参数身份、独立重算审计和防覆盖原子写入；
7. 没有针对 mixing 脚本本身的专项测试；
8. 运行时间只按整个状态汇总，不能清楚区分不同 sweep 深度的增量成本；
9. 当前主机只有系统 Python 3.10，且没有项目依赖；正式实现和运行前必须建立满足
   `pyproject.toml` 所要求的 Python 3.11+ 隔离环境。

因此后续应保留历史默认协议不变，以显式的新协议模式或新入口实现 Issue #49 阶段 A。

## 3. 冻结的小表输入

本阶段只使用：

```text
schema：configs/test_300x10/schema.yaml
queries/target：configs/test_300x10/measured_50query.json
marginals：configs/test_300x10/init_marginals.json
N：从 query 文件 record_count 读取并与 marginals 的 n_records 交叉核对，必须为 300
属性数：10
查询数：50
```

预注册时的输入 SHA-256：

```text
schema：58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792
queries：7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88
marginals：1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4
```

生成与探针都只读取上述公开配置和精确 target，不读取真实 CSV，不评价真实联合 TVD，
也不属于 DP 实验。

## 4. 共同状态库

### 4.1 状态来源

使用开发 seeds `0,1,2`。这些 seeds 已用于历史机制探索，本阶段明确把它们继续视为
开发状态，不会在后续长期确认中冒充新确认 seeds。

对每个 seed 生成两条 1000 轮独立方向无门控轨迹：

```text
source tau：4、8
factor Gibbs sweeps：0
generation acceptance：关闭，所有 proposal 成为下一 current state
rho=0.01，eta=0.5，mu=0.01
donor：geometric，alpha 在完整 1000 轮中按 2→10 调度
normalization：initial_rms
```

同 seed 的两条轨迹必须从相同 marginal 初始表和同一主 RNG 起点出发。首轮 donor 和
方向尺度计算发生在温度影响 mask 之前，因此两条轨迹的正有限 `s0` 必须精确一致；否则
状态生成门禁失败。

### 4.2 快照

每个 seed 保存：

- round 0：共同 marginal 初始 current state，只保留一份；
- round 500：分别保存 source `tau=4` 和 `tau=8` 的 current state；
- round 1000：分别保存 source `tau=4` 和 `tau=8` 的 current state。

合计 `3 × (1 + 2 × 2) = 15` 个 seed-state。不得使用历史 best 表代替 current state。
每个快照记录完整表哈希、current loss、来源温度、轮次、对应 donor alpha、固定 `s0`
和主 RNG 状态哈希。

冻结探针复用来源轨迹的 `s0`，不得在每个快照重新定标。round 0、500、1000 的 probe
alpha 分别使用完整 1000 轮日程对应的 2、中间值和 10；不得把所有非初始状态统一设为
10。

## 5. 阶段 A 网格与随机耦合

正式检查网格：

```text
evaluation tau：4、8
sweeps：0、8、16、32
每个 seed-state：200 个冻结 proposal 条件
rho=0.01，eta=0.5，probe mutation=0
最大查询因子阶数：3
logit clip：30
device：numpy / CPU
```

`sweeps=0` 是同温度独立方向分布，只用于测量初始差距和恢复比例；真正候选为
`8/16/32`。

对每个固定 seed-state-proposal：

- donor、参与行、基础 Bernoulli 核和 Gumbel tape 在所有 evaluation tau/sweeps 间
  共享；每个 evaluation tau 仍按自己的方向强度定义相应的独立初始 mask 分布；
- 随机源采用地址化派生，不因增加 sweeps 而错位其他 proposal；
- 每个 sweep 定义为 `k` 个随机、有放回的坐标微步，其中 `k` 是该行 active bit 数；
- 分布由有限状态转移精确传播，不以有限 Monte Carlo 链估计 TVD；
- 另用小规模采样器对拍确认正式 `random_scan_gibbs_mask` 与精确传播语义一致。

## 6. 指标与汇总

### 6.1 正确性指标

- 稀疏因子能量与完整 mask oracle 的最大绝对误差；
- one-hot 因子方向与独立方向分数的最大绝对误差；
- 每次传播后的概率总和、有限性和非负性；
- TVD 随累计微步的最大反向变化；
- 未截断条件 logit 的最大绝对值、clip 命中次数；
- 条件概率范围、条件熵和双向数值支持；
- 固定种子下采样器与精确传播的一致性。

### 6.2 混合指标

- 到同温度联合 oracle 的 TVD；
- `KL(candidate || oracle)`；
- 从 0 sweep 到 oracle 的期望方向差距恢复比例；
- mask 熵、条件熵、到参考 Bernoulli 核的 KL；
- 负方向完整 mask 质量及其 oracle 值。

### 6.3 proposal 与成本诊断

- 精确 workload gain、线性收益、二次惩罚；
- 正/零/负 gain 比例；
- 改变单元格数和记录数；
- factor 构造、精确传播和完整 probe 的墙钟；
- 每个候选的微步数与相对 0 sweep/oracle 的配对差值。

精确 proposal 诊断从有限状态传播后的边缘分布抽样；另对同一批冻结因子运行正式
`random_scan_gibbs_mask` 的采样器对拍与计时，不能用精确传播墙钟冒充生产采样成本。

proposal gain 只用于验证“分布更接近 oracle”是否转化为有意义的编辑，不参与 sweeps
选择。

### 6.4 汇总层级

至少同时保留：

1. 每个 seed-state；
2. initial、mid/source-4、mid/source-8、late/source-4、late/source-8 五个状态族的
   活跃 recipient 行加权汇总；
3. initial / mid / late 三阶段及 source `tau=4/8` 的补充分层汇总；
4. 全部状态的活跃 recipient 行加权总汇总。

不得把 proposal 或 recipient 行当作独立 seed 做显著性推断。

## 7. 语义门禁与最小充分 sweeps

正式结果按以下顺序判断：

1. 稀疏能量和 one-hot 方向最大误差均不超过 `1e-10`；
2. 概率传播全部有限、非负、归一；TVD 快照反向变化不超过 `1e-12`；
3. 未截断条件 logit 在正式范围内不得命中 `|logit|=30` 护栏。若命中，停止本次
   分类，先明确是修改温度范围、关闭护栏，还是定义截断核自己的平稳 oracle；
4. 状态库必须来自无门控 current state，两个来源温度的初始表和 `s0` 逐 seed 精确
   一致，所有来源轨迹跑满 1000 轮；
5. 对某个 evaluation tau，候选 sweep 必须在全局汇总以及 initial、mid/source-4、
   mid/source-8、late/source-4、late/source-8 五个状态族汇总中同时满足：
   - 到联合 oracle 的 TVD 不高于 `0.05`；
   - 期望方向差距相对 0 sweep 至少恢复 `80%`；
6. 在 `8、16、32` 中选择第一个满足全部门禁的最小 sweeps。

若某个温度到 32 sweeps 仍不通过，该温度不进入阶段 B；不得看过结果后自动追加 64、
修改状态、删除失败 seed-state 或放宽门槛。若两个温度都不通过，阶段 A 记为失败，另行
预注册新的混合策略。

阶段 A 不根据 proposal loss 在 `tau=4/8` 之间选温度。若两者各自得到充分 sweeps，二者
都进入阶段 B 的同温度 `independent vs factor Gibbs` 长期无门控对照。

## 8. 正式运行与审计要求

正式运行前必须：

- 建立 Python 3.11+ 隔离环境并记录 Python、NumPy、pandas、SciPy、PyYAML 和 torch
  版本；
- 新代码和测试提交后，在 clean worktree 的固定 commit 上运行；
- 先通过核心 factor Gibbs 测试、阶段 A 脚本专项测试和单 seed 短轨迹冒烟；
- 输出参数必须与本协议完全匹配，否则标记为非正式；
- 输出采用排他创建，默认不得覆盖已有结果；
- JSON 保存完整协议、输入 SHA、commit、命令、环境、状态身份、逐状态原始汇总和
  自动分类；
- 独立 audit 必须重新读取正式输入、核对协议身份并重算全部汇总与最小 sweeps 选择，
  不能相信 JSON 自带的 `passed=true`；
- 记录正式 JSON 的 SHA-256。失败结果同样保留。

计划输出目录：

```text
outputs/issue49_high_temperature_factor_gibbs/
```

## 9. 下一步代码清单

1. 为无门控独立方向轨迹增加不改变既有默认行为的 current-state snapshot 能力，或在
   新研究入口中复用相同更新语义；
2. 扩展 mixing 探针，使其接受外部状态库、固定 `s0` 和阶段 alpha；
3. 增加高温 raw-logit/clip、条件熵、阶段汇总和最小 sweeps 分类；
4. 增加正式协议身份、输入哈希、防覆盖和独立审计；
5. 增加脚本专项测试，并确认旧 `tau=1/2` 默认命令和历史核心语义不被改变；
6. 建立本地 Python 3.11+ CPU 环境，先运行既有 factor Gibbs 回归，再开始实现。

完成上述实现与门禁后，才运行本协议的正式 `tau=4/8 × sweeps=0/8/16/32` 小表检查。
