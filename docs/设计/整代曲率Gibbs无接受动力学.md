# 整代曲率 Gibbs 的无接受长期动力学

## 1. 研究动机

Issue #18 在固定 proposal 上验证了整代曲率条件式：`gamma=1` 同时降低逐行自身
二次项与跨行交叉项，三个晚期状态的相对净收益和正收益率全部改善。但 baseline 与
candidate 的晚期平均净收益仍分别为 `-5.0917/-3.9775`。这只说明 candidate 在
固定状态上相对少坏，不能推出连续无接受更新会进入更低 loss 的稳定区域。

本阶段把同一个曲率算子放进长期 Markov 轨迹，每个 proposal 无条件成为下一状态，
不执行 loss acceptance、重试、回滚、早停或 best checkpoint 选择。研究问题是：

> 完整曲率是否改善扩散动力学本身，而不是只改善冻结状态的一次 proposal？

本设计在编码前同步到 Issue #24。正式结果不能反过来修改这里的种子、轮数、参数、
主终点或判断门槛。

## 2. 算法变量与退化关系

沿用 Issue #18 的整代复制 mask 能量

\[
V_\gamma(M)
=\langle e,\Delta q(M)\rangle
-\frac{\gamma}{2N}\lVert\Delta q(M)\rVert_2^2.
\]

baseline 和 candidate 都调用同一个
`evolve_step_generation_curvature_gibbs`：

- baseline：`gamma=0`，精确退化为既有最高三阶因子 Gibbs；
- candidate：`gamma=1`，复制 mask 能量等于平方 workload proposal 收益除以 `N`；
- 两侧都保留 `mu=0.01` 的独立变异层。曲率能量只定义复制 mask，不把后续变异
  冒充为 `V_1` 恒等式的一部分。

唯一算法变量是 `gamma=0/1`。不能同时改变温度、sweep、参与率、变异率、donor、
初始化或查询 workload。

## 3. 无接受轨迹语义

每轮固定执行：

1. 从当前合成表计算预定义 workload 的查询答案、比例残差和 fitness；
2. 使用 geometric donor 机制抽取 donor；
3. 在首轮非零方向 RMS 固定尺度下计算方向分数；
4. 执行 8-sweep 整代曲率 Gibbs 复制与独立变异；
5. 不查看 proposal loss 是否更好，直接令 proposal 成为下一状态。

loss、best 和 gain 可以记录为诊断，但不参与接受、停止、重试、参数调度或输出表
选择。即使残差暂时全部为 0，也继续跑满固定 1000 轮，避免数据依赖控制流改变
随机数和正式样本量。

两侧使用相同初始化种子与连续主 RNG。主 RNG 负责 donor、参与、独立初始 mask 和
mutation；在跑满相同轮数时最终端点必须对齐。Gibbs RNG 按公开
`seed/round_index` 地址化，每轮重新创建，避免上一轮活跃 bit 数不同导致下一轮
随机流整体错位。

轨迹分叉后，两侧状态、fitness、donor 概率、实际 donor 与活跃 bit 本来就可以
不同；这些是候选动力学的下游结果，不能伪造 donor 对齐。配对含义是相同初始表、
算法外参数和地址化随机源，不是强迫不同状态使用同一实际 donor。

## 4. 一次性正式协议

- 数据配置：`test_300x10`；
- 运行时信息：公开 schema、公开记录数 300、预定义 50-query 精确 target、公开
  1-way marginal；
- 种子：一次性固定 `0..19` 共 20 个配对种子，不顺序扩样；
- 轮数：每个变体 1000 轮，全部跑满；
- donor：geometric、排除 self，`alpha` 从 2 线性增加到 10；
- 更新：`rho=0.01`、`eta=0.5`、`mu=0.01`；
- 方向：`initial_rms`、`tau=2`；
- Gibbs：8 sweep、最高查询因子阶数 3、条件 logit 护栏 30；
- baseline/candidate：`gamma=0/1`；
- 短轨迹退化预检：正式运行前用 seed 0、20 轮逐轮对拍 `gamma=0` 与既有因子
  Gibbs 的表、loss、主/Gibbs RNG 和共同诊断；
- 设备：Conda `gsd`、CUDA 0；
- 同轮数比较；时间只报告，不冒充同墙钟结论；
- 不读取真实训练/测试表，不报告联合 TVD；
- 每个 seed 独立保留完整 loss/gain 轨迹，失败种子不得删除或替换。

正式脚本必须要求 tracked 工作树干净、记录 commit/命令/环境/GPU/公开输入哈希，
并拒绝覆盖正式输出。

## 5. 预注册主终点与判断

主终点是每个种子最后 250 个**更新后当前状态**的平均 workload loss。每条轨迹
记录初始状态与 1000 个更新后状态；末 250 轮明确取最后 250 个更新后状态，包括
最终当前表。

选择末段平均而不是单点 final loss，是在正式运行前固定的稳定性口径。最终当前
loss 仍是重要副指标；best loss 只能作为离线诊断，不能成为返回状态。

通过全部语义门禁后：

- candidate 的 20 种子聚合末 250 轮均值相对 baseline 至少降低 5%，且至少
  14/20 个配对种子更低：`supports_unfiltered_curvature_dynamics`；
- 聚合均值降低但相对不足 5%，或只有 11--13 个种子改善：
  `curvature_dynamics_inconclusive`；
- 聚合均值不降，或至多 10/20 个种子改善：
  `curvature_dynamics_not_supported`。

不以 round/proposal 级 p 值代替种子级规则，也不在观察结果后调
`rho/gamma/tau/sweep/mu`、删除失败种子、延长轮数或改变末段窗口。

## 6. 指标与风险标记

每个种子、每个变体记录：

- 初始、最终、best（仅诊断）、全轨迹、末 100/250 轮当前 loss；
- 每轮原始 gain，正/零/负比例与正负幅度，最大瞬时 loss；
- 最终/平均唯一记录数、改变单元格数；
- 条件概率范围、微步加权条件熵、最大有效 `|logit|`、护栏命中次数与双向支持；
- 方向、因子构造、Gibbs 抽样和总墙钟，参与行、活跃 bit、因子表项与微步数；
- 初始/最终表哈希、主 RNG 端点、地址化 Gibbs 端点摘要和完整 loss/gain 轨迹。

预先声明三个风险标记：

- candidate 全轨迹微步加权条件熵相对 baseline 下降超过 10%：探索集中风险；
- candidate 最终唯一记录数聚合均值相对下降超过 5%：支持集收缩风险；
- candidate 每种子最大瞬时 loss 的聚合均值比 baseline 增加超过 20%：阶段爆炸
  风险。

风险不事后替代主判断，但会限制“稳定扩散”的表述。复制量、正收益比例或 best
loss 不能在结果出现后升级为新的成功终点。

## 7. 语义与复现门禁

正式判断前必须全部满足：

1. seed 0、20 轮 `gamma=0` 退化预检逐轮表、loss、主/Gibbs RNG 和共同诊断精确
   一致；
2. 两侧初始表、初始 loss、首轮方向尺度和主 RNG 最终端点 20/20 对齐；
3. 40 条轨迹全部完成 1000 轮，没有 loss/残差驱动的停止或分支；
4. `gamma=0` 微步条件概率对既有条件式最大误差为 0；
5. 正式 logit 护栏命中 0 次，所有条件概率严格位于 `(0,1)`；
6. 正式输出严格有限，20 个种子及所有不利轨迹完整保留；
7. 独立审计从原始轨迹复算每种子指标、配对汇总、主判断、风险和哈希。

任一门禁失败时，结论只能是实现或实验失败，不能解释算法效果。

## 8. 实现与隐私边界

第一阶段只新增研究脚本、测试和本设计文档，不接入 `run_evolution`、
`scripts/run.py` 或默认参数。因子预编译和批量构造继续由 Issue #15 独立处理；
只有本阶段通过后才另开标准接受闭环 Issue。

生成运行时只使用公开 schema、公开记录数、预定义查询、精确 target、公开 marginal
和显式种子/参数。当前 target 仍是无噪声精确计数，所以本研究不是 DP；查询选择、
测量噪声、预算 accountant 与带噪一致性都不在本阶段范围。

关联 Issue #24、#18、#17、#16、#14。
