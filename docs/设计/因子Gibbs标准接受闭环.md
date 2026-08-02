# 因子 Gibbs 标准接受闭环

## 1. 研究问题

Issue #14 已在冻结状态和关闭整代接受的动力学中证明：最高三阶稀疏因子加有限步
随机扫描 Gibbs 能逼近完整联合扩散核，并且不依赖后置接受筛选就能改善长期
workload loss。本阶段只回答下一层问题：把同一个候选接入现有 `run_evolution`
的标准整代接受闭环后，它是否仍优于同温度的独立单块扩散核。

本阶段不同时优化因子构造性能，不进入 nltcs、跨 workload、同墙钟调度或 DP
pipeline，也不改变任何默认参数。关联 Issue #16；因子构造性能单独由 Issue #15
跟踪。

## 2. 候选与唯一变量

两侧共享现有残差驱动扩散的单块方向、首轮 RMS 固定定标和独立 Bernoulli 初始
mask：

- baseline：`initial_rms, tau=2, sweep=0`，即现有独立单块复制核；
- candidate：`initial_rms, tau=2, sweep=8`，在同一个独立初始 mask 上执行最高
  三阶随机扫描 Gibbs；
- 唯一算法变量是 sweep 数。

两侧都保留现有 donor 抽样、变异、整代 loss 接受、缓存和历史最优表。candidate
没有正收益门槛、argmax、top-k 或额外 checkpoint 选择。额外 Gibbs 微步只使用从
公开主 seed 确定性派生的独立随机流，不消费主随机流。

主循环新增选项必须默认关闭。`sweep=0` 不构造查询因子、不消费附加 RNG，并与接入
前 `run_evolution` 的生成表、loss/接受/donor 轨迹和主 RNG 端点精确一致。

## 3. 正式实验协议

- 配置：`test_300x10`；公开 schema、公开记录数 300、预定义 50-query 精确 target；
- 初始化：公开 1-way `marginal`；
- 种子：一次性固定配对种子 `0..19`，不根据中途结果扩样或删种子；
- 轮数：每侧 500 轮；若精确残差归零则保留现有提前停止语义并如实记录；
- 更新：`rho=0.01`、`eta=0.5`、`mu=0.01`；
- donor：geometric，`lambda=0.5`、`alpha=2→10`、`delta=0.05`、winsorize
  `(0.01, 0.99)`、排除自身；
- 接受：现有 `proposal_loss <= current_loss + 1e-9`；`max_retries=0`；
- 方向：`residual_directed_diffusion=True`、`initial_rms`、`tau=2`；
- candidate：8 sweep、最高因子阶数 3；
- 设备：Conda `gsd`、CUDA 卡 0；正式运行前确认设备空闲；
- 顺序：偶数 seed 先 baseline、奇数 seed 先 candidate，减轻固定热身顺序对墙钟的
  系统影响；算法输出不应受运行先后影响；
- 预算口径：主结论只回答同轮数，不冒充同墙钟比较。

正式运行必须基于已提交且 tracked 工作树干净的 commit，输出记录完整命令、环境、
GPU、commit、参数、逐种子表哈希、初始 loss、首轮方向尺度以及主/附加 RNG 端点。

## 4. 生成与离线评价隔离

生成阶段只能读取 schema、记录数、查询定义、已发布 target、1-way marginal、seed
和算法参数。必须先完成全部 40 次 `run_evolution` 调用并保存结果，之后才允许读取
`data/test_300x10/test_300x10.csv`。

离线评价把公开 marginal 配置中的 age 五个区间作为固定离散化，不根据生成结果或
真实表重新分箱。报告：

- 训练 workload：best loss 与 normalized L1；
- 接受动力学：接受率、接受前原始 gain、正收益率、正负 gain 幅度和末 25% 进展；
- 未测量高阶：全部离散 3-way 单元格中排除 workload 已测量的 5 个三阶单元格，
  以及全部离散 4-way 单元格的 mean/median/P90/max L1；
- 联合分布：原始十属性经验联合 TVD、固定分箱后的联合 TVD；
- 支持集：唯一状态数、真实质量遗漏、合成新增质量和支持交集；
- 成本：总墙钟、方向评价、因子构造、Gibbs 抽样、因子数、表项数和微步数。

`test_300x10` 没有独立 train/test 划分，因此上述联合 TVD 只能称为对单一真实参考
表的离线指标，不能包装成 held-out 测试结果。正式 train/test 与跨 workload 结论
留到后续 nltcs 阶段。

## 5. 主终点与判断规则

主终点是 500 轮标准闭环返回的 best workload loss。固定 20 个配对差值
`candidate - baseline`，报告两侧均值/标准差、平均与相对差、配对 95% t 区间、
双侧配对 t 检验和胜/平/负。这里只设置一个确认性主终点；其他指标用于解释边界，
不替代主终点。

- 若主终点平均差小于 0、95% 区间上界也小于 0，且至少 14/20 种子改善，则当前
  小表标准闭环支持 candidate；
- 若均值改善但区间跨 0 或胜种子不足，则结论为不确定；
- 若均值不改善，则当前标准闭环不支持 candidate，不通过换温度、删种子或追加轮数
  覆盖失败结果。

即使主终点支持 candidate，只要离散联合 TVD、未测量 3-way/4-way mean L1 中任一
项的聚合均值相对恶化超过 5%，或唯一状态数下降超过 5%，结论也必须限制为“训练
workload 优化改善”，不能称为完整生成质量改善。5% 只作为预先声明的风险标记，
不作新的显著性检验。

同轮数 candidate 已知更慢。如果结果支持 candidate，下一步是在 Issue #15 完成
严格等价的因子预编译/批量构造后，再单独预注册同墙钟或 nltcs 实验；本阶段不按
观察到的耗时动态补跑 baseline。

## 6. 运行前门禁

正式 20 种子实验前必须依次通过：

1. 参数、零轮、初始即收敛、`eta` 端点、非法阶数和模式组合测试；
2. `sweep=0` 相对接入前路径的同种子表、决策轨迹和主 RNG 端点回归；
3. 非零 sweep 的固定种子复现、独立 Gibbs RNG 生命周期和重试路径测试；
4. 小表 NumPy 冒烟与 CUDA 冒烟；
5. 语法检查、针对性测试、完整 CPU/torch/CUDA 测试和 `git diff --check`；
6. 正式运行前提交实现与测试，使输出能指向固定 commit。

本阶段仍是固定精确 target 的无噪声原型，不是 DP。真实参考表只能用于生成完成后
的离线评价，不得进入初始化、接受、早停、sweep 选择或 checkpoint 选择。
