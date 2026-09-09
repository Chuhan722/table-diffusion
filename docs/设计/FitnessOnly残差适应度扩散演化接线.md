# Fitness-only 残差适应度扩散演化接线

状态：代码路径与专项测试已接通；已完成 seed 9908 开发归因与 test 覆盖诊断，
尚未进入正式多种子质量声明。

## 1. 方法边界

每轮只允许一条残差信息流：

```text
current table -> measured counts -> residual -> row fitness
                                              |
                                              v
                                  fitness + structural distance
                                              |
                                              v
                                        donor sampling
                                              |
                                              v
                   blind independent copy (rho, eta) + mutation (mu)
                                              |
                                              v
                                      next current table
```

donor 选定后，独立核不读取 residual、target、loss 或候选收益。候选表无条件
成为下一状态。历史最好表和所有 loss/residual 轨迹仅作只读诊断，不参与状态
推进、终止或最终选表。

这里的准确主张是“fitness 是演化阶段唯一依赖 residual 的驱动力”，而不是
“fitness 是唯一驱动力”：结构距离仍约束局部 donor，固定随机复制和变异仍提供
扩散与探索。

## 2. 两条配对归因臂

| 臂 | 行适应度 | donor 的其他信息 | 更新核 |
|---|---|---|---|
| `residual` | 每轮由当前 residual 重算 | 结构距离 | 盲独立核 |
| `equal` | 所有行恒为 0 | 与实验臂完全相同的结构距离 | 同一盲独立核 |

`equal` 不是均匀 donor：它保留结构距离，只删除 residual fitness。这样两臂的
唯一处理差异是行适应度，能够回答 residual fitness 本身是否产生增量价值。

两臂使用同一 seed、初始化、轮数和随机调用地址。配对入口会核验初始表哈希、
初始化后 RNG 状态、最终 RNG 状态、轮数和候选评价数；任一不一致即失败。

## 3. Fail-closed 合同

`run_evolution(..., fitness_only_mode=...)` 只在下列条件全部满足时运行：

- `distance_mode='geometric'`；
- 尺度不变 donor 选择开启；
- alpha 固定，不读取轮数进度或停滞状态；
- `residual_directed_diffusion=False`；
- factorized Gibbs 与 C/gap 核均为 0；
- 显式无条件状态接续，不使用 loss 接受门；
- 不重试、不做 residual self-cooling；rho 恒定，或走预冻结的绝对轮数
  纯时间驱动时间表（见文末 2026-09-04 修订；全程式退火仍被拒绝）；
- 不做精确残差早停或 P=6 等结果相关早停；
- `candidate_budget=None`，只以固定 `n_rounds` 结束；
- 不允许直接读取 target 的 `pairwise_maxent` 初始化；
- 主返回恒为 terminal current，不返回历史最好表。

高层入口位于 `src/table_diffevo/fitness_only.py`，负责构造上述配置并在运行后
再次审计。通用主循环的 legacy 默认行为保持不变。

## 4. 当前沿用而不重新选择的基线

第一轮归因实验不同时修改 residual 几何。沿用此前高阶 measured workload 已冻结
的 development baseline：

```text
residual_geometry       relative
residual_geometry_floor 8
rho                     0.01
eta                     0.5
mu                      0.01
fixed alpha             16
lambda                  0.5
scale-invariant floor   1e-3
initialization          frozen 1-way marginals
```

这些数值是首轮归因的共同条件，不代表已经证明为 fitness-only 架构下的全局最优值。

## 5. 当前证据边界

专项测试已证明：禁止的控制路径会在运行前被拒绝；`equal` 臂更换 target 后生成
轨迹保持不变；两臂的初始化与随机流地址一致；无门控轨迹跑满固定轮数并返回最后
状态，即使历史中间状态更好也不会回滚。

这些是代码与因果设计正确性证据，不是质量结论。正式质量主张必须等待两数据、
多配对种子的结果前协议和离线评价完成。

当前验证记录：

```text
fitness-only 专项                                      19 passed
主循环/尺度不变选择/方向核/向量化相关回归              264 passed
全仓                                                    2357 passed, 40 failed
```

全仓 40 个失败均属于旧证据的预期护栏或工作树产物状态：旧 B+C/R8 协议冻结了
`evolution.py` 的 SHA-256，新分支修改核心后它们按设计 fail closed；其余失败是
旧测试要求某些历史输出必须存在或尚不存在，而当前工作树的产物状态不满足。没有
fitness-only 专项或通用演化行为断言失败。不得据此写成“全仓通过”，也不得更新旧
协议哈希来伪装兼容；旧协议应继续在原分支、原提交上复现。

## 6. 已完成的开发运行

可复现入口为：

```text
scripts/run_fitness_only_attribution.py
```

seed `9908`、两数据集、两臂各固定 3000 轮的报告：

```text
outputs/fitness_only_attribution_dev_seed9908_v1/report.json
```

该报告的 `formal_claim_allowed` 为 `false`。test 上 residual 只改善被测 B，
held-out 与 one-way safety 变差；NLTCS 上四类质量指标均改善。因此它证明了
归因流程可执行，但单 seed、跨数据集方向不一致，不能直接进入正式确认。

## 7. 覆盖诊断

为检查 test 的 one-way 退化，另有结果后诊断入口：

```text
scripts/run_fitness_only_test_coverage_diagnostic.py
```

它把 B 的 50 条查询与固定 25 条 one-way 查询合并为 75 条生成输入，其他条件
不变。报告位于：

```text
outputs/fitness_only_test_coverage_diagnostic_seed9908_v1/report.json
```

补回 one-way 后，residual 相对 equal 的 one-way safety 差值由 `+0.038933` 变为
`-0.014667`，但 held-out 3/4-way 差值仍分别为 `+0.002760`、`+0.000964`。
这支持“查询覆盖不足是主要原因之一”，但不支持“补 one-way 后全部泛化问题
消失”。该产物明确标记为 `diagnostic_only`，不承担 promotion gate。

当前路线：先冻结这些现场，设计仍只改变 row fitness 的覆盖/聚合诊断并记录
donor 距离与终点波动；在得到稳定方向前，不启动预注册 5-seed 正式确认，也不
引入早停、接受门或历史最好选表。
