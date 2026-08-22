# Issue #53：`test_300x10` 查询 workload A/B 正式结果与结果后解释

日期：2026-08-18

## 1. 当前结论

本实验应以 **workload B 内部的 residual geometry 比较**作为当前研究主问题，而不以
“B 的公共评价误差是否低于 A”决定正常查询设计是否成立。

- workload A 是旧的 `25×1-way + 20×2-way + 5×3-way` 查询集。其中 25 条
  1-way target 与 marginal 初始化统计 25/25 精确一致，初态残差为零，并继续占据固定
  measured objective 的一半；它适合作为解释旧 `test_300x10` 结果的机制对照。
- workload B 使用同一份 1-way marginal 初始化，但后续 measured objective 固定为
  `30×2-way + 15×3-way + 5×4-way`。它对应本项目当前要研究的正常内层语义：
  1-way 提供初始单变量分布，演化阶段集中拟合尚未满足的高阶关系。
- A、B 接受的持续监督不同。A 在 1-way 及其相关公共指标上更准，不能反向证明 B 的
  查询设计错误，也不能作为选择 residual geometry 的必要门槛。
- 在 B 内部，`relative` 相对 `absolute` 的三个 primary group mean 与 1-way safety
  mean 全部改善，2-way 和 3-way 都达到预先冻结的 4/5 paired-seed 稳定改善。因此当前
  正式证据支持 B 下的 `relative`，不支持把 `sqrt_relative` 作为统一中间答案。

由此修正旧解释：原 `test_300x10` 上偏向 `absolute` 的结论依赖旧 workload A 的特殊
组成；它不能否定 `relative` 在正常高阶 measured workload 下的适用性。当前 residual
板块采用 `relative`、`residual_geometry_floor=8` 作为后续 development baseline；通用
`run_evolution` 默认值仍保持 `absolute`，避免在独立兼容性决策前改变公共 API。

## 2. 结果身份与不可变边界

```text
collection report    outputs/issue53_test_query_workload_ab_v1/collection_report.json
collection SHA       67f3ebbcf06100b0ba508b465dd4aea7b6ee69825a46b5eec5a768245b69e44a
evaluation report    outputs/issue53_test_query_workload_ab_v1/evaluation_report.json
evaluation SHA       a389504c92e87461d84c4eb8322b659afea0dabb58a256bedcd6c19f78c06651
query-seed CSV       outputs/issue53_test_query_workload_ab_v1/query_seed_errors.csv
query-seed CSV SHA   2bbcfba869187cfdd1b7198f9d2e675437f38d8a6e4081c73f9b03289b6c467c
seeds                318, 319, 320, 321, 322
cases                30
termination          30/30 early_stopped; 0 resource caps
```

本次结果后解释没有重新生成表、增加 seed、修改查询、重算 reference answer、调整门禁或
覆盖正式 evaluator 输出。正式报告中的
`mixed_no_workload_replacement`、`supports_geometry_under_workload_B` 等冻结分类原样保留；
这里只修正哪个问题与当前方法选择相关，不把结果后解释冒充结果前判定。

## 3. 公共查询评价

四组公共评价查询的 mean absolute count error：

| 查询组 | A abs | B abs | A sqrt | B sqrt | A relative | B relative |
|---|---:|---:|---:|---:|---:|---:|
| 1-way safety | 0.8560 | 16.1840 | 0.8560 | 15.2400 | 0.9280 | **13.0080** |
| common unseen 2-way | 7.4779 | 10.8088 | 7.1328 | 10.4791 | 7.6891 | **9.1708** |
| fixed held-out 3-way | 4.2859 | 5.1699 | 4.1855 | 5.1148 | 4.5813 | **4.5902** |
| fixed held-out 4-way | 1.7578 | 1.9848 | 1.7715 | 1.9516 | 1.8809 | **1.8949** |

加粗只标识 B 内三种 geometry 的最低 mean，不跨 A/B 选择方法。

### 3.1 冻结的 A/B 辅助判定

原结果前协议要求先计算 `B - A`。三种 geometry 在四组上的 mean delta 都为正，正式
分类均为 `mixed_no_workload_replacement`。这个数值结果有效，回答的是：

> 移除持续 1-way measured supervision 并同时换入更多高阶查询后，能否仍在 A 直接或间接
> 监督的公共统计上不劣于 A？

答案是否定的。但它不是当前需要回答的“正常高阶 workload 下哪种 residual geometry 更好”。
A、B 的 measured targets 不同，A 的额外持续监督本来就会帮助 1-way，并可通过边缘关系帮助
部分 unseen 2-way。因而不能从 `B > A` 的公共误差推出“B 查询设置失败”，也不能据此要求
在正常内层中重新加入持续 1-way anchor。

由于 B 还同时把 10 条新 2-way、10 条新 3-way 和 5 条新 4-way 换入固定预算，本实验也
没有单变量隔离“25 条初始零残差 1-way”这一因素。它能够直接证明的是 geometry 排序随
workload 改变；结合 25/25 初始精确匹配，零残差 1-way 是旧 A 偏向 `absolute` 的明确机制，
但不单独宣称它解释全部 A/B 数值差异。

### 3.2 当前主判定：B 内 geometry

候选相对 B/`absolute` 的 mean delta；负数表示候选更好：

| candidate | 1-way safety | unseen 2-way | held-out 3-way | held-out 4-way | paired 稳定性 | 正式分类 |
|---|---:|---:|---:|---:|---|---|
| `sqrt_relative` | -0.9440 | -0.3298 | -0.0551 | -0.0332 | 3/5、3/5、2/5 | `mixed_no_unified_geometry_candidate` |
| `relative` | **-3.1760** | **-1.6380** | **-0.5797** | **-0.0898** | 4/5、4/5、3/5 | `supports_geometry_under_workload_B` |

`relative` 同时降低了 B 的 1-way safety drift；所以本结果不支持“relative 改善高阶但必然
牺牲初始化边缘”的解释。B 的绝对 1-way error 仍明显高于 A，但是否需要单独的一维质量门槛
属于产品质量契约，不能用 A/B 方法选择替代。

B 内的生成成本也与质量方向一致：

| geometry | mean terminal measured L1 | mean normalized work | mean rounds |
|---|---:|---:|---:|
| `absolute` | 0.0024267 | 16.6040 | 1660.0 |
| `sqrt_relative` | 0.0024133 | 13.8047 | 1382.8 |
| `relative` | **0.0023600** | **9.8053** | **984.0** |

质量与成本没有合成单一分数；表格仅说明 B 下 `relative` 并非依靠更多工作量换取公共查询改善。

## 4. 与既有 nltcs 结果的关系

既有 P=6 三臂实验中，nltcs 的 measured workload 不含 1-way，`relative` 在 3/3 paired
seeds 下取得最低 terminal measured L1；旧 test workload A 则在 3/3 seeds 下偏向
`absolute`。本次把 test 改为 B 后，`relative` 在公共 unseen 2-way 和 held-out 3-way
均达到 4/5 稳定改善，方向与 nltcs 一致。

因此当前可以冻结的 development 结论是：

```text
1-way marginal initialization
+ higher-order measured workload
+ relative residual geometry (floor=8)
```

这是后续 donor/alpha 研究的固定 baseline，不是跨所有数据、查询类、带噪阶段或公共 API
默认值的全局定理。若以后加入独立数据，应先保证 geometry 比较不会再次让已由初始化精确满足的
1-way 查询占据大量固定 objective，再做一次结果前冻结的确认；不按新结果回头调整 floor 或查询比例。

## 5. 与 AIM、Private-GSD 的边界

两篇论文只能提供设计参照，不能当作本项目 A/B 协议的同义实现：

- [AIM](https://arxiv.org/abs/2201.12677) 在初始化阶段测量候选中的 1-way marginals，随后
  从 workload downward closure 自适应选择边缘；初始化测量继续参与 Private-PGM 拟合，后续
  候选技术上也仍可包含 1-way。
- [Private-GSD](https://proceedings.mlr.press/v202/liu23ag/liu23ag.pdf) 的遗传种群均匀随机
  初始化；论文的主要离散边缘实验直接匹配 2-way queries，并不是 1-way marginal 初始化。

两者都不能证明“A 或 B 必然正确”。与当前实验直接相关的共同启发只是：方法评价应围绕实际要
拟合的查询类进行，不能让一批由特定初始化已经精确满足的标量查询无意中主导 residual geometry
排序。

## 6. 收口与下一步

本 residual geometry 板块不再增加公式、seed 或 A/B workload 变体。当前只做以下收口：

1. 保留正式 A/B artifacts、SHA、冻结判定与失败历史；
2. 将 workload A 标记为机制对照，workload B 内比较标记为当前主结论；
3. 后续研究固定 `relative`、floor=8 和高阶 measured workload；
4. 下一科学板块进入 donor/alpha，另写结果前协议，并且一次只改变该板块允许的变量。

本结果文档不修改代码默认值，不 push，不创建、更新、审查或合并 PR。
