# Issue #53：`test_300x10` 残差几何 fresh-seed 确认结果

日期：2026-08-18

## 1. 结论

结果前冻结的统一候选门禁没有选出新方法：`sqrt_relative` 和 `relative` 均为
`mixed_no_unified_test_candidate`，总体分类为
`no_unified_test_candidate_under_frozen_rule`。

- `absolute` 在 measured 1-way 安全组和三个 primary 未测量组上，平均绝对计数误差全部最低。
- `sqrt_relative` 在这些组上始终比 `relative` 更接近 `absolute`，说明平方根归一化确实是几何上的
  中间点；但它相对 `absolute` 仍全部退化，不能成为 `test_300x10` 的统一候选。
- seed 310–312 上 `sqrt_relative` 在未测量 2-way 和 held-out 4-way 的微小改善，没有在全新的
  seed 313–317 上复现，方向分别变成 `+0.2267` 和 `+0.0512` count/query。
- 因而原 measured workload 中 25 条 1-way 虽会放大差异，却不是候选失败的唯一原因：完全排除
  measured 1-way 后，三个 primary 组仍一致偏向 `absolute`。

按冻结规则，当前保留 `absolute` 作为 test 参考，不再根据本批结果调 floor、gamma、混合权重或新公式。
本结果不改变 nltcs 上既有的 `sqrt_relative`/`relative` Pareto 证据，也不作跨数据 canonical 声明。

## 2. 冻结身份与审计链

```text
protocol commit       abf676e93b07837ced96ac4a311a5b401364770d
protocol doc SHA      4abe06d07f2eb59e880f8e2a16ff40e803c33a0e989487d732a1121b5e8bb785
collection commit     9f1873c1ebf7466e781687b7a17ea028f310b9cb
collection protocol   9708f994c6c479b8e08c75cc662d0f79ec3ab5ec39cd9322e2ba5e8b7b30373b
collection report     98e1b09bea3691d2c1d10b1ff6fc8830f4f5782b6f7d3b6ef49060dc82e98da8
evaluation commit     f7775dde2c6fdef67e0a9ed7fbb4ac21f279b8d3
evaluation report     54f586462c13e23a285d91d238d25246c8e7afd86016b8ee82ff6704bc5fe60f
query-seed CSV        1f158acd491add3164fb93ab0219d1323761cd7c12ec2f5c09b72d047a77466b
```

固定矩阵只有 `test_300x10`，三臂为 `absolute`、`sqrt_relative`、`relative`，fresh seeds 为
313–317，共 15 cases。除 seed 和 residual geometry 外，完全复用 P=6、C=6000、rho=0.01、fixed
alpha=16、scale-invariant donor、`initial_rms` direction、eta=0.5、mu=0.01、Gibbs=0、
terminal-current 的既有配置。

正式采集在用户指定的 `root@10.8.176.53:6006` A6000 服务器运行；生成器使用 NumPy，显式设置
`CUDA_VISIBLE_DEVICES` 为空，按 5 个 seed shard 并行、每个 shard 内三臂串行。GPU 3 上既有的约
34 GiB 无关任务未触碰。远端 collection 目录共 36 个文件，回收到本地后逐文件 SHA-256 完全一致。

采集阶段没有读取 raw reference。评价阶段先复算六组查询身份，再读取固定 SHA 的 reference 附加答案；
报告记录 `raw_reference_data_accessed=true`、`privacy_budget_consumed=false`、
`cross_group_aggregate_present=false` 和 `canonical_selection_performed=false`。

## 3. 生成结束与成本

15/15 cases 全部由 B/`early_stopped` 正常结束，没有 A/达标结束或 C/资源上限结束，满足正式判定资格。

| geometry | mean terminal measured L1 | mean normalized work | mean rounds | 正常结束 |
|---|---:|---:|---:|---:|
| `absolute` | **0.0028133** | **14.4073** | **1455.2** | 5/5 |
| `sqrt_relative` | 0.0030667 | 14.8047 | 1488.0 | 5/5 |
| `relative` | 0.0033067 | 16.8047 | 1687.0 | 5/5 |

相对 `absolute`，`sqrt_relative` 的 measured L1/work 分别高 9.00%/2.76%；`relative` 分别高
17.54%/16.64%。质量和成本没有合成总分；这里只说明本批 terminal measured workload 上
`absolute` 同时更准、更省。

## 4. 六组查询结果

下表单位为每条查询在 5 个 paired seed 上汇总的平均绝对计数误差。查询组分别报告，不跨组加权。

| 查询组 | 条数 | `absolute` | `sqrt_relative` | `relative` | 本组最低 |
|---|---:|---:|---:|---:|---|
| measured 1-way | 25 | **0.8000** | 0.9200 | 1.0080 | `absolute` |
| measured 2-way | 20 | **0.8600** | 0.8700 | 1.0200 | `absolute` |
| measured 3-way | 5 | 1.0000 | 1.1200 | **0.8000** | `relative`，仅描述 |
| 全部未测量 2-way | 531 | **6.7910** | 7.0177 | 7.4614 | `absolute` |
| 冻结 held-out 3-way | 512 | **3.9477** | 4.1637 | 4.4484 | `absolute` |
| 冻结 held-out 4-way | 512 | **1.6930** | 1.7441 | 1.8082 | `absolute` |

measured 2/3-way 只作描述，不参与统一候选门禁。measured 3-way 只有 5 条；它偏向 `relative` 不能抵消
三个独立 primary 组的同向结果。

## 5. 冻结门禁

差值均为“候选减 `absolute`”；正数表示候选误差更大。

| 候选 | 查询组 | mean delta | paired seed 更好 / 更差 / 平局 | 门禁结果 |
|---|---|---:|---:|---|
| `sqrt_relative` | measured 1-way | +0.1200 | 0 / 4 / 1 | safety fail |
| `sqrt_relative` | 全部未测量 2-way | +0.2267 | 1 / 4 / 0 | primary fail |
| `sqrt_relative` | held-out 3-way | +0.2160 | 0 / 5 / 0 | primary fail |
| `sqrt_relative` | held-out 4-way | +0.0512 | 1 / 4 / 0 | primary fail |
| `relative` | measured 1-way | +0.2080 | 1 / 4 / 0 | safety fail |
| `relative` | 全部未测量 2-way | +0.6704 | 1 / 4 / 0 | primary fail |
| `relative` | held-out 3-way | +0.5008 | 1 / 4 / 0 | primary fail |
| `relative` | held-out 4-way | +0.1152 | 2 / 3 / 0 | primary fail |

两个候选都不满足“三个 primary mean delta 全部 `<=0`”；也没有任何 primary 组同时满足 mean
delta `<0` 且 paired-seed better `>=4/5`。两者的 measured 1-way mean delta 也都大于零。因此
无需使用边界解释或事后 margin，结论直接失败。

## 6. 与上一批小样本的关系

seed 310–312 的只读分阶诊断中，`sqrt_relative` 相对 `absolute` 的三个 primary delta 为：

```text
unmeasured 2-way  -0.2166
held-out 3-way    +0.1341
held-out 4-way    -0.0111
```

fresh seed 313–317 对应变为：

```text
unmeasured 2-way  +0.2267
held-out 3-way    +0.2160
held-out 4-way    +0.0512
```

原先两个负差值都很小，且分别只有 2/3 paired seed 更好；fresh 结果方向反转，并分别只有 1/5 更好。
这正是 fresh-seed 门禁要排除的情况：不能用已见 seed 上的微弱均值差选择统一公式。

`relative` 在两批结果的三个 primary 组均没有形成优势；fresh 批次中除 4-way 为 2/5 更好外，另外
两组都只有 1/5 更好。因此“不实现 order-aware relative、不继续调 relative floor”得到更强支持。

## 7. 评价报告元数据勘误

第一次正式评价生成的报告 SHA 为
`e00e898c186ba5c20e3da104ddeb3f3865f5a042d4c91caed9c08c4900600bc3`。归档审计发现它把 plan 的
`mode=plan_only_no_collection_or_reference_read` 原样带入正式报告，同时又正确记录
`raw_reference_data_accessed=true`，属于自相矛盾的元数据标签。

提交 `f7775dde2c6fdef67e0a9ed7fbb4ac21f279b8d3` 只做两项修正：正式报告使用独立 evaluate mode；
collection commit 与 evaluation commit 分开记录，并要求前者是后者祖先。随后对同一份不可变 collection
重放评价：新旧 `query_seed_errors.csv` SHA 完全相同；删除 `mode` 和 `evaluation_git_commit` 后，
两份 JSON 逐位一致。查询、指标、门禁和结论均未改变，旧报告在本地 ignored 输出中保留作审计。

## 8. 验证与下一步

- 42 个 residual/query/held-out/confirmation 相关回归通过。
- 新 confirmation collector/evaluator 的 12 个定向测试在本机与 A6000 环境均通过。
- Ruff 0.16.3 对本研究新增的 runner、分析器、评价器及对应测试通过；`git diff --check` 通过。
- 正式评价 CSV 的 24,075 行又用 Python 标准库独立重聚合，六组均值及全部 paired-seed 胜/负/平
  与 JSON 报告一致。

本 residual 板块到此收口。下一步不再新增 residual 公式，而应先冻结跨 workload、分查询层报告的
质量—计算选择门禁，明确 test 与 nltcs 允许 workload 级不同 Pareto 选择的边界；之后再进入
Issue #53 的 donor/alpha 板块。本 PR 只归档实现、协议、诊断与 fresh-seed 证据，不修改默认值，
不自行 review 或 merge。
