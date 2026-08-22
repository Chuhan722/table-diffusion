# Issue #53：`test_300x10` 分阶 held-out 只读诊断协议

## 1. 目的与边界

本诊断只回答一个问题：在已经完成的 P=6 三臂早停实验中，`test_300x10` 的结论是否被测量查询里的 1-way 查询主导。

- 只读取已经生成的 9 张 `terminal_current` 表，不重新生成，不改变早停或残差几何参数。
- 这是看过主实验结果后的机制诊断，不能作为新的 canonical 方案选择证据。
- 各阶查询分别报告，禁止把不同查询组加权、平均或标量化成一个总分。
- 本诊断会在查询身份冻结后读取 `test_300x10` 原表，为未测量 2-way 查询附加答案；因此必须明确记录 `raw_reference_data_accessed=true`。

## 2. 冻结输入

主实验固定为：

- source report：`outputs/issue53_sqrt_residual_earlystop_comparison_v1/report.json`
- source report SHA-256：`241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143`
- source execution commit：`fe8fb797a718bf0e9a89668d46fbd5726c1c3082`
- arms：`absolute`、`sqrt_relative`、`relative`
- paired seeds：`310`、`311`、`312`
- 每个 case 的表身份必须与 source report 中的 `terminal_table_sha256` 相同。

`test_300x10` 输入固定为：

| 输入 | SHA-256 |
|---|---|
| `configs/test_300x10/schema.yaml` | `58087cbba7eb90e82974bc9ffc2222510705b97599f00ae207765e03b60cf792` |
| `configs/test_300x10/init_marginals.json` | `1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4` |
| `configs/test_300x10/measured_50query.json` | `7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88` |
| `configs/test_300x10/heldout_issue53_v1.json` | `300bffea1f3d9105ad8f1840d50a900616115659065efec35b3c02f7a38cc1e0` |
| `data/test_300x10/test_300x10.csv` | `c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb` |

## 3. 查询分组

查询阶数定义为一个合取查询中不同属性的数量。每组必须与其他组分别报告。

1. `measured_1way`：现有 50 条测量查询中的 1-way，预期 25 条。
2. `measured_2way`：现有测量查询中的 2-way，预期 20 条。
3. `measured_3way`：现有测量查询中的 3-way，预期 5 条。
4. `unmeasured_2way_all`：根据公开 `init_marginals.json` 域枚举全部标准 2-way cell，按语义指纹排除与 50 条测量查询完全相同的查询后保留全部候选，预期 531 条，不抽样。
5. `heldout_3way_512`：直接使用已冻结 `heldout_issue53_v1.json` 的 512 条 3-way。
6. `heldout_4way_512`：直接使用同一文件的 512 条 4-way。

未测量 2-way 的身份选择只能使用公开域和测量查询的语义指纹。必须先在内存中冻结全部 531 个身份并验证无重复、与测量查询不相交，然后才能加载原表附加 `result`。不得依据真实答案、稀有程度或任何终态表误差筛选。

现有 3/4-way held-out 文件必须复核：总数 1024、每阶 512、与测量查询语义不相交，且确定性重建的查询身份与文件一致。

不另造所谓“未测量 1-way”：这里的标准 1-way cell 已由公开边缘分布完整给出；1-way 仅作为 `measured_1way` 非回归诊断。

## 4. 冻结指标

对每个查询组、每个 arm 分别报告：

- `query_count` 和 `query_seed_count`；
- 三个 paired seed 各自的平均绝对计数误差；
- 全部 query×seed 的绝对计数误差 mean、median、p90、max；
- mean 除以 300 的 normalized L1；
- exact-match rate。

固定报告三组 paired 对比：`sqrt_relative - absolute`、`relative - sqrt_relative`、`relative - absolute`。对每个查询组报告：

- query×seed 平均绝对误差差值；
- query×seed 的 candidate better / tie / worse 数量；
- 每个 paired seed 的组平均误差差值，以及 seed better / tie / worse 数量。

差值小于 0 表示前一个 arm 更好，大于 0 表示更差。这里不设事后容差、不做显著性声明，也不把六组汇成总分。

## 5. 结果解释边界

结论按以下预先固定的方向解释：

- 若某 arm 主要在 `measured_1way` 变差，而在 `unmeasured_2way_all`、`heldout_3way_512`、`heldout_4way_512` 都不变差，则支持“原 50 查询汇总受 1-way 主导”的解释。
- 若它在三个未测量组都变差，则支持“该 arm 在 N=300 的未测量联合查询上确有弱点”的解释。
- 其他组合一律称为 mixed：指出具体在哪一阶好、哪一阶差，不宣布普遍赢家。
- `measured_2way` 和 `measured_3way` 用于区分训练内拟合与训练外泛化，但不能替代三个未测量组。
- 只有 3 个 paired seed，本诊断只提供描述性机制证据；任何新算法选择仍需另行冻结新实验。

## 6. 预期产物

输出目录为 `outputs/issue53_test_ordered_heldout_diagnostic_v1/`，至少包含：

- `report.json`：输入审计、查询身份、六个独立分组及 paired 对比；
- `query_seed_errors.csv`：每条查询、每个 seed、每个 arm 的答案与误差；
- `unmeasured_2way_queries.json`：身份先冻结、之后附加答案的 531 条查询，供复核。

输出目录只作本地可复算产物，不覆盖已有目录，不消耗隐私预算。
