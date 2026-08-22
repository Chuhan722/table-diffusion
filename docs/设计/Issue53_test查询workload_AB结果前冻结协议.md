# Issue #53：`test_300x10` 查询 workload A/B 结果前冻结协议

## 1. 目的与结论边界

本实验只回答两个按顺序分开的问题：

1. 旧 generation workload 中 25 条 1-way 与 1-way marginal 初始化重复后，是否使得
   `test_300x10` 的几何比较偏向 absolute？
2. 在查询总数仍为 50 的前提下，把 generation workload 改为
   `30×2-way + 15×3-way + 5×4-way` 后，absolute、sqrt-relative 和 relative
   在公共未见查询上的方向是否改变？

本协议不宣称这个 30/15/5 比例是通用最优设计，不扫描比例，不扫描
floor/gamma/rho/alpha/P，不形成跨数据集 canonical 结论。所有查询身份、seed、
生成参数、评价组和判定规则必须在读取新查询答案和生成结果前冻结。

## 2. 固定 A/B generation workload

本项目中的一条 query 是一个标量 conjunction cell count，不是一整个边际向量。
两组都使用同一份 `init_marginals.json` 做 1-way 初始化；两组唯一改变是后续
measured generation workload。

### 2.1 Workload A：旧设计

```text
25 条 1-way + 20 条 2-way + 5 条 3-way = 50
source = configs/test_300x10/measured_50query.json
query identity = cbb501f5c2f8c230b6d68d85baf40be7b17be713d41c5b97f54ac30457e90fc8
```

### 2.2 Workload B：新设计

保留 A 的 D01–D20 和 T01–T05，删除 S01–S25，再加入结果盲选取的
10 条 2-way、10 条 3-way、5 条 4-way：

```text
30 条 2-way + 15 条 3-way + 5 条 4-way = 50
query identity = 602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1
```

新增查询分阶身份：

| 阶数 | 数量 | query identity SHA-256 |
|---|---:|---|
| 2-way | 10 | `c87b8dd421c21b799e218c204eb5f3d87c708e3eb40b3fccef3404a32d751681` |
| 3-way | 10 | `c422642908450206e82bef8b0c6aca474b6e9af61824e833cbcd408a5720fcdd` |
| 4-way | 5 | `394f9dddd68c38638d81c10f0c3f06d7f2159cafec2c4a772d5f2e856cecbdc6` |

具体条件、语义指纹和 selection SHA 写入
`configs/test_300x10/issue53_query_workload_ab_v1.json`；该文件只冻结身份，不允许有
`result` 字段。

## 3. 结果盲选取算法

冻结输入：

| 输入 | SHA-256 |
|---|---|
| `init_marginals.json` | `1e0fb0413c5ed53907a760d491fda84aec8162642a39cf8eadc577d7d1ec9ee4` |
| `measured_50query.json` | `7cccd58400a8e7bf74aed6efe01069f3142dde166b37a39cd3d18408b8cecb88` |

候选只使用 `init_marginals.attributes` 中的属性顺序、`type`、`values` 和 `bins`。
禁止使用 marginal `counts`、旧/新 query `result`、raw reference、terminal error、稀有度或
任何已见实验结果。

对 order=2/3/4，先枚举公开域的全部标准 conjunction cells，用忽略 ID、type、
expression、result 和 condition 顺序的规范 JSON 语义指纹去重。新查询排序键为：

```text
SHA256("issue53-test-query-workload-ab-v1\0test_300x10\0<order>\0<query_fingerprint>")
```

按 `(selection_sha256, query_fingerprint)` 升序排序后分别取前 10/10/5。排除规则为：

- 三个阶数都排除与 A 语义重复的查询；
- 3-way 和 4-way 额外排除既有 `issue53-heldout-v1` 的各 512 条固定身份；
- 固定 held-out 用旧 A 和原 namespace 结果盲重建，不允许改用 B 重建。

候选空间审计：

| 阶数 | 公开 cell | 与 A 精确重叠 | 固定 held-out | 新查询可选 | 实选 |
|---|---:|---:|---:|---:|---:|
| 2-way | 548 | 17 | 0 | 531 | 10 |
| 3-way | 5,056 | 5 | 512 | 4,539 | 10 |
| 4-way | 30,450 | 0 | 512 | 29,938 | 5 |

旧 D04、D05、D07 不是公开网格中的单个标准 cell，但仍保留在 B；上表的
2-way 精确重叠因此是 17 而不是 20。

## 4. 固定公共评价查询

四组始终分开，不做跨组 aggregate：

| 查询组 | 数量 | 角色 | query identity SHA-256 |
|---|---:|---|---|
| `one_way_safety` | 25 | 1-way 安全性；对 A 是 measured，对 B 只有相同 marginal init | `b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c` |
| `common_unseen_2way` | 521 | primary；548 个公开 cell 排除 A∪B 语义并集 | `fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2` |
| `fixed_heldout_3way` | 512 | primary；原 result-blind held-out 身份 | `d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a` |
| `fixed_heldout_4way` | 512 | primary；原 result-blind held-out 身份 | `2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171` |

`one_way_safety` 不是两组都未见的 held-out，因此只作安全性边界；三个 primary
组才是 A/B 共同未见集。新 3/4-way 已从固定 held-out 中排除，不得因 B
的加入而换一批 held-out。

## 5. 固定生成矩阵

```text
dataset   = test_300x10
workloads = [A, B]
geometries= [absolute, sqrt_relative, relative]
seeds     = [318, 319, 320, 321, 322]
cases     = 2 × 3 × 5 = 30
```

同一 seed 的六臂使用相同 1-way marginal 初始化和同一初始 RNG seed，必须审计
initial table SHA 六臂一致。除 workload 和 residual geometry 外，生成参数完全复用
已冻结 P=6 实验：

```text
rho                                  = 0.01
P / inner patience                   = 6 natural-work ticks
n_rounds / candidate_budget          = 6000 / 6000
selection_scale_invariant            = true
selection_scale_invariant_min_spread = 1e-3
alpha_schedule_mode / fixed_alpha    = fixed / 16
diffusion direction normalization    = initial_rms
eta / mu                             = 0.5 / 0.01
factorized_gibbs_sweeps              = 0
tol                                  = +inf
max_retries                          = 0
output                               = terminal_current
```

三种 geometry 保持既有定义，不改 floor。每个 seed shard 内串行顺序固定为：

```text
A/absolute -> B/absolute -> A/sqrt_relative -> B/sqrt_relative
-> A/relative -> B/relative
```

可以在 A6000 服务器上并行 5 个 seed shard，但 shard 内不并行；生成器仍使用
NumPy，`CUDA_VISIBLE_DEVICES` 为空。在 30 条轨迹全部物化前，在线停止不得读取
raw reference 或四个离线评价组。

## 6. 答案附加与信息流

本协议首先物化不含 `result` 的身份文件。只有在以下审计全部通过后，
下一步才可以读取固定 SHA 的 raw reference：

1. A/B 查询数和阶数构成精确；
2. B 无 1-way，且 50 条内无语义重复；
3. 新 3/4-way 与固定 held-out 语义不相交；
4. `common_unseen_2way` 与 A∪B 语义不相交；
5. 身份文件递归检查不存在 `result` 字段；
6. 结果值、marginal counts 和 terminal errors 的改变不能改变查询选取身份。

附加 B 的 50 个答案后，必须再次计算 query identity，与
`602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1` 一致；答案文件是后续
runner 的新输入，不得反过来影响查询选取。

## 7. 固定指标与比较顺序

每个评价组、workload、geometry 分别报告：

- 5 个 paired seed 各自的 mean absolute count error；
- 全部 query×seed 的 mean、median、p90、max absolute count error；
- mean 除以 300 的 normalized L1 和 exact-match rate；
- B 相对 A、以及 B 内候选 geometry 相对 absolute 的 query×seed 和 paired-seed
  better/tie/worse。

生成成本另行报告 terminal measured L1、squared loss、rounds、normalized work、
elapsed 和 A/B/C 终止理由。质量与成本不标量化，measured workload 的 aggregate 只描述各自
拟合，不用两套不同查询的 aggregate 直接判 A/B 输赢。

比较顺序固定为：

1. 先在每个 geometry 内比较 B 相对 A 的四个公共评价组；
2. 再在 B 内比较 sqrt-relative、relative 相对 absolute；
3. 禁止先挑一个结果最好的 geometry，再反向宣称 workload B 有效。

## 8. 结果前冻结的判定规则

执行资格要求 30/30 都以 A/`fit_target_reached` 或 B/`early_stopped` 结束。任意一条
C/`resource_cap_reached` 都使对应的比较记为 `inconclusive_resource_cap`，不增加 seed。

### 8.1 查询设计：B 相对 A

对 absolute、sqrt-relative、relative 分别判定。B 的公共未见门禁要求：

1. `common_unseen_2way`、`fixed_heldout_3way`、`fixed_heldout_4way` 三组 mean delta
   `B - A <= 0`；
2. 至少一组 mean delta `< 0`，且该组 paired-seed B-better 数量 `>= 4/5`。

1-way 安全门禁要求 `one_way_safety` mean delta `B - A <= 0`。不设事后
non-inferiority margin。

- 两个门禁都通过：`supports_workload_B_under_geometry`；
- 公共未见通过但 1-way 失败：`higher_order_gain_with_1way_tradeoff`；
- 其他正常完成组合：`mixed_no_workload_replacement`。

不跨 geometry 汇总或投票。只有三种 geometry 的方向分类都一致，才可描述为
`geometry_independent_workload_effect`；否则固定为 `geometry_dependent_workload_effect`。

### 8.2 Workload B 内的 residual geometry

对 sqrt-relative、relative 分别相对 absolute 使用同一个三组公共未见 Pareto 门禁
和 1-way 安全门禁：三个 primary mean delta 均 `<= 0`，至少一组 `< 0` 且
paired-seed candidate-better `>= 4/5`，同时 1-way mean delta `<= 0`。这一阶段只解释新
workload 下的 geometry，不反向修改 8.1 的查询设计判定。

## 9. 产物与版本库边界

```text
configs/test_300x10/issue53_query_workload_ab_v1.json  # 结果前身份
outputs/issue53_test_query_workload_ab_v1/             # 后续 30-case 产物
```

本步只冻结协议和查询身份，不附加答案、不实现 runner、不运行实验。
后续未经用户明确要求，不 push、不创建或更新 PR、不自行 review/merge。
