# Issue #53 残差几何查询级诊断结果

日期：2026-08-18

## 1. 结论

跨数据反转不能只解释为“`nltcs` 稀有查询多、`test_300x10` 稀有查询少”。固定频率分箱后，
`relative` 在 `nltcs` 的 rare、medium、common 三档里都降低了平均绝对计数误差；真正区分两个
workload 的更强信号是查询阶数与初始化边缘：

- `test_300x10` 有 25/50 条 1-way 查询，25 条的 target 全部逐条等于 generation-visible
  marginal count，因此 marginal 初始化时本来就是精确约束。terminal 上 `relative` 明显破坏了这批
  已精确边缘。
- `nltcs` measured workload 没有 1-way，479 条 2-way、522 条 3-way；`relative` 在每个
  frequency×order 格子中都优于 `absolute`，不存在一阶边缘被扰动的同类代价。
- `test_300x10` 已进入整数分辨率区：三臂平均每查询绝对计数误差均约 1 条记录，`absolute` 的 exact
  match rate 为 40.67%，`relative` 降到 26.00%。这不是仍有大量稀有查询没有拟合，而是在相互耦合
  的离散查询间重新分配少量整数误差。

因此，不应继续扫固定归一化指数，也不应把下一候选做成简单 target-frequency selector。当前最有
根据、最小且可证伪的候选是 **order-aware / marginal-protected relative geometry**：1-way 残差保留
absolute 恢复力，2/3-way 使用 relative；实现前必须先冻结两块信号的共尺度化与 fresh-seed 门禁。

## 2. 输入、身份和边界

```text
analysis commit     deb659f3346f3dac92763a4479418b619027b061
source report SHA   241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143
diagnostic report   876b7cc2f75ddf315800dd36853ca617fbbbbbf6258bc908709bec49c251e48b
query-seed rows     9459
query summary rows  1051
```

固定入口和口径：

```text
scripts/analyze_issue53_residual_geometry_queries.py
docs/设计/Issue53_残差几何查询级诊断口径.md
```

本诊断在三臂聚合结果可见后进行，只读取原 18 张 terminal-current 表及 generation-visible measured
queries/target。它不生成新表、不读取 raw reference、不消耗隐私预算，不是 canonical 或结果前选择
证据。入口逐项核验 source report、query 文件、18 张终态表 SHA，并将逐查询复算的 overall L1 与原
报告逐臂交叉核对。

频率固定为：`rare <5%`、`medium 5%–20%`、`common >=20%`，另预留 target=0 和 target<floor=8
两档；当前两个 workload 均没有后两类。结构重叠是每条查询属性集合与其他查询属性集合的平均 Jaccard，
按输入 q25/q75 分 low/middle/high，不读取误差。

## 3. Workload 组成

| 数据 | rare | medium | common | 1-way | 2-way | 3-way |
|---|---:|---:|---:|---:|---:|---:|
| `test_300x10` | 0 | 33 | 17 | 25 | 20 | 5 |
| `nltcs` | 454 | 268 | 279 | 0 | 479 | 522 |

`test_300x10` 的 25 条 1-way target 与 `init_marginals.json` 对应 count 为 25/25 精确一致。这说明
relative 在该数据上的主要风险不是“没把一阶边缘学好”，而是演化过程中把本已精确的一阶边缘换成了
高阶查询收益。

## 4. Overall 与整数分辨率

| 数据 | 残差 | mean abs count/query | exact-match rate | fractional query-seed win rate |
|---|---|---:|---:|---:|
| `test_300x10` | `absolute` | 0.7800 | 40.67% | 39.44% |
| `test_300x10` | `sqrt_relative` | 0.8933 | 34.67% | 33.78% |
| `test_300x10` | `relative` | 1.1267 | 26.00% | 26.78% |
| `nltcs` | `absolute` | 18.3197 | 2.63% | 14.09% |
| `nltcs` | `sqrt_relative` | 7.9780 | 5.29% | 35.74% |
| `nltcs` | `relative` | 5.6224 | 6.33% | 50.17% |

在 `nltcs` 的 3003 个 query×seed 配对中，`relative` 对 `absolute` 为 2197 better / 145 tie /
661 worse，即 73.16% 的配对更好；改善不是只靠少量查询。但绝对残差同时留下了一些数百计数误差的
大尾部，relative 将它们压低，因此均值改善幅度仍会大于简单胜率所暗示的程度。

`test_300x10` 的同一配对是 39 better / 49 tie / 62 worse。这里每查询平均误差已低于或接近一条记录，
exact match 数量比连续近似更能说明离散取舍。

## 5. 按频率分解

下表为三 seed、每档查询的平均绝对计数误差：

| 数据 | 频率档 | query 数 | `absolute` | `sqrt_relative` | `relative` |
|---|---|---:|---:|---:|---:|
| `test_300x10` | medium | 33 | **0.838** | 0.899 | 0.980 |
| `test_300x10` | common | 17 | **0.667** | 0.882 | 1.412 |
| `nltcs` | rare | 454 | 18.009 | 7.105 | **4.943** |
| `nltcs` | medium | 268 | 14.469 | 7.139 | **5.072** |
| `nltcs` | common | 279 | 22.524 | 10.204 | **7.257** |

`nltcs` 的三个频率档方向一致，所以“只在 rare 上使用 relative”不能解释或复现已有收益。common
查询也随 rare/medium 单元格的修正而改善，这与同一二元/三元结构中的单元格计数相互耦合相符。

## 6. 按查询阶数分解

| 数据 | 阶数 | query 数 | `absolute` | `sqrt_relative` | `relative` |
|---|---:|---:|---:|---:|---:|
| `test_300x10` | 1 | 25 | **0.693** | 0.733 | 1.373 |
| `test_300x10` | 2 | 20 | **0.850** | 1.083 | 0.900 |
| `test_300x10` | 3 | 5 | 0.933 | 0.933 | **0.800** |
| `nltcs` | 2 | 479 | 16.547 | 7.482 | **5.814** |
| `nltcs` | 3 | 522 | 19.946 | 8.434 | **5.446** |

`test_300x10` 的 1-way 占 relative 总绝对误差 60.95%，而在 absolute 中只占 44.44%；其
exact-match rate 从 absolute 的 45.33% 降到 relative 的 17.33%。2-way 两端接近，3-way 虽然
relative 均值最低，但只有 5 条查询、15 个 query×seed 观察，不能据此作强结论。

继续交叉 frequency×order 后：

- `nltcs` 的 rare/medium/common × 2/3-way 共六格全部是 relative 最低；
- `test_300x10` 的 medium/common × 1-way 两格都偏 absolute，2-way 基本打平，唯一 3-way 格略偏
  relative。

按结构重叠 low/middle/high 分层后，两个数据各自的方向均未翻转。该粗粒度共享属性指标不是当前反转
的主要解释，但不能排除更细的互补单元格结构。

## 7. 下一步

下一步先写结果前设计，不直接实现或跑新矩阵：

```text
候选：order_aware_relative
1-way query   -> absolute residual
order >= 2    -> relative residual, floor=8
```

设计必须先解决一个关键问题：absolute 与 relative 的原始尺度不同，直接拼接可能让 1-way 块重新压过
全部高阶查询。应在不增加可调扫描的前提下，明确块内/块间标准化，并冻结以下可检验性质：

1. 当 workload 没有 1-way 时，候选应与现有 relative 路径数值等价；
2. 当 1-way residual 为零时，不凭空产生保护力；一旦高阶移动破坏 marginal，恢复信号必须可见；
3. 使用 fresh seeds 在 `test_300x10` 比较 absolute/relative/candidate，不能用当前 310–312 调尺度；
4. `nltcs` 先做路径等价测试，再决定是否需要冗余重跑；
5. 仍报告 P=6 terminal measured L1/work，并增加按阶误差；canonical 前再补 held-out、高阶联合、
   支持集和多样性。

若该候选无法同时保护 test 的一阶边缘和保留高阶收益，应停止 residual 指数/分流路线，转向显式
marginal-preserving update/projection 设计，而不是继续增加频率阈值。
