# PGM 基线 plants 生成对比结果前协议

状态：**诊断协议（diagnostic_only）**。用户已授权运行。与 nltcs 基线协议
（`docs/设计/PGM基线nltcs生成对比结果前协议.md`）同构，本文件只记差异与
plants 专有事实。本协议在看到任何 plants PGM 质量结果之前起草并冻结。

## 1. 研究问题

> 给定与 plants 诊断完全相同的测量集（980 条精确 cell 查询 + 69 个一维
> 初始化边缘），Private-PGM 在同一评价口径下落在哪里——尤其是 heldout
> 3/4-way（我们的 residual 臂在此均值劣于盲臂）？

nltcs 基线已证 v3 核在覆盖充分主场全面达标（五线全不输 PGM）。plants
是覆盖受限画像（2-way 4.9% / 3-way 0.5%）+ 预算截断（6000 轮未收敛）的
数据集，是外部达标线真正要校准的战场。

## 2. 基线定义（与 nltcs 协议差异项）

```text
数据     ：plants，N=17412，69 个二值属性
测量输入 ：(a) measured_1000query.json 980 条 cell 查询
           → 375 个 clique（115 对 + 260 三元组）的 CellSubsetQuery
           (b) init_marginals.json 69 个一维完整边缘（138 格）
推断     ：MirrorDescent 同 nltcs（iters=1000、known_total=17412、
           stddev=1.0、float64、CPU、零调参）
采样     ：synthetic_data(rows=17412, method="round")，确定性
可行性   ：junction_tree.hypothetical_model_size 预检，上限 4096 MB；
           超限 → 记 baseline_infeasible，不尝试推断（防 OOM）。
           结构预检已知（非质量结果）：JT ≈ 1.1 MB，最大团 14 属性
           （16384 格）、48 个极大团——完全可行，正式运行时复算入档
```

## 3. 冻结输入身份（SHA-256）

```text
schema     configs/plants/schema.yaml
           7eaf404bfe6d3a8824a5d29cebacac704a0fae5fc5cb2227a275a35a989e0a9b
marginals  configs/plants/init_marginals.json
           4e302b18e1e6e34871a2651378ec357b6fd35edcccfc42385a0b773b1d701535
measured   configs/plants/measured_1000query.json
           f93c2d9717e2f4f87536e703019f5b48230657db227656d1af62a8640babf649
heldout    configs/plants/heldout_issue53_v1.json
           d65401761bade19ad40d9588eaa57609de0bdbc3c26835a5343ab8cc332eca85
reference  data/plants/plants.csv
           c7b5cf1e2230df3facf8d4b5d4a077d747a4f84e2dd47ce8779335f553599532
对照报告   outputs/fitness_only_plants_diagnostic_seed9908_v1/report.json
           26a8934270e9042d05e90d0a77ba1859aebd34a48ca1082291dbee9330d95565
```

对照报告的 `quality.residual` / `quality.equal`（顶层按臂，无数据集层）
直接引用，不重跑我们的臂。

## 4. 运行与解释边界

```text
runner ：scripts/run_baseline_pgm_plants_diagnostic.py（plan / run 两模式）
输出   ：outputs/baseline_pgm_plants_v1/（fail-closed 不覆盖）
```

其余（评价同码同口径、对比表五主线 pgm_minus_residual /
pgm_minus_equal、失败预案、diagnostic_only、结果不回流调参）与 nltcs
协议第 3/5/6/7 节逐字同义。
