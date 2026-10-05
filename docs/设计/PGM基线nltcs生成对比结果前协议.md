# PGM 基线 nltcs 生成对比结果前协议

状态：**诊断协议（diagnostic_only）**。用户已授权 nltcs 先行；plants 对比
待后续单独授权。本协议在看到任何 PGM 结果之前起草并冻结。

## 1. 目的与研究问题

fitness-only 生成核（v3 冻结配置）至今只有内部对照（配对 equal 盲臂），
缺一条**外部达标线**：同样的已发布测量，领域标准生成器能做到什么分数。

本对比只回答一个描述性问题：

> 给定与 v3 屏完全相同的 nltcs 测量集（1001 条精确 cell 查询 +
> 16 个一维初始化边缘），Private-PGM 的 mirror descent 推断 + 采样
> 产出的合成表，在同一套评价口径下（measured / held-out 3&4-way /
> one-way safety）各指标落在哪里？

不设门控、不做正式效果声明、不据此立即改核——结果先如实归档。

## 2. 基线定义（结果前冻结，零调参）

```text
实现     ：private-pgm（mbi 包），SSH 克隆 GitHub 官方仓库
           commit 07635f9a0f1150237a1da1083a29b38aa4ff8645，
           pip 安装进 .venv（第三方代码不 vendor 入本仓库）
推断     ：estimation.MirrorDescent().estimate(...)
           iters=1000（库默认）、stepsize 自动、known_total=16181
           float64 开启、JAX 编译缓存关闭、CPU 后端
测量输入 ：(a) measured_1000query.json 的 1001 条 cell 查询，按 clique
           分组为线性测量（CellSubsetQuery 选格，0/1 指示、op_norm=1）
           (b) init_marginals.json 的 16 个一维完整边缘
           ——信息对等：我们的核同样以 (a) 为 fitness、(b) 为初始化
           全部精确答案（无噪声），stddev=1.0 等权
采样     ：model.synthetic_data(rows=16181, method="round")
           （受控舍入，确定性，无随机 seed）
禁止     ：任何一侧调参；重跑我们的臂；解读为门控证据
```

## 3. 对照与评价口径

- 我们侧数字**直接引用 v3 冻结产物**，不重跑：
  `outputs/fitness_only_schedule_T6000_dev_seed9908_v3/report.json`
  （SHA-256 见第 4 节）中 `quality.nltcs.residual` 与 `quality.nltcs.equal`。
- PGM 合成表用**同一条评价代码路径**打分：`evaluate_table` +
  `evaluate_quality_snapshot` + 同一 heldout 文件 + 同一 one-way 构造
  （init_marginals 的 values/counts）+ 同一 N=16181 归一。
- 对比表输出五条主线的 normalized_l1（mean/median/p90）与差值：
  measured、heldout_3way、heldout_4way、heldout_combined、one_way_safety，
  各给 `pgm_minus_residual` 与 `pgm_minus_equal`。

## 4. 冻结输入身份（SHA-256）

```text
schema     configs/nltcs/schema.yaml
           5765de90ea97bb6617c960f9cf81fee97ca4975296bfdd67686667729cc4e7f4
marginals  configs/nltcs/init_marginals.json
           a5e63ea80c49cfb1ac7cdb88662ce54641f4dab33ac60bda53e332cd123ea25e
measured   configs/nltcs/measured_1000query.json
           b34eb2d5a16ce1deeafbdcda7af9a9b971a490e59df0099d7c7c55ce70f0468f
heldout    configs/nltcs/heldout_issue53_v1.json
           a025b5b4d2d44261b03075d4aca030cc10fc0c1d99051241a671f465644003eb
reference  data/nltcs/nltcs.csv
           7d185b8a065e051341e581ba65b27007d72e5d2b67adfffacf213117799edf7c
v3 对照    outputs/fitness_only_schedule_T6000_dev_seed9908_v3/report.json
           0a613bb4bdb695f53469d5b7bb132725eb4a3df66f9404be707f81280d14dd48
```

marginals/measured/heldout 三值与 v3 报告内 `evaluation_inputs` 已交叉
核对一致（one_way_safety_sha256 = marginals SHA）。

## 5. 运行与输出

```text
runner   ：scripts/run_baseline_pgm_nltcs_diagnostic.py
           plan 模式盲计划（不读输入不读结果）；run 模式需确认协议 SHA
输出     ：outputs/baseline_pgm_nltcs_v1/report.json
           staging 目录写完整才落名；已存在则 fail-closed 拒绝覆盖
记录     ：推断耗时、采样耗时、mbi 运行时版本、逐 clique 测量审计
           （clique 数、测量值总数、y 与 target 逐条一致）
```

## 6. 失败模式预案（结果前声明）

- junction tree 内存不可行 / 推断发散 / NaN → 如实记录
  `baseline_infeasible`，不解读质量数字；
- 评价与 v3 口径任何 SHA 漂移 → fail-closed 中止；
- PGM 表行数/列域漂移 → fail-closed 中止。

## 7. 解释边界

单配置、单数据集、确定性采样的一次描述性对比。它回答"标准方法在这份
考卷上考多少分"，**不回答**"哪个方法更好"——后者需要多种子、多数据集
与预注册判据。对比结果不得直接回流为核参数调整依据（防结果泄漏）。
