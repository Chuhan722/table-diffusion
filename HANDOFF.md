# 项目交接

> 最后更新：2026-08-01（Asia/Shanghai）
>
> 本文件记录当前可接手状态；长期规则见 `AGENTS.md`，历史证据见
> `PROJECT_STATUS.md`。

## 1. 当前 Git 与 GitHub 状态

- 本地仓库：`/home/qianqiu/my_life/chuhan`
- 上游仓库：`Chuhan722/table-diffusion`
- 个人 fork：`houyuwushang/table-diffusion`
- 当前文档分支：`docs/project-agent-handoff`，基于 `origin/master`
- PR #1：二阶最大熵初始化与可选缩步重试，已合并。
- PR #2：性能修复，当前 Open、Ready、CLEAN、MERGEABLE：
  <https://github.com/Chuhan722/table-diffusion/pull/2>
- PR #3：本文档分支对应的项目代理规范与交接文档 PR：
  <https://github.com/Chuhan722/table-diffusion/pull/3>
- 本文档分支不包含 PR #2 的代码；后续若 PR #2 先合并，文档 PR 应在合入前同步
  最新 `origin/master` 并重新检查冲突。

开始新任务时必须重新查询 GitHub，以上状态可能已经变化。

## 2. 已完成并验证的工作

### PR #1：二阶最大熵初始化

- 从已测量的完整二阶等值边缘用 IPF 拟合有限状态最大熵分布，再抽样生成 `S_0`。
- nltcs 65,536 状态、120 个属性对，87 次扫描约 3.6 秒。
- 三种子 600 轮相对旧 marginal 1500 轮，训练/测试联合 TVD 从约
  `0.3147/0.3957` 降到 `0.2352/0.3261`。
- 仓库仍不是 DP；该功能只属于测量后的后处理原型。

### PR #2：主循环性能修复

- 拒绝提案后复用当前表的查询答案、残差、fitness、loss 和距离；接受后失效。
- GPU 只 gather 被选中的 donor 距离，不再回传完整约 1 GiB 距离矩阵。
- nltcs 种子 0、600 轮：`268.88s → 31.18s`，加速 `8.62×`。
- 优化前后合成 CSV SHA-256 完全相同；loss、接受轨迹和质量指标完全一致。
- 8 组旧版/新版深度对拍覆盖 NumPy、legacy、重试、linear、允许自身、torch
  CPU/CUDA，轨迹和最终表全部一致；torch 距离诊断最大差 `5.96e-8`。
- 完整测试 `283 passed`。短程内存对照：RSS 约 `3.12 GiB → 1.14 GiB`，GPU
  峰值约 `10.0 GiB → 9.0 GiB`。

## 3. 当前结论应如何解释

- “演化后 workload L1 降低约 18.5%”的 baseline 是同种子的二阶最大熵初始表
  `S_0`，不是旧 marginal。
- 同一组对照中，演化把 workload L1 从约 `0.001716` 降到 `0.001398`，未测量
  3-way L1 降约 10.1%；但训练/测试联合 TVD 分别约上升 2.86%/2.83%。
- 这说明演化改善了它直接优化的 workload，但没有改善完整联合分布。不能把
  pairwise 成品相对旧 marginal 的大幅改善全部归因于 600 轮演化。
- nltcs 训练集与测试集经验联合 TVD 本身约 0.2834；当前 TVD 仍有改进空间，
  但不能期望 16 维经验联合 TVD 接近零。

## 4. 已发现但尚未实现的改进

### 优先级 A：低方差最大熵离散化

当前 `rng.choice` 的普通多项式抽样重新引入较大有限样本误差。10 种子只读探索：

| 方法 | workload L1 | 训练 TVD | 测试 TVD |
|------|------------:|---------:|---------:|
| 当前普通抽样 | 0.001913 | 0.22765 | 0.31825 |
| residual resampling | 0.001133 | 0.20769 | 0.31025 |
| 随机排列后系统重采样 | 0.001020 | 0.20119 | 0.30983 |
| 连续最大熵模型 | 0.000639 | 0.20115 | 0.30989 |

这是探索结果，不应直接改默认。下一步应单独建方法分支：先做可选实现和排列敏感性
测试，再跑配对多种子及演化后的正式对照。

### 优先级 B：正式 DP 基础

- 当前 target 是精确计数，没有 ε/δ、噪声机制或 accountant。
- `scripts/gen_nltcs_queries.py` 根据原始数据计数和相关性选择查询；正式 DP 时必须
  固定公开 workload，或把数据依赖选择计入预算。
- 独立噪声会造成二阶边缘互相不一致。示例中计数噪声标准差仅 1，IPF 200 次仍
  未收敛，最大残差约 4.4 条；标准差 20 时约 89.8 条。需要统一一致性投影和
  `converged=False` 的明确处理策略。
- 被省略的 `attr_9=1, attr_10=0` 精确计数 7 可由公开 N 与其他三个精确格子反推。
  这不是 IPF 新增泄露，而是说明低计数阈值不能当隐私机制。

### 优先级 C：剩余规模瓶颈

- geometric 全对全路径在 nltcs 上仍有约 9 GiB GPU 峰值，核心是 `N×N` 距离与
  概率临时量。
- 成品通常只有约 2,721 个唯一状态；若按唯一状态和重复次数精确合并候选，矩阵
  理论上可缩小约 35 倍。实现时必须正确处理 `exclude_self` 下本状态 multiplicity
  从 `m` 变成 `m-1`，并使用带权分位数保持 geometric 归一化等价。
- 最大熵初始化的 `max_states` 只限制状态数。100 万状态、20 个二值属性时核心数组
  理论小计已超过 816 MiB；后续应增加“状态数×属性对数”的工作量/内存护栏。

## 5. 已知一致性与可复现问题

- `scripts/gen_nltcs_queries.py` 声称生成 1000 个查询、含 32 个 single、双属性阈值
  50；实际 `configs/nltcs/measured_1000query.json` 是 1001 个、无 single，并包含
  计数 43 的双属性查询。生成器无法复现当前正式 workload。
- 初始即收敛时，`rounds_run/loss_history/alpha_history` 包含终止检查轮，但 donor、
  accept 和 proposal 历史为空；这些“每轮历史”的长度语义不完全一致。
- `n_records=1` 且 `exclude_self=True` 时没有合法 donor，概率归一化会产生 NaN。
- 缩步重试会复用 donor，但每次重新抽参与、复制和变异掩码；它是“更小 rho 的新
  随机提案”，不是对同一个失败掩码做严格缩步。相关叙述应保持准确。
- 三种子配对 t 检验样本量很小，现有 p 值应视为探索性证据而非最终统计结论。

这些问题不是 PR #2 引入的。不要把它们混入纯性能 PR，应分别立项。

## 6. 推荐的下一步顺序

1. 等待并处理 PR #2 的人工 review；不要在该 PR 混入方法变化。
2. PR #2 合并后，从最新上游 master 更新本交接文档分支并确认无冲突。
3. 单独开展“低方差最大熵离散化”方法 PR，先保留旧抽样作为默认或显式开关。
4. 修复 workload 生成器与正式 JSON 的不可复现问题。
5. 在声称 DP 前，先完成威胁模型、测量机制、accountant 和一致性投影设计。
6. 之后再研究唯一状态压缩或分块抽样，解决剩余 `N×N` 显存瓶颈。

## 7. 常用命令

```bash
git status --short --branch
git fetch origin master
.conda/gh-cli/bin/gh pr list --repo Chuhan722/table-diffusion
PYTHONPATH=src conda run -n qdte python -m pytest tests/test_evolution.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n gsd python -m pytest -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src conda run -n de \
  python scripts/run_pairwise_maxent_nltcs.py --seeds 0 1 2 --rounds 600
```

正式长实验前先运行 `nvidia-smi`，确认卡号、显存和其他进程。不要默认文档中的
环境路径永远有效。

## 8. 本文档分支的验证

- `AGENTS.md`、`HANDOFF.md` 与 `CLAUDE.md` 的标题层级、Markdown 表格、本地文件
  引用和行尾空白检查通过。
- 文档中给出的 `qdte`/`gsd` conda 命令已实际验证可执行。
- 当前分支只改文档；基于 `origin/master` 的完整测试为 **277 passed**。
- PR #2 的性能分支另有 6 个新增回归测试，完整测试为 **283 passed**；两者不是
  同一 commit，测试数不能混用。
- GitHub 实时复核：PR #1 已合并；PR #2 当前 Ready、CLEAN、MERGEABLE；本文档
  对应 PR #3 已创建。

## 9. 交接检查清单

下一位协作者开始时：

- 阅读 `AGENTS.md`、本文件和 `PROJECT_STATUS.md`。
- 查询 PR #2 的实时状态与 review 意见。
- 确认当前分支和工作树，不要在错误分支修改。
- 若修改算法，先写 baseline 和评价计划；若修改性能，先准备旧新等价对拍。
- 任务结束时更新本文件的日期、分支、PR、测试、已知问题和下一步。

当前没有依赖聊天记录才能理解的未提交代码；以 `git status` 的实时输出为准。
