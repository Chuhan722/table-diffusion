# 项目进度

## 当前阶段
环境搭建完成，准备进入算法实现前的阶段

## 已完成
- 创建 Git 仓库
- 项目基础目录结构（按用户清单的简化结构，已确认保留，不必与文档逐字对齐）
- 收到设计文档，放于 docs/：
  - 表格扩散演化生成器_完整方案.pdf（28 页，方法与公式，作参考）
  - 扩散演化生成器_从零实现与实验计划.pdf（38 页，阶段 0–20 执行手册，作参考）
  - temp.md（适应度设计讨论稿，已定为适应度设计的准绳）
- 项目专用 Python 3.11 环境已创建（prefix 环境，位于 ./.conda，Python 3.11.15；.conda/ 已加入 .gitignore）
- 最小依赖已安装：正式依赖 numpy(2.4.6)，开发依赖 pytest(9.1.1)；项目以 editable 方式安装（pip install -e ".[dev]"）
- Python 包可以正常导入：import table_diffevo 与 import numpy 均成功
- 测试框架已可正常运行：tests/test_environment.py 冒烟测试通过
  （conda run -p ./.conda python -m pytest -q → 1 passed）
- 测试数据已到位（data/ 目录，已加入 .gitignore，符合铁律 6）：
  - test_300x10.csv（300 条记录 × 10 属性，19KB）
  - attribute_value_meanings.csv（属性值含义说明，1.5KB）
  - 数据属性：age, education, employment, income, marital, children, housing, vehicle, health, region
- 查询定义已完成（configs/measured_50query.json，50 个查询，17KB）：
  - 包含 single/double/triple 三类查询，涵盖 ==、>=、between 三种算子
  - 每个查询带有在原数据上的真实计数（当前作为无噪目标）
- 随机种子工具已实现并通过测试（src/table_diffevo/utils.py + tests/test_utils.py）：
  - set_seed(seed) 固定全局随机状态，确保实验可复现
  - 6 个测试全部通过，验证了相同种子 → 相同结果
- 查询评价器已实现并通过验证（src/table_diffevo/queries.py + tests/test_queries.py）：
  - evaluate_table(df, queries) 在给定表上评价所有查询，返回计数向量
  - 支持 ==、>=、between 三种算子，支持单条件和多条件（AND）查询
  - 在原数据上验证通过：50 个查询的计算结果与预期完全一致（9/9 测试通过）
  - 符合铁律 6：评价器不绑定原数据，可用于评价合成表
- 残差计算已实现并通过测试（src/table_diffevo/objective.py + tests/test_objective.py）：
  - compute_residual(target, current, n_records, sigma, kappa) 计算比例残差 ε_j
  - 无噪声阶段行为 = (y - q) / N；保留 σ/κ 噪声容忍区接口，为 DP 阶段铺路
  - 方向语义：偏低为正、偏高为负、达标为零；残差落在 [-1, 1]
  - 11 个测试全部通过，含与查询评价器的集成测试
- 适应度计算已实现并通过测试（fitness.py + 重构 queries.py）：
  - queries.py 新增 eval_query_mask(df, query) 返回单个查询的布尔掩码
  - evaluate_table 改为内部调用 eval_query_mask，逻辑统一
  - fitness.py 实现 compute_fitness，采用纯方向适应度公式（temp.md）
  - 逐查询累加策略：内存 O(N) 与查询数无关，支持几万查询的大规模场景
  - 9 个适应度测试通过，含 temp.md 四状态例子验证（00/01/10/11 方向正确）
- 属性 schema 和距离计算已实现（schema.py + distance.py）：
  - configs/schema.yaml 从 attribute_value_meanings.csv 自动生成（公开 schema 信息）
  - age 范围用领域常识 18-100（遵守严格 DP）
  - schema.py 提供 load_schema，支持属性块定义和查询
  - distance.py 实现归一化 Hamming 距离：age 数值块 + 9 个类别块，等权重
  - 接口：pairwise_block_distance(rows, donor_rows, schema) → (N, M) 距离矩阵
  - 支持全对全（玩具）和小池子（大规模）两种场景
  - 10 个测试通过，含对称性、自距离为0、真实数据集成
  - 当前全套测试：49 passed（新增 10 个测试）

## 文档要点（供后续参考，暂不实现）
- 六条铁律：主线只做扩散演化生成器；每条记录每轮只产生一个下一状态；一轮内所有记录用同一份旧残差；先用 NumPy + 小玩具验证；每个随机实验固定种子；运行期不读真实私有答案。
- 核心流程：固定 S_t → 算一次 residual_t → 用它算全部记录适应度 → 全表同步生成 S_{t+1} → 重算残差。
- 个体适应度（附录 A）：directional 项 e^T W (a(z) - ā) 减去 1/2 ||a(z)-ā||²_W。temp.md 建议只保留方向项。
- 阶段 0：先跑通官方 diffusion-evolution，理解原方法，再写表格代码。

## 已确定的设计决策
- 两份 PDF 计划文档只作参考，不必与其完全一致；现有简化目录结构保留。
- 适应度设计以 temp.md 为准：主适应度只保留残差方向项 e^T W (a(z)-ā)，
  删除二次步幅项；防过冲交给更新率/变异率与整代损失检查。
- 其余实现细节（模块拆分、命名等）到对应阶段再逐步讨论确定。

## 当前未完成
- 尚未实现任何算法
- 后续如需 pandas/scipy/matplotlib/pyyaml 等依赖，到对应阶段再按需加入（当前仅 numpy + pytest）

## 下一步
1. 阶段 0：跑通官方 diffusion-evolution 示例并写理解笔记，再进入表格代码。
2. 或按需先实现最基础模块（如查询评价器），到时再讨论。
