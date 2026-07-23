# 项目进度

## 当前阶段
**核心算法第一版已实现并验证可行，已加可选 GPU 加速并在 nltcs 大数据上跑通**

最简完整闭环已跑通：从随机初始表 S_0 经过扩散演化主循环降低查询残差损失，
玩具数据（300×10）测试 loss 降低约 79%（28932→6102），接受率 60%，方向正确。
nltcs 大数据（16181 条 × 1000 查询）100 轮实跑：loss 9.23e9 → 7.83e8（降约 91.5%），
未提前停止，方向正确。当前版本使用固定参数，尚未实现衰减调度、重试机制、诊断可视化等增强功能。

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
- 参考记录抽样已实现（sampling.py）：
  - 依据"抽样分数+抽样概率.pdf"实现 logit 和 softmax
  - compute_sampling_probs：ℓ_ik = β·F(z_k) − d²/(2h²)，按行 softmax → (N,K) 概率矩阵
  - sample_donors：每行按 Categorical 抽一个 donor 索引，固定种子可复现
  - β_t（选择强度）和 h_t（邻域尺度）作为显式参数，初值占位 β=1.0、h=0.8
  - 不对适应度做 /std 标准化（与 temp.md 第六节"不要除以 std"一致）
  - 相近程度采用高斯核 exp(−d²/2h²)，距离来自 distance.py
  - 允许记录抽到自己（玩具阶段全对全，保持不变是合法一步）
  - 23 个测试通过，含边界情况（β=0、h大/小、适应度/距离均匀）和复现性验证
- 向参考记录靠近一步已实现（update.py）：
  - 依据完整方案第 7 节，evolve_step 全表同步生成 S_{t+1}
  - 7.2 记录参与：U_i ~ Bernoulli(ρ_t)，ρ=0 全不变
  - 7.3 属性块复制：与参考不同的块以概率 η_t 复制，相同的块保持
  - 7.4 变异：参与记录以概率 μ_t 变异一个块，值从 schema 合法值均匀抽样
  - ρ/η/μ 作为显式入参（占位 ρ=0.1、η=0.5、μ=0.01），衰减调度留给主循环
  - 玩具阶段简化：变异用均匀分布；暂不做合法性检查与回退（7.5）
  - donors 已按行对齐（取 donor 的逻辑在上游），本函数职责单一
  - 15 个测试通过，含 ρ=0/1、η=0/1、变异合法性、复现性、真实数据集成
- 合成表初始化已实现（generator.py）：
  - init_synthetic_table(n_records, schema, rng) 生成起点 S_0
  - 纯随机：每格从 schema 合法域均匀抽样（类别值集合 / 数值范围整数含端点）
  - 与源数据一致：记录条数 N、列名列序、每列类型
  - 不复刻源数据实际取值范围（只需落在 schema 合法域，符合严格 DP）
  - 抽样口径与 update.py 的 _sample_legal_value 一致
  - 12 个测试通过，含结构、合法性、端点可达、复现性、下游查询可用性
- 监控损失已实现（objective.py 新增 compute_loss）：
  - E(S) = ½ Σ w_j·[max(|y−q|−κσ, 0)]²，用计数残差（非比例残差）
  - 无噪声阶段（κ=0、w=1）简化为 ½·Σ(y−q)²
  - 越小越好，E=0 表示全部查询达标
  - 9 个测试通过（基本值、达标为0、权重、噪声容忍、排序一致性）
- 扩散演化主循环已实现（evolution.py）：
  - run_evolution 串起完整一轮：评价→残差→适应度→距离→抽donor→靠近一步→整代检查
  - 只接收 target（目标计数）不接收源数据，守铁律 6
  - 整代检查：loss(proposal) ≤ loss(S)+tol 接受，否则保持原表（第一版不重试）
  - 参数 β/h/ρ/η/μ 用固定值（第一版不衰减），T 默认 100
  - 终止条件：残差全 0（达标提前停）或达到 T
  - best_S 保底 + 诊断（loss_history/best_loss/rounds_run/stopped_early/accept_history）
  - 12 个测试通过，含复现性、loss 单调不增、真实数据端到端
  - 真实数据实跑：loss 28932 → 6102（降约 79%），接受率 60%，方向正确
- 结果落盘已实现（io.py + scripts/run.py）：
  - save_run(best_S, diagnostics)：在 outputs/ 下新建 YYYY-MM-DD_HHMM_N 文件夹
    存 best_synthetic.csv（最优合成表）+ diagnostics.json（全部诊断）
  - 重名加数字后缀从 0 起递增；numpy 类型可序列化
  - 主循环保持只算不写，落盘由独立函数负责
  - scripts/run.py：一键跑演化+落盘入口，参数写死在顶部常量（调参改这里）
  - outputs/ 已在 .gitignore（结果不进 git，与 data/ 同类）
  - 9 个测试通过（文件结构、内容一致性、后缀递增、真实运行）
  - 当前全套测试：128 passed（新增 30 个测试）
- 可选 GPU 加速已实现（distance.py + evolution.py + utils.py + scripts/run.py）：
  - pairwise_block_distance 增加 device 参数，按 'numpy'/'cuda'/'cpu' 分派实现
  - _pairwise_distance_numpy（原实现，默认，兼容性保证）与 _pairwise_distance_torch（GPU）并存
  - 设计原则：所有随机操作仍留在 NumPy，只有确定性的距离计算可选上 GPU
    → 同种子 + 同 device 结果一致，可复现性不受影响（用户核心诉求）
  - utils.set_seed 同步固定 torch 种子（未装 torch 时静默跳过，不影响 NumPy 功能）
  - run.py 顶部新增 DEVICE 常量（'cuda'/'numpy'/'cpu'）一处切换
  - 新增 tests/test_distance_gpu.py（7 个测试：torch CPU/CUDA 与 numpy 一致性、
    自距离为0、对称性、范围、device 参数校验），全套 135 passed
  - 环境：torch 2.13.0+cu130，硬件 4× RTX A6000
  - nltcs 100 轮实跑观察：GPU 利用率仅约 24%，瓶颈已转移到仍在 CPU/NumPy 上的
    环节（1000 查询 × 16181 记录的计数评估、fitness、采样、更新），故加速有限
    （real 23m52s）。距离计算已非瓶颈；进一步提速需评估把查询评估也搬 GPU（待讨论）
- 可选 GPU 采样加速已实现（sampling.py + distance.py + evolution.py，2026-07-23）：
  - 依据单轮分段计时（见下），真瓶颈是采样 softmax(48%)+donor抽样(16.6%)≈65%，
    此前只有距离(6%)上了 GPU。本次把采样也搬 GPU。
  - distance.py 加 return_tensor 参数：torch 路径可让距离留在显存不搬回 CPU
    （默认 False=原行为，numpy 路径始终返回 array）
  - sampling.py：compute_sampling_probs / sample_donors 各加 device 参数与
    _*_torch 实现。numpy 路径原样不动；torch 路径 softmax/cumsum 在设备上算。
  - **可复现关键**：donor 抽样的随机数仍用 numpy rng.uniform（与 numpy 路径消耗
    相同随机状态），GPU 只做确定性的 cumsum+比较，只回传 N 个索引。
    → 同种子同 device 可复现；(u<cumprobs).argmax 的 torch 语义与 numpy 一致（已测）
  - evolution.py：device 为 cuda/cpu 时距离→采样全程留显存，省掉 GPU→CPU 搬运；
    numpy 时原路径。切换仍靠 run.py 的 DEVICE 常量，无新增参数。
  - 新增 tests/test_sampling_gpu.py（22 个：torch↔numpy 概率数值接近、同种子抽样
    索引一致、torch 自身可复现、端到端可复现、numpy 路径回归），全套 157 passed
  - **nltcs 实测**：单轮 49s → 5.2s（≈9.4×），loss 曲线与 numpy 路径逐位一致
    （9.2290e9→8.6095e9→8.0076e9→7.4143e9→6.8412e9，best_loss 均 6.322e9），
    演化行为不变。注：float32(cuda) vs float64(numpy) 有极小数值差，保证的是
    "cuda 自身同种子可复现"，非"cuda 与 numpy 逐比特相同"（同距离上 GPU 之性质）。

## 实验观察：nltcs 100 轮收敛分析（2026-07-22）

**现象：** nltcs（16181 条 × 1000 查询，固定参数 β=1/h=0.8/ρ=0.1/η=0.5/μ=0.01）
100 轮实跑，loss 9.23e9 → 7.83e8（降 91.5%），但**第 1-19 轮轮轮接受快速下降，
第 20-100 轮提案全部被拒、loss 卡在 7.83e8 不动**。

**根因（已用停滞表实验证实，非 bug）：步长太大 + 贪心接受 → 卡在局部最优。**
- 接受规则是纯贪心爬山（evolution.py：proposal_loss ≤ loss+tol 才接受，一步都不许退）
- ρ=0.1 时每轮改动约 1600 条记录，同时扰动上千查询；接近谷底时过冲损失 > 靠近收益，
  净变化恒为正 → 必被拒
- 前 19 轮离最优远、满地下坡，大步怎么走都往下，所以飞快收敛；
  步长过大的问题只在接近谷底时暴露（正好是第 19-20 轮转折点）

**停滞表上的步长扫描实验（从 best_synthetic.csv 出发，同一 fitness/距离/概率，只变 ρ）：**

| ρ | 每轮改动记录数 | 3 次随机提案接受次数 |
|------|-----------|------------|
| 0.1（当前值） | ~1600 | 0/3（delta 恒为 +2~4e7） |
| 0.05 | ~800 | 2/3 |
| 0.01 | ~155 | 2/3 |
| 0.005 | ~75 | 3/3（每次都降几百万） |
| 0.001 | ~13 | 2/3 |

**结论：** 停滞表并非真正最优，仍有下降空间；ρ 从 0.1 降到 0.005 后接受率 0→100%。
这直接印证了"参数衰减调度"的必要性（前期大步快降、后期小步精收），
与设计文档规划一致。另外第 20-100 轮明知被拒仍做全量距离+softmax+查询评价，
空转 81 轮——衰减调度或 patience 早停可同时解决"卡住"和"空转"。

## 实验观察：单轮分段计时（2026-07-23，纠正瓶颈判断）

**动机：** GPU 利用率仅 0-24%，需查清整轮时间到底花在哪，再定优化方向。
用临时脚本对 nltcs 稳态单轮（16181 行 × 1000 查询，device=cuda）分段计时，只测量不改主代码。

**稳态单轮 ≈ 12.8 秒，分布：**

| 环节 | 耗时 | 占比 | 在哪跑 |
|------|------|------|--------|
| 采样概率 softmax | 6155 ms | **48.0%** | CPU |
| donor 抽样（cumsum+searchsorted） | 2135 ms | **16.6%** | CPU |
| fitness | 1223 ms | 9.5% | CPU(pandas) |
| 查询评价(提案) | 1199 ms | 9.3% | CPU(pandas) |
| 查询评价(当前表) | 1195 ms | 9.3% | CPU(pandas) |
| 距离 | 784 ms | 6.1% | GPU |
| 更新 evolve_step | 11 ms | 0.1% | CPU |

**关键结论（纠正此前"距离是瓶颈"的直觉判断）：**
1. **真凶是采样，不是距离也不是查询评价**：softmax(48%) + donor抽样(16.6%) ≈ 65%，
   全在 CPU；两步都在处理 (N,N)=2.6亿 元素的大矩阵。
2. **当初唯一上 GPU 的"距离"只占 6%**——优化了最不该优化的地方。而且距离在 GPU
   算完，softmax 立刻把这 2.6亿 元素拉回 CPU，白白多一次 GPU→CPU 搬运。
3. 三处查询评价合计 ≈ 28%；向量化实测可 5× 提速（分块 batch=100 峰值内存仅 78MB，
   结果逐元素一致），但只优化这 28%，非最大头。

**据此重排优化优先级（实测驱动，非猜测）：**
1. ~~**首选：采样(softmax + donor抽样)搬 GPU**（占 65%）~~ **✅ 已完成（2026-07-23）**：
   距离留显存、softmax+cumsum 在设备上算、随机数仍用 numpy 抽只回传索引。
   nltcs 单轮 49s→5.2s（≈9.4×），结果与 numpy 逐位一致。详见上文"已完成"。
2. 次选（待讨论）：向量化查询评价（占 28%，5× 提速，分块控内存）。
3. 距离：已在 GPU，不再动。

## 文档要点（供后续参考，暂不实现）
- 六条铁律：主线只做扩散演化生成器；每条记录每轮只产生一个下一状态；一轮内所有记录用同一份旧残差；先用 NumPy + 小玩具验证；每个随机实验固定种子；运行期不读真实私有答案。
- 核心流程：固定 S_t → 算一次 residual_t → 用它算全部记录适应度 → 全表同步生成 S_{t+1} → 重算残差。
- 个体适应度（附录 A）：directional 项 e^T W (a(z) - ā) 减去 1/2 ||a(z)-ā||²_W。temp.md 建议只保留方向项。
- 阶段 0：先跑通官方 diffusion-evolution，理解原方法，再写表格代码。

## 已确定的设计决策
- 两份 PDF 计划文档只作参考，不必与其完全一致；现有简化目录结构保留。
- 适应度设计以 temp.md 为准：主适应度只保留残差方向项 e^T W (a(z)-ā)，
  删除二次步幅项；防过冲交给更新率/变异率与整代损失检查。
- 抽样分数采用"抽样分数+抽样概率.pdf"定义：β·F − d²/(2h²) 后 softmax，
  不对适应度做 /std 标准化（与完整方案 5.6 冲突，以新文档和 temp.md 为准）。
- β_t、h_t 的具体数值调度（如 h 从 0.8→0.15 线性衰减）留给主循环，
  抽样函数只接收当前轮的标量值，职责单一。
- 其余实现细节（模块拆分、命名等）到对应阶段再逐步讨论确定。

## 当前未完成
- 主循环第一版已跑通，但仍是最简版本，下列增强尚未做：
  - 参数随轮次的衰减调度（h/ρ/η/μ/β 从大到小线性计划）
  - 整代检查失败时的重试 + ρ 缩减（第一版仅"保持原表"）
  - 终止条件 patience / min_change_rate（第一版仅"达标 或 达到 T"）
  - 更丰富的诊断字段（文档第 12 节：fitness 分布、donor 距离、变异次数等）
- 尚未做的更大方向：DP 噪声阶段（σ/κ 接口已预留）、大规模共享参考池（M=512）
- 已在用依赖：numpy、pytest、pandas、pyyaml（后续 scipy/matplotlib 按需再加）

## 下一步（候选，待讨论）
- 全套零件已实现并跑通：generator / queries / objective / fitness / distance /
  sampling / update / evolution。主线的最简闭环已经能把 loss 降下来。
- 可选增强方向（按价值排序，需先讨论再动手）：
  1. 参数线性衰减调度（预期能进一步降低 loss、改善收敛）
  2. 诊断与可视化（画 loss 曲线、观察演化过程）
  3. 整代检查重试 / patience 早停
  4. 调参实验（β/h/ρ/η/μ 的消融）
- 性能方向（nltcs 实跑后暴露，待讨论）：
  1. 把查询计数评估搬到 GPU（当前最大 CPU 瓶颈，需保证结果一致）
  2. run.py 增加逐轮进度输出（现在只有跑完才有摘要，中途看不到进度）
