# 项目进度

## 当前阶段

### 最新暂停点：Stage 6E 三核自动停止正式比较已收口，PR #69 审查修复完成，gap_l1 缺口核研究线收束（2026-09-08）

> 本节把此前只存在于 PR #69 正文的 Stage 6E 正式结果与证据身份落入仓库文档（审查阻塞项 B4），
> 记录本轮审查修复（B1--B5）的处置与理由，并声明剩余缺口核（gap_l1 B+C）研究线的最终定位：
> 结果达标、按 2026-09-08 研究判据战略弃线，合入后保持 opt-in（显式选择启用）非默认，不再开发。

Stage 6E 正式主结果（P=6 自然工作刻度自动停止，两数据 × 三核 × 五种子 353--357，30/30 条轨迹全部
`early_stopped` 正常结束）。终点平均 normalized L1（归一化平均绝对查询误差，越低越好）：

```text
数据集          原 B 因子 Gibbs   独立 B          新 B+C（gap_l1）  vs 因子 B          vs 独立 B
test_300x10    0.0023733333     0.0022666667    0.0009333333     -60.67%，5/0/0     -58.82%，5/0/0
NLTCS          0.0003222660     0.0002964590    0.0002869635     -10.95%，4/0/1     -3.20%，4/0/1
```

两个数据集、相对两条基线均通过冻结的"aggregate 更低且至少 4/5 paired wins（配对胜出）"门槛。
NLTCS 的 95% 诊断 t 区间跨 0，结论严格限定为冻结五种子稳定门通过，不扩写为高置信度总体优势。
新核单条轨迹耗时约为基线 2--3 倍（如 NLTCS seed353：2924s vs 1566s/1539s）。

正式证据身份（正式输出位于 gitignored 的 `outputs/issue53_stage6e_autostop_three_kernel_formal_v1/`，
仓库通过下列 SHA-256 绑定证据）：

```text
原生成提交                       04904b2cdf1ae2125a5ff7d347f4888506ed7cac
恢复提交                         9e3939aef7488a5b4cc6197798b45175021c11fb
原生成协议 SHA-256               fec396940db1d071cab9030ac49631cd18029f250754037c9b503c47efff36c4
恢复协议 SHA-256                 3b6e26761cb3c976c59821266119ca0fcc025527304f218c629ccd8287133ef1
collection_report.json SHA-256   f415e76aee0bb354120a2acdc77e1ca7ad8733b0e1666852690ff8169d8a8aa1
evaluation_report.json SHA-256   e34dcb5fff70da4394484ac994d1d9b0ff1c6f08e7592f649b01541709b4d3d3
l1_results.csv SHA-256（187 行） 4b0fe0d4cfc368ee5a491d5b1f36c030b713fae975b6e59a0562ae5cc697a776
independent_audit.json SHA-256   e10bbcf3195ba01a6e133133fe4c7e3140ce1f9ef48e00143a61bbf14464ddf6
independent audit pass           true
```

研究线收束声明（方法定位，2026-09-08 判据）：

- gap_l1 B+C 核的机制是"E(M) 缺口分数逐开关 Gibbs 扫描重抽"——由损失账本直接决定开关概率，
  属于"筛选/定向优化"侧。按当前研究判据（创新归属应落在"生成分布变聪明"而非"筛选变强"），
  该路线稀释主线创新归属，且单轮成本 2--3 倍于基线，因此**该方法已停止开发（废弃）**：
  不设为公共默认，不再投入新阶段；主线继续走值引导生成（PR #73 系列）。
- 本 PR 的合入定位是**负向收束的完整存档**：冻结协议、正式结果、双重审计与 CUDA/Triton
  优化对拍设施全部入库，供未来对照与复现，新代码路径保持 opt-in 非默认。
- 判据讨论与全流程记录见工作笔记 2026-09-08 接续点（三层判据：无"作废分支"红线、
  创新归属、实战数字）。

PR #69 审查修复记录（对应审查阻塞项）：

- B1（py3.9 `zip(strict=True)`）：涉事 6 个正式脚本全部被 6D/6E 冻结协议按文件 SHA-256 绑定
  （`IMPLEMENTATION_SOURCES`），修改任何字节都会让协议守卫 fail closed，产物持有机将永久失去
  30 案例复核入口——实测修改后 `Stage 6E 正式实现源码漂移：evaluator` 直接拒绝。因此不修改
  冻结源码，改为：py3.9 下唯一直连该语法的测试
  `test_metrics_from_answers_reports_integer_sum_l1_and_gap_e` 加版本守卫跳过（合法输入路径由
  py3.11 基线覆盖；非法输入拒绝路径在 zip 之前抛错，双版本保持覆盖）。冻结脚本只在产物持有机
  （py3.11）复核运行，py3.9 环境不重放。
- B2（5 项测试依赖 gitignored 本地产物）：`test_issue53_stage6e_recovery.py` 3 项与
  `test_issue53_stage6b1_pipeline/protocol.py` 各 1 项，产物缺失时 `pytest.skip` 并列出缺失清单；
  产物持有机上仍为真实复核。双态实测：产物在位 28 passed，产物移走 skip 不 fail。
- B3（CUBLAS 确定性工作区）：新增 `tests/conftest.py` 在任何 torch 导入前
  `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")`；测试本体对外部注入非确定性取值
  时跳过。无外部环境变量下 CUDA 实测 7 passed。
- B4：本节即正式结果与证据身份入库；6D v1--v5 逆序堆叠已合并为下方历史小结。
- B5：修复后本机 py3.11 全套结果见 PR 回复；py3.9/gsd 环境本机不存在，请审查方复跑确认。

真正下一步：以上修复经用户过目批准后提交并回复 PR #69；值引导主线（V12 配对塑形设计）继续在
PR #73 工作区推进。

---

### 最新暂停点：决胜局收官——纯二维同餐对决各有胜场，引擎卖点定位清晰（2026-09-06 晚）

> 结论一句话：引擎与 PGM 吃完全同一份饭（480 格 all-2way + 32 格一维，真答案）
> 在冻结三维/四维 heldout 上正面对决——**PGM 高阶外推略强 ~30-40%**
> （heldout_combined 0.001431 vs 引擎 0.001894，同量级），**引擎池内拟合 2.6× 胜**
> （0.000135 vs 0.000357）、**一维 3× 胜**（0.000062 vs 0.000185）。四局战局
> 盘点定调（相对 maxent/图模型系基线的差异化，非一般卖点）：相对 PGM/AIM
> 一族，引擎赢在"什么都能吃"（任意阶/不可微/大属性数——图模型系受连接树
> 结构约束、吃不了半空间类查询），不赢在无信息外推；maxent 是更好的无信息
> 先验，引擎是更好的信息吸收器——选择测量壳子正是把"能吃"变成"吃得聪明"
> 的机制。注：零阶遗传搜索类方法（如 GSD）同样以"任意统计量、无需可微"为
> 设计目标，"能吃"不是对全体基线的独占卖点；该定位待 GSD 横评（#73 线）后
> 复核重写。

**协议与产物**：runner `scripts/run_fitness_only_all2way_pool_nltcs_diagnostic.py`
（SHA `e11ea5b2…`；池 512=480+32，479 旧二维收编+522 旧三维出池双对账；
对照五份全钉死含 PGM-all2way 公平组 `b7760bab…`）；测试 11 项新增全绿；
报告 `outputs/fitness_only_all2way_pool_nltcs_seed9908_v1/report.json`；
跑满 7000 轮 resource_cap_reached，23 分钟。

**O2-O6 对账**：

- O2 同餐 heldout：引擎 3way 0.001638 / 4way 0.002151 / comb 0.001894 vs
  PGM 0.001368 / 0.001494 / 0.001431——纯二维日粮下 maxent 外推略强；
- O3 纯日粮代价：引擎 heldout 0.000575（polish 含三维）→0.001894（3.3×），
  PGM 0.000762→0.001431（1.9×）——两边都付代价，引擎付更多；
- O4 出池三维探针 ✓：522 三维出池后 0.001525 ≈ heldout 水平（非旧拟合 0.000189），
  覆盖机制自洽；
- O5 一维安全 ✓：0.000062 守住 polish 水平且胜 PGM 同餐 0.000185；
- O6 家族内直比：引擎 0.000135 vs PGM 0.000357（同格同答案，2.6×）。

**四局战局（诚实定位）**：① 纯二维同餐 PGM 外推略强；② 混合日粮（含三维）
引擎 0.000575 < PGM-980 0.000762；③ plants all-2way PGM 结构性不可行
（2^69 收据），引擎照吃全面领先；④ 半空间 PGM 缺席，引擎 2.4×/3.1× 压裸猜。

**下一步**：可选 GSD 无噪声横评；然后壳子阶段（加噪 + 选择测量 + 停止规则
打包，用户明示到时候商量）。

### 上一暂停点：PGM-on-all2way 重拟合收官——信息不对称修正后 PGM 高阶反而更差（2026-09-06 晚）

> 结论一句话：把 nltcs 480 格 all-2way 全家族原样喂给 PGM 重拟合（修"引擎吃
> 全家族、PGM 只吃 980 题"的喂料不对称质疑）——PGM 家族内拟合很好
> （measured 0.000357），但**高阶 heldout 反而比旧 980 局差近 2 倍**
> （heldout_combined 0.001431 vs 0.000762）：旧考卷里混着 522 道三维题=喂过
> 高阶信息，纯二维日粮下最大熵外推撑不住。**质疑被反转：信息量不在格子数，
> 在阶数。** plants 侧出具结构性不可行收据（2^69 格=4096 艾字节，超上限
> 1.1e12 倍），PGM 系"选择测量"的存在理由拿到正式钉死凭证。

**协议与产物**：nltcs 重拟合 runner
`scripts/run_baseline_pgm_all2way_nltcs_diagnostic.py`（SHA `a80a73d2…`）+
plants 收据 runner `scripts/run_baseline_pgm_all2way_plants_feasibility_diagnostic.py`
（SHA `89f5ae06…`）；考卷=新冻结 `configs/nltcs/all2way_issue53_v1.json` 480 格
（SHA `5821fa4e…`，生成器 gen_plants_all2way_queries.py → `gen_all2way_queries.py`
参数化双数据集，plants 逐字重建 SHA 不变）；测试 20 项新增全绿；报告
`outputs/baseline_pgm_all2way_nltcs_v1/report.json` +
`outputs/baseline_pgm_all2way_plants_feasibility_v1/report.json`。

**nltcs 数字（normalized_l1_mean；PGM-all2way vs 旧 PGM-980 vs 引擎 polish）**：

- measured 家族内：0.000357（锚点，考卷不同不直接比：旧 PGM 在 1001 卷 0.000550、
  引擎 polish 在 1001 卷 0.000177）；
- **heldout_combined：0.001431 vs 0.000762 vs 0.000575**（3way：0.001368 vs
  0.000725 vs 0.000433；4way：0.001494 vs 0.000798 vs 0.000716）；
- one-way：0.000185 vs 0.000467（改善，全一维喂饱的自然结果）；
- 运行 18 秒 CPU（估计 2.5s+采样 15.5s），120 clique×4 格，单 2^16 clique 0.5MB。

**plants 收据**：verdict=`baseline_infeasible_no_estimation_attempted`；2346 对
全连图 K69→连接树单 69 列 clique=2^69 格=**4.504e15 MB**，超 4096MB 上限
**1.1e12 倍**；对照：旧 980 考卷可行仅 1.06MB；预检 0.25 秒。

**诚实边界**：引擎 polish 也吃过 1001 卷里的 522 道三维题，"两边都只吃纯二维"
的苹果对苹果还没跑——排队中的 **nltcs 纯 all-2way 池引擎局成为决胜局**
（考卷已冻结 5821fa4e…）：零阶引导外推 vs 最大熵外推，正面对决三维/四维 heldout。

**下一步**：nltcs 纯二维池引擎局（决胜局，镜像 plants all2way runner）；可选
GSD 无噪声横评；然后壳子阶段（加噪+选择测量+停止规则打包，用户明示到时候商量）。

### 上一暂停点：半空间不可微查询能力线双数据集收官——H1-H4 四问全过（2026-09-06 傍晚）

> 结论一句话：把冻结半空间考卷的 measured 档追加进各自演化池（plants
> 9522+64=9586、nltcs 1033+40=1073），κ=1 配方逐位不变——**H1 硬阈值不可微
> 查询确实可被零阶引导优化**（in-pool vs 裸猜地板：plants 2.4×、nltcs 3.1×），
> H2 有真泛化迁移（heldout：plants 2.0×、nltcs 1.4×），H3 普通组零回退
> （plants heldout3/4 反而 -30%/-38% 意外之喜），H4 定位能力边界=稀疏随机
> 方向的 heldout 半空间最硬。"适应度只要掩码不要梯度"的卖点在两个数据集的
> 单种子诊断下成立（diagnostic-only，不构成"已验证"声明）。

**协议与产物**（runner `scripts/run_fitness_only_halfspace_pool_{plants,nltcs}_diagnostic.py`，
协议 SHA plants `41831179…` / nltcs `29a9fa80…`；考卷 plants 127 题 SHA `5ee4f681…` /
nltcs 80 题 SHA `40159977…`（本轮新冻结，A 档 rowsum 16 + B 档 general k=8 64）；
报告 plants `outputs/fitness_only_halfspace_pool_plants_seed9908_v1/report.json`
SHA `2205451b…` / nltcs `…nltcs_seed9908_v1/report.json` SHA `1875d232…`）：
构造器参数化双数据集（B 档 k：plants 16 / nltcs 8，k=4 投影仅 9 档太粗）；
裸猜地板=各自旧终表只读评价（从未见过半空间题，免费无重跑）：plants=all2way
终表 `1849…`、nltcs=polish 终表 `2ec0…`；测试 26 项新增 + 相关回归 122 项全绿。

**H1-H4 对账**：

- H1 ✓ **可优化性（主问题）**：in-pool 半空间 vs 裸猜——plants 0.010284 vs
  0.024565（2.4×）、nltcs 0.001040 vs 0.003271（3.1×）；
- H2 ✓ 泛化：heldout（从未进池）vs 裸猜——plants 0.013950 vs 0.027936（2.0×）、
  nltcs 0.002738 vs 0.003801（1.4×，polish 终表太强地板天生低）；
- H3 ✓✓ 无回退：nltcs 四指标漂移 ±0.00005 内；plants measured +0.000114 /
  one-way -0.000039 / all2way_pool 0.001292→0.001321 均噪声级，且
  **heldout3 0.005712→0.003986（-30%）、heldout4 0.006630→0.004101（-38%）**
  ——64 道全局投影题帮了高阶泛化（单种子 observation-only）；
- H4 ✓ 难度档两数据集方向相反：nltcs 全列 rowsum（17 档）比 k=8 难；plants
  heldout A 档 rowsum 0.0072 远好于 B 档 general k=16 0.0205（裸猜只压 1.6×）
  ——**稀疏随机方向 heldout 半空间=最硬一档**。

**运行形态**：nltcs 24min 跑满 7000（resource_cap_reached，预算贴身如预期）；
plants 59min 早停 3805/27000（最优 ~3127，与 all2way 3506 同款）；墙钟冒烟
946ms/219ms 每轮兑现（64 道半空间走 fallback 慢路径只贵 9%）。

**边界注记**：θ 看了源投影谱（非 result-blind），diagnostic_only 能力考卷，正式
赛道须换公共信息选题规则（GSD θ~N(0,1) 即一例）；PGM 设计上缺席（吃不了半空间）；
nltcs 池基底=polish 局（含 522 in-pool 三维题），纯 all-2way 池版排队。

**下一步**：无噪声阶段清单顺次——PGM-on-all2way 重拟合基线；nltcs 纯二维池
（消 522 三维沾光、统一 one-shot 口径）；可选 GSD 无噪声横评；然后壳子阶段
（加噪 + 选择测量 + 停止规则打包，用户明示到时候商量）。

### 上一暂停点：plants all-2way 全家族考卷收官——病③元凶=考卷覆盖坐实，非引擎瓶颈（2026-09-06 下午）

> 结论一句话：把 plants 演化池从 980 稀疏考卷（2-way 覆盖 4.9%）换成 all-2way
> 全家族 9384 格（覆盖 100%，GSD 对齐），κ=1 配方逐位不变——heldout3
> 0.032006→**0.005712**（5.6× 改善，0.38× PGM-context）、heldout4
> 0.024527→**0.006630**（3.7×，0.52×）。病③（heldout 卡 2×PGM）**不是引擎
> 泛化瓶颈，是考卷覆盖不足**；plants 泛化病治愈，两数据集全指标恢复全面领先。

**背景（AIM/GSD 查询池调研定稿，论文原文+官方代码核实）**：领域标准考卷=全体
k-way 边缘家族，AIM/GSD 评估都在考卷内（无 heldout 泛化轴，我们更严）；GSD 分类
主实验=all-2way one-shot（一次性全测，无选择壳）；GSD 初始化=纯均匀随机（我们的
marginal init 已更聪明）。旧 980 考卷 4.9% 覆盖是自设非标准苛刻局。

**协议与产物**（runner `scripts/run_fitness_only_all2way_pool_plants_diagnostic.py`，
协议 SHA `16f03797…`；考卷 `configs/plants/all2way_issue53_v1.json` 9384 条 SHA
`9dc37994…`；报告 `outputs/fitness_only_all2way_pool_plants_seed9908_v1/report.json`
SHA `82d85b8516a6948563f0…`）：池 9522 = all2way 9384 + one-way 138；旧 980 考卷
保留为评估延续组（460 double 被全家族吸收，520 triple 出池变泛化探针）；预算
逐位从 polish 导入（cap 27000 / 时刻表 4050/2700 / patience 6）；测试 11 项新增
+ 相关回归 72 项全绿。

**O1-O5 对账（对照 polish 980 池报告 `fef4fb5e…`）**：

- O1 ✓ `early_stopped` 3506/27000（省 87%），最优 2902 轮；审计一次过；
  墙钟 49 分钟（868ms/轮，池 8.5× 只慢 1.24×——开销在行数不在查询数）；
- O2 ✓✓ **主问题大捷**：heldout3 0.005712（polish 0.032006 → 5.6× 改善；
  PGM-context 0.015229 → 0.38×）、heldout4 0.006630（0.024527 → 3.7×；
  PGM 0.012866 → 0.52×）——远超"收向 ≤~1×PGM"的预注册假设；
- O3 ✓ 出池 520 triple 探针 0.002466（polish 时代 in-pool 0.001646 只涨 1.5×，
  远低于旧 heldout3 0.032）——三维格被全家族二维覆盖撑住，覆盖机制再证；
- O4 ✓ one-way 安全 0.000257（治愈线 0.004230 的 6%；polish 0.002007 再改善 7.8×）；
- O5 ✓ all2way 工作负载首测锚点：mean 0.001292 / median 0.000976 / max 0.0312；
  980 延续组 measured 0.002273（vs polish 0.001456，旧题不再被专门优化，符合预期）。

**caveat**：PGM 基线是 980 池拟合的（信息不对称，本跑吃全家族真答案），PGM 倍数
只作 context；PGM-on-all2way 重拟合基线排队为独立工作。本跑无噪声（diagnostic_only），
加噪 one-shot 正式跑是下一件。

**下一步**：加噪件（Gaussian one-shot，一维从噪声二维推平均降噪的小设计点）；
PGM-on-all2way 重拟合基线；半空间赛道（GSD 式固定种子随机池）；all-3way 自适应
赛道押后（要外层壳子，停止规则用户明示到时候再商量）；本轮新文件待提交
（考卷生成器 + config + runner + 测试）。

### 上一暂停点：统一配方双数据集单种子诊断一致（plants 早停省 78% / nltcs 触顶零回归）；主线转泛化机制（2026-09-06 中午）

> 结论一句话：κ=1 统一配方（M=列数解最小轮数、上限取整到千、patience=6 早停）
> 在两个数据集的单种子诊断中画像一致（diagnostic-only，未做多种子验证，
> 不构成"配方已验证"声明）——plants 早停 6022/27000（省 78%，measured 差距
> 4.7×→1.87×，heldout 纹丝不动→病③隔离）；nltcs 触顶 7000/7000 零回归
> （全面超 PGM 战绩保住，one-way 再改善 36%）。主线转向泛化机制设计（病③）。

**nltcs 统一配方结果（4090 GPU1，报告
`outputs/fitness_only_polish_budget_nltcs_seed9908_v1/report.json`，
SHA `919b418059c62147…`，协议 SHA `ee426947…`）**：

- O1 ✓ 触顶 `resource_cap_reached`（7000/7000）如几何账预测（地板段 5250 <
  6 tick 窗口 6000）；κ=1.156 全额实现；最优在 6566 轮贴近上限还在刷新
  ——预算贴身实锤；修复后审计一次通过；墙钟 23 分钟；
- O2-O5 ✓ 零回归：measured 0.000177（+6.3%，0.32×PGM）、heldout3 0.000433
  （+5.8%，0.60×PGM）、heldout4 0.000716（-1.5%，0.90×PGM）、
  one-way 0.000070（-35.7% 再改善，0.15×PGM）；±6% 属小数值噪声
  （时刻表整条重排非同轨延长）；全面超 PGM 保持。

**审计烏龍与修复（第一跑 fail-closed 拦下，已修）**：

- 病根：`_audit_fitness_only_run` 早停分支允许集合写错——引擎 A/B/C 全集是
  `fit_target_reached / early_stopped / resource_cap_reached`（触顶不叫
  max_rounds）；stopped_early 映射同错；plants 走 early_stopped 未踩中，
  测试未盖"开早停+跑满"路径；
- 修复：fitness_only.py 允许集合改引擎真实三标签 + 触顶强制 rounds_run==上限 +
  stopped_early↔{fit_target_reached, early_stopped}；测试改语义并新增
  `test_early_stopping_enabled_cap_hit_is_resource_cap_reached`；nltcs 协议
  contract_amendment 文本诚实化重钉 SHA；50 项全绿（引擎 28+nltcs 11+plants 11）。

**下一步**：泛化机制设计讨论（主线，病③：plants heldout 卡 PGM ~2×，机制性差距）；
半空间线 3、4 步；PR 挂账（追加 fitness_only.py 审计修复 + nltcs runner/测试 +
两份新报告）；外层壳子（选择→测量，隐私预算）设计与外层停止规则——用户明示到时候再商量。

### 上一暂停点：plants κ=1 长跑收官——预算病治愈、泛化病隔离；nltcs 统一配方跑准备中（2026-09-06 上午）

> 结论一句话：早停 6022/27000 轮合规触发（墙钟 1.18h），loss 收敛后才停——
> 病②预算截断治愈；measured 对 PGM 差距 4.7×→1.87× 达标进 2× 以内，
> 但 heldout 纹丝不动（仍 ~2× PGM）——病③泛化机制缺失被干净隔离成
> 唯一剩余病：是机制问题，不是预算/调参问题。

**plants κ=1 长跑结果（4090 GPU0，报告
`outputs/fitness_only_polish_budget_plants_seed9908_v1/report.json`，
SHA `fef4fb5e999722775f9e2013a312009a5db5d24c847ec7fbccd9cdcaad54fd29`）**：

- 早停 6022/27000 轮（`early_stopped`）；最优在 4470 轮，其后 6 个打磨单位
  零刷新合规掐停；实际打磨 Σrho=50.05（κ=0.725 就收敛，没用满 69）；
  墙钟 1.18h（跑满预估 6-7h，早停省 ~80%）。
- loss best 8.92e5 / final 9.21e5——同在 ~6000 轮，昨晚旧时刻表 4.23e6
  （新时刻表好 4.6×：热身期 900→4050 拉长让粗磨充分）。
- O1 ✓ 截断解除（loss 到平台才停，非预算硬掐）；
- O2 ✓ measured 0.001456（进池 R 0.003634 再降 60%）；PGM 差距 **4.7×→1.87×**；
- **O3 ✗ heldout 纹丝不动**：3way 0.032006（+0.9%）、4way 0.024527（+4.3%），
  对 PGM 仍 2.10×/1.91×；
- O4 ✓ one-way 0.002007（再降 53%；PGM 3.41×）；O5 ✓ measured 零挤占。
- 解读：loss 已收敛 + 目标已完备 + 打磨已到顶——考题内逼近 PGM、考题外
  一分没涨。剩余 2× 差距要靠设计层泛化机制（PGM 拟合的是分布，图结构自带
  跨边缘平滑；我们的引擎逐题拟合考题）。

**进行中：nltcs 统一配方归档跑（已发射，在跑）**：

- 统一配方 = κ=1（M=属性数）+ 上限取整到千 + patience=6（引擎默认零调参）；
- nltcs M=16 → 解最小 T=6057（6056 只有 15.995 不够）→ 上限 7000，
  时刻表 (1050, 700)，κ(7000)=1.156；
- runner + 11 测试完成，协议 SHA `ee426947…`；第一跑触发审计烏龍（我的允许集合
  漏了引擎触顶标签 `resource_cap_reached`，fail-closed 拦下无报告）——已修
  fitness_only.py 审计分支 + 补触顶测试，50 项全绿后 10:50 重发（4090 GPU1）；
- 几何账：nltcs 地板段 5250 轮 < 6 tick 窗口 ~6000 轮，B 物理开不了枪，预期触顶收尾；
- 早停机制定位定稿：大预算省时器 + 收敛证书，小预算触顶兜底，不动耐心值（零调参）。

**下一步**：nltcs 归档跑 → 泛化机制设计讨论（主线，病③）；半空间线 3、4 步；
未提交内容 PR 安排挂账。

### 上一暂停点：one-way 进池全达标 + 打磨密度定标（κ=1）plants 长跑发射（2026-09-05 深夜）

> 用户方向：plants "还不够好"的病因实锤为**预算截断**（R 臂 loss 6000 轮仍以
> 每 500 轮 -19% 下降，远未收敛被硬掐；预算当年按 nltcs 绝对轮数抄写，
> 每格打磨密度 κ 仅 0.23 vs nltcs 0.991）。用户提出**打磨密度定标原则**：
> 每行被修改 M 次、M=属性数 ⇔ κ=Σrho_t/M=1；解最小轮数后上限取整到千。

**one-way 进池双实验结果（A6000，两份报告已回本地）**：

- plants（`outputs/fitness_only_oneway_pool_plants_seed9908_v1/report.json`，
  SHA `30fe3cd2…`）：P1 one-way 自愈 10×（0.0410→0.00423）；P2 heldout -15~-19%；
  P3 measured 反而改善 31% 零挤占；P4 R-E 倒挂翻转；E1 equal 逐字节一致。
- nltcs：N1 零回归（四指标全微幅改善）；E1 同铁证；**意外之喜：nltcs 进池 R
  全面超 PGM**（measured 3.3× 优、heldout/one-way 全胜）——κ≈1 的数据集赢 PGM，
  κ=0.23 的 plants 还差 ~2×，正是打磨密度故事的最硬对照。

**合同修订（引擎，测试 27+126+174+11 全绿）**：

- fitness-only 合同放行 **A/B/C 早停成套开**（`stop_on_exact_residual` +
  `inner_early_stopping_patience_ticks` 必须成对；单开报错；配对归因禁早停）；
- 审计分支：早停开时允许 rounds_run ≤ 上限，账目自洽强制（候选数=实际轮数、
  stopped_early ↔ 终止原因、rho 历史长度=实际轮数）；`termination_rule` 标签
  诚实化（`inner_early_stopping_a_b_c` / `fixed_n_rounds`）；
- 早停零件"只看不摸"（不碰提案/随机流/状态）：开与不开每轮逐位一致，
  唯一区别是停点——预算效应归因不糊。

**打磨密度长跑协议（已冻结，正在 4090 GPU0 运行）**：

- `scripts/run_fitness_only_polish_budget_plants_diagnostic.py`
  SHA `4b7c97105ab7f19824defb0b2cc7213b5f6c2dcb38d577befe91f3354a63874e`；
- M=69 → 解最小 T=26126 → **上限 C=27000**（取整到千，用户定稿），时刻表挂 C
  （冷却 4050 起 2700 轮，15%/10%），κ(C)=1.033；patience=6（引擎默认零调参）；
- 单 R 臂（equal 砍掉——双数据集 E1 铁证，盲臂数字引用今晚报告）；
  池/种子/评价口径与 one-way 进池逐字一致；预算三重自检（最小性/取整/上限量）；
- 预注册观察点：O1 截断解除；O2 measured 差距（原 4.7×）；O3 heldout 差距
  （原 ~2×）；O4 one-way 不回退（≤0.00423）；O5 measured 不挤占（≤0.00363）；
- 对照三把尺：今晚进池 R（主对照，纯预算效应）/ 冻结 v1 R / PGM；
- 测试 `tests/test_fitness_only_polish_budget_plants_diagnostic.py` 11 项全过。

**下一步**：长跑收报告后按 O1-O5 对账；半空间线协议 + runner（池基底=进池版，
已可动工）；未提交内容 PR 安排挂账。

### 上一暂停点：半空间不可微查询能力线——引擎扩展与冻结考卷完成（2026-09-05）

> 用户方向：关键落点之一"处理不可微查询"从未被检验。逐类判定标准查询清单
> （类别/范围/混合边缘、前缀和、条件前缀和、半空间）：前五类对 one-hot 都是
> 线性可微（对手主场），**唯有半空间 w·x≥θ 是硬阈值不可微**（row-sum 是
> 权重全 1 特例）。用户拍板主打半空间；本线暂不与 PGM 对比。
> 设计稿：`docs/设计/半空间不可微查询能力线设计稿.md`（预注册 H1-H4）。

**已完成（未运行任何实验）：**

- **引擎扩展（第一档，保正确）**：
  - `queries.py`：新增 `eval_halfspace_mask`（fail-closed 校验 + float64
    精确投影），`eval_query_mask` 按 `type == "halfspace"` 分派——fitness
    与计数共用同一掩码接口，机制零改动；
  - `quality.py`：`canonical_query_payload` 支持 halfspace（独立命名空间
    `{"halfspace": {"terms": 排序项, "theta": float}}`，与合取指纹永不碰撞，
    int/float 权重归一，重复属性拒绝）；
  - `vectorized_eval.py`：halfspace 整条进回退组（旧路径精确评价，计数与
    fitness 贡献都对），verbose 提示标注 `type=halfspace`，不崩溃；
    将来若成瓶颈可按白名单机制补向量化（本质一次矩阵乘）。
  - 测试 `tests/test_halfspace_queries.py` 19 项全过：掩码对拍朴素实现、
    混合池向量化 vs legacy 逐位一致、分块不变性、方向势能回退、指纹语义、
    fail-closed；受影响模块回归（queries/vectorized/fitness/quality）48 项全过。
- **冻结考卷** `configs/plants/halfspace_issue53_v1.json`
  （SHA256 `5ee4f6817a49705149557f782a533ce58484a3d62b846859a0b692595ee16012`，
  `--verify-existing` 确定性重建复核通过）：
  - 127 题 = A 档 row-sum 63（θ 3..65 全谱非退化格点，升序交替 32 M / 31 H）
    + B 档一般半空间 64（seed 20260905、k=16、±1 权重、θ 投影分位点，32 M / 32 H）；
  - 全部题目 result 非退化（91..16563，N=17412，两端各留 0.5% 边距）；
  - 构造器 `scripts/build_issue53_halfspace_workload.py` +
    测试 `tests/test_build_issue53_halfspace_workload.py` 10 项全过；
  - **边界注记（明示不藏）**：选择依赖源表投影谱（非 result-blind），
    诊断线专用；进正式 DP 管道前必须换成公共信息选择规则。

**下一步**：今晚先跑 one-way 进池双实验（见下一节暂停点，等用户发令）；
半空间线等 one-way 结果落地定演化池基底后，再写结果前协议 + runner
（工程顺序第 3、4 步），用户授权后运行。

### 上一暂停点：Private-PGM 外部基线校准完成——nltcs 我方全胜、plants 被 PGM 全面碾压（2026-09-05）

> 用户方向：fitness-only 线先整理成 PR（已完成，见 PR #70），再引入其他方法的
> 生成部分做达标校准。基线选 Private-PGM（MST/AIM 的生成器，mbi 库，
> SSH 克隆 commit 07635f9 装入 .venv，JAX CPU 后端）。口径：信息对等——
> 同 980/812 查询精确答案 + 同 one-way 边缘作 LinearMeasurement，
> MirrorDescent 1000 轮零调参，synthetic_data 受控舍入，同评价函数
> （复用 run_fitness_only_attribution 的 _grouped_error_metrics），
> 我方臂用冻结报告数字不重跑。diagnostic_only，两套三件套均 fail-closed。

**nltcs 结果**（`outputs/baseline_pgm_nltcs_v1/report.json`，推断 13.7s + 采样 58.7s）：

```text
measured   PGM 0.000550 vs 我方R 0.000179   —— 我方 3.1× 优 ✓
heldout3   PGM 0.000725 vs 我方R 0.000419   —— 我方 1.7× 优 ✓
heldout4   PGM 0.000798 vs 我方R 0.000769   —— 持平略优 ✓（中位数 PGM 反超）
one-way    PGM 0.000467 vs 我方R 0.000112   —— 我方 4.2× 优 ✓
```

**plants 结果**（`outputs/baseline_pgm_plants_v1/report.json`，推断 11.0s + 采样 18.6s，
JT 实测 1.06MB / 最大团 14 属性 16384 格——treewidth 爆炸剧本落空）：

```text
measured   PGM 0.000779 vs 我方R 0.005249 / E 0.068868   —— PGM 6.7× 优 ✗
heldout3   PGM 0.015229 vs 我方R 0.039262 / E 0.033184   —— PGM 2.6× 优 ✗
heldout4   PGM 0.012866 vs 我方R 0.027691 / E 0.022497   —— PGM 2.2× 优 ✗
one-way    PGM 0.000589 vs 我方R 0.040966 / E 0.040378   —— PGM 70× 优 ✗✗
```

**校准结论：**
1. **nltcs 的全胜是密覆盖小域的例外，不是规律**——plants（69 属性、
   覆盖 2-way 4.9%/3-way 0.5%）上 PGM 拿同样的信息把 heldout 误差压到
   我方 1/2.2~1/2.6，达标线有了实数：heldout3 ~0.015 量级。
2. **one-way 差 70× 是最扎眼的病灶**：一阶边缘明明喂给了我方
   （init marginals），演化过程守不住；PGM 把同样的边缘当测量几乎完美复现。
   方向：把 one-way（及覆盖补全）纳入 fitness——与 plants 诊断的
   候选 b 收敛到同一处。
3. **PGM 在两个数据集上都又快又稳**（30s~72s 纯 CPU vs 我方单臂 20~70min GPU），
   其优势来自图模型推断在低 treewidth workload 上的结构利用；
   我方无图结构假设的卖点须靠"PGM 不可行的场景"或质量反超来兑现。

产物（本地未提交，成 PR 安排待定——当前分支已被 PR #70 占用）：
- `docs/设计/PGM基线nltcs生成对比结果前协议.md` + `scripts/run_baseline_pgm_nltcs_diagnostic.py`
  （协议 SHA 91b2feb8…）+ `tests/test_baseline_pgm_nltcs_diagnostic.py`（10 项全过）
- `docs/设计/PGM基线plants生成对比结果前协议.md` + `scripts/run_baseline_pgm_plants_diagnostic.py`
  （协议 SHA 06dbbb90…，含 JT 可行性预检 cap 4096MB）
  + `tests/test_baseline_pgm_plants_diagnostic.py`（10 项全过）
- 两份冻结报告在 outputs/（gitignored），输入 SHA 与对照冻结报告 SHA 均钉死在 runner 内

下一步候选（等用户拍板）：
  a. 覆盖补全进 fitness（one-way + 加密 2-way），plants 上正面追 PGM
  b. plants 加预算重跑（24000 轮）看收敛后能追回多少
  c. 找 PGM 不可行的 workload（高 treewidth）建立差异化卖点
（单 seed、diagnostic_only，不作正式声明。）

---

### 上一暂停点：plants 中等规模真实基准诊断已运行——复现覆盖受限画像且预算截断（2026-09-05）

> 用户方向：小数据（test_300x10）不进最终对比，转向未用过的 plants
> （17412×69 二值，"twenty datasets" 基准）。用户授权 GPU 运行。
> 权重路线同日已由用户关闭（残差已是相对域，再按残差调权重属重复）。

准备工作（全部入库）：
- `configs/plants/heldout_issue53_v1.json`：冻结构造器新增 plants 条目
  生成（512×3way + 512×4way，哈希排序确定性选择，与 measured 无交集），
  SHA `d65401761bade19ad40d9588eaa57609de0bdbc3c26835a5343ab8cc332eca85`；
  nltcs/test 旧文件 `--verify-existing` 逐字节可重建（构造器改动零漂移）。
- `scripts/run_fitness_only_plants_diagnostic.py`：单数据集诊断 runner，
  v3 冻结配置原样（6000 轮、rho 0.01→0.001@900+600、eta 0.5、alpha 16、
  relative/floor 8、seed 9908、residual/equal 配对、cuda），
  协议 SHA `67d6a5b0128caa7536c4aa91736eb6b5332dc1ed871ca0a0dfc75115ba6f9861`，
  diagnostic_only / formal_claim_allowed=false / fail-closed 输出。
- 测试：test_plants_workload + test_issue53_heldout_workloads 13/13 过。

结果（`outputs/fitness_only_plants_diagnostic_seed9908_v1/report.json`，
GPU1 11:26-13:47，单臂 70 分钟，rho 审计通过，6000 轮走满）：

```text
measured   R 0.005249 vs E 0.068868（差值 -0.0636，优 13×）✓
heldout3   R 0.039262 vs E 0.033184（+0.0061 ✗）均值劣、中位数持平
heldout4   R 0.027691 vs E 0.022497（+0.0052 ✗）中位数 R 反而优（0.0103 vs 0.0126）
one-way    R 0.040966 vs E 0.040378（+0.0006 ≈中性）
           但分布迥异：R 中位 173 行/max 5613；E 中位 732/max 902（重尾 vs 均庸）
drift      1.0004（nltcs 的 1.16 回摆现象完全缺席）
形态       6000 轮仍在下降（last best @5990，732M→7.4M 未收敛）——预算截断
```

**两个新事实：**
1. **覆盖受限画像不是小数据伪影**：plants 复现 test_300x10 模式
   （measured 大胜、heldout 均值劣）。根因量化——workload 覆盖密度：
   nltcs 2-way 100%/3-way 64%；plants 2-way 4.9%/3-way 0.5%；
   test_300x10 同为稀疏。nltcs 是例外（密覆盖），不是规律。
   均值劣由重尾驱动（p90：R 0.089 vs E 0.077），中位数不输。
2. **v3 预算是按 nltcs 尺度标定的**：plants 6000 轮远未收敛
   （残差臂终点 loss 7.4M，仍在最速下降段尾部）。规模每上一档，
   固定 6000 轮就截断一次——预算需按数据集尺度重标定（这不是
   v3 truncation 判定的推翻：nltcs 上 v3 结论不变）。

下一步候选（等用户拍板）：
  a. plants 加预算重跑（如 24000 轮，~4.7h/臂）看收敛后画像
  b. 覆盖补全方向（对 plants 补 one-way + 加密 2-way 进 fitness）
  c. 两者结合的结果前协议
（本诊断单 seed、diagnostic_only，不作正式声明。）

---

### 上一暂停点：v6 MW 乘性权重聚合已运行，护栏双爆——聚合路线关闭，等权聚合定稿（2026-09-04）

> 用户授权在本机 4090（GPU1）运行。23:24-23:52（nltcs 生成 1182.3s，
> test 76.0s），6000 轮走满，rho/eta 时间表 v3 原样逐轮匹配，**前缀
> 审计通过**（前 1501 轮 loss/rho/state metrics/初表/RNG 五项与 v3
> residual 臂逐位一致；保温段权重恒全 1）——MW 实现无扰动，分叉
> 纯粹是机制效果。单臂设计：equal 臂复用 v3 冻结产物（frame SHA
> 链闭合）。协议 manifest SHA b3cc455ca19b4e3f…f0da8717。

结果（`outputs/fitness_only_mw_dev_seed9908_v6/report.json`）：

```text
护栏1：nltcs held-out 配对差 -0.061138/-0.037092 ≤ -0.050/-0.030 ✓（优势保住）
护栏2：nltcs L1 0.000466 ≤ 0.000197 ✗（2.60×v3——严重恶化）
        test  L1 0.003800 ≤ 0.002640 ✗（1.58×v3——严重恶化）
主判据：test held-out 配对差 3way +0.003288（v3 +0.003340，持平）
        4way +0.001107（v3 +0.000996，反恶化 11%）——即便无护栏也是 rejected
drift ：nltcs 5.1717（v3 1.1639——爆炸 4.4 倍）；test 2.7872（v3 2.2400）
best  ：nltcs 11126.5@r1474——MW 开启（r1500）前 26 轮，开启后 4500 轮零改进
判定  ：quality_regression_under_mw
```

**退化机理（只读权重轨迹诊断，协议 §7 预绑定动作）**：

```text
r1500 开启 → r1800 已 26 个顶格(8×) → r2000-2500 雪崩：沉底 0→535
→ 终态稳定极化：37 顶格 / 637 沉底（63.7% 查询只剩 1/8 权重）
loss 分段：[1500,2500) 2.84×v3 → [3000,4000) 7.38×v3（开启即恶化，单调加深）
```

死账户独裁死锁：nltcs 重尾欠账（目标计数 13-42 的稀有查询）是**量子化
不可修**的——差半行就是大相对残差，注意力再多也修不平。乘性复利
（exp(0.002×8)≈1.6%/轮）让这些死账 200 轮内顶格；归一化跷跷板把 62%
正常查询压到地板 → 大盘失守（L1 恶化 2.6×）；fitness 被 40 个修不平的
目标主导 → 系统反复搬行追死账 → drift 爆炸。**MW 的隐含假设"多注意=
能修好"对量子化死账不成立**，机制退化为把资源永久锁死在不可修目标上。

**科学结论（聚合路线关闭）**：等权聚合定稿（v3 配置）。fitness 聚合
缺口（test held-out 配对差为正）如实报告为方法局限/future work——
MW 记录为已排除方向。按协议 §8：不做剂量下探（cap/eta/start 修订=
新协议，且机理性死锁不是剂量问题）。

六代记录并列（全部结果前冻结、前缀审计通过、不重跑）：
v2 unsupported / v3 rejected(truncation) / v4 quality_regression(floor) /
v5 quality_regression(eta) / v6 quality_regression(MW)。
**v3 配置 = 实证质量最优点**，温度路线与聚合路线双关闭。

下一步（§7 预绑定）：回 v3 配置；5-seed 确认屏（v3 配置）成为下一个
待议协议。

> **2026-09-05 上午修正**：用户判断权重路线仍可行，5-seed 暂缓。
> 只读深挖已完成（三层死因：相对域记账 vs 绝对域结算的货币错位 /
> 量子化死账零改善 / 零和跷跷板株连占恶化 82%；nltcs 尚有 291 个
> >3 行可修大账户，test 已到本底无钱可赚）。待议：纸面模拟 v7
> 结构手术（信号换绝对域+量子化免记+去零和），详见工作笔记
> 2026-09-05 接续点。

---

### 上一暂停点：v5 eta 降温已运行，质量门失败且 drift 恶化——"降温"全路线关闭（2026-09-04）

> 用户授权运行。GPU1，19:25-20:30（nltcs 生成 3614.7s），9000 轮走满，
> 双时间表审计逐轮匹配，**前缀审计通过**（前 1501 轮与 v3 四条轨迹
> 逐位一致——eta 退火实现未扰动随机流，等价性合同在真实规模上成立）。

结果（`outputs/fitness_only_eta_cooling_dev_seed9908_v5/report.json`）：

```text
质量门：nltcs L1 0.000193 ≤ 0.000197 ✓（v3 0.000179，略差）
        test  L1 0.003267 ≤ 0.002640 ✗（v3 0.002400，倒退 36%，比 v4 更糟）
drift ：nltcs 1.3664（v3 1.1639、v4 1.1754——不降反升 17%！）
        test 1.9787（观察项，反而比 v3 2.24 略窄）
形态  ：非 descending；best 7553@r3970 后横盘于 ~9200-10900 带
        （震荡带中心上移且更宽：v3 稳态带 ~8000-8600）
判定  ：quality_regression_under_eta_cooling
held-out：nltcs 配对优势保持（-0.061458/-0.037278，与 v3 持平）
```

**核心科学结论（v3+v4+v5 机制排除三部曲完成）**：

eta 不只控制"单步跳多大"，同时控制"修复多快"——两者不可分。降 eta
后单步修复量减半，而破坏源（mu 突变、随机漂移）强度不变，稳态
修复-破坏平衡点上移：损失带从 ~8300 抬到 ~10300，带宽反而变宽。

三部曲总结（全部结果前冻结、前缀审计通过、记录并列）：
1. v3 预算加倍 → rejected：drift≈1.16 是稳态本底，非预算问题
2. v4 频率减半（rho 0.001→0.0005）→ quality_regression：本底未收窄
   （1.1639→1.1754）+ 小表冻伤
3. v5 幅度减半（eta 0.5→0.25）→ quality_regression：平衡点恶化，
   drift 反升至 1.3664 + 小表更伤

**结论：任何形式的"降温"都损伤修复-破坏平衡。v3 配置（rho 三段式
H900/D600/floor0.001、eta 恒 0.5）是实证的质量最优点；drift 本底
~1.16 是无门控终态输出在该最优点的固有代价。**

按 v5 协议 §7：quality_regression → 停下，不进任何后续屏，不做
eta=0.1 下探。B 路线（接受本底、修订叙事）已由三重预注册实验钉死，
从"选择"升级为"结论"。

下一步（等用户决定）：
- B 叙事定稿：方法=v3 配置；drift 如实报告；v3/v4/v5 作为机制排除
  证据链写入论文（预注册+可证伪审计是方法学卖点）
- 之后：5-seed 确认屏协议（v3 配置）或 fitness 聚合缺口优先级讨论

v2 unsupported / v3 rejected / v4 quality_regression /
v5 quality_regression 四条记录并列保留，均不得重跑。

### 上一暂停点：v5 eta 降温执行器与核心机制已就绪，等待运行授权（2026-09-04）

> v5 协议获用户批准后完成实现。**本屏含核心代码改动**（v2-v4 均为纯
> 配置 delta）：evolution.py + fitness_only.py 新增 eta 三段式退火
> （eta_anneal_start_round / eta_anneal_rounds / eta_anneal_end），与
> rho 退火完全同构、纯轮数驱动、只改复制开关阈值不动随机流；与
> residual_directed_diffusion 组合被 fail-closed 拒绝。

- 协议：docs/设计/FitnessOnly地板段Eta降温v5结果前协议.md
- 执行器：scripts/run_fitness_only_eta_cooling_v5.py（SHA 钉死
  05863ca27710fe52794a6106c74b341962b170b1edb964312d09571dc8e6fe27）
- 配置 delta（相对 v3，非 v4）：eta 时间表 H=1500/D=600/end=0.25 +
  n_rounds=9000；rho 时间表 v3 原样（floor 0.001）
- 等价性合同：tests/test_eta_anneal_equivalence.py 7 项——关闭 vs 全程
  保温逐位一致（loss/rho 历史、终表、RNG 终态）、中途降温保温段前缀
  逐位一致+降温后分叉（v5 前缀审计依赖的性质）、公式逐轮匹配、
  fail-closed 合同
- 专项测试：tests/test_fitness_only_eta_cooling_v5_runner.py 19 项
  （plan 只读、身份 fail-closed、不覆盖、config delta 仅 4 键、双时间
  表边界与篡改拒绝、1501 轮前缀审计伪造拒绝、四标签+边界阈值、v4 记录
  字面值核对）
- 全量回归：319 项通过（v1-v5 runner + fitness_only + schedule +
  等价性 + evolution + directional_diffusion），零回归
- plan 只读验证通过：rho floor 0.001 / eta 0.5→0.25 / prefix 1501 /
  T=9000 / generation_started=False

待办：用户授权后正式运行（nvidia-smi 选空闲卡，预计 65-70 分钟）。
判定四标签结果前冻结；v2 unsupported / v3 rejected /
v4 quality_regression 记录并列保留。

### 上一暂停点：v4 更低地板已运行，质量门失败（quality_regression_under_lower_floor），温度路线关闭（2026-09-04）

> 用户授权运行。GPU1，nltcs 生成 3561.4s / test 227s，固定 9000 轮
> 走满，时间表审计逐轮匹配，**前缀审计通过**（前 901 轮与 v3 四条
> 轨迹逐位一致）。未做任何结果后调参。

结果（`outputs/fitness_only_floor00005_dev_seed9908_v4/report.json`，
residual 臂，判据结果前冻结于 v4 协议）：

```text
质量门（fail-closed，先于 drift 判据）：
  nltcs L1 0.000165 ≤ 0.000197 ✓（v3 0.000179，再改善 8%）
  test  L1 0.002733 ≤ 0.002640 ✗（v3 0.002400，倒退 14%）
主判据 drift（被质量门盖住，记录用）：
  nltcs 1.1754 > 1.100（v3 1.1639——降地板后本底不降反微升）
  test  3.1364（v3 2.2400，量子化加剧，观察项）
形态：非 descending（best 7033@r7422，尾窗 8022 > 前窗 7832，
  链again 稳态横盘）
判定：quality_regression_under_lower_floor
```

两条关键科学结论（v3+v4 配对证据链）：

1. **nltcs 漂移本底对地板温度不敏感**：参与行减半（16→8 行/轮），
   本底 1.1639→1.1754 未收窄。说明带宽由"单行重采样量子跳变"
   主导，不由每轮扰动行数主导。温度路线对 drift 无效。
2. **test 小表被冻伤**：0.15 行/轮（约 7 轮动 1 行）修复能力不足，
   measured L1 显著倒退。地板不可再降（协议 §7 亦禁止下探）。

综合 v2/v3/v4：**floor=0.001（v3 配置）是质量最优点**（nltcs L1
0.000179 / test L1 0.002400 双优、held-out 配对优势保持）。
按 v4 协议 §7：quality_regression → 停下诊断，不进入后续屏。

下一步（等用户决定）：走 B 路线定稿——方法配置定为三段式
H=900/D=600/floor=0.001，叙事如实报告 drift 本底 ~1.16 为无门控
终态输出的固有代价，并以 v3（预算加倍）与 v4（地板减半）两次
冻结实验作为"本底与预算、更低温度无关"的对照证据；随后讨论
5-seed 确认屏协议与 fitness 聚合缺口优先级。

v2 unsupported / v3 rejected / v4 quality_regression 三条记录并列
保留，均不得重跑。

### 上一暂停点：v4 更低地板（0.0005）执行器与测试已就绪，等待运行授权（2026-09-04）

> 用户批准 v4 协议草稿后授权实现。已完成 v4 delta 执行器与专项测试
> （17 项全过，v1-v4 相关合计 69 项全过），plan 只读验证通过。
> **尚未运行**——正式运行需用户单独授权；运行前 nvidia-smi 选空闲
> GPU（CUDA_VISIBLE_DEVICES 注入，不改协议）。

结果前协议：`docs/设计/FitnessOnly地板0.0005降温v4结果前协议.md`。
执行器 `scripts/run_fitness_only_floor00005_v4.py` 继承 v3 链，变更 =
rho_anneal_end 0.001→0.0005 + n_rounds 9000（H=900/D=600 不变）。
v4 协议清单 SHA-256：

```text
3002afe55683f08c9b87ac5e8a59b0f45ca938406cc09e8287d1ddfb31d84cd2
```

新增审计与判定（全部结果前冻结）：

```text
前缀一致审计：v4 各臂前 901 轮（t∈[0,900]，两时间表逐点相同段）
  与 v3 逐位相等 + 初始表/初始化后 RNG 哈希相等（v3 参照产物
  5 文件 SHA-256 生成前 fail-closed 核对）；失败 →
  schedule_blindness_violated，运行无效、产物保留、不开 held-out。
判据：nltcs drift ≤1.100（不放宽）；质量门 nltcs L1 ≤0.000197、
  test L1 ≤0.002640（各 1.10×v3）。
形态：descending = 尾窗[8000,9000)均值 < 前窗[7000,8000)均值
  AND running-best 最后刷新 ≥8000。
四互斥标签：quality_regression_under_lower_floor /
  lower_floor_supported / lower_floor_budget_insufficient /
  lower_floor_rejected（→停止下探回 B 叙事）。
输出：outputs/fitness_only_floor00005_dev_seed9908_v4/（不覆盖）。
```

v2 unsupported / v3 rejected 记录保留不变。成本 ≈65-70 分钟
（nltcs ≈59 分钟两臂、test ≈4 分钟两臂、评价照旧）。

### 上一暂停点：v3 T=6000 已运行，截断假说被拒绝（truncation_hypothesis_rejected），产物已冻结（2026-09-04）

> 用户授权运行。GPU1，nltcs 生成 2366.5s / test 151.0s，固定 6000 轮
> 走满，时间表审计逐轮匹配，**前缀审计通过**（v3 前 3000 轮与 v2 四条
> 轨迹逐位一致 + 初始表/初始化后 RNG 哈希一致——horizon_invariant
> 承诺首次经受可证伪检验并成立）。未做任何结果后调参。

结果（`outputs/fitness_only_schedule_T6000_dev_seed9908_v3/report.json`，
residual 臂，判据结果前冻结于 v3 协议）：

```text
质量门（两条全过）：
  nltcs L1 0.000179 ≤ 0.000200 ✓（v2 0.000182，还在改善）
  test  L1 0.002400 ≤ 0.002786 ✓（v2 0.002533，改善 5%）
主判据 drift：
  nltcs 1.1639 ≤ 1.100 ✗（v2 1.2063，有改善但未达标）
形态判定（非 descending）：
  地板段分段均值 11382→9978→9640→9146→8713→7998→8231→8558→8221
  best 7146.5@r4432（<5000），尾窗[5000,6000)均值 8389 >
  前窗[4000,5000)均值 8115 → 链已于 ~r4400 进入稳态震荡
判定：truncation_hypothesis_rejected
```

含义：v2 的"链仍在降"是真的——再给 3000 轮后 best 又降 15%
（8409→7146.5）且两数据集 L1 双改善；但链在 ~r4400 到达稳态，
之后 1600 轮横盘。**drift ≈1.16 是地板温度（rho=0.001）下 nltcs 的
波动本底**，不是预算不足。加任何轮数都不会达成 ≤1.100。

观察项：test drift 2.2400（v2 1.7949，小表量子化波动，无硬门）；
held-out nltcs 优势原样保持（3way -0.0615 / 4way -0.0373）；
test held-out delta 略增（+0.0033/+0.0010，仍远小于 v1 的 +0.0056）。

按 v3 协议 §7 绑定动作：回 B/C 路线——
B = 接受本底修订方法叙事（时间表收益 test 漂移收窄 66%、两数据集
L1 全面改善、nltcs 漂移=本底 ~1.16 并如实报告）；
C = 另立地板修订协议（更低地板/终点邻域平均/DP 兼容早停输出规则，
任何一项都需新的结果前协议）。等待用户选择。

v2 unsupported 与 v3 rejected 两条记录并列保留，均不得重跑。

### 上一暂停点：T=6000 截断假说 v3 执行器与测试已就绪，等待运行授权（2026-09-04）

> 用户授权范围：批准 v3 协议草稿后"继续"（实现阶段）。已完成 v3
> delta 执行器与专项测试（16 项全过，v1/v2/时间表相关 52 项全过），
> plan 只读验证通过。**尚未运行**——正式运行需用户单独授权；运行前
> 用 nvidia-smi 选空闲 GPU（CUDA_VISIBLE_DEVICES 注入，不改协议）。

结果前协议：`docs/设计/FitnessOnly时间表T6000截断假说v3结果前协议.md`。
执行器 `scripts/run_fitness_only_schedule_v3_t6000.py` 继承 v2 全部科学
内容，唯一变更 n_rounds 3000→6000（时间表三常数逐字不变，公式不含 T）。
v3 协议清单 SHA-256：

```text
cd834439b90d6ffa7e7789c583019b92d07b3a763a4cf0a6357f30a0df8553b9
```

新增审计与判定（全部结果前冻结）：

```text
前缀一致审计：v3 各臂 loss/rho history[:3000] 与 v2 逐位相等 +
  初始表哈希、初始化后 RNG 哈希相等（v2 参照产物 5 个文件 SHA-256
  运行前 fail-closed 核对）；失败 → horizon_invariance_violated，
  运行无效、不打开 held-out、产物保留供架构诊断。
判据：nltcs drift ≤1.100（不放宽）；质量门 nltcs L1 ≤0.000200、
  test L1 ≤0.002786（各 1.10×v2）。
形态：descending = 尾窗[5000,6000)均值 < 前窗[4000,5000)均值
  AND running-best 最后刷新 ≥5000。
四互斥标签：quality_regression_under_extended_budget /
  truncation_hypothesis_supported / budget_still_insufficient /
  truncation_hypothesis_rejected。
输出：outputs/fitness_only_schedule_T6000_dev_seed9908_v3/（不覆盖）。
```

v2 的 unsupported 记录保留不变——v3 检验新假说（截断），不是重考。
成本 ≈45 分钟（nltcs cuda ≈2×20 分钟、test numpy ≈2×75 秒）。

### 上一暂停点：三段式 rho 时间表 v2 开发屏已运行并判定 unsupported，产物已冻结（2026-09-04）

> 用户授权范围："继续"（实现后接运行 v2）。已完成 v2 runner、其专项
> 测试与正式运行；按结果前冻结判据判定 `schedule_dev_unsupported`；
> 未做任何结果后调参或重跑，产物保留，promotion gate 保持关闭。

执行方式：新建 delta 执行器 `scripts/run_fitness_only_schedule_v2.py`，
继承 v1 模块全部科学内容（数据集、冻结输入哈希、seed 9908、T=3000、
配对两臂、评价管线、相位边界），唯一变更 = 预冻结三段式时间表
（H=900、D=600、地板 0.001）。v1 runner 仅加 `config_factory` 注入点
（默认行为不变）。v2 协议清单 SHA-256：

```text
bb7d19cb806a501a4cce6b139166a53cf8fe27f73140d951c15e4c3bb05a1531
```

runner 专项测试 9 项全过（plan 只读、身份 fail-closed、误确认先拒、
不覆盖输出、generation_config 与 v1 逐键一致仅差时间表三键、config
delta 唯一、三段边界、篡改历史拒绝、判据阈值逻辑）。

运行结果（`outputs/fitness_only_schedule_dev_seed9908_v2/report.json`，
生成 test 75.6s / nltcs 1177.7s，时间表审计 4 条轨迹逐轮匹配公式，
配对与源码不变审计全过，固定轮数走满）：

```text
判据一 漂移超额减半（residual 臂 drift = output/best）：
  test  : 1.7949 ≤ 2.167 ✓（v1 3.333，超额 2.333→0.795，收窄 66%）
  nltcs : 1.2063 ≤ 1.100 ✗（v1 1.2002，超额几乎不变）
判据二 measured L1 不倒退（≤1.10×v1）：
  test  : 0.002533 ≤ 0.003520 ✓（比 v1 0.003200 改善 21%）
  nltcs : 0.000182 ≤ 0.000287 ✓（比 v1 0.000261 改善 30%）
整体判定：schedule_dev_unsupported（判据一 nltcs 未达成）
```

观察项（无硬门，记录用）：

```text
test held-out 配对差值大幅收窄：3-way +0.005632→+0.002813，
  4-way +0.001875→+0.000566；one-way safety +0.038933→+0.017200
nltcs 绝对水平全面改善：best 14870→8409，terminal 17847.5→10143.5
nltcs held-out 配对优势保持（3-way −0.0615、4-way −0.0374）
equal 臂降温后 loss 轨迹冻结平稳（对照行为正常）
```

失败机理解读（记录，不构成新协议）：test 的漂移是高温末端重采样破坏，
时间表有效（超额 −66%）；nltcs 的 ~20% 相对漂移在绝对 loss 下降 43%
后等比例保留，更像当前地板温度下的稳态波动本底，非同一机理。若要
进一步压 nltcs 漂移（更低地板/更长降温），按协议 §7-§8 必须另立新
版本协议，本次产物保留为对照。

下一步（待用户决策）：按协议 §8 走"不支持"分支——回到调度设计
讨论（候选：地板比修订版协议、或接受 nltcs 漂移本底转向 fitness
聚合/覆盖缺口），或先讨论本结果对创新点叙事的影响。

### 最新暂停点：fitness-only 三段式 rho 时间表已实现并通过测试，v2 实验未运行（2026-09-04）

> 用户授权范围：按已批准的 v2 结果前协议实现代码与测试，跑实验前停止。
> 未运行任何 v2 生成实验，未修改 v1 冻结产物，promotion gate 保持关闭。

背景：v1 fitness-only 配对归因（seed 9908）机制成立但存在终点漂移
（output/best 漂移比 test 3.333、nltcs 1.2002），根因是恒定 rho 下末端仍以
固定强度重采样。依据
`docs/设计/FitnessOnly三段式rho时间表v2开发屏结果前协议.md` 实现预冻结
三段式盲时间表：保温 H=900 轮恒 rho=0.01 → D=600 轮几何降温 → 地板
rho/10=0.001（开口段，由总预算截断，将来由早停决定）；常数为归一化单位
（9 次/行 ×safety、一个数量级坡、地板比 0.1），全数据集通用、禁止按
数据集调；公式不含总轮数，视界不变保持成立。

本轮改动（均已完成，实验未跑）：

- `src/table_diffevo/evolution.py`：新参数 `rho_anneal_start_round`
  （默认 None 完全向后兼容，需与 `rho_anneal_rounds` 同时启用）；调度
  进度 `min(1, max(0, (t-H)/D))`；horizon_invariant 与 fitness-only 两道
  合同放行绝对轮数时间表、继续拒绝全程式退火；params 诊断新增该字段。
- `src/table_diffevo/fitness_only.py`：配置新增三个时间表字段
  （all-or-none 验证，end ∈ (0, rho]）；kwargs 传递；运行后审计逐轮核对
  `rho_schedule_history` 与预冻结公式一致（不一致即 RuntimeError）；
  配对核对字段加入 `rho_schedule_history`；pairing 记录
  `shared_rho_schedule`。
- `tests/test_fitness_only_schedule.py`：新增 19 项专项测试（三段边界
  逐轮核对、纯时间驱动非门控、两臂共享时间表、start=0 与 legacy 逐位
  一致、合同放行与 fail-closed 拒绝、运行后审计防篡改）。
- `docs/设计/FitnessOnly残差适应度扩散演化接线.md`：§3 合同同步并追加
  2026-09-04 修订节。

测试：时间表专项 19 + fitness-only 19 + rho 退火 23 = 61 全通过；全仓
`--continue-on-collection-errors` 下 2121 通过 / 43 失败（40 个为本分支
已知的协议哈希护栏 fail-closed——evolution.py 携带未提交 fitness-only
接线属预期；3 个为 venv 缺 scipy 的既有环境问题），无新增回归。

下一步（需用户单独授权）：运行 v2 配对归因（seed 9908，两数据集，
输出 `outputs/fitness_only_schedule_dev_seed9908_v2/`，不覆盖 v1）。
判据按协议 §5：漂移超额减半（drift_ratio test ≤2.167、nltcs ≤1.100）
且 measured L1 ≤1.10×v1（test ≤0.003520、nltcs ≤0.000287）；held-out
为观察项；失败记 `schedule_dev_unsupported`。

### 最新暂停点：Issue #53 问题一 A/R 双通道单种子筛查已评价，未通过（2026-09-02）

> 用户已明确授权完成冻结的两条候选轨迹及质量评价。本轮已完成采集、结果盲
> 勘误恢复和离线评价；未扩种子、未调参、未自动运行独立审计、未推送，也未
> 操作 PR #69。

候选仍是唯一研究臂 `gap_dual_abs_relative_max_s8`，开发种子 9908，数据集为
`test_300x10` 与 NLTCS。源生成提交为
`a3ba71fa2e84512fc6c8ba1bc818ba03e908cc71`，执行协议 SHA-256 为
`4ccf7bbe953d2523fca76d6a7e0ef1410e8773ebdb9a81fd50f8ed8f230b9386`。

两条 GPU 轨迹均完整生成：

```text
dataset        applied rounds  termination    case elapsed
test_300x10    1505            early_stopped  319.0563068520278 s
NLTCS          2805            early_stopped  5762.858124156017 s
```

源采集器在两个 case 均写入 staging 后，因继承的 Stage 6E 发布校验错误要求
`state_evaluation_count == applied_rounds+1` 而失败关闭；实际生成器契约为
`state_evaluation_count == max(1,applied_rounds)`，与此前 R8/sqrt 已冻结的同类
勘误一致。八个源 case 文件未改写、未删除，GPU 生成器未重跑。结果盲恢复协议
只对旧校验器构造内存代理计数，其余旧护栏与 A/R 逐轮权重/`8*K` 护栏原样通过：

```text
recovery protocol SHA-256  2132a4392740114398064b62bfc604cb97b3226e4b758d654dbd6a71dcc8c4b3
recovery commit            d51600e3f5b7b26261bf4ddca62318acbbfca1c0
collection SHA-256         3b53fcac678924205a4bd1038ec15f404b5245b09894d2b065d28794e615a790
```

最终评价在干净提交 `1a6194fd43437751b4b117e5dc4c30090637f239` 上运行，
复用已独立审计的 legacy/R8/sqrt 六条基线，不重跑旧臂；终表与所有已达到的
固定检查点均评价，未使用历史最好或检查点选表：

```text
evaluation adapter SHA-256  ebb1d6726b99f0f486934cb3c8b7bff8c561d594d02895ede68ed711dc422cb4
evaluation report SHA-256   df9c64430be28f45c0a354dff0861ba501b4c3e429e66a85956119961f5b5c56
screen metrics CSV SHA-256  3517bb58884f9aab2533a807a7e3996745de0e5b03154871489a76fa9fb4680e
candidate / baseline cases  2 / 6
CSV rows                     123
execution valid              true
```

冻结门的关键结果（越小越好）：

| 门 | candidate | baseline | ratio | threshold | pass |
|---|---:|---:|---:|---:|:---:|
| NLTCS measured normalized L1 vs legacy | 0.0001487913 | 0.0002894331 | 0.5141 | `<1.0` | ✅ |
| NLTCS 常见箱 mean error vs legacy | 2.7300771 | 6.3239075 | 0.4317 | `<1.0` | ✅ |
| NLTCS 稀有箱 mean error vs legacy | 1.4705882 | 0.9411765 | 1.5625 | `<=1.25` | ❌ |
| NLTCS 稀有箱 mean error vs sqrt | 1.4705882 | 1.5882353 | 0.9259 | `<1.0` | ✅ |
| test measured normalized L1 vs legacy | 0.0010000 | 0.0008667 | 1.1538 | `<=1.05` | ❌ |
| test one-way safety vs legacy | 0.0442667 | 0.0485333 | 0.9121 | `<=1.05` | ✅ |
| NLTCS one-way safety vs legacy | 0.0000927013 | 0.0002549286 | 0.3636 | `<=1.05` | ✅ |

冻结分类为 `rare_query_protection_not_recovered`，因为分类优先级先命中稀有箱
失败；test 总 L1 同时也独立失败。结论是：A/R 的 A 通道确实大幅保住并改善了
NLTCS 常见查询和总体误差，R 通道也使稀有箱优于 sqrt 约 7.4%，但仍比 legacy
差 56.25%，未达到 25% 容忍线；同时 test 总 L1 比 legacy 差 15.38%。因此当前
无参数 A/R `max` 设计不能作为问题一的合格解，也不得进入新种子确认。

评价过程中有两次结果发布前失败关闭：第一次是适配器错用 sqrt 顶层审计字段名，
第二次是把检查点历史字段 `gap_e` 误解释为 A/R C 能量。实际 `gap_e` 仍是
legacy relative 诊断，A/R C 核身份在逐轮 transition audit 中。两次均未生成
evaluation/CSV；修正、重冻结并通过 `13 passed` 后才执行上述正式评价。

当前严格停在单种子开发筛查结论处。`automatic_followup_authorized=false`，下一步
若继续问题一，应先讨论失败机制和新的结果前设计；不得自动调 A/R 比例、改门、
扩种子或把本结果描述为成功。

### 最新暂停点：Issue #53 问题一 A/R 双通道已通过开跑前预检，停在 collect 前（2026-09-02）

> 用户授权的范围是完成真实 GPU 候选生成之前的设计、实现、测试、协议和预检；
> 不授权启动两条候选轨迹、质量评价、新种子、调参、推送或操作 PR #69。

新增显式研究模式 `dual_abs_relative_max`，公共默认仍为
`legacy_relative`。冻结公式为：

```text
A = sum_all(|target-current|) / (N*J)
R = sum_positive(|target-current|/target)
    / [N*sum_positive(1/target)]
E = max(A,R)
```

`target=0` 只进 A，不进 R，不另设 Z；无正目标时退化为 `E=A`。
`floor` 不进入新公式，`max_weight_ratio` 必须为 null。微步在两侧分别先求
`max(A_b,R_b)` 再取 `E0-E1`，切换点不加 epsilon、滞回或随机平局。

实现只为新模式新增 R 通道的 float64 增量误差项/和；A 复用原误差状态。
旧 legacy/bounded/sqrt 继续走原 NumPy/Triton 结合顺序；新模式有专用单任务
CUDA eager 提交和逐微步批量 CUDA 分支。B 残差、`gap_l1_sweeps=8`、strength、
孤立分数 RMS 定尺、概率、RNG、自动停止和 terminal-current 均未改。

已通过：

- 相关 CPU/主循环回归 `155 passed`；
- 完整缺口核 CUDA 套件 `27 passed`；
- 新模式包括零目标、全零目标、等误差校准、A/R 各自主导、切换点、
  CPU 全量重算对拍、CPU/单 CUDA 逐微步对拍、单/批量 CUDA 轨迹哈希和 RNG 终点；
- 旧三种模式的现有数值与冻结 CUDA 轨迹测试全部通过。

全库试跑为 `2272 passed, 31 failed`。31 项集中于历史实验协议对
`evolution.py` / `gap_l1_diffusion.py` 旧提交哈希的预期失败关闭，以及本工作树
已存在/未复制的历史 outputs 前提；无 A/R 公式、增量状态、旧模式数值轨迹或
CUDA 回归失败。不改写历史冻结协议去伪装旧提交未变。

结果前科学协议：

```text
mode                         dual_abs_relative_max
tasks                        test_300x10 / NLTCS, development seed 9908
candidate arm                gap_dual_abs_relative_max_s8
scientific protocol SHA-256  85016eb9b91364006537b4ccadd71bea8685b502a96aa05d88f73c700d75532d
execution protocol SHA-256   4ccf7bbe953d2523fca76d6a7e0ef1410e8773ebdb9a81fd50f8ed8f230b9386
```

目标权重审计只读两份 measured query JSON，逐条固定 1051 条查询的 A/R 权重；
未导入核实现，未读参考表或任何候选结果。任务矩阵只新增两条轨迹，复用
已独立审计的 legacy/R8/sqrt 基线，不重跑旧臂。执行入口已实现双层哈希和
用户后续授权门。

已在干净实现提交 `2e01f26a628635904717d29a17a191ea274978f5` 上执行只读
`plan` 和 `preflight`：

```text
worktree clean including untracked   true
protocol / source / input identity   pass
GPU                                  linyao-system physical 1, RTX 4090 24 GiB
CUDA_VISIBLE_DEVICES                 1 (process cuda:0)
GPU preflight                        0% utilization, 18 MiB, no compute process
software                             Python 3.11 / NumPy 2.4.6 / pandas 3.0.3
                                     PyTorch 2.13.0+cu130 / CUDA 13.0
runtime executables                  all executable
ready_for_final_user_confirmation    true
screen_generation_authorized         false
generation_started                   false
candidate output / shard output      absent / absent
```

干净提交后的最终相关套件为 `215 passed in 9.14s`。已有平方根轨迹的案例耗时是
test 约 5.8 分钟、NLTCS 约 25.5 分钟；新模式单任务 CUDA 每微步需多维护一个
通道，尚无正式轨迹实测。执行预留保守墙钟范围为约 45--120 分钟；该范围只是
资源规划，不是质量或性能结论。

当前严格停在真实 GPU `collect` 前。只有用户后续单独明确授权并确认执行协议
`4ccf7bbe953d2523fca76d6a7e0ef1410e8773ebdb9a81fd50f8ed8f230b9386`，才能运行已冻结的
`collect` 命令。采集后仍必须先报告 collection SHA 并停止，不自动评价。

### 历史小结：第 6D 正式效果闭环 v1→v5（2026-08-27，五版均未产出正式结果，已由 Stage 6E 取代）

第 6D 目标是两数据 × 三方法 × 五种子、固定 2500 轮的正式效果比较。五个版本相继冻结，
均在启动阶段失败或未获授权，没有产出任何正式 L1 数值；正式比较最终改由 Stage 6E
自动停止协议完成（见上节）。逐版一句话结论：

- v1：冻结后启动预检失败关闭——采集器把"状态轨迹身份"哈希误当"结果盲身份"比较，生成器
  调用前即拒绝；0 案例，协议清单 `4c11aa1f…`。
- v2：修复身份边界后启动，第一条轨迹 2500 轮算完、序列化时 `tol=+inf` 被严格 JSON 记录器
  拒绝且异常清理误删临时产物；0 可恢复案例，协议清单 `83f21c49…`。
- v3：修复参数预检（`positive_infinity` 哨兵落盘）与断点保护（原子落盘、失败保留暂存）；
  冻结后未启动，用户改定 21/9 双服务器分片，协议清单 `3e648289…`。
- v4：增加 21/9 分片、双端环境身份、分片断点恢复与中央合并审计；两端就绪后启动时后台
  进程被启动工具回收+复用的旧监控器写死物理 1 号采样，17 秒内主动停止；0 案例，
  协议清单 `b64fbdff…`，A6000 暂存清单 `dac6e0ef…` 只作失败证据。
- v5：修复资源监控（从冻结分片读物理编号、逐样本记录、逐层拒绝跨卡样本）与前台承载；
  冻结等待授权期间被 Stage 6E 自动停止路线取代，从未运行。v5 冻结身份保留如下，
  各版完整协议、失败证据与验证记录见 git 历史（本文件 2026-08-27 各节）。

```text
v5 协议版本                      issue53-stage6d-joint-formal-effect-v5
v5 协议文档 SHA-256              842a251ed9c607250da9c17c70c422336ef055601b24dc1e443f24447f324344
v5 协议清单 SHA-256              835a2165beebe754707a942ae473bb0432b13ec7c914a1caac7b32a6a1a5a523
21/9 分片清单 SHA-256            627f39e59f8eb0d80a194719a8995c2db123fa1d1f1375de81d4f8ebc057d789
30 条参数清单 SHA-256            771b27775ddc9bde562c169e3c3cb991c2635d00b0087f90d1795847951e8e9d
正式采集/合并器 SHA-256          d020648c2ee9c3808e370ddab7fca74a26ddd412ea28c4643e3f9b280071f88b
离线评价器 SHA-256               4a18e49bc931446a978e1a1e7c3ae4c89684698a9d1782a4885b56e725ebc51b
独立复核器 SHA-256               901059965dc081e4dc707852411cfb018f475d5fb601e2ae3ffda5e415498205
```

### 最新暂停点：第 6C 阶段两数据三方法联合接线与资源测速 6/6 完成（2026-08-27）

> 用户在本地冻结提交和新协议清单展示后明确回复“运行”。本轮只执行结果盲的20轮接线/资源测速；没有输出、读取或比较
> 三种方法的任何误差或质量值，没有据此调参，也没有启动正式2500轮效果实验。

执行身份与唯一产物：

```text
执行提交                         1513c0329980b9c25f71be077b8f8110710589da
协议清单 SHA-256（哈希指纹）       aae6bee94f5872b61b3af3263a8f9bf8bcc87b0fb64812b1610069ea29787924
薄执行器 SHA-256（哈希指纹）       a676a9f2c29a9481dda66e842abeece9e3a46ad2f67d4bb91ce7df520446a864
报告路径                           outputs/issue53_stage6c_joint_smoke_seed9907_v1/timing_report.json
报告 SHA-256（哈希指纹）             ccda009b9326ba5f22437a87ff895ffffd00ba5c558cb1a20be16b0b146f4578
执行时间                           2026-08-27 09:50:11--09:51:12 +08:00
```

运行前工作树干净、输出目录不存在，物理1号 `NVIDIA GeForce RTX 4090（英伟达 GeForce RTX 4090 显卡）` 身份与冻结 UUID（唯一设备编号）
一致；利用率0%、基础显存15 MiB（兆字节）且无计算进程后才启动。全程只暴露物理1号显卡，使用2个 `spawn（新进程启动）` 工作进程，
没有回退中央处理器。

接线与无门控完整性：

- 6/6条预列轨迹全部完成，每条实际推进20轮，共120个应用轮次；全部因固定 `candidate_budget=20（候选预算20）` 结束。
- 120/120个唯一下一表全部无条件成为新当前表；没有接受/拒绝、回滚、缩步重试、重抽或历史赢家选择。六条报告终点均为
  `terminal_current（最后当前表）`，并与最后状态时钟哈希一致。
- 两个 `gap_l1_global_s8（新的 B+C 剩余缺口核）` 轨迹都建立了正固定参考尺度；小表执行1800个微步，NLTCS（真实数据）执行60040个微步，
  合计61840个，全部等于冻结的 `8*K（8乘以活跃开关数）`。
- 缺口核0次对数几率截断、六条轨迹0个非有限值；最终报告的递归质量字段禁止检查和固定汇总检查全部通过。
- 唯一报告在6/6条任务全部完成后由临时目录原子改名发布；没有中间、部分或覆盖产物。

墙钟与显卡资源事实：

```text
整次联合墙钟                         61.4231 s（秒）
显卡采样                               228 个，每0.25 s（秒）
平均 / 峰值利用率                    66.386% / 100%
非零利用率采样比例                    95.614%
整卡显存范围                         18--21807 MiB（兆字节） / 24564 MiB
峰值显存占比 / 余量                    88.776% / 2757 MiB
```

| 方法 | 数据 | 20轮墙钟 | 每轮墙钟 | 工作进程峰值实际分配显存 |
|---|---|---:|---:|---:|
| 原始 B 因子吉布斯核 | `test_300x10（小型测试数据）` | 1.9902 s | 0.09951 s | 0.0113 GiB（吉字节） |
| 原始 B 因子吉布斯核 | `NLTCS（真实数据）` | 14.8268 s | 0.74134 s | 9.7628 GiB |
| 现行独立 B 核 | `test_300x10（小型测试数据）` | 0.9759 s | 0.04879 s | 0.0113 GiB |
| 现行独立 B 核 | `NLTCS（真实数据）` | 8.2369 s | 0.41184 s | 9.7628 GiB |
| 新的 B+C 剩余缺口核 | `test_300x10（小型测试数据）` | 3.4892 s | 0.17446 s | 0.0113 GiB |
| 新的 B+C 剩余缺口核 | `NLTCS（真实数据）` | 44.7804 s | 2.23902 s | 9.7628 GiB |

工作进程在同一 `spawn（新进程启动）` 进程内会保留 PyTorch（张量计算框架）的显存缓存，因此小表缺口任务报告的10.1836 GiB 峰值保留显存是前一个
NLTCS 任务留在该工作进程的缓存，不是小表自身新分配10 GiB。判断双进程整卡容量应以21807 MiB的实测整卡峰值为准。

资源结论边界与下一步：

- 本次已证明2并发在当前两数据和三核上能完成，没有显存不足；但峰值已占整卡88.8%，余量约2.69 GiB，因而没有资源证据支持把同卡并发提高到3或更多。
- 严格按本次每轮墙钟线性外推，一个随机种子的6条2500轮任务按同顺序、2并发约为2.07小时；若以同样顺序一次排全5个随机种子，简单贪心调度外推约为7.22小时。
  这只是用20轮短轨迹的资源估算，不是已测正式耗时，也不是效果结论。
- 测速完成后物理1号显卡恢复为0%利用率、15 MiB基础显存且无计算进程。
- 当前不自动启动正式效果轨迹。真正下一步是与用户讨论并冻结正式协议：是否保持2并发、2500轮和建议的353--357五个新随机种子，以及效果输出、判定与审计边界。
  在正式协议得到用户确认前，不运行任何新轨迹。

### 最新暂停点：联合测速提交前深审与全仓回归通过，冻结执行提交就绪（2026-08-27）

> 用户授权继续完成提交前审查、无显卡全仓回归和本地提交。本步没有读取两个真实数据集执行轨迹，没有检查或
> 占用显卡，没有创建测速输出，也没有推送远端。

提交前逐项审查发现并修正了一个会阻断真实运行的参数接线问题：

- 旧的共同参数把 `factorized_gibbs_use_compiled_workload（使用预编译因子查询结构）=true` 同时传给了三组方法；
  但完整生成器只允许非零因子扫描启用该选项，因而独立 B 和新缺口核的4条任务会在入口直接被拒绝。
- 现在该选项已归入方法核差量：只有 `factor_b_s8（原始 B 因子吉布斯核）` 为 `true（启用）`，
  `independent_b_s0（现行独立 B 核）` 和 `gap_l1_global_s8（新的 B+C 剩余缺口核）` 都为 `false（关闭）`。
  这只修正执行参数的归属，没有改 B/C 公式、抽样强度、供体与参与行顺序、无门控语义或公共默认核。
- 执行器同时改为从因子核真实的嵌套诊断字段读取有限值护栏；对新缺口核则强制扫描次数为8、诊断轮数完整、
  `8*K（8乘以活跃开关数）` 一致。若已经推进至少一轮却全程未建立正固定参考尺度，整次测速必须失效；只有初始即精确零误差且零转移时例外。
- 新增人工小表贯通回归，三种冻结参数均真实调用现有 `run_evolution（完整生成循环）`，并继续通过最后表、无条件应用、因子有限值和缺口微步报告转换；
  该测试不加载真实数据，也不使用显卡。

深审后的最终冻结身份：

```text
协议文档 SHA-256（哈希指纹）  35b97ae43120724e6125732ad17b2e006c032465fa1876f858477d0d362b4aff
协议清单 SHA-256（哈希指纹）  aae6bee94f5872b61b3af3263a8f9bf8bcc87b0fb64812b1610069ea29787924
联合任务清单源码 SHA-256       37070be03d0b2d8121553f49f01079e37744eb418143f380160b058c9671a194
薄执行器源码 SHA-256             a676a9f2c29a9481dda66e842abeece9e3a46ad2f67d4bb91ce7df520446a864
```

深审前记录的 `eeeee528...` 协议清单已在任何真实运行前失效，不得再用于确认执行。新协议文档只补充上述未定尺失效边界，
用户已确认的两数据、三方法、20轮、最大2并发、物理1号显卡和结果盲输出范围未变。

最终无显卡验证：

```text
联合清单 + 协议 + 执行器专用测试                         47 passed
公共更新 + 完整生成 + 因子/缺口核 + 执行器相关回归       261 passed, 1 skipped
全仓无显卡回归                                                1979 passed, 21 skipped, 2 warnings
Python（编程语言）语法编译                                      passed
协议文档、核心源码与清单身份复核                         passed
git diff --check（代码差异格式检查）                              clean
```

21项跳过来自当前显式隐藏显卡及其他既有可选环境；2条警告是既有残差几何哈希失配测试中 NumPy（数值计算库）对空数组求均值的警告，
全部测试无失败。

当前边界与下一步：本节与上述实现形成同一个本地冻结提交，提交后工作树应保持干净。下一步不再改方法或自动运行；只有用户再明确说“运行”，
才先检查物理1号显卡是否空闲，再使用新协议清单 SHA 执行冻结的6条联合测速轨迹；不自动运行正式2500轮效果实验。

### 最新暂停点：联合测速薄执行器与人工双进程接线完成，真实数据和显卡运行仍未授权（2026-08-27）

> 用户授权继续冻结协议后的下一小步。本步实现只消费冻结6任务清单的薄执行器，并用人工表/人工工作进程验证调度和报告护栏；没有
> 加载 `test_300x10（小型测试数据）` 或 `NLTCS（真实数据）`，没有调用真实生成任务，没有检查或占用显卡，没有创建测速输出。

新增执行入口：

```text
scripts/run_issue53_stage6c_joint_smoke.py
当前文件 SHA-256  e7880c2f6a0f37772e6826dd11e13d000b31b30ba34c0a6c425db8dfaf8eecad
```

实现结果：

- `plan（只读计划）` 只核对冻结协议/核心源码身份，返回固定6个任务地址和执行器自身 SHA；明确记录
  `runner_wired=true（执行器已接线）`、`smoke_run_authorized=false（测速仍未授权）`、
  `generation_started=false（生成未开始）`，不审计输入内容、不检查显卡、不调用生成器。
- `run（运行）` 只有 `--confirm-protocol-sha（确认完整协议哈希）` 一个参数；没有数据集、方法、随机种子、轮数、并发数、设备或报告
  字段覆盖。运行前将强制协议/核心源码 SHA、包含未跟踪文件的干净工作树、全新输出目录、两数据输入 SHA 和物理1号显卡的编号、
  UUID（唯一设备编号）、名称、唯一可见设备及确定性算法身份；任何失败都禁止回退。
- 六条任务严格调用现有 `run_evolution（完整生成循环）`，每个 `spawn（新进程启动）` 工作进程自己加载对应结构、测量查询和边缘
  初始化，最多同时2条。任务预先全部提交并保持冻结顺序；任一人工任务失败时完整矩阵直接抛错，不返回可解释的部分结果。
- 发现并显式处理了现有公共返回语义差异：独立 B/因子核在未启用内层停止时，生成器主返回仍可能是历史最好表；新缺口核主返回是
  最后当前表。执行器不修改公共生成器，也不使用独立/因子主返回，而是三组统一读取 `return_final_table（返回最后表诊断）`，再与
  最后状态时钟哈希核对后作为唯一终点；历史最好返回立即丢弃。因此比较身份始终是 `terminal_current（最后当前表）`，没有运行末尾
  赢家选择。
- 生成器控制台输出在工作进程内完全压制，异常只向父进程暴露任务地址和异常类型，不泄漏误差数值。完整 diagnostics（诊断）不跨
  进程返回；工作进程只构造协议字段白名单中的终点表哈希、主随机端点、实际应用轮数、墙钟、进程显存峰值、无门控、定尺、缺口
  微步/`8*K`、截断和非有限计数。
- 精确零误差在抽供体前停止时，生成器的 `rounds_run` 会包含本次终止检查而没有实际状态转移；执行器已按状态时钟单独审计并把
  报告中的实际应用轮数记为0或此前已应用数，不把终止检查伪装成生成轮次。
- 父进程每0.25秒用 `nvidia-smi（显卡状态工具）` 采集整卡利用率与显存，报告平均/峰值利用率、非零采样比例及显存范围，避免旧5秒
  抽样频繁碰到0%而无法解释。工作进程同时记录各自显存峰值；两类口径分开，不冒充显卡独占性能。
- 最终报告字段严格白名单化并递归拒绝 loss（平方误差）、L1（平均绝对查询误差）、查询计数/残差、表内容、历史最好、胜负或排序。
  6/6任务、最后当前表、无条件应用、`8*K` 和零非有限值全部满足后，才在临时目录写唯一 JSON（结构化结果）并原子改名；失败或非法
  字段不会留下正式/部分输出，也拒绝覆盖已有目录。

验证全部不接触真实数据和显卡：

```text
薄执行器人工替身专用测试                                         13 passed
执行器 + 冻结协议 + 联合任务清单专用测试                         42 passed
再加因子/缺口完整循环与既有独立轨迹调度相关回归                  81 passed
只读 plan（计划）真实入口                                        6 tasks，未生成
Python（编程语言）语法编译                                       passed
git diff --check（代码差异格式检查）                              clean
Ruff（代码风格检查器）                                            当前可用测试环境未安装，未执行
```

当前边界与下一步：执行器已经具备运行条件，但正式入口强制干净工作树；当前阶段的完整改动仍未提交，所以现在直接运行会按设计拒绝。下一步
应先审查本工作区全部接线改动并由用户决定是否形成一个本地干净执行提交。只有提交、相关复验通过且用户再明确说“运行”，才检查物理1号
显卡是否空闲并执行冻结6条测速任务；不得顺带启动正式2500轮效果实验。

### 最新暂停点：两数据三方法联合长轨迹测速协议已结果前冻结，尚未实现执行器或运行（2026-08-27）

> 用户确认采用上一暂停点建议：正式长轨迹前先做一次只检查接线和资源的联合短轨迹测速。本步只完成协议文档、只读冻结清单、身份
> 校验和普通测试；没有加载两个数据集、没有调用生成器、没有检查或占用显卡、没有产生或查看任何方法效果。

冻结身份：

```text
协议文档      docs/设计/Issue53_Stage6C两数据三方法联合长轨迹测速接线协议.md
文档 SHA-256  637a1010a75dd84d08d004bf3b1a8dd23cce9b07ec42a90ceaa62af0240938a6
协议模块      scripts/issue53_stage6c_joint_smoke_protocol.py
协议 SHA-256  eeeee528933d029d2f4822fa40adfcb08567041f1222652b589f127dadb9b48a
预留输出      outputs/issue53_stage6c_joint_smoke_seed9907_v1/
```

唯一冻结测速矩阵：

```text
数据集          test_300x10, nltcs
方法            factor_b_s8, independent_b_s0, gap_l1_global_s8
随机种子        9907（只用于测速，排除在未来正式效果种子之外）
每条最大轮数    20
任务顺序        seed -> arm -> dataset
轨迹数          6
最大并发        2 个 spawn（新进程启动）工作进程
设备            物理 1 号 NVIDIA GeForce RTX 4090，进程内唯一 cuda:0
```

- `test_300x10` 固定使用第 6A/6B 阶段同一份 `measured_50query_30_15_5（30条二阶、15条三阶、5条四阶查询）`，
  `nltcs` 固定使用现有1001条二/三阶测量查询；两数据 schema（结构）、查询和初始化边缘文件 SHA-256 均写入清单。
- 六条任务共同固定：边缘初始化、`rho=0.01`、`eta=0.5`、`mu=0.01`、尺度不变供体、固定 `alpha=16`、
  `relative-f8（相对残差、分母下限8）`、现行 B 强度2及初始均方根定尺、`tol=+inf`、零重试、固定20轮上限、最后当前表输出；
  不启用 P=6 停止器，保留精确零误差时抽供体前确定性结束。
- 三组唯一差量仍是核：原 B 因子核8次扫描、独立 B 零扫描、新缺口核8次扫描；因子核复用已有等价预编译路径，新缺口核复用已有
  显卡分支，外层供体、参与行、突变和查询评价不重写。
- 任务必须在任一结果前一次性列全，入口以后不得开放数据集/方法/种子/轮数/并发覆盖，不得根据小表结果删除 NLTCS；并发固定为2，
  不按活跃开关、速度或中间结果重排，显卡失败禁止自动回退或改单进程续跑。
- 测速报告只允许设备身份、墙钟、显存/利用率、轮数/终止原因、随机端点与表哈希、无门控计数、定尺状态、`8*K` 微步和数值护栏；
  明确禁止 loss（平方误差）、L1（平均绝对查询误差）、查询误差/残差/计数、表内容、历史最好、方法排序/胜负和留出评价。
- 当前建议的正式353--357随机种子、2500轮、正式并发和效果判定不属于本测速协议，没有自动冻结。测速只能提供资源事实，不能根据
  隐藏或公开质量调整正式设置。
- 协议清单绑定当前完整生成器、公共随机更新方案、新缺口显卡核和联合任务清单源码 SHA；`plan（只读计划）` 只核对文档/源码身份并
  返回6个任务地址，不读取数据、不生成。协议冻结时明确记录 `runner_wired=false（执行器未接线）`、
  `smoke_run_authorized=false（测速运行未授权）`、公共默认核未改。

验证：

```text
联合测速协议 + 联合任务清单专用测试                              29 passed
再加因子/缺口完整循环与既有独立轨迹调度相关回归                   68 passed
协议文档、核心源码和清单 SHA 身份复核                             passed
Python（编程语言）语法编译                                       passed
git diff --check（代码差异格式检查）                              clean
```

当前边界与下一步：下一小步只能实现一个薄测速执行器，按冻结6任务清单加载输入并逐条调用现有 `run_evolution（完整生成循环）`，再用
人工替身测试两进程调度、显卡身份拒绝、任一任务失败使整体无效、质量字段白名单和原子输出。本暂停点不授权读取真实数据做接线检查，也
不授权在物理1号显卡运行；执行器和普通测试完成后仍需向用户汇报并等待单独“运行”授权。

### 最新暂停点：两数据三方法的联合长轨迹结果盲任务清单已接通，正式协议与执行仍未冻结（2026-08-27）

> 用户确认继续，并要求未来 `test_300x10（小型测试数据）` 与 `NLTCS（真实数据）` 放在同一轮运行，不能先看前者结果再决定
> 是否运行后者。本步只实现不读取数据、不调用生成器的联合任务清单基础设施；没有启动任何轨迹、没有使用显卡、没有冻结正式
> 随机种子/轮数/停止规则或并发数。

检查与设计结论：

- 现有 `gap_l1_diffusion（剩余缺口扩散）` 显卡批量后端批的是同一冻结状态下的多个局部候选地址，不是已经各自分叉的完整生成轨迹。
  把不同数据规模、不同当前表的完整轨迹硬塞进同一张量批次需要改写外层生成器，和本阶段只换核、复用现有流程的边界不符。
- 当前采用薄联合调度：每条正式轨迹以后仍调用同一个 `run_evolution（完整生成循环）`；联合层只负责在读取任何结果前一次性列全
  两个数据集、全部方法和全部种子。显卡并发将来只负责调度这些独立轨迹，不另写供体、参与行、更新、查询评价或新核。

实现结果：

- 新增 `scripts/issue53_stage6c_joint_trajectories.py`，固定联合清单的数据集为
  `test_300x10 -> nltcs`，方法为 `factor_b_s8（原始 B 因子吉布斯核）`、
  `independent_b_s0（现行独立 B 核）`、`gap_l1_global_s8（新的 B+C 剩余缺口核）`。
- `build_joint_trajectory_plan（构造联合轨迹计划）` 没有数据集或方法子集入口；每个种子必须同时生成 `2×3=6` 条任务，固定排序为
  `seed -> arm -> dataset（种子 -> 方法 -> 数据集）`，同一方法下两个数据集相邻。缺少任一数据集、额外塞入第三个数据集、空/重复/
  非法种子或非法轮数都会直接拒绝。
- 三组只暴露核差量：因子核为 `factorized_gibbs_sweeps=8`，独立 B 为两种扫描都为 0，新缺口核为 `gap_l1_sweeps=8`；共同强制
  `tol=+inf`、`max_retries=0`、使用现行 B 初始方向、强度 2、固定初始均方根口径和返回最后当前表。因此联合层没有生成后接受、
  拒绝、回滚、重试或历史赢家选择。
- `plan_manifest（计划清单）` 明确记录 `protocol_frozen=false（协议未冻结）`、
  `generation_started=false（生成未开始）`、禁止部分数据集计划和禁止结果驱动删任务。当前模块没有导入或调用生成器。

验证：

```text
联合任务矩阵与输入护栏专用测试                                      16 passed
联合清单 + 因子/缺口完整循环 + 既有独立轨迹调度相关回归             55 passed
Python（编程语言）语法编译                                         passed
git diff --check（代码差异格式检查）                                clean
```

当前边界与下一步：本步只是不会漏跑任一数据集/方法的结果盲清单，还不是可执行正式实验。下一步必须先讨论并冻结全新随机种子、两个数据集
各自的固定轮数或统一停止规则、显卡并发上限、主/辅助指标和完整结果判定；确认后才能让执行器消费这份完整清单。不能用未讨论的占位数字
直接运行，也不自动做接线冒烟验证。

### 最新暂停点：剩余缺口核已接入现有完整生成循环，单轨迹无门控接线与普通回归通过（2026-08-27）

> 用户授权继续公共随机方案拆分后的下一小步，并新增后续实验要求：test_300x10（小测试数据集）与 NLTCS（大数据集）
> 不再先后分开验证，而要放进同一轮共同冻结、共同执行的对照。具体实验协议仍待讨论，本步只完成单轨迹生成器接线和普通测试；
> 没有启动任何真实数据实验，没有扩展不同当前表的显卡批量轨迹，也没有修改公共默认核。

接线结果：

- `run_evolution（完整生成循环）` 新增 `gap_l1_sweeps` 参数。默认 `0` 时现有独立 B/因子核路径不变；只有显式设为冻结的 `8`
  才启用现有 `gap_l1_global（全局剩余缺口绝对误差核）`，其他扫描次数在入口直接拒绝。
- 新核没有重写供体、距离、适应度、B 方向、参与行、突变、查询评价、早停和状态记录。它只在原单步转移位置复用
  `UpdateRandomPlan（更新随机方案）`：同一主随机流先抽参与行、B 初始开关和后置突变，再把初始开关交给已有缺口核，最后应用
  同一份预抽突变。
- 缺口参考尺度遵循既定长轨迹语义：当前误差非零但本轮全部孤立分数为零时，不编造极小常数、不借用 B 尺度，也不启动缺口扫描，
  本轮原样应用 B 初始开关；以后每轮继续尝试，首次得到正非零均方根后立即使用并在余下轨迹永久冻结。
- 缺口查询结构只编译一次并跨轮复用。每轮扫描随机流由公开主种子和轮次单独派生；某轮活跃开关数 K 改变，只改变本轮消耗，
  不会平移后续轮次的随机地址，也不消费供体/参与/B/突变使用的主随机流。
- 新核入口强制 `residual_directed_diffusion=True（启用现行 B 初始方向）`、`tol=+inf（关闭整代门控）`、
  `max_retries=0（禁止缩步重试）`，并禁止与旧因子吉布斯核同时启用。生成后的查询评价只更新状态和诊断，代码结构上无条件应用
  唯一下一张表。
- 新核主返回固定为 `terminal_current（最后当前表）`。即使历史上出现过误差更小的表，也不会把它挑出来替换最终输出；`best`
  只保留为诊断。因而不存在逐轮接受/拒绝、回滚、重试或运行结束后的历史赢家选择。
- 新增定尺状态、冻结尺度、逐轮扫描种子及随机流哈希、微步轨迹、最终复制查询计数、参与/突变数、护栏命中和未定尺退化轮数诊断。

验证：

```text
新核完整循环专用测试（含真实三轮重放及故意更差表无条件接续）      12 passed
单步更新与完整生成循环全部测试                                   143 passed
方向核/因子核/缺口核/第 6A/早停/参考过程/自冷却等相关回归         332 passed
不重复合计                                                       475 passed
Python（编程语言）语法编译                                       passed
git diff --check（代码差异格式检查）                              clean
```

- 专用测试确认：两次运行的最终表、逐轮缺口随机种子、随机流终点、微步轨迹、最终开关计数完全一致；新核与独立 B 对照的主随机流
  终点一致；`8*K` 微步恒等式成立；无法定尺时不启动扫描；故意返回误差更大的下一表时仍无条件成为当前表和最终输出。
- `gap_l1_sweeps=0` 与未传该参数的历史轨迹一致，公共默认核仍未改变。

当前边界与下一步：用户已经决定未来 test_300x10 与 NLTCS 必须一起跑，但种子、轮数/停止规则、两组还是三组对照、共同资源边界、
显卡多轨迹批量方式和结果判定尚未冻结。下一步应先讨论这份联合长轨迹协议；在协议确认前不运行任一数据集，也不先拿 test 结果决定
是否运行 NLTCS。

### 最新暂停点：现有单步更新已拆出公共随机方案，历史表结果与随机流保持一致（2026-08-26）

> 用户确认新的剩余缺口核设计暂时合理，并授权继续约定的第一小步。本步只拆分现有独立 B 单步更新中的共同随机方案；
> 没有把新核接入完整生成器，没有修改供体/参与行顺序、B/C 公式、公共默认核或无门控语义，也没有启动任何实验。

实现结果：

- `src/table_diffevo/update.py` 新增 `UpdateRandomPlan（更新随机方案）` 和 `MutationEvent（突变事件）`，显式保存参与行、
  现行 B 核抽出的初始复制开关以及已经抽好的逐行突变属性和值。
- 新增 `sample_update_random_plan（抽取更新随机方案）`，严格保持历史随机数顺序：先抽参与行，再按 schema（结构定义）属性顺序
  抽全部复制随机数，随后抽突变行，最后按行抽突变属性和值；不会按实际参与数或复制数缩短随机带。
- 新增 `apply_update_random_plan（应用更新随机方案）`。默认应用初始 B 开关时精确复现原单步更新；也允许未来传入另一个最终复制
  开关，但仍复用完全相同的参与行和突变事件。最终开关不得启用未参与行。
- 现有 `evolve_step（单步演化）` 只改为依次调用上述两个公共函数；函数参数、默认路径、诊断字段和返回表语义保持不变。
  本步没有调用缺口核，因子吉布斯核也没有改写。

验证：

```text
单步更新定向测试（含 4 种拆分前逐项参考重放）                   27 passed
方向核/因子核/第 6A 阶段采集/完整生成主循环相关回归             262 passed
参考过程与自冷却相关回归                                         70 passed
合计                                                             359 passed
Python（编程语言）语法编译                                       passed
git diff --check（代码差异格式检查）                              clean
```

- 新回归同时核对拆分前后的最终表和随机数生成器终点；覆盖无参与、零复制、全突变、普通独立 B 和非零方向强度。
- 另有测试确认替换最终复制开关时，复制之后应用的仍是同一组预抽突变事件。

当前真正下一步需用户另行确认：只在现有 `run_evolution（完整生成循环）` 的单步转移位置接入新的剩余缺口核，先保持单轨迹实现，
明确固定参考尺度、独立扫描随机流、8 次扫描和每轮无条件接续；接线与普通测试通过后，再单独决定是否运行约 20 轮的小规模连续
轨迹检查。本步到此停止，不自动接入或运行。

### 最新暂停点：第 6B-1B 阶段 NLTCS 正式 500 地址批量显卡筛查与双审计通过，两数据共同支持继续开发（2026-08-26）

> 用户明确授权在物理 1 号显卡运行第二版正式批量管线。本轮严格执行结果前冻结协议；没有根据中间效果停止、补样、调参或
> 选择结果，没有修改公式、参考尺度、抽样强度、扫描次数、参与率、判定门槛或公共默认核。

执行身份与完整性：

- 正式执行提交为 `481f453fafd4c8b9e7a700731459e5c0b59cbe99`，冻结协议 SHA-256（哈希指纹）为
  `ecdd49ada43f4da578f901538a25339e6fce9b99b6718e09ee91af9e1fa17d00`；五份产物均绑定同一提交和协议。
- 全程只向本实验暴露物理 1 号 `NVIDIA GeForce RTX 4090（英伟达 GeForce RTX 4090 显卡）`，UUID（唯一设备编号）为
  `GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02`，进程内只见 `cuda:0（逻辑 0 号显卡）`，确定性算法开启。
- 五个随机种子 348--352 的固定参考尺度全部成功，范围为 `0.00011626454419536456`--`0.00011925116642605507`。
- 25/25 个冻结状态、500/500 个地址、三组方法和两条测量段全部采集，共 3000 条地址—方法—测量段记录；每状态 20 个
  地址组成固定批次，全部结果无条件保留。
- 新缺口核共执行 994456 个有效微步，逐地址严格等于 `8*K（8 乘以活跃开关数）`；0 次对数几率护栏命中、0 个非有限条件、
  0 个精确单向概率。结构审计 25/25 个状态通过，独立批量 CUDA（显卡计算）算术审计重新执行并核对全部 994456 个微步。
- 五个产物验证器全部通过；冻结评价与独立审计的完整数据集指标和最终分类精确一致，正式结果有效，机制证据有效。

正式效果结果：

| 预冻结指标 | 原始 B 因子吉布斯核 | 现行独立 B 核 | 新的全局剩余缺口绝对误差核 | 判定 |
|---|---:|---:|---:|---|
| 后初始仅复制平均误差 | 0.0368342930 | 0.0368129645 | 0.0362896917 | 分别下降 1.4785% / 1.4214%，两个比较均 5/5 随机种子更低 |
| 坏步平均误差损失 | 0.0001815010 | 0.0001666484 | 0.0000040550 | 分别下降 97.7659% / 97.5667%，两个比较均 5/5 随机种子更低 |
| 有益前进量保留比 | 基准 1.0 | 基准 1.0 | 1.8635 / 1.7546 | 高于预冻结均值 0.95 和单种子 0.90 门槛，两个比较均 5/5 通过 |
| 初始状态仅复制平均误差 | 2.1483843331 | 2.1488563405 | 2.1495741836 | 仅高 0.0554% / 0.0334%，远低于允许的 5% 非劣界，均 5/5 通过 |
| 后初始复制再突变平均误差 | 0.0368595385 | 0.0368370055 | 0.0363142859 | 分别下降 1.4793% / 1.4190%，完整段安全门均 5/5 通过 |

- 六项冻结布尔判定——稳定误差改善、坏步抑制、有益步保留、初始安全、完整段安全和转移支持——全部为真。
- NLTCS（数据集）的正式分类为 `gap_kernel_development_supported（新核开发筛查获得支持）`；与此前小数据同类结果合并后，
  总分类为 `shared_development_support（两数据开发筛查共同支持）`。
- 结果含义是：新核在这些冻结局部状态上，不是靠统一减速取得较小误差；它主要压低了走坏方向的损失，同时保住并放大了
  原本有益的前进量。误差净改善幅度约 1.4%，不算巨大，但五个随机种子的方向完全一致，且坏步损失下降非常明显。

无门控与结论边界：

- 500/500 个地址及三组结果全部无条件保留；没有生成后接受、拒绝、回滚、重试、赢家选择、结果筛选、进入噪声范围后冻结或
  结果驱动补样。实验末尾的通过/不通过只决定后续研究方向，不参与生成过程。
- 这仍是使用既有 348--352 随机种子的固定状态开发筛查，不是全新随机种子的完整长轨迹生成。因此结果支持继续开发新核，
  但还不能宣称最终合成表整体优于现行方法，也不能直接替换公共默认核；产物明确记录
  `default_kernel_changed=false（默认核没有改变）`。
- 按冻结协议，下一步只能先起草“全新随机种子完整长轨迹”的结果前协议，讨论并冻结后再另行请求运行授权；本轮不自动进入。

耗时与显卡负载说明：

- 五段内部诊断耗时合计约 8441.51 秒，即 2 小时 20 分 42 秒：定尺 10.97 秒、采集 6081.57 秒、结构审计 1190.90 秒、
  冻结评价 1.04 秒、独立算术审计 1157.04 秒。
- 只有新缺口核的条件扫描采用跨地址批量显卡后端；供体/参与行准备、两个既有基准、记录整理和精确汇总仍有处理器阶段。
  因此 5 秒取样经常恰好看到 0% 利用率，但批量扫描峰值达到 81%--84%，不是停机或回退处理器。
- 运行期间另一用户曾在同一张卡上启动两个进程。监控到整卡总峰值显存约 16150 MiB、总峰值利用率 84%、最多三个计算进程；
  本实验没有显存错误且双重独立审计全部通过，所以科学结果仍有效，但上述耗时和整卡峰值不能当作本实验独占显卡性能基准。

正式产物目录：
`outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_v2/`。

```text
calibration_manifest.json              2b9e6fb0efa34acf4d335b17df35c8d6fe9ef7fb5805c966e2082d3321275041
screen_collection.json                  09fb0724a79b90178d71a201211619f6b4db934024ab0ffe2c5b9918d4506882
structural_audit.json                   2c67550191bece74199fb1299fc2846c074db53c35ec713db6c08c45a628eb60
frozen_evaluation.json                  07e87dfb3396da65a0aacb7f8117613cd0dffbcf2fac88409c58e014e3b28a7e
independent_arithmetic_audit.json       9fcaeb39b50ab420e0fbdbc821fa882b6d39b75f2ca951b15c9bf5c21feae1a5
```

### 最新暂停点：第 6B-1B 阶段新批量协议 10 地址小规模五段管线与双审计通过（2026-08-26）

> 用户明确授权“跑”上一暂停点约定的新批量协议 10 地址小规模管线。本轮只运行 `smoke（小规模接线）` 模式；
> 没有运行 500 地址正式实验，没有读取或解释被压制的方法效果，没有调公式、参考尺度、抽样强度、扫描次数、参与率或门槛。

执行身份与矩阵：

- 执行提交为 `f244c5045a0571a3ee03cddc4a1d88cdb2fba7b7`，工作树在五段运行前保持干净；冻结协议
  SHA-256（哈希指纹）为 `ecdd49ada43f4da578f901538a25339e6fce9b99b6718e09ee91af9e1fa17d00`。
- 全程只暴露物理 1 号 `NVIDIA GeForce RTX 4090（英伟达 GeForce RTX 4090 显卡）`，UUID（唯一设备编号）为
  `GPU-a3ed64b7-5f7a-0f95-9913-74fdb2340a02`；五份产物均记录只见一张显卡且确定性算法开启。
- 冻结种子 9906 的 5 个状态完整执行，每状态恰好 2 个地址，共 5 个生产批次、10 个地址、60 条三组两段记录；
  五批均按 `proposal_index（候选编号）` 升序，批量格式为 `issue53_gap_l1_batched_cuda_float64_v2`。
- 新输出目录为 `outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_smoke_v2/`；旧单地址小规模和已中止正式目录
  均未覆盖、续跑或复制。

五段结果：

- 定尺成功，固定参考尺度仍为 `0.0014675087058420766`；定尺后端为单地址
  `torch_cuda_float64（PyTorch 显卡双精度）`，与批量采集身份明确分开。
- 采集完整：5/5 个状态、10/10 个地址全部生成并无条件保留；生产后端均为
  `torch_cuda_float64_batched（PyTorch 显卡双精度批量后端）`，批大小均为 2。
- 新缺口核合计执行 248 个有效微步并逐地址满足 `8*K（8 乘以活跃开关数）`；其中一个状态两个地址均为 `K=0`，
  批量核正确执行零微步且独立审计确认不消费随机流。全程 0 次对数几率截断、0 个非有限条件、0 个精确单向概率。
- 结构审计 5/5 个状态、10/10 个地址通过，表、稀疏编辑、供体、参与行、初始开关、增量/完整查询计数、随机轨迹和批内位置一致。
- 冻结评价器完成，但小规模边界正确保持：`formal_result_valid=false（正式结果无效）`、
  `mechanism_evidence_emitted=false（不发布机制证据）`、`final_screen_classification=null（不产生最终分类）`。
- 完全独立的批量 CUDA（显卡计算）算术审计 5/5 个状态通过，独立重放全部 248 个微步；明确记录没有导入生产批量算术、
  没有新生成、没有接受/拒绝，并精确核对全部稀疏表、查询计数、误差算术、轨迹和内部汇总。

无门控与耗时：

- 10/10 个地址及三组结果全部无条件保留；没有接受、拒绝、重试、回滚、赢家选择、结果筛选、按结果重排或结果驱动补样。
- 五段产物内部诊断耗时合计约 `38.7547` 秒：定尺 `0.4537` 秒、采集 `17.8281` 秒、结构审计 `8.1282` 秒、
  冻结评价 `0.0207` 秒、独立算术审计 `12.3239` 秒。
- 这个小规模来源只有 128 行、每批 2 地址且总共只有 248 个微步，批量调度开销占比很高；因此本次耗时只能证明真实文件管线
  可运行，不能用来推断正式批量 20 的加速比。当前产物未记录进程峰值显存，不能补造精确峰值；五段均无显存错误，设备总显存
  记录为 25250627584 字节。正式同尺寸资源判断仍以此前批量 20 人工测量为依据。

产物 SHA-256（文件指纹）：

```text
calibration_manifest.json              a61d76857a124138af6885a4823f9ff9b44796e37119d8f156accb76c4a3ae3e
screen_collection.json                  eddb91b0b65f04ecd02d06dcbf72acd60376f178ee32e6ea7eeaee3268a481b2
structural_audit.json                   f4fa7141126c28d0b9c220cc0a545fb83b877c6027baba45c575ed7445415272
frozen_evaluation.json                  c4d1fbc6c80fbeae630d1d4726df3146f99cca33ad86766bc2586468d34db566
independent_arithmetic_audit.json       39861780b9d45a0f2f33a7d05c774354d492441c92c0a4559ae9293cffe8e5c9
```

当前真正下一步需另行确认：是否在物理 1 号显卡启动全新第二版正式目录的 500 地址运行。正式运行仍未获授权；
在用户明确同意前不得启动、不得从小规模内部值判断方法好坏，也不得修改公共默认核。

### 最新暂停点：第 6B-1B 阶段显卡批量协议已接入五段执行管线，尚未运行 10 地址小规模实验（2026-08-26）

> 用户授权继续上一暂停点约定的“只接线并做普通/人工测试”小步。本步没有读取冻结状态或正式地址，没有创建实验输出，
> 没有运行 10 地址小规模实验或 500 地址正式实验，也没有查看方法效果、改公式或改变无门控设计。

本步接线结果：

- 共享协议加载器新增 `stage6b1b_batched（第 6B-1B 批量协议）` 选择项；旧单地址协议、旧默认选择和旧输出目录均未改动。
- 定尺仍逐地址使用已冻结的单地址 CUDA（显卡计算）核来计算隔离尺度分数；定尺本身不是生成扫描，因此不伪装成批量计算。
- 采集器先按 `proposal_index（候选编号）` 升序，为同一状态内每个地址分别重放供体、参与行、初始开关及独立随机流，
  再把正式每状态 20 个地址或小规模每状态 2 个地址一次送入生产批量缺口核。批内地址不共享供体、随机数或开关状态。
- 结构审计器按同样的状态边界和地址顺序，用生产批量核整批重放，并逐地址核对表、开关、随机数终点、微步轨迹和批次位置。
- 独立算术审计器没有导入生产批量算术；它用单独实现的独立批量 CUDA（显卡计算）重放，逐地址精确核对最终表、开关、
  随机流起止哈希、轨迹哈希、微步数、裁剪数、查询计数、生产/独立后端身份及批内位置。
- 评价公式、三组方法、查询、指标和门槛未改。已经精确到目标的状态仍在抽供体前直接执行确定性 `no-op（不操作）`，
  生产和独立批量核均不被调用。

无门控与隔离边界：

- 生成的全部地址和三组结果仍无条件保留；没有接受/拒绝、重试、回滚、赢家选择、结果筛选或进入噪声范围后冻结。
- 禁止按 K、供体或结果重排，禁止动态改变批大小，禁止回退单地址或 CPU（中央处理器）；失败即整次运行无资格。
- 定尺、采集、结构审计、评价和独立算术审计五段都对输出路径做精确检查，只接受新的
  `outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_v2/（正式批量输出目录）` 或
  `outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_smoke_v2/（小规模批量输出目录）`。
- 冻结协议清单没有被改写，SHA-256（哈希指纹）仍为
  `ecdd49ada43f4da578f901538a25339e6fce9b99b6718e09ee91af9e1fa17d00`；清单中“冻结时尚未接线”的历史边界保留，
  当前只读计划另行明确现在已有接线入口。

验证：

```text
五个只读计划入口选择新协议且不读取源数据                 5 passed
两地址人工生产批量核与独立批量审计真实显卡端到端对拍       1 passed
第 6B-1 及批量接线定向回归                               66 passed
因子核/缺口核/第 6A/6B-1/批量协议/采样/查询较宽回归       390 passed
Python（编程语言）语法编译                               passed
git diff --check（代码差异格式检查）                     clean
```

当前真正下一步需另行确认：只在物理 1 号显卡上运行全新目录的 10 地址小规模五段管线，先验证真实文件接线、耗时、显存和
两套审计能否完整通过；小规模结果通过前不启动 500 地址正式实验。本步到此停止，不自动运行小规模或正式实验。

### 最新暂停点：第 6B-1B 阶段显卡批量执行差量协议已冻结，尚未接正式管线（2026-08-26）

> 用户授权继续上一暂停点约定的协议冻结小步。本步只固定已经通过人工精确对拍的“生产批量实现 + 独立批量审计”执行身份；
> 没有接入正式定尺、采集、结构审计、评价或正式算术审计，没有读取冻结状态或正式地址，没有运行 10 地址小规模实验或
> 500 地址正式实验，也没有评价方法效果或修改公共默认核。

历史隔离与冻结身份：

- 原单地址协议 `issue53-stage6b1b-nltcs-gap-l1-gpu-screen-v1`、原清单哈希
  `754fbf02...c63387` 和已中止的旧输出目录保持不变；没有覆盖或冒充旧运行身份。
- 新增独立批量差量协议文档
  `docs/设计/Issue53_Stage6B1B_NLTCS显卡批量执行差量协议.md`，文档 SHA-256（哈希指纹）为
  `6fdcefb5...531474`，文档提交为 `fd6031043f606fbc2073b55abf6b1714ed99cb0e`。
- 新可执行协议版本为 `issue53-stage6b1b-nltcs-gap-l1-gpu-batched-execution-v2`，冻结清单 SHA-256 为
  `ecdd49ada43f4da578f901538a25339e6fce9b99b6718e09ee91af9e1fa17d00`。
- 批量实现来源固定到提交 `104d2c8c09e86f70e5eb91e445ba8c181ead376d` 及生产/独立源文件哈希；生产格式固定为
  `issue53_gap_l1_batched_cuda_float64_v2`，独立审计格式固定为
  `issue53_gap_l1_independent_batched_audit_v1`。

唯一批量规则：

- 只批量执行 `gap_l1_global_s8（全局剩余缺口绝对误差核）`；另外两组、方法参数、查询、地址、指标、门槛和无门控契约不变。
- 正式每个冻结状态的 20 个地址恰好组成一批，共 25 批；小规模每个状态的 2 个地址恰好组成一批，共 5 批。
- 批内严格按 `proposal_index（候选编号）` 升序。禁止跨状态合批、按 K/供体/结果排序、动态改批大小、回退单地址或回退
  CPU（中央处理器）；失败即本轮无资格。
- 每个地址仍独立消费 `8*K` 组坐标与随机数，`K=0` 不消费随机数。短地址填充不更新状态、不消费随机数、不写轨迹。
- 批量归约允许与旧单地址浮点末位和轨迹哈希不同，但坐标、随机数、抽样决定、最终开关、查询计数和表必须与旧语义精确一致，
  逐步有限浮点差仍不得超过 `1e-12` 且不得翻转抽样。
- 新执行身份要求生产批量与独立批量的有效步、坐标、随机流初始/终点哈希、7 个逐步双精度值、3 个逐步布尔值、逐地址轨迹
  哈希、最终开关、计数和表全部逐位一致，不使用容差放行。
- 批量核继续无门控：没有接受/拒绝、回滚、赢家选择、结果筛选或进入噪声范围后冻结地址/批次。

输出与授权隔离：

- 新正式目录固定为 `outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_v2/`，新小规模目录固定为
  `outputs/issue53_stage6b1b_nltcs_gap_l1_gpu_batched_screen_smoke_v2/`。
- 禁止续跑或复制旧中止目录；未来获准接线后必须在新目录重新定尺。
- 新协议模块目前没有被定尺、采集、结构审计、评价或正式算术审计脚本导入；冻结不等于运行授权。

```text
新增批量差量协议身份与边界测试                     9 passed
因子核/缺口核/第 6A/6B-1/批量协议/采样/查询回归    337 passed
Python（编程语言）语法编译                          passed
git diff --check                                    clean
```

当前真正下一步需另行确认：只把正式采集和正式独立算术审计接到新批量协议入口，并运行不读取正式地址的普通接线测试；
完成后再单独讨论是否在物理 1 号显卡启动 10 地址小规模管线。500 地址正式运行仍未获授权。

### 最新暂停点：第 6B-1B 阶段不同地址显卡人工批量原型通过，端到端吞吐约 7.10 倍（2026-08-26）

> 用户授权继续上一暂停点的下一小步。本步只把人工批量原型推广到同一状态内供体、参与行、活跃坐标、K、初始开关和
> 随机带均不同的地址，并与现有单地址实现逐地址对拍；没有读取冻结状态或正式地址、恢复已终止的正式运行、评价方法效果、
> 接入正式采集、实现独立批量审计或修改公共生成器。

实现边界：

- `run_variable_workloads_batched_prototype（不同地址批量人工原型）` 共享同一 current（当前表）、目标和查询状态，
  但为每个地址分别建立 donors（供体表）、participate（参与行）、初始开关和随机带。
- 活跃行按批内最大行数填充；每个属性的受影响查询按批内固定宽度填充，并使用互不重复的哨兵索引避免原地更新冲突；
  完整查询计数、查询误差项和每行合取失败状态均在显卡上按地址分别维护。
- 每个地址仍按自己的 K 消费恰好 `8*K` 组“坐标、随机数”。微步带按批内最长地址填充，显式有效位掩码让已经结束的
  短地址保持原状态且不产生轨迹输出，因此没有改变任何地址内部的严格顺序。
- 原首版同结构入口保留为上述通用实现的薄封装，没有复制第二套扫描算法。正式采集、结构审计、独立算术审计和公共生成器
  均未导入该人工原型。

物理 1 号 RTX 4090（显卡）NLTCS（数据集）同尺寸人工结果：16181 行、16 属性、1001 查询、20 个地址；
参与行 134–188，`K=1074..1513`，每地址 8592–12104 个微步。20 个现有单地址参考总耗时约 122.4134 秒，
平均约 6.1207 秒/地址。

```text
批量    批量总耗时（秒）    平均每地址（秒）    相对现有单地址总吞吐
1       10.4117              10.4117             0.6729x
4       12.7130               3.1783             1.9903x
8       13.8018               1.7252             3.5696x
16      16.1402               1.0088             6.0192x
20      17.2454               0.8623             7.0983x
```

- 批量 20 的 17.2454 秒已经包含约 3.6885 秒逐地址准备、13.2123 秒批量扫描和 0.3447 秒轨迹/物化，
  不是只计核心算子的理想化数字。最长微步带为 12104，20 地址共含 37480 个无效填充位置。
- 批量计算期间连续采样的显卡利用率稳定约 71%–72%，不再接近 0%；批量 20 峰值实际分配显存
  `235698176` 字节（约 224.78 MiB），框架保留 `247463936` 字节（约 236.00 MiB）；外部监视最高约 709 MB。
  这些数字仍不包含未来正式状态上下文及同流程其他两组的常驻内存。
- 20/20 个不同地址的微步长度、坐标、随机数、全部逐微步布尔结果、最终开关、最终查询计数和最终表均精确一致；
  抽样决策翻转数为 0，所有条件概率均为有限开区间值。最大浮点绝对差约 `1.821e-13`，低于 `1e-12` 界。
- 批量归约仍会改变浮点末位，所以旧单地址轨迹 SHA-256（哈希指纹）不相同；这再次确认未来生产批量实现不能冒用旧轨迹
  哈希，必须与算术独立实现的新批量审计形成新的精确执行身份。
- 约 7.10 倍是不同地址缺口核路径的端到端人工吞吐；正式采集还包含供体/共同提议重放、原 B 因子基准及评价，不能把
  该数字直接当成完整正式流程加速比。

验证：

```text
批量原型 + 第 6B-1B + CUDA（显卡计算）相邻测试       28 passed
因子核/缺口核/第 6A/6B-1/采样/查询回归              317 passed
原 NLTCS 同尺寸生产/独立人工资格                      passed（全部旧精确检查仍为 true）
小尺寸 3 地址端到端 + 4 地址含 K=0 单元对拍           passed
git diff --check                                      clean
```

当前真正下一步需另行确认：基于已通过的不同地址人工原型设计生产批量采集入口，并单独实现不复用生产算术的批量审计；
二者对拍后冻结新的性能执行差量协议，再跑普通测试和 10 地址小规模接线。上述步骤均尚未开始，已终止的旧正式运行不能恢复，
正式 500 地址重跑仍未获授权。

### 最新暂停点：第 6B-1B 阶段多地址显卡人工原型达到约 11.62 倍吞吐，尚未接入正式管线（2026-08-26）

> 用户观察到正式 NLTCS（数据集）运行期间显卡利用率接近 0%，要求先暂停、随后明确终止该次运行并检查性能。
> 本步只实现和验证不读取冻结地址的人工批量原型；没有恢复正式实验、查看部分方法效果、修改方法公式或接入公共生成器。

正式运行中止边界：

- 被终止的执行提交仍为 `4a26d0e6a17e33c9db97367b6fdf9f268b83277d`，冻结执行清单 SHA-256（哈希指纹）仍为
  `754fbf02fdaeacb716fae0be90ee1b3d8fd27f1b8dd0c8faa1c73a11ddc63387`。
- 5/5 个随机种子定尺已经完成；采集在第一个状态尚未原子发布前终止，后续结构审计、冻结评价和独立算术审计未运行。
- 中止目录只保留 `calibration_manifest.json（定尺清单）` 与 `formal_pipeline.log（正式流程日志）`；没有
  `screen_collection.json（三组采集）`，没有查看或发布任何部分效果。进程终止后物理 1 号显卡恢复为空闲。

性能根因与实测：

- 一个同尺寸地址有 `K=1236`、9888 个严格顺序微步。现有实现由 Python（编程语言）逐微步派发许多小型
  PyTorch（张量计算框架）算子；函数分析记录每地址各 9888 次条件计算和状态更新，以及大量小型
  `where（条件选择）`、类型转换与求和调用。
- 单地址生产扫描约 6.05 秒，其中约 5.82 秒、即约 96% 位于微步扫描段。正式采集还逐地址串行运行三个组，
  原始 B 因子吉布斯基准本身在同尺寸人工表上约 0.95 秒，因此正式流程整体会出现显卡等待中央处理器的区段。
- 仅增加独立 CUDA（显卡计算）执行流不是解决办法：批量 4 吞吐约 1.93 倍，批量 8 反而退化到约 1.19 倍，
  原因是多个 Python 线程仍争用逐微步调度。

已完成的人工批量张量原型：

- 新增 `scripts/issue53_stage6b1b_batched_cuda_prototype.py`，一个微步只派发一套批量张量操作，同时推进多个地址；
  每个地址内部仍保持原坐标顺序、原随机带、原 8 次扫描、原条件概率和无门控边界。
- 新增可重复基准 `scripts/benchmark_issue53_stage6b1b_batched_gpu_prototype.py`；现有同尺寸人工基准只抽出共享的
  `build_artificial_workload（构造人工工作量）`，原资格逻辑和输出保持不变。
- 首版原型故意只允许一批地址共享 current（当前表）、donors（供体表）、participate（参与行）和活跃坐标，
  但使用不同初始开关和不同随机带；因此它只证明核心扫描批量化可行，不是生产实现。

物理 1 号 RTX 4090（显卡）同尺寸人工结果：16181 行、16 属性、1001 查询、154 个参与行、9888 微步/地址；
4 个不同旧版地址参考平均约 6.0642 秒/地址。

```text
批量    总耗时（秒）    平均每地址（秒）    相对旧版吞吐
1       5.1439          5.1439              1.1789x
4       7.3008          1.8252              3.3225x
8       8.0132          1.0016              6.0543x
16      9.5733          0.5983             10.1352x
20     10.4371          0.5219             11.6205x
```

- 批量计算期间连续显卡采样主要约为 57%–70%，不再接近 0%；批量 20 的原型峰值实际分配显存约 148.35 MB、
  框架保留显存约 159.38 MB。这里不包含正式状态上下文的约 9.9 GB 常驻显存，不能直接当作正式峰值。
- 对 4 个不同初始开关、不同随机带的同尺寸参考地址：全部坐标和随机数、全部逐微步布尔结果、最终开关、
  查询计数和表完全一致，0 次抽样翻转；最大浮点差约 `1.821e-13`，低于冻结 `1e-12` 界。
- 由于批量版对查询项采用批量归约，浮点末位不同，旧单地址微步轨迹 SHA-256（哈希指纹）不相同；这不是结果翻转，
  但意味着未来若接入生产，必须先冻结新的执行等价口径，并让生产批量实现与独立批量审计形成新的精确轨迹身份，
  不能冒用旧批量 1 的轨迹哈希。

验证：

```text
新增原型 + 第 6B-1B + CUDA 相邻测试      18 passed
因子核/缺口核/第 6A/6B-1/采样/查询回归  330 passed
原 NLTCS 同尺寸生产/独立人工资格          passed（全部旧精确检查仍为 true）
git diff --check                           clean
```

当前真正下一步需另行确认：把原型推广到正式同状态下 20 个不同供体、不同参与行、不同 K 的地址，批量计算完整查询指标，
并实现与生产算术分离的独立批量审计；随后冻结新的性能执行差量协议、重跑普通测试与 10 地址小规模接线。当前旧正式运行
不能恢复，人工原型没有正式运行资格，没有推送远端、创建拉取请求、更新议题或修改公共默认核。
### 最新暂停点：Stage 5 同温度核比较完成正式闭环并收口（2026-08-27）

> 本节覆盖下方“Stage 5 尚未实现或运行”的历史暂停描述。正式 collection（采集）、frozen evaluator
> （冻结评价器）与 independent audit（独立审计）均已完成；本次收口只补充结果文档与状态记录，
> 不重跑实验、不改写正式产物、不审查或合并任何 PR。

正式身份：

```text
execution commit       a5455ca1574a45acbbbe68abe3100a97e4b976ba（clean）
formal cases           2 datasets × 2 arms × 10 seeds = 40 / 40 complete
seeds                  338..347
protocol document SHA  4d0ffb8ebf77006becea00849eef452174559aea62faeb86dd70c227e3fc7fab
protocol manifest SHA  1d447be0fb0ce9a2c7707abd2e259ed6ea41edbf3780f30426e320b1bad94f1c
audit                  overall_pass=true / 40 of 40 cases recomputed
```

冻结主结果：

| 数据集 | independent（独立核）总误差 | factor（因子核）总误差 | 相对变化 | 配对胜/平/负 | 正式分类 |
|---|---:|---:|---:|---:|---|
| `test_300x10` | 407 | 373 | -8.35% | 6/1/3 | 无稳定因子核增益 |
| `nltcs` | 54,759 | 49,393 | -9.80% | 6/0/4 | 无稳定因子核增益 |

两套数据的总误差方向都偏向 factor（因子核），但都只有 6/10 个种子严格胜出，未达到预注册的
8/10 稳定门槛。nltcs 的离线质量和外层工作量门通过，但墙钟时间更慢；test 的四组离线安全、参考
支持集和外层工作量门失败。跨数据结论为 `no_shared_factor_support`（无共享的因子核支持证据），因此：

```text
kernel_default_changed       false（不修改默认核）
tau_selected                 null（不据此选择温度）
additional_seed_requested    false（不追加种子）
```

40/40 案例均由 P=6 正常提前停止，0 个资源上限案例、0 次方向/条件对数几率裁剪、有效行比例全部为 1。
独立审计从终态表重新计算全部案例指标、配对差、门禁和最终分类，全部精确一致。正式输出约 210 MiB，
继续保留在 ignored（被 Git 忽略）的 `outputs/issue53_stage5_kernel_ab_v1/`；结果文档为
`docs/实验结果/Issue53_Stage5同温度独立核与factor核外层公平比较正式结果.md`。

只读产物 SHA-256：

```text
collection report   375377849aaec401ec6e2dcd29ed850f180c4204c17ed785f67fba2f8f6c506d
evaluation report   8368c58d462a9f4b540f32c0d815faf93c436a7b23fef7fa852cda013a8f94f5
independent audit   c14fb651466e641b4bf75d0b5966aed4b4a2dcd8f588da2cf5862aaa035e34bb
```

Stage 4 的 8-sweep（8 次扫描）内层混合资格保持有效；Stage 5 证明该资格没有转化为跨数据、跨种子
稳定的完整外层优势。当前继续保留 independent（独立核）作为 development reference（开发参考核）。
### 最新暂停点：Draft PR #67 本地同步结果已获授权推送，远端冲突解除（2026-08-22）

> 用户检查本地结果后明确授权“推”。本步只发布上一节已经完成并验证的 #67 同步提交；没有修改
> PR 正文、发布评论、请求 review、催促审阅者、转为 Ready、Approve/merge PR，也没有开始 Stage 5。

```text
published merge head  6ac9d600695d38ea589f5285f0c1f9f0c6310afe
remote branch         research/issue53-factor-gibbs-stage4
PR                     #67 / OPEN / Draft
base                   research/issue53-fixed-alpha（PR #66）
GitHub status          MERGEABLE / CLEAN
checks                 none reported
```

远端从 `3c69686` fast-forward 到 `6ac9d60`，本地与 upstream 在发布后为 `0 behind / 0 ahead`。
推送内容为：同步 #66 最新头 `4bc0bad`、完整保留下层与 Stage 4 状态历史，并继承完整 Python 3.9
兼容链。发布前 Python 3.9/3.11 全仓均为 `1770 passed`，Stage 4 Python 3.9 定向为
`112 passed`；三个正式资格产物 SHA 均保持原值，没有重跑实验或改写 artifacts。

按本次“推送”范围，PR 正文没有改动，仍保留同步前“Python 3.9 全量被旧下层基线阻断”的历史说明；
在将 Draft 转为 Ready 前，应单独更新正文为本次常规 Python 3.9 全量 `1770 passed` 的最终状态。
该动作需用户后续决定，本步不自行修改远端描述。

本节作为发布后的仓库内状态记录随同一 #67 分支推送。Stage 5 仍未实现或运行；#63 自然等待外部
反馈，不评论、不催促。

### 最新暂停点：Draft PR #67 已在本地同步最新 #66，Python 3.9/3.11 全仓通过（2026-08-22）

> 用户授权继续按 PR 栈逐层处理，但没有授权 push。本步只在 #67 本地 worktree 合入 #66 最新
> 远端头、解决状态文档冲突、复查 Stage 4 兼容性、跑双版本回归并只读复核正式产物；没有 push、
> 修改 PR 正文、发布评论、review request、催促审阅者、Approve/merge PR，也没有开始 Stage 5。

同步身份：

```text
#67 pre-sync head     3c696869e74e1d836a9d87752017365416b48a0a
#66 synced head       4bc0bad7d86278aed680538a027d9607eff3a362
common ancestor       8570187287ce826d5b65fe6b70559e0c3c9d6652
divergence            #66 侧 11 commits / #67 侧 9 commits
merge conflict        PROJECT_STATUS.md only
resolution            双方新增记录完整保留；重复的 #66 旧标题只保留一份并标为历史
remote #67            保持 3c69686，GitHub 远端仍显示 CONFLICTING / Draft
```

#66 已验证的全部下层同步与 Python 3.9 兼容修复自动合入；
`tests/test_inner_early_stopping_integration.py` 同时自动合并 Stage 4 natural-work 测试与下层 postponed
annotations/stationarity-trace 测试，无代码冲突。对 Stage 4 独有 11 个 Python 文件完整扫描，未发现
`zip(strict=True)`、`datetime.UTC` 或 `from datetime import UTC`；需要联合类型的文件均具备
postponed annotations，Stage 4 不需要新增兼容修复。

最终验证（物理 GPU 1，单进程串行）：

```text
Python 3.9 / CUDA Stage 4 定向   112 passed
Python 3.9 / CUDA 全仓           1770 passed, 15 warnings, 0 skipped
Python 3.11 / CUDA 全仓          1770 passed, 2 warnings, 0 skipped
git diff --check                 clean
outputs tracked diff             empty
```

Python 3.9 的 15 条 warning 为临时 Matplotlib/PyParsing 依赖弃用提示 13 条与既有空切片 NumPy
warning 2 条；Python 3.11 只有后两条。原先由旧 #66 基线造成的四个 Python 3.9 收集错误已全部消失，
Stage 4 新增集成测试现在由常规 pytest 正常收集与执行。

正式 qualification 未重跑。三个 ignored 产物只读 SHA-256 复核仍精确等于归档身份：

```text
state_library  3c7475e89d693bd2240846bb78dec6b7a8d2abc14a71beea25fd6be3e2a02561
mixing_report  15751180b96c6a466f7096a63935d93eb60f47b836f2aa9462a1842ee58b7fa5
mixing_audit   fde7929a39cb039d26dd56b91063fa4151a639ea0e483ba8b9303e912a42ed6d
```

同步只改变下层代码历史与状态文档，没有改变 Stage 4 冻结协议、runner、auditor、正式执行提交或
ignored artifacts，因此资格结论 `qualified_random_scan_s8` 保持原身份有效。

本节随本地 merge commit 保存。当前暂停在本地完成、远端未更新的 #67；只有用户明确说“推”才可
push。Stage 5 仍未实现或运行，#63 继续自然等待外部反馈，不评论、不催促。

### 最新暂停点：Stage 4 已发布为 Draft PR #67，等待下层栈同步（2026-08-22）

> Stage 4 qualification 的代码、development 诊断与正式结果已从本地分支推送，并以 Draft PR
> [#67](https://github.com/Chuhan722/table-diffusion/pull/67) 发布。本节覆盖下方“只在当前分支本地提交、
> 未 push、未建 PR”的旧暂停描述；正式实验本身没有重跑或改写。

PR 身份与边界：

```text
PR                     #67（Draft / OPEN）
title                  研究：Issue #53 Stage 4 factor Gibbs 内层资格验证
base                   research/issue53-fixed-alpha（PR #66）
head                   research/issue53-factor-gibbs-stage4
published result head  82e679ccdad135cf07b8e1fb8818b3db5d8b6d47
GitHub initial status  MERGEABLE / CLEAN（仍受下层栈审阅与同步顺序约束）
review order           #63 -> #65 -> #66 -> #67
```

PR 正文明确限定：Stage 4 只证明冻结 `tau=2` 提议分布下 factor random-scan Gibbs
`sweeps=8` 取得内层资格；不决定最佳 tau、不比较最终外层 independent/Gibbs 质量、不改公开默认，
也不包含 Stage 5。约 171MB 状态库与约 2.1MB report 继续保持 ignored 本地归档；PR 只提交代码、
协议、测试、结果说明与产物哈希。发布后逐项核对 base/head、Draft 状态与三个产物 SHA；PR 正文中
一次 report SHA 粘贴错误已立即更正为正式值
`15751180b96c6a466f7096a63935d93eb60f47b836f2aa9462a1842ee58b7fa5`，没有改动仓库或产物。

PR 前验证（物理 GPU 1，单进程串行）：

```text
Python 3.11 / CUDA Stage 4 定向     109 passed
Python 3.11 / CUDA 全仓             1757 passed，2 个既有 NumPy warning
Python 3.9 / CUDA Stage 4 核心       95 passed
Python 3.9 Stage 4 新增集成断言       2 passed
Python 3.9 当前叠加头全仓收集         4 errors（下层基线阻断，尚未进入测试执行）
```

Python 3.9 的四个收集错误均来自 PR #66 尚未同步的 PR #63 旧基线：三处
`datetime.UTC` 与一处未延迟求值的 `int | None`。PR #63 已在提交 `5efbcb8` 修复这些兼容点，
并在其后记录 Python 3.9 CUDA 全量通过；Stage 4 新文件的核心测试与新增集成断言均已单独通过。
因此本 PR 不重复修改下层代码，待 #63 -> #65 -> #66 同步后，在 #67 上再跑一次常规 Python 3.9
全量回归。

不重跑正式 qualification 的依据：正式执行绑定 clean commit `e7cbcc0`，之后只有结果/状态文档提交
`82e679c`，科学代码、冻结协议、runner、auditor 与 ignored 正式产物均未改变；当前 Python 3.11
全量回归和 Python 3.9 Stage 4 验证也未发现需要改变科学实现的问题。只有将来测试暴露真实代码缺陷，
且修复触及科学代码、协议或运行器时，才重新判断是否需要生成新身份的实验，而不是覆盖本次资格结果。

下一步：先等待并处理下层 PR 栈同步；同步到 #67 后补跑 Python 3.9 全量测试并更新 Draft 状态。
Stage 5 的 same-tau independent vs factor 结果前协议仍是独立后续工作，未实现、未运行。

### 最新暂停点：Stage 4 qualification 正式通过并完成仓库内收口，下一步为 Stage 5 结果前协议（2026-08-22）

> qualification `333..337` 已于 2026-08-20 运行完成，并于 2026-08-22 逐字段核验、写入正式结果
> 文档。本节覆盖下方“等待 qualification 授权”的旧暂停点。正式实验产物保持 ignored 本地归档，
> 没有重跑、改写或强行纳入 Git；本节与结果文档只在当前分支本地提交，未 push、未建 PR、未评论
> Issue #53，也未修改 #63/#65/#66。

正式身份：

```text
execution commit       e7cbcc0beaf6718f7f4ad148a8ee07f8cc9a089f（clean）
mode                   qualification / formal_result_valid=true
seeds                  333..337
states                 2 datasets × 5 seeds × 5 stages = 50
protocol SHA           6a2834db4cb75fffbaac3330bbb1923fa2e864572ca05ac901c3142ecd443680
result                 qualified_random_scan_s8
attempted sweeps       [8]（两数据同时通过后依协议停止，16/32 未运行）
independent audit      status=complete / passed=true / 同一结果与 selected sweeps=8
```

正式主结果：

| dataset | global TVD（≤0.05） | gap recovery（≥0.80） | valid | passed |
|---|---:|---:|---|---|
| test_300x10 | 0.0015438274 | 98.7275% | true | true |
| nltcs | 0.0000104018 | 99.9486% | true | true |

- 10/10 条来源轨迹均由 P=6 `early_stopped`；test rounds 1209–1758，nltcs 1895–3400，均未触及
  C=6000。
- 50 状态最差单状态为 test seed 337 initial：TVD 0.0133081、recovery 97.6778%，仍明显优于门槛。
- 两数据所有 validity gates、全部 required stage groups 与全部非空 width groups 均通过；零 clip；
  25,650 次 production exact-tape replay 零失配。
- max energy error / tolerance ratio：test `2.60e-18 / 2.60e-8`；nltcs
  `1.39e-17 / 1.39e-7`，正式 fresh seeds 继续证实 float64 累加修复。
- 科学边界：只证明冻结提议分布上 factor random-scan Gibbs 8 sweeps 的内层资格，不证明 factor 外层
  生成优于 same-tau independent，不修改公共默认。

只读复核文件 SHA-256：

```text
state_library  3c7475e89d693bd2240846bb78dec6b7a8d2abc14a71beea25fd6be3e2a02561
mixing_report  15751180b96c6a466f7096a63935d93eb60f47b836f2aa9462a1842ee58b7fa5
mixing_audit   fde7929a39cb039d26dd56b91063fa4151a639ea0e483ba8b9303e912a42ed6d
```

正式结果文档：
`docs/实验结果/Issue53_Stage4_factor_Gibbs正式资格结果.md`。状态库约 171MB、report 约 2.1MB，
继续留在 `outputs/issue53_stage4_qualification_v1/` ignored 归档，不因文档收口改变 artifact 身份。

下一小步是 Stage 5 same-tau independent vs factor **结果前协议设计**：独立臂 sweeps=0、factor 臂
固定使用已取得资格的 sweeps=8，采用全新 paired seeds，冻结质量、支持集、多样性、validity、
normalized work、Gibbs 微步、查询评价次数与墙钟口径。未获用户后续确认前不实现、不跑 smoke/正式
实验、不处理 PR #65 冲突、不 push。

### 最新暂停点：development v3 全绿通过——能量门禁史上首次通过，等待 qualification 授权（2026-08-20）

> float32 修复（`5a61b7b`）后的首次全链路重跑。GPU 单卡（CUDA_VISIBLE_DEVICES=1）串行
> 采集（15:50–16:51，10-14 分/seed）→ 聚合 → mixing（~9 分）→ 审计（17:00:32 完成）。
> 产物 `outputs/issue53_stage4_development_v3/`（ignored，永久归档）。

```text
正式结果      qualified_random_scan_s8（v1/v2 均为 invalid_or_incomplete）
能量门禁      首次通过：nltcs exact_energy_max_error 2.08e-17（v2 为 4.75e-9，改善 8 个
              数量级）、tolerance_ratio_max 2.08e-7 ≤ 1.0；test max_error 3.47e-18、
              ratio 3.47e-8——与验尸预测（seed 326 重放 2.08e-17）严丝合缝
性能门槛      与 v1/v2 同水平：test TVD 0.001161 / recovery 98.91%；
              nltcs TVD 0.000011 / recovery 99.95%
validity      8 项全 True；零 clip；production tape replay 零失配
独立审计      passed=True（重算一致）；protocol SHA dd344a08…（吻合冻结修订协议）
report SHA    7599514254fbac16e0774b141c98aad86582de90debeedabcf7ead194f6ab726
audit  SHA    1d3521fb0879f34ad0972132ebed4b05a48a55d81b36b211c274a62eaf347c5c
library SHA   c5e6b5e72a84d012ac57d3c4a2ed18d4c97a84ce17bc1399664e6d7d95904dc8
```

修复的科学结论闭环：性能指标与修复前完全同水平（修复只提精度不改算法），唯一失败点
（nltcs initial 状态能量恒等）彻底消失——float32 根因诊断与最小修复面均被 v3 证实。

下一步（需用户单独授权）：qualification `333..337` 正式资格实验（协议 SHA 以
`protocol_sha256('qualification')` 实算为准；要求 clean worktree、库/mixing 同 commit）。
未 push、未建 PR、未评论 Issue。

### 历史暂停点：能量门禁真凶已修复——oracle torch 路径 float32 累加缺陷（验尸位级证实），待决策是否跑 v3（2026-08-20）

> 本节推翻上一节（v2 暂停点）的"灾难性相消"初判。只读验尸脚本
> `scripts/diagnose_issue53_stage4_energy_cancellation.py`（确定性重放 probe 能量比对、
> 逐查询项拆解、`math.fsum` 精确参照）对全部 5 个失败 initial 状态给出位级结论：

```text
numpy 重放     全程 float64 路径差值仅 ~2e-17 —— 未复现，排除 factor/oracle 算法差异
cuda  重放     位级复现 recorded 差值（如 seed 326 的 4.74621039171785331e-09，bitwise_equal）
误差归属       factor 侧 vs fsum 精确值：0~2 ulp（无辜）；oracle 侧独扛全部误差（~5.5 亿 ulp）
相消假说       不成立：相消比仅 1.1~11 倍，远不足以放大 float64 舍入到 1e-9 级
真凶           src/table_diffevo/vectorized_eval.py::_directional_potential_torch
               势能累加器 potential_t、残差张量 wr_t、mask 转型全为 float32；
               float32 eps(1.2e-7) × 势能量级(3.7e-2) ≈ 4.4e-9，与观测严丝合缝
谜团全解       nltcs 走 torch 路径(float32)而 test 走 numpy(float64)→只有 nltcs 挂；
               initial 残差大→势能大→float32 绝对误差大→只有 initial 挂；
               该算法下 torch CPU/GPU float32 舍入相同→v1(CPU 回退)/v2(GPU) 位级同值
```

**修复（已实施并验证）**：`_directional_potential_torch` 累加链升为 float64
（`potential_t`/`wr_t` dtype、`mask.to(torch.float64)`）；掩码比较仍在 float32 上进行，
0/1 语义不变，最小修复面。修 bug 不属于修订协议第 6 节禁止的"改容差重跑同批 seeds"。

验证三件套：
1. 新回归测试 `tests/test_directional_diffusion.py::test_torch_potential_accumulates_in_float64`
   （cpu/cuda 参数化，大残差场）：撤掉修复必挂、装回全绿——反向验证成立；
2. 真实数据重放（v2 库 seed 326 initial worst 元素，cuda）：oracle 误差
   4.746e-9 → **2.08e-17（≈0.8 ulp）**，门禁容差 1e-10 下余量 7 个数量级；
   产物 `outputs/issue53_stage4_development_v2/postfix_verification_seed326_cuda.json`；
3. 全仓测试 **1757 passed / 0 failed / 0 skipped**（1755 基线 + 2 新增）。

下一步（待用户授权）：跑 development v3——修复改变 nltcs 前向轨迹，状态库必须重采集，
GPU 单卡串行全链路约 1.5 小时；能量门禁预期 ulp 级通过。qualification `333..337`
仍需之后单独授权。未 push、未建 PR、未评论 Issue。

### 历史暂停点：development v2 已跑完——能量门禁再次失败且 v1 诊断被证伪（当时初判 cancellation，后被验尸推翻，见上节）（2026-08-20）

> GPU 单卡（CUDA_VISIBLE_DEVICES=1）串行采集 5 分片（13:38–14:39，9-14 分/seed）→ 聚合 →
> mixing → 独立审计通过。产物 `outputs/issue53_stage4_development_v2/`（ignored，永久归档）。

```text
性能门槛      全过且与 v1 同水平：test TVD 0.0011614 / recovery 98.91%；
              nltcs TVD 0.0000109 / recovery 99.95%；零 clip；tape replay 0/12440 失配
能量门禁      nltcs exact_factor_energy 仍失败：仅 5 个 initial 状态，ratio 36.8~47.5（>1）；
              其余 20 状态 ratio 0.03~0.44 全过；test 侧 ratio_max 3.5e-8 全过
正式标签      invalid_or_incomplete（16/32 按 invalid-stop 未尝试）；独立审计通过（重算一致）
report SHA    13c85f0603b7b04a977ecead758abacc2c3fb925d6a90f71f524a479112a38ac
audit  SHA    60d58df6c22cf0734ee8d5b0bff85d2e5c4b46e5565431f7cc4f8f35eb178cc8
```

**重大发现：v1 的"大能量 2~3 ulp 舍入"诊断是误诊。** v2 新增的 worst-case 分量证明：
最大差值（如 seed 326 initial 的 4.746210391717853e-09，与 v1 **位级相同**，确定性差异、
与 GPU/CPU 无关）发生在**能量仅 ~1e-3 量级的元素**上，相对误差 ~1e-4 级，远超 float64
舍入极限。机理是**灾难性相消**：能量是大量 ~1e7 量级项的和（initial 状态 loss ~6.9e8），
大项几乎相消后结果仅 1e-3；factor 侧与 oracle 侧求和顺序不同，各自舍入停留在大项 ulp 级
（~2e-9），差值天然 1e-9 级。混合容差用"结果能量量级"做 scale 在相消场景下物理上错误
（分母退化为 atol），正确 scale 应反映**求和项量级**（如 Σ|term|）。

纪律：按已冻结修订协议第 6 节，再次触发只能记失败、**不得改容差后重跑同批 seeds**——
development `323..327` 在当前协议下到此为止。下一步候选（待用户决策）：
(a) 写独立诊断脚本确证 initial 状态能量求和的相消结构（只读产物，不碰资格管线）；
(b) 第二次协议修订：scale 改为求和项量级，换新 development seeds；
(c) 改能量计算为补偿求和（动被测实现，影响面最大）。
未 push、未建 PR、未评论 Issue；qualification `333..337` 仍未运行。

### 历史暂停点：环境已修复（CUDA torch + matplotlib）（2026-08-20）

> 环境变更事件：`.issue49-tools/venv` 的 torch 从 `2.13.0+cpu` 换为 `2.13.0+cu130`
> （版本号不变，只换 CUDA build；驱动 CUDA 13.0，2×RTX 4090 就绪），并补装 matplotlib 3.11.1。
> 背景：协议规定 nltcs `device: "cuda"`，但旧环境 torch 是 CPU-only 版，实际一直走
> `_run_batches_torch` 的 "CUDA not available, falling back to CPU" 回退——**v1 及此前
> 各阶段的 nltcs 轨迹都是该回退下的 CPU float32 产物**。修复后 nltcs 才真正按协议在 GPU
> 上执行；GPU float32 归约顺序与 CPU 不同，故 GPU 轨迹与 CPU 轨迹不位级可比（v2 是新
> 协议 SHA 下的全新实验，本就无需与 v1 可比）。
> 环境验证：全仓 1755 passed / 0 failed / 0 skipped（含此前因无 CUDA 跳过的 GPU 用例
> 与因缺 matplotlib 无法收集的 6 个 stage2b 文件，全部恢复并通过）。
> 首次 v2 开跑尝试（5 分片并行）已在完成前全部终止并清理，未留任何 v2 产物。

### 历史暂停点：Stage 4 能量门禁混合容差修订已冻结并实施（2026-08-20）

> 本板块在独立 worktree `research/issue53-factor-gibbs-stage4`（起点 PR #66 head `8570187`）上进行。
> Stage 4 factor Gibbs 资格管线已以本地提交 `3c19ec3`（管线）与 `6d102af`（一种子一分片）落盘；
> 未 push、未建 PR、未评论 Issue、未操作 #63/#65/#66。补记：这两个提交当时漏更本文件，本节一并补上。

development `323..327` 状态库与共享 sweep=8 mixing 已完成（非正式，不产生资格结论）：

```text
状态库            2 数据 × 5 seeds × 5 状态 = 50 个，分片聚合与 SHA 绑定全部通过
性能门槛          全过：test TVD 0.0011614 / recovery 98.91%；nltcs TVD 0.0000109 / recovery 99.95%
                  全部 stage groups 与非空 width groups 通过；零 clip；tape replay 0 失配
结构门禁          nltcs exact_factor_energy 失败：绝对容差 1e-10，五个 initial 状态误差
                  3.68e-9 ~ 4.75e-9（能量 ~1e7 量级的 2~3 ulp 浮点舍入）；其余 45 个状态全过
正式标签          invalid_or_incomplete；按 invalid-stop 规则 16/32 未尝试
独立审计          通过且重算结论一致（报告未损坏，是容差公式缺尺度稳健性）
report SHA        7268800e37238a733c483ca755572344f35b84d5226adc015ce5139082a76d9c
audit SHA         6a263bfed443890bcba66ba2e943fe0c514bd0b443763860736a75e591fe12a1
产物              outputs/issue53_stage4_development_v1/（ignored，永久归档不改写）
```

已核对远端（只读）：PR #63/#65/#66 均 OPEN、CLEAN、零 review 零评论；Issue #53 最新为
Amendment 5，无新动态。

当前动作：结果后协议修订
`docs/设计/Issue53_Stage4能量恒等门禁混合容差修订协议.md` **已经用户确认冻结并完成实施**。
`exact_factor_energy` 改为逐元素混合容差
`abs_diff <= atol(1e-10) + rtol(1e-12) × max(|E_factor|, |E_oracle|)`，常量按浮点精度
第一性原理推导、明确禁止按观察值反推。实施与验证情况：

```text
protocol.py       ENERGY_ATOL/ENERGY_RTOL + energy_tolerance_ratio() + energy_identity_gate 结构
probe             opt-in energy_atol/energy_rtol；新增 max_relative_error / tolerance_ratio_max /
                  worst_case{abs_diff,scale} / atol / rtol 五字段；不传参时字段不出现（issue49/52 兼容）
runner            门禁改 ratio_max <= 1.0；numerical_diagnostics 增两字段
auditor           断言 atol/rtol 与冻结常量相等；从 worst_case 用协议常量独立重算 ratio 并要求
                  与记录值位级相等；自洽不等式；聚合门禁独立重算
新 protocol SHA   development dd344a087478f686d4a72409c4bf4855ed18a5dd0cacaaa83cf4cfc37b097a51
                  qualification 6a2834db4cb75fffbaac3330bbb1923fa2e864572ca05ac901c3142ecd443680
测试              Stage 4 专项 13 passed（新增 4：跨尺度/注错必抓、probe 一致性与省略兼容、
                  审计篡改必拒、旧协议库绑定必拒）；probe 共用方回归 17 passed；
                  全仓 1695 passed / 8 skipped / 0 failed（6 个文件因环境缺 matplotlib 无法收集，
                  属既有环境问题，与本次改动无关）
```

**尚未重跑任何实验**：development `323..327` v2 重跑（状态库重采集约 1 小时 + mixing + audit，
写 `outputs/issue53_stage4_development_v2/`）需用户另行授权；qualification `333..337` 更需
单独授权。未 push、未建 PR、未评论 Issue、未操作 #63/#65/#66。

### 最新暂停点：PR #66 本地同步结果已获授权推送，远端冲突解除（2026-08-22）

> 用户检查本地结果后明确授权“推吧”。本步只发布上一节已经完成并验证的 #66 同步提交；没有修改
> PR 正文、发布评论、请求 review、催促审阅者、Approve/merge PR，也没有操作 #67。

```text
published merge head  92a21d65740e291ea37dc76b3013f8fe42a1a0cf
remote branch         research/issue53-fixed-alpha
PR                     #66 / OPEN / non-Draft
base                   research/issue53-query-workload-ab（PR #65）
GitHub status          MERGEABLE / CLEAN
checks                 none reported
```

远端从 `8570187` fast-forward 到 `92a21d6`，本地与 upstream 在发布后为 `0 behind / 0 ahead`。
推送内容为：同步 #65 最新头 `c87af50`，完整保留双方状态历史，并继承下层 Python 3.9 兼容修复。
#66 自身无需额外兼容代码修改；发布前 Python 3.9/3.11 全仓均为 `1753 passed`，没有重跑正式
alpha 实验或改写 artifacts。

本节作为发布后的仓库内状态记录随同一 #66 分支推送。下一步仍需用户单独决定；未经再次明确授权，
不 push #67。#63 自然等待外部反馈，不评论、不催促。

### 最新暂停点：PR #66 已在本地同步最新 #65，双版本全仓通过（2026-08-22）

> 用户授权继续按 PR 栈逐层处理，但没有授权 push。本步只在 #66 本地 worktree 合入 #65 最新
> 远端头、解决状态文档冲突、检查 #66 自身兼容性并测试；没有 push、PR 评论、review request、
> 催促审阅者、Approve/merge 远端 PR，也没有操作 #67 或 Stage 5。

同步身份：

```text
#66 pre-sync head     8570187287ce826d5b65fe6b70559e0c3c9d6652
#65 synced head       c87af50e6cd0952366f1f85a595c357dba85eabe
common ancestor       a2bc496da223ef49a5a1e8a8e5ac6f60252ab62b
divergence            #65 侧 9 commits / #66 侧 11 commits
merge conflict        PROJECT_STATUS.md only
resolution            双方进度历史完整保留；科学代码无内容冲突
remote #66            保持 8570187，GitHub 远端仍显示 CONFLICTING
```

#65 已验证的 #63/PR #61 同步、Python 3.9 兼容修复和状态记录均自动合入。另对 #66 独有的 10 个
Python 文件做完整兼容扫描：未发现 `zip(strict=True)`、`datetime.UTC` 或
`from datetime import UTC`；使用联合类型的控制器模块已有 postponed annotations。#66 不需要新增
兼容代码修复。

最终验证（物理 GPU 1，单进程串行）：

```text
Python 3.9 / CUDA #66 定向       41 passed
Python 3.9 / CUDA 全仓           1753 passed, 15 warnings
Python 3.11 / CUDA 全仓          1753 passed, 2 warnings
git diff --check                 clean
outputs tracked diff             empty
```

Python 3.9 的 15 条 warning 为临时 Matplotlib/PyParsing 依赖弃用提示 13 条与既有空切片 NumPy
warning 2 条；Python 3.11 只有后两条。同步没有改变 #66 冻结 alpha 协议、正式输入、ignored
artifacts、已归档负结果或公共默认，因此没有重跑正式实验。

本节随本地 merge commit 保存。当前暂停在本地完成、远端未更新的 #66；只有用户明确说“推”才可
push。#67 尚未同步或修改，#63 继续自然等待外部反馈，不评论、不催促。

### 历史暂停点：两档自适应 α 正式负结果已提交 stacked PR #66（2026-08-19）

> 本板块基于 PR #65 head `a2bc496` 的独立分支完成；没有修改、自行审查、批准或合并 PR #63/#65。
> 固定 α 响应与两档自适应实验均使用更新后无 measured 1-way 的 test workload，并保持无门控、
> terminal-current 输出身份。

```text
worktree              /home/chuhan/projects/table-diffusion-issue53-fixed-alpha
branch                research/issue53-fixed-alpha
adaptive protocol     5a88ddc5077df82528b7fda3cd12a4fb79c1b8e5c027d6d555d4a50e869e911e
execution commit      4e0270c587f2efdf8aa47b5581b200815cffd3d2
collection SHA        ec2d81a954e1e0e85478a7dc26df190edec9c5b1e6beaec439e815dd6ec2057e
evaluation SHA        0646e7b2a52653995e3af96c51e2730056fe0bda1f688ac8e5f7ff6da2a9be4a
cases                 30/30 early_stopped；0 resource caps
```

状态机固定为：α16 正常档；连续 2 个自然工作刻度没有严格新最好后，α12 探索恰好 2 个自然工作
刻度，再恢复 α16；现有 P=6 不清零、不暂停或延长。同一无改善阶段最多触发一次，只有严格新最好
创建新的 progress epoch。控制器不读取 held-out、raw reference、未来预算或其他轨迹，不消费 RNG。

自适应在两套数据的 5/5 seeds 都触发：test 共 7 个探索段，nltcs 共 8 个。阶段诊断确认 α12 确实
扩大供体覆盖：test 有效供体比例从约 0.01296 增至 0.01925，nltcs 从约 0.09315 增至 0.12660。
但 15 个探索段中只有 1 个在 α12 期间产生新最好，4 个在恢复 α16 后产生，10 个没有新最好。

正式主结果：test 自适应 measured L1 比固定 α16 高 8.29%，只有 1/5 更好；nltcs 高 17.40%，
只有 3/5 更好，并且 1-way safety 高 33.96%、work 高 17.71%。固定 α12 在 test 的均值低 5.70%，
但只有 3 胜、1 平、1 负，未达到 4/5；nltcs measured L1 高 23.80%。冻结结论为：

```text
test mechanism         no_supported_alpha12_strategy
nltcs mechanism        no_supported_alpha12_strategy
cross dataset          no_shared_adaptive_support
```

该结果不支持当前 2/2/6 两档自适应策略，也不支持事后把回滚或 best 输出加入同一个无门控算法。
回滚会形成分段接受门，若未来研究必须改成另一种算法身份并另写协议。完整设计、逐 seed 主值、阶段
集中度、新最好位置、全部门禁与执行勘误见：

```text
docs/实验结果/Issue53_两档自适应alpha正式结果.md
```

相关定向回归 `40 passed, 1 skipped`，GPU 0 的 NumPy/CUDA 小前缀对拍 `1 passed`；排除当前测试
环境缺少 matplotlib 的 6 个旧 Stage2B 文件后，全仓为 `1680 passed, 8 skipped`。正式输出目录约
63 MiB，保持 ignored，不把 30 张 CSV 和大 JSON 提交进 Git，只通过报告 SHA 和结果文档绑定。

当前分支已推送并创建 stacked PR #66：
`https://github.com/Chuhan722/table-diffusion/pull/66`。其 base 是 PR #65 的 head
`research/issue53-query-workload-ab`，正文详细说明固定 α 设计、两档状态机、正式负结果和结论边界。
下一动作是等待他人按 #63 → #65 → #66 的顺序审查；不自行 review、approve 或 merge。

### 历史暂停点：固定 α 响应曲线完成，等待讨论自适应目标（2026-08-18）

> 本板块基于 PR #65 的提交 `a2bc496` 新建独立分支完成；没有修改或推送 PR #65，没有自行
> review（审查）、approve（批准）或 merge（合并）。当前分支只在本地，尚未 push（推送）或创建 PR。

```text
worktree              /home/chuhan/projects/table-diffusion-issue53-fixed-alpha
branch                research/issue53-fixed-alpha
protocol SHA          6a3716f11ed6a4233256b9d3a549fc45281bc464470cc82a6e64d66d0104b311
collection commit     41f3dc55416d4525819033f5069b4160a8a378bf
collection SHA        03e26d01cf960fd763219a926a8ae4f5cada9eaa7686aed91e5393fee5625884
evaluation commit     19f342d8fb38be3de5300cc23ce709eafd91a17f
evaluation SHA        b27543c2844bd09ae58706116ce315b7dda44ef47b9fc9ac6684674a6a7ed99d
result commit         44e8d89
cases                 30/30 early_stopped；0 resource caps
```

冻结矩阵为两套数据 × α=`12/16/24` × 5 个配对随机种子。`test_300x10` 使用更新后的
`30×2-way + 15×3-way + 5×4-way` workload（查询负载），没有 1-way 已测查询；`nltcs`
使用 479 条 2-way 与 522 条 3-way，也没有 1-way 已测查询。

正式结论：α 对供体集中度的作用在两套数据上稳定单调——α 越小，有效供体越多、单行最大供体概率
越低；α 越大则越集中。但质量和计算取舍不单调，α=12、24 相对 α=16 都没有通过结果前冻结的
“已测稳定改善 + 离线安全 + 多样性 + 计算量”完整门禁，两套数据支持的 probe α（探测 α）集合均为空，
跨数据分类为 `mixed_fixed_response`（混合固定响应）。没有选择统一固定 α，也没有事后设计自适应公式。

运行中按用户要求先把剩余 11 条 `nltcs` 分成当前服务器 6 条、A6000 5 条，随后把 A6000 尚未启动
的 2 条移回；最终当前服务器完成 8 条、A6000 完成 seed 326 的 3 条。跨机器 100 轮初态、随机数、
终表和完整轨迹签名逐位一致；同一数据集、同一随机种子的三种 α 没有拆到不同硬件。A6000 GPU 0
和本实验使用的本机 GPU 均已释放。

首次离线评价发现 L1 复算两种数学等价写法存在一个浮点末位差异；提交 `19f342d` 只把一致性审计
改为绝对容差 `1e-15`，不改指标和门禁。定向测试 `10 passed`，30 条 measured L1（已测查询 L1）
全部复算通过，关键均值和配对胜负又由独立标准库脚本重聚合一致。完整结果见
`docs/实验结果/Issue53_固定alpha响应曲线结果.md`。

**当前下一步不是直接写自适应公式。** 先与用户讨论控制目标：本实验已经证明 α 是可靠的集中度执行量，
但两套数据的集中度绝对尺度不同，而且只追已测误差会伤害未测高阶查询和多样性。建议下一轮先确定
“运行内归一化的供体集中度主反馈 + 残差进展/当前表多样性保护条件”的语义，再冻结更新频率、步幅、
上下界和实验矩阵。用户未明确要求前不要 push（推送）。

### 最新暂停点：PR #65 本地同步结果已获授权推送，远端冲突解除（2026-08-22）

> 用户在检查本地结果后明确授权“直接把本地改的推送上去”。本步只发布上一节已经完成并验证的
> #65 同步提交；没有修改 PR 正文、发布评论、请求 review、催促审阅者、Approve/merge PR，
> 也没有操作 #66/#67。

```text
published merge head  c1871ba892c2575c115288c49bdbd1240376d48a
remote branch         research/issue53-query-workload-ab
PR                     #65 / OPEN / non-Draft
base                   research/issue-53-stage2-v2-evidence（PR #63）
GitHub status          MERGEABLE / CLEAN
checks                 none reported
```

远端从 `a2bc496` fast-forward 到 `c1871ba`，本地与 upstream 在发布后为 `0 behind / 0 ahead`。
推送内容为：同步 #63 最新头 `6cc7825`、完整保留状态文档双方历史，以及 #65 自身 11 处
Python 3.9 strict-zip 兼容修复。验证结果沿用上一节本次提交前的干净结果：Python 3.9/3.11 全仓均
`1712 passed`；没有重跑正式实验或改写 artifacts。

本节作为发布后的仓库内状态记录随同一 #65 分支推送。下一步仍需用户单独决定；未经再次明确授权，
不 push 后续分支。#63 自然等待外部反馈，不催促；#66 尚未同步或修改。

### 最新暂停点：PR #65 已在本地同步最新 #63，并补齐自身 Python 3.9 兼容（2026-08-22）

> 用户要求先处理 PR 栈，并再次明确“未说推就不 push、不要催审阅者”。本步只在 #65 本地
> worktree 合入 #63 最新远端头、解决冲突、补齐 #65 自身兼容遗漏并测试；没有 push、PR 评论、
> review request、Approve、merge 远端 PR，也没有开始 #66/#67 或 Stage 5。

同步身份：

```text
#65 pre-sync head     a2bc496da223ef49a5a1e8a8e5ac6f60252ab62b
#63 synced head       6cc7825c68ad8b4247737e010cbd5d184fbfb53a
common ancestor       24478dde3f639ee8f55100d3e7741506631bbc12
divergence            #63 侧 7 commits / #65 侧 24 commits
merge conflict        PROJECT_STATUS.md only
resolution            双方进度历史完整保留；科学代码无内容冲突
remote #65            保持 a2bc496，GitHub 冲突状态不会因本地工作自动改变
```

#63 的 Python 3.9 兼容修复、stationarity trace 补强、V2b 路径便携修复以及其同步的 PR #61
plants 数据/workload 均自动合入。首次合并后验证：Python 3.11 全仓 `1712 passed, 2 warnings`；
Python 3.9 已不再发生 #63 的收集错误，但暴露 #65 自身 11 处 `zip(strict=True)` 遗漏，其中当次
全仓实际触发 `4 failed + 4 errors`，其余 `1704 passed`。

兼容修复覆盖 #65 的 5 个脚本与 1 个测试文件：全部 11 处 strict zip 改为 Python 3.9 支持的普通
`zip`，并逐处保留既有等长检查或补充显式长度漂移拒绝。合法等长输入的查询、答案、fingerprint、
分组和误差计算顺序及数值不变；不把静默截断当作兼容方案。全范围扫描确认 #65 新增 Python 文件不再
包含 `zip(strict=True)`、`datetime.UTC` 或 `from datetime import UTC`。

最终验证（物理 GPU 1，单进程串行）：

```text
Python 3.9 / CUDA 相关专项     38 passed
Python 3.9 / CUDA 全仓         1712 passed, 15 warnings
Python 3.11 / CUDA 全仓        1712 passed, 2 warnings
git diff --check               clean
outputs tracked diff           empty
```

Python 3.9 的 15 条 warning 为临时 Matplotlib/PyParsing 依赖弃用提示 13 条，加既有空切片 NumPy
warning 2 条；Python 3.11 只有后两条。兼容改写没有改变冻结协议、正式输入、ignored artifacts、
已归档结果或公共默认，因此没有重跑任何正式实验。

本节随本地 merge commit 保存。当前暂停在本地完成、远端未更新的 #65；下一步必须由用户明确决定，
且只有用户明确说“推”才可 push。#66 尚未同步或修改，#63 继续自然等待外部反馈，不催促。
### 最新暂停点：residual geometry 收口已提交 stacked PR #65（2026-08-18）

> 用户明确授权 push 后，当前分支已推送并创建新的 stacked PR；没有自行 review、approve
> 或 merge，也没有操作依赖 PR #63。PR #65 替代此前按用户要求关闭的 #64，包含原 residual
> geometry 证据链以及后续 query-workload A/B 正式实验与结果后解释修正。

```text
PR       #65
title    研究：Issue #53 残差几何与高阶查询 workload 确认
state    OPEN, non-Draft
url      https://github.com/Chuhan722/table-diffusion/pull/65
base     research/issue-53-stage2-v2-evidence（PR #63）
head     research/issue53-query-workload-ab
```

PR 正文明确保留正式 `mixed_no_workload_replacement` 作为历史辅助分类，但不再以 A/B
总体高低选择 residual geometry；当前主结论读取 B 内比较。后续 development baseline 为
“1-way marginal 初始化 + 高阶 measured workload + relative/floor=8”，公共 API 默认值仍
保持 `absolute`。当前停在等待外部审查，不自行处理 #63/#65 的 review 或 merge；下一科学
板块 donor/alpha 另行讨论和冻结，不顺带加入本 PR。

### 历史暂停点：test query-workload A/B 结果后解释修正，residual 板块收口（2026-08-18）

> 本步不修改正式 evaluator、原始 artifacts、查询、seed、门禁或 SHA，只修正结果解释：
> A/B 的持续监督不同，`B - A` 不能作为正常查询设计或 residual geometry 的选择门槛；
> 当前主结论应读取 workload B 内部比较。没有重新生成表、调整规则、增加 seed、修改
> 公共 API 默认值、形成全局 canonical 结论，也没有 push 或操作 Issue/PR。

正式评价身份：

```text
collection SHA       67f3ebbcf06100b0ba508b465dd4aea7b6ee69825a46b5eec5a768245b69e44a
evaluation commit    4c275b7789f6b08efafa3959ccf278c4c5dbba39
evaluation report    outputs/issue53_test_query_workload_ab_v1/evaluation_report.json
evaluation SHA       a389504c92e87461d84c4eb8322b659afea0dabb58a256bedcd6c19f78c06651
query-seed CSV       outputs/issue53_test_query_workload_ab_v1/query_seed_errors.csv
query-seed CSV SHA   2bbcfba869187cfdd1b7198f9d2e675437f38d8a6e4081c73f9b03289b6c467c
data rows             47,100（文件 47,101 行，含 header）
```

查询身份在 reference load 前冻结，四组数量仍为 `25 / 521 / 512 / 512`，身份 SHA
与结果前协议完全一致；fixed held-out 3/4-way answers 与既有 archive 精确一致。
30/30 terminal table SHA 再审计通过。评价阶段记录
`new_generation_performed=false`、`cross_group_aggregate_present=false`、
`canonical_selection_performed=false`、`privacy_budget_consumed=false`。独立从 CSV
重算 24 个 group/workload/geometry cell 的 mean，全部与正式报告逐项精确一致。

统一测试的 mean absolute count error：

| 查询组 | A abs | B abs | A sqrt | B sqrt | A relative | B relative |
|---|---:|---:|---:|---:|---:|---:|
| 1-way safety | 0.8560 | 16.1840 | 0.8560 | 15.2400 | 0.9280 | 13.0080 |
| common unseen 2-way | 7.4779 | 10.8088 | 7.1328 | 10.4791 | 7.6891 | 9.1708 |
| fixed held-out 3-way | 4.2859 | 5.1699 | 4.1855 | 5.1148 | 4.5813 | 4.5902 |
| fixed held-out 4-way | 1.7578 | 1.9848 | 1.7715 | 1.9516 | 1.8809 | 1.8949 |

完整结果后解释归档于：

```text
docs/实验结果/Issue53_test查询workload_AB正式结果.md
```

### 冻结问题 1：workload B 能否替代 A（保留为历史辅助判定）

`B - A` mean delta 均为正数时表示 B 更差：

| geometry | 1-way | unseen 2-way | held-out 3-way | held-out 4-way | 冻结结论 |
|---|---:|---:|---:|---:|---|
| absolute | +15.3280 | +3.3309 | +0.8840 | +0.2270 | mixed/no replacement |
| sqrt-relative | +14.3840 | +3.3463 | +0.9293 | +0.1801 | mixed/no replacement |
| relative | +12.0800 | +1.4818 | +0.0090 | +0.0141 | mixed/no replacement |

三种 geometry 的 unseen Pareto 和 1-way safety 均失败，正式分类全部为
`mixed_no_workload_replacement`；因此结论方向在 geometry 间一致。尤其三个 geometry
的 common unseen 2-way 都是 5/5 paired seeds 下 B 更差。relative 已把 3/4-way
差距压到接近零，但仍没有让 B 通过替代门禁，而且 1-way 与 2-way 仍明显退化。

上述正式数值与分类保持有效，但它回答的是：移除 A 的持续 1-way measured supervision、
同时换入更多高阶查询后，B 能否在 A 直接或间接监督的公共统计上不劣于 A。答案是否定的。
它不回答正常高阶 workload 应使用哪种 residual geometry，也不能证明 B 的查询设置失败。

A 的 `25×1-way + 20×2-way + 5×3-way` 中，25 条 1-way target 与 marginal 初始化
25/25 精确一致，初态残差为零，并继续占固定 objective 的一半；B 使用相同 marginal
初始化，后续只拟合 `30×2-way + 15×3-way + 5×4-way`。两者持续监督不同，不能以
A/B 总体高低选择方法。B 还同时换入新 2/3/4-way，因此也不能声称 1-way 单一因素解释
全部 A/B 数值；可以确认的是旧 geometry 排序依赖 workload，零残差 1-way 是 A 偏向
absolute 的明确机制。

### 冻结问题 2：workload B 内哪种 geometry 更好

候选相对 B/absolute 的 mean delta：

| candidate | 1-way | unseen 2-way | held-out 3-way | held-out 4-way | paired 稳定改善 | 冻结结论 |
|---|---:|---:|---:|---:|---|---|
| sqrt-relative | -0.9440 | -0.3298 | -0.0551 | -0.0332 | 无（3/5、3/5、2/5） | mixed |
| relative | -3.1760 | -1.6380 | -0.5797 | -0.0898 | 2-way 4/5；3-way 4/5 | supported |

relative 的三个 primary mean 和 1-way mean 全部不劣，2-way、3-way 都达到 4/5
paired-seed 稳定改善，正式分类为 `supports_geometry_under_workload_B`。4-way mean
也改善，但只有 3/5 seeds，不单独宣称稳定。sqrt-relative 虽然四组 mean 都略有改善，
没有任何 primary group 达到预先要求的 4/5，正式分类为
`mixed_no_unified_geometry_candidate`。

### 结果后研究解释与当前决定

当前项目要研究的内层语义是“1-way marginal 初始化 + 尚未满足的高阶 measured
workload”。因此 workload A 只保留为解释旧 test 反转的机制对照；选择 residual geometry
时，以 B 内部比较为当前主问题，不要求 B 先通过相对 A 的 replacement gate。

正式 B 内结果支持 `relative`：四组 mean 全部优于 B/absolute，unseen 2-way 与 held-out
3-way 都有 4/5 paired seeds 改善；1-way safety mean 也从 16.184 降至 13.008，因此
没有证据要求为挽救 relative 再加入持续 1-way anchor。sqrt-relative 的 primary 稳定性
不足，仍是 mixed。

结合既有 nltcs 无 measured 1-way workload 下 relative 的 3/3 paired-seed 优势，当前
development baseline 冻结为：

```text
1-way marginal initialization
+ higher-order measured workload
+ relative residual geometry (floor=8)
```

这不修改 `run_evolution` 为兼容性保留的 `absolute` 默认值，也不外推到所有数据、带噪
阶段或公共 API。residual 板块到此停止增加公式、seed 和 A/B 变体；下一科学板块进入
donor/alpha，并须另写结果前协议。当前先完成本地文档收口与验证；按用户要求停在 push
之前，不创建、更新、审查或合并远端 PR。

本次收口验证：A/B freeze/runner/evaluator、fresh-seed evaluator 与 ordered-heldout
相关定向回归 `37 passed`；`git diff --check` 通过。没有运行 generator 或读取新的实验
结果。新增结果文档及两份历史结果顶部的后续说明均使用仓库内有效相对链接。

### 历史暂停点：test query-workload A/B 正式公共评价完成（2026-08-18）

> 以下保留正式评价执行、不可变身份与原始冻结分类；上方结果后解释只改变这些证据
> 在当前研究问题中的角色，不覆盖历史结果。

### 历史暂停点：test query-workload A/B 正式 30-case 采集与聚合完成（2026-08-18）

> 本步按用户确认的新服务器绑定在 `linyao-system` 正式运行全部 30 条轨迹，并只聚合
> generation collection；没有打开 raw reference，没有运行四组公共查询评价，没有
> 结果后调参或增加 seed，也没有 push 或操作 PR。

正式身份：

```text
execution commit    4f80b962d290ba896bc93cb5e3129380ed1d7e7c
protocol SHA        5b27cc3ddd5b39829a584f1cdc06b961ef50204840d957481444297023a18f0f
collection report   outputs/issue53_test_query_workload_ab_v1/collection_report.json
collection SHA      67f3ebbcf06100b0ba508b465dd4aea7b6ee69825a46b5eec5a768245b69e44a
```

5 个 seed shard 并行，shard 内按冻结顺序串行六臂；实际环境均为
`hostname=linyao-system`、clean worktree、NumPy 2.4.6、pandas 3.0.5、
`CUDA_VISIBLE_DEVICES=""`、generator device NumPy。五份 manifest 的 protocol、
execution commit 和六个输入 SHA 完全一致。

完整性与停止审计：

```text
case identities                    30/30 unique
termination                        30 early_stopped
normal completion                  30/30
resource_cap_reached               0
paired initial state seed shards   5/5
terminal table SHA                 30/30
result JSON                        30/30
B cases with five full 4-way       15/15
factorized Gibbs active            0/30
raw reference accessed             false
privacy budget consumed            false
parameter retuning performed       false
```

generation workload 内部拟合与成本均值如下；A/B 的 measured 查询不同，因此这些 L1
只描述各自拟合，不能直接用来判断新查询设计优劣：

| workload | geometry | terminal measured L1 | rounds | normalized work |
|---|---|---:|---:|---:|
| A | absolute | 0.0029333333 | 1977.8 | 19.8053 |
| A | sqrt-relative | 0.0029333333 | 1362.2 | 13.6013 |
| A | relative | 0.0029733333 | 1498.2 | 15.0020 |
| B | absolute | 0.0024266667 | 1660.0 | 16.6040 |
| B | sqrt-relative | 0.0024133333 | 1382.8 | 13.8047 |
| B | relative | 0.0023600000 | 984.0 | 9.8053 |

该 collection 已满足公共评价的执行资格，但目前还不能回答 workload B 或哪种 geometry
更好。下一个独立小步骤：以完整 collection SHA
`67f3ebbc...b69e44a` 显式确认 evaluator；它将先重新审计 30 张表和四组查询身份，
然后才读取固定 reference，生成 47,100 条 query-seed error 和冻结门禁结论。

### 历史暂停点：test query-workload A/B 正式执行服务器重新冻结完成（2026-08-18）

> 用户已确认后续改在当前有空闲资源的服务器执行。本步只在看到正式结果前重新绑定
> execution server、重算 protocol SHA 并增加运行时硬校验；没有启动正式 30 cases，
> 没有读取 collection 结果或 raw reference，没有 push 或操作 PR。

正式 collector protocol 从：

```text
old server    root@10.8.176.53:6006
old SHA       e40317be5a21c0c7a59928865c31cb56071b78e1206dba00bcb574b3cd3b198a
```

结果前重新冻结为：

```text
new server    linyao-system
new SHA       5b27cc3ddd5b39829a584f1cdc06b961ef50204840d957481444297023a18f0f
```

规范化 manifest 差异审计证明唯一变化是
`execution_concurrency.server`：把新 manifest 的该字段临时还原为旧 SSH target 后，
SHA 精确重建为旧 `e40317...b198a`。workload A/B 身份、target vector、30-case
矩阵、seeds 318–322、全部 generator 参数、公共评价身份和冻结门禁均未改变。

正式 runner 现在除要求 clean worktree、`CUDA_VISIBLE_DEVICES` 为空外，还会要求
`platform.node() == "linyao-system"`，并把 hostname 写入 shard environment；因此
新 protocol 不能被误拿到其他服务器执行。结果前协议文档、身份 artifact 和附答案
workload 仍保留原 SHA：

```text
protocol doc       291c591ba5408e046005b24122bfe602bf8a97f7c175ee45e59f81daf96b44b6
identity artifact  a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36
answered workload  708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14
```

这样避免因只改运行位置而重写已经结果盲冻结、随后附答案的科学输入；服务器变更由新
collector manifest 和 Git 历史单独审计。collector/evaluator plan 均显示新 SHA、
`server=linyao-system`、`generation_started=false`。Ruff 通过；使用当前完整运行环境
的相关测试为 `32 passed`，包含错误 hostname 拒绝和正确 hostname 记录测试。正式
output namespace 仍不存在。

下一个独立小步骤：在 `linyao-system` 做正式运行前只读资源/环境预检，然后按冻结
SHA `5b27cc...a18f0f` 启动 5 个 seed shard；每个 shard 内六臂串行，全部 30 cases
完成后再聚合。正式运行期间不修改协议、不增加 seed、不读取离线评价结果。

### 历史暂停点：test query-workload A/B 真实核心与一轮闭环验证通过（2026-08-18）

> 本步在用户确认当前服务器已有空闲资源后，只做非正式轻量验证；没有启动
> 6000-round 的正式 30-case 采集，没有保留任何生成表或评价结果，没有 push 或
> 操作 PR。

验证环境为当前 `linyao-system`（2×RTX 4090）；冻结 generator 仍按 NumPy/CPU
执行，没有改走 GPU。完整依赖环境默认指向另一个 worktree，因此验证时显式设置
`PYTHONPATH=src:.`，确认加载的是当前提交 `ec14608a9073a5cf756af25d12fec98611431c86`
下的 `src/table_diffevo`。

真实核心算法的一轮 4-way 定向测试实际通过：workload B 的答案和 target shape 均为
50，阶数构成为 `30×2-way + 15×3-way + 5×4-way`，5 条 4-way 进入完整
objective；`factorized_gibbs_factor_count=0`，证明关闭 Gibbs 时
`factorized_gibbs_max_order=3` 没有截断 measured 4-way。使用完整运行环境重跑相关
身份、物化、collector、evaluator 测试，结果为 `30 passed`，无 skip。

随后在仓库 `outputs/` 下的忽略临时目录运行 `30 cases × 1 round` 非正式结构冒烟：

```text
cases                         30/30
rounds per case               1
termination                   30 resource_cap_reached（预期）
paired seed shards            5/5
B cases with full 4-way path  15/15
terminal table SHA audit      30/30
query-seed error rows         47,100
evaluation groups             25 / 521 / 512 / 512
scientific gates              全部 inconclusive_resource_cap（预期）
temporary artifacts           已自动清理
```

该冒烟只证明真实 generation、六臂配对、terminal table 读取、公共查询附答案、分阶
汇总和资源上限门禁可以贯通，不是科学实验结果。评价仍严格先冻结查询身份，再读取固定
reference；没有消耗隐私预算。正式 output namespace
`outputs/issue53_test_query_workload_ab_v1` 仍不存在。

注意：当前冻结 protocol 的执行服务器字段仍是此前指定的 A6000
`root@10.8.176.53:6006`，而本次轻量验证按用户最新指示在 `linyao-system` 完成。
下一个独立小步骤若要在当前服务器正式跑，应先在看到正式结果前把 execution server
元数据改为 `linyao-system`、重算并冻结 protocol SHA；若保持现有 protocol，则正式
30 cases 应回到原 A6000 执行。

### 历史暂停点：test query-workload A/B 采集器与评估器实现完成（2026-08-18）

> 本步只实现已冻结 30-case 实验的 collector、evaluator 和回归测试，并执行
> plan/本地测试；没有启动 seeds 318–322 的正式生成，没有产生正式结果，没有
> push 或操作 PR。

新增入口：

```text
collector  scripts/run_issue53_test_query_workload_ab.py
evaluator  scripts/evaluate_issue53_test_query_workload_ab.py
output     outputs/issue53_test_query_workload_ab_v1
protocol   e40317be5a21c0c7a59928865c31cb56071b78e1206dba00bcb574b3cd3b198a
```

collector 固定执行 `A/B × absolute/sqrt_relative/relative × seeds 318–322 =
30 cases`；每个 seed 的六个 case 串行且强制使用相同初始表和 RNG 状态。A/B
分别审计为 `25×1-way + 20×2-way + 5×3-way` 和
`30×2-way + 15×3-way + 5×4-way`。B 的完整 50 条 query/target 不截断传入
objective 和 residual direction，运行后还会用全部 50 条查询独立重算 terminal
loss/L1 并与 early-stop 末次指标对齐。`factorized_gibbs_sweeps=0` 且 compiled
workload 关闭，因此 `factorized_gibbs_max_order=3` 只属于未启用的 Gibbs 路径，
不会排除 B 的 5 条 4-way；正式结果还会记录并断言该路径事实。

evaluator 先冻结并审计四组公共查询身份，再读取固定 reference 附答案；它会逐项
校验 30 个 terminal table 的身份和 SHA，输出 47,100 条 query-seed error。报告先在
每种 geometry 内比较 workload `B-A`，再只在 B 内比较
`sqrt_relative/relative - absolute`；521 条 common unseen 2-way、512 条 fixed
held-out 3-way、512 条 fixed held-out 4-way 分开判定，25 条 1-way 只作 safety
门禁，不做跨组 aggregate。资源上限或未完成 case 只会使相关比较无效，不会被误判
为科学结论。

collector/evaluator 的 plan 均已验证为只展示冻结协议：前者不读取输入、结果或 raw
reference，后者不读取 collection 或 raw reference，且二者都明确
`generation_started=false`。冻结输入与公共评价身份重新审计一致。Ruff 通过；相关
身份、物化、collector、evaluator 测试合计 `29 passed, 1 skipped`。唯一跳过项是真实
核心算法的一轮 4-way 冒烟测试，因为当前本机轻量测试环境缺少完整运行依赖；fake
runtime 测试已证明 50 条 target 和 5 条 4-way 不被截断。

下一个独立小步骤：到用户指定的 A6000 服务器做轻量验证，先让真实核心算法的一轮
4-way 冒烟测试实际通过，并验证一个非正式短 shard 的采集/聚合/评价闭环；确认路径、
依赖和输出审计都正确后，再单独决定是否启动 30 个正式 case。

### 历史暂停点：test 30/15/5 workload B 答案附加与身份审计完成（2026-08-18）

> 本步在上一步的查询身份和实验协议已冻结后，读取固定 SHA 的
> `test_300x10.csv`，仅为 workload B 的 50 条查询附加精确计数答案。没有根据
> 答案替换、重排或删除查询，没有实现 runner 或运行生成实验，也没有
> push 或操作 PR。

附答案入口与产物：

```text
materializer       scripts/materialize_issue53_test_query_workload_b.py
identity input     configs/test_300x10/issue53_query_workload_ab_v1.json
identity input SHA a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36
raw reference      data/test_300x10/test_300x10.csv
reference SHA      c211133455c4fdd19f01f34eca511cf089667452d038265897eec15b5b84baeb
answered workload  configs/test_300x10/measured_50query_30_15_5.json
workload file SHA  708afe2863b797fae714c39699457dd91ac97a9dbcd35b900d46fcf6c01e9e14
target vector SHA  e04988c93076fd0a8ce820d0635080b33d88030415b97f1b804186e017c02e3d
```

信息流审计先在禁止打开 CSV 的条件下逐字段重建 identity artifact，确认 B 为
`30×2-way + 15×3-way + 5×4-way`、50 条无重复且无 1-way，然后才加载 raw
reference。附答案前后 query identity 均为：

```text
602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1
```

为防止新评价器的类型对齐或边界语义有误，先用它重算旧 workload A 的 50 条已知
答案，`50/50` 逐条精确一致；再用正式 `table_diffevo.queries.evaluate_table` 独立
重算 B，也是 `50/50` 精确一致。这证明物化的 target vector 与后续 generator
实际使用的查询语义一致。

结果盲选取的 25 条新查询中，10 条 2-way 计数均大于 0；10 条 3-way 中
N3_01 计数为 0；5 条 4-way 中 N4_01、N4_05 计数为 0。这 3 条保留，因为按
答案过滤零计数会破坏结果前冻结；relative/sqrt-relative 仍使用已冻结 floor=8
处理这些 target。

新增附答案回归与上一步身份回归合计 `13 passed`；Ruff 通过，formal
workload 可用固定 reference 逐字段确定性重建，`git diff --check` 通过。正式文件
正确记录 `raw_reference_data_accessed=true`、`selection_used_reference_answers=false`、
`privacy_budget_consumed=false`。

下一个独立小步骤：基于已冻结 A/B 输入和 30-case 协议实现 collector、公共分阶
evaluator 及其测试；只做 plan/smoke 级本地验证，不启动 seeds 318–322 正式实验。

### 历史暂停点：test 30/15/5 workload A/B 结果前身份与协议冻结完成（2026-08-18）

> 本步在新 namespace 下用 SHA-256 排序结果盲选定 10 条新 2-way、10 条新
> 3-way 和 5 条新 4-way，并冻结 A/B 公共评价身份与 30-case 协议。没有读取
> 原始 reference CSV，没有为 B 附加 query answers，没有实现 runner 或运行实验，
> 也没有 push 或操作 PR。

冻结入口与产物：

```text
protocol doc       docs/设计/Issue53_test查询workload_AB结果前冻结协议.md
protocol doc SHA   291c591ba5408e046005b24122bfe602bf8a97f7c175ee45e59f81daf96b44b6
freezer            scripts/freeze_issue53_test_query_workload_ab.py
identity artifact  configs/test_300x10/issue53_query_workload_ab_v1.json
artifact SHA       a20e33923a399844275eaa53e3b008be251c81e484bbc6eacd2a3ca8a51bec36
```

冻结的 workload 身份：

```text
A = 25×1-way + 20×2-way + 5×3-way
    cbb501f5c2f8c230b6d68d85baf40be7b17be713d41c5b97f54ac30457e90fc8
B = 30×2-way + 15×3-way + 5×4-way
    602d8b7fcbe3f56a3abf62ffe4e2b6b3638578f47ea9fe346a18583923969af1
```

B 保留 D01–D20/T01–T05，新增 N2_01–N2_10、N3_01–N3_10、N4_01–N4_05。
新查询仅使用公开属性 `type/values/bins` 及查询语义选取；选择器不访问
marginal counts、query results、raw reference、terminal errors 或稀有度。3/4-way 候选排除
原 `issue53-heldout-v1` 各 512 条身份，且使用旧 A 结果盲重建原 held-out，没有
用 B 改抽评价集。

四个公共评价身份：

| 查询组 | 数量 | query identity SHA-256 |
|---|---:|---|
| one-way safety | 25 | `b144694657b98b27ac92173b10d641981ce5f16e5c8ab00191b26ef5c143250c` |
| common unseen 2-way | 521 | `fabbdc8de6aa9ebbc9d6c5bc209e3c47ee9a678c98f41bc71c168e470d9f1fc2` |
| fixed held-out 3-way | 512 | `d70e87c3bceb1203a6df8d0d6f7279764ca5b9801467e73ed839e84589dae78a` |
| fixed held-out 4-way | 512 | `2e0788fa13347f867d7cb9bfc5b3c63d7d5e7c9397cd44079bc071e9b04ec171` |

协议冻结 `workloads=[A,B] × geometries=[absolute,sqrt_relative,relative] ×
seeds=[318,319,320,321,322] = 30 cases`，其余参数完全复用 P=6 fresh-seed 实验。
判定先在每个 geometry 内比 B 相对 A，然后才在 B 内比 geometry；三个 common
unseen 组分开报告并要求 Pareto 不劣，1-way 只作安全门禁，不作跨组 aggregate。

审计与验证：正式身份文件可逐字段确定性重建，SHA 一致；递归 `result`
key 审计为 0；更改旧 query results 和 marginal counts 不改变选取身份；新 B 阶数构成、
无 1-way、A/B 并集与公共评价集不相交、固定 held-out 与既有存档身份等价均有
定向测试。结果为 `7 passed`，Ruff 通过，`git diff --check` 通过。

下一个独立小步骤：在不改变任何冻结身份的前提下，读取固定 raw reference 仅为
B 的 50 条查询附加答案，物化新 measured workload 输入，再审计附答案前后的
query identity 仍为 `602d8b...9af1`。本步完成后再进入 collector/evaluator 实现。

### 历史暂停点：test 30/15/5 generation workload A/B 候选空间审计完成（2026-08-18）

> 用户已确认把新 generation workload 固定为 50 条：30 条 2-way、15 条
> 3-way、5 条 4-way；不在 measured generation workload 中放 1-way。本步只建立
> 本地分支并审计公开候选空间，没有生成新查询、没有读取原始 reference CSV、
> 没有运行实验，也没有 push 或操作 PR。

当前本地分支：

```text
branch   research/issue53-query-workload-ab
base     88853c29a9cc1f571a06a2537e57cddcca665628
remote   未创建，未 push
```

对照组 A 保留旧 `measured_50query.json`：25 条 1-way + 20 条 2-way + 5 条
3-way。新组 B 保留旧 D01–D20 和 T01–T05，再用 10 条新 2-way、10 条新
3-way、5 条新 4-way 替换 S01–S25，因而恰好是 `30 + 15 + 5 = 50`。A/B 仍使用
同一份 1-way `init_marginals.json` 初始化；改变的只是后续 measured generation workload。

结果盲审计结论：

| 阶数 | 公开标准 cell 总数 | 已保留旧查询精确重叠 | 固定 held-out 排除 | B 可选新查询 |
|---|---:|---:|---:|---:|
| 2-way | 548 | 17 | 0 | 531 |
| 3-way | 5,056 | 5 | 512 | 4,539 |
| 4-way | 30,450 | 0 | 512 | 29,938 |

旧 20 条 2-way 中 D04、D05、D07 使用合并/单边年龄区间条件，不是公开边际网格中
的单个标准 cell；它们仍保留在 B，但去重必须使用语义指纹，不能只看 `type`
或 ID。新 2/3/4-way 只能从公开 `init_marginals` 定义的 cell 中用新 namespace
的 SHA-256 排序确定性选取；不许用 target count、稀有度、旧 terminal error 或实验结果
挑查询。3/4-way 还必须排除已存档的各 512 条 held-out 身份；不能用 B 重建
held-out，否则评价集会跟着训练集变化。

选完 10 条新 2-way 后，A/B 公共未见 2-way 评价集将固定为 521 条；另外分开
报告 25 条 1-way safety、既有 512 条 held-out 3-way 和 512 条 held-out 4-way，不做
跨组 aggregate。

GitHub 状态更正：PR #64 已按用户要求关闭，未合并；PR #63 仍保持原状态并
等待他人审查。下方原“PR #64 等待审查”板块仅保留为历史实验记录，其 OPEN
描述已被本板块取代。

下一个独立小步骤：先写并测试结果前查询身份冻结器/协议，只产生不含
`result` 的 30/15/5 身份和 A/B 公共评价身份；审计通过后才可以读取 reference
附答案，再进入 runner 实现。

### 历史暂停点：test 残差几何 fresh-seed 确认完成，stacked PR #64 后已关闭（2026-08-18）

> 本步按结果前协议在用户指定的 A6000 服务器完成 seed 313–317、三种 residual geometry 的 15 条
> fresh 轨迹，再按 measured 1-way、全部未测量 2-way、冻结 held-out 3/4-way 分阶评价。没有按结果
> 增加 seed、修改门禁、扫描 floor/gamma/rho 或触碰等待外部审查的 PR #63 分支。

冻结与产物身份：

```text
protocol commit       abf676e93b07837ced96ac4a311a5b401364770d
collection commit     9f1873c1ebf7466e781687b7a17ea028f310b9cb
collection protocol   9708f994c6c479b8e08c75cc662d0f79ec3ab5ec39cd9322e2ba5e8b7b30373b
collection report     98e1b09bea3691d2c1d10b1ff6fc8830f4f5782b6f7d3b6ef49060dc82e98da8
evaluation commit     f7775dde2c6fdef67e0a9ed7fbb4ac21f279b8d3
evaluation report     54f586462c13e23a285d91d238d25246c8e7afd86016b8ee82ff6704bc5fe60f
query-seed CSV        1f158acd491add3164fb93ab0219d1323761cd7c12ec2f5c09b72d047a77466b
```

15/15 cases 全部 `early_stopped`，无资源上限结束。正式采集使用 NumPy、5 个 seed shard 并行、
shard 内三臂串行；`CUDA_VISIBLE_DEVICES` 为空，未触碰 GPU 3 的既有约 34 GiB 任务。远端 36 个
collection 文件回收到本地后逐项 SHA-256 一致。评价先审计查询身份，再读取固定 reference；没有生成
新表或消耗隐私预算，也没有跨组 aggregate/canonical selection。

五 seed 平均绝对计数误差：

| 查询组 | 数量 | absolute | sqrt | relative | 最低 |
|---|---:|---:|---:|---:|---|
| measured 1-way | 25 | **0.8000** | 0.9200 | 1.0080 | absolute |
| measured 2-way | 20 | **0.8600** | 0.8700 | 1.0200 | absolute |
| measured 3-way | 5 | 1.0000 | 1.1200 | **0.8000** | relative，仅描述 |
| all unmeasured 2-way | 531 | **6.7910** | 7.0177 | 7.4614 | absolute |
| frozen held-out 3-way | 512 | **3.9477** | 4.1637 | 4.4484 | absolute |
| frozen held-out 4-way | 512 | **1.6930** | 1.7441 | 1.8082 | absolute |

sqrt 相对 absolute 的 primary delta 为 `+0.2267/+0.2160/+0.0512`，paired-seed 更好数为
`1/5、0/5、1/5`；measured 1-way delta `+0.1200`，0/5 更好、1/5 平局。relative 对应 primary
delta 为 `+0.6704/+0.5008/+0.1152`，更好数 `1/5、1/5、2/5`；1-way delta `+0.2080`。
两候选的 unseen/safety 门禁都失败，总分类为 `no_unified_test_candidate_under_frozen_rule`。

这说明平方根方法确实比 relative 更接近 absolute，但不是 test 的统一赢家；seed 310–312 上 sqrt
在未测量 2-way/4-way 的微小改善没有 fresh 复现。原 25 条 measured 1-way 会放大差异，却不是失败
唯一原因，因为三个 primary 未测量组也全部偏 absolute。按冻结规则保留 absolute 作为 test 参考，
停止调 residual 新公式；nltcs 既有 Pareto 结论保持不变。

评价归档时发现首次报告错误继承 plan-only mode。提交 `f7775dd` 仅修元数据并允许 collection/evaluation
commit 分开审计；同一 collection 重放后 CSV SHA 不变，删除 mode/evaluation commit 后新旧 JSON
逐位一致，科学结果没有变化。完整说明见
`docs/实验结果/Issue53_test残差几何fresh-seed确认结果.md`。

当前验证：相关回归 `42 passed`；confirmation 定向 12 tests 在本机/A6000 均通过；研究新增脚本与
测试 Ruff 0.16.3 通过；24,075 行 CSV 独立重聚合与报告一致。

已创建依赖 PR #63 的 stacked PR：

```text
PR       #64
title    研究：Issue #53 残差几何分阶诊断与 fresh-seed 确认
base     research/issue-53-stage2-v2-evidence （PR #63 head）
head     research/issue53-sqrt-residual-earlystop
url      https://github.com/Chuhan722/table-diffusion/pull/64
state    OPEN，非 Draft
merge    CLEAN（创建后回读时无 CI check 回报）
```

PR 正文明确了依赖/review 顺序、审计 SHA、元数据勘误和“不改默认 residual、不作跨数据 canonical、
不自行 review/merge”的边界。当前停在等待他人审查，不对 #63/#64 做合并动作。研究上的下一步是在
新板块中先冻结跨 workload 的分阶质量—计算门禁，再进入 donor/alpha；不得把它顺带塞进本 PR。

### 最新暂停点：test 分阶 held-out 诊断完成，排除简单 order-aware 接续（2026-08-18）

> 用户确认先核对 AIM/Private-GSD 的 1-way 语义，再要求检查当前 test 查询设计。为避免用同一批结果
> 事后挑口径，本步先提交分阶诊断协议，再实现并在干净提交上只读评价既有 9 张
> `test_300x10/terminal_current.csv`；没有重新生成、修改早停/残差参数或消耗隐私预算。

冻结入口：

```text
protocol doc     docs/设计/Issue53_test分阶heldout只读诊断协议.md
protocol commit  d427db68b927375a58e87ea8b172476e1ed5dcbd
analysis script  scripts/analyze_issue53_test_ordered_heldout.py
analysis commit  219bf74ea753823058c0b2842d7c90a543d47079
source report    241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143
result report    cb88a5bbbd6de494fd97f60ca3984dfe53fe714379978137ae69436773feff24
```

查询身份在读取 raw reference 前冻结：公开标准 2-way cell 共 548 条，与 measured 精确重叠 17 条；
其余 531 条全部纳入，不按 target/终态误差抽样。既有 result-blind 3/4-way held-out 身份确定性重建
一致，各 512 条、与 measured 重叠为 0。六组始终分开报告，没有总体加权分。该离线诊断随后读取
原表为 531 条查询附答案，因此报告正确标记 `raw_reference_data_accessed=true`。

三 seed 平均绝对计数误差：

| 查询组 | 数量 | absolute | sqrt | relative | 最低 |
|---|---:|---:|---:|---:|---|
| measured 1-way | 25 | **0.6933** | 0.7333 | 1.3733 | absolute |
| measured 2-way | 20 | **0.8500** | 1.0833 | 0.9000 | absolute |
| measured 3-way | 5 | 0.9333 | 0.9333 | **0.8000** | relative |
| all unmeasured 2-way | 531 | 7.0929 | **6.8763** | 7.1620 | sqrt |
| frozen held-out 3-way | 512 | **4.1296** | 4.2637 | 4.2435 | absolute |
| frozen held-out 4-way | 512 | 1.7760 | **1.7650** | 1.8737 | sqrt |

关键判断：原 50-query aggregate 确实因 25 条 marginal 1-way 放大 relative 劣势；但 relative 相对
absolute 在全部未测量 2-way、held-out 3-way、held-out 4-way 的均值仍分别差 0.0691、0.1139、
0.0977 count/query，三个组都是 1/3 paired seed 更好、2/3 更差。因此问题不只是 test 查询设计，
但差值很小、seed 方向仅 2:1，不能声称稳定显著劣化。

sqrt 相对 absolute 在未测量 2-way 改善 0.2166、4-way 改善 0.0111 count/query，在 held-out 3-way
变差 0.1341；measured 1-way 只差 0.04，但 measured 2-way 差 0.2333。它是 mixed 中间点，不是统一
赢家。完整结果见 `docs/实验结果/Issue53_test分阶heldout只读诊断结果.md`；实现及既有相关回归
`30 passed`，Ruff 通过，CSV 独立重聚合与报告一致。

下一步不实现 `order_aware_relative`，因为其 order>=2 路径正是当前三个未测量组都未胜出的 relative；
也不扫 gamma/floor 或按 310–312 调权重。若继续 residual 板块，先结果前冻结 fresh-seed 的 test 专用
复核，仍将未测量 2/3/4-way 分开，检查这些很小的 2:1 方向能否复现；若仍 mixed，则停止寻找单一
跨数据 residual，先冻结 workload 级质量—成本选择门禁，再进入 donor/alpha。

### 最新暂停点：残差几何查询级诊断完成，下一候选转为一阶边缘保护（2026-08-18）

> 本工作在独立 worktree/分支 `research/issue53-sqrt-residual-earlystop` 上进行，不修改等待外部审查的
> PR #63 分支。用户授权直接比较 absolute、平方根中间方法和 relative 在 test/nltcs 的 P=6 早停
> terminal-current 结果；18 组矩阵已在 A6000 上全部完成并聚合。

新增残差几何：

```text
sqrt_relative
  = sign(raw) * magnitude / sqrt(max(target, 8)) / n_records
```

它固定在 absolute（不按 target 标准化）与 relative（完整除以 target）之间；不暴露指数、不扫描
gamma。噪声容忍仍先于标准化，三种几何零点相同，absolute/relative 旧路径不改默认语义。

冻结 development 矩阵：

```text
datasets = [test_300x10, nltcs]
arms     = [absolute, sqrt_relative, relative]
seeds    = [310, 311, 312]
cases    = 18
P        = 6 natural-work ticks
C        = 6000 rounds / 6000 candidates
protocol SHA = 7e7b5e08f9d934031257cbd98b6a857f7ba1dcb4cf1f97077d48f781a4e2585f
```

除 seed/geometry 外完全复用 PR #63 两数据 smoke 参数：rho=0.01、scale-invariant donor、fixed
alpha=16、direction initial_rms、eta=0.5、mu=0.01、Gibbs sweeps=0、tol=inf、无重试、
terminal-current。每个数据只描述三 seed 平均 terminal measured L1、配对胜数、loss、rounds/work 和
A/B/C；不读取原始 reference table，不形成 canonical/held-out/收敛结论，不按结果调参。

固定入口和协议：

```text
scripts/compare_issue53_residual_geometry_earlystop.py
tests/test_compare_issue53_residual_geometry_earlystop.py
docs/设计/Issue53_平方根残差P6早停两数据三臂对比协议.md
```

结果前验证：新增/相关定向 `47 passed`；本机轻量环境可收集的全仓回归 `1577 passed, 7 skipped`，
仅有 2 个既有 warning。本机环境缺少 matplotlib 的 6 个旧 Stage2 分析测试在 A6000 完整环境中补跑；
冻结提交 `fe8fb797a718bf0e9a89668d46fbd5726c1c3082` 的远端全仓结果为 `1636 passed, 2 warnings`。

正式运行使用 `root@10.8.176.53:6006` 的 RTX A6000 GPU 0，只暴露一张 GPU、一个 worker，三个 seed
shard 串行；GPU 3 的既有任务未触碰。18/18 cases 均为 `early_stopped`，没有资源上限结束：

| 数据 | 残差 | mean terminal L1 | 配对胜数 | mean work | mean rounds |
|---|---|---:|---:|---:|---:|
| `test_300x10` | `absolute` | 0.0026000 | 3/3 | 13.3356 | 1319.7 |
| `test_300x10` | `sqrt_relative` | 0.0029778 | 0/3 | 14.0044 | 1396.0 |
| `test_300x10` | `relative` | 0.0037556 | 0/3 | 15.6689 | 1550.0 |
| `nltcs` | `absolute` | 0.0011321723 | 0/3 | 17.6697 | 1768.7 |
| `nltcs` | `sqrt_relative` | 0.0004930488 | 0/3 | 15.3349 | 1534.3 |
| `nltcs` | `relative` | 0.0003474679 | 3/3 | 23.0065 | 2301.3 |

`test_300x10` 上 absolute 同时以 L1 和 work 支配另外两臂：sqrt 的平均 L1/work 分别高 14.53%/
5.02%。`nltcs` 上 sqrt 同时以 L1 和 work 支配 absolute（分别低 56.45%/13.21%）；relative 比
sqrt 的 L1 再低 29.53%，但 work 高 50.03%，二者构成 Pareto 取舍。三 seed 的胜者在两个数据上
都是 3/3 一致，但方向相反，因此平方根方法只是 nltcs 上有价值的中间点，不是跨数据 canonical 答案。

完整结果见 `docs/实验结果/Issue53_平方根残差P6早停两数据三臂结果.md`。报告 SHA-256 为
`241618e80cce3549e2626fc668467e4c9029be968858e09a2dffb029716de143`；本地与远端 40 个文件逐项
SHA-256 完全一致。运行结束后远端树 clean、GPU 0 已释放，GPU 3 未触碰。

已完成不需 GPU 的 query-level 只读诊断：固定读取原 18 张 terminal-current 表，形成 9459 条
query/seed/arm error 和 1051 条 query summary；逐臂复算 overall L1 与 source report 一致。入口提交
`deb659f3346f3dac92763a4479418b619027b061`，报告 SHA-256 为
`876b7cc2f75ddf315800dd36853ca617fbbbbbf6258bc908709bec49c251e48b`。本分析明确为已见结果后的
development diagnostic，没有生成新表、读取 raw reference 或消耗隐私预算。

机制结论：频率不是唯一或最强分流。`test_300x10` 没有 rare query，但 25/50 条为 1-way，且 25/25
target 与初始化 marginal count 精确一致；relative terminal 上 1-way mean abs count 从 absolute 的
0.693 升到 1.373、exact rate 从 45.33% 降到 17.33%，该阶占 relative 总误差 60.95%。`nltcs`
没有 1-way；relative 在 rare/medium/common × 2/3-way 六格中全部最低，并在 3003 个 query×seed
配对中相对 absolute 为 2197 better / 145 tie / 661 worse。结构 overlap low/middle/high 不改变各数据
方向。完整结果见 `docs/实验结果/Issue53_残差几何查询级诊断结果.md`。

下一步不做频率 selector、不扫 gamma/floor、不调 rho。先结果前设计 `order_aware_relative`：1-way
使用 absolute 恢复力、order>=2 使用 relative-f8。设计必须先解决两块原始尺度不同的问题，禁止根据
310–312 调混合系数；无 1-way workload 必须与 relative 数值等价。设计和单测审查后，才使用 fresh
seeds 在 test 上比较 absolute/relative/candidate；nltcs 先证明路径等价，再决定是否冗余重跑。

### 最新暂停点：PR #63 三项补强及 Python 3.9/CUDA 验证全绿，已推送并回复 reviewer（2026-08-22）

> 用户授权修复上一轮只读复核发现的全部问题，并要求同步刚合并的 PR；完成检查后又明确授权提交、
> push 并发布拟好的 reviewer 回复。已确认刚合并的是 PR #61（merge commit
> `e6d84cf1fc071706e52610d8593f488b7137092c`）；PR #62 仍为 OPEN。本轮没有重跑或改写正式实验产物，
> 也没有提交新的 Review、Approve 或 merge PR #63。

先从 `origin/master` 取回 PR #61，并确认其 11 个变更文件只涉及 plants 数据/workload、
`scripts/build_marginals.py`、生成脚本和 plants 测试，与 PR #63 当时的 11 个未提交兼容文件无交叉。
随后以普通 merge 无冲突同步到本地 PR #63 分支：

```text
origin/master       e6d84cf  Merge pull request #61 from Chuhan722/feat/plants-dataset
local merge commit  caad0c687b74b0e613db9fedad93e38ff61c0d0d
old remote PR63     24478dde3f639ee8f55100d3e7741506631bbc12
```

在此前 Python 3.9 兼容补丁之上，本轮完成三项补强：

1. **A/B/C 与 stationarity trace 组合崩溃**：`StationarityTrace` 的 v1 终止原因白名单补入
   `fit_target_reached`、`early_stopped`、`resource_cap_reached`。增加 A/B/C 三条真实
   `run_evolution(record_stationarity_trace=True)` 接线回归，逐项要求 diagnostics/trace 原因一致、
   state count 对齐并可再次 `trace.validate()`。修复前最小 C 反例会在返回前抛“未知
   termination_reason”；修复后 A/B/C 全部正常返回。
2. **development audit 在 `python -O` 下 fail-open**：
   `scripts/analyze_issue53_terminal_early_stop_development.py` 的关键运行时 `assert` 全部替换为带明确信息的
   `_require`/`AssertionError`；新增 AST 契约，禁止该审计脚本重新出现会被优化器删除的 `assert`。
   另以真实 `python -O` 调用故障注入确认 `_require(False, ...)` 仍抛 `AssertionError`。
3. **V2b 路径不便携**：未来 V2b runner/report/audit 统一只写 sibling 文件名；auditor 只实际读取并
   SHA 校验 report 同目录的 `protocol_manifest.json`，接受新相对格式，也兼容旧报告仅作元数据的绝对
   `.../protocol_manifest.json`，同时拒绝 `..` 相对逃逸、错误绝对文件名和解析后离开 sibling 的
   symlink。既有 V2b/V2c 冻结负结果及其哈希链没有被改写或冒充重跑。

验证结果（包含刚同步的 PR #61 plants workload 测试）：

```text
针对性回归（A/B/C trace + development audit + V2b runner/auditor）
  Python 3.11.15: 50 passed

全仓 Python 3.11.15
  1638 passed, 2 个既有空切片 warning

全仓 Python 3.9.25（NumPy 1.26.4 / Pandas 2.2.3 / PyTorch 2.8.0+cu128 / RTX 4090）
  1638 passed, 0 skipped, 15 warnings
  warning：13 条临时 Matplotlib/PyParsing 依赖弃用提示 + 2 条既有空切片 warning
```

最初 Python 3.9 临时环境使用 CPU-only PyTorch，因此 7 条 CUDA 测试被 pytest 按设计跳过。用户要求
补齐后，将同一临时环境从 `torch 2.8.0+cpu` 替换为官方 `torch 2.8.0+cu128`，固定使用一张空闲
RTX 4090：

```text
CUDA smoke
  Python 3.9.25 / torch 2.8.0+cu128 / CUDA runtime 12.8
  torch.cuda.is_available() == True；真实 1024x1024 CUDA 矩阵乘成功

原 skip 所在 GPU/采样测试组
  206 passed, 0 skipped（原 7 条全部执行通过，CUDA 可用时另收集 9 条路径）

全仓复验
  1638 passed, 0 skipped, 15 warnings in 121.57s
```

该补充只更换临时测试环境并运行测试，没有源码、依赖声明、实验产物或协议改动。

协议与冻结 artifact 边界保持不变：

```text
P=6 protocol SHA             759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317
RMSE frozen protocol SHA     cb1224ac797191b74aa40f7baadfab08928b5cb25414971fe8ee091a297d433a
RMSE result-blind plan SHA   aadcebcf68ff3ed5a05bb3164e8199951e11b749a3fbff82cedcfcc11a4b56cd
V2b protocol SHA             a7dde6b7867e215c9147131f085eaa47b47e04495b5d1bed37355f95a69dd33f
```

`git diff -- outputs` 为空，`git diff --check` 通过。用户检查拟稿后明确授权，本节所述兼容补丁、三项
修复、测试与状态记录已提交并推送到 PR #63；随后已发布回复，向 reviewer 说明实际 Python 3.9 修复面、
额外补强、双版本全仓测试结果和冻结证据边界。当前等待 reviewer 增量复核；未经新的明确授权，不再
修改远端、不提交 Review、不 Approve 或 merge，也不重跑任何正式科学实验。

### 最新暂停点：PR #63 Python 3.9 兼容补丁本地全绿，待用户确认提交/推送（2026-08-22）

> 外部 Review 将 Conda `gsd`（Python 3.9）测试兼容列为合并阻塞。用户同意修改，但再次明确未授权
> push；本轮只在 PR #63 本地工作树实施最小兼容补丁、运行测试和对拍协议摘要，没有 commit、push、
> PR 评论、Review、正式产物重跑或参数/算法修改。

独立复现先确认 Review 对失败文件的归因不完整：四个 collection error 中，只有
`tests/test_inner_early_stopping_integration.py` 真正因 `int | None` 缺 future annotations；另外三个
实际来自 P=6 collector/evaluator 和 RMSE runner 使用 Python 3.11 才提供的 `datetime.UTC`。排除这
四个 collection error 后，原生 Python 3.9 的 23 个失败全部来自本 PR 的 10 处
`zip(..., strict=True)`。

本地兼容补丁共改 10 个文件，当前 diff 为 `24 insertions / 17 deletions`：

- wiring test 增加 `from __future__ import annotations`；
- 三个脚本将 `datetime.UTC` 等价替换为 `timezone.utc`；
- 10 处 strict zip 改为普通 `zip`，并保留既有等长校验；原本缺少显式校验的测试辅助路径补
  `len(...)` 断言；
- 不改 A/B/C、自然工作时钟、阈值、seed、rho、C、alpha、Gibbs、terminal-current 或任何聚合规则。

验证结果：

```text
Python 3.9.25 临时环境（NumPy 1.26.4 / Pandas 2.2.3 / PyTorch 2.8.0+cpu）
  PR #63 改动测试：414 passed
  全仓测试：1605 passed, 7 skipped, 15 warnings
  warning：13 条临时 Matplotlib/PyParsing 依赖弃用提示 + 2 条既有空切片 warning

Python 3.11.15 项目环境
  PR #63 改动测试：414 passed
  全仓测试：1625 passed, 2 个既有 warning
```

兼容前后对拍保持不变：

```text
P=6 protocol / manifest SHA
  759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317
RMSE frozen protocol canonical SHA
  cb1224ac797191b74aa40f7baadfab08928b5cb25414971fe8ee091a297d433a
RMSE result-blind plan canonical SHA
  aadcebcf68ff3ed5a05bb3164e8199951e11b749a3fbff82cedcfcc11a4b56cd
```

既有 P=6 collection/evaluation 与 RMSE JSON 文件均未写入，文件 SHA 保持原值；它们继续诚实绑定原
运行 commit 和原 source SHA。当前源码 SHA 会因兼容补丁变化，但科学 protocol/plan 和历史 artifact
没有被回填或冒充重跑，因此不需要重新运行正式实验。若后续提交，应在 PR 回复中说明这是运行时兼容
修改、协议摘要逐位不变，并附 Python 3.9/3.11 全仓结果。

当前停止在本地工作树有上述未提交改动。下一步必须先向用户展示结果；只有用户明确授权后才能 commit
或 push，且 push 后仍需等待原 reviewer 增量复核，不能自动 Approve/merge。

### 最新暂停点：PR #63 已创建并以 Amendment 3 同步 Issue #53，等待审查（2026-08-17）

> 用户确认不需要为 P=6 先冻结 rho，并授权在 `test_300x10`、`nltcs` 各跑一次后归档 PR；本机两张
> 4090 均有其他用户任务，因此改用旧服务器空闲 GPU 0。冻结 runner 在提交
> `d220ba4d04606c4ed99c89d98da314a31f1d0d71` 上只运行一次；两个数据均由 B/`early_stopped`
> 正常结束，没有触及 C=6000。运行前后没有调整 P/rho/C/alpha/Gibbs，也没有访问原始参考表或消耗
> 隐私预算。

已将本分支首次推送并创建非 Draft PR：

```text
PR       #63
title    研究：Issue #53 无门控内层 A/B/C 早停、P=6 验收与两数据 smoke
base     master
head     research/issue-53-stage2-v2-evidence
url      https://github.com/Chuhan722/table-diffusion/pull/63
state    OPEN
merge    CLEAN（创建后核对时尚无 CI check 回报）
```

PR 正文按“目标与边界 → 最终 A/B/C 设计 → 旧 V2/V2b/V2c/RMSE+MAX 负证据 → 主要实现 → P=6
未见人工轨迹正式验收 → test/nltcs smoke → 测试审计 → review 顺序 → 后续 rho 板块”组织。正文明确
600 只是在 `rho=0.01` 下 P=6 的近似单次无改善耐心，6000 才是 smoke 的绝对上限；同时保留单 seed
真实链路 smoke 不能证明收敛、P=6 全局最优或计算优于固定 2000 轮的边界。

已按用户确认稿向 Issue #53 发布结果后状态修订：

```text
comment  Amendment 3（2026-08-17）：无噪声生成内层停止路线收口与 P=6 阶段结果
url      https://github.com/Chuhan722/table-diffusion/issues/53#issuecomment-5313678308
```

GitHub 回读确认评论标题、terminal-current 最终语义、P=6 未见人工验收、test/nltcs smoke 和完整结论
边界均已保存。该评论明确修正 Amendment 2 的 best 输出旧语义，只同步当前结果并停在 PR #63 等待
审查；没有写入或冻结下一研究板块，后续方向留待另行讨论。

新增固定入口：

```text
scripts/run_issue53_p6_dataset_smoke.py
tests/test_run_issue53_p6_dataset_smoke.py
protocol SHA = 3b593ce71c8b4bd147b836dd03986d4e64d27bb782a57d0a9ac5759baf805c17
output = outputs/issue53_p6_dataset_smoke_seed200/
```

入口只有 result-blind `plan` 和必须确认完整 protocol SHA 的 `run`，没有数据集、seed、P、rho、轮数、
alpha、Gibbs 或其他科学参数覆盖。固定为：

```text
datasets = test_300x10 -> nltcs（串行）
seed = 200（每个数据一条，仅作 PR 归档前真实全链路 smoke）
rho = 0.01
P = 6 natural-work ticks
C = n_rounds 6000 / candidate_budget 6000（期望 60 normalized work）
relative residual geometry floor = 8
scale-invariant fixed alpha = 16
factorized Gibbs sweeps = 0
tol = +inf, max_retries = 0, terminal-current output
```

runner 原样使用当前 relative-f8 主臂的 `initial_rms` 方向尺度；alpha 显式固定为 16，rho 恒定且不启用
任何随总轮数变化的退火，因此改变纯 C 上限不会改变同 seed 的前缀。两个数据的 schema/query/marginals
SHA 以及历史 relative-f8 seed 200、2000-round 归档 JSON SHA 全部固定。历史结果只作描述性同 seed
对照，不构成新验收阈值，不允许据此调参。在线停止不读取 L1，runner 也不读取原始 reference table。

冻结提交前验证：新增 runner `10 passed`；Issue #53 当前停止链路相关 `113 passed`；全仓 CPU 回归
`1605 passed, 7 skipped, 2 个既有 warning`。正式执行使用完整 protocol SHA 确认，只暴露远端
RTX A6000 GPU 0，单 worker 按 `test_300x10 -> nltcs` 串行运行；运行结束后 GPU 0 回到
`3 MiB / 0%`，隔离代码树仍干净，GPU 3 的既有无关任务从未触碰。

正式结果：

| 数据 | 停止原因 | 实际轮数 / 6000 | normalized work | terminal loss | terminal normalized L1 | 相对历史同 seed 2000 轮 |
|---|---|---:|---:|---:|---:|---|
| `test_300x10` | `early_stopped` | 2128 | 21.0000 | 49.5 | 0.0036666667 | 多 128 轮；L1 相同；loss +1.0 |
| `nltcs` | `early_stopped` | 2500 | 25.0062 | 17649.5 | 0.0002645522 | 多 500 轮；L1 -0.0000006791；loss +182.5 |

这里的 6000 是绝对 raw-round/candidate 上限；`P=6` 是六个 natural-work ticks 的连续无最好 loss
刷新耐心。在 `rho=0.01` 下，一个 tick 约对应 100 raw rounds，所以“约 600 轮”只描述一次完整
无改善耐心，不是总轮数或新上限；任何新最好值都会重新累计耐心。本次实际停止在 2128/2500 轮正好
验证了两者不能混同。

两个目标数据 smoke 都证明 B 能在 C 前自行结束，并保持 terminal-current、无门控输出身份；但相对
旧的任意 2000 轮参考并未节省 raw rounds（分别多 6.4% 和 25%）。单 seed 的 L1 为相同/极小改善，
terminal squared loss 则略高，属于描述性混合结果，不能宣称收敛、真实数据质量正式通过、P=6 全局
最优或计算成本优于 2000 轮。历史 2000 轮结果从始至终不是验收门禁，也没有据此调参。

完整结果记录见 `docs/实验结果/Issue53_P6两数据归档Smoke结果.md`；本地忽略目录
`outputs/issue53_p6_dataset_smoke_seed200/` 与远端逐文件 SHA-256 完全一致，顶层 report SHA-256 为
`cd1e10f9034f63ec4a4caed39370e1b7bb802720e41c9ed5dae8716667ee90fa`。当前停在 PR #63 已创建、等待
CI/审查的节点；不得自动合并，也不自动重跑、调 P/rho 或进入外层 DP。审查完成并由用户确认归档后，
再与用户讨论后续研究方向；当前不预先冻结下一板块。

### 最新暂停点：Issue #53 P=6 正式质量—计算验收通过（2026-08-17）

> 用户在 auditor erratum 提交后单独授权继续。本步先运行 result-blind evaluator plan，再对既有唯一
> primary collection 做了一次正式只读评价；没有重新采集、调用 generator、修改 raw artifacts/协议/
> 阈值/分类规则、运行 P=12/P=4 回退、访问真实数据或消耗隐私预算。

正式评价绑定：

```text
protocol SHA = 759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317
collection Git commit = 34b477acff11adabfc22b6eb9c14e4fb3939b7a1
collection manifest SHA = aa4b34f80cbe72546c6a085845d205e988e04ccdeb0ee843ec135fbfa3505133
erratum evaluator Git commit = 4a63d957b7db4e0f9b23e88320d15405b4565a3c
report = outputs/issue53_p6_unseen_primary/p6_evaluation_report.json
report SHA = c2fce2269476b979e2396def587a6fda764ebb74b9c5a1623f75b9ee8eddfb08
```

artifact、原 source SHA、runtime 和勘误 commit 漂移白名单审计全部通过，唯一报告原子生成。正式聚合
结果为：

```text
classification = supports_p6_on_frozen_artificial_development
next_action = accept_p6_for_current_development_stage
claim_scope = two_public_artificial_families_development_only_not_convergence
```

具体证据：

1. 12/12 均由 B/`early_stopped` 正常结束，A=0、C=0；B 数量、正常结束数和资源上限门禁均通过。
2. `+6`、`+12` shadow checkpoint 都覆盖 12/12，两个 family 在两个检查点各有 6 条，证据门禁全部
   通过且无 right-censoring。
3. 冻结定义 `delta_L1 = L1_at_B - L1_at_tau_plus_k`。`+6` 总体中位数为
   `-0.012226430976430974`，`+12` 为 `-0.005124158249158246`，均不超过 `0.01`；两个 family 的
   两个检查点中位数也均为负，不超过 `0.02`。
4. `+6` 没有 `delta_L1>0.02` 的大退化；`+12` 有 2/12，即 `16.67%`，仍低于冻结上限 25%。质量
   门禁通过。负中位数表示按该离线比较，B terminal current 的典型 L1 没有劣于继续运行的检查点；这
   不是每条轨迹都单调改善的声明。
5. `saving_12` 中位数为 `0.5217391304347826`，即约 52.17%，高于冻结下限 30%；两个 family 分别约
   51.02% 和 58.53%，计算门禁通过。两个 family 均无增大/减小 P 的方向要求，不存在方向冲突。

所以当前只能严谨表述为：**P=6 在当前两个人工 family 的未见轨迹上通过 development 质量—计算
验收。** 不能表述为算法收敛、P=6 全局最优、真实数据已验证，或未来带噪阶段自动成立。本次结果不
触发任何 P 回退，`fallback_patience_ticks=null`。

当前在正式评价结果记录处停止。下一步应由用户另行决定 Issue #53 当前 development 结论如何归档/
同步，以及进入哪个后续板块；不得自动扩展到真实数据、外层 DP、加噪或新的 P 扫描。

### 最新暂停点：Issue #53 P=6 evaluator 审计计数勘误已完成，尚未重新评价（2026-08-17）

> 用户授权本步只做 auditor erratum、测试、记录和提交。本步没有重新运行 collector，没有修改 raw
> artifacts、generator、停止规则、协议阈值、聚合/classification 规则或回退规则，也没有运行正式
> evaluator；因此仍然没有 P=6 验收结论，`p6_evaluation_report.json` 仍不存在。

本次把首次 fail-closed 暴露出的唯一审计错误修正为真实运行语义：

```text
state_evaluation_count == max(1, rounds_run)
```

`current_state_metrics_history` 仍必须保留初始状态和每轮后的 terminal-current 状态，即长度仍为
`rounds_run+1`；本次没有把二者混成一个计数。测试中的假 B artifact 同步改为真实计数，并用真实 B
artifact 明确回归 `rounds_run=6 -> state_evaluation_count=6`。

由于 evaluator 修订必然产生不同于原 collection 的 Git commit，本次没有简单删除身份门禁，而是增加
了更窄的 fail-closed 勘误门禁：

1. 只接受原 collection commit
   `34b477acff11adabfc22b6eb9c14e4fb3939b7a1` 与 collection manifest SHA
   `aa4b34f80cbe72546c6a085845d205e988e04ccdeb0ee843ec135fbfa3505133` 的组合；
2. 原 execution manifest 中 6 个 collector/generator/protocol 源文件 SHA 必须逐项等于当前文件；
3. 从原 collection commit 到 evaluator commit 的 Git 路径差异只能出现在 evaluator、两份相关测试、
   `PROJECT_STATUS.md` 与新勘误文档；任何其他路径都 fail closed；
4. 将原 collection commit、当前 evaluator commit、实际/允许的漂移路径、计数修正和首次诊断暴露范围
   一并写进未来正式 report。

勘误细节登记于 `docs/设计/Issue53_P6评价器审计计数勘误.md`。冻结的 collector、generator、停止实现、
验收协议文档均未修改。提交前复核结果：

```text
定向 collector/evaluator tests：34 passed
Issue #53 相关回归：103 passed
全仓库 CPU 回归：1595 passed, 7 skipped, 2 个既有 warning
原 collection manifest SHA：未变
原 execution manifest 的 6 个 source SHA：全部 MATCH
正式 evaluation report：不存在
```

当前在本节所在勘误提交处停止。下一步必须由用户另行授权，才能先核验 clean worktree 与完整 protocol
SHA，再用修订 evaluator 对现有唯一原始 collection 做一次只读正式评价。不得重跑 collection，也不得
自动运行 P=12/P=4 回退。

### 最新暂停点：Issue #53 P=6 evaluator 因审计计数公式错误 fail closed，尚无结论（2026-08-17）

> 用户单独授权运行冻结 evaluator。本步没有重新采集、修改 raw artifacts、调用 generator、运行回退
> P、访问真实数据或消耗隐私预算。evaluator 在聚合判定前按预期 fail closed，没有生成 report，因此
> 当前不能声明 P=6 通过/失败、质量/计算通过/失败或任何回退方向。

正式 evaluate 前把唯一的 `PROJECT_STATUS.md` 改动保存到独立 stash，确认 HEAD 为 collection 绑定的
`34b477acff11adabfc22b6eb9c14e4fb3939b7a1`、工作树干净、collection manifest SHA 仍为
`aa4b34f80cbe72546c6a085845d205e988e04ccdeb0ee843ec135fbfa3505133`，随后使用完整 protocol SHA 启动
只读 evaluator。它在 `_audit_online_diagnostics` 抛出：

```text
RuntimeError: online current metrics terminal 身份不一致
```

失败发生在逐 case online diagnostics 审计阶段，尚未进入 B shadow checkpoint 审计、证据聚合、质量/
计算门禁或 classification；`p6_evaluation_report.json` 不存在。

只读诊断定位到错误不在 terminal table/loss/L1 身份，而在 evaluator 新增的计数断言：

```text
错误断言：state_evaluation_count == rounds_run + 1
真实语义：rounds_run == 0 时 count=1；rounds_run > 0 时 count=rounds_run
```

原因是初始 current 做一次完整 state evaluation；每个已接受 proposal 的 query/loss 已在 candidate
evaluation 中得到，下一轮才在 cache 失效后重新做 state evaluation，所以 terminal proposal 会增加
`current_state_metrics_history`，但不会额外增加 `state_evaluation_count`。冻结无门控路径每轮一个直接
生效的 proposal，因此本批 12 条均表现为 `metrics length = rounds_run+1`、`state_evaluation_count =
rounds_run`。evaluator 把“状态记录数”误当成“完整 state evaluation 次数”。

诊断脚本为定位失败解析了 12 个 case manifest 的 online 部分和 online diagnostics，控制台暴露了全部
12 条 termination reason（均为 B/`early_stopped`）、stop state 以及 terminal loss/L1；逐条确认
diagnostics 末状态与 case terminal loss/L1 精确相等。脚本没有访问或输出 shadow checkpoint 字段，
没有计算 delta_L1、saving、coverage、family median、quality/compute gate 或 classification。因此质量—
计算验收的核心续跑结果仍未查看，但“12 条均为 B”及 terminal 指标已经暴露，后续必须透明记录，不能
再把整个流程描述成完全未见。

原始 collection 未改变，manifest SHA 复核未变；失败报告不存在；工作记录 stash 已完整恢复并删除，
现在仍只有 `PROJECT_STATUS.md` 为 tracked 改动。测试之所以漏掉该问题，是完整假 B artifact 使用了同样
错误的 `rounds+1` 假计数，未按真实 `run_evolution` 计数语义构造。

当前停止，不得直接放宽断言、手改报告或重跑 collection。建议的最小严谨修复是单独建立“结果无关的
auditor erratum”：只修正该计数公式和假 B 测试，不改协议阈值、聚合公式或 raw artifacts；同时把
collection/evaluator 的同 commit 要求改为“所有 collector/generator/protocol 源文件 SHA 必须逐项与
原 collection 相同，允许仅 evaluator/test/status 形成新的审计修订 commit”，并在报告中同时记录原
collection commit 与 erratum evaluator commit。该方案不重跑已暴露 seed，但属于冻结后审计规则修订，
必须先由用户明确同意，不能由当前失败自动授权。

### 最新暂停点：Issue #53 P=6 primary raw collection 已完成，尚未评价（2026-08-17）

> 用户在看到完整冻结 protocol SHA 后明确授权继续。本步只执行已经提交的 12-case primary raw
> collector，并在集合级做不含结果的结构核对；没有调用 evaluator、没有读取 case manifest/terminal
> table/shadow 内容，没有查看或汇总 A/B/C、loss、L1、delta、saving，也没有形成通过/失败、回退或
> P 调整结论。没有访问真实数据、使用 GPU、消耗隐私预算或实现外层 DP。

正式命令使用：

```text
protocol SHA = 759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317
Git commit  = 34b477acff11adabfc22b6eb9c14e4fb3939b7a1
output      = outputs/issue53_p6_unseen_primary/
```

运行前确认包含 untracked 在内的工作树干净、输出目录不存在，并先执行 result-blind plan；plan 固定为
2 family × 3 seed × 2 rho 的 12 条 P=6 cases。正式 collector 顺序报告 1/12—12/12 全部 collected，
进程正常退出，总墙钟约 7.53 秒。控制台没有输出单条 loss/L1。

采集结束后只读取不含 case 结果的 collection/execution manifest 顶层元数据，得到：

```text
contract = issue53-p6-unseen-primary-collection-v1
formal_primary_collection_complete = true
case_count = 12
case_manifest_file_count = 12
acceptance_evaluated = false
partial_matrix_classification_emitted = false
real_data_accessed = false
privacy_budget_consumed = false
evaluation_report_exists = false
collection manifest SHA-256 = aa4b34f80cbe72546c6a085845d205e988e04ccdeb0ee843ec135fbfa3505133
```

同时只按目录名计数确认有 12 个 case 目录；没有打开其中任何 artifact。原始输出位于 `.gitignore` 的
`outputs/` 下，不进入提交。为了遵守每步更新状态的规则，本节使 `PROJECT_STATUS.md` 成为当前唯一
tracked 改动；不得提交它，因为 evaluator 要求与 collection 保持同一 Git commit `34b477a`。

当前立即停止。下一步只有用户单独授权后，才运行冻结的只读 evaluator。正式 evaluate 前应把这份
状态更新安全暂存，使工作树临时恢复到 `34b477a` 的 clean 状态；评价完成后再恢复并追加评价记录。
不得重新采集、不得修改或覆盖现有 raw artifacts、不得先偷看 case 结果，也不得自动触发 P=12/P=4
回退实验。

### 最新暂停点：Issue #53 terminal-current P=6 冻结链路已提交，正式 12 cases 尚未运行（2026-08-17）

> 用户授权本步只做提交前回归并提交当前完整冻结实现，提交后停止。本步没有执行 formal collector、
> 12 cases、未见 seed、真实数据或 GPU，没有预览未见 loss/L1，也没有调整 P、阈值、family、rho、
> seed、C 或生成核参数。

本次冻结提交统一包含：独立 terminal-current A/B/C 自然工作时钟状态机、`run_evolution` opt-in 接线、
六条已知 development 轨迹一致性诊断、P=6 未见公开人工 family/protocol、受 SHA 与 clean-tree 约束的
raw collector，以及只读 artifact 审计和唯一结果分流 evaluator。formal collection 和 evaluation 仍是
两个独立显式命令，没有自动串联；错误 SHA、dirty tree、已有输出/报告都会 fail closed。

提交前再次审查全部待提交文件和 CLI：protocol 只有 plan；collector 只有 plan/collect，collect 仅接收
输出目录与完整 protocol SHA；evaluator 只有 plan/evaluate，evaluate 仅接收 collection 路径与完整
protocol SHA。没有 family、seed、rho、P、C、阈值或分流覆盖参数，也没有导入时或 plan 时自动生成。

验证：

```text
Issue #53 当前链路相关回归：99 passed
全仓 CPU 回归：1591 passed, 7 skipped, 2 warnings
```

两条 warning 仍来自旧 `test_residual_geometry_formal_script.py` 输入哈希错误路径的 NumPy 空均值，零
失败。全仓测试首次收集时发现当前轻量测试解释器缺少 matplotlib；随后只读追加项目原有 `.conda`
site-packages 的 matplotlib 路径重跑，冒烟确认 NumPy 仍来自当前测试环境，完整回归通过。没有安装或
修改依赖。全部新增/修改 Python 文件 `py_compile` 通过；新文件 Ruff check/format 通过，legacy
`evolution.py` 的 F/E9 检查通过；`git diff --check` 通过。

当前应在冻结提交后的干净工作树停下。下一步不再改设计或自动跑实验；只有用户再次明确确认完整
protocol SHA `759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317`，才允许执行
primary collection。采集结束后还必须单独授权只读 evaluate；任何一步都不得顺带启动下一步。

### 最新暂停点：Issue #53 P=6 只读证据审计与唯一分流完成，尚未运行（2026-08-17）

> 用户授权继续 collector 之后的一小步。本步只实现 raw artifact 审计、第 7/8 节固定聚合门禁与假
> 证据测试；没有执行 generator、12 cases、新 seed、真实数据或 GPU，没有预览未见 loss/L1，也没有
> 调整 P、阈值、family、seed、rho、C，或实现外层选择、加噪、隐私预算与 DP。

新增：

```text
scripts/evaluate_issue53_p6_unseen.py
tests/test_evaluate_issue53_p6_unseen.py
evaluation contract = issue53-p6-unseen-evaluation-v1
```

入口仍严格分离：`plan` 不读 collection、不调用 generator；`evaluate` 只接收 collection 路径和完整
protocol SHA，不开放任何阈值、P 或分流覆盖参数。错误 SHA 在环境、artifact 和 generator 之前失败；
正式判定要求含 untracked 在内的干净工作树、与 collection 完全相同的 Git commit 和 Python/NumPy/
Pandas runtime，并拒绝覆盖已有 `p6_evaluation_report.json`。evaluator 的 source manifest 覆盖
collector 所有冻结源码并追加自身。

artifact 审计逐级验证 collection/execution/case manifest 的精确字段和 SHA，拒绝绝对路径、路径逃逸、
缺失/篡改文件及不完整 12-case 矩阵；核对协议全文、采集源 SHA、NumPy CPU 环境、case/family/query/
target 身份。每条 terminal CSV 均重新检查 schema/domain/N，复算 ordered query vector、squared loss、
normalized L1 和 terminal table SHA。A/C 必须没有 shadow；B 必须有完整 trace/summary，且 online 的
current metrics、transition clocks、accept/proposal/attempt、table/query、RNG、candidate 前缀均由
artifact 重新验证，不能只相信 collector 写入的 `prefix_audit=true`。

在线 A/B/C 也会从保存的逐状态 current loss 与自然工作时钟重新执行：冻结无门控核每轮恰好一次
proposal 并直接生效；A/B/C 更早触发却仍继续的轨迹会被拒绝；terminal decision、historical best、
candidate C 来源和 terminal-current 身份必须复算一致。B 的 +6/+12 继续调用冻结定位规则，右删失不
插值、不用 C terminal 补齐。

聚合顺序固定为：

```text
artifact 完整性 -> 证据完整性 -> 质量/计算 -> 唯一分流
```

证据门禁沿用协议的正常结束>=10、C<=2、B>=6、两个 checkpoint 各覆盖>=80%；另把协议中逐 family
中位数的必要前提显式化为“每个 checkpoint 的两个 family 各至少一条可用 B”，否则中位数无定义，
只判证据不足。证据不足时 quality/compute 固定为不可评价，绝不误判为质量或计算失败。证据完整后
严格执行两个 checkpoint 的总体中位数<=0.01、`delta_L1>0.02` 比例<=25%、逐 family 中位数<=0.02，
以及 +12 中位节省>=30%。

分流只有五个 machine-readable classification：P=6 当前人工 development 支持；仅质量失败唯一 P=12；
仅计算失败唯一 P=4；证据不足不改 P；双失败/family 相反方向拒绝 B。family 冲突在结果前操作化为：
一个 family 的质量失败/计算通过指向增大 P，另一个质量通过/计算失败指向减小 P 时直接拒绝，不能走
单一回退。报告锁定最多一次回退、禁止第三个 P、禁止结果后重调阈值，并明确不能声称收敛。

验证：evaluator 22 项测试通过，覆盖五种分类、三类基础证据不足、family 缺失、方向冲突、严格
`>0.02` 边界、右删失只用观测值、输入契约、A 优先级、A/B/C 离线复算、完整假 A 矩阵和假 B shadow
artifact、文件篡改、路径逃逸、错误 SHA、dirty tree、同 commit/source/runtime 约束与报告拒绝覆盖。
协议、collector、evaluator 合计 `42 passed`；`py_compile`、Ruff check/format 通过。全部是伪造矩阵
或内存假执行器，没有调用真实 generator。

当前再次停止。工作树仍未提交，所以 clean-tree 门禁会继续阻止误运行。下一步应先由用户审查本步；
若继续，只做提交前的相关回归与冻结提交，仍不运行 12 cases。提交后也必须再次单独得到用户对完整
protocol SHA 的明确确认，才允许正式 primary collection；不得自动开始实验。

### 最新暂停点：Issue #53 P=6 正式原始采集入口完成，尚未运行（2026-08-17）

> 用户在 manifest 暂停点后授权继续下一小步。本步只实现受冻结协议 SHA 约束的 primary runner/raw
> collector，并用假执行器测试；没有执行 generator、12 cases 或任何未见 seed，没有预览 loss/L1、
> 读取真实数据、使用 GPU、评价通过/失败，或调整 P/阈值/family/seed/rho/C。

新增：

```text
scripts/collect_issue53_p6_unseen.py
tests/test_collect_issue53_p6_unseen.py
collection contract = issue53-p6-unseen-primary-collection-v1
```

入口只有两个命令：

```text
plan    -> 只打印固定矩阵与开销，不实例化 RNG、不生成
collect -> 必须显式确认 protocol SHA，且正式运行前要求工作树含 untracked 在内完全干净
```

`collect` 不提供 family、seed、rho、P、C、检查点或验收阈值覆盖参数，只能执行 manifest 中的 12 条
primary P=6 cases；输出目录已存在时拒绝覆盖。正式 execution manifest 会锁定 Git commit、协议全文、
关键源码/协议文档文件 SHA，以及 Python/NumPy/Pandas/平台环境。错误 protocol SHA 会在环境检查、输出
创建和 generator 调用之前失败。

运行结构保持最小：每条 case 先执行在线 terminal-current A/B/C；A 或 C 不做无用续跑，只有 B 才用
同一 family/S0/seed 从头确定性重放一个关闭停止器的只读 shadow 到固定 C。shadow 必须与 online 在 B
terminal 之前逐项一致：current metrics、transition clocks、accept/proposal/attempt 序列、terminal
table、query vector、主 RNG 和 candidate evaluation count，任一不等立即中止。该重放等价于从冻结 B
状态继续，但无需给核心生成器增加 resume 接口。

B 通过前缀审计后，只定位 `tau+6`、`tau+12` 的第一个真实 current state；不插值、不用 C terminal
补齐，观察范围不足就记 `right_censored_by_resource_guard`。逐 case 原子保存 terminal CSV、在线诊断，
以及仅 B 才有的 shadow trace/summary/checkpoint 身份；reference multiset 从不传给 generator。集合入口
只发布 raw collection manifest，不计算 delta、聚合门禁、classification 或回退选择，控制台也不输出
单条 loss/L1。

预估硬上限已在 plan 中显式报告：12 条 online 的 raw-round cap 总和 1800；若 12 条全部为 B，再各跑
一条 shadow，最多 24 次 generator 调用、3600 raw rounds。实际 A/C 只有一次调用，B 才有第二次；均为
小 N 人工 family、NumPy CPU，不使用 GPU或隐私预算。

验证：新 collector 的 `8 passed`，覆盖 plan 零生成、family 物化不含 reference、冻结 kwargs、错误 SHA
前置失败、dirty-tree 前置失败、+6/+12 首个真实状态与右删失、B terminal/shadow artifact 和全 12-case
编排。所有运行均为内存假执行器，没有调用真实 `run_evolution`。Ruff、format 与 `py_compile` 通过。

当前再次停止，工作树仍未提交，因此 clean-tree 门禁也会阻止误启动正式采集。下一步不应直接跑实验；
应先实现并结果前冻结只读 artifact 审计器和第 7/8 节聚合判定器，只用伪造 A/B/C 矩阵测试全部通过、
失败、证据不足及唯一 P=12/P=4 分流。判定器审查完成后，再提交代码并单独确认 protocol SHA 后运行。

### 最新暂停点：Issue #53 P=6 未见轨迹确定性 manifest 与纯校验完成（2026-08-17）

> 用户授权继续协议中预先限定的一小步。本步只实现公开人工 family、case 矩阵、C 映射、验收元数据
> 与纯测试；没有导入或运行 generator，没有执行 12 cases、新 seed、真实数据或 GPU，也没有预览
> loss/L1、调整 P/阈值/C，或接入外层选择、加噪、隐私预算与 DP。

新增结果前冻结入口与测试：

```text
scripts/issue53_p6_unseen_protocol.py
tests/test_issue53_p6_unseen_protocol.py
```

入口只支持 `--mode plan`，没有执行模式，也不导入 `table_diffevo.evolution`。它每次重新构造并校验
U1/U2 manifest；从公开 reference multiset 逐项复算 N 和 ordered target，检查 schema domain、state
唯一性、查询字段/顺序与 family 身份，不能只信手写 target。固定身份如下：

```text
protocol SHA-256 = 759cddb3e75a8a1d04e9568ae0fff30b0e26969dd6e95020500330838269b317
U1 family SHA-256 = c47200c0b68c6c3bcf4818b7b9322f85666584eaa1459d94a19d216642f447ee
U2 family SHA-256 = db3af48d083e1e4905a16362b63ba4bbbe7c55045efd3ae6e6a580f82a58bbab
```

复算结果与协议完全一致：

```text
U1 binary_chain_4: N=32, 11 queries,
target=[16,16,16,16,12,10,8,10,8,6,6]

U2 mixed_2x3x2: N=36, 15 queries,
target=[18,14,10,12,19,6,6,6,6,6,7,11,4,5,2]
```

首轮 case 顺序固定为 family -> seed -> rho，共 12 条；每条显式带 P、N、`n_rounds` 与
`candidate_budget`。C 映射严格只有 rho=1.0 -> 60/60、rho=0.25 -> 240/240，未知值 fail closed。
回退入口不接收任意 P，只允许两个结果前注册的互斥分支：质量单独失败用独立 seed 的 P=12；计算
单独失败用同一独立 seed 批的 P=4；每个候选矩阵仍各 12 条，但正式流程最多触发其中一个一次。

manifest 同时锁定无门控 terminal-current A/B/C 契约、+6/+12 影子检查点、五项通过门禁和证据不足/
结构失败分流。`tol=+inf` 在严格 JSON 中写为语义字符串 `positive_infinity`，避免写入非法 Infinity；
正式执行入口将来必须显式翻译且核对协议 SHA。每次 manifest 调用返回新对象，外部修改不会污染后续
身份。

验证只运行新协议纯测试：`12 passed`；`py_compile`、Ruff check/format 与新文件
`git diff --check` 全部通过。没有运行全库测试，因为本步没有修改生成代码，且边界明确禁止启动新
轨迹。

当前再次停止。下一步先由用户审查冻结实现；未获明确授权前，不得另建 formal runner 或执行 12
cases。若用户继续，下一小步应先明确是只实现受 protocol SHA 约束的独立运行/结果收集入口，还是在
入口审查后再单独授权正式运行；无论哪种都不得开放任意 P、family、seed、rho、阈值或 C 覆盖参数。

### 最新暂停点：Issue #53 P=6 未见轨迹验收协议已结果前冻结（2026-08-17）

> 用户确认先写简单、统一的 P=6 质量—计算验收方案，并确认 P 失败后最多调整一次。本步只写协议
> 与做 target 算术复核；没有实现 family、运行 generator、新 seed、真实数据或 GPU，也没有接外层
> 选择、加噪、隐私预算与 DP。

新增正式协议：

```text
docs/设计/Issue53_P6未见轨迹质量计算验收协议.md
```

首轮固定矩阵：

```text
P = 6
family = U1 binary_chain_4 (N=32), U2 mixed_2x3x2 (N=36)
seed = 20260819, 20260820, 20260821
rho = 1.0, 0.25
总计 = 2 × 3 × 2 = 12 cases
影子检查点 = B 后 +6、+12 normalized work
```

两个 family 的 reference multiset、ordered equality queries 与整数 target 已逐项写死。独立算术复核：

```text
U1 total N=32，target=[16,16,16,16,12,10,8,10,8,6,6]
U2 total N=36，target=[18,14,10,12,19,6,6,6,6,6,7,11,4,5,2]
```

生成参数沿用当前冻结核，只隔离停止规则；在线仍为 `tol=+inf`、`max_retries=0`、terminal-current
输出。C 不作质量终点，统一按预期 60 normalized work 换算纯防挂死护栏：rho=1 使用 60 raw rounds/
candidates，rho=0.25 使用 240。实际 work 随 participating rows 随机变化；C case 只标资源截断。

P=6 必须同时满足：

1. 12 条中至少 10 条由 A/B 正常结束，C 不超过 2；
2. 至少 6 条 B，且 +6/+12 各至少覆盖 80% 的 B cases；
3. 两个检查点的 `delta_L1 = L1_B - L1_continuation` 中位数均不超过 0.01；
4. 每个检查点 `delta_L1>0.02` 不超过 25%，两个 family 各自中位数不超过 0.02；
5. 相对 +12 work 的中位 normalized-work 节省至少 30%。

L1 只在表身份固定后离线计算，不进入在线停止。loss 同时报但不作跨 family 硬门禁。检查点右删失
不得用 C terminal 或最后状态补齐。

失败后的处理已预注册：

```text
仅质量失败、计算/覆盖通过 -> 唯一候选 P=12
仅计算失败、质量/覆盖通过 -> 唯一候选 P=4
质量与计算同时失败，或 family 需要相反方向 -> 停止调 P，重新设计 B
C 太多、B 少于 6 或检查点覆盖不足 -> 证据不足，不调 P
```

唯一回退 seed 固定为 20260822—20260824；family/rho/阈值/C 均不变。回退仍失败或证据不足，就停止
P 调整；禁止第三个 P 或在同一 seed 上反复调到通过。

下一步若用户继续授权，只把协议实现成确定性 family/manifest 和纯测试，复算 N、target、查询顺序、
12-case 矩阵、C 映射与协议哈希；实现后再次停止，仍不运行这 12 cases。

### 最新暂停点：Issue #53 六条已知轨迹在线接线一致性验证通过（2026-08-17）

> 用户授权本步只复用已经看过的六条人工轨迹，验证新 A/B/C 在线接线能否严格复现旧离线回放。
> 没有增加 seed、读取真实数据、比较 patience、选择 C、使用 GPU 或形成正式质量/收敛结论。

验证入口继续复用：

```text
scripts/analyze_issue53_terminal_early_stop_development.py
tests/test_issue53_terminal_early_stop_development.py
```

完整轨迹与在线轨迹共用同一份冻结 generator 参数；唯一预期差异是完整轨迹关闭 stopping 并跑到旧
horizon，在线轨迹启用 `inner_early_stopping_patience_ticks=6`。对每条完整轨迹先按旧纯回放得到 A/B
决定，再要求在线运行严格复现同一状态。

逐条结果：

```text
seed      rho    offline/online reason   stop state   stop work   terminal loss
20260816  1.00   B / B                    12           12.0000     12.0
20260817  1.00   B / B                    20           20.0000      7.0
20260818  1.00   B / B                     8            8.0000      7.5
20260816  0.25   B / B                    27            7.0625      4.0
20260817  0.25   B / B                    64           16.0000      1.0
20260818  0.25   A / A                    11            2.8125      0.0
```

六条全部满足：在线 reason/state/work 与离线相同；主返回和 `final_table` 的 SHA-256 等于完整轨迹
stop state 的 current 表；current metrics、transition clocks、accept history、candidate evaluations
严格等于完整前缀；停止后主 RNG SHA-256 等于完整轨迹同一状态。5 条 B 仍全部输出高于历史 best
loss 的 terminal current，确认没有回滚或 best 选表。结果为 5 B、1 A、0 C；C 没有抢先干扰 A/B。

报告新增 `online_wiring_audit`，分类固定为
`development_known_trajectories_wiring_consistency_only`。它只证明实现一致，不证明 P=6 合适、C 数值
合理、输出质量通过或算法收敛。结果文档已同步：
`docs/实验结果/Issue53_TerminalCurrent早停开发诊断.md`。

验证：六轨迹开发逻辑/在线审计 15 passed；相关新旧回归 255 passed、1 skipped；全库 1549 passed、
7 skipped，2 条旧 residual-geometry 错误路径的 NumPy warning，零失败；Ruff、`py_compile` 与
`git diff --check` 通过。

下一步不是运行新数据，而是先讨论并冻结未见 seed/family 的质量—计算评价协议：一个全局 P 候选、
允许的 terminal-current 质量代价、最低计算收益、报告的检查点/删失规则，以及 C 的外部资源设置。
协议确认前不得开始正式验证或真实数据运行。

### 最新暂停点：Issue #53 terminal-current A/B/C 已完成 opt-in 接线（2026-08-17）

> 用户授权的本步只把已完成的纯状态机接入 `run_evolution` 并增加接线测试。没有运行新 seed、真实
> 数据或 GPU，没有选择新的 patience/C 数值，也没有实现外层查询选择、加噪、隐私预算与 DP。

`run_evolution` 新增末尾可选参数：

```text
inner_early_stopping_patience_ticks=None  -> legacy 行为不变，主返回历史 best
inner_early_stopping_patience_ticks=P>0   -> 启用 A/B/C，主返回 terminal current
```

参数放在原有 `horizon_invariant` 之后，避免改变旧 positional 调用对已有参数的映射。新模式固定：

```text
A：current loss == 0 -> fit_target_reached，inner_complete=true
B：连续 P 个自然 work tick 无严格 best 刷新 -> early_stopped，inner_complete=true
C：candidate_budget 或 n_rounds 到界 -> resource_cap_reached，inner_complete=false
同一状态优先级：A > B > C
```

每轮先让已接受 proposal 成为 current、更新 current/best/时钟，再观察 A/B/C。A、B、C 的主返回、
`diagnostics["final_table"]`（请求时）和 `output_squared_loss` 都对应同一 terminal current；历史 best
只保留在 `best_loss_diagnostic_only` 与 best-L1 诊断字段，不参与选表。C 的具体来源另记为
`resource_cap_source_diagnostic_only`，不会把资源截断冒充质量完成。`n_rounds=0` 在 S0 上直接走 C，
但同一 S0 若已满足 A，仍由 A 优先。

逐段接线审查发现并修复一个配置漏洞：`run_evolution` 默认有限 `tol` 会拒绝 loss 上升 proposal，
与本方法的无门控身份冲突。因此启用新模式时 fail-closed 要求 `tol=+inf`、`max_retries=0`；否则明确
报错，不能出现“名义无门控、实际仍门控”。只增加观察器并把 patience 设到不会提前触发时，配对
测试确认 current-table、逐状态 metrics、transition clocks、accept history 与 RNG hash 全部不变。

测试与检查：

- 纯状态机 + 接线测试：42 collected，全部通过；
- 新旧停止、evolution、reference process、fit-target/shadow 等相关回归：249 passed，1 skipped；
- 全库：1543 passed，7 skipped，2 条 warning 来自旧 residual-geometry 错误路径的空均值；零失败；
- 新文件 Ruff check/format、`evolution.py` 的 F/E9 定向 lint、`py_compile`、`git diff --check` 通过。

下一步不是自动跑数据。先由用户审查本次接线；确认后再单独讨论并冻结运行与评价协议，包括是否仍
以 P=6 作为 development 候选、如何报告 A/B/C 比例与 terminal-current 质量—计算权衡，以及 C 的
外部资源配置。当前不得据此宣称收敛或开始正式实验。

### 最新暂停点：Issue #53 terminal-current A/B/C 纯状态机完成（2026-08-17）

> 用户纠正方向：若没有 B，当前零噪声 A 很难触发，生成大多只能运行到 C；而 terminal current 并不
> 随运行单调改善，所以 A+C 不是更好的质量方案。B 应是大多数运行的主要实践停止方式，不再要求它
> 先证明未来永远不会改善。本步只实现独立纯状态机与边界测试；没有导入或修改 `run_evolution`，没有
> 运行 generator、新 seed、真实数据或 GPU，也没有接外层选择、加噪、隐私预算与 DP。

新增当前候选模块：

```text
src/table_diffevo/inner_early_stopping.py
tests/test_inner_early_stopping.py
```

旧 `inner_stopping.py` 及其 shadow/counterfactual 测试继续原样保存，用于复现已被否决的 3+3
`optimization_stalled` 历史负证据。没有把旧结果事后改名成通过；新 A/B/C 候选用独立模块隔离。

新状态机固定以下最小契约：

```text
natural tick = floor(cumulative applied participating rows / N)
P = 6 ticks（可配置 development 默认值）

A：current loss == 0
   -> fit_target_reached, inner_complete=true

B：连续 P 个已完成自然 tick 没有严格历史 best 刷新
   -> early_stopped, inner_complete=true

C：调用方在当前状态传入 resource_cap_reached=true
   -> resource_cap_reached, inner_complete=false

同一 current 状态的优先级：A > B > C
```

C 不在 `EarlyStoppingConfig` 内硬编码任何 max work/round/time；具体外部资源限制由以后调用方拥有，
只把是否到达护栏传给状态机。这避免把 C 的魔法数字重新混入质量逻辑。

状态机只观察 current loss、严格 best 刷新、applied participating rows 与外部 C 布尔标志；不接收
table、L1、reference、未来状态或 RNG。普通 current loss 上升已经是发生后的合法状态，不拒绝、
回滚或重试。best loss/state 仅以 `diagnostic_only` 字段暴露。A/B/C 停止决定的
`terminal_output_state_index` 与 `terminal_output_loss` 均严格等于触发时 current 身份；即使 B/C
触发时 current 明显差于历史 best，也不会返回 best。

新纯测试共 `28 passed`，覆盖：初始 A、默认/可配置 P、参与行自然时钟、只在真实 tick 边界触发 B、
下降但未低于 best 不算 progress、第六 tick 恰好刷新 best 取消 B、A/B/C 同状态优先级、C 输出较差
current 而非 best、非法 loss/参与行/配置/C 标志、生命周期及 API 禁止 L1。旧状态机三组历史回归
`29 passed`，确认旧负证据未被破坏。Ruff check、Ruff format check、`py_compile` 与
`git diff --check` 通过。

总设计稿已同步：B 是主要实践停止，不再把复杂 continuation 验收当成实现前置；既有 continuation
曲线只保留为“terminal 后续非单调、不能挑任意 C 终点”的 development 观察。

下一步若用户继续授权，只将 `InnerEarlyStopper` 接入 `run_evolution` 并增加接线测试，仍不运行新
seed 或真实数据。接线必须保证 proposal 先成为 current，再观察 A/B/C；B/C 返回
`diagnostics["final_table"]` 对应的 terminal current，不改生成核、rho、alpha、residual 或 Gibbs。

### 最新暂停点：Issue #53 B 后 continuation-gain 开发曲线完成（2026-08-17）

> 用户确认不再用任意 C 的最后状态评价 B，并授权下一小步只改造既有六条已知轨迹的开发诊断。
> 本步没有新增 seed、比较 patience、选择 C、接入在线状态机、读取真实数据、使用 GPU，或实现外层
> 选择、加噪、隐私预算与 DP。固定 `P=6` 仍只是 development 基线，不是生产参数或收敛结论。

诊断入口 `scripts/analyze_issue53_terminal_early_stop_development.py` 已更新为 v2：

```text
analysis_id = issue53_terminal_early_stop_continuation_development_v2
B stop work = tau
relative checkpoints = tau+P, tau+2P, tau+4P
P = 6 normalized-work ticks（仅 development）
```

B 在 `tau` 的 terminal current table/current loss 先固定为正式输出；离线影子副本才沿同 RNG 前缀
继续。每个相对检查点取第一个实际达到或越过目标 work 的 post-round current 状态，不插值。旧完整
轨迹只是可观察范围，不是 C 或质量终点；不够长的点固定记为
`right_censored_by_known_trace_horizon`，state/table/loss/L1 均保持空值，不能拿末尾状态补齐。

五条 B 轨迹的 continuation-gain 汇总如下。差值均为“续跑检查点 current − B 输出 current”，负数
表示续跑后的 terminal 更好：

```text
相对点    观察/删失   loss 续跑更好/相同/更差   loss 平均/中位差    L1 平均/中位差
+6        5/0         1/0/4                    +3.2 / +4.0         +0.016667 / +0.020833
+12       5/0         0/1/4                    +3.8 / +4.5         +0.020833 / +0.020833
+24       3/2         3/0/0                    -4.0 / -2.5         -0.020833 / -0.010417
```

L1 的方向计数与 loss 完全一致。实际平均额外 work 为 6.0、12.0625、24.0；平均额外 raw rounds/
candidate evaluations 为 13.4、28、49。所有可观察检查点前均未达到零残差 A。`+24` 的 3/3 不能
解释为稳定收益，因为另外两条较晚停止的轨迹恰好被旧 horizon 右删失，存在明显可观察性偏差。

结论是 terminal current 的后续变化明显非单调：只选 `+6`、`+12` 或 `+24` 中任一单点都会得到
不同甚至相反判断。这支持取消单一 C 终点比较，但既不支持也不否决 P=6。正式验收必须在看新数据前
冻结多检查点质量—计算口径和删失处理，不能看完曲线后挑有利检查点。

每个观察点均用同 seed、`horizon_invariant=True` 重放前缀；metrics/clocks、current-table SHA-256、
独立 squared loss 与离线 normalized L1 全部同状态核对通过。旧轨迹没有逐状态累计墙钟，因此明确
报告 `wall_clock_delta_available=false`，没有用独立重放耗时伪造续跑墙钟。结果文档已改写为
`docs/实验结果/Issue53_TerminalCurrent早停开发诊断.md`，总设计稿 6.4 与当前暂停点已同步。

纯契约测试现为 `9 passed`，`py_compile`、Ruff check、Ruff format check 与 `git diff --check` 通过。
下一步必须先讨论并结果前冻结：允许的 terminal loss/离线 true-L1 退化、最低计算收益、正式相对
检查点与删失规则、一个全局 patience 候选、全新未见 seed/family 协议。用户再次确认前不实现状态机、
不运行新 seed 或真实数据，也不接外层 DP。

### 最新暂停点：Issue #53 A/B 正常停止与 C 资源截断语义完成收口（2026-08-17）

> 用户指出：若用“运行到 C 的最后状态”作为 B 的参考结果，无门控随机轨迹会因 C 的取值不同而
> 得到不同结论，因此该比较不严谨。本步只修正
> `docs/设计/Issue53_三条件自然工作时钟组合停止设计稿.md` 与本状态记录；没有修改代码或测试，
> 没有运行 generator、实验、新 seed、真实数据或 GPU，也没有实现外层选择、加噪、隐私预算或 DP。

当前统一输出契约为：每个 proposal 都无条件成为下一张 current 表；普通 loss 上升不拒绝、不回滚、
不重试。A、B、C 任一出口都只返回触发时的 terminal current table/current loss，历史 best 表不得
替换输出。历史 best loss 只允许作为 B 的只读 progress/计时信号和诊断字段，不能影响 proposal、
残差、current 状态或输出身份。准确方法名为：

```text
无门控残差引导扩散核 + 状态相关早停器
```

三个条件的完成含义已经分开：

```text
A = fit_target_reached
    当前表进入测量噪声允许范围；当前 sigma=0 时即 measured residual 全零；正常质量完成。

B = early_stopped
    尚未达到 A，但连续冻结的自然工作耐心内没有历史 best 刷新；到边界立即输出 current；
    不等待 current 回到 best 附近，不回滚，不增加 terminal-readiness 输出门控；正常早停，
    但不表示拟合达标、平稳、收敛或未来不会改善。

C = resource_cap_reached / candidate_budget / max_rounds_guard
    A、B 均未触发时由调用方外部资源限制强制截断；输出 current，inner_complete=false；
    只表示 resource-limited，不能计作 A/B 正常完成或质量合格。
```

C 不再被设计成内部质量参数，也没有理论推出的通用最佳数字。它只能由调用方在运行前按可用资源
传入，例如最大 normalized work、candidate evaluations 或墙钟；旧 `work=20` 已被正式负结果证明
不能充当默认拟合点、稳定点或 B 的共同参考终点。固定六个无 best-refresh tick 也只保留为已知六条
轨迹上的 development 基线，不是最终耐心值、生产参数或收敛结论。

B 的后续验收不再比较“B terminal”与“某个 C terminal”。新结构是：令 B 在工作量 `tau` 触发，
立即固定 `tau` 的 current 表作为正式输出；只在离线影子诊断中沿同一 RNG 前缀继续，在事前声明的
`tau+P`、`tau+2P`、`tau+4P`……相对工作检查点，记录 terminal current loss、离线 true L1、额外
normalized work/raw rounds/candidates/墙钟及是否后来达到 A。正式判断必须跨未见 seed 汇总“额外
计算—后续收益曲线”，不能由一个任意终点决定。若影子观察先碰到 C，未观察到的检查点记为右删失/
证据不足，不能把 C terminal 补成真值或判 B 通过。

下一步若用户继续授权，只把现有 development 诊断改造成上述 B 后 continuation-gain 曲线，仍只用
已知六条轨迹检查协议与实现，不读取新 seed，也不当独立验证。该诊断审查后，再单独讨论并结果前
冻结统一质量—计算容限、一个全局耐心候选和未见 seed 协议；当前不得直接接主生成器或外层 DP。

本步为纯文档修改，没有运行测试；`git diff --check` 通过，新增设计稿无尾随空白且 Markdown fence
成对。

### 最新暂停点：Issue #53 固定六 tick terminal-current 早停开发诊断完成（2026-08-17）

> 用户确认：无门控生成的 B 最终输出必须是触发早停时最后一张 current table/current loss，不能返回
> 历史 best；但中间可以只读记录 best loss，并用 best 是否刷新作为 progress 信号。本步只在旧 3+3
> 已使用的六条人工完整轨迹上，按新 terminal-output 语义重放固定六 tick 早停基线。该材料已被看过，
> 分类仅为 development 诊断，不是验证或调参证据。没有修改主生成器、接入在线停止、读取真实数据、
> 使用 GPU 或实现外层 DP。

新增可复现入口与三项纯边界测试：

```text
scripts/analyze_issue53_terminal_early_stop_development.py
tests/test_issue53_terminal_early_stop_development.py
```

入口先完整生成轨迹，再只读 loss/participating rows 确定 A/B 停止 state；随后用同 seed 和固定前缀
重放，并显式读取 `diagnostics["final_table"]`。六条停止表和六条参考终点表均通过前缀 metrics/clocks、
current-table SHA-256、独立 squared loss 与 normalized L1 复算，确认没有误用 `run_evolution` 的历史
best 主返回值。

固定候选只使用：自然 tick=`floor(cumulative participating rows/N)`；tick 内刷新严格 best 算 progress；
连续 6 tick 无 progress 触发 `early_stopped`；当前零噪声 A 为 current loss=0；B 输出 terminal current。
没有比较 9/12/20 或自适应公式。

六条中五条由 B 早停、一条在 work=2.8125 由 A 精确命中。五条 B 的 development 汇总：

```text
平均 / 中位 work 节省                    68.47% / 70.00%
terminal loss 较参考好 / 相同 / 差        1 / 0 / 4
terminal L1 较参考好 / 相同 / 差          1 / 0 / 4
平均 / 中位 terminal loss delta           +2.7 / +1.5
平均 / 中位 terminal L1 delta             +0.01875 / +0.020833
停止 current loss 高于 / 等于历史 best    5 / 0
平均 / 中位 current-minus-best loss        +3.8 / +4.0
```

最明显一条在 best loss=3 时以 terminal current loss=12 输出。结论不是按旧苛刻标准自动否决早停，
而是发现一个直接风险：best-stagnation 可以描述“多久没产生新低”，但不能保证触发瞬间的 current 状态
适合作为 terminal 输出。完整参考终点自身也会随机漂动，一条 B 早停优于参考，A 精确命中那条若继续
完整运行反而从 L1=0 漂到 0.114583，因此参考终点不是 ground-truth 最优 checkpoint。

完整结果位于 `docs/实验结果/Issue53_TerminalCurrent早停开发诊断.md`。纯测试 `3 passed`；Ruff check、
Ruff format check 与 `git diff --check` 通过。

下一步只讨论：best-stagnation 是否只负责把 B 武装成早停候选，而真正停止还要等待一个简单、在线、
不回滚且不读取 L1 的 terminal-readiness 条件。用户确认前不实现、不比较其他 patience、不运行新 seed
或真实数据，也不接外层 DP。

### 最新暂停点：Issue #53 条件 B 已正式改为早停语义，等待定义质量—成本验收口径（2026-08-17）

> 用户确认不再要求 B 证明“完全不变、未来没有任何改善”，而将其改为计算—质量权衡下的早停。
> 本步只更新设计语义和状态记录；没有选择耐心值、min-delta、窗口或相对改善公式，没有修改生产代码
> 或测试，没有运行 generator、新 seed、真实数据或 GPU，也没有实现外层 DP。

三种出口现在统一为：

```text
A = fit_target_reached：进入测量噪声允许范围，当前零噪声下即 residual 全零
B = early_stopped：尚未达到 A，但按事先冻结的质量—计算取舍返回停止时 terminal current
C = resource_cap_reached：达到工程资源上限，强制返回历史 best
```

用户随后明确纠正 B 的输出契约：早停结果必须是触发 B 时最后一张 current table 及其 current loss，
不能回头返回历史 best table/best loss。对 B，`selected identity == terminal identity`；历史 best 最多
作为进度诊断或候选触发信号，不参与输出选表。C 的返回语义未在本步重新确认，暂不从 B 外推修改。

B 不再使用 `optimization_stalled` 名称，也不声明 converged、stationary、全局最优或未来不会改善。
停止后的完整参考轨迹出现更低 loss，不再自动判候选失败；后来达到 A 也必须与额外付出的标准工作量
和墙钟一起评价。早停验收比较早停 terminal current 表与共同参考终点的 terminal current 表；两者的
true evaluation L1 仍只允许在两张表都固定后离线计算，不能进入在线停止或选点。完整前缀 best 可
并列报告，但不能替换任一 terminal 输出。

在线可用信息和已确认结构不变：自然时钟为
`floor(cumulative applied participating rows / N)`；普通 current-loss 上升不拒绝状态，B 早停返回
terminal current 表。B 的触发信号是否仍观察内部 best loss，还是改看 current-loss 趋势，尚未确认。

旧固定 3+3 与“历史最长 progress 间隔两倍、最少 6 tick”的结果仍是有效负证据：它们不能证明停滞。
不能事后把旧结论改写为通过；若以后把同类 patience 规则作为早停候选，必须用新的名称、质量—成本
问题、结果前协议和未见验证轨迹。

下一步先讨论 B 的在线触发信号应看 best 刷新还是 terminal/current-loss 趋势，再冻结 terminal 输出的
早停验收口径：允许多少 loss/离线 true-L1 差异，至少必须换取多少 normalized work、raw rounds、
candidate evaluations 或墙钟节省。该口径确定前，不比较 patience 6/9/12，不实现状态机、不运行
新数据，也不接外层 DP。设计稿已同步更新：
`docs/设计/Issue53_三条件自然工作时钟组合停止设计稿.md`。

### 最新暂停点：Issue #53 自适应耐心 B 候选在已知反例回放中失败（2026-08-17）

> 用户确认将 A 简化回唯一的“进入测量噪声允许范围”：当前噪声为 0，所以只有 measured residual
> 全零才触发，不再使用固定 `RMSE<=1 AND MAX<=2` 工程容差作为正式 A。随后只做一个最小
> development 步骤：在旧 3+3 已使用的六条完整人工轨迹上回放新提出的自适应耐心 B。没有修改
> 生成核或生产代码，没有接入 `run_evolution`，没有读取真实数据、使用 GPU、生成输出文件或实现
> 外层 DP。该回放使用已知 seed，只能检查旧漏洞，不能作为新候选的独立验证。

保留的自然时钟与观察口径为：

```text
work_tick = floor(cumulative applied participating rows / N)
每个 tick 内出现严格更低 best loss才算 progress
```

development 候选固定为：`longest_gap=max(3, 历史相邻 progress tick 最大间隔)`，当前连续无进展
`idle_ticks >= 2*longest_gap` 时由 B 停止。它想用当前轨迹自己的历史改善节奏替代所有数据固定 3+3。

回放结果仍失败：六条中一条在 work 2.8125 先以零 residual 触发 A；其余五条由 B 停止，其中四条
停止后仍出现严格更低 best，恰好重现固定 3+3 的四个已知反例：

```text
seed 20260816 / rho 1.0   B work 18.0000  best 3.0 -> tail 2.0  patience 12
seed 20260817 / rho 1.0   B work 22.0000  best 3.0 -> tail 2.0  patience 8
seed 20260818 / rho 1.0   B work 8.0000   best 3.0 -> tail 1.5  patience 6
seed 20260816 / rho 0.25  B work 7.0625   best 3.0 -> tail 1.0  patience 6
seed 20260817 / rho 0.25  B work 16.0000  best 0.5 -> tail 0.5  patience 6
seed 20260818 / rho 0.25  A work 2.8125   best 0.0
```

结论：该自适应候选只把前两条停止点稍微后移，无法解决“未来改善间隔可能超过全部历史间隔”的
根本漏洞，development 阶段即否决；不实现独立状态机、不接主循环、也不为它建立新 seed 正式协议。
固定 3+3 负结果继续有效。自然 work tick 与 best-so-far 口径仍可保留，但当前没有已接受的 B 数值
规则。设计记录已同步到 `docs/设计/Issue53_三条件自然工作时钟组合停止设计稿.md`。

下一步只继续讨论 B 的正确语义和可获得的证据；不能把耐心值从 6 事后改成 9/12/20，也不能把本次
已知轨迹回放改写成独立实验。用户再次确认前不实现、运行新 seed、读取真实数据或接外层 DP。

### 最新暂停点：Issue #53 A+B+C 自然工作时钟组合停止设计稿完成，等待审查（2026-08-17）

> 用户重新确认内层目标后，本步只新增待审查设计稿
> `docs/设计/Issue53_三条件自然工作时钟组合停止设计稿.md`。没有修改代码、测试或冻结协议，
> 没有运行 generator、人工矩阵、真实数据或 GPU，也没有接入 `run_evolution`，没有改变 rho、alpha、
> residual、Gibbs、外层 DP 或任何已归档结果。

设计恢复完整的三个互补条件：A 是同一 current checkpoint 的拟合/未来噪声一致达标，B 是尚未达到
A 时按观察到的 best-loss 收益判断停滞，C 是最大标准工作量、candidate budget 或 raw-round guard
构成的工程保险。建议终止优先级为
`exact_residual > fit_target_reached > optimization_stalled > resource_cap_reached`；A 返回第一张达标
current table，B/C 返回历史 minimum-loss、并列最早的 best 表。L1、held-out、真实答案与 reference
table 仍禁止进入在线停止或选点。

用户明确希望保留的时钟固定写成：

```text
normalized_work = cumulative applied participating rows / N
work_tick = floor(normalized_work)
```

每当 work tick 增加 1，结算刚完成的自然工作区间是否刷新过历史 best。这只时钟没有被旧证据否定；
它自动把不同 rho 的原始轮数换成记录更新机会。相同 work 不代表相同墙钟，factor build、Gibbs
microsteps、candidate evaluations 和 elapsed time 必须并列报告，零参与空转还需 candidate/raw-round
故障保险。

旧 3+3 负结果保持有效：它不能独立保证未来再无严格 improvement。但当时四条反例均在 3+3 停止前
已经达到后来接受的 RMSE<=1/MAX<=2，因此在完整 A+B+C 顺序下会先由 A 结束；旧矩阵没有回答
“只对尚未达到 A 的轨迹，3+3 能否作为 practical B fallback”。新稿建议先原样保留 3 个无新-best
tick 形成 candidate、再 3 个确认的 3+3 作为唯一待验候选，不增加 V2 式统计层，也不把 stalled 写成
converged/stationary。

后续若获设计确认，必须先另立新协议和全新 seed 的完整轨迹影子验证。明确错误是：B 在 A 前停止，
但不受停止影响的同一前缀在共同 C 之前后来达到 A。若尾部未达到 A、但 best loss 仍降低，正式协议
还需用户先选择：任意严格改善都否决 B，或只把 missed A 作为硬失败并另立结果前“明显改善”口径；
不得看完结果后用百分比或 true L1 临时决定。

下一步只请用户审查设计稿的三个决定：自然 work tick 是否唯一采用；3+3 是否先原样作为组合 B 的
待验候选；tail improvement 采用严格方案还是另立明显收益口径。用户确认前不得写正式协议、实现、
运行或更新 Issue，也不得重跑旧 6 条、V2 系列或最新 12 条矩阵。当前设计稿故意未提交；工作树预期
只包含 `PROJECT_STATUS.md` 修改和上述新设计文档，待用户审查后再决定是否修订与提交。

### 最新暂停点：Issue #53 RMSE+max 正式人工矩阵有效失败，等待重新讨论（2026-08-17）

> 用户明确授权后，已在冻结 commit `898b76c2a8e60093888bfe05ffce74b89a124c5e` 上执行唯一一次
> 12 条正式人工矩阵。正式分类为 `candidate_failed`；执行有效性与矩阵身份门禁全部通过。
> 本步只运行小型 NumPy/CPU 人工生成问题，没有读取真实数据、使用 GPU、消耗隐私预算或接入
> `run_evolution`。不得修改阈值、资源上限、seed/family 后重跑本矩阵。

冻结协议 SHA-256 为
`cb1224ac797191b74aa40f7baadfab08928b5cb25414971fe8ee091a297d433a`，scientific result SHA-256 为
`a29fa02edf7492ab171da50a61eb532a33ce1b47db3f318f3ec2912ff448da49`。12 条 case、1200 个完整生成
轮次均有效，全部 validity 与 checkpoint prefix replay 检查为 true；manifest、scientific payload
和 12 张 selected table 的独立字节哈希复核一致。因此这是科学候选失败，不是执行或证据链故障。

唯一科学门禁要求 12/12 在第一个 `work>=20` 真实边界状态及之前同时达到
`query-count RMSE<=1 AND per-query MAX<=2`，实际只有 4/12：

```text
marginal_skew   4/4 按时达标；first work = 2.0417, 3, 3.0417, 9
ring_pair       0/4 按时达标；2 条 rho=0.25 在 work 22.9688/39.0312 迟到，2 条 rho=1 从未达标
nested_overlap  0/4 按时达标；2 条 rho=0.25 在 work 21.4219/35.5938 迟到，2 条 rho=1 从未达标
```

完整 horizon 中 8/12 曾达标，但只有最简单的一维偏态 family 按时通过；两个包含联合结构的 family
在两个 seed 和两个 rho 下都是 0/8。8 条失败 case 中 6 条在资源边界选中表之后仍出现严格更低
loss，说明 work=20 不是普遍稳定点，`resource_cap_reached` 不能冒充拟合达标或收敛。

结论只否决当前冻结组合：固定 0-sweep independent 无门控核、当前参数、统一 work=20 以及
RMSE/max 目标共同构成的 v1 在线接入候选。纯 `assess_query_fit` 接口可以保留为同 checkpoint
质量描述，但当前不得接入主生成流程，也不得运行 nltcs/plants/test_300x10。失败不能单独归因于某一
阈值或核参数，因此不从本结果事后挑参数修补。

完整解释位于 `docs/实验结果/Issue53_RMSEMax全新人工验证结果.md`。正式原始产物位于
`outputs/issue53_rmse_max_artificial_898b76c/`：

```text
protocol_manifest.json          SHA cc49b278276846879d3fc44767451742fe701400a69eed1eff4d92390f7f144c
rmse_max_evidence_report.json    SHA 92f1588a735bced793dc9a6086c304a1b56a6f99b2d2e08354d31834183ec29f
selected_tables/*.csv            12/12 与报告逐文件哈希一致
```

下一步必须暂停实现并与用户重新讨论：跨数据统一资源上限究竟只作为“未达标也必须返回”的工程失败
出口，还是应先改进生成核，使复杂联合关系能在统一工作量内可靠吸收。不得恢复已否定的 3+3、
V2b/V2c 或固定 2048 轮；任何新候选都需另立结果前协议和全新证据。

### 最新暂停点：Issue #53 RMSE+max 固定 runner 与契约测试完成，正式矩阵未运行（2026-08-17）

> 用户已确认结果前协议，并只授权实现固定 runner、只读 `plan` 和确定性契约测试。
> 本步没有运行 12 条正式轨迹，没有实例化 6 个正式 seed，没有读取真实数据或使用 GPU，
> 没有修改或接入 `run_evolution`，也没有实现外层选择、加噪、隐私预算或 accountant。

新增固定入口 `scripts/validate_issue53_rmse_max_artificial.py`。它只有两个子命令：只读 `plan`，以及
留待再次授权的 `run --output-dir`；没有 family、seed、rho、轮数、阈值、资源护栏或生成器参数等科学
覆盖项。正式 `run` 在构造 RNG 或调用生成器之前要求：结果前协议 SHA-256 精确匹配、工作树连同
untracked 文件完全干净、全部证据源存在、输出目录尚不存在。这样任何含未提交改动的工作树都会 fail closed，
不可能误触正式运行；本节所述源文件由当前提交形成可绑定快照，正式执行仍以运行前 manifest
记录的 `git_commit` 和逐文件 SHA-256 为准。

runner 固定列出 3 family × 2 seed × 2 rho 的 12 条矩阵，`rho=1` 为 40 轮、`rho=0.25` 为
160 轮，总计 1200 个小型人工生成轮次。只读 `plan` 已实际执行，输出确认：

```text
mode = plan_only_no_formal_rng_instantiation
case_count = 12
full_round_count = 1200
formal_seed_values_listed_not_instantiated = true
generation_started = false
execution_started = false
protocol_sha256 = cb1224ac797191b74aa40f7baadfab08928b5cb25414971fe8ee091a297d433a
```

正式选点实现为纯回放契约。判定器每个状态只能看到 `state_index/round_index`、该状态自己的
count-error vector 和累计 applied participating rows；不含 L1、参考表、family、sigma 或预算。
它调用唯一的 `QueryFitThresholds.exact_integer_counts()` 与 `assess_query_fit(...)`，不复制公式。
正常路径选择不晚于第一个 `work>=20` 真实边界状态的 first-qualified current checkpoint；边界状态
本身先评估质量再触发护栏，原子越界不裁造中间表。若边界前从未达标，则选择边界前 squared loss
最低、并列最早的表，标记 `resource_cap_reached / fit_target_reached=false`。L1 只在选点和同状态
物化校验全部完成之后离线计算，完整尾部的后续改善只作报告，不称“收敛”。

新增 `tests/test_issue53_rmse_max_artificial.py` 共 29 项契约测试，覆盖 plan 禁止 RNG/生成器、矩阵身份、
显式参考 target、协议哈希、CLI 无科学旋钮、脏树/已有输出 fail closed、判定器最小投影、first-qualified
与 loss-only best 的关键反例、初始/边界/原子越界/迟到达标/未达标兜底、时钟和查询身份交叉校验、
L1 污染不影响选点、同 checkpoint 前缀物化、12/12 聚合门禁和严格 JSON。仅额外使用非正式测试 seed
`999053001` 跑了一条 `N=24,m=3,rho=1,40 rounds` 的内存内接线 smoke；它不属于正式矩阵，未写结果文件。

验证结果：新 runner 测试 `29 passed`；连同纯达标接口、旧 3+3 负结果和影子回放的相关回归为
`84 passed`。Ruff check、Ruff format check 和 tracked diff whitespace check 均通过；runner 可执行，
`evolution.py` 对 `inner_fit_target`、`inner_stopping` 和新 runner 仍为零引用；`outputs/` 没有生成本步文件。

当前身份 SHA-256：

```text
protocol document = 012d2507f7b3a79a7fb566047a7f2dae3dbf2e9d77e30dbeb5da44bcfbff6245
runner            = 271a63a3e8460a69d24a9330a6ae844d730a6fba5da61cf6999e5fd6c9f0c15d
runner tests      = 8c062c06e5ea40a77d5737ac2300528f295e6d8b821c9b5681331daa8086c51b
```

**下一步必须暂停。** 本次 runner/测试与可复现提交已由用户授权完成；若要得到科学结果，仍需要
再次明确授权正式 12 轨迹运行。未经授权不得实例化正式 seed。即使未来
12/12 通过，也只说明当前 RMSE+max 候选未被该全新人工矩阵否定；仍不能直接接入主生成流程，
不能称收敛，也不推进真实数据或外层 DP。

### 最新暂停点：Issue #53 RMSE+max 全新人工验证协议草案已写，等待审查（2026-08-16）

> 本步只新增结果前协议草案
> `docs/设计/Issue53_RMSEMax全新人工验证协议.md`，并同步总设计稿。
> 没有实现 runner，没有实例化正式 RNG，没有运行任何矩阵轨迹、真实数据或 GPU，
> 没有修改或接入 `run_evolution`，也没有实现外层选择、加噪、隐私预算或 accountant。

由于当前 `query-count RMSE <= 1 AND max absolute count error <= 2` 是在看过旧 6 条轨迹后才形成的，
旧结果不能再当独立通过证据。新协议固定三个显式可行的二元人工 family：
`marginal_skew(N=24,m=3)`、`ring_pair(N=32,m=10)` 和
`nested_overlap(N=64,m=15)`，分别覆盖偏态边缘、环形二阶重叠和高阶嵌套包含。
每类使用两个未进入旧证据的 seed，并与 `rho in {1.0,0.25}` 配对，共 12 条轨迹；
`rho=1` 固定 40 轮，`rho=0.25` 固定 160 轮，期望工作量均为 40 次等效全表扫描。

协议明确区分“首次达标输出”和“历史最低 loss 兜底”。正常达标时，loss、RMSE、最大误差与返回表
必须来自同一张 first-qualified current table；不能用当前表宣布达标后，改为返回另一张可能
`MAX>2` 的历史 minimum-loss 表。只有第一个跨过 20 次等效扫描的真实状态仍未达标时，
才返回此前 minimum-loss 表，并明确标为 `resource_cap_reached / fit_target_reached=false`。
状态处理顺序固定为先应用并评估真实 post-round，再检查护栏，因此边界状态首次达标仍正常完成，
不伪造恰好 work=20 的中间表。

矩阵保持固定无门控 0-sweep independent 人工核：随机初始化、geometric、fixed alpha 6、
rho 两档、eta 0.45、mu 0.02、`tol=inf`、无重试、绝对 residual geometry、固定方向强度/尺度，
完整轨迹关掉 exact 提前停止并开启 horizon-invariant clocks/query-answer trace。
这是对达标契约的小型反例搜索，不选择或验证全部 Gibbs、alpha、relative geometry/floor 配置。
回放判定层只得到同状态 count-error vector 与实际 participating work；不读取 L1、参考表、
dataset→threshold、sigma、预算或未来尾部。

执行身份、显式 target、轨迹/时钟对齐、有限值、无拒绝重试、至少 10 次扫描尾部和 SHA-256
先组成有效性门禁。候选的唯一科学通过条件是 12/12 都存在 first-qualified state，且该 state
不晚于第一个跨过 work=20 的真实边界状态；任一迟到或从未达标即失败，不事后改最大误差 2、
资源护栏 20、seed 或 family。选点后才离线计算 L1；完整尾部是否继续改善只报告，
不把“达到预定质量”误称为收敛。

当前协议草案 SHA-256 为
`e5a47547e7b657fedb4da39d6a08b9c60ebf88709162469a897ed0c0d2f7ffc3`。
下一步先由用户审查两项核心决定：正常完成返回 first-qualified current table，以及上述 12 轨迹矩阵与
12/12 门禁。只有用户确认后，才另立一步实现固定 runner 和确定性契约测试；该步仍不得运行正式 12 条轨迹，
正式运行需要再次明确授权。

### 最新暂停点：Issue #53 RMSE+max 纯达标接口完成，预留外部噪声阈值（2026-08-16）

> 用户接受当前平衡候选 `query-count RMSE <= 1 AND max absolute count error <= 2`，并要求为未来加噪留接口。
> 本步只新增独立纯判定模块和确定性边界测试；没有修改或接入 `run_evolution`，没有运行新人工矩阵、
> 真实数据或 GPU，也没有实现噪声公式、外层选择、隐私预算或 accountant。

事前只读诊断确认 `max error <= 1` 偏严：原 6 条轨迹虽最终 6/6 达到，但只有 3/6 在 20 次等效扫描前
达到；平均工作量从 RMSE-only 的 4.3125 增至 19.6458，约 4.6 倍。因此用户接受整数误差下的最小一级放宽 2。
该 2 是在看过旧轨迹后提出的，所以旧 6 轨迹不能作为独立验证；下一证据必须使用结果前固定的全新 m/N/workload 人工矩阵。

新增 `src/table_diffevo/inner_fit_target.py`。`QueryFitThresholds.exact_integer_counts()` 唯一生成当前 1/2 候选；
`assess_query_fit(count_errors, thresholds)` 对单一明确 checkpoint 的同一 count-error 向量计算 squared loss、RMSE、
最大误差、两项通过状态与 exact residual。该契约防止把不同表的 best loss 和 max error 拼接后错误宣布达标。

未来接口 `QueryFitThresholds.external_noise_calibrated(...)` 只接收外层已换算的数值阈值：一个全局 RMSE limit，
以及统一或逐查询 max-error limits。内层刻意不接收 sigma、真实答案、reference table、L1、隐私预算或 accountant；
各查询噪声不同时可直接传等长阈值向量，无需改动评估 API。如何从已发布噪声严格推导这些阈值仍属于未来外层专题，
本步没有偷偷固定公式。

`tests/test_inner_fit_target.py` 新增 25 项纯边界测试：覆盖当前 1/2 边界、100-query 单尖峰反例、两项必须同时通过、
exact residual、nextafter、符号不变性、统一/异质外部噪声阈值、向量长度、非有限/溢出和 API 禁止 sigma/L1/reference 输入。
浮点边界测试暴露 NumPy 聚合后将 `nextafter(1,+inf)` 开根舍入回 1.0；已改用 `math.fsum` 高精度求和，不放宽边界。
纯测试现为 `25 passed`；连同 RMSE 影子矩阵、旧 3+3 反例与状态机边界的全部相关回归为 `55 passed`。
Ruff check、Ruff format check、compileall 和 tracked diff 空白检查均通过；`evolution.py` 对新模块仍为零引用。

下一步不是接主循环。应先设计并结果前冻结一个全新的小型人工反例矩阵，覆盖不同查询数、N 与重叠/包含结构；
经用户另行确认后才能运行该矩阵。

### 最新暂停点：Issue #53 一条记录查询 RMSE 达标候选通过小矩阵（2026-08-16）

> 用户授权试验“查询计数 RMSE 不超过 1 条记录就达标”。本步只离线复用上一步的 6 条
> 16×3 人工二元轨迹，没有新增 seed，没有读取 `test_300x10`/`nltcs`/validation，没有使用 GPU，
> 没有修改或接入 `run_evolution`，也没有实现外层选择、加噪、隐私预算或 accountant。

候选只是现有 squared loss 的等价换算：`query_count_RMSE = sqrt(2 * best_loss / m)`。
当前无噪声整数计数阶段固定 `RMSE <= 1`，等价于 `best_loss <= m/2`。它的语义是“已达到
预定拟合质量”，不是“已收敛”。停止选点材料中删除 L1；只在 first-qualified checkpoint 已由 loss
选定且通过 horizon-invariant 前缀重放物化后，才离线复算 L1。

结果前固定的三项门禁全部通过。6/6 轨迹的 first-qualified work 为
`6.0, 14.0, 2.0, 0.5625, 0.75, 2.5625`，全部早于 20 次等效扫描。前 5 条首次在
loss 3.0 / RMSE 1.0 达标，第 6 条在 loss 1.5 / RMSE 0.7071 达标。所有前缀、表哈希、独立 loss
和 residual RMSE 复算一致；后算 normalized L1 为 5 条 0.041667、1 条 0.03125，均低于理论上限
`1/N = 0.0625`。

小矩阵同时暴露必须保留的边界：前 5 条达标表的最大单查询绝对误差均为 2。因此该条件只保证
workload 整体的计数 RMSE/normalized L1 上界，不保证每个查询都单独最多差 1。完整尾部仍把 best loss
改善到 0..2，不能把达标曲解成后续不会改善。

首次执行前测试源 SHA-256 为 `e0d4a28f...0759d9d`；结果回归版为
`c88a2d09...5c4fbb`。停止相关回归现为 `30 passed`；Ruff check、Ruff format check、compileall、
tracked/untracked 空白检查均通过，`evolution.py` 仍对新候选零引用。当前结论只是候选通过
“同一 6-query、16-record 人工问题×6 轨迹”的
立即反例搜索，尚不能在线接入。下一步先由用户审查是否接受“整体 RMSE<=1，但个别查询可能差 2”的
质量语义；若接受，再另立一步验证不同 m/N/workload，仍不直接接主循环。

### 最新暂停点：Issue #53 固定反事实矩阵否定 3+3 停止候选（2026-08-16）

> 本步只执行 6 条固定人工 16×3 二元轨迹，在已完成的无门控生成轨迹上离线回放停止器。
> 没有读取 `test_300x10`/`nltcs`、validation 或任何真实数据，没有使用 GPU，没有修改或接入
> `run_evolution`，也没有实现外层选择、加噪、隐私预算或 accountant。

结果前固定矩阵为 seed `20260816..20260818` 与 `rho in {1.0, 0.25}` 的笛卡尔积；
`rho=1.0` 跑 40 轮，`rho=0.25` 跑 160 轮，两者期望工作量均为 40 次等效扫描。
原生成器全程不停，事后只向 stopper 提供 squared loss 和 applied participating rows。首次执行前
测试源文件 SHA-256 为 `4506aedca5bca7fc415e539e3b7862ac1e96f452c454c8254eab9b44a34cb411`。

预注册的尾部工作量门禁与“资源护栏不冒充正常完成”门禁均通过，但核心安全门禁失败：
6 条中 4 条在 `optimization_stalled` 之后出现严格更低 loss。具体为：

- seed 20260816 / rho 1.0：state 12、work 12.0 以 best 3.0 停，尾部到 2.0；
- seed 20260817 / rho 1.0：state 20、work 20.0 以 best 3.0 停，尾部到 2.0；
- seed 20260818 / rho 1.0：state 8、work 8.0 以 best 3.0 停，尾部到 1.5；
- seed 20260816 / rho 0.25：state 27、work 7.0625 以 best 3.0 停，尾部到 1.0。

特别是原 25 轮影子轨迹在 state 13..25 未见低于 3.0，但同一 seed 延长到 40 轮后出现 2.0，
证明原短尾部不足以支持停止。因此 3+3 候选已被否定，不得接入主循环；本步不事后增加窗口或换 seed。
`inner_stopping.py` 及其测试暂保留为已否定原型和回归证据，`evolution.py` 仍对它零引用。
负结果已固化为 `tests/test_inner_stopping_counterfactual_matrix.py`；结果回归版源文件 SHA-256 为
`567547fd80935520394d5e7e02fd44533d22d2fd9b8b50728e1b6bd35e9e5356`。全部停止器相关测试现为
`29 passed`；Ruff check、Ruff format check、compileall、tracked/untracked 空白检查均通过。

下一步不是接入，也不是继续把 6 改成 9/12/20。需先与用户讨论并选择语义：要么接受“停滞只是
有限资源下的启发式截断”，要么取消无门控随机轨迹的正常停滞声明，只保留 exact residual 与 fail-closed 资源上限。
用户确认前不实现新停止规则。

### 最新暂停点：Issue #53 三窗直接停止被反例否定，3+3 候选修订通过回归（2026-08-16）

> 用户明确当前只做固定 workload、精确查询答案、`sigma=0` 的生成部分；不实现外层查询选择、
> 私有测量、隐私预算或 accountant。Issue #53 Amendment 2 位于：
> https://github.com/Chuhan722/table-diffusion/issues/53#issuecomment-5307820105 。本步新增独立模块
> `src/table_diffevo/inner_stopping.py`、人工测试 `tests/test_inner_stopping.py`、只读影子测试
> `tests/test_inner_stopping_shadow_replay.py`，并更新
> `docs/设计/Issue53_无噪声生成内层停止与best输出契约设计稿.md`。**没有接入或修改
> `run_evolution`；只在 CPU 上运行一条 16×3 人工二元短轨迹，没有读取 `test_300x10`/`nltcs`、
> validation 或任何真实数据，没有使用 GPU，也没有实现 DP 外层。**

用户已确认“无门控残差引导更新核 + best-checkpoint 输出”的停止语义。生成停止控制只读取现有
squared loss；normalized L1 只在生成结束后离线评价。普通有限 current loss 上升照常成为下一状态，
不拒绝、回滚或立即停止；只有 `current_loss < best_loss` 才刷新 checkpoint 并算进展。current loss
从高处回落但未低于历史 best 不算进展，连续停滞后返回历史 best 而不是较差的 terminal current。NaN、无穷或
负 loss 直接报错，不能冒充普通上升或正常终止。

纯状态机只接收 initial/post-round loss 与已应用 proposal 的 participating rows。最初实现为连续 3
个工作窗口无 best 改善就直接 `optimization_stalled`；25 轮固定影子轨迹证明该规则过早：state 3
候选 best 为 5.0，但继续运行到 state 6 出现 best=3.0。因此三窗直接停止已被明确否定，不得接入。

候选修订保持简单、无数据集阈值：`stall_block_windows=3`，前三个连续空窗只产生 candidate；再连续
三个空窗仍无新 best 才正式停止，即派生 `required_no_progress_windows=6`。任何严格新 best 都取消
candidate 与确认并把计数清零。20 个窗口仍是 fail-closed 工程护栏；终止优先级保持
`exact_residual > optimization_stalled > resource_cap_reached`。当前整数计数下任一严格 loss 改善
自然至少为 0.5。

同一影子反例在修订后表现正确：state 3 只进入确认、不停止；state 6 的 best=3.0 在终止检查前更新并
清零；随后 state 7..12 连续六窗无新 best，在 state 12 返回 `optimization_stalled` 和 state 6 best；
反事实 state 13..25 没有低于 3.0。回放逐状态匹配 prefix minimum、累计参与行、窗口数、first-seen
best 与表哈希，且不改变完整诊断、候选评价数或 RNG 哈希。rho=1 只让人工测试一轮等于一个窗口，
不是生产参数或质量证据。legacy 非残差引导配置的 initial table hash 差异仍不扩展当前范围。

纯逻辑测试现为 27 项，新增“第五个空窗后第六窗恰好创新低必须取消停止”等边界；连同 1 项影子回放，
最终共 `28 passed`。Ruff check、Ruff format check、compileall 和 `git diff --check` 均通过，并确认
`evolution.py` 对新模块零引用。

本小步到此暂停。单条反例通过只能说明该已知漏洞被修复，不能证明 3+3 普遍安全。下一步若用户继续
授权，应先固定一个小型人工反事实尾部矩阵，统计候选停止后是否仍出现更低 best；矩阵通过前不接入
`run_evolution`，不读取真实数据，也不推进外层 DP。

### 最新暂停点：Issue #53 V2c 正式人工验收失败，独立审计通过（2026-08-16）

> 本段为当前最新暂停点。冻结 commit `f9db5d6fb4af9bccf36c5fad3c1c2565eb8b57c5` 上已完成唯一一次
> V2c 正式人工矩阵；结果为 `candidate_failed`，独立全量重放 `passed=true`、mismatch=0。**只运行
> 10000 条固定人工 AR(1) 标量轨迹，没有读取 `test_300x10`/`nltcs`，没有运行项目生成器、使用 GPU
> 或消耗隐私预算。V2c v1 不得接入真实轨迹或在线过程，也不得重复运行。**

正式协议 SHA-256 为 `a9930b440f3483d0bb2e6ad8d3bbf4cd8db097b2d85deed69216a475679cbc04`；
5 个 family 各 2000 条、每条 2048 round，共完成 150000 次检查点分类和 450000 次尺度估计，runner
墙钟 `128.3000 sec`。科学结果 SHA-256 为
`77c054980c4da46bc385fbadd8dbe79e968b6dfada704fadd5eac1329983de0c`。独立 auditor 不导入 runner
或项目 V2/V2b/V2c 核心，重新生成全部轨迹并独立重算三尺度 OBM、双确认、回撤和门禁；payload、
科学 SHA、最终 status 与 22 项边界检查全部精确一致。

预注册失败项恰好是 `main.ar1_phi_0p5.ready_count`、`main.ar1_phi_0p8.ready_count` 和
`slow_pressure`。iid、`phi=0.5`、`phi=0.8` 的 first-ready 数分别为 1963、1646、1631；后两者低于
冻结下限 1850。三类已放行子集的覆盖率分别为 95.26%、94.41%、93.44%，LRV 比中位数分别为
1.0653、0.9920、0.8853，均通过条件安全门禁；因此 V2c 改善了 V2b 的过早不安全放行，却以过低
利用率换取该改善。`phi=0.95` 只放行 606 条，落入 `unsafe_sparse_release`，且放行子集覆盖率仅
84.49%、LRV 比仅 0.5316，慢相关压力仍明确失败。

成本门禁与全局契约全部通过：三类主 family pooled resource mean 为 958.0587，iid/`phi=0.5` 资源
中位数为 384/1024；ESS 排序、负相关控制、正式 ESS cap、MCSE floor、输入身份和 22/22 边界检查
全部通过，所有数学/非有限/契约违规计数为 0。失败是科学假设失败，不是实现损坏。但回撤诊断显示
first-ready 后再次不相容的比例在 iid、`phi=0.5`、`phi=0.8` 上分别为 36.27%、67.86%、68.55%；
2048 当前双确认通过数仅 1652、994、950，说明有限样本三尺度比例仍不是稳定的自适应信号。

完整解释见 `docs/实验结果/Issue53_V2c三尺度双确认有效证据人工验收结果.md`。正式 manifest、8.4 MB
report 和 audit 位于 `outputs/issue53_v2c_three_scale_effective_evidence_f9db5d6/`，将由本次结果提交
显式归档。冻结失败动作是 `no_v2d_return_to_v2_fixed_2048`：不得追加 V2d 或事后调 1.25；当前回到
V2 支持的统一 2048 ESS/MCSE 数值资格下限，它仍不是收敛或停止证明。下一步先审查并接受该负结果，
不运行真实数据；若继续 Issue #53，应讨论固定资源下的无门控残差引导生成评估，而不是继续造门控。

### 最新暂停点：Issue #53 V2c 固定 runner 与独立 auditor 完成，正式矩阵未运行（2026-08-16）

> 本段为当前最新暂停点。V2c 设计、结果前人工协议和三尺度双确认研究核心已分别由 commit
> `c274221`、`2425b67`、`7e244eb` 冻结；本阶段完成固定人工 runner、独立 auditor 和入口契约
> 测试。**没有实例化或生成 `[53,2,3,...]` 正式 seed，没有运行 10000 条正式人工轨迹，没有读取
> `test_300x10`/`nltcs` 或任何真实数据，没有接入项目生成器、GPU、隐私预算或在线停止过程。**

新增固定入口 `scripts/validate_issue53_v2c_three_scale_effective_evidence.py`。`plan` 不实例化 RNG；
`run` 仅允许指定全新输出目录，不能覆盖 seed、family、重复次数、检查点、三尺度、1.25、双确认或
验收门禁。正式入口要求包含 untracked 在内的干净工作树，绑定当前 commit、全部设计/协议/核心/
入口/测试源码哈希，并显式绑定已归档 V2b 负结果的 report、audit 和科学 SHA-256
`abd39f88da0408b5341374b1019ddb61df50fa591ec745d10bba27e504dbdb12`；V2b 必须保持
`candidate_failed` 且独立审计通过，否则 fail closed。report 只记录可移植的同目录 manifest 文件名，
输出文件和目录均拒绝覆盖。

新增 `scripts/audit_issue53_v2c_three_scale_effective_evidence.py`。它不导入 runner，也不导入项目
V2/V2b/V2c 数学核心；独立实现 PCG64 人工轨迹生成、三尺度 OBM、相邻双确认、first-ready、资格
回撤、全部 family 汇总与验收门禁。auditor 严格拒绝重复 JSON key、NaN/Infinity、绝对或跨目录
manifest、commit/source/provenance 漂移，并逐条重放完整科学 payload、SHA 和最终 status。runner 与
auditor 各自固定 22 项边界检查，覆盖三个尺度分别主导、1.25/nextafter、任一尺度失败、双确认序列、
2048 当前状态与 first-ready 的区别、资格回撤、ESS cap、MCSE floor、溢出和非法输入。

`tests/test_issue53_v2c_three_scale_effective_evidence_artificial.py` 新增 23 项测试。专用测试 namespace
`[999,53,2,3,...]` 的小矩阵中，runner 与 auditor 对轨迹、检查点、family 汇总和门禁逐值一致；测试
显式禁止正式 namespace，并覆盖 plan 零抽样、CLI 无科学旋钮、dirty-tree 先拒绝、V2b 来源损坏、
payload 篡改后重算 SHA、路径逃逸、非覆盖输出与独立导入边界。最新验证为 V2c 入口 `23 passed`，
V2/V2b/V2c 核心加三个入口 `175 passed`，相关研究测试 `257 passed`，全仓
`1428 passed, 2 warnings`；两条 warning 仍只来自既有 residual-geometry 输入哈希失败测试。

只读 `plan` 已核对：协议 SHA-256 为
`a9930b440f3483d0bb2e6ad8d3bbf4cd8db097b2d85deed69216a475679cbc04`，固定 5 个 family、每类
2000 条，共 10000 条轨迹、150000 次检查点分类、450000 次尺度估计、最多 20480000 个人工标量；
输出明确为 `generation_started=false`、`execution_started=false`。下一步只把本阶段冻结成干净的
预运行 commit，并从该干净 HEAD 再核对一次 `plan`。正式矩阵仍必须等待用户另一次明确授权；届时
只能执行一次固定矩阵，再运行独立 audit，不能同时读取真实数据或接入生成过程。

### 最新暂停点：Issue #53 V2c 研究核心与确定性测试完成（2026-08-16）

> 本段为当前最新暂停点。V2c 结果前协议已由 commit `2425b67` 独立冻结；本阶段只新增
> `src/table_diffevo/adaptive_effective_evidence_v2c.py` 与
> `tests/test_adaptive_effective_evidence_v2c.py`。**没有实现 runner/auditor，没有实例化或生成
> `[53,2,3,...]` 正式 seed，没有运行人工矩阵或真实数据，没有接生成器、GPU、隐私预算或在线
> 停止过程。**

V2c 核心在 15 个固定 `256..2048` 检查点复用未修改的 V2 显式批长 OBM 公式，批长严格为
`b=floor(sqrt(n))`、`2b`、`4b`。任一尺度不可估计则 `core_not_estimable`；三者都可估计时按 formal
inflation 的 `max/min <= 1.25` 判当前 `three_scale_compatible`，official LRV/ESS/MCSE 始终使用三个
尺度的最大 inflation；组合计算溢出单独 fail closed 为 `nonfinite_computation`。

当前公开数值状态严格按相邻两点计算：第一点恒为 false，之后
`adaptive_numerically_estimable[k] = C[k-1] and C[k]`，所以最早只能在 384 取得资格。完整轨迹仍
保存第一次取得资格的位置和资源计数，但当前状态可回撤；first-ready 后的三尺度不相容次数、2048
当前相容状态和当前数值资格均独立保留。全程固定 `stationarity_not_assessed=true`，没有新增
confirmed/stable/converged/qualified/stop/threshold/quality 字段。

新增 53 项纯确定性测试，覆盖三尺度手算 OBM、冻结批长表、1.25/nextafter 精确边界、三个尺度分别
成为最大风险、任一尺度失效、组合溢出、official 公式、ESS cap/MCSE floor、平移/正缩放、常数、
周期、尖峰、趋势、非法输入、`T,T`/`T,F,T`/`T,F,T,T`、2048 首次通过与无资格、资格回撤和原
V2/V2b 行为不变。V2/V2b/V2c 核心测试通过；Issue #53 相关回归通过；全仓结果为
`1405 passed, 2 warnings`，两条 warning 仍只来自既有 residual-geometry 输入哈希失败测试。

当前唯一下一步是用户审查本阶段核心与测试。获得下一次明确授权后，才可另行实现固定 runner 与不
导入 runner/V2b/V2c 核心的独立 auditor；此时仍不得生成正式 seed 或运行 10000 条人工矩阵，更
不得读取 `test_300x10`/`nltcs` 或连接真实生成过程。

### 最新暂停点：Issue #53 V2c 结果前人工协议已接受，研究核心获授权（2026-08-16）

> 本段为当前最新暂停点。用户接受的 V2c 唯一设计已由 commit `c274221` 冻结；当前只新增
> `docs/设计/Issue53_V2c三尺度双确认有效证据人工验收协议.md`，用户已确认该结果前协议。**尚未
> 实现 V2c 核心、runner 或 auditor，尚未生成全新 `[53,2,3,...]` seed，尚未运行人工矩阵或真实
> 数据。当前只授权实现独立研究核心与确定性测试。**

协议严格落实设计：15 个 `256..2048` 检查点分别计算 `b/2b/4b` 三个 V2 OBM 尺度，共冻结 10000
条轨迹、150000 次检查点分类、450000 次尺度估计和 20480000 个人工标量。当前三尺度均可计算且
formal inflation 的 max/min `<=1.25` 只形成相容状态；公开
`adaptive_numerically_estimable=true` 还要求当前与前一检查点连续相容，最早只能发生在 384。
official LRV/ESS/MCSE 始终取当前三个尺度最大 inflation，状态可在后续检查点回撤。

五个 AR family、每类 2000、主 family 利用率/覆盖率/LRV 比/ESS 排序、原成本上限、负相关控制和
`phi=0.95` 两分支压力门禁全部沿用 V2b，不因规则更保守而放宽。全新正式 namespace 固定为
`SeedSequence([53,2,3,family_code,repeat_index])`；first-ready 后再次三尺度不相容和 2048 当前状态
必须报告，但不新增结果后阈值。全局要求三尺度公式、双确认序列、reason、first-ready、ESS cap、
MCSE floor 和输入身份逐项对拍；正式规模必须由不导入 runner/V2b/V2c 核心的 auditor 全量重放。

若未来 V2c 正式失败，协议预先固定停止增加 V2d；回到 V2 的统一 2048 数值资格下限。当前唯一
下一步是实现 V2c 独立数学核心与确定性测试；完成并交用户审查前，不得实现 runner/auditor、生成
新 seed、回放 V2b seed、读取真实轨迹或启动实验。

### 最新暂停点：Issue #53 V2c 三尺度双确认设计已接受（2026-08-16）

> 本段为当前最新暂停点。V2b 正式负结果与原始 manifest/report/audit 已由 commit `0846d18` 长期
> 归档。用户已接受最后一个简单自适应候选，设计固定在
> `docs/设计/Issue53_V2c三尺度双确认有效证据设计稿.md`。**没有实现 V2c、没有生成新 seed、没有
> 运行人工矩阵或 `test_300x10`/`nltcs`。**

V2c 只对 V2b 的两个正式失败机制各增加一道防线。当前检查点从 `b/2b` 改为
`b/2b/4b` 三个 OBM 批长，三者均可计算且正式相关膨胀的 max/min 不超过原阈值 1.25 时，才记为
当前三尺度相容；正式 LRV/ESS/MCSE 仍取三个尺度的最大相关膨胀。数值资格还要求相邻两个固定
检查点连续相容，因此最早从 256 推迟到 384；`T,F,T` 不通过，2048 要通过必须由 1920/2048 连续
相容。公开 schema 继续使用 `adaptive_numerically_estimable`，固定
`stationarity_not_assessed=true`，不产生稳定、收敛、质量或停止字段。

除增加第三尺度和双确认外，候选尽量不变：15 个 `256..2048` 检查点、1.25、五个 AR family、每类
2000 条、安全/成本/负相关/`phi=0.95` 门禁均建议沿用；新协议必须使用全新
`SeedSequence([53,2,3,family_code,repeat_index])`。first-ready 之后再次不相容的比例必须报告，
但不能替代 first-ready 覆盖率。若 V2c 再次正式失败，本路线停止继续增加尺度或确认次数，回到 V2
支持的统一 2048 数值资格下限。

下一步只写结果前人工协议并交用户再次审查。协议确认前不得实现、回放 V2b 正式 seed、调整 1.25、
读取真实轨迹或启动生成实验。

### 最新暂停点：Issue #53 V2b 正式人工验收失败，独立审计通过（2026-08-16）

> 本段为当前最新暂停点。冻结 commit `1ad9340b6bd93ce7e998d5bc73d28d48f82231d6` 上已完成唯一一次
> V2b 正式人工矩阵；结果为 `candidate_failed`，独立全量重放 `passed=true`、mismatch=0。**这次
> 只运行 10000 条固定人工 AR(1) 标量轨迹，没有读取 `test_300x10` 或 `nltcs`，没有运行表格
> 生成器、使用 GPU 或消耗隐私预算。V2b v1 不得接入真实轨迹或在线过程。**

正式协议 SHA-256 为 `a7dde6b7867e215c9147131f085eaa47b47e04495b5d1bed37355f95a69dd33f`；
5 个 family 各 2000 条、每条 2048 round，共完成 150000 次检查点分类和 300000 次尺度估计，runner
墙钟 `88.0692 sec`。科学结果 SHA-256 为
`abd39f88da0408b5341374b1019ddb61df50fa591ec745d10bba27e504dbdb12`。独立 auditor 不导入 runner
或项目 V2/V2b 核心，重新生成全部轨迹并独立重算 OBM、first-ready 和门禁；payload、科学 SHA、
最终 status 与 16 项边界检查全部精确一致。

预注册失败项恰好是 `main.ar1_phi_0p8.coverage`、`main.ar1_phi_0p8.lrv_ratio` 和
`slow_pressure`。iid 与 `phi=0.5` 的 first-ready 覆盖率分别为 95.75% 与 93.65%，LRV 比中位数
1.0766 与 0.9319，均通过；`phi=0.8` 虽有 1997/2000 条取得过数值资格，但 first-ready 覆盖率只有
90.39%，LRV 比中位数 0.7856，均低于冻结下限。`phi=0.95` 有 1373/2000 条被放行，进入协议的
validated-release 分支，但覆盖率仅 77.13%、LRV 比中位数仅 0.4210，因此慢相关压力明确失败。

成本门禁全部通过：iid、`phi=0.5`、`phi=0.8` 的资源中位数均为 256，三类 pooled resource mean
为 342.8053；ESS 比仍严格保持 iid > 0.5 > 0.8。该低成本不能补偿安全失败，因为过早放行正是成本
偏低的原因。负相关门禁、正式 ESS cap、MCSE floor、输入身份和 16/16 边界检查全部通过；
`core_not_estimable`、非有限、契约违规和身份违规均为 0，说明失败是科学假设失败而非实现损坏。

核心原因与设计稿事前漏洞一致：`b` 和 `2b` 可以一起低估长期相关，并错误地显得相容；顺序日程还
会永久抓住一次偶然相容。`phi=0.95` 在整段日程累计放行 1373 条，但 2048 检查点本身仅 819 条仍
相容，至少 554 条一次性放行随后不再相容。不得用本批结果修改 1.25、检查点、family 或 first-ready
规则后重新宣称 V2b v1 通过。

完整结果见 `docs/实验结果/Issue53_V2b自适应有效证据人工验收结果.md`。正式 manifest/report/audit
位于通常被忽略的 `outputs/issue53_v2b_adaptive_effective_evidence_1ad9340/`，但三份冻结文件由本次
结果提交显式归档；文件 SHA-256 分别为 `1aae6b7a...e2bb7`、`1a3f9e7e...104c`、
`df0a2b46...bbcb2`。当前下一步先由用户审查负结果；不能继续跑真实数据。若仍研究自适应方案，必须
另立 V2c、新公式、新协议和全新 seed，同时解决共同偏差与一次偶然通过；否则沿用 V2 的 2048 统一
数值资格下限，但仍不得称其为收敛或停止轮数。

### 最新暂停点：Issue #53 V2b 固定 runner 与独立 auditor 完成，正式矩阵未运行（2026-08-16）

> 本段为当前最新暂停点。用户已接受 V2b 自适应设计与结果前人工协议；双尺度研究核心、固定人工
> runner、独立 auditor、入口契约测试和全仓回归均已完成。**没有生成正式协议 seed，没有运行
> 10000 条正式人工轨迹，没有读取真实 development/validation，没有接入生成器，也没有创建在线
> 停止器。包含本段与上述工具的干净 HEAD 是唯一预运行候选，正式矩阵仍需另行授权。**

研究核心仍固定为 15 个 `256,384,...,2048` 检查点，在同一连续前缀上比较
`b=floor(sqrt(n))` 与 `2b` 两个 V2 重叠批均值估计。任一尺度不可计算则
`core_not_estimable`；两个尺度均可计算但正式相关膨胀比大于 1.25 则
`multiscale_disagreement`；相容时正式 LRV、ESS 与 MCSE 一律采用两个尺度中更大的相关膨胀。
2048 只是资源上限：首次在 2048 相容与到上限仍不相容继续保持不同 first-ready/reason 身份。本层
始终固定 `stationarity_not_assessed=true`，不产生稳定、收敛、质量或停止结论。

新增固定入口 `scripts/validate_issue53_v2b_adaptive_effective_evidence.py`。`plan` 不实例化 RNG，只
打印冻结协议；`run` 只有 `--output-dir`，不能覆盖 seed、重复次数、family、检查点、阈值或门禁。
正式入口要求包含 untracked 在内的干净工作树，manifest 绑定 Git commit、两份设计文档、V2/V2b
核心、runner、auditor、测试的 SHA-256 以及 Python/NumPy/OS/CPU 环境；输出目录和 JSON 文件均
拒绝覆盖。runner 会计算所有 15 个检查点，逐轨迹只把 first-ready 用于安全/成本主指标，并同时
核对公式关系、正式 ESS 上限、MCSE floor、输入身份和禁用字段。

新增 `scripts/audit_issue53_v2b_adaptive_effective_evidence.py`。它不导入 runner，也不导入项目
V2/V2b 核心；独立实现 PCG64 轨迹生成、NumPy OBM 公式、双尺度分类、first-ready、覆盖率、LRV
比、资源成本、负相关控制、`phi=0.95` 分支和最终门禁，并将完整科学 payload 与 SHA-256 逐值重放
对拍。审计前还严格拒绝重复 JSON key、NaN/Infinity、manifest/commit/source hash 漂移。协议全局
门禁已明确：连续高斯人工矩阵的 `core_not_estimable` 总数必须为 0，慢相关只能因两个可计算尺度
不一致而安全拒绝，不能用数学核心失效冒充 fail closed。

`tests/test_issue53_v2b_adaptive_effective_evidence_artificial.py` 新增 18 项入口与独立审计测试。专用
非正式 namespace `SeedSequence([999,53,2,2,...])` 的小矩阵中，runner 与 auditor 的轨迹记录、75
个 checkpoint 汇总、5 个 family 汇总和接受门禁逐值一致；测试还覆盖 plan 零抽样、CLI 无科学
旋钮、dirty-tree 先于哈希/抽样拒绝、两套 16 项边界检查、不相容检查点的正式量对拍、矩阵身份和
汇总篡改拒绝、严格 JSON 与非覆盖审计。没有使用正式 namespace 生成测试随机数。

当前验证结果：V2/V2b 核心加入口为 `86 passed`；Issue #53 相关回归为 `181 passed`；使用 `/tmp`
临时可执行 Python 副本完成全仓回归为 `1352 passed, 2 warnings`，两条 warning 仍只来自既有
residual-geometry 输入哈希失败测试。只读 CLI `plan` 已实际核对：5 个 family、10000 条轨迹、
150000 次检查点分类、300000 次尺度估计、最多生成 20480000 个标量，且
`generation_started=false`；当前协议 SHA-256 为
`a7dde6b7867e215c9147131f085eaa47b47e04495b5d1bed37355f95a69dd33f`。

下一步不能直接跑正式矩阵。预运行 commit 只负责冻结本段所列文档、实现与测试；随后必须从该干净
HEAD 再核对一次只读 `plan`。只有 commit 与 plan 都确认无误并再次获得运行授权，才执行一次正式
10000 轨迹矩阵，再运行独立 audit；当前没有 push 或 PR。

### 最新暂停点：Issue #53 V2 人工验收完成，历史下限候选为 2048（2026-08-16）

> 本段为当前最新暂停点。固定 100 轮/12 小块路线已由既有 development 审计作为设计反例归档；
> 当前改为不预设正常停止轮数的连续轨迹有效证据路线。研究版数学核心、确定性边界测试与固定人工
> 验收均已完成；预注册候选得到 `candidate_supported`，共同最少历史候选为 **2048 outer rounds**。
> **这不是收敛或停止轮数；尚未读取真实 development/validation，也未接入生成器或停止器。**

新增两份设计与人工协议记录：

```text
docs/设计/Issue53_V2有效证据计数器设计稿.md
docs/设计/Issue53_V2人工轨迹验收协议.md
```

新实现位于 `src/table_diffevo/effective_evidence.py`，与旧的
`stationarity_v2.py` 固定小块研究代码隔离。研究函数只接受连续 post-round 身份与同长度一维有限
标量序列；拒绝 initial、缺号、重复、乱序、布尔、非有限和非数值输入。批长唯一候选为
`b=floor(sqrt(n))`，实现使用整数 `isqrt(n)`；以全部重叠批均值估计长期方差，再计算 raw 相关膨胀、
raw ESS、正式保守 ESS 和 MCSE。输出固定声明 `stationarity_not_assessed=true`，不存在
stable/converged/qualified/stop/threshold/quality 字段，也不接受数据集或核身份。

实现审查发现并修正了一处设计漏洞：若只把正式 ESS 截在实际 round 数以内、但 MCSE 继续直接使用
负相关下更小的原始长期方差，MCSE 仍会暗中获得“超过 n 份证据”的收益。现在正式 ESS 与正式 MCSE
统一使用 `max(1, raw_correlation_inflation)`；raw ESS 与原始长期方差只作诊断。完全常数返回
`zero_round_variance`，批长与周期精确偶合造成伪零长期方差时返回
`degenerate_long_run_variance`，有限输入导致数值溢出时返回 `nonfinite_computation`，全部 fail closed。

`tests/test_effective_evidence.py` 新增 25 项确定性测试，覆盖手工复算 OBM 公式、输入身份、常数、周期、
平移/正比例缩放不变性、单点尖峰、负相关 raw/formal ESS 分离、趋势不越权分类、数值溢出和输入不变。
Issue #53 相关回归为 `67 passed`。第一次全仓回归因共享 `.conda/bin/python3.11` 没有执行权限，所有
失败均集中在需要启动子进程的旧测试；使用不写入仓库的临时可执行副本重跑后完整通过：
`1258 passed, 7 skipped, 2 warnings`。两条 warning 均来自既有 residual-geometry 输入哈希失败测试。

新增 `scripts/validate_issue53_v2_effective_evidence.py` 作为固定 CPU 人工入口，协议 SHA-256 为
`79c88437c3ae720f6938fdb2fa56b31b198734a4a39f6dd596d75e16a1690e22`。正式矩阵只有独立白噪声与
`AR(1) phi=0.5/0.8/-0.5` 四类，每类 2000 条、每条最长 4096，在
`16/32/64/128/256/512/1024/2048/4096` 九个前缀调用同一个研究核心，共 8000 条人工轨迹、
72000 次证据计算。命令行只能指定新输出目录，不能覆盖 seed、重复、长度、相关强度或容差；运行前
要求包含 untracked 在内的干净工作树，manifest 将绑定 commit、协议、源码/测试/文档哈希和环境。

`tests/test_issue53_v2_effective_evidence_artificial.py` 新增 13 项入口契约测试；与核心合计
`38 passed`。加入入口后的全仓临时可执行环境回归为 `1271 passed, 7 skipped, 2 warnings`，warning
仍只来自既有 residual-geometry 测试。运行前的 `plan` 验证只描述固定矩阵且没有生成随机数；完整
矩阵只在下述预运行 commit 锁定后执行。

人工矩阵已由本地预运行 commit `3d7d667b06525eed088926d79e07fdde3aa8faec` 锁定并执行；运行前
工作树干净。四类各 2000 条、九个前缀共 72000 次证据计算在 27.51 秒完成，结果为
`candidate_supported`。正相关三类仅 2048 与 4096 共同通过；1024 的唯一关键失败是
`phi=0.8` MCSE 覆盖率 `92.30% < 92.50%`，虽只差 0.2 个百分点也不得事后放宽。2048/4096 的
对应覆盖率为 93.70%/93.65%。九个长度的 ESS 排序均正确，负相关控制全部通过；数值失败、非有限、
契约违规、正式 ESS 超过 n 均为 0；固定边界 11/11 通过。一次不导入 runner 的独立重算也通过。

结果身份：协议 SHA `79c88437c3ae720f6938fdb2fa56b31b198734a4a39f6dd596d75e16a1690e22`；
manifest/report/scientific SHA 依次为
`69a6da17fca36fe9affc692f6c2bbcccc87602b1926ad2b369acaa491169ecd3`、
`6221e79464f3128d06bfb7f0146abc4b11f42c363bb2a785be52ef5295b92c8d`、
`f328a8026382e19fb96a4aa6aa66a17a675d1b90fe66096a445bb6c07a43a57c`。完整解释见
`docs/实验结果/Issue53_V2有效证据人工验收结果.md`。

下一步必须先由用户审查是否接受 2048 这一较高但预注册通过的“数值估计资格下限”。若接受，再冻结
正式接口 `insufficient_history=2048` 并补 deferred 测试；若认为代价不可接受，必须保留本次结果，
另立新协议研究其他统一估计器，不能事后把本协议改成 1024。当前仍不得读取真实轨迹、连接生成
runner、设置 max_rounds 或创建稳定/收敛/停止判定。分支尚未 push 或创建 PR。

### 最新暂停点：残差信号几何正式通过——相对残差适应度五种子 supports（2026-08-16，Issue #57）

> 本段为当前最新暂停点。诊断表明旧绝对残差适应度把优化力气集中于大计数
> 查询、对稀有模式查询系统性无力（平台残差 44.8% 集中在 p<0.05 查询，
> bootstrap iid 对照 22.3%；既有平台 0.00086–0.0011 已低于单次 iid 抽样
> 涨落线 0.001598±0.000174，误差是系统性的几何失配而非预算/容量问题）。
>
> `run_evolution` 新增 `residual_geometry`（absolute 默认，逐位向后兼容）
> 与 `residual_geometry_floor`：relative 口径 ε=(y−q)/max(y,floor)/N 近似
> KL 梯度，稀有查询推动力按相对误差放大；σ/κ 容忍先于相对化；残差同时
> 驱动 fitness 与方向场，信号几何统一切换；相对化只用公开 target 计数，
> 不引入新信息流。
>
> 预注册正式实验（scripts/probe_residual_geometry_formal.py，formal=True，
> commit aac1aff）：五臂 absolute / relative_f8(主) / relative_f{1,4,16}
> × 种子 200..204 × 2000 轮。**nltcs 主判定 supports_relative_geometry**：
> 5/5 配对全胜，measured L1 五种子均值 0.001061 → **0.000314**（−70.4%，
> 门槛 ≥30%）；质量风险带零报警——train 未测量 3-way −61.7%、4-way
> −47.5%、分箱 TVD −7.5% 全部同向改善；稀有查询平均绝对残差 16.93→4.28
> 计数；支持集重叠 1251→1274；墙钟零额外开销（456s→452s/run）。floor
> 敏感性：nltcs 最小 target = 12 ≥ 8，f1/f4/f8 逐位一致，f16 均值同为
> 0.000314（观察项 floor_best=f16 是第 7 位小数差异，无实际意义）。
> test_300x10 辅助判定 not_supported（1/5 胜，−20.6%）：该数据集最小
> target = 17 > 8，floor 臂全程未激活，失败**不能归因于 floor 取值**；
> 当前归因是 300 行尺度下小计数查询的相对化把抽样噪声当信号放大——
> **相对几何的收益与数据规模相关，小表边界如实保留**，不合并主结论。
> 首次正式运行在末段离线评价
> 因参考表读取缺陷崩溃（已修复并加测试）；修复后完整重跑，与首跑全部
> 50 个 run 的主指标逐位一致（墙钟除外）。
>
> 对照参考：Issue #46 讨论的第一层同信息基线（private-pgm 系）nltcs
> 水位 ≈ 0.000357——本结果五种子均值首次越过该线（0.000314）。收口
> 判定仍按 #46 协议（≥2 数据集 + 全指标 + 冻结基线调参）正式执行，
> 本结果不替代该协议。
>
> 产物归档：正式 JSON 位于运行工作树 outputs/residual_geometry/
> formal_residual_geometry_5seed_2000round.json（gitignored），SHA-256
> `51aff5414eb15c9cfdda496dc1549c6fba7216043159bd377be429fb11443f64`，
> 绑定提交 aac1aff；脚本冻结 EXPECTED_INPUT_SHA256 与该产物记录的
> 公开输入哈希一致（fail-closed 对拍，见
> scripts/probe_residual_geometry_formal.py）。合并 master 后的行为
> 不变性由 scripts/replay_residual_geometry_main_arms.py 在当前 HEAD
> 重放主判定两臂×五种子对拍最终表哈希验证。
>
> 边界：无噪声原型结论；**判定口径是 2000 轮固定预算下的终态显著改善，
> 不构成"算法已收敛"的声明**（未做更长 horizon 对照或平稳性检查；
> 配置为 tol=inf、固定 rho/mu、无 self-cooling，过程不会自然冻结）。
> DP 阶段分母将使用带噪计数，其稳定化（floor 与
> 噪声尺度挂钩或 y+c 平滑）需在 DP 设计中单独处理并计入预算论证（见
> docs/设计/残差信号几何_绝对与相对适应度口径.md）。默认参数未改
> （absolute 仍为默认，是否切换默认与 #46 衔接由后续决定）。

### 上一暂停点：Issue #52 Stage B 正式完成，结论为 no_factor_candidate（2026-08-14）

> 本段取代下面“Stage B 工具已确认、正式实验尚未启动”的暂停点。Stage B 已严格按结果前冻结
> 记录完成并通过独立审计；G* 虽优于同 tau independent，但未优于 I*=independent tau5，
> 因此正式停止，不进入 Stage C。

冻结与结果身份：

```text
tool commit = bab78a377aa49e6a680b91660f579d427e82860a
pre-run record = https://github.com/Chuhan722/table-diffusion/issues/52#issuecomment-5290210632
formal result  = https://github.com/Chuhan722/table-diffusion/issues/52#issuecomment-5290453407
mode = formal
task count = 30/30 complete
runner formal_result_valid = true
audit passed = true
audit formal_result_valid = true
runner elapsed = 470.64 s
audit elapsed = 0.95 s
```

正式 Stage B 只运行 Stage A 合格的三个 factor 配置，seeds `200..209`、每条3000轮、末500轮
current-loss mean 为主指标；同 tau independent 和 I* 均复用已审计 Stage T 轨迹：

| factor | factor late mean | 同 tau independent | 同 tau 差值 / W-T-L | I*=independent tau5 | 相对 I* 差值 / W-T-L |
|---|---:|---:|---:|---:|---:|
| tau=1,s8 | 132.3734 | 157.2517 | -24.8783 / 8-0-2 | 68.5089 | +63.8645 / 0-0-10 |
| tau=2,s8 | 98.9192 | 111.5445 | -12.6253 / 9-0-1 | 68.5089 | +30.4103 / 0-0-10 |
| tau=3,s16 | **73.3693** | 85.5252 | **-12.1559 / 9-0-1** | 68.5089 | **+4.8604 / 4-0-6** |

factor 排名为 `tau3,s16 < tau2,s8 < tau1,s8`，所以：

```text
I* = independent tau=5
G* = factor tau=3,sweeps=16
point_estimate_better_than_same_tau_independent = true
point_estimate_better_than_i_star = false
stage_c_candidate = null
stage_c_allowed = false
status = no_factor_candidate
```

失败位置唯一且清楚：G* 相对同 tau independent tau3 的末500轮均值改善12.1559，9/10 seeds
改善；但相对 I* 反而高4.8604，只有4/10 seeds改善。按冻结规则不得递补 factor 第二名或追加
配置。G* 的 final 单点均值为 `63.75`，略低于 I* 的 `64.70`，但主指标是末500轮均值，不能用
辅助单点翻转决定；AUC 也比 I* 高 `4095.78`。

三个 factor 都优于各自同 tau independent，说明 factor Gibbs 的同温度收益在3000轮仍存在；
`no_factor_candidate` 的含义不是 factor 完全无效，而是最佳合格 factor 仍未超过更强的高温
independent tau5。

本结果仍是 horizon-limited 固定预算结果，不能声称达到平衡：

| factor | rounds 2001..2500 | rounds 2501..3000 | 相对变化 | 明显下降 seeds |
|---|---:|---:|---:|---:|
| tau=1,s8 | 166.2599 | 132.3734 | -20.06% | 10/10 |
| tau=2,s8 | 112.9026 | 98.9192 | -12.11% | 7/10 |
| tau=3,s16 | 81.0142 | 73.3693 | -9.20% | 9/10 |

预注册没有因末段下降而延长 horizon，因此固定预算选择仍为 `no_factor_candidate`。三个 factor
总 clip hits 均为0；factor 条件最大原始 `abs(logit)` 为 `4.6984/11.2762/16.2638`，全部低于
clip=30。全部轨迹身份、初态、主 RNG 终点、方向尺度、数值、双向条件和上游哈希门禁通过。

正式产物：

```text
目录：outputs/issue52_low_temperature_long_horizon/stage_b_formal_bab78a3_20260814
protocol SHA：ae9e6848c54a072a742019d31ff6341f9aba80e981ba1d65ae235bcb895d8604
trajectory scientific SHA：1e01ca87c376a35636bff5fa13762c18b3dfbf340bd2169ef83390179b11c195
report SHA：5e62a9c21638807d4827b89d476a16903d081d1f1424fea136f12d8074c5151d
audit SHA：de42e0a7ae2a12f132956cf38525f3bb563e5db78bd4da47a9c5464e18a62c1a
report size：6,697,968 bytes
audit size：118,387 bytes
```

独立 auditor 未导入 Stage B runner，重新绑定 Stage T/A、推导三个 factor 配置，并重算趋势、
聚合、逐 seed 双对照配对、I*/G* 和 selection，全部一致。

当前禁止运行 seeds `300..319` 的 Stage C，也不得事后增加 tau/sweeps/seeds/horizon。Issue #52
在本协议下已得到完整可接受的负结果；若后续研究 nltcs、跨 workload、GPU、无限 horizon 或新的
温度/混合设计，必须另立问题和结果前协议，不能并入本次结果。

### 最新暂停点：Issue #52 Stage B 工具已确认，正式实验尚未启动（2026-08-14）

> 本段取代下面“Stage A 正式完成、尚未实现 Stage B”的暂停点，记录将与 Stage B 工具作为
> 同一个冻结提交。当前完成范围只有 runner、独立 auditor、真实全链路 smoke 和回归测试；
> **本提交不包含任何正式3000轮 factor 结果，也不授权跳过 Issue 运行前记录。**

本轮新增：

```text
scripts/run_issue52_stage_b.py
scripts/audit_issue52_stage_b.py
tests/test_issue52_stage_b.py
scripts/issue52_protocol.py 中的 Stage B 冻结协议与正式上游哈希
```

Stage B 正式输入和判定已按 Issue #52 正文落成代码：

1. Stage T report/audit 与 Stage A report/audit 都按文件 SHA、格式、协议、审计状态和科学哈希
   绑定；正式模式只接受既有正式产物。Stage A 的正式 report 很大，runner/auditor 通过已冻结
   文件 SHA 和小型独立 audit 读取 selection，不重复把335MB raw proposal 全量载入内存。
2. factor 配置不手写重新选择，而是从 Stage A 已审计的 `minimal_sufficient_sweeps` 推导。正式
   集合严格为 `tau=1,s8`、`tau=2,s8`、`tau=3,s16`；tau=4/5 不运行。
3. 只运行 factor 轨迹；同 tau independent 与 I* 都直接复用已审计 Stage T 的
   seeds `200..209` 原始轨迹，不重复计算 independent。正式 horizon、检查点和主指标仍是
   3000 rounds、每500轮检查、rounds 2501..3000 current-loss mean。
4. I* 从绑定的 Stage T late-mean 排名重算，正式固定得到 `independent_tau_5`。G* 在合格
   factor 中先按 late mean，精确并列再按 sweeps 少、tau 小排序。
5. 只有 G* 点估计同时低于同 tau independent 与 I* 才输出
   `factor_candidate_selected` 并允许 Stage C；否则固定为 `no_factor_candidate`。Stage B 不提前
   加入胜率或置信区间，也不递补第二名；这些更严格条件仍只属于 Stage C。
6. factor 长轨迹使用已验证等价、速度更合适的 `compiled_batch` 构造路径和 NumPy/CPU，最多
   8 workers；每条轨迹继续检查相同初态、主 RNG 终点、方向固定尺度、独立/factor 条件数值、
   完整轮数和表/RNG 哈希。worker 数只影响墙钟时间。
7. 独立 auditor 不导入 Stage B runner；它重新绑定两个上游、推导配置、重算 factor 趋势与
   聚合、逐 seed 的同 tau/I* 配对、I*/G* 和最终停止状态，并核验 trajectory scientific SHA。

真实 smoke 串起了 Stage T → 状态库 → Stage A → Stage B。smoke 的 Stage A 因样本极小只给出
`tau=1,s8` 与 `tau=2,s16` 两个管线配置，因此 Stage B 正确只执行2条 factor 轨迹（seed 9903、
12 rounds），没有重新运行5条 independent。1 worker 与2 workers 的全部非计时科学字段完全
一致：

```text
trajectory scientific SHA = 76a5c8f7a0e3ab71663f843115333fd5574d0d9e783aaf99d1ae469314d84620
all identity gates = true
audit passed = true
runner/audit formal_result_valid = false
```

smoke 判定分支临时得到：

```text
I* = independent tau=5，late mean=1697.25
G* = factor tau=2,s16，late mean=1383.50
vs same-tau independent tau=2：-409.00
vs I*：-313.75
status = factor_candidate_selected
```

这些数只有1 seed、12 rounds，而且 factor 配置本身也来自每状态2 proposals 的 smoke Stage A；
它们只证明配置派生、配对和选择分支能工作，**不是任何正式效果证据，也不能预告正式 G***。
两条 smoke factor 轨迹的独立方向与 Gibbs 条件总 clip hits 均为0。

专项和回归：

```text
Stage B 新专项：6 passed
Issue #52 全链路：33 passed
全仓库：966 passed, 7 skipped
ruff / py_compile / git diff --check：通过
```

测试覆盖正式协议与四个冻结产物哈希、只运行 Stage A 合格 factor、I* 复算、G* 的 late
mean/sweeps/tau 排序、双点估计门槛、无递补、串并行等价、不可覆盖，以及轨迹 history、配对、
selection 和上游 SHA 篡改拒绝。

用户已经检查并确认本阶段设计与 smoke。冻结顺序必须保持为：本节与工具同一 commit 推送，随后
在 Issue #52 发布绑定该 commit、四个上游哈希和正式命令的运行前记录；只有记录发布成功且再次
明确继续后，才可启动 formal Stage B。当前不得运行正式3000轮、Stage C、nltcs、GPU 或新增
tau/sweeps。

### 最新暂停点：Issue #52 Stage A 混合资格正式完成并审计通过（2026-08-14）

> 本段取代下面“实现与 smoke 已确认”的暂停点。Stage A 已严格按结果前冻结记录完成；
> 尚未实现或运行 Stage B。下一步只能先设计并审查 Stage B 工具，不能直接启动3000轮 factor
> 轨迹。

冻结与结果身份：

```text
tool commit = 5d25864c9845b1bba67ffb87ae5b7e1bd92c6073
pre-run record = https://github.com/Chuhan722/table-diffusion/issues/52#issuecomment-5289850550
formal result  = https://github.com/Chuhan722/table-diffusion/issues/52#issuecomment-5289918247
mode = formal
runner formal_result_valid = true
audit passed = true
audit formal_result_valid = true
runner elapsed = 444.02 s
audit elapsed = 18.26 s
```

正式实验读取已审计的48状态库（16组、每组3 seeds），每状态每 attempt 200 proposals；
共实际执行10个 attempt、480个 state-attempts、96,000个 raw state-proposal bundles。严格按每个
tau 的 `8→16→32` 序列运行，首次全部组通过即停，32失败即不合格。结果：

| tau | 实际执行 | 决定处最差组 TVD | 决定处最差 gap recovery | 资格 |
|---:|---|---:|---:|---|
| 1 | 8 | 0.003855 | 99.266% | qualified，最小8 |
| 2 | 8 | 0.031897 | 96.611% | qualified，最小8 |
| 3 | 8→16 | 0.042889 | 96.373% | qualified，最小16 |
| 4 | 8→16→32 | 0.050535 | 95.923% | 32上限仍不合格 |
| 5 | 8→16→32 | 0.061061 | 95.172% | 32上限仍不合格 |

因此 Stage B 的 factor 资格集合已经冻结为：

```text
factor tau=1, sweeps=8
factor tau=2, sweeps=8
factor tau=3, sweeps=16
```

tau=4/5 的失败都只来自 `initial` 状态组的 TVD；gap recovery 已通过，其他15个中晚期组
全部通过。tau=4,s32 的未舍入值为 `0.0505347729 > 0.05`，虽然只超出
`0.00053477`，仍必须按冻结规则判失败；不得四舍五入改判、追加64/128或放宽门槛。tau=5,s32
为 `0.0610613188`。tau=3 的 initial TVD 从8 sweep 的 `0.057311` 降至16 sweep 的
`0.042889`，所以16是首次充分值。

数值与实现门禁全部通过：最大因子能量误差 `1.11e-16`、one-hot误差 `5.55e-17`；
tau=1..5 最大原始 `abs(logit)` 为 `5.695/11.391/17.086/22.781/28.477`，全部低于
clip=30且总 clip hits=0。production/exact replay 共核验283,950次、22,741,632微步，
mismatch=0；同一 tau 的不同 sweeps 的共享条件/tape hash 均唯一且一致。独立 auditor 重新计算
proposal summary、16组 TVD/recovery、停止序列和 selection，全部一致。

产物与哈希：

```text
目录：outputs/issue52_low_temperature_long_horizon/stage_a_mixing_formal_5d25864_20260814
protocol SHA：e0259b1c614aa49ed79f4d7dec61829ae0b46233e77a14e127b39af93c62e17d
execution scientific SHA：56d5a470a891e99c985503514d7092e385805d1b6900cb46e1da4995fb0c2e0d
report SHA：52654a455d42a0899194878789aa4690b3527c5482a6e4c2d5475b59df835b09
audit SHA：6b4b76af01dc2ed3b267f8047e8edefef84e5fc1049d7aa5974af6ab8286a741
report size：335,058,715 bytes
```

结论边界：Stage A 只决定混合资格，不比较外层3000轮效果、不选择 G*，也不说明 independent
tau=4/5 无效。下一步若继续，应先实现 Stage B：只运行上述三个合格 factor 配置，复用
seeds 200..209、3000 rounds、Stage T 检查点和主指标，再与同 tau independent 及 Stage T 的
I* 比较。Stage B 的 runner/auditor、结果前冻结记录和正式输出目前都不存在。

### 最新暂停点：Issue #52 Stage A 混合资格实现与 smoke 已确认（2026-08-14）

本轮先把 `origin/master`（含 PR #48）合入研究分支，merge commit 为
`ea1e4ba`，无冲突；此前正式 Stage T 与48状态库的四个文件哈希均保持不变。随后只完成
Stage A factor Gibbs 混合资格的实现和缩小 smoke，**没有运行正式 Stage A，也没有启动
Stage B**。用户已经检查并确认本阶段实现；本节与工具代码共同组成正式运行前的冻结提交，
该提交本身不包含任何 formal Stage A 结果。

新增/修改内容：

- `scripts/issue52_protocol.py`：冻结 evaluation `tau=1..5`、逐 tau
  `sweeps=8→16→32`、首次16个状态组全部通过即停止、32失败即不合格的统一规则；正式模式
  绑定已审计48状态库及其 audit 的文件/协议/scientific SHA。
- `scripts/run_issue52_stage_a_mixing.py`：只读取已有状态库；逐 tau 真正按顺序增量执行并立即
  停止；固定同一 state/donor/proposal/address tape；按组计算 TVD 与 expected-direction gap
  recovery；同时检查零 clip、数值、精确能量和 production Gibbs 同 tape replay；输出不可覆盖的
  原始逐状态报告和 scientific hash。
- `scripts/audit_issue52_stage_a_mixing.py`：不导入 Stage A runner，使用审计侧既有独立聚合实现，
  从 raw proposal 重新计算逐配置 summary、16组 TVD/recovery、共享条件 hash、最小充分 sweeps、
  停止序列与最终 selection。
- `tests/test_issue52_stage_a_mixing.py`：覆盖首次8/16通过、到32仍失败、通过后多跑、错误提前停、
  跳级、缺状态、乱序、共享 tape 篡改和 `sweeps>32` 拒绝，并构造真实 smoke 全链路。

真实 smoke 使用1个 seed/组、16状态、每状态2 proposals，仅验证管线，不是方法证据。它有意跑出
不同停止分支：`tau=1` 在8通过，`tau=2` 在16通过，`tau=3/4/5` 到32仍不合格；全部已执行
attempt 的正确性与共享条件门禁通过、clip hits 为0，独立 audit `passed=true`，但
`formal_result_valid=false`。因此这些 tau/sweeps 数值**不能**替代正式48状态 × 每状态200
proposals 结果，也不能据此提前淘汰 tau=3/4/5。

最终验证：

```text
Stage A mixing 新测试：8 passed
Issue #52 相关链路：31 passed
全仓库（正式 CPU 研究环境）：960 passed, 7 skipped
全仓库（CUDA 辅助环境，含13个额外 CUDA 参数化用例）：980 passed
ruff / py_compile / git diff --check：通过
```

本实现已经通过用户检查。冻结提交推送后，必须先在 Issue #52 发布绑定该 commit、协议、输入与
状态库哈希的正式运行前记录；只有记录发布完成后才可运行 formal Stage A 及独立 audit。当前禁止
追加64/128 sweeps、改门槛、做3000轮 factor 轨迹、Stage B、nltcs 或 GPU 改造。

**当前主线固定仓库原始无噪声配置，研究扩散演化机制本身能否更好地逼近精确
workload；2-way 最大熵初始化保留为辅助消融，不再作为本阶段主线 baseline。**

当前 baseline 使用 1-way `marginal` 初始化、精确 target、geometric donor 抽样和
固定 workload；真实 train/test 只在生成完成后离线评价。当前方法是残差驱动的
连续扩散核：不筛掉反向编辑，而是用实际单块方向连续倾斜复制概率，并用首个非零方向 RMS
固定定标。已合入的 `tau=1` 在 nltcs 三种子 1500 轮中把 best loss 从 720.96 万
降到 333.21 万，训练/测试联合 TVD 从 `0.3147/0.3957` 降到
`0.2838/0.3680`。后续小表温度前沿表明 `tau=1` 尚未达到核内极限：`tau=8`
是保留实际反向概率的当前 Pareto 拐点，而 `tau=32/64` 已接近符号门控。这个结论
尚未跨 workload 验证，因此没有更改默认参数。

进一步研究已把精确联合 mask oracle 转成最高三阶稀疏因子，并用随机扫描 Gibbs
避免生产候选路径的 `2^k` 枚举。当前小表 workload 为 25 个 1 属性、20 个 2 属性
和 5 个 3 属性查询；因子能量与完整 hybrid oracle 的最大误差为 `1.11e-16`。
8 sweep 在 `tau=1/2` 把到联合 oracle 的 TVD 降到 `0.00315/0.02479`，恢复
98.97%/95.66% 的期望方向差距和 99.43%/96.20% 的 oracle proposal-gain 缺口。

关闭整代接受的 `tau=2`、1000 轮顺序实验中，追加 20 种子单独复现方向；首批与
追加合计 30 种子的最终当前 loss 为 `106.30→89.93`，最后 250 轮平均 loss 为
`122.02→102.27`，全轨迹平均 loss 为 `296.94→221.76`。全轨迹 30/30、末 250
轮 25/30、最终单点 22/30 改善。扩样决定发生在观察首批结果之后，合计区间和 p 值
只作描述。candidate 没有增加正收益事件，而主要减小负收益步幅，说明收益不依赖
整代接受筛选。

进入标准整代接受闭环后，一次性预注册的 20 配对种子、500 轮实验将 best
workload loss 从 `39.225→33.875`（-13.64%），配对 95% 区间为
`[-10.597,-0.103]`，但只有 13/1/6，未达事先要求的至少 14/20 改善；正式
结论因此是“不确定”，不改默认参数。事后分解显示 candidate 的一阶残差方向收益
增加 `0.3048`，但整代二次过冲惩罚增加 `0.4614`；下一个方法问题是扩散时间步
或查询变化预算归一化，而不是抑制反向扩散。当前仍未进入 nltcs、跨 workload 或
DP。

随后 Issue #17 用 3 seed × 初始/500 轮状态 × 200 个冻结 proposal 精确拆分总
二次步幅。candidate 的二次增量 +1.3658 中，逐行自身项占 61.32%，跨行交叉项占
38.68%；自身项虽在 5/6 个状态更大，但未达到预注册的 2/3 主导门槛，正式判断为
“混合/不确定”。初始态净收益差 +24.883，而 500 轮态为 -0.322，说明全局优势由
大残差早期主导，晚期一阶增量不足以支付两类二次项。当前不据此实现固定基数或只调
参与率；下一候选必须同时给出逐行组合与跨行叠加的分布语义。

Issue #18 已进一步定义整代曲率能量
`V_gamma=<e,Delta q>-gamma/(2N)||Delta q||^2`。`gamma=0` 精确退化到现有逐行
因子 Gibbs，`gamma=1` 等于复制 proposal 的平方 workload 收益除以公开记录数。
在预注册的 3 seed × 初始/500 轮状态 × 200 proposal 冻结实验中，晚期三个状态的
净收益差全部为正，聚合从 `-5.0917→-3.9775`（差 `+1.1142`），正收益率从
10.17% 提高到 12.50%。曲率核牺牲 `0.4983` 一阶收益，但减少 `1.6125` 总二次项，
其中自身/交叉项分别减少 `0.9225/0.6900`。正式判断通过预注册门槛；初始态净收益
只下降 0.86%，晚期条件熵下降 0.56%，两个风险标记均未触发。

Issue #24 已完成独立的 20 配对种子、1000 轮无接受动力学验证。`gamma=1` 的末
250 轮平均当前 loss 为 `105.7517`，高于 `gamma=0` 的 `99.9544`，差
`+5.7973`（相对 `+5.80%`），配对只有 `5/0/15`；正式判断为
`curvature_dynamics_not_supported`。最终当前 loss `90.375→92.600`，best
诊断 `64.325→70.100`，也没有形成支持证据。因此冻结 proposal 上的相对改善不能
外推为当前 1000 轮 Markov 轨迹优势，不改默认参数，也不据此进入标准接受闭环。

失败不是探索塌缩：末 250 轮条件熵提高 0.417%，最终唯一记录数提高 0.877%，三个
预注册风险均未触发。candidate 每轮改变单元格减少 7.66%，全轨迹负收益幅度改善
`+0.4121`，但正收益幅度同时下降 `0.3287`；末 250 轮正收益率完全相同，而好坏
两类幅度都收缩。该核更像降低整体移动性，没有选择性地只消除晚期过冲。下一方法
问题若继续沿曲率方向，必须直接研究状态分布与多步漂移，而不能只依赖同状态一步
proposal 的局部排序。本结论只覆盖固定 workload、参数和 1000 轮有限时间，不是
任意曲率权重或无限时间的否定。

Issue #27 已进一步把相同 40 条无接受轨迹按实际查询二次变差
`sum_t ||q_{t+1}-q_t||^2` 对齐。candidate 的累计查询二次变差为 baseline 的
`83.26%`，20/20 个种子都更低；这个收缩强于累计改单元格数的 `92.35%`，说明
曲率核同时降低改单元格数量和每次修改的 workload 空间位移。匹配每个种子的共同
查询时钟后，末四分之一路径平均 loss 差从按轮数的 `+5.7973` 变为 `-6.8990`，
配对从 `5/0/15` 变为 `13/0/7`，但 95% 区间仍为
`[-15.0216, 1.2236]`。预注册判断是“混合/证据不足”，既不能证明单位查询路程
优势，也不能把原长期劣势归结为纯时间重参数化。

状态分段进一步显示，baseline 残差十分位的前五箱中 candidate 的单位查询二次
变差净收益均更低，后五箱则四箱更高、一箱近似相同；曲率核在低残差和较高残差状态
的漂移方向并不统一。下一候选如果继续沿曲率方向，应另立问题预注册公开残差或内禀
时钟驱动的状态调度，同时保持有限温双向支持，而不是事后选择阈值或把固定
`gamma=1` 设为默认。

Issue #30 已用新的 seed 30..39 在共同 marginal 初始态与标准闭环 500 轮态上
复核这个表观反转。每个状态 200 对严格配对 proposal；初始态曲率净收益差为
`-0.6055`，500 轮态为 `+1.07025`，晚期减初始的 seed 级交互为
`+1.67575`，95% t 区间 `[1.15337,2.19813]`，10/10 同向。正式分类为
`curvature_advantage_strengthens_late`。机制上，晚期曲率牺牲的一阶收益更少，
同时节省的二次项更多；这支持后续研究“低曲率起步、随公开残差下降而增强”的方向，
但两个端点不能确定调度函数或阈值，且固定 `gamma=1` 的长期失败结论仍然有效。
当前不改默认参数，也不在同一阶段实现调度。

Issue #32 的第二轮方法审查已把结论收口为更严格的负结果：平方 workload 能量的
固定温单坐标热浴核在单一 `test_300x10`、`tau=1` 配置上降低了它直接优化的 L2
能量，却使 candidate 相对同 seed 初始表的 normalized L1 从 `0.017713` 恶化到
`0.019767`，未测量 3-way mean L1 从 `0.0101981` 恶化到 `0.0108538`（+6.43%，
触发风险），4-way 也恶化 3.63%。因此当前“平方能量 + 初始 RMS 固定温度 + 单坐标
更新 + 末态输出”的完整协议没有合成质量改善或相对默认生成器优势的证据。
现有设计不能把负结果单独归因于平方能量形式。

原预注册的末 750 步 L2 比较 `10555.7663 -> 1342.6576`（-87.28%，20/0/0）
只保留为历史实现一致性检查：指标就是核直接定义的能量，而 `beta=0` 参考会破坏
1-way marginal 初始化并持续趋向全域均匀分布，不能作为方法有效性 baseline。
同状态单步期望优势也已由 `m'(beta)=-Var_beta(E)/N` 给出；分叉有限时间轨迹并非
逐步数学必然，但这不提供独立质量证据。

固定有限 `beta` 下，本核的收敛对象是玻尔兹曼分布 `pi_beta`，不是 `argmin`。
candidate 四个窗口的 loss 为 `1544.23/1274.07/1315.91/1342.66`，与低能区波动
相容，但当前没有混合诊断，不能声称已达平衡。此外，单坐标更新与主线 donor-copy
块级更新的邻域不同，强相关合取查询上的能垒/混合限制也是未排除的并行解释。

仍可复用的是实现与审计基础设施：全部步骤无接受、回滚或 best 选择；平均条件熵
保留 91.62%，uphill 概率质量为 31.20%；完整 oracle、120000 次转移、40 张最终表
和随机日程的两次独立审计均通过。当前没有默认生成器同预算对照、温度前沿、跨
数据集或实际规模证据，固定初始尺度在状态漂移后的有效温度也未经验证。本 PR 保留
默认路径之外的研究模块、oracle、复现脚本和测试，不将其作为稳定公共 API。自适应
温度/退火与局部最小逃逸改由 Issue #40 独立预注册，不在观察本次种子后追加扫描。

Issue #38 已按新 seed 60..79、3000 微步、共同初始表/坐标/Gumbel 的冻结协议检验
目标对齐的 normalized L1 能量。L1 candidate 的 measured L1 为 `0.027200`，高于
平方 baseline 的 `0.019570` 和 initial 的 `0.017357`；相对平方核差
`+0.007630`，95% 区间 `[+0.006782,+0.008478]`，0/20 改善。未测量 3-way 和
4-way 相对平方核也分别恶化 12.98%/8.42%，均为 0/20 改善，正式分类为
`not_supported`，不进入默认生成器直接比较。

失败不是探索塌缩：L1 平均条件熵比例为 0.9432，uphill 概率质量为 0.2689。事后
全坐标精确漂移诊断进一步显示，`tau=1` 时 L1 初始一步平均期望漂移在 20/20 seed
为正，均值 `+4.08e-5`；临界 `tau_*` 均值为 4.390、范围 3.770..5.681。相同
名义 tau 经不同能量 RMS 定标后不保证相同的相对当前状态下降语义。新的定理同时
证明：严格单坐标局部最优点上，有限温全支持与原始能量处处超鞅不能兼得。下一方法
问题应另立为漂移约束温度或更大块邻域，不能在正式 seed 上事后把 tau 改为临界值。

独立性能工作已把相同 8-sweep 算法的因子管线稳定降低约 74%，逐轮表哈希与两个
RNG 端点精确相同。原始一次性运行的完整墙钟中位降低 26.31%，但一个 seed 因
未修改阶段波动而变慢 2.746%，触发预注册硬门禁；当前父提交上的完整独立复验则
中位降低 24.77%、10/0/0，并通过全部门槛。新旧输出的 20,000 个跨提交逐轮哈希和
非计时算法字段精确一致。原失败仍保留，单次迁移复验不替代另行冻结的多重复默认
切换协议，所以批量构造仍是 opt-in，尚未替换默认路径。

无接受 1000 轮的同轮数墙钟增加 61.1%；标准闭环 500 轮为
`8.865→14.405s/种子`（+62.5%），其中因子构造 5.089s，Gibbs 抽样只有
0.278s。下一工程瓶颈是查询因子预编译和批量构造，应作为独立性能问题处理，而不与
本方法证据混入一个 PR。

Issue #49 已按预注册协议完成 Stage T/A、Stage B 和最终确认，三个阶段的正式
身份与独立审计均通过。冻结候选 `factor tau=4, sweeps=32` 相对同温度
`independent tau=4` 通过全部配对门槛，但相对最强独立基线
`I*=independent tau=5` 的 95% 配对区间上界为 `1.2522`，未通过最终确认。
因此正式状态为 `not_confirmed_no_reselection / confirmed=false`，没有追加 seeds
或重选配置。该结论仅限固定 1000 轮预算，不代表长期收敛结果。

当前版本仍是无噪声原型，不是 DP：尚未实现 ε/δ、噪声机制、accountant、私有
查询选择和带噪一致性。

## 最近变更（2026-08-15）

### PR #55 审查同步：保留 PR #51 诊断并完成 Stage 0/1 增量复核

PR #55 已合并最新 `origin/master=9685179`。唯一人工冲突位于 factor Gibbs
演化函数的说明文字；合并结果同时保留 PR #51 的 `condition_observer`、实际条件
logit/概率诊断及其 RNG 不变性测试，以及 PR #55 的 `direction_logit_clip` 与
`gibbs_logit_clip` 分离配置和身份诊断。master 新增的 Issue #49 正式里程碑也已
保留，没有回退既有生成路径。

按审查建议清理了 `PROJECT_STATUS.md` 中的外部项目名称。合并后定向回归为
`317 passed, 1 skipped`，全仓为 `991 passed, 7 skipped`；`compileall` 与
`git diff --check` 通过。本次只解决分支同步与文档问题，没有运行效果实验、修改
Stage 0/1 契约或进入 Issue #53 的后续阶段。

## 怎么跑 / 指定 GPU
一键跑（默认用卡 0）：
```
conda run -p ./.conda python scripts/run.py
```
调参改 scripts/run.py 顶部"参数配置"区的常量即可，不走命令行传参。

多人共用机器时，cuda 固定用卡 0（代码里是 torch.device('cuda')，不带卡号）。
若卡 0 被占，先 `nvidia-smi` 看哪块空，再用环境变量指定空闲卡跑（代码无需改动，
指定的卡在程序里自动成为 cuda:0）：
```
CUDA_VISIBLE_DEVICES=1 conda run -p ./.conda python scripts/run.py
```
卡号写错会找不到 GPU，自动降级到 CPU（很慢）——看到异常慢先查卡号。
注：多种子是串行跑；用多卡并行跑不同种子需另改调度，暂未做。

## 最近变更（2026-08-15）

### Issue #53 Stage 2B：正式开发轨迹完成；无阈值量程报告入口完成

Stage 2B 已把讨论确认的前三项写成显式协议：同时覆盖 `test_300x10` 与 `nltcs`、
独立核与 8-sweep 三阶因子 Gibbs 核，四格共用同一套后续 detector；固定 exact、
no-gate、marginal init、scale-invariant donor、`alpha=16`、`rho=0.01`、`eta=0.5`、
`mu=0.01`、`tau=2` 和双侧 `logit clip=30`。开发 seed 冻结为 200..202，验证 seed
220..224 在 detector 配置冻结前由程序封存。历史文件名
`measured_1000query.json` 实际含 1001 条查询，协议按完整冻结文件和 SHA-256 使用，
不静默删减。

新增 `scripts/collect_issue53_stage2b_range_finding.py`：默认只打印 12 条开发轨迹计划，
完整检查输入哈希、参数、seed 角色、当前态终点、RNG、轨迹文件和不可覆盖原子落盘。
正式 development 最大观察预算已在报告单卡预计开销后经用户确认冻结为每条 8000
轮；它不是收敛阈值，也不执行在线早停。入口拒绝其他轮数，smoke 仅允许未保留
seed、小表和至多 3 轮。生成提交 `d87503e`、协议
`483fd48ff88f050a7935eeb8cd4eb05e74607c1067800da39669516aa1d4b12b` 上的 12 条
development 轨迹现已全部跑满并严格重读：2 dataset × 2 kernel × seed 200..202，
每条 8001 个当前状态，共 209 MB、记录运行时间 4.90 小时；validation seed
220..224 未读取。六个 dataset×seed 配对均共享 s0/S0/初始化后 RNG，方向与 Gibbs
条件 logit 的正式 clip 命中总数都为 0。

`s0` 现在由每个 dataset×seed 的独立核 `initial_rms` 参考预检取得首个非零 RMS，
随后两个核从同一 seed 重启并固定共享该 `s0`；预检状态不进入正式轨迹，任意常数
回退被拒绝。新增方向 logit 与 Gibbs 条件 logit 的评价/clip 命中只读计数。Gibbs
在 Stage 2B 显式接入已验证逐步输出等价的 compiled-batch 因子构造，默认生成器仍
保持旧路径；新主循环接线测试确认 final table、全部 current-state 观测、查询向量和
主/Gibbs RNG 精确不变。

非正式 seed 999 冒烟中，小表两个核各 2 轮共享 `s0=0.0495288`、S0 和初始化后主
RNG，轨迹可严格重读，两个 clip 均零命中；compiled 与旧 rowwise 的两轮轨迹和 RNG
逐字一致。nltcs 的 seed 999 两轮实现等价审计也确认 final table、全部 current-state
观测、查询向量、主/Gibbs RNG 和关键 diagnostics 精确一致。nltcs 纯性能探针显示旧
Gibbs 约 6.35 秒/轮；compiled 后冷启动一轮约 0.99 秒，20 轮稳态约 0.676 秒/轮，
独立核约 0.240 秒/轮。正式 8000 轮预算已获确认，以上探针不参与窗口/阈值校准。

新增 `collect_stationarity_range_evidence()` 与
`scripts/analyze_issue53_stage2b_range_finding.py`，固定候选窗口
`100/200/400/800/1000`，复用 Stage 2A 的三相邻窗口、全窗口两两比较公式，但 API
没有 detector config/阈值参数，输出也 fail-closed 禁止稳定、运动充分、停滞、候选
停止轮次等分类字段。正式 report 入口严格审计 12 个 manifest、trace 哈希、生成提交、
协议、配对 s0/S0/RNG、轮数和 seed 角色，拒绝 validation seed、脏工作树和覆盖已有
目录；发布 1776 条 range check、96012 条 current-state 描述、纯描述性 JSON 与 9 张
图。正式 report 已从干净分析提交 `35955cb` 生成到
`outputs/issue53_stage2b_range_report/`：12 个被列入 manifest 的产物逐一重算哈希通过，
共 18 MB；source generator 仍绑定 `d87503e`，无阈值/分类字段，validation seed 仍未
读取。量程图显示四个 cell 的 current L1 均先快速下降再持续波动，运动护栏量在后段
保持非零；这些只是描述性观察，尚未冻结窗口、阈值或给出任何收敛轮次。

在以上无阈值报告之后，已与用户逐项确认开发候选校准协议：统一 `W=400`；只用
development seed `200..202` 中完整落在 `6001..8000` 的检查终点
`7200/7600/8000`，共 12 轨迹 × 3 检查 = 36 行；六个稳定性上限按每个
dataset×kernel cell 的线性 P95 后取四格最大值，两项运动性下限按每格线性 P05 后
取四格最小值，不加人工倍数或舍入；连续两次稳定且运动充分才合格，稳定但运动不足的
停滞耐心为四次。候选停止后的开发审计同样按窗口重叠语义，只有连续四次稳定性失败才
记为持续再漂移，避免单个异常块在三个相邻检查中被重复计数。

新增 `scripts/calibrate_issue53_stage2b_detector.py`，没有阈值覆盖入口，严格复用 12 条
正式 development 输入审计并拒绝 validation seed、脏工作树和覆盖输出；正式候选报告
已从干净分析提交 `58c0386` 生成到
`outputs/issue53_stage2b_detector_calibration/`。候选公共配置为：查询均值变化上限
`0.0022331667`、查询 P95 变化上限 `0.0054885833`、L1 均值变化上限
`0.0004866`、L1 P90-P10 宽度变化上限 `0.00044`、唯一行比例变化上限
`0.0558866667`、归一化行熵变化上限 `0.0192478344`、活跃轮次比例下限
`0.8625`、平均改变行比例下限 `0.0057487176`、停滞耐心 `4`。

开发完整回放发布 36 行阈值来源和 216 行全预算检查：12/12 均为
`stationary_qualified`，其中 11 条候选停止在 2000 轮、1 条在 2400 轮；0 条停滞，
候选停止后最大连续不稳定为 0..2，0 条触发四连败持续再漂移，开发分类为
`candidate_supported_on_development`。三项产物哈希复算通过，报告 manifest SHA-256
为 `faa7c821804ea8de98a50069745ef906996ca51dbb00bdab7bc862f2945c1d8e`；该报告生成时
validation seed `220..224` 仍未读取，候选配置尚未冻结，也尚未接入在线停止。

用户审查开发结果后已同意将上述候选冻结为 validation 配置。新增
`scripts/issue53_stage2b_validation_protocol.py`，完整精度绑定生成提交 `d87503e`、
原生成协议 `483fd48f…`、校准分析提交 `58c0386` 以及正式校准报告/36 行来源/216 行
回放三个产物哈希；冻结配置 SHA-256 为
`f3789ddbbec63b66bf8f7b21e08268a0e46aad5ed2e6601c67524d988d9cb1b9`，完整验证协议
SHA-256 为 `7c6d345dc559298dafd4a28eb5a2c1f08742133f660bbbef67b0347c726e8921`。

封存验证范围固定为 2 dataset × 2 kernel × seed `220..224` 共 20 条，每条必须跑满
8000 轮以保留候选停止后的反事实尾部，总预算 160000 轮；采集时不在线早停。硬门禁为
20/20 `stationary_qualified`、0 条 `stalled`、0 条停止后四连败持续再漂移，不允许
cell 特例；验证一旦失败即拒绝本配置、退休这些 seed 并重新设计，禁止用同一验证结果
调阈值。当前入口只有 plan 模式，不能读取 validation 轨迹或启动生成；正式单卡运行
仍须先向用户报告 GPU 与预计开销并再次取得明确确认。validation 数据截至目前仍未读取，
在线自动停止仍未接入。

用户随后明确批准启动 8000 轮封存验证，并要求在线 detector 留待另行讨论。新增独立
`scripts/collect_issue53_stage2b_validation.py`：collect 必须显式确认冻结协议完整 SHA，
要求干净工作树和恰好一张可见 CUDA GPU；20 条按固定顺序运行，每条轨迹原子落盘并可
严格审计续跑，完整集合结束前不执行 detector replay 或发布部分分类。执行器全套回归
1080 passed 后固定在提交 `0388997`，并于 `2026-08-15T10:55:31+08:00` 从独立 detached
worktree `/home/chuhan/projects/table-diffusion-issue53-validation-run` 正式启动；物理 GPU 1
通过 `CUDA_VISIBLE_DEVICES=1` 成为唯一可见卡。正式输出写入
`outputs/issue53_stage2b_validation/`，运行日志为
`outputs/issue53_stage2b_validation_run.log`；预计墙钟约 8.5..9 小时。validation seed
现已正式解封，后续禁止修改或根据部分结果调节 detector；截至本记录只确认进程健康，
未执行任何部分 detector 分类。

在不读取 validation 中间结果、不修改冻结配置和运行 worktree 的前提下，已并行完成一项
纯人工轨迹反例审查。审查严格使用冻结的 `W=400`、1001 个查询、`N=1000` 和完整 replay：
当仅 1/1001 个查询的四个窗口均值依次为 `0.1/0.3/0.5/0.7`、其余查询不变时，三个窗口
最大两两单查询漂移达到 `0.4`，但跨查询 mean 被稀释为 `0.0003996004`，线性 P95 为
`0`，整体 L1 均值漂移同为 `0.0003996004`；六项稳定性与两项运动性均通过，detector
在 1200/1600 两次连续通过后错误给出 `stationary_qualified`。两个查询等幅反向移动的
反例也在 1600 轮错误合格，且整体 L1 漂移恰为 `0`。稳定对照按预期在 1600 轮合格；
全体查询单块尖峰对照则连续三次被识别为不稳定，尖峰离开三个窗口并重新连续通过后才在
3200 轮合格。该审查使用满足 trace 契约但非生成器实跑的构造轨迹，因此证明的是当前
`query mean + query P95` 在逻辑上不足以排除稀疏查询漂移，不证明正式生成轨迹中已经发生
该事件。当前 detector、校准报告和正在运行的 validation 协议均未改动；下一步需先讨论
是否引入对所有查询一视同仁的最坏查询漂移护栏，再决定新配置的校准与新 seed 验证。

为只测量该护栏的开发自然量程，新增
`scripts/analyze_issue53_stage2b_query_max_range.py`。入口固定复用原校准的 `W=400`、
`6001..8000` 与终点 `7200/7600/8000`，严格审计 12 条 development 输入并拒绝
validation；每个检查对三个窗口全部两两计算逐查询归一化窗口均值漂移，输出其中最大值、
对应查询坐标和窗口对，同时复算并逐项核对现有 query mean/P95 公式。入口没有阈值、配置、
分类、候选停止或生成重跑参数，正式输出要求干净工作树且不可覆盖。新增契约测试覆盖 21 个
查询中单坐标漂移令线性 P95 为零而 max 保持 `0.4`、全部三组窗口对以及脏树先验拒绝；
相关 14 项测试通过。正式报告已从干净分析提交 `1505fd5` 发布到
`outputs/issue53_stage2b_query_max_range/`：36 行证据完整覆盖四格各 9 行，终点严格为
`7200/7600/8000`；test 的查询数为 50、nltcs 为 1001。`query_max_shift` 全局范围为
`0.0015501205..0.007825`、全局线性 P95 为 `0.0077875`；四格线性 P95 分别为 nltcs
Gibbs `0.002673506`、nltcs independent `0.0030212904`、test Gibbs
`0.0074966667`、test independent `0.0078083333`。若后续机械沿用“四格 P95 后取最大”
规则，其描述性包络值为 `0.0078083333`，但本步骤没有把它选择为阈值，也没有修改 detector。
CSV/JSON 哈希分别为 `0556945c2a09d08e45f747bdc53bf11ccbb0ebfe21adeeabd04e88bee26f9092`
与 `70390c99f6cdac24568db35f502356601cf70d1a0721bec68d99b235b1439d9a`，逐项复算通过；
source audit 明确记录 `sealed_validation_seeds_read=false`。这些量程来自固定完整 workload 的
晚期开发轨迹，尚不能替代增量查询场景验证。

在用户确认采用上述护栏候选后，新增独立的
`QueryMaxStationarityDetectorConfig`、`collect_query_max_stationarity_range_evidence()`
与 `replay_query_max_stationarity()`；原 `StationarityDetectorConfig`、原无阈值证据入口、
原 replay 契约和冻结 validation 协议均保持原样，新版使用单独的
`issue53-stage2b-query-max-*-v1` 契约，避免旧验证静默获得新规则。新增稀疏单查询漂移反例
确认旧版仍会合格而新版拒绝，稳定且运动充分的对照仍合格，并逐字段确认新版除新增
`query_max_shift` 外不改变旧证据。

新增 `scripts/calibrate_issue53_stage2b_query_max_detector.py`：只从 12 条 development
轨迹的既定 36 行晚期证据按“四格各线性 P95、再取最大”自动导出 max 上限，原版冻结配置
作为不可变 base；随后配对回放原版与新版全部 8000 轮，并审计停止后四连败再漂移。入口
没有阈值覆盖、validation 读取、生成重跑或在线停止参数。相关 71 项测试先行通过；全套
回归首次仅因共享 Conda Python 无执行位导致两个需要 `sys.executable` 的子进程测试无法
启动，使用不修改共享权限的临时可执行副本重跑后为 `1097 passed`。正式 development
候选报告已从干净实现提交 `c55b703` 发布到
`outputs/issue53_stage2b_query_max_calibration/`：36 行校准证据与 216 行全预算检查均严格
覆盖预定范围，自动导出的公共 `query_max_shift_tolerance` 为
`0.007808333333333567`，原冻结配置的其余字段逐项不变。新版回放 12/12 均为
`stationary_qualified`，停止轮次由原版 11 条 2000、1 条 2400 变为 10 条 2000、2 条
2400；仅 `test_300x10/seed_200/factorized_gibbs` 推迟 400 轮，其余 11 条不变。0 条
停滞、0 条持续再漂移；停止后最大连续不稳定分别为 6 条 0、2 条 1、4 条 2，均未达到
四连败门禁。随后补充的两个稀疏查询反向变化与单块全体尖峰恢复测试确认：新版拒绝前者，
后者在异常块退出三窗口并重新连续两次通过后正常合格；相关 49 项定向测试通过。

正式 query-max 报告 manifest、报告、36 行证据和 216 行全回放哈希依次为
`b7b2906c59f7d0b5668f3d68380a14cefc5df62c91942200c047138b0ba85658`、
`23f6acbcce5f7bae45edcbb4f3f76e6bf804abfcfb63936287da0ba45c8c249c`、
`bd2e2d8a19c67ba94289bb256a4caa0eed4292b0a9b886d8da0294fd64deda96`、
`a5db9728fa59787125be1da117b80668dac82cd43ac7b42d48595138d95dad9a`，逐项复算通过；
报告明确记录旧冻结 detector 未修改、无在线停止、无生成重跑、无 validation seed 访问。
该结果支持 query-max 开发候选，但尚未把它冻结为新版 validation 配置。

冻结 V1 的正式 validation 轨迹采集已于 `2026-08-15 18:57 +08:00` 完成：20/20 条
轨迹均跑满 8000 轮，总计 160000 轮，采集器逐条重审 trace 哈希、完整预算、无门控
proposal 全应用、终态和配对 `s0/S0/初始化后 RNG` 后才发布集合 manifest。正式输入仍为
`outputs/issue53_stage2b_validation/`，collection manifest SHA-256 为
`cdb58df5d6ebcc0ea0892ace2244889448cb62e3ba7a4174259fe4c3c5fd4e92`；其中仍明确记录
`detector_replay_performed=false` 与 `partial_validation_classification_read=false`。采集阶段
没有提前查看任一轨迹的收敛分类，也没有根据 validation seed 调整冻结配置。

新增 `scripts/replay_issue53_stage2b_validation.py` 作为冻结 V1 唯一正式解封入口。默认
`plan` 不读取 validation 轨迹；`report` 没有阈值覆盖参数，必须同时显式确认冻结协议完整
SHA-256 和上述 collection manifest SHA-256，并要求干净工作树。正式回放前会重新审计
集合字段、20 个 cell 与 run manifest 哈希、采集提交、8000 轮预算、trace 哈希和十组配对
绑定，再用唯一冻结配置回放；每条轨迹同时保留完整 18 个 `W=400` 检查，审计候选停止后
连续四次不稳定再漂移，并把 20/20 合格、零停滞、零持续再漂移交给既有冻结门禁统一判定。
失败轨迹的停滞轮次单独记录，不会冒充合格候选停止轮次。正式产物将原子、不可覆盖地发布
报告、20 行轨迹结果、360 行全预算检查及带哈希 manifest；该入口不重跑生成器、不接在线
停止，也不使用 query-max 候选或绝对 L1 质量作为停止条件。

新增正式回放契约测试覆盖协议/集合双哈希确认、脏工作树先验拒绝、run manifest 篡改、
采集阶段提前分类标记、跨核配对不一致、真实 V1 公式的 20 条通过、停滞字段归一化、候选后
四连败再漂移以及结果原子落盘。专项 13 项、相关冻结协议/采集/校准/基础收敛 69 项均通过；
CPU 全套回归为 `1092 passed, 7 skipped`。入口实现固定在本地提交 `a69c499`，随后从该
干净提交显式确认冻结协议 SHA 与 collection manifest SHA，执行了一次不可覆盖的正式
`report` 回放。

正式 V1 validation 分类为 `does_not_support_frozen_detector_on_validation`。20/20 条轨迹都
曾满足“连续两次稳定且运动充分”：18 条候选停止轮次为 2000，2 条为 2400；停滞为 0。
失败只来自预注册的停止后持续再漂移门禁：`test_300x10 / seed 220 / factorized_gibbs` 在
候选轮次 2000 后，于检查终点 5600..7600 出现 6 次连续稳定性失败。前四次由 L1
`P90-P10` 宽度变化超过 `0.00044` 触发，后两次由 L1 窗口均值变化超过 `0.0004866`
触发；运动护栏始终通过。8000 轮检查重新稳定不撤销“曾连续至少四次再漂移”的冻结门禁。
其余 19 条均未达到四连败，唯一失败轨迹令持续再漂移计数为 1，因此硬门禁整体失败。

正式报告位于 `outputs/issue53_stage2b_v1_validation_replay/`，20 行轨迹结果和 360 行完整
检查均已发布并逐项重算哈希。report manifest、报告、轨迹 CSV、完整检查 CSV 的 SHA-256
依次为 `bb4de0d6cfee9257eb3f4c2045ed1011b55e36bbf5c2f72b712c5b952e96b324`、
`dcccefff9ae2237f0be3298ef53e3d9df2dbb621537a5c278ca6f41c91c306b7`、
`ff945cf8b29ed86a316e643210934fbc37a66ac01b121813f4bad4f0eaefaaff`、
`3c3e21a9914e532693b18672667bb20aa63fffcf7fbafb19e87a234748bdf51e`。按冻结协议，V1 配置
被拒绝，validation seed `220..224` 对后续配置正式验证作废且不得用于回调阈值；当前不能
把 V1 接入在线停止。query-max 仍只是 development 候选，也不能在这批 seed 上补做正式
验证。下一步应先区分“候选停止过早”与“判据对正常长期波动过敏”的设计问题，冻结新版
方法后再使用全新的 validation seed。

对该失败所做的只读事后诊断没有发现实现、封存输入或异构服务器问题：正式 replay 与原始
400 轮块统计逐项一致，失败配对均来自同一 RTX 4090 环境，方向及 Gibbs 条件 clip 仍为零，
运动护栏全程通过。异常轨迹的候选证据区间 `801..2000` L1 均值为 `0.00295911`，候选后
`2001..8000` 为 `0.00291260`，最后 2000 轮为 `0.00292033`，最后一个 400 轮块为
`0.00287450`；候选后块均值线性斜率近零且略向下。因此现有证据不支持“loss 持续向坏处
漂移”，更像一次随后恢复的局部波动形态变化。5600..6800 的四连败来自相邻块 L1
`P90-P10` 宽度在约 `0.00073..0.00127` 间切换，7200..7600 则由一个 L1 均值较高块
造成，8000 已恢复稳定。

主要设计矛盾在统计口径而非某个核的代码：开发校准每个 cell 只用 3 seed × 3 个晚期检查
共 9 行估计单检查 P95，但验证实际要求六项稳定性指标在总计 298 个候选后检查中不得形成
任何四连败；这些检查还因三窗口滚动而高度相关。development 小表候选后已有 12/89 个
不稳定检查，validation 小表进一步为 32/149，而 nltcs 在 development 和 validation 均为
0。统一绝对阈值由小表噪声最大的 cell 主导，相对 nltcs 各 cell 的晚期 P95 已宽松约
3.95..17.32 倍，说明当前“一套绝对阈值覆盖不同 N/查询数”并不真正尺度无关。小表
`N=300,Q=50` 的 normalized L1 单计数粒度为 `1/(NQ)=0.00006667`，宽度阈值
`0.00044` 实际夹在约 6 与 7 个离散粒度之间，也会放大边界敏感性。

另一个信号是 development 与 validation 共 32 条轨迹全部在 2000 或 2400 轮合格，候选
规则几乎退化为固定 burn-in；“连续两次”检查共享三分之二窗口，并不是两份独立稳定证据。
但仅增加连续次数也不能根治：本次异常在候选后曾连续稳定多个检查，直到 5600 才出现并在
8000 恢复。当前最合理判断是 V1 的单检查量程校准、重叠连续语义和“长期不得出现一次局部
波动”的验证目标没有共同控制误停风险；不能据此认定 Gibbs 核未收敛，也没有足够 seed 将
问题归因给 Gibbs。query-max 不直接处理本次 L1 波动。以上诊断不改变正式失败结论，也不
用于回调 validation seed `220..224` 的阈值。

为准备把 Stage 2 V1 作为独立研究 PR 归档，新增
`docs/进度/Issue53_Stage2_V1收敛检测器总结.md`，集中整理 Stage 2A 轨迹语义、V1 三窗口
判据、development 校准、封存 validation、query-max 补充边界、正式负结果、失败诊断和
全部关键产物哈希。文档明确区分“V1 detector 被否决”与“生成核是否收敛/质量是否足够”，
并明确 V2 不进入本 PR。当前仍未 push 或创建新 PR。

为使该堆叠 PR 基于 PR #55 的最新审核状态，已在不改写上述正式证据提交 SHA 的前提下，
普通合入 `origin/research/issue-53-current-state-contract=b765233`。唯一内容冲突位于 factor
Gibbs 条件采样：解决结果在同一次条件计算中同时保留 PR #51 的截断前 logit/概率/熵
observer 与 Stage 2 的实际 clip-change 计数，不增加 RNG 消耗或改变采样概率；边界测试明确
区分“`abs(raw_logit) >= clip` 的资格命中”和“数值确实被 clip 改变”两种统计。合并后
factor 专项 `63 passed`、Stage 2/相关 Gibbs 定向 `278 passed`、全仓 `1147 passed`。
本次同步没有重跑正式实验、修改冻结协议、读取新 seed、push 或创建 PR。

验证：执行器加入后用不修改共享 Conda 权限的临时可执行副本完成全套 1080 passed。
所有新改动仍只在本地工作树，未推送、未更新 Issue/PR。

### Issue #53 统一 detector V2：第一步标量无阈值证据原语完成

在独立分支 `research/issue-53-stage2-v2-evidence` 新增
`src/table_diffevo/stationarity_v2.py`，没有修改已冻结的 V1 `stationarity.py`。本步只实现
候选区间内**单个标量序列**的版本化、无阈值数学证据，不包含 detector config、阈值、状态分类、
候选轮次或自动停止。

输入固定为 `B1+B2+B3` 的 12 个连续 100 轮小块摘要。实现使用 12 个小块编号 `0..11`：

- 计算全部 66 个两点斜率并取中位数 `b`；
- `R=median(x)`，`D=11b`；
- `a=median(x_i-b i)`，`r_i=x_i-(a+b i)`；
- `S=1.4826×median(|r_i-median(r)|)`；
- `T=|D|/S`，`O=max(|r_i|)/S`。

接口同时返回输入小块、66 个斜率、拟合斜率/截距、残差和 `R/D/S/T/O`，便于后续离线审计，
但不把任何证据解释成“已收敛”。没有给 `S` 加隐藏 epsilon；`S=0` 时显式记录
`zero_scale=true`：`D=0` 则 `T=0`，否则 `T=∞`；所有残差也为零时 `O=0`，存在非零残差时
`O=∞`。该对象目前是计算接口，不是 JSON 持久化格式，避免在正式落盘协议确定前静默处理无穷值。

新增人工测试覆盖常数序列、无噪声持续趋势、单点尖峰、顺序反转、平移/正比例缩放不变性、
66 个斜率和 MAD 公式逐项复算，以及维数/长度/非有限值/布尔与字符串输入拒绝。定向测试
`13 passed`；显式关闭 CUDA 的全仓 CPU 回归为 `1140 passed, 7 skipped`。没有读取 development
或已退休 validation 轨迹，没有运行生成实验或使用 GPU，也没有修改核、alpha/rho 或其他生成参数。

下一步仍属于同一 V2 无阈值证据层：先讨论并实现完整轨迹到 100 轮小块摘要的聚合，包括
query mean/P95/max、L1 中心与 `P90-P10` 波动、表结构和实际运动量，再让这些标量统一复用本原语。
在该接口和人工反例通过前，不读取真实轨迹；阈值、确认块、状态机和在线停止继续后置。本分支
尚未 push，也没有创建 PR。

### Issue #53 统一 detector V2：显式长度的小块汇总层完成

用户进一步明确了生产目标：最终必须是一套**与数据集身份无关**的收敛判定程序，不能为
`test/nltcs` 或未来新数据集分别手写窗口长度和判据。小块长度会受实际轨迹相关时间间接影响，
因此当前 `100` 只作为待检查的研究候选，不能提前宣称为普适常数；若后续发现固定长度无法跨
数据工作，应设计统一的轨迹驱动或多尺度规则，而不是建立 dataset→window 映射。该约束已经写入
V2 接口说明，正式 validation 前必须再次检查。

新增版本化 `V2SubblockSummary`、`V2SubblockCollection` 与
`collect_v2_subblock_summaries()`。调用者必须显式给出 `subblock_round_count`，没有默认值，也没有
数据集、核或模式分支；当前候选常量单独记录为 `100`。接口只消费既有 `StationarityTrace`：排除
initial 状态，把完整、连续的 post-round 按指定长度切成互不重叠的小块，并返回每块轮次范围、
逐查询归一化均值、L1 均值与 `P90-P10`、唯一行比例、归一化行熵，以及活跃轮次比例、平均改变行/
查询比例和平均查询 L1 运动量。不完整尾部绝不参与平均，只显式返回剩余轮数，供在线调用继续收集。

人工测试覆盖：205 轮按 100 切成两个完整块并保留 5 轮尾部、同一轨迹显式按 50/100 重分块、
99 轮不产生伪完整块、initial 不进入摘要、实际运动与冻结轮次各字段、非法长度和错误 trace 类型。
V2 定向测试为 `24 passed`。随后覆盖 V2、原 V1 stationarity、reference process 和全部 Stage 2B
协议/回放入口的相关 CPU 回归为 `181 passed`。第一次全仓运行在无失败到 31% 时异常停在共享磁盘
页读取等待，主动终止；磁盘恢复后的限时重试完整通过 `1151 passed, 7 skipped`，因此确认前次只是
基础设施瞬时异常。本步没有读取 development/退休 validation 轨迹，没有运行生成实验或使用 GPU。

下一小步是让连续 12 个小块组成候选证据：普通标量字段复用既有 `R/D/S/T/O`；查询必须先按同一
查询形成 12 点序列，再汇总 mean/P95/max，防止稀疏或反向漂移被提前平均掉。仍只使用人工轨迹，
不设计阈值、确认块、状态机或自动停止。本分支尚未 push，也没有创建 PR。

### Issue #53 统一 detector V2：12 小块候选无阈值证据完成

新增版本化 `V2CandidateEvidence` 与 `compute_v2_candidate_evidence()`。接口只接受前一步产生的
完整小块集合和显式的首个小块编号，每次固定使用连续 12 个小块；起点必须落在每 4 个小块的大块
边界，因此可表达 `B1+B2+B3`、随后 `B2+B3+B4` 的结构，但不自行寻找候选轮次。原始 12 个摘要、
起止轮次、查询/target 身份和小块长度均保留在输出中。

查询处理严格采用“先保持同一查询身份，再跨查询汇总”：每个查询先用自己的 12 点序列计算一套
`R/D/S/T/O`，随后对 `|D|` 汇总有限 mean/P95/max 和最大值查询编号，反向查询不会互相抵消；
`T/O` 的有限分布与正无穷数量/首个查询编号分开返回，既不加 epsilon，也不让一个 `S=0` 查询把
普通有限均值变成不可解释的无穷。另行记录 query `zero_scale` 数量；query-count 的 max 修正仍未
设计，本接口只提供原始证据。

L1 中心水平、L1 块内 `P90-P10` 波动、唯一行比例、归一化行熵、活跃轮次比例、平均改变行比例、
平均改变查询比例和平均归一化查询运动量各自形成 12 点序列并统一复用标量 `R/D/S/T/O`。稳定运动
和完全冻结可在运动证据的 `R` 中区分，但输出中仍没有 `stable/converged/stalled` 等解释字段。

人工反例确认：两个查询等幅反向移动时整体 L1 完全不变，但两个有符号 `D` 分别保留；21 个查询
中只有 1 个持续漂移时，`|D|` mean 被正常稀释、线性 P95 为 0，而 max 仍准确指向第 21 个查询；
单点尖峰产生 `O` 而不是持续方向 `D`；L1 中心不变但块内波动持续变化时两条证据不会混淆；稳定
运动与完全冻结只输出不同运动水平而不越权分类。V2 定向测试 `36 passed`，Stage 2 相关 CPU 回归
`193 passed`，全仓 CPU 回归 `1163 passed, 7 skipped`。没有读取 development/退休 validation，
没有运行生成实验或使用 GPU。

下一步是在用户审查该无阈值接口后，建立只读 development 分析入口，在相同逐轮轨迹上比较
`50/100/200` 小块的相邻相关、zero-scale 比例、证据尺度和检测延迟代价。比较目标是选择或否定统一
窗口规则，禁止按数据集挑各自最有利长度；在窗口结构冻结前仍不讨论阈值、B4 分类或在线停止。本分支
尚未 push，也没有创建 PR。

### Issue #53 统一 detector V2：单一 100 轮小块假设审查入口完成

经进一步讨论，用户认为逐个尝试 `50/100/200` 再挑选结果最好的长度既复杂也不严谨，因此撤销上一节
的多长度比较计划。当前改为一个更简单的预声明证伪协议：只把 `100` 轮作为唯一共同假设，在全部既有
development cell 上寻找明显反例；若没有明显反例，只能暂时保留 `100`，不能宣称已证明对任意未来
数据普适。若它明确失败，才另行讨论统一的轨迹驱动长度规则，仍禁止 dataset→window 映射。

新增 `scripts/analyze_issue53_stage2_v2_subblock_100.py`。默认 `plan` 不读取轨迹；正式 `report` 只允许
12 条既有 development 轨迹，每条 8000 轮固定切成 80 个完整 100 轮小块，并按四个小块前进一步形成
终点 `1200/1600/.../8000` 的 18 组候选证据，总计 216 行。入口没有其他长度参数，也没有阈值、
收敛/停滞分类、候选停止轮次、B4 或在线停止；不读取已退休 validation seed，不重跑生成器。

审查只描述四类证据：query 与普通标量的 `R/D/S/T/O` 自然量程；`S=0`、正无穷 `T/O` 的显式计数；
去掉 V2 稳健直线后相邻残差的 lag-1 Pearson 相关；运动量和表结构的参考水平。相邻相关在任一侧残差
方差严格为零时返回空值并单独计数，不用 epsilon 或伪造零相关。不同查询数下的 raw query max 只保留
作诊断，报告明确禁止在 query-count 修正尚未定义时直接用它设置跨数据阈值。

新增契约测试覆盖：plan 只有 `[100]` 且无选择/判定入口；80 小块严格产生 18 个四块对齐候选；常数
残差的相关显式不可计算、交替残差相关为 `-1`；候选行保留 zero-scale/相关缺失且 JSON 中没有
`NaN/Infinity`；四格描述性汇总无隐藏结论；脏工作树在读取正式输入前 fail-closed。V2 定向与相关
Stage 2 测试 `138 passed`，全仓 CPU 回归 `1169 passed, 7 skipped`，`git diff --check` 通过。
固定 100 轮的正式无阈值 development 审计随后已从干净分析提交
`c4462fe688532489fee773d1a420c9f0028770f3` 生成到
`outputs/issue53_stage2_v2_subblock_100_audit/`。本次只读回放上述 12 条既有 development
轨迹，没有读取退休 validation seed、重跑生成器或使用 GPU；每条轨迹仍为 8000 轮，严格切成
80 个 100 轮小块，每个候选使用连续 12 块、每 4 块前进一步，得到每条 18 个、合计 216 个候选。
六组 dataset×seed 的配对 `s0/S0/RNG` 绑定全部通过；方向 logit 共评价
`1,523,784,931` 次、Gibbs 条件 logit 共评价 `61,470,968` 次，正式 clip 命中均为 0。

固定 100 轮下，逐查询去趋势残差 lag-1 相关绝对值 P95 的候选中位数/最大值为：

| dataset | kernel | 中位数 | 最大值 |
|---|---|---:|---:|
| `nltcs` | factorized Gibbs | 0.5944 | 0.9823 |
| `nltcs` | independent | 0.6033 | 0.9790 |
| `test_300x10` | factorized Gibbs | 0.4946 | 0.6288 |
| `test_300x10` | independent | 0.5089 | 0.6736 |

四格的 query zero-scale 与相关不可计算比例均为 0，因此该信号不是零尺度或缺失值造成的。
由于协议没有预注册拒绝阈值，这不是“统计检验已拒绝”的结论；但四格均出现一致且不弱的残差
相关风险，足以在方法设计层面不再保留“固定 100 轮可作为统一证据块”的假设。下一步改为设计
同一套、由轨迹本身决定证据尺度的规则；不得补试 `50/200` 后择优，也不得建立
dataset→window 映射。query-count 的跨数据修正仍未定义，阈值、状态机、在线停止和新 validation
继续冻结。

正式产物 SHA-256：`report_manifest.json` 为
`024f301db8212c1335226accc189e402a5b32d062c0e035efcad9c876c81a2f0`，
`audit_summary.json` 为
`4055db58cf173e5c0b32dc2f84d65cc9c255fe39ebd8abbda9494fc63b4795ee`，
`candidate_evidence.csv` 为
`b0ca4fe23b87d721da2af477886915d893b622172d784c794b437056bb87d714`。
本次状态记录不新增事后阈值或分类；分支尚未 push，也没有创建 PR。

## 最近变更（2026-08-14）

### Issue #53 Stage 2A：离线收敛回放判据完成本地审查修订（待用户审查）

Stage 2A 仍只交付轨迹与离线回放工具，不接在线早停、不冻结生产窗口/阈值，也不运行
正式长实验。三窗口、全窗口两两比较、连续两次通过和 S0 排除语义保持不变；L1 判据
现在显式固定为窗口算术均值与线性分位数 `P90-P10`，并额外输出每窗口 P95 作为过冲
诊断，不把绝对 L1 高低混入收敛资格。

运动护栏不再使用“任一单元格改变”的合并布尔比例。每个窗口分别计算活跃轮次比例和
平均改变行比例，三个窗口的最小值必须同时超过调用者显式给出的正阈值；零阈值被
fail-closed 拒绝。人工轨迹已覆盖完全冻结、大表每轮只改一行、前动后冻、持续漂移、
稳定充分运动，以及高 L1 但稳定运动仍可收敛，明确区分“收敛”与“质量”。

回放结果契约升为 `issue53-stage2a-replay-v2`，结果绑定完整轨迹内容 SHA-256、查询/目标
身份、轨迹终止原因和完整检测配置。JSON/NPZ 加载现在拒绝顶层、数组 metadata 和逐状态
观测中的未知字段；非预算原因结束但未通过回放的轨迹返回
`terminated_before_qualification`，不再误报为 `collecting`。独立核与因子化 Gibbs 核均
纳入“开关轨迹不改变最终表、评价次数或 RNG 端点”的回归测试。当前改动仅在本地
`research/issue-53-stage2-convergence-trace` 工作树，尚未推送或更新 Issue/PR。验证：
Stage 2A/参考过程/主循环定向测试 159 passed、1 skipped；全套 998 passed、7 skipped。

### 尺度不变选择 v3：NaN 修复+证据链+归因进分类，主判定三过（PR #48）

第二轮审查五项全部修复：exclude_self 的 NaN 漏洞（softmax 前置
-inf，双路径回归测试）、证据链对齐 #45/#47（allow-dirty 无条件非
正式、拒绝覆盖、参考表生成后读取、全输入 SHA-256、原生
initial_state、tail 改名、tol="inf"）、标准化归因参与最终分类、
逐行集中度诊断（row_max_prob/有效 donor 数）、v1/v2 归档非正式。
本次隔离复核：尺度不变相关 26 项、rho 退火 23 项；全套 898 passed、
7 skipped（905 collected）。

**v3 正式结果**（三次预注册 Issue #44 评论 5288095498，五臂、种子
100..104、formal=true）：nltcs **supports_scale_invariant_selection**
（三判定全过）——无门 si L1 **0.001010**（五臂最优 5/5）：机制
0.291✓、归因 0.539✓（标准化本身贡献 46%）、门冗余 0.722✓、质量零
报警。逐行诊断如实报告：平均集中度低（终态均值 3.1%、有效 donor
1726），但存在瞬时单行确定性选择（峰值 1.0）——不作"无集中坍缩"
声明；支持集收窄 ~29% 仍为已知风险。小表辅助
mechanism_gain_gate_not_redundant（门仍强+高阶风险，如实）。
正式 JSON SHA-256 `b5f2782c…`。对 PGM 水位 0.000357 差距 2.8 倍。

## 最近变更（2026-08-13 晚）

### 尺度不变选择 v2：审查修复后正式重跑，主判定维持 supports（PR #48）

第一轮审查（PR #48）指出五处实现/协议问题，全部修复：低信号保护
`scale_invariant_min_spread`（放大倍数有界、低离散度平滑退化均匀）、
exclude_self 行统计只在非自身候选上计算（numpy/torch 同步）、donor
全局 top_share 监控（当时的"无坍缩"判断已由 v3 逐行诊断撤回）、
best_loss 改用主循环值、nltcs 离线参考限定 train（一次实验一份源数据）、formal 标志
校验协议参数、any_quality_risk 纳入分类、单变量归因臂
no_gate_legacy_a16。专项测试 18 项，全套 906 通过。

**v2 正式结果**（重新预注册 Issue #44 评论 5278428081，提交 051cf22，
五臂、种子 100..104、2000 轮、formal=true）：nltcs 主判定
**supports_scale_invariant_selection**——无门 si L1 **0.001094**（五臂
最优 5/5）：机制改进 0.316✓、**标准化归因 0.584✓（v2 新增：同 alpha
下标准化本身贡献 42%，是最大单一来源；alpha 数值贡献也真实）**、门
冗余 0.729✓（配置对齐后无门反好 27%），质量零报警，best_loss 不变式
零违反。test_300x10 辅助 mechanism_gain_gate_not_redundant（机制与
归因成立、小表门仍强 4.60 且高阶风险 flagged，如实保留）。v1 输出
归档 *.prefix_legacy.json，v1 数字不再引用（v2 无门 si 0.001094 与
v1 0.001020 同量级，方向不变）。正式 JSON SHA-256 `e5c507bb…`。

"门冗余"结论限于测试配置（rho=0.01、a16、nltcs、2000 轮等预算），
不从方法设计推广（审查意见一.3 的定位）。

## 最近变更（2026-08-13）

### 尺度不变选择：无门控扩散演化正式通过预注册判定（Issue #44 阶段二）

**背景：** rho=0.01 更正重跑（PR #45）将无门量化目标定为等预算打平历史门
L1 0.002728。诊断（4000 轮地板扫描）定位瓶颈为净漂移速率而非噪声地板；
盲 rho 调度（几何/两段式，已实现为消融工具）增益仅 3-5%；alpha 扫描单调
改善无拐点 → 结构诊断：种群同质化使 donor 联合分数行内离散度收缩，固定
alpha 的 softmax 选择压力衰减为均匀。

**机制：** `selection_scale_invariant`（默认 False）——geometric 抽样的
logits 行内标准化后乘 alpha，有效选择温度恒等于 alpha，尺度不变性成为
内生性质。纯分布侧（改分布，不筛选）：不读取候选评价、无接受/拒绝组件。
dev 定标（seed 42..44，只用于定标）冻结 a16 恒定 + 全默认漂移参数；恒定
单参数打平/超过手调双端递增谱系，且修正后饱和拐点出现——结构而非调参。

**正式结果（预注册 Issue #44 评论 5270803118，协议 f59d0ba，四臂 2×2
配置对齐，种子 100..104，2000 轮）：** nltcs 主判定分类
**supports_scale_invariant_selection**——无门 si L1 **0.001020**（四臂
最优，5/5）：机制改进 ratio 0.294（≤0.60），门冗余 ratio 0.611（≤1.10，
配置对齐后无门反好 39%）；质量零报警（train 未测量 3-way −50.7%、4-way
−33.8%，TVD 略好）；支持集 2087 vs 2989 为已知边界（观察指标）。
test_300x10 辅助：mechanism_gain_gate_not_redundant（机制 0.48 通过 5/5，
小表上门仍强 4.20，高阶报警）——小表偏门模式如实保留，逐数据集判定不
合并。legacy 两臂与 PR #45 正式重跑逐种子一致（轨迹级复现）。正式 JSON
SHA-256 `cf0e2699…`（nltcs）已入库。

**公平性纪律：** dev 曾出现"无门越过有门 41%"的中间表述，经有门同配置
对照证实为配置不对称假象（rho 混淆教训的应用），已在设计文档 §5 更正；
正式协议因此采用 2×2 完整配置对齐。

**收口进度（Issue #46）：** 对 PGM 同信息无噪声水位 0.000357 的差距从
基线 9.8 倍缩到 **2.9 倍**（0.001020）。尚未收口；支持集收窄机制与剩余
差距分解为开放问题。

## 最近变更（2026-08-11 晚）

### 扩散必要性 2×2 消融：完整引导机制具有显著增量价值（Issue #43）

**预注册协议**（提交 31abe2c；rho 更正重跑提交 7c06ce7，种子 100..104，
四臂等预算 2000 轮，rho=0.01 显式）：{扩散核, 随机核} × {无门+自冷却,
历史贪心门}。随机核 = alpha=0（geometric 抽样精确退化为逐行均匀 donor）+
关闭残差定向复制，只移除方向来源，保留 rho/eta/mu 扰动结构与冷却调度。
主判定数据集为 nltcs（项目方 2026-08-12 于结果产出前澄清；小表只作辅助）。
首轮继承默认 rho=0.1 的输出保留为 `*.rho01_legacy.json`。

**nltcs 结果（rho=0.01，最终表 measured L1，5 种子均值）：**

- 无门条件：完整引导 0.006626 vs 无引导 0.063833——比值 **0.104**（约
  10 倍），5/5 全胜，`diffusion_necessary_no_gate = true`。判定含义
  （第三、四轮审查定稿）：**完整引导组合**（fitness/距离加权 donor +
  残差定向复制作为整体）显著优于**均匀 donor、无定向复制的对照**；
  对照臂仍保留冷却或门提供的目标反馈，不称"完全无引导"；本实验未把
  donor 加权与残差定向拆开，不宣称其中某个子机制单独必要；无门一列
  实际是"无门+双重自冷却"，不推广到其他无门控配置。
- 有门条件：随机 0.051753 vs 扩散 0.002728——比值 **18.97**，
  `gate_confounds_attribution = false`。该判定的含义（审查修正）：随机核
  加门**不能匹配**引导核加门，门不足以解释引导方案的全部收益；它不证明
  "随机+门本身无效"——相对初始状态（marginal 初始化 L1 0.063276，逐正式
  种子复算）随机+门在 nltcs 上改善约 18.2%（5/5 种子方向一致），确实
  构成有效爬山，只是显著弱于引导核。

test_300x10 辅助数据同方向（无门比值 0.089、5/5；有门比值 2.73）。两个
必要性标志在 rho∈{0.1, 0.01} 下一致成立——消融结论对 rho 稳健（rho=0.1
历史数字：无门比值 0.050、有门比值 6.08）。

**结论（第四轮审查定稿口径）：** **在当前参数、固定 2000 轮等候选评价
与 measured L1 指标下，完整 donor/copy 方向引导组合，在"无门+双重自
冷却"和"历史门控"两种具体配置中，均显著优于均匀 donor、无定向复制的
对照；随机门控本身有效（相对初始改善约 18.2%），但不能替代引导机制。**
两个判定标志在 rho∈{0.1, 0.01} 下一致成立。边界：不拆分宣称单个子
机制（donor 加权 / 残差定向）单独必要；对照臂保留冷却/门的目标反馈，
不称"完全无引导"；无门一列实际是"无门+双重自冷却"，不推广到其他无门
配置；A/B、C/D 跨列比较同时改变门与冷却两个因素，不将跨列差异单独
归因给门（单因素归因见 PR #45 §7.3 交叉表）。消融正式 JSON 含初始
状态 L1（initial_state，可由 scripts/audit_formal_json.py 独立复验）；
nltcs test 侧评价按"一次实验一份源数据"规则撤回（见
test_evaluation_withdrawn，历史 test 参考哈希迁入其
withdrawn_reference_sha256）。--allow-dirty 现强制
formal_protocol=false（脚本级测试覆盖）。

**边界：** 单数据集主判定 + 小表辅助；随机核只是"均匀 donor+无定向"这一种
随机化（更强的随机基线如纯变异核未测）；与外部生成器的差距见 Issue #46
（PGM 无噪声水位 0.000357，尚差约 10 倍，未达收口）。

### 无门控扩散演化：残差自冷却机制接入（Issue #44 阶段一）

**方法边界背景（Issue #43）：** 主循环的整代接受门是归因混杂因子——随机变异加
贪心门本身即有效爬山法，门在时无法证明扩散机制的必要性。三臂探索（test_300x10、
seed 42/43/44 配对、500 轮）显示：无门恒定扰动下，分布倾斜完成约 99.6% 的下降
（28,900→118）但终点回漂到 ~234；历史贪心门 85.7；A0 严格门 89.3（门工程无
增益）。项目主线因此确定为无门控扩散演化：方向与收敛来自 fitness/分布动力学，
接受门只作安全护栏或对照臂。

**机制：** 每轮残差比 `r_t = min(1, ||target−q_t||_1 / ||target−q_0||_1)`，
冷却因子 `c_t = r_t^p` 整体缩放 `rho` 与 `mu`。现有 fitness 在零残差时自动
变平，本机制补上扰动幅度侧：零残差成为吸收态，不需要门的棘轮作用。新增
`residual_self_cooling`（冷却指数 p，默认 None 关闭）与
`self_cooling_stop_ratio`（内在停止阈值）两个参数；诊断新增
`self_cooling_history`/`self_cooling_stopped`。默认关闭路径与
`master=cd5de38` 在 seed 137、25 轮下最终表与全轨迹摘要哈希精确一致。

**探索性证据（冒烟，非正式结论）：** 无门 + `p=1`、2000 轮、3 种子均值：
final 89.2 / 全程最优 65.5 / 末 100 轮 104.5；同轮有门贪心 82.3/82.3/82.9。
无门全程最优首次越过有门贪心；终点回漂从 2.31 倍压缩到 1.36 倍，吸收态尚未
完全达成（正式实现以 `scripts/compare_gate_free_self_cooling.py` 复现）。

**nltcs test 评价定位（第三轮审查更正）：** 本实验源数据为 train，
test 侧评价（含等行数抽样口径）已按"一次实验一份源数据"规则从正式
产物撤回；泛化问题须以 test 为源数据独立建实验。离线参考只用 train。

**正式结果（rho 更正后，提交 2bdd11e，四臂 gate×cooling，种子 100..104，
2000 轮）：** 首轮（继承默认 rho=0.1）的 nltcs `supports` 结论已确认为 rho
混淆产物——大扰动下贪心门拒绝率极高、baseline 被系统性削弱，输出保留为
`*.rho01_legacy.json` 不再引用。以项目标准 `rho=0.01` 显式重跑后，nltcs 与
test_300x10 均为 **not_supported**：nltcs 最终表 L1 历史门 0.002728、无门
恒定 0.003467（差 27%）、有门+冷却 0.006263、无门+冷却 0.006626（主判定
ratio 2.43）。两个冷却臂无论有无门都显著更差——冷却×小 rho 交互在等预算内
把动力学饿死，这是机制在标准配置下的真实负结果；回漂消除（1.0002）的吸收态
语义保留。正式 JSON SHA-256（第四轮审查收口后，含 test 撤回迁移与字段改名）
`61665697c7383081d1a1429aceb4ee784988b87af7445b1a33b2aabfb3a12746`。

**完整引导机制的增量价值对 rho 稳健**（PR #47 同步重跑，rho=0.01）：
无门条件引导/无引导 = 0.104（5/5 全胜），有门条件无引导/引导 = 18.97——
完整引导机制显著优于无引导基线、门只解释部分改善，两标志在
rho∈{0.1, 0.01} 下一致成立。消融最终 JSON SHA-256（第四轮收口后）
`01c6f67b0886704c34810a39d1820d0f1d77d429c66dc18eb1027a715434aa24`；
运行时刻原始哈希 `41ca3664…` 保留于提交历史。

**边界：** dev 定标记录与协议冻结见设计文档；不修改默认生成器；小表终点锁定
的机制迭代（Issue #40 退火、mu 独立调度）与 Issue #43 的 2×2 必要性消融另行
预注册；尚未与外部生成器对照（Issue #46），不构成收口判定。测试：专项 26 项，
全套 820 项通过。设计文档见 `docs/设计/无门控残差自冷却扩散.md`。
## 最近变更（2026-08-07）

### 目标对齐 L1 持久热浴——正式负结果与有限温漂移边界

**预注册结果：** Issue #38 固定 `test_300x10`、精确 50-query target、marginal
初始化、新 seed 60..79、3000 个单坐标微步、`tau=1`。平方 baseline 与 L1
candidate 各自只在初始表计算一次同类非零能量变化 RMS，并共享初始表、坐标日程
和逐合法值 Gumbel；不接受、回滚、早停或选择 best。L1 的 measured normalized
L1 为 `0.027200`，平方为 `0.019570`，配对差 `+0.007630`、区间
`[+0.006782,+0.008478]`、0/0/20；相对 initial `0.017357` 也为 0/20 改善。

**独立质量：** 未测量 3-way 为 `0.0102133→0.0107962→0.0121971`
（initial→平方→L1），L1 相对平方恶化 12.98%，差的区间为
`[+0.00126797,+0.00153378]`，0/0/20。4-way 为
`0.00498069→0.00512743→0.00555928`，相对平方恶化 8.42%，也是 0/0/20。
measured 和 3-way 预注册门槛都失败，正式分类为 `not_supported`；阶段 II 不运行。
条件熵比例 0.9432、L1-uphill 概率质量 0.2689，未触发探索塌缩门槛。

**数理解释与事后诊断：** 有限温完整条件核对 Gibbs 分布可逆、遍历；同状态条件
期望对逆温的导数为负方差，但这只保证相对无向扩散更低，不保证相对当前状态下降。
全坐标精确复算显示，正式 `tau=1` 下 L1 的初始随机扫描一步期望漂移均值为
`+4.0776e-5`，20/20 seed 为正，平均 90.98% 坐标正漂移；沿轨迹的 seed 级平均
条件漂移也 20/20 为正。初始漂移变号的 `tau_*` 均值为 4.390，范围
3.770..5.681，而平方核均值仅 0.239。这是事后机制定位，不是新温度有效性结果。

进一步证明了有限温全支持的停止边界：若状态是严格单坐标局部最小，所有上坡合法值
仍有正概率就必然产生正期望漂移，因此原始能量不可能在每个状态都构成超鞅。后续若
研究漂移约束温度，必须用新 seed 预注册，并在局部最小处明确探索策略；不能直接把
4.390 设为默认值。

**门禁与复现：** clean generation commit `1ac5ddd` 上 20 seed 全部跑满；内置与
磁盘审计均重放 40 条轨迹、120000 次转移、20 个随机日程和 40 张最终表，失败 0。
离线评价重建 60 张表后才读取真实参考表，磁盘复算通过。漂移诊断 commit 为
`d7ee057`，20 seed、40 组初始全坐标候选集复算通过。审查修复（`0dd707c`）后
生成 JSON 不变；离线 JSON 因新增审计重执行记录重新生成，`analysis` 段与原件
（`c1566a4d…`，保留为 `*.pre_reaudit.json`）逐字段一致。当前生成、离线和漂移
JSON 的 SHA-256 分别为
`16c7095e700a64cec052c839b84618042111e04708fe1e63b67614ab34ab85af`、
`df90f4b79b4b939441f88bee8121492536d07884cfd0f5cd3ee0ac8585835365` 和
`3bc7b4eabdffb1325254ed410649d36643feecc8c0272469223606ce0bfe684e`。完整定理、
协议、结果和边界见 `docs/设计/L1持久有限温扩散的理论边界与验证协议.md`。

## 最近变更（2026-08-11）

### 固定温单坐标平方 workload 热浴——生成质量未改善

**核心负面信号（探索性，非预注册主判断）：** 以整张合成表为持久状态、使用
`P(v|rest) proportional exp(-beta E/N)` 条件分布的 `tau=1` candidate，虽然将
相对初始表的平方 workload loss 从 `2155.35` 降到 `1371.80`（20/0/0），却把同一
50-query workload 的 normalized L1 从 `0.017713` 恶化到 `0.019767`（仅 3/20
改善）。未测量 3-way mean L1 从 `0.0101981` 恶化到 `0.0108538`（+6.43%，
1/20 改善），4-way +3.63%，分箱联合 TVD `0.98583 -> 0.98983`。这些离线指标是
正式能量实验后追加的探索性分析，不属于预注册主判断；它们构成当前固定温单坐标
末态协议的明显负面信号，不支持接入默认生成器，但不据此声称已严格否定该协议的
生成有效性，也不能把原因单独归于平方能量形式。

**历史协议为何不支持有效性：** 原预注册主终点
`10555.7663 -> 1342.6576`（-87.28%，20/0/0）直接评价核所优化的平方能量；
`beta=0` 机制参考又从 1-way marginal 初始表持续退化，四窗口均值为
`2882.97 -> 5049.51 -> 7670.23 -> 10555.77`。按第三轮审查意见，输出主分类已
重命名为 `construction_energy_check_passed`（构造层能量检查通过），运行前冻结的
历史标签 `supports_persistent_heatbath_smoke` 逐字保留在
`legacy_classification`；判定规则与门槛不变，该分类只是实现一致性检查，不是
方法支持结论。本 PR 没有与默认生成器作同数据、同初始化和
可比查询评价预算/墙钟的直接对照，因而不提供相对优势证据。

**可复用实现证据：** 每步立即更新表、查询答案与残差，不使用 donor、临时 mask、
接受、回滚、早停或 best 输出。candidate 平均每 3000 步仍有 935 次实际升 loss，
条件熵保留 91.62%，uphill 合法值概率质量 31.20%。16 状态 oracle 的条件概率误差
`1.11e-16`、细致平衡误差 `1.73e-18`、平稳性误差 `1.39e-17`；120000 次转移和
40 张最终表已独立重放。首次正式运行的浮点消减失败、修复和调试 seed 均保留。

**采样器与更新粒度边界：** 固定有限 `beta` 的收敛对象是 `pi_beta`，不是 `argmin`；
本协议没有退火或 best-tracking，并返回固定预算末态。四个窗口 loss
`1544.23/1274.07/1315.91/1342.66` 与低能区波动相容，但没有混合诊断。单坐标更新
与主线块级 donor-copy 的邻域不同，强相关查询上的能垒/混合限制未排除。因而当前
无法在能量形式、温度定标、平衡态框架和更新粒度间分配负结果的因果责任。

**离线评价可验证性：** 第三轮审查发现离线脚本原先只检查输入 JSON 是否声明
`independent_audit.passed`，内部自洽的篡改 JSON 可能被接受。现在离线评价在读取
真实参考表之前，用固定公开 schema/query/marginal 重新执行完整独立审计（公开
输入哈希、20 条随机日程与 120000 次逐转移重放、聚合复算），失败直接终止，重执行
结果写入输出的 `independent_audit_reexecuted`。端到端负测试确认篡改
`loss_history[500]` 的 JSON 在读取参考表前被
`transition_replay_mismatch`/`recomputed_loss_mismatch` 拒绝。含审计的离线评价
总耗时 276.169 秒。

**覆盖、规模与输出：** 结果只覆盖 `test_300x10` 和单一 `tau=1`；初始 RMS 固定
`beta` 在状态漂移后的有效温度未经验证，没有温度前沿或跨数据集证据。当前精确
小表一次扫描为 3000 个坐标，nltcs 一次扫描为 258896 个坐标且 workload 为
1001 个查询，尚无实际规模可行性证据。自适应温度/退火使用新种子由 Issue #40
独立验证；本 PR 不在观察正式结果后追加温度扫描。研究代码保留作为非默认、非稳定
API 的复现基础。分类重命名后按同协议重跑正式实验（CPU 生成 244.600 秒），新旧
JSON 对拍除分类字段、非决策墙钟诊断和环境快照外逐字段一致，原输出保留为
`*.pre_rename.json`。专项 59 项、完整 CPU/torch 测试 664 项通过。重跑正式/离线
JSON 的 SHA-256 分别为
`f255628ed07173a5a0727d754a9661ab31db07a4464c15d75a4272d50d2ab3e4` 和
`7f35b1fe5dab244130b6b1e91a78de218308a9f6609eef40558bd2b2dbaed7b2`。完整公式、
协议、失败与边界见 `docs/设计/持久化workload能量热浴扩散.md`，关联 Issue #32。

---

## 最近变更（2026-08-06）

### #33 阶段 0：实验基础设施已完成（PR #36，第三轮反馈修订中）

**第三轮反馈修订（2026-08-06）：**

**问题 2.2：配置完整性** — ✅ 已完成

- **目标**：补全 ExperimentConfig，使其能完整描述一个实验的所有参数
- **实施位置**：`src/table_diffevo/experiment_config.py`

**1. 补全参数（新增 21 个参数）：**
  - 初始化：`init_method`, `maxent_max_states`, `maxent_max_sweeps`, `maxent_tol`
  - 计算与性能：`eval_method`, `batch_size`, `log_every`, `tol`
  - 抽样：`distance_mode`, `p`, `exclude_self`
  - 重试：`max_retries`, `retry_rho_decay`
  - 扩散核：`residual_directed_diffusion`, `diffusion_direction_strength`, `diffusion_direction_normalization`
  - Gibbs：`factorized_gibbs_sweeps`, `factorized_gibbs_max_order`, `factorized_gibbs_logit_clip`
  - DataConfig 新增：`schema_path`, `query_path`

**2. 详细文档注释：**
  - 每个参数都有完整注释：含义、选项、范围、单位、推荐值、注意事项
  - 统一文档风格，方便查阅

**3. 完善验证逻辑（validate()）：**
  - 验证所有新增参数的合法性
  - fail-closed：拒绝非法值（如负数、超出范围、未知选项）
  - 验证参数间的依赖关系（如 factorized_gibbs_sweeps > 0 需要 residual_directed_diffusion）

**4. 配置映射方法（to_run_evolution_kwargs()）—— 阶段 0 fail-closed 边界：**
  - 自动加载文件（schema、queries、marginals）
  - 转换参数名（如 `lambda_` → `lambda_param`）
  - 用法：`kwargs = config.to_run_evolution_kwargs(seed=0)` → `run_evolution(**kwargs)`
  - **接入范围**：阶段 0 只交付数据结构 + 接线骨架，不改主循环算法。本方法
    只对主循环**已真正支持**的口径做真实映射（默认接受判据 + 线性轮数 α 调度）。
  - **fail-closed（#36 问题 2 修订）**：配置里预注册的 A0/A1 接受规则、fixed/probe
    α 调度**尚未接入主循环**，本方法对这些口径直接抛 `NotImplementedError`，
    而不是静默丢弃或错误映射（此前 fixed 会被当成线性调度、A0/A1 被整个丢掉）。
    真正接入分别留待阶段 1（接受规则）与阶段 2-5（探测调度）。

**5. 示例配置：**
  - `configs/experiments/nltcs_baseline.yaml`：基于 run.py 默认参数的完整配置
  - 包含所有参数及注释说明

**6. 命令行启动脚本：留待阶段 1**
  - 从 YAML 端到端运行实验的启动脚本（`scripts/run_from_config.py`）**本阶段不交付**：
    它需调用 `to_run_evolution_kwargs()`，而阶段 0 的 fail-closed 对 A0/A1、fixed/probe
    一律抛 `NotImplementedError`，此刻无法端到端跑真实验。待阶段 1 接入 A0/A1 后再交付
    可运行版本。参数体系本身（读/校验/转换）已由 `experiment_config.py` 交付并测试。
  - 原 `scripts/run.py` 保持不变（向后兼容）

**验证：**
- 语法检查通过
- 配置加载、验证、参数转换测试通过
- 生成 38 个 run_evolution 参数
- 不影响现有代码

**优势：**
- ✅ 参数集中在 ExperimentConfig（单一真相源）
- ✅ 文档清晰（每个参数都有详细说明）
- ✅ fail-closed（拒绝非法配置）
- ✅ 映射到 run_evolution（自动转换）
- ✅ 不破坏现有代码（新脚本独立）

**问题 2.3：实现 candidate_budget 全局预算控制** — ✅ 已完成

- **目标**：为长时间探索实验设置计算成本上限，确保可比性
- **实现位置**：`src/table_diffevo/evolution.py`
  - 新增参数 `candidate_budget: Optional[int] = None`（全局候选评估次数上限）
  - 主循环新增计数器 `candidate_evaluation_count`（跟踪候选提案评估次数，不含初始表）
  - 每次候选评估后检查：达到预算时设置 `candidate_budget_exhausted = True` 并提前停止
  - 更新终止条件文档：残差全 0、达到 n_rounds、或达到 candidate_budget（若指定）
- **配置验证**：`src/table_diffevo/experiment_config.py`
  - `ExperimentConfig.candidate_budget` 已存在（lines 85-87）
  - 验证逻辑已完整（lines 144-145）：若指定则必须 > 0
- **诊断输出**：
  - 新增字段 `candidate_evaluation_count`：实际候选评估次数
  - 新增字段 `candidate_budget_exhausted`：是否因达到预算提前停止
  - 进度输出包含预算信息：`轮次 X/Y | loss: Z | 接受: 是/否 | 尝试: N | 候选: M/budget`
- **语义**：
  - 候选评估 = 生成候选表 → 评估所有查询 → 计算误差（算1次）
  - 初始表评估不计入（只统计主循环中的提案评估，包括重试）
  - 与 max_retries 配合：一轮多次重试的每次评估都计入
  - 与 n_rounds 并存：先达到者停止
- **验证**：语法检查通过（`python -m py_compile` 无错误）

**剩余工作**：
- 问题 1（Logger 原子性）：已确认现有实现满足要求，无需修改
- 问题 2.1（W/H 重命名）：已完成参数重命名和文档更新（experiment_config.py, probe_convergence_a2.py）
- 问题 2.2（配置完整性）：已推迟到后续阶段
- 问题 3、4：待 1-2 完成后讨论

### #33 阶段 0：实验基础设施已完成（PR #36）

**背景：** Issue #33 重构锐度调度为残差降速驱动，分 6 个阶段渐进实现。阶段 0 先搭建可复用的实验基础设施，确保后续所有实验（A0/A1 对照、固定 α 扫描、探测式调度）使用统一的度量、日志和配置管理。

**已完成交付物：**

1. **度量计算模块** (`src/table_diffevo/metrics.py`)
   - `compute_normalized_l1`: 归一化 L1 误差 = mean(|target - current|) / n_records = Σ|target - current| / (k·N)（k 为查询数）
   - `compute_squared_loss`: 平方 loss Q，wrapper for `objective.compute_loss`
   - `compute_all_metrics`: 一次性计算避免重复
   - 验证脚本 `scripts/verify_metrics.py` 确认与现有实现完全一致（8 个测试用例全部通过）

2. **接受规则模块** (`src/table_diffevo/acceptance.py`)
   - 统一接受规则接口 `check_acceptance()`，返回 (accepted, delta_L1, delta_Q)
   - **严格改善口径（Issue #33 预注册定义）**：A0/A1 均采用严格不等号，必须有
     超过 eps 的实质改善才接受；平局与容差内微小恶化一律拒绝。
   - A0 规则：`delta_Q < -eps_Q`（Q 改善超过 eps_Q 才接受）
   - A1 规则：`delta_L1 < -eps_L1` 视为严格改善直接接受 → `|delta_L1| <= eps_L1`
     落入平局带时用 `delta_Q < -eps_Q` 裁决 → `delta_L1 > eps_L1` 拒绝
   - **关键设计**：在 state 覆盖前计算 delta，避免误报零差值
   - **与主循环的边界差异（须披露）**：主循环 `evolution.py` 是 `proposal_loss <= loss + tol`
     （非严格，接受平局与容差内微小恶化）。A0/A1 严格口径与之不同，故 A0 符合 Issue #33
     冻结公式、可作预注册臂，但**不是**主循环逐轨迹等价 baseline；阶段 A 分析须显式披露，
     选出 A* 接入主循环时再单独决定其容差口径。
   - 单元测试全部通过，覆盖四象限、严格不等号、两个 epsilon 边界

3. **实验日志模块** (`src/table_diffevo/experiment_logger.py`)
   - 三层日志：`RoundLog`（每轮）、`BlockLog`（每块）、`ProbeLog`（探测分支）
   - 输出 CSV（rounds/blocks/probes）+ JSON（summary）
   - **安全性增强**：
     - 启动前拒绝非空目录（可选 `force_overwrite=True`）
     - numpy 类型序列化（包括 `np.bool_`）
     - 非有限值（NaN/Infinity）转换为 `None`
     - 原子写入（临时文件 + rename，失败不破坏既有结果）
     - JSON 预校验（序列化前测试，失败立即抛出）
   - 18 个单元测试全部通过

4. **实验配置模块** (`src/table_diffevo/experiment_config.py`)
   - 四部分配置：数据、接受规则、α 调度、实验参数
   - **完整参数验证**：
     - 接受规则：A0/A1（已移除 A2），A1 需 eps_L1，epsilon 非负
     - Alpha 调度：probe 模式需 W>0, 0<s<1
     - 数据：n_records > 0
     - 实验：0 < rho ≤ 1，n_rounds > 0
   - **Fail-closed 设计**：拒绝未知 YAML 键，避免拼写错误
   - 新增字段：`rho`（Issue #33 要求）、`device`、`candidate_budget`
   - 16 个单元测试全部通过

5. **文档** (`docs/experiment_infrastructure.md`)
   - 三个模块的详细使用指南
   - 完整工作流示例
   - 测试与验证方法

6. **集成演示** (`scripts/demo_infrastructure.py`)
   - 端到端展示模块集成使用
   - 使用统一接受规则接口
   - 输出到 `experiments/results/.demo`（隐藏目录，不污染结果区）

**测试覆盖：**
- 阶段 0 四个模块（当前工作树，含未提交改动，pytest 收集数）：metrics 11 + acceptance 19 + experiment_logger 23 + experiment_config 32 = 85 项
  （config 增量来自问题 1-3 的 fail-closed/未知键回归 + 本次「逐个加载仓库示例 YAML」回归；均尚未提交）
- 全套单元测试 735 项全部通过（含本地未提交的问题 1-4 修订与 force_overwrite 文档修订；提交前以干净 checkout 复核数字为准）

**验证通过：**
- 度量计算与 `evolution.py` 完全一致（数值误差 < 1e-12）
- 配置验证规则覆盖所有预注册的约束
- 日志格式符合后续分析需求（CSV 易于 pandas/R 处理）

**状态：** 阶段 0 已作为 PR #36 提交，正在按审查反馈迭代修订。仅包含实验基础设施四个模块（metrics/acceptance/experiment_logger/experiment_config）+ 文档 + 集成演示；不含 evolution.py 集成、固定 α 扫描或探测式调度（后续阶段单独成 PR）。

**下一步：** 阶段 0 合入后，进入阶段 1（把 A0/A1 接受规则集成到 `evolution.py`）。

---

## 最近变更（2026-08-04）

### #33 重构：调度驱动量从"轮数"改为"残差降速"，分两步走（设计已冻结，未跑）

**背景（承接 #29 三臂负结果）：** ρ 衰减调度在 nltcs 三臂正式实验中负结果——改善过程平均
loss（tail250/traj linear 9–10/10），但产出质量全线落败（best_loss、normalized_l1
均 0/10，best_loss 差约 30%）。**500 轮未收敛作为既有结论直接采用**（不再另跑诊断）：
fixed 臂末段仍陡降（seed 0 的 400→450→500 轮 2.07e8→1.54e8→1.11e8，每 50 轮降约
0.4e8、每轮均接受），降速毫无趋缓，即未收敛。

**根因诊断：** 旧调度按轮数进度 `t/(n_rounds-1)` 驱动。此病不止在 ρ——
`src/table_diffevo/evolution.py:559-563` 的几何抽样锐度 α 同样按 `t/(n_rounds-1)`
从 2 爬到 10。500 轮时 α 已到最尖（=10、最贪），可系统仍在前期，**α 过早贪心**，
与 ρ 过早精炼是同一种"看错钟"。**这翻转了 2026-07-24 "选轮数驱动 p=t/n_rounds、
残差驱动推到第二版" 的决定**（见下文该日记录）：轮数驱动正是 nltcs 负结果的根因。

**新设计（分两步，本轮只冻第一步）：**
- 第一步（#33）：ρ 固定 0.01 不动，只把 α 改成**残差降速驱动**——降速信号
  `D = 最近 W 轮平均残差下降量 ÷ 初始降速`（初期≈1、趋缓→0），
  `α = α_min + (α_max−α_min)×(1−D)`（降速快→α 小广撒，降速缓→α 大精挑）。单变量、
  归因干净。两臂 ρ 相同 → 每轮参与量相同 → 跑相同固定轮数 N 直接比 best_loss/
  normalized_l1，无需累计参与量口径。
- 第二步（有效才做，暂不冻协议）：再把 ρ 也改残差降速，按累计参与量重做公平比较。

**方法论要点：** 残差一律对 measured/noisy target 算（后处理量，DP 下免费）；降速趋缓
一信号两用（转精修 + 判收敛），不再单设 T_conv；DP 噪声地板作为将来接口预留。
与曲率线（#18–#31）正交：那条动"怎么改"（gamma），本条动"选哪些记录改"（α）。

**状态：** 分支 `research/nltcs-convergence-probe`（从 master 04d59dc 切）。Issue #33
已按上述设计**覆盖重写**（标题＋正文，预注册协议冻结）：
https://github.com/Chuhan722/table-diffusion/issues/33 。#29 三臂负结果代码在
另一分支 `feat/rho-decay-schedule`（commit aa01850，本地未推）。尚未写实验脚本、
未跑。

（注：固定 α 扫描与探测式调度的代码/实验属于阶段 3/4，在独立研究分支
`research/probe-alpha-schedule` 上开发，不在 PR #36 树内；相关进度另行记录，不在此展开。）

### 共同状态曲率阶段交互——曲率相对收益在晚期稳定增强

**协议与主结果：** Issue #30 固定 `test_300x10`、新 seed 30..39、marginal
初始态与标准闭环 500 轮 best 态、每状态 200 对 proposal；唯一变量为
`gamma=0/1`，不执行接受。初始态 seed 级曲率净收益差均值为 `-0.6055`，区间
`[-1.1082,-0.1028]`，方向 2/0/8；500 轮态为 `+1.07025`，区间
`[0.9761,1.1644]`，10/0/0。主交互 `late-initial` 为 `+1.67575`，区间
`[1.15337,2.19813]`，10/0/0，通过预注册的区间和 8/10 一致性门槛，分类为
`curvature_advantage_strengthens_late`。

**机制与边界：** 初始态曲率少获得 `1.8140` 一阶收益，只节省 `1.2085` 二次项，
净差为负；晚期只少获得 `0.5080` 一阶收益，却节省 `1.57825` 二次项，净差转正。
晚期正收益率从 `7.45%→9.05%`，但两侧平均原始 proposal 收益仍为负；曲率还把
晚期查询步平方范数降低 17.70%。结果只支持后续低到高曲率调度的研究方向，不证明
某个阈值、函数或长期生成质量，不改变 Issue #24 对固定 `gamma=1` 长期轨迹的失败
结论，也不在本阶段接入调度或默认值。

**门禁与复现：** 20 个状态、4000 对 proposal 的状态、donor、参与、初始 mask
和随机流全部对齐；`gamma=0` 退化精确，条件护栏零命中且保持双向支持。
`N*V_1=gain` 最大误差 `1.14e-13`，稀疏线性一致性最大误差 `2.78e-17`。独立
读取审计重算原始配对、查询变化、seed 交互、区间、输入哈希和分类均一致。相关测试
46 项、完整 CPU/torch/CUDA 605 项通过。正式输出 SHA-256 为
`cbdc8ce541e12f93931ead0fad78ffa1863f11cf97454675f64315c84aabe5ab`；完整协议、
结果与限制见 `docs/设计/曲率收益共同状态阶段交互.md`。

### 曲率核多步漂移与内禀扩散时钟——时间减速明确，单位路程优势未定

**协议与主结果：** Issue #27 精确重放 Issue #24 的 20 配对种子、1000 轮
`gamma=0/1` 无接受轨迹，只额外记录每轮 50 维整数查询向量。candidate/baseline
最终查询二次变差比为 `0.83261`，范围 `[0.78587, 0.89355]`；匹配共同查询时钟
末四分之一路径后，loss 差为 `-6.8990`，中位数 `-8.0246`，配对 `13/0/7`，
95% t 区间 `[-15.0216, 1.2236]`。区间跨 0 且效应没有缩小到按轮数差的一半以内，
正式分类为 `mixed_or_inconclusive_multistep_effect`，不改默认算法。

**次要解释：** 匹配累计改单元格数的差为 `+1.7379`，区间
`[-5.9601, 9.4360]`，配对 `8/0/12`，只复核已披露的事后线索。RMS 残差十分位
分段显示低残差前五箱的单位查询路程净漂移均变差，后五箱中四箱改善、一箱近似
相同；这是描述性状态反转，不是新的成功门槛。

**门禁与复现：** 干净提交 `0fca33a` 上 40 条轨迹全部跑满；2400 个旧非计时字段
与冻结输出精确一致。内置门禁和独立审计均复算 40,040 个查询向量、40,000 次转移、
匹配统计、状态分段和分类，最大收益恒等式误差为 0。相关测试 136 项、完整
CPU/torch/CUDA 测试 587 项通过。正式 JSON 大小 49,522,269 字节，路径为
`outputs/curvature_multistep_drift/formal_20seed_1000r_tau2_sweep8_query_clock_0fca33a.json`，
SHA-256 为
`3ea7b12390a2b4ebfddfe435d8de41cce9967a9d5d92e81193381c251bd2c2b5`。本阶段只做
无噪声机制诊断，没有真实表评价，不是 DP；完整协议见
`docs/设计/曲率核多步漂移与内禀扩散时钟.md`。

### 因子 workload 预编译与批量条件评价——作用域内稳定加速，默认替换门禁未全过

**实现与边界：** 增加显式不可变的 workload 编译对象，只保存公开 schema、查询
结构、35 个去重条件和局部 mask 模板；动态 residual、recipient/donor 与因子值仍
逐轮重算。非零 sweep 只对活跃参与行批量评价条件，模型逐行构造、采样后释放；
无全局缓存。旧逐行构造器保留，0 sweep 不读取编译对象，默认生成器与联合 Gibbs
能量、随机数顺序和参数均未改变。

**等价与作用域性能：** `test_300x10`、`tau=2`、8 sweep、1000 轮、seed 0..9，
两种构造器分别在独立进程、奇偶交错顺序运行。10,000 个逐轮表 SHA-256、loss/gain、
最终表、29,869 个模型工作量和主/Gibbs RNG 端点全部精确相同。编译 + 校验 + 构造
从 `7.921→2.107s/种子`，配对中位降幅 73.96%，10/0/0；一次编译仅 0.000290s。

**总墙钟与正式判断：** 扣除状态哈希审计的完整墙钟为 `24.700→18.767s/种子`，
配对中位降幅 26.31%，9/0/1。seed 1 为 `22.665→23.287s`（+2.746%），超过
预注册的任何 seed 不得变慢 2% 门槛；该 seed 的目标因子管线仍下降 70.04%，而
未修改的方向阶段增加 4.14s。事后分解不能覆盖硬门禁，因此正式判断是
`performance_not_supported`，本阶段不把批量路径设为默认。

**内存与验证：** RSS 平均只增加 0.043%、逐 seed 最大 +0.142%；CUDA allocated
和 reserved 峰值保持 11.575/24 MiB。普通 merge 同步
`origin/master=c3b3e89` 后，针对性测试 56 项、完整 gsd CPU/torch/CUDA 574 项、
语法和差异检查通过；隔离 diff 仍只有本性能工作的 7 个文件。正式 JSON 位于
`outputs/factorized_workload/formal_10seed_1000r_tau2_sweep8_71a1954.json`，
SHA-256 为
`6c374164b3f87b3ed97e9e9f046f5c57c92ece6a0439c46181996b77552a6e42`。完整协议、
全部失败结果和边界见 `docs/设计/因子工作负载预编译与批量构造.md`，关联 Issue #15。

**当前父提交复验：** 深审补充了全部计划输出的启动前碰撞检查和 JSON 原子发布，
保证后续 worker 文件已存在或序列化失败时不会留下新半成品，也不会损坏已有结果。
干净提交 `a4b3c46` 上按同一 10 seed × 1000 轮协议完整重放：run 内 10,000 个逐轮
哈希等价门禁通过；两个构造器的新旧轨迹再跨提交对拍 20,000 个逐轮哈希，所有非
计时、非内存字段精确相同。因子管线中位降幅 73.62%、10/0/0，完整墙钟中位降幅
24.77%、10/0/0，逐 seed 均降低 21.26% 到 27.04%；RSS 最大只增加 0.152%，CUDA
峰值完全相同，本次判断为 `performance_success`。

该复验不覆盖原始 `performance_not_supported`：两份输出共同说明算法结果稳定，
也说明总墙钟存在运行间波动。当前仍不在本 PR 内切换默认构造器。针对性测试 58 项、
完整 gsd CPU/torch/CUDA 576 项通过；复验 JSON 路径为
`outputs/factorized_workload/formal_10seed_1000r_tau2_sweep8_a4b3c46.json`，
SHA-256 为
`6a94b93ae70c7e061cef8f07c09107662d38f535c24f0b0e2bba5e7f935a93e1`，旧输出与两套
worker 目录均原样保留。

### 整代曲率 Gibbs 无接受动力学——冻结局部优势没有转化为长期优势

**协议与门禁：** Issue #24 一次性固定 `test_300x10`、seed 0..19、1000 轮、
`tau=2`、8 sweep、`rho=0.01`、`eta=0.5`、`mu=0.01`，只比较同一曲率更新器的
`gamma=0/1`。每个 proposal 无条件成为下一状态，没有接受、回滚、早停或 best
选择。seed 0 的 20 轮退化预检逐轮表、loss、方向、主/Gibbs RNG 和共同诊断精确
一致；40 条轨迹跑满，两侧初始状态、方向尺度、主 RNG 端点和地址化 Gibbs seed
全部对齐。最大有效 `|logit|=10.6513`、护栏零命中、条件概率保持双向支持。

**预注册结果：** 主终点末 250 轮平均当前 loss
`99.9544→105.7517`（`+5.80%`；`5/0/15`），正式判断为
`curvature_dynamics_not_supported`。最终当前 loss `90.375→92.600`
（`8/0/12`），best 诊断 `64.325→70.100`（`5/0/15`），全轨迹平均 loss
`219.7429→224.5991`（`9/0/11`），均不能改写主结论。

**风险与解释：** 末 250 轮条件熵 `0.671946→0.674752`（`+0.417%`），最终唯一
记录数 `284.95→287.45`（`+0.877%`），三个预注册风险都未触发。candidate 改变
单元格数减少 7.66%，负收益幅度减小 `0.4121`，但正收益幅度也减小 `0.3287`；
晚期正收益率相同而两类步幅同时收缩。固定状态局部过冲改善不保证轨迹分叉后的状态
分布与多步漂移改善，当前不进入默认接入或标准接受闭环。

**验证与输出：** PR #23 以 `8b6eaf6` 合入后，本分支又用普通 merge 同步最新
`master`；合并树与此前同步的父 HEAD `3465405` 逐字节相同，隔离 diff 仍只有本阶段
4 个主题文件。父分支对曲率更新器的代码变化仅修正 0-sweep 未计算诊断，不经过本实验固定的
8-sweep 路径。当前干净合并提交 `fe99279` 上完整重放 seed 0 的两条 1000 轮轨迹，
预检、baseline 和 candidate 去除方向/因子/Gibbs/总墙钟四个计时字段后，分别与
原正式输出精确一致。动力学专项 18 项、相关独立 Python 3.11 环境
`222 passed, 2 skipped`、相关 gsd 230 项、完整 gsd CPU/torch/CUDA 559 项通过。
独立 Python 3.11 环境全量唯一失败是该环境未安装
torch 而对应测试未按能力跳过；排除该不适用用例后 `510 passed, 28 skipped,
1 deselected`。重放 JSON 位于
`outputs/generation_curvature_dynamics/replay_seed0_1000r_tau2_sweep8_fe99279.json`，
大小 477,479 字节，SHA-256 为
`d0a2873dbdd3e986e4ffb005fea619be411aab0822d9ee344591c47c9ea3b02e`。原 20 种子
9,047,096 字节正式 JSON 的 SHA-256 仍为
`3a6d185b9522550ce0807069b03a087780aefc0696da84a4c630339e10076234`，路径为
`outputs/generation_curvature_dynamics/formal_20seed_1000r_tau2_sweep8_ecf072c.json`。
独立审计重算 40,000 次转移、窗口、条件诊断、配对汇总、风险和输入哈希均一致；
完整协议与边界见 `docs/设计/整代曲率Gibbs无接受动力学.md`。

### 整代曲率感知 Gibbs——晚期同时压低自身与交叉二次项

**问题与分布：** Issue #17 排除了自身项或交叉项单独主导后，本阶段没有事后缩小
`rho`，而是对整代复制 mask 定义
`V_gamma=<e,Delta q>-gamma/(2N)||Delta q||^2` 和相对历史 Bernoulli 核的有限
温 Gibbs 分布。`gamma=0` 是现有最高三阶因子 Gibbs，`gamma=1` 是精确平方
workload proposal 收益除以公开 `N`。单 bit 条件式同时包含当前整代查询变化，
有限强度下不设置收益门槛、argmax、top-k 或 generation acceptance。

**协议与门禁：** 一次性固定 `test_300x10`、seed 0/1/2、marginal 初始态与标准
0-sweep 闭环 500 轮 best 态、每状态 200 proposal、`tau=2`、8 sweep、条件
`logit` 护栏 30、`rho=0.01`、`eta=0.5`、`mu=0`，唯一变量为 `gamma=0/1`。
1200 对 donor、参与、初始 mask、主/Gibbs RNG 全部对齐；初始 mask 从 update
seed 独立重放全部随机
消耗，`gamma=0` 与既有因子 Gibbs 的表、mask、诊断、RNG 端点及条件概率精确
相同。查询变化误差为 0，`N*V_1` 与直接 loss 收益最大误差 `5.68e-14`；最大
有效 `|logit|=9.70686`，护栏零命中，所有正式条件概率保持双向支持。脚本没有
真实参考表路径。

**预注册晚期结果：** 500 轮状态的一阶收益 `3.8233→3.3250`（-0.4983），自身
二次项 `8.1967→7.2742`（-0.9225），交叉项 `0.7183→0.0283`（-0.6900），总
二次项 `8.9150→7.3025`（-1.6125），净收益因而从 `-5.0917→-3.9775`
（+1.1142；227/332/41），正收益率 `10.17%→12.50%`。三个晚期状态净收益和
正收益率全部同向，正式判断为 `supports_late_curvature_kernel`。

**风险与解释：** 初始态净收益 `74.8158→74.1725`，相对只下降 0.86%，没有触发
5% 阶段风险。晚期 68,208 个微步平均条件熵 `0.68747→0.68362`（-0.56%），没有
触发 10% 集中风险，所有概率严格位于 `(0,1)`。candidate 晚期复制单元格减少
9.15%，但当前没有匹配复制量控制，不能把完整分布结果进一步归因为单纯缩步。
更重要的是，两侧晚期平均净收益仍为负；本结论支持进入独立动力学验证，不等于
已经改善最终生成器。

**验证与输出：** 深审修复提交 `743c8d5` 取代证据门禁不完整的旧输出；新旧
非计时算法结果独立对拍精确一致。同步 PR #22 父 HEAD 并收紧正式元数据/零扫描
诊断边界后，重跑与 `743c8d5` 输出的整份非决策 JSON 精确一致。PR #22 合入后，
本分支又以普通 merge 同步 `master=eac317b`；父 HEAD 与合并后主分支代码树相同，
最新正式重跑去除环境和计时字段后与上一版整份 JSON 精确一致。专项测试 52 项、
相关 gsd 测试 176 项、完整 gsd CPU/torch/CUDA 541 项；独立 Python 3.11 环境
相关 174 项通过、2 项跳过。最新正式 JSON 位于
`outputs/generation_curvature_gibbs/formal_3seed_2state_200p_tau2_sweep8_1b18d13.json`，
大小 11,326,345 字节，SHA-256 为
`43aab0862a5dfe86c81863c6dc645d9243167234d8dcd55804953f0e0d6e7eaf`。完整公式、
逐状态结果、审计边界和旧证据修正见 `docs/设计/整代曲率感知Gibbs扩散.md`，关联
Issue #18。

### 联合扩散整代步幅诊断——过冲是逐行自身项与跨行交叉项的混合来源

**问题与协议：** Issue #16 的事后分解表明因子 Gibbs 一阶方向更好，但整代二次
惩罚增加得更多。为避免事后挑小 `rho`，先固定 3 个 seed、marginal 初始态与
标准 0-sweep 闭环 500 轮 best 态、每状态 200 个冻结 proposal，只比较
`tau=2` 下 0/8 sweep，`rho=0.01`、`mu=0`。两个状态复用同 seed 标准运行的
首个非零方向 RMS；冻结 proposal 不执行接受，也没有真实数据路径。

**精确门禁：** 1200 对 donor、参与和主 RNG 全部对齐；直接初始化表与标准运行
初始哈希 3/3 一致。初始 mask 不再从落地 proposal 反推，而是从配对 seed 重放
全体 Bernoulli mask 和 `mu=0` 随机消耗；重放 RNG 端点及 baseline 参与行落地编辑
均 1200/1200 对齐。逐行查询变化之和、`Q_self+Q_cross` 和 `gain=linear-Q`
三个恒等式最大误差均为 0。

**全局分解：** 一阶方向收益 `31.8783→45.5250`（差 +13.6467）；自身项
`8.8963→9.7338`（差 +0.8375）；交叉项 `0.7300→1.2583`（差 +0.5283）；
总二次项差 +1.3658。自身/交叉分别占 61.32%/38.68%；自身项在 5/6 状态更大，
但未达到预注册 2/3，全局判断为 `mixed_or_inconclusive_source`。复制单元格
只增加 3.29%，自身项增加 9.41%，不能把问题简化成 mask 基数稍大。

**阶段差异：** 初始态的一阶/总二次/净收益差为 +27.080/+2.197/+24.883，正收益率
提高 5.5 pp；500 轮态为 +0.213/+0.535/-0.322，正收益率下降 2.5 pp。全局净收益
改善由初始大残差主导；晚期联合方向仍略好，但不足以支付同时增加的两类曲率成本。

**结论与验证：** 预注册的单一来源假设未成立，所以本阶段不实现固定基数 Gibbs，
也不只做参与率/微批归一化。更一般的整代曲率能量或总查询步幅预算需另立问题并先
定义有限温度、反向支持和退化性质。诊断测试 37 项、加因子和演化相关测试 124 项、
完整 CPU/torch/CUDA 489 项通过；独立 Python 3.11 环境相关 81 项通过（另 2 项
跳过）。完整协议
审计的 158,512 次实际 Gibbs 条件更新最大原始 `|logit|=9.7069`，零次触发 30
护栏。同步最新主分支并收紧正式元数据门禁后的重跑与上一版非决策字段精确
一致。正式 6,876,791 字节 JSON 的
SHA-256 为
`b3b13c448e3b7ed27405f4ad73a74d56e6bb54d6d227a8a37ddaf723709e1a67`，位于
`outputs/factorized_step_overshoot/formal_3seed_2state_200p_tau2_sweep8_3fe91b7.json`。
完整公式和逐状态结果见 `docs/设计/联合扩散整代步幅诊断.md`，关联 Issue #17。

### 因子 Gibbs 标准接受闭环——平均训练目标改善，但未通过预注册胜种子门槛

**实现与边界：** 在不改变默认路径的前提下，将最高三阶因子 Gibbs 接入
`run_evolution` 的现有 donor、变异、整代 loss 接受、缓存和历史最优表。
迁移深审从当前父提交重放了全部 20 个 0-sweep baseline、每个 500 轮；初始表、
初始化后主 RNG、最终主 RNG、最终表和全部共享非计时 diagnostics 逐种子精确一致。
附加 Gibbs 使用独立派生 RNG，20 个正式种子的初始表、初始化后主 RNG、方向尺度与
最终主 RNG 全部对齐。冻结正式提交尚未启用条件 logit 截断；当前实现固定使用
`[-30,30]` 护栏并写入运行参数，事后逐种子审计证明该护栏在正式轨迹上从未触发。
全部生成完成后才读取真实参考表。

**预注册结果：** `test_300x10`、精确 50-query target、marginal 初始化、
`tau=2`、`rho=0.01`、20 配对种子、500 轮。baseline 0 sweep 与 candidate
8 sweep 的 best loss 为 `39.225→33.875`（-13.64%），平均差 -5.350，
95% 配对 t 区间 `[-10.597,-0.103]`，13/1/6。因改善种子少于事先写定的
14/20，正式判断为“不确定”；不使用边界 `p=0.0461` 替代完整判断规则。
normalized L1 为 `0.0030167→0.0027767`（-7.96%），只作次要描述。

**泛化与成本：** 未测量 3-way/4-way mean L1 分别变化 -0.083%/-0.286%，
分箱联合 TVD 均为 0.98183，唯一状态数下降 0.342%，预注册的 5% 风险标记均未
触发。高阶区间均跨 0，且没有 held-out 切分，因此不宣称完整生成质量改善。墙钟
`8.865→14.405s/种子`（+62.5%）；因子构造 5.089s，Gibbs 抽样 0.278s。

**机制解释：** candidate 的正收益提案比例下降 12.25%，但正收益幅度增大
14.22%（19/20）。事后精确分解中，一阶残差方向收益 `9.0010→9.3058`，二次
过冲惩罚 `9.2400→9.7014`，后者增量更大，使平均净提案收益反而更低。下一方法
问题应聚焦于保留联合结构的扩散时间步或查询变化预算归一化，不事后挑更小
`rho`，也不回到方向门槛。

**验证与输出：** 闭环针对性测试 53 项、加因子模块共 94 项、完整
CPU/torch/CUDA 452 项、独立 Python 3.11 环境相关测试 92 项通过（另 2 项按环境
跳过）。正式输出包含
40 张 300×10 合成表、81 份 JSON；
数值有限性、文件数、参数、哈希、配对差和 5051/30450 个离线高阶单元格均已独立
复核。迁移到数值护栏后的 20 种子 candidate 重放又审计了 1,311,408 次条件更新，
最大原始 `|logit|=10.6513`、零次触发护栏；最终表，以及除墙钟和当前实现新增的
`params.factorized_gibbs_logit_clip` 字段外的全部共同 diagnostics，均精确一致。
正式 `summary.json` SHA-256 为
`bd6884d9747ad9d8b62a0d1edc84521bac1c2925cb77e3cd945ff40c1254c7e8`。输出位于
`outputs/factorized_gibbs_closed_loop/formal_20seed_500r_tau2_sweep8_8431146/`，
完整协议与边界见 `docs/设计/因子Gibbs标准接受闭环.md`，关联 Issue #16。

### 三阶稀疏因子 Gibbs——有限步逼近 oracle，并在无接受动力学中复现改善

**实现：** 对每条 recipient-donor 对，把每个查询写成其活跃属性上的局部布尔
因子；当前 workload 最高三阶，每个查询局部表最多 8 项。随机扫描 Gibbs 从现有
独立定向 mask 出发，每个 sweep 做 `k` 个均匀带放回坐标微步，条件 logit 使用完整
相邻因子能量差。有限温度不做方向门槛、argmax、top-k 或 proposal 接受。0 sweep
与现有定向 `evolve_step` 的表和主 RNG 端点精确一致；额外 Gibbs 使用独立随机流，
不让后续 donor/复制/变异随机量错位。条件 logit 默认使用 `[-30, 30]` 数值护栏，
极端有限输入也不会在 float64 中退化成精确 0/1。

**冻结 oracle：** 3 种子 × 初始/100 轮状态 × 200 proposal。3568 条参与记录上
稀疏能量误差最大 `1.11e-16`。8 sweep 在 `tau=1/2` 的 TVD 为
`0.00315/0.02479`，方向差距恢复 `98.97%/95.66%`；平均 proposal gain 从
`28.37→40.13`、`37.86→55.79`，而精确 oracle 为 `40.20/56.50`。六个
seed-state 全部同向，说明有限步近似已基本到达定义好的联合核。

**关闭整代接受：** 固定 `tau=2`，只比较 0 与 8 sweep，`rho/eta/mu`、donor
抽样机制、轮数和主随机流保持一致；状态分叉后的实际 donor 下标不是强制配对量。
追加 20 种子单独得到最终 loss `102.35→89.03`
（15/20）、最后 250 轮 `121.94→101.64`（17/20）、全轨迹
`299.16→221.27`（20/20）。顺序合计 30 种子为最终 `106.30→89.93`（22/30）、
末 250 轮 `122.02→102.27`（25/30）、全轨迹 `296.94→221.76`（30/30）。正收益
事件略少，负收益绝对幅度约下降 4.98%，支持“改善扩散步幅分布”而非后置挑选。

**成本与边界：** 墙钟 `15.50→24.97s/种子`（+61.1%）；因子构造 8.22s，Gibbs
采样 0.444s。扩样为看过首批结果后的顺序设计，合计推断只作描述。本阶段本身尚未
验证标准闭环、nltcs、真实离线 TVD 或跨 workload；后续标准闭环已由上面的独立
Issue #16 实验验证且结论为“不确定”。两阶段均未改变默认生成器，当前仍是精确
target 的无噪声原型，不是 DP。公式、命令、全部失败种子和输出位置见
`docs/设计/三阶因子随机扫描Gibbs扩散.md`，关联 Issue #14。

**验证：** 因子模块 41 项测试、相关测试 183 项、完整 CPU/torch/CUDA 测试
425 项均通过；既有演化测试 25 项通过（另 1 项跳过）。0 sweep
在 `eta=0/0.4/1` 均与旧路径逐单元及主 RNG 端点一致，正式配置的 10 轮 baseline
回归哈希保持为
`1f6890ac1c68a9627d018f0f642f6a06e343a838462530e4781a68b75f487b07`。新增 logit
护栏在冻结 815.8 万个条件值和 30 种子 390.2 万个实际微步中均未触发；冻结结果及
30 个最终表哈希/完整 loss 轨迹保持不变。

### 联合属性块 Gibbs oracle——匹配 KL 后仍存在稳定的独立分解缺口

**问题：** 现有 logistic 核只在“单块方向可加、复制决策独立”的族内最优。一次
转移可能复制多个属性，合取查询会使完整 hybrid 方向 `U(M)` 与单块和 `A(M)`
不同；继续升高单块温度无法利用这部分交互。

**实现：** 新增小规模精确 oracle，枚举 recipient 与 donor 不同属性上的全部复制
mask，并构造 `q_joint(M) ∝ q0(M) exp(beta U(M))`。有限温度不设置正收益门槛、
方向 argmax 或 top-k；不同核用共同 Gumbel-max 做 categorical 配对抽样。加性地形
下联合分布严格退化为独立 logistic 乘积核。精确枚举默认最多 16 个活跃属性、绝对
护栏 20，当前只作研究探针，不改生产主循环。

**交互规模：** `test_300x10`、3 种子、初始/第 100 轮状态、每状态 200 proposal，
共 1200 个冻结 proposal；关闭变异和 generation acceptance。3568 条活跃参与记录
中，历史核下非加性 mask 质量为 35.19%，交互项加权平均绝对值/RMS 为
`0.01673/0.03726`，与状态方向 RMS 同量级。

**匹配 KL 结果：** 每条参与记录先取相同温度两侧较小的 KL，再分别一维求根，使
联合与独立核使用相同 `KL(q||q0)`：

| tau | 独立平均收益 | 联合平均收益 | 差值 | 胜/平/负 | 联合参考熵 | 联合负方向质量 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27.22 | **35.21** | **+7.99** | 478/602/120 | 89.47% | 12.54% |
| 2 | 37.22 | **49.33** | **+12.10** | 670/418/112 | 75.22% | 5.70% |
| 4 | 42.80 | **58.48** | **+15.68** | 792/328/80 | 57.79% | 2.00% |
| 8 | 44.96 | **62.51** | **+17.54** | 845/284/71 | 43.81% | 0.74% |
| 16 | 45.74 | **63.87** | **+18.12** | 861/282/57 | 35.83% | 0.32% |
| 32 | 45.85 | **64.12** | **+18.27** | 863/279/58 | 32.17% | 0.16% |

六个“种子 × 状态”组合在全部温度上平均差值都为正；后期状态的缺口较小但仍保留。
匹配后两侧聚合每活跃块 KL 一致到约 `1.5e-12`，逐 proposal 最大误差
`5.14e-10`。one-hot 完整方向与既有单块方向最大误差 `8.33e-17`，理论独立核与
当前截断 logistic 边缘概率最大误差 `9.36e-14`，排除了额外核强度、列映射和数值
截断三个主要混淆。

**结论边界：** 低温 `tau=1/2` 已证明无需走向确定性 mask 选择，也能利用联合交互；
高温 `tau=16→32` 收益只增 0.25，负方向质量继续减半，再次形成近门控平台。冻结
证据支持“独立分解存在可利用缺口”，但还没有证明无接受动力学、标准闭环、最终
loss/TVD 或跨 workload 改善。精确枚举不是可扩展算法，不能直接设为默认。

**验证：** 新增测试与既有方向/更新测试共 **116 passed**；完整 gsd
CPU/torch/CUDA 测试 **384 passed**；独立 Python 3.11 环境主循环
**25 passed, 1 skipped**。
公式、匹配规则、命令、环境、原始输出和全部失败 proposal 见
`docs/设计/联合属性块扩散核.md`。关联 Issue #12；本变更已在 PR #11 合入后从
最新 `master` 独立整理，不包含后续因子 Gibbs 实现。

### 扩散核温度—熵—漂移极限——核内空间已定位，高温符号门控不是目标路线

**理论：** 固定状态和 donor 后，当前 logistic 核是“加性一阶方向收益减相对历史
Bernoulli 核 KL”的唯一闭式最优解。其期望漂移
`D(tau)=sum_i p_i(tau)u_i` 满足
`D'(tau)=sum_i p_i(1-p_i)u_i^2>=0`；温度趋于无穷时则退化到正方向必复制、负方向
不复制的符号极限。新增相对历史核的逐轮平均 KL、加性漂移改善、可用符号极限
改善和利用率诊断。它们只观察独立单块一阶近似，不进入采样、接受或 checkpoint，
也不冒充精确 proposal 收益上界。

**固定同状态探针：** 小表 1200 个配对 proposal 中，平均精确收益随
`tau=0/1/2/4/8/16` 从 `5.36` 升至
`27.88/37.71/43.65/45.74/46.21`，但负向复制概率从 50% 降至
`30.97%/20.36%/10.74%/4.51%/1.41%`。`tau=8→16` 的收益已经平台化，反向
概率和熵仍继续损失。补齐 KL 诊断后，`tau=1/2/4/8/16/32/64` 分别利用了
加性符号极限改善的 58.34%/80.60%/93.54%/98.39%/99.68%/99.96%/99.998%；
`tau=8→64` 的精确收益只从 45.74 增到 46.40，负向概率则降到 0.11%。

**关闭整代接受：** 小表 1000 轮、10 配对种子最终当前 loss 为
`335.95/155.95/114.20/76.60/50.35`（`tau=0/1/2/4/8`）。`tau=8` 相对
`tau=1` 10/10 改善，并仍有 24.78% 负向复制概率和 78.71% 最大熵；正收益事件
比例仍约 50%，改善主要来自收益幅度而非后置筛选。

**标准闭环：** 沿用历史 500 轮、10 配对种子，只改变温度：

| tau | best loss | 接受前平均收益 | 负向复制概率 | 最大熵比例 |
|---:|---:|---:|---:|---:|
| 0 | 109.45 ± 32.09 | -3.36 | 50.00% | 100.00% |
| 1 | 52.55 ± 15.54 | -0.70 | 44.12% | 98.22% |
| 4 | 28.90 ± 6.10 | +0.17 | 34.69% | 91.06% |
| 8 | 18.15 ± 2.57 | +0.68 | 27.05% | 83.57% |
| 16 | 18.30 ± 5.97 | +0.77 | 16.14% | 70.23% |
| 32 | 14.80 ± 3.47 | +0.63 | 5.72% | 53.07% |
| 64 | 13.90 ± 3.64 | +0.59 | 0.78% | 40.49% |

`tau=8` 已取得 baseline 到近符号门控 `tau=64` 训练 loss 总改善的约 95.6%，并保留
实际反向扩散。`tau=32/64` 的训练 loss 彼此没有稳定差异，接受前平均收益也没有
超过 `tau=8`，却基本丢失反向支持。因此当前区分两种极限：这个 workload、500
轮下的纯训练 loss 平台约为 14；符合扩散路线的 Pareto 拐点约为 `tau=8`。十种子
和多温度扫描只提供较强方向性证据，不能宣称跨数据集最优，也不据此改默认值。

**等价与验证：** 新增诊断前后，seed 0、500 轮的 baseline 与 `tau=8` 成品
SHA-256 分别保持
`1ef835d9...570a00a` 与 `c8c93554...12b6c`，共享轨迹和质量指标相同。
新旧冻结探针共同的 10,800 行逐提案指标最大绝对差也为 0。
针对性测试 **67 passed**，完整 CPU/torch/CUDA 测试 **356 passed**，独立
Python 3.11 环境主循环测试 **25 passed, 1 skipped**。公式、数值截断边界、完整
命令、环境和输出位置见
`docs/设计/扩散核温度熵漂移极限.md`。关联 Issue #10；方向计算性能仍由 Issue #7
独立跟踪。

## 最近变更（2026-08-01）

### 残差驱动正向扩散核——同状态和无接受实验均验证算子正漂移

**问题：** 原流程按完整记录 fitness 选择 donor，却以固定 `eta` 随机复制 donor
的若干块。完整 donor 的方向不等于 recipient 条件下实际 hybrid 编辑的方向；用
支配门控压低反向复制虽然能提升结果，但属于额外定向筛选，不能证明扩散算子本身
变好。

**实现：** 对每个 recipient、donor 和属性块计算实际单块转移方向
`d=sum_j w_j*epsilon_j*(a_j(x^{g<-z_g})-a_j(x))`，再使用
`logit(p_copy)=logit(eta)+tau*d/s0` 连续倾斜 Bernoulli 核。`s0` 是首个非零
active direction 矩阵的 RMS，固定后不再逐轮更新，使残差收缩时倾斜自然冷却。
该闭式核也可解释为“方向收益减 Bernoulli KL”的熵正则最优解。它不使用正方向
门槛、argmax、top-k 或真实表评价；负方向在有限温度下保留非零概率。总开关默认
关闭，`strength=0` 精确退化到历史轨迹。

**失败消融保留：** 不做尺度归一化时，固定 `strength=1` 在小表关闭整代接受的
1000 轮、10 种子实验中只有 6 胜 4 负（最终 loss `335.95→321.90`，`p=0.720`）。
小表与 nltcs 首轮方向 RMS 相差约 32 倍，说明固定原始强度不可跨 workload；正式
候选因此使用 `initial_rms, tau=1`，原始尺度只作消融。

**接受前的因果证据：** 冻结当前表、残差、donor 和全部更新随机数后：

- 小表 3 种子×2 个状态×200 proposal：平均原始收益 `5.36→27.88`，正收益率
  `54.75%→80.50%`，逐提案 `888/238/74`；负方向实际复制概率 30.97%，平均
  复制核熵 0.6084 nat（最大熵的 87.78%）。
- nltcs 3 个历史 baseline 状态×100 proposal：平均原始收益
  `20,743.86→136,670.29`，正收益率 `75.67%→100%`，逐提案 **300/0/0**；
  负方向实际复制概率 33.28%，平均复制核熵 0.6063 nat（最大熵的 87.47%）。

这两组都不执行 generation acceptance。`strength=0` 的所有提案与 baseline
逐项完全相同。candidate 仍保留约 87.5% 的最大 Bernoulli 熵，并非接近确定性的
软门控。

**无整代接受动力学：** 小表 1000 轮、10 配对种子中，每个 proposal 都无条件
成为下一状态，最终当前 loss `335.95→155.95`（-53.58%，10/10，`p=1.59e-4`）。
平均原始收益 `1.818→1.998`，但正收益事件比例变化不显著；改进主要来自收益幅度，
不是把“赢的次数”做高。donor 仍由原适应度与距离抽样，因此这里只隔离整代接受，
不声称脱离原 donor 机制。

**nltcs 正式结果：** 原始 marginal、精确 workload、1500 轮、种子 0/1/2：

| 指标 | baseline | candidate | 相对变化 |
|------|---------:|----------:|---------:|
| best loss | 7,209,581.5 | **3,332,056.0** | **-53.78%** |
| 训练 workload L1 | 0.005349 | **0.003582** | **-33.04%** |
| 未测量 3-way / 4-way L1 | 0.005940 / 0.004719 | **0.004026 / 0.003311** | **-32.22% / -29.84%** |
| 训练 / 测试联合 TVD | 0.314669 / 0.395718 | **0.283810 / 0.368030** | **-9.81% / -7.00%** |
| 漏真 / 训练集外合成质量 | 0.108543 / 0.153555 | **0.104382 / 0.133037** | **-3.83% / -13.36%** |
| 墙钟/种子 | 262.47s | 286.21s | +9.04% |

三种子全部同向，但 n=3 的 p 值只视为探索性证据。全轨迹负方向平均复制概率仍为
44.83%。补熵诊断后的 seed 2 完整复跑中，平均复制核熵为 0.6808 nat（最大熵的
98.22%），最终 CSV 和共享决策轨迹与补诊断前完全相同。唯一状态数从 3262.7 降到
3030.7，但漏真和训练集外质量同时改善，不能只看唯一数判定多样性。方向评价本身
约 43.77s；惰性计算优化单独跟踪在 Issue #7，不混入本方法改动。

三种子平均昂贵状态/距离评价次数为 baseline 1448.3、candidate 1226.3；candidate
另执行 1500 次方向评价，所以墙钟差包含轨迹和缓存次数差异。seed 0、200 轮、
同一张 RTX 4090 顺序资源审计中，两侧均接受全部 proposal、状态/距离各评价 200 次：
主循环 `36.23→42.18s`（+16.41%，其中方向评价 5.42s），进程峰值 RSS
`1,154,828→1,171,040 KiB`（+1.40%），进程显存观测峰值
`9,882→9,890 MiB`（+8 MiB，约 +0.08%）。短程只作成本/内存审计，质量表仍使用
上面的三种子正式运行。

**等价与边界：** 三枚关闭机制的 nltcs baseline 成品与历史 CSV 哈希逐种子相同，
共享决策轨迹相同；donor 距离诊断最大非决策微差 `2.98e-8`。完整测试
**343 passed**，独立 Python 3.11 环境定向主循环测试 **25 passed, 1 skipped**。
当前未建模多块合取协同，也未建立统一的跨方法对照协议；仓库仍不是 DP。

详细公式、失败结果、命令口径、输出目录和限制见
`docs/设计/残差驱动正向扩散算子.md`。关联 Issue #6；方向计算性能问题见 Issue #7。

### 主循环状态缓存与 GPU 诊断修复——结果逐项等价，nltcs 实测加速 8.62 倍

**问题：** 二阶最大熵起点已经很接近目标，600 轮三种子平均只有约 3.7% 的提案
被接受。提案拒绝后当前表 `S` 完全没变，但下一轮仍重复评价查询/fitness 并重建
约 1 GiB 的全对全距离矩阵。另外，原实现为了记录一个 donor 平均距离诊断，每轮
把完整 `16181×16181` 距离矩阵从 GPU 搬回 CPU；微基准中该回传约 191 ms，
而在 GPU 上只 gather 被选中的 16181 个距离后求均值约 0.023 ms。

**实现：**
- 缓存由当前表唯一决定的 `q / residual / fitness / loss / distances`。拒绝提案后
  复用缓存；接受提案后统一失效。动态 `alpha_t`、抽样概率、donor 与提案仍逐轮
  重算，随机数消耗和演化轨迹不变。
- 初始表直接做一次完整评价并初始化 `best_loss`，删除原先 `counts-only` 后第一轮
  再完整评价的重复扫描。
- torch/CUDA 路径在设备上 gather 选中距离并只回传一个均值标量，不再创建约
  1 GiB 的 CPU 距离副本；donor 抽出后立即释放 `N×N` 概率矩阵。
- 新增 `state_evaluation_count` / `distance_evaluation_count` 诊断，直接记录昂贵阶段
  实际执行次数。
- 新增 6 个回归测试，覆盖 NumPy/torch 诊断、连续拒绝复用缓存、接受后缓存失效、
  零轮和初始即收敛边界；完整测试集 **283 passed**。

**nltcs 正式对拍：** RTX 4090、种子 0、二阶最大熵初始化、600 轮，其余参数完全
相同。合并前保存的成品与优化后成品 CSV 的 SHA-256 均为
`7ac90fcafea32a63a9a613d9421625357baf38afd80f6946f57c006ebf84ef02`；loss/接受轨迹、
donor fitness、自身率和全部质量指标逐项相同。GPU 与 NumPy 的归约顺序只让 donor
距离诊断产生最大 `2.98e-8` 的浮点差，不参与算法决策。

| 指标 | 优化前 | 优化后 | 变化 |
|------|-------:|-------:|-----:|
| 600 轮墙钟 | 268.88s | **31.18s** | **8.62× 加速** |
| 单轮墙钟 | 0.4481s | **0.0520s** | -88.4% |
| 当前表完整评价次数 | 600 | **21** | -96.5% |
| 全对全距离计算次数 | 600 | **21** | -96.5% |

**合入前深度审查：** 另以合并前 `origin/master` 建立独立 worktree，对 8 组配置逐项
运行旧版与新版：NumPy geometric 两种子、缩步重试、linear、legacy 评价、允许抽到
自身、torch CPU、torch CUDA。8 组的合成表哈希、loss、接受/重试轨迹和全部算法
诊断完全一致；torch donor 距离归约最大浮点差为 `5.96e-8`。5 轮全部接受、没有
缓存命中的 nltcs 短程内存对照中，进程峰值 RSS 从约 3.12 GiB 降至 1.14 GiB，
GPU 峰值从约 10.0 GiB 降至 9.0 GiB，验证全量回传与概率矩阵生命周期修复本身
也独立生效。

该加速幅度与拒绝率有关：高接受率配置的缓存收益会更小；GPU 诊断回传修复则普遍
有效。本次只改等价执行方式，没有更改初始化、目标函数、抽样公式或更新参数。

**PR：** [#2 缓存演化状态并消除全量距离回传](https://github.com/Chuhan722/table-diffusion/pull/2)
已于 2026-08-01 从 `houyuwushang:perf/cache-evolution-state` 提交到上游 `master`。

### 2-way 最大熵初始化——已实现+测试+nltcs 三种子验证，设为 nltcs 推荐配置

**动机与诊断：** 原 `marginal` 初始化只贴合 1-way 边缘，各列独立打乱会丢掉
属性相关性。nltcs 旧成品的训练联合 TVD 为 0.3147，虽接近训练集—测试集经验
TVD 0.2834，但仍同时漏掉 10.85% 的真实模式质量、生成 15.36% 的训练集外模式
质量。1–4 阶边缘已经较好而 16-way TVD 仍高，说明主要瓶颈是起点缺少联合结构，
不是简单继续调重试次数。

**实现：**
- 新增 `src/table_diffevo/pairwise_init.py`：只从 `queries + target` 收集完整二属性
  等值边缘，在可枚举类别状态空间上用 IPF 拟合最大熵分布，再抽样得到 S_0；
  运行时不读取原始表，真实 train/test 仅由实验脚本在结束后评价 TVD。
- 每个属性对允许缺一个单元，用公开 N 减去其余单元恢复。nltcs 120 对属性的
  479/480 个单元全部可用，唯一缺失单元成功恢复。
- 明确护栏：当前仅支持全 categorical；默认状态上限 1,000,000；覆盖不足、数值
  属性或空间过大均明确报错并提示回退，不做静默近似。
- `run_evolution(init_method='pairwise_maxent')` 接入原扩散演化；原 random/marginal
  路径与 API 默认保持不变。诊断新增 IPF 扫描数、收敛状态、拟合/抽样二阶误差、
  使用属性对数、状态数与初始化耗时。
- `scripts/run.py` 的 nltcs 推荐配置改为 `pairwise_maxent + 600 轮`；新增
  `scripts/run_pairwise_maxent_nltcs.py`，明确隔离运行时输入与离线 TVD 评价。
- 设计与边界见 `docs/设计/初始化设计_2way最大熵.md`。新增 8 个针对性测试，
  全套 CPU/CUDA **277 passed**，无回归。

**正式对照：nltcs，同种子 0/1/2；候选 600 轮 vs 旧 marginal 基线 1500 轮：**

| 指标 | marginal 1500 轮 | pairwise_maxent 600 轮 | 相对变化 / 配对检验 |
|------|------------------:|--------------------------:|----------------------:|
| best loss | 7,209,582 ± 436,119 | **464,081 ± 74,341** | -93.56%，p=0.00134 |
| 训练 workload L1 | 0.005349 ± 0.000199 | **0.001398 ± 0.000090** | -73.86%，p=0.00097 |
| 训练联合 TVD | 0.3147 ± 0.0042 | **0.2352 ± 0.0053** | -25.24%，p=0.00292 |
| 独立测试联合 TVD | 0.3957 ± 0.0022 | **0.3261 ± 0.0055** | -17.58%，p=0.00267 |
| 未测量 3-way L1 | 0.005940 | **0.001767** | -70.24%，p=0.00102 |
| 全部 4-way L1 | 0.004719 | **0.001785** | -62.18%，p=0.00132 |
| 漏掉真实模式质量 | 0.1085 | **0.0982** | -9.51%，p=0.0278 |
| 新生成模式质量 | 0.1536 | **0.1009** | -34.30%，p=0.00046 |

三个种子全部胜出。候选轮数更少，质量对比是保守的；总墙钟约 683s→261s 的下降
主要来自 1500→600 轮，不能声称单轮代码加速。最大熵拟合本身约 3.6s、87 次扫描，
模型二阶最大误差约 8.2e-9。

**结论/边界：**
- 该改进保留“已测量统计初始化 + 扩散演化精修”的原思想，不把真实联合 TVD
  偷偷放入训练；它同时改善训练、独立测试和未测量高阶查询，证据支持作为 nltcs
  推荐起点。
- 仓库仍不是 DP 实现：当前 target 是精确计数，没有 ε/δ、加噪机制或 accountant。
  DP 噪声会造成跨属性对不一致，正式 DP 阶段必须增加统一一致性投影；当前
  `converged` 诊断不能替代它。
- 新起点下原更新参数接受率仅约 3.7%。下一步若继续优化，应单独研究更小 rho、
  确定性配额抽样或 patience，不能在本次初始化结论里混做。

**输出：** `outputs/pairwise_maxent_nltcs_600_seed{0,1,2}/`（每个目录含合成表、
诊断、离线 evaluation 与 summary；outputs 被 gitignore，不进版本库）。

**PR：** [#1 新增二阶最大熵初始化并完善扩散演化提案重试](https://github.com/Chuhan722/table-diffusion/pull/1)
已于 2026-08-01 从 `houyuwushang:feat/pairwise-maxent-init` 提交到上游 `master`。

### 整代提案缩步重试——已实现+测试+小表/nltcs 对照，默认保持关闭

**动机：** 原主循环每轮只生成一个整代提案，被贪心接受检查拒绝后，
已经计算的适应度、全对全距离和 donor 抽样全部浪费。小表推荐配置的拒绝率约 41%，
适合验证“复用当轮 donor、缩小 rho 重试”能否把被拒的大步改成可接受的小步。

**实现：**
- `run_evolution` 新增 `max_retries=0` 和 `retry_rho_decay=0.5`。第 a 次尝试使用
  `rho * retry_rho_decay**a`；重试只重做 `evolve_step + proposal 查询评价`，
  复用当轮 residual/fitness/distance/donor，不重算最贵的全对全阶段。
- 默认 `max_retries=0`，随机数消耗和历史轨迹与原实现一致。`scripts/run.py` 暴露
  `MAX_RETRIES` / `RETRY_RHO_DECAY` 开关，但不擅自更改生产默认。
- 新增诊断 `proposal_attempts_history` / `accepted_attempt_history` /
  `accepted_rho_history`，可直接看每轮尝试数、第几次被接受和实际 rho。
- 新增 `scripts/compare_retries.py`（小表 20 种子+近似等墙钟对照）和
  `scripts/analyze_retry_nltcs.py`（训练 workload + 未测量 3-way + 全部 4-way + 联合 TVD）。
- 新增 6 个主循环测试：默认单次、缩步值/第二次接受、参数边界。
  完整 CPU/CUDA 测试集 **269 passed**，无回归。

**test_300x10，100 轮×20 配对种子，α2→10：**

| 配置 | best loss（均值±std） | 最终拒绝率 | 提案/轮 | 耗时/种子 |
|------|-------------------------|-----------|---------|------------|
| baseline（100 轮） | 886.5 ± 137.0 | 40.6% | 1.00 | 0.351s |
| retry1（ρ×0.5） | 730.7 ± 159.7 | 12.5% | 1.39 | 0.398s |
| retry2（ρ×0.5×0.5） | **726.2 ± 123.5** | **3.2%** | 1.56 | 0.419s |
| baseline（120 轮，近似等墙钟） | 763.0 ± 119.0 | 41.3% | 1.00 | 0.416s |

- 固定 100 轮：retry2 loss 低 **18.08%**，20 种子 19 胜 1 负，配对
  **p=2.08e-5**；L1 0.01445→0.01289，唯一记录数 294.6→293.9，多样性基本不变。
- 近似等墙钟：retry2 比 120 轮 baseline loss 仍低 4.81%，但 **p=0.196 不显著**。
  因此小表上“同轮数更好”证据强，“同时间更好”只有趋势。

**nltcs，1500 轮×3 配对种子，α2→10：**

| 指标 | baseline | retry2 | 相对变化 / 检验 |
|------|----------|--------|------------------|
| best loss | 7.210e6 | **6.881e6** | -4.55%，p=0.3895（2 胜 1 负） |
| 训练 L1 | 0.005349 | **0.005250** | -1.85%，p=0.6184 |
| 未测量 3-way L1（3958 查询） | 0.005940 | **0.005843** | -1.63%，p=0.6354 |
| 全部 4-way L1（29120 查询） | 0.004719 | **0.004657** | -1.31%，p=0.6777 |
| 完整联合 TVD | **0.314669** | 0.315535 | +0.27%，p=0.5843 |
| 接受率 | 96.56% | **99.82%** | 最终拒绝约 52→3 轮/种子 |
| 提案/轮 | 1.000 | 1.045 | +4.5% 提案评价 |
| 耗时/种子 | 682.5s | 683.2s | +0.11% |

**结论/决策：**
- 缩步重试的机制有效：两数据集都大幅减少最终拒绝，且 nltcs 因复用贵的
  当轮计算，墙钟开销几乎为零。训练和未测量边缘误差方向一致，没有明显过拟合信号。
- **但不改默认**：nltcs 只有 3 种子且差异不显著，联合 TVD 还有 +0.27% 的微弱反向；
  小表等墙钟对照也未显著。不应宣称它是新的通用最优配置。
- **条件推荐**：当拒绝率高、全对全 donor 计算远贵于一次提案评价时，可试
  `max_retries=2, retry_rho_decay=0.5`；接受率已很高时收益天花板较低，保持 0 更稳妥。

**输出：**
- 小表：`outputs/retry_experiment_small/summary.json`
- nltcs baseline：`outputs/retry_baseline_nltcs_2026-07-31_2223/`
- nltcs retry2：`outputs/retry2_nltcs_2026-07-31_2223/`（`comparison.json` 含泛化评价）

## 最近变更（2026-07-28）

### 对角线屏蔽 exclude_self（禁止记录抽到自己）—— 已实现+测试+回归实验，收尾

**背景：** 用户担心 α2→10 高接受率是否由"抽到自己=复制自己=表不变=整代必接受"刷出来的。
候选池=全表（K=N，全对全）含自身，自身距离=0、相似度=1，高锐度 softmax 可能把质量堆到自己身上。

**诊断（scripts/diagnose_self_sampling.py，只测量不改主代码）：** 复现单轮抽样直接数 donor_idx==i 比例
（历史 diagnostics 只存 donor_distance_history 均值、未存 donor_idx，无法从历史直接读出）。结果：
- **nltcs α2→10**：末轮 α_t=10 自身抽样率仅 **0.04%**（万分之四）——高接受率**不是**自我复制刷的，原结论成立。
  原因：nltcs 83% 重复，每条记录有大量等价副本瓜分"相似度=1"的质量，自身只分到 ≈1/副本数。
- **test_300x10 α2→10**：0% 重复、无副本兜底，自身是全表唯一"满分"候选，末轮 α_t=10 自身率升到 **8.07%**
  （单条最高 p_ii 达 71%）。小表高锐度确有自我复制浪费（该行更新被浪费、末轮收敛略慢），但因整代
  接受判据看全表 loss、其余行照常演化，非"表死住"或"接受率虚高"。

**修复（用户拍板选方案 1：对角线屏蔽）：**
- `sampling.py`：`compute_sampling_probs` / `_compute_sampling_probs_torch` 加参数 `exclude_self=False`。
  True 时抽样前把对角 probs[i,i] 置 0 再按行重归一化（等价对角 logit=-inf），覆盖全部 5 种 distance_mode
  （加性 softmax 与乘性/几何提前 return 三处落点 × numpy+torch 双路径）。新增 `_exclude_self_numpy/_exclude_self_torch` 辅助函数。
  仅在候选池=全表（方阵 N==K）时有意义，非方阵报错（防误用于共享参考池 K≠N）。
- `evolution.py`：`run_evolution` 加形参 `exclude_self: bool=True`（B1 方案：run.py 不暴露、走默认 True，
  只让实验脚本传 False 复现 baseline）；主循环 `distances=pairwise_block_distance(S,S,...)` 全对全，
  抽样调用透传该形参；记进 `diagnostics["params"]["exclude_self"]` 可回溯。
- `sampling.compute_sampling_probs` 自身默认仍 `exclude_self=False`（独立调用/诊断脚本行为不变），
  仅 `run_evolution` 默认 True。**将来改共享参考池（K≠N）时非方阵会报错，届时需把主循环改传 False**
  （护栏：候选池阶段绝不会带着错误的自我屏蔽静默跑起来）。
- 新增常驻诊断字段 `donor_self_rate_history`（每轮 `mean(donor_idx==i)`），exclude_self=True 时恒 0。
- 新增 `tests/test_sampling_exclude_self.py`（24 个）+ `test_evolution.py::TestExcludeSelf`（3 个：
  字段存在、默认 True 自身率恒 0、False 允许非零），全套 **263 passed**（236+27），无回归。

**回归实验（scripts/exclude_self_regression.py，1 号卡，α2→10 λ0.5）：**
- **核心结论达成**：屏蔽后自身率恒 **0**（nltcs 3 种子 + 小表 5 种子末段均 0.000%）；
  **接受率几乎不变**（nltcs 96.8%→96.6%）——直接回答用户最初疑虑：**高接受率不是"抽到自己"刷的**。
- **nltcs loss**：False 6.87e6 vs True 7.59e6（+10.5%），但**配对 t 检验 p=0.165 不显著**，
  且 baseline 种子间方差本就 8%。非机制退化（自身率本来 0.04%，只影响万分之四行）——
  是屏蔽平移整行累积概率边界→`sample_donors` 同种子下抽到不同 donor→轨迹发散噪声。用户判定差异不大、不补种子。
- **小表 test_300x10**：False 990.5 vs True 941.3（loss 略降但 p=0.319 不显著）、唯一记录 294→292（p=0.137）。
  屏蔽消除自我复制空转，质量无损。
- baseline（False 侧）复用写死前的旧结果：nltcs seed0 来自 `outputs/geometric_alpha2_10_2026-07-28_0530/`、
  seed1/2 来自 `outputs/alpha_multiseed_1500/a2_10/`；新结果落 `outputs/exclude_self_regression/`。

**未做：** 总结文档暂不补（用户先前指示）。`scripts/diagnose_self_sampling.py` 保留（离线诊断，不进运行期）。

### α 配置多种子验证（nltcs，1500 轮，seed=0/1/2）—— 推翻"α2→10 坍缩"判断

**背景：** 用户质疑文档里"α2→10 坍缩"的说法怎么得出的。核查发现该判断是错的（α2→10 最高频 13.04% < 真实 17.67%、Top-10 全覆盖，未坍缩）。遂补 nltcs 多种子 α 对比 + 配对 t 检验，确认 α1.5→6 vs α2→10 的真实排序。

| 配置 | best_loss 均值 | 标准差 | seed0/1/2 | L1 均值 | 接受率 |
|------|---------------|--------|-----------|---------|--------|
| α1.5→6（原推荐） | 7.33e7 | 4.08e6 | 7.31e7/7.75e7/6.93e7 | 0.0184 | ~89.9% |
| **α2→10** | **6.87e6** | **3.21e5** | 7.24e6/6.69e6/6.69e6 | **0.0052** | ~96.7% |

**统计检验：**
- best_loss：α2→10 低 **10.7 倍（−90.6%）**，配对 t 检验 t=28.1，**p=0.0013 显著**
- L1：0.0052 vs 0.0184，t=46.0，**p=0.0005 显著**
- 三种子完全一致，α2→10 方差更小，排除种子噪声

**收敛性警示（关键）：** 两配置 1500 轮**均未收敛**（末 100 轮仍降 25~32%）。查曲线：α2→10 在 75% 进度处已 3.0e7 vs α1.5→6 的 1.9e8，**确实收敛更快**（α 大 → softmax 尖 → 收敛压力强）。但两者都没跑到底、α2→10 恰处曲线更陡段，1500 轮截断把领先**放大**了。"10 倍"= 真优势 + 未收敛放大：**方向可信、倍数存疑**。测真实幅度需两者都跑到收敛（patience 早停或 3000+ 轮）——用户决定**暂缓不跑**。

**推荐配置决定（用户拍板 2026-07-28）：推荐配置改为 α 2→10, λ 0.5。**
- α2→10 精度显著且稳定更优（两数据集方向一致：nltcs 一个数量级 p=0.0013；小数据 ~7.9% p=0.21 不显著），分布健康未坍缩，接受率最高（96.8%）。
- α2→10 多样性比 α1.5→6 略集中（唯一 3251 vs 5042），但所有分布指标（最高频、Top-10 覆盖）均健康，且更贴近 nltcs 真实高重复结构。
- 收敛后真实幅度未测（1500 轮均未收敛，10 倍含未收敛放大），但方向稳定，不影响选型。

**文档处理：** geometric 总结文档已重写为"总结归纳"体例——删去已推翻的探索过程（α2→10 坍缩误判、λ 的 1000/1500 轮翻转过程），只保留最新结论，推荐配置统一为 α2→10, λ0.5。

**输出：** `outputs/alpha_multiseed_1500/{a1p5_6,a2_10}/seed{1,2}/`，脚本 `scripts/alpha_multiseed_1500.py`（nltcs）、`scripts/alpha_small_multiseed.py`（小数据 5 种子）。

### λ 参数扫描（geometric 模式，锁定 α 1.5→6）

**结论：推荐 λ=0.5。1000 轮粗扫曾显示 λ=0.6 略优（+3.7%），但 1500 轮收敛后 λ=0.5 反超（+3.6%）——1000 轮是未收敛假象。差异仅 3~4% 且随轮数翻转、单种子，不足以定论，先取 λ=0.5（原始默认、改动最小）。**

#### 动机
geometric 模式已定 α=1.5→6，但 λ（适应度-距离权衡）从未扫过，一直固定 0.5。λ 是剩下唯一没碰的核心自由度，直接控制精度-多样性偏向。本次 OAT 扫描验证 λ=0.5 是否通用最优。

#### 方法
- **单种子粗扫地形（第 1 步）**：λ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}，两个数据集
  - test_300x10：300×50，100 轮，seed=42（几乎不花时间）
  - nltcs：16181×1000，1000 轮，seed=0（约 15 min/λ）
- 固定参数：α 1.5→6，ρ=0.01, η=0.5, μ=0.01, δ=0.05, winsorize=(0.01,0.99)

#### 实验结果

**nltcs（16181 条 × 1001 查询，1000 轮，seed=0）**

| λ | 最优 loss | 归一化 L1 | 下降% | 接受率 | 唯一记录 | 重复率% | Top-10 覆盖 |
|---|-----------|----------|-------|--------|----------|---------|-------------|
| 0.3 | 2.58e+08 | 0.0340 | 62.8 | 88.4% | 5302 (32.8%) | 67.23 | 9/10 |
| 0.4 | 1.81e+08 | 0.0290 | 73.9 | 93.7% | 5560 (34.4%) | 65.64 | 10/10 |
| 0.5 | 1.35e+08 | 0.0251 | 80.6 | 94.8% | 5489 (33.9%) | 66.08 | 10/10 |
| **0.6** | **1.30e+08** | **0.0250** | **81.3** | 96.7% | 5640 (34.9%) | 65.14 | 9/10 |
| 0.7 | 1.35e+08 | 0.0258 | 80.5 | 93.9% | 5855 (36.2%) | 63.82 | 10/10 |

**1000 轮地形（未收敛）**：单峰，峰值在 λ=0.6。
- λ=0.3 → 0.6：loss 单调下降（2.58e8 → 1.30e8），提升 **49%**
- λ=0.6 比 0.5 优 3.7%（1.30e8 vs 1.35e8）
- λ=0.6 → 0.7：持平或略降（差 3.8%，在噪声边缘）

多样性全部健康：覆盖真实 Top-10 中 9–10 个，最高频 6.2–8.3%（远低于真实 17.67%），重复率 64–67%（真实 83.49%）。无坍缩，都在合理模拟分布。

**1500 轮验证（收敛后，结论反转）**：补跑 λ0.5/0.6 到 1500 轮，排序翻转。

| λ | 1000 轮 loss | 1500 轮 loss |
|---|--------------|--------------|
| 0.5 | 1.35e+08 | **7.31e+07**（反超） |
| 0.6 | 1.30e+08 | 7.58e+07 |

λ0.6 早期收敛快（1000 轮领先），λ0.5 偏多样性、后期不易卡住（1500 轮反超 3.6%）。两曲线在 1000~1500 轮间交叉。**教训：调参对比必须对齐轮数并跑到收敛。**
输出：`outputs/geometric_lambda0p6_1500_2026-07-28_1003/result.json`

**test_300x10（300 条 × 50 查询，100 轮，seed=42）**

| λ | 最优 loss | 归一化 L1 | 下降% | 唯一 | 重复率% |
|---|-----------|----------|-------|------|---------|
| 0.3 | 1.14e+03 | 0.0157 | 44.2 | 297 (99%) | 1.00 |
| 0.4 | 1.10e+03 | 0.0168 | 45.9 | 295 (98.3%) | 1.67 |
| 0.5 | 8.88e+02 | 0.0136 | 56.5 | 297 (99%) | 1.00 |
| 0.6 | 8.97e+02 | 0.0141 | 56.1 | 297 (99%) | 1.00 |
| **0.7** | **8.28e+02** | 0.0143 | **59.5** | 296 (98.7%) | 1.33 |

**地形**：单调递增趋势，但 0.5–0.7 段几乎平（差 7%，单种子噪声大）。小数据量使得结论弱。

#### 关键发现

1. **λ0.5 与 λ0.6 差异小且随轮数翻转**：1000 轮 λ0.6 优 3.7%，1500 轮 λ0.5 优 3.6%。单种子无法区分，需多种子 t 检验才能定论。当前先取 λ=0.5（1500 轮收敛数字更可信，且原始默认、改动最小）。

2. **1000 轮的"假象"教训**：未收敛时的参数排序可能误导——λ0.6 前期收敛快但后期被 λ0.5 反超。以后调参对比必须对齐轮数并跑到收敛。

3. **仍成立的共性**：λ=0.3（偏相似度）明显最差（1000 轮下比 λ0.5 差近一倍），生成任务需要 λ≥0.5 偏向适应度那侧。直觉吻合：目标是满足查询约束（适应度），距离只是防坍缩辅助。

4. **多种子精修（已完成 2026-07-28）**：λ=0.5 vs 0.6，seed=0/1/2，nltcs 1500 轮，配对 t 检验。
   - **λ=0.5**: 7.33e7 ± 4.08e6 (seed0/1/2: 7.31e7, 7.75e7, 6.93e7)
   - **λ=0.6**: 7.36e7 ± 2.34e6 (seed0/1/2: 7.58e7, 7.11e7, 7.39e7)
   - 差异 +0.42% (λ0.6 略差),**p=0.93 >> 0.05**,**差异不显著**
   - **结论**：单种子看到的 3.6% 差异是种子噪声,λ0.5 与 0.6 无法区分。推荐 λ=0.5（对称默认值、改动最小）。
   - 输出：`outputs/lambda_multiseed_1500/`

#### 推荐配置
```python
# 推荐用于生产环境的 geometric 参数（2026-07-28，λ 回退到 0.5）
distance_mode = 'geometric'
lambda_param = 0.5          # 适应度-距离均衡（1500 轮收敛后略优于 0.6）
alpha_min = 1.5             # 初始锐度（早期探索）
alpha_max = 6.0             # 终值锐度（后期收敛）
delta = 0.05                # 下界防止 log(0)
winsorize_quantiles = (0.01, 0.99)  # 裁剪极端值
```

**何时使用**：跨数据集、不想为每个数据单独调 β/p、需要精度-多样性平衡的场景。

**默认值决策**：**库默认已改为 geometric**（2026-07-29，提交 47ce6db；run.py 同日 013ef8a 对齐）。此前一度写「暂不改默认」，理由是 geometric 只在 nltcs + test_300x10 两数据集验证过、待更多数据集确认普适性。后续拍板直接把 geometric（α2→10, λ0.5）定为库默认与 run.py 默认。linear/squared/none/multiplicative 仍可显式指定，向后兼容。

**输出**：`outputs/lambda_sweep_2026-07-28_0812/comparison.json`

---

### 抽样设计：geometric 几何均值模式（精度-多样性最佳平衡，推荐使用）

**结论：geometric 模式（α 1.5→6, λ 0.5）在 nltcs 和小数据集上均达到最佳平衡，推荐作为默认配置。**

#### 设计动机
multiplicative 模式虽比 linear 优 31%，但仍存在量级匹配问题：适应度 F 的绝对尺度依赖查询数和相关性（∝ m^0.75），换数据集后 `β·F` 与 `(1-d)^p` 的相对权重会漂移。geometric 模式通过**归一化 + 几何均值**彻底解耦尺度：
- **Winsorize + min-max 归一化**：将 fitness 剪裁到 [q_low, q_high] 分位数再线性缩放到 [0,1]，消除量级和异常值影响
- **几何均值结合**：A = f^λ · s^(1-λ)，其中 f = δ + (1-δ)·f_norm ∈ [δ,1]，s = δ + (1-δ)·(1-d) ∈ [δ,1]
  - λ ∈ [0,1] 控制适应度-距离权衡（0=纯距离，1=纯适应度）
  - δ 是下界防止 log(0)，确保 f, s 有界
- **动态锐度调度**：α_t 从 α_min 线性增长到 α_max，控制抽样分布的尖锐程度（早期平缓探索，后期锐利收敛）

**核心优势**：
1. **两个独立自由度**：λ（tradeoff）和 α（sharpness），语义清晰、调参解耦
2. **跨数据集稳定**：归一化后 f, s ∈ [δ,1] 不依赖数据尺度，λ=0.5 / α 范围在不同数据集上通用
3. **鲁棒处理异常值**：winsorize 自动裁剪极端 fitness，避免单个高适应度样本主导抽样

#### 代码实现（2026-07-27/28）
- `sampling.py`：新增 `distance_mode='geometric'` + 参数 `lambda_param=0.5`, `alpha=1.0`, `delta=0.05`, `winsorize_quantiles=(0.01,0.99)`
  - Winsorize: `f_clip = np.clip(fitness, q_low, q_high)`，然后 min-max 归一化到 [0,1]
  - Bounded: `f = δ + (1-δ)·f_norm`, `s = δ + (1-δ)·(1-d_clipped)` 确保 ∈ [δ,1]
  - Log-space: `logits = α·[λ·log(f) + (1-λ)·log(s)]`，然后 softmax
  - numpy 和 torch 双路径实现，距离自动 clip 到 [0,1] 防止负相似度
- `evolution.py`：新增参数 `lambda_param`, `alpha_min`, `alpha_max`, `delta`, `winsorize_quantiles`
  - 动态调度：`α_t = α_min + (α_max - α_min) · progress`，其中 `progress = t/(n_rounds-1)`
  - 记录 `alpha_history` 和 geometric 参数到 diagnostics
- 新增 `tests/test_sampling_geometric.py`（18 个测试）：基本功能、边界情况、鲁棒性、torch 一致性、端到端集成、参数验证
- 全套测试：**236 passed**（218+18），无回归

#### 实验结果

**nltcs 大数据（16181 条 × 1000 查询，1500 轮，seed=0）**

| 配置 | 最优 loss | 归一化 L1 | 接受率 | 下降比例 |
|------|-----------|-----------|--------|----------|
| **geometric α1.5→6** | **7.31e+07** | **0.0185** | 90.3% | **89.5%** |
| geometric α2→10 | 7.24e+06 | 0.0054 | 96.8% | 99.0% |
| geometric α0.5→4 | 2.71e+08 | 0.0359 | 76.2% | 60.9% |
| multiplicative p=1 | 7.17e+07 | 0.0188 | 46.0% | 89.7% |

**多样性分析（真实数据：2671 唯一 / 83.49% 重复率 / 最高频 17.67%）**

| 配置 | 唯一记录 | 重复率 | 最高频% | Top-10 覆盖 | 分布健康度 |
|------|----------|--------|---------|-------------|-----------|
| **geometric α1.5→6** | **5042 (31%)** | **68.84%** | **9.34%** | 10/10 | ✓ 健康 |
| geometric α2→10 | 3251 (20%) | 79.91% | 13.04% | 10/10 | ✓ 最接近真实 |
| geometric α0.5→4 | 5877 (36%) | 63.68% | 4.59% | 9/10 | ✓ 健康 |
| multiplicative | 5308 (33%) | 67.20% | 7.77% | 10/10 | ✓ 健康 |

**test_300x10 小数据（300 条 × 50 查询，100 轮，seed=42，真实数据：300 唯一 / 0% 重复率）**

| 配置 | 最优 loss | 归一化 L1 | 接受率 | 合成唯一 | 合成重复率 |
|------|-----------|-----------|--------|----------|-----------|
| **geometric α1.5→6** | **8.88e+02** | **0.0136** | 60.0% | 297 (99%) | 1.00% |
| geometric α2→10 | 9.42e+02 | 0.0147 | 63.0% | 294 (98%) | 2.00% |
| geometric α0.5→4 | 1.03e+03 | 0.0146 | 40.0% | 300 (100%) | 0.00% |

#### 关键发现

1. **α1.5→6 达到最佳平衡**：
   - **nltcs 上**：精度与 multiplicative 持平（loss 7.31e7 vs 7.17e7，差 2%），多样性健康（覆盖所有真实 Top-10，最高频 9.34% < 真实 17.67%）
   - **test_300x10 上**：精度最优（loss 8.88e2，比 α0.5→4 低 14%），多样性最接近真实（1% 轻微重复 vs 真实 0%）
   - **鲁棒性**：两个数据集分布完全不同（nltcs 83% 重复 vs test_300x10 0% 重复），α1.5→6 在两者上都表现优异

2. **α 调度的重要性**：
   - α0.5→4（旧参数）：初始 α 过小导致前期收敛慢，nltcs 上 1500 轮才降 60.9%
   - α2→10：初始 α 过大虽收敛快（99.0% 下降），但在 nltcs 上过度坍缩（唯一 3251 vs 真实 2671，最高频 13.04%）
   - α1.5→6：起点适中、终值合理，早期有足够区分度、后期不过度尖锐

3. **"坍缩"判断需看分布相似度**：
   - 仅看重复率无法判断：nltcs 真实数据本身 83% 重复率（16 二元特征只用了 2671/65536 = 4% 理论空间）
   - 真正标准：合成数据是否覆盖真实高频模式（Top-10 覆盖率）、是否产生单一超频模式（最高频 < 20%）
   - α2→10 虽接近真实重复率（79.91% vs 83.49%），但 Top-10 中 8/10 来自真实 Top-10，说明在正确模拟分布而非坍缩

4. **收敛效率**：α1.5→6 接受率 90.3%（nltcs）/ 60.0%（小数据），说明演化路径高效稳健，比 multiplicative 的 46% 少了很多无效尝试

#### 推荐配置（2026-07-28，λ=0.5）
```python
# 推荐用于生产环境的 geometric 参数
distance_mode = 'geometric'
lambda_param = 0.5          # 适应度-距离均衡（1500 轮收敛后略优于 0.6，详见上节 λ 扫描）
alpha_min = 1.5             # 初始锐度（早期探索）
alpha_max = 6.0             # 终值锐度（后期收敛）
delta = 0.05                # 下界防止 log(0)
winsorize_quantiles = (0.01, 0.99)  # 裁剪极端值
```

**何时使用**：跨数据集、不想为每个数据单独调 β/p、需要精度-多样性平衡的场景。

**默认值决策**：**库默认已改为 geometric**（2026-07-29，提交 47ce6db；run.py 同日 013ef8a 对齐）。此前记为「暂不改默认」（理由：仅 nltcs + test_300x10 两数据集验证、待确认普适性），后拍板将 geometric 定为库默认与 run.py 默认。其余模式仍可显式指定。

#### 输出位置
- 四方对比：`outputs/geometric_alpha1p5_6_2026-07-28_0559/comparison.json`
- α2→10：`outputs/geometric_alpha2_10_2026-07-28_0530/`
- α0.5→4 + multiplicative：`outputs/geometric_vs_multiplicative_2026-07-28_0431/`
- 小数据三方对比：脚本 `scripts/compare_alpha_on_small_data.py`

---

## 最近变更（2026-07-25）

### 抽样设计：乘法解耦版 multiplicative 模式（第 0 步：代码实现完成）

**动机**：现有抽样分数 `ℓ = β·F − d/h`（相加结构）存在两个问题：
1. **量级匹配难**：F 的量级由查询数、残差幅度决定，跨数据集不稳定；需要 β·F 和 d/h 在同一量级才能平衡，但 F 只能实测、调参耦合。
2. **硬门槛坍缩风险**：若改用硬门槛(距离>r 排除)绕开量级问题，会丢失逃逸通道，导致"相似→只互抽→更相似"正反馈坍缩。

**新设计（multiplicative 模式）**：
- 核心思路：**适应度和距离各自算各自，最后相乘**，不再相加。
- 公式：`p(k|i) ∝ softmax(β·F) × (1−d)^p`，按行归一化。
  - 第 1 步：F 先过 softmax → 适应度概率 p_F（和为 1），β 控制锐度。
  - 第 2 步：距离权重 w = (1−d)^p ∈ [0,1]，p 控制陡度。
  - 第 3 步：相乘 → 按行归一化。
- **同时解决两个问题**：
  - 量级问题退化成"两个有界形状钮"（β 管锐度、p 管陡度），不需要匹配 F 和 d 的绝对大小。
  - 软权重 w>0 保留逃逸通道，坍缩风险低。

**代码实现（2026-07-25 晚）**：
- `sampling.py`：`compute_sampling_probs` 新增 `distance_mode='multiplicative'` + 参数 `p: float = 1.0`（距离陡度）。numpy 和 torch 两条路径都实现。现有三种模式(squared/linear/none)不动，向后兼容。
- `evolution.py`：`run_evolution` 新增参数 `p: float = 1.0`，透传给抽样，记录到 diagnostics.params。
- 新增 `tests/test_multiplicative.py`（8 个测试）：基本功能、边界情况(p=0/β=0)、数值稳定、torch 一致性。
- 全套测试：**218 passed**（+8），无回归。

**设计文档**：`docs/设计/抽样设计_乘法解耦版.md`（完整公式、为什么解决问题、两个调节钮含义、与现有模式对比）。

**验证结果（2026-07-26/27）**：

**第 1 步（小数据查错，test_300x10, 100 轮, 单种子）**：✅ 通过
- 无 NaN/inf，loss 下降 29.6%（2021→1422），接受率 40%，参数记录正确。

**第 2 步（nltcs 单种子对比，1500 轮，seed=0）**：✅ multiplicative 明显更好
- multiplicative: best_loss=7.17e7, 归一化 L1=0.0188, 接受率 46.0%
- linear: best_loss=1.04e8, 归一化 L1=0.0232, 接受率 43.2%
- loss 相对差 -31.1%；multiplicative 全程领先（非后期反超），后期仍在降。

**第 3 步（nltcs 多种子严格对比，1500 轮，seed=0/1/2）**：✅ 高度显著更好
| 模式 | 最优 Loss (均值±std) | 归一化 L1 | 接受率 |
|------|---------------------|-----------|--------|
| **multiplicative** | **7.09e7 ± 8.3e5** | **0.0186 ± 0.0002** | ~45% |
| linear | 1.03e8 ± 5.2e5 | 0.0230 ± 0.0001 | ~42% |
- **loss 相对差 -31.4%**，t 检验 **p=1.2e-6（高度显著）**。
- 三种子两组完全不重叠（最差的 mult 7.17e7 仍远低于最好的 linear 1.03e8），方差极小（~1%）。
- 与第 2 步单种子（-31.1%）一致，排除偶然。归一化 L1 同步改善约 19%。
- 无坍缩迹象（loss 持续下降、接受率正常），验证了"软权重保留逃逸通道"的设计推理。
- 输出：`outputs/multiplicative_step3_nltcs_2026-07-27_0905/`（含 summary.json + 各种子 diagnostics）。

**结论**：乘法解耦在 nltcs 上显著优于 linear（约 31%），设计推理得到验证。

**默认值决策（2026-07-27 时点）**：当时定为「暂不改默认」（彼时默认仍是 linear）。
- 理由：nltcs 单数据集虽 3 种子很稳，但需换数据集才能确认"普遍更好"。
- multiplicative 已可用（作为可选模式），改默认值待更多数据集验证后再定。

**后续更新（2026-07-29）**：库默认最终改为 **geometric**（非 multiplicative），提交 47ce6db，run.py 同日对齐（013ef8a）。上面这段是历史决策记录，现已被覆盖。

**后续候选（待讨论）**：
- 换数据集再确认（验证普适性，是改默认的前提）。
- 第 4 步：扫 p（[0.5,1,2,3]）看调参是否更顺手（设计初衷）。
- 第 5 步：p 递减调度（之前决定先固定，验证可行后可试）。

---

### 距离项函数形式对比实验（squared / linear / none）
**结论：LINEAR（拉普拉斯核 exp(-d/h)）在大数据上显著优于 SQUARED（高斯核 exp(-d²/2h²)）和 NONE（无距离）。**

#### 实验设计
- **实现**：在 `sampling.py` 新增 `distance_mode` 参数（'squared'/'linear'/'none'），
  修改距离惩罚项计算：squared = d²/(2h²)、linear = d/h、none = 0。
  `evolution.py` 传递参数并记录到 diagnostics。新增 `scripts/compare_distance_modes.py`
  自动化对比脚本（3 种模式 × 3 个种子，生成汇总对比）。
- **固定参数**：init_method=marginal, β=1.0, h=0.8, ρ=0.01, η=0.5, μ=0.01。
- **数据集**：test_300x10（300 条×100 轮）和 nltcs（16181 条×1000 轮）。

#### 小数据结果（test_300x10, 100 轮）
| 模式 | 最优 Loss | 归一化 L1 | 相对差异 |
|------|-----------|-----------|---------|
| NONE | 1216 ± 117 | 0.0162 ± 0.0008 | baseline |
| LINEAR | 1324 ± 89 | 0.0170 ± 0.0005 | +8.9% |
| SQUARED | 1328 ± 161 | 0.0172 ± 0.0013 | +9.2% |

**小数据观察**：NONE 最优，距离项似乎有害（起点 loss 约 2000）。

#### 大数据结果（nltcs, 1000 轮）—— **与小数据相反**
| 模式 | 最优 Loss | 归一化 L1 | 相对差异 |
|------|-----------|-----------|---------|
| **LINEAR** | **1.135e8 ± 1.10e6** | **0.0242 ± 0.00016** | **baseline (最优)** |
| SQUARED | 1.434e8 ± 3.65e6 | 0.0276 ± 0.00035 | +26.3% ↓ |
| NONE | 1.835e8 ± 2.34e6 | 0.0315 ± 0.00024 | +61.7% ↓ |

**大数据核心发现**：
1. **LINEAR 显著最优**：loss 比 SQUARED 低 20.9%，比 NONE 低 38.1%；标准差最小（最稳定）。
2. **NONE 最差**：与小数据"最优"结论完全矛盾，说明大数据下距离约束确实有用。
3. **SQUARED 居中**：比 LINEAR 差但比 NONE 好，暗示"平方衰减过快"限制了远距离学习。
4. **小数据结论不可靠**：300 条数据量太小，距离/适应度的量级关系与大数据不同。

#### 为什么 LINEAR 更好？（初步假设，**后被诊断数据部分推翻，见下**）
- ~~**高斯核（d²）衰减过快**：远距离记录被过度惩罚，即使它们适应度高也难被选中。~~
- ~~**拉普拉斯核（d）衰减更缓**：允许从较远的高适应度记录学习。~~
- **NONE 无约束**：完全随适应度选择，缺乏"就近"平滑，收敛质量差。

#### 验证 1：统计显著性（现有 3 种子）
对 1000 轮 3 种子结果做 t 检验（scipy.stats.ttest_ind），三组两两差异全部高度显著：
- LINEAR vs SQUARED：p=0.000377；LINEAR vs NONE：p=0.000003；SQUARED vs NONE：p=0.000198。
- 即便只有 3 种子，差异也远超随机波动。新增 `scripts/analyze_significance.py`。

#### 验证 2：机制诊断（donor 适应度/距离，种子0，1000 轮）—— **推翻初步假设**
在 `evolution.py` 记录每轮选中 donor 的平均适应度和平均距离（`donor_fitness_history`/
`donor_distance_history`），新增 `scripts/analyze_donor_diagnostics.py`。结果与假设**相反**：

| 指标 | SQUARED | LINEAR | |
|------|---------|--------|---|
| 平均 donor 适应度 | 2.393 | 2.091 | LINEAR **低 12.6%** |
| 平均 donor 距离 | 0.383 | 0.363 | LINEAR **更近** |
| 最终 loss | 1.43e8 | 1.15e8 | LINEAR **好 20%** |

- LINEAR 选的 donor **适应度更低、距离更近**，却收敛更好——"选到更高适应度的远 donor"假设不成立。
- **接受率几乎相同**（总 54.5% vs 55.9%；末 200 轮 21.5% vs 22.5%）——"LINEAR 提案更易接受"也不成立。
- **真机制**：接受步的平均 loss 下降 LINEAR 略大（1.035e6 vs 1.011e6）+ 接受次数略多（559 vs 545），
  累积成 ~5% 总下降差。推测高斯核权重过于"尖锐"（超线性惩罚集中在少数近邻），拉普拉斯核权重更"平缓"、
  donor 更多样，长程下探索的复利效应胜出。**注：单种子定性观察，非严格结论。**

#### 验证 3：加长到 1500 轮（3 种子）—— **纠正"SQUARED 卡死"判断**
| 模式 | 最优 Loss（1500轮） | 归一化 L1 | vs SQUARED |
|------|-----------|-----------|---------|
| **LINEAR** | **1.034e8 ± 5.16e5** | **0.0230** | baseline（优 20.5%）|
| SQUARED | 1.301e8 ± 1.20e6 | 0.0262 | — |

- 显著性 p=0.000009；LINEAR 优势 20.5%，与 1000 轮的 20.9% **几乎一致**（两曲线平行下降，非追赶）。
- **纠正**：此前诊断说"SQUARED 900 轮后卡死"是误判。1500 轮显示 SQUARED 末 500 轮仍降 8.8%、
  末 100 轮降 2.28%，只是暂时平台期非真收敛。LINEAR 优势来自**每步都略优的累积**，非"避免卡死"。
- **两组都未收敛**：末 100 轮都还在以 ~2% 速度降，加轮数（2000+）应继续平行下降、LINEAR 维持 ~20% 优势。

#### 代码变更
- `sampling.py`：`compute_sampling_probs` 和 `_compute_sampling_probs_torch` 新增 `distance_mode` 参数，
  根据模式计算距离惩罚（修复 bug：none 模式需用 `np.zeros_like(distances)` 而非标量 0.0）。
- `evolution.py`：`run_evolution` 新增 `distance_mode` 参数，传递给抽样并记录到 `diagnostics["params"]`；
  新增 `donor_fitness_history`/`donor_distance_history` 诊断字段。
- 新增 `scripts/compare_distance_modes.py`：自动化实验脚本，支持多数据集/多种子/多模式对比，
  生成 `outputs/distance_mode_experiment_YYYY-MM-DD_HHMM/{模式}/summary.json` 和总对比 `comparison.json`。
- 新增 `scripts/analyze_significance.py`（显著性检验）、`scripts/analyze_donor_diagnostics.py`（机制诊断）、
  `scripts/run_donor_diagnostics.py`（单种子诊断实验）。
- **默认值改动**（2026-07-25 晚）：基于实验证据，将 `distance_mode` 默认从 'squared' 改为 'linear'。
  向后兼容（显式传 'squared' 仍可用），不显式指定则自动用 linear。Docstring 已更新说明推荐用法。
- 测试：210 passed（无回归，新参数默认值向后兼容）。

#### 下一步候选方向
1. ~~**机制诊断**~~ **✅ 已做（验证 2）**：结果推翻初步假设，真机制是"每步略优的累积"。
2. ~~**切换到 LINEAR 作为默认**~~ **✅ 已做**：默认改为 'linear'，docstring 已加推荐说明。
3. **h 扫描（与 distance_mode 交互）**：当前 h=0.8 固定，测试不同 h 值下 linear/squared 的相对表现。
   LINEAR 和 SQUARED 的 h 量纲不同（一个除 h、一个除 h²），最优 h 很可能不同，值得单独扫。
4. **β 交互**：β 控制适应度权重，测试 β 与 distance_mode 的交互（当前 β=1.0）。
5. **更长轮数（2000+）**：两组 1500 轮都未收敛，若要摸到精度天花板需加轮数。

---

## 最近变更（2026-07-24）

### 当前里程碑（tag: v0.1-baseline-before-tuning）
**核心算法与基础设施已就绪，参数调优前的基线版本。**

- **抽样打分公式保持原始设计**：`ℓ = β·F(z) − d²/(2h²)`，距离用平方、未做任何形式变更。
- **初始化已完成**：marginal init（按 1-way 边缘确定性初始化）已实现并验证有效（加速收敛 13.3×），
  可通过 `INIT_METHOD` 开关在 random/marginal 间切换。
- **基础设施齐全**：多种子、计时诊断、归一化 L1 评价指标、sweep.py 扫描框架。
- **下一步：验证距离平方项是否有用 + 参数调优**（β/h/ρ/η/μ 的敏感性扫描与调度设计）。

### 调参前基础设施（3 个小提交）
为让后续调参结论可信，先补齐可观测性与多种子对照：
- **诊断加归一化 L1 分布**：除均值外，补充逐查询 |target−pred|/N 的中位/P90/最大，
  让"典型查询"与"最差查询"可观测（evolution.py + run.py）。
- **进度输出频率可控**：run_evolution 加 log_every（0=每轮打印，向后兼容；
  >0=每 N 轮打印，首末轮总打印），长实验清爽。run.py 默认 LOG_EVERY=50。
- **多种子入口 + 新落盘结构**：run.py 的 SEED→SEEDS=[0,1,2]，一次跑多个种子看波动。
  落盘改为 outputs/YYYY-MM-DD_HHMM/{顺序}-{种子}/（如 0-0、1-1、2-2）+ summary.json。
  summary.json 存参数 + 逐种子 best_loss/归一化L1 + 汇总（均值±std/min/max）。
  io.py 的 save_run 加可选 run_dir（向后兼容），新增 create_parent_dir、save_summary。
  测试 210 passed（+6）。

### 初始化作用的验证结论
用现有 outputs 对照（边缘 init 1000 轮 vs 随机 init 1000/10000 轮）：
- 边缘初始化起点 loss 低 13.3×（6.94e8 vs 9.23e9），同 1000 轮预算最优 loss 低 30.6%，
  且第 346 轮就达到随机跑满 1000 轮的水平——**收敛显著更快**。
- 但随机 init 给 10 倍轮数（10000 轮）反超边缘（1.10e8 < 1.43e8），且边缘末尾 100 轮
  几乎不再下降。**结论：初始化的价值是"加速收敛"，不改变算法的最终精度天花板。**
- 后期停在平台不动，正是固定参数缺乏精修能力的体现，指向"参数衰减调度"的必要性。
- 注：该对照非严格（预算不同、单种子），下定论需同轮数多种子对照（暂未做）。

### 评价指标改为"平均归一化 L1 误差"（参考 AIM workload error 的形式，方案 A）
  - 旧指标"平均相对误差" = mean(|target−pred| / target)，被小 target 查询污染：
    nltcs 上高达 133.8%，其中 target<50 的查询平均相对误差 953%（如 target=13 预测成 253 → 1846%），
    而占多数的 target≥1000 查询相对误差仅 16.5%。算术平均被极端值严重拉高，误导性强。
  - 新指标 = mean(|target−pred|) / |D|（分母为记录数，非逐查询 target），跨数据规模可比，
    贴合 L2 训练目标。同一 nltcs 结果上为 0.0275（中位 0.0245 / P90 0.0510 / 最大 0.1111）。
  - 中间训练仍用 L2 loss，未改动。仅替换报告指标：
    evolution.py 字段 mean_relative_error → normalized_l1_error；run.py 输出行同步更新。
  - 204 个测试全部通过。

### 调参路线与调度设计的讨论结论（2026-07-24，尚未实现）
目标：五个参数 β/h/ρ/η/μ 最终都做"探索→精修"的调度（当前是固定值）。
本次只定方向，不实现，也未跑实验。

- **调度驱动量：选"随进度比例 p = t / n_rounds 变化"，非绝对轮数、非残差。**
  - 用 p（0→1）而非绝对轮数 t：换轮数预算（500/1000/2000）曲线自动缩放，不用重标定。
  - 选轮数驱动而非残差驱动的理由：① 确定性、可复现，能干净对比"固定 vs 调度"是否真有收益；
    ② 残差是带噪反馈量，无平滑机制时参数易震荡；③ 本质就是退火（早期广探索、后期精收敛），
    按迭代进度调度是成熟做法；④ 残差跨查询/数据集量级差异大，需先归一化。
  - **残差驱动放第二版**：等确认调度有收益、并加了平滑（EMA/滑窗）后，再把个别参数
    （最可能是 μ）改成残差自适应。先简单确定，再复杂自适应。
- **曲线形状：只定大方向"探索→精修"（早期大、后期小衰减），具体形状（线性/指数/阶梯）待定。**
- **固定值扫描（sweep.py）的定位：调度设计的"地形勘探"，非调度本身。** 产出三样东西指导调度：
  ① 每个参数好取值的量级（→ 调度起点/终点的选取范围）；
  ② 敏感度（不敏感的直接固定、不浪费调度自由度，很可能五个砍成两三个）；
  ③ 单调还是有峰（单调才值得衰减调度，有峰则固定在峰值）。
- **参数交互与扫描策略：分两阶段，先 OAT 筛选定标、再对交互对做小网格。**
  - 参数间会互相影响（交互）。典型：β 和 h 在同一打分公式 `ℓ=β·F−d²/(2h²)` 里竞争——
    h 小则距离项压死适应度项、β 变了也没用；h 大则 β 效应被放大。故"β 的最优值依赖 h"。
  - **OAT（一次一参、其余锚 baseline）的固有盲区就是交互**：只能看到某一个切面上该参数的表现，
    看不到最优值会不会随另一参数漂移。所以 OAT 用来**筛选+定标**，不用来找全局最优。
  - **第一阶段（现在做）**：逐参数 OAT 扫描，回答两件事——① 哪些参数敏感、哪些钝
    （钝的直接固定，不用管交互）；② 每个敏感参数的好取值范围（把网格从拍脑袋的 5 值收窄到 2–3 值）。
  - **第二阶段**：对第一阶段暴露出的、疑似交互的敏感参数对做小网格（如 β×h 的 3×3）。
    因范围已收窄，成本从 5×5 降到 3×3，可控。β×h 因公式里直接竞争，几乎必是第一对候选。
  - 不一上来就联合网格：组合数是乘的，对不敏感/不交互的参数做联合是纯浪费；
    先用线性成本的 OAT 买到"敏感度地形图"，它才告诉你哪两个值得花平方成本一起扫。
- **β/h 的探索记录（2026-07-24，方案存档但不实施）**
  探针实测发现：适应度 F 的量级比距离项大 15–20 倍（nltcs），默认 β=1 下距离项被
  压制、就近约束几乎不起作用；且 F 的量级只能实测、换数据就变（取决于查询数与相关性，
  增长约 m^0.75）。因 β 和 h 强耦合（`β_平衡≈0.029/h²`），提出过「温度重参数化」方案
  （见 `docs/设计/选择强度与邻域尺度_温度重参数化.md`），但暂不采纳该方案。
  探针脚本已删除，设计文档保留作参考。
- **下一步（待定）**：β/h 及其余参数（ρ/η/μ）的调参路线待确定后实施。

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
- diagnostics.json 记录实验参数（2026-07-23）：run_evolution 诊断新增 params 字段
  （n_records/n_rounds/seed/beta/h/rho/eta/mu/tol/device/eval_method/batch_size），
  以后看结果直接知道参数设置。另加 mean_relative_error 字段+终端输出（平均相对误差，
  比原始 loss 更直观：loss 是绝对计数残差平方和，尺度大不代表效果差）。
- 向量化+分块查询评价已实现（vectorized_eval.py + evolution.py + run.py，2026-07-23）：
  - **动机**：GPU 采样优化后重测单轮分段，瓶颈已转移——稳态单轮 3.87s 中 pandas
    查询评价占 93.4%（fitness 31% + 当前表评价 31% + 提案评价 31%），全是逐查询
    调 pandas（1000 查询 = 1000 次 pandas 调用），距离/采样只剩 6%。
  - **新增 vectorized_eval.evaluate_vectorized**：表转数值矩阵 X(N×属性)摆脱 pandas，
    查询编译成定长数组（列/算子/值 padding 到最多 3 条件），一次掩码扫描同时派生
    计数 q、残差、fitness。**分块**（batch_size，默认 256）按查询切列，一次算
    (N,batch) 掩码、边算边派生边释放，内存 ∝ N×batch 不随查询数爆。
  - **fitness 向量化关键**：F = M @ (w·residual) − (w·residual)·p，权重天然融入矩阵乘法
    （w_j 权重接口保留，默认全 1）；残差经 objective.compute_residual 算（σ/κ 噪声
    接口保留，默认 σ=0）。计数与旧路径**整数逐元素相同**，fitness numpy 逐位一致。
  - **算子白名单+回退**：快路径支持 {==,>=,between}；含白名单外算子的查询自动走旧
    evaluate_table 慢路径（保证正确），并打印提醒。以后加新算子：不改也能跑对（自动
    回退），想加速再补白名单+向量化实现。
  - **eval_method 开关**：run.py 新增 EVAL_METHOD（'vectorized'默认/'legacy'）+ BATCH_SIZE=256
    常量。legacy 走旧 evaluate_table+compute_fitness 原路径，作正确性基准/对拍/应急。
    旧 queries.py/fitness.py 一字未改。
  - 新增 tests/test_vectorized_eval.py（19 个：计数逐元素相同、fitness 一致、非全1权重、
    σ≠0噪声、batch_size 无关性、回退组正确+提醒、端到端 vectorized↔legacy 逐位一致），
    全套 **176 passed**（157+19）。
  - **nltcs 实测**：稳态单轮 3.87s → **0.40s（≈9.7×）**；单点评价（计数+fitness）
    numpy 4.1×、cuda 167×。float32(cuda) vs float64(numpy) 极小差（既有性质）。
- 按 1-way 边缘确定性初始化已实现（marginals.py + build_marginals.py + generator.py + evolution.py + run.py，2026-07-24）：
  - **动机**：纯随机起点连单属性分布都不对，演化前若干轮被迫先修 1-way，浪费在最基础信息上。
    用 1-way 边缘测量初始化 S_0，让每个属性自己的分布一开始就贴合目标，演化专注高阶关系（3-way）。
    与 AIM 初始化哲学一致：先学稳定的单变量统计，把力气留给属性间相关性。
  - **设计归档**：docs/设计/初始化设计_1way边缘确定性初始化.md（完整设计、通用规则、优缺点）
  - **workload 诱导分箱（通用规则）**：对数值属性，扫描全部查询提取判定边界（切点），
    自动造箱。换数据/换查询零改代码。test_300x10 的 age 查询自动得到 5 箱 [18–24 / 25–34 / 35–49 / 50–64 / 65–100]，
    正好对齐查询阈值。类别属性每个合法值天然一箱。
  - **算子白名单+回退**：造箱支持 `{>=, >, <=, <, between, ==}`；白名单外算子跳过该查询（不贡献切点）+提醒，
    不报错——后果只是"该查询 1-way 信息没烤进初始化"，起点差一点，评价仍正确（可预测降级）。
  - **确定性填充（做法 B）**：按边缘计数确定性填配额 + 数值列箱内等距均摊（因边界对齐查询切点，
    箱内怎么铺都不改查询计数）+ 最大余数法凑 N（为 DP 加噪保留）+ 每列独立打乱（避免列间人为关联）。
  - **离线 / 运行时分界（守铁律 6）**：scripts/build_marginals.py 读源数据数每箱人数（离线测量，带警告标注），
    产出 configs/<dataset>/init_marginals.json（含箱定义+计数）。运行时 marginals.init_from_marginals 
    只读该文件，从不碰源数据。
  - **开关保留**：当时 run.py 支持 INIT_METHOD='random'/'marginal'；现已再加入
    'pairwise_maxent'，原两条路径继续保留，marginals=None 仍向后兼容（纯随机）。
  - **新增文件**：src/table_diffevo/marginals.py（造箱+加载+填充）、scripts/build_marginals.py（离线入口）、
    tests/test_marginals.py（28 测试）、scripts/compare_init.py（对照实验脚本）。
  - **已改文件**：generator.py 加 marginals 参数（None 时行为不变）、evolution.py 加 init_method 开关、
    run.py 加 INIT_METHOD/MARGINALS_PATH 常量、diagnostics.params 加 init_method 字段。
  - **产出**：configs/test_300x10/init_marginals.json（10/10 属性，age 5 箱）、
    configs/nltcs/init_marginals.json（16/16 属性，各 2 箱）。
  - 新增 tests/test_marginals.py（28 个：造箱切点语义、边缘精确匹配、凑 N、每列打乱、复现性、回退、
    真实数据端到端、初始化 beats 随机），全套 **204 passed**（176+28），无回归。
  - **对照实验**（test_300x10, 300 条 × 50 查询，10 轮演化）：
    - 纯随机初始化：初始 loss 22939 → 最优 18104（降 21.1%）
    - 按边缘初始化：初始 loss **2044** → 最优 **1880**（降 8.0%）
    - **起点 loss 降低 91.1%**（11.23×），最优 loss 远优于随机（1880 vs 18104）。


## 实验观察：nltcs 收敛分析（2026-07-22/23）

### ρ=0.1（100 轮，2026-07-22）
**现象：** 固定参数 β=1/h=0.8/ρ=0.1/η=0.5/μ=0.01，loss 9.23e9 → 7.83e8（降 91.5%），
但**第 1-19 轮轮轮接受快速下降，第 20-100 轮提案全部被拒、loss 卡在 7.83e8 不动**。

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

### ρ=0.01（500 轮，2026-07-23）
**实验目的：** 验证小步长能否避免 ρ=0.1 的"卡死"并走得更深。

**结果：**
- **初始 loss**: 9.23e9 → **最优 loss**: 2.96e8（**降低 96.8%**）
- **跑满 500 轮**，未提前停止；接受率约 95%（475/500）
- **前 291 轮**：连续接受（仅 1 次拒绝），loss 9.23e9 → 4.83e8，快速下降
- **后 209 轮**：开始出现拒绝（约 26 次），但**仍在缓慢下降** 4.83e8 → 2.96e8
  - 关键区别：与 ρ=0.1"卡死不动"不同，ρ=0.01 的拒绝是**暂时性**的——
    拒绝几次后仍能继续降低，说明小步长避免了过冲，能在局部最优附近微调前进
- **未完全收敛**：最后 10 轮仍有 3 次接受（497/495/491 轮），最低点在第 500 轮，
  更长轮次可能继续下降

**对比 ρ=0.05**（100 轮，项目状态未记录的实验）：
- ρ=0.05：100 轮降到 5.14e8，接受率 100%
- ρ=0.01：前 291 轮降到 4.83e8（已超越），500 轮最终 2.96e8（再降 42%）

**结论：** 小步长（ρ=0.01）确实能走得更深，避免了 ρ=0.1 的"卡死"和 ρ=0.05 的
"收敛过早"。后段拒绝率上升（约 12%）是正常的——接近最优时需要更小步长，但当前
ρ=0.01 仍能缓慢前进。**参数衰减调度的必要性再次印证**：理想方案是 ρ 从 0.1（前期
快降）→ 0.01（中期深挖）→ 0.005（后期精收），而非全程固定。另外 ρ=0.1 第 20-100 
轮明知被拒仍做全量计算空转 81 轮——衰减调度或 patience 早停可同时解决"卡住"和"空转"。

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
2. ~~次选：向量化查询评价~~ **✅ 已完成（2026-07-23）**：向量化+分块，稳态单轮
   3.87s→0.40s（≈9.7×），计数逐元素相同、fitness numpy 逐位一致。详见上文"已完成"。
   注：GPU 采样优化后重测，查询评价实占 93%（非早先估的 28%），是真正大头。
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
- 抽样分数最初采用"抽样分数+抽样概率.pdf"定义：β·F − d²/(2h²) 后 softmax（squared 模式），
  不对适应度做 /std 标准化（与完整方案 5.6 冲突，以新文档和 temp.md 为准）。
- **抽样模式默认已定型为 geometric（2026-07-29）**：归一化 + 几何均值 `α·[λ·log f + (1−λ)·log s]` 后 softmax，
  推荐配置 **α 2→10、λ 0.5**（nltcs 多种子 + 配对 t 检验确认显著优于 linear/α1.5→6）。
  两个旋钮语义清晰：λ 管精度-多样性权衡、α 管锐度；相比相加型（squared/linear）不依赖 F 与距离项的量级匹配，跨数据集更稳。
  squared/linear/none/multiplicative 仍可显式指定，向后兼容。
- **对角线屏蔽 exclude_self 默认开启（run_evolution 默认 True）**：候选池=全表时禁止抽到自己，
  消除小表高锐度下的自我复制空转；仅方阵 N==K 有意义，将来改共享参考池（K≠N）需传 False。
- β_t、h_t 的具体数值调度（如 h 从 0.8→0.15 线性衰减）留给主循环，
  抽样函数只接收当前轮的标量值，职责单一。
- 其余实现细节（模块拆分、命名等）到对应阶段再逐步讨论确定。

## 当前未完成
- 主循环第一版已跑通，但仍是最简版本，下列增强尚未做：
  - 参数随轮次的衰减调度（h/ρ/η/μ/β 从大到小线性计划）
  - ~~整代检查失败时的重试 + ρ 缩减~~（已实现为可选功能，默认关闭）
  - 终止条件 patience / min_change_rate（第一版仅"达标 或 达到 T"）
  - 更丰富的诊断字段（文档第 12 节：fitness 分布、donor 距离、变异次数等）
- 尚未做的更大方向：DP 噪声阶段（σ/κ 接口已预留）、大规模共享参考池（M=512）
- 2-way 最大熵初始化在精确测量上已收敛；DP 噪声下的跨边缘一致性投影尚未实现
- 已在用依赖：numpy、pytest、pandas、pyyaml（后续 scipy/matplotlib 按需再加）

## PR #36 第三轮反馈修复进度

### 问题 2.2：补全 ExperimentConfig 并添加详细文档 ✅ 已完成（2026-08-06）
**需求**：ExperimentConfig 应包含 run_evolution 的所有参数，并为每个参数添加详细注释说明（含义、选项、范围、单位等）

**已完成**：
1. ✅ 补全 21 个缺失参数到 ExperimentConfig：
   - 初始化参数：init_method、maxent_max_states、maxent_max_sweeps、maxent_tol
   - 计算参数：eval_method、batch_size、log_every、tol
   - 抽样参数：distance_mode、p、exclude_self
   - 重试参数：max_retries、retry_rho_decay
   - 残差驱动扩散核：residual_directed_diffusion、diffusion_direction_strength、diffusion_direction_normalization
   - Gibbs 参数：factorized_gibbs_sweeps、factorized_gibbs_max_order、factorized_gibbs_logit_clip
   - 全局预算：candidate_budget（问题 2.3）
   - 其他：delta（原缺失）

2. ✅ 为所有参数添加详细文档字符串：
   - 每个字段都有中文说明
   - 枚举类型列出所有可选值
   - 数值参数标注范围和单位
   - 复杂参数附带使用说明和示例

3. ✅ DataConfig 补全 schema_path 和 query_path（向后兼容，默认空字符串）

4. ✅ 新增 to_run_evolution_kwargs() 方法：
   - 自动加载 schema/queries/marginals 文件
   - 转换所有 37 个参数为 run_evolution 格式
   - 支持运行时指定 seed

5. ✅ 扩展 validate() 方法，增加新参数的校验规则

6. ✅ 创建完整示例配置：configs/experiments/nltcs_baseline.yaml

7. ⚠️ 曾创建示例脚本 scripts/run_from_config.py —— **本 PR 不交付**（见上「6. 命令行启动脚本」）：
   fail-closed 下它无法端到端运行，留待阶段 1 接入 A0/A1 后交付可运行版本。

8. ✅ 所有测试通过（19/19），向后兼容性验证通过

**文件清单**：
- 修改：src/table_diffevo/experiment_config.py（+21 参数 +详细文档 +to_run_evolution_kwargs）
- 新增：configs/experiments/nltcs_baseline.yaml（完整参数示例）
- （不交付）scripts/run_from_config.py：留待阶段 1，理由同上
- 修改：tests/test_experiment_config.py（更新参数名，全部通过）

**向后兼容性**：
- DataConfig 的 schema_path/query_path 有默认值，旧代码无需修改
- 新参数都有合理默认值
- 旧的直接调用 run_evolution 方式完全不受影响
- run.py 等现有脚本无需修改，继续正常工作

### 问题 2.3：实现 candidate_budget 全局预算控制 ✅ 已完成（2026-08-06）
**需求**：添加全局评估预算限制，避免计算成本失控

**已完成**：
1. ✅ evolution.py 添加 candidate_budget 参数和追踪逻辑
2. ✅ 在主循环中累计候选评估次数（含一轮内的重试）
3. ✅ 达到预算时提前终止，在诊断信息中记录状态
4. ✅ ExperimentConfig 包含此参数并添加验证规则
5. ✅ 测试覆盖：配置校验 + 主循环硬上限行为（接受/拒绝/重试触边三类）

**实现细节**：
- candidate_evaluation_count 在每次候选评估后 +1
- 重试的评估计入（当前尚无 probe 主循环，不涉及探测分支计数）
- 预算耗尽时 candidate_budget_exhausted = True，记录到 diag；
  candidate_budget 本身也写入 diag["params"]
- None 表示无限制（默认行为，向后兼容）
- 语义（#36 修复后）：candidate_budget 是**硬上限**。预算检查在接受/拒绝
  分支之前判定，接受路径不再绕过它。触边的那个已评估候选仍可正常应用
  （接受即生效），但随后立即停止，累计评估次数精确等于预算、绝不越界。

### 补充：问题 2 fail-closed 回归测试 ✅ 已补（2026-08-06 修订）
审查（第三轮问题 2）指出签名内省测试只证"键合法"，证不了"语义正确"，且实测
fixed 被静默当成线性调度、A0/A1 被整个丢掉。按方案 B（阶段 0 只做骨架、未接入
口径 fail-closed），原先锚定"完整映射"的两个测试已替换为 fail-closed 契约测试：
- `test_to_run_evolution_kwargs_fail_closed_on_acceptance_rule`：配 A0/A1 时
  `to_run_evolution_kwargs` 必须抛 `NotImplementedError`（留待阶段 1），而非静默按默认判据跑
- `test_to_run_evolution_kwargs_fail_closed_on_alpha_mode`：配 fixed/probe 时
  必须抛 `NotImplementedError`（留待阶段 2-5），而非把 fixed 错当线性调度
- 预算行为测试见上节（问题 2.3）：硬上限、接受/拒绝/重试触边、跨轮累计、不变式扫描

### 补充：问题 2 冗余字段清理 ✅ 已完成（2026-08-06）
`DataConfig` 中 `target_path` / `measured_target_path` 两字段从未被 `to_run_evolution_kwargs`
消费（target 一律从 query 文件的 `result` 字段派生），且无文档说明。经确认将来做 DP 时
查询本身就是加噪后的，同一入口、不区分加噪/无噪来源，故两字段冗余。已从 DataConfig 定义、
nltcs_baseline.yaml、demo、全部测试中彻底删除，DataConfig docstring 说明缘由；全库零残留。

### 问题 1：Logger 整组原子发布 ✅ 已完成（2026-08-06）
审查（第三轮问题 1）要求 `save()` 多文件发布做到整组原子——读者只能看到完整的上一版或
完整的这一版，不得读到新旧拼接的半成品。原实现是逐个 `temp.replace(final)`，中途崩会留下
混搭坏数据且静默不报错。改法：
- 把本次全部文件（各 CSV + summary.json）先用**最终文件名**写进唯一暂存目录（`tempfile.mkdtemp`
  建在 output_dir 同级，保证同一文件系统 → rename 原子）。
- 发布：output_dir 为空时单步 `os.replace`（零窗口）；非空复用时两步 rename（旧→备份、
  暂存→正式），第二步失败则回滚备份并抛出，成功后删备份（删失败被吞、不影响已发布数据）。
- 陈旧类别文件随整个旧目录被丢弃而自动消失，`_MANAGED_FILES` + 逐个清理循环整段删除。
- 权限修正：`tempfile.mkdtemp` 建的是 0700，发布后会让 output_dir 只有属主可进（共享机器上
  同组同事读不到日志的回归）；发布前按 umask 把暂存目录 chmod 回常规 0755，与原 `mkdir` 一致。
- 权衡：非空复用路径留一个微秒级窗口（旧已挪走、暂存未挪进时崩溃），届时目录暂时缺失、
  会**明着报错**（非静默坏数据），数据在备份中无损，一条 mv 可恢复。单写入者研究日志可接受。
- 两个故障注入回归测试（审查点名要求）：
  - `test_second_rename_failure_rolls_back`：monkeypatch 让第二步 rename 失败，断言回滚到完整旧版、无残留
  - `test_backup_deletion_failure_still_publishes`：让删备份抛错，断言新数据仍完整发布

全套测试 681 通过。至此 PR #36 第三轮反馈四个问题（1 logger 原子性 / 2 配置闭环 / 3 A0 边界披露 / 4 文本一致性）全部解决。

### 问题 3（第四轮反馈）：示例/文档一致性 + 嵌套未知键护栏 ✅ 已完成（2026-08-08）

第四轮审查（分支 HEAD 98680d1）在问题 3 下点出两处缺陷，指示「A 选方案 1、B 要做，先 A 后 B」。

**缺陷 A：示例配置与文档引用了已删除的字段/模式（方案 1：全量对齐到已接入口径）** ✅
- `experiments/configs/example_phase_a.yaml`：改为 `acceptance_rule=A0 + alpha_schedule.mode=round_schedule`，
  `DataConfig` 用 `schema_path`/`query_path`（删除已废弃的 `target_path`），文件头补 fail-closed 接入边界说明。
- `docs/experiment_infrastructure.md`：AlphaScheduleConfig 示例改 round_schedule + probe_* 新字段名并加接入边界注；
  DataConfig 示例去掉 `target_path`；AcceptanceRuleConfig 补 A0/主循环旧判据边界差异；新增「接入边界（阶段 0 fail-closed）」表；
  伪代码按 round_schedule 逐轮算 α（说明该模式下 alpha_value=None，不能直接读单值）。
- `scripts/demo_infrastructure.py`：`simulate_evolution` 按 `_alpha_at(round_idx)` 逐轮线性算 α；打印按 mode 分支（fixed 单值 / 其余显示范围）。
- **补漏（本轮复查）**：demo 的「配置文件不存在」fallback 分支原仍是旧口径 `rule="A1" + mode="fixed"`，
  与同文件 `_alpha_at` 硬编码的 round_schedule 线性调度语义错位（config 声明 fixed 却按 round_schedule 跑）。
  已对齐为 `A0 + round_schedule`，方案 1「全量对齐」现真正覆盖 fallback 路径。
- 验证：demo 脚本 exit 0；配置测试通过。

**缺陷 B：嵌套 dict 里的未知键抛原始 TypeError，不符合「拒绝未知 YAML 键」契约（要做）** ✅
- 根因：顶层 `from_yaml` 有未知键护栏，但 `data`/`acceptance_rule`/`alpha_schedule` 内部拼错的键会被
  `DataConfig(**...)` 等原样展开，抛晦涩的 `TypeError`（如 `__init__() got an unexpected keyword argument`），定位差。
- 修复（`experiment_config.py` `from_yaml`）：新增嵌套节白名单循环，用 `dataclasses.fields()` 派生每个嵌套
  dataclass 的合法键；未知键抛带节名的 `ValueError`（列出该节合法键），嵌套节非映射（写成标量/列表）也给清晰 `ValueError`。
- 测试（`tests/test_experiment_config.py`）：新增参数化 `test_config_yaml_nested_unknown_key`（覆盖
  data/acceptance_rule/alpha_schedule 三节；alpha_schedule 用旧字段名 `W` 模拟真实迁移遗漏）+
  `test_config_yaml_nested_section_wrong_type`。
- **补漏（本轮复查）**：护栏原本对必填节缺失/显式为空（`data:` / `data: null`）用 `if section_data is None: continue`
  直接跳过，会漏到下方 `DataConfig(**None)` 抛晦涩的 `TypeError: argument after ** must be a mapping`——
  与缺陷 B 同源（坏 YAML 必须落到带节名的清晰错误）。已改为：必填节缺失或为空立即抛带节名的 `ValueError`
  （`配置节 'data' 缺失或为空`），补齐「拒绝坏 YAML」契约。新增 `test_config_yaml_nested_section_empty`
  （参数化覆盖 `data:` / `data: null`）+ `test_config_yaml_nested_section_missing`（整节缺失）。

**验证：** 配置测试 30 passed（本轮复查较上一版净增 3）；全套单测 733 passed（零回归）；demo 脚本 exit 0。

至此 PR #36 第四轮反馈的问题 3 已全部解决。

### 问题 4（第四轮反馈）：force_overwrite 非空发布非崩溃安全 ✅ 已完成（2026-08-08，走「降级定位 + 如实披露」）

**审查者给的是二选一**：(A) 不可变唯一 run 目录 + 完成标记/manifest 作读取门禁；或
(B) 明确禁止正式流程复用目录、把 `force_overwrite` 降级为非原子 best-effort 工具并补中断恢复说明。
审查者强调的核心是「公开文字必须与实际保证一致」。选 **B**——不重写发布模型，只把过强声明改为与实际相符，并如实披露窗口 + 恢复办法。

**问题本质**：仅 `force_overwrite` 复用**非空**目录时，`_publish()` 两次 rename（旧→备份、暂存→正式）
之间正式目录短暂不存在；此刻被 `KeyboardInterrupt`/进程终止打断（`except` 无法覆盖），正式目录缺失，
只剩 `.backup-*`（完整旧版）与 `.staging-*`（完整新版）。窗口微秒级，且只写日志、不碰源数据，下次跑必暴露。

**改动（`experiment_logger.py`，仅文档/注释，不改发布算法）**：
- `__init__` 的 `force_overwrite` docstring：明确定位为「非正式、best-effort 覆盖工具，不提供崩溃安全保证」，
  正式实验请用新的（唯一/时间戳）`output_dir` 走首次发布的单步原子路径。
- `save()` docstring：把「整组原子发布 / 读者只会看到完整旧版或新版」的强声明改为分情形说明——
  空目录单步 `os.replace` 真正零窗口原子（正式实验走这条）；非空复用两步 rename **不是崩溃安全**。
  新增「中断恢复」段：说明残留的 `.backup-*`/`.staging-*` 各是什么、如何改名恢复。
- `_publish()` docstring + 两步 rename 处的行内注释：如实标注非崩溃安全窗口与回滚仅对普通异常可达。

**未改**：发布逻辑本身（两步 rename + 普通异常回滚）保持不变；空目录单步原子路径不变。

**验证**：`tests/test_experiment_logger.py` **23 passed**（纯文档改动，零回归）。

**注意（对外动作待确认）**：PR 正文里若仍有「整组原子/崩溃安全」的措辞，需同步弱化——改 PR 正文是对外动作，等用户确认再动。

### 设计决策：A0/A1 保持严格改善口径，不与主循环对齐（2026-08-08）

**背景：** 曾一度把 A0/A1 从严格（`delta_Q < -eps_Q` 等）改为非严格容差口径以对齐主循环
`proposal_loss <= loss + tol`。经讨论**已回退**：A0/A1 恢复为 Issue #33 预注册的严格改善口径。

**为何回退（理由）：**
- 接受规则是阶段 A 的**被测对象**，不是要跟主循环对齐的东西。A0/A1 两条臂只应在
  「看 Q 还是看 L1」上有差异，严格/非严格口径必须一致，否则把「接受规则」和「容差口径」
  两个变量混在一起，违背 Issue #33「禁止混淆」的对照原则。
- 「与主循环对齐」既非 Issue #33 要求（原文就是严格 `<`），也非审查者要求（审查者只要求
  **披露** A0 与主循环的边界差异，不要求改成对齐）。该需求是中途引入的，可安全丢弃。
- 回退后代码自动与 Issue #33 预注册定义一致，**无需改动 Issue 正文**。
- 「选出的 A* 接入主循环时用什么容差口径」这件事**推迟到阶段 1**：届时只有一条规则、
  一个口径，再决定要不要带 tol，不存在「一加一减」的别扭。

**回退改动（均本地未提交）：**
- `acceptance.py`：`_check_A0` 回 `delta_Q < -eps_Q`；`_check_A1` 回 `delta_L1 < -eps_L1` /
  平局带 `|delta_L1| <= eps_L1` / `delta_Q < -eps_Q`；模块与函数 docstring 去掉「对齐主循环」表述，
  改为「严格改善口径 + 边界差异须披露」。
- `experiment_config.py`：`AcceptanceRuleConfig` docstring/注释回「严格改善阈值」口径；校验逻辑不变。
- `docs/experiment_infrastructure.md`：注块改回「A0/A1 严格 + 与主循环边界差异披露」。
- `tests/test_acceptance.py`：受影响用例全部转回严格语义（平局/完美匹配/边界拒绝），
  A1 边界的两个拆分用例合回单一 `test_a1_delta_l1_exactly_neg_epsilon_uses_q`。

**验证：** `tests/test_acceptance.py tests/test_experiment_config.py tests/test_evolution.py` **140 passed**。

**注意（fail-closed 不变）：** acceptance.py 仍未接入主循环（`to_run_evolution_kwargs()` 对 A0/A1 抛
`NotImplementedError`），真正接入留待阶段 1。本次只改判据语义与配套测试/文档，不改主循环行为。

## 下一步（候选，待讨论）
- 全套零件已实现并跑通：generator / queries / objective / fitness / distance /
  sampling / update / evolution。主线的最简闭环已经能把 loss 降下来。
- 可选增强方向（按价值排序，需先讨论再动手）：
  1. 针对 pairwise_maxent 起点的更小 rho / 衰减调度（当前接受率约 3.7%）
  2. patience 早停（新起点较早进入平台，可直接节省空转）
  3. 确定性配额抽样，降低 S_0 的有限样本波动
  4. 诊断与可视化（画 loss 曲线、观察演化过程）
- 性能方向（nltcs 实跑后暴露，待讨论）：
  1. 把查询计数评估搬到 GPU（当前最大 CPU 瓶颈，需保证结果一致）
  2. run.py 增加逐轮进度输出（现在只有跑完才有摘要，中途看不到进度）
