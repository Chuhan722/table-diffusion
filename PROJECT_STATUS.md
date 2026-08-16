# 项目进度

## 当前阶段

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
