# Table Diffusion

当前研究入口是 **`resevo`：残差驱动的有限种群概率演化**。新核与旧实现已经放在同一
代码树中，不需要为了使用新核切换到独立研究分支。

| 用途 | 入口 | 状态 |
|---|---|---|
| 新方案的基础生成 | `python -m resevo` / `resevo-generate` | 只接受显式公开 schema、记录数和已提供查询目标 |
| 新方案的数学核与可选研究模块 | `src/resevo/` | CPU、CuPy GPU；附加策略需单独说明适用边界 |
| 旧路线与历史复现 | `src/table_diffevo/`、`scripts/run.py` | 保留既有接口和默认行为，不被新核静默替换 |
| 研究分支历史实验适配器 | `scripts/run_plants.py` 等 | 仅用于已声明条件的基准复现，仍有原表元数据依赖，不是推荐的发布入口 |

本仓库不是已经完成认证的端到端 DP 系统。给定合规 DP 统计量后的后处理，不代表
上游选择、测量、schema、公开记录数和发布产物已经满足完整隐私要求。

## 安装

项目声明 Python **3.11+**。在自己的虚拟环境中：

```bash
python -m pip install -e '.[dev]'
```

新核 CPU 依赖 NumPy/SciPy。CUDA 12 环境可选择安装 CuPy：

```bash
python -m pip install -e '.[dev,resevo-gpu]'
```

GPU 运行仍需可用的 NVIDIA 驱动、匹配的 CUDA 运行时及 NVRTC；缺少这些条件会明确报错，
不会偷偷退回 CPU。不要在共享机器上直接修改别人的环境。

## 不读取原表的最小运行

仓库提供一个手工构造的公开二元示例；它不来自训练或测试记录：

```bash
python -m resevo \
  --schema configs/resevo/smoke_schema.json \
  --queries configs/resevo/smoke_queries.json \
  --rows 64 --rounds 100 --seed 7 \
  --output outputs/resevo/smoke.csv
```

从源码运行而未安装时，在命令前加 `PYTHONPATH=src`。
GPU 版本加 `--backend gpu`，并由 `CUDA_VISIBLE_DEVICES=0` 显式指定可用设备。
输出路径必须不存在；生成器不会覆盖原来的表。

schema 格式为 `{"fields": [...], "domains": [[...], ...]}`，字段名和域值均为字符串，
每个域非空且无重复。查询文件沿用 `resevo.dataset.load_queries` 的计数格式：
`result_unit="records"`、`record_count` 与 `--rows` 一致，`queries` 中每题给出
`id`、`conditions` 和 `result`。

基础入口使用固定单位权重，默认从完整二阶目标导出一阶初始化；查询覆盖不足或带噪边缘
不一致时不会自动猜测修复方式，可明确选择 `--init uniform`。它不自动做逆方差加权、
清洗、结构先验、配对救援、纤维第二阶段、噪声地板或快照挑选。

`alpha=0.5`、`stay=0.9`、`damping=1`。阻尼必须有限且处于 `(0,1]`；
`--max-expected-rows` 必须为正有限数，省略表示无该预算。
`--retries`、`--work-rows`、`--work-below`、`--gpu-entry-budget` 是显式可选控制项，
默认都为 0。种子流固定为初始化 `seed`、菜单 `100000+seed`、
工作批选择 `200000+seed`、落表 `300000+seed`。

默认输出是停止时的当前表；启用下述硬后处理时交付其固定预算终点表，都不是用真实答案选出的最优历史表。原始训练/测试表只应在生成
完成之后用于独立离线评价，不能送回初始化、运行调度或停止选择。

发生已识别的 `numerical_indeterminate` 时，程序仍异常退出且不发布正常终表，
但会在输出文件旁创建唯一的 `*.failed-*` 诊断目录，保存 `current.csv`、
`records.csv` 和标明失败原因的 `failure.json`；不覆盖此前失败。只有有效的
`failure.json` 才表示诊断完整写入。快照是失败点之前的当前合成表，不是合格终表，
也不包含恢复所需的全部随机数/提供器状态，不能当作自动续跑点。保存失败同样报错。
这包括 β 校准耗尽括根次数的错误：保存最新进度后仍失败退出，不自动切换冷启动重试。

正且有限的外部参数不保证派生核可表示：零/非有限最终步长、非法最终概率和非有限
预测账本会在采样前按数值中断处理；真实无正方向仍按原规则冻结。
范围与稳定账本公式见 [最终核数值后置条件](docs/设计/最终核数值后置条件.md)。
可选 beta 网格诊断的数值失败与主核分开：受限点告警并记录原因，全部不可用时
明确标记诊断不可用，不中断有效主核，也不把无效诊断值当作零或反馈给生成。

采样采用 `numpy-finite-precision-v1` 合同。若所有活跃段的正非保持质量都被保持项
舍入掩盖，采样前数值中断；部分受限则告警并在返回结果/CLI JSON/曲线中记录，
不暗中抬高概率。批量入口恢复可检测的正参数，但不承诺任意微小概率严格可采样；
没有告警也不是所有尾部均保真的证明。详见 [有限精度采样合同](docs/设计/有限精度采样合同.md)。

## 可选硬纤维后处理

`--hard-fiber-steps K` 显式启用生成结束后的CPU后处理，默认0不执行、不增加随机调用，
默认输出格式和原有四条随机流保持。它严格保住第一段合成表的**完整查询字段集合边缘**
和全部一阶，不是只固定卷上出现的格子，也不是继续拟合带噪目标或伪目标。

```bash
python -m resevo \
  --schema configs/resevo/smoke_schema.json \
  --queries configs/resevo/smoke_queries.json \
  --rows 64 --rounds 100 --seed 7 \
  --hard-fiber-steps 20000 --hard-fiber-seed 7301 \
  --output outputs/resevo/smoke-hard.csv
```

此两字段全边缘例只演示入口，联合直方图已经被固定，可以只发生行序变化；
不能用它宣称解冻或质量提升。四环反例与目标分布说明见
[有限表合同与连通性](docs/设计/纤维第二阶段有限表合同与连通性.md)。

K是含显式自环的马尔可夫步数，不是接受次数、整表扫描次数或混合证书。
N>=4时以1/2保持、1/4单次交换、1/4四行原子双交换；N=2/3只用保持和单次交换，
N=1明确报告`no_row_pairs`且不消耗后处理随机流。查询约束不满足的提议也成为自环，
没有按损失接受、最优回滚或接受率早停。

仅支持等值查询；输入题号以`P2_`开头也不忽略。常量题可以与含字段的等值题混合，
但现有第一阶段对只含常量的卷有独立缺陷（#93），启用此选项时会在生成前明确拒绝。
库函数 `resevo.hard_fiber.hard_fiber_sample` 可对已有合成表只固定一阶，不依赖该入口限制。
旧 `run_plants.py` 的软正则路线保持原样，没有将其重新命名为硬锁。

后处理使用新的随机生成器，默认种子`400000+seed`，可用`--hard-fiber-seed`单独覆盖。
GPU生成之后也在CPU执行此阶段；JSON新增的`hard_fiber`对象单独记录后端、种子、
保持集合、实际步数、约束拒绝、恒等／仅换行序／直方图变化次数。
原有`rounds`、`stop_reason`、`backend`和采样精度信息仍描述第一段。
完整边缘复核失败会报错，不把第一段表作为成功结果代替输出。

理想对称提议保持有序可行表的均匀测度，诱导直方图质量与逆阶乘成正比；
不是均匀直方图、独立样本或逐步最大熵优化。
不保证所有纤维连通、有限预算充分混合、未测真实关系改善或端到端DP。
稀疏整数计数不分配值域乘积张量，但完整复核成本仍随行数和约束集合增长。

## 验证

```bash
PYTHONPATH=src python scripts/run_anchor_audit.py
PYTHONPATH=src python -m pytest tests/test_resevo_step_parameters.py tests/test_resevo_cli.py -q
PYTHONPATH=src python -m pytest tests/test_hard_fiber.py tests/test_hard_fiber_cli.py -q
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python -m pytest -q
```

GPU 测试在无 CuPy/可用设备时会跳过；部分历史集成测试需要未入库的实验产物。
必须查看 skip 原因，不能把未运行场景写成通过。

迁移来源、保证范围和已知后续事项见
[残差演化主线迁移与边界](docs/设计/残差演化主线迁移与边界.md)；
小例精确数值见 [锚点数据](docs/锚点数据.md)，阶段事实见 `PROJECT_STATUS.md`。