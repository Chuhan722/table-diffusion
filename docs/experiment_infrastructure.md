# 实验基础设施使用指南

## 概述

本文档说明如何使用实验基础设施模块进行接受规则对照实验和 α 调度实验。

阶段 0 提供了三个核心模块：
1. **度量计算** (`metrics.py`) - 统一的 L1、Q、残差计算
2. **实验日志** (`experiment_logger.py`) - 结构化日志记录
3. **实验配置** (`experiment_config.py`) - YAML 配置管理

---

## 度量计算 (metrics.py)

### 可用函数

#### `compute_normalized_l1(target, current, n_records)`
计算归一化 L1 误差（与 `evolution.py:904` 完全一致）。

**公式**：`normalized_l1 = mean(|target - current|) / n_records`

**参数**：
- `target`: 目标计数向量 (np.ndarray)
- `current`: 当前合成表的查询答案 (np.ndarray)
- `n_records`: 记录总数 N (int)

**返回**：`float`，范围 [0, 1]

#### `compute_squared_loss(target, current)`
计算平方 loss Q（wrapper for `objective.compute_loss`）。

**公式**：`Q = ½ Σ(target - current)²`

**参数**：
- `target`: 目标计数向量 (np.ndarray)
- `current`: 当前合成表的查询答案 (np.ndarray)

**返回**：`float`，≥ 0

#### `compute_all_metrics(target, current, n_records)`
一次性计算所有度量，避免重复计算。

**返回**：`(normalized_l1, squared_loss, residual)`
- `normalized_l1`: float
- `squared_loss`: float
- `residual`: np.ndarray（比例残差向量）

### 使用示例

```python
import numpy as np
from table_diffevo.metrics import compute_all_metrics

target = np.array([180, 95, 42])
current = np.array([170, 100, 42])
n_records = 300

l1, q, residual = compute_all_metrics(target, current, n_records)

print(f"Normalized L1: {l1:.6f}")
print(f"Squared Loss Q: {q:.2f}")
print(f"Residual: {residual}")
```

### 与现有代码的一致性保证

所有度量计算与 `evolution.py` 和 `objective.py` 完全一致：
- `compute_normalized_l1` 复制自 `evolution.py:904` 的逻辑
- `compute_squared_loss` 直接调用 `objective.compute_loss`
- 单元测试验证了计算结果的一致性

**验证方法**：
```bash
python -m pytest tests/test_metrics.py -v
```

---

## 实验日志 (experiment_logger.py)

### 日志层级

实验日志分为三个层级：

#### 1. 每轮日志 (RoundLog)
记录每一轮的详细信息：
- `seed`: 随机种子
- `arm`: 实验臂名称（如 "A0", "A1", "B2"）
- `round`: 轮数
- `block`: 当前块编号
- `alpha`: 当前 α 值
- `u`: 归一化 α（范围 [0, 1]）
- `L1_current`: 当前轮的 L1
- `best_L1`: 迄今最佳 L1
- `Q_current`: 当前轮的 Q
- `accepted`: 是否接受候选
- `delta_L1`: L1 变化量
- `delta_Q`: Q 变化量
- `candidate_evaluations`: 累计候选评估次数

#### 2. 每块日志 (BlockLog)
记录每个块的汇总信息：
- `seed`, `arm`, `block`: 标识
- `block_start_L1`, `block_end_L1`: 块开始/结束的 best_L1
- `block_improvement`: 块内改善量（正数表示改善）
- `acceptance_rate`: 块内接受率
- `stall_count`: 当前停滞计数
- `cooldown_remaining`: 剩余冷却块数
- `probe_triggered`: 是否触发探测

#### 3. 探测日志 (ProbeLog)
记录探测分支的详细信息：
- `seed`, `arm`, `probe_id`: 标识
- `checkpoint_block`: 触发探测时的块编号
- `checkpoint_L1`: checkpoint 时的 best_L1
- `direction`: "DOWN", "HOLD", "UP"
- `branch_seed`: 分支随机种子（可选）
- `branch_budget`: 分支消耗的候选评估数
- `branch_final_L1`, `branch_final_Q`: 分支最终结果
- `winner`: 是否为获胜分支
- `winner_reason`: 获胜原因（仅获胜分支有值）

### 使用示例

```python
from pathlib import Path
from table_diffevo.experiment_logger import ExperimentLogger

# 创建日志记录器
logger = ExperimentLogger(Path("experiments/results/test"))

# 记录每轮
logger.log_round(
    seed=42, arm="A0", round=1, block=0,
    alpha=2.0, u=0.0, L1_current=0.1, best_L1=0.1,
    Q_current=5000.0, accepted=True,
    delta_L1=0.0, delta_Q=0.0, candidate_evaluations=100
)

# 记录每块
logger.log_block(
    seed=42, arm="A0", block=0,
    block_start_L1=0.1, block_end_L1=0.09,
    block_improvement=0.01, acceptance_rate=0.8,
    stall_count=0, cooldown_remaining=0,
    probe_triggered=False
)

# 添加统计信息
logger.add_stat("final_best_L1", 0.05)
logger.add_stat("total_rounds", 200)

# 保存到文件
logger.save()
```

### 输出格式

调用 `logger.save()` 后，会生成以下文件：

- **`rounds.csv`**: 每轮详细日志（CSV 格式，易于分析）
- **`blocks.csv`**: 每块汇总日志
- **`probes.csv`**: 探测详细日志（仅 probe 模式有此文件）
- **`summary.json`**: 统计信息（JSON 格式）

**CSV 文件示例**（rounds.csv）：
```csv
seed,arm,round,block,alpha,u,L1_current,best_L1,Q_current,accepted,delta_L1,delta_Q,candidate_evaluations
42,A0,1,0,2.0,0.0,0.1,0.1,5000.0,True,0.0,0.0,100
42,A0,2,0,2.0,0.0,0.09,0.09,4500.0,True,-0.01,-500.0,200
```

**JSON 文件示例**（summary.json）：
```json
{
  "final_best_L1": 0.05,
  "total_rounds": 200
}
```

---

## 实验配置 (experiment_config.py)

### 配置结构

实验配置由四个部分组成：

#### 1. 数据配置 (DataConfig)
```yaml
data:
  dataset_name: "nltcs"
  target_path: "configs/nltcs/measured_1000query.json"
  measured_target_path: "configs/nltcs/measured_1000query.json"
  init_marginals_path: "configs/nltcs/init_marginals.json"
  n_records: 16181
```

#### 2. 接受规则配置 (AcceptanceRuleConfig)
```yaml
acceptance_rule:
  rule: "A1"  # "A0" 或 "A1"
  eps_L1: 1.0e-5  # A1 需要
  eps_Q: 0.0
```

#### 3. α 调度配置 (AlphaScheduleConfig)
```yaml
alpha_schedule:
  mode: "fixed"  # "fixed", "round_schedule", "probe"
  alpha_value: 5.0  # fixed 模式需要
  alpha_min: 2.0
  alpha_max: 10.0
  # probe 模式参数（可选）
  W: 20  # 块大小
  P: 3   # 停滞触发块数
  H: 2   # 探测分支块数
  s: 0.10  # 归一化步长
  C: 2   # 冷却块数
```

#### 4. 实验参数
```yaml
seeds: [42, 43, 44]
n_rounds: 200
output_dir: "experiments/results/phase_a_pilot"

# 演化参数
beta: 1.0
eta: 0.5
h: 0.8
mu: 0.01
lambda_: 0.5
delta: 0.05
winsorize_limits: [0.01, 0.99]
```

### YAML 配置示例

完整配置文件示例见 `experiments/configs/example_phase_a.yaml`。

### 使用示例

```python
from pathlib import Path
from table_diffevo.experiment_config import ExperimentConfig

# 加载配置
config = ExperimentConfig.from_yaml(Path("experiments/configs/example_phase_a.yaml"))

# 配置会自动验证
# 若验证失败，会抛出 ValueError 并列出所有错误

# 访问配置
print(f"实验名称: {config.experiment_name}")
print(f"接受规则: {config.acceptance_rule.rule}")
print(f"α 模式: {config.alpha_schedule.mode}")
print(f"种子: {config.seeds}")

# 保存配置（可用于记录实际运行的参数）
config.to_yaml(Path("experiments/results/actual_config.yaml"))
```

### 参数验证规则

配置加载时会自动验证以下规则：

1. **接受规则验证**：
   - A1 必须指定 `eps_L1`
   - A0 只需要 `eps_Q`

2. **α 调度验证**：
   - `fixed` 模式必须指定 `alpha_value`
   - `probe` 模式必须指定 `W`
   - `alpha_min` 必须小于 `alpha_max`

3. **实验参数验证**：
   - `seeds` 列表至少包含 1 个种子
   - `n_rounds` 必须为正数

**验证失败示例**：
```python
config = ExperimentConfig.from_yaml(Path("bad_config.yaml"))
# ValueError: 配置验证失败:
#   - A1 需要指定 eps_L1
#   - fixed 模式需要指定 alpha_value
```

---

## 完整工作流示例

```python
from pathlib import Path
import numpy as np
from table_diffevo.experiment_config import ExperimentConfig
from table_diffevo.experiment_logger import ExperimentLogger
from table_diffevo.metrics import compute_all_metrics

# 1. 加载配置
config = ExperimentConfig.from_yaml(Path("experiments/configs/example_phase_a.yaml"))

# 2. 创建日志记录器
logger = ExperimentLogger(Path(config.output_dir))

# 3. 运行实验（伪代码）
for seed in config.seeds:
    for round in range(config.n_rounds):
        # 演化一轮...
        target = ...  # 从配置加载
        current = ...  # 当前合成表的查询答案

        # 计算度量
        l1, q, residual = compute_all_metrics(target, current, config.data.n_records)

        # 记录日志
        logger.log_round(
            seed=seed,
            arm=config.acceptance_rule.rule,
            round=round,
            block=round // 10,  # 假设 10 轮为一块
            alpha=config.alpha_schedule.alpha_value,
            u=(config.alpha_schedule.alpha_value - config.alpha_schedule.alpha_min) /
              (config.alpha_schedule.alpha_max - config.alpha_schedule.alpha_min),
            L1_current=l1,
            best_L1=...,  # 维护最佳值
            Q_current=q,
            accepted=...,  # 根据接受规则判断
            delta_L1=...,
            delta_Q=...,
            candidate_evaluations=round * 100
        )

# 4. 保存日志
logger.add_stat("config", config.experiment_name)
logger.save()
```

---

## 测试

所有模块都有完整的单元测试：

```bash
# 测试度量计算
python -m pytest tests/test_metrics.py -v

# 测试日志记录
python -m pytest tests/test_experiment_logger.py -v

# 测试配置管理
python -m pytest tests/test_experiment_config.py -v

# 运行所有测试
python -m pytest tests/test_metrics.py tests/test_experiment_logger.py tests/test_experiment_config.py -v
```

---

## 下一步

阶段 0 完成后，后续阶段将使用这些基础设施：

- **阶段 1**：把已实现的 A0/A1 接受规则（`acceptance.py`）集成到 `evolution.py`
- **阶段 2**：运行 A0 vs A1 对照实验，使用 `ExperimentLogger` 记录结果
- **阶段 3**：扫描固定 α，使用 `ExperimentConfig` 管理参数
- **阶段 4**：实现探测式 α 控制器
- **阶段 5**：运行 B0 vs B1 vs B2 正式对照实验
