# 阶段 3 实现计划：固定 α 基线 (B0) 与 W 校准

## 目标
1. 扫描固定 α ∈ {2, 4, 6, 8, 10}，在开发集（3种子）上找出最优 α*
2. 根据改善时间尺度校准块大小 W（用于阶段 4 探测机制）
3. 在测试集（10种子）上建立 B0 基线（固定 α*），作为科学对照

## 当前基础设施

### 已有组件
- ✅ `ExperimentConfig`: 支持 `alpha_schedule.mode = "fixed"` + `alpha_value`
- ✅ `ExperimentLogger`: 支持 round/block/probe 三级日志
- ✅ `evolution.py`: α 调度在 line 563: `alpha_t = alpha_min + (alpha_max - alpha_min) * progress`
- ✅ YAML 配置模板: `experiments/configs/example_phase_a.yaml`
- ✅ 实验脚本模板: `experiments/nltcs_pilot_acceptance.py`

### α 调度修改点
当前 `evolution.py` 的 α 计算逻辑：
```python
# Line 563
alpha_t = alpha_min + (alpha_max - alpha_min) * progress
```
需要支持：
- **固定模式**: `alpha_t = alpha_value` (常数)
- **轮次模式**: `alpha_t = alpha_min + (alpha_max - alpha_min) * progress` (当前默认)
- **探测模式**: 由外部 probe 逻辑控制（阶段 4）

## 实现方案

### 步骤 3.1: 修改 `evolution.py` 支持固定 α
**文件**: `src/table_diffevo/evolution.py`

**修改位置**: line 563 附近

**当前代码**:
```python
alpha_t = alpha_min + (alpha_max - alpha_min) * progress
```

**修改后**:
```python
# 根据 alpha_schedule_mode 决定 α 计算方式
if alpha_schedule_mode == "fixed":
    alpha_t = alpha_value
elif alpha_schedule_mode == "round_schedule":
    alpha_t = alpha_min + (alpha_max - alpha_min) * progress
else:  # probe 模式由外部控制
    alpha_t = alpha_value  # 临时值，后续阶段 4 实现
```

**需要添加的函数参数**:
- `alpha_schedule_mode: str = "round_schedule"` (默认保持当前行为)
- `alpha_value: Optional[float] = None` (fixed 模式必需)

**向后兼容**: 默认参数保持当前行为，现有实验不受影响

---

### 步骤 3.2: 创建固定 α 扫描脚本
**文件**: `experiments/phase_b_fixed_alpha_scan.py`

**功能**:
- 在 nltcs 数据集上运行 5 个固定 α 值: {2, 4, 6, 8, 10}
- 使用开发种子: [42, 43, 44]
- 每个配置运行 500 轮
- 记录最终 L1 误差、Q 损失、接受率

**输出**:
- `experiments/results/phase_b_alpha_scan/summary.csv`: 汇总表
- `experiments/results/phase_b_alpha_scan/rounds.csv`: 逐轮详细日志
- `experiments/results/phase_b_alpha_scan/alpha_comparison.png`: α vs L1 曲线图

---

### 步骤 3.3: 分析结果选择 α*
**文件**: `experiments/analyze_alpha_scan.py`

**分析内容**:
1. 计算每个 α 的平均 L1 误差（3种子均值）
2. 绘制 α vs L1 散点图 + 误差棒
3. 选择最优 α*（L1 最小）
4. 输出统计显著性检验结果（配对 t 检验）

**输出**:
- `experiments/results/phase_b_alpha_scan/analysis_report.md`: 分析报告
- 最优 α* 写入配置文件 `experiments/configs/phase_b0_baseline.yaml`

---

### 步骤 3.4: W 校准实验
**文件**: `experiments/calibrate_w.py`

**目标**: 确定"一个块"的合适轮数 W

**方法**:
1. 使用最优 α* 运行长时程实验（1000轮）
2. 分析 best_L1 时间序列的改善特征:
   - 计算滑窗改善率 `(L1[t-w] - L1[t]) / w` 对不同 w 的分布
   - 找到改善率从"快速下降"转为"停滞"的特征时间尺度
3. 根据探测逻辑需求确定 W:
   - W 太小：频繁误触发探测（噪声敏感）
   - W 太大：响应迟钝（错过真实停滞）

**输出**:
- `experiments/results/w_calibration/improvement_timescale.png`: 改善率 vs 窗口大小
- `experiments/results/w_calibration/recommended_w.txt`: 推荐 W 值
- 将 W 写入 `experiments/configs/phase_b2_probe.yaml`

---

### 步骤 3.5: B0 基线实验
**文件**: `experiments/phase_b0_baseline.py`

**配置**:
- α = α* (步骤 3.3 确定)
- 数据集: nltcs
- 种子: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (测试集)
- 轮数: 500
- 接受规则: A0 (已确认 A0 ≡ A1)

**输出**:
- `experiments/results/phase_b0_baseline/summary.csv`: 10 种子汇总
- `experiments/results/phase_b0_baseline/rounds.csv`: 完整轨迹
- `experiments/results/phase_b0_baseline/convergence.png`: 收敛曲线

---

### 步骤 3.6: 文档更新
**文件**: `PROJECT_STATUS.md`, `/root/projects/工作笔记.md`

**记录内容**:
1. 最优 α* 值及其选择依据
2. W 校准结果及物理解释
3. B0 基线性能指标（10种子均值±标准差）
4. 冻结的实验协议（为阶段 4 对照）

---

## 预期成果

### 定量指标
- 最优 α* ∈ {2, 4, 6, 8, 10}
- B0 基线 L1 误差: μ ± σ (10种子)
- 块大小 W ∈ [10, 50] 轮（预估）

### 科学意义
- **B0**: 固定 α 基线，代表"不自适应"的最优静态策略
- **W**: 探测机制的时间分辨率，平衡响应性与鲁棒性
- **对照设计**: B1 (旧轮次调度) vs B0 (最优固定) vs B2 (探测自适应)

---

## 风险与依赖

### 技术风险
- **α 扫描成本**: 5 × 3 = 15 次实验，每次 ~500轮，预估 GPU 时间 2-3 小时
- **W 校准主观性**: 改善时间尺度可能无明显分界，需结合经验判断

### 前置依赖
- ✅ 阶段 0: 基础设施已就绪
- ✅ 阶段 1-2: A0 ≡ A1 结论已确认

### 后续阶段依赖
- 阶段 4 (探测自适应) 依赖: α*, W, B0 基线

---

## 实施顺序

1. **修改 `evolution.py`** (代码修改，无实验)
2. **运行 α 扫描** (GPU 密集，2-3 小时)
3. **分析 α 扫描结果** (快速，10 分钟)
4. **W 校准实验** (GPU 密集，1 小时)
5. **B0 基线实验** (GPU 密集，3-4 小时)
6. **文档更新** (快速，15 分钟)

**总预估时间**: 6-8 小时（主要是 GPU 计算）
