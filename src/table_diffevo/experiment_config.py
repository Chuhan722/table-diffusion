"""
实验配置管理

标准化实验参数结构，支持 YAML 序列化/反序列化和参数验证。
"""
import yaml
import numpy as np
from pathlib import Path
from typing import Optional, Literal, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict, fields


@dataclass
class AcceptanceRuleConfig:
    """接受规则配置

    A0/A1 采用严格改善口径（Issue #33 预注册定义）：eps_Q 是"必须改善的最小
    幅度"阈值，仅当 delta_Q < -eps_Q 才接受，Q 平局与恶化一律拒绝。A1 的 L1
    主判同为严格。此口径与主循环 `proposal_loss <= loss + tol`（非严格）在边界
    处理上不同，故 A0 非主循环逐轨迹等价 baseline。详见 acceptance.py。

    A1 的 `eps_L1` 是 L1 平局带半宽（单位「每记录平均绝对误差」），必须显式
    指定——缺省会把 A1 静默降级成另一条判据，属于换掉被测对象。对照实验取 0.0：
    normalized_l1 的分子是整数计数之差，ΔL1 只能以 `1/(m * n_records)` 为步长
    跳变，"恰好相等"是可靠可判的事件，不需要容差吸收浮点毛刺。

    取非零值时须按落带率（|ΔL1| 落进平局带的候选比例）复核松紧：带宽小于一个
    步长时等价于 0，带宽大于 L1 总量时 A1 在算术上退化为 A0，两端都让对照失效。
    同一数值在不同规模数据集上的落带率可差两个数量级，不能跨数据集直接沿用。
    """
    rule: Literal["A0", "A1"]  # 移除 A2（已删除）
    eps_L1: Optional[float] = None  # A1 必填：L1 平局带半宽（>=0）
    eps_Q: float = 0.0  # Q 严格改善阈值（delta_Q < -eps_Q 才接受，>=0）


@dataclass
class AlphaScheduleConfig:
    """α 调度配置"""
    mode: Literal["fixed", "round_schedule", "probe"]
    alpha_value: Optional[float] = None  # fixed 模式需要
    alpha_min: float = 2.0
    alpha_max: float = 10.0
    # probe 模式参数
    probe_block_candidate_budget: Optional[int] = None
    """块大小W：检测停滞的候选评估次数窗口。

    每执行W次候选评估后，检查这个窗口内的平均L1改善是否 < eps_L1。
    注意：单位是"候选评估次数"，不是"轮数"。
    - 候选评估 = 生成候选表 → 评估所有查询 → 计算误差（算1次）
    - 当 max_retries>0 时，一轮可能包含多次候选评估（初次+重试）
    """

    probe_P: int = 3
    """停滞触发块数P：连续P个块改善不足时触发探测"""

    probe_H_candidate_budget: int = 2
    """探测预算H：每个探测分支（UP/HOLD/DOWN）的候选评估次数上限。

    每个分支最多评估H次候选，不是"跑H轮"。
    用于限制探测分支的计算成本。
    """

    probe_s: float = 0.10
    """归一化步长s：探测时alpha调整幅度，范围(0,1)"""

    probe_C: int = 2
    """冷却块数C：探测后冷却C个块才能再次探测"""


@dataclass
class DataConfig:
    """数据配置

    说明：不区分「真实 target」与「加噪 target」。算法只有一个查询答案入口
    ——query_path 文件里每个 query 的 result 字段。无 DP 版 result 为无噪值；
    将来 DP 版把同一字段换成加噪值即可，算法代码与本配置无需改动，也不存在
    真值泄漏问题（算法从头到尾只见这一个来源）。
    """
    dataset_name: str
    init_marginals_path: str
    n_records: int

    schema_path: str = ""
    """Schema 文件路径（YAML 格式）

    示例：configs/nltcs/schema.yaml
    """

    query_path: str = ""
    """查询定义文件路径（JSON 格式）

    示例：configs/nltcs/measured_1000query.json
    """

    device: str = "cpu"
    """计算设备

    选项：
      - 'numpy': NumPy CPU 实现（默认）
      - 'cpu': PyTorch CPU
      - 'cuda': PyTorch GPU（需要 CUDA）
    """


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    experiment_name: str
    data: DataConfig
    acceptance_rule: AcceptanceRuleConfig
    alpha_schedule: AlphaScheduleConfig
    seeds: List[int]
    n_rounds: int
    output_dir: str
    # 演化参数（从现有实验继承）
    rho: float = 0.01  # 固定值（Issue #33 要求）
    beta: float = 1.0
    eta: float = 0.5
    h: float = 0.8
    mu: float = 0.01
    lambda_: float = 0.5
    delta: float = 0.05
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    # 可选的总候选评估预算上限。n_rounds 始终是主停止条件；candidate_budget
    # 若给定，则作为额外的评估次数上限（先到者停止）。二者不是互斥关系。
    candidate_budget: Optional[int] = None

    # ===== 初始化参数 =====
    init_method: Literal["random", "marginal", "pairwise_maxent"] = "random"
    """初始化方式

    选项：
      - 'random': 纯随机（每格从 schema 合法域均匀抽样）
      - 'marginal': 按 1-way 边缘确定性初始化（需提供 init_marginals_path）
      - 'pairwise_maxent': 二阶最大熵（nltcs 推荐，全类别数据集）
    """

    maxent_max_states: int = 1_000_000
    """pairwise_maxent 可枚举的最大联合状态数

    单位：状态数
    默认：1,000,000
    用途：防止意外耗尽内存
    """

    maxent_max_sweeps: int = 200
    """pairwise_maxent 的 IPF 最大扫描轮数

    单位：轮
    默认：200
    """

    maxent_tol: float = 1e-8
    """pairwise_maxent 的最大二阶单元概率误差收敛阈值

    单位：概率
    默认：1e-8
    """

    # ===== 计算与性能参数 =====
    eval_method: Literal["vectorized", "legacy"] = "vectorized"
    """查询评价方式（性能开关，不改变结果）

    选项：
      - 'vectorized': 向量化+分块评价（默认，快）
      - 'legacy': 原始逐查询 pandas 路径（慢，用于对拍/应急）
    """

    batch_size: int = 256
    """向量化评价的分块大小（一次算多少个查询）

    单位：查询数
    范围：> 0
    默认：256
    注意：仅 eval_method='vectorized' 生效；内存峰值 ∝ n_records × batch_size
    """

    log_every: int = 0
    """逐轮进度打印频率

    单位：轮
    选项：
      - 0: 每轮都打印（默认，向后兼容）
      - >0: 每 N 轮打印一次（首轮与末轮总会打印）
    """

    tol: float = 1e-9
    """整代检查的数值容差

    单位：loss
    默认：1e-9
    含义：loss(proposal) ≤ loss(S) + tol 时接受
    """

    # ===== 抽样参数 =====
    distance_mode: Literal["geometric", "linear", "squared", "multiplicative", "none"] = "geometric"
    """距离项的处理方式

    选项：
      - 'geometric': 归一化+几何均值+动态锐度调度（推荐，nltcs 最优）
      - 'linear': exp(-d/h)，拉普拉斯核
      - 'squared': exp(-d²/2h²)，高斯核
      - 'multiplicative': softmax(β·F) × (1-d)^p
      - 'none': 不考虑距离，只用适应度
    """

    p: float = 1.0
    """multiplicative 模式的距离陡度参数

    范围：≥ 0
    默认：1.0
    注意：仅 distance_mode='multiplicative' 生效
    """

    exclude_self: bool = True
    """是否禁止记录抽到自己（对角线屏蔽）

    默认：True
    含义：候选池=全表时，禁止抽到自己可消除自我复制空转
    注意：共享参考池（K≠N）时必须设为 False
    """

    # ===== 重试参数 =====
    max_retries: int = 0
    """整代提案被拒后的最大重试次数

    范围：≥ 0
    默认：0（不重试）
    含义：大于 0 时复用当轮 donor，逐次缩小 rho 重新生成提案
    """

    retry_rho_decay: float = 0.5
    """每次重试的参与率缩放因子

    范围：(0, 1)
    默认：0.5
    含义：第 a 次尝试使用 rho × retry_rho_decay^a
    """

    # ===== 残差驱动扩散核参数 =====
    residual_directed_diffusion: bool = False
    """是否启用残差驱动的局部扩散核

    默认：False（关闭）
    含义：让实际单块 donor 复制的比例残差方向量连续倾斜复制概率
    注意：不执行正负门控或逐候选 top-k
    """

    diffusion_direction_strength: float = 1.0
    """残差驱动扩散的非负有限强度

    范围：≥ 0
    默认：1.0
    含义：0 在启用机制时也精确退化到历史固定 eta；正值越大，复制概率对实际局部方向越敏感
    """

    diffusion_direction_normalization: Literal["none", "initial_rms"] = "initial_rms"
    """方向强度的尺度口径

    选项：
      - 'none': 直接使用原始比例残差方向量
      - 'initial_rms': 用本次运行首个非零方向矩阵的 RMS 固定定标（默认）
    """

    # ===== Gibbs 参数 =====
    factorized_gibbs_sweeps: int = 0
    """每条参与记录在独立定向初始 mask 后执行的随机扫描 Gibbs sweep 数

    范围：≥ 0
    默认：0（完全保留既有独立单块更新）
    注意：正数只允许与 residual_directed_diffusion 一起启用
    """

    factorized_gibbs_max_order: int = 3
    """查询局部因子的最高允许属性阶数

    范围：≥ 1
    默认：3
    注意：仅在 factorized_gibbs_sweeps > 0 时使用
    """

    factorized_gibbs_logit_clip: Optional[float] = 30.0
    """Gibbs 条件 logit 的对称数值护栏

    范围：> 0 或 None
    默认：30.0
    含义：正有限数值保留极端有限温度下的双向 float64 支持；显式传入 None 可关闭
    """

    def validate(self):
        """验证配置的合理性"""
        errors = []

        # 验证接受规则
        if self.acceptance_rule.rule not in ["A0", "A1"]:
            errors.append(f"未知接受规则: {self.acceptance_rule.rule}，仅支持 A0 或 A1")

        # A1 的平局带半宽必须显式给出：缺了它语义就变成「纯 L1 严格改善」（另一条
        # 判据），静默回落等于换掉被测对象。0.0 是合法值，不可与 None 混为一谈。
        _abs_given = self.acceptance_rule.eps_L1 is not None
        if self.acceptance_rule.rule == "A1" and not _abs_given:
            errors.append("A1 需要显式指定 eps_L1（L1 平局带半宽，可为 0）")
        if _abs_given and self.acceptance_rule.eps_L1 < 0:
            errors.append("eps_L1 必须 >= 0")

        if self.acceptance_rule.eps_Q < 0:
            errors.append("eps_Q 必须 >= 0")

        # 验证 α 调度
        if self.alpha_schedule.mode not in ["fixed", "round_schedule", "probe"]:
            errors.append(f"未知 alpha_schedule.mode: {self.alpha_schedule.mode}")

        if self.alpha_schedule.mode == "fixed":
            if self.alpha_schedule.alpha_value is None:
                errors.append("fixed 模式需要指定 alpha_value")
            elif not (self.alpha_schedule.alpha_min <= self.alpha_schedule.alpha_value <= self.alpha_schedule.alpha_max):
                errors.append(f"alpha_value ({self.alpha_schedule.alpha_value}) 必须在 [{self.alpha_schedule.alpha_min}, {self.alpha_schedule.alpha_max}] 范围内")

        elif self.alpha_schedule.mode == "probe":
            if self.alpha_schedule.probe_block_candidate_budget is None:
                errors.append("probe 模式需要指定 probe_block_candidate_budget（块大小W）")
            elif self.alpha_schedule.probe_block_candidate_budget <= 0:
                errors.append("probe_block_candidate_budget 必须 > 0")

            if self.alpha_schedule.probe_P <= 0:
                errors.append("probe_P 必须 > 0")
            if self.alpha_schedule.probe_H_candidate_budget <= 0:
                errors.append("probe_H_candidate_budget 必须 > 0")
            if self.alpha_schedule.probe_C < 0:
                errors.append("probe_C 必须 >= 0")
            if not (0 < self.alpha_schedule.probe_s < 1):
                errors.append("probe_s 必须在 (0, 1) 范围内")

        # 验证 alpha 范围
        if self.alpha_schedule.alpha_min >= self.alpha_schedule.alpha_max:
            errors.append("alpha_min 必须小于 alpha_max")

        # 验证种子数量
        if len(self.seeds) == 0:
            errors.append("至少需要 1 个 seed")

        # 验证轮数
        if self.n_rounds <= 0:
            errors.append("n_rounds 必须 > 0")

        # 验证候选预算：给定时必须为正
        if self.candidate_budget is not None and self.candidate_budget <= 0:
            errors.append("candidate_budget 若指定则必须 > 0")

        # 验证数据配置
        if self.data.n_records <= 0:
            errors.append("n_records 必须 > 0")

        # 验证设备白名单
        valid_devices = {"cpu", "cuda", "numpy"}
        if self.data.device not in valid_devices:
            errors.append(
                f"未知 device: {self.data.device}，仅支持 {sorted(valid_devices)}"
            )

        # 验证 rho
        if not (0 < self.rho <= 1):
            errors.append("rho 必须在 (0, 1] 范围内")

        # 验证初始化参数
        if self.init_method not in ["random", "marginal", "pairwise_maxent"]:
            errors.append(f"未知 init_method: {self.init_method}")

        if self.maxent_max_states <= 0:
            errors.append("maxent_max_states 必须 > 0")

        if self.maxent_max_sweeps <= 0:
            errors.append("maxent_max_sweeps 必须 > 0")

        if self.maxent_tol <= 0:
            errors.append("maxent_tol 必须 > 0")

        # 验证计算与性能参数
        if self.eval_method not in ["vectorized", "legacy"]:
            errors.append(f"未知 eval_method: {self.eval_method}")

        if self.batch_size <= 0:
            errors.append("batch_size 必须 > 0")

        if self.log_every < 0:
            errors.append("log_every 必须 >= 0")

        if self.tol < 0:
            errors.append("tol 必须 >= 0")

        # 验证抽样参数
        if self.distance_mode not in ["geometric", "linear", "squared", "multiplicative", "none"]:
            errors.append(f"未知 distance_mode: {self.distance_mode}")

        if self.p < 0:
            errors.append("p 必须 >= 0")

        # 验证重试参数
        if self.max_retries < 0:
            errors.append("max_retries 必须 >= 0")

        if not (0 < self.retry_rho_decay < 1):
            errors.append("retry_rho_decay 必须在 (0, 1) 范围内")

        # 验证扩散核参数
        if not isinstance(self.residual_directed_diffusion, bool):
            errors.append("residual_directed_diffusion 必须是布尔值")

        if self.diffusion_direction_strength < 0:
            errors.append("diffusion_direction_strength 必须 >= 0")

        if self.diffusion_direction_normalization not in ["none", "initial_rms"]:
            errors.append(f"未知 diffusion_direction_normalization: {self.diffusion_direction_normalization}")

        # 验证 Gibbs 参数
        if self.factorized_gibbs_sweeps < 0:
            errors.append("factorized_gibbs_sweeps 必须 >= 0")

        if self.factorized_gibbs_max_order < 1:
            errors.append("factorized_gibbs_max_order 必须 >= 1")

        if self.factorized_gibbs_logit_clip is not None and self.factorized_gibbs_logit_clip <= 0:
            errors.append("factorized_gibbs_logit_clip 若指定则必须 > 0")

        if self.factorized_gibbs_sweeps > 0 and not self.residual_directed_diffusion:
            errors.append("factorized_gibbs_sweeps > 0 要求启用 residual_directed_diffusion")

        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """从 YAML 文件加载配置"""
        with open(path) as f:
            data = yaml.safe_load(f)

        # 定义合法的顶层键
        valid_keys = {
            "experiment_name", "data", "acceptance_rule", "alpha_schedule",
            "seeds", "n_rounds", "output_dir", "rho", "beta", "eta", "h",
            "mu", "lambda_", "delta", "winsorize_limits", "candidate_budget",
            # 新增参数
            "init_method", "maxent_max_states", "maxent_max_sweeps", "maxent_tol",
            "eval_method", "batch_size", "log_every", "tol",
            "distance_mode", "p", "exclude_self",
            "max_retries", "retry_rho_decay",
            "residual_directed_diffusion", "diffusion_direction_strength",
            "diffusion_direction_normalization",
            "factorized_gibbs_sweeps", "factorized_gibbs_max_order",
            "factorized_gibbs_logit_clip"
        }

        # 检查顶层未知键
        unknown_keys = set(data.keys()) - valid_keys
        if unknown_keys:
            raise ValueError(f"配置文件包含未知键: {sorted(unknown_keys)}")

        # 检查嵌套节的未知键。顶层护栏拦不住嵌套 dict 里的拼写错误——
        # 那些键会被 DataConfig(**...) 等原样展开，抛出原始 TypeError（定位差、
        # 不符合“拒绝未知 YAML 键”的契约）。这里对每个嵌套节按其 dataclass 字段
        # 白名单校验，提前给出带节名定位的 ValueError。
        _NESTED_SECTIONS = {
            "data": DataConfig,
            "acceptance_rule": AcceptanceRuleConfig,
            "alpha_schedule": AlphaScheduleConfig,
        }
        for section, section_cls in _NESTED_SECTIONS.items():
            # 三节均为必填。缺失或显式为空（`data:` / `data: null`）若放行，会落到
            # 下方 DataConfig(**None) 抛晦涩的 TypeError——与“坏 YAML 给清晰错误”的
            # 契约相悖。这里提前拦成带节名的 ValueError（与未知键/类型错误同一风格）。
            if section not in data or data[section] is None:
                raise ValueError(
                    f"配置节 {section!r} 缺失或为空，必须提供键值映射"
                )
            section_data = data[section]
            if not isinstance(section_data, dict):
                raise ValueError(
                    f"配置节 {section!r} 必须是键值映射，实际为 {type(section_data).__name__}"
                )
            allowed = {f.name for f in fields(section_cls)}
            unknown_nested = set(section_data.keys()) - allowed
            if unknown_nested:
                raise ValueError(
                    f"配置节 {section!r} 包含未知键: {sorted(unknown_nested)}"
                    f"（合法键：{sorted(allowed)}）"
                )

        # 递归构建嵌套 dataclass
        config = cls(
            experiment_name=data["experiment_name"],
            data=DataConfig(**data["data"]),
            acceptance_rule=AcceptanceRuleConfig(**data["acceptance_rule"]),
            alpha_schedule=AlphaScheduleConfig(**data["alpha_schedule"]),
            seeds=data["seeds"],
            n_rounds=data["n_rounds"],
            output_dir=data["output_dir"],
            **{k: v for k, v in data.items() if k in [
                "rho", "beta", "eta", "h", "mu", "lambda_", "delta",
                "winsorize_limits", "candidate_budget",
                "init_method", "maxent_max_states", "maxent_max_sweeps", "maxent_tol",
                "eval_method", "batch_size", "log_every", "tol",
                "distance_mode", "p", "exclude_self",
                "max_retries", "retry_rho_decay",
                "residual_directed_diffusion", "diffusion_direction_strength",
                "diffusion_direction_normalization",
                "factorized_gibbs_sweeps", "factorized_gibbs_max_order",
                "factorized_gibbs_logit_clip"
            ]}
        )

        config.validate()
        return config

    def to_yaml(self, path: Path):
        """保存配置到 YAML 文件"""
        # 转换为字典
        data = {
            "experiment_name": self.experiment_name,
            "data": asdict(self.data),
            "acceptance_rule": asdict(self.acceptance_rule),
            "alpha_schedule": asdict(self.alpha_schedule),
            "seeds": self.seeds,
            "n_rounds": self.n_rounds,
            "output_dir": self.output_dir,
            "rho": self.rho,
            "beta": self.beta,
            "eta": self.eta,
            "h": self.h,
            "mu": self.mu,
            "lambda_": self.lambda_,
            "delta": self.delta,
            "winsorize_limits": list(self.winsorize_limits),
            # 初始化参数
            "init_method": self.init_method,
            "maxent_max_states": self.maxent_max_states,
            "maxent_max_sweeps": self.maxent_max_sweeps,
            "maxent_tol": self.maxent_tol,
            # 计算与性能参数
            "eval_method": self.eval_method,
            "batch_size": self.batch_size,
            "log_every": self.log_every,
            "tol": self.tol,
            # 抽样参数
            "distance_mode": self.distance_mode,
            "p": self.p,
            "exclude_self": self.exclude_self,
            # 重试参数
            "max_retries": self.max_retries,
            "retry_rho_decay": self.retry_rho_decay,
            # 扩散核参数
            "residual_directed_diffusion": self.residual_directed_diffusion,
            "diffusion_direction_strength": self.diffusion_direction_strength,
            "diffusion_direction_normalization": self.diffusion_direction_normalization,
            # Gibbs 参数
            "factorized_gibbs_sweeps": self.factorized_gibbs_sweeps,
            "factorized_gibbs_max_order": self.factorized_gibbs_max_order,
        }

        if self.candidate_budget is not None:
            data["candidate_budget"] = self.candidate_budget

        if self.factorized_gibbs_logit_clip is not None:
            data["factorized_gibbs_logit_clip"] = self.factorized_gibbs_logit_clip

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    def to_run_evolution_kwargs(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """转换为 run_evolution 的参数字典（仅覆盖阶段 0 已真实接入的口径）。

        此方法加载必要的文件（schema、queries）并将配置参数转换为
        run_evolution() 函数需要的格式。

        **接入范围（fail-closed）**：阶段 1 已把 A0/A1 接受规则接入主循环，
        故本方法现在真实映射 `acceptance_rule` / `eps_L1` / `eps_Q`。α 调度仍
        只支持线性轮数调度（alpha_min→alpha_max）；配置里预注册的 fixed/probe
        尚未接入——留待阶段 2-5（探测调度）。为避免"配置填了却静默按另一套算法
        运行"，本方法对尚未接入的口径**直接报错**，而不是悄悄丢弃或错误映射。

        **注意（口径差异须披露）**：显式指定 A0/A1 时主循环走 acceptance.py 的
        严格改善口径（拒绝平局），与主循环历史默认判据
        `proposal_loss <= loss + tol`（非严格）在边界处理上不同。因此 A0 **不是**
        历史 baseline 的逐轨迹等价复现——要复现历史轨迹需让 `acceptance_rule=None`，
        而本方法总会显式给出规则名（配置里 rule 是必填项）。

        Parameters
        ----------
        seed : int, optional
            随机种子。如果不提供，调用者需要在返回的字典中手动设置。

        Returns
        -------
        dict
            包含 run_evolution 已接入参数的字典

        Raises
        ------
        NotImplementedError
            当配置使用尚未接入主循环的口径时（alpha_schedule.mode 为
            fixed/probe）。

        Usage
        -----
        >>> config = ExperimentConfig.from_yaml("xxx.yaml")
        >>> kwargs = config.to_run_evolution_kwargs(seed=0)
        >>> best_S, diag = run_evolution(**kwargs)
        """
        # Fail-closed：只放行主循环“已真正接入”的口径，其余一律报错——绝不
        # 静默按默认算法运行。用白名单（而非黑名单）判定：拼错/新增的口径会落到
        # 白名单之外而被挡下，不会 fail-open 溜进映射。
        # 先查 α 模式，再查接受规则——这样 fixed/probe 配置能看到 α 模式的报错，
        # round_schedule 配置则落到接受规则的报错，两道护栏都可观测、可测试。
        #
        # 阶段推进时放开对应白名单即可：
        #   阶段 1 已接入接受规则，A0/A1 在白名单内；
        #   阶段 2-5 接入探测调度后，把 fixed/probe 加入 _WIRED_ALPHA_MODES。
        _WIRED_ALPHA_MODES = {"round_schedule"}          # 阶段 0 只接了线性轮数调度
        _WIRED_ACCEPTANCE_RULES = {"A0", "A1"}           # 阶段 1 接入 A0/A1

        mode = self.alpha_schedule.mode
        if mode not in _WIRED_ALPHA_MODES:
            raise NotImplementedError(
                f"alpha_schedule.mode={mode!r} 尚未接入主循环，留待阶段 2-5（探测调度）。"
                f"当前 run_evolution 仅支持线性轮数调度（round_schedule，alpha_min→alpha_max）；"
                f"fixed/probe 的语义（固定 α、停滞触发三岔路探测）需后续阶段实现。"
            )
        rule = self.acceptance_rule.rule
        if rule not in _WIRED_ACCEPTANCE_RULES:
            # 白名单之外：拼错的、或未来新增而尚未接线的规则名。一律挡下，
            # 绝不 fail-open 静默按主循环默认判据跑。
            raise NotImplementedError(
                f"acceptance_rule={rule!r} 未接入主循环。"
                f"当前仅支持 {sorted(_WIRED_ACCEPTANCE_RULES)}（阶段 1 已接入）；"
                f"配置可记录其他值作为预注册意图，但不能据此运行。"
            )
        # A1 的平局带半宽是主判容差，缺了它语义就变成"纯 L1 严格改善"（另一条
        # 判据）。这里必须报错而不是回落到 0.0——否则等于静默换掉被测对象。
        if rule == "A1" and self.acceptance_rule.eps_L1 is None:
            raise ValueError(
                "acceptance_rule.rule='A1' 必须显式指定 eps_L1 作为 L1 平局带半宽"
                "（0 是合法取值）；缺省会把 A1 静默降级为纯 L1 严格改善判据，"
                "属于换掉被测对象。"
            )

        from table_diffevo.schema import load_schema
        from table_diffevo.queries import load_queries
        from table_diffevo.marginals import load_marginals

        # 加载数据文件
        schema = load_schema(self.data.schema_path)
        queries = load_queries(self.data.query_path)
        target = np.array([q["result"] for q in queries])

        # 加载边缘（如果需要）
        marginals = None
        if self.init_method == "marginal" and self.data.init_marginals_path:
            marginals = load_marginals(self.data.init_marginals_path)

        # 构建参数字典
        kwargs = {
            "target": target,
            "queries": queries,
            "schema": schema,
            "n_records": self.data.n_records,
            "n_rounds": self.n_rounds,
            "rho": self.rho,
            "beta": self.beta,
            "eta": self.eta,
            "h": self.h,
            "mu": self.mu,
            "tol": self.tol,
            "device": self.data.device,
            "eval_method": self.eval_method,
            "batch_size": self.batch_size,
            "init_method": self.init_method,
            "marginals": marginals,
            "log_every": self.log_every,
            "distance_mode": self.distance_mode,
            "p": self.p,
            "lambda_param": self.lambda_,
            "alpha_min": self.alpha_schedule.alpha_min,
            "alpha_max": self.alpha_schedule.alpha_max,
            "delta": self.delta,
            "winsorize_quantiles": self.winsorize_limits,
            "exclude_self": self.exclude_self,
            "max_retries": self.max_retries,
            "retry_rho_decay": self.retry_rho_decay,
            "maxent_max_states": self.maxent_max_states,
            "maxent_max_sweeps": self.maxent_max_sweeps,
            "maxent_tol": self.maxent_tol,
            "residual_directed_diffusion": self.residual_directed_diffusion,
            "diffusion_direction_strength": self.diffusion_direction_strength,
            "diffusion_direction_normalization": self.diffusion_direction_normalization,
            "factorized_gibbs_sweeps": self.factorized_gibbs_sweeps,
            "factorized_gibbs_max_order": self.factorized_gibbs_max_order,
            "factorized_gibbs_logit_clip": self.factorized_gibbs_logit_clip,
            "candidate_budget": self.candidate_budget,
            # 阶段 1 接入：显式规则名 → 主循环走 acceptance.py 的严格改善口径。
            # A0 不读平局带，eps_L1 缺省时补 0.0（run_evolution 的默认值）。
            # 注意不能用 `or 0.0` 兜底：0.0 是合法且有语义的取值（平局带退化为
            # 「ΔL1 恰好为 0」），`or` 会把它和 None 混为一谈。
            "acceptance_rule": rule,
            "eps_L1": (
                float(self.acceptance_rule.eps_L1)
                if self.acceptance_rule.eps_L1 is not None
                else 0.0
            ),
            "eps_Q": float(self.acceptance_rule.eps_Q),
        }

        # 如果提供了种子，添加到参数中
        if seed is not None:
            kwargs["seed"] = seed

        return kwargs
