"""
实验配置管理

标准化实验参数结构，支持 YAML 序列化/反序列化和参数验证。
"""
import yaml
from pathlib import Path
from typing import Optional, Literal, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class AcceptanceRuleConfig:
    """接受规则配置"""
    rule: Literal["A0", "A1"]  # 移除 A2（已删除）
    eps_L1: Optional[float] = None  # A1 需要
    eps_Q: float = 0.0  # 默认值，所有规则都用


@dataclass
class AlphaScheduleConfig:
    """α 调度配置"""
    mode: Literal["fixed", "round_schedule", "probe"]
    alpha_value: Optional[float] = None  # fixed 模式需要
    alpha_min: float = 2.0
    alpha_max: float = 10.0
    # probe 模式参数
    W: Optional[int] = None  # 块大小，单位为轮数（每 W 轮结束时检测停滞）
    P: int = 3  # 停滞触发块数
    H: int = 2  # 探测分支块数
    s: float = 0.10  # 归一化步长
    C: int = 2  # 冷却块数


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str
    target_path: str
    measured_target_path: str
    init_marginals_path: str
    n_records: int
    device: str = "cpu"  # 运行设备


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

    def validate(self):
        """验证配置的合理性"""
        errors = []

        # 验证接受规则
        if self.acceptance_rule.rule not in ["A0", "A1"]:
            errors.append(f"未知接受规则: {self.acceptance_rule.rule}，仅支持 A0 或 A1")

        if self.acceptance_rule.rule == "A1":
            if self.acceptance_rule.eps_L1 is None:
                errors.append("A1 需要指定 eps_L1")
            elif self.acceptance_rule.eps_L1 < 0:
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
            if self.alpha_schedule.W is None:
                errors.append("probe 模式需要指定 W（块大小）")
            elif self.alpha_schedule.W <= 0:
                errors.append("W 必须 > 0")

            if self.alpha_schedule.P <= 0:
                errors.append("P 必须 > 0")
            if self.alpha_schedule.H <= 0:
                errors.append("H 必须 > 0")
            if self.alpha_schedule.C < 0:
                errors.append("C 必须 >= 0")
            if not (0 < self.alpha_schedule.s < 1):
                errors.append("s 必须在 (0, 1) 范围内")

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
            "mu", "lambda_", "delta", "winsorize_limits", "candidate_budget"
        }

        # 检查未知键
        unknown_keys = set(data.keys()) - valid_keys
        if unknown_keys:
            raise ValueError(f"配置文件包含未知键: {unknown_keys}")

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
                "winsorize_limits", "candidate_budget"
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
            "winsorize_limits": list(self.winsorize_limits)
        }

        if self.candidate_budget is not None:
            data["candidate_budget"] = self.candidate_budget

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
